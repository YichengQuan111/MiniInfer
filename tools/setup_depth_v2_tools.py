#!/usr/bin/env python3
"""Fetch CUDA 12.4.1 compiler and compatible profilers; install locally in WSL.

Windows: python tools/setup_depth_v2_tools.py fetch --downloads tmp/v2-tools
WSL: python3 tools/setup_depth_v2_tools.py install --downloads tmp/v2-tools \
    --report docs/validation/v21_toolchain.json
No driver/package-manager/venv changes. Installation performs version checks and
a compile-only CUDA smoke test. The explicit profile-smoke action executes only
one tiny CUDA kernel to verify real kernel/memcpy tracing; it does not run TRT.
"""
import argparse
import concurrent.futures
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import sqlite3
import subprocess
import tarfile
import urllib.request

BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
MANIFEST = "redistrib_12.4.1.json"
MANIFEST_SHA = "9cd815f3b71c2e3686ef2219b7794b81044f9dcefaa8e21dacfcb5bc4d931892"
COMPONENTS = ("cuda_nvcc", "cuda_cudart", "cuda_cccl", "cuda_sanitizer_api",
              "nsight_compute", "nsight_systems")
NSYS_RELEASES = {
    "12.4.1": MANIFEST_SHA,
    "12.6.3": "9c598598457a6463eb92889080c16b2b9dc04150e501b8bfc1536d403ba70aaf",
    "13.0.3": "d73344990be5d826cf6eedc5495c858eb5923e5ffe7ede7fd6ebdd7be928ffc9",
    "13.4.1": "291178934b2139727407c76f697bb0dfbac4014753a8094d7c290b0f4124a302",
    "cli-2026.5.1": "35623a7aaa47b9f8c99f3a904ef4966891e5f52768447d6db1753a5e9d24087a",
}
NSYS_CLI_BASE = "https://developer.download.nvidia.com/devtools/repos/ubuntu2404/amd64/"
NSYS_CLI_FILE = "NsightSystems-linux-cli-public-2026.5.1.161-3889610.deb"
NSYS_CLI_SHA = "61829db6392e5c293ada1319df86a97c3356bc810335a1975cb551fcfe3eca08"


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def fetch_file(url, path, expected):
    if path.exists() and digest(path) == expected:
        return
    partial = path.with_suffix(path.suffix + ".partial")
    curl = shutil.which("curl.exe" if platform.system() == "Windows" else "curl")
    if curl:
        subprocess.run([curl, "--fail", "--location", "--silent", "--show-error",
                        "--retry", "2", "--connect-timeout", "15", "--max-time", "600",
                        "--output", str(partial), url], check=True, timeout=1900)
    else:
        with urllib.request.urlopen(url, timeout=60) as source, partial.open("wb") as dest:
            shutil.copyfileobj(source, dest, 1024 * 1024)
    if digest(partial) != expected:
        raise ValueError(f"SHA-256 mismatch: {path.name}")
    partial.replace(path)
    print(f"Verified {path.name}", flush=True)


def metadata(downloads, fetch=False):
    path = downloads / MANIFEST
    if fetch:
        fetch_file(BASE + MANIFEST, path, MANIFEST_SHA)
    if digest(path) != MANIFEST_SHA:
        raise ValueError("Unrecognized CUDA release manifest")
    return json.loads(path.read_text())


def nsys_metadata(args, fetch=False):
    if args.nsys_release == "cli-2026.5.1":
        name = "nsight-packages.gz"
        path = args.downloads / name
        expected = NSYS_RELEASES[args.nsys_release]
        if fetch:
            fetch_file(NSYS_CLI_BASE + "Packages.gz", path, expected)
        if digest(path) != expected:
            raise ValueError("Unrecognized NVIDIA repository snapshot")
        import gzip
        blocks = gzip.decompress(path.read_bytes()).decode().split("\n\n")
        block = next(block for block in blocks if f"Filename: {NSYS_CLI_FILE}\n" in block + "\n")
        if f"SHA256: {NSYS_CLI_SHA}\n" not in block + "\n":
            raise ValueError("Nsight CLI hash differs from official repository snapshot")
        return {"_download_base": NSYS_CLI_BASE, "_source_manifest": NSYS_CLI_BASE + "Packages.gz",
                "_source_sha256": expected, "nsight_systems": {"version": "2026.5.1.161",
                "linux-x86_64": {"relative_path": NSYS_CLI_FILE, "sha256": NSYS_CLI_SHA,
                                  "size": "225594886"}}}
    name = f"redistrib_{args.nsys_release}.json"
    path = args.downloads / name
    expected = NSYS_RELEASES[args.nsys_release]
    if fetch:
        fetch_file(BASE + name, path, expected)
    if digest(path) != expected:
        raise ValueError("Unrecognized Nsight Systems release manifest")
    value = json.loads(path.read_text())
    value.update(_download_base=BASE, _source_manifest=BASE + name, _source_sha256=expected)
    return value


def run(command):
    completed = subprocess.run(command, capture_output=True, text=True, timeout=90)
    return {"command": command, "returncode": completed.returncode,
            "stdout": completed.stdout.strip(), "stderr": completed.stderr.strip()}


def is_x86_64_elf(path):
    with path.open("rb") as handle:
        header = handle.read(20)
        return header[:6] == b"\x7fELF\x02\x01" and header[18:20] == b"\x3e\x00"


def trace_tables(path):
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        names = [row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        counts = {name: connection.execute('SELECT COUNT(*) FROM "' + name.replace('"', '""') + '"').fetchone()[0]
                  for name in names if name.startswith("CUPTI_") or name == "NVTX_EVENTS"}
        diagnostics = {name: connection.execute('SELECT * FROM "' + name.replace('"', '""') + '" LIMIT 20').fetchall()
                       for name in names if "DIAGNOSTIC" in name or name == "ANALYSIS_DETAILS"}
    return {"path": str(path.resolve()), "sha256": digest(path), "activity_counts": counts,
            "diagnostics": diagnostics,
            "has_kernel": any("KERNEL" in name and count > 0 for name, count in counts.items()),
            "has_memcpy": any("MEMCPY" in name and count > 0 for name, count in counts.items())}


def profile_smoke(args):
    if platform.system() != "Linux":
        raise RuntimeError("Profile smoke requires Linux/WSL")
    output = args.profile_output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output / "profile.cu"
    source.write_text(r'''#include <cuda_runtime.h>
#include <cstdio>
#define CHECK(call) do { auto e=(call); if(e!=cudaSuccess) { std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e)); return 1; } } while(0)
__global__ void add_one(float* p) { int i=threadIdx.x; p[i]+=1.0f; }
int main() {
  float *host=nullptr,*device=nullptr; constexpr int n=256;
  CHECK(cudaMallocHost(&host,n*sizeof(float))); CHECK(cudaMalloc(&device,n*sizeof(float)));
  for(int i=0;i<n;++i) host[i]=static_cast<float>(i);
  CHECK(cudaMemcpy(device,host,n*sizeof(float),cudaMemcpyHostToDevice));
  add_one<<<1,n>>>(device); CHECK(cudaGetLastError()); CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(host,device,n*sizeof(float),cudaMemcpyDeviceToHost));
  for(int i=0;i<n;++i) if(host[i]!=static_cast<float>(i+1)) return 2;
  CHECK(cudaFree(device)); CHECK(cudaFreeHost(host)); CHECK(cudaDeviceReset());
  std::puts("CUDA smoke PASS: one kernel, 1024-byte H2D and D2H"); return 0;
}
''')
    nvcc = args.root.expanduser().resolve() / "cuda-12.4.1/bin/nvcc"
    application = output / "profile-smoke"
    report = {"recorded_utc": datetime.now(timezone.utc).isoformat(),
              "nsys_version": run([str(args.nsys), "--version"]),
              "scope": "one 256-thread kernel and two 1024-byte transfers; no TRT workload",
              "compile": run([str(nvcc), "-std=c++17", "-arch=sm_89", "--fmad=false",
                              str(source), "-o", str(application)])}
    if args.baseline_sqlite:
        report["original_trace"] = trace_tables(args.baseline_sqlite)
    if report["compile"]["returncode"] == 0:
        prefix = output / "timeline"
        report["profile"] = run([str(args.nsys), "profile", "--trace=cuda,nvtx", "--sample=none",
                                 "--cpuctxsw=none", "--force-overwrite=false", "--output", str(prefix),
                                 str(application)])
        report["export"] = run([str(args.nsys), "export", "--type=sqlite", "--output",
                                str(prefix.with_suffix(".sqlite")), str(prefix.with_suffix(".nsys-rep"))])
        if prefix.with_suffix(".sqlite").exists():
            report["trace"] = trace_tables(prefix.with_suffix(".sqlite"))
    report["passed"] = bool(report.get("trace", {}).get("has_kernel") and
                            report.get("trace", {}).get("has_memcpy") and
                            report.get("profile", {}).get("returncode") == 0)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "report": str(args.report),
                      "nsys_version": report["nsys_version"]["stdout"],
                      "trace": {k: report.get("trace", {}).get(k) for k in
                          ("path", "sha256", "activity_counts", "has_kernel", "has_memcpy")},
                      "profile_stderr": report.get("profile", {}).get("stderr")}, indent=2))
    return int(not report["passed"])


def install(args, manifest):
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Installation requires Linux x86_64 (run inside WSL)")
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    evidence = {"schema_version": 1, "recorded_utc": datetime.now(timezone.utc).isoformat(),
                "cuda_release": "12.4.1", "platform": platform.platform(),
                "source_manifest": BASE + MANIFEST, "source_manifest_sha256": MANIFEST_SHA,
                "root": str(root), "components": {}, "tools": {},
                "initial_path_tools": {t: shutil.which(t) for t in
                    ("nvcc", "nsys", "ncu", "compute-sanitizer")},
                "sources": ["https://docs.nvidia.com/cuda/archive/12.4.1/cuda-installation-guide-linux/index.html",
                            "https://docs.nvidia.com/cuda/wsl-user-guide/index.html",
                            "https://docs.nvidia.com/nsight-compute/2024.1/ReleaseNotes/index.html"],
                "limitations": ["Compile-only smoke test; GPU execution, profiler capture and sanitizer checks pending.",
                                "Nsight Compute counter permissions must be verified by actual capture.",
                                "No driver, existing runtime, TensorRT or PyTorch package was replaced."]}
    directories = {}
    systems_manifest = nsys_metadata(args) if "nsight_systems" in args.components else manifest
    for name in args.components:
        selected = systems_manifest if name == "nsight_systems" else manifest
        item = selected[name]["linux-x86_64"]
        archive = args.downloads / Path(item["relative_path"]).name
        if digest(archive) != item["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {archive}")
        destination = root / (f"{name}-{selected[name]['version']}" if name == "nsight_systems" else name)
        marker = destination / ".archive-sha256"
        if not marker.exists():
            destination.mkdir(parents=True, exist_ok=True)
            if archive.suffix == ".deb":
                subprocess.run(["dpkg-deb", "--extract", str(archive), str(destination)],
                               check=True, timeout=180)
            else:
                with tarfile.open(archive) as package:
                    package.extractall(destination, filter="data")
            marker.write_text(item["sha256"])
        elif marker.read_text() != item["sha256"]:
            raise ValueError(f"Different installation already exists: {destination}")
        directories[name] = destination if archive.suffix == ".deb" else next(
            p for p in destination.iterdir() if p.is_dir())
        evidence["components"][name] = {"version": selected[name]["version"],
            "url": selected.get("_download_base", BASE) + item["relative_path"], "sha256": item["sha256"],
            "source_manifest": selected.get("_source_manifest", BASE + MANIFEST),
            "source_manifest_sha256": selected.get("_source_sha256", MANIFEST_SHA),
            "archive_bytes": int(item["size"]), "installed": str(directories[name])}
    toolkit = root / "cuda-12.4.1"
    toolkit.mkdir(exist_ok=True)
    for name in ("cuda_nvcc", "cuda_cudart", "cuda_cccl"):
        shutil.copytree(directories[name], toolkit, dirs_exist_ok=True)
    # Redistributable archives use lib/; nvcc's 64-bit link search uses lib64/.
    if not (toolkit / "lib64").exists():
        (toolkit / "lib64").symlink_to("lib", target_is_directory=True)
    paths = {"nvcc": toolkit / "bin/nvcc"}
    for name, tool in (("cuda_sanitizer_api", "compute-sanitizer"),
                       ("nsight_compute", "ncu"), ("nsight_systems", "nsys")):
        if name not in directories:
            continue
        candidates = [p for p in directories[name].rglob(tool)
                      if p.is_file() and os.access(p, os.X_OK)]
        # Redist bin/ wrappers contain unresolved installer placeholders. Prefer
        # the actual ELF executable while preserving its bundled sibling files.
        paths[tool] = next((p for p in candidates if is_x86_64_elf(p)),
                           candidates[0])
    for tool, path in paths.items():
        evidence["tools"][tool] = {"path": str(path), **run([str(path), "--version"])}
    if "nsys" in paths:
        evidence["nsys_environment"] = run([str(paths["nsys"]), "status", "--environment"])
        evidence["limitations"].append("Nsight Systems system-wide CPU sampling depends on host perf permissions; see nsys_environment.")
    python_env = Path(os.environ.get("MINIINFER_DEPTH_ENV", str(Path.home() / ".venvs/miniinfer-depth")))
    evidence["existing_runtime_packages"] = run([str(python_env / "bin/python"), "-c",
        "import importlib.metadata as m, json; print(json.dumps({n:m.version(n) for n in "
        "['torch','torchvision','nvidia-cuda-runtime-cu12','nvidia-cuda-nvcc-cu12','tensorrt_cu12']}))"])
    environment = root / "env.sh"
    path_dirs = ":".join(dict.fromkeys(str(p.parent) for p in paths.values()))
    environment.write_text(
        f"export MINIINFER_CUDA_TOOLKIT={shlex.quote(str(toolkit))}\n"
        f"export CUDACXX={shlex.quote(str(paths['nvcc']))}\n"
        f"export PATH={shlex.quote(path_dirs)}:\"$PATH\"\n")
    evidence["environment_script"] = str(environment)
    evidence["gpu"] = run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    evidence["host_compiler"] = run(["g++", "--version"])
    smoke = root / "smoke"
    smoke.mkdir(exist_ok=True)
    source = smoke / "compile.cu"
    source.write_text("#include <cuda_runtime.h>\n__global__ void copy_value(float* p) { p[threadIdx.x] = 1.0f; }\nint main() { return 0; }\n")
    evidence["compile_smoke"] = run([str(paths["nvcc"]), "-std=c++17", "-arch=sm_89",
        "--fmad=false", "-c", str(source), "-o", str(smoke / "compile.o")])
    evidence["link_smoke"] = run([str(paths["nvcc"]), "-std=c++17", "-arch=sm_89",
        "--fmad=false", str(source), "-o", str(smoke / "compile-linked")])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"tools": evidence["tools"], "compile_smoke": evidence["compile_smoke"],
                      "link_smoke": evidence["link_smoke"],
                      "environment_script": str(environment)}, indent=2))
    return int(any(v["returncode"] for v in evidence["tools"].values()) or
               evidence["compile_smoke"]["returncode"] != 0 or
               evidence["link_smoke"]["returncode"] != 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("fetch", "install", "profile-smoke"))
    parser.add_argument("--downloads", type=Path, default=Path("tmp/v2-tools"))
    parser.add_argument("--root", type=Path, default=Path("~/.cache/miniinfer-depth/v2-tools"))
    parser.add_argument("--report", type=Path, default=Path("docs/validation/v21_toolchain.json"))
    parser.add_argument("--components", nargs="+", choices=COMPONENTS, default=COMPONENTS,
                        help="Subset for independent installation; retain nvcc/cudart/cccl for install")
    parser.add_argument("--nsys-release", choices=NSYS_RELEASES, default="cli-2026.5.1",
                        help="Nsight Systems only; compiler/runtime remain CUDA 12.4.1")
    parser.add_argument("--nsys", type=Path, help="Explicit Nsight Systems executable for profile-smoke")
    parser.add_argument("--profile-output", type=Path, help="Exclusive output directory for profile-smoke")
    parser.add_argument("--baseline-sqlite", type=Path, help="Read-only original trace evidence")
    args = parser.parse_args()
    if args.action == "profile-smoke":
        if args.nsys is None or args.profile_output is None:
            parser.error("profile-smoke requires --nsys and --profile-output")
        return profile_smoke(args)
    if args.action == "install" and not {"cuda_nvcc", "cuda_cudart", "cuda_cccl"}.issubset(args.components):
        parser.error("install requires cuda_nvcc, cuda_cudart and cuda_cccl")
    args.downloads.mkdir(parents=True, exist_ok=True)
    manifest = metadata(args.downloads, args.action == "fetch")
    if args.action == "fetch":
        systems_manifest = nsys_metadata(args, True) if "nsight_systems" in args.components else manifest
        def download(name):
            selected = systems_manifest if name == "nsight_systems" else manifest
            item = selected[name]["linux-x86_64"]
            fetch_file(selected.get("_download_base", BASE) + item["relative_path"], args.downloads / Path(item["relative_path"]).name,
                       item["sha256"])
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(download, args.components))
        return 0
    return install(args, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
