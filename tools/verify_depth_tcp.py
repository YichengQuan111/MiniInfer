#!/usr/bin/env python3
"""Verify real depth TCP apps against local SDK output, then verify SIGTERM drain.

Linux/WSL only. Each --package uses its golden/original.ppm unless --image is
specified. Comparisons are exact FP32 bytes from the same backend/package;
Python-golden numerical accuracy is covered separately by verify_depth_sdk.py.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import struct
import subprocess
import sys
import time


def positive_seconds(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("timeout must be positive and finite")
    return result


def positive_count(value):
    result = int(value)
    if not 1 <= result <= 100:
        raise argparse.ArgumentTypeError("request count must be in [1,100]")
    return result


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def events_from_bytes(data):
    events = []
    for line in data.decode("utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and "event" in event:
            events.append(event)
    return events


def wait_ready(process, timeout, stdout_log):
    """Read bytes through pollable pipes so a missing newline cannot hang us."""
    deadline = time.monotonic() + timeout
    captured = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            for key, _ in selector.select(max(0, remaining)):
                chunk = os.read(key.fd, 65536)
                if not chunk:
                    raise RuntimeError("depth_server exited before ready JSON")
                stdout_log.write(chunk)
                stdout_log.flush()
                captured.extend(chunk)
                if len(captured) > 1024 * 1024:
                    raise RuntimeError("depth_server startup stdout exceeds 1 MiB")
                # Only parse complete lines: a short read may split its JSON.
                complete = bytes(captured[:captured.rfind(b"\n") + 1])
                for event in events_from_bytes(complete):
                    if event.get("event") == "ready":
                        port = event.get("port")
                        if type(port) is not int or not 1 <= port <= 65535:
                            raise RuntimeError("invalid depth_server ready port")
                        if event.get("pid") != process.pid:
                            raise RuntimeError("depth_server ready PID mismatch")
                        return event, bytes(captured)
            if process.poll() is not None:
                raise RuntimeError("depth_server exited before ready JSON")
    raise TimeoutError("depth_server ready timeout")


def run_command(command, output_dir, stem, timeout, expected_success=True):
    stdout_path, stderr_path = output_dir / (stem + ".stdout.log"), output_dir / (stem + ".stderr.log")
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        # subprocess.run kills and reaps this child if its timeout expires.
        result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=timeout, check=False)
    record = {"command": command, "returncode": result.returncode,
              "stdout": str(stdout_path), "stderr": str(stderr_path)}
    write_json(output_dir / (stem + ".command.json"), record)
    if (result.returncode == 0) != expected_success:
        raise RuntimeError(stem + " returned unexpected status " + str(result.returncode))
    return record


def image_pixels(path):
    # Read only four P6 header tokens. Raster data must never be treated as
    # whitespace/comment syntax. The owning C++ reader validates the full file.
    with path.open("rb") as stream:
        tokens = []
        while len(tokens) < 4:
            token = bytearray()
            while True:
                byte = stream.read(1)
                if not byte:
                    raise ValueError("incomplete PPM header: " + str(path))
                if byte == b"#":
                    stream.readline()
                    continue
                if byte not in b" \t\r\n\v\f":
                    token.extend(byte)
                    break
            while True:
                byte = stream.read(1)
                if not byte or byte in b" \t\r\n\v\f":
                    break
                token.extend(byte)
                if len(token) > 128:
                    raise ValueError("invalid PPM header token")
            tokens.append(bytes(token))
    if tokens[0] != b"P6" or tokens[3] != b"255":
        raise ValueError("expected P6 RGB8 image: " + str(path))
    width, height = int(tokens[1]), int(tokens[2])
    if not 1 <= width <= 4096 or not 1 <= height <= 4096 or width * height > 2097152:
        raise ValueError("image dimensions outside depth contract")
    return width, height


def fp32_bytes(path, pixels):
    data = path.read_bytes()
    if len(data) != 4 * pixels:
        raise ValueError("FP32 element count mismatch: " + str(path))
    if not all(math.isfinite(value) for (value,) in struct.iter_unpack("<f", data)):
        raise ValueError("non-finite FP32 output: " + str(path))
    return data


def require_count(event, name, expected):
    actual = event.get(name)
    if type(actual) is not int or actual != expected:
        raise RuntimeError(f"stopped {name} expected {expected}, got {actual!r}")


def verify_stopped(event, requests):
    for name in ("requests", "responses", "accepted", "completed", "sdk_accepted", "sdk_succeeded", "sdk_delivered"):
        require_count(event, name, requests)
    for name in ("active", "pending", "rejected", "expired", "discarded", "bad_requests", "protocol_errors",
                 "session_failures", "accept_errors", "busy", "inference_failures", "io_failures",
                 "network_bytes", "network_inflight", "sdk_bytes", "sdk_inflight", "sdk_queued", "sdk_executing",
                 "sdk_failed", "sdk_rejected_busy", "sdk_rejected_closed", "sdk_rejected_invalid"):
        require_count(event, name, 0)
    if event.get("sdk_accepting") is not False or event.get("sdk_shutdown_complete") is not True:
        raise RuntimeError("SDK did not reach completed shutdown state")
    peak = event.get("network_peak_bytes")
    if type(peak) is not int or not 0 < peak <= 128 * 1024 * 1024:
        raise RuntimeError("invalid network reservation peak")


def verify_package(args, root, directory, executables):
    directory.mkdir(exist_ok=False)
    images = [Path(name).resolve(strict=True) for name in args.image] if args.image else [root / "golden/original.ppm"]
    process = None
    record = {"package": str(root), "passed": False, "samples": [], "errors": []}
    stdout_path, stderr_path = directory / "server.stdout.log", directory / "server.stderr.log"
    command = [str(executables["server"]), "--package", str(root), "--host", "127.0.0.1", "--port", "0",
               "--read-timeout-ms", str(args.client_timeout_ms), "--write-timeout-ms", str(args.client_timeout_ms)]
    record["server_command"] = command
    captured = b""
    reaped_output = False
    try:
        with stdout_path.open("xb") as stdout_log, stderr_path.open("xb") as stderr_log:
            try:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=stderr_log)
                ready, captured = wait_ready(process, args.startup_timeout, stdout_log)
                record["ready"] = ready
                package = json.loads((root / "package.json").read_text(encoding="utf-8"))
                if ready.get("backend") != package.get("backend") or ready.get("precision") != package.get("precision"):
                    raise RuntimeError("ready backend/precision differs from package")
                requests = 0
                for index, image in enumerate(images):
                    image = image.resolve(strict=True)
                    width, height = image_pixels(image)
                    sample_dir = directory / (f"image-{index:03d}")
                    sample_dir.mkdir(exist_ok=False)
                    local = sample_dir / "sdk.f32"
                    run_command([str(executables["sdk"]), str(root), str(image), str(local)],
                                sample_dir, "sdk", args.command_timeout)
                    baseline = fp32_bytes(local, width * height)
                    sample = {"image": str(image), "width": width, "height": height,
                              "sdk_sha256": hashlib.sha256(baseline).hexdigest(), "requests": []}
                    record["samples"].append(sample)
                    for repeat in range(args.requests_per_image):
                        target = sample_dir / (f"tcp-{repeat:03d}.f32")
                        client_command = [str(executables["client"]), "--image", str(image), "--output", str(target),
                                          "--host", "127.0.0.1", "--port", str(ready["port"]),
                                          "--timeout-ms", str(args.client_timeout_ms)]
                        run_command(client_command, sample_dir, f"tcp-{repeat:03d}",
                                    args.client_timeout_ms / 1000 + 5)
                        candidate = fp32_bytes(target, width * height)
                        exact = candidate == baseline
                        sample["requests"].append({"output": str(target), "sha256": hashlib.sha256(candidate).hexdigest(),
                                                   "exact_fp32": exact})
                        if not exact:
                            raise RuntimeError("TCP result differs from local SDK bytes")
                        requests += 1
                record["requests"] = requests
                if process.poll() is not None:
                    raise RuntimeError("depth_server stopped before SIGTERM")
                process.send_signal(signal.SIGTERM)
                tail, _ = process.communicate(timeout=args.shutdown_timeout)
                reaped_output = True
                stdout_log.write(tail); stdout_log.flush()
                captured += tail
                record["server_returncode"] = process.returncode
                stopped = [event for event in events_from_bytes(captured) if event.get("event") == "stopped"]
                if process.returncode != 0 or len(stopped) != 1:
                    raise RuntimeError("server must exit successfully with one stopped JSON after SIGTERM")
                record["stopped"] = stopped[0]
                verify_stopped(stopped[0], requests)
                record["passed"] = True
            finally:
                if process is not None and not reaped_output:
                    if process.poll() is None:
                        # On any failure, force cleanup. Success always uses
                        # SIGTERM and the stricter shutdown validation above.
                        process.kill()
                    tail, _ = process.communicate(timeout=5)
                    stdout_log.write(tail)
                if process is not None:
                    record.setdefault("server_returncode", process.returncode)
    except Exception as error:
        record["errors"].append(str(error))
    record.update(stdout=str(stdout_path), stderr=str(stderr_path))
    write_json(directory / "report.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", required=True, help="directory containing depth_server, depth_client and depth_sdk_cli")
    parser.add_argument("--package", action="append", required=True, help="SDK model package; repeat for multiple backends")
    parser.add_argument("--image", action="append", default=[], help="optional P6 image; repeat to test each image on every package")
    parser.add_argument("--output", required=True, help="new output directory; existing paths are rejected")
    parser.add_argument("--requests-per-image", type=positive_count, default=2)
    parser.add_argument("--startup-timeout", type=positive_seconds, default=180, help="server ready timeout in seconds")
    parser.add_argument("--command-timeout", type=positive_seconds, default=180, help="local SDK CLI timeout in seconds")
    parser.add_argument("--client-timeout-ms", type=int, default=180000, help="client connect/send/receive timeout, 1..600000 ms")
    parser.add_argument("--shutdown-timeout", type=positive_seconds, default=30, help="SIGTERM drain and process exit timeout in seconds")
    args = parser.parse_args()
    if not 1 <= args.client_timeout_ms <= 600000:
        parser.error("--client-timeout-ms must be in [1,600000]")
    output = Path(args.output).resolve()
    owns_output = False
    report = {"schema_version": 1, "passed": False, "comparison": "same-package exact little-endian FP32 bytes",
              "packages": [], "errors": []}
    try:
        if os.name != "posix" or not hasattr(signal, "SIGTERM"):
            raise ValueError("run this verifier in Linux/WSL")
        output.mkdir(parents=True, exist_ok=False)
        owns_output = True
        bin_dir = Path(args.bin_dir).resolve(strict=True)
        executables = {"server": bin_dir / "depth_server", "client": bin_dir / "depth_client", "sdk": bin_dir / "depth_sdk_cli"}
        for executable in executables.values():
            if not executable.is_file() or not os.access(executable, os.X_OK):
                raise ValueError("missing/non-executable app: " + str(executable))
        packages = [Path(name).resolve(strict=True) for name in args.package]
        if len(set(packages)) != len(packages):
            raise ValueError("duplicate model package")
        # Invalid port is rejected during option parsing, before model loading.
        invalid = run_command([str(executables["server"]), "--package", str(packages[0]), "--port", "65536"],
                              output, "invalid-server-port", args.command_timeout, expected_success=False)
        if events_from_bytes(Path(invalid["stdout"]).read_bytes()) or "invalid numeric option" not in Path(invalid["stderr"]).read_text(encoding="utf-8"):
            raise RuntimeError("invalid server CLI was not rejected before startup")
        report["invalid_server_cli"] = invalid
        for index, package in enumerate(packages):
            record = verify_package(args, package, output / (f"package-{index:03d}"), executables)
            report["packages"].append(record)
            report["errors"].extend(f"package[{index}]: {error}" for error in record["errors"])
        report["passed"] = bool(report["packages"]) and all(item["passed"] for item in report["packages"])
    except Exception as error:
        report["errors"].append(str(error))
    report_path = output / "report.json"
    # Never overwrite a pre-existing output directory or a previous report.
    try:
        if owns_output and not report_path.exists():
            write_json(report_path, report)
        else:
            report_path = None
    except Exception as error:
        report["errors"].append("report write failed: " + str(error))
        report["passed"] = False
        report_path = None
    print(json.dumps({"passed": report["passed"], "report": str(report_path) if report_path else None,
                      "errors": report["errors"]}, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
