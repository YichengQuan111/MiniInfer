"""Exercise benchmark child cleanup, the report contract and overwrite refusal."""
import json
import pathlib
import subprocess
import sys
import tempfile

binary_dir, reference = map(pathlib.Path, sys.argv[1:3])
script = pathlib.Path(__file__).resolve().parents[1] / "tools" / "run_tcp_baseline.py"
with tempfile.TemporaryDirectory(prefix="tcp-report-test-") as directory:
    output = pathlib.Path(directory) / "report.json"
    command = [sys.executable, str(script), "--bin-dir", str(binary_dir), "--reference", str(reference),
               "--output", str(output), "--iterations", "2", "--warmup", "1", "--repeats", "1"]
    run = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if run.returncode:
        raise AssertionError((run.stdout, run.stderr))
    saved = output.read_bytes()
    report = json.loads(saved)
    assert [r["batch"] for r in report["runs"]] == [1, 8, 32, 64, 256]
    assert all(r["runtime"]["result"]["successes"] == r["tcp"]["result"]["successes"] == 2 for r in report["runs"])
    assert all(r["tcp"]["result"]["latency_samples"] == 2 for r in report["runs"])
    again = subprocess.run(command, capture_output=True, text=True, timeout=5)
    assert again.returncode != 0 and "output already exists" in again.stderr
    assert output.read_bytes() == saved
print("TCP baseline driver: local/remote metrics, reference validation and overwrite protection passed")
