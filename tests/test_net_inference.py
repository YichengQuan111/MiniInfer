"""D16.3: independent Python wire oracle and C++ executables on loopback TCP."""
import json
import math
import pathlib
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile

from test_net_cli import cleanup, frame, ready_line, response


def check_result(logits, labels, reference_logits, reference_labels, batch):
    assert len(logits) == batch * 10 and len(labels) == batch
    for i, value in enumerate(logits):
        expected = reference_logits[i % 640]
        assert math.isfinite(value) and abs(value - expected) <= 1e-4 + 1e-4 * abs(expected)
    assert list(labels) == [reference_labels[i % 64] for i in range(batch)]


def main(binary_dir, reference):
    inputs = (reference / "inputs.f32").read_bytes()
    logits = struct.unpack("<640f", (reference / "logits.f32").read_bytes())
    labels = struct.unpack("<64i", (reference / "labels.i32").read_bytes())
    with tempfile.TemporaryDirectory(prefix="miniinfer-net-") as temp:
        root = pathlib.Path(temp)
        model = root / "model"
        model.mkdir()
        for name in ("w1.f32", "b1.f32", "w2.f32", "b2.f32"):
            shutil.copyfile(reference / name, model / name)
        process = subprocess.Popen(
            [str(binary_dir / "infer_server"), "--model", str(model), "--port", "0", "--connections", "10"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            port = ready_line(process)
            # Proof of load-once: serving still works after the weight files disappear.
            for path in model.iterdir():
                path.unlink()
            for batch in (1, 8, 32, 64, 256):
                path = root / f"input_{batch}.f32"
                path.write_bytes((inputs * 4)[:batch * 256])
                run = subprocess.run(
                    [str(binary_dir / "infer_client"), "--port", str(port), "--input", str(path),
                     "--batch", str(batch), "--requests", "3"],
                    capture_output=True, text=True, timeout=10,
                )
                assert run.returncode == 0, run.stderr
                result = json.loads(run.stdout)
                assert result["status"] == 0 and result["requests_completed"] == 3
                check_result(result["logits"], result["labels"], logits, labels, batch)
            first = struct.unpack("<64f", inputs[:256])
            good = struct.pack("!I64f", 1, *first)
            with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
                overflow = struct.pack("!I", 1) + struct.pack("!I", 0x7f7fffff) * 64
                sock.sendall(frame(2, 1, overflow))
                assert struct.unpack("!I", response(sock, 2, 1)[:4])[0] == 4
                sock.sendall(frame(2, 2, good))
                sock.shutdown(socket.SHUT_WR)
                body = response(sock, 2, 2)
                assert struct.unpack("!II", body[:8]) == (0, 1)
                check_result(struct.unpack("!10f", body[8:48]), struct.unpack("!I", body[48:]), logits, labels, 1)
                assert sock.recv(1) == b""
            with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
                invalid = good[:4] + struct.pack("!I", 0x7fc00000) + good[8:]
                sock.sendall(frame(2, 4, invalid))
                assert struct.unpack("!I", response(sock, 2, 4)[:4])[0] == 1
                assert sock.recv(1) == b""
            with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
                sock.sendall(frame(1, 1, b"echo"))
                assert response(sock, 1, 1) == b"\0\0\0\0echo"
                sock.sendall(frame(1, 1, b"again"))
                assert struct.unpack("!I", response(sock, 1, 1)[:4])[0] == 1
                assert sock.recv(1) == b""
            for malformed in (frame(0x8002, 1, b""), struct.pack("!4sHHII", b"MINI", 1, 2, 1, 0xffffffff)):
                with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
                    sock.sendall(malformed)
                    assert sock.recv(1) == b""
            out, err = process.communicate(timeout=5)
            assert process.returncode == 0, (out, err)
        finally:
            cleanup(process)
    print("Inference TCP: batches 1/8/32/64/256 match reference; reuse, load-once, errors, recovery and half-close passed")


def signal_shutdown(binary_dir, reference):
    for signal_number, partial in ((signal.SIGINT, False), (signal.SIGTERM, True)):
        server = subprocess.Popen(
            [str(binary_dir / "infer_server"), "--model", str(reference), "--port", "0", "--read-timeout-ms", "600000"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            port = ready_line(server)
            with socket.create_connection(("127.0.0.1", port), timeout=3) as sock:
                sock.sendall(frame(1, 1, b"ready"))
                assert response(sock, 1, 1) == b"\0\0\0\0ready"
                if partial:
                    sock.sendall(frame(2, 2, b"x")[:7])
                server.send_signal(signal_number)
                out, err = server.communicate(timeout=5)
                assert server.returncode == 0, (server.returncode, out, err)
        finally:
            cleanup(server)
    print("SIGINT/SIGTERM: active idle and partial-frame connections stop before long read deadlines")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
    signal_shutdown(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
