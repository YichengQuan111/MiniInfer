"""Real executable smoke tests; all children have deadlines and final cleanup."""
import pathlib
import selectors
import socket
import struct
import subprocess
import sys


def ready_line(process, timeout=5):
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        if not selector.select(timeout):
            raise AssertionError("server did not report readiness")
        line = process.stdout.readline().strip()
    if not line.startswith("Listening on "):
        raise AssertionError(f"unexpected readiness: {line!r}")
    return int(line.rsplit(":", 1)[1])


def cleanup(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    process.communicate(timeout=5)


def echo_smoke(binary_dir):
    process = subprocess.Popen(
        [str(binary_dir / "tcp_echo_server"), "--port", "0", "--connections", "3"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        port = ready_line(process)
        for message in ("hello MiniInfer", "", "TCP UTF-8 测试"):
            result = subprocess.run(
                [str(binary_dir / "tcp_echo_client"), "--port", str(port),
                 "--message", message], capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0 or result.stdout != message + "\n":
                raise AssertionError((result.returncode, result.stdout, result.stderr))
        stdout, stderr = process.communicate(timeout=5)
        if process.returncode != 0:
            raise AssertionError((process.returncode, stdout, stderr))
    finally:
        cleanup(process)
    print("Echo CLI: readiness, repeated connections, empty body and UTF-8 passed")


def exact(sock, size):
    result = b""
    while len(result) < size:
        part = sock.recv(size - len(result))
        if not part:
            raise AssertionError("truncated response")
        result += part
    return result


def frame(kind, request_id, body):
    return struct.pack("!4sHHII", b"MINI", 1, kind, request_id, len(body)) + body


def response(sock, kind, request_id):
    magic, version, actual_kind, actual_id, size = struct.unpack("!4sHHII", exact(sock, 16))
    if (magic, version, actual_kind, actual_id) != (b"MINI", 1, kind | 0x8000, request_id) or size > 65540:
        raise AssertionError("invalid response header")
    return exact(sock, size)


def echo_frames(binary_dir):
    process = subprocess.Popen(
        [str(binary_dir / "tcp_echo_server"), "--port", "0", "--connections", "7"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        port = ready_line(process)
        with socket.create_connection(("127.0.0.1", port), timeout=3) as sock:
            # Connectivity check only; deterministic chunk boundaries are tested with IoHooks.
            sock.sendall(frame(1, 1, b"abc") + frame(1, 2, b""))
            sock.shutdown(socket.SHUT_WR)
            assert response(sock, 1, 1) == b"\0\0\0\0abc"
            assert response(sock, 1, 2) == b"\0\0\0\0"
            assert sock.recv(1) == b""
        bad_headers = [
            struct.pack("!4sHHII", b"FAIL", 1, 1, 1, 0),
            struct.pack("!4sHHII", b"MINI", 2, 1, 1, 0),
            struct.pack("!4sHHII", b"MINI", 1, 3, 1, 0),
            struct.pack("!4sHHII", b"MINI", 1, 1, 1, 0xffffffff),
        ]
        for header in bad_headers:
            with socket.create_connection(("127.0.0.1", port), timeout=3) as sock:
                sock.sendall(header)
                assert sock.recv(1) == b""
        with socket.create_connection(("127.0.0.1", port), timeout=3) as sock:
            sock.sendall(frame(1, 1, b"x" * 1025))
            assert struct.unpack("!I", response(sock, 1, 1)[:4])[0] == 1
            assert sock.recv(1) == b""
        with socket.create_connection(("127.0.0.1", port), timeout=3) as sock:
            sock.sendall(frame(1, 1, b"a"))
            assert response(sock, 1, 1) == b"\0\0\0\0a"
            sock.sendall(frame(1, 1, b"b"))
            assert struct.unpack("!I", response(sock, 1, 1)[:4])[0] == 1
            assert sock.recv(1) == b""
        stdout, stderr = process.communicate(timeout=5)
        if process.returncode != 0:
            raise AssertionError((process.returncode, stdout, stderr))
    finally:
        cleanup(process)
    print("Echo frames: reuse, half-close, malformed headers, oversized bodies and repeated IDs passed")


if __name__ == "__main__":
    echo_smoke(pathlib.Path(sys.argv[1]))
    echo_frames(pathlib.Path(sys.argv[1]))
