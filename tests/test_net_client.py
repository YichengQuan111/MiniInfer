"""Actual infer_client accounting against bounded, controlled TCP peers."""
import json
import math
import pathlib
import socket
import struct
import subprocess
import sys
import tempfile
import threading


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def exact(sock, count):
    result = bytearray()
    while len(result) < count:
        part = sock.recv(count - len(result))
        require(part, "client request ended mid-frame")
        result.extend(part)
    return bytes(result)


def frame(kind, request_id, body):
    return struct.pack("!4sHHII", b"MINI", 1, kind, request_id, len(body)) + body


def successful(request_id):
    # Batch 1: status, batch, ten binary32 logits, one uint32 label.
    return frame(0x8002, request_id, struct.pack("!II10fI", 0, 1, *([0.0] * 10), 0))


def reject_constant(value):
    raise AssertionError("non-finite JSON constant: " + value)


def finite_tree(value):
    if isinstance(value, float):
        require(math.isfinite(value), "non-finite metric")
    elif isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)


def metrics(result, expected, code):
    require(result.returncode == code,
            f"exit={result.returncode}, expected={code}, stderr={result.stderr}")
    document = json.loads(result.stdout, parse_constant=reject_constant)
    finite_tree(document)
    for field, value in expected.items():
        require(document[field] == value, f"{field}: {document[field]} != {value}: {document}")
    require(document["attempts"] == sum(document[field] for field in (
        "successes", "business_errors", "protocol_errors", "connection_errors", "timeouts")),
        "attempts must count each outcome exactly once")
    require(document["requests_planned"] == document["attempts"] + document["skipped"],
            "planned requests must equal attempts plus skipped")
    require(document["business_errors"] == sum(document[field] for field in (
        "bad_request", "busy", "shutting_down", "inference_failed")), "business status totals")
    require(document["requests_completed"] == document["successes"], "completed count")
    require(document["latency_samples"] == document["successes"], "only successes enter latency samples")
    require(document["warmup_attempts"] == document["warmup_successes"] + document["warmup_failures"],
            "warmup outcome totals")
    require(0 <= document["p50_ms"] <= document["p95_ms"] <= document["p99_ms"], "latency ordering")
    if not document["latency_samples"]:
        require(document["p99_ms"] == 0, "empty latency samples must report zero")
    return document


def command(binary_dir, input_file, port, requests, warmup, timeout_ms):
    return [str(binary_dir / "infer_client"), "--input", str(input_file), "--batch", "1",
            "--port", str(port), "--requests", str(requests), "--warmup", str(warmup),
            "--timeout-ms", str(timeout_ms)]


def exercise(binary_dir, input_file, name, actions, expected, requests=1, warmup=0,
             code=1, timeout_ms=1000):
    errors = []
    observed = []
    release = threading.Event()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.settimeout(3)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)  # Explicit readiness, before launching infer_client.
        port = listener.getsockname()[1]

        def serve():
            try:
                with listener.accept()[0] as peer:
                    peer.settimeout(3)
                    for action in actions:
                        header = struct.unpack("!4sHHII", exact(peer, 16))
                        magic, version, kind, request_id, count = header
                        require((magic, version, kind, count) == (b"MINI", 1, 2, 260), "MLP request header")
                        body = exact(peer, count)
                        require(struct.unpack("!I", body[:4])[0] == 1, "request batch")
                        require(request_id == len(observed) + 1, "request IDs must advance without replay")
                        observed.append(request_id)
                        if isinstance(action, int):
                            peer.sendall(frame(0x8002, request_id, struct.pack("!I", action)))
                        elif action == "ok":
                            peer.sendall(successful(request_id))
                        elif action == "wrong-id":
                            peer.sendall(successful(request_id + 1))
                        elif action == "wrong-type":
                            peer.sendall(frame(0x8001, request_id, struct.pack("!I", 0)))
                        elif action == "bad-length":
                            peer.sendall(struct.pack("!4sHHII", b"MINI", 1, 0x8002, request_id, 65541))
                        elif action == "nonfinite-logit":
                            body = struct.pack("!II10fI", 0, 1, float("nan"), *([0.0] * 9), 0)
                            peer.sendall(frame(0x8002, request_id, body))
                        elif action == "half-eof":
                            peer.sendall(successful(request_id)[:19])
                            return
                        elif action == "eof":
                            return
                        elif action == "timeout":
                            # Hold the peer open until client timeout completes;
                            # an Event prevents accidental EOF classification.
                            require(release.wait(5), "timeout peer was not released")
                            return
                        else:
                            raise AssertionError("unknown test action")
                    require(peer.recv(1) == b"", "unexpected extra request/retry after scenario")
            except BaseException as error:
                errors.append(error)

        worker = threading.Thread(target=serve, name="client-contract-peer")
        worker.start()
        try:
            result = subprocess.run(command(binary_dir, input_file, port, requests, warmup, timeout_ms),
                                    capture_output=True, text=True, timeout=5)
        finally:
            release.set()
            worker.join(timeout=6)
        require(not worker.is_alive(), "fake server failed to finish")
        if errors:
            raise errors[0]
        require(len(observed) == len(actions), f"{name}: wrong number of actual wire requests")
        metrics(result, expected, code)
    print(name + " passed")


def main(binary_dir, reference_dir):
    with tempfile.TemporaryDirectory(prefix="miniinfer-client-") as temporary:
        input_file = pathlib.Path(temporary) / "input.f32"
        with (reference_dir / "inputs.f32").open("rb") as source:
            first_row = source.read(64 * 4)
        require(len(first_row) == 256, "reference first input row missing")
        input_file.write_bytes(first_row)
        exercise(binary_dir, input_file, "success", ["ok"],
                 {"successes": 1, "attempts": 1, "skipped": 0, "status": 0}, code=0)
        for status, field in ((2, "busy"), (4, "inference_failed")):
            exercise(binary_dir, input_file, field + " then next request succeeds", [status, "ok"],
                     {field: 1, "business_errors": 1, "successes": 1, "latency_samples": 1,
                      "attempts": 2, "skipped": 0}, requests=2, code=3)
        for status, field in ((1, "bad_request"), (3, "shutting_down")):
            exercise(binary_dir, input_file, field + " stops remaining requests", [status],
                     {field: 1, "business_errors": 1, "successes": 0, "attempts": 1,
                      "skipped": 2}, requests=3, code=3)
        for action in ("wrong-id", "wrong-type", "bad-length", "nonfinite-logit", "half-eof"):
            exercise(binary_dir, input_file, action, [action],
                     {"protocol_errors": 1, "connection_errors": 0, "attempts": 1,
                      "successes": 0, "skipped": 1}, requests=2)
        exercise(binary_dir, input_file, "clean EOF", ["eof"],
                 {"connection_errors": 1, "protocol_errors": 0, "attempts": 1, "skipped": 1}, requests=2)
        exercise(binary_dir, input_file, "controlled read timeout", ["timeout"],
                 {"timeouts": 1, "connection_errors": 0, "attempts": 1, "skipped": 1},
                 requests=2, timeout_ms=100)
        exercise(binary_dir, input_file, "successful warmup excluded", ["ok"] * 4,
                 {"warmup_attempts": 2, "warmup_successes": 2, "warmup_failures": 0,
                  "attempts": 2, "successes": 2, "latency_samples": 2, "skipped": 0},
                 requests=2, warmup=2, code=0)
        exercise(binary_dir, input_file, "failed warmup prevents formal requests", [2],
                 {"warmup_attempts": 1, "warmup_successes": 0, "warmup_failures": 1,
                  "attempts": 0, "successes": 0, "business_errors": 0, "busy": 0,
                  "skipped": 2, "latency_samples": 0}, requests=2, warmup=2)
        # Reserved but non-listening port guarantees refusal without a freed-port race.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as reservation:
            reservation.settimeout(3)
            reservation.bind(("127.0.0.1", 0))
            result = subprocess.run(command(binary_dir, input_file, reservation.getsockname()[1], 2, 0, 1000),
                                    capture_output=True, text=True, timeout=5)
            metrics(result, {"connect_errors": 1, "connect_timeouts": 0, "attempts": 0,
                             "connection_errors": 0, "skipped": 2}, 1)
        print("connect failure accounting passed")
    print("test_net_client passed")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
