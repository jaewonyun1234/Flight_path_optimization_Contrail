"""
progress.py — ZMQ pub/sub helpers for streaming solver progress.

The solver service publishes one small message per improved CP-SAT incumbent;
the GUI subscribes and live-plots the convergence curve. This is the
event-driven-messaging layer that decouples "the solve is making progress"
from "something is watching" — the publisher does not care whether anyone is
listening, and subscribers can come and go.

WIRE FORMAT
===========
Each message is a two-frame ZMQ multipart:

    frame 0 : topic   (UTF-8 bytes; subscribers filter on a prefix of this)
    frame 1 : payload (UTF-8 JSON: {"improvement": int, "objective": float})

Two frames (rather than "topic payload" in one string) means a topic
containing spaces or JSON-like characters can never corrupt parsing.

SLOW-JOINER CAVEAT
==================
ZMQ PUB/SUB drops messages sent before a subscriber has finished connecting
and subscribing. Subscribe BEFORE triggering the solve, and tolerate the
occasional missed early frame — progress is advisory, not a source of truth
(the final SolveResponse is authoritative).
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator

import zmq

DEFAULT_PUB_ADDRESS = "tcp://*:5556"
DEFAULT_SUB_ADDRESS = "tcp://localhost:5556"


# =============================================================================
# PUBLISHER
# =============================================================================

class ProgressPublisher:
    """Binds a ZMQ PUB socket and publishes (topic, improvement, objective).

    One publisher is bound per running server and shared across solves;
    `publish` is guarded by a lock because the CP-SAT callback fires from a
    worker thread and pyzmq sockets are not thread-safe.
    """

    def __init__(self, address: str = DEFAULT_PUB_ADDRESS) -> None:
        self.address = address
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.PUB)
        self._lock = threading.Lock()
        self._bound = False

    def bind(self) -> ProgressPublisher:
        """Bind the socket. Returns self so callers can do `Publisher(addr).bind()`."""
        self._socket.bind(self.address)
        self._bound = True
        return self

    def publish(self, topic: str, improvement: int, objective: float) -> None:
        """Send one progress event on `topic`."""
        payload = json.dumps({"improvement": int(improvement), "objective": float(objective)})
        with self._lock:
            self._socket.send_multipart([topic.encode("utf-8"), payload.encode("utf-8")])

    def close(self) -> None:
        with self._lock:
            self._socket.close(0)


# =============================================================================
# SUBSCRIBER
# =============================================================================

def subscribe(
    topic: str,
    address: str = DEFAULT_SUB_ADDRESS,
    *,
    poll_timeout_ms: int = 200,
    stop: Callable[[], bool] | None = None,
) -> Iterator[tuple[int, float]]:
    """Yield decoded (improvement, objective) tuples published on `topic`.

    Used by the GUI worker thread. The generator blocks on a poller with a
    short timeout; when idle it checks `stop()` (if given) and returns once
    that becomes true. With no `stop`, it streams forever until the caller
    closes the generator.
    """
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.connect(address)
    sock.setsockopt_string(zmq.SUBSCRIBE, topic)

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)
    try:
        while True:
            events = dict(poller.poll(poll_timeout_ms))
            if sock in events:
                frames = sock.recv_multipart()
                data = json.loads(frames[1].decode("utf-8"))
                yield int(data["improvement"]), float(data["objective"])
            elif stop is not None and stop():
                return
    finally:
        poller.unregister(sock)
        sock.close(0)
