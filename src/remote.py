"""Remote control server and webhook utilities for the signal runner.

Provides:
- ``serve()``: lightweight HTTP server that accepts remote trigger requests.
- ``post_webhook()``: POST a signal payload to a remote URL.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080

# Shared in-process cache for the most-recently generated signal payload.
_latest_signal: dict[str, Any] | None = None
_signal_lock = threading.Lock()


def set_latest_signal(payload: dict[str, Any]) -> None:
    """Store a copy of *payload* in the in-process cache."""
    global _latest_signal
    with _signal_lock:
        _latest_signal = dict(payload)


def get_latest_signal() -> dict[str, Any] | None:
    """Return the most-recently generated signal payload, or ``None``."""
    with _signal_lock:
        return dict(_latest_signal) if _latest_signal is not None else None


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class _Handler(BaseHTTPRequestHandler):
    """Minimal HTTP handler exposing three routes:

    ``GET  /health``        – liveness probe.
    ``GET  /signal/latest`` – return cached signal JSON.
    ``POST /signal``        – trigger signal generation (delegated via callback).
    """

    # Injected by ``serve()`` before the server starts.
    trigger_callback: Any = None  # Callable[[dict], dict[str, Any]]

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D102
        # Suppress default noisy access log; callers can add their own logging.
        pass

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_error_json(self, status: int, message: str) -> None:
        self._send_json(status, {"error": message})

    def _read_body_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        elif self.path in ("/signal/latest", "/signal/latest/"):
            cached = get_latest_signal()
            if cached is None:
                self._send_error_json(404, "No signal generated yet.")
            else:
                self._send_json(200, cached)
        else:
            self._send_error_json(404, f"Unknown route: {self.path}")

    def do_POST(self) -> None:  # noqa: N802
        if self.path in ("/signal", "/signal/"):
            options = self._read_body_json()
            cb = self.__class__.trigger_callback
            if cb is None:
                self._send_error_json(503, "No trigger callback registered.")
                return
            try:
                result = cb(options)
                set_latest_signal(result)
                self._send_json(200, result)
            except Exception as exc:  # pragma: no cover – server must not crash
                self._send_error_json(500, str(exc))
        else:
            self._send_error_json(404, f"Unknown route: {self.path}")


def serve(
    trigger_callback: Any,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    output_path: Path | None = None,
) -> None:
    """Start a blocking HTTP server for remote control.

    Parameters
    ----------
    trigger_callback:
        A callable ``(options: dict) -> dict`` that generates and returns a
        signal payload.  It will be called each time ``POST /signal`` is
        received.  The *options* dict may contain any of the keys accepted by
        ``generate_latest_signal`` (``threshold``, ``size_pct``, etc.) and may
        be empty.
    host:
        Interface to bind (default ``"0.0.0.0"``).
    port:
        TCP port to listen on (default ``8080``).
    output_path:
        If given, the server prints its listen address and this path on startup.
    """
    _Handler.trigger_callback = trigger_callback

    httpd = HTTPServer((host, port), _Handler)
    print(f"[remote] Listening on http://{host}:{port}")
    if output_path is not None:
        print(f"[remote] Signal output: {output_path}")
    print("[remote] Routes: GET /health  GET /signal/latest  POST /signal")
    print("[remote] Press Ctrl-C to stop.")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[remote] Shutting down.")
    finally:
        httpd.server_close()


# ---------------------------------------------------------------------------
# Webhook
# ---------------------------------------------------------------------------

def post_webhook(
    payload: dict[str, Any],
    url: str,
    timeout: int = 10,
) -> dict[str, Any]:
    """POST *payload* as JSON to *url*.

    Returns the parsed JSON response body (or ``{"status": "sent"}`` for
    non-JSON 2xx replies).  Raises ``RuntimeError`` on HTTP errors.
    """
    encoded = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return {"status": "sent", "http_status": resp.status}
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Webhook POST to {url!r} failed: HTTP {exc.code} {exc.reason}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Webhook POST to {url!r} failed: {exc.reason}"
        ) from exc
