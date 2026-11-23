"""Shared helpers: start a real Lantern process wired to the fake llama-server."""
import http.client
import json
import os
import socket
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FAKE = Path(__file__).resolve().parent / "fake_llama_server.py"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_wrapper(directory):
    """An executable that runs the fake server with the same arguments Lantern passes."""
    if os.name == "nt":
        path = Path(directory) / "llama-server.bat"
        path.write_text('@echo off\r\n"%s" "%s" %%*\r\n' % (sys.executable, FAKE))
    else:
        path = Path(directory) / "llama-server"
        path.write_text('#!/bin/sh\nexec "%s" "%s" "$@"\n' % (sys.executable, FAKE))
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


class LanternProcess:
    """Runs `python app.py` against a throwaway lab folder."""

    def __init__(self, models=("test-model.gguf",)):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.home = base / "lab"
        (self.home / "models").mkdir(parents=True)
        for name in models:
            path = self.home / "models" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\0" * 1024)
        self.ui_port = free_port()
        self.llama_port = free_port()
        env = dict(os.environ,
                   LANTERN_HOME=str(self.home),
                   LANTERN_PORT=str(self.ui_port),
                   LANTERN_LLAMA_PORT=str(self.llama_port),
                   LANTERN_LLAMA_SERVER=str(make_wrapper(base)))
        self.proc = subprocess.Popen([sys.executable, str(ROOT / "app.py")], env=env,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._wait_up()

    @property
    def base(self):
        return "http://127.0.0.1:%d" % self.ui_port

    def _wait_up(self):
        for _ in range(100):
            try:
                self.request("GET", "/api/status")
                return
            except OSError:
                time.sleep(0.1)
        self.close()
        raise RuntimeError("Lantern did not start")

    def request(self, method, path, body=None, headers=None, port=None):
        conn = http.client.HTTPConnection("127.0.0.1", port or self.ui_port, timeout=15)
        h = dict(headers or {})
        data = None
        if body is not None:
            data = body if isinstance(body, (bytes, str)) else json.dumps(body)
            h.setdefault("Content-Type", "application/json")
        conn.request(method, path, data, h)
        resp = conn.getresponse()
        raw = resp.read()
        conn.close()
        try:
            return resp.status, json.loads(raw)
        except ValueError:
            return resp.status, raw

    def wait_state(self, want, timeout=20):
        end = time.time() + timeout
        status = {}
        while time.time() < end:
            _, status = self.request("GET", "/api/status")
            if status.get("state") == want:
                return status
            time.sleep(0.15)

        log_path = self.home / "llama-server.log"
        try:
            log_tail = "\\n".join(log_path.read_text(errors="replace").splitlines()[-30:])
        except OSError:
            log_tail = "<log unavailable>"
        raise AssertionError(
            "never reached state %r, last status: %r, server pid: %r, "
            "server return code: %r, server log:\\n%s"
            % (want, status, self.proc.pid, self.proc.poll(), log_tail)
        )

    def load(self, model="test-model.gguf", **kw):
        body = {"model": model, "mode": "cpu", "ctx": 4096, "threads": 2}
        body.update(kw)
        return self.request("POST", "/api/load", body)

    def close(self):
        try:
            self.request("POST", "/api/stop", {})
        except OSError:
            pass
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.tmp.cleanup()
