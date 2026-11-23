#!/usr/bin/env python3
"""Lantern: a tiny, dependency-free web UI for llama.cpp's llama-server.

Serves the browser UI, discovers GGUF models, starts/stops llama-server,
and proxies chat requests (streaming) to it. Standard library only.
"""
import argparse
import atexit
import http.client
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

VERSION = "1.0.0"
HERE = Path(__file__).resolve().parent
WEB_DIR = HERE / "web"

def default_home():
    """Use a sibling llama.cpp/models workspace when Lantern is cloned into one."""
    override = os.environ.get("LANTERN_HOME")
    if override:
        return Path(override).expanduser()

    workspace = HERE.parent
    if (workspace / "llama.cpp").is_dir() and (workspace / "models").is_dir():
        return workspace

    return Path.home() / "lantern-lab"


HOME = default_home()
MODELS_DIR = Path(os.environ.get("LANTERN_MODELS", HOME / "models")).expanduser()
RUNTIME_DIR = HOME / "llama.cpp"
LOG_FILE = Path(os.environ.get("LANTERN_LOG_FILE", HOME / "lantern-server.log")).expanduser()
UI_PORT = int(os.environ.get("LANTERN_PORT", "8080"))
LLAMA_PORT = int(os.environ.get("LANTERN_LLAMA_PORT", str(UI_PORT + 1)))
EXE_NAME = "llama-server.exe" if os.name == "nt" else "llama-server"
MAX_BODY = 32 * 1024 * 1024


class ApiError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


# --------------------------------------------------------------------------
# Discovery helpers
# --------------------------------------------------------------------------

def find_server():
    """Locate llama-server: env override, then the lab folder, then PATH."""
    override = os.environ.get("LANTERN_LLAMA_SERVER")
    if override and Path(override).is_file():
        return Path(override)
    if RUNTIME_DIR.is_dir():
        for hit in sorted(RUNTIME_DIR.rglob(EXE_NAME)):
            if hit.is_file():
                return hit
    found = shutil.which(EXE_NAME)
    return Path(found) if found else None


def _is_secondary_shard(name):
    # Split models look like model-00002-of-00005.gguf; only load shard 1.
    stem = name[:-5].lower()
    if "-of-" not in stem:
        return False
    head, _, total = stem.rpartition("-of-")
    idx = head.rsplit("-", 1)[-1]
    return total.isdigit() and idx.isdigit() and int(idx) != 1


def list_models():
    models = []
    if MODELS_DIR.is_dir():
        for p in sorted(MODELS_DIR.rglob("*.gguf")):
            if not p.is_file() or "mmproj" in p.name.lower() or _is_secondary_shard(p.name):
                continue
            models.append({
                "name": p.relative_to(MODELS_DIR).as_posix(),
                "size": p.stat().st_size,
            })
    return models


def resolve_model(name):
    # Only accept names that appear in the scan, which rules out path traversal.
    for m in list_models():
        if m["name"] == name:
            return MODELS_DIR / m["name"]
    raise ApiError(404, f"Model not found: {name!r}. Click Refresh after adding files.")


def tail(path, n=40):
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 16384))
            data = fh.read().decode("utf-8", "replace")
        return "\n".join(data.splitlines()[-n:])
    except OSError:
        return ""


# --------------------------------------------------------------------------
# Engine: owns the llama-server child process
# --------------------------------------------------------------------------

class Engine:
    def __init__(self):
        self.lock = threading.Lock()
        self.proc = None
        self.logfh = None
        self.info = {}
        self.last_exit = None

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def _health(self):
        conn = http.client.HTTPConnection("127.0.0.1", LLAMA_PORT, timeout=1)
        try:
            conn.request("GET", "/health")
            response = conn.getresponse()
            response.read()
            return response.status == 200
        except (OSError, http.client.HTTPException):
            return False
        finally:
            conn.close()

    def ready(self):
        return self.running() and self._health()

    def state(self):
        if self.proc is None:
            return "stopped"
        if self.proc.poll() is not None:
            return "crashed" if self.last_exit not in (0, None) else "stopped"
        return "ready" if self._health() else "loading"

    def start(self, model, mode, ctx, threads):
        exe = find_server()
        if exe is None:
            raise ApiError(
                400,
                f"{EXE_NAME} not found. Put a llama.cpp build under {RUNTIME_DIR} "
                "or add it to PATH.",
            )
        path = resolve_model(model)
        if mode not in ("cpu", "gpu"):
            raise ApiError(400, "mode must be 'cpu' or 'gpu'")
        try:
            ctx, threads = int(ctx), int(threads)
        except (TypeError, ValueError):
            raise ApiError(400, "ctx and threads must be integers")
        if not 256 <= ctx <= 1_048_576:
            raise ApiError(400, "ctx must be between 256 and 1048576")
        if not 1 <= threads <= 512:
            raise ApiError(400, "threads must be between 1 and 512")

        with self.lock:
            self._stop_locked()
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            self.logfh = open(LOG_FILE, "wb")
            args = [
                str(exe), "-m", str(path),
                "--host", "127.0.0.1", "--port", str(LLAMA_PORT),
                "-c", str(ctx), "-t", str(threads),
                "-ngl", "99" if mode == "gpu" else "0",
                "--jinja",
            ]
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            try:
                self.proc = subprocess.Popen(
                    args, stdout=self.logfh, stderr=subprocess.STDOUT,
                    cwd=str(exe.parent), creationflags=flags,
                )
            except OSError as exc:
                raise ApiError(500, f"Could not start {exe}: {exc}")
            self.last_exit = None
            self.info = {"model": model, "mode": mode, "ctx": ctx, "threads": threads}

    def _stop_locked(self):
        if self.proc is not None and self.proc.poll() is None:
            if os.name == "nt":
                # Popen may own a cmd.exe wrapper (e.g. a .bat launcher) whose
                # Python/server child inherits the log handle. Kill the tree.
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(self.proc.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        check=False,
                        timeout=10,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    self.proc.kill()
            else:
                self.proc.terminate()

            try:
                self.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        if self.proc is not None:
            self.last_exit = 0  # deliberate stop is not a crash
        self.proc = None
        self.info = {}
        if self.logfh is not None:
            self.logfh.close()
            self.logfh = None

    def stop(self):
        with self.lock:
            self._stop_locked()

    def snapshot(self):
        if self.proc is not None and self.proc.poll() is not None:
            self.last_exit = self.proc.returncode
        state = self.state()
        out = {
            "state": state,
            "cpu_count": os.cpu_count() or 4,
            "server_found": find_server() is not None,
            "server_pid": self.proc.pid if self.proc is not None else None,
            "models_dir": str(MODELS_DIR),
            "runtime_dir": str(RUNTIME_DIR),
            "version": VERSION,
            **self.info,
        }
        if state == "crashed":
            out["exit_code"] = self.last_exit
            out["log_tail"] = tail(LOG_FILE)
        return out


engine = Engine()


# --------------------------------------------------------------------------
# HTTP handler
# --------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "Lantern/" + VERSION
    protocol_version = "HTTP/1.0"

    def log_message(self, *args):
        pass

    # -- security: localhost only, same-origin only (blocks DNS rebinding/CSRF)
    def _allowed_hosts(self):
        return {f"127.0.0.1:{UI_PORT}", f"localhost:{UI_PORT}"}

    def _guard(self):
        if self.headers.get("Host", "") not in self._allowed_hosts():
            self._json(403, {"error": "Forbidden host"})
            return False
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc not in self._allowed_hosts():
            self._json(403, {"error": "Forbidden origin"})
            return False
        return True

    # -- helpers
    def _json(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ApiError(400, "Bad Content-Length")
        if length > MAX_BODY:
            raise ApiError(413, "Request too large")
        return self.rfile.read(length) if length else b""

    def _read_json(self):
        raw = self._read_body()
        try:
            data = json.loads(raw or b"{}")
        except ValueError:
            raise ApiError(400, "Invalid JSON")
        if not isinstance(data, dict):
            raise ApiError(400, "Expected a JSON object")
        return data

    # -- routes
    def do_GET(self):
        if not self._guard():
            return
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                page = (WEB_DIR / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(page)
            elif path == "/api/status":
                self._json(200, engine.snapshot())
            elif path == "/api/models":
                self._json(200, {"models": list_models(), "models_dir": str(MODELS_DIR)})
            elif path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            else:
                self._json(404, {"error": "Not found"})
        except FileNotFoundError:
            self._json(500, {"error": "web/index.html is missing"})
        except ApiError as exc:
            self._json(exc.status, {"error": exc.message})

    def do_POST(self):
        if not self._guard():
            return
        path = urlparse(self.path).path
        try:
            if path == "/api/load":
                data = self._read_json()
                engine.start(
                    data.get("model"), data.get("mode", "cpu"),
                    data.get("ctx", 4096), data.get("threads", os.cpu_count() or 4),
                )
                self._json(200, engine.snapshot())
            elif path == "/api/stop":
                engine.stop()
                self._json(200, engine.snapshot())
            elif path == "/v1/chat/completions":
                self._proxy_chat(self._read_body())
            else:
                self._json(404, {"error": "Not found"})
        except ApiError as exc:
            self._json(exc.status, {"error": exc.message})

    def _proxy_chat(self, body):
        if not engine.ready():
            raise ApiError(503, "No model is loaded yet.")
        conn = http.client.HTTPConnection("127.0.0.1", LLAMA_PORT, timeout=900)
        sent = False
        try:
            conn.request("POST", "/v1/chat/completions", body,
                         {"Content-Type": "application/json"})
            resp = conn.getresponse()
            self.send_response(resp.status)
            self.send_header("Content-Type", resp.getheader("Content-Type", "application/json"))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            sent = True
            while True:
                chunk = resp.read1(4096)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass  # browser pressed Stop; closing upstream halts generation
        except OSError as exc:
            if not sent:
                raise ApiError(502, f"Model server error: {exc}")
        finally:
            conn.close()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Lantern: local LLM studio for llama.cpp")
    parser.add_argument("--open", action="store_true", help="open the UI in your browser")
    args = parser.parse_args()

    for d in (MODELS_DIR, RUNTIME_DIR):
        d.mkdir(parents=True, exist_ok=True)

    atexit.register(engine.stop)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    try:
        httpd = ThreadingHTTPServer(("127.0.0.1", UI_PORT), Handler)
    except OSError as exc:
        sys.exit(f"Cannot bind 127.0.0.1:{UI_PORT} ({exc}). Set LANTERN_PORT to use another port.")
    httpd.daemon_threads = True

    url = f"http://127.0.0.1:{UI_PORT}"
    print(f"Lantern {VERSION}")
    print(f"  UI      {url}")
    print(f"  models  {MODELS_DIR}")
    print(f"  runtime {find_server() or 'llama-server NOT FOUND (see README)'}")
    print("Press Ctrl+C to stop.")
    if args.open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        httpd.server_close()
        engine.stop()


if __name__ == "__main__":
    main()
