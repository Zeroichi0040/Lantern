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
import secrets
import socket
import time
import signal
import subprocess
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from process_job import WindowsJob
from gguf_metadata import layer_count

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


def list_models(include_metadata=False):
    models = []
    if MODELS_DIR.is_dir():
        for p in sorted(MODELS_DIR.rglob("*.gguf")):
            if not p.is_file() or "mmproj" in p.name.lower() or _is_secondary_shard(p.name):
                continue
            item = {"name": p.relative_to(MODELS_DIR).as_posix(), "size": p.stat().st_size}
            if include_metadata:
                blocks = layer_count(p)
                if blocks is not None:
                    item["layers"] = blocks
            models.append(item)
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
        self.lock = threading.RLock()
        self.proc = None
        self.logfh = None
        self.info = {}
        self.last_exit = None
        self.job = None
        self.instance = None
        self.key = None
        self.model_path = None
        self.problem = None

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def _port_free(self):
        # Windows can time out on a closed loopback port instead of returning
        # ECONNREFUSED within a short connect timeout. Probe a real bind first.
        address = ("127.0.0.1", LLAMA_PORT)
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                if os.name == "nt":
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                else:
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(address)
                return True
        except OSError:
            pass

        # A previous owned server can leave sockets in TIME_WAIT. A failed
        # bind alone does not prove a live listener; allow a refused connect.
        # This is a preflight only: readiness still requires launch identity.
        try:
            with socket.create_connection(address, timeout=2):
                return False
        except ConnectionRefusedError:
            return True
        except OSError:
            return False

    def _get_json(self, path):
        conn = http.client.HTTPConnection("127.0.0.1", LLAMA_PORT, timeout=1)
        try:
            conn.request("GET", path, headers={"Authorization": "Bearer " + self.key})
            response = conn.getresponse()
            if response.status != 200:
                return None
            return json.loads(response.read())
        except (OSError, ValueError, http.client.HTTPException):
            return None
        finally:
            conn.close()

    def state(self):
        # Called under the lifecycle lock. Identity must be checked again
        # before every chat, not cached from a previous successful load.
        self.problem = None
        if self.proc is None:
            if not self._port_free():
                self.problem = f"Inference port {LLAMA_PORT} is occupied. Stop the other server or use LANTERN_LLAMA_PORT."
                return "blocked"
            return "stopped"
        if not self.running():
            self.last_exit = self.proc.returncode
            return "crashed" if self.last_exit else "stopped"
        if self._get_json("/health") is None:
            return "loading"
        models = self._get_json("/v1/models")
        props = self._get_json("/props")
        entries = models.get("data", []) if isinstance(models, dict) else []
        reported_path = props.get("model_path") if isinstance(props, dict) else None
        if (not isinstance(entries, list) or len(entries) != 1
                or not isinstance(entries[0], dict) or entries[0].get("id") != self.instance
                or not isinstance(reported_path, str)
                or os.path.normcase(os.path.abspath(reported_path)) != self.model_path):
            self.problem = "Inference server identity or loaded model could not be verified. Stop and check the llama-server log."
            return "blocked"
        if not self.running():
            self.last_exit = self.proc.returncode
            return "crashed" if self.last_exit else "stopped"
        generation = props.get("default_generation_settings")
        actual_ctx = generation.get("n_ctx") if isinstance(generation, dict) else None
        if type(actual_ctx) is int and actual_ctx > 0:
            self.info["context_per_slot"] = actual_ctx
        return "ready"

    def ready(self):
        with self.lock:
            return self.state() == "ready"

    def start(self, model, mode, ctx, threads, gpu_layers=8, batch=2048,
              ubatch=512, slots=1, batch_threads=0):
        exe = find_server()
        if exe is None:
            raise ApiError(
                400,
                f"{EXE_NAME} not found. Put a llama.cpp build under {RUNTIME_DIR} "
                "or add it to PATH.",
            )
        path = resolve_model(model)
        if mode not in ("cpu", "hybrid", "gpu"):
            raise ApiError(400, "mode must be 'cpu', 'hybrid' or 'gpu'")

        def integer(name, value, low, high):
            # Accept integer JSON values and legacy numeric strings, not bools or floats.
            if type(value) is int:
                parsed = value
            elif (isinstance(value, str) and value.strip().isascii()
                  and value.strip().isdigit() and len(value.strip()) <= 10):
                parsed = int(value)
            else:
                raise ApiError(400, f"{name} must be an integer")
            if not low <= parsed <= high:
                raise ApiError(400, f"{name} must be between {low} and {high}")
            return parsed

        ctx = integer("ctx", ctx, 256, 1048576)
        threads = integer("threads", threads, 1, 512)
        gpu_layers = integer("gpu_layers", gpu_layers, 1, 1024)
        batch = integer("batch", batch, 1, 1048576)
        ubatch = integer("ubatch", ubatch, 1, batch)
        slots = integer("slots", slots, 1, 64)
        batch_threads = integer("batch_threads", batch_threads, 0, 512)
        if ctx < 256 * slots:
            raise ApiError(400, "Context must provide at least 256 tokens per parallel slot")
        blocks = layer_count(path) if mode == "hybrid" else None
        if blocks is not None and gpu_layers > blocks:
            raise ApiError(400, f"This model has {blocks} repeating layers; use GPU mode for full offload")
        offload = 0 if mode == "cpu" else 999 if mode == "gpu" else gpu_layers
        effective_batch_threads = batch_threads or threads

        with self.lock:
            self._stop_locked()
            if not self._port_free():
                raise ApiError(409, f"Inference port {LLAMA_PORT} is occupied. Stop the other server or use LANTERN_LLAMA_PORT.")
            instance = "lantern-" + secrets.token_hex(16)
            key = secrets.token_hex(32)
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            self.logfh = open(LOG_FILE, "wb")
            args = [
                str(exe), "-m", str(path.resolve()),
                "--alias", instance, "--api-key", key,
                "--host", "127.0.0.1", "--port", str(LLAMA_PORT),
                "-c", str(ctx), "-t", str(threads),
                "-ngl", str(offload), "-b", str(batch), "-ub", str(ubatch),
                "-np", str(slots), "-tb", str(effective_batch_threads),
                "--jinja",
            ]
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            try:
                if os.name == "nt":
                    self.job = WindowsJob()
                self.proc = subprocess.Popen(
                    args, stdout=self.logfh, stderr=subprocess.STDOUT,
                    cwd=str(exe.parent), creationflags=flags,
                )
                if self.job is not None:
                    self.job.assign(self.proc.pid)
            except OSError as exc:
                self._stop_locked()
                raise ApiError(500, f"Could not start or contain {exe}: {exc}")
            self.last_exit = None
            self.instance, self.key = instance, key
            self.model_path = os.path.normcase(os.path.abspath(path.resolve()))
            self.info = {"model": model, "mode": mode, "ctx": ctx, "threads": threads,
                         "gpu_layers": gpu_layers, "offload_layers": offload,
                         "batch": batch, "ubatch": ubatch, "slots": slots,
                         "batch_threads": batch_threads,
                         "context_per_slot": ctx // slots}
            return self.snapshot()

    def _stop_locked(self):
        proc = self.proc
        if proc is not None and proc.poll() is None:
            if os.name == "nt":
                # Also kill a launcher tree, including children created before
                # the launcher could be assigned to the Windows job.
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        check=False, timeout=10,
                    )
                except (OSError, subprocess.TimeoutExpired):
                    pass
            else:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
        if self.job is not None:
            self.job.close()
            self.job = None
        if proc is not None:
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                try:
                    proc.kill()
                    proc.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    raise ApiError(503, f"Could not stop the owned inference process: {exc}")
            self.last_exit = 0
        self.proc = None
        self.info = {}
        self.instance = self.key = self.model_path = None
        if self.logfh is not None:
            self.logfh.close()
            self.logfh = None
        # A listener may take a moment to disappear after a process tree exits.
        if proc is not None:
            deadline = time.monotonic() + 2
            while not self._port_free() and time.monotonic() < deadline:
                time.sleep(0.05)

    def stop(self):
        with self.lock:
            self._stop_locked()
            return self.snapshot()

    def snapshot(self):
        with self.lock:
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
            if state == "ready":
                out["instance"] = self.instance
            if self.problem:
                out["problem"] = self.problem
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
                self._json(200, {"models": list_models(include_metadata=True), "models_dir": str(MODELS_DIR)})
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
                self._json(200, engine.start(
                    data.get("model"), data.get("mode", "cpu"),
                    data.get("ctx", 4096), data.get("threads", os.cpu_count() or 4),
                    gpu_layers=data.get("gpu_layers", 8), batch=data.get("batch", 2048),
                    ubatch=data.get("ubatch", 512), slots=data.get("slots", 1),
                    batch_threads=data.get("batch_threads", 0),
                ))
            elif path == "/api/stop":
                self._json(200, engine.stop())
            elif path == "/v1/chat/completions":
                self._proxy_chat(self._read_body())
            else:
                self._json(404, {"error": "Not found"})
        except ApiError as exc:
            self._json(exc.status, {"error": exc.message})

    def _proxy_chat(self, body):
        conn = http.client.HTTPConnection("127.0.0.1", LLAMA_PORT, timeout=900)
        sent = False
        try:
            with engine.lock:
                if not engine.ready():
                    raise ApiError(503, engine.problem or "No verified model is loaded yet.")
                expected = self.headers.get("X-Lantern-Instance")
                if expected is not None and expected != engine.instance:
                    raise ApiError(409, "The model changed before the request was sent. Try again.")
                # Bind dispatch to this launch before another load/stop can run.
                # The lock is released before waiting for streamed generation.
                conn.connect()
                conn.request("POST", "/v1/chat/completions", body,
                             {"Content-Type": "application/json",
                              "Authorization": "Bearer " + engine.key})
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
