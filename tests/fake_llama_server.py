#!/usr/bin/env python3
"""A fake llama-server for Lantern's tests. It is not a real model.

Understands the few flags Lantern passes (-m, --port, -c) and serves:
  GET  /health                 -> 200
  GET  /last-request           -> the last chat request body (test helper)
  POST /v1/chat/completions    -> a canned streaming reply

Behaviour is chosen by the model file name or by keywords in the last user message:
  model name contains "crash"   -> prints an error and exits with code 3
  message contains "OVERFLOW"   -> 400 exceed_context_size_error
  each "BIGMSG" in a user message counts as 1500 prompt tokens, so a few of them
                                -> a natural context overflow (for trim-and-retry tests)
  message contains "TABLE"      -> reply includes a markdown table
  message contains "LONG"       -> reply streams ~200 words slowly (for Stop tests)
"""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import socketserver

print("fake llama-server entrypoint reached", flush=True)
args = sys.argv[1:]


def opt(name, default=None):
    return args[args.index(name) + 1] if name in args else default


PORT = int(opt("--port"))
N_CTX = int(opt("-c", "4096"))
MODEL = opt("-m", "")

if "crash" in MODEL:
    print("error: failed to load model (fake crash)", flush=True)
    sys.exit(3)

LAST = {}


def count_tokens(messages):
    """Rough word count; each "BIGMSG" in a *user* message counts as 1500 extra tokens."""
    total = 4
    for m in messages:
        text = str(m.get("content", ""))
        total += len(text.split())
        if m.get("role") == "user":
            total += 1500 * text.count("BIGMSG")
    return total


class LocalThreadingHTTPServer(ThreadingHTTPServer):
    """Avoid HTTPServer's reverse-DNS lookup during startup.

    The test server only binds to loopback. Resolving that address to an FQDN
    is unnecessary and can stall on isolated macOS CI runners.
    """
    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name = self.server_address[0]
        self.server_port = self.server_address[1]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def _send_json(self, status, obj):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        elif self.path == "/last-request":
            self._send_json(200, LAST)
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        LAST.clear()
        LAST.update(req)
        messages = req.get("messages", [])
        user_text = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        prompt_tokens = count_tokens(messages)

        if "OVERFLOW" in user_text or prompt_tokens > N_CTX:
            self._send_json(400, {"error": {
                "code": 400,
                "type": "exceed_context_size_error",
                "message": "the request exceeds the available context size",
                "n_prompt_tokens": max(prompt_tokens, N_CTX + 123),
                "n_ctx": N_CTX,
            }})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def emit(obj):
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
            self.wfile.flush()

        pieces = []
        continuing = req.get("continue_final_message") in (True, "content")
        if continuing:
            pieces.append(("content", " Continued scene."))
        else:
            if req.get("chat_template_kwargs", {}).get("enable_thinking"):
                for w in ("Let me ", "think."):
                    pieces.append(("reasoning_content", w))
            pieces.append(("content", "You said: %s\n\n" % user_text))
            if "TABLE" in user_text:
                pieces.append(("content", "| Name | Size |\n|---|---|\n| tiny | 1 GB |\n| big | 9 GB |\n\n"))
            if "LONG" in user_text:
                pieces.extend(("content", "word%d " % i) for i in range(200))
            else:
                pieces.append(("content", "```python\ndef hi():\n    return 'x'  # greet\n```\n"))

        completion = 0
        try:
            for key, text in pieces:
                emit({"choices": [{"delta": {key: text}}]})
                completion += 1
                time.sleep(0.05 if "LONG" in user_text else 0.01)
            emit({
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion,
                          "total_tokens": prompt_tokens + completion},
                "timings": {"predicted_per_second": 42.5},
            })
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass


print("fake llama-server binding 127.0.0.1:%d" % PORT, flush=True)
server = LocalThreadingHTTPServer(("127.0.0.1", PORT), Handler)
print("fake llama-server listening on 127.0.0.1:%d" % PORT, flush=True)
server.serve_forever()
