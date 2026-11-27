# Lantern

A small, local-first web UI for chatting with GGUF models through
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`.
Local by default, with an optional experimental private Tailscale connection. There is nothing to `pip install` or `npm install`.

## Features

- Discovers `.gguf` models (including subfolders; multi-part models show only shard 1; `mmproj` files are hidden)
- Starts and stops `llama-server` from the browser, CPU or GPU offload
- Context length and thread controls, remembered separately for each model
- Streaming chat with a Stop button (or Esc), Regenerate, and Continue (a hidden prompt) to extend a reply in separate assistant bubbles
- Edit completed assistant responses; changes are saved into conversation history and used by subsequent turns
- Configurable message actions (Edit, Continue, Delete), optional cascading deletion with a confirmation step, and a model-aware composer placeholder
- Delete individual user or assistant bubbles, optionally deleting every message after the selected bubble
- Context meter showing how full the model's context window is; if a chat gets too long,
  one click trims the oldest messages and retries
- Edit and resend an earlier message
- Thinking toggle, with reasoning shown in a collapsible "Thought process" block
  (works with a separate reasoning field or inline `<think>` tags)
- A reorganized, category-based Settings screen with a decorative username, assistant identity, message controls, reusable system prompts, and generation settings
- Live and completed response durations, with an optional separate thinking-time breakdown
- System prompt library with per-conversation activation, reusable prompt editing, and no more than one active prompt per chat
- Multiple chats saved in your browser: search, rename (double-click), delete, Markdown export
- Markdown tables, syntax-highlighted code blocks with copy buttons, tokens/sec per reply
- Light and dark themes, usable on narrow screens
- Windows, macOS and Linux

## Requirements

- Python 3.9+
- A llama.cpp build that includes `llama-server` (use the build matching your hardware: CPU, Vulkan, CUDA, Metal)
- One or more `.gguf` model files

## Setup

Lantern can use a shared workspace. When you clone it as a folder beside
`models/` and `llama.cpp/`, it automatically uses that parent folder:

```
llm-lab/
├── lantern/      # this repository
├── models/       # put .gguf files here
├── llama.cpp/    # llama-server (+ its DLLs/libs)
└── studio/       # optional, unrelated projects
```

That layout is detected automatically. Otherwise Lantern uses
`~/lantern-lab` (created on first run). Set `LANTERN_HOME` to choose
another workspace explicitly.

If `llama-server` is already on your `PATH`, you can skip the `llama.cpp/` folder.

## Run

```bash
./run.sh --open            # macOS / Linux
```
```powershell
.\run.ps1 --open           # Windows
```

Then open <http://127.0.0.1:8080> (`--open` does it for you).

## Using it

1. Pick a model (press ↻ after adding files).
2. Choose **CPU** or **GPU**, set context and threads, click **Load model**.
3. Chat. Press **Stop** to halt a reply; use **Stop** in the Model panel before switching models.

**GPU** passes `-ngl 99` (offload all layers). It only does something if your llama.cpp build has a GPU backend.

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `LANTERN_HOME` | `~/lantern-lab` | Base folder |
| `LANTERN_MODELS` | `$LANTERN_HOME/models` | Models folder |
| `LANTERN_LLAMA_SERVER` | auto-detected | Explicit path to `llama-server` |
| `LANTERN_LOG_FILE` | `$LANTERN_HOME/lantern-server.log` | Lantern's server log file |
| `LANTERN_PORT` | `8080` | UI port |
| `LANTERN_LLAMA_PORT` | UI port + 1 | llama-server port |
| `LANTERN_REMOTE_ORIGIN` | unset | Exact Tailscale Serve HTTPS origin (experimental) |
| `LANTERN_REMOTE_LOGIN` | unset | Exact authorized Tailscale user login (experimental) |

## Privacy and security

- Both services bind to `127.0.0.1` only.
- Requests are rejected unless the `Host` and `Origin` headers point at Lantern itself.
  Experimental remote access additionally requires the configured Tailscale identity and exact HTTPS origin.
- Chats live in your browser's local storage. Lantern writes its model server log to `$LANTERN_HOME/lantern-server.log` by default, so it does not overwrite an existing `llama-server.log`.

## Experimental remote access (Tailscale Serve)

Install Tailscale on your computer and phone and sign into the same account.
From an administrator PowerShell window on Windows, run (use your configured
Lantern port if it differs from 8080):

```powershell
tailscale serve --bg 8080
tailscale serve status
```

Follow any HTTPS setup prompt. Copy the private HTTPS URL shown by Serve and
use your exact Tailscale login (usually your email). In your Lantern terminal:

```powershell
$env:LANTERN_REMOTE_ORIGIN = "https://your-pc.your-tailnet.ts.net"
$env:LANTERN_REMOTE_LOGIN = "you@example.com"
.\run.ps1
```

On macOS/Linux use the same variables with `export` and run `./run.sh`.
Keep Lantern and the PC running and connected. Open the HTTPS address in your
phone browser while connected to Tailscale. Test with Wi-Fi off before leaving.
Both services remain bound to loopback; the app trusts Tailscale Serve's
identity header only on its local reverse-proxy connection. Do not use Funnel,
public reverse proxies or port forwarding. Local software can still access
the local service. Chat history and settings remain per-browser/origin, not
synchronized. Long generations may be interrupted by mobile disconnections.

Stop Serve with `tailscale serve --https=443 off` (if using the default HTTPS
port), and restart Lantern without either remote environment variable to
return to localhost-only use.

## Development and tests

The backend is plain standard-library Python; the UI is one HTML file (`web/index.html`).
Tests run against a fake `llama-server` (`tests/fake_llama_server.py`), so no real model is needed:

```bash
python -m unittest discover -s tests -p "test_backend.py" -v   # backend, no extra packages

pip install playwright && python -m playwright install chromium
python -m unittest discover -s tests -p "test_ui.py" -v        # drives the UI in a real browser
```

Set `LANTERN_TEST_CHROME=/path/to/chrome` to use an installed Chrome instead of Playwright's download.
GitHub Actions runs both suites on every push.

## License

MIT. See [LICENSE](LICENSE).

## Troubleshooting

- **No models found**: files must end in `.gguf` and live under the models folder.
- **llama-server not found**: check the runtime folder or set `LANTERN_LLAMA_SERVER`.
- **Model fails to load**: the sidebar shows the last log lines; try CPU mode or a smaller context.
- **Thinking toggle does nothing**: it depends on the model's chat template.
- **Port in use**: set `LANTERN_PORT`.
