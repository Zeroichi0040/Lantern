# Lantern

A small, local-first web UI for chatting with GGUF models through
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`.
Nothing leaves your machine, and there is nothing to `pip install` or `npm install`.

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

## Privacy and security

- Both services bind to `127.0.0.1` only.
- Requests are rejected unless the `Host` and `Origin` headers point at Lantern itself,
  which blocks other websites (and DNS-rebinding tricks) from driving your local model.
- Chats live in your browser's local storage. Lantern writes its model server log to `$LANTERN_HOME/lantern-server.log` by default, so it does not overwrite an existing `llama-server.log`.

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

### Inference server lifecycle

Lantern verifies a unique launch alias and the loaded model path before serving chat. A conflicting listener on the configured inference port is not adopted or killed; stop it yourself or set `LANTERN_LLAMA_PORT` to a free port. On Windows the owned process is assigned to a kill-on-close job object (in addition to explicit process-tree shutdown). The runtime must support llama-server's `--alias`, `--api-key`, `/v1/models`, and `/props` model path; older incompatible builds will not be marked ready.

### Advanced inference

The model sidebar has CPU, Hybrid and GPU modes. Hybrid allows selecting the number of repeating GPU layers (GGUF metadata supplies the limit when readable); GPU requests full offload. Under Advanced inference, configure logical batch size (`-b`, default 2048), physical batch size (`-ub`, default 512), parallel slots (`-np`, default 1), and prompt threads (`-tb`, 0 follows generation threads). All are saved per model. Changes require reloading; generation sampling can still change between replies. For constrained memory, start with one slot, context 1024, 4 threads, Hybrid 8 layers, batch 128 and physical batch 64, then tune. These are not automatic hardware recommendations. Multiple slots may divide the available context; the context meter uses the runtime-reported slot context when available. Older saved CPU/GPU profiles remain valid; new settings default as above. No additional dependencies.
