# Lantern

A small, local-first web UI for chatting with GGUF models through
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`.
Nothing leaves your machine, and there is nothing to `pip install` or `npm install`.

## Features

- Discovers `.gguf` models (including subfolders; multi-part models show only shard 1; `mmproj` files are hidden)
- Starts and stops `llama-server` from the browser, CPU or GPU offload
- Context length and thread controls
- Streaming chat with a Stop button and Regenerate
- Thinking toggle, with reasoning shown in a collapsible "Thought process" block
  (works with a separate reasoning field or inline `<think>` tags)
- Temperature, top-p, max tokens and system prompt
- Multiple chats saved in your browser, Markdown export, code copy buttons, tokens/sec
- Light and dark themes, usable on narrow screens
- Windows, macOS and Linux

## Requirements

- Python 3.9+
- A llama.cpp build that includes `llama-server` (use the build matching your hardware: CPU, Vulkan, CUDA, Metal)
- One or more `.gguf` model files

## Setup

Lantern keeps everything in one folder (default `~/lantern-lab`, created on first run):

```
lantern-lab/
├── models/       # put .gguf files here
└── llama.cpp/    # extract llama-server (+ its DLLs/libs) here
```

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
| `LANTERN_PORT` | `8080` | UI port |
| `LANTERN_LLAMA_PORT` | UI port + 1 | llama-server port |

## Privacy and security

- Both services bind to `127.0.0.1` only.
- Requests are rejected unless the `Host` and `Origin` headers point at Lantern itself,
  which blocks other websites (and DNS-rebinding tricks) from driving your local model.
- Chats live in your browser's local storage. The model server log is `$LANTERN_HOME/llama-server.log`.

## Troubleshooting

- **No models found**: files must end in `.gguf` and live under the models folder.
- **llama-server not found**: check the runtime folder or set `LANTERN_LLAMA_SERVER`.
- **Model fails to load**: the sidebar shows the last log lines; try CPU mode or a smaller context.
- **Thinking toggle does nothing**: it depends on the model's chat template.
- **Port in use**: set `LANTERN_PORT`.
