# Open Self-Play Template

This is a minimal, extensible template inspired by `lm-selfplay`, designed to:

- Use pluggable language models, including local open models (e.g., Llama via Hugging Face Transformers).
- Support custom turn-based games via a simple Game interface (you implement your own game later).
- Provide a minimal Flask web UI you can redesign later.

## Quick Start

1. Create and activate a Python 3.10+ environment.
2. Install dependencies:

```
pip install -r requirements.txt
```

3. Configure environment (optional):

- Copy `.env.example` to `.env` and adjust values.
- To use a local HF model, set `MODEL_BACKEND=huggingface`, `HF_MODEL_ID` to a model id or local path.

4. Run the web app:

```
export FLASK_APP=web/app.py
flask run --debug
```

Then open http://127.0.0.1:5000

## Structure

- `models/`
  - `base.py`: Model provider interface and loader.
  - `huggingface_provider.py`: Local/remote HF model (Llama etc.).
  - `noop_provider.py`: Dev stub model (chooses first available action or "pass").
- `games/`
  - `base_game.py`: Abstract game interface for your custom game.
  - `template_game.py`: Minimal, commented skeleton to copy for your game.
- `engine/`
  - `session.py`: Orchestrates turn-taking between players and the game.
  - `prompts.py`: Prompt helpers (lightweight, optional).
- `web/`
  - `app.py`: Flask app with simple routes.
  - `templates/`: `index.html`, `play.html` minimal UI.
  - `static/`: `styles.css` and a tiny `app.js`.
- `config.py`: Centralized config via env vars.

## Implement Your Game

Implement a new game by subclassing `SimpleTurnGame` in `games/base_game.py`.
See `games/template_game.py` for a guided starting point. Plug your new game into the web app by updating the factory in `web/app.py` or via config.

## Model Backends

- `huggingface` (recommended for local open models):
  - Set `MODEL_BACKEND=huggingface`
  - Set `HF_MODEL_ID` to a model id (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`) or a local directory path.
  - Optional: `HF_DEVICE` (e.g., `cpu`, `cuda:0`), `HF_MAX_NEW_TOKENS`, `HF_TEMPERATURE`.
- `noop` (dev): always available and fast.

You can add more providers by implementing `ModelProvider` in `models/base.py` and extending `load_model_provider()`.

## CLI (optional)

Run a quick self-play loop without the UI:

```
python -m engine.session --backend noop --steps 5
```

## Notes

- Torch installs can be large; if you're only testing the UI, start with `MODEL_BACKEND=noop`.
- The project is now root-based (no `open-selfplay/` subfolder). If you still have that directory from a previous layout, you can remove it.