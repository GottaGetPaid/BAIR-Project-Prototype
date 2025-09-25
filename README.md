## BAIR Project Prototype — Web App Guide

This repository contains a simple Flask web app for experimenting with LLM prompts using BFCL-style JSON inputs. It lets you:

- Upload a JSON describing a prompt and its callable tools (toolboxes and tools).
- Preview the JSON on the left (including a grouped Tools & Toolboxes view and a raw JSON toggle).
- Auto-submit the JSON’s question to a chat on the right.
- Save the chat as structured JSON under `test-sessions/`.
- Optional: record a short voice query (local Whisper or Gemini STT if configured).

The “model” response defaults to a placeholder (“please insert an api key for a model”) unless a valid API key and SDK is configured.

## Configuration

To connect to a real model, you'll need to provide an API key. Create a file named `.env` in the root of the project and add your key like this:

```env
# .env
# Google API Key
GOOGLE_API_KEY="<your_api_key>"
```

The application will load this file automatically. The `.env` file is included in `.gitignore` and should not be committed to version control.

---

## Quick Start (Bash / Zsh)

1) Create and activate a virtual environment (Python 3.10+):

```bash
python3 -m venv venv
source venv/bin/activate
```

2) Install dependencies:

```bash
pip install -r requirements.txt
```

3) Run the app (either command works):

```bash
python web/app.py
# or
export FLASK_APP="web/app.py"; flask run --debug
```

Open http://127.0.0.1:5000 in your browser.

---

## Quick Start (Windows PowerShell)

1) Create and activate a virtual environment (Python 3.10+):

```powershell
python -m venv .\venv
.\venv\Scripts\Activate.ps1
```

2) Install dependencies:

```powershell
pip install -r requirements.txt
```

3) Run the app (either command works):

```powershell
python .\web\app.py
# or
$env:FLASK_APP = "web/app.py"; flask run --debug
```

Open http://127.0.0.1:5000 in your browser.

---

## Using the Site

Left panel: Prompt toolbox Preview
- Theme button toggles light/dark (persists via localStorage).
- Tools & Toolboxes: functions are grouped by toolbox (prefix before the dot). Each toolbox is collapsible, and each tool shows its description and parameter schema.
- Raw JSON: hidden by default; click “Show/Hide” to toggle.

Right panel: Chat & Upload
- Query Interface: type a message or use the mic button for voice input (optional; see Voice Input below).
- Upload a JSON file: choose a BFCL-style JSON file and click Upload. The app will:
  - Display the JSON details on the left (ID, Question, Tools & Toolboxes, Raw JSON toggle).
  - Auto-fill and submit the question into the chat on the right.
- Save Chat: persists the current conversation to disk (see Where chats are saved).

Sample JSON
- A ready-to-use example is provided: `queries/BFCL_single.json`.
- Expected shape (simplified):
  - `id`: string
  - `question`: nested list(s) of turns with `{ role: 'user', content: '...' }`
  - `function` (or `functions`): array of tool specs with `name`, `description`, and `parameters` schema
- Example tool naming: `math_toolkit.sum_of_multiples` where `math_toolkit` is the toolbox and `sum_of_multiples` is the tool.

Model response
- If no `GOOGLE_API_KEY` is present or the `google-generativeai` package is not installed, the backend will return a default response: "please insert an api key for a model".

---

## Where chats are saved (and how)

When you click “Save Chat”, the app writes a session folder and JSON file here:

```
test-sessions/
  <N>-<YYYYMMDD-HHMMSS>-<model>-<user>/
    session-data.json
```

- N is an incrementing number.
- model and user are slugified from session info (defaults to `gemini-1.5-flash-latest` and `anonymous`).

File contents follow this structure:

```json
{
  "sessionInfo": {
    "sessionId": "...",
    "userId": "anonymous",
    "startTimestamp": "...",
    "endTimestamp": "...",
    "llmModel": "gemini-1.5-flash-latest"
  },
  "context": {
    "initialContent": "...",  
    "finalContent": "..."     
  },
  "conversationLog": [
    { "turn": 1, "role": "user", "inputType": "text",  "content": "...", "timestamp": "..." },
    { "turn": 2, "role": "assistant", "inputType": "model", "content": "...", "timestamp": "..." }
  ],
  "evaluation": {
    "surveyResponses": {},
    "userComments": ""
  }
}
```

---

## JSON format tips

- The app supports either a single object or an array of objects (it will pick the first).
- `question` is parsed for user content (nested arrays of `{ role, content }`). That content is auto-submitted to the chat.
- Tools are grouped by the prefix in `name` before the first `.` to form a toolbox.

---

## Optional: Voice input

- Mic button records short audio, sends it to the backend, and the transcription is auto-submitted as the query.
- Local Whisper (default path in code): requires FFmpeg and the `openai-whisper` Python package.
    - **Install FFmpeg**: FFmpeg must be installed on your system and available in your PATH.
    - **macOS (using Homebrew):** `brew install ffmpeg`
    - **Debian/Ubuntu:** `sudo apt update && sudo apt install ffmpeg`
    - **Windows (using Chocolatey):** `choco install ffmpeg`
  - Install Whisper: `pip install openai-whisper`
  - Set `STT_BACKEND=whisper_local` in `.env`.
- Gemini STT (alternative): set `STT_BACKEND=gemini` in `.env` and export `GOOGLE_API_KEY`, plus install the module: `pip install google-generativeai`.

If you don’t need voice, you can ignore the mic button.

---

## Optional: Configure a real model

By default, the app returns a placeholder message. To use Gemini for text responses:

```powershell
# In PowerShell (current session)
$env:GOOGLE_API_KEY = "<your_api_key>"
pip install google-generativeai
```

Rerun the app. The `/query` endpoint will call `gemini-1.5-flash-latest` when the key and package are available.

---

## Minimal self-play (for later)

The repository includes a minimal turn-based “self-play” engine (not wired to the web UI yet):

- `engine/` orchestrates turn-taking.
- `games/template_game.py` shows how to implement a game.

---

## Repository layout

- `web/` — Flask app
  - `app.py` — backend routes for JSON upload/parse, chat, voice STT, and saving chats
  - `templates/upload.html` — two-panel UI with toolbox visualization and dark mode
  - `static/` — shared assets
- `queries/` — sample JSONs (`BFCL_single.json`)
- `test-sessions/` — generated chat sessions
- `engine/`, `games/` — basic self-play scaffolding 
- `models/` — model provider scaffolding (optional for now)
- `config.py` — environment configuration

If you only want to test the web features, you don’t need any large model downloads.