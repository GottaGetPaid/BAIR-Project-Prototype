from __future__ import annotations
from flask import Flask, render_template, request, redirect, url_for, session as flask_session, jsonify
from flask_sock import Sock
import uuid

import sys
import os
import json
import re
from collections import Counter
try:
    import google.generativeai as genai
except ImportError:
    genai = None  # Allow app to run without the package; endpoints will fallback
from huggingface_hub import InferenceClient
try:
    import asyncio
    import websockets
    import json as json_lib
    deepgram_available = True
except ImportError:
    deepgram_available = False
import openai
from werkzeug.utils import secure_filename
from datetime import datetime
from typing import Optional
import time
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from config import AppConfig
from models.base import load_model_provider

app = Flask(__name__)
app.secret_key = AppConfig.SECRET_KEY
sock = Sock(app)

# In-memory store for demo purposes
SESSIONS = {}

# Global Whisper model (load once at startup)
WHISPER_MODEL = None

def get_whisper_model():
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        try:
            import whisper
            print("Loading Whisper model (one-time setup)...")
            WHISPER_MODEL = whisper.load_model("tiny")
            print("Whisper model loaded successfully!")
        except ImportError:
            print("Whisper not available")
            WHISPER_MODEL = False
    return WHISPER_MODEL if WHISPER_MODEL is not False else None

# Setup folders
QUERIES_FOLDER = 'queries'
VOICE_METADATA_FOLDER = 'voice_metadata'
os.makedirs(QUERIES_FOLDER, exist_ok=True)
os.makedirs(VOICE_METADATA_FOLDER, exist_ok=True)
REPO_ROOT = parent_dir
TEST_SESSIONS_FOLDER = os.path.join(REPO_ROOT, 'test-sessions')
os.makedirs(TEST_SESSIONS_FOLDER, exist_ok=True)

MODEL_PROVIDER_CACHE = None
TOPIC_DATA_PATH = os.path.join(current_dir, 'static', 'data', 'example_functions.json')
TOPIC_CATALOG_CACHE = None
TOPIC_INDEX_CACHE = None
DEFAULT_TOPICS = [
    {"id": "travel", "title": "Travel", "description": "Flights, hotels, itineraries."},
    {"id": "science", "title": "Science", "description": "Facts, summaries, lookups."},
    {"id": "math", "title": "Math", "description": "Arithmetic, primes, and geometry."},
    {"id": "weather", "title": "Weather", "description": "Forecasts and conditions."},
    {"id": "finance", "title": "Finance", "description": "Budgets, markets, and analysis."},
    {"id": "coding", "title": "Coding", "description": "APIs, debugging, code examples."},
    {"id": "health", "title": "Health", "description": "Wellness and nutrition guidance."},
    {"id": "education", "title": "Education", "description": "Study help and quizzes."}
]
TOPIC_SYNONYM_MAP = {
    "physics": ["physics", "mechanics", "mechanic", "quantum", "gravity", "force", "forces", "energy", "motion", "optics", "electromagnetism", "relativity", "thermodynamics", "waves"],
    "science": ["science", "scientific", "biology", "chemistry", "research", "experiment", "experiments", "lab", "laboratory"],
    "search": ["search", "lookup", "google", "bing", "find", "discover", "locate", "query", "information", "research"],
    "trivia": ["trivia", "facts", "fact", "quiz", "questions", "curiosity", "knowledge"],
    "math": ["math", "mathematics", "algebra", "geometry", "calculus", "statistics", "arithmetic", "numbers", "equations"],
    "weather": ["weather", "forecast", "temperature", "rain", "climate", "conditions", "storm", "humidity"],
    "finance": ["finance", "stocks", "markets", "budget", "investing", "investment", "money", "economy", "economic"],
    "coding": ["coding", "programming", "debug", "debugging", "code", "software", "api", "development", "compute"],
    "travel": ["travel", "trip", "journey", "flights", "flight", "hotel", "vacation", "itinerary", "tourism"],
    "health": ["health", "wellness", "nutrition", "fitness", "exercise", "diet", "medical", "medicine"],
    "sports": ["sports", "sport", "game", "games", "scores", "athletics", "leagues", "teams", "tournament"],
    "news": ["news", "headline", "headlines", "breaking", "articles", "media", "press", "reports", "reporting"],
    "education": ["education", "study", "studies", "learning", "school", "teaching", "quiz", "lesson", "class"]
}
STOPWORDS = {
    'the', 'a', 'an', 'and', 'or', 'for', 'with', 'to', 'of', 'in', 'on', 'about', 'is', 'are', 'was', 'were',
    'be', 'being', 'been', 'that', 'this', 'those', 'these', 'over', 'under', 'between', 'how', 'what', 'where',
    'which', 'when', 'who', 'why', 'you', 'your', 'please', 'can', 'could', 'would', 'should', 'tell', 'me',
    'give', 'show', 'list', 'explain', 'provide', 'find', 'from', 'into', 'at', 'by', 'it', 'as'
}
WORD_PATTERN = re.compile(r"[A-Za-z0-9']+")


def _get_model_provider():
    """Load and cache the local Hugging Face provider if configured."""
    global MODEL_PROVIDER_CACHE
    if AppConfig.MODEL_BACKEND != 'huggingface':
        return None
    if MODEL_PROVIDER_CACHE is None:
        try:
            MODEL_PROVIDER_CACHE = load_model_provider()
        except Exception as exc:  # noqa: BLE001 - log and continue
            print(f"Unable to load local model provider: {exc}")
            MODEL_PROVIDER_CACHE = False
    return MODEL_PROVIDER_CACHE if MODEL_PROVIDER_CACHE else None


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    tokens = []
    for match in WORD_PATTERN.findall(text):
        token = match.lower()
        if token not in STOPWORDS:
            tokens.append(token)
    return tokens


def _load_topic_catalog() -> list[dict]:
    global TOPIC_CATALOG_CACHE
    if TOPIC_CATALOG_CACHE is None:
        try:
            with open(TOPIC_DATA_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            topics = data.get('topics') if isinstance(data, dict) else []
            if not isinstance(topics, list):
                topics = []
            TOPIC_CATALOG_CACHE = topics
        except FileNotFoundError:
            TOPIC_CATALOG_CACHE = DEFAULT_TOPICS.copy()
        except Exception as exc:  # noqa: BLE001 - log only
            print(f"Failed to load topic catalog: {exc}")
            TOPIC_CATALOG_CACHE = DEFAULT_TOPICS.copy()
    return TOPIC_CATALOG_CACHE


def _build_topic_index() -> list[dict]:
    global TOPIC_INDEX_CACHE
    if TOPIC_INDEX_CACHE is not None:
        return TOPIC_INDEX_CACHE
    catalog = _load_topic_catalog()
    index = []
    for topic in catalog:
        text_parts = [topic.get('title', ''), topic.get('description', '')]
        for fn in topic.get('functions') or []:
            text_parts.append(fn.get('name', ''))
            text_parts.append(fn.get('description', ''))
        tokens = set(_tokenize(' '.join(text_parts)))
        synonyms = set(token.lower() for token in TOPIC_SYNONYM_MAP.get(topic.get('id', ''), []))
        index.append({
            "topic": topic,
            "tokens": tokens | synonyms,
            "synonyms": synonyms,
        })
    TOPIC_INDEX_CACHE = index
    return TOPIC_INDEX_CACHE


def _topic_summary(topic: dict) -> dict:
    return {
        "title": (topic or {}).get('title', ''),
        "description": (topic or {}).get('description', ''),
    }


def _score_topics(prompt: str) -> list[dict]:
    tokens = _tokenize(prompt)
    if not tokens:
        return [_topic_summary(t) for t in _load_topic_catalog()[:3]]
    counts = Counter(tokens)
    scored = []
    for entry in _build_topic_index():
        base = entry['tokens']
        synonyms = entry['synonyms']
        score = 0.0
        for tok, freq in counts.items():
            if tok in base:
                score += freq * 1.0
            if tok in synonyms:
                score += freq * 1.5
        scored.append((score, entry['topic']))
    scored.sort(key=lambda tup: tup[0], reverse=True)
    picks = [topic for score, topic in scored if score > 0][:3]
    if not picks:
        picks = _load_topic_catalog()[:3]
    return [_topic_summary(t) for t in picks]


def _extract_json_object(text: str):
    if not text:
        return None
    decoder = json.JSONDecoder()
    stripped = text.strip()
    try:
        obj, _ = decoder.raw_decode(stripped)
        return obj
    except json.JSONDecodeError:
        pass
    for idx, char in enumerate(stripped):
        if char == '{':
            try:
                obj, _ = decoder.raw_decode(stripped[idx:])
                return obj
            except json.JSONDecodeError:
                continue
    return None


def _normalize_topics(value) -> list[dict]:
    topics: list[dict] = []
    if isinstance(value, dict):
        value = value.get('items') or value.get('topics') or value.get('list')
    if isinstance(value, list):
        for entry in value:
            if isinstance(entry, dict):
                title = str(entry.get('title') or entry.get('topic') or entry.get('name') or '').strip()
                desc = str(entry.get('description') or entry.get('detail') or entry.get('summary') or '').strip()
            else:
                title = str(entry).strip()
                desc = ''
            if title:
                topics.append({"title": title, "description": desc})
    return topics


def _normalize_followups(value) -> list[str]:
    followups: list[str] = []
    if isinstance(value, dict):
        value = value.get('items') or value.get('questions') or value.get('followUps')
    if isinstance(value, list):
        for entry in value:
            text = str(entry).strip()
            if text:
                followups.append(text)
    return followups


def _llm_related_topics(prompt: str) -> dict | None:
    provider = _get_model_provider()
    if not provider:
        return None
    system_prompt = (
        "You help a conversation facilitator suggest relevant directions. "
        "Given the user's original prompt, respond ONLY with minified JSON containing two keys: "
        "'topics' (array of up to 3 objects with 'title' and 'description') and 'followUps' (array of up to 2 example questions)."
    )
    user_message = (
        "Original prompt:\n"
        f"{prompt}\n"
        "Return JSON now."
    )
    try:
        raw = provider.generate([
            {"role": "user", "content": user_message}
        ], system_prompt=system_prompt, max_new_tokens=256, temperature=0.4)
    except Exception as exc:  # noqa: BLE001
        print(f"Local LLM related-topics failure: {exc}")
        return None
    parsed = _extract_json_object(raw)
    if not isinstance(parsed, dict):
        return None
    topics = _normalize_topics(parsed.get('topics') or parsed.get('related_topics'))
    followups = _normalize_followups(parsed.get('followUps') or parsed.get('followups') or parsed.get('follow_up_questions'))
    if not topics:
        return None
    return {
        "topics": topics[:3],
        "followUps": followups[:2],
    }

@app.route("/")
def index():
    return render_template("upload.html", stt_backend=AppConfig.STT_BACKEND)

@app.route("/test")
def test_mic():
    return render_template("test_mic.html")

# Removed unused /upload route - we use Deepgram streaming, no file uploads needed

@app.route('/query', methods=['POST'])
def query():
    query_text = request.form.get('query', '').strip()
    if not query_text: return jsonify({"error": "No query provided"}), 400

    backend = AppConfig.MODEL_BACKEND
    default_msg = "Model backend not configured or API key is missing."
    response_text = default_msg

    try:
        if backend == 'gemini':
            print("Sending query to Gemini 1.5 Flash model...")
            api_key = os.environ.get("GOOGLE_API_KEY") if genai else None
            if api_key and genai is not None:
                try:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-1.5-flash-latest')
                    response = model.generate_content(f"Respond to this in exactly 1 sentence: {query_text}")
                    response_text = getattr(response, 'text', None) or default_msg
                except Exception as model_err:
                    print(f"Model error: {model_err}")
                    response_text = default_msg
            else:
                if not api_key: print("No GOOGLE_API_KEY found; returning default message.")
                if genai is None: print("google-generativeai is not installed; returning default message.")

        elif backend == 'huggingface':
            print(f"Sending query to Hugging Face model: {AppConfig.HF_MODEL_ID}...")
            hf_token = AppConfig.HUGGING_FACE_API_TOKEN
            if hf_token and AppConfig.HF_MODEL_ID:
                try:
                    client = InferenceClient(token=hf_token)
                    response = client.chat_completion(
                        messages=[
                            {
                                "role": "user",
                                "content": f"Respond to this in exactly 1 sentence: {query_text}"
                            }
                        ],
                        model=AppConfig.HF_MODEL_ID,
                        max_tokens=AppConfig.HF_MAX_NEW_TOKENS,
                        temperature=AppConfig.HF_TEMPERATURE or None, # Temp must be > 0, so pass None if 0
                    )
                    if response.choices:
                        response_text = response.choices[0].message.content or default_msg
                except Exception as model_err:
                    print(f"Hugging Face model error: {model_err}")
                    response_text = default_msg
            else:
                if not hf_token: print("No HUGGING_FACE_API_TOKEN found; returning default message.")
                if not AppConfig.HF_MODEL_ID: print("No HF_MODEL_ID found; returning default message.")
        else:
            print(f"Model backend is '{backend}'. No action taken.")
            response_text = "No model backend selected. Please set MODEL_BACKEND in your .env file."

        response_text = response_text.strip()
        print(f'Query: {query_text}\nResponse: {response_text}')
        with open(os.path.join(QUERIES_FOLDER, 'queries.csv'), 'a', encoding='utf-8') as f:
            f.write(f'{UPLOAD_FILE_COUNT},"{query_text}","{response_text}"\n')
        return jsonify({"response": response_text}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------
# JSON upload/parse route
# ----------------------------

def _extract_question_text(question_field) -> str:
    """Extract concatenated user question content from the BFCL-like nested question structure."""
    try:
        texts = []
        if isinstance(question_field, list):
            # It may be list of lists of dicts
            def walk(node):
                if isinstance(node, dict):
                    if node.get('role') == 'user' and 'content' in node:
                        texts.append(str(node['content']))
                elif isinstance(node, list):
                    for item in node:
                        walk(item)
            walk(question_field)
        elif isinstance(question_field, dict):
            if question_field.get('role') == 'user' and 'content' in question_field:
                texts.append(str(question_field['content']))
        # Join with double newline to separate turns if multiple
        return "\n\n".join(texts).strip()
    except Exception:
        return ""

@app.route('/upload_json', methods=['POST'])
def upload_json():
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400
    try:
        raw_text = file.read().decode('utf-8')
        entries = []
        if file.filename.endswith('.jsonl'):
            for line in raw_text.strip().split('\n'):
                if line.strip():
                    entries.append(json.loads(line))
        else:
            data_obj = json.loads(raw_text)
            if isinstance(data_obj, list):
                entries.extend(data_obj)
            else:
                entries.append(data_obj)

        parsed_list = []
        for entry in entries:
            q_text = _extract_question_text(entry.get('question'))
            functions = entry.get('function') or entry.get('functions') or []
            parsed = {
                'id': entry.get('id'),
                'questionText': q_text,
                'functions': functions,
                'raw': entry
            }
            parsed_list.append(parsed)

        sid = flask_session.get('ui_sid') or str(uuid.uuid4())
        flask_session['ui_sid'] = sid
        SESSIONS[f'last_json:{sid}'] = parsed_list

        return jsonify({"message": f"{len(parsed_list)} entries parsed successfully", "parsed_list": parsed_list}), 200
    except json.JSONDecodeError as je:
        return jsonify({"error": f"Invalid JSON: {je}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/related_topics', methods=['POST'])
def related_topics():
    payload = request.get_json(silent=True) or {}
    prompt = (payload.get('prompt') or '').strip()
    if not prompt:
        return jsonify({"topics": [], "followUps": [], "source": "none"}), 200

    llm_payload = _llm_related_topics(prompt)
    if llm_payload:
        llm_payload['source'] = 'llm'
        return jsonify(llm_payload), 200

    topics = _score_topics(prompt)
    return jsonify({"topics": topics, "followUps": [], "source": "fallback"}), 200

@app.route('/get_bfcl_tools', methods=['GET'])
def get_bfcl_tools():
    """Load BFCL tools from queries folder and return sample tools with example question."""
    import random
    
    bfcl_path = os.path.join(parent_dir, 'queries', 'BFCL_multiple.jsonl')
    
    if not os.path.exists(bfcl_path):
        return jsonify({"tools": [], "exampleQuestion": "", "error": "BFCL file not found"}), 404
    
    try:
        # Load all entries from BFCL
        entries = []
        with open(bfcl_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        
        if not entries:
            return jsonify({"tools": [], "exampleQuestion": ""}), 200
        
        # Pick a random entry to get example question and tools
        selected_entry = random.choice(entries)
        
        # Extract example question
        example_question = ""
        if selected_entry.get('question') and len(selected_entry['question']) > 0:
            if len(selected_entry['question'][0]) > 0:
                example_question = selected_entry['question'][0][0].get('content', '')
        
        # Extract tools/functions
        tools = []
        if selected_entry.get('function'):
            for func in selected_entry['function'][:3]:  # Limit to 3 tools
                tool_info = {
                    "name": func.get('name', 'Unknown'),
                    "description": func.get('description', ''),
                    "parameters": func.get('parameters', {})
                }
                tools.append(tool_info)
        
        return jsonify({
            "tools": tools,
            "exampleQuestion": example_question,
            "entryId": selected_entry.get('id', '')
        }), 200
        
    except Exception as e:
        return jsonify({"tools": [], "exampleQuestion": "", "error": str(e)}), 500

# ----------------------------
# Save chat route
# ----------------------------

def _slugify(value: str) -> str:
    value = value or ''
    value = re.sub(r'[^A-Za-z0-9_.-]+', '-', value)
    return value.strip('-') or 'unknown'

@app.route('/save_voice_metadata', methods=['POST'])
def save_voice_metadata_route():
    """Accept voice metadata from frontend and save it."""
    try:
        payload = request.get_json(force=True, silent=False) or {}
        metadata = payload.get('metadata', {})
        transcript = payload.get('transcript', '')
        
        filepath = _save_voice_metadata(metadata, transcript)
        
        if filepath:
            return jsonify({"message": "Voice metadata saved", "path": filepath}), 200
        else:
            return jsonify({"error": "Failed to save metadata"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/save_chat', methods=['POST'])
def save_chat():
    try:
        payload = request.get_json(force=True, silent=False) or {}
        # Ensure structure and fill defaults
        now_iso = datetime.utcnow().isoformat() + 'Z'
        session_info = payload.get('sessionInfo') or {}
        session_info.setdefault('sessionId', str(uuid.uuid4()))
        session_info.setdefault('userId', 'anonymous')
        session_info.setdefault('startTimestamp', now_iso)
        session_info.setdefault('endTimestamp', now_iso)
        session_info.setdefault('llmModel', 'gemini-1.5-flash-latest')

        context = payload.get('context') or {}
        context.setdefault('initialContent', '')
        context.setdefault('finalContent', '')

        conversation_log = payload.get('conversationLog') or []
        evaluation = payload.get('evaluation') or {"surveyResponses": {}, "userComments": ""}

        final_payload = {
            "sessionInfo": session_info,
            "context": context,
            "conversationLog": conversation_log,
            "evaluation": evaluation,
        }

        # Build directory name: session-number-date-model-user
        # Count existing sessions to create next number
        existing = [d for d in os.listdir(TEST_SESSIONS_FOLDER) if os.path.isdir(os.path.join(TEST_SESSIONS_FOLDER, d))]
        next_num = len(existing) + 1
        date_str = datetime.utcnow().strftime('%Y%m%d-%H%M%S')
        model_slug = _slugify(session_info.get('llmModel', 'model'))
        user_slug = _slugify(session_info.get('userId', 'user'))
        dir_name = f"{next_num}-{date_str}-{model_slug}-{user_slug}"
        out_dir = os.path.join(TEST_SESSIONS_FOLDER, dir_name)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'session-data.json')
        with open(out_path, 'w', encoding='utf-8') as w:
            json.dump(final_payload, w, ensure_ascii=False, indent=2)
        return jsonify({"message": "Chat saved", "path": out_path, "dirName": dir_name}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ----------------------------
# Speech-to-Text (STT) routes
# ----------------------------
# NOTE: Chunked recording routes below are DISABLED - we use Deepgram streaming only
# Uncomment if you need Whisper/Gemini chunked recording

# def _save_chunk(file_storage) -> str:
#     filename = secure_filename(file_storage.filename or f"chunk_{datetime.utcnow().timestamp()}.webm")
#     path = os.path.join(AUDIO_TMP, filename)
#     file_storage.save(path)
#     return path

# def _concat_chunks(paths: list[str], out_path: str) -> str:
#     with open(out_path, 'wb') as w:
#         for p in paths:
#             with open(p, 'rb') as r: w.write(r.read())
#     return out_path

# def _convert_to_wav(input_path: str) -> Optional[str]:
#     """Converts an audio file to a standard WAV format using FFmpeg."""
#     if not os.path.exists(input_path):
#         return None
#     output_path = input_path.rsplit('.', 1)[0] + ".wav"
#     print(f"Converting {input_path} to {output_path} using FFmpeg...")
#     try:
#         command = [
#             'ffmpeg',
#             '-i', input_path,
#             '-ac', '1',
#             '-ar', '16000',
#             '-y',
#             output_path
#         ]
#         subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#         print("...Conversion successful.")
#         return output_path
#     except (subprocess.CalledProcessError, FileNotFoundError) as e:
#         print(f"!!! FFmpeg ERROR: {e}")
#         print(f"!!! Is FFmpeg installed and in your system's PATH?")
#         return None

# def _genai_upload_and_wait(path: str, *, timeout_s: int = 60):
#     """Upload a file to Gemini and wait for it to become ACTIVE. Returns the File or None."""
#     print(f"Uploading {path} to Gemini...")
#     try:
#         f = genai.upload_file(path)
#         start = time.time()
#         while True:
#             f = genai.get_file(f.name)
#             if f.state.name == 'ACTIVE':
#                 print("...Upload successful and file is active.")
#                 return f
#             if f.state.name == 'FAILED':
#                 print("...Upload failed.")
#                 return None
#             if time.time() - start > timeout_s:
#                 print("...Upload timed out while waiting for 'ACTIVE' state.")
#                 return None
#             time.sleep(1)
#     except Exception as e:
#         print(f"...An exception occurred during upload: {e}")
#         return None

# @app.route('/stt/start', methods=['POST'])
# def stt_start():
#     sid = str(uuid.uuid4())
#     flask_session["stt_sid"] = sid
#     SESSIONS[f"stt:{sid}"] = {"chunks": []}
#     return jsonify({"stt_sid": sid})

# @app.route('/stt/chunk', methods=['POST'])
# def stt_chunk():
#     sid = flask_session.get("stt_sid")
#     stt_key = f"stt:{sid}" if sid else None
#     if not stt_key or not SESSIONS.get(stt_key): return jsonify({"ok": False, "error": "no stt session"}), 409
#     if 'audio' not in request.files: return jsonify({"error": "no audio chunk"}), 400
#     fs = request.files['audio']
#     path = _save_chunk(fs)
#     SESSIONS[stt_key]["chunks"].append(path)
#     return jsonify({"ok": True, "chunks": len(SESSIONS[stt_key]["chunks"])})

# @app.route('/stt/stop', methods=['POST'])
# def stt_stop():
#     sid = flask_session.get("stt_sid")
#     stt_key = f"stt:{sid}" if sid else None
#     if not stt_key or not SESSIONS.get(stt_key): return jsonify({"text": ""})

#     data = SESSIONS.pop(stt_key)
#     chunks: list[str] = data.get("chunks", [])
#     if not chunks: return jsonify({"text": ""})

#     webm_path = os.path.join(AUDIO_TMP, f"{sid}.webm")
#     _concat_chunks(chunks, webm_path)
    
#     wav_path = _convert_to_wav(webm_path)
#     if not wav_path:
#         return jsonify({"text": "", "error": "Failed to convert audio file."}), 200

#     text = ""
#     try:
#         backend = AppConfig.STT_BACKEND
#         print(f"STT backend is set to: '{backend}'")
        
#         if backend == 'gemini':
#             if genai is None:
#                 return jsonify({"text": "", "error": "Gemini STT requires google-generativeai. Please install it or switch backend."}), 200
#             genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
#             stt_model = genai.GenerativeModel("gemini-1.5-flash-latest")
#             uploaded_file = _genai_upload_and_wait(wav_path)
#             if uploaded_file:
#                 print("Transcribing with Gemini...")
#                 response = stt_model.generate_content([
#                     uploaded_file, "Transcribe the spoken audio to plain text only."
#                 ])
#                 text = getattr(response, 'text', '') or ''
#                 genai.delete_file(uploaded_file.name)
        
#         elif backend == 'whisper':
#             print("Transcribing with Whisper...")
#             client = openai.OpenAI()
#             with open(wav_path, 'rb') as audio_file:
#                 tr = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
#                 text = tr.text if hasattr(tr, 'text') else ''
#         elif backend == 'whisper_local':
#             print("Transcribing with local Whisper...")
#             import whisper
#             model = whisper.load_model("tiny") 
#             result = model.transcribe(wav_path)
#             text = result.get("text", "")
        
#         print(f"Transcription result: '{text}'")
#         flask_session.pop("stt_sid", None)
#         return jsonify({"text": text})
        
#     except Exception as e:
#         print(f"An error occurred during transcription: {e}")
#         return jsonify({"text": "", "error": str(e)}), 200
#     finally:
        # Cleanup all temporary files
        files_to_delete = chunks + [webm_path, wav_path]
        for p in files_to_delete:
            if p and os.path.exists(p):
                try: os.remove(p)
                except Exception: pass

# ----------------------------
# Voice Metadata Helper Function
# ----------------------------

def _save_voice_metadata(metadata: dict, transcript: str) -> str:
    """Save voice metadata to a JSON file without saving audio."""
    try:
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        filename = f"voice_metadata_{timestamp}.json"
        filepath = os.path.join(VOICE_METADATA_FOLDER, filename)
        
        # Build full transcript from words to ensure consistency
        words_list = metadata.get('words', [])
        if words_list:
            # Always generate transcript from word metadata to ensure it matches
            generated_transcript = ' '.join(w.get('punctuated_word', w.get('word', '')) for w in words_list)
            # Use generated transcript if provided transcript doesn't match word count
            if not transcript or len(transcript.split()) != len(words_list):
                transcript = generated_transcript
        
        # Add transcript and analysis
        full_metadata = {
            "transcript": transcript,
            "session_start": metadata.get('session_start'),
            "total_duration_seconds": metadata.get('total_duration', 0.0),
            "word_count": len(words_list),
            "utterance_count": len(metadata.get('utterances', [])),
            "words": words_list,
            "utterances": metadata.get('utterances', []),
            "saved_at": datetime.utcnow().isoformat() + 'Z'
        }
        
        # Calculate pauses (gaps between words)
        words = metadata.get('words', [])
        pauses = []
        for i in range(len(words) - 1):
            gap = words[i + 1]['start'] - words[i]['end']
            if gap > 0.3:  # Significant pause (>300ms)
                pauses.append({
                    'after_word': words[i]['punctuated_word'],
                    'before_word': words[i + 1]['punctuated_word'],
                    'duration_seconds': gap,
                    'timestamp': words[i]['end']
                })
        full_metadata['pauses'] = pauses
        
        # Create descriptive transcription with pause markers
        descriptive_transcript = ""
        for i, word_data in enumerate(words):
            descriptive_transcript += word_data['punctuated_word']
            if i < len(words) - 1:
                gap = words[i + 1]['start'] - word_data['end']
                if gap > 1.0:  # Long pause
                    descriptive_transcript += "... "
                elif gap > 0.5:  # Medium pause
                    descriptive_transcript += ".. "
                elif gap > 0.3:  # Short pause
                    descriptive_transcript += ". "
                else:
                    descriptive_transcript += " "
        full_metadata['descriptive_transcript'] = descriptive_transcript.strip()
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(full_metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Voice metadata saved to {filepath}")
        return descriptive_transcript.strip()  # Return descriptive transcript instead of filepath
    except Exception as e:
        print(f"Error saving voice metadata: {e}")
        return ""

# ----------------------------
# Deepgram STT (Live) route
# ----------------------------

class ConversationManager:
    def __init__(self):
        self.transcript = ""
        self.last_speech_time = time.time()
        self.silence_threshold = 2.0  # seconds of silence before considering interjection
        self.conversation_context = []
        self.voice_metadata = {  # Store detailed voice metadata
            "words": [],  # word-level timestamps
            "utterances": [],  # utterance boundaries
            "session_start": datetime.utcnow().isoformat() + 'Z',
            "total_duration": 0.0
        }
        
    def should_interject(self, transcript_text, is_final):
        """Determine if the LLM should interject based on speech patterns"""
        current_time = time.time()
        
        # Update last speech time if we got new content
        if transcript_text.strip():
            self.last_speech_time = current_time
            
        # Check for natural pause points
        silence_duration = current_time - self.last_speech_time
        
        # Interject if:
        # 1. There's been silence for threshold duration AND we have content
        # 2. User asked a direct question
        # 3. User said something that seems to expect a response
        
        if silence_duration > self.silence_threshold and self.transcript.strip():
            return True
            
        if is_final and transcript_text.strip():
            # Check for question patterns
            question_indicators = ['?', 'what do you think', 'right?', 'you know?', 'can you', 'would you', 'should i']
            text_lower = transcript_text.lower()
            if any(indicator in text_lower for indicator in question_indicators):
                return True
                
        return False
    
    async def get_llm_response(self, transcript):
        """Get response from the configured LLM"""
        try:
            backend = AppConfig.MODEL_BACKEND
            
            if backend == 'huggingface':
                hf_token = AppConfig.HUGGING_FACE_API_TOKEN
                if hf_token and AppConfig.HF_MODEL_ID:
                    client = InferenceClient(token=hf_token)
                    response = client.chat_completion(
                        messages=[
                            {"role": "system", "content": "You are having a natural conversation. Respond naturally and briefly to what the user just said. Keep responses conversational and under 2 sentences."},
                            {"role": "user", "content": transcript}
                        ],
                        model=AppConfig.HF_MODEL_ID,
                        max_tokens=100,
                        temperature=AppConfig.HF_TEMPERATURE or None,
                    )
                    if response.choices:
                        return response.choices[0].message.content
                        
            elif backend == 'gemini':
                api_key = os.environ.get("GOOGLE_API_KEY") if genai else None
                if api_key and genai is not None:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-1.5-flash-latest')
                    response = model.generate_content(f"Respond naturally and briefly to: {transcript}")
                    return getattr(response, 'text', None)
                    
        except Exception as e:
            print(f"LLM response error: {e}")
            
        return None

@sock.route('/stt/deepgram')
def stt_deepgram(ws):
    if not AppConfig.DEEPGRAM_API_KEY:
        print("No DEEPGRAM_API_KEY found; closing WebSocket.")
        ws.send(json.dumps({"error": "Deepgram not configured"}))
        return
        
    print("Deepgram streaming WebSocket connected")
    conversation = ConversationManager()
    
    # Create a thread to handle the Deepgram connection
    import threading
    
    # Event to signal when Deepgram connection is ready
    deepgram_ready = threading.Event()
    
    def deepgram_handler():
        try:
            # Connect to Deepgram's streaming API with word-level timestamps and utterances
            # filler_words=true captures "uh", "um", etc.
            # disfluencies=true captures stutters and repetitions
            deepgram_url = f"wss://api.deepgram.com/v1/listen?model=nova-2&language=en-US&smart_format=true&interim_results=true&endpointing=300&utterances=true&punctuate=true&diarize=false&filler_words=true&disfluencies=true"
            
            import websocket
            
            def on_message(ws_dg, message):
                try:
                    data = json.loads(message)
                    
                    # Handle metadata messages (utterance end events)
                    if data.get('type') == 'UtteranceEnd':
                        # Capture utterance metadata
                        if 'channel' in data:
                            utterance_data = {
                                'type': 'utterance_end',
                                'timestamp': datetime.utcnow().isoformat() + 'Z'
                            }
                            conversation.voice_metadata['utterances'].append(utterance_data)
                    
                    if 'channel' in data:
                        alternative = data['channel']['alternatives'][0]
                        transcript = alternative.get('transcript', '')
                        is_final = data.get('is_final', False)
                        
                        # Extract word-level timestamps if available
                        words_data = None
                        if is_final and 'words' in alternative:
                            words_data = []
                            for word_data in alternative['words']:
                                word_metadata = {
                                    'word': word_data.get('word', ''),
                                    'start': word_data.get('start', 0.0),
                                    'end': word_data.get('end', 0.0),
                                    'confidence': word_data.get('confidence', 0.0),
                                    'punctuated_word': word_data.get('punctuated_word', word_data.get('word', ''))
                                }
                                conversation.voice_metadata['words'].append(word_metadata)
                                words_data.append(word_metadata)
                                # Update total duration
                                if word_data.get('end', 0.0) > conversation.voice_metadata['total_duration']:
                                    conversation.voice_metadata['total_duration'] = word_data.get('end', 0.0)
                        
                        if transcript:
                            # Log filler words and disfluencies for debugging
                            if any(filler in transcript.lower() for filler in ['uh', 'um', 'hmm', 'ah', 'er']):
                                print(f"📝 Filler words detected: {transcript}")
                            
                            # Update conversation transcript
                            if is_final:
                                if conversation.transcript:
                                    conversation.transcript += " " + transcript
                                else:
                                    conversation.transcript = transcript
                            
                            # Send transcript to client with word-level metadata
                            message = {
                                "transcript": conversation.transcript + (" " + transcript if not is_final else ""),
                                "is_final": is_final,
                                "interim": not is_final
                            }
                            
                            # Include word-level data if available
                            if words_data:
                                message["words"] = words_data
                            
                            ws.send(json.dumps(message))
                            
                            # DISABLED: Automatic interjection - now using manual control
                            # User will press mic button again to send prompt
                            # if conversation.should_interject(transcript, is_final):
                            #     ... automatic response logic ...
                                
                except Exception as e:
                    print(f"Deepgram message error: {e}")
            
            def on_error(ws_dg, error):
                print(f"Deepgram WebSocket error: {error}")
            
            def on_close(ws_dg, close_status_code, close_msg):
                print("Deepgram WebSocket closed")
            
            def on_open(ws_dg):
                print("Connected to Deepgram")
                deepgram_ready.set()  # Signal that connection is ready
                ws.send(json.dumps({"status": "connected", "message": "Real-time transcription with AI interjection ready"}))
            
            # Create Deepgram WebSocket connection
            deepgram_ws = websocket.WebSocketApp(
                deepgram_url,
                header={"Authorization": f"Token {AppConfig.DEEPGRAM_API_KEY}"},
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            # Forward audio from client to Deepgram
            def forward_audio():
                try:
                    # Wait for Deepgram connection to be ready (with timeout)
                    if not deepgram_ready.wait(timeout=10):
                        print("Deepgram connection timeout")
                        return
                    
                    print("Starting audio forwarding loop...")
                    while True:
                        try:
                            # Use timeout to avoid blocking forever
                            data = ws.receive(timeout=30)  # 30 second timeout for receiving data
                            if data is None:
                                print("Received None from client WebSocket, ending forwarding")
                                break
                            
                            # Handle control messages (text/JSON)
                            if isinstance(data, str):
                                try:
                                    control_msg = json.loads(data)
                                    if control_msg.get('action') == 'finalize':
                                        print(f"User manually finalized prompt: '{conversation.transcript}'")
                                        
                                        # Save voice metadata (includes descriptive transcript for metadata file)
                                        _save_voice_metadata(conversation.voice_metadata, conversation.transcript)
                                        
                                        # Use plain transcript for display (without pause markers)
                                        current_transcript = conversation.transcript
                                        
                                        # Reset for next prompt
                                        conversation.transcript = ""
                                        conversation.voice_metadata = {
                                            "words": [],
                                            "utterances": [],
                                            "session_start": datetime.utcnow().isoformat() + 'Z',
                                            "total_duration": 0.0
                                        }
                                        
                                        # Get LLM response
                                        def get_response():
                                            import asyncio
                                            loop = asyncio.new_event_loop()
                                            asyncio.set_event_loop(loop)
                                            response = loop.run_until_complete(conversation.get_llm_response(current_transcript))
                                            if response:
                                                try:
                                                    ws.send(json.dumps({
                                                        "llm_response": response,
                                                        "user_transcript": current_transcript,  # Plain transcript without pause markers
                                                        "type": "manual_submit"
                                                    }))
                                                    print(f"✓ Sent LLM response successfully")
                                                except Exception as e:
                                                    print(f"Failed to send LLM response (connection may be closed): {e}")
                                            loop.close()
                                        
                                        threading.Thread(target=get_response, daemon=True).start()
                                except json.JSONDecodeError:
                                    print(f"Invalid control message: {data}")
                            
                            # Handle audio data (binary)
                            elif isinstance(data, bytes):
                                if deepgram_ws.sock and deepgram_ws.sock.connected:
                                    deepgram_ws.send(data, websocket.ABNF.OPCODE_BINARY)
                                else:
                                    print("Deepgram socket not connected")
                                    break
                        except Exception as e:
                            # Check if it's just a timeout or actual error
                            if "timeout" in str(e).lower():
                                continue  # Keep waiting for data
                            else:
                                raise
                except Exception as e:
                    print(f"Audio forwarding error: {e}")
                finally:
                    print("Closing Deepgram WebSocket")
                    deepgram_ws.close()
            
            # Start audio forwarding in a separate thread
            audio_thread = threading.Thread(target=forward_audio, daemon=True)
            audio_thread.start()
            
            # Run Deepgram WebSocket
            deepgram_ws.run_forever()
            
        except Exception as e:
            print(f"Deepgram handler error: {e}")
            ws.send(json.dumps({"error": str(e)}))
    
    # Start Deepgram handler in a separate thread
    dg_thread = threading.Thread(target=deepgram_handler, daemon=True)
    dg_thread.start()
    
    # Keep the main WebSocket alive
    try:
        dg_thread.join()
    except Exception as e:
        print(f"Main WebSocket error: {e}")


if __name__ == "__main__":
    # Note: `app.run` is not compatible with Flask-Sock
    # Use a production WSGI server like Gunicorn or uWSGI instead.
    # For development, you can use the werkzeug development server directly.
    from werkzeug.serving import run_simple
    run_simple('127.0.0.1', 5001, app, use_debugger=True, use_reloader=True)