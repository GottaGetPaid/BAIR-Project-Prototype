from __future__ import annotations
from flask import Flask, render_template, request, redirect, url_for, session as flask_session, jsonify
from flask_sock import Sock
import uuid

import sys
import os
import json
import re
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
import subprocess # <-- NEW IMPORT

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from config import AppConfig

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
UPLOAD_FOLDER = 'uploaded_files'
QUERIES_FOLDER = 'queries'
AUDIO_TMP = os.path.join(UPLOAD_FOLDER, 'audio_tmp')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(QUERIES_FOLDER, exist_ok=True)
os.makedirs(AUDIO_TMP, exist_ok=True)
REPO_ROOT = parent_dir
TEST_SESSIONS_FOLDER = os.path.join(REPO_ROOT, 'test-sessions')
os.makedirs(TEST_SESSIONS_FOLDER, exist_ok=True)
UPLOAD_FILE_COUNT = len(os.listdir(UPLOAD_FOLDER))

@app.route("/")
def index():
    return render_template("upload.html", stt_backend=AppConfig.STT_BACKEND)

# Your existing /upload and /query routes are fine...
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({"error": "No selected file"}), 400
    if file:
        try:
            raw_text = file.read().decode('utf-8')
            global UPLOAD_FILE_COUNT
            with open(os.path.join(UPLOAD_FOLDER, f"context_file_{UPLOAD_FILE_COUNT}"), 'w', encoding='utf-8') as f:
                f.write(raw_text)
                UPLOAD_FILE_COUNT += 1
            return jsonify({"message": "Context file successfully saved"}), 200
        except Exception as e: return jsonify({"error": str(e)}), 500
    return jsonify({"error": "An unexpected error occurred."}), 500

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

# ----------------------------
# Save chat route
# ----------------------------

def _slugify(value: str) -> str:
    value = value or ''
    value = re.sub(r'[^A-Za-z0-9_.-]+', '-', value)
    return value.strip('-') or 'unknown'

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

def _save_chunk(file_storage) -> str:
    filename = secure_filename(file_storage.filename or f"chunk_{datetime.utcnow().timestamp()}.webm")
    path = os.path.join(AUDIO_TMP, filename)
    file_storage.save(path)
    return path

def _concat_chunks(paths: list[str], out_path: str) -> str:
    with open(out_path, 'wb') as w:
        for p in paths:
            with open(p, 'rb') as r: w.write(r.read())
    return out_path

# --- NEW HELPER FUNCTION TO CONVERT AUDIO ---
def _convert_to_wav(input_path: str) -> Optional[str]:
    """Converts an audio file to a standard WAV format using FFmpeg."""
    if not os.path.exists(input_path):
        return None
    output_path = input_path.rsplit('.', 1)[0] + ".wav"
    print(f"Converting {input_path} to {output_path} using FFmpeg...")
    try:
        # This command converts the audio to mono, 16000Hz sample rate WAV
        command = [
            'ffmpeg',
            '-i', input_path,
            '-ac', '1',
            '-ar', '16000',
            '-y', # Overwrite output file if it exists
            output_path
        ]
        # Use DEVNULL to hide ffmpeg's verbose output from the console
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("...Conversion successful.")
        return output_path
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"!!! FFmpeg ERROR: {e}")
        print("!!! Is FFmpeg installed and in your system's PATH?")
        return None

def _genai_upload_and_wait(path: str, *, timeout_s: int = 60):
    """Upload a file to Gemini and wait for it to become ACTIVE. Returns the File or None."""
    print(f"Uploading {path} to Gemini...")
    try:
        f = genai.upload_file(path)
        start = time.time()
        while True:
            f = genai.get_file(f.name)
            if f.state.name == 'ACTIVE':
                print("...Upload successful and file is active.")
                return f
            if f.state.name == 'FAILED':
                print("...Upload failed.")
                return None
            if time.time() - start > timeout_s:
                print("...Upload timed out while waiting for 'ACTIVE' state.")
                return None
            time.sleep(1)
    except Exception as e:
        print(f"...An exception occurred during upload: {e}")
        return None

@app.route('/stt/start', methods=['POST'])
def stt_start():
    sid = str(uuid.uuid4())
    flask_session["stt_sid"] = sid
    SESSIONS[f"stt:{sid}"] = {"chunks": []}
    return jsonify({"stt_sid": sid})

@app.route('/stt/chunk', methods=['POST'])
def stt_chunk():
    sid = flask_session.get("stt_sid")
    stt_key = f"stt:{sid}" if sid else None
    if not stt_key or not SESSIONS.get(stt_key): return jsonify({"ok": False, "error": "no stt session"}), 409
    if 'audio' not in request.files: return jsonify({"error": "no audio chunk"}), 400
    fs = request.files['audio']
    path = _save_chunk(fs)
    SESSIONS[stt_key]["chunks"].append(path)
    return jsonify({"ok": True, "chunks": len(SESSIONS[stt_key]["chunks"])})

@app.route('/stt/stop', methods=['POST'])
def stt_stop():
    sid = flask_session.get("stt_sid")
    stt_key = f"stt:{sid}" if sid else None
    if not stt_key or not SESSIONS.get(stt_key): return jsonify({"text": ""})

    data = SESSIONS.pop(stt_key)
    chunks: list[str] = data.get("chunks", [])
    if not chunks: return jsonify({"text": ""})

    webm_path = os.path.join(AUDIO_TMP, f"{sid}.webm")
    _concat_chunks(chunks, webm_path)
    
    # --- CONVERT TO WAV BEFORE UPLOADING ---
    wav_path = _convert_to_wav(webm_path)
    if not wav_path:
        return jsonify({"text": "", "error": "Failed to convert audio file."}), 200

    text = ""
    try:
        backend = AppConfig.STT_BACKEND
        print(f"STT backend is set to: '{backend}'")
        
        if backend == 'gemini':
            if genai is None:
                return jsonify({"text": "", "error": "Gemini STT requires google-generativeai. Please install it or switch backend."}), 200
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
            stt_model = genai.GenerativeModel("gemini-1.5-flash-latest")
            uploaded_file = _genai_upload_and_wait(wav_path) # Upload the .wav file
            if uploaded_file:
                print("Transcribing with Gemini...")
                response = stt_model.generate_content([
                    uploaded_file, "Transcribe the spoken audio to plain text only."
                ])
                text = getattr(response, 'text', '') or ''
                genai.delete_file(uploaded_file.name)
        
        elif backend == 'whisper':
            print("Transcribing with Whisper...")
            client = openai.OpenAI()
            with open(wav_path, 'rb') as audio_file:
                tr = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
                text = tr.text if hasattr(tr, 'text') else ''
        elif backend == 'whisper_local':
            print("Transcribing with local Whisper...")
            import whisper
            model = whisper.load_model("tiny") 
            result = model.transcribe(wav_path)
            text = result.get("text", "")
        
        print(f"Transcription result: '{text}'")
        flask_session.pop("stt_sid", None)
        return jsonify({"text": text})
        
    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        return jsonify({"text": "", "error": str(e)}), 200
    finally:
        # Cleanup all temporary files
        files_to_delete = chunks + [webm_path, wav_path]
        for p in files_to_delete:
            if p and os.path.exists(p):
                try: os.remove(p)
                except Exception: pass

# ----------------------------
# Deepgram STT (Live) route
# ----------------------------

class ConversationManager:
    def __init__(self):
        self.transcript = ""
        self.last_speech_time = time.time()
        self.silence_threshold = 2.0  # seconds of silence before considering interjection
        self.conversation_context = []
        
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
    
    def deepgram_handler():
        try:
            # Connect to Deepgram's streaming API
            deepgram_url = f"wss://api.deepgram.com/v1/listen?model=nova-2&language=en-US&smart_format=true&interim_results=true&endpointing=300"
            
            import websocket
            
            def on_message(ws_dg, message):
                try:
                    data = json.loads(message)
                    if 'channel' in data:
                        transcript = data['channel']['alternatives'][0]['transcript']
                        is_final = data.get('is_final', False)
                        
                        if transcript:
                            # Update conversation transcript
                            if is_final:
                                if conversation.transcript:
                                    conversation.transcript += " " + transcript
                                else:
                                    conversation.transcript = transcript
                            
                            # Send transcript to client
                            ws.send(json.dumps({
                                "transcript": conversation.transcript + (" " + transcript if not is_final else ""),
                                "is_final": is_final,
                                "interim": not is_final
                            }))
                            
                            # Check if LLM should interject
                            if conversation.should_interject(transcript, is_final):
                                print(f"LLM interjecting based on: '{conversation.transcript}'")
                                
                                # Get LLM response in a separate thread to avoid blocking
                                def get_response():
                                    import asyncio
                                    loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(loop)
                                    response = loop.run_until_complete(conversation.get_llm_response(conversation.transcript))
                                    if response:
                                        ws.send(json.dumps({
                                            "llm_response": response,
                                            "type": "interjection"
                                        }))
                                        # Reset transcript after response
                                        conversation.transcript = ""
                                    loop.close()
                                
                                threading.Thread(target=get_response, daemon=True).start()
                                
                except Exception as e:
                    print(f"Deepgram message error: {e}")
            
            def on_error(ws_dg, error):
                print(f"Deepgram WebSocket error: {error}")
            
            def on_close(ws_dg, close_status_code, close_msg):
                print("Deepgram WebSocket closed")
            
            def on_open(ws_dg):
                print("Connected to Deepgram")
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
                    while True:
                        data = ws.receive()
                        if data and deepgram_ws.sock and deepgram_ws.sock.connected:
                            deepgram_ws.send(data, websocket.ABNF.OPCODE_BINARY)
                        else:
                            break
                except Exception as e:
                    print(f"Audio forwarding error: {e}")
                finally:
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