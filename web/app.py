from __future__ import annotations
from flask import Flask, render_template, request, redirect, url_for, session as flask_session, jsonify
import uuid

import sys
import os
import google.generativeai as genai
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

# In-memory store for demo purposes
SESSIONS = {}

# Setup folders
UPLOAD_FOLDER = 'uploaded_files'
QUERIES_FOLDER = 'queries'
AUDIO_TMP = os.path.join(UPLOAD_FOLDER, 'audio_tmp')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(QUERIES_FOLDER, exist_ok=True)
os.makedirs(AUDIO_TMP, exist_ok=True)
UPLOAD_FILE_COUNT = len(os.listdir(UPLOAD_FOLDER))

@app.route("/")
def index():
    return render_template("upload.html")

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
    try:
        with open(os.path.join(QUERIES_FOLDER, 'queries.csv'), 'a', encoding='utf-8') as f:
            print("Sending query to Gemini 1.5 Flash model...")
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(f"Respond to this in exactly 1 word: {query_text}")
            print(f'Query: {query_text}\nResponse: {response.text}')
            f.write(f'{UPLOAD_FILE_COUNT},"{query_text}","{response.text}"\n')
            return jsonify({"response": response.text}), 200
    except Exception as e: return jsonify({"error": str(e)}), 500

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
        backend = 'whisper_local'
        print(f"STT backend is set to: '{backend}'")
        
        if backend == 'gemini':
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

if __name__ == "__main__":
    app.run(debug=True)