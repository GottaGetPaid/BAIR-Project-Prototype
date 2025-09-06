from __future__ import annotations
from flask import Flask, render_template, request, redirect, url_for, session as flask_session, jsonify
import uuid

import sys
import os
import google.generativeai as genai


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from config import AppConfig
from games.template_game import MyTemplateGame
from engine.session import GameSession, PlayerConfig

app = Flask(__name__)
app.secret_key = AppConfig.SECRET_KEY

# In-memory store for demo purposes
SESSIONS = {}

#number of uploaded context files
#UPLOAD_FILE_COUNT = 0
if not os.path.exists('uploaded_files'):
    os.makedirs('uploaded_files')
UPLOAD_FILE_COUNT = os.listdir('uploaded_files').__len__()

#context file storage dir
UPLOAD_FOLDER = 'uploaded_files'

#queries file storage dir
QUERIES_FOLDER = 'queries'


def new_session():
    game = MyTemplateGame()  # Replace with your game class later
    game.reset()
    sess = GameSession(game, PlayerConfig("human"), PlayerConfig("model"))
    return sess


@app.route("/")
def index():
    sid = flask_session.get("sid")
    sess = SESSIONS.get(sid)
    snap = sess.snapshot() if sess else None
    return render_template("upload.html", snapshot=snap)


# @app.route("/start", methods=["POST"]) 
# def start():
#     sid = str(uuid.uuid4())
#     flask_session["sid"] = sid
#     SESSIONS[sid] = new_session()
#     return redirect(url_for("play"))


# @app.route("/play")
# def play():
#     sid = flask_session.get("sid")
#     if not sid or sid not in SESSIONS:
#         return redirect(url_for("index"))
#     sess = SESSIONS[sid]
#     snap = sess.snapshot()
#     return render_template("play.html", snapshot=snap)


# @app.route("/step", methods=["POST"])  # model step or apply human action
# def step():
#     sid = flask_session.get("sid")
#     if not sid or sid not in SESSIONS:
#         return jsonify({"error": "no session"}), 400
#     sess = SESSIONS[sid]

#     action = request.form.get("action", "").strip()
#     if action:
#         out = sess.apply_human_action(action)
#     else:
#         out = sess.step()
#     return jsonify({"turn": out, "snapshot": sess.snapshot()})


@app.route('/upload', methods=['POST'])
def upload():
    """Receives the file and prints its raw content to the console."""
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        try:
            # Read the file as a string
            raw_text = file.read().decode('utf-8')

            global UPLOAD_FILE_COUNT
            global UPLOAD_FOLDER

            if not os.path.exists(UPLOAD_FOLDER):
                os.makedirs(UPLOAD_FOLDER)

            with open(os.path.join(UPLOAD_FOLDER, f"context_file_{UPLOAD_FILE_COUNT}"), 'w', encoding='utf-8') as f:
                f.write(raw_text)
                UPLOAD_FILE_COUNT += 1
            
            #print(raw_text)
            #print(f'Recieved context file {file.filename}')

            return jsonify({"message": "Context file successfully saved"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return jsonify({"error": "An unexpected error occurred."}), 500

@app.route('/query', methods=['POST'])
def query():
    """Receives the query, stores it in a csv file and returns a string."""
    query_text = request.form.get('query', '').strip()

    if not query_text:
        return jsonify({"error": "No query provided"}), 400

    try:

        if not os.path.exists(QUERIES_FOLDER):
            os.makedirs(QUERIES_FOLDER)

        with open(QUERIES_FOLDER + '/queries.csv', 'a', encoding='utf-8') as f:

            #response_text = f"Query received: {query_text}"

            print("Sending query to Gemini 1.5 Flash model...")

            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(f"Respond to this in exactly 1 word: {query_text}")

            print(f'Query: {query_text}\nResponse: {response.text}')

            f.write(f'{UPLOAD_FILE_COUNT},"{query_text}","{response.text}"\n')

        
            return jsonify({"response": response.text}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    

if __name__ == "__main__":
    app.run(debug=True)
