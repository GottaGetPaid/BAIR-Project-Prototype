from __future__ import annotations
from flask import Flask, render_template, request, redirect, url_for, session as flask_session, jsonify
import uuid

from config import AppConfig
from games.template_game import MyTemplateGame
from engine.session import GameSession, PlayerConfig

app = Flask(__name__)
app.secret_key = AppConfig.SECRET_KEY

# In-memory store for demo purposes
SESSIONS = {}


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
    return render_template("index.html", snapshot=snap)


@app.route("/start", methods=["POST"]) 
def start():
    sid = str(uuid.uuid4())
    flask_session["sid"] = sid
    SESSIONS[sid] = new_session()
    return redirect(url_for("play"))


@app.route("/play")
def play():
    sid = flask_session.get("sid")
    if not sid or sid not in SESSIONS:
        return redirect(url_for("index"))
    sess = SESSIONS[sid]
    snap = sess.snapshot()
    return render_template("play.html", snapshot=snap)


@app.route("/step", methods=["POST"])  # model step or apply human action
def step():
    sid = flask_session.get("sid")
    if not sid or sid not in SESSIONS:
        return jsonify({"error": "no session"}), 400
    sess = SESSIONS[sid]

    action = request.form.get("action", "").strip()
    if action:
        out = sess.apply_human_action(action)
    else:
        out = sess.step()
    return jsonify({"turn": out, "snapshot": sess.snapshot()})


if __name__ == "__main__":
    app.run(debug=True)
