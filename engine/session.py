from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any

from models.base import load_model_provider
from games.base_game import SimpleTurnGame
from engine.prompts import build_turn_messages


@dataclass
class PlayerConfig:
    kind: str  # 'model' or 'human'


class GameSession:
    def __init__(self, game: SimpleTurnGame, p0: PlayerConfig, p1: PlayerConfig):
        self.game = game
        self.players = [p0, p1]
        self.model = None

    def ensure_model(self):
        if self.model is None:
            self.model = load_model_provider()

    def step(self) -> Dict[str, Any]:
        """Advance one turn if the current player is a model; otherwise wait for human input."""
        if self.game.is_over():
            return {"status": "done", "message": "Game is over."}

        current = self.game.current_player
        cfg = self.players[current]
        if cfg.kind == "human":
            return {"status": "awaiting_human"}

        # model turn
        self.ensure_model()
        state_text = self.game.render()
        actions = self.game.get_available_actions()
        messages = build_turn_messages(state_text, actions)
        action = self.model.generate(messages)
        ok, msg = self.game.apply_action(action)
        return {
            "status": "ok" if ok else "invalid",
            "action": action,
            "message": msg,
        }

    def apply_human_action(self, action: str) -> Dict[str, Any]:
        if self.game.is_over():
            return {"status": "done", "message": "Game is over."}
        current = self.game.current_player
        cfg = self.players[current]
        if cfg.kind != "human":
            return {"status": "error", "message": "Not human's turn."}
        ok, msg = self.game.apply_action(action)
        return {
            "status": "ok" if ok else "invalid",
            "action": action,
            "message": msg,
        }

    def snapshot(self) -> Dict[str, Any]:
        return {
            "over": self.game.is_over(),
            "state": self.game.get_state(),
            "render": self.game.render(),
            "to_move": self.game.current_player,
            "actions": self.game.get_available_actions(),
        }


if __name__ == "__main__":
    import argparse
    from games.template_game import MyTemplateGame

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="noop")
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    # Allow quick override via env
    import os
    os.environ.setdefault("MODEL_BACKEND", args.backend)

    game = MyTemplateGame()
    game.reset()
    session = GameSession(game, PlayerConfig("model"), PlayerConfig("model"))

    for i in range(args.steps):
        print("\n== Step", i)
        print(json.dumps(session.snapshot(), indent=2))
        out = session.step()
        print("Turn result:", out)
        if session.game.is_over():
            print("Game over!")
            break
