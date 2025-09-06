import os
from dotenv import load_dotenv

load_dotenv()

class AppConfig:
    MODEL_BACKEND = os.getenv("MODEL_BACKEND", "noop").lower()

    # HF
    HF_MODEL_ID = os.getenv("HF_MODEL_ID", "")
    HF_DEVICE = os.getenv("HF_DEVICE", "cpu")
    HF_MAX_NEW_TOKENS = int(os.getenv("HF_MAX_NEW_TOKENS", "128"))
    HF_TEMPERATURE = float(os.getenv("HF_TEMPERATURE", "0.7"))

    # Flask
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "dev-secret")

    # Speech-to-Text (STT)
    # Options: "gemini" (Google Generative AI), "whisper" (OpenAI Whisper)
    STT_BACKEND = os.getenv("STT_BACKEND", "gemini").lower()

