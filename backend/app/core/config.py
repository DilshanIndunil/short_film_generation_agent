import os
from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    PROJECT_NAME: str = "FILM AGENT — FLUX.1 / FLUX.2 AI Studio"
    API_V1_STR: str = "/api/v1"

    # Storage & Paths
    DATA_DIR: Path = BASE_DIR.parent / "data"
    CHARACTERS_DIR: Path = DATA_DIR / "characters"
    KEYFRAMES_DIR: Path = DATA_DIR / "keyframes"
    VIDEOS_DIR: Path = DATA_DIR / "videos"
    GENERATED_DIR: Path = DATA_DIR / "generated_images"
    HISTORY_DIR: Path = DATA_DIR / "history"
    MODELS_DIR: Path = BASE_DIR / "models"

    # Also check shared local GPU models in D:\video-generation-ai\backend\models
    SHARED_MODELS_DIR: Path = Path("D:/video-generation-ai/backend/models")

    # AI Planner / LLM
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-2.0-flash"

    # Local GPU FLUX Settings
    FLUX_MODEL_VARIANT: str = "FLUX.1-schnell (4-bit NF4 / FP8 Local GPU)"
    COMFYUI_URL: str = "http://127.0.0.1:8188"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()

# Ensure directories exist
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(settings.CHARACTERS_DIR, exist_ok=True)
os.makedirs(settings.KEYFRAMES_DIR, exist_ok=True)
os.makedirs(settings.VIDEOS_DIR, exist_ok=True)
os.makedirs(settings.GENERATED_DIR, exist_ok=True)
os.makedirs(settings.HISTORY_DIR, exist_ok=True)
os.makedirs(settings.MODELS_DIR, exist_ok=True)

