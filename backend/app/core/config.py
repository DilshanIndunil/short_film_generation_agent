import os
from pydantic_settings import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ROOT_DIR = BASE_DIR.parent


class Settings(BaseSettings):
    # Project & Server Settings
    PROJECT_NAME: str = "FILM AGENT — Qwen-Image-2.1 & FLUX.1 AI Studio"
    API_V1_STR: str = "/api/v1"
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000

    # Storage & Paths
    DATA_DIR: Path = ROOT_DIR / "data"
    CHARACTERS_DIR: Path = DATA_DIR / "characters"
    KEYFRAMES_DIR: Path = DATA_DIR / "keyframes"
    VIDEOS_DIR: Path = DATA_DIR / "videos"
    GENERATED_DIR: Path = DATA_DIR / "generated_images"
    HISTORY_DIR: Path = DATA_DIR / "history"
    MODELS_DIR: Path = BASE_DIR / "models"
    SHARED_MODELS_DIR: Path = BASE_DIR / "models"

    # AI Planner / LLM (Loaded securely from .env)
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3-flash-preview"
    HF_TOKEN: str = ""

    # Local GPU Generation Settings (Loaded from .env)
    FLUX_MODEL_VARIANT: str = "Qwen-Image-2.1 (Local GPU High-Precision)"
    COMFYUI_URL: str = "http://127.0.0.1:8188"
    QWEN_DEFAULT_STEPS: int = 16
    FLUX_SCHNELL_DEFAULT_STEPS: int = 4
    FLUX_DEV_DEFAULT_STEPS: int = 20
    LORA_TRIGGER_MALE: str = "arjun_male"
    LORA_TRIGGER_FEMALE: str = "maya_female"

    class Config:
        env_file = (str(ROOT_DIR / ".env"), str(BASE_DIR / ".env"), ".env")
        env_file_encoding = "utf-8"
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
