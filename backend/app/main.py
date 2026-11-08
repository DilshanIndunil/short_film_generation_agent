import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse

from app.core.config import settings
from app.api.routes_scenes import router as scenes_router
from app.api.routes_generation import router as generation_router
from app.api.routes_history import router as history_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Film_Agent_FLUX_Studio")

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Scenario -> AI Optimized Prompt -> FLUX.1 / FLUX.2 Local GPU Image Generation Pipeline",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount /data directory (Characters, Keyframes, Generated Images, History)
if os.path.exists(settings.DATA_DIR):
    app.mount("/data", StaticFiles(directory=str(settings.DATA_DIR)), name="data")

# Mount Production React Frontend Assets if built
dist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist"))
dist_assets = os.path.join(dist_dir, "assets")
if os.path.exists(dist_assets):
    app.mount("/assets", StaticFiles(directory=dist_assets), name="assets")

app.include_router(scenes_router, prefix=settings.API_V1_STR)
app.include_router(generation_router, prefix=settings.API_V1_STR)
app.include_router(history_router, prefix=settings.API_V1_STR)

@app.on_event("startup")
def startup_banner():
    banner = """
========================================================================
     FILM AGENT — FLUX.1 / FLUX.2 LOCAL GPU STUDIO (RTX 4060)
========================================================================
   Frontend Studio UI  : http://localhost:5173/ (or http://127.0.0.1:8000/)
   Backend API Gateway : http://127.0.0.1:8000/
   Swagger API Docs    : http://127.0.0.1:8000/docs
   Health Check        : http://127.0.0.1:8000/health
========================================================================
    """
    print(banner)

@app.get("/")
def root():
    dist_index = os.path.join(dist_dir, "index.html")
    if os.path.exists(dist_index):
        return FileResponse(dist_index)
    return RedirectResponse(url="http://localhost:5173/")

@app.get("/health")
def health():
    return {"status": "healthy", "studio": "Film Agent FLUX Local GPU"}

