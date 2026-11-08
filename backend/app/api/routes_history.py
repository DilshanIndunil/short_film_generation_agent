import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/history", tags=["Generation History"])

class HistoryItemSummary(BaseModel):
    run_id: str
    title: str
    created_at: str
    aspect_ratio: str = "16:9"
    flux_model: str = "FLUX.1-dev"
    scenes_count: int = 1
    thumbnail_url: Optional[str] = None
    folder_path: Optional[str] = None

class SaveHistoryRequest(BaseModel):
    run_id: str
    title: Optional[str] = None
    aspect_ratio: str = "16:9"
    flux_model: str = "FLUX.1-dev"
    created_at: Optional[str] = None
    storyboard: Optional[Dict[str, Any]] = None
    planned_scenes: Optional[List[Dict[str, Any]]] = None
    keyframes: Optional[List[Dict[str, Any]]] = None
    videos: Optional[List[Dict[str, Any]]] = None
    folder_path: Optional[str] = None

def _get_history_dir() -> Path:
    settings.HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    return settings.HISTORY_DIR

@router.get("", response_model=List[HistoryItemSummary])
def get_all_history():
    history_dir = _get_history_dir()
    runs_map: Dict[str, HistoryItemSummary] = {}

    for f in history_dir.glob("*.json"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                run_id = data.get("run_id") or f.stem
                frames = data.get("keyframes") or []
                thumb = None
                if frames and len(frames) > 0:
                    thumb = frames[0].get("imageUrl") or frames[0].get("url")

                runs_map[run_id] = HistoryItemSummary(
                    run_id=run_id,
                    title=data.get("title") or f"FLUX Scene ({data.get('aspect_ratio', '16:9')})",
                    created_at=data.get("created_at") or datetime.now().isoformat(),
                    aspect_ratio=data.get("aspect_ratio", "16:9"),
                    flux_model=data.get("flux_model", "FLUX.1-dev"),
                    scenes_count=len(frames) or 1,
                    thumbnail_url=thumb,
                    folder_path=data.get("folder_path")
                )
        except Exception as e:
            logger.warning(f"Error loading history file {f}: {e}")

    if settings.KEYFRAMES_DIR.exists():
        for run_folder in settings.KEYFRAMES_DIR.glob("run_*"):
            if run_folder.is_dir() and run_folder.name not in runs_map:
                meta_file = run_folder / "metadata.json"
                first_img = run_folder / "scene_1_master.png"
                thumb_url = f"/data/keyframes/{run_folder.name}/scene_1_master.png" if first_img.exists() else None

                created_at = datetime.fromtimestamp(run_folder.stat().st_mtime).isoformat()
                aspect_ratio = "16:9"
                flux_model = "FLUX.1-dev"
                scenes_count = 1

                if meta_file.exists():
                    try:
                        with open(meta_file, "r", encoding="utf-8") as mfp:
                            meta = json.load(mfp)
                            aspect_ratio = meta.get("aspect_ratio", "16:9")
                            flux_model = meta.get("flux_model", "FLUX.1-dev")
                            scenes_count = meta.get("total_scenes", 1)
                            created_at = meta.get("created_at", created_at)
                    except Exception:
                        pass

                runs_map[run_folder.name] = HistoryItemSummary(
                    run_id=run_folder.name,
                    title=f"{flux_model} Run ({aspect_ratio})",
                    created_at=created_at,
                    aspect_ratio=aspect_ratio,
                    flux_model=flux_model,
                    scenes_count=scenes_count,
                    thumbnail_url=thumb_url,
                    folder_path=f"data/keyframes/{run_folder.name}"
                )

    return sorted(runs_map.values(), key=lambda x: x.created_at, reverse=True)

@router.get("/{run_id}")
def get_history_detail(run_id: str):
    history_dir = _get_history_dir()
    file_path = history_dir / f"{run_id}.json"
    if file_path.exists():
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    run_folder = settings.KEYFRAMES_DIR / run_id
    if run_folder.exists() and run_folder.is_dir():
        meta_file = run_folder / "metadata.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as mfp:
                meta = json.load(mfp)
            frames = [
                {
                    "scene_number": fr.get("scene_number", idx + 1),
                    "duration": fr.get("duration", "5s"),
                    "camera": fr.get("camera", "50mm Anamorphic"),
                    "lighting": fr.get("lighting", "Cinema Light"),
                    "imageUrl": fr.get("url"),
                    "prompt_used": fr.get("optimized_image_prompt", ""),
                    "raw_prompt": fr.get("user_prompt", ""),
                    "engine_used": f"{meta.get('flux_model', 'FLUX.1-dev')} • RTX 4060 Local GPU",
                    "seed": fr.get("seed")
                }
                for idx, fr in enumerate(meta.get("frames", []))
            ]
            planned = [
                {
                    "id": idx + 1,
                    "scene_number": fr.get("scene_number", idx + 1),
                    "duration": fr.get("duration", "5s"),
                    "raw_prompt": fr.get("user_prompt", ""),
                    "optimized_image_prompt": fr.get("optimized_image_prompt", ""),
                    "character_description": fr.get("character_description", ""),
                    "action": fr.get("action", ""),
                    "camera": fr.get("camera", ""),
                    "camera_motion": "Smooth dolly push-in",
                    "lighting": fr.get("lighting", ""),
                    "environment": fr.get("environment", ""),
                    "flux_model": meta.get("flux_model", "FLUX.1-dev")
                }
                for idx, fr in enumerate(meta.get("frames", []))
            ]
            return {
                "run_id": run_id,
                "title": f"{meta.get('flux_model', 'FLUX.1-dev')} Run ({meta.get('aspect_ratio', '16:9')})",
                "created_at": meta.get("created_at"),
                "aspect_ratio": meta.get("aspect_ratio", "16:9"),
                "flux_model": meta.get("flux_model", "FLUX.1-dev"),
                "folder_path": f"data/keyframes/{run_id}",
                "storyboard": {"scenes": meta.get("storyboard_scenes", []), "aspect_ratio": meta.get("aspect_ratio", "16:9")},
                "planned_scenes": planned,
                "keyframes": frames
            }
    raise HTTPException(status_code=404, detail="Run not found")

@router.post("")
def save_history_snapshot(req: SaveHistoryRequest):
    history_dir = _get_history_dir()
    created_at = req.created_at or datetime.now().isoformat()
    title = req.title or f"{req.flux_model} Scene ({req.aspect_ratio})"
    snapshot_data = {
        "run_id": req.run_id,
        "title": title,
        "created_at": created_at,
        "aspect_ratio": req.aspect_ratio,
        "flux_model": req.flux_model,
        "folder_path": req.folder_path or f"data/keyframes/{req.run_id}",
        "storyboard": req.storyboard or {},
        "planned_scenes": req.planned_scenes or [],
        "keyframes": req.keyframes or [],
        "videos": req.videos or []
    }
    file_path = history_dir / f"{req.run_id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(snapshot_data, f, indent=2)
    return {"status": "success", "run_id": req.run_id, "snapshot": snapshot_data}

