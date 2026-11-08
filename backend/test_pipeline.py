import json
from datetime import datetime
from app.services.scene_planner import scene_planner_agent
from app.services.flux_service import flux_service
from app.api.routes_history import save_history_snapshot, SaveHistoryRequest

def run_coro_sync(coro):
    try:
        coro.send(None)
    except StopIteration as e:
        return e.value

def run_test():
    print("1. Testing Scenario -> FLUX Optimized Prompt (Card 1 -> Card 2)...", flush=True)
    plan = run_coro_sync(scene_planner_agent.plan_multi_scenes(
        scenes=[{
            "id": 1,
            "text": "Two characters meeting at a cozy rain-streaked coffee shop window at night, talking warmly over steaming coffee",
            "duration": "5s"
        }],
        aspect_ratio="16:9",
        flux_model="FLUX.1-dev",
        characters=["male", "female"]
    ))
    scene = plan.scenes[0]
    print(f"Raw Scenario     : {scene.raw_prompt}", flush=True)
    print(f"Optimized Prompt : {scene.optimized_image_prompt}", flush=True)
    print(f"Camera           : {scene.camera}", flush=True)

    print("\n2. Synthesizing Master Image on Local RTX 4060 GPU (Card 2 -> Card 3)...", flush=True)
    kf_res = run_coro_sync(flux_service.generate_run_keyframes(
        scenes=[scene.model_dump()],
        flux_model="FLUX.1-dev",
        aspect_ratio="16:9",
        steps=28
    ))
    print(f"Generated Run ID : {kf_res.run_id}", flush=True)
    print(f"Saved Image Path : {kf_res.frames[0].file_path}", flush=True)

    save_history_snapshot(SaveHistoryRequest(
        run_id=kf_res.run_id,
        title="FLUX.1-dev (16:9 • Coffee Shop Scene)",
        aspect_ratio="16:9",
        flux_model="FLUX.1-dev",
        folder_path=kf_res.folder_path,
        created_at=datetime.now().isoformat(),
        storyboard={
            "scenes": [{"id": 1, "text": scene.raw_prompt, "duration": "5s"}],
            "aspect_ratio": "16:9"
        },
        planned_scenes=[scene.model_dump()],
        keyframes=[f.model_dump() for f in kf_res.frames]
    ))
    print("[SUCCESS] End-to-End Local GPU Pipeline Verified & Saved to History!", flush=True)

if __name__ == "__main__":
    run_test()
