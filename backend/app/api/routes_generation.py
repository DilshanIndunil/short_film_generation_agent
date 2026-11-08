from fastapi import APIRouter
from typing import Optional
from pydantic import BaseModel
from app.services.flux_service import flux_service
from app.services.scene_planner import scene_planner_agent

router = APIRouter(prefix="/generate", tags=["FLUX Local GPU Engine"])

class GenerateImageRequest(BaseModel):
    prompt: str
    optimized_prompt: Optional[str] = None
    flux_model: str = "FLUX.1-dev"
    aspect_ratio: str = "16:9"
    steps: int = 28
    guidance_scale: float = 6.5
    seed: Optional[int] = None

@router.post("/image")
async def generate_single_image(payload: GenerateImageRequest):
    """
    Direct endpoint to optimize prompt (if needed) and generate image on Local GPU.
    """
    opt_prompt = payload.optimized_prompt
    scene_info = {}
    if not opt_prompt:
        plan = await scene_planner_agent.plan_multi_scenes(
            scenes=[{"id": 1, "text": payload.prompt, "duration": "5s"}],
            aspect_ratio=payload.aspect_ratio,
            flux_model=payload.flux_model
        )
        if plan.scenes:
            sc = plan.scenes[0]
            opt_prompt = sc.optimized_image_prompt
            scene_info = sc.model_dump()

    result = await flux_service.generate_single_image(
        user_prompt=payload.prompt,
        optimized_prompt=opt_prompt or payload.prompt,
        scene_info=scene_info,
        aspect_ratio=payload.aspect_ratio,
        flux_model=payload.flux_model,
        num_inference_steps=payload.steps,
        guidance_scale=payload.guidance_scale,
        seed=payload.seed
    )
    return result

@router.get("/progress")
def get_live_generation_progress():
    """Returns real-time step-by-step progress for Card 2 optimization and Card 3 Local GPU image inference."""
    return flux_service.progress

@router.get("/image/status")
def get_image_engine_status():
    """Returns GPU device, VRAM, and model status."""
    return flux_service.get_status()

