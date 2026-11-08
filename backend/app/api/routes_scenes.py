from fastapi import APIRouter, HTTPException
from app.schemas.schemas import (
    MultiScenePlanRequest,
    MultiScenePlanResponse,
    GenerateKeyframesRequest,
    GenerateKeyframesResponse,
    RegenerateSingleKeyframeRequest,
    KeyframeItem,
    GenerateVideosRequest,
    GenerateVideosResponse,
    RegenerateSingleVideoRequest,
    VideoClipItem
)
from app.services.scene_planner import scene_planner_agent
from app.services.flux_service import flux_service
from app.services.wan_video_service import wan_video_service

router = APIRouter(prefix="/scenes", tags=["Film Agent Scene Planner & FLUX Studio"])

@router.post("/plan-multi", response_model=MultiScenePlanResponse)
async def plan_multi_scenes_endpoint(payload: MultiScenePlanRequest):
    """
    Card 1 -> Card 2: Converts raw user scenario(s) into FLUX.1 / FLUX.2 optimized cinematic prompts.
    """
    try:
        response = await scene_planner_agent.plan_multi_scenes(
            scenes=[s.model_dump() if hasattr(s, "model_dump") else s.dict() for s in payload.scenes],
            aspect_ratio=payload.aspect_ratio,
            flux_model=payload.flux_model,
            characters=payload.characters
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-keyframes", response_model=GenerateKeyframesResponse)
async def generate_keyframes_endpoint(payload: GenerateKeyframesRequest):
    """
    Card 2 -> Card 3: Synthesizes high-res images on Local GPU (RTX 4060) from optimized FLUX prompts.
    """
    try:
        result = await flux_service.generate_run_keyframes(
            scenes=payload.scenes,
            flux_model=payload.flux_model,
            aspect_ratio=payload.aspect_ratio,
            steps=payload.steps
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/regenerate-keyframe", response_model=KeyframeItem)
async def regenerate_keyframe_endpoint(payload: RegenerateSingleKeyframeRequest):
    """
    Card 3: Regenerates a single scene image on Local GPU and replaces it in the run folder.
    """
    try:
        updated_frame = await flux_service.regenerate_single_frame(
            run_id=payload.run_id,
            scene_number=payload.scene_number,
            prompt=payload.prompt,
            raw_prompt=payload.raw_prompt,
            camera=payload.camera,
            lighting=payload.lighting,
            aspect_ratio=payload.aspect_ratio or "16:9",
            flux_model=payload.flux_model or "FLUX.1-dev",
            steps=payload.steps
        )
        return updated_frame
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-videos", response_model=GenerateVideosResponse)
async def generate_videos_endpoint(payload: GenerateVideosRequest):
    """
    Card 3 + Card 4 -> Wan 2.2 Video Synthesis:
    Combines each scene's Master Frame Image with Card 4's Wan 2.2 Motion & Voice-Over Script
    to synthesize scene-by-scene .mp4 video clips.
    """
    try:
        result = await wan_video_service.generate_run_videos(
            run_id=payload.run_id,
            wan_model=payload.wan_model,
            aspect_ratio=payload.aspect_ratio,
            scenes=payload.scenes,
            frames=payload.frames
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/regenerate-video", response_model=VideoClipItem)
async def regenerate_video_endpoint(payload: RegenerateSingleVideoRequest):
    """
    Card 4: Regenerates a single scene's Wan 2.2 video clip from its Master Frame Image + edited Motion/VO script.
    """
    try:
        updated_video = await wan_video_service.regenerate_single_video(
            run_id=payload.run_id,
            scene_number=payload.scene_number,
            wan_model=payload.wan_model,
            aspect_ratio=payload.aspect_ratio,
            source_image_url=payload.source_image_url,
            source_image_path=payload.source_image_path,
            video_motion_prompt=payload.video_motion_prompt,
            camera_motion=payload.camera_motion,
            voiceover_speaker=payload.voiceover_speaker,
            voiceover_sinhala=payload.voiceover_sinhala,
            voiceover_english=payload.voiceover_english,
            voice_tone=payload.voice_tone,
            sfx_cue=payload.sfx_cue
        )
        return updated_video
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


