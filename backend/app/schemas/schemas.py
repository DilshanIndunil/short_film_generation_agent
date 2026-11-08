from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# Multi-Scene Storyboard & Scenario Input Schemas
class SceneInputItem(BaseModel):
    id: int = 1
    text: str = Field(default="", description="Raw scenario or scene description typed by user")
    duration: str = Field(default="5s", description="Target shot duration")
    characters: List[str] = Field(default_factory=lambda: ["male", "female"], description="Selected characters: 'male', 'female', 'both', or 'none'")

class MultiScenePlanRequest(BaseModel):
    flux_model: str = "FLUX.1-dev"
    aspect_ratio: str = "16:9"
    characters: List[str] = Field(default_factory=lambda: ["male", "female"])
    scenes: List[SceneInputItem]

class OptimizedSceneItem(BaseModel):
    id: int
    scene_number: int
    duration: str = "5s"
    raw_prompt: str
    optimized_image_prompt: str = Field(..., description="FLUX.1 / FLUX.2 T5-XXL optimized natural language master prompt")
    character_description: str = Field(default="", description="Consistent character visual profile injected into prompt")
    action: str
    camera: str
    camera_motion: str
    lighting: str
    environment: str
    flux_model: str = "FLUX.1-dev"
    # Card 4: Short Film Video Motion & Voice-Over Fields
    video_motion_prompt: str = Field(default="", description="Wan 2.2 Image-to-Video motion & action prompt")
    voiceover_speaker: str = Field(default="Arjun (Male Lead)", description="Character or Narrator speaking in this scene")
    voiceover_sinhala: str = Field(default="", description="Sinhala / Singlish voice-over line for the short film")
    voiceover_english: str = Field(default="", description="English voice-over / subtitle line for the short film")
    voice_tone: str = Field(default="Warm cinematic tone", description="Emotional tone & delivery direction")
    sfx_cue: str = Field(default="Natural ambient cinema soundscape", description="Ambient sound effects & background score cue")
    wan_model: str = Field(default="Wan2.2-TI2V-5B", description="Target Wan 2.2 Image-to-Video model")

class MultiScenePlanResponse(BaseModel):
    flux_model: str = "FLUX.1-dev"
    aspect_ratio: str = "16:9"
    total_duration_seconds: int = 5
    scenes: List[OptimizedSceneItem]

class SinglePlanInput(BaseModel):
    prompt: str
    flux_model: str = "FLUX.1-dev"
    aspect_ratio: str = "16:9"
    characters: List[str] = Field(default_factory=lambda: ["male", "female"])

# Keyframe / Image Generation Schemas
class KeyframeItem(BaseModel):
    scene_number: int
    duration: str = "5s"
    camera: str = "Cinematic 35mm Anamorphic"
    lighting: str = "Natural Cinematic"
    prompt_used: Optional[str] = None
    raw_prompt: Optional[str] = None
    imageUrl: str
    file_path: Optional[str] = None
    engine_used: str = "FLUX.1 Local GPU (RTX 4060)"
    seed: Optional[int] = None

class GenerateKeyframesRequest(BaseModel):
    flux_model: str = "FLUX.1-dev"
    aspect_ratio: str = "16:9"
    steps: int = 28
    guidance_scale: float = 3.5
    scenes: List[Dict[str, Any]]

class GenerateKeyframesResponse(BaseModel):
    run_id: str
    folder_path: str
    frames: List[KeyframeItem]

class RegenerateSingleKeyframeRequest(BaseModel):
    run_id: str
    scene_number: int
    prompt: Optional[str] = None
    raw_prompt: Optional[str] = None
    camera: Optional[str] = None
    lighting: Optional[str] = None
    aspect_ratio: Optional[str] = "16:9"
    flux_model: Optional[str] = "FLUX.1-dev"
    steps: int = 28
    guidance_scale: float = 3.5

# Card 4: Wan 2.2 Video & Voice-Over Generation Schemas
class VideoClipItem(BaseModel):
    scene_number: int
    duration: str = "5s"
    wan_model: str = "Wan2.2-TI2V-5B"
    videoUrl: str
    file_path: Optional[str] = None
    source_image_url: Optional[str] = None
    video_motion_prompt: str = ""
    camera_motion: str = "Slow dolly push-in"
    voiceover_speaker: str = "Arjun (Male Lead)"
    voiceover_sinhala: str = ""
    voiceover_english: str = ""
    voice_tone: str = "Warm cinematic tone"
    sfx_cue: str = ""
    engine_used: str = "Wan 2.2 I2V • RTX 4060 Local GPU"

class GenerateVideosRequest(BaseModel):
    run_id: str
    wan_model: str = "Wan2.2-TI2V-5B"
    aspect_ratio: str = "16:9"
    scenes: List[Dict[str, Any]]
    frames: List[Dict[str, Any]]

class GenerateVideosResponse(BaseModel):
    run_id: str
    video_folder_path: str
    videos: List[VideoClipItem]

class RegenerateSingleVideoRequest(BaseModel):
    run_id: str
    scene_number: int
    wan_model: str = "Wan2.2-TI2V-5B"
    aspect_ratio: str = "16:9"
    source_image_url: Optional[str] = None
    source_image_path: Optional[str] = None
    video_motion_prompt: str = ""
    camera_motion: str = "Slow dolly push-in"
    voiceover_speaker: str = "Arjun (Male Lead)"
    voiceover_sinhala: str = ""
    voiceover_english: str = ""
    voice_tone: str = "Warm cinematic tone"
    sfx_cue: str = ""


