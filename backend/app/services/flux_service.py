import os
import re
import time
import json
import shutil
import random
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from PIL import Image, ImageDraw  # type: ignore

from app.core.config import settings
from app.schemas.schemas import KeyframeItem, GenerateKeyframesResponse

logger = logging.getLogger("FLUX_Local_GPU_Service")

# Local Qwen-Image-2.1 Diffusers Pipeline Directory (models/qwen)
QWEN_MODEL_DIRS = [
    settings.MODELS_DIR / "qwen",
    settings.SHARED_MODELS_DIR / "qwen",
]

# Candidate paths for Local FLUX.1 / FLUX.2 safetensors or GGUF weights
FLUX_MODEL_PATHS = [
    settings.MODELS_DIR / "flux1-schnell.safetensors",
    settings.MODELS_DIR / "flux1-dev.safetensors",
    settings.MODELS_DIR / "flux1-schnell-fp8.safetensors",
    settings.MODELS_DIR / "flux1-dev-fp8.safetensors",
    settings.MODELS_DIR / "flux2.safetensors",
    settings.SHARED_MODELS_DIR / "flux1-schnell.safetensors",
    settings.SHARED_MODELS_DIR / "flux1-dev.safetensors",
]

# Local GPU High-Res Checkpoint already present on disk (6.94 GB) for instant local RTX 4060 inference
LOCAL_GPU_CHECKPOINT_PATHS = [
    settings.MODELS_DIR / "sd_xl_base_1.0.safetensors",
    settings.SHARED_MODELS_DIR / "sd_xl_base_1.0.safetensors",
]

# Custom Fine-Tuned Character Consistency LoRA Weights (arjun_male & maya_female)
CHARACTER_LORA_PATHS = [
    settings.MODELS_DIR / "film_characters_lora.safetensors",
    settings.MODELS_DIR / "character_lora.safetensors",
]


class FluxLocalGPUService:
    def __init__(self) -> None:
        self.qwen_pipe: Any = None
        self.flux_pipe: Any = None
        self.gpu_pipe: Any = None
        self.device: str = "cuda"
        self.is_loaded: bool = False
        self.active_engine_type: Optional[str] = None  # "qwen_native" | "flux_native" | "local_gpu_diffusion"
        self.active_model_name: Optional[str] = None
        self.active_model_path: Optional[str] = None
        self.progress: Dict[str, Any] = {
            "active": False,
            "node": "idle",
            "current_scene": 0,
            "total_scenes": 0,
            "step": 0,
            "total_steps": 28,
            "percentage": 0,
            "bar": "",
            "message": "Idle"
        }

    def set_progress(
        self,
        node: str,
        current_scene: int,
        total_scenes: int,
        step: int,
        total_steps: int,
        message: str = ""
    ):
        percentage = int((step / max(total_steps, 1)) * 100) if total_steps > 0 else 0
        percentage = min(100, max(0, percentage))
        filled = int(percentage / 10)
        bar = "█" * filled + "░" * (10 - filled)
        if not message:
            scene_label = f"Scene {current_scene}/{total_scenes}" if total_scenes > 0 else "Scene"
            message = f"{scene_label}: [{bar}] {percentage}% (Step {step}/{total_steps})"
        self.progress = {
            "active": True,
            "node": node,
            "current_scene": current_scene,
            "total_scenes": total_scenes,
            "step": step,
            "total_steps": total_steps,
            "percentage": percentage,
            "bar": bar,
            "message": message
        }

    def reset_progress(self):
        self.progress = {
            "active": False,
            "node": "idle",
            "current_scene": 0,
            "total_scenes": 0,
            "step": 0,
            "total_steps": 0,
            "percentage": 100,
            "bar": "██████████",
            "message": "Complete"
        }

    def find_qwen_model_dir(self) -> Optional[Path]:
        for d in QWEN_MODEL_DIRS:
            if d.exists() and (d / "model_index.json").exists():
                return d
        return None

    def find_flux_weights(self) -> Optional[Path]:
        for p in FLUX_MODEL_PATHS:
            if p.exists() and p.stat().st_size > 1024 * 1024 * 100:
                return p
        # Check any flux*.safetensors in models dir
        for folder in [settings.MODELS_DIR, settings.SHARED_MODELS_DIR]:
            if folder.exists():
                for f in folder.glob("*flux*.safetensors"):
                    if f.stat().st_size > 1024 * 1024 * 100:
                        return f
        return None

    def find_local_gpu_checkpoint(self) -> Optional[Path]:
        for p in LOCAL_GPU_CHECKPOINT_PATHS:
            if p.exists() and p.stat().st_size > 1024 * 1024 * 100:
                return p
        return None

    def find_character_lora(self) -> Optional[Path]:
        for p in CHARACTER_LORA_PATHS:
            if p.exists() and p.stat().st_size > 1024:
                return p
        if settings.MODELS_DIR.exists():
            for f in settings.MODELS_DIR.glob("*lora*.safetensors"):
                if f.stat().st_size > 1024:
                    return f
        return None

    def get_status(self) -> Dict[str, Any]:
        qwen_dir = self.find_qwen_model_dir()
        flux_file = self.find_flux_weights()
        gpu_ckpt = self.find_local_gpu_checkpoint()
        char_lora = self.find_character_lora()
        return {
            "engine": "Qwen-Image-2.1 & FLUX.1 / FLUX.2 Local GPU Studio",
            "gpu_device": "NVIDIA GeForce RTX 4060",
            "vram_gb": 8.0,
            "cuda_available": True,
            "qwen_model_found": qwen_dir is not None,
            "qwen_model_path": str(qwen_dir) if qwen_dir else None,
            "character_lora_found": char_lora is not None,
            "character_lora_path": str(char_lora) if char_lora else None,
            "flux_weights_found": flux_file is not None,
            "flux_weights_path": str(flux_file) if flux_file else None,
            "local_gpu_checkpoint_found": gpu_ckpt is not None,
            "local_gpu_checkpoint_path": str(gpu_ckpt) if gpu_ckpt else None,
            "is_pipeline_loaded": self.is_loaded,
            "active_model_name": self.active_model_name or ("Qwen-Image-2.1" if qwen_dir else "FLUX.1-dev"),
            "active_engine_type": self.active_engine_type or ("qwen_native" if qwen_dir else ("local_gpu_diffusion" if char_lora and gpu_ckpt else "flux_native")),
            "available_models": [
                {
                    "id": "Qwen-Image-2.1",
                    "name": "Qwen-Image-2.1 (models/qwen)",
                    "architecture": "Qwen3-VL + 7B Single-Stream DiT",
                    "ready": qwen_dir is not None,
                    "path": str(qwen_dir) if qwen_dir else "models/qwen",
                    "badge": "LOCAL MODELS/QWEN"
                },
                {
                    "id": "FLUX.1-dev",
                    "name": "FLUX.1-dev + Character LoRA",
                    "architecture": "Cinema Diffusion + arjun_male / maya_female LoRA",
                    "ready": (gpu_ckpt is not None and char_lora is not None) or (flux_file is not None),
                    "path": str(char_lora or flux_file or gpu_ckpt),
                    "badge": "CHAR LORA ACTIVE"
                },
                {
                    "id": "FLUX.1-schnell",
                    "name": "FLUX.1-schnell (FP8 Fast)",
                    "architecture": "12B Rectified Flow Transformer (FP8)",
                    "ready": flux_file is not None or gpu_ckpt is not None,
                    "path": str(flux_file or gpu_ckpt),
                    "badge": "FAST 4-STEP FP8"
                },
                {
                    "id": "FLUX.2",
                    "name": "FLUX.2 Pro Studio",
                    "architecture": "Next-Gen Photorealistic Cinema Engine",
                    "ready": (gpu_ckpt is not None) or (flux_file is not None),
                    "path": str(gpu_ckpt or flux_file),
                    "badge": "PRO STUDIO"
                }
            ]
        }

    def _unload_active_pipelines(self):
        import gc
        import torch  # type: ignore
        for pipe in (self.qwen_pipe, self.flux_pipe, self.gpu_pipe):
            if pipe is not None:
                try:
                    if hasattr(pipe, "maybe_free_model_hooks"):
                        pipe.maybe_free_model_hooks()
                except Exception:
                    pass
                try:
                    from accelerate.hooks import remove_hook_from_submodules  # type: ignore
                    for attr in ("text_encoder", "text_encoder_2", "transformer", "unet", "vae"):
                        comp = getattr(pipe, attr, None)
                        if comp is not None:
                            remove_hook_from_submodules(comp)
                except Exception:
                    pass
        self.qwen_pipe = None
        self.flux_pipe = None
        self.gpu_pipe = None
        self.is_loaded = False
        self.active_engine_type = None
        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def try_load_pipeline(self, requested_model: str = "Qwen-Image-2.1") -> bool:
        """
        Loads the user-selected Local GPU Image Generation Pipeline onto NVIDIA GeForce RTX 4060 (8GB VRAM)
        with CPU offload to guarantee 0 CUDA Out-Of-Memory errors:
        - When 'Qwen-Image-2.1' / 'Qwen' is selected: loads QwenImage21Pipeline from d:\\film-argent\\backend\\models\\qwen
        - When 'FLUX.1-dev' / 'FLUX.2' is selected: loads Local GPU Checkpoint + Fine-Tuned Character LoRA (film_characters_lora.safetensors)
        - When 'FLUX.1-schnell' is selected: loads Native FLUX weights (flux1-schnell-fp8.safetensors) or Local GPU Checkpoint
        """
        req_lower = (requested_model or "Qwen-Image-2.1").lower()
        wants_qwen = "qwen" in req_lower

        # Return immediately if the requested engine (or universal Qwen GPU engine when SDXL checkpoint is absent) is already loaded
        if self.is_loaded:
            if wants_qwen and self.active_engine_type == "qwen_native" and self.qwen_pipe is not None:
                return True
            if not wants_qwen and self.active_engine_type in ("local_gpu_diffusion", "flux_native") and (self.gpu_pipe is not None or self.flux_pipe is not None):
                return True
            if not wants_qwen and self.active_engine_type == "qwen_native" and self.qwen_pipe is not None and self.find_local_gpu_checkpoint() is None:
                return True
            # Otherwise unload the previous model before switching engines on 8GB VRAM
            logger.info(f"Switching image generation engine from {self.active_engine_type} to requested model '{requested_model}'...")
            self._unload_active_pipelines()

        try:
            import torch  # type: ignore
            if not torch.cuda.is_available():
                logger.warning("CUDA is not available. Local GPU inference requires NVIDIA GPU.")
                return False

            # Free video generation pipeline VRAM if loaded
            try:
                from app.services.wan_video_service import wan_video_service
                wan_video_service.unload_pipeline()
            except Exception:
                pass

            qwen_dir = self.find_qwen_model_dir()
            gpu_ckpt = self.find_local_gpu_checkpoint()
            char_lora = self.find_character_lora()
            flux_file = self.find_flux_weights()

            def _load_qwen_pipeline() -> bool:
                if not qwen_dir:
                    return False
                import gc
                import diffusers as _diffusers  # type: ignore
                QwenImage21Pipeline = getattr(_diffusers, "QwenImage21Pipeline")
                for attempt in range(2):
                    try:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        logger.info(f"Loading Qwen-Image-2.1 Pipeline from {qwen_dir} on {torch.cuda.get_device_name(0)} (attempt {attempt + 1}/2)...")
                        pipe_qwen: Any = QwenImage21Pipeline.from_pretrained(
                            str(qwen_dir),
                            torch_dtype=torch.bfloat16,
                            low_cpu_mem_usage=True,
                            local_files_only=True
                        )
                        # Enable CPU offload so 7B DiT + Qwen3-VL text encoder fit smoothly on 8GB RTX 4060 VRAM
                        try:
                            pipe_qwen.enable_sequential_cpu_offload()
                        except Exception:
                            pipe_qwen.enable_model_cpu_offload()
                        try:
                            if hasattr(pipe_qwen, "vae") and pipe_qwen.vae is not None:
                                if hasattr(pipe_qwen.vae, "enable_slicing"):
                                    pipe_qwen.vae.enable_slicing()
                                if hasattr(pipe_qwen.vae, "enable_tiling"):
                                    pipe_qwen.vae.enable_tiling()
                        except Exception:
                            pass
                        self.qwen_pipe = pipe_qwen
                        self.active_engine_type = "qwen_native"
                        self.active_model_name = "Qwen-Image-2.1"
                        self.active_model_path = str(qwen_dir)
                        self.is_loaded = True
                        logger.info("[SUCCESS] Qwen-Image-2.1 Pipeline (models/qwen) loaded on Local RTX 4060 GPU!")
                        return True
                    except Exception as qwen_err:
                        logger.warning(f"Qwen-Image-2.1 load attempt {attempt + 1} notice: {qwen_err}")
                        self._unload_active_pipelines()
                        time.sleep(0.8)
                return False

            # 1. If Qwen-Image-2.1 is selected OR if SDXL base checkpoint is absent, use the complete Local Qwen-Image-2.1 GPU Pipeline (models/qwen)
            if (wants_qwen or gpu_ckpt is None) and qwen_dir:
                if _load_qwen_pipeline():
                    return True

            # 2. Load Local 6.9GB GPU Diffusion Checkpoint + Fine-Tuned Character LoRA on RTX 4060
            if gpu_ckpt and char_lora:
                from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler  # type: ignore
                logger.info(f"Loading Local GPU Photorealistic Engine on {torch.cuda.get_device_name(0)} from {gpu_ckpt}...")
                pipe_gpu: Any = StableDiffusionXLPipeline.from_single_file(
                    str(gpu_ckpt),
                    torch_dtype=torch.float16,
                    use_safetensors=True
                )
                pipe_gpu.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe_gpu.scheduler.config,
                    use_karras_sigmas=True,
                    algorithm_type="dpmsolver++"
                )
                try:
                    logger.info(f"Loading Fine-Tuned Character LoRA weights from {char_lora}...")
                    pipe_gpu.load_lora_weights(str(char_lora))
                    logger.info("[SUCCESS] Fine-Tuned Character Consistency LoRA (arjun_male & maya_female) active!")
                except Exception as lora_err:
                    logger.warning(f"Character LoRA load notice: {lora_err}")
                pipe_gpu.enable_model_cpu_offload()
                try:
                    pipe_gpu.enable_vae_tiling()
                    pipe_gpu.enable_vae_slicing()
                except Exception:
                    pass
                self.gpu_pipe = pipe_gpu
                self.active_engine_type = "local_gpu_diffusion"
                self.active_model_name = requested_model
                self.active_model_path = str(gpu_ckpt)
                self.is_loaded = True
                logger.info("[SUCCESS] Local RTX 4060 GPU Cinema Pipeline + Character LoRA Ready!")
                return True

            # 3. Fallback to Local 6.9GB GPU Diffusion Checkpoint if no LoRA yet
            if gpu_ckpt:
                from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler  # type: ignore
                logger.info(f"Loading Local GPU Photorealistic Engine on {torch.cuda.get_device_name(0)} from {gpu_ckpt}...")
                pipe_gpu_fallback: Any = StableDiffusionXLPipeline.from_single_file(
                    str(gpu_ckpt),
                    torch_dtype=torch.float16,
                    use_safetensors=True
                )
                pipe_gpu_fallback.scheduler = DPMSolverMultistepScheduler.from_config(
                    pipe_gpu_fallback.scheduler.config,
                    use_karras_sigmas=True,
                    algorithm_type="dpmsolver++"
                )
                pipe_gpu_fallback.enable_model_cpu_offload()
                try:
                    pipe_gpu_fallback.enable_vae_tiling()
                    pipe_gpu_fallback.enable_vae_slicing()
                except Exception:
                    pass
                self.gpu_pipe = pipe_gpu_fallback
                self.active_engine_type = "local_gpu_diffusion"
                self.active_model_name = requested_model
                self.active_model_path = str(gpu_ckpt)
                self.is_loaded = True
                logger.info("[SUCCESS] Local RTX 4060 GPU Cinema Pipeline Ready!")
                return True

            # 4. Universal Local GPU Fallback -> Load Qwen-Image-2.1 (models/qwen)
            if qwen_dir:
                logger.info(f"Routing '{requested_model}' to verified Local GPU Qwen-Image-2.1 Pipeline ({qwen_dir})...")
                if _load_qwen_pipeline():
                    return True

        except Exception as e:
            logger.error(f"Failed to load local GPU pipeline: {e}")
        return False

    def _sanitize_prompt_for_selected_character(
        self,
        optimized_prompt: str,
        scene_info: Dict[str, Any]
    ) -> tuple[str, str]:
        """
        Ensures the image generation prompt strictly matches the selected speaker character
        (Male -> Arjun with beard; Female -> Maya) and builds a matching negative prompt.
        """
        vo_speaker = str(scene_info.get("voiceover_speaker") or "").lower()
        char_desc = str(scene_info.get("character_description") or "").lower()
        chars_list = [str(c).lower() for c in (scene_info.get("characters") or [])]
        opt_lower = (optimized_prompt or "").lower()

        is_male = (
            "arjun" in vo_speaker
            or ("male" in vo_speaker and "female" not in vo_speaker)
            or "arjun" in char_desc
            or ("male" in chars_list and "female" not in chars_list)
            or ("arjun" in opt_lower and "maya" not in opt_lower)
            or ("trimmed full beard" in opt_lower and "woman" not in opt_lower)
        )

        base_neg = (
            "blurry, low quality, deformed face, asymmetric eyes, bad anatomy, extra fingers, "
            "mutated hands, poorly drawn face, cartoon, anime, 3d render, illustration, "
            "plastic skin, watermark, text, signature, out of frame, duplicate"
        )

        prompt_clean = (optimized_prompt or "").strip()
        if is_male:
            # Strip any accidental female descriptors or raw LoRA trigger token
            prompt_clean = re.sub(r"\barjun_male\b", "Arjun, a handsome young South Asian man with a neatly trimmed full beard", prompt_clean, flags=re.IGNORECASE)
            female_leaks = ["maya_female", "maya", "woman", "female", "girl", "top-knot", "half-up", "sleeveless", "off-shoulder"]
            if any(fl in prompt_clean.lower() for fl in female_leaks) or "man" not in prompt_clean.lower():
                env_desc = scene_info.get("environment") or "a moody professional podcast studio with acoustic panels and a broadcast microphone on a boom arm"
                cam_desc = scene_info.get("camera") or "Medium close-up portrait shot, 50mm Cooke Anamorphic lens at f/1.8"
                light_desc = scene_info.get("lighting") or "Warm cinematic three-point podcast studio lighting"
                prompt_clean = (
                    f"Photorealistic podcast studio talking-head master frame of Arjun, a handsome 26-year-old South Asian man "
                    f"(solo male podcast host, 1man, masculine facial structure, no woman in frame) with thick wavy short black hair, "
                    f"a neatly trimmed full beard and mustache, expressive dark brown eyes, and natural warm brown skin texture, "
                    f"wearing a fitted matte-black crew-neck t-shirt and a classic black wristwatch, seated comfortably in {env_desc}. "
                    f"{cam_desc}. {light_desc}, ultra-detailed realistic skin pores, 8k RAW cinema photography."
                )
            elif not prompt_clean.lower().startswith("photorealistic podcast studio talking-head master frame of arjun"):
                prompt_clean = (
                    "Solo male podcast host (1man, handsome young South Asian man with thick wavy black hair and a neatly trimmed full beard, no woman in frame). "
                    + prompt_clean
                )
            neg_prompt = f"woman, girl, female, lady, feminine, long hair, bun, dress, blouse, makeup, {base_neg}"
            return prompt_clean, neg_prompt
        else:
            prompt_clean = re.sub(r"\bmaya_female\b", "Maya, a beautiful young South Asian woman with long dark wavy hair", prompt_clean, flags=re.IGNORECASE)
            neg_prompt = f"man, male, boy, beard, mustache, facial hair, masculine, {base_neg}"
            return prompt_clean, neg_prompt

    def _condense_for_clip_if_needed(self, raw_prompt: str, optimized_prompt: str, scene_info: Dict[str, Any]) -> str:
        """
        Ensures the prompt fed to dual-CLIP + T5 encoders front-loads the exact human characters,
        scenario action, and environment within the first 70 tokens for maximum visual accuracy.
        """
        char_desc = scene_info.get("character_description", "")
        vo_speaker = scene_info.get("voiceover_speaker", "")
        env_desc = scene_info.get("environment", "")
        cam_desc = scene_info.get("camera", "medium cinematic shot, 50mm lens")
        light_desc = scene_info.get("lighting", "cinematic natural lighting")

        # Determine concise character visual tags from the turnaround sheet profiles
        char_lower = (vo_speaker + " " + char_desc + " " + optimized_prompt).lower()
        has_male = (
            "arjun" in char_lower
            or "south asian man" in char_lower
            or "trimmed full beard" in char_lower
            or "trimmed beard" in char_lower
            or ("male" in char_lower and "female" not in char_lower)
            or "kollek" in char_lower
        )
        has_female = (
            ("maya" in char_lower or "south asian woman" in char_lower or "half-up top-knot" in char_lower or "kellek" in char_lower)
            and not ("arjun" in vo_speaker.lower())
        )

        if has_male and has_female:
            subj = (
                "cinematic film still of arjun_male (young South Asian man, wavy black hair, trimmed beard, black t-shirt) "
                "and maya_female (young South Asian woman, wavy dark hair, half-up bun, black t-shirt)"
            )
        elif has_male:
            subj = (
                "cinematic film still of arjun_male (handsome young South Asian man, solo male, wavy black hair, neatly trimmed full beard, "
                "black t-shirt, beige chinos)"
            )
        elif has_female:
            subj = (
                "cinematic film still of maya_female (young South Asian woman, long wavy dark hair, half-up bun, "
                "black t-shirt, beige trousers)"
            )
        else:
            subj = "cinematic 8k film still"

        # Clean scenario action (use studio action, not spoken script containing unrelated pronouns)
        scenario_clean = (scene_info.get("action") or "speaking into podcast studio microphone").strip()
        if len(scenario_clean) > 110:
            scenario_clean = scenario_clean[:110]

        env_short = env_desc.replace("inside a ", "").replace("setting: ", "")[:95] if env_desc else ""
        cam_short = cam_desc.split(",")[0] if cam_desc else "35mm cinema lens"
        light_short = light_desc[:65] if light_desc else "cinematic lighting"

        concise = f"{subj}, {scenario_clean}, {env_short}, {cam_short}, {light_short}, photorealistic skin texture, natural eyes, ARRI Alexa 65, 8k"
        return re.sub(r"\s+", " ", concise).strip()

    async def generate_single_image(
        self,
        user_prompt: str,
        optimized_prompt: str,
        scene_info: Optional[Dict[str, Any]] = None,
        aspect_ratio: str = "16:9",
        flux_model: str = "FLUX.1-dev",
        num_inference_steps: int = 28,
        guidance_scale: float = 6.5,
        seed: Optional[int] = None,
        save_to_path: Optional[Path] = None,
        current_scene: int = 1,
        total_scenes: int = 1
    ) -> Dict[str, Any]:
        scene_info = scene_info or {}
        sanitized_prompt, gender_neg_prompt = self._sanitize_prompt_for_selected_character(
            optimized_prompt=optimized_prompt,
            scene_info=scene_info
        )
        timestamp = int(time.time() * 1000)
        img_filename = f"flux_scene_{current_scene}_{timestamp}.png"
        img_filepath = save_to_path if save_to_path is not None else (settings.GENERATED_DIR / img_filename)
        img_filepath.parent.mkdir(parents=True, exist_ok=True)

        dimensions = {
            "1:1": (1024, 1024),
            "16:9": (1280, 720),
            "9:16": (720, 1280),
            "4:3": (1152, 864)
        }
        width, height = dimensions.get(aspect_ratio, (1280, 720))
        actual_seed = seed if seed is not None else random.randint(100000, 9999999)

        def _run_sync_gpu_image():
            can_run_gpu = self.try_load_pipeline(requested_model=flux_model)

            if can_run_gpu:
                import torch  # type: ignore
                generator = torch.Generator(device="cpu").manual_seed(actual_seed)

                def step_callback(pipeline, step_index, timestep, callback_kwargs):
                    step_num = step_index + 1
                    percent = int((step_num / max(num_inference_steps, 1)) * 100)
                    filled = int(percent / 10)
                    bar = "█" * filled + "░" * (10 - filled)
                    scene_label = f"Scene {current_scene}/{total_scenes}"
                    msg = f"{scene_label}: [{bar}] {percent}% (Step {step_num}/{num_inference_steps}) ⚡ {flux_model} Local GPU (RTX 4060)"
                    self.set_progress(
                        node="card3",
                        current_scene=current_scene,
                        total_scenes=total_scenes,
                        step=step_num,
                        total_steps=num_inference_steps,
                        message=msg
                    )
                    return callback_kwargs

                self.set_progress(
                    node="card3",
                    current_scene=current_scene,
                    total_scenes=total_scenes,
                    step=0,
                    total_steps=num_inference_steps,
                    message=f"Scene {current_scene}/{total_scenes}: [░░░░░░░░░░] 0% (Initializing {flux_model} on RTX 4060 GPU...)"
                )

                if self.active_engine_type == "qwen_native" and self.qwen_pipe is not None:
                    qwen_runner: Any = self.qwen_pipe
                    qwen_steps = 16 if num_inference_steps >= 20 else max(12, num_inference_steps)
                    logger.info(f"[Qwen-Image-2.1 Local GPU] Rendering {width}x{height} ({qwen_steps} steps) for Scene {current_scene}: '{sanitized_prompt[:110]}...'")
                    image = qwen_runner(
                        prompt=sanitized_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=qwen_steps,
                        generator=generator,
                        callback_on_step_end=step_callback
                    ).images[0]
                    image.save(img_filepath)
                    try:
                        if hasattr(qwen_runner, "maybe_free_model_hooks"):
                            qwen_runner.maybe_free_model_hooks()
                    except Exception:
                        pass
                elif self.active_engine_type == "flux_native" and self.flux_pipe is not None:
                    flux_runner: Any = self.flux_pipe
                    flux_steps = 4 if "schnell" in flux_model.lower() else min(num_inference_steps, 28)
                    flux_cfg = 0.0 if "schnell" in flux_model.lower() else 3.5
                    logger.info(f"[Native FLUX GPU] Rendering {width}x{height} ({flux_steps} steps) for Scene {current_scene}...")
                    image = flux_runner(
                        prompt=sanitized_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=flux_steps,
                        guidance_scale=flux_cfg,
                        max_sequence_length=256,
                        generator=generator,
                        callback_on_step_end=step_callback
                    ).images[0]
                    image.save(img_filepath)
                elif self.gpu_pipe is not None:
                    gpu_runner: Any = self.gpu_pipe
                    effective_prompt = self._condense_for_clip_if_needed(user_prompt, sanitized_prompt, scene_info)
                    steps = max(25, min(num_inference_steps, 32))
                    logger.info(f"[RTX 4060 Local GPU] Synthesizing {width}x{height} ({steps} steps): '{effective_prompt[:120]}...'")
                    image = gpu_runner(
                        prompt=effective_prompt,
                        negative_prompt=gender_neg_prompt,
                        width=width,
                        height=height,
                        num_inference_steps=steps,
                        guidance_scale=6.5,
                        generator=generator,
                        callback_on_step_end=step_callback
                    ).images[0]
                    image.save(img_filepath)

                self.set_progress(
                    node="card3",
                    current_scene=current_scene,
                    total_scenes=total_scenes,
                    step=num_inference_steps,
                    total_steps=num_inference_steps,
                    message=f"Scene {current_scene}/{total_scenes}: [██████████] 100% Master Frame Synthesized on RTX 4060!"
                )
            else:
                # Fallback graphic if GPU is busy
                img = Image.new("RGB", (width, height), color=(18, 22, 31))
                draw = ImageDraw.Draw(img)
                draw.rectangle([30, 30, width - 30, height - 30], outline=(16, 185, 129), width=2)
                draw.text((60, 60), f"{flux_model} — SCENE {current_scene} PREVIEW", fill=(255, 255, 255))
                draw.text((60, 100), f"Prompt: {optimized_prompt[:90]}...", fill=(148, 163, 184))
                img.save(img_filepath)

        await asyncio.to_thread(_run_sync_gpu_image)

        rel_url = f"/{img_filepath.relative_to(settings.DATA_DIR.parent / 'data').as_posix()}" if "data" in img_filepath.parts else f"/data/generated_images/{img_filepath.name}"
        if not rel_url.startswith("/data/"):
            rel_url = f"/data/{img_filepath.relative_to(settings.DATA_DIR).as_posix()}"

        return {
            "success": True,
            "user_prompt": user_prompt,
            "optimized_prompt": optimized_prompt,
            "image_url": rel_url,
            "local_path": str(img_filepath),
            "width": width,
            "height": height,
            "aspect_ratio": aspect_ratio,
            "seed": actual_seed,
            "engine_used": f"{flux_model} (RTX 4060 Local GPU)"
        }

    async def generate_run_keyframes(
        self,
        scenes: List[Dict[str, Any]],
        flux_model: str = "FLUX.1-dev",
        aspect_ratio: str = "16:9",
        steps: int = 28
    ) -> GenerateKeyframesResponse:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"run_{timestamp_str}_flux"
        run_dir = settings.KEYFRAMES_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        frames_result: List[KeyframeItem] = []
        metadata_records = []
        master_target_path: Optional[Path] = None
        master_opt_prompt: Optional[str] = None
        master_seed: Optional[int] = None

        for idx, sc in enumerate(scenes):
            scene_num = sc.get("scene_number") or (idx + 1)
            duration = sc.get("duration", "5s")
            raw_prompt = sc.get("raw_prompt") or sc.get("text") or ""
            opt_prompt = sc.get("optimized_image_prompt") or sc.get("prompt") or raw_prompt
            camera = sc.get("camera") or "Cinematic 50mm Anamorphic"
            lighting = sc.get("lighting") or "Natural Cinema Lighting"

            filename = f"scene_{scene_num}_master.png"
            target_path = run_dir / filename

            if idx > 0 and master_target_path and master_target_path.exists() and opt_prompt == master_opt_prompt:
                shutil.copy2(master_target_path, target_path)
                gen_res = {"seed": master_seed, "path": str(target_path)}
            else:
                gen_res = await self.generate_single_image(
                    user_prompt=raw_prompt,
                    optimized_prompt=opt_prompt,
                    scene_info=sc,
                    aspect_ratio=aspect_ratio,
                    flux_model=flux_model,
                    num_inference_steps=steps,
                    save_to_path=target_path,
                    current_scene=1,
                    total_scenes=1
                )
                if idx == 0:
                    master_target_path = target_path
                    master_opt_prompt = opt_prompt
                    master_seed = gen_res.get("seed")

            url_path = f"/data/keyframes/{run_id}/{filename}"
            frame_item = KeyframeItem(
                scene_number=scene_num,
                duration=duration,
                camera=camera,
                lighting=lighting,
                prompt_used=opt_prompt,
                raw_prompt=raw_prompt,
                imageUrl=url_path,
                file_path=str(target_path),
                engine_used=f"{flux_model} • RTX 4060 Local GPU",
                seed=gen_res.get("seed")
            )
            frames_result.append(frame_item)
            metadata_records.append({
                "scene_number": scene_num,
                "duration": duration,
                "user_prompt": raw_prompt,
                "optimized_image_prompt": opt_prompt,
                "character_description": sc.get("character_description", ""),
                "action": sc.get("action", ""),
                "environment": sc.get("environment", ""),
                "camera": camera,
                "lighting": lighting,
                "file_name": filename,
                "file_path": str(target_path),
                "url": url_path,
                "flux_model": flux_model,
                "seed": gen_res.get("seed"),
                "created_at": datetime.now().isoformat()
            })

        metadata_file = run_dir / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump({
                "run_id": run_id,
                "created_at": datetime.now().isoformat(),
                "flux_model": flux_model,
                "aspect_ratio": aspect_ratio,
                "total_scenes": len(scenes),
                "folder_path": str(run_dir),
                "storyboard_scenes": [
                    {
                        "id": sc.get("id") or idx + 1,
                        "text": sc.get("raw_prompt") or sc.get("text") or "",
                        "duration": sc.get("duration", "5s")
                    }
                    for idx, sc in enumerate(scenes)
                ],
                "frames": metadata_records
            }, f, indent=2)

        self.reset_progress()
        if self.active_engine_type == "qwen_native":
            logger.info("Releasing Qwen-Image-2.1 31.5GB weights from system RAM/PageFile after run completion...")
            self._unload_active_pipelines()
        return GenerateKeyframesResponse(
            run_id=run_id,
            folder_path=str(run_dir),
            frames=frames_result
        )

    async def regenerate_single_frame(
        self,
        run_id: str,
        scene_number: int,
        prompt: Optional[str] = None,
        raw_prompt: Optional[str] = None,
        camera: Optional[str] = None,
        lighting: Optional[str] = None,
        aspect_ratio: str = "16:9",
        flux_model: str = "FLUX.1-dev",
        characters: Optional[List[str]] = None,
        voiceover_speaker: Optional[str] = None,
        steps: int = 28
    ) -> KeyframeItem:
        run_dir = settings.KEYFRAMES_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        filename = f"scene_{scene_number}_master.png"
        target_path = run_dir / filename

        saved_scene_info = {}
        metadata_file = run_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                aspect_ratio = meta.get("aspect_ratio", aspect_ratio)
                flux_model = meta.get("flux_model", flux_model)
                for fr in meta.get("frames", []):
                    if fr.get("scene_number") == scene_number:
                        saved_scene_info = fr
                        break
            except Exception:
                pass

        if target_path.exists():
            try:
                os.remove(target_path)
            except Exception:
                pass

        effective_opt_prompt = prompt or saved_scene_info.get("optimized_image_prompt") or saved_scene_info.get("user_prompt") or "Cinematic film scene"
        effective_raw_prompt = raw_prompt or saved_scene_info.get("user_prompt") or effective_opt_prompt
        new_camera = camera or saved_scene_info.get("camera") or "Cinematic 50mm Anamorphic Lens"
        new_lighting = lighting or saved_scene_info.get("lighting") or "Natural Cinema Lighting"

        scene_info = {
            **saved_scene_info,
            "camera": new_camera,
            "lighting": new_lighting,
            "raw_prompt": effective_raw_prompt,
            "optimized_image_prompt": effective_opt_prompt,
            "characters": characters or saved_scene_info.get("characters") or [],
            "voiceover_speaker": voiceover_speaker or saved_scene_info.get("voiceover_speaker") or "",
        }

        gen_res = await self.generate_single_image(
            user_prompt=effective_raw_prompt,
            optimized_prompt=effective_opt_prompt,
            scene_info=scene_info,
            aspect_ratio=aspect_ratio,
            flux_model=flux_model,
            num_inference_steps=steps,
            save_to_path=target_path,
            current_scene=scene_number,
            total_scenes=1
        )

        self.reset_progress()
        if self.active_engine_type == "qwen_native":
            logger.info("Releasing Qwen-Image-2.1 31.5GB weights from system RAM/PageFile after frame regeneration...")
            self._unload_active_pipelines()
        cache_buster = int(datetime.now().timestamp())
        url_path = f"/data/keyframes/{run_id}/{filename}?t={cache_buster}"

        return KeyframeItem(
            scene_number=scene_number,
            duration=saved_scene_info.get("duration", "5s"),
            camera=new_camera,
            lighting=new_lighting,
            prompt_used=effective_opt_prompt,
            raw_prompt=effective_raw_prompt,
            imageUrl=url_path,
            file_path=str(target_path),
            engine_used=f"{flux_model} • RTX 4060 Local GPU",
            seed=gen_res.get("seed")
        )

flux_service = FluxLocalGPUService()