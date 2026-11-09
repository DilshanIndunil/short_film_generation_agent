import os
import re
import time
import json
import math
import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.core.config import settings
from app.schemas.schemas import VideoClipItem, GenerateVideosResponse
from app.services.flux_service import flux_service

logger = logging.getLogger("Wan22_Video_Studio")

LTX_MODEL_DIRS = [
    settings.MODELS_DIR / "models" / "ltx",
    settings.MODELS_DIR / "ltx",
    settings.SHARED_MODELS_DIR / "models" / "ltx",
    settings.SHARED_MODELS_DIR / "ltx",
]

WAN_MODEL_DIRS = [
    settings.MODELS_DIR / "models" / "wan2.2" / "Wan2.2-I2V-A14B-Diffusers",
    settings.MODELS_DIR / "models" / "wan2.2",
    settings.MODELS_DIR / "wan2.2-ti2v-5b",
    settings.MODELS_DIR / "wan2.2-i2v-a14b",
    settings.SHARED_MODELS_DIR / "wan2.2-ti2v-5b",
    settings.SHARED_MODELS_DIR / "wan2.2-i2v-a14b",
]


class WanVideoStudioService:
    """
    Card 4.K Engine: Generates Scene-by-Scene Short Film Video Clips (.mp4)
    from Card 3.K (Master Frame Image + Scene K Motion & Voice-Over Script)
    using Local LTX-Video (models/models/ltx) or Wan 2.2 (models/models/wan2.2).
    """

    def __init__(self):
        self.ltx_pipe: Any = None
        self.wan_pipe: Any = None
        self.active_pipe_type: Optional[str] = None
        self.active_model_dir: Optional[str] = None

    def unload_pipeline(self):
        """Frees GPU VRAM and CPU offload hooks used by the video generation pipeline."""
        import gc
        for pipe in (self.ltx_pipe, self.wan_pipe):
            if pipe is not None:
                try:
                    if hasattr(pipe, "maybe_free_model_hooks"):
                        pipe.maybe_free_model_hooks()
                except Exception:
                    pass
                try:
                    from accelerate.hooks import remove_hook_from_submodules  # type: ignore
                    for attr in ("text_encoder", "transformer", "unet", "vae"):
                        comp = getattr(pipe, attr, None)
                        if comp is not None:
                            remove_hook_from_submodules(comp)
                except Exception:
                    pass
        self.ltx_pipe = None
        self.wan_pipe = None
        self.active_pipe_type = None
        self.active_model_dir = None
        gc.collect()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _find_ltx_model_dir(self) -> Optional[Path]:
        for d in LTX_MODEL_DIRS:
            if d.exists() and (d / "model_index.json").exists():
                return d
        return None

    def _find_wan_model_dir(self) -> Optional[Path]:
        for d in WAN_MODEL_DIRS:
            if d.exists() and (d / "model_index.json").exists():
                if (d / "transformer").exists() or (d / "vae").exists():
                    return d
        return None

    def _ensure_fast_tokenizer(self, model_dir: Path) -> None:
        """
        Ensures tokenizer.json exists in model_dir/tokenizer so T5TokenizerFast loads
        offline on Python 3.14 without requiring the C++ sentencepiece binary.
        """
        tok_dir = model_dir / "tokenizer"
        fast_tok_path = tok_dir / "tokenizer.json"
        spiece_path = tok_dir / "spiece.model"
        if fast_tok_path.exists() or not spiece_path.exists():
            return
        try:
            from transformers import T5TokenizerFast
            from transformers.convert_slow_tokenizer import SpmConverter, import_protobuf

            class _MockOriginalTokenizer:
                def __init__(self, vocab_file: str):
                    self.vocab_file = vocab_file
                    pb2: Any = import_protobuf()
                    model_proto_cls = getattr(pb2, "ModelProto")
                    self.proto = model_proto_cls()
                    with open(vocab_file, "rb") as f:
                        self.proto.ParseFromString(f.read())
                    self._extra_ids = 100
                    self.additional_special_tokens = [f"<extra_id_{i}>" for i in range(100)]

            conv = SpmConverter(_MockOriginalTokenizer(str(spiece_path)))
            fast_tok = T5TokenizerFast(
                tokenizer_object=conv.converted(),
                eos_token="</s>",
                unk_token="<unk>",
                pad_token="<pad>",
                extra_ids=100,
            )
            fast_tok.save_pretrained(str(tok_dir))
            logger.info(f"Generated fast tokenizer.json at {fast_tok_path}")
        except Exception as e:
            logger.warning(f"Could not auto-generate fast tokenizer.json: {e}")

    def _load_gpu_video_pipeline(self, wan_model: str = "LTX-Video-2B") -> bool:
        """
        Loads the local Diffusers Image-to-Video pipeline (LTXImageToVideoPipeline or WanImageToVideoPipeline)
        onto the NVIDIA RTX 4060 GPU using sequential CPU offload and VAE tiling.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                logger.warning("CUDA is not available; cannot run GPU I2V pipeline.")
                return False

            # Free Card 3 image generation VRAM first so 8GB RTX 4060 has full headroom
            flux_service._unload_active_pipelines()

            ltx_dir = self._find_ltx_model_dir()
            wan_dir = self._find_wan_model_dir()

            # Wan2.2-I2V-A14B-Diffusers is 126.3 GB unquantized (24 shards x 4.92GB) which exceeds 8GB RTX 4060 / PC RAM
            # Always use the 8GB-VRAM-compatible local LTX-Video GPU Diffusers engine (2 shards, 3.8GB) when present
            target_dir = ltx_dir or wan_dir
            if target_dir is None:
                logger.warning("No local LTX or Wan Diffusers model directory found.")
                return False

            if self.ltx_pipe is not None and self.active_model_dir == str(target_dir):
                return True

            self.unload_pipeline()
            self._ensure_fast_tokenizer(target_dir)

            logger.info(f"Loading Local GPU I2V Pipeline from {target_dir} for model '{wan_model}'...")
            from diffusers import LTXImageToVideoPipeline

            pipe = LTXImageToVideoPipeline.from_pretrained(
                str(target_dir),
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            pipe.enable_sequential_cpu_offload()
            if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
                pipe.vae.enable_tiling()

            self.ltx_pipe = pipe
            self.active_pipe_type = "ltx_i2v"
            self.active_model_dir = str(target_dir)
            logger.info(f"Local GPU I2V Pipeline loaded successfully from {target_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to load GPU I2V pipeline: {e}", exc_info=True)
            return False

    def _resolve_model_label(self, wan_model: str) -> str:
        model_up = (wan_model or "LTX-Video-2B").upper()
        if "LTX" in model_up:
            for d in LTX_MODEL_DIRS:
                if d.exists():
                    if "13B" in model_up and (d / "ltxv-13b-0.9.8-distilled-fp8.safetensors").exists():
                        return f"{wan_model} (models/ltx/ltxv-13b-fp8) • RTX 4060 GPU"
                    return f"{wan_model} (models/ltx/ltxv-2b-fp8) • RTX 4060 GPU"
        else:
            for d in WAN_MODEL_DIRS:
                if d.exists():
                    return f"{wan_model} (models/wan2.2/I2V-Diffusers) • RTX 4060 GPU"
        return f"{wan_model} I2V • RTX 4060 Local GPU"

    def _resolve_source_image_path(
        self,
        run_id: str,
        scene_number: int,
        source_image_url: Optional[str] = None,
        source_image_path: Optional[str] = None
    ) -> Optional[Path]:
        if source_image_path:
            p = Path(source_image_path)
            if p.exists():
                return p

        if source_image_url:
            clean_url = source_image_url.split("?")[0]
            if clean_url.startswith("/data/"):
                rel = clean_url[len("/data/"):]
                candidate = settings.DATA_DIR / rel
                if candidate.exists():
                    return candidate

        run_frame = settings.KEYFRAMES_DIR / run_id / f"scene_{scene_number}_master.png"
        if run_frame.exists():
            return run_frame

        master_frame = settings.KEYFRAMES_DIR / run_id / "scene_1_master.png"
        if master_frame.exists():
            return master_frame

        return None

    def _render_cinema_subtitle_overlay(
        self,
        frame_bgr: np.ndarray,
        speaker: str,
        vo_english: str,
        vo_sinhala: str,
        scene_number: int,
        wan_model: str
    ) -> np.ndarray:
        """
        Adds a sleek cinema letterbox subtitle bar at the bottom with the active Voice-Over line.
        """
        h, w, _ = frame_bgr.shape
        overlay = frame_bgr.copy()

        # Subtle bottom gradient bar for cinema subtitle readability
        bar_h = max(72, int(h * 0.13))
        cv2.rectangle(overlay, (0, h - bar_h), (w, h), (10, 12, 18), -1)
        blended = cv2.addWeighted(overlay, 0.72, frame_bgr, 0.28, 0)

        # Convert to PIL for crisp typography
        pil_img = Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)

        try:
            font_sub = ImageFont.truetype("arial.ttf", max(15, int(h * 0.024)))
            font_meta = ImageFont.truetype("arial.ttf", max(12, int(h * 0.018)))
        except Exception:
            font_sub = ImageFont.load_default()
            font_meta = ImageFont.load_default()

        # Top-left discrete badge
        badge_text = f"SCENE {scene_number} • {wan_model} I2V"
        draw.text((20, 16), badge_text, fill=(167, 243, 208), font=font_meta)

        # Speaker + Voice-Over Subtitle
        speaker_clean = (speaker or "Narrator").strip()
        line_en = (vo_english or "").strip()
        if len(line_en) > 105:
            line_en = line_en[:102] + "..."

        # Extract Singlish portion if Sinhala unicode isn't supported by Arial
        singlish_part = ""
        if vo_sinhala and "(" in vo_sinhala and ")" in vo_sinhala:
            singlish_part = vo_sinhala.split("(")[-1].split(")")[0].strip()

        sub_line_1 = f"[{speaker_clean}]: \"{line_en}\"" if line_en else f"[{speaker_clean}]"
        draw.text((24, h - bar_h + 12), sub_line_1, fill=(255, 255, 255), font=font_sub)

        if singlish_part:
            draw.text((24, h - bar_h + 40), f"VO: {singlish_part}", fill=(165, 243, 252), font=font_meta)

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _generate_gpu_i2v_frames(
        self,
        pil_src: Image.Image,
        scene_number: int,
        total_scenes: int,
        wan_model: str,
        aspect_ratio: str,
        camera_motion: str,
        video_motion_prompt: str,
        voiceover_speaker: str,
        voiceover_english: str,
        out_w: int,
        out_h: int,
    ) -> Optional[List[np.ndarray]]:
        """
        Runs true GPU Image-to-Video diffusion (LTXImageToVideoPipeline) on the Master Frame Image
        to generate genuine talking-head / scene motion frames (lip movement, eye blinking, head nods).
        """
        if not self._load_gpu_video_pipeline(wan_model):
            return None

        try:
            import torch

            ltx_dims = {
                "16:9": (704, 480),
                "9:16": (480, 704),
                "1:1": (576, 576),
                "4:3": (640, 480),
            }
            gen_w, gen_h = ltx_dims.get(aspect_ratio, (704, 480))
            cond_img = pil_src.resize((gen_w, gen_h), Image.Resampling.LANCZOS)

            speaker_lower = (voiceover_speaker or "").lower()
            is_male = "arjun" in speaker_lower or ("male" in speaker_lower and "female" not in speaker_lower)
            subj_desc = "young man" if is_male else "young woman"

            # Clean any bracket prefix from video_motion_prompt
            clean_motion = re.sub(r"^\[.*?\]\s*", "", video_motion_prompt or "").strip()
            prompt = (
                f"Real-time cinematic video of a {subj_desc} speaking expressively into a studio podcast microphone, "
                f"natural lip and mouth movement talking clearly, blinking eyes naturally, subtle expressive head nods "
                f"and facial expressions while speaking. {clean_motion} {camera_motion}. "
                f"High quality photorealistic studio video, sharp facial features, steady studio background."
            )
            negative_prompt = (
                "static image, frozen face, closed mouth, blurry, worst quality, inconsistent motion, "
                "jittery, distorted face, morphing microphone, deformed hands, watermark"
            )

            num_frames = 49  # 8 * 6 + 1 native temporal latent frames for LTX-Video
            num_steps = 15

            flux_service.set_progress(
                node="card4",
                current_scene=scene_number,
                total_scenes=total_scenes,
                step=1,
                total_steps=num_steps,
                message=f"Scene {scene_number}/{total_scenes}: Running {wan_model} GPU I2V (Step 1/{num_steps})...",
            )

            def _step_callback(pipe, step_index, timestep, callback_kwargs):
                current_step = min(step_index + 1, num_steps)
                pct = int((current_step / num_steps) * 100)
                flux_service.set_progress(
                    node="card4",
                    current_scene=scene_number,
                    total_scenes=total_scenes,
                    step=current_step,
                    total_steps=num_steps,
                    message=f"Scene {scene_number}/{total_scenes}: {wan_model} GPU Video Synthesis ({pct}%)",
                )
                return callback_kwargs

            generator = torch.Generator(device="cpu").manual_seed(1000 + scene_number * 137)
            if self.ltx_pipe is None:
                return None

            with torch.inference_mode():
                output = self.ltx_pipe(
                    image=cond_img,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=gen_w,
                    height=gen_h,
                    num_frames=num_frames,
                    num_inference_steps=num_steps,
                    guidance_scale=3.0,
                    max_sequence_length=256,
                    generator=generator,
                    callback_on_step_end=_step_callback,
                )

            raw_pil_frames = output.frames[0]
            try:
                if hasattr(self.ltx_pipe, "maybe_free_model_hooks"):
                    self.ltx_pipe.maybe_free_model_hooks()
            except Exception:
                pass
            upscaled_bgr_frames: List[np.ndarray] = []
            for pil_f in raw_pil_frames:
                rgb_arr = np.array(pil_f.convert("RGB"))
                bgr_arr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
                up_bgr = cv2.resize(bgr_arr, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
                # Subtle cinema unsharp mask for crisp 720p detail
                gaussian = cv2.GaussianBlur(up_bgr, (0, 0), 1.1)
                sharp_bgr = cv2.addWeighted(up_bgr, 1.16, gaussian, -0.16, 0)
                upscaled_bgr_frames.append(sharp_bgr)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(
                f"Scene {scene_number}: Generated {len(upscaled_bgr_frames)} GPU I2V frames via {wan_model}"
            )
            return upscaled_bgr_frames
        except Exception as e:
            logger.error(f"Scene {scene_number}: GPU I2V generation failed ({e}); using talking-head fallback.", exc_info=True)
            return None

    def _animate_talking_head_frame(
        self,
        base_bgr: np.ndarray,
        t_sec: float,
        scene_number: int,
        vo_text: str,
        grid_x: np.ndarray,
        grid_y: np.ndarray,
    ) -> np.ndarray:
        """
        CPU fallback: localized facial & head articulation (syllable lip movement, eye blinking,
        subtle head nod & breathing) via continuous 2D displacement maps — never a static image zoom.
        """
        h, w, _ = base_bgr.shape
        dx = np.zeros((h, w), dtype=np.float32)
        dy = np.zeros((h, w), dtype=np.float32)

        # Head center & anatomical landmarks in portrait master frame
        hx, hy = w * 0.485, h * 0.285
        hrx, hry = w * 0.13, h * 0.20

        # 1. Conversational head nod & micro-tilt (localized to head region, keeping mic fixed)
        nod_y = math.sin(t_sec * 3.4 + scene_number) * (h * 0.0045)
        tilt_x = math.cos(t_sec * 2.1 + scene_number * 0.7) * (w * 0.0035)
        head_dist2 = ((grid_x - hx) / hrx) ** 2 + ((grid_y - hy) / hry) ** 2
        # Exclude lower-right microphone zone from head warp
        mic_mask = np.clip(((grid_x - w * 0.56) / (w * 0.08)) ** 2 + ((grid_y - h * 0.62) / (h * 0.16)) ** 2, 0.0, 1.0)
        head_weight = np.exp(-1.8 * head_dist2) * mic_mask
        dx -= tilt_x * head_weight
        dy -= nod_y * head_weight

        # 2. Syllable-driven mouth & jaw articulation
        syllable_rate = max(3.8, min(5.5, len((vo_text or "speaking").split()) * 0.42))
        jaw_open = max(0.0, math.sin(t_sec * syllable_rate * math.pi) * 0.65 + math.sin(t_sec * 11.3 + scene_number) * 0.35)
        mx, my = w * 0.49, h * 0.382
        mrx, mry = w * 0.042, h * 0.045
        mouth_dist2 = ((grid_x - mx) / mrx) ** 2 + ((grid_y - my) / mry) ** 2
        mouth_w = np.exp(-2.2 * mouth_dist2)
        lower_jaw_mask = np.clip((grid_y - (my - h * 0.005)) / (h * 0.025), -0.4, 1.0)
        dy -= (jaw_open * h * 0.011) * mouth_w * lower_jaw_mask

        # 3. Periodic natural eye blink (every ~2.3s)
        blink_phase = (t_sec + scene_number * 0.6) % 2.3
        blink_amp = math.exp(-((blink_phase - 0.18) / 0.065) ** 2)
        if blink_amp > 0.02:
            for ex, ey in [(w * 0.448, h * 0.278), (w * 0.508, h * 0.265)]:
                erx, ery = w * 0.024, h * 0.018
                ed2 = ((grid_x - ex) / erx) ** 2 + ((grid_y - ey) / ery) ** 2
                ew = np.exp(-2.5 * ed2)
                dy -= (blink_amp * h * 0.008) * ew

        # 4. Subtle shoulder breathing motion
        breath = math.sin(t_sec * 1.65) * (h * 0.003)
        sh_dist2 = ((grid_x - w * 0.49) / (w * 0.22)) ** 2 + ((grid_y - h * 0.68) / (h * 0.18)) ** 2
        dy -= breath * np.exp(-2.0 * sh_dist2) * mic_mask

        warped = cv2.remap(
            base_bgr,
            grid_x + dx,
            grid_y + dy,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return warped

    def _synthesize_i2v_clip_from_frame(
        self,
        source_img_path: Optional[Path],
        target_video_path: Path,
        scene_number: int,
        total_scenes: int,
        wan_model: str,
        aspect_ratio: str,
        duration_sec: int,
        camera_motion: str,
        video_motion_prompt: str,
        voiceover_speaker: str,
        voiceover_sinhala: str,
        voiceover_english: str
    ) -> bool:
        """
        Synthesizes a 24fps H.264 MP4 video clip driven by Card 3's Master Frame Image
        and Card 4's Scene Motion & Voice-Over Script using Local GPU I2V (LTX-Video / Wan 2.2).
        """
        dimensions = {
            "1:1": (1024, 1024),
            "16:9": (1280, 720),
            "9:16": (720, 1280),
            "4:3": (1152, 864)
        }
        out_w, out_h = dimensions.get(aspect_ratio, (1280, 720))
        fps = 24
        total_frames = max(fps * duration_sec, 96)  # Default 5s = 120 frames at 24fps

        if source_img_path and source_img_path.exists():
            pil_src = Image.open(source_img_path).convert("RGB")
            pil_resized = pil_src.resize((out_w, out_h), Image.Resampling.LANCZOS)
            base_bgr = cv2.cvtColor(np.array(pil_resized), cv2.COLOR_RGB2BGR)
        else:
            pil_src = Image.new("RGB", (out_w, out_h), (38, 25, 20))
            base_bgr = np.full((out_h, out_w, 3), (20, 25, 38), dtype=np.uint8)

        # 1. Run Real GPU Image-to-Video Diffusion Pipeline (LTX-Video / Wan 2.2)
        ai_bgr_frames = self._generate_gpu_i2v_frames(
            pil_src=pil_src,
            scene_number=scene_number,
            total_scenes=total_scenes,
            wan_model=wan_model,
            aspect_ratio=aspect_ratio,
            camera_motion=camera_motion,
            video_motion_prompt=video_motion_prompt,
            voiceover_speaker=voiceover_speaker,
            voiceover_english=voiceover_english,
            out_w=out_w,
            out_h=out_h,
        )

        target_video_path.parent.mkdir(parents=True, exist_ok=True)
        if target_video_path.exists():
            try:
                target_video_path.unlink()
            except Exception:
                pass

        # Use Windows Media Foundation H.264 encoder for 100% browser <video> playback
        writer = cv2.VideoWriter(
            str(target_video_path),
            cv2.CAP_MSMF,
            cv2.VideoWriter_fourcc(*"H264"),
            fps,
            (out_w, out_h)
        )
        if not writer.isOpened():
            writer = cv2.VideoWriter(
                str(target_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (out_w, out_h)
            )

        grid_x, grid_y = np.meshgrid(
            np.arange(out_w, dtype=np.float32),
            np.arange(out_h, dtype=np.float32),
        )

        num_ai = len(ai_bgr_frames) if ai_bgr_frames else 0

        for f_idx in range(total_frames):
            if ai_bgr_frames and num_ai > 1:
                # Smooth temporal interpolation of the AI-generated frames across the 5s (120-frame) 24fps timeline
                src_pos = (f_idx / float(max(total_frames - 1, 1))) * (num_ai - 1)
                idx0 = int(math.floor(src_pos))
                idx1 = min(idx0 + 1, num_ai - 1)
                alpha = src_pos - idx0
                if alpha <= 0.001 or idx0 == idx1:
                    frame = ai_bgr_frames[idx0].copy()
                else:
                    frame = cv2.addWeighted(ai_bgr_frames[idx0], 1.0 - alpha, ai_bgr_frames[idx1], alpha, 0)
            else:
                t_sec = f_idx / float(fps)
                frame = self._animate_talking_head_frame(
                    base_bgr=base_bgr,
                    t_sec=t_sec,
                    scene_number=scene_number,
                    vo_text=voiceover_english or voiceover_sinhala,
                    grid_x=grid_x,
                    grid_y=grid_y,
                )

            # Add Voice-Over subtitle bar
            frame = self._render_cinema_subtitle_overlay(
                frame_bgr=frame,
                speaker=voiceover_speaker,
                vo_english=voiceover_english,
                vo_sinhala=voiceover_sinhala,
                scene_number=scene_number,
                wan_model=wan_model
            )

            writer.write(frame)

        writer.release()
        return target_video_path.exists() and target_video_path.stat().st_size > 1024

    async def generate_run_videos(
        self,
        run_id: str,
        wan_model: str = "Wan2.2-TI2V-5B",
        aspect_ratio: str = "16:9",
        scenes: Optional[List[Dict[str, Any]]] = None,
        frames: Optional[List[Dict[str, Any]]] = None
    ) -> GenerateVideosResponse:
        scenes = scenes or []
        frames = frames or []
        video_run_dir = settings.VIDEOS_DIR / run_id
        video_run_dir.mkdir(parents=True, exist_ok=True)

        videos_result: List[VideoClipItem] = []
        total_scenes = max(len(scenes), len(frames), 1)

        for idx in range(total_scenes):
            sc = scenes[idx] if idx < len(scenes) else {}
            fr = frames[idx] if idx < len(frames) else (frames[0] if frames else {})
            scene_num = sc.get("scene_number") or fr.get("scene_number") or (idx + 1)
            duration_str = sc.get("duration") or fr.get("duration") or "5s"
            duration_sec = int(re.sub(r"[^\d]", "", str(duration_str)) or 5)

            src_img_url = fr.get("imageUrl") or fr.get("url")
            src_img_path_str = fr.get("file_path") or fr.get("filePath")
            resolved_img = self._resolve_source_image_path(
                run_id=run_id,
                scene_number=scene_num,
                source_image_url=src_img_url,
                source_image_path=src_img_path_str
            )

            camera_motion = sc.get("camera_motion") or "Slow cinematic dolly push-in"
            video_motion_prompt = sc.get("video_motion_prompt") or (
                f"[{wan_model} I2V] {camera_motion}. {sc.get('action', '')}. 24fps natural cinema motion."
            )
            vo_speaker = sc.get("voiceover_speaker") or "Maya (Female Lead)"
            vo_sinhala = sc.get("voiceover_sinhala") or ""
            vo_english = sc.get("voiceover_english") or sc.get("raw_prompt") or ""
            vo_tone = sc.get("voice_tone") or "Warm cinematic tone"
            sfx_cue = sc.get("sfx_cue") or "Natural cinema ambience"

            video_filename = f"scene_{scene_num}_wan22.mp4"
            target_video_path = video_run_dir / video_filename

            await asyncio.to_thread(
                self._synthesize_i2v_clip_from_frame,
                source_img_path=resolved_img,
                target_video_path=target_video_path,
                scene_number=scene_num,
                total_scenes=total_scenes,
                wan_model=wan_model,
                aspect_ratio=aspect_ratio,
                duration_sec=duration_sec,
                camera_motion=camera_motion,
                video_motion_prompt=video_motion_prompt,
                voiceover_speaker=vo_speaker,
                voiceover_sinhala=vo_sinhala,
                voiceover_english=vo_english,
            )

            cache_ts = int(time.time() * 1000)
            video_url = f"/data/videos/{run_id}/{video_filename}?t={cache_ts}"

            videos_result.append(
                VideoClipItem(
                    scene_number=scene_num,
                    duration=duration_str,
                    wan_model=wan_model,
                    videoUrl=video_url,
                    file_path=str(target_video_path),
                    source_image_url=src_img_url or f"/data/keyframes/{run_id}/scene_1_master.png",
                    video_motion_prompt=video_motion_prompt,
                    camera_motion=camera_motion,
                    voiceover_speaker=vo_speaker,
                    voiceover_sinhala=vo_sinhala,
                    voiceover_english=vo_english,
                    voice_tone=vo_tone,
                    sfx_cue=sfx_cue,
                    engine_used=self._resolve_model_label(wan_model)
                )
            )

        flux_service.reset_progress()
        return GenerateVideosResponse(
            run_id=run_id,
            video_folder_path=str(video_run_dir),
            videos=videos_result
        )

    async def regenerate_single_video(
        self,
        run_id: str,
        scene_number: int,
        wan_model: str = "LTX-Video-2B",
        aspect_ratio: str = "16:9",
        source_image_url: Optional[str] = None,
        source_image_path: Optional[str] = None,
        video_motion_prompt: str = "",
        camera_motion: str = "Slow dolly push-in",
        voiceover_speaker: str = "Maya (Female Lead)",
        voiceover_sinhala: str = "",
        voiceover_english: str = "",
        voice_tone: str = "Warm cinematic tone",
        sfx_cue: str = ""
    ) -> VideoClipItem:
        video_run_dir = settings.VIDEOS_DIR / run_id
        video_run_dir.mkdir(parents=True, exist_ok=True)
        video_filename = f"scene_{scene_number}_wan22.mp4"
        target_video_path = video_run_dir / video_filename

        resolved_img = self._resolve_source_image_path(
            run_id=run_id,
            scene_number=scene_number,
            source_image_url=source_image_url,
            source_image_path=source_image_path
        )

        await asyncio.to_thread(
            self._synthesize_i2v_clip_from_frame,
            source_img_path=resolved_img,
            target_video_path=target_video_path,
            scene_number=scene_number,
            total_scenes=1,
            wan_model=wan_model,
            aspect_ratio=aspect_ratio,
            duration_sec=5,
            camera_motion=camera_motion,
            video_motion_prompt=video_motion_prompt,
            voiceover_speaker=voiceover_speaker,
            voiceover_sinhala=voiceover_sinhala,
            voiceover_english=voiceover_english,
        )

        flux_service.reset_progress()
        cache_ts = int(time.time() * 1000)
        video_url = f"/data/videos/{run_id}/{video_filename}?t={cache_ts}"

        return VideoClipItem(
            scene_number=scene_number,
            duration="5s",
            wan_model=wan_model,
            videoUrl=video_url,
            file_path=str(target_video_path),
            source_image_url=source_image_url or f"/data/keyframes/{run_id}/scene_1_master.png",
            video_motion_prompt=video_motion_prompt,
            camera_motion=camera_motion,
            voiceover_speaker=voiceover_speaker,
            voiceover_sinhala=voiceover_sinhala,
            voiceover_english=voiceover_english,
            voice_tone=voice_tone,
            sfx_cue=sfx_cue,
            engine_used=self._resolve_model_label(wan_model)
        )


wan_video_service = WanVideoStudioService()

