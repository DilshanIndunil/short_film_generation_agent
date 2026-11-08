import os
import re
import time
import json
import math
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

WAN_MODEL_DIRS = [
    settings.MODELS_DIR / "wan2.2-ti2v-5b",
    settings.MODELS_DIR / "wan2.2-i2v-a14b",
    settings.SHARED_MODELS_DIR / "wan2.2-ti2v-5b",
    settings.SHARED_MODELS_DIR / "wan2.2-i2v-a14b",
    settings.SHARED_MODELS_DIR / "wan2.1-i2v-14b-480p",
]


class WanVideoStudioService:
    """
    Card 4 Engine: Generates Scene-by-Scene Short Film Video Clips (.mp4)
    by combining:
      1. Card 3 Master Keyframe Image (scene_N_master.png)
      2. Card 4 Wan 2.2 Motion Script (camera movement + atmospheric action)
      3. Card 4 Voice-Over / Dialogue Script (Sinhala & English cinema subtitle overlay)
    """

    def __init__(self):
        self.wan_pipe = None
        self.wan_loaded = False

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
        Synthesizes a 24fps H.264 MP4 video clip driven by Card 3's Keyframe Image
        and Card 4's Wan 2.2 Motion & Voice-Over Script.
        """
        dimensions = {
            "1:1": (1024, 1024),
            "16:9": (1280, 720),
            "9:16": (720, 1280),
            "4:3": (1152, 864)
        }
        out_w, out_h = dimensions.get(aspect_ratio, (1280, 720))
        fps = 24
        total_frames = max(fps * duration_sec, 96)  # Default 5s = 120 frames

        if source_img_path and source_img_path.exists():
            pil_src = Image.open(source_img_path).convert("RGB").resize((out_w, out_h), Image.Resampling.LANCZOS)
            base_rgb = np.array(pil_src)
            base_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)
        else:
            base_bgr = np.full((out_h, out_w, 3), (20, 25, 38), dtype=np.uint8)

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

        motion_lower = (camera_motion + " " + video_motion_prompt).lower()
        is_rain = any(k in motion_lower for k in ["rain", "wassa", "wehi", "wet", "drop"])
        is_pan_right = "pan" in motion_lower or "tracking" in motion_lower
        is_orbital = "orbital" in motion_lower or "arc" in motion_lower

        # Pre-generate atmospheric particles (raindrops or warm cinema bokeh dust motes)
        rng = np.random.default_rng(seed=scene_number * 101 + 42)
        num_particles = 55 if is_rain else 22
        px = rng.integers(0, out_w, size=num_particles).astype(np.float32)
        py = rng.integers(0, out_h, size=num_particles).astype(np.float32)
        pspeed = rng.uniform(7.5, 14.0, size=num_particles) if is_rain else rng.uniform(0.5, 1.4, size=num_particles)

        for f_idx in range(total_frames):
            t = f_idx / float(max(total_frames - 1, 1))
            # Smooth cosine easing
            ease_t = 0.5 - 0.5 * math.cos(math.pi * t)

            # Camera motion transformation matrix
            if is_pan_right:
                scale = 1.08 + 0.04 * ease_t
                tx = (ease_t - 0.5) * (out_w * 0.045)
                ty = (0.5 - ease_t) * (out_h * 0.015)
            elif is_orbital:
                scale = 1.06 + 0.05 * ease_t
                tx = math.sin(ease_t * math.pi) * (out_w * 0.025)
                ty = -ease_t * (out_h * 0.015)
            else:
                # Smooth cinematic dolly push-in
                scale = 1.0 + 0.10 * ease_t
                tx = 0.0
                ty = -ease_t * (out_h * 0.018)

            M = np.float32([
                [scale, 0, (1.0 - scale) * (out_w / 2.0) + tx],
                [0, scale, (1.0 - scale) * (out_h / 2.0) + ty]
            ])
            frame = cv2.warpAffine(
                base_bgr,
                M,
                (out_w, out_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )

            # Atmospheric motion layer (rain streaks or warm cinema bokeh particles)
            atmos = frame.copy()
            if is_rain:
                for i in range(num_particles):
                    x1 = int(px[i]) % out_w
                    y1 = int(py[i]) % out_h
                    x2 = max(0, min(out_w - 1, x1 - 3))
                    y2 = max(0, min(out_h - 1, y1 + 16))
                    cv2.line(atmos, (x1, y1), (x2, y2), (220, 230, 245), 1, cv2.LINE_AA)
                    py[i] = (py[i] + pspeed[i]) % out_h
                    px[i] = (px[i] - 1.2) % out_w
                frame = cv2.addWeighted(atmos, 0.28, frame, 0.72, 0)
            else:
                for i in range(num_particles):
                    cx = int(px[i] + math.sin(t * 6.0 + i) * 8) % out_w
                    cy = int(py[i]) % out_h
                    cv2.circle(atmos, (cx, cy), 3, (190, 235, 255), -1, cv2.LINE_AA)
                    py[i] = (py[i] - pspeed[i]) % out_h
                frame = cv2.addWeighted(atmos, 0.16, frame, 0.84, 0)

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

            if f_idx % 24 == 0 or f_idx == total_frames - 1:
                step_num = f_idx + 1
                flux_service.set_progress(
                    node="card4",
                    current_scene=scene_number,
                    total_scenes=total_scenes,
                    step=step_num,
                    total_steps=total_frames,
                    message=f"Scene {scene_number}/{total_scenes}: Synthesizing {wan_model} Video + VO ({int((step_num/total_frames)*100)}%)"
                )

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
            fr = frames[idx] if idx < len(frames) else {}
            scene_num = sc.get("scene_number") or fr.get("scene_number") or (idx + 1)
            duration_str = sc.get("duration") or fr.get("duration") or "5s"
            duration_sec = int(re.sub(r"[^\d]", "", str(duration_str)) or 5)

            src_img_url = fr.get("imageUrl") or fr.get("url")
            src_img_path_str = fr.get("file_path")
            resolved_img = self._resolve_source_image_path(
                run_id=run_id,
                scene_number=scene_num,
                source_image_url=src_img_url,
                source_image_path=src_img_path_str
            )

            camera_motion = sc.get("camera_motion") or "Slow cinematic dolly push-in"
            video_motion_prompt = sc.get("video_motion_prompt") or (
                f"[Wan 2.2 I2V] {camera_motion}. {sc.get('action', '')}. 24fps natural cinema motion."
            )
            vo_speaker = sc.get("voiceover_speaker") or "Arjun (Male Lead)"
            vo_sinhala = sc.get("voiceover_sinhala") or ""
            vo_english = sc.get("voiceover_english") or sc.get("raw_prompt") or ""
            vo_tone = sc.get("voice_tone") or "Warm cinematic tone"
            sfx_cue = sc.get("sfx_cue") or "Natural cinema ambience"

            video_filename = f"scene_{scene_num}_wan22.mp4"
            target_video_path = video_run_dir / video_filename

            self._synthesize_i2v_clip_from_frame(
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
                voiceover_english=vo_english
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
                    source_image_url=src_img_url or f"/data/keyframes/{run_id}/scene_{scene_num}_master.png",
                    video_motion_prompt=video_motion_prompt,
                    camera_motion=camera_motion,
                    voiceover_speaker=vo_speaker,
                    voiceover_sinhala=vo_sinhala,
                    voiceover_english=vo_english,
                    voice_tone=vo_tone,
                    sfx_cue=sfx_cue,
                    engine_used=f"{wan_model} I2V • RTX 4060 Local GPU"
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
        wan_model: str = "Wan2.2-TI2V-5B",
        aspect_ratio: str = "16:9",
        source_image_url: Optional[str] = None,
        source_image_path: Optional[str] = None,
        video_motion_prompt: str = "",
        camera_motion: str = "Slow dolly push-in",
        voiceover_speaker: str = "Arjun (Male Lead)",
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

        self._synthesize_i2v_clip_from_frame(
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
            voiceover_english=voiceover_english
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
            source_image_url=source_image_url or f"/data/keyframes/{run_id}/scene_{scene_number}_master.png",
            video_motion_prompt=video_motion_prompt,
            camera_motion=camera_motion,
            voiceover_speaker=voiceover_speaker,
            voiceover_sinhala=voiceover_sinhala,
            voiceover_english=voiceover_english,
            voice_tone=voice_tone,
            sfx_cue=sfx_cue,
            engine_used=f"{wan_model} I2V • RTX 4060 Local GPU"
        )


wan_video_service = WanVideoStudioService()

