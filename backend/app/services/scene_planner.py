import os
import re
import json
import time
import random
import asyncio
import logging
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional, Tuple
from app.core.config import settings
from app.schemas.schemas import OptimizedSceneItem, MultiScenePlanResponse

logger = logging.getLogger("FLUX_Prompt_Optimizer")

# ============================================================================
# EXACT CHARACTER VISUAL PROFILES (Derived from Studio Turnaround Sheets)
# Plus Custom LoRA Trigger Tokens: `arjun_male` and `maya_female`
# ============================================================================
CHARACTER_PROFILES = {
    "male": (
        "arjun_male, a handsome young South Asian man in his mid-20s with thick wavy black hair, "
        "a neatly trimmed full beard and mustache, warm brown skin, athletic build, wearing a fitted "
        "plain black crew-neck short-sleeve t-shirt, beige chino trousers, a black wristwatch on his "
        "left wrist, and clean white sneakers"
    ),
    "female": (
        "maya_female, a beautiful young South Asian woman in her early 20s with long dark wavy hair "
        "styled in a half-up top-knot bun with soft strands framing her face, expressive dark eyes, "
        "warm natural smile, wearing a fitted black crew-neck short-sleeve t-shirt, high-waisted "
        "wide-leg beige tailored trousers, a delicate silver wristwatch, and clean white sneakers"
    )
}


class FilmScenePlannerAgent:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.model_name = settings.GEMINI_MODEL or "gemini-3-flash-preview"
        self.active = False
        self.model = None
        self._init_gemini()

    def _init_gemini(self):
        self.api_key = (settings.GEMINI_API_KEY or "").strip()
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            self.active = False
            return
        # Use direct HTTPS REST API with gemini-3-flash-preview (supports both AQ.* and AIza* keys)
        self.active = True
        logger.info(f"Gemini Film Director REST Agent configured for model: {self.model_name}")

    def _call_gemini_rest(self, prompt_text: str) -> Optional[str]:
        """Calls Gemini REST API directly with gemini-3-flash-preview / gemini-3.1-flash-lite-preview."""
        if not self.api_key or self.api_key == "your_gemini_api_key_here":
            return None
        candidates_to_try = []
        for m in [self.model_name, "gemini-3-flash-preview", "gemini-3.1-flash-lite-preview", "gemma-4-26b-a4b-it"]:
            if m and m not in candidates_to_try and "2.0" not in m and "2.5" not in m and "1.5" not in m:
                candidates_to_try.append(m)

        for candidate in candidates_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{candidate}:generateContent?key={self.api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt_text}]}],
                "generationConfig": {"temperature": 0.9, "topP": 0.95}
            }
            try:
                req = urllib.request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                with urllib.request.urlopen(req, timeout=25) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    candidates = data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        for p in parts:
                            if "text" in p and p["text"].strip():
                                logger.info(f"[Gemini AI] Successfully synthesized scene plan using {candidate}!")
                                return p["text"]
            except Exception as e:
                logger.warning(f"Gemini REST call notice ({candidate}): {e}")
        return None

    def _translate_singlish_hints(self, raw_text: str) -> str:
        """
        Enhances Singlish / Sinhala phrases into clear English cinematic descriptions
        while 100% preserving all English words and scenario details typed by the user.
        """
        if not raw_text or not raw_text.strip():
            return raw_text

        text_lower = raw_text.lower()
        pure_singlish_map = {
            "kollekui kellekui": "a young man and a young woman together",
            "kollek": "a young man",
            "kellek": "a young woman",
            "dennama": "both the young man and young woman",
            "katha karanawa": "having an engaging emotional conversation face to face",
            "katha krnwa": "having an engaging emotional conversation face to face",
            "hina wenawa": "smiling warmly with genuine emotion",
            "wassa": "falling rain with wet reflective ground",
            "wehi": "rainy atmosphere with glistening raindrops",
            "raa": "at night with moody cinematic streetlights",
            "ude": "in the early morning golden hour sunlight",
            "para": "along a scenic urban street road",
            "pare": "walking along a cinematic street",
            "muhuda": "by the ocean shoreline with gentle waves",
            "kanda": "on a misty scenic mountain viewpoint overlooking lush hills",
            "gedara": "inside a modern sunlit living room interior",
            "duwanawa": "running dynamically with cinematic motion",
            "yanawa": "walking side by side naturally",
            "balan innawa": "looking thoughtfully into the distance",
            "awidinawa": "walking gracefully side by side",
            "adare": "romantic tender emotional connection",
            "dukai": "melancholic emotional expression",
            "sathutin": "joyful and radiant expression",
        }

        translated_parts = []
        for singlish, eng in sorted(pure_singlish_map.items(), key=lambda x: len(x[0]), reverse=True):
            if re.search(rf"\b{re.escape(singlish)}\b", text_lower):
                if singlish in ("kollek", "kellek") and "kollekui kellekui" in text_lower:
                    continue
                translated_parts.append(eng)

        if translated_parts:
            return f"{raw_text.strip()} ({', '.join(translated_parts)})"
        return raw_text.strip()

    def _build_character_string(self, characters: List[str], raw_text: str) -> str:
        raw_lower = raw_text.lower()
        chars = [c.lower() for c in (characters or [])]

        has_male = "male" in chars or "both" in chars
        has_female = "female" in chars or "both" in chars

        if not chars or "auto" in chars:
            if any(w in raw_lower for w in ["couple", "both", "two", "together", "kollekui kellekui", "dennama", "man and woman", "boy and girl"]):
                has_male = True
                has_female = True
            elif any(w in raw_lower for w in ["woman", "girl", "female", "she", "her", "kellek", "lady"]):
                has_female = True
                has_male = False
            elif any(w in raw_lower for w in ["man", "boy", "male", "he", "his", "kollek", "guy"]):
                has_male = True
                has_female = False
            else:
                has_male = True
                has_female = True

        if has_male and has_female:
            return f"Featuring two consistent lead characters: (1) {CHARACTER_PROFILES['male']}, and (2) {CHARACTER_PROFILES['female']}"
        elif has_male:
            return f"Featuring lead male character: {CHARACTER_PROFILES['male']}"
        elif has_female:
            return f"Featuring lead female character: {CHARACTER_PROFILES['female']}"
        return "Cinematic human subjects with realistic skin texture and natural expressions"

    def _synthesize_custom_scene_elements(self, raw_input: str, enhanced_scenario: str, idx: int) -> Dict[str, str]:
        """
        Deeply analyzes the user's exact scenario text and synthesizes a brand-new,
        context-accurate Environment, Lighting, Camera Rig, Wan 2.2 Motion Script,
        and Sinhala + English Voice-Over Script tailored to the user's words.
        """
        raw_lower = enhanced_scenario.lower()
        clean_topic = raw_input.strip().rstrip(".")

        # 1. Camera & Lens Rig Pool (Fresh selection on every generation)
        camera_setups = [
            ("Medium two-shot at eye level, 50mm Cooke Anamorphic/i prime lens, shallow f/1.8 depth of field", "Slow cinematic dolly push-in at 24fps"),
            ("Cinematic medium close-up, 85mm Zeiss Master Prime f/1.4 lens, creamy bokeh background", "Subtle handheld cinema breathing motion with slow focus pull"),
            ("Wide cinematic establishing shot, 35mm ARRI Master Anamorphic lens, rich spatial depth", "Smooth lateral Steadicam tracking pan across the frame"),
            ("Over-the-shoulder dramatic cinema angle, 50mm Panavision Primo lens, razor-sharp eye focus", "Slow orbital arc camera movement around the characters"),
            ("Low-angle expressive cinema shot, 40mm Cooke S4 prime lens, subtle warm anamorphic flare", "Steady gimbal forward tracking movement"),
            ("Intimate eye-level widescreen frame, 65mm ARRI Prime DNA lens, soft background separation", "Gentle crane push-in with natural foreground parallax")
        ]

        if any(w in raw_lower for w in ["close-up", "close up", "portrait", "face"]):
            cam, motion = random.choice([
                ("Intimate close-up shot, 85mm f/1.4 prime lens, ultra-shallow depth of field", "Slow subtle push-in on facial expressions"),
                ("Emotional medium close-up, 75mm anamorphic prime lens, soft background bokeh", "Gentle focus pull from eyes to background")
            ])
        elif any(w in raw_lower for w in ["wide", "full body", "walking", "running", "landscape"]):
            cam, motion = random.choice([
                ("Wide cinematic full-frame shot, 35mm anamorphic lens, deep cinema composition", "Smooth gimbal tracking shot following the characters"),
                ("Sweeping wide establishing shot, 28mm cinema prime lens, expansive depth", "Slow lateral dolly tracking movement")
            ])
        else:
            cam, motion = random.choice(camera_setups)

        cinema_styles = [
            "shot on ARRI Alexa 65 (Kodak Vision3 500T 5219 film stock grade)",
            "shot on RED V-Raptor 8K VV (Cooke Anamorphic theatrical grade)",
            "shot on Sony VENICE 2 8K (Panavision Primo 70 organic cinema grade)",
            "shot on ARRI Alexa Mini LF (IMAX 70mm photorealistic color science)"
        ]
        chosen_style = random.choice(cinema_styles)

        # 2. Time-of-day & Weather detection from user scenario
        is_night = any(w in raw_lower for w in ["night", "raa", "evening", "dark", "dinner", "moon", "neon", "streetlights"])
        is_rain = any(w in raw_lower for w in ["rain", "rainy", "rain-streaked", "wassa", "wehi", "wet", "umbrella", "storm"])
        is_morning = any(w in raw_lower for w in ["morning", "ude", "sunrise", "dawn"])
        is_sunset = any(w in raw_lower for w in ["sunset", "golden hour", "dusk", "hawasa"])

        # 3. Context-specific Environment, Lighting, Motion & Sinhala/English Voice-Over
        if any(w in raw_lower for w in ["coffee", "cafe", "restaurant", "tea", "bistro", "table", "cup"]):
            env = random.choice([
                f"inside a cozy, atmospheric artisan coffee shop depicting '{clean_topic}', with warm timber tables, ceramic cups with rising steam, and rain-streaked glass windows overlooking glowing city bokeh",
                f"at an intimate corner window table inside a modern vintage cafe ('{clean_topic}'), featuring warm Edison pendant bulbs reflecting on the glass pane and authentic cafe props",
                f"inside a warm cinematic bistro setting for '{clean_topic}', with soft condensation on the window glass, handcrafted wooden furniture, and deep bokeh background separation"
            ])
            light = random.choice([
                "warm 2700K tungsten pendant lamp glow balanced with cool nocturnal blue rain ambience through the window, soft rim light on faces",
                "golden amber cafe table lamp illumination contrasting with moody teal city bokeh reflections on the glass",
                "soft diffused warm interior key light with natural catchlights in the eyes and subtle cinematic shadows"
            ])
            vo_pool = [
                (
                    "Arjun (Male Lead)",
                    f"කෝපි කෝප්පයක උණුසුමත් එක්ක, ඔයාගේ ඇස්වල තිබුණු ඒ නිහඬ කතාව මට හැමදේම කිව්වා. (Kopi koppayaka unusumath ekka, oyage aswala thibunu e nihada kathawa mata hamadema kiwwa.)",
                    f"Over the warmth of steaming coffee, the quiet look in your eyes told me everything words never could.",
                    f"Delicate wisps of steam curl upward from the coffee cups while raindrops trickle down the window glass; Arjun leans in slightly, looking at Maya with a warm, genuine smile as they converse."
                ),
                (
                    "Maya (Female Lead)",
                    f"පිටත වැස්ස කොච්චර සැර වුණත්, ඔයා ළඟදී මගේ හිතට දැනෙන්නේ පුදුමාකාර නිදහසක්. (Pitatha wassa kochchara sara wunath, oya lagadi mage hithata danenne pudumakara nidahasak.)",
                    f"No matter how hard the rain falls outside, sitting here with you feels like the safest place in the world.",
                    f"Maya gently wraps her hands around the warm ceramic cup, lifting her gaze to meet Arjun's eyes with a tender expression as city lights shimmer outside."
                ),
                (
                    "Arjun & Maya (Dialogue)",
                    f"සමහර හමුවීම් ජීවිතේට එන්නේ අහම්බෙන් වුණාට, ඒවා හිතේ නවතින්නේ සදහටම. (Samahara hamuweem jeewitheta enne ahamben wunata, ewa hithe nawathinne sadahatama.)",
                    f"Some meetings happen by chance, yet they stay in the heart forever.",
                    f"Both Arjun and Maya share an authentic, unhurried conversation across the table, with natural head nods, expressive eye contact, and rising coffee steam at 24fps."
                ),
                (
                    "Arjun (Male Lead)",
                    f"මේ නිහඬ රාත්‍රියේ, අපි කතා කරපු හැම වචනයක්ම අලුත් බලාපොරොත්තුවක් වුණා. (Me nihada rathriye, api katha karapu hama wachanayakma aluth balaporothththuwak wuna.)",
                    f"In the quiet glow of this cafe tonight, every word we shared became a brand new promise.",
                    f"Warm pendant light catches their expressive eyes as Arjun speaks softly and Maya smiles in response, while soft bokeh headlights glide past the rainy window."
                )
            ]
            vo_speaker, vo_si, vo_en, motion_detail = random.choice(vo_pool)
            vo_tone = "Warm, intimate & romantic short-film tone"
            sfx = "Soft rain tapping on cafe glass window, gentle acoustic guitar notes, subtle ceramic coffee cup clink"

        elif any(w in raw_lower for w in ["beach", "ocean", "sea", "muhuda", "waves", "shore", "sand", "coast"]):
            env = random.choice([
                f"along a breathtaking tropical beach shoreline depicting '{clean_topic}', with gentle turquoise ocean waves washing over wet golden sand and a glowing horizon",
                f"by the shimmering ocean edge ('{clean_topic}') with foamy tide ripples, coastal breeze, and atmospheric golden-hour mist"
            ])
            light = random.choice([
                "warm golden-hour sunset backlighting with anamorphic sun flare and soft natural fill on skin tones",
                "amber-magenta twilight horizon reflections over wet sand with crisp rim lighting on hair and shoulders"
            ])
            vo_pool = [
                (
                    "Maya (Female Lead)",
                    "මුහුදු රළ වගේම සමහර මතකයන් කවදාවත් හිතෙන් මැකෙන්නේ නෑ. (Muhudu rala wagema samahara mathakayan kawadawath hithen makenne naa.)",
                    "Just like the endless ocean tide, some memories never fade from the heart.",
                    "Foamy ocean waves roll gently onto the wet sand as a coastal breeze stirs their dark wavy hair while Arjun and Maya walk side by side."
                ),
                (
                    "Arjun (Male Lead)",
                    "ඉර බැහැගෙන යන මේ වෙරළේදී, ඔයාගේ හිනාව මගේ මුළු ලෝකයම එළිය කළා. (Ira bahagena yana me weraledi, oyage hinawa mage mulu lokayama eliya kala.)",
                    "Along this golden shoreline at sunset, your smile lit up my entire world.",
                    "Arjun turns toward Maya as golden sunset light glints across the ocean waves behind them, capturing natural wind movement in their hair and clothes."
                )
            ]
            vo_speaker, vo_si, vo_en, motion_detail = random.choice(vo_pool)
            vo_tone = "Dreamy, emotional golden-hour narration"
            sfx = "Gentle ocean waves breaking on shore, coastal wind breeze, warm ambient strings"

        elif any(w in raw_lower for w in ["mountain", "hill", "kanda", "forest", "nature", "valley", "mist", "tea estate"]):
            env = random.choice([
                f"at a scenic misty mountain viewpoint depicting '{clean_topic}', surrounded by lush emerald highland hills and drifting clouds",
                f"along a winding highland trail overlooking misty valleys ('{clean_topic}') with natural foliage and deep atmospheric perspective"
            ])
            light = random.choice([
                "soft diffused morning golden sunlight piercing through mountain mist with volumetric rays",
                "ethereal highland natural daylight with warm rim light and crisp shadow detail"
            ])
            vo_pool = [
                (
                    "Maya (Female Lead)",
                    "මීදුම අතරින් පේන මේ කඳු යාය වගේම, ඔයා එක්ක ගෙවන හැම තත්පරයක්ම හරිම නිස්කලංකයි. (Meeduma atharin pena me kandu yaya wagema, oya ekka gewana hama thathparayakma harima niskalankai.)",
                    "Like these quiet misty hills, every second spent with you brings peace to my soul.",
                    "Wisps of white mountain mist drift across the lush green valley behind them while a gentle breeze moves their hair and they exchange a heartfelt look."
                ),
                (
                    "Arjun (Male Lead)",
                    "මේ නිහඬ කඳු මුදුනේදී, අපේ හිත්වල තිබුණු හැම ප්‍රශ්නයකටම උත්තර හම්බවුණා. (Me nihada kandu mudunedi, ape hithwala thibunu hama prashnayakatama uththara hambawuna.)",
                    "Standing high above the clouds, every question in our hearts finally found its answer.",
                    "Arjun and Maya stand overlooking the misty peaks, turning smoothly toward each other with natural breathing and expressive eye contact."
                )
            ]
            vo_speaker, vo_si, vo_en, motion_detail = random.choice(vo_pool)
            vo_tone = "Peaceful, airy & heartfelt cinema narration"
            sfx = "Soft mountain wind breeze, distant morning birdsong, emotional cello & piano melody"

        elif is_rain or any(w in raw_lower for w in ["street", "road", "para", "pare", "city", "walk", "bus", "train", "station"]):
            env = random.choice([
                f"along an atmospheric urban street setting depicting '{clean_topic}', with realistic architectural depth, glowing streetlamps, and wet pavement reflections",
                f"in a cinematic outdoor city location ('{clean_topic}') with rich background bokeh, authentic street textures, and moody depth of field"
            ])
            light = random.choice([
                "dramatic warm streetlight key illumination balanced with cool evening ambient fill and glistening rim highlights",
                "naturalistic cinematic street lighting with rich shadow contrast and soft bokeh highlights"
            ])
            vo_pool = [
                (
                    "Arjun (Male Lead)",
                    f"කලබලකාරී නගරය මැද්දේ වුණත්, ඔයා එක්ක යන මේ ගමන මට විශේෂයි. (Kalabalakari nagaraya madde wunath, oya ekka yana me gamana mata visheshai.)",
                    f"Even in the middle of a busy city street, walking beside you makes everything else fade away.",
                    f"Natural 24fps walking and head-turning motion as Arjun and Maya move through '{clean_topic}', with realistic fabric motion and shifting streetlamp bokeh."
                ),
                (
                    "Maya (Female Lead)",
                    f"ඒ මොහොතේ අපි දෙන්නා අතරේ තිබුණු නිහඬ බැල්ම වචන දහසකට වඩා බලවත් වුණා. (E mohothe api denna athare thibunu nihada balma wachana dahasakata wada balawath wuna.)",
                    f"In that fleeting moment on the street, one quiet glance spoke louder than a thousand words.",
                    f"Smooth Steadicam tracking motion capturing Arjun and Maya in '{clean_topic}', exchanging an emotional glance as city lights shimmer in the background."
                )
            ]
            vo_speaker, vo_si, vo_en, motion_detail = random.choice(vo_pool)
            vo_tone = "Moody, reflective & cinematic voice-over"
            sfx = "Ambient city atmosphere, gentle footsteps, soft emotional piano score"

        else:
            # Custom scenario-driven synthesis for any unique user scenario!
            time_lighting = (
                "moody nocturnal cinema lighting with warm practical lamp glow and cool shadow contrast"
                if is_night else
                "warm golden-hour sunlight with soft volumetric rays and natural skin highlights"
                if (is_morning or is_sunset) else
                random.choice([
                    "naturalistic Hollywood three-point cinema lighting with soft key light, subtle rim light, and rich anamorphic color grading",
                    "dramatic warm-amber key lighting balanced with soft ambient fill and natural eye catchlights",
                    "high-dynamic-range cinema illumination with volumetric atmospheric depth and realistic shadow roll-off"
                ])
            )
            env = random.choice([
                f"in a richly detailed, photorealistic film environment specifically staged for '{clean_topic}', with authentic props, realistic architecture, and shallow depth-of-field separation",
                f"inside a cinematically art-directed location depicting '{clean_topic}', featuring lifelike surface textures, natural spatial depth, and atmospheric background bokeh",
                f"in an authentic live-action short-film setting for '{clean_topic}', with realistic environmental details and theatrical production design"
            ])
            light = time_lighting
            vo_pool = [
                (
                    "Arjun (Male Lead)",
                    f"ජීවිතයේ සමහර මොහොතවල් තියෙනවා, කවදාවත් අමතක කරන්න බැරි විදිහට හිතේ ඇඳෙන... ({clean_topic})",
                    f"There are certain moments in life that etch themselves into the heart forever — {clean_topic}.",
                    f"Starting from the Scene {idx + 1} master frame, Arjun and Maya perform '{clean_topic}' with lifelike 24fps facial micro-expressions, natural eye contact, and subtle body movement."
                ),
                (
                    "Maya (Female Lead)",
                    f"අපි දෙන්නාගේ කතාවේ ලස්සනම පරිච්ඡේදය ලියවුණේ මේ සුවිශේෂී මොහොතේදීයි. ({clean_topic})",
                    f"The most unforgettable chapter of our story came alive right in this moment — {clean_topic}.",
                    f"Fluid 24fps cinematic motion bringing '{clean_topic}' to life, with natural breathing, expressive glances between the characters, and smooth depth-of-field parallax."
                ),
                (
                    "Arjun & Maya (Narrator)",
                    f"වචනවලින් කියන්න බැරි හැඟීම් ගොඩක් ඒ නිහඬ තත්පර කිහිපය ඇතුළේ තිබුණා. ({clean_topic})",
                    f"More feelings than words could ever express lived inside those few quiet seconds — {clean_topic}.",
                    f"Realistic live-action character movement depicting '{clean_topic}', with gentle head turns, authentic fabric physics, and cinematic lighting shifts at 24fps."
                )
            ]
            vo_speaker, vo_si, vo_en, motion_detail = random.choice(vo_pool)
            vo_tone = "Expressive cinematic short-film narrator tone"
            sfx = "Rich ambient room/location atmosphere with a warm, emotional instrumental film score"

        return {
            "style": chosen_style,
            "camera": cam,
            "camera_motion": motion,
            "environment": env,
            "lighting": light,
            "vo_speaker": vo_speaker,
            "vo_sinhala": vo_si,
            "vo_english": vo_en,
            "vo_tone": vo_tone,
            "sfx_cue": sfx,
            "motion_detail": motion_detail
        }

    async def plan_multi_scenes(
        self,
        scenes: List[Dict[str, Any]],
        aspect_ratio: str = "16:9",
        flux_model: str = "Qwen-Image-2.1",
        characters: Optional[List[str]] = None
    ) -> MultiScenePlanResponse:
        """
        Converts user scenario(s) from Card 1 into fresh, newly synthesized Card 2 Master Prompts
        and Card 4 Video Motion & Voice-Over Scripts with live step-by-step progress reporting.
        """
        total_scenes = max(len(scenes), 1)
        default_chars = characters if characters is not None else ["male", "female"]

        def report_step(step_num: int, total_st: int, msg: str):
            try:
                from app.services.flux_service import flux_service
                flux_service.set_progress(
                    node="card2",
                    current_scene=1,
                    total_scenes=total_scenes,
                    step=step_num,
                    total_steps=total_st,
                    message=msg
                )
            except Exception:
                pass

        report_step(1, 4, f"Step 1/4: Analyzing {total_scenes} Scenario(s) & Character Turnaround Profiles...")
        await asyncio.sleep(0.35)

        optimized_list: List[OptimizedSceneItem] = []
        total_seconds = 0
        variation_nonce = random.randint(1000, 9999)

        # 1. Try Gemini LLM (via SDK or REST API) if a valid key is available
        if self.active and self.api_key:
            try:
                report_step(2, 4, f"Step 2/4: Synthesizing fresh {flux_model} Master Prompt(s) & Scripts (Variation #{variation_nonce})...")
                char_context = self._build_character_string(default_chars, " ".join([s.get("text", "") for s in scenes]))
                system_prompt = f"""
You are an award-winning Short Film Director and AI Prompt Engineer specializing in {flux_model}.
Generate a COMPLETELY FRESH, UNIQUE creative variation (Variation Seed #{variation_nonce}) for the user's scenario(s).
Maintain 100% Character Consistency using these Turnaround Sheet profiles:
{char_context}

For each scene, expand the user's exact scenario into:
1. 'optimized_image_prompt': A vivid 75-115 word natural-language cinematic prompt specifying camera rig, lens focal length, character appearance, exact action from the user's scenario, setting, and lighting.
2. 'video_motion_prompt': A detailed Wan 2.2 24fps Image-to-Video motion script describing camera motion and character movement.
3. 'voiceover_sinhala' & 'voiceover_english': Fresh, emotional Sinhala (with Singlish phonetics) and English short-film voice-over lines matching the user's scenario.

Return ONLY a valid JSON array with {len(scenes)} objects matching this exact schema:
[
  {{
    "id": 1,
    "scene_number": 1,
    "duration": "5s",
    "raw_prompt": "<original user scenario>",
    "optimized_image_prompt": "<master image prompt>",
    "character_description": "<character summary>",
    "action": "<specific character action>",
    "camera": "<camera shot & lens>",
    "camera_motion": "<camera motion>",
    "lighting": "<lighting setup>",
    "environment": "<detailed setting>",
    "flux_model": "{flux_model}",
    "video_motion_prompt": "<Wan 2.2 motion script>",
    "voiceover_speaker": "<e.g. Arjun (Male Lead) or Maya (Female Lead)>",
    "voiceover_sinhala": "<Sinhala + Singlish voiceover>",
    "voiceover_english": "<English voiceover>",
    "voice_tone": "<Voice delivery tone>",
    "sfx_cue": "<Ambient sound effects cue>",
    "wan_model": "Wan2.2-TI2V-5B"
  }}
]
"""
                full_query = f"{system_prompt}\n\nUser Scenarios:\n{json.dumps(scenes, indent=2)}\n\nJSON Output:"
                raw_reply = None
                if self.model is not None:
                    resp = self.model.generate_content(full_query)
                    raw_reply = resp.text
                else:
                    raw_reply = self._call_gemini_rest(full_query)

                if raw_reply:
                    text = raw_reply.strip()
                    if "```json" in text:
                        text = text.split("```json")[1].split("```")[0].strip()
                    elif "```" in text:
                        text = text.split("```")[1].split("```")[0].strip()
                    parsed = json.loads(text)
                    for idx, item in enumerate(parsed):
                        orig_raw = scenes[idx].get("text", "") or scenes[idx].get("raw_prompt", "")
                        item["raw_prompt"] = orig_raw
                        item["flux_model"] = flux_model
                        optimized_list.append(OptimizedSceneItem(**item))
            except Exception as e:
                logger.warning(f"Gemini LLM notice (using Local AI Film Director Synthesizer): {e}")
                optimized_list = []

        # 2. Local AI Film Director Synthesizer (Progressive, Context-Aware & Fresh on Every Call)
        if not optimized_list:
            report_step(2, 4, f"Step 2/4: Synthesizing fresh {flux_model} Cinematic Master Prompts (Card 2)...")
            await asyncio.sleep(0.45)

            for idx, sc in enumerate(scenes):
                raw_input = (sc.get("text") or sc.get("raw_prompt") or "").strip()
                if not raw_input:
                    raw_input = "Two characters meeting at a cozy rain-streaked coffee shop window at night, talking warmly over coffee"

                scene_chars = sc.get("characters") or default_chars
                enhanced_scenario = self._translate_singlish_hints(raw_input)
                char_desc = self._build_character_string(scene_chars, enhanced_scenario)

                synth = self._synthesize_custom_scene_elements(raw_input, enhanced_scenario, idx)

                report_step(3, 4, f"Step 3/4: Choreographing Wan 2.2 24fps Motion & Sinhala/English VO Script (Scene {idx + 1}/{total_scenes})...")
                await asyncio.sleep(0.35)

                flux_prompt = (
                    f"Cinematic 8K short film still, {synth['style']}, {synth['camera']}. "
                    f"Scene scenario & action: {enhanced_scenario}. "
                    f"{char_desc}. "
                    f"Setting: {synth['environment']}. "
                    f"Lighting: {synth['lighting']}. "
                    f"Ultra-realistic human skin micro-texture, natural expressive eyes, authentic fabric folds, "
                    f"anamorphic cinema color grading, crisp focus, masterpiece photorealism."
                )

                wan_video_prompt = (
                    f"[Wan 2.2 I2V Motion Script — Scene {idx + 1}] {synth['camera_motion']}. "
                    f"{synth['motion_detail']} "
                    f"Scenario action: {enhanced_scenario}. "
                    f"Maintain 100% facial identity and outfit consistency for arjun_male and maya_female at 24fps."
                )

                dur = sc.get("duration", "5s")
                dur_int = int(re.sub(r"[^\d]", "", dur) or 5)
                total_seconds += dur_int

                optimized_list.append(OptimizedSceneItem(
                    id=sc.get("id", idx + 1),
                    scene_number=idx + 1,
                    duration=dur,
                    raw_prompt=raw_input,
                    optimized_image_prompt=flux_prompt,
                    character_description=char_desc,
                    action=enhanced_scenario,
                    camera=synth["camera"],
                    camera_motion=synth["camera_motion"],
                    lighting=synth["lighting"],
                    environment=synth["environment"],
                    flux_model=flux_model,
                    video_motion_prompt=wan_video_prompt,
                    voiceover_speaker=synth["vo_speaker"],
                    voiceover_sinhala=synth["vo_sinhala"],
                    voiceover_english=synth["vo_english"],
                    voice_tone=synth["vo_tone"],
                    sfx_cue=synth["sfx_cue"],
                    wan_model="Wan2.2-TI2V-5B"
                ))

        report_step(4, 4, f"Step 4/4: Card 2 Master Prompts & Card 4 Motion/VO Scripts Ready!")
        await asyncio.sleep(0.2)

        try:
            from app.services.flux_service import flux_service
            flux_service.reset_progress()
        except Exception:
            pass

        return MultiScenePlanResponse(
            flux_model=flux_model,
            aspect_ratio=aspect_ratio,
            total_duration_seconds=total_seconds or 5,
            scenes=optimized_list
        )

    def _build_fresh_studio_master_prompt(
        self,
        speaker_gender: str,
        script_text: str,
        aspect_ratio: str,
        flux_model: str
    ) -> Dict[str, str]:
        """
        Generates a brand-new, visually distinct Podcast / Talking-Head Studio Master Frame Prompt
        on every single generation run so no two runs ever share the same static studio setup.
        """
        studio_backdrops = [
            "deep maroon velvet acoustic studio curtains with subtle warm rim light",
            "warm dark walnut wood-slat acoustic studio panels with a soft amber LED accent strip",
            "emerald-green soundproof plush studio panels with moody cinematic warm tungsten key lighting",
            "charcoal-grey modern podcast booth walls with warm golden practical sconces in the background",
            "cozy dark brick and blurred wooden bookshelf podcast studio corner with warm Edison bulb bokeh",
            "midnight-blue broadcast studio acoustic backdrop with soft warm copper key light and subtle rim glow",
            "terracotta and dark espresso acoustic fabric studio backdrop with soft diffused cinema lighting",
        ]
        studio_mics = [
            "a sleek rose-gold and copper broadcast condenser podcast microphone mounted on an articulated studio boom arm in the foreground",
            "a professional matte-black dynamic broadcast podcast microphone on a low-profile studio boom arm",
            "a vintage brushed-silver studio condenser microphone with a metallic shock mount positioned near the speaker",
            "a warm champagne-gold broadcast podcast microphone on a studio desk arm",
        ]
        studio_mugs = [
            "a matte black ceramic coffee mug with clean white motivational lettering resting on the foreground studio table",
            "a dark stoneware artisan coffee mug with gentle wisps of steam on the corner of the wooden podcast desk",
            "a sleek matte-charcoal studio tumbler and a closed leather notebook on the foreground table edge",
            "a deep burgundy ceramic coffee cup sitting in soft foreground bokeh on the studio desk",
        ]
        camera_framings = [
            "Medium close-up portrait shot, 85mm f/1.4 cinema prime lens, shallow depth of field",
            "Intimate medium talking-head studio shot, 50mm Cooke Anamorphic lens at f/1.8, crisp eye focus",
            "Editorial podcast medium close-up, 75mm Leica Summilux cinema lens, creamy background bokeh",
        ]
        lighting_setups = [
            "Warm soft-box studio key light at 45 degrees, subtle golden hair rim light, rich cinematic contrast",
            "Moody broadcast studio lighting with warm amber key light, soft fill, and gentle specular highlights on the microphone",
            "Cinematic three-point podcast studio lighting with warm skin tones and deep velvet background shadows",
        ]

        backdrop = random.choice(studio_backdrops)
        mic = random.choice(studio_mics)
        mug = random.choice(studio_mugs)
        camera = random.choice(camera_framings)
        lighting = random.choice(lighting_setups)

        if speaker_gender == "female":
            wardrobes = [
                "wearing a stylish dark chocolate-brown draped sleeveless studio top and a delicate silver flower pendant necklace",
                "wearing an elegant black fitted crew-neck knit top with a subtle gold pendant necklace",
                "wearing a tailored deep burgundy studio blazer over a dark silk top with minimalist jewelry",
                "wearing a chic charcoal-black off-shoulder studio top and a dainty diamond clover necklace",
            ]
            wardrobe = random.choice(wardrobes)
            subject_desc = (
                f"Maya, a gorgeous young South Asian woman in her early 20s (solo female podcast host, 1girl, no man in frame) "
                f"with long flowing dark brown-black wavy hair parted in the middle, "
                f"expressive dark brown eyes, natural warm olive skin texture, and a confident, warm subtle smile, {wardrobe}"
            )
            char_summary = f"Maya (Female Lead): young South Asian woman, long dark wavy hair, expressive eyes, {wardrobe}."
        else:
            wardrobes = [
                "wearing a fitted matte-black crew-neck t-shirt and a classic black wristwatch",
                "wearing a tailored charcoal-grey henley shirt with rolled sleeves and a dark chronograph watch",
                "wearing a sleek midnight-navy casual studio jacket over a black crew-neck tee",
                "wearing a dark olive-brown knit studio sweater and a minimalist black watch",
            ]
            wardrobe = random.choice(wardrobes)
            subject_desc = (
                f"Arjun, a handsome young South Asian man in his mid-20s (solo male podcast host, 1man, masculine male speaker, no woman in frame) "
                f"with thick wavy short black hair, a neatly trimmed full beard and mustache, strong masculine jawline, "
                f"expressive dark brown eyes, and natural warm brown skin texture, {wardrobe}"
            )
            char_summary = f"Arjun (Male Lead): handsome young South Asian man, thick wavy black hair, neatly trimmed full beard, {wardrobe}."

        ratio_note = "vertical 9:16 social reel framing" if aspect_ratio == "9:16" else f"{aspect_ratio} cinema framing"
        master_prompt = (
            f"Photorealistic podcast studio talking-head master frame of {subject_desc}, "
            f"seated comfortably in a professional podcast studio speaking into {mic}. "
            f"{camera}, {ratio_note}. Behind the speaker is {backdrop}. In the lower foreground rests {mug}. "
            f"{lighting}, ultra-detailed realistic skin pores, natural hair strands, 8k RAW cinema photography."
        )

        return {
            "optimized_image_prompt": master_prompt,
            "character_description": char_summary,
            "camera": camera,
            "lighting": lighting,
            "environment": f"Podcast studio with {backdrop}, {mic}, and {mug}",
        }

    def _enforce_speaker_gender_on_master_prompt(
        self,
        candidate_prompt: str,
        candidate_char_desc: str,
        speaker_gender: str,
        fresh_studio_master: Dict[str, str],
    ) -> Tuple[str, str]:
        """
        Guarantees that the Master Frame Image Prompt and Character Description strictly match
        the user's selected speaker gender ('male' -> Arjun, 'female' -> Maya), even if the
        spoken script is about a mother/woman ('she', 'her') or Gemini hallucinates the wrong gender.
        """
        p_clean = (candidate_prompt or "").strip()
        c_clean = (candidate_char_desc or "").strip()
        p_low = p_clean.lower()
        c_low = c_clean.lower()

        if speaker_gender == "male":
            female_terms = [
                "maya", "maya_female", "woman", "women", "female", "girl", "lady",
                "mother", "top-knot", "half-up", "sleeveless", "off-shoulder", "blouse",
                "dress", "she ", "her ", "hers "
            ]
            has_female_leak = any(ft in p_low for ft in female_terms)
            has_male_anchor = any(mt in p_low for mt in ["man", "male", "arjun", "beard"])
            if not p_clean or has_female_leak or not has_male_anchor:
                p_clean = fresh_studio_master["optimized_image_prompt"]
            else:
                # Replace any raw LoRA token with clear natural language for Qwen/FLUX VLM encoders
                p_clean = re.sub(r"\barjun_male\b", "Arjun, a handsome young South Asian man with a neatly trimmed full beard", p_clean, flags=re.IGNORECASE)
                if "beard" not in p_clean.lower():
                    p_clean = (
                        "Photorealistic medium talking-head studio shot of Arjun, a handsome young South Asian man "
                        "in his mid-20s with thick wavy black hair and a neatly trimmed full beard (solo male podcast host, no woman in frame). "
                        + p_clean
                    )

            if not c_clean or any(ft in c_low for ft in female_terms) or "arjun" not in c_low:
                c_clean = fresh_studio_master["character_description"]
        else:
            male_terms = [
                "arjun", "arjun_male", "beard", "mustache", "moustache", "henley",
                "handsome young man", "south asian man", "he ", "his ", "him "
            ]
            has_male_leak = any(mt in p_low for mt in male_terms) or bool(re.search(r"\b(man|male|boy|guy)\b", p_low))
            has_female_anchor = any(ft in p_low for ft in ["woman", "female", "maya", "girl"])
            if not p_clean or has_male_leak or not has_female_anchor:
                p_clean = fresh_studio_master["optimized_image_prompt"]
            else:
                p_clean = re.sub(r"\bmaya_female\b", "Maya, a beautiful young South Asian woman with long dark wavy hair", p_clean, flags=re.IGNORECASE)

            if not c_clean or any(mt in c_low for mt in male_terms) or "maya" not in c_low:
                c_clean = fresh_studio_master["character_description"]

        return p_clean, c_clean

    def _segment_story_into_beats(self, story_text: str) -> List[Dict[str, Any]]:
        """
        Automatically splits a spoken script (English, Sinhala, or Singlish) into sequential
        5-second spoken dialogue segments based on sentences, newlines, and natural speech pauses.
        """
        cleaned = (story_text or "").strip()
        if not cleaned:
            cleaned = (
                "Success never comes overnight; it is built in the quiet hours when nobody is watching. "
                "Every small step you take today shapes the person you become tomorrow. "
                "Keep your mindset strong and never stop believing in your journey."
            )

        # Split by newlines or sentence-ending punctuation (. ! ? or Sinhala/line breaks)
        raw_chunks = [
            rc.strip()
            for rc in re.split(r"(?:\r?\n)+|(?<=[.!?।])\s+", cleaned, flags=re.IGNORECASE)
            if rc and rc.strip()
        ]
        beats = []
        for chunk in raw_chunks:
            c = re.sub(r"^\s*(?:scene\s*\d+\s*[:.-]?|part\s*\d+\s*[:.-]?|\d+\s*[.):-])\s*", "", chunk.strip(), flags=re.IGNORECASE)
            if not c:
                continue
            words_in_c = c.split()
            # Only sub-split a sentence by clauses if it is very long (> 26 words or > 165 chars) or if the whole script is 1 sentence
            if len(c) > 165 or len(words_in_c) > 26 or (len(raw_chunks) == 1 and (len(c) > 70 or len(words_in_c) > 14)):
                sub_clauses = [
                    sc.strip()
                    for sc in re.split(
                        r"[;]|,\s+|\s+(?=\b(?:and|but|so|because|while|when|then|eth|namuth|habai|mokada|eeta passe|ita passe)\b)",
                        c,
                        flags=re.IGNORECASE,
                    )
                    if len(sc.strip()) >= 8
                ]
                if len(sub_clauses) >= 2:
                    merged_clauses = []
                    for sc in sub_clauses:
                        if merged_clauses and len(merged_clauses[-1].split()) < 6:
                            merged_clauses[-1] = f"{merged_clauses[-1]} {sc}"
                        else:
                            merged_clauses.append(sc)
                    beats.extend(merged_clauses)
                    continue
                if len(words_in_c) > 26:
                    chunk_size = 15
                    for w_i in range(0, len(words_in_c), chunk_size):
                        piece = " ".join(words_in_c[w_i : w_i + chunk_size]).strip()
                        if piece:
                            beats.append(piece)
                    continue
            if len(c) >= 3:
                beats.append(c)

        if not beats:
            beats = [cleaned]

        beats = beats[:30]
        return [{"id": i + 1, "text": b, "duration": "5s"} for i, b in enumerate(beats)]

    async def breakdown_story_to_scenes(
        self,
        story_text: str,
        aspect_ratio: str = "16:9",
        flux_model: str = "Qwen-Image-2.1",
        characters: Optional[List[str]] = None
    ) -> MultiScenePlanResponse:
        """
        Card 1 (Character Selection [Girl or Man] + Spoken Script) -> Card 2:
        1. Generates ONE brand-new Master Frame Image Prompt per generation run (fresh studio backdrop,
           fresh podcast microphone, fresh mug/props, fresh wardrobe & lighting every run).
        2. Automatically breaks the user's spoken script into Scene 1..N sequential 5-second dialogue parts,
           sharing the run's Master Frame Prompt across all scenes while giving each scene its own broken-down
           Voice-Over script + Wan 2.2 talking-head motion prompt.
        """
        default_chars = [str(c).lower().strip() for c in (characters or []) if str(c).strip()] or ["female"]
        speaker_gender = "male" if "male" in default_chars and "female" not in default_chars else (
            "female" if "female" in default_chars and "male" not in default_chars else default_chars[0]
        )
        speaker_label = "Maya (Female Lead)" if speaker_gender == "female" else "Arjun (Male Lead)"

        cleaned_script = (story_text or "").strip()
        if not cleaned_script:
            cleaned_script = (
                "Success never comes overnight; it is built in the quiet hours when nobody is watching. "
                "Every small step you take today shapes the person you become tomorrow. "
                "Keep your mindset strong and never stop believing in your journey."
            )

        def report_step(step_num: int, total_st: int, msg: str):
            try:
                from app.services.flux_service import flux_service
                flux_service.set_progress(
                    node="card2",
                    current_scene=1,
                    total_scenes=1,
                    step=step_num,
                    total_steps=total_st,
                    message=msg
                )
            except Exception:
                pass

        report_step(1, 4, f"Step 1/4: AI Analyzing {speaker_label} Spoken Script & Designing Fresh Master Frame...")
        await asyncio.sleep(0.2)

        variation_nonce = random.randint(10000, 99999)
        extracted_beats = self._segment_story_into_beats(cleaned_script)
        target_scene_count = len(extracted_beats)
        fresh_studio_master = self._build_fresh_studio_master_prompt(
            speaker_gender=speaker_gender,
            script_text=cleaned_script,
            aspect_ratio=aspect_ratio,
            flux_model=flux_model
        )

        optimized_list: List[OptimizedSceneItem] = []
        story_title = f"{'Maya' if speaker_gender == 'female' else 'Arjun'} Studio Session #{variation_nonce % 1000}"
        story_synopsis = cleaned_script[:180]

        if self.active and self.api_key:
            try:
                report_step(2, 4, f"Step 2/4: Gemini 3 Flash Creating Fresh Master Frame & Breaking Script into {target_scene_count} Scene(s)...")
                char_profile_desc = CHARACTER_PROFILES.get(speaker_gender, CHARACTER_PROFILES["female"])
                beats_json_str = json.dumps(extracted_beats, ensure_ascii=False, indent=2)
                system_prompt = f"""
You are an award-winning Podcast Video Director & AI Prompt Engineer specializing in {flux_model} and Wan 2.2.
The user has selected ONE speaker character ({speaker_label}) in Card 1 and entered ONLY the SPOKEN SCRIPT (in English, Sinhala, or Singlish).

CRITICAL RULES:
1. SINGLE NEWLY GENERATED MASTER FRAME PER RUN (Variation Seed #{variation_nonce}):
   - Invent ONE brand-new, photorealistic Podcast / Studio Talking-Head Master Image Prompt ('master_image_prompt') featuring ONLY {speaker_label} ({char_profile_desc}).
   - STRICT GENDER RULE: Even if the spoken script talks about a mother, woman, 'she', or 'her' (or a father/man), the visual speaker in 'master_image_prompt' and 'character_description' MUST BE EXCLUSIVELY {speaker_label} (a {"handsome young South Asian MAN with thick wavy black hair and a neatly trimmed full beard — NEVER depict a woman, girl, or Maya" if speaker_gender == "male" else "beautiful young South Asian WOMAN with long dark wavy hair — NEVER depict a man or Arjun"}).
   - Make this studio setup COMPLETELY FRESH and UNIQUE for this generation run (use this fresh studio design direction as inspiration: {fresh_studio_master['environment']}; {fresh_studio_master['lighting']}; {fresh_studio_master['camera']}).
   - Do NOT repeat the same studio colors, mug text, or outfit from previous runs.
   - ALL scenes in your JSON output MUST use this EXACT SAME 'master_image_prompt' in their 'optimized_image_prompt' field, because the video uses ONE consistent Master Frame Image while the script plays across scenes!
2. EXACT SPOKEN SCRIPT SCENE BREAKDOWN ({target_scene_count} SCENES):
   - The user's spoken script has been segmented into these exact {target_scene_count} sequential spoken segments:
{beats_json_str}
   - Your JSON 'scenes' array MUST contain EXACTLY {target_scene_count} scene objects (Scene 1 .. Scene {target_scene_count}), one for each segment above in the exact same order!
   - For each scene:
     - 'raw_prompt': The exact spoken script segment text from the list above.
     - 'optimized_image_prompt': MUST be identical to the top-level 'master_image_prompt' for all scenes in this run.
     - 'voiceover_speaker': "{speaker_label}"
     - 'voiceover_sinhala': The spoken line for this scene in natural Sinhala with Singlish phonetic script in parentheses.
     - 'voiceover_english': The spoken line for this scene in natural English.
     - 'video_motion_prompt': A 5-second Wan 2.2 talking-head motion prompt describing {speaker_label} speaking this specific line into the studio podcast microphone with natural lip movements, subtle head nods, eye contact, and expressive micro-gestures.

Return ONLY a valid JSON object matching this exact schema:
{{
  "story_title": "<Short title for this spoken script session>",
  "story_synopsis": "<1-sentence summary of the script>",
  "master_image_prompt": "<85-115 word photorealistic {flux_model} Master Frame Prompt for {speaker_label} in this newly generated podcast studio>",
  "master_camera": "<Camera & lens for the master frame>",
  "master_lighting": "<Studio lighting setup>",
  "master_environment": "<Studio backdrop, podcast microphone, and foreground mug/prop details>",
  "scenes": [
    {{
      "id": 1,
      "scene_number": 1,
      "duration": "5s",
      "raw_prompt": "<Spoken script segment for Scene 1>",
      "optimized_image_prompt": "<Identical to master_image_prompt>",
      "character_description": "<{speaker_label} studio appearance>",
      "action": "<Speaking this line into the podcast mic with specific expression/gesture>",
      "camera": "<Identical to master_camera>",
      "camera_motion": "<Subtle talking-head camera motion, e.g. Locked-off studio tripod or ultra-subtle push-in>",
      "lighting": "<Identical to master_lighting>",
      "environment": "<Identical to master_environment>",
      "flux_model": "{flux_model}",
      "video_motion_prompt": "<Wan 2.2 24fps talking-head motion script for Scene 1>",
      "voiceover_speaker": "{speaker_label}",
      "voiceover_sinhala": "<Sinhala + (Singlish) spoken dialogue for Scene 1>",
      "voiceover_english": "<English spoken dialogue for Scene 1>",
      "voice_tone": "<Delivery tone for Scene 1>",
      "sfx_cue": "<Subtle studio room tone / breath cue>",
      "wan_model": "Wan2.2-TI2V-5B"
    }}
  ]
}}
"""
                full_query = f"{system_prompt}\n\nSpeaker: {speaker_label}\nUser Spoken Script:\n\"\"\"\n{cleaned_script}\n\"\"\"\n\nJSON Output:"
                raw_reply = self._call_gemini_rest(full_query)
                if raw_reply:
                    text = raw_reply.strip()
                    if "```json" in text:
                        text = text.split("```json")[1].split("```")[0].strip()
                    elif "```" in text:
                        text = text.split("```")[1].split("```")[0].strip()
                    parsed = json.loads(text)
                    scenes_arr = parsed.get("scenes", parsed) if isinstance(parsed, dict) else parsed
                    raw_master_img_prompt = (
                        (parsed.get("master_image_prompt") if isinstance(parsed, dict) else None)
                        or fresh_studio_master["optimized_image_prompt"]
                    )
                    master_img_prompt, enforced_char_desc = self._enforce_speaker_gender_on_master_prompt(
                        candidate_prompt=raw_master_img_prompt,
                        candidate_char_desc=fresh_studio_master["character_description"],
                        speaker_gender=speaker_gender,
                        fresh_studio_master=fresh_studio_master,
                    )
                    master_cam = (
                        (parsed.get("master_camera") if isinstance(parsed, dict) else None)
                        or fresh_studio_master["camera"]
                    )
                    master_light = (
                        (parsed.get("master_lighting") if isinstance(parsed, dict) else None)
                        or fresh_studio_master["lighting"]
                    )
                    master_env = (
                        (parsed.get("master_environment") if isinstance(parsed, dict) else None)
                        or fresh_studio_master["environment"]
                    )
                    if isinstance(parsed, dict):
                        story_title = parsed.get("story_title") or story_title
                        story_synopsis = parsed.get("story_synopsis") or story_synopsis
                    if isinstance(scenes_arr, list) and len(scenes_arr) > 0:
                        gestures = [
                            "speaking expressively into the podcast microphone with natural lip movements, warm eye contact, and a subtle nod",
                            "delivering the line with calm conviction into the studio microphone, tilting the head slightly with a gentle hand gesture",
                            "speaking warmly into the condenser microphone, eyebrows lifting naturally with a confident subtle smile",
                            "leaning slightly toward the podcast microphone, articulating each word clearly with natural breathing and blinking",
                        ]
                        count_matches = len(scenes_arr) == len(extracted_beats)
                        for idx, beat in enumerate(extracted_beats):
                            line_text = beat["text"]
                            gesture = gestures[idx % len(gestures)]
                            raw_item = scenes_arr[idx] if idx < len(scenes_arr) else scenes_arr[-1]
                            item = dict(raw_item) if isinstance(raw_item, dict) else {}
                            item["id"] = idx + 1
                            item["scene_number"] = idx + 1
                            item["duration"] = beat.get("duration") or item.get("duration") or "5s"
                            item["flux_model"] = flux_model
                            item["raw_prompt"] = line_text
                            item["optimized_image_prompt"] = master_img_prompt
                            _, item_char_desc = self._enforce_speaker_gender_on_master_prompt(
                                candidate_prompt=master_img_prompt,
                                candidate_char_desc=str(item.get("character_description") or enforced_char_desc),
                                speaker_gender=speaker_gender,
                                fresh_studio_master=fresh_studio_master,
                            )
                            item["character_description"] = item_char_desc
                            if not count_matches or not item.get("action"):
                                item["action"] = f"{speaker_label} {gesture}."
                            if not count_matches or not item.get("video_motion_prompt"):
                                item["video_motion_prompt"] = (
                                    f"24fps realistic podcast studio video. {speaker_label} is {gesture} while speaking the line: \"{line_text}\". "
                                    f"Background curtain and foreground studio microphone remain steady and sharp."
                                )
                            if not count_matches or not item.get("voiceover_english"):
                                item["voiceover_english"] = line_text
                            if not count_matches or not item.get("voiceover_sinhala"):
                                item["voiceover_sinhala"] = line_text
                            if speaker_gender == "male":
                                if item.get("action"):
                                    item["action"] = re.sub(r"\bMaya\b", "Arjun", str(item["action"]), flags=re.IGNORECASE)
                                if item.get("video_motion_prompt"):
                                    item["video_motion_prompt"] = re.sub(r"\bMaya\b", "Arjun", str(item["video_motion_prompt"]), flags=re.IGNORECASE)
                            else:
                                if item.get("action"):
                                    item["action"] = re.sub(r"\bArjun\b", "Maya", str(item["action"]), flags=re.IGNORECASE)
                                if item.get("video_motion_prompt"):
                                    item["video_motion_prompt"] = re.sub(r"\bArjun\b", "Maya", str(item["video_motion_prompt"]), flags=re.IGNORECASE)
                            item["camera"] = master_cam
                            item["lighting"] = master_light
                            item["environment"] = master_env
                            item["voiceover_speaker"] = speaker_label
                            optimized_list.append(OptimizedSceneItem(**item))
            except Exception as e:
                logger.warning(f"Gemini script breakdown notice (using local script segmenter): {e}")
                optimized_list = []

        if not optimized_list:
            gestures = [
                "speaking expressively into the podcast microphone with natural lip movements, warm eye contact, and a subtle nod",
                "delivering the line with calm conviction into the studio microphone, tilting the head slightly with a gentle hand gesture",
                "speaking warmly into the condenser microphone, eyebrows lifting naturally with a confident subtle smile",
                "leaning slightly toward the podcast microphone, articulating each word clearly with natural breathing and blinking",
            ]
            for idx, beat in enumerate(extracted_beats):
                line_text = beat["text"]
                gesture = gestures[idx % len(gestures)]
                optimized_list.append(
                    OptimizedSceneItem(
                        id=idx + 1,
                        scene_number=idx + 1,
                        duration="5s",
                        raw_prompt=line_text,
                        optimized_image_prompt=fresh_studio_master["optimized_image_prompt"],
                        character_description=fresh_studio_master["character_description"],
                        action=f"{speaker_label} {gesture}.",
                        camera=fresh_studio_master["camera"],
                        camera_motion="Locked-off studio tripod with ultra-subtle breathing motion",
                        lighting=fresh_studio_master["lighting"],
                        environment=fresh_studio_master["environment"],
                        flux_model=flux_model,
                        video_motion_prompt=(
                            f"24fps realistic podcast studio video. {speaker_label} is {gesture} while speaking the line: \"{line_text}\". "
                            f"Background curtain and foreground studio microphone remain steady and sharp."
                        ),
                        voiceover_speaker=speaker_label,
                        voiceover_sinhala=line_text,
                        voiceover_english=line_text,
                        voice_tone="Confident, warm, and engaging podcast delivery",
                        sfx_cue="Clean studio room tone, warm vocal presence",
                        wan_model="Wan2.2-TI2V-5B"
                    )
                )

        total_seconds = sum(int(re.sub(r"[^\d]", "", s.duration) or 5) for s in optimized_list)
        report_step(4, 4, f"Step 4/4: Generated 1 Fresh Master Frame Prompt + {len(optimized_list)} Script Scenes!")
        await asyncio.sleep(0.15)

        try:
            from app.services.flux_service import flux_service
            flux_service.reset_progress()
        except Exception:
            pass

        return MultiScenePlanResponse(
            flux_model=flux_model,
            aspect_ratio=aspect_ratio,
            total_duration_seconds=total_seconds or 5,
            story_title=story_title,
            story_synopsis=story_synopsis,
            scenes=optimized_list
        )


scene_planner_agent = FilmScenePlannerAgent()
