import json
import re
import random
import logging
from typing import Dict, Any, Optional, List
from app.core.config import settings
from app.schemas.schemas import OptimizedSceneItem, MultiScenePlanResponse

logger = logging.getLogger("FilmAgent_Planner")

# Exact Character Consistency Profiles extracted from d:\film-argent Turnaround Sheets (Trigger words: arjun_male, maya_female)
CHARACTER_PROFILES = {
    "male": (
        "arjun_male, a handsome young South Asian man in his mid-20s with thick wavy black hair, "
        "a neatly trimmed full beard and mustache, warm brown skin, athletic build, "
        "wearing a fitted plain black crew-neck t-shirt, beige chino trousers, a black wristwatch, and clean white sneakers"
    ),
    "female": (
        "maya_female, a beautiful young South Asian woman in her early 20s with long dark wavy hair styled in a half-up top-knot bun, "
        "expressive dark eyes, warm natural smile, wearing a fitted black short-sleeve t-shirt, "
        "high-waisted wide-leg beige tailored trousers, silver wristwatch, and clean white sneakers"
    )
}

SINGLISH_DICTIONARY = {
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
    "coffee shop": "inside a cozy warm-lit modern artisan coffee shop with glass windows",
    "cafe": "inside a stylish warm-lit cafe with wooden tables and bokeh lights",
    "para": "along a scenic urban street road",
    "pare": "walking along a cinematic street",
    "beach": "on a scenic tropical beach shoreline at sunset",
    "muhuda": "by the ocean shoreline with gentle waves",
    "kanda": "on a misty scenic mountain viewpoint overlooking lush hills",
    "gedara": "inside a modern sunlit living room interior",
    "office": "inside a sleek modern glass-walled office workspace",
    "duwanawa": "running dynamically with cinematic motion",
    "yanawa": "walking side by side naturally",
    "balan innawa": "looking thoughtfully into the distance",
}

class FilmScenePlannerAgent:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.active = False
        self._init_gemini()

    def _init_gemini(self):
        if self.api_key and self.api_key.startswith("AIzaSy"):
            try:
                import google.generativeai as genai  # type: ignore
                genai.configure(api_key=self.api_key)
                for candidate in [settings.GEMINI_MODEL, "gemini-2.0-flash", "gemini-1.5-flash"]:
                    if not candidate:
                        continue
                    try:
                        self.model = genai.GenerativeModel(candidate)
                        self.model_name = candidate
                        self.active = True
                        logger.info(f"Gemini Film Director Agent active with model: {candidate}")
                        break
                    except Exception:
                        continue
            except Exception as e:
                logger.warning(f"Gemini initialization skipped: {e}")
                self.active = False

    def _translate_singlish_hints(self, raw_text: str) -> str:
        """
        Enhances Singlish / Sinhala phrases into clear English cinematic descriptions
        while 100% preserving all English words and scenario details typed by the user.
        """
        if not raw_text or not raw_text.strip():
            return raw_text

        text_lower = raw_text.lower()
        # Only translate pure Singlish words (not English words like 'coffee shop', 'beach', 'office')
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

        # Auto-detect from text if needed
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

    async def plan_multi_scenes(
        self,
        scenes: List[Dict[str, Any]],
        aspect_ratio: str = "16:9",
        flux_model: str = "FLUX.1-dev",
        characters: Optional[List[str]] = None
    ) -> MultiScenePlanResponse:
        """
        Converts user scenario(s) from Card 1 into FLUX.1 / FLUX.2 T5-XXL optimized prompts in Card 2.
        """
        try:
            from app.services.flux_service import flux_service
            flux_service.set_progress(
                node="card2",
                current_scene=1,
                total_scenes=len(scenes),
                step=1,
                total_steps=max(len(scenes), 2),
                message=f"Card 2: AI Director Optimizing {len(scenes)} Scenario(s) for {flux_model}... [█████░░░░░] 50%"
            )
        except Exception:
            pass

        optimized_list: List[OptimizedSceneItem] = []
        total_seconds = 0
        default_chars = characters if characters is not None else ["male", "female"]

        # 1. Try Gemini LLM if active
        if self.active:
            try:
                char_context = self._build_character_string(default_chars, " ".join([s.get("text", "") for s in scenes]))
                system_prompt = f"""
You are an award-winning Short Film Director and AI Prompt Engineer specializing in {flux_model} (Black Forest Labs FLUX.1 / FLUX.2 architecture).
Unlike older models that use comma-separated tags, {flux_model} uses a T5-XXL text encoder that excels at rich, natural-language, descriptive paragraphs specifying subject appearance, exact pose/action, environment, lighting, lens focal length, and film stock.

Character Turnaround Sheet Reference (Maintain 100% Character Consistency):
{char_context}

The user has provided {len(scenes)} scenario(s) (may be in English, Sinhala, or Singlish).
For each scene, translate/expand the user's scenario into a vivid, photorealistic cinematic {flux_model} prompt ('optimized_image_prompt') of 65-110 words.

Return ONLY a valid JSON array with {len(scenes)} objects matching this exact schema:
[
  {{
    "id": <integer>,
    "scene_number": <integer 1..N>,
    "duration": "<e.g. 5s>",
    "raw_prompt": "<original user scenario>",
    "optimized_image_prompt": "<Rich natural-language {flux_model} master prompt describing the exact scene, character appearance from turnaround sheet, action, environment, camera angle, lens, and cinematic lighting>",
    "character_description": "<summary of the character(s) in this shot>",
    "action": "<specific character action and emotional expression>",
    "camera": "<camera shot type & lens, e.g. Medium Close-Up, 50mm Anamorphic Prime>",
    "camera_motion": "<camera movement for future video step, e.g. Slow dolly push-in>",
    "lighting": "<lighting setup, e.g. Warm tungsten cafe glow with rainy window rim-light>",
    "environment": "<detailed setting & background atmosphere>",
    "flux_model": "{flux_model}",
    "video_motion_prompt": "<Wan 2.2 Image-to-Video (I2V) motion prompt describing exact 5-second visual movement starting from the generated frame: character facial expressions, subtle eye/head movement, environmental motion, and camera motion at 24fps>",
    "voiceover_speaker": "<e.g. Arjun (Male Lead) or Maya (Female Lead)>",
    "voiceover_sinhala": "<Emotional Sinhala/Singlish short film voice-over or dialogue line for this scene>",
    "voiceover_english": "<Emotional English voice-over or dialogue translation for this scene>",
    "voice_tone": "<Voice delivery style, e.g. Warm, soft romantic cinema tone>",
    "sfx_cue": "<Ambient sound effects and background music cue for this scene>",
    "wan_model": "Wan2.2-TI2V-5B"
  }}
]
"""
                resp = self.model.generate_content(
                    f"{system_prompt}\n\nUser Scenarios:\n{json.dumps(scenes, indent=2)}\n\nJSON Output:"
                )
                text = resp.text.strip()
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
                logger.warning(f"Gemini optimization fallback triggered: {e}")
                optimized_list = []

        # 2. Intelligent Film Director Prompt Optimizer Engine (Generates a FRESH variation on every single call!)
        if not optimized_list:
            camera_setups = [
                ("Medium two-shot at eye level, 50mm anamorphic prime lens, shallow f/1.8 depth of field", "Slow cinematic dolly push-in"),
                ("Cinematic medium close-up, 85mm f/1.4 portrait prime lens, creamy bokeh background", "Subtle handheld breathing motion with slow focus pull"),
                ("Wide cinematic establishing frame, 35mm ARRI Master Anamorphic lens, rich depth", "Smooth lateral Steadicam tracking pan"),
                ("Over-the-shoulder dramatic cinema angle, 50mm Cooke Anamorphic lens, sharp eye focus", "Slow orbital arc camera movement"),
                ("Low-angle expressive medium shot, 40mm cinema prime lens, golden lens flare", "Steady gimbal forward tracking movement"),
                ("Intimate eye-level cinema frame, 65mm prime lens, soft background separation", "Gentle crane push-in with subtle parallax")
            ]

            cinema_styles = [
                "shot on ARRI Alexa 65, Kodak Vision3 500T film stock aesthetic, natural skin micro-texture",
                "shot on RED V-Raptor 8K VV, Cooke Anamorphic/i Full Frame Plus lens, rich theatrical color grade",
                "shot on Sony VENICE 2 8K, Panavision Primo 70 lens, organic film grain and lifelike eye catchlights",
                "shot on ARRI Alexa Mini LF, Master Anamorphic optics, high-dynamic-range cinema lighting"
            ]

            for idx, sc in enumerate(scenes):
                raw_input = (sc.get("text") or sc.get("raw_prompt") or "").strip()
                if not raw_input:
                    raw_input = "Two characters meeting at a cozy rain-streaked coffee shop window at night, talking warmly over coffee"

                scene_chars = sc.get("characters") or default_chars
                enhanced_scenario = self._translate_singlish_hints(raw_input)
                raw_lower = enhanced_scenario.lower()

                char_desc = self._build_character_string(scene_chars, enhanced_scenario)
                chosen_style = random.choice(cinema_styles)

                # Dynamic pools of Environment, Lighting, Motion & Voice-Over variations per theme
                if any(w in raw_lower for w in ["coffee", "cafe", "restaurant", "tea", "shop"]):
                    env = random.choice([
                        f"inside a cozy, warm-lit artisan coffee shop ({enhanced_scenario}) with rain-streaked glass windows, wooden tables, steaming ceramic cups, and glowing city bokeh outside",
                        f"at a corner window booth inside an intimate vintage cafe ({enhanced_scenario}) with warm Edison bulb reflections on wet window glass and freshly brewed coffee steam",
                        f"inside a stylish modern bistro cafe at night ({enhanced_scenario}) with soft rain droplets on the glass pane, warm timber interior, and cinematic depth"
                    ])
                    light = random.choice([
                        "warm tungsten pendant lamp glow balanced with cool blue evening rain ambience through the window, soft rim light on faces",
                        "golden amber cafe table lamp illumination contrasting with moody teal city bokeh reflections on the glass",
                        "soft diffused warm interior key light with natural catchlights in the eyes and subtle cinematic shadows"
                    ])
                    vo_options = [
                        (
                            "Arjun (Male Lead)",
                            "සමහර හමුවීම් තියෙනවා... වචන ගොඩක් නැතුවත් හිතටම දැනෙන. (Samahara hamuweem thiyenawa, wachana godak nathuwath hithatama danena.)",
                            "Some meetings don't need many words... they quietly speak straight to the heart.",
                            "Warm wisps of steam curl upward from the coffee cups as raindrops slide down the cafe glass; Arjun leans forward slightly, smiling warmly at Maya as they speak softly."
                        ),
                        (
                            "Maya (Female Lead)",
                            "වැස්ස වැටෙන මේ නිහඬ හවසේ, ඔයාගේ කතාව අහන් ඉන්න එකම ලොකු සැනසීමක්. (Wassa watena me nihada hawase, oyage kathawa ahan inna ekama loku sanaseemak.)",
                            "On this quiet rainy evening, just listening to your voice feels like home.",
                            "Maya gently wraps her fingers around the warm ceramic cup, looking up at Arjun with a soft, genuine smile while city lights shimmer through the rainy window."
                        ),
                        (
                            "Arjun & Maya (Dialogue)",
                            "කෝපි කෝප්පයක උණුසුමත් එක්ක, අපේ අලුත් කතාව මෙතනින් පටන් ගත්තා. (Kopi koppayaka unusumath ekka, ape aluth kathawa methanin patan gaththa.)",
                            "Over the warmth of a simple cup of coffee, a brand new chapter of our story began.",
                            "Both Arjun and Maya share a candid laugh across the wooden cafe table as soft rain patters against the window glass and warm pendant lights glow above them."
                        )
                    ]
                    vo_speaker, vo_si, vo_en, motion_detail = random.choice(vo_options)
                    vo_tone = "Warm, intimate & emotional cinema tone"
                    sfx = "Soft rain tapping on cafe window glass, gentle acoustic guitar notes, subtle coffee cup clink"
                elif any(w in raw_lower for w in ["rain", "wassa", "wehi", "umbrella", "storm", "wet"]):
                    env = random.choice([
                        f"on a rain-slicked urban city street ({enhanced_scenario}) with glistening wet asphalt reflections, glowing shopfronts, and raindrops in mid-air",
                        f"along a moody rain-soaked boulevard at night ({enhanced_scenario}) with shimmering neon puddles and atmospheric mist",
                        f"under the warm awning of a city street during a gentle downpour ({enhanced_scenario}) with wet pavement reflections"
                    ])
                    light = random.choice([
                        "dramatic neon and warm streetlight reflections bouncing off wet pavement, moody cinematic rim lighting",
                        "cool cyan rainlight mixed with warm amber streetlamps illuminating falling raindrops",
                        "high-contrast nocturnal cinema lighting with glistening highlights on wet hair and fabric"
                    ])
                    vo_options = [
                        (
                            "Maya (Female Lead)",
                            "වැස්සත් එක්ක ආපු ඒ රාත්‍රිය, අපේ කතාවේ ලස්සනම පිටුවක් වුණා. (Wassath ekka aapu e rathriya, ape kathawe lassanama pituwak wuna.)",
                            "That rainy night in the city became the most unforgettable page of our story.",
                            "Glistening raindrops fall in slow motion as ripples spread across the neon-lit puddles; Arjun and Maya walk side by side, turning toward each other with a warm smile."
                        ),
                        (
                            "Arjun (Male Lead)",
                            "හැම වැහි බිඳුවක්ම කියන්නේ අපි දෙන්නාගේ හිතේ තිබුණු නිහඬ හැඟීමක්. (Hama wahi binduwakma kiyanne api dennage hithe thibunu nihada hageemak.)",
                            "Every falling raindrop echoed the quiet feelings we never said out loud.",
                            "Arjun glances at Maya through the softly falling rain as warm streetlight reflections dance across the wet pavement around them."
                        )
                    ]
                    vo_speaker, vo_si, vo_en, motion_detail = random.choice(vo_options)
                    vo_tone = "Soft, reflective & romantic voice-over"
                    sfx = "Steady city rainfall on wet pavement, distant soft traffic hum, gentle piano melody"
                elif any(w in raw_lower for w in ["beach", "ocean", "sea", "muhuda", "sunset", "coast"]):
                    env = random.choice([
                        f"along a scenic tropical coastline beach ({enhanced_scenario}) with gentle turquoise ocean waves, wet golden sand, and a glowing sunset horizon",
                        f"by the shimmering ocean shoreline at golden hour ({enhanced_scenario}) with foamy tide waves and warm coastal breeze"
                    ])
                    light = random.choice([
                        "warm golden hour sunset backlighting with soft anamorphic sun flare and natural fill light on skin tones",
                        "amber-violet twilight sky reflections over the water with soft golden rim light"
                    ])
                    vo_options = [
                        (
                            "Arjun & Maya (Narrator)",
                            "ඉර බැහැගෙන යන හැම හවසකම, අලුත් බලාපොරොත්තුවක් හිතේ ඉතුරු වෙනවා. (Ira bahagena yana hama hawasakama, aluth balaporothththuwak hithe ithuru wenawa.)",
                            "With every golden sunset by the ocean, a quiet new hope stays behind.",
                            "Gentle ocean waves wash onto the wet golden sand while a coastal breeze stirs their hair as Arjun and Maya walk slowly along the shoreline."
                        ),
                        (
                            "Maya (Female Lead)",
                            "මුහුදු රළ වගේම සමහර මතකයන් කවදාවත් හිතෙන් මැකෙන්නේ නෑ. (Muhudu rala wagema samahara mathakayan kawadawath hithen makenne naa.)",
                            "Like the endless ocean tide, some memories never fade from the heart.",
                            "Maya looks out toward the glowing sunset horizon before turning to Arjun with a peaceful, emotional smile as the breeze moves her hair."
                        )
                    ]
                    vo_speaker, vo_si, vo_en, motion_detail = random.choice(vo_options)
                    vo_tone = "Dreamy, uplifting golden-hour narration"
                    sfx = "Gentle ocean waves washing ashore, soft coastal wind breeze, warm ambient strings"
                elif any(w in raw_lower for w in ["mountain", "hill", "kanda", "forest", "nature", "jungle", "mist"]):
                    env = random.choice([
                        f"at a breathtaking misty mountain viewpoint ({enhanced_scenario}) surrounded by lush emerald green hills and drifting morning fog",
                        f"along a scenic highland trail overlooking misty valleys ({enhanced_scenario}) with golden sunlight breaking through the clouds"
                    ])
                    light = random.choice([
                        "soft diffused morning golden hour sunlight piercing through mountain mist, natural cinematic contrast",
                        "ethereal highland daylight with warm rim light and atmospheric depth"
                    ])
                    vo_options = [
                        (
                            "Maya (Female Lead)",
                            "මීදුම අතරින් පේන මේ කඳු යාය වගේම, ජීවිතයත් හරිම නිස්කලංකයි. (Meeduma atharin pena me kandu yaya wagema, jeewithayath harima niskalankai.)",
                            "Up here above the drifting mountain mist, time finally slows down for us.",
                            "Soft morning mist drifts across the emerald hills in the background as a gentle mountain breeze moves their hair and they share a quiet smile."
                        ),
                        (
                            "Arjun (Male Lead)",
                            "මේ නිහඬ කඳු මුදුනේදී, හිතට දැනෙන්නේ පුදුමාකාර නිදහසක්. (Me nihada kandu mudunedi, hithata danenne pudumakara nidahasak.)",
                            "Standing above the quiet clouds, everything finally feels clear and free.",
                            "Arjun and Maya stand side by side overlooking the misty valley, turning to exchange an expressive, heartfelt glance at 24fps."
                        )
                    ]
                    vo_speaker, vo_si, vo_en, motion_detail = random.choice(vo_options)
                    vo_tone = "Peaceful, airy & heartfelt emotional tone"
                    sfx = "Soft mountain wind breeze, distant morning birdsong, swelling cinematic cello & piano"
                else:
                    env = random.choice([
                        f"in a richly detailed, photorealistic cinematic setting matching '{enhanced_scenario}' with authentic props and atmospheric depth",
                        f"inside a realistic, cinematically staged environment for '{enhanced_scenario}' with natural depth-of-field separation",
                        f"in an atmospheric wide-dynamic-range film location depicting '{enhanced_scenario}' with lifelike textures"
                    ])
                    light = random.choice([
                        "naturalistic Hollywood cinema lighting with soft key light, subtle rim light, and rich anamorphic color grading",
                        "dramatic golden-amber key lighting balanced with soft ambient fill and natural eye catchlights",
                        "chiaroscuro-inspired soft cinema illumination with volumetric atmospheric depth"
                    ])
                    vo_options = [
                        (
                            "Arjun (Male Lead)",
                            f"මේ මොහොත අපේ කතාවේ කවදාවත් අමතක නොවන හැරවුම් ලක්ෂ්‍යයක් වුණා. ({raw_input})",
                            f"This very moment became a turning point we would remember forever — {enhanced_scenario}.",
                            f"Starting from the master frame, natural 24fps human motion and expressive facial micro-movements bring '{enhanced_scenario}' to life with realistic eye contact and fabric movement."
                        ),
                        (
                            "Maya (Female Lead)",
                            f"හිතේ තිබුණු හැම බලාපොරොත්තුවක්ම සැබෑ වුණු ඒ සුවිශේෂී මොහොත... ({raw_input})",
                            f"Every quiet hope we carried finally came alive in this moment — {enhanced_scenario}.",
                            f"Smooth 24fps cinematic motion as the characters interact naturally in '{enhanced_scenario}', with subtle head turns, authentic emotion, and shallow depth-of-field parallax."
                        ),
                        (
                            "Arjun & Maya (Narrator)",
                            f"ජීවිතයේ සමහර තත්පර කිහිපයක් මුළු කතාවම වෙනස් කරනවා. ({raw_input})",
                            f"Sometimes just a few seconds can change the entire story — {enhanced_scenario}.",
                            f"Dynamic 24fps short-film movement capturing '{enhanced_scenario}' with lifelike breathing, natural posture shifts, and atmospheric background motion."
                        )
                    ]
                    vo_speaker, vo_si, vo_en, motion_detail = random.choice(vo_options)
                    vo_tone = "Expressive cinematic storyteller tone"
                    sfx = "Rich cinematic ambient atmosphere with subtle emotional score"

                # Randomly pick a fresh camera angle & movement on EVERY optimization call
                user_raw_lower = raw_input.lower()
                if any(w in user_raw_lower for w in ["close-up", "close up", "portrait", "face shot"]):
                    cam, motion = random.choice([
                        ("Intimate close-up shot, 85mm f/1.4 prime lens, ultra-shallow depth of field", "Slow subtle push-in on facial expression"),
                        ("Emotional medium close-up, 75mm anamorphic prime lens, soft bokeh", "Gentle focus pull with subtle breathing motion")
                    ])
                elif any(w in user_raw_lower for w in ["wide", "full body", "walking", "running", "landscape"]):
                    cam, motion = random.choice([
                        ("Wide cinematic full-frame shot, 35mm anamorphic lens, deep cinema composition", "Smooth gimbal tracking shot"),
                        (" sweeping wide establishing shot, 28mm cinema prime lens", "Slow lateral dolly tracking movement")
                    ])
                else:
                    cam, motion = random.choice(camera_setups)

                action_desc = enhanced_scenario

                # Construct a fresh natural-language Master Prompt for Card 2
                flux_prompt = (
                    f"Cinematic 8K short film still, {chosen_style}, {cam}. "
                    f"Scene scenario & action: {enhanced_scenario}. "
                    f"{char_desc}. "
                    f"Setting: {env}. "
                    f"Lighting: {light}. "
                    f"Ultra-realistic human skin micro-texture, natural expressive eyes, authentic fabric folds, "
                    f"anamorphic cinema color grading, crisp focus, masterpiece photorealism."
                )

                # Construct a fresh Wan 2.2 Image-to-Video (I2V) Motion Prompt for Card 4
                wan_video_prompt = (
                    f"[Wan 2.2 I2V Motion Script — Scene {idx + 1}] {motion}. "
                    f"{motion_detail} "
                    f"Action: {enhanced_scenario}. "
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
                    action=action_desc,
                    camera=cam,
                    camera_motion=motion,
                    lighting=light,
                    environment=env,
                    flux_model=flux_model,
                    video_motion_prompt=wan_video_prompt,
                    voiceover_speaker=vo_speaker,
                    voiceover_sinhala=vo_si,
                    voiceover_english=vo_en,
                    voice_tone=vo_tone,
                    sfx_cue=sfx,
                    wan_model="Wan2.2-TI2V-5B"
                ))

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

scene_planner_agent = FilmScenePlannerAgent()
