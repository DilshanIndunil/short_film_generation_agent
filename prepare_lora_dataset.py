r"""
Auto-Slicer & LoRA Dataset Builder for Film Agent Character Turnaround Sheets.
Extracts 13 distinct multi-angle views (Full-Body, Medium Waist-Up, and Close-Up Face Angles)
per character from the two 360 Turnaround Sheets (26 total training images + .txt captions),
strips out bottom text labels so the model never learns text artifacts, pads/resizes to 1024x1024,
and packages everything into `d:\film-argent\character_lora_dataset.zip` for LoRA Fine-Tuning.
"""
import os
import zipfile
from pathlib import Path
from PIL import Image, ImageOps

ROOT_DIR = Path(__file__).resolve().parent
DATASET_DIR = ROOT_DIR / "data" / "lora_dataset"
ZIP_PATH = ROOT_DIR / "character_lora_dataset.zip"

# Neutral studio background color matching the turnaround sheet backdrop (#bcbec4)
BG_COLOR = (188, 190, 196)

# Normalized crop boxes (x1, y1, x2, y2) for 1774 x 887 Turnaround Sheets
# Carefully excludes the dark text badges beneath each pose/face box!
TURNAROUND_CROPS = [
    # 5 Full-Body Poses
    ("01_full_front",         (25,   15, 335,  832), "full body front view standing pose"),
    ("02_full_right_side",    (345,  15, 600,  832), "full body right side profile standing view"),
    ("03_full_back",          (605,  15, 895,  832), "full body back view from behind"),
    ("04_full_left_side",     (900,  15, 1150, 832), "full body left side profile standing view"),
    ("05_full_three_quarter", (1155, 15, 1435, 832), "full body three-quarter angle standing pose"),

    # 4 Medium / Upper-Body Cinema Shots (extracted from the upper half of full-body poses)
    ("06_medium_front",         (35,   15, 325,  465), "medium waist-up front shot"),
    ("07_medium_right_profile", (350,  15, 595,  465), "medium waist-up right side profile shot"),
    ("08_medium_left_profile",  (905,  15, 1145, 465), "medium waist-up left side profile shot"),
    ("09_medium_three_quarter", (1165, 15, 1425, 465), "medium waist-up three-quarter angle shot"),

    # 4 Close-Up Face & Head Angles (from the right-hand inset boxes, excluding border & label)
    ("10_face_closeup_front", (1474, 24,  1740, 219), "close-up front face portrait, detailed eyes and facial features"),
    ("11_face_closeup_right", (1474, 266, 1740, 424), "close-up right side profile face portrait, jawline and hair detail"),
    ("12_face_closeup_left",  (1474, 470, 1740, 628), "close-up left side profile face portrait, jawline and hair detail"),
    ("13_head_closeup_back",  (1474, 676, 1740, 840), "close-up back of head hairstyle detail"),
]

CHARACTERS_CONFIG = [
    {
        "folder_name": "10_arjun_male",
        "trigger_word": "arjun_male",
        "source_image": ROOT_DIR / "data" / "characters" / "male_turnaround.png",
        "base_description": (
            "photo of arjun_male, a handsome young South Asian man in his mid-20s with thick wavy black hair, "
            "neatly trimmed full beard and mustache, warm brown skin, wearing a plain black crew-neck t-shirt, "
            "beige chino trousers, and white sneakers"
        )
    },
    {
        "folder_name": "10_maya_female",
        "trigger_word": "maya_female",
        "source_image": ROOT_DIR / "data" / "characters" / "female_turnaround.png",
        "base_description": (
            "photo of maya_female, a beautiful young South Asian woman in her early 20s with long dark wavy hair "
            "styled in a half-up top-knot bun, warm smile, expressive dark eyes, wearing a fitted black t-shirt, "
            "high-waisted wide-leg beige trousers, and white sneakers"
        )
    }
]


def pad_to_square_1024(crop_img: Image.Image) -> Image.Image:
    """Pads any aspect-ratio crop onto a clean studio square canvas and resizes to 1024x1024 with LANCZOS."""
    w, h = crop_img.size
    max_side = max(w, h)
    # Add 6% breathing room around the subject
    canvas_size = int(max_side * 1.06)
    square = Image.new("RGB", (canvas_size, canvas_size), BG_COLOR)
    offset_x = (canvas_size - w) // 2
    offset_y = (canvas_size - h) // 2
    square.paste(crop_img, (offset_x, offset_y))
    return square.resize((1024, 1024), Image.Resampling.LANCZOS)


def build_dataset():
    print("=" * 78)
    print("   FILM AGENT - CHARACTER TURNAROUND SHEET AUTO-SLICER & LORA DATASET BUILDER")
    print("=" * 78)
    os.makedirs(DATASET_DIR, exist_ok=True)
    total_saved = 0

    for char_cfg in CHARACTERS_CONFIG:
        src_path = char_cfg["source_image"]
        if not src_path.exists():
            print(f"[!] Source image not found: {src_path}")
            continue

        out_dir = DATASET_DIR / char_cfg["folder_name"]
        os.makedirs(out_dir, exist_ok=True)

        with Image.open(src_path) as sheet_img:
            sheet_rgb = sheet_img.convert("RGB")
            # Normalize to 1774x887 if different
            if sheet_rgb.size != (1774, 887):
                sheet_rgb = sheet_rgb.resize((1774, 887), Image.Resampling.LANCZOS)

            print(f"\n[+] Slicing Character: {char_cfg['trigger_word']} ({src_path.name})")
            for crop_name, box, angle_desc in TURNAROUND_CROPS:
                cropped = sheet_rgb.crop(box)
                final_img = pad_to_square_1024(cropped)

                img_file = out_dir / f"{char_cfg['trigger_word']}_{crop_name}.png"
                txt_file = out_dir / f"{char_cfg['trigger_word']}_{crop_name}.txt"

                final_img.save(img_file, "PNG", quality=98)

                caption = (
                    f"{char_cfg['base_description']}, {angle_desc}, "
                    f"clean studio lighting, sharp focus, photorealistic skin texture, 8k"
                )
                with open(txt_file, "w", encoding="utf-8") as tf:
                    tf.write(caption)

                total_saved += 1
                print(f"    [OK] Saved {img_file.name} (1024x1024) + .txt caption")

    # Package into character_lora_dataset.zip for 1-click upload to Google Colab / Trainer
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(DATASET_DIR):
            for file in files:
                full_path = Path(root) / file
                rel_path = full_path.relative_to(DATASET_DIR)
                zf.write(full_path, rel_path)

    print("\n" + "=" * 78)
    print(f"[SUCCESS] Prepared {total_saved} High-Res Character Training Images + Captions!")
    print(f"Dataset Folder : {DATASET_DIR}")
    print(f"Colab ZIP File : {ZIP_PATH} ({round(ZIP_PATH.stat().st_size / (1024*1024), 2)} MB)")
    print("=" * 78)


if __name__ == "__main__":
    build_dataset()
