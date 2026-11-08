import os
import sys
import time
import json
import urllib.request
import urllib.error
from pathlib import Path

# Fix Windows console UTF-8 encoding
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Paths configuration inside d:\film-argent\backend\models
ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "backend" / "models"
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_FILENAME = "flux1-schnell-fp8.safetensors"
DEST_FILE = MODELS_DIR / MODEL_FILENAME
PART_FILE = MODELS_DIR / f"{MODEL_FILENAME}.part"
STATUS_FILE = MODELS_DIR / "download_status.json"

# Public HuggingFace CDN URLs for FLUX.1 (8GB VRAM FP8 Quantized Checkpoint)
MODEL_URLS = [
    "https://huggingface.co/Comfy-Org/flux1-schnell/resolve/main/flux1-schnell-fp8.safetensors",
    "https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-schnell-fp8-e4m3fn.safetensors",
]
EXPECTED_SIZE = 11900000000  # ~11.9 GB


def update_status(status: str, downloaded_bytes: int, total_bytes: int, speed_mb: float = 0.0, eta_sec: float = 0.0):
    pct = round((downloaded_bytes / total_bytes * 100), 1) if total_bytes > 0 else 0
    data = {
        "status": status,
        "downloaded_mb": round(downloaded_bytes / (1024 * 1024), 2),
        "total_mb": round(total_bytes / (1024 * 1024), 2),
        "downloaded_gb": round(downloaded_bytes / (1024 * 1024 * 1024), 2),
        "total_gb": round(total_bytes / (1024 * 1024 * 1024), 2),
        "percent": pct,
        "speed_mb_s": round(speed_mb, 2),
        "eta_seconds": round(eta_sec, 1),
        "dest_file": str(DEST_FILE),
        "model_name": MODEL_FILENAME,
        "timestamp": time.time()
    }
    try:
        with open(STATUS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass


def format_time(seconds: float) -> str:
    if seconds <= 0 or seconds > 86400 * 3:
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def download_flux():
    print("=" * 80, flush=True)
    print("       FLUX.1 (FP8 8GB VRAM) LOCAL GPU MODEL DOWNLOADER", flush=True)
    print("=" * 80, flush=True)
    print(f"Target File : {DEST_FILE}")
    print(f"Destination : {MODELS_DIR}")
    print("=" * 80, flush=True)

    # 1. Check if model already exists
    if DEST_FILE.exists():
        existing_size = DEST_FILE.stat().st_size
        if existing_size >= 10 * 1024 * 1024 * 1024:
            mb_done = existing_size / (1024 * 1024)
            gb_done = existing_size / (1024 * 1024 * 1024)
            print(f"[OK] FLUX.1 model already exists and is fully downloaded!")
            print(f"    Size: {mb_done:,.1f} MB ({gb_done:.2f} GB)")
            print(f"    Path: {DEST_FILE}")
            update_status("completed", existing_size, existing_size)
            return
        else:
            print(f"[!] Found incomplete model file ({round(existing_size / (1024 * 1024), 1)} MB). Moving to .part to resume...")
            if PART_FILE.exists():
                PART_FILE.unlink()
            DEST_FILE.rename(PART_FILE)

    # 2. Check partial download for resuming
    downloaded_bytes = 0
    if PART_FILE.exists():
        downloaded_bytes = PART_FILE.stat().st_size
        print(f"[+] Found partial download: {downloaded_bytes / (1024 * 1024):,.2f} MB. Resuming from where it stopped...")

    max_retries = 20
    retry_delay = 5
    url_idx = 0

    for attempt in range(1, max_retries + 1):
        active_url = MODEL_URLS[url_idx % len(MODEL_URLS)]
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0 Safari/537.36"
        }
        if downloaded_bytes > 0:
            headers["Range"] = f"bytes={downloaded_bytes}-"

        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"

        try:
            print(f"\n[Connecting to Hugging Face CDN (Attempt {attempt}/{max_retries})...]", flush=True)
            req = urllib.request.Request(active_url, headers=headers)
            with urllib.request.urlopen(req, timeout=35) as response:
                status_code = response.getcode()

                if status_code == 200:
                    mode = "wb"
                    downloaded_bytes = 0
                    total_bytes = int(response.headers.get("Content-Length", EXPECTED_SIZE))
                elif status_code == 206:
                    mode = "ab"
                    content_len = int(response.headers.get("Content-Length", EXPECTED_SIZE - downloaded_bytes))
                    total_bytes = downloaded_bytes + content_len
                else:
                    raise RuntimeError(f"Unexpected HTTP status: {status_code}")

                total_mb = total_bytes / (1024 * 1024)
                total_gb = total_bytes / (1024 * 1024 * 1024)
                print(f"[+] Total Model Size : {total_mb:,.1f} MB ({total_gb:.2f} GB)")
                print(f"[+] Download Mode    : {'Resuming (.part)' if mode == 'ab' else 'Fresh Download'}")
                print("-" * 80, flush=True)

                chunk_size = 2 * 1024 * 1024  # 2 MB chunks
                last_update_time = time.time()
                bytes_window = 0
                speed_mb_s = 0.0

                update_status("downloading", downloaded_bytes, total_bytes, speed_mb_s)

                with open(PART_FILE, mode) as f:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded_bytes += len(chunk)
                        bytes_window += len(chunk)

                        current_time = time.time()
                        elapsed = current_time - last_update_time

                        if elapsed >= 0.8:
                            speed_mb_s = (bytes_window / elapsed) / (1024 * 1024)
                            remaining_bytes = max(0, total_bytes - downloaded_bytes)
                            eta_sec = remaining_bytes / (speed_mb_s * 1024 * 1024) if speed_mb_s > 0 else 0
                            pct = (downloaded_bytes / total_bytes * 100) if total_bytes > 0 else 0

                            bar_len = 24
                            filled = int(bar_len * (pct / 100))
                            bar = "=" * filled + ">" if filled < bar_len else "=" * bar_len
                            bar = bar.ljust(bar_len, " ")

                            mb_done = downloaded_bytes / (1024 * 1024)
                            mb_total = total_bytes / (1024 * 1024)
                            sys.stdout.write(
                                f"\r[{bar}] {pct:5.1f}% | {mb_done:8.1f} / {mb_total:8.1f} MB | {speed_mb_s:5.2f} MB/s | ETA: {format_time(eta_sec)}  "
                            )
                            sys.stdout.flush()

                            update_status("downloading", downloaded_bytes, total_bytes, speed_mb_s, eta_sec)
                            last_update_time = current_time
                            bytes_window = 0

                if downloaded_bytes >= total_bytes:
                    print("\n" + "=" * 80)
                    print("[OK] FLUX.1 model download complete! Finalizing file...")
                    if PART_FILE.exists():
                        if DEST_FILE.exists():
                            DEST_FILE.unlink()
                        PART_FILE.rename(DEST_FILE)

                    update_status("completed", downloaded_bytes, total_bytes, 0.0, 0.0)
                    final_mb = DEST_FILE.stat().st_size / (1024 * 1024)
                    final_gb = DEST_FILE.stat().st_size / (1024 * 1024 * 1024)
                    print(f"[SUCCESS] FLUX.1 Model Ready for Local GPU!")
                    print(f"Location : {DEST_FILE}")
                    print(f"Size     : {final_mb:,.1f} MB ({final_gb:.2f} GB)")
                    print("=" * 80, flush=True)
                    return
                else:
                    print(f"\n[!] Stream interrupted at {downloaded_bytes / (1024 * 1024):.1f} MB. Reconnecting...")

        except urllib.error.HTTPError as he:
            if he.code in (401, 403, 404):
                print(f"\n[!] Mirror {url_idx + 1} returned HTTP {he.code}, switching to backup mirror...")
                url_idx += 1
                continue
            elif he.code == 416:
                downloaded_bytes = 0
                continue
            print(f"\n[!] HTTP Error on attempt {attempt}: {he}")
            time.sleep(retry_delay)
        except Exception as e:
            print(f"\n[!] Network issue on attempt {attempt}: {e}")
            if attempt < max_retries:
                print(f"[+] Waiting {retry_delay}s before resuming...", flush=True)
                time.sleep(retry_delay)
                if PART_FILE.exists():
                    downloaded_bytes = PART_FILE.stat().st_size
            else:
                update_status(f"error: {str(e)}", downloaded_bytes, EXPECTED_SIZE)
                print("\n[ERROR] Max retries reached. Run `python download_flux.py` again anytime to resume.")
                sys.exit(1)


if __name__ == "__main__":
    if "--help" in sys.argv:
        print("Usage: python download_flux.py")
        sys.exit(0)
    try:
        download_flux()
    except KeyboardInterrupt:
        print("\n\n[!] Download paused by user (Ctrl+C). Run `python download_flux.py` again anytime to resume!")
        sys.exit(0)

