import os
import sys
import time
import json
import requests
from pathlib import Path

# Fix Windows console UTF-8 encoding
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Paths configuration
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
os.makedirs(MODELS_DIR, exist_ok=True)

MODEL_FILENAME = "flux1-schnell-fp8.safetensors"
DEST_FILE = MODELS_DIR / MODEL_FILENAME
PART_FILE = MODELS_DIR / f"{MODEL_FILENAME}.part"
STATUS_FILE = MODELS_DIR / "download_status.json"

# Public HuggingFace CDN URLs for FLUX.1 (8GB VRAM FP8 Quantized Checkpoint - No Token Required)
MODEL_URLS = [
    "https://huggingface.co/Comfy-Org/flux1-schnell/resolve/main/flux1-schnell-fp8.safetensors",
    "https://huggingface.co/Kijai/flux-fp8/resolve/main/flux1-schnell-fp8-e4m3fn.safetensors",
]
EXPECTED_SIZE = 11900000000  # ~11.9 GB (FP8 Quantized for 8GB RTX 4060 GPU)


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
    print("       FLUX.1 (FP8 8GB VRAM) LOCAL GPU MODEL DOWNLOADER — RTX 4060", flush=True)
    print("=" * 80, flush=True)
    print(f"Target File : {DEST_FILE}")
    print(f"Destination : {MODELS_DIR}")
    print(f"Source CDN  : {MODEL_URLS[0]}")
    print("=" * 80, flush=True)

    # 1. Check if model already exists
    if DEST_FILE.exists():
        existing_size = DEST_FILE.stat().st_size
        if existing_size >= 10 * 1024 * 1024 * 1024:  # >= 10 GB
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

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0 Safari/537.36"
    }

    # Optional HF_TOKEN support if set in environment
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    max_retries = 20
    retry_delay = 5
    url_idx = 0

    for attempt in range(1, max_retries + 1):
        active_url = MODEL_URLS[url_idx % len(MODEL_URLS)]
        if downloaded_bytes > 0:
            headers["Range"] = f"bytes={downloaded_bytes}-"
        else:
            headers.pop("Range", None)

        try:
            print(f"\n[Connecting to Hugging Face CDN (Attempt {attempt}/{max_retries})...]", flush=True)
            with requests.get(active_url, headers=headers, stream=True, timeout=35, allow_redirects=True) as response:
                if response.status_code == 416:
                    print("[!] Range already satisfied or reset needed. Restarting range check...")
                    downloaded_bytes = 0
                    headers.pop("Range", None)
                    continue

                if response.status_code in (401, 403, 404):
                    print(f"[!] Mirror {url_idx + 1} returned HTTP {response.status_code}, switching to backup mirror...")
                    url_idx += 1
                    continue

                if response.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP Server responded with status code: {response.status_code}")

                if response.status_code == 200:
                    mode = "wb"
                    downloaded_bytes = 0
                    total_bytes = int(response.headers.get("content-length", EXPECTED_SIZE))
                else:  # 206 Partial Content
                    mode = "ab"
                    content_len = int(response.headers.get("content-length", EXPECTED_SIZE - downloaded_bytes))
                    total_bytes = downloaded_bytes + content_len

                total_mb = total_bytes / (1024 * 1024)
                total_gb = total_bytes / (1024 * 1024 * 1024)
                print(f"[+] Total Model Size : {total_mb:,.1f} MB ({total_gb:.2f} GB)")
                print(f"[+] Download Mode    : {'Resuming (.part)' if mode == 'ab' else 'Fresh Download'}")
                print("-" * 80, flush=True)

                chunk_size = 2 * 1024 * 1024  # 2MB chunks for smooth MB counter updates
                last_update_time = time.time()
                bytes_window = 0
                speed_mb_s = 0.0

                update_status("downloading", downloaded_bytes, total_bytes, speed_mb_s)

                with open(PART_FILE, mode) as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
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
                    print(f"[SUCCESS] FLUX.1 Model Ready for Local GPU (RTX 4060)!")
                    print(f"Location : {DEST_FILE}")
                    print(f"Size     : {final_mb:,.1f} MB ({final_gb:.2f} GB)")
                    print("=" * 80, flush=True)
                    return
                else:
                    print(f"\n[!] Stream interrupted at {downloaded_bytes / (1024 * 1024):.1f} MB. Reconnecting...")

        except (requests.exceptions.RequestException, Exception) as e:
            print(f"\n[!] Network issue on attempt {attempt}: {e}")
            if attempt < max_retries:
                print(f"[+] Waiting {retry_delay}s before resuming...", flush=True)
                time.sleep(retry_delay)
                if PART_FILE.exists():
                    downloaded_bytes = PART_FILE.stat().st_size
            else:
                update_status(f"error: {str(e)}", downloaded_bytes, EXPECTED_SIZE)
                print("\n[ERROR] Max retries reached. Run the script again anytime to resume from current MB.")
                sys.exit(1)


if __name__ == "__main__":
    try:
        download_flux()
    except KeyboardInterrupt:
        print("\n\n[!] Download paused by user (Ctrl+C). Run the script again anytime to resume from current MB!")
        sys.exit(0)
