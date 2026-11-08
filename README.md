# 🎬 FILM AGENT — FLUX.1 / FLUX.2 Local GPU Studio

A node-based React Flow + FastAPI AI Studio designed for **Short Film Pre-Visualization & Character-Consistent Keyframe Generation** on your **Local NVIDIA GeForce RTX 4060 GPU (8GB VRAM)**.

---

## 🔗 3-Node Visual Workflow (`@xyflow/react`)

```text
[Card 1: Scenario & Character Input] ──► [Card 2: AI Prompt Optimizer Agent] ──► [Card 3: FLUX.1 / FLUX.2 Local GPU Studio]
```

1. **Card 1 — Scenario & Character Input (`NodePrompt`)**:
   - Type any short scenario in **English, Singlish, or Sinhala** (e.g., *"Two characters meeting at a cozy rain-streaked coffee shop window at night"* or *"kollekui kellekui coffee shop ekaka katha krnwa"*).
   - Select **FLUX Engine** (`FLUX.1-dev`, `FLUX.1-schnell`, `FLUX.2`), **Aspect Ratio** (`16:9`, `9:16`, `1:1`), and **Scene Count** (`1–3 Scenes`).
   - Toggle **Character Turnaround Reference Sheets** (`Male Lead` & `Female Lead` from `d:\film-argent`).
2. **Card 2 — AI Prompt Optimizer Agent (`NodePlanner`)**:
   - Converts your short scenario into a rich, natural-language **FLUX T5-XXL Master Prompt** with 100% character consistency, camera lens (`50mm Anamorphic Prime`, `85mm Portrait`), lighting, and environment breakdown.
   - Supports **Step 1 Only (Optimize Prompt)**, live prompt editing inside the card, and **Variation Prompt** regeneration.
3. **Card 3 — FLUX.1 / FLUX.2 Image Studio (`NodeKeyframeGenerator`)**:
   - Executes local GPU inference on your **NVIDIA GeForce RTX 4060 (8GB VRAM)** with CPU Offloading (`0 OOM errors`).
   - Displays a real-time step-by-step GPU progress bar (`0% ➔ 100%`), high-res image preview, seed tracking, single-frame GPU regeneration, and 1-click download.

---

## 🚀 Quick Start

### 1. Run Backend Server (FastAPI + Local GPU)
```powershell
cd d:\film-argent\backend
D:\video-generation-ai\backend\.venv\Scripts\python.exe run.py
```

### 2. Run Frontend Studio (React + React Flow)
```powershell
cd d:\film-argent\frontend
pnpm install
pnpm run dev
```
Open **[http://localhost:5173/](http://localhost:5173/)** (or **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** when built).

