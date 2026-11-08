@echo off
title FILM AGENT - FLUX.1 / FLUX.2 Local GPU Studio (RTX 4060)
echo ========================================================================
echo      FILM AGENT - FLUX.1 / FLUX.2 LOCAL GPU STUDIO (RTX 4060)
echo ========================================================================
echo Starting FastAPI Backend on http://127.0.0.1:8000 ...
start "Film Agent Backend (GPU)" cmd /k "cd /d d:\film-argent\backend && d:\film-argent\backend\.venv\Scripts\python.exe run.py"
echo Starting React Flow Studio UI on http://localhost:5173 ...
start "Film Agent Frontend" cmd /k "cd /d d:\film-argent\frontend && npm run dev"
echo Both servers launched! Open http://localhost:5173 in your browser.
