import os
import sys
import socket
import subprocess
from pathlib import Path

# Ensure we always execute inside the backend's local GPU .venv environment
BASE_DIR = Path(__file__).resolve().parent
VENV_PYTHON = BASE_DIR / ".venv" / "Scripts" / "python.exe"
if VENV_PYTHON.exists() and Path(sys.executable).resolve() != VENV_PYTHON.resolve():
    sys.exit(subprocess.call([str(VENV_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]]))

import uvicorn
from app.core.config import settings


def _free_port_if_in_use(host: str, port: int) -> None:
    """If an older backend process is still holding the port, terminate it cleanly before starting."""
    check_host = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        if sock.connect_ex((check_host, port)) != 0:
            return
    try:
        out = subprocess.check_output(
            ["netstat", "-ano", "-p", "TCP"], text=True, stderr=subprocess.DEVNULL
        )
        current_pid = os.getpid()
        for line in out.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                pid = int(parts[-1])
                if pid != current_pid and pid > 0:
                    print(f"[run.py] Stopping previous backend server on port {port} (PID {pid})...")
                    subprocess.run(
                        ["taskkill", "/F", "/PID", str(pid)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
    except Exception:
        pass


if __name__ == "__main__":
    os.chdir(str(BASE_DIR))
    _free_port_if_in_use(settings.BACKEND_HOST, settings.BACKEND_PORT)
    uvicorn.run(
        "app.main:app",
        host=settings.BACKEND_HOST,
        port=settings.BACKEND_PORT,
        reload=False,
        log_level="info",
    )

