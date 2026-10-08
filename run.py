"""
run.py
------
Single-command launcher for Graph-Based Attack Path Predictor AI.
Serves both the Frontend Dashboard and Backend REST API simultaneously.
Binds to 0.0.0.0 by default to allow network-wide access across your LAN.

Self-healing: Automatically verifies and installs missing dependencies if
started in an environment without pre-installed packages.

Usage:
    python3 run.py
    HOST=0.0.0.0 PORT=8080 python3 run.py
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys

# Ensure project root is in working directory and sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def ensure_dependencies() -> None:
    """Check for required dependencies and auto-install if missing."""
    required = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("sklearn", "scikit-learn"),
        ("networkx", "networkx"),
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("sqlalchemy", "sqlalchemy"),
        ("pydantic", "pydantic"),
        ("torch", "torch"),
        ("torch_geometric", "torch-geometric"),
    ]

    missing = []
    for mod_name, pkg_name in required:
        try:
            __import__(mod_name)
        except ImportError:
            missing.append((mod_name, pkg_name))

    if not missing:
        return

    missing_names = [pkg for _, pkg in missing]
    print("=" * 65)
    print(f"[*] Missing dependencies detected: {', '.join(missing_names)}")
    print(f"[*] Auto-installing into current Python ({sys.executable})...")
    print("=" * 65)

    pip_cmd = [sys.executable, "-m", "pip", "install"]

    # Special handling for torch on CPU to avoid multi-gigabyte CUDA downloads
    missing_mods = {mod for mod, _ in missing}
    if "torch" in missing_mods:
        print("[*] Installing lightweight CPU PyTorch...")
        cmd_torch = pip_cmd + ["torch", "--index-url", "https://download.pytorch.org/whl/cpu"]
        try:
            subprocess.run(cmd_torch + ["--break-system-packages"], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                subprocess.run(cmd_torch, check=True)
            except Exception as e:
                print(f"[!] Warning: Torch install via CPU index failed: {e}")

    # Install remaining requirements from requirements.txt
    req_file = os.path.join(PROJECT_ROOT, "requirements.txt")
    if os.path.exists(req_file):
        print("[*] Installing requirements from requirements.txt...")
        cmd_req = pip_cmd + ["-r", req_file]
        try:
            subprocess.run(cmd_req + ["--break-system-packages"], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            try:
                subprocess.run(cmd_req, check=True)
            except Exception as e:
                print(f"[!] Warning: Requirements install failed: {e}")
                print("[!] If your system restricts pip (PEP 668), run:")
                print("    python3 -m pip install -r requirements.txt --break-system-packages")
                print("    or create a virtual environment: python3 -m venv venv && source venv/bin/activate")

    print("[*] Dependency check complete. Continuing startup...\n")


def get_lan_ip() -> str:
    """Detect local network IP for convenient LAN access."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def find_available_port(host: str, start_port: int, max_tries: int = 50) -> int:
    """Find the first available port starting from start_port."""
    for p in range(start_port, start_port + max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, p))
                return p
            except OSError:
                continue
    return start_port


if __name__ == "__main__":
    ensure_dependencies()

    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    desired_port = int(os.environ.get("PORT", "8000"))
    port = find_available_port(host, desired_port)
    if port != desired_port:
        print(f"[!] Port {desired_port} is already in use by another process.")
        print(f"[*] Automatically falling back to available port {port}.\n")

    lan_ip = get_lan_ip()

    print("=" * 65)
    print("🛡️  Graph-Based Attack Path Predictor AI")
    print("=" * 65)
    print(f"  Local Access:       http://localhost:{port}")
    print(f"  Network / LAN:      http://{lan_ip}:{port}")
    print(f"  Interactive Docs:   http://{lan_ip}:{port}/docs")
    print(f"  Binding Address:    {host}:{port}")
    print("=" * 65)

    uvicorn.run("backend.app.main:app", host=host, port=port, reload=False)
