"""
graphrag/utils/ollama_launcher.py

Utility functions for ensuring that a local Ollama server is running.

This module is intended purely as an experimental utility to support local verbalization.
For cloud-based settings, this module can be safely ignored or replaced by external model
orchestration.
"""

import subprocess
import socket
import time

_ollama_initialized = False
_ollama_process = None

def is_ollama_running(host="127.0.0.1", port=11434, timeout=1) -> bool:
    """
    Check if Ollama server is running.
    """
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def ensure_ollama_running():
    """
    Ensure that a local Ollama server is running.
    If no Ollama service is detected, this function will attempt to start one automatically
    by invoking the `ollama serve` command.

    This function is NOT required for the core method and can be
    disabled or replaced.
    """
    global _ollama_initialized, _ollama_process

    if _ollama_initialized:
        return

    if is_ollama_running():
        print("Ollama server is already running.")
    else:
        print("Starting Ollama server...")
        _ollama_process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

        for _ in range(10):
            if is_ollama_running():
                print("Ollama server is now running.")
                break
            time.sleep(1)
        else:
            raise RuntimeError("Failed to start Ollama server.")

    _ollama_initialized = True


if __name__ == "__main__":
    ensure_ollama_running()




