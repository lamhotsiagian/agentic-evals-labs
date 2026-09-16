"""Testing helpers and Streamlit app server fixtures for Playwright E2E tests."""

from __future__ import annotations
import os
import socket
import subprocess
import sys
import time
from typing import Generator
import requests


def find_free_port() -> int:
    """Finds an available TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        return s.getsockname()[1]


class StreamlitServerRunner:
    """Spins up a Streamlit app in headless mode on an ephemeral port for Playwright tests."""

    def __init__(self, script_path: str, port: int = None):
        self.script_path = os.path.abspath(script_path)
        self.port = port or find_free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.process: subprocess.Popen = None

    def start(self, timeout: int = 60):
        labs_root = os.path.dirname(os.path.dirname(self.script_path))
        env = os.environ.copy()
        # Direct local model calling - no mock
        env.pop("MOCK_LLM", None)
        env["MOCK_LLM"] = "0"
        env["HOME"] = labs_root
        env["STREAMLIT_SERVER_PORT"] = str(self.port)
        env["STREAMLIT_SERVER_HEADLESS"] = "true"
        env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
        env["PYTHONPATH"] = labs_root

        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            self.script_path,
            "--server.port",
            str(self.port),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ]

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
        )

        # Wait for server health endpoint
        start_time = time.time()
        health_url = f"{self.base_url}/_stcore/health"
        while time.time() - start_time < timeout:
            if self.process.poll() is not None:
                out, err = self.process.communicate()
                raise RuntimeError(f"Streamlit server exited prematurely with code {self.process.returncode}:\n{err}\n{out}")
            try:
                resp = requests.get(health_url, timeout=1)
                if resp.status_code == 200:
                    return self.base_url
            except Exception:
                time.sleep(0.3)

        self.stop()
        raise TimeoutError(f"Streamlit server at {self.script_path} failed to start within {timeout}s.")

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)


import contextlib


@contextlib.contextmanager
def run_streamlit_app(script_path: str) -> Generator[str, None, None]:
    """Context manager yielding the base URL of a running Streamlit app."""
    runner = StreamlitServerRunner(script_path)
    try:
        url = runner.start()
        yield url
    finally:
        runner.stop()
