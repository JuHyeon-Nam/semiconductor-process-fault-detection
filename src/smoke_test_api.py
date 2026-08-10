from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib import error, request


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_INPUT = ROOT / "artifacts" / "sample_input.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the FastAPI app locally and smoke-test core endpoints.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    return parser.parse_args()


def get_json(url: str) -> dict:
    with request.urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(http_request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_until_ready(base_url: str, server: subprocess.Popen, timeout_seconds: int = 15) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if server.poll() is not None:
            stderr = server.stderr.read().decode("utf-8") if server.stderr else ""
            raise RuntimeError(f"API server exited early.\n{stderr}")
        try:
            get_json(f"{base_url}/health")
            return
        except (error.URLError, TimeoutError):
            time.sleep(0.5)
    raise TimeoutError(f"API server did not become ready within {timeout_seconds} seconds.")


def main() -> None:
    args = parse_args()
    if not SAMPLE_INPUT.exists():
        raise FileNotFoundError("Run `python src/make_sample_input.py` first.")

    base_url = f"http://{args.host}:{args.port}"
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "src.api:app",
            "--host",
            args.host,
            "--port",
            str(args.port),
            "--log-level",
            "warning",
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        wait_until_ready(base_url, server)
        health = get_json(f"{base_url}/health")
        print("health:", health)

        info = get_json(f"{base_url}/model-info")
        print(
            "model-info:",
            {
                "model_name": info["model_name"],
                "threshold": info["threshold"],
                "feature_count": info["feature_count"],
                "recall_fail": info["metrics"]["recall_fail"],
                "f2_fail": info["metrics"]["f2_fail"],
            },
        )

        sample = json.loads(SAMPLE_INPUT.read_text(encoding="utf-8"))
        prediction = post_json(f"{base_url}/predict", {"sensors": sample["sensors"]})
        print("predict:", prediction)
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)


if __name__ == "__main__":
    main()
