#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import requests


def print_stream(response: requests.Response) -> None:
    ttft = None
    tps = None

    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue

        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break

        event = json.loads(payload)
        if event.get("object") == "audio.transcription.error":
            raise RuntimeError(event.get("error", "audio transcription failed"))

        ttft = event.get("ttft", ttft)
        tps = event.get("tps", tps)

        text = event.get("text", "")
        if text:
            print(text, end="", flush=True)

    print()
    if ttft is not None:
        print(f"server ttft: {ttft:.4f}s")
    if tps is not None:
        print(f"server tps: {tps:.2f} tokens/s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Send an audio transcription request to GenAIServer.")
    parser.add_argument("audio_file", type=Path)
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=9998)
    parser.add_argument("--model", default="asr", help="Served ASR model name")
    parser.add_argument("--language", default="en")
    args = parser.parse_args()

    url = f"http://{args.server_ip}:{args.server_port}/v1/audio/transcriptions"
    print(f"model: {args.model}")
    with args.audio_file.open("rb") as audio:
        response = requests.post(
            url,
            data={"model": args.model, "language": args.language, "stream": "true"},
            files={"file": (args.audio_file.name, audio, "audio/wav")},
            timeout=120,
            stream=True,
        )
    response.raise_for_status()
    print_stream(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
