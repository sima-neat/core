#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import requests


def print_stream(response: requests.Response) -> None:
    ttft = None
    tps_samples = []
    language = None
    no_speech_prob = None
    avg_logprob = None

    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue

        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break

        event = json.loads(payload)
        if event.get("object") in {"audio.transcription.error", "audio.translation.error"}:
            raise RuntimeError(event.get("error", "audio request failed"))

        ttft = event.get("ttft", ttft)
        language = event.get("language", language)
        no_speech_prob = event.get("no_speech_prob", no_speech_prob)
        avg_logprob = event.get("avg_logprob", avg_logprob)
        if "tps" in event:
            tps_samples.append(float(event["tps"]))

        text = event.get("text", "")
        if text:
            print(text, end="", flush=True)

    print()
    if ttft is not None:
        print(f"server ttft: {ttft:.4f}s")
    if tps_samples:
        avg_tps = sum(tps_samples) / len(tps_samples)
        print(
            f"server tps: avg={avg_tps:.2f} min={min(tps_samples):.2f} "
            f"max={max(tps_samples):.2f} tokens/s"
        )
    if language is not None:
        print(f"detected source language: {language}")
    if no_speech_prob is not None:
        print(f"no_speech_prob: {float(no_speech_prob):.6f}")
    if avg_logprob is not None:
        print(f"avg_logprob: {float(avg_logprob):.6f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Send an audio transcription request to GenAIServer.")
    parser.add_argument("audio_file", type=Path)
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=9998)
    parser.add_argument("--model", default="asr", help="Served ASR model name")
    parser.add_argument("--language", default="auto")
    parser.add_argument(
        "--translate", action="store_true", help="Translate speech into English"
    )
    args = parser.parse_args()

    operation = "translations" if args.translate else "transcriptions"
    url = f"http://{args.server_ip}:{args.server_port}/v1/audio/{operation}"
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
