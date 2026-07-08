#!/usr/bin/env python3
import argparse
import json

import requests


def print_stream(response: requests.Response) -> None:
    ttft = None
    tps_samples = []

    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue

        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break

        event = json.loads(payload)
        if "error" in event:
            error = event["error"]
            if isinstance(error, dict):
                error = error.get("message", error)
            raise RuntimeError(error)

        ttft = event.get("ttft", ttft)
        if "tps" in event:
            tps_samples.append(float(event["tps"]))

        choice = event.get("choices", [{}])[0]
        content = choice.get("delta", {}).get("content", "")
        if content:
            print(content, end="", flush=True)

    print()
    if ttft is not None:
        print(f"server ttft: {ttft:.4f}s")
    if tps_samples:
        avg_tps = sum(tps_samples) / len(tps_samples)
        print(
            f"server tps: avg={avg_tps:.2f} min={min(tps_samples):.2f} "
            f"max={max(tps_samples):.2f} tokens/s"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Send a text chat request to GenAIServer.")
    parser.add_argument("prompt", nargs="*", help="Prompt text")
    parser.add_argument("--server-ip", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=9998)
    parser.add_argument("--model", default="llm", help="Served LLM model name")
    parser.add_argument("--max-tokens", type=int, default=None)
    args = parser.parse_args()

    prompt = " ".join(args.prompt) if args.prompt else "Give me three tips for designing a small REST API."
    url = f"http://{args.server_ip}:{args.server_port}/v1/chat/completions"
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
    }
    if args.max_tokens is not None:
        payload["max_tokens"] = args.max_tokens

    print(f"model: {args.model}")
    response = requests.post(url, json=payload, stream=True, timeout=120)
    response.raise_for_status()
    print_stream(response)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
