#!/usr/bin/env python3
import argparse
import time

import pyneat as neat


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9998)
    parser.add_argument("--llm", help="Path to an LLM model directory")
    parser.add_argument("--vlm", help="Path to a VLM model directory")
    parser.add_argument("--asr", help="Path to an ASR model directory")
    args = parser.parse_args()

    if not any([args.llm, args.vlm, args.asr]):
        raise RuntimeError("provide at least one of --llm, --vlm, or --asr")

    # STEP configure-server
    options = neat.genai.GenAIServerOptions()
    options.host = args.host
    options.port = args.port
    server = neat.genai.GenAIServer(options)
    # END STEP

    # STEP register-models
    if args.llm:
        server.add_model(args.llm, "llm")
    if args.vlm:
        server.add_model(args.vlm, "vlm")
    if args.asr:
        server.add_model(args.asr, "asr")

    print("registered models:", ", ".join(server.model_names()))
    # END STEP

    # STEP start-serving
    print(f"serving on http://{options.host}:{options.port}")
    print(f"try: curl http://<modalix-ip>:{options.port}/v1/models")
    server.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
    # END STEP

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
