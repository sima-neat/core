#!/usr/bin/env python3
import argparse

import pyneat as neat


def sample_text(sample) -> str:
    if sample.kind == neat.SampleKind.Tensor and sample.tensor is not None:
        return sample.tensor.to_text()
    if sample.kind == neat.SampleKind.TensorSet and len(sample.tensors) == 1:
        return sample.tensors[0].to_text()
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to a deployed LLiMa model directory")
    args = parser.parse_args()

    # STEP create-fragment
    model = neat.genai.VisionLanguageModel(args.model)

    options = neat.genai.VisionLanguageOptions()
    options.system_prompt = "You are concise."
    options.max_new_tokens = 96
    options.streaming = True

    genai_fragment = neat.genai.graphs.vision_language(model, options, "genai_stage")
    # END STEP

    # STEP compose-graph
    app = neat.Graph("genai_app")
    app.add(genai_fragment)
    print(app.describe())
    # END STEP

    # STEP push-prompt
    run = app.build()
    prompt = neat.make_text_sample("prompt", "Explain what an API gateway does.")
    if not run.push("prompt", [prompt]):
        raise RuntimeError(f"push(prompt) failed: {run.last_error()}")
    # END STEP

    # STEP pull-results
    print("assistant: ", end="", flush=True)
    for _ in range(256):
        token = run.pull("tokens", 250)
        if token is not None:
            print(sample_text(token), end="", flush=True)
            continue
        done = run.pull("done", 10)
        if done is not None:
            break
        error = run.pull("error", 10)
        if error is not None:
            raise RuntimeError(sample_text(error))
    print()
    run.close()
    # END STEP

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
