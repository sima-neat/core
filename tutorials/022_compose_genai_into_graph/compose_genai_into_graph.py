#!/usr/bin/env python3
import argparse

import cv2
import numpy as np
import pyneat as neat


def load_rgb_image(path: str) -> np.ndarray:
    image_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"failed to read image: {path}")
    return np.asarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def sample_text(sample) -> str:
    if sample.kind == neat.SampleKind.Tensor and sample.tensor is not None:
        return sample.tensor.to_text()
    if sample.kind == neat.SampleKind.TensorSet and len(sample.tensors) == 1:
        return sample.tensors[0].to_text()
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to a deployed VLM model directory")
    parser.add_argument("--image", required=True, help="Path to an image file")
    args = parser.parse_args()

    # STEP create-fragment
    model = neat.genai.VisionLanguageModel(args.model)

    options = neat.genai.VisionLanguageOptions()
    options.system_prompt = "You are concise."
    options.max_new_tokens = 96
    options.streaming = True
    options.encode_images_on_input = False

    genai_fragment = neat.genai.graphs.vision_language(model, options, "genai_stage")
    # END STEP

    # STEP compose-graph
    app = neat.Graph("genai_app")
    app.add(genai_fragment)
    print(app.describe())
    # END STEP

    # STEP push-prompt
    run = app.build()
    image_tensor = neat.Tensor.from_numpy(
        load_rgb_image(args.image), copy=True, image_format=neat.PixelFormat.RGB
    )
    image = neat.make_tensor_sample("image", image_tensor)
    if not run.push("image", [image]):
        raise RuntimeError(f"push(image) failed: {run.last_error()}")

    prompt = neat.make_text_sample("prompt", "Describe this image in one sentence.")
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
