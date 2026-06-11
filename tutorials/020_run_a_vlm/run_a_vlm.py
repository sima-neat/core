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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to a deployed VLM model directory")
    parser.add_argument("--image", required=True, help="Path to an image file")
    args = parser.parse_args()

    # STEP load-inputs
    model = neat.genai.VisionLanguageModel(args.model)
    image = load_rgb_image(args.image)
    # END STEP

    # STEP direct-image
    direct = neat.genai.GenerationRequest()
    direct.prompt = "Describe this image in one sentence."
    direct.images = [image]
    direct.max_new_tokens = 96

    first = model.run(direct)
    print(f"direct image: {first.text}\n")
    # END STEP

    # STEP cache-image
    if not model.encode([image]):
        raise RuntimeError("VLM did not accept the image for caching")
    print(f"cached_images={model.cached_image_count()}")
    # END STEP

    # STEP follow-up
    cached = neat.genai.GenerationRequest()
    cached.prompt = "What details should I inspect more closely?"
    cached.use_cached_images = True
    cached.max_new_tokens = 96

    follow_up = model.run(cached)
    print(f"cached image: {follow_up.text}\n")

    second_cached = neat.genai.GenerationRequest()
    second_cached.prompt = "Summarize the image in three keywords."
    second_cached.use_cached_images = True
    second_cached.max_new_tokens = 48

    second_follow_up = model.run(second_cached)
    print(f"cached image keywords: {second_follow_up.text}\n")
    # END STEP

    # STEP message-image
    image_message = neat.genai.ChatMessage()
    image_message.role = "user"
    image_message.content = "What is the main subject of this image?"
    image_message.images = [image]

    message_request = neat.genai.GenerationRequest()
    message_request.messages = [image_message]
    message_request.max_new_tokens = 96

    message_result = model.run(message_request)
    print(f"message image: {message_result.text}")
    # END STEP

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
