"""Read an image on the host and ask a VLM about its RGB pixels over PCIe."""

import argparse

import cv2
import numpy as np
from pyneatpcie import genai


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="prepared model directory on the host")
    parser.add_argument("--image", required=True, help="image file on the host")
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--card-host", default="")
    parser.add_argument("--user", default="sima")
    parser.add_argument("--ssh-key", default="")
    parser.add_argument("--prompt", default="Describe this image briefly.")
    parser.add_argument("--max-tokens", type=int, default=128)
    args = parser.parse_args()
    if args.card < 0 or args.max_tokens <= 0:
        parser.error("--card must be nonnegative and --max-tokens must be positive")
    connection = genai.ConnectionOptions()
    connection.card_id = args.card
    connection.card_host = args.card_host
    connection.user = args.user
    connection.ssh_key = args.ssh_key
    connection.startup_timeout_ms = 900000
    connection.request_timeout_ms = 300000

    # STEP prepare-image
    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        parser.error(f"cannot read image: {args.image}")
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    # END STEP

    # STEP image-request
    with genai.GenAIModel(args.model, connection) as model:
        if not model.accepts_image():
            raise ValueError("choose a vision-language model directory")
        request = genai.GenerationRequest()
        request.prompt = args.prompt
        request.images = [rgb]
        request.max_new_tokens = args.max_tokens
        print(model.run(request).text)
    # END STEP


if __name__ == "__main__":
    main()
