"""Run one host-side text request over PCIe, optionally streaming the answer."""

import argparse

from pyneatpcie import genai


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="prepared model directory on the host")
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--card-host", default="")
    parser.add_argument("--user", default="sima")
    parser.add_argument("--ssh-key", default="")
    parser.add_argument("--prompt", default="Explain PCIe in one sentence.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--stream", action="store_true")
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

    # STEP load-llm
    with genai.GenAIModel(args.model, connection) as model:
        if not model.accepts_text() or model.task() == genai.GenAITask.ASR:
            raise ValueError("choose a text-generation model directory")
        # END STEP

        # STEP text-request
        request = genai.GenerationRequest()
        request.prompt = args.prompt
        request.max_new_tokens = args.max_tokens
        if not args.stream:
            result = model.run(request)
            print(result.text)
            print(f"Generated tokens: {result.metrics.generated_tokens}")
        # END STEP
        else:
            # STEP stream-answer
            for sample in model.stream(request):
                print(sample.text, end="", flush=True)
                if sample.is_final:
                    print(f"\nGenerated tokens: {sample.metrics.generated_tokens}")
            # END STEP
        # STEP close-model
        # The context manager closes the model, including on exceptions.
        # END STEP


if __name__ == "__main__":
    main()
