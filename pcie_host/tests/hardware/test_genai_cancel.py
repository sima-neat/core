"""Run explicitly against a configured PCIe GenAI worker and an LLM model."""
import argparse
from pathlib import Path

from pyneatpcie import genai


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--card-host", default="10.0.0.2")
    parser.add_argument("--user", default="sima")
    parser.add_argument("--receive-root", default="/srv/simaai/incoming")
    args = parser.parse_args()
    connection = genai.ConnectionOptions()
    connection.card_host = args.card_host
    connection.user = args.user
    connection.card_receive_directory = Path(args.receive_root)
    connection.startup_timeout_ms = 180000
    connection.request_timeout_ms = 120000
    with genai.GenAIModel(args.model, connection) as model:
        request = genai.GenerationRequest()
        request.prompt = "Count from one to one hundred."
        request.max_new_tokens = 256
        for iteration in range(3):
            stream = model.stream(request)
            first = stream.next()
            assert first is not None and not first.is_final, "Need an active stream to cancel"
            stream.cancel()
            finals = [sample for sample in stream if sample.is_final]
            assert len(finals) == 1, "Cancellation must publish exactly one terminal sample"
            assert stream.next() is None, "Stream must end after the terminal sample"
            print(f"Cancel {iteration + 1}: terminal reason={finals[0].finish_reason}", flush=True)
        request.prompt = "What is two plus two? Answer briefly."
        request.max_new_tokens = 24
        assert model.run(request).text.strip(), "Model must remain usable after cancellation"
    print("Cancellation, reuse and close passed", flush=True)


if __name__ == "__main__":
    main()
