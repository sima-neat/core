"""Transcribe a host audio file over PCIe, or translate its speech into English."""

import argparse
from pathlib import Path

from pyneatpcie import genai


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="prepared Whisper directory on the host")
    parser.add_argument("--audio", required=True, help="audio file on the host")
    parser.add_argument("--card", type=int, default=0)
    parser.add_argument("--card-host", default="")
    parser.add_argument("--user", default="root")
    parser.add_argument("--ssh-key", default="")
    parser.add_argument("--language", default="auto")
    parser.add_argument("--translate", action="store_true", help="translate speech into English")
    args = parser.parse_args()
    if args.card < 0:
        parser.error("--card must be nonnegative")
    if not Path(args.audio).is_file():
        parser.error(f"audio file does not exist: {args.audio}")
    connection = genai.ConnectionOptions()
    connection.card_id = args.card
    connection.card_host = args.card_host
    connection.user = args.user
    connection.ssh_key = args.ssh_key
    connection.startup_timeout_ms = 900000
    connection.request_timeout_ms = 300000

    # STEP audio-request
    with genai.GenAIModel(args.model, connection) as model:
        if model.task() != genai.GenAITask.ASR or not model.accepts_audio():
            raise ValueError("choose a Whisper model directory")
        request = genai.GenerationRequest()
        request.audio_file = args.audio
        request.language = args.language
        request.asr_task = genai.ASRTask.Translate if args.translate else genai.ASRTask.Transcribe
        result = model.run(request)
        print(result.text)
        print(f"Language: {result.language}")
        if result.no_speech_prob is not None:
            print(f"No-speech probability: {result.no_speech_prob}")
    # END STEP


if __name__ == "__main__":
    main()
