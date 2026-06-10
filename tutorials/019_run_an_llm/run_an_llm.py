#!/usr/bin/env python3
import argparse

import pyneat as neat


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to a deployed LLiMa model directory")
    args = parser.parse_args()

    # STEP load-model
    model = neat.genai.GenAIModel(args.model)
    # END STEP

    # STEP send-prompt
    request = neat.genai.GenerationRequest()
    request.prompt = "Give me three practical tips for designing a small REST API."
    request.max_new_tokens = 96

    first = model.run(request)
    print(f"assistant: {first.text}\n")
    # END STEP

    # STEP system-prompt
    system_prompt = "You are concise and practical."

    concise_request = neat.genai.GenerationRequest()
    concise_request.system_prompt = system_prompt
    concise_request.prompt = "Give me one rule of thumb for designing a small REST API."
    concise_request.max_new_tokens = 64

    concise = model.run(concise_request)
    print(f"assistant: {concise.text}\n")
    # END STEP

    # STEP store-history
    messages = []

    system_message = neat.genai.ChatMessage()
    system_message.role = "system"
    system_message.content = system_prompt
    messages.append(system_message)

    first_user = neat.genai.ChatMessage()
    first_user.role = "user"
    first_user.content = "Give me three practical tips for writing API documentation."
    messages.append(first_user)

    chat_request = neat.genai.GenerationRequest()
    chat_request.messages = messages
    chat_request.max_new_tokens = 96

    chat_result = model.run(chat_request)
    print(f"assistant: {chat_result.text}\n")

    chat_answer = neat.genai.ChatMessage()
    chat_answer.role = "assistant"
    chat_answer.content = chat_result.text
    messages.append(chat_answer)
    # END STEP

    # STEP follow-up
    follow_up_user = neat.genai.ChatMessage()
    follow_up_user.role = "user"
    follow_up_user.content = "Which tip should I apply first for a prototype?"
    messages.append(follow_up_user)

    follow_up = neat.genai.GenerationRequest()
    follow_up.messages = messages
    follow_up.max_new_tokens = 96

    second = model.run(follow_up)
    print(f"assistant: {second.text}\n")

    second_answer = neat.genai.ChatMessage()
    second_answer.role = "assistant"
    second_answer.content = second.text
    messages.append(second_answer)
    # END STEP

    # STEP stream-answer
    final_user = neat.genai.ChatMessage()
    final_user.role = "user"
    final_user.content = "Turn that advice into a short checklist."
    messages.append(final_user)

    streaming_request = neat.genai.GenerationRequest()
    streaming_request.messages = messages
    streaming_request.max_new_tokens = 96

    stream_handle = model.stream(streaming_request)
    print("assistant: ", end="", flush=True)
    for token in stream_handle:
        print(token.text, end="", flush=True)
    print()
    # END STEP

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
