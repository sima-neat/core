// Transcribe a host audio file on the card, or translate its speech into English.
#include "tutorial_args.h"

#include <filesystem>

namespace genai = simaai::neat::pcie::genai;

int main(int argc, char** argv) {
  try {
    const auto args = tutorial::parse_args(argc, argv, "whisper");
    if (!std::filesystem::is_regular_file(args.media))
      throw std::invalid_argument("audio file does not exist: " + args.media);

    // STEP audio-request
    genai::GenAIModel model(args.model, args.connection);
    if (model.task() != genai::GenAITask::ASR || !model.accepts_audio())
      throw std::invalid_argument("choose a Whisper model directory");
    genai::GenerationRequest request;
    request.audio_file = args.media;
    request.language = args.language;
    request.asr_task = args.translate ? genai::ASRTask::Translate : genai::ASRTask::Transcribe;
    const auto result = model.run(request);
    std::cout << result.text << '\n';
    std::cout << "Language: " << result.language << '\n';
    if (result.no_speech_prob)
      std::cout << "No-speech probability: " << *result.no_speech_prob << '\n';
    model.close();
    // END STEP
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Whisper tutorial: " << error.what() << '\n';
    return 1;
  }
}
