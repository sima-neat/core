#include "genai/ASRModel.h"
#include "genai/GenAIModel.h"
#include "genai/GenAITypes.h"
#include "genai/GraphFragments.h"
#include "genai/GenAIServer.h"
#include "genai/VisionLanguageModel.h"
#include "asset_utils.h"
#include "test_main.h"

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <type_traits>

// Verifies that the public GenAI headers are self-contained and expose the
// lightweight value/handle types expected by external C++ users.
RUN_TEST("unit_genai_header_compile_surface_test", ([] {
           using namespace simaai::neat::genai;

           const std::filesystem::path root = sima_test::test_source_root();
           require(!std::filesystem::exists(root / "include/neat/graph.h"),
                   "neat/graph.h must not be part of the source public header surface");
           require(!std::filesystem::exists(root / "include/genai/nodes/VisionLanguage.h"),
                   "genai/nodes/VisionLanguage.h must not be part of the source public GenAI "
                   "header surface");
           require(!std::filesystem::exists(root / "include/genai/nodes/SpeechTranscriber.h"),
                   "genai/nodes/SpeechTranscriber.h must not be part of the source public GenAI "
                   "header surface");

           GenAITask task = GenAITask::VisionLanguage;
           GenerationRequest request;
           request.prompt = std::string{"hello"};
           GenerationResult result;
           result.text = "world";
           result.no_speech_prob = 0.1F;
           result.avg_logprob = -0.2F;
           GenerationMetrics metrics;
           TokenSample token;
           token.text = "tok";
           token.no_speech_prob = 0.3F;
           token.avg_logprob = -0.4F;
           ChatMessage message{"user", "hello"};
           ImageList images;
           VisionLanguageOptions vision_language_options;
           SpeechTranscriberOptions speech_transcriber_options;
           GenAIServerOptions genai_server_options;
           genai_server_options.port = 9998;
           request.audio_file = std::filesystem::path{"audio.wav"};
           request.language = "en";
           request.tools = Json::array(
               {{{"type", "function"},
                 {"function", {{"name", "lookup"}, {"parameters", {{"type", "object"}}}}}}});
           request.tool_choice = "auto";
           message.tool_calls =
               Json::array({{{"id", "call_0"},
                             {"type", "function"},
                             {"function", {{"name", "lookup"}, {"arguments", "{}"}}}}});
           message.tool_call_id = "call_0";
           message.name = "lookup";
           result.tool_calls = message.tool_calls;
           token.tool_calls = message.tool_calls;

           static_assert(std::is_default_constructible_v<ImageList>);
           static_assert(std::is_move_constructible_v<GenerationStream>);
           static_assert(!std::is_copy_constructible_v<GenerationStream>);
           static_assert(std::is_copy_constructible_v<TokenSample>);
           static_assert(std::is_default_constructible_v<VisionLanguageOptions>);
           static_assert(std::is_default_constructible_v<SpeechTranscriberOptions>);
           static_assert(std::is_move_constructible_v<VisionLanguageModel>);
           static_assert(!std::is_copy_constructible_v<VisionLanguageModel>);
           static_assert(std::is_move_constructible_v<ASRModel>);
           static_assert(!std::is_copy_constructible_v<ASRModel>);
           static_assert(std::is_move_constructible_v<GenAIModel>);
           static_assert(!std::is_copy_constructible_v<GenAIModel>);
           static_assert(std::is_move_constructible_v<GenAIServer>);
           static_assert(!std::is_copy_constructible_v<GenAIServer>);

           auto vlm = static_cast<VisionLanguageModel*>(nullptr);
           auto asr = static_cast<ASRModel*>(nullptr);
           auto genai = static_cast<GenAIModel*>(nullptr);
           auto genai_server = static_cast<GenAIServer*>(nullptr);
           auto genai_run = &GenAIModel::run;
           auto genai_stream = &GenAIModel::stream;
           bool vision_language_rejected_null = false;
           try {
             (void)graphs::VisionLanguage(nullptr);
           } catch (const std::invalid_argument&) {
             vision_language_rejected_null = true;
           }
           if (!vision_language_rejected_null) {
             throw std::runtime_error("genai::graphs::VisionLanguage should reject nullptr model");
           }
           bool speech_rejected_null = false;
           try {
             (void)graphs::SpeechTranscriber(nullptr);
           } catch (const std::invalid_argument&) {
             speech_rejected_null = true;
           }
           if (!speech_rejected_null) {
             throw std::runtime_error(
                 "genai::graphs::SpeechTranscriber should reject nullptr model");
           }

           (void)task;
           (void)request;
           (void)result;
           (void)metrics;
           (void)token;
           (void)message;
           (void)images;
           (void)vision_language_options;
           (void)speech_transcriber_options;
           (void)genai_server_options;
           (void)vlm;
           (void)asr;
           (void)genai;
           (void)genai_server;
           (void)genai_run;
           (void)genai_stream;
         }));
