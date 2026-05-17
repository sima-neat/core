#include "genai/ASRModel.h"
#include "genai/GenAIModel.h"
#include "genai/GenAITypes.h"
#include "genai/VisionLanguageModel.h"
#include "genai/nodes/VisionLanguage.h"
#include "test_main.h"

#include <memory>
#include <stdexcept>
#include <type_traits>

// Verifies that the public GenAI headers are self-contained and expose the
// lightweight value/handle types expected by external C++ users.
RUN_TEST("unit_genai_header_compile_surface_test", ([] {
           using namespace simaai::neat::genai;

           GenAITask task = GenAITask::VisionLanguage;
           GenerationRequest request;
           request.prompt = std::string{"hello"};
           GenerationResult result;
           result.text = "world";
           GenerationMetrics metrics;
           TokenSample token;
           token.text = "tok";
           ChatMessage message{"user", "hello"};
           ImageList images;
           nodes::VisionLanguageOptions vision_language_options;

           static_assert(std::is_default_constructible_v<ImageList>);
           static_assert(std::is_move_constructible_v<GenerationStream>);
           static_assert(!std::is_copy_constructible_v<GenerationStream>);
           static_assert(std::is_copy_constructible_v<TokenSample>);
           static_assert(std::is_default_constructible_v<nodes::VisionLanguageOptions>);
           static_assert(std::is_move_constructible_v<VisionLanguageModel>);
           static_assert(!std::is_copy_constructible_v<VisionLanguageModel>);
           static_assert(std::is_move_constructible_v<ASRModel>);
           static_assert(!std::is_copy_constructible_v<ASRModel>);
           static_assert(std::is_move_constructible_v<GenAIModel>);
           static_assert(!std::is_copy_constructible_v<GenAIModel>);

           auto vlm = static_cast<VisionLanguageModel*>(nullptr);
           auto asr = static_cast<ASRModel*>(nullptr);
           auto genai = static_cast<GenAIModel*>(nullptr);
           bool vision_language_rejected_null = false;
           try {
             (void)nodes::VisionLanguage(nullptr);
           } catch (const std::invalid_argument&) {
             vision_language_rejected_null = true;
           }
           if (!vision_language_rejected_null) {
             throw std::runtime_error("genai::nodes::VisionLanguage should reject nullptr model");
           }

           (void)task;
           (void)request;
           (void)result;
           (void)metrics;
           (void)token;
           (void)message;
           (void)images;
           (void)vision_language_options;
           (void)vlm;
           (void)asr;
           (void)genai;
         }));
