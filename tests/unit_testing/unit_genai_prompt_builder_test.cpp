#include "genai/GenAIInternal.h"
#include "test_main.h"

#include <functional>
#include <stdexcept>
#include <string>

// Verifies prompt/message/image request validation and conversion without
// loading real LLiMa model artifacts.
namespace {

void require_throws_contains(const std::function<void()>& fn, const std::string& expected) {
  try {
    fn();
  } catch (const std::exception& e) {
    require_contains(e.what(), expected, "unexpected exception text");
    return;
  }
  throw std::runtime_error("expected exception containing: " + expected);
}

} // namespace

RUN_TEST("unit_genai_prompt_builder_test", ([] {
           using simaai::neat::genai::ChatMessage;
           using simaai::neat::genai::GenerationRequest;
           namespace internal = simaai::neat::genai::internal;

           require_throws_contains(
               [] { internal::validate_text_generation_request(GenerationRequest{}); },
               "requires prompt");

           GenerationRequest multiple;
           multiple.prompt = std::string{"hello"};
           multiple.messages.push_back(ChatMessage{.role = "user", .content = "hello"});
           require_throws_contains([&] { internal::validate_text_generation_request(multiple); },
                                   "exactly one");

           GenerationRequest prompt;
           prompt.system_prompt = std::string{"You are concise."};
           prompt.prompt = std::string{"Explain NEAT."};
           const auto prompt_messages = internal::build_text_messages(prompt);
           require(prompt_messages.size() == 2U, "prompt messages size mismatch");
           require(prompt_messages[0].role == "system", "system role mismatch");
           require(prompt_messages[0].content == "You are concise.", "system content mismatch");
           require(prompt_messages[1].role == "user", "user role mismatch");
           require(prompt_messages[1].content == "Explain NEAT.", "user content mismatch");

           GenerationRequest messages;
           messages.messages.push_back(ChatMessage{.role = "system", .content = "Be brief."});
           messages.messages.push_back(ChatMessage{.role = "user", .content = "Hi"});
           messages.messages.push_back(ChatMessage{.role = "assistant", .content = "Hello"});
           const auto converted = internal::build_text_messages(messages);
           require(converted.size() == 3U, "converted messages size mismatch");
           require(converted[0].role == "system", "converted system role mismatch");
           require(converted[1].role == "user", "converted user role mismatch");
           require(converted[2].role == "assistant", "converted assistant role mismatch");
           require(converted[2].content == "Hello", "converted assistant content mismatch");

           GenerationRequest empty_prompt;
           empty_prompt.prompt = std::string{};
           const auto empty_prompt_messages = internal::build_text_messages(empty_prompt);
           require(empty_prompt_messages.size() == 1U, "empty prompt messages size mismatch");
           require(empty_prompt_messages[0].role == "user", "empty prompt role mismatch");
           require(empty_prompt_messages[0].content.empty(),
                   "empty prompt content should be preserved");

           GenerationRequest message_with_top_level_images;
           message_with_top_level_images.messages.push_back(
               ChatMessage{.role = "user", .content = "hello"});
           message_with_top_level_images.images = {simaai::neat::Tensor{}};
           require_throws_contains(
               [&] { internal::validate_text_generation_request(message_with_top_level_images); },
               "top-level images");

           GenerationRequest direct_and_cached;
           direct_and_cached.prompt = std::string{"hello"};
           direct_and_cached.images = {simaai::neat::Tensor{}};
           direct_and_cached.use_cached_images = true;
           require_throws_contains(
               [&] { internal::validate_text_generation_request(direct_and_cached); },
               "direct images");

           GenerationRequest two_cached_insertions;
           two_cached_insertions.messages.push_back(
               ChatMessage{.role = "user", .content = "first", .use_cached_images = true});
           two_cached_insertions.messages.push_back(
               ChatMessage{.role = "user", .content = "second", .use_cached_images = true});
           require_throws_contains(
               [&] { internal::validate_text_generation_request(two_cached_insertions); },
               "one cached image");
         }));
