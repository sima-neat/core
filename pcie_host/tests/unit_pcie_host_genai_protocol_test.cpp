#include "genai/GenAIProtocol.h"

#include <nlohmann/json.hpp>

#include <iostream>
#include <stdexcept>
#include <string>

namespace pgi = simaai::neat::pcie::genai::internal;
using simaai::neat::Tensor;
using simaai::neat::genai::GenerationRequest;
using simaai::neat::genai::ImageList;

namespace {

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Exception, typename Fn> bool throws(Fn&& fn) {
  try {
    fn();
  } catch (const Exception&) {
    return true;
  }
  return false;
}

} // namespace

int main() {
  try {
    // The prompt payload is the wire contract with the card. The card-side
    // test (LLiMa tests/runtime/pcie_genai_protocol_test.cpp) parses this exact
    // string, so the two sides cannot drift apart unnoticed.
    {
      GenerationRequest request;
      request.prompt = "Hi";
      request.system_prompt = "Be brief.";
      request.max_new_tokens = 64;
      require(
          pgi::encode_prompt("h1-1", request) ==
              R"({"id":"h1-1","prompt":"Hi","system_prompt":"Be brief.","max_new_tokens":64,"enable_thinking":false})",
          "encode_prompt must produce the golden payload");
    }

    // Optional fields are left out when unset.
    {
      GenerationRequest request;
      request.prompt = "Hi";
      require(pgi::encode_prompt("h1-2", request) ==
                  R"({"id":"h1-2","prompt":"Hi","enable_thinking":false})",
              "encode_prompt must omit unset optional fields");
    }

    // Quotes, newlines and non-ASCII text survive the trip unchanged.
    {
      GenerationRequest request;
      request.prompt = "He said \"hi\"\nnaïve 🙂";
      const auto j = nlohmann::json::parse(pgi::encode_prompt("h1-3", request));
      require(j["prompt"] == "He said \"hi\"\nnaïve 🙂", "special characters must round-trip");
    }

    // A VLM prompt names its images (relative to the data serve root), in order.
    {
      GenerationRequest request;
      request.prompt = "what is on the image";
      const auto with =
          pgi::encode_prompt("h1-4", request, {"pcie-genai/h1-4-0.jpg", "pcie-genai/h1-4-1.png"});
      const auto jw = nlohmann::json::parse(with);
      require(jw.contains("images") && jw["images"].size() == 2 &&
                  jw["images"][0] == "pcie-genai/h1-4-0.jpg" &&
                  jw["images"][1] == "pcie-genai/h1-4-1.png",
              "the image names are written in order");
      require(!jw.contains("image"), "the old single \"image\" key is gone");
      const auto without = pgi::encode_prompt("h1-5", request);
      require(!nlohmann::json::parse(without).contains("images"), "no images key when none given");
    }

    // Pixel-image tensors are not sent over PCIe: point the caller at image_files.
    {
      GenerationRequest request;
      request.prompt = "Hi";
      request.images = ImageList(std::vector<Tensor>(1));
      bool named_image_file = false;
      try {
        (void)pgi::encode_prompt("x", request);
      } catch (const std::invalid_argument& e) {
        named_image_file = std::string(e.what()).find("image_files") != std::string::npos;
      }
      require(named_image_file, "pixel images are refused with a message naming image_files");
    }

    // PCIe GenAI takes one prompt, text and image files only: refuse what the card cannot do,
    // before anything is sent.
    {
      GenerationRequest none;
      require(throws<std::invalid_argument>([&] { pgi::encode_prompt("x", none); }),
              "a request without a prompt must be refused");
      GenerationRequest empty;
      empty.prompt = "";
      require(throws<std::invalid_argument>([&] { pgi::encode_prompt("x", empty); }),
              "an empty prompt must be refused");
      GenerationRequest audio;
      audio.prompt = "Hi";
      audio.audio_file = "question.wav";
      require(throws<std::invalid_argument>([&] { pgi::encode_prompt("x", audio); }),
              "audio must be refused");
      GenerationRequest multi;
      multi.prompt = "Hi";
      multi.messages.push_back({"user", "Hello", {}, false});
      require(throws<std::invalid_argument>([&] { pgi::encode_prompt("x", multi); }),
              "multi-turn messages must be refused");
      GenerationRequest tools;
      tools.prompt = "Hi";
      tools.tools.push_back({{"type", "function"}});
      require(throws<std::invalid_argument>([&] { pgi::encode_prompt("x", tools); }),
              "tools must be refused");
    }

    require(pgi::encode_cancel("h1-1") == R"({"id":"h1-1"})", "encode_cancel golden payload");

    // Card -> host payloads (golden strings from the card-side encoder test).
    {
      const pgi::FinalEvent f = pgi::parse_final(
          R"({"id":"h1-1","finish_reason":"stop","generated_tokens":2,"ttft":0.5,"tps":2.0})");
      require(f.id == "h1-1", "final id");
      require(f.finish_reason == "stop", "final finish_reason");
      require(f.generated_tokens == 2, "final generated_tokens");
      require(f.ttft_s == 0.5, "final ttft");
      require(f.tps == 2.0, "final tps");

      const pgi::ErrorEvent e = pgi::parse_error(R"({"id":"h1-1","message":"busy"})");
      require(e.id == "h1-1" && e.message == "busy", "error id and message");

      const pgi::MetricEvent m = pgi::parse_metric(R"({"type":"ttft","value":0.5})");
      require(m.type == "ttft" && m.value == 0.5, "metric type and value");
    }

    {
      // parse_token reads the "<seq>\n<text>" wire format the card sends.
      const pgi::TokenEvent a = pgi::parse_token("0\nHel");
      require(a.has_seq && a.seq == 0 && a.text == "Hel", "seq 0, text Hel");
      const pgi::TokenEvent b = pgi::parse_token("42\nlo");
      require(b.has_seq && b.seq == 42 && b.text == "lo", "multi-digit seq");
      // Review focus: the text may itself contain newlines; split on the first only.
      const pgi::TokenEvent c = pgi::parse_token("7\na\nb");
      require(c.has_seq && c.seq == 7 && c.text == "a\nb", "text keeps later newlines");
      const pgi::TokenEvent e = pgi::parse_token("3\n");
      require(e.has_seq && e.seq == 3 && e.text.empty(), "empty text still has a seq");
      // Review focus: an old-style payload with no numeric prefix is all text.
      const pgi::TokenEvent old = pgi::parse_token("Hello");
      require(!old.has_seq && old.text == "Hello", "no prefix: whole payload is text");
      const pgi::TokenEvent nonnum = pgi::parse_token("x\ny");
      require(!nonnum.has_seq && nonnum.text == "x\ny", "non-numeric prefix: all text");
    }

    require(throws<std::runtime_error>([] { pgi::parse_final("not json"); }),
            "a malformed final must throw");
    require(throws<std::runtime_error>([] { pgi::parse_error("[1,2]"); }),
            "a non-object error must throw");

    // Chat commands. Golden strings shared with the card test
    // (LLiMa tests/runtime/pcie_genai_protocol_test.cpp).
    {
      require(
          pgi::encode_chat("h1-4", pgi::ChatOp::Reset, std::string("Be brief."), false) ==
              R"({"id":"h1-4","op":"reset","system_prompt":"Be brief.","enable_thinking":false})",
          "reset golden (with system prompt)");
      require(pgi::encode_chat("h1-5", pgi::ChatOp::Reset, std::nullopt, true) ==
                  R"({"id":"h1-5","op":"reset","enable_thinking":true})",
              "reset golden (model default system prompt)");
      require(pgi::encode_chat("h1-6", pgi::ChatOp::Reset, std::string(""), false) ==
                  R"({"id":"h1-6","op":"reset","system_prompt":"","enable_thinking":false})",
              "reset golden (no system prompt)");
      require(pgi::encode_chat("h1-7", pgi::ChatOp::Print) == R"({"id":"h1-7","op":"print"})",
              "print golden");

      const pgi::ReplyEvent ok = pgi::parse_reply(R"({"id":"h1-4","ok":true,"text":""})");
      require(ok.id == "h1-4" && ok.ok && ok.text.empty(), "ok reply");
      const pgi::ReplyEvent no = pgi::parse_reply(
          R"({"id":"h1-6","ok":false,"text":"Thinking is not supported for this model."})");
      require(!no.ok && no.text == "Thinking is not supported for this model.", "not-ok reply");
      require(!pgi::parse_reply(R"({"id":"x"})").ok, "a reply with no ok is not ok");
      require(throws<std::runtime_error>([] { pgi::parse_reply("nope"); }),
              "a malformed reply must throw");

      const pgi::FinalEvent cleared = pgi::parse_final(
          R"({"id":"h1-1","finish_reason":"cancelled","generated_tokens":1,"ttft":0.5,"tps":0.0,"history_cleared":true})");
      require(cleared.history_cleared, "final carries history_cleared");
      const pgi::FinalEvent kept = pgi::parse_final(
          R"({"id":"h1-1","finish_reason":"stop","generated_tokens":2,"ttft":0.5,"tps":2.0})");
      require(!kept.history_cleared, "absent history_cleared = false");
      const pgi::ErrorEvent err =
          pgi::parse_error(R"({"id":"h1-1","message":"pull failed","history_cleared":true})");
      require(err.history_cleared && err.message == "pull failed", "error carries history_cleared");
      require(!pgi::parse_error(R"({"id":"h1-1","message":"busy"})").history_cleared,
              "absent history_cleared on error = false");
    }

    std::cout << "[PASS] host GenAI wire protocol\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
