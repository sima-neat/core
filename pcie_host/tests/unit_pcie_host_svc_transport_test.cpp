#include "genai/GenAIModelInternal.h"
#include "genai/GenAIProtocol.h"
#include "genai/SvcTransport.h"
#include "genai_fake_svc_client.h"

#include "simaai/neat/pcie/genai/GenAIModel.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace pgenai = simaai::neat::pcie::genai;
namespace pgi = simaai::neat::pcie::genai::internal;
using pgi::test::FakeSvcClient;

namespace {

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

pgi::SvcTransportOptions fast_options() {
  pgi::SvcTransportOptions o;
  o.model_id = "llama";
  o.recv_timeout_ms = 10;
  o.idle_timeout_ms = 2000;
  o.cancel_timeout_ms = 2000;
  return o;
}

pgenai::GenerationRequest request(const std::string& prompt) {
  pgenai::GenerationRequest r;
  r.prompt = prompt;
  return r;
}

std::string id_of(const std::string& payload) {
  return nlohmann::json::parse(payload)["id"].get<std::string>();
}

std::string final_json(const std::string& id, const std::string& reason) {
  return R"({"id":")" + id + R"(","finish_reason":")" + reason +
         R"(","generated_tokens":2,"ttft":0.5,"tps":2.0})";
}

struct Run {
  std::vector<pgenai::TokenSample> samples;
  std::string error;
};

// Drive one generate() and collect what it emits (or the error it throws).
Run generate(pgi::SvcTransport& t, const pgenai::GenerationRequest& r,
             std::function<bool(const Run&)> cancel_when = nullptr) {
  Run run;
  try {
    t.generate(
        r, [&] { return cancel_when && cancel_when(run); },
        [&](const pgenai::TokenSample& s) { run.samples.push_back(s); });
  } catch (const std::exception& e) {
    run.error = e.what();
  }
  return run;
}

// The card's happy path: ttft metric, two tokens, final.
void play_happy_card(FakeSvcClient& c, const std::string& tag, const std::string& payload) {
  if (tag != pgi::kTagPrompt) {
    return;
  }
  c.push(pgi::kTagMetrics, R"({"type":"ttft","value":0.5})");
  c.push(pgi::kTagToken, "0\nHel");
  c.push(pgi::kTagToken, "1\nlo");
  c.push(pgi::kTagFinal, final_json(id_of(payload), "stop"));
}

} // namespace

int main() {
  try {
    // The constructor subscribes to the four card->host tags, so no event can
    // arrive before we listen (a notification with no listener is dropped).
    {
      auto fake = std::make_unique<FakeSvcClient>();
      FakeSvcClient* f = fake.get();
      pgi::SvcTransport t(std::move(fake), fast_options());
      const auto subs = f->subscribed();
      for (const char* tag :
           {pgi::kTagToken, pgi::kTagMetrics, pgi::kTagFinal, pgi::kTagError, pgi::kTagReply}) {
        require(std::find(subs.begin(), subs.end(), tag) != subs.end(),
                std::string("constructor must subscribe to ") + tag);
      }
      require(t.model_id() == "llama", "model_id comes from the options");
    }

    // Happy path: tokens in order, then one final carrying the card's metrics.
    {
      auto fake = std::make_unique<FakeSvcClient>();
      FakeSvcClient* f = fake.get();
      f->on_notify = play_happy_card;
      pgi::SvcTransport t(std::move(fake), fast_options());
      const Run run = generate(t, request("Hi"));
      require(run.error.empty(), "happy path must not throw: " + run.error);
      require(run.samples.size() == 3, "expected 2 tokens + 1 final");
      require(run.samples[0].text == "Hel" && run.samples[1].text == "lo", "tokens in order");
      const pgenai::TokenSample& fin = run.samples[2];
      require(fin.is_final && fin.finish_reason == "stop", "final with finish_reason stop");
      require(fin.metrics.generated_tokens == 2, "final carries generated_tokens");
      require(fin.metrics.time_to_first_token_s == 0.5, "final carries ttft");
      require(fin.metrics.tokens_per_second == 2.0, "final carries tps");
      require(fin.metrics.dropped_events == 0, "a gap-free run drops nothing");
      const auto sent = f->sent();
      require(sent.size() == 1 && sent[0].tag == pgi::kTagPrompt, "exactly one prompt sent");
      require(nlohmann::json::parse(sent[0].payload)["prompt"] == "Hi", "prompt text sent");
    }

    // A seq gap in the token stream is counted as dropped_events
    // on the final sample. seq jumps 0 -> 2, so exactly one token was lost.
    {
      auto fake = std::make_unique<FakeSvcClient>();
      fake->on_notify = [](FakeSvcClient& c, const std::string& tag, const std::string& payload) {
        if (tag != pgi::kTagPrompt) {
          return;
        }
        c.push(pgi::kTagToken, "0\nA");
        c.push(pgi::kTagToken, "2\nC"); // seq 1 never arrived
        c.push(pgi::kTagFinal, final_json(id_of(payload), "stop"));
      };
      pgi::SvcTransport t(std::move(fake), fast_options());
      const Run run = generate(t, request("Hi"));
      require(run.error.empty(), "a gap must not end the run: " + run.error);
      const pgenai::TokenSample& fin = run.samples.back();
      require(fin.is_final, "the run still ends on a final");
      require(fin.metrics.dropped_events == 1, "one missing seq is one dropped event");
      require(run.samples.size() == 3 && run.samples[0].text == "A" && run.samples[1].text == "C",
              "both delivered tokens still reach the caller");
    }

    // A bigger jump counts every missing seq: 0 -> 5 is four dropped events.
    {
      auto fake = std::make_unique<FakeSvcClient>();
      fake->on_notify = [](FakeSvcClient& c, const std::string& tag, const std::string& payload) {
        if (tag != pgi::kTagPrompt) {
          return;
        }
        c.push(pgi::kTagToken, "0\nA");
        c.push(pgi::kTagToken, "5\nB");
        c.push(pgi::kTagFinal, final_json(id_of(payload), "stop"));
      };
      pgi::SvcTransport t(std::move(fake), fast_options());
      const Run run = generate(t, request("Hi"));
      require(run.samples.back().metrics.dropped_events == 4, "0->5 loses four tokens");
    }

    // A repeated seq is not a drop (and does not underflow the counter).
    {
      auto fake = std::make_unique<FakeSvcClient>();
      fake->on_notify = [](FakeSvcClient& c, const std::string& tag, const std::string& payload) {
        if (tag != pgi::kTagPrompt) {
          return;
        }
        c.push(pgi::kTagToken, "0\nA");
        c.push(pgi::kTagToken, "1\nB");
        c.push(pgi::kTagToken, "1\nB"); // duplicate delivery of the same seq
        c.push(pgi::kTagFinal, final_json(id_of(payload), "stop"));
      };
      pgi::SvcTransport t(std::move(fake), fast_options());
      const Run run = generate(t, request("Hi"));
      require(run.samples.back().metrics.dropped_events == 0, "a duplicate seq is not a drop");
    }

    // An old-style stream with no seq prefix reports zero drops, never a false gap.
    {
      auto fake = std::make_unique<FakeSvcClient>();
      fake->on_notify = [](FakeSvcClient& c, const std::string& tag, const std::string& payload) {
        if (tag != pgi::kTagPrompt) {
          return;
        }
        c.push(pgi::kTagToken, "Hel");
        c.push(pgi::kTagToken, "lo");
        c.push(pgi::kTagFinal, final_json(id_of(payload), "stop"));
      };
      pgi::SvcTransport t(std::move(fake), fast_options());
      const Run run = generate(t, request("Hi"));
      require(run.error.empty(), "a no-seq stream must still work: " + run.error);
      require(run.samples.size() == 3 && run.samples[0].text == "Hel" &&
                  run.samples[1].text == "lo",
              "no-seq tokens are delivered as their whole payload");
      require(run.samples.back().metrics.dropped_events == 0, "no seq means no drop count");
    }

    // The drop counter resets per run: a gap in run 1 does not carry into run 2.
    {
      auto fake = std::make_unique<FakeSvcClient>();
      FakeSvcClient* f = fake.get();
      f->on_notify = [](FakeSvcClient& c, const std::string& tag, const std::string& payload) {
        if (tag != pgi::kTagPrompt) {
          return;
        }
        c.push(pgi::kTagToken, "0\nA");
        c.push(pgi::kTagToken, "3\nB"); // two dropped in run 1
        c.push(pgi::kTagFinal, final_json(id_of(payload), "stop"));
      };
      pgi::SvcTransport t(std::move(fake), fast_options());
      const Run run1 = generate(t, request("Hi"));
      require(run1.samples.back().metrics.dropped_events == 2, "run 1 counts its two drops");

      f->on_notify = play_happy_card; // run 2 is gap-free (seq 0, 1)
      const Run run2 = generate(t, request("Hello"));
      require(run2.samples.back().metrics.dropped_events == 0, "run 2 starts its own count");
    }

    // A drop is still counted when the missing seq is noticed after a cancel:
    // the cancelled final must report the true loss.
    {
      auto fake = std::make_unique<FakeSvcClient>();
      fake->on_notify = [](FakeSvcClient& c, const std::string& tag, const std::string& payload) {
        if (tag == pgi::kTagPrompt) {
          c.push(pgi::kTagToken, "0\nA");
        } else if (tag == pgi::kTagCancel) {
          c.push(pgi::kTagToken, "2\nC"); // arrives after cancel; seq 1 was lost
          c.push(pgi::kTagFinal, final_json(id_of(payload), "cancelled"));
        }
      };
      pgi::SvcTransport t(std::move(fake), fast_options());
      const Run run = generate(t, request("Hi"), [](const Run& r) { return !r.samples.empty(); });
      require(run.error.empty(), "cancel must not throw: " + run.error);
      require(run.samples.back().finish_reason == "cancelled", "ends on the cancelled final");
      require(run.samples.back().metrics.dropped_events == 1,
              "a gap seen after the cancel is still counted");
    }

    // Review focus 2: nobody listens for genai.prompt -> fail at once, clearly.
    {
      auto fake = std::make_unique<FakeSvcClient>();
      fake->listeners = 0;
      pgi::SvcTransport t(std::move(fake), fast_options());
      const Run run = generate(t, request("Hi"));
      require(run.error.find("0 subscribers") != std::string::npos,
              "zero listeners must raise an error naming the cause, got: " + run.error);
      require(run.samples.empty(), "nothing may be emitted without a listener");
    }

    // A card error for this run ends it with an exception.
    {
      auto fake = std::make_unique<FakeSvcClient>();
      fake->on_notify = [](FakeSvcClient& c, const std::string& tag, const std::string& payload) {
        if (tag == pgi::kTagPrompt) {
          c.push(pgi::kTagError, R"({"id":")" + id_of(payload) + R"(","message":"busy"})");
        }
      };
      pgi::SvcTransport t(std::move(fake), fast_options());
      const Run run = generate(t, request("Hi"));
      require(run.error == "card error: busy", "card error must surface, got: " + run.error);
    }

    // Review focus 5: a final or error from an older run is ignored.
    {
      auto fake = std::make_unique<FakeSvcClient>();
      fake->on_notify = [](FakeSvcClient& c, const std::string& tag, const std::string& payload) {
        if (tag != pgi::kTagPrompt) {
          return;
        }
        c.push(pgi::kTagFinal, final_json("old-1", "cancelled"));
        c.push(pgi::kTagError, R"({"id":"old-1","message":"stale"})");
        c.push(pgi::kTagToken, "A");
        c.push(pgi::kTagFinal, final_json(id_of(payload), "stop"));
      };
      pgi::SvcTransport t(std::move(fake), fast_options());
      const Run run = generate(t, request("Hi"));
      require(run.error.empty(), "stale events must not end the run: " + run.error);
      require(run.samples.size() == 2 && run.samples[0].text == "A" && run.samples[1].is_final,
              "only the current run's token and final are delivered");
    }

    // Cancel: the transport sends genai.cancel with the run id, drops later
    // tokens, and ends on the card's "cancelled" final.
    {
      auto fake = std::make_unique<FakeSvcClient>();
      FakeSvcClient* f = fake.get();
      fake->on_notify = [](FakeSvcClient& c, const std::string& tag, const std::string& payload) {
        if (tag == pgi::kTagPrompt) {
          c.push(pgi::kTagToken, "Hel");
        } else if (tag == pgi::kTagCancel) {
          c.push(pgi::kTagToken, "late");
          c.push(pgi::kTagFinal, final_json(id_of(payload), "cancelled"));
        }
      };
      pgi::SvcTransport t(std::move(fake), fast_options());
      const Run run = generate(t, request("Hi"), [](const Run& r) { return !r.samples.empty(); });
      require(run.error.empty(), "cancel must not throw: " + run.error);
      require(run.samples.size() == 2 && run.samples[0].text == "Hel", "token before cancel kept");
      require(run.samples[1].is_final && run.samples[1].finish_reason == "cancelled",
              "the card's cancelled final ends the run");
      const auto sent = f->sent();
      require(sent.size() == 2 && sent[1].tag == pgi::kTagCancel, "one cancel sent");
      require(id_of(sent[1].payload) == id_of(sent[0].payload), "cancel carries the run id");
    }

    // Review focus 1: after a cancel the card never answers -> give up after
    // cancel_timeout_ms instead of hanging.
    {
      auto fake = std::make_unique<FakeSvcClient>();
      fake->on_notify = [](FakeSvcClient& c, const std::string& tag, const std::string&) {
        if (tag == pgi::kTagPrompt) {
          c.push(pgi::kTagToken, "Hel");
        }
      };
      pgi::SvcTransportOptions o = fast_options();
      o.cancel_timeout_ms = 50;
      pgi::SvcTransport t(std::move(fake), o);
      const auto start = std::chrono::steady_clock::now();
      const Run run = generate(t, request("Hi"), [](const Run& r) { return !r.samples.empty(); });
      const auto waited = std::chrono::steady_clock::now() - start;
      require(run.error.empty(), "a cancel with no final must return quietly: " + run.error);
      require(waited < std::chrono::milliseconds(1500), "cancel wait must be bounded");
    }

    // Review focus 3: the card goes silent -> clear error after idle_timeout_ms.
    {
      auto fake = std::make_unique<FakeSvcClient>();
      pgi::SvcTransportOptions o = fast_options();
      o.idle_timeout_ms = 50;
      pgi::SvcTransport t(std::move(fake), o);
      const Run run = generate(t, request("Hi"));
      require(run.error.find("no event from the card") != std::string::npos,
              "silence must time out with a clear error, got: " + run.error);
    }

    // Review focus 3: the local daemon goes away -> clear error.
    {
      auto fake = std::make_unique<FakeSvcClient>();
      fake->on_notify = [](FakeSvcClient& c, const std::string& tag, const std::string&) {
        if (tag == pgi::kTagPrompt) {
          c.disconnect();
        }
      };
      pgi::SvcTransport t(std::move(fake), fast_options());
      const Run run = generate(t, request("Hi"));
      require(run.error.find("lost the host simaai_svc daemon") != std::string::npos,
              "a daemon loss must raise a clear error, got: " + run.error);
    }

    // A non-text request is refused before anything is sent.
    {
      auto fake = std::make_unique<FakeSvcClient>();
      FakeSvcClient* f = fake.get();
      pgi::SvcTransport t(std::move(fake), fast_options());
      pgenai::GenerationRequest r = request("Hi");
      r.audio_file = "q.wav";
      const Run run = generate(t, r);
      require(!run.error.empty(), "an audio request must be refused");
      require(f->sent().empty(), "nothing may be sent for a refused request");
    }

    // Through the real GenAIModel: a card error thrown on the stream worker
    // reaches the caller's next() (GenerationStream rethrows producer errors).
    {
      auto fake = std::make_unique<FakeSvcClient>();
      fake->on_notify = [](FakeSvcClient& c, const std::string& tag, const std::string& payload) {
        if (tag == pgi::kTagPrompt) {
          c.push(pgi::kTagError,
                 R"({"id":")" + id_of(payload) + R"(","message":"mla load failed"})");
        }
      };
      pgenai::GenAIModel model = pgi::make_model_with_transport(
          std::make_unique<pgi::SvcTransport>(std::move(fake), fast_options()));
      pgenai::GenerationStream stream = model.stream(request("Hi"));
      std::string error;
      try {
        while (stream.next()) {
        }
      } catch (const std::exception& e) {
        error = e.what();
      }
      require(error == "card error: mla load failed", "stream.next() must rethrow, got: " + error);
    }

    // Fix round 1: abandoned run's late tokens must not leak into the next run.
    {
      auto fake = std::make_unique<FakeSvcClient>();
      FakeSvcClient* f = fake.get();
      fake->on_notify = [](FakeSvcClient& c, const std::string& tag, const std::string&) {
        if (tag == pgi::kTagPrompt) {
          c.push(pgi::kTagToken, "old");
        }
        // no response to cancel: the card keeps running after we give up
      };
      pgi::SvcTransportOptions o = fast_options();
      o.cancel_timeout_ms = 50;
      pgi::SvcTransport t(std::move(fake), o);

      // Run 1: cancel after one token, but the card never sends the cancelled final.
      const Run run1 = generate(t, request("Hi"), [](const Run& r) { return !r.samples.empty(); });
      require(run1.error.empty(), "run 1 must return quietly: " + run1.error);
      require(run1.samples.size() == 1 && run1.samples[0].text == "old", "run 1 got one token");

      // Before run 2: simulate the card finishing run 1 late.
      const std::string run1_id = id_of(f->sent()[0].payload);
      f->push(pgi::kTagToken, "stale-1");
      f->push(pgi::kTagToken, "stale-2");
      f->push(pgi::kTagFinal, final_json(run1_id, "cancelled"));

      // Run 2: normal happy card. The stale tokens must be drained, not emitted.
      f->on_notify = play_happy_card;
      const Run run2 = generate(t, request("Hello"));
      require(run2.error.empty(), "run 2 must succeed: " + run2.error);
      require(run2.samples.size() == 3, "run 2: 2 tokens + final");
      require(run2.samples[0].text == "Hel" && run2.samples[1].text == "lo",
              "run 2 tokens must be fresh, not stale");
      require(run2.samples[2].is_final, "run 2 final");
    }

    // Fix round 1: if the old final never comes, the drain gives up and run 2 still works.
    {
      auto fake = std::make_unique<FakeSvcClient>();
      FakeSvcClient* f = fake.get();
      f->on_notify = [](FakeSvcClient& c, const std::string& tag, const std::string&) {
        if (tag == pgi::kTagPrompt) {
          c.push(pgi::kTagToken, "old");
        }
      };
      pgi::SvcTransportOptions o = fast_options();
      o.cancel_timeout_ms = 50;
      pgi::SvcTransport t(std::move(fake), o);

      // Run 1: abandoned (cancel timeout).
      const Run run1 = generate(t, request("Hi"), [](const Run& r) { return !r.samples.empty(); });
      require(run1.error.empty() && run1.samples.size() == 1, "run 1 abandoned");

      // Run 2: the old final never arrives, so the drain times out, but run 2 still completes.
      f->on_notify = play_happy_card;
      const auto start = std::chrono::steady_clock::now();
      const Run run2 = generate(t, request("Hello"));
      const auto waited = std::chrono::steady_clock::now() - start;
      require(run2.error.empty(), "run 2 must succeed even if drain timed out: " + run2.error);
      require(run2.samples.size() == 3, "run 2: full response");
      require(waited < std::chrono::milliseconds(1500), "drain timeout must be bounded");
    }

    // Final review: a malformed payload ends the run with an error AND marks it
    // abandoned, so its leftovers are drained instead of leaking into the next run.
    {
      auto fake = std::make_unique<FakeSvcClient>();
      FakeSvcClient* f = fake.get();
      f->on_notify = [](FakeSvcClient& c, const std::string& tag, const std::string&) {
        if (tag == pgi::kTagPrompt) {
          c.push(pgi::kTagToken, "old");
          c.push(pgi::kTagFinal, "not json");
        }
      };
      pgi::SvcTransport t(std::move(fake), fast_options());

      const Run run1 = generate(t, request("Hi"));
      require(!run1.error.empty(), "a malformed final must end the run with an error");

      // The card's run 1 goes on: late tokens, then its real final.
      const std::string run1_id = id_of(f->sent()[0].payload);
      f->push(pgi::kTagToken, "stale-1");
      f->push(pgi::kTagToken, "stale-2");
      f->push(pgi::kTagFinal, final_json(run1_id, "stop"));

      f->on_notify = play_happy_card;
      const Run run2 = generate(t, request("Hello"));
      require(run2.error.empty(), "run 2 must succeed: " + run2.error);
      require(run2.samples.size() == 3 && run2.samples[0].text == "Hel" &&
                  run2.samples[1].text == "lo" && run2.samples[2].is_final,
              "run 2 must deliver only its own tokens and final");
    }

    // A VLM request stages its image (under the configured stage dir), sends the
    // staged name in the prompt, and the copy is gone once generate() returns.
    {
      namespace fs = std::filesystem;
      const fs::path root = fs::temp_directory_path() / ("svc_img_" + std::to_string(::getpid()));
      const fs::path stage = root / "pcie-genai";
      const fs::path src = root / "in.jpg";
      fs::create_directories(root);
      std::ofstream(src) << "JPEG";

      auto fake = std::make_unique<FakeSvcClient>();
      FakeSvcClient* f = fake.get();
      bool existed_during_send = false;
      f->on_notify = [&](FakeSvcClient& c, const std::string& tag, const std::string& payload) {
        if (tag != pgi::kTagPrompt) {
          return;
        }
        const auto j = nlohmann::json::parse(payload);
        if (j.contains("images")) {
          existed_during_send = fs::exists(root / j["images"][0].get<std::string>());
        }
        c.push(pgi::kTagFinal, final_json(id_of(payload), "stop"));
      };
      pgi::SvcTransportOptions o = fast_options();
      o.image_stage_dir = stage.string();
      pgi::SvcTransport t(std::move(fake), o);
      pgenai::GenerationRequest r = request("what is on the image");
      r.image_files = {src};
      const Run run = generate(t, r);
      require(run.error.empty(), "an image request generates without error: " + run.error);
      const auto sent = f->sent();
      require(sent.size() == 1, "exactly one prompt sent for an image request");
      const auto j = nlohmann::json::parse(sent[0].payload);
      require(j.contains("images") &&
                  j["images"][0].get<std::string>().rfind("pcie-genai/", 0) == 0,
              "the prompt carries the staged image name");
      require(existed_during_send, "the staged copy exists while the prompt is sent");
      require(!fs::exists(stage) || fs::is_empty(stage),
              "the staged copy is gone after generate() returns");
      fs::remove_all(root);
    }

    // Two images: both staged under distinct names, both exist during the send,
    // both are gone after generate() returns.
    {
      namespace fs = std::filesystem;
      const fs::path root = fs::temp_directory_path() / ("svc_img2_" + std::to_string(::getpid()));
      const fs::path stage = root / "pcie-genai";
      fs::create_directories(root);
      std::ofstream(root / "a.jpg") << "JPEG";
      std::ofstream(root / "b.png") << "PNG";

      auto fake = std::make_unique<FakeSvcClient>();
      FakeSvcClient* f = fake.get();
      std::vector<std::string> names;
      bool all_existed = true;
      f->on_notify = [&](FakeSvcClient& c, const std::string& tag, const std::string& payload) {
        if (tag != pgi::kTagPrompt) {
          return;
        }
        // Keep the parsed JSON alive: a range-for over a temporary's member dangles.
        const nlohmann::json j = nlohmann::json::parse(payload);
        for (const auto& n : j["images"]) {
          names.push_back(n.get<std::string>());
          all_existed = all_existed && fs::exists(root / names.back());
        }
        c.push(pgi::kTagFinal, final_json(id_of(payload), "stop"));
      };
      pgi::SvcTransportOptions o = fast_options();
      o.image_stage_dir = stage.string();
      pgi::SvcTransport t(std::move(fake), o);
      pgenai::GenerationRequest r = request("compare them");
      r.image_files = {root / "a.jpg", root / "b.png"};
      const Run run = generate(t, r);
      require(run.error.empty(), "a two-image request generates without error: " + run.error);
      require(names.size() == 2 && names[0] != names[1], "two distinct staged names");
      require(names[0].size() > 4 && names[0].substr(names[0].size() - 4) == ".jpg" &&
                  names[1].substr(names[1].size() - 4) == ".png",
              "each staged name keeps its extension, in order");
      require(all_existed, "both staged copies exist while the prompt is sent");
      require(!fs::exists(stage) || fs::is_empty(stage), "both copies are gone afterwards");
      fs::remove_all(root);
    }

    // image_files set but no stage dir configured -> refused, nothing sent.
    {
      auto fake = std::make_unique<FakeSvcClient>();
      FakeSvcClient* f = fake.get();
      pgi::SvcTransport t(std::move(fake), fast_options()); // image_stage_dir empty
      pgenai::GenerationRequest r = request("hi");
      r.image_files = {"/tmp/whatever.jpg"};
      const Run run = generate(t, r);
      require(run.error.find("image") != std::string::npos,
              "an image with no stage dir must be refused, got: " + run.error);
      require(f->sent().empty(), "nothing may be sent when the image cannot be staged");
    }

    // Chat reset sends genai.chat and returns the card's reply.
    {
      auto fake = std::make_unique<FakeSvcClient>();
      FakeSvcClient* f = fake.get();
      f->on_notify = [](FakeSvcClient& c, const std::string& tag, const std::string& payload) {
        if (tag == pgi::kTagChat) {
          c.push(pgi::kTagReply, R"({"id":")" + id_of(payload) + R"(","ok":true,"text":""})");
        }
      };
      pgi::SvcTransport t(std::move(fake), fast_options());
      const pgenai::ChatReply r = t.reset_chat(std::string("Be brief."), false);
      require(r.ok, "reset is ok");
      const auto sent = f->sent();
      require(sent.size() == 1 && sent[0].tag == pgi::kTagChat, "one genai.chat sent");
      const auto j = nlohmann::json::parse(sent[0].payload);
      require(j["op"] == "reset" && j["system_prompt"] == "Be brief." &&
                  j["enable_thinking"] == false,
              "reset carries the settings");
    }

    // A late reply with another id is ignored; the matching one is used.
    {
      auto fake = std::make_unique<FakeSvcClient>();
      FakeSvcClient* f = fake.get();
      f->on_notify = [](FakeSvcClient& c, const std::string& tag, const std::string& payload) {
        if (tag == pgi::kTagChat) {
          c.push(pgi::kTagReply, R"({"id":"old-1","ok":false,"text":"stale"})");
          c.push(pgi::kTagReply, R"({"id":")" + id_of(payload) + R"(","ok":true,"text":"[]"})");
        }
      };
      pgi::SvcTransport t(std::move(fake), fast_options());
      const pgenai::ChatReply r = t.chat_history();
      require(r.ok && r.text == "[]", "the reply with our id wins, got: " + r.text);
    }

    // No reply: bounded wait, then a clear error.
    {
      auto fake = std::make_unique<FakeSvcClient>();
      pgi::SvcTransportOptions o = fast_options();
      o.chat_timeout_ms = 100;
      pgi::SvcTransport t(std::move(fake), o);
      std::string error;
      try {
        (void)t.reset_chat(std::nullopt, false);
      } catch (const std::exception& e) {
        error = e.what();
      }
      require(error.find("no reply from the card") != std::string::npos,
              "a missing reply times out, got: " + error);
    }

    // An old card (no genai.chat subscriber): fail fast, no wait.
    {
      auto fake = std::make_unique<FakeSvcClient>();
      fake->listeners = 0;
      pgi::SvcTransport t(std::move(fake), fast_options());
      std::string error;
      try {
        (void)t.chat_history();
      } catch (const std::exception& e) {
        error = e.what();
      }
      require(error.find("0 subscribers") != std::string::npos,
              "a card without genai.chat is reported, got: " + error);
    }

    // history_cleared from the final / error is reported per run.
    {
      auto fake = std::make_unique<FakeSvcClient>();
      FakeSvcClient* f = fake.get();
      int run_no = 0;
      f->on_notify = [&](FakeSvcClient& c, const std::string& tag, const std::string& payload) {
        if (tag != pgi::kTagPrompt) {
          return;
        }
        ++run_no;
        const std::string id = id_of(payload);
        if (run_no == 1) {
          c.push(
              pgi::kTagFinal,
              R"({"id":")" + id +
                  R"(","finish_reason":"stop","generated_tokens":0,"ttft":0.0,"tps":0.0,"history_cleared":true})");
        } else if (run_no == 2) {
          c.push(pgi::kTagFinal, final_json(id, "stop"));
        } else {
          c.push(pgi::kTagError,
                 R"({"id":")" + id + R"(","message":"pull failed","history_cleared":true})");
        }
      };
      pgi::SvcTransport t(std::move(fake), fast_options());
      (void)generate(t, request("a"));
      require(t.last_run_cleared_history(), "run 1: the card cleared the history");
      (void)generate(t, request("b"));
      require(!t.last_run_cleared_history(), "run 2: history kept");
      const Run run3 = generate(t, request("c"));
      require(!run3.error.empty() && t.last_run_cleared_history(),
              "run 3: an error that cleared the history is reported");
    }

    // print history maps card paths back to host paths; a reset forgets them.
    {
      namespace fs = std::filesystem;
      const fs::path root = fs::temp_directory_path() / ("svc_hist_" + std::to_string(::getpid()));
      fs::create_directories(root);
      const fs::path src = root / "cat.jpg";
      std::ofstream(src) << "JPEG";

      auto fake = std::make_unique<FakeSvcClient>();
      FakeSvcClient* f = fake.get();
      std::string sent_name;
      f->on_notify = [&](FakeSvcClient& c, const std::string& tag, const std::string& payload) {
        const std::string id = id_of(payload);
        const nlohmann::json j = nlohmann::json::parse(payload);
        if (tag == pgi::kTagPrompt) {
          sent_name = j["images"][0].get<std::string>();
          c.push(pgi::kTagFinal, final_json(id, "stop"));
        } else if (tag == pgi::kTagChat) {
          if (j["op"] == "print") {
            nlohmann::json text = nlohmann::json::array(
                {{{"role", "user"},
                  {"content", {{{"type", "image"}, {"image", "/data/recv/" + sent_name}}}}}});
            nlohmann::json reply = {{"id", id}, {"ok", true}, {"text", text.dump()}};
            c.push(pgi::kTagReply, reply.dump());
          } else {
            c.push(pgi::kTagReply, R"({"id":")" + id + R"(","ok":true,"text":""})");
          }
        }
      };
      pgi::SvcTransportOptions o = fast_options();
      o.image_stage_dir = (root / "pcie-genai").string();
      pgi::SvcTransport t(std::move(fake), o);
      pgenai::GenerationRequest r = request("what");
      r.image_files = {src};
      require(generate(t, r).error.empty(), "image run ok");
      const pgenai::ChatReply before = t.chat_history();
      require(before.text.find(src.string()) != std::string::npos,
              "print shows the host path, got: " + before.text);
      require(t.reset_chat(std::nullopt, false).ok, "reset ok");
      const pgenai::ChatReply after = t.chat_history();
      require(after.text.find("/data/recv/") != std::string::npos,
              "after a reset the name table is empty, got: " + after.text);
      fs::remove_all(root);
    }

    std::cout << "[PASS] host SvcTransport over a fake svc client\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
