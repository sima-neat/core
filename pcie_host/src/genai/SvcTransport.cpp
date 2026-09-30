// SvcTransport: one GenAI run over simaai_svc. Send genai.prompt, then turn
// genai.token / genai.metrics / genai.final / genai.error into TokenSamples.
// Tokens carry no id, so the code keeps a strict "one run at a time" rule and
// cleans up after a run that ended early (see drain_abandoned_run).

#include "genai/SvcTransport.h"

#include "genai/GenAIProtocol.h"
#include "genai/HistoryPaths.h"
#include "genai/ImageStage.h"

#include <chrono>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

namespace simaai::neat::pcie::genai::internal {

SvcTransport::SvcTransport(std::unique_ptr<SvcClient> client, SvcTransportOptions options)
    : client_(std::move(client)), options_(std::move(options)) {
  if (!client_) {
    throw std::invalid_argument("SvcTransport requires an svc client");
  }
  // Subscribe here, before any prompt can be sent. The daemon drops a
  // notification that has no listener, so if we subscribed later, the first
  // tokens (or a fast final) of the first run could be lost.
  for (const char* tag : {kTagToken, kTagMetrics, kTagFinal, kTagError, kTagReply}) {
    client_->subscribe(tag);
  }
}

std::string SvcTransport::model_id() const {
  return options_.model_id;
}

void SvcTransport::cancel() {
  cancel_requested_.store(true);
}

// "h<pid>-<n>": unique per host process and per run, so the card's final and
// error can be matched to the run that asked for them.
std::string SvcTransport::next_request_id() {
  return "h" + std::to_string(static_cast<long long>(::getpid())) + "-" +
         std::to_string(++request_counter_);
}

// Drop id-less tokens/metrics left over from a run that ended without receiving its final.
// Why: tokens and metrics carry no id. If the old run is still sending, its
// tokens would look like the new run's answer. So before the next prompt we
// read and drop events until the old run's final or error arrives.
// If it does not arrive in time, throw and keep the run marked abandoned: the
// old run may still be sending, so no new request may start yet. The next
// request tries the drain again.
void SvcTransport::drain_abandoned_run() {
  using clock = std::chrono::steady_clock;
  const std::string old_id = abandoned_id_;

  const clock::time_point start = clock::now();
  for (;;) {
    if (clock::now() - start >= std::chrono::milliseconds(options_.cancel_timeout_ms)) {
      throw std::runtime_error("the card is still finishing an earlier run; try again, or "
                               "restart pcie-genai if this does not go away");
    }
    SvcNote note;
    const RecvStatus status = client_->recv(note, options_.recv_timeout_ms);
    if (status == RecvStatus::Disconnected) {
      throw std::runtime_error("lost the host simaai_svc daemon during generation");
    }
    if (status == RecvStatus::Timeout) {
      continue;
    }
    // Drop everything until we see the old run's final or error.
    // A malformed leftover is just more junk to drop.
    try {
      if (note.tag == kTagFinal) {
        if (parse_final(note.payload).id == old_id) {
          abandoned_id_.clear();
          return;
        }
      } else if (note.tag == kTagError) {
        if (parse_error(note.payload).id == old_id) {
          abandoned_id_.clear();
          return;
        }
      }
    } catch (const std::exception&) {
    }
  }
}

void SvcTransport::generate(const GenerationRequest& request,
                            const std::function<bool()>& is_cancelled,
                            const std::function<void(const TokenSample&)>& emit) {
  using clock = std::chrono::steady_clock;
  const std::string id = next_request_id();
  cancel_requested_.store(false);
  expected_token_seq_ = 0;
  dropped_events_ = 0;
  last_history_cleared_ = false;

  // Stage the images (if any) before encoding, so their names go in the prompt.
  // The StagedImages live for the whole run and delete their copies when we
  // return (success, error, or cancel), because generate() owns them.
  std::vector<std::unique_ptr<StagedImage>> staged;
  std::vector<std::string> image_names;
  if (!request.image_files.empty()) {
    if (options_.image_stage_dir.empty()) {
      throw std::invalid_argument(
          "this connection has no image stage dir: the daemon config has no [serve] 'data' "
          "root, so images cannot be sent");
    }
    for (std::size_t i = 0; i < request.image_files.size(); ++i) {
      staged.push_back(std::make_unique<StagedImage>(
          options_.image_stage_dir, id + "-" + std::to_string(i), request.image_files[i]));
      image_names.push_back(staged.back()->relative_name());
      sent_images_[image_names.back()] = request.image_files[i].string();
    }
  }

  // Encode after staging: a request the card cannot serve is still refused
  // before the notify (the staged copies, if any, are cleaned up on throw).
  const std::string prompt = encode_prompt(id, request, image_names);

  // Drop any leftovers from a previous run that ended without receiving its final.
  if (!abandoned_id_.empty()) {
    drain_abandoned_run();
  }

  // On the host, notify() returns the real number of card apps that got the
  // note (only the card side always reports 0). 0 here means no backend is
  // listening, so no answer will ever come: fail now instead of waiting for
  // the idle timeout.
  if (client_->notify(kTagPrompt, prompt) == 0) {
    throw std::runtime_error(
        "genai.prompt has 0 subscribers on the card: is pcie-genai-backend running and READY?");
  }

  double live_ttft_s = 0.0;
  bool cancel_sent = false;
  clock::time_point last_event = clock::now();
  clock::time_point cancel_sent_at{};

  // A malformed payload ends this run with an error. The card's run may still be
  // going, so mark it abandoned: its later events are drained before the next run.
  const auto parse_or_abandon = [&](auto parse, const std::string& payload) {
    try {
      return parse(payload);
    } catch (...) {
      abandoned_id_ = id;
      throw;
    }
  };

  for (;;) {
    // Checked once per recv() wait, so a cancel is noticed within
    // recv_timeout_ms. genai.cancel is sent only once.
    if (!cancel_sent && (cancel_requested_.load() || is_cancelled())) {
      client_->notify(kTagCancel, encode_cancel(id));
      cancel_sent = true;
      cancel_sent_at = clock::now();
    }

    SvcNote note;
    const RecvStatus status = client_->recv(note, options_.recv_timeout_ms);
    const clock::time_point now = clock::now();
    if (status == RecvStatus::Disconnected) {
      abandoned_id_ = id;
      throw std::runtime_error("lost the host simaai_svc daemon during generation");
    }
    if (status == RecvStatus::Timeout) {
      // The cancel wait is bounded: if the card never sends its final (for
      // example it is stuck or gone), the caller must not wait forever. The
      // run is marked abandoned, so its late events are drained before the
      // next prompt.
      if (cancel_sent &&
          now - cancel_sent_at >= std::chrono::milliseconds(options_.cancel_timeout_ms)) {
        abandoned_id_ = id;
        return; // the stream is already cancelled; do not hold its worker forever
      }
      if (now - last_event >= std::chrono::milliseconds(options_.idle_timeout_ms)) {
        abandoned_id_ = id;
        throw std::runtime_error("no event from the card for " +
                                 std::to_string(options_.idle_timeout_ms) + " ms");
      }
      continue;
    }
    last_event = now;

    if (note.tag == kTagToken) {
      const TokenEvent token = parse_token(note.payload);
      // A seq ahead of what we expected means the notifications in between were
      // dropped in transit. Count them, even after a cancel, so the final still
      // reports the true loss. A repeated or older seq is not a drop.
      if (token.has_seq) {
        if (token.seq > expected_token_seq_) {
          dropped_events_ += static_cast<std::uint32_t>(token.seq - expected_token_seq_);
          expected_token_seq_ = token.seq + 1;
        } else if (token.seq == expected_token_seq_) {
          ++expected_token_seq_;
        }
      }
      // After a cancel, drop tokens: the user asked to stop, and the card may
      // still send a few tokens before its stop takes effect. We keep reading
      // only to get the final.
      if (!cancel_sent) {
        TokenSample sample;
        sample.text = token.text;
        emit(sample);
      }
    } else if (note.tag == kTagMetrics) {
      const MetricEvent metric = parse_or_abandon(parse_metric, note.payload);
      if (metric.type == "ttft") {
        live_ttft_s = metric.value;
      }
    } else if (note.tag == kTagFinal) {
      const FinalEvent final_event = parse_or_abandon(parse_final, note.payload);
      // final and error carry the request id. One with another id belongs to
      // an older run (for example one we gave up on) and must not end this one.
      if (final_event.id != id) {
        continue; // left over from an earlier run
      }
      last_history_cleared_ = final_event.history_cleared;
      if (last_history_cleared_) {
        sent_images_.clear();
      }
      TokenSample sample;
      sample.is_final = true;
      sample.finish_reason = final_event.finish_reason;
      sample.metrics.generated_tokens = final_event.generated_tokens;
      sample.metrics.time_to_first_token_s =
          final_event.ttft_s > 0.0 ? final_event.ttft_s : live_ttft_s;
      sample.metrics.tokens_per_second = final_event.tps;
      // Only gaps revealed by a later, higher seq are counted. Token
      // notifications dropped after the last one we received leave no gap, so
      // trailing-token loss is not reported (like a dropped final). This is a
      // known limitation; the fix is to send the token count in genai.final.
      sample.metrics.dropped_events = dropped_events_;
      emit(sample);
      return;
    } else if (note.tag == kTagError) {
      const ErrorEvent error = parse_or_abandon(parse_error, note.payload);
      // An empty id is kept: the card sends it when it could not read the
      // prompt's id, and that error is about our prompt.
      if (!error.id.empty() && error.id != id) {
        continue; // left over from an earlier run
      }
      last_history_cleared_ = error.history_cleared;
      if (last_history_cleared_) {
        sent_images_.clear();
      }
      throw std::runtime_error("card error: " + error.message);
    }
  }
}

ChatReply SvcTransport::reset_chat(const std::optional<std::string>& system_prompt,
                                   bool enable_thinking) {
  return chat_op(ChatOp::Reset, system_prompt, enable_thinking);
}

ChatReply SvcTransport::chat_history() {
  return chat_op(ChatOp::Print, std::nullopt, false);
}

bool SvcTransport::last_run_cleared_history() const {
  return last_history_cleared_;
}

// One chat command: send genai.chat, wait (bounded) for the genai.reply with
// the same id. Anything else that arrives meanwhile is ignored.
ChatReply SvcTransport::chat_op(ChatOp op, const std::optional<std::string>& system_prompt,
                                bool enable_thinking) {
  using clock = std::chrono::steady_clock;
  const std::string id = next_request_id();
  if (!abandoned_id_.empty()) {
    drain_abandoned_run();
  }
  if (client_->notify(kTagChat, encode_chat(id, op, system_prompt, enable_thinking)) == 0) {
    throw std::runtime_error("genai.chat has 0 subscribers on the card: is a pcie-genai-backend "
                             "with chat support running?");
  }
  const clock::time_point start = clock::now();
  for (;;) {
    if (clock::now() - start >= std::chrono::milliseconds(options_.chat_timeout_ms)) {
      throw std::runtime_error("no reply from the card");
    }
    SvcNote note;
    const RecvStatus status = client_->recv(note, options_.recv_timeout_ms);
    if (status == RecvStatus::Disconnected) {
      throw std::runtime_error("lost the host simaai_svc daemon");
    }
    if (status != RecvStatus::Ok || note.tag != kTagReply) {
      continue;
    }
    ReplyEvent reply;
    try {
      reply = parse_reply(note.payload);
    } catch (const std::exception&) {
      continue; // a malformed reply is not ours to act on
    }
    if (reply.id != id) {
      continue; // a late reply to an older command
    }
    if (op == ChatOp::Reset && reply.ok) {
      sent_images_.clear();
    }
    if (op == ChatOp::Print && reply.ok) {
      reply.text = map_history_image_paths(reply.text, sent_images_);
    }
    return ChatReply{reply.ok, reply.text};
  }
}

} // namespace simaai::neat::pcie::genai::internal
