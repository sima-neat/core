#include "genai/GenAIModel.h"
#include "genai/GenAIInternal.h"
#include "genai/ScopedFileProvider.h"
#include "pcie_genai/PcieFileProvider.h"
#include "Protocol.h"
#include "Service.h"
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <iostream>
#include <mutex>
#include <thread>

namespace wire = simaai::neat::pcie::genai::wire;
namespace local = simaai::neat::genai;
using Clock = std::chrono::steady_clock;
namespace {
volatile std::sig_atomic_t interrupted = 0;
void signal_stop(int) {
  interrupted = 1;
}

local::GenerationRequest request_from_json(const wire::Json& j, wire::PcieFileProvider& files,
                                           uint64_t request) {
  local::GenerationRequest out;
  if (j.contains("prompt"))
    out.prompt = j.at("prompt").get<std::string>();
  if (j.contains("system_prompt"))
    out.system_prompt = j.at("system_prompt").get<std::string>();
  out.max_new_tokens = j.at("max_new_tokens");
  out.enable_thinking = j.at("enable_thinking");
  out.language = j.at("language");
  const auto task = j.at("asr_task").get<std::string>();
  if (task != "transcribe" && task != "translate")
    throw std::invalid_argument("Invalid ASR task");
  out.asr_task = task == "translate" ? local::ASRTask::Translate : local::ASRTask::Transcribe;
  out.tools = j.at("tools");
  out.tool_choice = j.at("tool_choice");
  std::size_t index = 0;
  auto media = [&](const wire::Json& descriptor) {
    auto path = files.media(request, descriptor.at("root"), descriptor.at("name"), index++);
    if (std::filesystem::file_size(path) != descriptor.at("bytes").get<uint64_t>())
      throw std::runtime_error("Media transfer size mismatch");
    return path;
  };
  auto images = [&](const wire::Json& list, local::ImageList& target) {
    for (const auto& descriptor : list) {
      auto path = media(descriptor);
      auto bgr = cv::imread(path.string(), cv::IMREAD_COLOR);
      if (bgr.empty())
        throw std::invalid_argument("Cannot decode image input");
      local::ImageList converted(std::vector<cv::Mat>{bgr});
      target.tensors().push_back(converted.tensors().front());
    }
  };
  images(j.at("images"), out.images);
  if (j.contains("audio_file"))
    out.audio_file = media(j.at("audio_file"));
  for (const auto& m : j.at("messages")) {
    local::ChatMessage message;
    message.role = m.at("role");
    message.content = m.at("content");
    message.tool_calls = m.at("tool_calls");
    if (m.contains("tool_call_id"))
      message.tool_call_id = m.at("tool_call_id").get<std::string>();
    if (m.contains("name"))
      message.name = m.at("name").get<std::string>();
    images(m.at("images"), message.images);
    out.messages.push_back(std::move(message));
  }
  return out;
}
} // namespace

int main(int argc, char** argv) {
  // Arguments are passed by the host as separate, shell-quoted values.
  if (argc != 6) {
    std::cerr << "usage: neat-pcie-genai-worker SESSION RECV_ROOT ASSET_TIMEOUT_MS TARGET_DIR "
                 "DRAFT_DIR\n";
    return 2;
  }
  const std::string session = argv[1];
  try {
    wire::validate_session(session);
    std::signal(SIGTERM, signal_stop);
    std::signal(SIGINT, signal_stop);
    wire::Service control(0, true);
    control.subscribe(wire::tag(session, false));
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<wire::Json> events;
    std::optional<wire::Json> command;
    std::optional<wire::Json> capabilities;
    std::string startup_error;
    std::shared_ptr<local::GenerationStream> stream;
    std::atomic<bool> stop{false}, cancel{false};
    uint64_t active = 0, next_sequence = 0;
    bool completed = true;

    std::thread executor([&] {
      try {
        auto files = std::make_shared<wire::PcieFileProvider>(argv[2], session, std::stoi(argv[3]),
                                                              [&] { return stop || interrupted; });
        std::shared_ptr<simaai::llima::FileProvider> target_files = files, draft_files;
        std::filesystem::path target_root = files->model_root();
        if (argv[4][0]) {
          const auto target = wire::relative_name(argv[4]);
          target_files = std::make_shared<local::internal::ScopedFileProvider>(files, target);
          target_root /= target;
        }
        if (argv[5][0])
          draft_files = std::make_shared<local::internal::ScopedFileProvider>(
              files, wire::relative_name(argv[5]));
        auto model = local::internal::ModelAccess::create(
            local::internal::provider_model_context(target_root, target_files, draft_files));
        {
          std::lock_guard lock(mutex);
          capabilities = {{"task", model.accepts_audio() ? "asr" : "vision_language"},
                          {"text", model.accepts_text()},
                          {"image", model.accepts_image()},
                          {"audio", model.accepts_audio()}};
        }
        while (!stop) {
          wire::Json job;
          uint64_t id;
          {
            std::unique_lock lock(mutex);
            changed.wait(lock, [&] { return stop || command.has_value(); });
            if (stop)
              break;
            job = std::move(*command);
            command.reset();
            id = active;
          }
          auto emit = [&](wire::Json event) {
            std::unique_lock lock(mutex);
            if (!changed.wait_for(lock, std::chrono::seconds(30), [&] {
                  return stop || cancel || events.size() < wire::event_window;
                }))
              throw std::runtime_error("Host did not acknowledge GenAI events");
            if (stop || cancel)
              return;
            event["sequence"] = next_sequence + 1;
            if (event.dump().size() > wire::max_message_bytes)
              throw std::length_error("Generated event too large");
            ++next_sequence;
            events.push_back(std::move(event));
          };
          std::optional<wire::Json> terminal;
          try {
            auto request = request_from_json(job.at("body"), *files, id);
            if (cancel)
              throw std::runtime_error("Request cancelled during loading");
            auto running = std::make_shared<local::GenerationStream>(model.stream(request));
            {
              std::lock_guard lock(mutex);
              stream = running;
              if (cancel || stop)
                running->cancel();
            }
            while (auto sample = running->next()) {
              auto event = wire::envelope(session, id, "sample");
              event["body"] = wire::encode(*sample);
              if (event.dump().size() > wire::max_message_bytes - 64)
                throw std::length_error("Generated event too large");
              if (sample->is_final)
                terminal = std::move(event);
              else
                emit(std::move(event));
            }
          } catch (const std::exception& e) {
            // Preserve prior sequence numbers: a terminal error must not turn
            // a missing event into a successful partial answer.
            auto error = wire::envelope(session, id, "error");
            error["message"] = std::string(e.what()).substr(0, 4096);
            terminal = std::move(error);
          }
          files->release_media(id);
          {
            std::lock_guard lock(mutex);
            stream.reset();
            completed = true;
            if (!terminal) {
              terminal = wire::envelope(session, id, "error");
              (*terminal)["message"] = "Generation ended without a terminal result";
            }
            (*terminal)["sequence"] = ++next_sequence;
            events.push_back(std::move(*terminal));
          }
        }
      } catch (const std::exception& e) {
        std::lock_guard lock(mutex);
        startup_error = e.what();
      }
    });
    auto lease = Clock::now(), resend = Clock::now();
    try {
      while (!interrupted && Clock::now() - lease < std::chrono::seconds(30)) {
        auto payload = control.receive(100);
        if (payload) {
          auto j = wire::parse(*payload, session);
          const auto kind = j.at("kind").get<std::string>();
          const uint64_t id = j.at("request");
          lease = Clock::now();
          std::lock_guard lock(mutex);
          if (kind == "hello") {
            auto reply = wire::envelope(session, 0, "hello");
            if (!startup_error.empty())
              reply["error"] = startup_error;
            else if (capabilities)
              reply["capabilities"] = *capabilities;
            control.send(wire::tag(session, true), reply.dump());
          } else if (kind == "generate") {
            // A new request can carry the previous terminal acknowledgement.
            // This prevents a lost final ACK from making a ready model busy.
            if (completed && j.value("previous_request", uint64_t{0}) == active &&
                j.value("previous_sequence", uint64_t{0}) == next_sequence)
              events.clear();
            if (id == active) {
              resend = Clock::time_point{};
            } else if (id <= active || !completed || !events.empty()) {
              auto busy = wire::envelope(session, id, "rejected");
              busy["message"] = "Model busy or stale request";
              control.send(wire::tag(session, true), busy.dump());
            } else if (capabilities && startup_error.empty()) {
              active = id;
              next_sequence = 0;
              completed = false;
              cancel = false;
              command = j;
              changed.notify_all();
            }
          } else if (kind == "ack" && id == active) {
            const uint64_t seq = j.at("sequence");
            if (seq > next_sequence)
              throw std::runtime_error("Invalid event acknowledgement");
            while (!events.empty() && events.front().at("sequence").get<uint64_t>() <= seq)
              events.pop_front();
            changed.notify_all();
          } else if (kind == "cancel" && id == active) {
            cancel = true;
            if (stream)
              stream->cancel();
            changed.notify_all();
          } else if (kind == "close") {
            break;
          }
        }
        if (Clock::now() - resend >= std::chrono::milliseconds(200)) {
          std::lock_guard lock(mutex);
          for (const auto& event : events)
            control.send(wire::tag(session, true), event.dump());
          resend = Clock::now();
        }
      }
    } catch (...) {
      {
        std::lock_guard lock(mutex);
        stop = true;
        cancel = true;
        if (stream)
          stream->cancel();
      }
      changed.notify_all();
      executor.join();
      throw;
    }
    {
      std::lock_guard lock(mutex);
      stop = true;
      cancel = true;
      if (stream)
        stream->cancel();
    }
    changed.notify_all();
    executor.join();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "neat-pcie-genai-worker: " << e.what() << '\n';
    return 1;
  }
}
