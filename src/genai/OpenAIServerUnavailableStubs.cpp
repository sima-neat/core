#include "genai/OpenAIServer.h"

#include <stdexcept>
#include <utility>

namespace simaai::neat::genai {
namespace {

[[noreturn]] void throw_openai_server_unavailable() {
  throw std::runtime_error(
      "NEAT OpenAI-compatible GenAI server is not available in this build. Reconfigure with "
      "SIMANEAT_ENABLE_OPENAI_SERVER=ON and a valid SimaLMM/cpp-httplib setup to use "
      "genai::OpenAIServer.");
}

} // namespace

struct OpenAIServer::Impl {};

OpenAIServer::OpenAIServer(OpenAIServerOptions) {
  throw_openai_server_unavailable();
}

OpenAIServer::~OpenAIServer() = default;
OpenAIServer::OpenAIServer(OpenAIServer&&) noexcept = default;
OpenAIServer& OpenAIServer::operator=(OpenAIServer&&) noexcept = default;

std::string OpenAIServer::add_model(std::filesystem::path) {
  throw_openai_server_unavailable();
}

std::string OpenAIServer::add_model(std::filesystem::path, std::string) {
  throw_openai_server_unavailable();
}

void OpenAIServer::add_model(std::string, std::shared_ptr<GenAIModel>) {
  throw_openai_server_unavailable();
}

bool OpenAIServer::remove_model(const std::string&) {
  throw_openai_server_unavailable();
}

std::vector<std::string> OpenAIServer::model_names() const {
  throw_openai_server_unavailable();
}

void OpenAIServer::serve() {
  throw_openai_server_unavailable();
}

void OpenAIServer::start() {
  throw_openai_server_unavailable();
}

void OpenAIServer::stop() {
  throw_openai_server_unavailable();
}

} // namespace simaai::neat::genai
