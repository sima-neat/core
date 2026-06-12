#include "genai/ASRModel.h"
#include "genai/GenAIModel.h"
#include "genai/GenAIServer.h"
#include "genai/GraphFragments.h"
#include "genai/VisionLanguageModel.h"
#include "genai/internal/SpeechTranscriberNodeFactory.h"
#include "genai/internal/VisionLanguageNodeFactory.h"

#include <stdexcept>
#include <utility>

namespace simaai::neat::genai {
namespace {

[[noreturn]] void throw_genai_unavailable() {
  throw std::runtime_error(
      "NEAT GenAI/LLiMa support is not available in this build. Reconfigure with "
      "SIMANEAT_REQUIRE_LLIMA_ARTIFACTS=ON and a valid SimaLMM package to use GenAI models or "
      "GenAI graph fragments or the GenAI server.");
}

} // namespace

struct GenAIModel::Impl {};
struct GenAIServer::Impl {};

GenAIModel::GenAIModel(std::filesystem::path) {
  throw_genai_unavailable();
}
GenAIModel::~GenAIModel() = default;
GenAIModel::GenAIModel(GenAIModel&&) noexcept = default;
GenAIModel& GenAIModel::operator=(GenAIModel&&) noexcept = default;
GenAITask GenAIModel::task() const {
  throw_genai_unavailable();
}
bool GenAIModel::accepts_text() const {
  throw_genai_unavailable();
}
bool GenAIModel::accepts_image() const {
  throw_genai_unavailable();
}
bool GenAIModel::accepts_audio() const {
  throw_genai_unavailable();
}
std::string GenAIModel::model_id() const {
  throw_genai_unavailable();
}
GenerationResult GenAIModel::run(const GenerationRequest&) {
  throw_genai_unavailable();
}
GenerationStream GenAIModel::stream(const GenerationRequest&) {
  throw_genai_unavailable();
}

GenAIServer::GenAIServer(GenAIServerOptions) {
  throw_genai_unavailable();
}
GenAIServer::~GenAIServer() = default;
GenAIServer::GenAIServer(GenAIServer&&) noexcept = default;
GenAIServer& GenAIServer::operator=(GenAIServer&&) noexcept = default;
std::string GenAIServer::add_model(std::filesystem::path) {
  throw_genai_unavailable();
}
std::string GenAIServer::add_model(std::filesystem::path, std::string) {
  throw_genai_unavailable();
}
void GenAIServer::add_model(std::string, std::shared_ptr<GenAIModel>) {
  throw_genai_unavailable();
}
bool GenAIServer::remove_model(const std::string&) {
  throw_genai_unavailable();
}
std::vector<std::string> GenAIServer::model_names() const {
  throw_genai_unavailable();
}
void GenAIServer::serve() {
  throw_genai_unavailable();
}
void GenAIServer::start() {
  throw_genai_unavailable();
}
void GenAIServer::stop() {
  throw_genai_unavailable();
}

ASRModel::ASRModel(std::filesystem::path) {
  throw_genai_unavailable();
}
ASRModel::~ASRModel() = default;
ASRModel::ASRModel(ASRModel&&) noexcept = default;
ASRModel& ASRModel::operator=(ASRModel&&) noexcept = default;
bool ASRModel::accepts_audio() const {
  throw_genai_unavailable();
}
std::string ASRModel::model_id() const {
  throw_genai_unavailable();
}
GenerationResult ASRModel::run(const GenerationRequest&) {
  throw_genai_unavailable();
}
GenerationStream ASRModel::stream(const GenerationRequest&) {
  throw_genai_unavailable();
}

VisionLanguageModel::VisionLanguageModel(std::filesystem::path) {
  throw_genai_unavailable();
}
VisionLanguageModel::~VisionLanguageModel() = default;
VisionLanguageModel::VisionLanguageModel(VisionLanguageModel&&) noexcept = default;
VisionLanguageModel& VisionLanguageModel::operator=(VisionLanguageModel&&) noexcept = default;
bool VisionLanguageModel::accepts_image() const {
  throw_genai_unavailable();
}
std::string VisionLanguageModel::model_id() const {
  throw_genai_unavailable();
}
std::size_t VisionLanguageModel::cached_image_count() const {
  throw_genai_unavailable();
}
bool VisionLanguageModel::encode(const Tensor&) {
  throw_genai_unavailable();
}
bool VisionLanguageModel::encode(const std::vector<Tensor>&) {
  throw_genai_unavailable();
}
#if defined(SIMA_WITH_OPENCV)
bool VisionLanguageModel::encode(const cv::Mat&) {
  throw_genai_unavailable();
}
bool VisionLanguageModel::encode(const std::vector<cv::Mat>&) {
  throw_genai_unavailable();
}
#endif
GenerationResult VisionLanguageModel::run(const GenerationRequest&) {
  throw_genai_unavailable();
}
GenerationStream VisionLanguageModel::stream(const GenerationRequest&) {
  throw_genai_unavailable();
}

namespace nodes {

std::shared_ptr<graph::Node> VisionLanguage(std::shared_ptr<VisionLanguageModel>,
                                            VisionLanguageOptions, std::string) {
  throw_genai_unavailable();
}

std::shared_ptr<graph::Node> SpeechTranscriber(std::shared_ptr<ASRModel>, SpeechTranscriberOptions,
                                               std::string) {
  throw_genai_unavailable();
}

} // namespace nodes

namespace graphs {

Graph VisionLanguage(std::shared_ptr<VisionLanguageModel> model, VisionLanguageOptions,
                     std::string) {
  if (!model) {
    throw std::invalid_argument("genai::graphs::VisionLanguage requires a non-null model");
  }
  throw_genai_unavailable();
}

Graph SpeechTranscriber(std::shared_ptr<ASRModel> model, SpeechTranscriberOptions, std::string) {
  if (!model) {
    throw std::invalid_argument("genai::graphs::SpeechTranscriber requires a non-null model");
  }
  throw_genai_unavailable();
}

} // namespace graphs
} // namespace simaai::neat::genai
