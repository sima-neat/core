#include "nodes/groups/ModelGroups.h"

#include "model/Model.h"
#include "nodes/sima/Preproc.h"
#include "nodes/sima/QuantTess.h"
#include "nodes/sima/DetessDequant.h"
#include "nodes/sima/SimaBoxDecode.h"

#include <memory>
#include <array>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat::nodes::groups {
namespace {

simaai::neat::Model load_model(const std::string& tar_gz, const InferOptions& opt) {
  simaai::neat::Model::Options model_opt;

  if (opt.input_format != PreprocessColorFormat::Auto) {
    model_opt.preprocess.kind = InputKind::Image;
    model_opt.preprocess.color_convert.input_format = opt.input_format;
  }
  if (!opt.upstream_name.empty()) {
    model_opt.upstream_name = opt.upstream_name;
  }

  model_opt.preprocess.normalize.enable = opt.normalize ? AutoFlag::On : AutoFlag::Off;
  if (!opt.mean.empty()) {
    std::array<float, 3> mean = {0.0f, 0.0f, 0.0f};
    if (opt.mean.size() == 1) {
      mean = {opt.mean[0], opt.mean[0], opt.mean[0]};
    } else if (opt.mean.size() >= 3) {
      mean = {opt.mean[0], opt.mean[1], opt.mean[2]};
    }
    model_opt.preprocess.normalize.enable = AutoFlag::On;
    model_opt.preprocess.normalize.mean = mean;
    model_opt.preprocess.normalize.has_explicit_stats = true;
  }
  if (!opt.stddev.empty()) {
    std::array<float, 3> stddev = {1.0f, 1.0f, 1.0f};
    if (opt.stddev.size() == 1) {
      stddev = {opt.stddev[0], opt.stddev[0], opt.stddev[0]};
    } else if (opt.stddev.size() >= 3) {
      stddev = {opt.stddev[0], opt.stddev[1], opt.stddev[2]};
    }
    model_opt.preprocess.normalize.enable = AutoFlag::On;
    model_opt.preprocess.normalize.stddev = stddev;
    model_opt.preprocess.normalize.has_explicit_stats = true;
  }

  return simaai::neat::Model(tar_gz, model_opt);
}

} // namespace

simaai::neat::NodeGroup preprocessing(const std::string& tar_gz, const InferOptions& opt) {
  auto model = load_model(tar_gz, opt);
  return model.preprocess();
}

simaai::neat::NodeGroup simple_infer(const std::string& tar_gz) {
  simaai::neat::Model model(tar_gz, simaai::neat::Model::Options{});
  return model.inference();
}

simaai::neat::NodeGroup postprocessing(const std::string& tar_gz) {
  simaai::neat::Model model(tar_gz, simaai::neat::Model::Options{});
  return model.postprocess();
}

simaai::neat::NodeGroup infer(const std::string& tar_gz) {
  simaai::neat::Model model(tar_gz, simaai::neat::Model::Options{});
  return model.inference();
}

simaai::neat::NodeGroup infer(const std::string& tar_gz, const InferOptions& opt) {
  auto model = load_model(tar_gz, opt);
  return model.inference();
}

simaai::neat::NodeGroup Preprocess(const simaai::neat::Model& model) {
  return model.preprocess();
}

simaai::neat::NodeGroup MLA(const simaai::neat::Model& model) {
  return model.inference();
}

simaai::neat::NodeGroup Postprocess(const simaai::neat::Model& model) {
  return model.postprocess();
}

simaai::neat::NodeGroup Infer(const simaai::neat::Model& model) {
  return model.inference();
}

} // namespace simaai::neat::nodes::groups
