#include "graph/nodes/StageModelExecutor.h"

#include "model/Model.h"
#include "pipeline/Tensor.h"
#include "pipeline/TensorCore.h"

#include <stdexcept>

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/core.hpp>
#endif

namespace simaai::neat::graph::nodes {
namespace {

Sample make_bbox_sample(const Sample& in, const std::vector<uint8_t>& raw) {
  auto holder = std::make_shared<std::vector<uint8_t>>(raw);

  simaai::neat::Tensor t;
  t.storage = simaai::neat::make_cpu_external_storage(holder->data(), holder->size(), holder, true);
  t.device = {simaai::neat::DeviceType::CPU, 0};
  t.read_only = true;
  t.byte_offset = 0;
  t.dtype = TensorDType::UInt8;
  t.layout = TensorLayout::HW;
  t.shape = {1, static_cast<int64_t>(holder->size())};
  t.strides_bytes = {static_cast<int64_t>(holder->size()), 1};

  Sample out;
  out.kind = SampleKind::Tensor;
  out.media_type = "application/vnd.simaai.tensor";
  out.payload_tag = "BBOX";
  out.format = "BBOX";
  out.tensor = std::move(t);
  out.frame_id = in.frame_id;
  out.stream_id = in.stream_id;
  out.pts_ns = in.pts_ns;
  out.dts_ns = in.dts_ns;
  out.duration_ns = in.duration_ns;
  out.input_seq = in.input_seq;
  out.orig_input_seq = in.orig_input_seq;
  return out;
}

} // namespace

StageModelExecutor::StageModelExecutor(StageModelExecutorOptions opt) : opt_(std::move(opt)) {
  if (!opt_.model) {
    throw std::invalid_argument("StageModelExecutor: model is null");
  }
}

void StageModelExecutor::set_ports(const StagePorts& ports) {
  const PortId only = ports.only_output();
  if (only != kInvalidPort) {
    out_port_ = only;
  }
}

void StageModelExecutor::on_input(StageMsg&& msg, std::vector<StageOutMsg>& out) {
  if (!msg.sample.tensor.has_value()) {
    throw std::invalid_argument("StageModelExecutor: sample missing simaai::neat::Tensor");
  }
  if (!opt_.model) {
    throw std::runtime_error("StageModelExecutor: model is null");
  }

  simaai::neat::Tensor current = *msg.sample.tensor;

  if (opt_.do_preproc) {
#if defined(SIMA_WITH_OPENCV)
    simaai::neat::ImageSpec::PixelFormat fmt = simaai::neat::ImageSpec::PixelFormat::BGR;
    if (current.semantic.image.has_value()) {
      fmt = current.semantic.image->format;
    }
    auto view = current.map_cv_mat_view(fmt);
    if (!view.has_value()) {
      throw std::runtime_error("StageModelExecutor: failed to map simaai::neat::Tensor to cv::Mat");
    }
    current = simaai::neat::stages::Preproc(view->mat, *opt_.model);
#else
    throw std::runtime_error("StageModelExecutor: preproc requires SIMA_WITH_OPENCV");
#endif
  }

  if (opt_.do_mla) {
    current = simaai::neat::stages::MLA(current, *opt_.model);
  }

  Sample out_sample;
  if (opt_.do_boxdecode) {
    const auto res = simaai::neat::stages::BoxDecode(current, *opt_.model, opt_.box_opt);
    out_sample = make_bbox_sample(msg.sample, res.raw);
  } else {
    out_sample.kind = SampleKind::Tensor;
    out_sample.tensor = std::move(current);
    out_sample.media_type = "application/vnd.simaai.tensor";
    out_sample.frame_id = msg.sample.frame_id;
    out_sample.stream_id = msg.sample.stream_id;
    out_sample.pts_ns = msg.sample.pts_ns;
    out_sample.dts_ns = msg.sample.dts_ns;
    out_sample.duration_ns = msg.sample.duration_ns;
    out_sample.input_seq = msg.sample.input_seq;
    out_sample.orig_input_seq = msg.sample.orig_input_seq;
  }

  const PortId out_port = (out_port_ == kInvalidPort) ? kInvalidPort : out_port_;
  out.push_back(StageOutMsg{.out_port = out_port, .sample = std::move(out_sample)});
}

std::shared_ptr<simaai::neat::graph::Node>
StageModelExecutorNode(const StageModelExecutorOptions& opt, std::string label,
                       StageNodeOptions node_opt) {
  StageNode::StageExecutorFactory factory = [opt]() {
    return std::make_unique<StageModelExecutor>(opt);
  };

  std::vector<PortDesc> inputs = {PortDesc{.name = "in", .spec = OutputSpec{}}};
  std::vector<PortDesc> outputs = {PortDesc{.name = "out", .spec = OutputSpec{}}};
  return std::make_shared<StageNode>("StageModelExecutor", std::move(factory), std::move(inputs),
                                     std::move(outputs), std::move(label),
                                     StageNode::OutputSpecFn{}, std::move(node_opt));
}

} // namespace simaai::neat::graph::nodes
