// The Dataflow backend's model-route planner as a pure function (DataflowRoutePlan.h): every
// decline it can return, each executed. The manifests are what Core compiles, renders and lowers
// for real models' nodes (dataflow_model_manifest), and copies of them changed in one field each;
// the planner reads them with the Input's options, the build's seed and the EV74's state given
// here, so no device is opened, no model is loaded onto the MLA and no board state is read.
//
// Models: a YOLOv8 detection package (--model-package, or SIMA_TEST_BOXDECODE_MODEL_PACKAGE,
// SIMA_YOLO_TAR): the image preproc, the MLA stage of six heads and BoxDecode, for BGR and NV12
// frames; and a model package whose pre and post stages are canonical transforms (--resnet-package,
// or SIMA_TEST_MODEL_PACKAGE; ResNet-50 is the one it was written with), placed on the A65.
// Without the first the test is skipped; without the second its kernel-stage rows are (said in
// the output). A third, optional package is one whose MLA buffers are longer than the tensors of
// the canonical transforms beside the MLA (--padded-package, or SIMA_TEST_PADDED_MODEL_PACKAGE;
// Inception-v3 is the one the rows were written with: its MLA input is 299x299x3 elements in a
// buffer rounded up to 16 bytes); without it the row `padding` is not run (said in the output).
// It downloads nothing.
//
// Rows: `accepted` (the unchanged manifests plan), `route` (the route's shape: stage kinds, the
// stage count, buffers handed on), `mla` (the MLA stage), `frames` (what the build says of the
// frames: seed, declaration), `image` (the image preproc stage), `boxdecode`, `kernel` (a
// canonical transform), `padding` (a kernel stage's tensor in an MLA buffer that is longer than
// it: the routes that plan, and each decline of a tensor that is not such a view), `invocations`
// (the YOLOv8 package with BoxDecode off: a post stage of six dequantizations behind the MLA, on
// the A65 and on the EV74; each decline of a stage of several invocations). Each decline's text
// is checked for the words that say what is not carried.

#include "model/Model.h"
#include "model/internal/ModelInternal.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "nodes/sima/internal/ImageTransformPolicy.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/NeatError.h"
#include "pipeline/internal/DataflowNotCarried.h"
#include "pipeline/internal/sima/KernelStageRecipe.h"
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include "pipeline/runtime/DataflowRoutePlan.h"
#include "cli_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <simaai/scheduler/client/kernel.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

using namespace simaai::neat;
namespace si = pipeline_internal::sima;
namespace client = ::simaai::scheduler::client;
using runtime::dataflow_model_manifest;
using runtime::plan_dataflow_route_stages;

std::size_t g_declines = 0;
// Every row's failure, reported together at the end (one run says all that is wrong).
std::vector<std::string> g_failures;

void check(bool ok, const std::string& what) {
  if (!ok) {
    std::cout << "[FAIL] " << what << "\n";
    g_failures.push_back(what);
  }
}

// What the planner is asked: a manifest, the Input's options, the seed and the EV74's state.
struct Ask {
  si::SimaPluginStaticManifest manifest;
  InputOptions declared;
  std::vector<Tensor> seed;
  std::string ev74_obstacle;
};

// What the planner answered: nothing when the route plans; otherwise who is not carried and the
// fact about it.
pipeline_internal::DataflowNotCarried ask_planner(const Ask& ask) {
  pipeline_internal::DataflowNotCarried why;
  const auto route =
      plan_dataflow_route_stages(ask.manifest, ask.declared, ask.seed, ask.ev74_obstacle, &why);
  if (route) {
    return {};
  }
  if (!why) {
    why.subject = "(declined without a subject)";
  }
  return why;
}

// "" when the route plans; otherwise the planner's answer as the error's sentence.
std::string plan(const Ask& ask) {
  const pipeline_internal::DataflowNotCarried why = ask_planner(ask);
  return why ? pipeline_internal::dataflow_not_carried_sentence(why) : std::string();
}

// The form every answer of the planner has (DataflowNotCarried.h): a subject, a fact that is
// one clause in the present tense without a final period, no promise and no word of the old
// texts.
void check_form(const pipeline_internal::DataflowNotCarried& why, const std::string& what) {
  check(!why.subject.empty() && !why.fact.empty(), what + ": a subject and a fact");
  check(why.fact.empty() || why.fact.back() != '.', what + ": the fact ends in a period");
  const std::string text = pipeline_internal::dataflow_not_carried_sentence(why);
  for (const char* gone : {" yet", "not supported", "RunBackend", "the runtime", "on Dataflow"}) {
    check(text.find(gone) == std::string::npos, what + ": the text has '" + gone + "': " + text);
  }
}

void accepts(const Ask& ask, const std::string& what) {
  const std::string why = plan(ask);
  check(why.empty(), what + ": declined: " + why);
  if (why.empty()) {
    std::cout << "[accepted] " << what << "\n";
  }
}

void declines(const Ask& ask, const std::string& text, const std::string& what) {
  const pipeline_internal::DataflowNotCarried answer = ask_planner(ask);
  const std::string why =
      answer ? pipeline_internal::dataflow_not_carried_sentence(answer) : std::string();
  check(!why.empty(), what + ": the route planned");
  if (answer) {
    check_form(answer, what);
  }
  check(why.empty() || why.find(text) != std::string::npos,
        what + ": " + why + " (missing: " + text + ")");
  if (!why.empty() && why.find(text) != std::string::npos) {
    ++g_declines;
    std::cout << "[declined] " << what << ": " << why << "\n";
  }
}

// A copy of stage `i`'s kernel recipe that the caller changes.
si::KernelStageRecipe& recipe_of(si::SimaPluginStaticManifest& manifest, std::size_t i) {
  auto copy = std::make_shared<si::KernelStageRecipe>(*manifest.stages.at(i).native_kernel);
  si::KernelStageRecipe& recipe = *copy;
  manifest.stages.at(i).native_kernel = std::move(copy);
  return recipe;
}

si::SimaPluginStaticManifest only(const si::SimaPluginStaticManifest& manifest,
                                  std::initializer_list<std::size_t> stages) {
  si::SimaPluginStaticManifest out = manifest;
  out.stages.clear();
  for (const std::size_t i : stages) {
    out.stages.push_back(manifest.stages.at(i));
  }
  return out;
}

// Frames as the planner reads them (shape, dtype, layout, planes); CPU memory.
Tensor bgr(int width, int height, int channels = 3) {
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(width) * height * channels, 0);
  Tensor t = Tensor::from_vector(bytes, {height, width, channels}, TensorMemory::CPU);
  t.layout = TensorLayout::HWC;
  return t;
}

Tensor nv12(int width, int height) {
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(width) * height * 3U / 2U, 0);
  Tensor t =
      Tensor::from_vector(bytes, {static_cast<std::int64_t>(bytes.size())}, TensorMemory::CPU);
  t.layout = TensorLayout::HW;
  t.shape = {height, width};
  t.strides_bytes = {width, 1};
  t.semantic.image = ImageSpec{ImageSpec::PixelFormat::NV12, ""};
  Plane luma;
  luma.role = PlaneRole::Y;
  luma.shape = {height, width};
  luma.strides_bytes = {width, 1};
  Plane chroma;
  chroma.role = PlaneRole::UV;
  chroma.shape = {height / 2, width};
  chroma.strides_bytes = {width, 1};
  chroma.byte_offset = static_cast<std::int64_t>(width) * height;
  t.planes = {luma, chroma};
  return t;
}

Model::Options image_options(PreprocessColorFormat format) {
  Model::Options o;
  o.preprocess.kind = InputKind::Image;
  o.preprocess.enable = AutoFlag::On;
  o.preprocess.normalize.enable = AutoFlag::On;
  o.preprocess.color_convert.input_format = format;
  o.decode_type = BoxDecodeType::YoloV8;
  o.score_threshold = 0.49F;
  o.nms_iou_threshold = 0.5F;
  o.top_k = 100;
  return o;
}

std::shared_ptr<Node> output() {
  OutputOptions out;
  out.max_buffers = 4;
  out.sync = false;
  return nodes::Output(out);
}

// The manifest of Input -> the model's stages -> Output, as the Dataflow backend asks for it.
si::SimaPluginStaticManifest manifest_of(const Model& model, const InputOptions& in, bool pre,
                                         bool inf, bool post, const GraphOptions& options) {
  std::vector<std::shared_ptr<Node>> nodes{nodes::Input(in)};
  const auto add = [&nodes](std::vector<std::shared_ptr<Node>> more) {
    nodes.insert(nodes.end(), more.begin(), more.end());
  };
  if (pre) {
    add(internal::ModelAccess::build_public_preprocess_nodes(model));
  }
  if (inf) {
    add(internal::ModelAccess::build_public_inference_nodes(model));
  }
  if (post) {
    add(internal::ModelAccess::build_public_postprocess_nodes(model));
  }
  nodes.push_back(output());
  pipeline_internal::DataflowNotCarried why;
  std::optional<si::SimaPluginStaticManifest> manifest =
      dataflow_model_manifest(nodes, options, &why);
  require(manifest.has_value(),
          "the model's manifest: " + pipeline_internal::dataflow_not_carried_sentence(why));
  return *manifest;
}

struct Yolo {
  Ask full, pre, inf, nv12_pre;
};

// ---- rows ----------------------------------------------------------------------------------

void accepted(const Yolo& y, const Ask* resnet) {
  accepts(y.full, "YOLOv8: preprocess, inference, postprocess");
  accepts(y.pre, "YOLOv8: preprocess alone");
  accepts(y.inf, "YOLOv8: inference alone (six heads)");
  accepts(y.nv12_pre, "YOLOv8 of NV12 frames: preprocess alone");
  Ask seeded = y.full;
  seeded.seed = {bgr(1280, 720)};
  accepts(seeded, "YOLOv8 seeded with a 1280x720 frame");
  Ask declared = y.full;
  declared.declared.width = 1280;
  declared.declared.height = 720;
  accepts(declared, "YOLOv8 with a declared 1280x720 frame");
  if (resnet) {
    accepts(*resnet, "canonical transforms on the A65 around the MLA");
  }
}

void route_shape(const Yolo& y, const Ask* resnet) {
  {
    Ask a = y.full;
    a.manifest.stages.clear();
    declines(a, "the model's nodes render no stage", "no stage");
  }
  {
    Ask a = y.pre;
    a.manifest = only(y.pre.manifest, {0, 0});
    declines(a, "two kernel stages in a row", "two kernel stages in a row");
  }
  {
    Ask a = y.full;
    a.manifest.stages[0].payload_kind = si::StagePayloadKind::None;
    a.manifest.stages[0].native_kernel = nullptr;
    declines(a, "is neither an MLA stage nor a kernel stage", "a stage of no kind");
  }
  if (resnet) {
    // A canonical transform alone (Model::preprocess() between the Input and the Output) is a
    // route when the tensor it publishes is dense; one into the MLA's tiled layout is not.
    Ask a = *resnet;
    a.manifest = only(resnet->manifest, {0});
    const auto& pre = *resnet->manifest.stages[0].native_kernel;
    if (!pre.requests.empty() && !pre.requests.front().outputs.empty() &&
        std::holds_alternative<client::Contiguous>(pre.requests.front().outputs.front().layout)) {
      accepts(a, "a kernel stage alone, its output dense");
    } else {
      declines(a, "publishes a tiled or strided tensor as the route's output",
               "a kernel stage alone, its output tiled");
    }
    a.manifest = only(resnet->manifest, {1, 1});
    declines(a, "with 2 MLA stages", "two MLA stages");
    a = *resnet;
    a.manifest.stages[1].physical_inputs[0].size_bytes += 64;
    declines(a, "not one whole buffer handed on", "a stage reads other bytes than are written");
  }
}

void mla(const Yolo& y) {
  const auto mla_stage = [&y](const std::function<void(si::StageStaticSpec&)>& change) {
    Ask a = y.inf;
    change(a.manifest.stages[0]);
    return a;
  };
  declines(mla_stage([](si::StageStaticSpec& s) { s.processmla.model_path.clear(); }),
           "names no model file", "no model file");
  declines(mla_stage([](si::StageStaticSpec& s) { s.processmla.batch_size = 2; }),
           "a batched MLA model (batch 2)", "a batch of 2");
  declines(mla_stage([](si::StageStaticSpec& s) {
             s.physical_inputs.push_back(s.physical_inputs.front());
           }),
           "input buffers", "two input buffers");
  declines(mla_stage([](si::StageStaticSpec& s) {
             s.physical_outputs.push_back(s.physical_outputs.front());
           }),
           "output buffers", "two output buffers");
  declines(mla_stage([](si::StageStaticSpec& s) { s.logical_outputs.clear(); }), "(0 tensors)",
           "no published tensor");
  declines(mla_stage([](si::StageStaticSpec& s) { s.logical_inputs[0].dtype = "NO_IN_DTYPE"; }),
           "the dtype 'NO_IN_DTYPE' of its input is not a tensor dtype of Core",
           "an input dtype without a Core dtype");
  declines(mla_stage([](si::StageStaticSpec& s) { s.logical_inputs[0].byte_offset = 16; }),
           "is not one whole buffer", "an offset input");
  declines(mla_stage([](si::StageStaticSpec& s) { s.logical_outputs[1].dtype = "NO_OUT_DTYPE"; }),
           "the dtype 'NO_OUT_DTYPE' of the MLA stage's published tensor",
           "an output dtype without a Core dtype");
  declines(mla_stage([](si::StageStaticSpec& s) {
             s.logical_outputs[2].byte_offset =
                 static_cast<std::int64_t>(s.physical_outputs[0].size_bytes);
           }),
           "does not lie in its output buffer", "a head outside the buffer");
  declines(mla_stage([](si::StageStaticSpec& s) { s.logical_outputs[1].logical_index = 7; }),
           "is not published in the stage's logical order", "heads out of order");
  {
    // Six heads read by a kernel stage that is not BoxDecode's program: the stage is planned as
    // any kernel stage, which runs canonical transforms of one input and one output only (the
    // decode is one invocation of six inputs, or several invocations without recorded requests).
    Ask a = y.full;
    si::KernelStageRecipe& decode = recipe_of(a.manifest, 2);
    decode.boxdecode_program.reset();
    declines(a,
             decode.members.size() > 1U ? "not a canonical transform"
                                        : "a kernel of several inputs or outputs",
             "six heads read by another kernel");
  }
}

void frames(const Yolo& y) {
  {
    Ask a = y.pre;
    a.seed = {bgr(1280, 720), bgr(1280, 720)};
    declines(a, "the seed is several tensors", "a seed of two tensors");
  }
  {
    Ask a = y.nv12_pre;
    a.seed = {bgr(1280, 720)};
    declines(a, "the seed is not an NV12 frame", "a packed seed for NV12 frames");
  }
  {
    Ask a = y.pre;
    a.seed = {nv12(1280, 720)};
    declines(a, "the seed is a planar image", "an NV12 seed for packed frames");
  }
  {
    Ask a = y.pre;
    std::vector<float> values(16U * 16U * 3U, 0.F);
    Tensor f = Tensor::from_vector(values, {16, 16, 3}, TensorMemory::CPU);
    f.layout = TensorLayout::HWC;
    a.seed = {f};
    declines(a, "the seed is not a packed UInt8 image", "a float seed");
  }
  {
    Ask a = y.pre;
    Tensor padded = bgr(1280, 720);
    padded.strides_bytes = {1280 * 3 + 64, 3, 1};
    a.seed = {padded};
    declines(a, "the seed's rows are padded", "a seed with padded rows");
  }
  {
    Ask a = y.pre;
    a.declared.dtype = TensorDType::UInt8;
    a.declared.shape = {720, 1280};
    declines(a, "the Input's declared shape is not a UInt8 frame", "a declared shape of rank 2");
    a = y.nv12_pre;
    a.declared.dtype = TensorDType::UInt8;
    a.declared.shape = {720, 1280, 3};
    declines(a, "declares a tensor shape where the model reads NV12 frames",
             "a declared shape for NV12 frames");
    a = y.pre;
    a.declared.dtype = TensorDType::UInt8;
    a.declared.shape = {-1, -1, 3};
    a.declared.max_shape = {5};
    declines(a, "max_shape is not the bound of a frame", "a max_shape that is no frame");
  }
}

void image(const Yolo& y) {
  const auto pre_stage = [&y](const std::function<void(si::StageStaticSpec&)>& change) {
    Ask a = y.pre;
    change(a.manifest.stages[0]);
    return a;
  };
  declines(pre_stage([](si::StageStaticSpec& s) { s.processcvu.graph_name = "resize"; }),
           "is an image stage with per-frame geometry", "an image stage other than preproc");
  {
    // Behind an MLA stage of one tensor (of six, the stage behind would have to be BoxDecode).
    Ask a = y.full;
    a.manifest = only(y.full.manifest, {1, 0});
    a.manifest.stages[0].logical_outputs.resize(1);
    declines(a, "is an image stage with per-frame geometry", "the image preproc behind the MLA");
  }
  declines(pre_stage([](si::StageStaticSpec& s) {
             s.processcvu.image_transform =
                 std::make_shared<const internal::ImageTransformPolicy>();
           }),
           "is an image frontend", "an image frontend");
  declines(pre_stage([](si::StageStaticSpec& s) { s.processcvu.input_img_type = "I420"; }),
           "only packed RGB and BGR frames and NV12 frames", "I420 frames");
  declines(pre_stage([](si::StageStaticSpec& s) { s.processcvu.batch_size = 2; }),
           "has a batch of 2", "a batched image stage");
  {
    Ask a = y.pre;
    recipe_of(a.manifest, 0).constant_data.push_back(std::byte{0});
    declines(a, "has several outputs, variants, constants or private buffers",
             "an image stage with constants");
  }
  declines(pre_stage([](si::StageStaticSpec& s) { s.processcvu.resolved_exec_backend = "A65"; }),
           "the stage's image policy does not check", "an image policy that does not check");
  {
    Ask a = y.pre;
    a.declared.max_width = 640;
    a.declared.max_height = 360;
    a.seed = {bgr(1280, 720)};
    declines(a, "the Input declares as its largest", "a seed above the declared bound");
  }
  {
    Ask a = y.nv12_pre;
    a.declared.max_width = 1280;
    a.declared.max_height = 720;
    a.seed = {nv12(641, 360)};
    declines(a, "frame (the seed) is not one the stage takes", "an NV12 seed of an odd width");
  }
  {
    Ask a = y.pre;
    a.declared.width = 100000;
    a.declared.height = 100000;
    declines(a, "frame (the Input's declaration) is not one the stage takes",
             "a declared size the stage does not take");
  }
  {
    Ask a = y.pre;
    a.seed = {bgr(1280, 720, 4)};
    declines(a, "the seed has 4 channels where the model's BGR input has 3",
             "a seed of four channels");
  }
  {
    // The chroma plane's alignment the kernel asks divides every even-sized luma plane (an
    // NV12 bound is even): a bound of 1918x1078, whose luma plane is four times an odd number of
    // bytes, plans. The planner's alignment decline guards a contract the Kernels package does
    // not give today.
    Ask a = y.nv12_pre;
    a.declared.max_width = 1918;
    a.declared.max_height = 1078;
    accepts(a, "an NV12 bound of 1918x1078");
  }
  declines(pre_stage([](si::StageStaticSpec& s) { s.logical_outputs[0].dtype = "FP32"; }),
           "the published tensor's dtype 'FP32' is not the kernel's",
           "a published dtype other than the kernel's");
  declines(pre_stage([](si::StageStaticSpec& s) { s.logical_outputs[0].byte_offset = 64; }),
           "is not the kernel's whole output buffer", "a published view inside the buffer");
  declines(pre_stage([](si::StageStaticSpec& s) {
             s.logical_outputs[0].publication_relation = si::PublicationViewRelation::Identity;
           }),
           "publishes its tiled output as another view than its packed storage",
           "a tiled output published as a view");
}

void boxdecode(const Yolo& y) {
  const auto decode =
      [&y](const std::function<void(si::StageStaticSpec&, si::KernelStageRecipe&)>& change) {
        Ask a = y.full;
        si::KernelStageRecipe& recipe = recipe_of(a.manifest, 2);
        change(a.manifest.stages[2], recipe);
        return a;
      };
  {
    Ask a = y.full;
    a.manifest = only(y.full.manifest, {2});
    declines(a, "is not the route's last stage behind the model's MLA stage",
             "BoxDecode on pushed heads");
    a.manifest = only(y.full.manifest, {1, 2, 2});
    declines(a, "is not the route's last stage behind the model's MLA stage", "BoxDecode not last");
  }
  // The decode families: a (profile, task) pair runs once a device row has shown its wire
  // byte-equal to GStreamer's; the stage's shape is the same for all of them.
  for (const char* profile : {"yolo_dfl_obb_v1", "yolo26_obb_v1", "yolo26_v1", "yolox_v1"}) {
    accepts(decode([profile](si::StageStaticSpec&, si::KernelStageRecipe& r) {
              r.boxdecode_program->profile = profile;
            }),
            std::string("the decode family ") + profile + "/detection");
  }
  accepts(decode([](si::StageStaticSpec&, si::KernelStageRecipe& r) {
            r.boxdecode_program->profile = "yolo26_v1";
            r.boxdecode_program->task = "pose";
          }),
          "the decode family yolo26_v1/pose");
  accepts(decode([](si::StageStaticSpec&, si::KernelStageRecipe& r) {
            r.boxdecode_program->profile = "yolo26_v1";
            r.boxdecode_program->task = "segmentation";
          }),
          "the decode family yolo26_v1/segmentation");
  for (const char* profile :
       {"ssd_v1", "yolov5_v1", "yolov6_v1", "yolov7_v1", "detr_v1", "yolox_seg_pose_v1"}) {
    declines(decode([profile](si::StageStaticSpec&, si::KernelStageRecipe& r) {
               r.boxdecode_program->profile = profile;
             }),
             "the BoxDecode stage runs the decode families yolo_dfl_v1/detection",
             std::string("a decode family without a device row: ") + profile);
  }
  // A decode that takes each frame's affine: the oriented-box families, on frames whose
  // letterbox has one scale for both axes.
  declines(decode([](si::StageStaticSpec&, si::KernelStageRecipe& r) {
             r.boxdecode_program->complete_geometry = true;
           }),
           "a complete-geometry profile takes each frame's preprocess affine",
           "a detection decode that takes each frame's affine");
  const auto oriented = [&decode](int width, int height) {
    Ask a = decode([](si::StageStaticSpec&, si::KernelStageRecipe& r) {
      r.boxdecode_program->profile = "yolo_dfl_obb_v1";
      r.boxdecode_program->complete_geometry = r.boxdecode_program->uniform_transform = true;
    });
    a.declared.width = width;
    a.declared.height = height;
    return a;
  };
  accepts(oriented(1280, 720), "oriented boxes on 1280x720 frames (one scale)");
  declines(oriented(642, 362), "oriented boxes require equal authoritative affine scales",
           "oriented boxes on 642x362 frames (two scales)");
  for (const char* task : {"pose", "segmentation", "segmentation_pose"}) {
    declines(decode([task](si::StageStaticSpec&, si::KernelStageRecipe& r) {
               r.boxdecode_program->task = task;
             }),
             std::string("task '") + task + "': the BoxDecode stage runs the decode families",
             std::string("a task without a device row: ") + task);
  }
  // A head stated as a strided tensor of which the kernel reads neither its own extent nor its
  // rows whole (here: rows pitched far wider than the bytes the kernel reads).
  declines(decode([](si::StageStaticSpec&, si::KernelStageRecipe& r) {
             client::TensorSpec& head = r.boxdecode_program->request.inputs.front();
             const std::size_t rank = head.shape.size();
             client::Strided pitched;
             std::int64_t pitch = 1024;
             for (std::size_t axis = rank; axis-- > 0U;) {
               pitched.strides_bytes[axis] = axis + 1U == rank ? 4 : pitch;
               pitch = axis + 1U == rank ? pitch : pitch * head.shape[axis];
             }
             head.layout = pitched;
           }),
           "which is neither the tensor's own extent",
           "a strided head whose kernel reads neither of its two byte counts");
  declines(decode([](si::StageStaticSpec&, si::KernelStageRecipe& r) {
             r.boxdecode_program->write = nullptr;
           }),
           "publishes another wire than BBOX", "no wire serializer");
  declines(decode([](si::StageStaticSpec&, si::KernelStageRecipe& r) {
             r.members.push_back(r.members.front());
           }),
           "detess invocations in front of the decode", "a detess member");
  declines(decode([](si::StageStaticSpec&, si::KernelStageRecipe& r) { r.members.pop_back(); }),
           "runs 1 invocations where it runs the decode and its serializer",
           "a decode without its serializer");
  declines(decode([](si::StageStaticSpec&, si::KernelStageRecipe& r) {
             r.constant_data.push_back(std::byte{0});
           }),
           "is not one decode on the A65", "a decode with constants");
  declines(decode([](si::StageStaticSpec& s, si::KernelStageRecipe&) {
             s.physical_inputs.push_back(s.physical_inputs.front());
           }),
           "reads its heads from 2 buffers", "heads in two buffers");
  declines(decode([](si::StageStaticSpec&, si::KernelStageRecipe& r) {
             r.boxdecode_program->resize = "stretch";
           }),
           "are not ones the decode takes", "frames the decode does not take");
  {
    // Without the image preproc in front, the frames' geometry is the decode's own when it
    // takes neither each frame's size nor its resize mode.
    Ask a = y.full;
    a.manifest = only(y.full.manifest, {1, 2});
    const auto& program = *y.full.manifest.stages[2].native_kernel->boxdecode_program;
    if (program.size_from_frame || program.resize_required) {
      declines(a, "the route has no image preproc stage in front", "BoxDecode without frames");
    } else {
      accepts(a, "BoxDecode of a fixed original size without the image preproc");
    }
  }
  {
    // Another kernel's executable under the decode's request: the contract's ports differ.
    Ask a = y.full;
    const client::KernelExecutable preproc =
        y.full.manifest.stages[0].native_kernel->members[0].executable;
    recipe_of(a.manifest, 2).members[0].executable = preproc;
    declines(a, "the kernel's contract has other ports than its request",
             "a decode whose contract is not its request's");
  }
  declines(decode([](si::StageStaticSpec&, si::KernelStageRecipe& r) {
             r.members[0].inputs[0].source = client::TensorStagePortSource::Constant;
           }),
           "reads a private buffer", "a head from a private buffer");
  declines(decode([](si::StageStaticSpec& s, si::KernelStageRecipe&) {
             s.logical_inputs[0].byte_offset =
                 static_cast<std::int64_t>(s.physical_inputs[0].size_bytes);
           }),
           "does not lie in the stage's one input buffer", "a head outside the buffer");
  declines(decode([](si::StageStaticSpec& s, si::KernelStageRecipe&) {
             s.logical_inputs[1].byte_offset = s.logical_inputs[0].byte_offset;
           }),
           "shares bytes with another head", "two heads on the same bytes");
  declines(decode([](si::StageStaticSpec& s, si::KernelStageRecipe&) {
             s.logical_outputs[0].size_bytes = 1;
           }),
           "is not the wire of", "a published output that is not the wire");
}

void kernel(const Yolo& y, const Ask* resnet) {
  {
    Ask a = y.full;
    a.ev74_obstacle = "the EV74 is running with firmware 'simaai_cvu'";
    declines(a,
             "is not carried by the Dataflow backend on this board: the EV74 is running with "
             "firmware 'simaai_cvu', and the Dataflow backend runs EV74 kernels on the firmware "
             "'sima-neat-native-cvu-fw' only",
             "an EV74 kernel stage on the stock firmware");
    // The board's fact is an error of its own: its code, and a second remedy.
    const pipeline_internal::DataflowNotCarried board = ask_planner(a);
    check(board.code != nullptr && std::string(board.code) == error_codes::kDispatcherUnavailable &&
              board.remedy == pipeline_internal::DataflowRemedy::BackendOrFirmware,
          "the stock firmware: infra.dispatcher_unavailable, with the firmware as a remedy");
    check(board.subject.rfind("stage 0 '", 0) == 0 &&
              board.subject.find("(kernel preproc on the EV74)") != std::string::npos,
          "the stock firmware: the subject is the stage, its kind and target: " + board.subject);
    try {
      pipeline_internal::throw_dataflow_not_carried("Graph::build", board);
      check(false, "throw_dataflow_not_carried returned");
    } catch (const NeatError& e) {
      const std::string text = e.what();
      check(text.rfind("[infra.dispatcher_unavailable] Graph::build: stage 0 '", 0) == 0,
            "the stock firmware: the error's head: " + text);
      check(text.find("only.\nHint: To run this program on the GStreamer backend, ") !=
                    std::string::npos &&
                text.find(" Or select the Kernels firmware on the board (package "
                          "simaai-kernels-firmware) and start the program again.") !=
                    std::string::npos,
            "the stock firmware: the fact, then both remedies: " + text);
    }
  }
  if (!resnet) {
    std::cout << "[kernel] the canonical-transform rows need --resnet-package or "
                 "SIMA_TEST_MODEL_PACKAGE: not run\n";
    return;
  }
  const auto pre =
      [resnet](const std::function<void(si::StageStaticSpec&, si::KernelStageRecipe&)>& change) {
        Ask a = *resnet;
        si::KernelStageRecipe& recipe = recipe_of(a.manifest, 0);
        change(a.manifest.stages[0], recipe);
        return a;
      };
  declines(pre([](si::StageStaticSpec&, si::KernelStageRecipe& r) {
             r.members.push_back(r.members.front());
           }),
           "kernel invocations per frame", "two kernel invocations");
  declines(pre([](si::StageStaticSpec&, si::KernelStageRecipe& r) { r.requests.clear(); }),
           "has no recorded kernel request", "no recorded request");
  declines(pre([](si::StageStaticSpec&, si::KernelStageRecipe& r) {
             r.constant_data.push_back(std::byte{0});
           }),
           "has per-frame variants, constants or private buffers", "a kernel with constants");
  declines(pre([](si::StageStaticSpec&, si::KernelStageRecipe& r) {
             r.members[0].inputs.push_back(r.members[0].inputs.front());
           }),
           "a kernel of several inputs or outputs", "a kernel of two inputs");
  declines(pre([](si::StageStaticSpec&, si::KernelStageRecipe& r) {
             r.members[0].inputs[0].byte_offset = 64;
           }),
           "reads or writes inside a shared buffer", "a kernel reading an offset view");
  declines(pre([](si::StageStaticSpec& s, si::KernelStageRecipe&) {
             s.logical_outputs.push_back(s.logical_outputs.front());
           }),
           "published tensors", "a kernel stage of two published tensors");
  {
    Ask a = *resnet;
    const client::KernelExecutable decode =
        y.full.manifest.stages[2].native_kernel->members[0].executable;
    recipe_of(a.manifest, 0).members[0].executable = decode;
    declines(a, "the kernel's contract has other ports than its request",
             "a kernel whose contract is not its request's");
  }
  declines(pre([](si::StageStaticSpec&, si::KernelStageRecipe& r) {
             r.requests[0].inputs[0].dtype = client::DType::Float16;
           }),
           "a tensor's dtype has no Core dtype", "a Float16 input");
  declines(pre([](si::StageStaticSpec&, si::KernelStageRecipe& r) {
             r.requests[0].inputs[0].layout = client::Strided{};
           }),
           "reads a strided or tiled tensor from the application", "a strided first input");
  {
    Ask a = *resnet;
    si::StageStaticSpec& post = a.manifest.stages[2];
    post.logical_outputs[0].dtype = post.logical_outputs[0].dtype == "INT8" ? "FP32" : "INT8";
    declines(a, "the published tensor's dtype", "a published dtype other than the kernel's");
    a = *resnet;
    a.manifest.stages[2].logical_outputs[0].byte_offset = 1 << 30;
    declines(a, "does not lie in the kernel's output buffer", "a published output outside");
  }
}

// A post stage of several invocations: YOLOv8 with BoxDecode off, its tensors in and out (the
// model package's default post stage: one dequantization per head of the MLA's buffer).
struct Invocations {
  Ask a65, ev74;
};

void invocations(const Invocations& v) {
  const si::StageStaticSpec& post = v.a65.manifest.stages.at(2);
  const si::KernelStageRecipe& recipe = *post.native_kernel;
  const std::size_t n = recipe.members.size();
  check(n == 6U && post.logical_outputs.size() == n,
        "YOLOv8 without BoxDecode: a post stage of six invocations, six published tensors");
  accepts(v.a65, "YOLOv8 without BoxDecode: six dequantizations on the A65 behind the MLA");
  accepts(v.ev74, "YOLOv8 without BoxDecode: six dequantizations on the EV74 behind the MLA");
  const auto change =
      [&v](const std::function<void(si::StageStaticSpec&, si::KernelStageRecipe&)>& edit) {
        Ask a = v.a65;
        edit(a.manifest.stages[2], recipe_of(a.manifest, 2));
        return a;
      };
  declines(change([](si::StageStaticSpec&, si::KernelStageRecipe& r) {
             r.members[3].dependencies.push_back(1);
           }),
           "reads another invocation's output", "an invocation that depends on another");
  declines(change([](si::StageStaticSpec&, si::KernelStageRecipe& r) {
             r.members[2].inputs[0].source = client::TensorStagePortSource::Output;
           }),
           "reads another invocation's output", "an invocation reading the stage's output");
  declines(change([](si::StageStaticSpec&, si::KernelStageRecipe& r) {
             r.members[1].inputs.push_back(r.members[1].inputs.front());
           }),
           "runs kernels of one input and one output", "an invocation of two inputs");
  declines(change([](si::StageStaticSpec&, si::KernelStageRecipe& r) {
             r.members[4].outputs[0].tensor = r.members[3].outputs[0].tensor;
           }),
           "a tensor another invocation writes", "two invocations writing one tensor");
  declines(change([](si::StageStaticSpec&, si::KernelStageRecipe& r) { r.requests.pop_back(); }),
           "no recorded kernel request for every invocation", "a request missing");
  declines(change([](si::StageStaticSpec&, si::KernelStageRecipe& r) {
             r.private_buffers.push_back(si::PhysicalBufferStaticSpec{});
           }),
           "per-frame variants, constants or private buffers", "a private buffer");
  declines(
      change([](si::StageStaticSpec& s, si::KernelStageRecipe&) { s.logical_outputs.pop_back(); }),
      "each invocation writes one published tensor", "five published tensors for six");
  declines(change([](si::StageStaticSpec& s, si::KernelStageRecipe&) {
             s.physical_inputs[3].source_byte_offset = 1 << 30;
           }),
           "does not lie in the output buffer", "a head outside the MLA's buffer");
  declines(change([](si::StageStaticSpec& s, si::KernelStageRecipe&) {
             s.physical_inputs[1].source_physical_index = 1;
           }),
           "reads output buffer 1 of the MLA stage", "a head in a second MLA buffer");
  declines(change([](si::StageStaticSpec& s, si::KernelStageRecipe&) {
             const std::string& was = s.logical_outputs[2].dtype;
             s.logical_outputs[2].dtype = was == "int8" || was == "INT8" ? "float32" : "int8";
           }),
           "not its invocation's", "a published dtype other than the kernel's");
  {
    Ask a = v.a65;
    a.manifest = only(v.a65.manifest, {2});
    declines(a, "in front of the MLA stage", "a stage of several invocations first");
  }
}

// The manifests of a model whose MLA buffers are longer than the tensors beside them.
struct Padded {
  Ask full, pre, inf, pre_inf, inf_post;
};

// What the kernel of stage `i`'s one invocation asks of its input tensor's alignment.
std::size_t input_alignment(const si::SimaPluginStaticManifest& manifest, std::size_t i) {
  const auto inspected =
      client::inspect_kernel(manifest.stages.at(i).native_kernel->members.at(0).executable);
  const auto* contract = std::get_if<client::KernelInspection>(&inspected);
  return contract && !contract->input_requirements().empty()
             ? contract->input_requirements().front().alignment_bytes
             : 0U;
}

void padding(const Padded& p) {
  const si::StageStaticSpec& mla = p.full.manifest.stages[1];
  const std::uint64_t in_buffer = mla.physical_inputs.at(0).size_bytes;
  const std::uint64_t out_buffer = mla.physical_outputs.at(0).size_bytes;
  const std::uint64_t pre_bytes = p.full.manifest.stages[0].physical_outputs.at(0).size_bytes;
  const std::uint64_t post_bytes = p.full.manifest.stages[2].physical_inputs.at(0).size_bytes;
  std::cout << "[padding] the pre stage writes " << pre_bytes << " bytes, the MLA reads "
            << in_buffer << " and writes " << out_buffer << ", the post stage reads " << post_bytes
            << "\n";
  check(pre_bytes < in_buffer && (pre_bytes + 15U) / 16U * 16U == in_buffer,
        "the padded model's MLA input buffer is its pre stage's tensor rounded up to 16 bytes");
  accepts(p.full, "padded: preprocess, inference, postprocess");
  accepts(p.pre_inf, "padded: preprocess and inference (the kernel writes the buffer's front)");
  accepts(p.inf_post, "padded: inference and postprocess (a pushed tensor is the buffer's front)");
  accepts(p.inf, "padded: inference alone");
  accepts(p.pre, "padded: preprocess alone (one kernel stage, its output dense)");
  {
    Ask a = p.full;
    a.ev74_obstacle = "the EV74 is running with firmware 'simaai_cvu'";
    if (a.manifest.stages[0].native_kernel->target != client::Hardware::A65) {
      declines(a, "is not carried by the Dataflow backend on this board",
               "padded: an EV74 kernel stage on the stock firmware");
    }
  }
  // In front of the MLA: a buffer that is not the kernel's tensor rounded up to 16 bytes.
  {
    Ask a = p.full;
    a.manifest.stages[1].physical_inputs[0].size_bytes += 16;
    declines(a, "nor a tensor at the front of an MLA input buffer",
             "the MLA reads 16 bytes more than its padding");
    a = p.full;
    a.manifest.stages[1].physical_inputs[0].size_bytes = pre_bytes - 1U;
    declines(a, "not one whole buffer handed on", "the MLA reads fewer bytes than are written");
  }
  // Behind the MLA: the kernel's tensor is a view of the MLA's output buffer.
  {
    // A buffer 16 bytes longer than the manifest's: the tensor is its front, or lies 8 bytes in.
    Ask longer = p.full;
    longer.manifest.stages[1].physical_outputs[0].size_bytes += 16;
    accepts(longer, "padded: a post tensor at the front of a longer MLA buffer");
    const std::size_t alignment = input_alignment(p.full.manifest, 2);
    Ask a = longer;
    a.manifest.stages[2].physical_inputs[0].source_byte_offset = 8;
    if (alignment != 0U && 8U % alignment == 0U) {
      accepts(a, "padded: a post tensor 8 bytes into a longer MLA buffer");
    }
    a = longer;
    a.manifest.stages[2].physical_inputs[0].source_byte_offset = 64;
    declines(a, "does not lie in the output buffer of", "a post tensor that ends past the buffer");
    a = longer;
    a.manifest.stages[2].logical_inputs[0].byte_offset = -1;
    declines(a, "does not lie in the output buffer of", "a post tensor at a negative offset");
    a = longer;
    a.manifest.stages[2].physical_inputs[0].source_physical_index = 1;
    declines(a, "reads output buffer 1 of the MLA stage", "a post tensor in a second MLA buffer");
    if (alignment > 1U) {
      a = longer;
      recipe_of(a.manifest, 2).members[0].inputs[0].byte_offset = 1;
      declines(a, "asks for an alignment of " + std::to_string(alignment) + " bytes",
               "a post tensor at an offset its kernel's alignment does not divide");
    } else {
      std::cout << "[padding] the post kernel reads bytes (alignment " << alignment
                << "): the alignment decline is not run\n";
    }
  }
}

} // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);
  std::string yolo;
  sima_test::get_arg(argc, argv, "--model-package", yolo);
  for (const char* name : {"SIMA_TEST_BOXDECODE_MODEL_PACKAGE", "SIMA_YOLO_TAR"}) {
    if (const char* staged = std::getenv(name); yolo.empty() && staged) {
      yolo = staged;
    }
  }
  if (yolo.empty() || !file_exists(yolo)) {
    return skip_test("unit_dataflow_route_plan_test needs a YOLOv8 detection model package: "
                     "--model-package <X_mpk.tar.gz>, SIMA_TEST_BOXDECODE_MODEL_PACKAGE or "
                     "SIMA_YOLO_TAR");
  }
  std::string resnet_archive;
  sima_test::get_arg(argc, argv, "--resnet-package", resnet_archive);
  if (const char* staged = std::getenv("SIMA_TEST_MODEL_PACKAGE");
      resnet_archive.empty() && staged) {
    resnet_archive = staged;
  }
  std::string padded_archive;
  sima_test::get_arg(argc, argv, "--padded-package", padded_archive);
  if (const char* staged = std::getenv("SIMA_TEST_PADDED_MODEL_PACKAGE");
      padded_archive.empty() && staged) {
    padded_archive = staged;
  }
  std::string only_row;
  sima_test::get_arg(argc, argv, "--row", only_row);
  return sima_test::run_test("unit_dataflow_route_plan_test", [&] {
    const auto row = [&](const char* name, const std::function<void()>& body) {
      if (!only_row.empty() && only_row != name) {
        return;
      }
      body();
      std::cout << "[row] " << name << " ok\n";
    };
    const Model bgr_model(yolo, image_options(PreprocessColorFormat::BGR));
    const Model nv12_model(yolo, image_options(PreprocessColorFormat::NV12));
    const InputOptions frame_input{};
    const GraphOptions defaults{};
    Yolo y;
    y.full.manifest = manifest_of(bgr_model, frame_input, true, true, true, defaults);
    y.pre.manifest = manifest_of(bgr_model, frame_input, true, false, false, defaults);
    y.inf.manifest = manifest_of(bgr_model, frame_input, false, true, false, defaults);
    y.nv12_pre.manifest = manifest_of(nv12_model, frame_input, true, false, false, defaults);
    require(y.full.manifest.stages.size() == 3U && y.pre.manifest.stages.size() == 1U &&
                y.inf.manifest.stages.size() == 1U && y.nv12_pre.manifest.stages.size() == 1U,
            "the YOLOv8 manifests have the stages the rows change");
    std::optional<Ask> resnet;
    if (!resnet_archive.empty() && file_exists(resnet_archive)) {
      Model::Options tensor;
      tensor.preprocess.kind = InputKind::Tensor;
      const Model model(resnet_archive, tensor);
      GraphOptions on_a65;
      on_a65.processcvu.pre_run_target = "A65";
      on_a65.processcvu.post_run_target = "A65";
      resnet.emplace();
      resnet->declared = model.input_appsrc_options(true);
      resnet->manifest = manifest_of(model, resnet->declared, true, true, true, on_a65);
      require(resnet->manifest.stages.size() == 3U,
              "the canonical-transform model's manifest has three stages");
    }
    const Ask* const r = resnet ? &*resnet : nullptr;
    {
      // The same package with BoxDecode off and its tensors in and out: a post stage of six
      // invocations, on the A65 and on the EV74.
      Model::Options tensor;
      tensor.preprocess.kind = InputKind::Tensor;
      const Model model(yolo, tensor);
      Invocations v;
      GraphOptions post;
      for (Ask* ask : {&v.a65, &v.ev74}) {
        post.processcvu.post_run_target = ask == &v.a65 ? "A65" : "EV74";
        ask->declared = model.input_appsrc_options(true);
        ask->manifest = manifest_of(model, ask->declared, true, true, true, post);
      }
      require(v.a65.manifest.stages.size() == 3U && v.a65.manifest.stages[2].native_kernel &&
                  v.ev74.manifest.stages.size() == 3U,
              "YOLOv8 without BoxDecode: three stages, a kernel stage behind the MLA");
      row("invocations", [&] { invocations(v); });
    }
    row("accepted", [&] { accepted(y, r); });
    row("route", [&] { route_shape(y, r); });
    row("mla", [&] { mla(y); });
    row("frames", [&] { frames(y); });
    row("image", [&] { image(y); });
    row("boxdecode", [&] { boxdecode(y); });
    row("kernel", [&] { kernel(y, r); });
    if (!padded_archive.empty() && file_exists(padded_archive)) {
      Model::Options tensor;
      tensor.preprocess.kind = InputKind::Tensor;
      const Model model(padded_archive, tensor);
      const InputOptions declared = model.input_appsrc_options(true);
      Padded p;
      for (Ask* ask : {&p.full, &p.pre, &p.inf, &p.pre_inf, &p.inf_post}) {
        ask->declared = declared;
      }
      p.full.manifest = manifest_of(model, declared, true, true, true, defaults);
      p.pre.manifest = manifest_of(model, declared, true, false, false, defaults);
      p.inf.manifest = manifest_of(model, declared, false, true, false, defaults);
      p.pre_inf.manifest = manifest_of(model, declared, true, true, false, defaults);
      p.inf_post.manifest = manifest_of(model, declared, false, true, true, defaults);
      require(p.full.manifest.stages.size() == 3U && p.full.manifest.stages[0].native_kernel &&
                  p.full.manifest.stages[2].native_kernel &&
                  p.full.manifest.stages[0].native_kernel->members.size() == 1U &&
                  p.full.manifest.stages[2].native_kernel->members.size() == 1U,
              "the padded model's manifest has a canonical transform on either side of the MLA");
      row("padding", [&] { padding(p); });
    } else {
      std::cout << "[padding] the row needs --padded-package or SIMA_TEST_PADDED_MODEL_PACKAGE: "
                   "not run\n";
    }
    std::cout << "[summary] " << g_declines << " declines executed, " << g_failures.size()
              << " failures\n";
    require(g_failures.empty(), std::to_string(g_failures.size()) + " rows failed (above)");
  });
}
