// One public Input of a Run on the Dataflow backend (DataflowInputPort.h): its Session, the
// bind-once state, and the one path every push of the input takes (publish).
//
// Copy or wrap. A push and a try_push of one tensor wrap its storage instead of copying it when
// that is safe (dataflow_wrap_layout, wrap_block): CPU storage (CpuOwned, CpuExternal) with an
// owning holder, laid out exactly as a slot of the input's port and exactly a slot long (every
// frame of a fixed-shape input, a frame at a bounded input's bound), aligned to the port's slot
// alignment (64 bytes or more), neither a lease pulled from a Dataflow Run nor a Run::alloc tensor
// (a wrap would pin a slot of a pool), with RunAdvancedOptions::copy_input off.
// Session::alloc then backs the slot with that storage (the runtime's Block), keeping the pushed
// Tensor's TensorBuffer alive (one count on its own control block: no allocation) until the last
// pulled tensor or queued message using it is gone; the pull returns that TensorBuffer. Anything
// else is copied into the slot on the pushing thread, inside push: then the caller's buffer is
// free when push returns. Either way the application must not change a pushed tensor's bytes
// while a Tensor shares its storage (GStreamer's push also retains the tensor and copies it later
// on Core's input thread); copy_input = true copies at the push.
//
// Device inputs: InputOptions::memory_policy Ev74 or Dms0 gives the input's port that memory kind,
// and its pool one DMA-BUF per slot from the runtime's Runner-backed source. A CPU tensor (and a
// cv::Mat, and anything a wrap is not safe for) is copied into the slot: Session::alloc opened the
// slot's write epoch, and the push ends it (the cache clean) before the message is visible. A
// TensorMemory::EV74 or MLA tensor of the port's kind is wrapped instead
// (dataflow_device::wrap_tensor): its DMA-BUF backs the slot. Run::alloc lends the slot's own
// mapping, its epoch open until the push or the drop. A host input copies device tensors.
//
// An item is one tensor (a leaf: a TensorSpec port) or several (a record: a RecordSchema port with
// positional field names). The first item binds which, as it binds the shape.
#if defined(SIMANEAT_WITH_DATAFLOW)

#include "pipeline/runtime/DataflowInputPort.h"

#include "pipeline/ErrorCodes.h"
#include "pipeline/internal/DataflowErrorMap.h"
#include "pipeline/internal/DataflowNotCarried.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/ErrorUtil.h"
#include "pipeline/internal/InputStream.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/runtime/DataflowModelRoute.h"
#include "pipeline/runtime/DataflowTensorCodec.h"
#include "pipeline/runtime/RunAllocation.h"
#include "pipeline/runtime/RunCore.h"

#include <array>
#include <stdexcept>
#include <utility>
#include <variant>

namespace simaai::neat::runtime {

namespace {

namespace error_util = pipeline_internal::error_util;
using dataflow_codec::throw_input_shape;

// Where an input's Plan is compiled and its Session started, which decides what an error there
// tells the application to do (DataflowErrorSite): at Graph::build (a seed or a declared shape
// gives the spec), or at the input's first push, where the Run exists already.
struct BindSite {
  const char* where;
  pipeline_internal::DataflowErrorSite site;
};

// The Net of input `input` of `config` for `binding`, compiled: one Feed, and one link and one
// Fetch per Output the input feeds (a fan-out when there are several: one OutPort, one link per
// Output, each with that Output's depth and policy). On a model route (`loaded`: its model and
// kernels), the Feed's one link goes to the first stage's operator, each operator's output link
// to the next one, and the last one's to the one Fetch. The Feed's link and the Fetch's have the
// Output's depth (within the runtime's range: the build declines a deeper one), the first with
// the Run's try_push policy; the links between operators hold one message; every link behind the
// Feed's is Block (the Worker waits for room: nothing a stage produced is dropped). Throws
// NeatError on a compile error.
struct DataflowNet {
  df::Plan plan;
  std::uint32_t feed = 0;             ///< the Feed's operator index
  std::vector<std::uint32_t> fetches; ///< the Fetches' operator indices, the input's output order
};

DataflowNet compile_dataflow_net(const DataflowRunConfig& config, std::size_t input,
                                 const DataflowInputBinding& binding, std::uint32_t stream,
                                 const std::shared_ptr<void>& loaded, const BindSite& site) {
  const DataflowInputConfig& in = config.inputs[input];
  df::Net net;
  const df::OperatorRef<df::Feed> feed = net.add<df::Feed>(
      "input", binding.spec,
      df::Feed::Options{.stream = df::StreamId{stream}, .loans = in.feed_loans, .max_links = 0});
  std::vector<std::uint32_t> fetches;
  for (std::size_t k = 0; k < in.outputs.size(); ++k) {
    const DataflowOutputConfig& output = config.outputs[in.outputs[k]];
    df::Overflow overflow = df::Overflow::Block;
    if (output.terminal_drop || config.link_policy == OverflowPolicy::KeepLatest) {
      overflow = df::Overflow::DropOldest;
    } else if (config.link_policy == OverflowPolicy::DropIncoming) {
      overflow = df::Overflow::DropIncoming;
    }
    if (in.model) {
      // The last operator's output link is connected first, so that the Session's report lists
      // the Output's link first, as for an Input that feeds its Output directly.
      df::PortSpec produced;
      const std::vector<std::uint32_t> ops =
          add_dataflow_model_ops(net, *in.model, loaded, binding.spec, produced);
      const auto op = [&ops](std::size_t i) { return df::OperatorRef<df::Operator>{ops[i]}; };
      const df::OperatorRef<df::Fetch> fetch =
          net.add<df::Fetch>("output", produced, df::Fetch::Options{.retained = 1});
      df::Status status = net.connect(op(ops.size() - 1U).out(), fetch.in(),
                                      df::LinkPolicy{.depth = output.link_depth,
                                                     .overflow = df::Overflow::Block,
                                                     .stale_after = {}});
      for (std::size_t i = ops.size() - 1U; !status && i > 0U; --i) {
        status = net.connect(
            op(i - 1U).out(), op(i).in(),
            df::LinkPolicy{.depth = 1, .overflow = df::Overflow::Block, .stale_after = {}});
      }
      if (!status) {
        status = net.connect(
            feed.out(), op(0).in(),
            df::LinkPolicy{.depth = output.link_depth, .overflow = overflow, .stale_after = {}});
      }
      if (status) {
        pipeline_internal::throw_dataflow_error(*status, site.where, {}, site.site);
      }
      fetches.push_back(fetch.index);
      continue;
    }
    // Operator names are the Plan's own (Plan::fetch(name)); Core addresses its Fetches by index.
    const df::OperatorRef<df::Fetch> fetch =
        net.add<df::Fetch>(k == 0U ? std::string("output") : "output" + std::to_string(k),
                           binding.spec, df::Fetch::Options{.retained = 1});
    if (const df::Status status = net.connect(
            feed.out(), fetch.in(),
            df::LinkPolicy{.depth = output.link_depth, .overflow = overflow, .stale_after = {}})) {
      pipeline_internal::throw_dataflow_error(*status, site.where, {}, site.site);
    }
    fetches.push_back(fetch.index);
  }
  df::PlanReport failure;
  df::CompileOptions options{
      .force_fine_clock = false, .retention_limit_bytes = 0, .failure_report = &failure};
#if defined(SIMANEAT_WITH_DATAFLOW_DEVICE_MEMORY)
  // Where a device port (InputOptions::memory_policy Ev74 or Dms0) takes its slots: the process's
  // Runner-backed source, which has no state and is never destroyed. A Plan of host ports never
  // reads it.
  options.blocks = &df::runner::block_source();
#endif
  df::Result<df::Plan> plan = std::move(net).compile(options);
  if (const df::Error* error = df::failure(plan)) {
    pipeline_internal::throw_dataflow_error(*error, site.where, failure.mismatch, site.site);
  }
  return DataflowNet{std::get<df::Plan>(std::move(plan)), feed.index, std::move(fetches)};
}

} // namespace

DataflowInputPort::Bound::Bound(DataflowInputBinding b, df::Session s, std::uint32_t feed_op,
                                const std::vector<std::uint32_t>& fetch_ops)
    : binding(std::move(b)), session(std::move(s)), feed(df::OperatorRef<df::Feed>{feed_op}) {
  fetches.reserve(fetch_ops.size());
  for (const std::uint32_t op : fetch_ops) {
    fetches.emplace_back(df::OperatorRef<df::Fetch>{op});
  }
}

DataflowInputPort::Bound::~Bound() = default;

DataflowInputPort::DataflowInputPort(const DataflowPortContext& run, std::size_t index)
    : run_(run), index_(index), config_(run.config.inputs[index]),
      stream_(run.streams.intern(config_.default_stream)),
      device_(config_.input.memory_policy == InputMemoryPolicy::Ev74 ||
              config_.input.memory_policy == InputMemoryPolicy::Dms0) {
  for (const std::size_t o : config_.outputs) {
    lossy_block_ = lossy_block_ || run_.config.outputs[o].terminal_drop;
  }
  if (config_.model) {
    // The route's model and kernels, loaded and checked at the build, where a file the MLA
    // refuses or a request the Kernels package refuses is the build's error.
    model_ = load_dataflow_model_route(*config_.model, run_.config.where.c_str());
    publish_times_ = std::make_unique<DataflowPublishTimes>();
    feed_depth_ = run_.config.outputs[config_.outputs.front()].link_depth;
  }
}

// The Session is destroyed with the port, after RunCore::close ran stop (a Session's destructor
// after a stop that timed out only polls). Messages still held by pulled tensors, and slots still
// lent by alloc(), keep their pool alive on the runtime's retirement list.
DataflowInputPort::~DataflowInputPort() = default;

void DataflowInputPort::bind_at_build(std::span<const Tensor> seed) {
  std::string why;
  Tensor view;
  if (config_.model && seed.size() == 1U &&
      &dataflow_model_input_view(*config_.model, seed.front(), view) == &view) {
    // A seed without the leading unit axes of the first stage's tensor binds as that tensor.
    seed = std::span<const Tensor>(&view, 1);
  }
  std::optional<DataflowInputBinding> binding = bind_dataflow_item(config_.input, seed, &why);
  if (!binding && why.empty() && seed.empty() && config_.model) {
    // A model route binds at the build from what its first stage takes, so that the model, its
    // kernels and the pools are the build's (a refusal is the build's error, never a push's).
    binding =
        bind_dataflow_item(dataflow_model_input_contract(*config_.model, config_.input), {}, &why);
  }
  if (!why.empty()) {
    pipeline_internal::throw_dataflow_not_carried(
        run_.config.where, {error_codes::kInputShape,
                            "Input " + dataflow_names_text(config_.names) +
                                (seed.empty() ? " as declared" : " with this seed"),
                            why});
  }
  if (binding && config_.model && !seed.empty()) {
    // A seed the model's first stage does not take binds nothing here: the build's preflight
    // pushes it and its push says what is wrong with its tensor, as GStreamer's preflight does
    // (without the preflight the first push says it). A declaration the model does not take is
    // the build's error, below.
    DataflowInputBinding probe = *binding;
    if (!bind_dataflow_model_input(*config_.model, probe).empty()) {
      return;
    }
  }
  if (binding) {
    std::lock_guard<std::mutex> lock(mu_);
    bind_locked(std::move(*binding), run_.config.where.c_str(), /*at_build=*/true);
  }
}

// The first push into the input binds it: its tensors' spec, under the declaration (WORKER.md
// 18.1). nullptr when the input closed or the Run stopped first.
DataflowInputPort::Bound* DataflowInputPort::bind_first(std::span<const Tensor> item,
                                                        const char* where) {
  std::lock_guard<std::mutex> lock(mu_);
  if (Bound* b = bound()) {
    return b;
  }
  if (run_.stopped.load(std::memory_order_relaxed) || closed_.load(std::memory_order_relaxed)) {
    return nullptr;
  }
  std::string why;
  std::optional<DataflowInputBinding> binding = bind_dataflow_item(config_.input, item, &why);
  if (!binding) {
    pipeline_internal::throw_dataflow_not_carried(
        where, {error_codes::kInputShape,
                "the first item pushed into Input " + dataflow_names_text(config_.names),
                why.empty() ? std::string("it binds no spec") : why});
  }
  return bind_locked(std::move(*binding), where, /*at_build=*/false);
}

DataflowInputPort::Bound* DataflowInputPort::bind_locked(DataflowInputBinding binding,
                                                         const char* where, bool at_build) {
  const BindSite site{where, at_build ? pipeline_internal::DataflowErrorSite::Build
                                      : pipeline_internal::DataflowErrorSite::FirstPush};
  if (config_.model) {
    // The input is the first stage's input, whatever the declaration allows beyond it.
    if (const std::string why = bind_dataflow_model_input(*config_.model, binding); !why.empty()) {
      throw_input_shape(std::string(where) + ": " + why + " (Dataflow backend)");
    }
  }
  DataflowNet net = compile_dataflow_net(run_.config, index_, binding, stream_, model_, site);
  df::PlanReport plan = net.plan.report();
  df::Result<df::Session> session = std::move(net.plan).instantiate(df::SessionOptions{});
  if (const df::Error* error = df::failure(session)) {
    pipeline_internal::throw_dataflow_error(*error, site.where, {}, site.site);
  }
  auto b = std::make_unique<Bound>(std::move(binding), std::get<df::Session>(std::move(session)),
                                   net.feed, net.fetches);
  b->plan = std::move(plan);
  if (!b->binding.is_record()) {
    b->slot_bytes = df::slot_bytes(b->binding.spec);
    b->wrap_mask = df::slot_alignment(b->binding.spec) - 1U;
    b->wraps = !run_.config.run.advanced.copy_input;
  }
  b->device = b->binding.spec.memory != df::MemoryKind::Host;
  // In front of a kernel on the EV74 a wrap is a tensor of the port's own kind (linux,cma)
  // by its descriptor alone; a tensor of the other device heap is copied into a slot.
  const bool ev74_first = config_.model && config_.model->ev74_input;
  b->any_device_memory = config_.model && !ev74_first;
  // An A65 kernel reads a wrap through the CPU's address; in front of an image preproc whose
  // frames change size the port is bounded, and the runtime takes a wrap into a bounded port only
  // with that address. Every push of such a route takes wrap_frame, of the other wrap_tensor.
  if (config_.model && ((config_.model->kernel_input && !ev74_first) ||
                        (config_.model->frames && !config_.model->planar_input))) {
    b->mapped_wraps = dataflow_device::make_mapped_wraps();
  }
  if (const df::Status status = b->session.start()) {
    pipeline_internal::throw_dataflow_error(*status, site.where, {}, site.site);
  }
  bound_storage_ = std::move(b);
  bound_.store(bound_storage_.get(), std::memory_order_release);
  cv_.notify_all();
  return bound_storage_.get();
}

DataflowInputPort::Bound* DataflowInputPort::wait_bound_slow(df::Deadline deadline) {
  std::unique_lock<std::mutex> lock(mu_);
  const auto ready = [&] {
    return bound() != nullptr || run_.stopped.load(std::memory_order_relaxed) ||
           closed_.load(std::memory_order_relaxed);
  };
  if (deadline == df::kNoDeadline) {
    cv_.wait(lock, ready);
  } else {
    cv_.wait_until(lock, deadline, ready);
  }
  return bound();
}

void DataflowInputPort::mark_closed() {
  {
    std::lock_guard<std::mutex> lock(mu_);
    closed_.store(true, std::memory_order_release);
  }
  cv_.notify_all();
}

void DataflowInputPort::close_feed() {
  if (Bound* b = bound()) {
    // Never waits behind a parked push: that push returns FR3 (Session::close_input). Queued
    // messages stay.
    (void)b->session.close_input(b->feed, df::kNoDeadline);
  }
}

void DataflowInputPort::wake() {
  // The Run's flag is set outside mu_: taking it once orders this notify after any waiter that
  // read the flag unset.
  { std::lock_guard<std::mutex> lock(mu_); }
  cv_.notify_all();
}

df::Status DataflowInputPort::stop(df::Deadline deadline) {
  Bound* b = bound();
  return b ? b->session.stop(deadline) : df::Status{};
}

std::optional<df::SessionReport> DataflowInputPort::report() const {
  Bound* b = bound();
  return b ? std::optional<df::SessionReport>(b->session.report()) : std::nullopt;
}

std::string DataflowInputPort::report_text() const {
  std::string text;
  if (Bound* b = bound()) {
    b->session.report().to_text(text, &b->plan);
  }
  return text;
}

void DataflowInputPort::enforce_transferable(const Sample& sample, const char* where) {
  if (pipeline_internal::env_bool("SIMA_ALLOW_CROSS_RUN_GSTSAMPLE_PUSH", false) ||
      !pipeline_internal::sample_has_device_gstsample_producer_lifetime(sample, false)) {
    return;
  }
  std::string reason;
  if (!pipeline_internal::sample_has_transferable_zero_copy_loan(sample, &reason)) {
    throw std::runtime_error(pipeline_internal::cross_run_zero_copy_sample_error(where) +
                             (reason.empty() ? std::string{} : " Reason: " + reason + "."));
  }
}

Tensor DataflowInputPort::alloc(const std::string& at) {
  const std::vector<std::int64_t> shape = declared_alloc_shape(config_.input, at);
  const auto closed_error = [&at] {
    return std::runtime_error(error_util::decorate_error(
        error_codes::kRuntimePull, at + ": the input is closed or the Run is stopped"));
  };
  Bound* b = bound();
  if (!b || run_.stopped.load(std::memory_order_acquire) || closed()) {
    // A declared shape binds at build, so no binding means a closed input or a stopped Run.
    throw closed_error();
  }
  df::Result<df::WriteLoan> allocated = b->session.alloc(b->feed, df::Acquire::Block);
  if (const df::Error* error = df::failure(allocated)) {
    if (error->code == df::ErrorCode::Closed) {
      throw closed_error();
    }
    pipeline_internal::throw_dataflow_error(*error, at);
  }
  auto loan = std::make_shared<SlotLoan>();
  loan->loan = std::get<df::WriteLoan>(std::move(allocated));
  const std::span<std::byte> bytes = loan->loan.bytes();
  auto allocation = std::make_shared<RunAllocation>();
  allocation->run = run_.core.weak_from_this();
  allocation->data = bytes.data();
  allocation->size = bytes.size();
  allocation->backing = std::move(loan);
  return make_allocated_tensor(allocation, b->binding.dtype, shape);
}

// A try_push that would wrap pays the wrap decision (the storage's type query, the holder's copy
// and release) before alloc can refuse it. So once FR14 has refused one (every link full:
// Bound::full_at), and while no pull has returned since, the links are still full, and the next
// try_push asks alloc first, without a wrap: its refusal (returned) costs what a copying
// try_push's refusal costs.
// If alloc finds room after all (a pull is between its take and its count), the slot goes back at
// once, before the push takes its own. Out of line: the blocking push never comes here.
[[gnu::noinline]] df::Status DataflowInputPort::still_full(Bound& b) {
  const df::Result<df::WriteLoan> room = b.session.alloc(b.feed, df::Acquire::Fail);
  if (const df::Error* error = df::failure(room)) {
    return *error;
  }
  b.full_at.store(Bound::kNoRefusal, std::memory_order_relaxed);
  return std::nullopt;
}

} // namespace simaai::neat::runtime

#endif // SIMANEAT_WITH_DATAFLOW
