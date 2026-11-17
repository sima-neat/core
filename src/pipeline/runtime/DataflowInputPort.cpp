// One public Input of a Run on the Dataflow backend (DataflowInputPort.h): its Session, the
// bind-once state, and the one path every push of the input takes (publish).
//
// Copy or wrap (W2, agent-work/h0b/plan.md 7.3, D14-D16). A push and a try_push of one tensor wrap
// its storage instead of copying it when that is safe (dataflow_wrap_layout, wrap_block): CPU
// storage (CpuOwned, CpuExternal) with an owning holder, laid out exactly as a slot of the input's
// port and exactly a slot long (every frame of a fixed-shape input, a frame at a bounded input's
// bound), aligned to the port's slot alignment (64 bytes or more), neither pulled from a Dataflow
// Run nor a Run::alloc one (D15), with RunAdvancedOptions::copy_input off.
// Session::alloc then backs the slot with that storage (the runtime's Block, W1), keeping the
// pushed Tensor's TensorBuffer alive (one count on its own control block: no allocation) until the
// last pulled tensor or queued message using it is gone; the pull returns that TensorBuffer (D14).
// Anything else is copied into the slot on the pushing thread, inside push, as before: then the
// caller's buffer is free when push returns. Either way the application must not change a pushed
// tensor's bytes while a Tensor shares its storage (D16; GStreamer's push also retains the tensor
// and copies it later on Core's input thread); copy_input = true copies at the push.
//
// Device inputs (H0b, agent-work/h0b/plan.md 3): InputOptions::memory_policy Ev74 or Dms0 gives the
// input's port that memory kind, and its pool one DMA-BUF per slot from the runtime's Runner-backed
// source. A CPU tensor (and a cv::Mat, and anything a wrap is not safe for) is copied into the
// slot: Session::alloc opened the slot's write epoch, and the push ends it (the cache clean) before
// the message is visible. A TensorMemory::EV74 or MLA tensor of the port's kind is wrapped instead
// (dataflow_device::wrap_tensor): its DMA-BUF backs the slot. Run::alloc lends the slot's own
// mapping, its epoch open until the push or the drop. A host input copies device tensors
// (GstSample storage), as before.
//
// An item is one tensor (a leaf: a TensorSpec port) or several (a record: a RecordSchema port with
// positional field names, H0 plan 3.5). The first item binds which, as it binds the shape.
#if defined(SIMANEAT_WITH_DATAFLOW)

#include "pipeline/runtime/DataflowInputPort.h"

#include "pipeline/ErrorCodes.h"
#include "pipeline/internal/DataflowErrorMap.h"
#include "pipeline/internal/EnvUtil.h"
#include "pipeline/internal/ErrorUtil.h"
#include "pipeline/internal/InputStream.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/runtime/DataflowMlaRoute.h"
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

// The Net of input `input` of `config` for `binding`, compiled: one Feed, and one link and one
// Fetch per Output the input feeds (a fan-out when there are several: one OutPort, one link per
// Output, each with that Output's depth and policy). On a model route (H1, `mla_model`), the Feed's
// one link goes to the MLA operator and the operator's output link to the one Fetch; both links
// have the Output's depth, the first with the Run's try_push policy, the second always Block (the
// Worker waits for room: nothing an inference produced is dropped). Throws NeatError on a compile
// error.
struct DataflowNet {
  df::Plan plan;
  std::uint32_t feed = 0;             ///< the Feed's operator index
  std::vector<std::uint32_t> fetches; ///< the Fetches' operator indices, the input's output order
};

DataflowNet compile_dataflow_net(const DataflowRunConfig& config, std::size_t input,
                                 const DataflowInputBinding& binding, std::uint32_t stream,
                                 const std::shared_ptr<void>& mla_model) {
  const DataflowInputConfig& in = config.inputs[input];
  df::Net net;
  const df::OperatorRef<df::Feed> feed = net.add<df::Feed>(
      "input", binding.spec,
      df::Feed::Options{.stream = df::StreamId{stream}, .loans = in.feed_loans, .max_links = 0});
  std::vector<std::uint32_t> fetches;
  for (std::size_t k = 0; k < in.outputs.size(); ++k) {
    const DataflowOutputConfig& output = config.outputs[in.outputs[k]];
    if (output.link_depth == 0U || output.link_depth > df::kMaxLinkDepth) {
      pipeline_internal::error_util::throw_session_error(
          error_codes::kOptionOutOfRange,
          "Graph::build: the link depth " + std::to_string(output.link_depth) +
              " (the larger of RunOptions::queue_depth " + std::to_string(output.ingress_depth) +
              " and OutputOptions::max_buffers " + std::to_string(output.terminal_depth) +
              ") is outside the Dataflow link's range 1.." + std::to_string(df::kMaxLinkDepth));
    }
    df::Overflow overflow = df::Overflow::Block;
    if (output.terminal_drop || config.link_policy == OverflowPolicy::KeepLatest) {
      overflow = df::Overflow::DropOldest;
    } else if (config.link_policy == OverflowPolicy::DropIncoming) {
      overflow = df::Overflow::DropIncoming;
    }
    if (in.mla) {
      // H1. The operator's output link is connected first, so that the Session's report lists the
      // Output's link first, as for an Input that feeds its Output directly.
      df::PortSpec produced;
      const df::OperatorRef<df::Operator> mla{
          add_dataflow_mla_op(net, *in.mla, mla_model, binding.spec, produced)};
      const df::OperatorRef<df::Fetch> fetch =
          net.add<df::Fetch>("output", produced, df::Fetch::Options{.retained = 1});
      df::Status status = net.connect(mla.out(), fetch.in(),
                                      df::LinkPolicy{.depth = output.link_depth,
                                                     .overflow = df::Overflow::Block,
                                                     .stale_after = {}});
      if (!status) {
        status = net.connect(
            feed.out(), mla.in(),
            df::LinkPolicy{.depth = output.link_depth, .overflow = overflow, .stale_after = {}});
      }
      if (status) {
        pipeline_internal::throw_dataflow_error(*status, "Graph::build");
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
      pipeline_internal::throw_dataflow_error(*status, "Graph::build");
    }
    fetches.push_back(fetch.index);
  }
  df::PlanReport failure;
  df::CompileOptions options{
      .force_fine_clock = false, .retention_limit_bytes = 0, .failure_report = &failure};
#if defined(SIMANEAT_WITH_DATAFLOW_DEVICE_MEMORY)
  // Where a device port (H0b: InputOptions::memory_policy Ev74 or Dms0) takes its slots: the
  // process's Runner-backed source, which has no state and is never destroyed. A Plan of host
  // ports never reads it.
  options.blocks = &df::runner::block_source();
#endif
  df::Result<df::Plan> plan = std::move(net).compile(options);
  if (const df::Error* error = df::failure(plan)) {
    pipeline_internal::throw_dataflow_error(*error, "Graph::build", failure.mismatch);
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
  if (config_.mla) {
    // H1: the route's model, loaded at the build, where a file the MLA refuses is the build's
    // error.
    mla_model_ = load_dataflow_mla_model(*config_.mla, "Graph::build");
  }
}

// The Session is destroyed with the port, after RunCore::close ran stop (A2: a destructor after a
// stop that timed out only polls). Messages still held by pulled tensors, and slots still lent by
// alloc(), keep their pool alive on the runtime's retirement list.
DataflowInputPort::~DataflowInputPort() = default;

void DataflowInputPort::bind_at_build(std::span<const Tensor> seed) {
  std::string why;
  std::optional<DataflowInputBinding> binding = bind_dataflow_item(config_.input, seed, &why);
  if (!why.empty()) {
    throw_input_shape("Graph::build: " + why + " (Dataflow backend)");
  }
  if (binding && config_.mla && !seed.empty()) {
    // H1: a seed the model's MLA input does not take binds nothing. GStreamer builds with such a
    // seed too; the first push says what is wrong with its tensor, as there. (A declaration the
    // MLA does not take is the build's error, below.)
    DataflowInputBinding probe = *binding;
    if (!bind_dataflow_mla_input(*config_.mla, probe).empty()) {
      return;
    }
  }
  if (binding) {
    std::lock_guard<std::mutex> lock(mu_);
    bind_locked(std::move(*binding), "Graph::build");
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
    throw_input_shape(std::string(where) + ": " +
                      (why.empty() ? std::string("the input cannot be bound") : why) +
                      " (Dataflow backend)");
  }
  return bind_locked(std::move(*binding), where);
}

DataflowInputPort::Bound* DataflowInputPort::bind_locked(DataflowInputBinding binding,
                                                         const char* where) {
  if (config_.mla) {
    // H1: the input is the MLA's input, whatever the declaration allows beyond it.
    if (const std::string why = bind_dataflow_mla_input(*config_.mla, binding); !why.empty()) {
      throw_input_shape(std::string(where) + ": " + why + " (Dataflow backend)");
    }
  }
  DataflowNet net = compile_dataflow_net(run_.config, index_, binding, stream_, mla_model_);
  df::Result<df::Session> session = std::move(net.plan).instantiate(df::SessionOptions{});
  if (const df::Error* error = df::failure(session)) {
    pipeline_internal::throw_dataflow_error(*error, "Graph::build");
  }
  auto b = std::make_unique<Bound>(std::move(binding), std::get<df::Session>(std::move(session)),
                                   net.feed, net.fetches);
  if (!b->binding.is_record()) {
    b->slot_bytes = df::slot_bytes(b->binding.spec);
    b->wrap_mask = df::slot_alignment(b->binding.spec) - 1U;
    b->wraps = !run_.config.run.advanced.copy_input;
  }
  b->device = b->binding.spec.memory != df::MemoryKind::Host;
  b->any_device_memory = static_cast<bool>(config_.mla);
  if (const df::Status status = b->session.start()) {
    pipeline_internal::throw_dataflow_error(*status, "Graph::build");
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
    // Never waits behind a parked push: that push returns FR3 (A1, A2). Queued messages stay.
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
// try_push asks alloc first, without a wrap: its refusal (returned) costs what it cost before W2.
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
