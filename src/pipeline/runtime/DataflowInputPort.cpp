// One public Input of a Run on the Dataflow backend (DataflowInputPort.h): its Session, the
// bind-once state, and the one path every push of the input takes (publish).
//
// Copy or wrap (W2, agent-work/h0b/plan.md 7.3, D14-D16). A push and a try_push of one tensor wrap
// its storage instead of copying it when that is safe (dataflow_wrap_layout, wrap_block): CPU
// storage (CpuOwned, CpuExternal) with an owning holder, laid out exactly as a slot of the input's
// port and exactly a slot long (every frame of a fixed-shape input, a frame at a bounded input's
// bound), aligned to the port's slot alignment (64 bytes or more), neither pulled from a Dataflow
// Run nor a Run::alloc one (D15), into a Run of one input, with RunAdvancedOptions::copy_input off.
// Session::alloc then backs the slot with that storage (the runtime's Block, W1), keeping the
// pushed Tensor's TensorBuffer alive (one count on its own control block: no allocation) until the
// last pulled tensor or queued message using it is gone; the pull returns that TensorBuffer (D14).
// Anything else is copied into the slot on the pushing thread, inside push, as before: then the
// caller's buffer is free when push returns. Either way the application must not change a pushed
// tensor's bytes while a Tensor shares its storage (D16; GStreamer's push also retains the tensor
// and copies it later on Core's input thread); copy_input = true copies at the push. EV74 and
// other device tensors (GstSample storage) are copied: every Dataflow port is Host memory until
// H0b's device pools.
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
// Output, each with that Output's depth and policy). Throws NeatError on a compile error.
struct DataflowNet {
  df::Plan plan;
  std::uint32_t feed = 0;             ///< the Feed's operator index
  std::vector<std::uint32_t> fetches; ///< the Fetches' operator indices, the input's output order
};

DataflowNet compile_dataflow_net(const DataflowRunConfig& config, std::size_t input,
                                 const DataflowInputBinding& binding, std::uint32_t stream) {
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
              " (RunOptions::queue_depth " + std::to_string(output.ingress_depth) +
              " + OutputOptions::max_buffers " + std::to_string(output.terminal_depth) +
              ") is outside the Dataflow link's range 1.." + std::to_string(df::kMaxLinkDepth));
    }
    df::Overflow overflow = df::Overflow::Block;
    if (output.terminal_drop || config.link_policy == OverflowPolicy::KeepLatest) {
      overflow = df::Overflow::DropOldest;
    } else if (config.link_policy == OverflowPolicy::DropIncoming) {
      overflow = df::Overflow::DropIncoming;
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
  df::Result<df::Plan> plan = std::move(net).compile(df::CompileOptions{
      .force_fine_clock = false, .retention_limit_bytes = 0, .failure_report = &failure});
  if (const df::Error* error = df::failure(plan)) {
    pipeline_internal::throw_dataflow_error(*error, "Graph::build", failure.mismatch);
  }
  return DataflowNet{std::get<df::Plan>(std::move(plan)), feed.index, std::move(fetches)};
}

// Sample::orig_input_seq of the messages of one input in a Run of several inputs, by the slot each
// message occupies (H0c). GStreamer's graph path, which every Run of several inputs takes there,
// numbers the pushes of the whole Run in one counter and writes it to orig_input_seq, or keeps a
// pushed orig_input_seq (or input_seq) without counting the push (RunPush.cpp:276-287); the
// runtime's Stamp has no field for it and its seq is per Feed. A Run of one input has no table:
// there orig_input_seq is input_seq (seq - 1), as before.
// The pusher writes a message's entry while it holds the slot's loan, before the publish; a puller
// reads it while it holds a lease on the message, so the slot cannot carry another message in
// between, and the publish's release and the take's acquire order the write before the read. The
// keys are atomics, so a lookup may run beside the insertion of another slot's key; a slot's key
// is inserted once (the pool's slots are fixed), and the table holds twice the pool's slots.
class SlotSeqTable {
public:
  explicit SlotSeqTable(std::uint64_t slots) {
    unsigned bits = 4;
    while ((std::uint64_t{1} << bits) < 2U * slots && bits < 40U) {
      ++bits;
    }
    shift_ = 64U - bits;
    mask_ = (std::size_t{1} << bits) - 1U;
    cells_ = std::make_unique<Cell[]>(mask_ + 1U);
  }

  void set(const void* slot, std::int64_t seq) noexcept {
    for (std::size_t i = index(slot);; i = (i + 1U) & mask_) {
      const void* key = cells_[i].key.load(std::memory_order_relaxed);
      if (key == nullptr &&
          cells_[i].key.compare_exchange_strong(key, slot, std::memory_order_relaxed)) {
        key = slot;
      }
      if (key == slot) {
        cells_[i].value.store(seq, std::memory_order_relaxed);
        return;
      }
    }
  }

  // -1 for a slot the table has never seen.
  std::int64_t get(const void* slot) const noexcept {
    for (std::size_t n = 0, i = index(slot); n <= mask_; ++n, i = (i + 1U) & mask_) {
      const void* key = cells_[i].key.load(std::memory_order_relaxed);
      if (key == slot) {
        return cells_[i].value.load(std::memory_order_relaxed);
      }
      if (key == nullptr) {
        return -1;
      }
    }
    return -1;
  }

private:
  struct Cell {
    std::atomic<const void*> key{nullptr};
    std::atomic<std::int64_t> value{-1};
  };
  std::size_t index(const void* slot) const noexcept {
    // Fibonacci hashing: the high bits of the product.
    return static_cast<std::size_t>(
        (reinterpret_cast<std::uintptr_t>(slot) * std::uint64_t{0x9E3779B97F4A7C15}) >> shift_);
  }
  std::unique_ptr<Cell[]> cells_;
  std::size_t mask_ = 0;
  unsigned shift_ = 60;
};

} // namespace

// Sample::orig_input_seq in a Run of several inputs, behind one interface (SlotSeqTable, until
// the runtime's message carries a word for it): the Run's push counter, kept per slot.
class DataflowInputPort::RunSeqs {
public:
  RunSeqs(std::uint64_t slots, std::atomic<std::uint64_t>& counter)
      : table_(slots), counter_(counter) {}
  // The orig_input_seq of the item being pushed: the pushed one, or the next value of the Run's
  // counter (one relaxed increment; the counter is shared by the inputs).
  std::int64_t next(std::int64_t pushed_seq) noexcept {
    return pushed_seq >= 0
               ? pushed_seq
               : static_cast<std::int64_t>(counter_.fetch_add(1, std::memory_order_relaxed));
  }
  void set(df::WriteLoan& loan, std::int64_t seq) noexcept {
    table_.set(loan.bytes().data(), seq);
  }
  std::int64_t get(const df::Message& message) const noexcept {
    return table_.get(message.bytes().data());
  }

private:
  SlotSeqTable table_;
  std::atomic<std::uint64_t>& counter_;
};

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
      stream_(run.streams.intern(config_.default_stream)) {
  for (const std::size_t o : config_.outputs) {
    lossy_block_ = lossy_block_ || run_.config.outputs[o].terminal_drop;
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
  if (binding) {
    std::lock_guard<std::mutex> lock(mu_);
    bind_locked(std::move(*binding));
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
  return bind_locked(std::move(*binding));
}

DataflowInputPort::Bound* DataflowInputPort::bind_locked(DataflowInputBinding binding) {
  DataflowNet net = compile_dataflow_net(run_.config, index_, binding, stream_);
  std::uint64_t slots = 0; // the Feed's pool: what the RunSeqs must hold
  for (const df::PlanPort& port : net.plan.report().ports) {
    if (port.op == net.feed) {
      slots = port.capacity;
    }
  }
  df::Result<df::Session> session = std::move(net.plan).instantiate(df::SessionOptions{});
  if (const df::Error* error = df::failure(session)) {
    pipeline_internal::throw_dataflow_error(*error, "Graph::build");
  }
  auto b = std::make_unique<Bound>(std::move(binding), std::get<df::Session>(std::move(session)),
                                   net.feed, net.fetches);
  if (run_.run_seq) {
    b->run_seqs = std::make_unique<RunSeqs>(slots, *run_.run_seq);
  }
  if (!b->binding.is_record()) {
    b->slot_bytes = df::slot_bytes(b->binding.spec);
    b->wrap_mask = df::slot_alignment(b->binding.spec) - 1U;
    b->wraps = !run_.config.run.advanced.copy_input && !b->run_seqs;
  }
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

std::int64_t DataflowInputPort::run_seq_of(const Bound& b, const df::Message& message) noexcept {
  return b.run_seqs->get(message);
}

std::int64_t DataflowInputPort::next_run_seq(Bound& b, std::int64_t pushed_seq) noexcept {
  return b.run_seqs->next(pushed_seq);
}

void DataflowInputPort::set_run_seq(Bound& b, df::WriteLoan& loan, std::int64_t seq) noexcept {
  b.run_seqs->set(loan, seq);
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
