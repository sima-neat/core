/**
 * @file
 * @ingroup graph
 * @brief Actor-like stage executor interface — the runtime-graph extension point.
 *
 * `StageExecutor` is to the runtime graph what `Node` is to the builder graph: the
 * abstract base every stage subclasses. The runtime drives stages by calling `on_input()`
 * for each arriving sample and `on_tick()` periodically; the stage emits zero or more
 * `StageOutMsg`s back to the runtime, which routes them on outgoing edges.
 *
 * @see Graph, GraphSession, GraphRun
 * @see "Runtime graph stages" (§73 / §84 of the design deep dive)
 */
#pragma once

#include "graph/GraphTypes.h"
#include "pipeline/SessionOptions.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace simaai::neat::graph {

/**
 * @brief A single input message handed to a `StageExecutor`.
 *
 * Carries the sample plus the input port it arrived on, so multi-input stages can
 * dispatch on `in_port`.
 * @ingroup graph
 */
struct StageMsg {
  PortId in_port = kInvalidPort; ///< Input port id this sample arrived on.
  Sample sample;                 ///< The sample payload.
};

/**
 * @brief A single output message produced by a `StageExecutor`.
 *
 * The runtime routes this to all edges originating at `out_port` on this stage.
 * @ingroup graph
 */
struct StageOutMsg {
  PortId out_port = kInvalidPort; ///< Output port id to emit on.
  Sample sample;                  ///< The sample payload.
};

/**
 * @brief Runtime-owned output handle for stages that stream while `on_input()` is active.
 *
 * Most stages can keep appending to the `out` vector passed to `on_input()`. Long-running
 * stages may use this emitter to route samples immediately, before `on_input()` returns.
 *
 * The pointer supplied through `StageExecutor::set_emitter()` is owned by the runtime and
 * remains valid until the executor is stopped.
 * @ingroup graph
 */
class StageEmitter {
public:
  virtual ~StageEmitter() = default;

  /// Route one output sample through the same graph paths used for returned `StageOutMsg`s.
  virtual bool emit(StageOutMsg msg) = 0;
  /// True once the graph runtime has begun stopping.
  virtual bool stop_requested() const = 0;
};

/**
 * @brief Resolved port-id table for a stage — populated by the runtime before `start()`.
 *
 * Stage authors should cache the relevant port-ids in `set_ports()` rather than looking
 * them up by name on every `on_input()` call.
 * @ingroup graph
 */
struct StagePorts {
  std::unordered_map<std::string, PortId> in;  ///< Input ports keyed by name.
  std::unordered_map<std::string, PortId> out; ///< Output ports keyed by name.

  /// Look up the id of an input port by name; returns `kInvalidPort` if missing.
  PortId in_port(const std::string& name) const {
    auto it = in.find(name);
    return it == in.end() ? kInvalidPort : it->second;
  }

  /// Look up the id of an output port by name; returns `kInvalidPort` if missing.
  PortId out_port(const std::string& name) const {
    auto it = out.find(name);
    return it == out.end() ? kInvalidPort : it->second;
  }

  /// Convenience: returns the sole input port if there's exactly one, else `kInvalidPort`.
  PortId only_input() const {
    return in.size() == 1 ? in.begin()->second : kInvalidPort;
  }

  /// Convenience: returns the sole output port if there's exactly one, else `kInvalidPort`.
  PortId only_output() const {
    return out.size() == 1 ? out.begin()->second : kInvalidPort;
  }
};

/**
 * @brief Actor-style executor base class — implement to add a new runtime-graph stage.
 *
 * Subclass this to define a stage's behavior. The runtime owns the lifetime, the threads,
 * the queues, and the routing; the stage just transforms incoming `StageMsg`s into
 * outgoing `StageOutMsg`s.
 *
 * **Lifecycle**: `set_ports()` → `start()` → repeated `on_input()` / `on_tick()` →
 * `stop()` → destruction.
 *
 * **Threading**: the runtime guarantees serial invocation per stage instance — `on_input`
 * and `on_tick` will not be called concurrently on the same stage. Stages may keep
 * non-atomic per-instance state without locks.
 *
 * @ingroup graph
 */
class StageExecutor {
public:
  /// Virtual destructor — required for the abstract base.
  virtual ~StageExecutor() = default;

  /// Provide resolved port ids for caching. Optional — default is a no-op.
  virtual void set_ports(const StagePorts& /*ports*/) {}

  /// Provide the runtime output emitter. Optional — default is a no-op.
  virtual void set_emitter(StageEmitter* /*emitter*/) {}

  /// Start any background work. Called once before the first `on_input`.
  virtual void start() {}
  /// Ask an active stage to cancel long-running work. Optional — default is a no-op.
  virtual void request_stop() {}
  /// Tear down. Called once after the last `on_input`/`on_tick`.
  virtual void stop() {}

  /**
   * @brief Handle one input message. Append zero or more outputs to `out`.
   * @param msg Input arriving on `msg.in_port`.
   * @param out Append outputs here; the runtime routes them on outgoing edges.
   */
  virtual void on_input(StageMsg&& msg, std::vector<StageOutMsg>& out) = 0;

  /**
   * @brief Optional periodic tick for timeout-driven stages.
   *
   * The first argument is a monotonic timestamp (ns); the second is the same output
   * vector convention as `on_input` — append outputs to it, the runtime routes them.
   */
  virtual void on_tick(std::int64_t /*now_ns*/, std::vector<StageOutMsg>& /*out*/) {}
};

} // namespace simaai::neat::graph
