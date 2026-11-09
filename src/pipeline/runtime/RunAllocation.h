#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

// The Tensor Run::alloc(name) hands out (H0 plan 3.3, APPLICATION_MODEL.md 2.5: "pushing it is the
// commit"), the same on both backends. Its storage is CpuExternal over the allocation's bytes and
// its map_fn is a RunAllocationMapper, which is how a push recognises it (as TensorUtil.cpp
// recognises DmaBufStorageMapper): no registry, no lookup on the push of any other Tensor.
//
// State, under `mu`:
//   writable  the application may map and write it; push(const Tensor&) into the Run that
//             allocated it takes it
//   pushing   that push is publishing it (a second push of it fails)
//   pushed    published: the application's storage is emptied (data null, size 0; map() gives no
//             data), and a second push(const Tensor&) fails with misconfig.input_shape
// That push requires no live Mapping of it (`maps` == 0), so nothing the application still holds
// can write a published message. Any other push (another Run; the TensorList and Sample verbs)
// reads it as any CPU Tensor. A push the Run refuses (stopped, closed input) returns it to
// writable. Dropping it unpushed releases `backing`: the Dataflow backend's slot goes back to
// its pool (or to the runtime's retirement list when its Run is gone), the GStreamer backend's
// memory is freed.

#include "nodes/io/Input.h"
#include "pipeline/TensorCore.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace simaai::neat::runtime {

struct RunCore;

struct RunAllocation {
  enum class State { Writable, Pushing, Pushed };

  std::mutex mu;
  State state = State::Writable;
  std::uint32_t maps = 0;           ///< live Mappings of the application's storage
  std::weak_ptr<const RunCore> run; ///< the Run that allocated it (identity only)
  std::byte* data = nullptr;        ///< the bytes; valid while `backing` lives
  std::size_t size = 0;
  std::shared_ptr<void> backing;       ///< owns the bytes (a pool slot's loan, or CPU storage)
  std::weak_ptr<TensorBuffer> storage; ///< the application's storage, emptied at the push
};

// The map_fn of an allocated Tensor's storage: maps while writable, counts the Mapping.
struct RunAllocationMapper {
  std::shared_ptr<RunAllocation> allocation;
  Mapping operator()(MapMode mode) const;
};

// The application's Tensor over `allocation` (its data, size and backing already set), with the
// given element type and shape, writable.
Tensor make_allocated_tensor(const std::shared_ptr<RunAllocation>& allocation, TensorDType dtype,
                             std::vector<std::int64_t> shape);

// The allocation behind `tensor` when it came from Run::alloc of `core`; nullptr otherwise (any
// other Tensor, or one allocated by another Run).
std::shared_ptr<RunAllocation> allocation_of(const Tensor& tensor, const RunCore& core);

// The shape Run::alloc gives a declared input: each fixed axis, each dynamic axis at its bound.
// Throws NeatError (misconfig.input_shape) when the input declares no shape (InputOptions::shape)
// or a dynamic axis has no bound. `where` names the call and the input.
std::vector<std::int64_t> declared_alloc_shape(const InputOptions& declared,
                                               const std::string& where);

// Throws what begin_allocated_push would (pushed already, still mapped), changing nothing: a push
// checks it before it validates the Tensor, whose storage a published allocation has emptied.
void require_pushable(RunAllocation& allocation, const char* where);
// A push into the allocating Run takes the allocation: writable -> pushing, or a NeatError
// (misconfig.input_shape) when it was pushed already or is still mapped.
void begin_allocated_push(RunAllocation& allocation, const char* where);
// The push published it (pushing -> pushed, the application's storage emptied), or it did not
// (pushing -> writable).
void end_allocated_push(RunAllocation& allocation, bool published);

} // namespace simaai::neat::runtime
