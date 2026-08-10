#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "ExecutionGraphPlan.h"
#include "pipeline/internal/DecoderAdmissionClient.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat::runtime {

class DecoderAdmissionBackend {
public:
  virtual ~DecoderAdmissionBackend() = default;

  virtual pipeline_internal::DecoderAdmissionResult
  admit(const std::vector<pipeline_internal::DecoderAdmissionStreamRequest>& streams,
        bool dry_run) = 0;
  virtual bool release(const std::array<std::uint8_t, 16>& group_uuid, std::string* error) = 0;
};

class DecoderAdmissionReservation {
public:
  DecoderAdmissionReservation() = default;
  DecoderAdmissionReservation(std::shared_ptr<DecoderAdmissionBackend> backend,
                              std::array<std::uint8_t, 16> group_uuid, std::size_t stream_count,
                              std::uint64_t reserved_bytes);
  ~DecoderAdmissionReservation();

  DecoderAdmissionReservation(const DecoderAdmissionReservation&) = delete;
  DecoderAdmissionReservation& operator=(const DecoderAdmissionReservation&) = delete;
  DecoderAdmissionReservation(DecoderAdmissionReservation&& other) noexcept;
  DecoderAdmissionReservation& operator=(DecoderAdmissionReservation&& other) noexcept;

  bool active() const noexcept;
  void release() noexcept;

private:
  std::shared_ptr<DecoderAdmissionBackend> backend_;
  std::array<std::uint8_t, 16> group_uuid_{};
  std::size_t stream_count_ = 0;
  std::uint64_t reserved_bytes_ = 0;
  bool active_ = false;
};

struct DecoderAdmissionPreparation {
  std::unique_ptr<DecoderAdmissionReservation> reservation;
  std::size_t eligible_decoders = 0;
  std::string warning;
};

DecoderAdmissionPreparation
prepare_decoder_admission(ExecutionGraphPlan& plan,
                          std::shared_ptr<DecoderAdmissionBackend> backend = nullptr);

} // namespace simaai::neat::runtime
