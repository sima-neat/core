/**
 * @file
 * @brief Internal helpers for Model (not part of the public API).
 */
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "model/Model.h"

namespace simaai::neat {
class NodeGroup;
} // namespace simaai::neat

namespace simaai::neat::internal {

class ModelPack;

struct ModelAccess {
  static const class ModelPack& pack(const Model& model);
  static const class ModelPack& pack_for_sync(const Model& model);
  static std::string model_id(const Model& model);
  static simaai::neat::NodeGroup build_preprocess_group(const Model& model, bool sync);
  static simaai::neat::NodeGroup build_infer_group(const Model& model, bool sync);
};

} // namespace simaai::neat::internal
