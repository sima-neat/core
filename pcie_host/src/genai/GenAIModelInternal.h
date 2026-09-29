#pragma once

#include "genai/GenAITransport.h"

#include "simaai/neat/pcie/genai/GenAIModel.h"

#include <memory>

namespace simaai::neat::pcie::genai::internal {

/**
 * @brief Build a GenAIModel over an explicit transport.
 *
 * The production entry points construct the real simaai_svc transport; this
 * factory injects any Transport, which lets tests drive the model with a scripted
 * fake and no card.
 */
GenAIModel make_model_with_transport(std::unique_ptr<Transport> transport);

} // namespace simaai::neat::pcie::genai::internal
