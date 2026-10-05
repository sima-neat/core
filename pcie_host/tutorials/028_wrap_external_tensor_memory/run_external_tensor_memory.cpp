// Submit application-owned tensor memory without a host staging copy.
//
// Usage:
//   tutorial_028_wrap_external_tensor_memory [--card 0]

#include <simaai/neat/pcie/Model.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace pcie = simaai::neat::pcie;

namespace {

constexpr int kBuildTimeoutMs = 180000;
constexpr int kPullTimeoutMs = 30000;
constexpr std::size_t kRingSlots = 3;
constexpr std::size_t kFrameCount = 8;
constexpr char kModelPath[] = "yolo_v8s_mpk.tar.gz";

int parse_card(const int argc, char** argv) {
  int card_id = 0;
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    if (arg == "--card" && index + 1 < argc) {
      card_id = std::stoi(argv[++index]);
    } else if (arg == "-h" || arg == "--help") {
      std::cout << "Usage: " << argv[0] << " [--card 0]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown or incomplete argument: " + arg);
    }
  }
  return card_id;
}

std::size_t checked_element_count(const std::vector<std::int64_t>& shape) {
  std::size_t count = 1;
  for (const std::int64_t dimension : shape) {
    if (dimension <= 0 ||
        count > std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(dimension)) {
      throw std::runtime_error("model input has an invalid or overflowing shape");
    }
    count *= static_cast<std::size_t>(dimension);
  }
  return count;
}

void validate_fp32_input(const pcie::TensorInfo& input) {
  if (input.dtype != "FP32" && input.dtype != "FLOAT32") {
    throw std::runtime_error("tutorial requires an FP32 model input, got " + input.dtype);
  }
  const std::size_t element_count = checked_element_count(input.shape);
  if (element_count > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
      element_count * sizeof(float) != input.size_bytes) {
    throw std::runtime_error("model input shape and byte size are inconsistent");
  }
}

struct InputSlot {
  std::shared_ptr<std::vector<float>> storage;
  pcie::Tensor tensor;
};

InputSlot make_input_slot(const pcie::TensorInfo& input) {
  auto storage = std::make_shared<std::vector<float>>(checked_element_count(input.shape), 0.0F);
  pcie::Tensor tensor = pcie::Tensor::from_external(storage->data(), storage->size(), storage,
                                                    input.shape, input.name);
  return {.storage = std::move(storage), .tensor = std::move(tensor)};
}

std::vector<InputSlot> make_input_ring(const pcie::TensorInfo& input) {
  std::vector<InputSlot> slots;
  slots.reserve(kRingSlots);
  for (std::size_t index = 0; index < kRingSlots; ++index) {
    slots.push_back(make_input_slot(input));
  }
  return slots;
}

std::size_t run_input_ring(pcie::Model& model, std::vector<InputSlot>& slots) {
  std::deque<std::size_t> available;
  std::deque<std::size_t> in_flight;
  for (std::size_t index = 0; index < slots.size(); ++index) {
    available.push_back(index);
  }

  std::size_t completed = 0;
  const auto complete_oldest = [&] {
    auto outputs = model.pull(kPullTimeoutMs);
    if (!outputs) {
      throw std::runtime_error("timed out waiting for an external-memory submission");
    }
    if (outputs->empty() || in_flight.empty()) {
      throw std::runtime_error("received an invalid external-memory completion");
    }
    available.push_back(in_flight.front());
    in_flight.pop_front();
    ++completed;
  };

  for (std::size_t frame = 0; frame < kFrameCount; ++frame) {
    if (available.empty()) {
      complete_oldest();
    }

    const std::size_t slot_index = available.front();
    available.pop_front();
    InputSlot& slot = slots[slot_index];

    // A slot is writable only while it is not in flight.
    std::fill(slot.storage->begin(), slot.storage->end(), static_cast<float>(frame % 10U) / 10.0F);
    if (!model.push(slot.tensor)) {
      throw std::runtime_error("push rejected frame " + std::to_string(frame));
    }
    in_flight.push_back(slot_index);
  }

  while (!in_flight.empty()) {
    complete_oldest();
  }
  return completed;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const int card_id = parse_card(argc, argv);
    if (!std::filesystem::is_regular_file(kModelPath)) {
      throw std::runtime_error(std::string("model does not exist: ") + kModelPath);
    }

    // STEP inspect-contract
    pcie::ConnectionOptions connection;
    connection.card_id = card_id;
    connection.max_inflight = static_cast<int>(kRingSlots);
    pcie::Model model(kModelPath, {}, connection);
    const pcie::ModelInfo info = model.info();
    if (info.inputs.size() != 1U) {
      throw std::runtime_error("tutorial requires a model with one input");
    }
    validate_fp32_input(info.inputs.front());
    // END STEP

    // STEP wrap-memory
    std::vector<InputSlot> slots = make_input_ring(info.inputs.front());
    // END STEP

    // STEP build-model
    model.build(kBuildTimeoutMs);
    // END STEP

    // STEP submit-ring
    std::size_t completed = 0;
    try {
      completed = run_input_ring(model, slots);
    } catch (...) {
      model.close();
      throw;
    }
    model.close();
    // END STEP

    std::cout << "input=" << info.inputs.front().name << '\n'
              << "ring_slots=" << slots.size() << '\n'
              << "completed=" << completed << '\n'
              << "[OK] 028_wrap_external_tensor_memory\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << '\n';
    return 1;
  }
}
