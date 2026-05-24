#include "pipeline/GraphOptions.h"
#include "pipeline/TensorCore.h"
#include "test_main.h"

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

// Verifies the text tensor contract: UTF-8 text is carried as a normal
// UInt8 rank-1 Tensor and survives Tensor/Sample round-trips.
namespace {

void require_throws(const std::function<void()>& fn, const std::string& label) {
  try {
    fn();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error("expected exception: " + label);
}

void require_text_roundtrip(const std::string& text) {
  simaai::neat::Tensor tensor = simaai::neat::Tensor::from_text(text);
  require(tensor.semantic.text.has_value(), "text semantic missing");
  require(tensor.semantic.text->encoding == "utf-8", "text encoding mismatch");
  require(tensor.dtype == simaai::neat::TensorDType::UInt8, "text dtype mismatch");
  require(tensor.layout == simaai::neat::TensorLayout::Unknown, "text layout mismatch");
  require(tensor.shape.size() == 1U, "text shape rank mismatch");
  require(tensor.shape[0] == static_cast<int64_t>(text.size()), "text byte count mismatch");
  require(tensor.to_text() == text, "text roundtrip mismatch");

  std::string err;
  require(tensor.validate(&err), "text tensor validation failed: " + err);
}

} // namespace

RUN_TEST("unit_genai_text_tensor_test", ([] {
           require_text_roundtrip("");
           require_text_roundtrip("hello neat");
           const std::string utf8_text =
               "Gr\303\274\303\237e, "
               "\343\201\223\343\202\223\343\201\253\343\201\241\343\201\257";
           require_text_roundtrip(utf8_text);

           simaai::neat::Tensor text = simaai::neat::Tensor::from_text("prompt");

           simaai::neat::Sample sample = simaai::neat::make_tensor_sample("prompt", text);
           require(sample.kind == simaai::neat::SampleKind::TensorSet,
                   "text sample should be TensorSet");
           require(sample.tensors.size() == 1U, "text sample tensor count mismatch");
           require(sample.tensors.front().to_text() == "prompt", "text sample roundtrip failed");

           simaai::neat::Tensor bad_dtype = text;
           bad_dtype.dtype = simaai::neat::TensorDType::Float32;
           std::string err;
           require(!bad_dtype.validate(&err), "bad dtype text tensor should fail validation");
           require_contains(err, "UInt8", "bad dtype validation message mismatch");
           require_throws([&] { (void)bad_dtype.to_text(); }, "to_text bad dtype");

           simaai::neat::Tensor bad_layout = text;
           bad_layout.layout = simaai::neat::TensorLayout::HWC;
           require(!bad_layout.validate(&err), "bad layout text tensor should fail validation");
           require_contains(err, "layout", "bad layout validation message mismatch");
           require_throws([&] { (void)bad_layout.to_text(); }, "to_text bad layout");

           simaai::neat::Tensor non_text = simaai::neat::Tensor::from_vector(
               std::vector<uint8_t>{'x'}, {1}, simaai::neat::TensorMemory::CPU);
           require_throws([&] { (void)non_text.to_text(); }, "to_text non-text");
         }));
