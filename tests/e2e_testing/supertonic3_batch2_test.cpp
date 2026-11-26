// Supertonic 3 from the Model Browser: the vector estimator is compiled at batch 2 and the vocoder
// at batch 1. The test feeds the vector estimator synthetic batch-2 inputs and checks its input and
// output contract, that one run of one batch-2 input set returns one batch-2 result, that results
// repeat, that the two batch slots do not affect each other, that async results match sync ones in
// order, that a retained output survives later runs and that malformed inputs are refused. It then
// feeds one guided latent update to the vocoder. No reference audio is compared.
#include "cli_utils.h"
#include "gst/GstHelpers.h"
#include "gst/GstInit.h"
#include "model/Model.h"
#include "test_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

using Shape = std::vector<int64_t>;
// One host buffer per vector-estimator input, in input_specs() order, batch slot 0 first.
using Inputs = std::vector<std::vector<float>>;

constexpr int kSlots = 2;
constexpr int kTimeoutMs = 120000;
// The vector estimator's public inputs: noisy latent, text embedding, style_ttl, style_key,
// latent mask, text mask, current step and total step, all float32 NHWC.
const std::vector<Shape> kVectorInputs = {{2, 1, 192, 144}, {2, 1, 192, 256}, {2, 1, 50, 256},
                                          {2, 1, 50, 256},  {2, 1, 192, 1},   {2, 1, 1, 192},
                                          {2, 1, 1, 64},    {2, 1, 192, 128}};
const Shape kVectorOutput = {2, 1, 192, 144};
const Shape kVocoderInput = {1, 1, 192, 144};
const Shape kVocoderOutput = {1, 1, 1152, 512};
constexpr std::size_t kLatentInput = 0;
constexpr std::size_t kStyleTtlInput = 2;
constexpr std::size_t kStyleKeyInput = 3;
constexpr std::size_t kLatentMaskInput = 4;
constexpr std::size_t kTextMaskInput = 5;
// The masks mark the first 120 positions of each slot as valid.
constexpr std::size_t kValidPositions = 120;

int failures = 0;

void check(bool ok, const std::string& name, const std::string& detail = {}) {
  std::cout << (ok ? "[PASS] " : "[FAIL] ") << name;
  if (!detail.empty())
    std::cout << ": " << detail;
  std::cout << "\n";
  if (!ok)
    ++failures;
}

std::string shape_string(const Shape& shape) {
  std::ostringstream os;
  os << "[";
  for (std::size_t i = 0; i < shape.size(); ++i)
    os << (i ? "," : "") << shape[i];
  os << "]";
  return os.str();
}

std::size_t count_of(const Shape& shape) {
  std::size_t n = 1;
  for (int64_t d : shape)
    n *= static_cast<std::size_t>(std::max<int64_t>(d, 0));
  return n;
}

bool float32_spec(const simaai::neat::TensorSpec& spec) {
  return spec.dtypes.size() == 1 && spec.dtypes[0] == simaai::neat::TensorDType::Float32;
}

bool all_finite(const std::vector<float>& values) {
  return std::all_of(values.begin(), values.end(), [](float v) { return std::isfinite(v); });
}

bool bitwise_equal(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

std::string max_diff(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.size() != b.size())
    return "sizes " + std::to_string(a.size()) + " and " + std::to_string(b.size());
  double diff = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    // std::max keeps the old value when d is NaN.
    if (!std::isfinite(d))
      return "max diff inf";
    diff = std::max(diff, d);
  }
  return "max diff " + std::to_string(diff);
}

std::vector<float> slot(const std::vector<float>& batch, int index) {
  const std::size_t n = batch.size() / kSlots;
  const auto begin = batch.begin() + static_cast<std::ptrdiff_t>(n * index);
  return {begin, begin + static_cast<std::ptrdiff_t>(n)};
}

std::vector<float> read_floats(const simaai::neat::Tensor& tensor) {
  if (tensor.dtype != simaai::neat::TensorDType::Float32)
    throw std::runtime_error("expected a float32 output, got dtype " +
                             std::to_string(static_cast<int>(tensor.dtype)));
  const std::size_t count = count_of(tensor.shape);
  simaai::neat::Mapping map = tensor.map(simaai::neat::MapMode::Read);
  if (!map.data || map.size_bytes < count * sizeof(float))
    throw std::runtime_error("output map is smaller than its shape " + shape_string(tensor.shape));
  const auto* p = static_cast<const float*>(map.data);
  return {p, p + count};
}

// Seeded normals times 0.5 for every input; the two masks are 0/1 instead.
Inputs make_inputs(unsigned seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> normal(0.0f, 1.0f);
  Inputs inputs;
  for (const Shape& shape : kVectorInputs) {
    std::vector<float> data(count_of(shape));
    for (float& v : data)
      v = normal(rng) * 0.5f;
    inputs.push_back(std::move(data));
  }
  for (std::size_t idx : {kLatentMaskInput, kTextMaskInput}) {
    std::vector<float>& mask = inputs[idx];
    const std::size_t per_slot = mask.size() / kSlots;
    std::fill(mask.begin(), mask.end(), 0.0f);
    for (int s = 0; s < kSlots; ++s)
      std::fill_n(mask.begin() + static_cast<std::ptrdiff_t>(per_slot * s), kValidPositions, 1.0f);
  }
  return inputs;
}

simaai::neat::TensorList to_tensors(const Inputs& inputs) {
  simaai::neat::TensorList list;
  for (std::size_t i = 0; i < inputs.size(); ++i)
    list.push_back(simaai::neat::Tensor::from_vector(inputs[i], kVectorInputs[i],
                                                     simaai::neat::TensorMemory::EV74));
  return list;
}

// Builds a new input set from `inputs` with slot `index` of every input taken from `from`.
Inputs with_slot(const Inputs& inputs, const Inputs& from, int index) {
  Inputs out = inputs;
  for (std::size_t i = 0; i < out.size(); ++i) {
    const std::size_t n = out[i].size() / kSlots;
    std::copy_n(from[i].begin() + static_cast<std::ptrdiff_t>(n * index), n,
                out[i].begin() + static_cast<std::ptrdiff_t>(n * index));
  }
  return out;
}

Inputs swapped_slots(const Inputs& inputs) {
  Inputs out = inputs;
  for (std::size_t i = 0; i < out.size(); ++i) {
    const std::size_t n = out[i].size() / kSlots;
    std::copy_n(inputs[i].begin(), n, out[i].begin() + static_cast<std::ptrdiff_t>(n));
    std::copy_n(inputs[i].begin() + static_cast<std::ptrdiff_t>(n), n, out[i].begin());
  }
  return out;
}

std::vector<float> single_output(const simaai::neat::TensorList& outs) {
  if (outs.size() != 1)
    throw std::runtime_error("expected one vector estimator output, got " +
                             std::to_string(outs.size()));
  return read_floats(outs[0]);
}

bool throws(const std::function<void()>& fn, std::string* what) {
  try {
    fn();
  } catch (const std::exception& e) {
    *what = e.what();
    return true;
  }
  return false;
}

std::string resolve_component(const char* var, const char* component, int argc, char** argv) {
  std::string path;
  if (const char* env = std::getenv(var); env && *env)
    path = env;
  std::string arg;
  if (sima_test::get_arg(argc, argv, std::string("--") + component, arg))
    path = arg;
  std::error_code ec;
  if (path.empty() || !fs::is_regular_file(path, ec) || fs::file_size(path, ec) == 0) {
    throw std::runtime_error(
        std::string("Supertonic 3 ") + component + " MPK is missing or empty: '" + path +
        "'. Set " + var + " (or pass --" + component + ") to the supertonic_3_w8a16_w16a16_" +
        component + "_mpk.tar.gz from 'sima-cli models download --id supertonic_3 --variant " +
        "w8a16_w16a16', or run scripts/ci/prepare_model_browser_fixture.py --id supertonic_3 " +
        "--variant w8a16_w16a16 --component vector_estimator --component vocoder --output <dir>");
  }
  return path;
}

} // namespace

int main(int argc, char** argv) {
  try {
    simaai::neat::gst_init_once();
    if (!simaai::neat::element_exists("neatkernelrunner") ||
        !simaai::neat::element_exists("neatprocessmla")) {
      return fail_test("missing SimaAI plugins");
    }

    std::string vector_path, vocoder_path;
    try {
      vector_path = resolve_component("SIMA_SUPERTONIC_VECTOR_TAR", "vector_estimator", argc, argv);
      vocoder_path = resolve_component("SIMA_SUPERTONIC_VOCODER_TAR", "vocoder", argc, argv);
    } catch (const std::exception& e) {
      return fail_test(e.what());
    }
    std::cout << "[MODEL] supertonic_3 w8a16_w16a16 vector_estimator=" << vector_path
              << " vocoder=" << vocoder_path << "\n";

    simaai::neat::Model vector(vector_path);
    simaai::neat::Model vocoder(vocoder_path);
    const auto vector_in = vector.input_specs();
    const auto vector_out = vector.output_specs();
    const auto vocoder_in = vocoder.input_specs();
    const auto vocoder_out = vocoder.output_specs();
    auto print_specs = [](const std::string& label, const auto& specs) {
      for (const auto& s : specs)
        std::cout << "[SPEC] " << label << " shape=" << shape_string(s.shape)
                  << " float32=" << float32_spec(s) << "\n";
    };
    print_specs("vector_estimator input", vector_in);
    print_specs("vector_estimator output", vector_out);
    print_specs("vocoder input", vocoder_in);
    print_specs("vocoder output", vocoder_out);

    check(vector.compiled_batch_size() == kSlots && vocoder.compiled_batch_size() == 1,
          "vector estimator is compiled at batch 2 and vocoder at batch 1",
          std::to_string(vector.compiled_batch_size()) + ", " +
              std::to_string(vocoder.compiled_batch_size()));
    {
      bool ok = vector_in.size() == kVectorInputs.size();
      for (std::size_t i = 0; ok && i < vector_in.size(); ++i)
        ok = vector_in[i].shape == kVectorInputs[i] && float32_spec(vector_in[i]);
      check(ok, "vector estimator has the eight float32 batch-2 inputs in order",
            std::to_string(vector_in.size()) + " inputs");
    }
    check(vector_out.size() == 1 && vector_out[0].shape == kVectorOutput &&
              float32_spec(vector_out[0]),
          "vector estimator has one float32 output [2,1,192,144]");
    check(vocoder_in.size() == 1 && vocoder_in[0].shape == kVocoderInput &&
              float32_spec(vocoder_in[0]),
          "vocoder has one float32 input [1,1,192,144]");
    check(vocoder_out.size() == 1 && vocoder_out[0].shape == kVocoderOutput &&
              float32_spec(vocoder_out[0]),
          "vocoder has one float32 output [1,1,1152,512]");

    const Inputs base = make_inputs(1);
    const Inputs other = make_inputs(2);

    auto runner = vector.build(to_tensors(base));
    simaai::neat::TensorList base_outs = runner.run(to_tensors(base), kTimeoutMs);
    for (std::size_t i = 0; i < base_outs.size(); ++i)
      std::cout << "[OUTPUT] vector_estimator index=" << i << " name='" << base_outs[i].route.name
                << "' dtype=" << static_cast<int>(base_outs[i].dtype)
                << " shape=" << shape_string(base_outs[i].shape) << "\n";
    check(base_outs.size() == 1 && base_outs[0].shape == kVectorOutput &&
              base_outs[0].dtype == simaai::neat::TensorDType::Float32,
          "one run of one batch-2 input set returns one float32 [2,1,192,144] output");
    const std::vector<float> a = single_output(base_outs);
    check(all_finite(a), "vector estimator output is finite");
    // Kept to check that later runs do not write into a returned output.
    simaai::neat::Tensor kept = base_outs[0];
    base_outs.clear();

    const std::vector<float> a2 = single_output(runner.run(to_tensors(base), kTimeoutMs));
    check(bitwise_equal(a, a2), "a repeated run is bitwise identical", max_diff(a, a2));

    const std::vector<float> b =
        single_output(runner.run(to_tensors(with_slot(base, other, 1)), kTimeoutMs));
    check(bitwise_equal(slot(a, 0), slot(b, 0)), "changing slot 1 leaves slot 0 bitwise unchanged",
          max_diff(slot(a, 0), slot(b, 0)));
    check(all_finite(b) && !bitwise_equal(slot(a, 1), slot(b, 1)),
          "changing slot 1 changes slot 1 to finite values");

    const std::vector<float> c =
        single_output(runner.run(to_tensors(swapped_slots(base)), kTimeoutMs));
    check(bitwise_equal(slot(c, 0), slot(a, 1)) && bitwise_equal(slot(c, 1), slot(a, 0)),
          "swapping the slots swaps the outputs bitwise",
          max_diff(slot(c, 0), slot(a, 1)) + ", " + max_diff(slot(c, 1), slot(a, 0)));

    {
      // Slot 1 takes slot 0 of every input.
      const Inputs same = with_slot(base, swapped_slots(base), 1);
      const std::vector<float> e = single_output(runner.run(to_tensors(same), kTimeoutMs));
      check(bitwise_equal(slot(e, 0), slot(e, 1)) && bitwise_equal(slot(e, 0), slot(a, 0)),
            "the same input in both slots gives identical outputs equal to slot 0 of the base run",
            max_diff(slot(e, 0), slot(e, 1)) + ", " + max_diff(slot(e, 0), slot(a, 0)));
    }

    {
      Inputs styles = base;
      std::swap(styles[kStyleTtlInput], styles[kStyleKeyInput]);
      const std::vector<float> f = single_output(runner.run(to_tensors(styles), kTimeoutMs));
      check(all_finite(f) && !bitwise_equal(a, f),
            "swapping style_ttl and style_key changes the output to finite values", max_diff(a, f));
    }

    check(bitwise_equal(read_floats(kept), a), "a retained output is unchanged after later runs");
    kept = simaai::neat::Tensor{};

    // Four input sets pushed before any pull must come back in order with their sync results.
    {
      std::vector<Inputs> sets;
      std::vector<std::vector<float>> sync;
      for (unsigned k = 0; k < 4; ++k) {
        sets.push_back(make_inputs(10 + k));
        sync.push_back(single_output(runner.run(to_tensors(sets.back()), kTimeoutMs)));
      }
      for (const Inputs& set : sets)
        if (!runner.push(to_tensors(set)))
          throw std::runtime_error("async push failed");
      bool ok = true;
      std::string detail;
      for (std::size_t i = 0; i < sets.size(); ++i) {
        simaai::neat::Sample sample = runner.pull(kTimeoutMs);
        const std::vector<float> got = single_output(simaai::neat::tensors_from_sample(sample));
        if (!bitwise_equal(got, sync[i])) {
          ok = false;
          detail += "#" + std::to_string(i) + " " + max_diff(got, sync[i]) + " ";
        }
      }
      check(ok, "four async results match the sync results bitwise, in order", detail);
    }
    runner.close();

    // One guided update of the slot 0 latent from the conditional (slot 0) and unconditional
    // (slot 1) outputs, then the vocoder on the result.
    {
      const std::vector<float> latent0 = slot(base[kLatentInput], 0);
      const std::vector<float> cond = slot(a, 0);
      const std::vector<float> uncond = slot(a, 1);
      std::vector<float> latent(latent0.size());
      for (std::size_t i = 0; i < latent.size(); ++i)
        latent[i] = latent0[i] + (4.0f * cond[i] - 3.0f * uncond[i]) / 8.0f;
      const simaai::neat::Tensor latent_tensor = simaai::neat::Tensor::from_vector(
          latent, kVocoderInput, simaai::neat::TensorMemory::EV74);
      auto vocoder_runner = vocoder.build(simaai::neat::TensorList{latent_tensor});
      simaai::neat::TensorList wave_outs = vocoder_runner.run({latent_tensor}, kTimeoutMs);
      check(wave_outs.size() == 1 && wave_outs[0].shape == kVocoderOutput,
            "vocoder returns one [1,1,1152,512] output",
            wave_outs.empty() ? "no output" : shape_string(wave_outs[0].shape));
      if (wave_outs.size() == 1) {
        const std::vector<float> wave = read_floats(wave_outs[0]);
        double sum = 0.0;
        for (float v : wave)
          sum += static_cast<double>(v) * v;
        const double rms = wave.empty() ? 0.0 : std::sqrt(sum / static_cast<double>(wave.size()));
        check(all_finite(wave) && rms > 0.0, "vocoder output is finite with a nonzero RMS",
              "rms " + std::to_string(rms));
      }
      vocoder_runner.close();
    }

    // Each malformed input set gets its own Runner, so a check cannot pass on an earlier error.
    auto check_rejected = [&](const std::string& name, const simaai::neat::TensorList& bad) {
      auto fresh = vector.build(to_tensors(base));
      std::string what;
      const bool rejected = throws([&] { (void)fresh.run(bad, kTimeoutMs); }, &what);
      check(rejected, name + " is rejected", what);
      try {
        const std::vector<float> again = single_output(fresh.run(to_tensors(base), kTimeoutMs));
        check(bitwise_equal(again, a), "valid run after " + name + " matches the base run",
              max_diff(again, a));
      } catch (const std::exception& e) {
        check(false, "valid run after " + name + " matches the base run", e.what());
      }
      fresh.close();
    };
    {
      simaai::neat::TensorList bad = to_tensors(base);
      bad[kLatentInput] = simaai::neat::Tensor::from_vector(
          slot(base[kLatentInput], 0), kVocoderInput, simaai::neat::TensorMemory::EV74);
      check_rejected("a batch-1 noisy latent", bad);
    }
    {
      simaai::neat::TensorList bad = to_tensors(base);
      std::vector<uint8_t> bytes(base[kLatentInput].size(), 0);
      bad[kLatentInput] = simaai::neat::Tensor::from_vector(bytes, kVectorInputs[kLatentInput],
                                                            simaai::neat::TensorMemory::EV74);
      check_rejected("a uint8 noisy latent", bad);
    }
    {
      simaai::neat::TensorList bad = to_tensors(base);
      bad.pop_back();
      check_rejected("seven inputs", bad);
    }

    if (failures > 0) {
      std::cerr << "[FAIL] supertonic3_batch2_test: " << failures << " check(s) failed\n";
      return 1;
    }
    std::cout << "[OK] supertonic3_batch2_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
