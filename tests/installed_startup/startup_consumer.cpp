// Installed-header consumer. Cold startup/admission only: no pipeline, model,
// allocator allocation, numerical submission, or device retirement claim.
#include <gst/GstInit.h>
#include <simaai/scheduler/kernel_preparation.h>
#include <gst/gst.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace {
namespace fs = std::filesystem;
namespace scheduler = simaai::scheduler;
namespace rt = simaai::runtime;
namespace k = simaai::kernels;

void require(bool value, const std::string& message) {
  if (!value)
    throw std::runtime_error(message);
}

class IsolatedRegistry {
public:
  IsolatedRegistry(const fs::path& installed, bool include_mla) {
    require(fs::is_directory(installed),
            "installed plugin directory is missing: " + installed.string());
    const auto allocator = fs::canonical(installed / "libgstneatallocator.so");
    // Check the real MLA fixture even in the negative: missing test inputs must
    // not be mistaken for proof that production rejects an omitted provider.
    const auto installed_mla = fs::canonical(installed / "libgstneatprocessmla.so");
    std::string pattern = (fs::temp_directory_path() / "neat-startup-contract-XXXXXX").string();
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    char* created = ::mkdtemp(buffer.data());
    require(created != nullptr, "cannot create isolated startup registry");
    root_ = created;
    try {
      fs::create_directory(root_ / "plugins");
      fs::create_directory(root_ / "no-system-plugins");
      // Core already links this allocator. Preserve its inode identity rather
      // than copying it and accidentally introducing a second memory authority.
      fs::create_symlink(allocator, root_ / "plugins/libgstneatallocator.so");
      require(fs::equivalent(allocator, root_ / "plugins/libgstneatallocator.so"),
              "isolated allocator must reference the installed memory authority");
      if (include_mla) {
        // Discovery is isolated, but NEAT-only provenance resolves symlinks.
        // Stage the real MLA bytes inside the selected root; dependency search
        // paths still name the original installed cohort (set before exec).
        mla_provider_ = root_ / "plugins/libgstneatprocessmla.so";
        fs::copy_file(installed_mla, mla_provider_);
        std::ifstream source(installed_mla, std::ios::binary);
        std::ifstream staged(mla_provider_, std::ios::binary);
        require(source.good() && staged.good() &&
                    fs::file_size(installed_mla) == fs::file_size(mla_provider_) &&
                    std::equal(
                        std::istreambuf_iterator<char>(source), std::istreambuf_iterator<char>(),
                        std::istreambuf_iterator<char>(staged), std::istreambuf_iterator<char>()) &&
                    !source.bad() && !staged.bad(),
                "isolated MLA plugin differs from the installed provider");
        mla_provider_ = fs::canonical(mla_provider_);
        require(mla_provider_.parent_path() == fs::canonical(root_ / "plugins"),
                "isolated MLA plugin must remain inside the selected root");
      }
      for (const char* key : {"SIMA_ALLOW_GST_INIT", "GST_PLUGIN_PATH", "GST_PLUGIN_PATH_1_0",
                              "GST_PLUGIN_SYSTEM_PATH", "GST_PLUGIN_SYSTEM_PATH_1_0",
                              "GST_REGISTRY", "GST_REGISTRY_1_0", "GST_PLUGIN_SCANNER"})
        ::unsetenv(key);
      set("SIMA_GST_PLUGIN_DIR", root_ / "plugins");
      set("GST_PLUGIN_SYSTEM_PATH_1_0", root_ / "no-system-plugins");
      set("GST_REGISTRY", root_ / "registry.bin");
      set("GST_REGISTRY_1_0", root_ / "registry.bin");
      // These supported isolation controls do not bypass Core startup validation.
      ::setenv("SIMA_GST_ALLOW_SYSTEM_PLUGINS", "1", 1);
      ::setenv("SIMA_GST_NEAT_ONLY", "1", 1);
      ::setenv("SIMA_GST_WRAP_SCANNER", "0", 1);
      ::setenv("GST_REGISTRY_FORK", "no", 1);
      ::setenv("SIMA_GST_SUPPRESS_DEVICE_LOGS", "0", 1);
    } catch (...) {
      fs::remove_all(root_);
      throw;
    }
  }

  ~IsolatedRegistry() {
    std::error_code ignored;
    fs::remove_all(root_, ignored);
  }

  const fs::path& mla_provider() const {
    return mla_provider_;
  }

private:
  static void set(const char* name, const fs::path& value) {
    require(::setenv(name, value.c_str(), 1) == 0, std::string("cannot set ") + name);
  }
  fs::path root_;
  fs::path mla_provider_;
};

void require_factory(const char* name, const fs::path* expected = nullptr) {
  GstElementFactory* factory = gst_element_factory_find(name);
  require(factory != nullptr, std::string("required actual factory missing: ") + name);
  GstPlugin* plugin = gst_plugin_feature_get_plugin(GST_PLUGIN_FEATURE(factory));
  const gchar* filename = plugin ? gst_plugin_get_filename(plugin) : nullptr;
  const std::string observed = filename ? filename : "";
  if (plugin)
    gst_object_unref(plugin);
  gst_object_unref(factory);
  if (expected) {
    require(!observed.empty() && fs::equivalent(observed, *expected),
            std::string("unexpected provider for ") + name + ": " + observed);
  } else {
    require(observed.empty(),
            "generic runner unexpectedly came from a discovered plugin: " + observed);
  }
}

void require_absent(const char* name) {
  GstElementFactory* factory = gst_element_factory_find(name);
  const bool absent = factory == nullptr;
  if (factory)
    gst_object_unref(factory);
  require(absent, std::string("isolated registry unexpectedly contains ") + name);
}

void check_real_boxdecode() {
  // Stable public request, not a provider factory or duplicated operation ID.
  const std::array inputs{rt::TensorSpec{{1, 1, 4}, rt::DType::BFloat16},
                          rt::TensorSpec{{1, 1, 2}, rt::DType::BFloat16}};
  const std::array outputs{
      rt::TensorSpec{{2, 4}, rt::DType::Int32}, rt::TensorSpec{{2}, rt::DType::Float32},
      rt::TensorSpec{{2}, rt::DType::Int32}, rt::TensorSpec{{1}, rt::DType::Int32}};
  const scheduler::ParameterRecords heads{{{"boxes_input", 0U}, {"scores_input", 1U}}};
  const std::vector<scheduler::Parameter> parameters{
      {"profile", "decoded_v1"},
      {"heads", heads},
      {"model_input_size", scheduler::ParameterArray{64, 64}},
      {"original_image_size", scheduler::ParameterArray{64, 64}},
      {"num_classes", 2},
      {"max_detections", 2},
      {"detection_threshold", .5F},
      {"nms_iou_threshold", .5F},
      {"score_activation", "identity"}};
  auto result = scheduler::check_kernel({"boxdecode", k::Target::A65, inputs, outputs, parameters});
  if (const auto* error = std::get_if<rt::Error>(&result))
    throw std::runtime_error("real BoxDecode admission: " + error->message);
  const auto executable = std::get<rt::KernelExecutable>(std::move(result));
  auto inspected = scheduler::inspect_kernel(executable);
  require(std::holds_alternative<scheduler::KernelInspection>(inspected),
          "real BoxDecode inspection failed");
  const auto& contract = std::get<scheduler::KernelInspection>(inspected);
  require(contract.inputs().size() == inputs.size() && contract.outputs().size() == outputs.size(),
          "BoxDecode admitted the wrong public port contract");
  for (std::size_t i = 0; i < inputs.size(); ++i)
    require(contract.inputs()[i].shape == inputs[i].shape &&
                contract.inputs()[i].dtype == inputs[i].dtype,
            "BoxDecode input geometry/dtype was not preserved");
  for (std::size_t i = 0; i < outputs.size(); ++i)
    require(contract.outputs()[i].shape == outputs[i].shape &&
                contract.outputs()[i].dtype == outputs[i].dtype,
            "BoxDecode output geometry/dtype was not preserved");
}
} // namespace

int main(int argc, char** argv) {
  try {
    const std::string mode = argc > 1 ? argv[1] : "positive";
    require(mode == "positive" || mode == "missing-mla" || mode == "missing-runner",
            "unsupported startup test mode");
    const char* configured = argc > 2 ? argv[2] : std::getenv("SIMA_GST_PLUGIN_DIR");
    require(configured && *configured, "set the installed cohort's SIMA_GST_PLUGIN_DIR");
    const fs::path installed = fs::canonical(configured);
    require(!gst_is_initialized(), "startup test process was already initialized");
    IsolatedRegistry registry(installed, mode != "missing-mla");
    try {
      simaai::neat::gst_init_once();
    } catch (const std::exception& error) {
      const std::string expected =
          mode == "missing-mla"      ? "Required NEAT factory is missing: neatprocessmla"
          : mode == "missing-runner" ? "Required NEAT factory is missing: neatkernelrunner"
                                     : "";
      require(!expected.empty() && std::string(error.what()).find(expected) != std::string::npos,
              "unexpected startup error: " + std::string(error.what()));
      std::cout << "[OK] exact startup rejection: " << error.what() << '\n';
      return 0;
    }
    require(mode == "positive", "startup incorrectly accepted an unavailable required factory");
    require_factory("neatkernelrunner");
    require_factory("neatprocessmla", &registry.mla_provider());
    require_absent("neatobjectdecode");
    require_absent("neatprocesscvu");
    check_real_boxdecode();
    std::cout << "[OK] public Core startup and real canonical BoxDecode admission without legacy "
                 "plugins\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "[FAIL] " << error.what() << '\n';
    return 1;
  }
}
