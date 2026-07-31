/**
 * @file
 * @brief `neat-model-archive`: validates and extracts .tar.gz model archives.
 *
 * Backs `neat model validate` and `neat model extract`. Links the archive loader alone, so the
 * command runs anywhere the Core package is installed without pulling in the GStreamer runtime.
 */
#include "model/internal/ModelArchiveLoader.h"
#include "pipeline/internal/sima/MpkContract.h"

#include <cstdlib> // mkdtemp

#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using simaai::neat::internal::model_archive_error_class_name;
using simaai::neat::internal::ModelArchiveError;
using simaai::neat::internal::ModelArchiveLoader;
using simaai::neat::internal::ModelArchiveLoaderOptions;
using simaai::neat::internal::ModelArchiveManifest;
using simaai::neat::internal::rewrite_model_paths;
using simaai::neat::pipeline_internal::sima::load_mpk_contract_from_pack_root;

namespace {

constexpr const char* kProgram = "neat-model-archive";

/// Runtime loading tolerates the auxiliary build artifacts real model packs carry, so the CLI
/// must relax the same two checks or it would reject archives `Model` accepts.
ModelArchiveLoaderOptions runtime_parity_options() {
  ModelArchiveLoaderOptions opt;
  opt.reject_unsupported_file_types = false;
  opt.require_pipeline_sequence = false;
  return opt;
}

void print_usage(std::ostream& out) {
  out << "Usage:\n"
         "  neat model validate <archive.tar.gz>\n"
         "  neat model extract <archive.tar.gz> --output <directory>\n";
}

fs::path create_staging_directory(const fs::path& parent) {
  std::string staging_template = (parent / ".neat-model-archive.XXXXXX").string();
  if (::mkdtemp(staging_template.data()) == nullptr) {
    throw std::runtime_error("failed to create staging directory under " + parent.string());
  }
  return staging_template;
}

void require_mpk_contract(const fs::path& package_root) {
  std::string error;
  if (!load_mpk_contract_from_pack_root(package_root.string(), &error).has_value()) {
    throw ModelArchiveError(simaai::neat::internal::ModelArchiveErrorClass::SchemaError,
                            "schema_error: invalid MPK contract: " + error);
  }
}

/// ModelArchiveError prefixes what() with the error class and most loader messages name the class
/// as well. Collapse the pair so the reported class reads once.
std::string error_line(const ModelArchiveError& e) {
  const std::string prefix = std::string(model_archive_error_class_name(e.code())) + ": ";
  std::string text = e.what();
  const bool doubled = text.compare(0, prefix.size(), prefix) == 0 &&
                       text.compare(prefix.size(), prefix.size(), prefix) == 0;
  return doubled ? text.substr(prefix.size()) : text;
}

/// Directory entries carry no payload, so summing every entry gives what extraction will occupy.
std::uint64_t extracted_bytes(const ModelArchiveManifest& manifest) {
  std::uint64_t total = 0;
  for (const auto& entry : manifest.entries) {
    total += entry.size_bytes;
  }
  return total;
}

void validate(const std::string& archive) {
  const fs::path staging = create_staging_directory(fs::temp_directory_path());
  ModelArchiveManifest manifest;
  try {
    ModelArchiveLoaderOptions opt = runtime_parity_options();
    opt.staging_base = staging.string();
    const auto extracted = ModelArchiveLoader::extract(archive, staging.string(), opt);
    rewrite_model_paths(extracted.etc_dir, extracted.package_root);
    require_mpk_contract(extracted.package_root);
    manifest = extracted.manifest;
  } catch (...) {
    std::error_code ec;
    fs::remove_all(staging, ec);
    throw;
  }
  std::error_code ec;
  fs::remove_all(staging, ec);
  // Reports the extracted size because that, not the entry count, decides whether `extract` fits.
  std::cout << manifest.package_name << ": valid model archive (" << manifest.entries.size()
            << " entries, " << std::fixed << std::setprecision(1)
            << static_cast<double>(extracted_bytes(manifest)) / (1024.0 * 1024.0)
            << " MiB extracted)\n";
}

void extract(const std::string& archive, const std::string& output) {
  fs::path package_root = fs::absolute(output).lexically_normal();
  if (package_root.filename().empty()) {
    // A trailing separator would otherwise make parent_path() name the output itself, putting
    // staging inside the directory it must be renamed onto.
    package_root = package_root.parent_path();
  }
  if (fs::exists(package_root)) {
    throw std::runtime_error("output already exists: " + package_root.string());
  }
  const fs::path parent = package_root.parent_path();
  fs::create_directories(parent);

  // Staging shares the destination filesystem, so publication is a rename and a reader never
  // observes a partial package at the requested path.
  const fs::path staging = create_staging_directory(parent);

  ModelArchiveLoaderOptions opt = runtime_parity_options();
  // Inflate beside the destination too. The snapshot is the size of the extracted package, and
  // TMPDIR is routinely a smaller filesystem than the one holding --output.
  opt.staging_base = staging.string();

  try {
    const auto extracted = ModelArchiveLoader::extract(archive, staging.string(), opt);
    // Anchored at the published root, not at staging, so the JSON never names a path that stops
    // existing at the rename below.
    rewrite_model_paths(extracted.etc_dir, package_root.string());
    require_mpk_contract(extracted.package_root);
    fs::rename(extracted.package_root, package_root);
  } catch (...) {
    std::error_code ec;
    fs::remove_all(staging, ec);
    throw;
  }
  std::error_code ec;
  fs::remove_all(staging, ec);
  std::cout << package_root.string() << "\n";
}

} // namespace

int main(int argc, char** argv) {
  const std::vector<std::string> args(argv + 1, argv + argc);
  if (!args.empty() && (args[0] == "--help" || args[0] == "-h")) {
    print_usage(std::cout);
    return 0;
  }

  try {
    if (args.size() == 2 && args[0] == "validate") {
      validate(args[1]);
      return 0;
    }
    if (args.size() == 4 && args[0] == "extract" && args[2] == "--output") {
      extract(args[1], args[3]);
      return 0;
    }
  } catch (const ModelArchiveError& e) {
    // The class stays in the message: it is the CLI's contract with the runtime loader.
    std::cerr << kProgram << ": " << error_line(e) << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << kProgram << ": " << e.what() << "\n";
    return 1;
  }

  print_usage(std::cerr);
  return 1;
}
