#include <simaai/neat/pcie/genai/GenAIModel.h>

#include <iostream>
#include <memory>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shlobj.h>
#endif

// Build against the staged Development package, not the Core source tree.
// Running with a model argument requires configured daemons and a card.
int main(int argc, char** argv) {
  namespace genai = simaai::neat::pcie::genai;
#ifdef _WIN32
  const genai::ConnectionOptions defaults;
  PWSTR common_application_data = nullptr;
  const HRESULT status = SHGetKnownFolderPath(FOLDERID_ProgramData, KF_FLAG_DEFAULT, nullptr,
                                              &common_application_data);
  std::unique_ptr<wchar_t, decltype(&CoTaskMemFree)> allocated(common_application_data,
                                                               &CoTaskMemFree);
  if (FAILED(status) || !allocated)
    return 2;
  const auto expected =
      std::filesystem::path(allocated.get()) / std::filesystem::path(L"SiMa.ai") / L"data";
  if (defaults.media_serve_root != "data" || defaults.media_directory != expected)
    return 3;
#endif
  if (argc < 2)
    return 0;
  genai::GenerationRequest request;
  request.prompt = "Hello";
  genai::GenAIModel model(argv[1]);
  std::cout << model.run(request).text << '\n';
  model.close();
}
