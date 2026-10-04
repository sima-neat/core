#include "simaai/neat/pcie/genai/GenAIModel.h"
#include <iostream>
int main(int argc, char** argv) {
  namespace genai = simaai::neat::pcie::genai;
  if (argc < 3) {
    std::cerr << "usage: pcie-genai MODEL PROMPT [--host HOST] [--user USER] [--image FILE] "
                 "[--audio FILE]\n";
    return 2;
  }
  try {
    genai::ConnectionOptions connection;
    genai::GenerationRequest request;
    request.prompt = argv[2];
    for (int i = 3; i < argc; i += 2) {
      if (i + 1 >= argc)
        throw std::invalid_argument("Missing option value");
      std::string flag = argv[i];
      if (flag == "--host")
        connection.card_host = argv[i + 1];
      else if (flag == "--user")
        connection.user = argv[i + 1];
      else if (flag == "--image")
        request.image_files.emplace_back(argv[i + 1]);
      else if (flag == "--audio") {
        request.audio_file = argv[i + 1];
        request.prompt.reset();
      } else
        throw std::invalid_argument("Unknown option: " + flag);
    }
    genai::GenAIModel model(argv[1], connection);
    auto stream = model.stream(request);
    for (const auto& sample : stream)
      std::cout << sample.text << std::flush;
    std::cout << '\n';
    model.close();
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
