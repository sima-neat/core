#include "neat/genai.h"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace genai = simaai::neat::genai;

struct Args {
  std::string host = "0.0.0.0";
  std::uint16_t port = 9998;
  std::filesystem::path llm;
  std::filesystem::path vlm;
  std::filesystem::path asr;
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--host" && i + 1 < argc) {
      args.host = argv[++i];
    } else if (arg == "--port" && i + 1 < argc) {
      args.port = static_cast<std::uint16_t>(std::stoi(argv[++i]));
    } else if (arg == "--llm" && i + 1 < argc) {
      args.llm = argv[++i];
    } else if (arg == "--vlm" && i + 1 < argc) {
      args.vlm = argv[++i];
    } else if (arg == "--asr" && i + 1 < argc) {
      args.asr = argv[++i];
    } else {
      throw std::runtime_error("usage: serve_genai_models [--host <host>] [--port <port>] "
                               "[--llm <dir>] [--vlm <dir>] [--asr <dir>]");
    }
  }
  if (args.llm.empty() && args.vlm.empty() && args.asr.empty()) {
    throw std::runtime_error("provide at least one of --llm <dir>, --vlm <dir>, or --asr <dir>");
  }
  return args;
}

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);

    // STEP configure-server
    genai::GenAIServerOptions options;
    options.host = args.host;
    options.port = args.port;
    genai::GenAIServer server(options);
    // END STEP

    // STEP register-models
    if (!args.llm.empty()) {
      server.add_model(args.llm, "llm");
    }
    if (!args.vlm.empty()) {
      server.add_model(args.vlm, "vlm");
    }
    if (!args.asr.empty()) {
      server.add_model(args.asr, "asr");
    }

    std::cout << "registered models:";
    for (const auto& name : server.model_names()) {
      std::cout << " " << name;
    }
    std::cout << "\n";
    // END STEP

    // STEP start-serving
    std::cout << "serving on http://" << options.host << ":" << options.port << "\n";
    std::cout << "try: curl http://<modalix-ip>:" << options.port << "/v1/models\n";
    server.serve();
    // END STEP

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
