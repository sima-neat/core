#include <simaai/neat/pcie/genai/GenAIModel.h>

#include <iostream>

// Build against the staged Development package, not the Core source tree.
// Running with a model argument requires configured daemons and a card.
int main(int argc, char** argv) {
  if (argc < 2)
    return 0;
  namespace genai = simaai::neat::pcie::genai;
  genai::GenerationRequest request;
  request.prompt = "Hello";
  genai::GenAIModel model(argv[1]);
  std::cout << model.run(request).text << '\n';
  model.close();
}
