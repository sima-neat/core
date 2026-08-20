#include <simaai/neat/pcie/Model.h>
#include <simaai/neat/pcie/Runtime.h>

int main(int argc, char** argv) {
  simaai::neat::pcie::Runtime runtime;
  runtime.close();
  if (argc > 1) {
    simaai::neat::pcie::Model model(argv[1]);
    return model.running() ? 0 : 1;
  }
  return 0;
}
