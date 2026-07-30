#include <simaai/neat/pcie/Model.h>

int main(int argc, char** argv) {
  if (argc > 1) {
    simaai::neat::pcie::Model model(argv[1]);
    return model.running() ? 0 : 1;
  }
  return 0;
}
