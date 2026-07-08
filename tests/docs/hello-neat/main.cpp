#include <iostream>
#include <pipeline/TensorCore.h>

int main() {
  auto storage = simaai::neat::make_cpu_owned_storage(64);
  if (!storage) {
    std::cerr << "Failed to allocate CPU tensor storage\n";
    return 1;
  }
  std::cout << "Hello from sima-neat\n";
  return 0;
}
