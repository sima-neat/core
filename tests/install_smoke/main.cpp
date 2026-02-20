#include "pipeline/TensorCore.h"

int main() {
  auto storage = simaai::neat::make_cpu_owned_storage(64);
  return storage ? 0 : 1;
}
