#include "neat/version.h"
#include "pipeline/TensorCore.h"

#include <cstring>

int main() {
  if (std::strcmp(sima_neat_abi_version(), "4") != 0) {
    return 1;
  }
  auto storage = simaai::neat::make_cpu_owned_storage(64);
  return storage ? 0 : 2;
}
