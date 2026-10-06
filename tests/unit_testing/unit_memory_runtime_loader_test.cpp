#include "pipeline/internal/SimaMemApi.h"

#include <iostream>

int main() {
  const auto& api = simaai::neat::pipeline_internal::sima_mem_api();
  if (!api.attach || !api.invalidate || !api.invalidate_part || !api.flush || !api.map ||
      !api.unmap || !api.get_size) {
    std::cerr << "Required memory runtime entry points are unavailable\n";
    return 1;
  }
  std::cout << "Memory runtime entry points resolved\n";
  return 0;
}
