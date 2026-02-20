#pragma once

#include "test_utils.h"

#include <functional>
#include <iostream>
#include <stdexcept>

namespace sima_test {

inline int run_test(const char* name, const std::function<void()>& fn) {
  try {
    fn();
    std::cout << "[OK] " << name << " passed\n";
    return 0;
  } catch (const SkipTest& e) {
    if (is_long_test_context()) {
      std::cout << "[SKIP-LONG] " << e.what() << "\n";
      return 77;
    }
    std::cerr << "[FAIL][SKIP_USED] " << e.what() << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "[FAIL] unknown exception\n";
    return 1;
  }
}

} // namespace sima_test

#define RUN_TEST(name_literal, body_lambda)                                                        \
  int main() {                                                                                     \
    return sima_test::run_test(name_literal, body_lambda);                                         \
  }
