#include "gst/GstInit.h"
#include "test_utils.h"

#include <gst/gst.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
  try {
    unsetenv("SIMA_ALLOW_GST_INIT");

    int argc = 0;
    char** argv = nullptr;
    gst_init(&argc, &argv);

    bool threw = false;
    std::string err;
    try {
      simaai::neat::gst_init_once();
    } catch (const std::exception& ex) {
      threw = true;
      err = ex.what();
    }
    require(threw, "expected gst_init_once to throw when gst_init called first");
    require(err.find("gst_init") != std::string::npos, "error message should mention gst_init");

    setenv("SIMA_ALLOW_GST_INIT", "1", 1);
    bool threw_override = false;
    std::string override_err;
    try {
      simaai::neat::gst_init_once();
    } catch (const std::exception& ex) {
      threw_override = true;
      override_err = ex.what();
    }
    require(!threw_override,
            "expected gst_init_once to allow when SIMA_ALLOW_GST_INIT=1: " + override_err);

    std::cout << "[OK] unit_gst_init_guard_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
