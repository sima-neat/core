#include "pipeline/Session.h"
#include "nodes/common/FileInput.h"
#include "nodes/common/ImageDecode.h"
#include "nodes/common/ImageFreeze.h"
#include "nodes/common/Output.h"

#include "test_utils.h"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  try {
    std::string image_path = "test.jpg";
    if (argc > 1)
      image_path = argv[1];

    simaai::neat::Session p;
    p.add(simaai::neat::nodes::FileInput(image_path));
    p.add(simaai::neat::nodes::ImageDecode());
    p.add(simaai::neat::nodes::ImageFreeze(1));
    p.add(simaai::neat::nodes::Output());

    const std::string save_path = "pipeline_saved.json";
    p.save(save_path);

    simaai::neat::Session loaded = simaai::neat::Session::load(save_path);
    const std::string original = p.describe_backend(false);
    const std::string restored = loaded.describe_backend(false);

    require(!original.empty(), "Original pipeline string empty");
    require(original == restored, "Loaded pipeline string mismatch");

    std::cout << "[OK] unit_pipeline_io_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
