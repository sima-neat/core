#include "yolov8_variant_route_matrix_common.inc"

namespace {

simaai::neat::RunOptions graph_model_run_options() {
  simaai::neat::RunOptions opt;
  // These graph outputs are re-pushed into a standalone SiMa device-transport
  // boxdecode (run_framework_boxdecode_accuracy). A device route can only accept
  // a device-backed (zero-copy GstSample) tensor; an Owned/CPU copy is not
  // device-re-pushable by design (the push guard fails fast on it). Use the same
  // async zero-copy contract as the Model::build path (run_canonical_model_sample),
  // so both entry points feed the boxdecode an equivalent device-backed output.
  opt.output_memory = simaai::neat::OutputMemory::Auto;
  opt.queue_depth = 1;
  if (const char* raw_depth = std::getenv("SIMA_E2E_RUN_QUEUE_DEPTH"); raw_depth && *raw_depth) {
    const int depth = std::atoi(raw_depth);
    if (depth > 0) {
      opt.queue_depth = depth;
    }
  }
  return opt;
}

simaai::neat::Model::Options graph_image_model_options() {
  auto opt = default_model_options();
  opt.preprocess.kind = simaai::neat::InputKind::Image;
  opt.preprocess.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.preset = simaai::neat::NormalizePreset::COCO_YOLO;
  opt.preprocess.resize.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.resize.width = 640;
  opt.preprocess.resize.height = 640;
  opt.preprocess.resize.mode = simaai::neat::ResizeMode::Letterbox;
  opt.preprocess.resize.pad_value = 114;
  opt.preprocess.resize.scaling_type = "BILINEAR";
  opt.preprocess.normalize.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.normalize.mean = {0.0f, 0.0f, 0.0f};
  opt.preprocess.color_convert.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.color_convert.input_format = simaai::neat::PreprocessColorFormat::BGR;
  opt.preprocess.color_convert.output_format = simaai::neat::PreprocessColorFormat::RGB;
  opt.preprocess.layout_convert.enable = simaai::neat::AutoFlag::On;
  opt.preprocess.layout_convert.perm = {0, 1, 2};
  opt.upstream_name = "decoder";
  return opt;
}

simaai::neat::Sample require_one_graph_output(const simaai::neat::Sample& outputs,
                                              const std::string& where) {
  require(outputs.size() == 1U,
          where + ": expected exactly one output sample, got " + std::to_string(outputs.size()));
  return outputs.front();
}

void require_tensor_contract_equivalent(const simaai::neat::Sample& expected,
                                        const simaai::neat::Sample& actual,
                                        const std::string& where) {
  const auto a = tensors_in_sample(expected);
  const auto b = tensors_in_sample(actual);
  require(a.size() == b.size(), where +
                                    ": tensor count mismatch expected=" + std::to_string(a.size()) +
                                    " actual=" + std::to_string(b.size()));
  for (std::size_t i = 0; i < a.size(); ++i) {
    require(a[i].shape == b[i].shape, where + ": tensor[" + std::to_string(i) +
                                          "] shape mismatch expected=" + shape_string(a[i].shape) +
                                          " actual=" + shape_string(b[i].shape));
    require(a[i].dtype == b[i].dtype, where + ": tensor[" + std::to_string(i) +
                                          "] dtype mismatch expected=" + dtype_name(a[i].dtype) +
                                          " actual=" + dtype_name(b[i].dtype));
  }
}

struct PreparedIngress {
  bool tensor = false;
  simaai::neat::InputOptions input_options;
  simaai::neat::Tensor tensor_input;
  cv::Mat image_input;
};

PreparedIngress prepare_canonical_tensor_ingress(const cv::Mat& img_bgr,
                                                 simaai::neat::Model& model) {
  PreparedIngress ingress;
  ingress.input_options = model.input_appsrc_options(true);
  require(upper_copy(simaai::neat::resolve_input_media_type(ingress.input_options)) ==
              "APPLICATION/VND.SIMAAI.TENSOR",
          "Graph::add(model) canonical coverage requires tensor ingress");
  ingress.tensor = true;
  ingress.tensor_input = build_canonical_preprocessed_input(img_bgr, model);
  return ingress;
}

template <typename BuildGraph>
simaai::neat::Sample run_graph_with_ingress(const PreparedIngress& ingress,
                                            BuildGraph&& build_graph, const std::string& where,
                                            int timeout_ms) {
  const int retry_timeout_ms = std::max(timeout_ms, kRunRetryTimeoutMs);
  auto run_once = [&](int pull_timeout_ms) -> simaai::neat::Sample {
    simaai::neat::Graph g = build_graph();
    auto opt = graph_model_run_options();
    if (ingress.tensor) {
      auto run = g.build(simaai::neat::TensorList{ingress.tensor_input}, opt);
      require(static_cast<bool>(run), where + ": build(tensor) failed");
      require(run.push(simaai::neat::TensorList{ingress.tensor_input}),
              where + ": push(tensor) failed");
      auto out = require_one_graph_output(run.pull_samples(pull_timeout_ms), where);
      run.close();
      return out;
    }

    auto run = g.build(std::vector<cv::Mat>{ingress.image_input}, opt);
    require(static_cast<bool>(run), where + ": build(image) failed");
    require(run.push(std::vector<cv::Mat>{ingress.image_input}), where + ": push(image) failed");
    auto out = require_one_graph_output(run.pull_samples(pull_timeout_ms), where);
    run.close();
    return out;
  };

  try {
    return run_once(timeout_ms);
  } catch (const std::exception&) {
    return run_once(retry_timeout_ms);
  }
}

simaai::neat::Sample run_graph_add_model_sample(const cv::Mat& img_bgr,
                                                simaai::neat::Model& model) {
  const PreparedIngress ingress = prepare_canonical_tensor_ingress(img_bgr, model);
  return run_graph_with_ingress(
      ingress,
      [&]() {
        simaai::neat::Graph g;
        g.add(simaai::neat::nodes::Input(ingress.input_options));
        g.add(model);
        g.add(simaai::neat::nodes::Output());
        return g;
      },
      "Graph::add(model)", default_model_run_timeout_ms());
}

simaai::neat::Sample run_graph_connected_model_sample(const cv::Mat& img_bgr,
                                                      simaai::neat::Model& model) {
  const PreparedIngress ingress = prepare_canonical_tensor_ingress(img_bgr, model);
  return run_graph_with_ingress(
      ingress,
      [&]() {
        simaai::neat::Graph source;
        source.add(simaai::neat::nodes::Input(ingress.input_options));
        source.add(model);

        simaai::neat::Graph sink;
        sink.add(simaai::neat::nodes::Output());

        simaai::neat::Graph app;
        app.connect(source, sink);
        return app;
      },
      "Graph::connect(Input+model, Output)", default_model_run_timeout_ms());
}

simaai::neat::Sample run_graph_image_add_model_sample(const cv::Mat& img_bgr,
                                                      simaai::neat::Model& model) {
  auto opt = graph_model_run_options();
  auto run_once = [&](int timeout_ms) -> simaai::neat::Sample {
    simaai::neat::Graph g;
    g.add(simaai::neat::nodes::Input());
    g.add(model);
    g.add(simaai::neat::nodes::Output());
    auto run = g.build(std::vector<cv::Mat>{img_bgr}, opt);
    require(static_cast<bool>(run), "image Graph::add(model): build failed");
    require(run.push(std::vector<cv::Mat>{img_bgr}), "image Graph::add(model): push failed");
    auto out = require_one_graph_output(run.pull_samples(timeout_ms), "image Graph::add(model)");
    run.close();
    return out;
  };
  try {
    return run_once(default_model_run_timeout_ms());
  } catch (const std::exception&) {
    return run_once(std::max(default_model_run_timeout_ms(), kRunRetryTimeoutMs));
  }
}

simaai::neat::Sample run_graph_explicit_stage_fragments_sample(const cv::Mat& img_bgr,
                                                               simaai::neat::Model& model) {
  auto opt = graph_model_run_options();
  auto run_once = [&](int timeout_ms) -> simaai::neat::Sample {
    simaai::neat::Graph g;
    g.add(simaai::neat::nodes::Input());
    g.add(model.preprocess());
    g.add(model.inference());
    g.add(model.postprocess());
    g.add(simaai::neat::nodes::Output());
    auto run = g.build(std::vector<cv::Mat>{img_bgr}, opt);
    require(static_cast<bool>(run), "stage-fragment Graph: build failed");
    require(run.push(std::vector<cv::Mat>{img_bgr}), "stage-fragment Graph: push failed");
    auto out = require_one_graph_output(run.pull_samples(timeout_ms), "stage-fragment Graph");
    run.close();
    return out;
  };
  try {
    return run_once(default_model_run_timeout_ms());
  } catch (const std::exception&) {
    return run_once(std::max(default_model_run_timeout_ms(), kRunRetryTimeoutMs));
  }
}

simaai::neat::Sample run_graph_lifetime_sample(const cv::Mat& img_bgr, const fs::path& tar) {
  simaai::neat::Graph g;
  PreparedIngress ingress;
  {
    simaai::neat::Model model(tar.string(), canonical_model_options(BoxDecodeRunMode::NoModel));
    ingress = prepare_canonical_tensor_ingress(img_bgr, model);
    g.add(simaai::neat::nodes::Input(ingress.input_options));
    g.add(model);
    g.add(simaai::neat::nodes::Output());
  }

  const auto opt = graph_model_run_options();
  if (ingress.tensor) {
    auto run = g.build(simaai::neat::TensorList{ingress.tensor_input}, opt);
    require(static_cast<bool>(run), "Graph::add(model) lifetime: build(tensor) failed");
    require(run.push(simaai::neat::TensorList{ingress.tensor_input}),
            "Graph::add(model) lifetime: push(tensor) failed");
    auto out = require_one_graph_output(run.pull_samples(default_model_run_timeout_ms()),
                                        "Graph::add(model) lifetime");
    run.close();
    return out;
  }

  auto run = g.build(std::vector<cv::Mat>{ingress.image_input}, opt);
  require(static_cast<bool>(run), "Graph::add(model) lifetime: build(image) failed");
  require(run.push(std::vector<cv::Mat>{ingress.image_input}),
          "Graph::add(model) lifetime: push(image) failed");
  auto out = require_one_graph_output(run.pull_samples(default_model_run_timeout_ms()),
                                      "Graph::add(model) lifetime");
  run.close();
  return out;
}

void run_repeated_build_check(const cv::Mat& img_bgr, simaai::neat::Model& model,
                              const std::string& model_name) {
  const PreparedIngress ingress = prepare_canonical_tensor_ingress(img_bgr, model);
  simaai::neat::Graph g;
  g.add(simaai::neat::nodes::Input(ingress.input_options));
  g.add(model);
  g.add(simaai::neat::nodes::Output());

  for (int i = 0; i < 3; ++i) {
    const auto opt = graph_model_run_options();
    simaai::neat::Sample out;
    if (ingress.tensor) {
      auto run = g.build(simaai::neat::TensorList{ingress.tensor_input}, opt);
      require(static_cast<bool>(run),
              "Graph repeated build iteration " + std::to_string(i) + ": build(tensor) failed");
      require(run.push(simaai::neat::TensorList{ingress.tensor_input}),
              "Graph repeated build iteration " + std::to_string(i) + ": push(tensor) failed");
      out = require_one_graph_output(run.pull_samples(default_model_run_timeout_ms()),
                                     "Graph repeated build iteration " + std::to_string(i));
      run.close();
    } else {
      auto run = g.build(std::vector<cv::Mat>{ingress.image_input}, opt);
      require(static_cast<bool>(run),
              "Graph repeated build iteration " + std::to_string(i) + ": build(image) failed");
      require(run.push(std::vector<cv::Mat>{ingress.image_input}),
              "Graph repeated build iteration " + std::to_string(i) + ": push(image) failed");
      out = require_one_graph_output(run.pull_samples(default_model_run_timeout_ms()),
                                     "Graph repeated build iteration " + std::to_string(i));
      run.close();
    }

    const AccuracyResult acc =
        run_framework_boxdecode_accuracy(out, model, img_bgr, BoxDecodeRunMode::NoModel);
    require(acc.ok,
            "Graph repeated build accuracy failed iter=" + std::to_string(i) + ": " + acc.note);
    std::cout << "GRAPH_MODEL_REBUILD model=" << model_name << " iter=" << i
              << " status=OK accuracy=\"" << acc.note << "\"\n";
  }
}

bool should_run_stress_cases(const fs::path& tar, std::size_t pack_count) {
  if (pack_count <= 1U)
    return true;
  if (sima_yolov8_test::env_bool("SIMA_GRAPH_MODEL_COMPOSITION_FULL_MATRIX", false))
    return true;
  return tar.filename().string() == "yolov8n_A_W_INT8_MLATess.tar.gz";
}

} // namespace

int main(int argc, char** argv) {
#if !defined(SIMA_WITH_OPENCV)
  (void)argc;
  (void)argv;
  return skip_long_test(
      "OpenCV required for graph_migration_unified_yolov8_graph_model_composition_test");
#else
  try {
    simaai::neat::gst_init_once();
    if (!simaai::neat::element_exists("neatprocessmla")) {
      return skip_long_test("missing NEAT MLA plugin (neatprocessmla)");
    }
    if (!simaai::neat::element_exists("neatprocesscvu")) {
      return skip_long_test("missing NEAT CVU plugin (neatprocesscvu)");
    }
    if (!simaai::neat::element_exists("neatobjectdecode")) {
      return skip_long_test("missing NEAT objectdecode plugin (neatobjectdecode)");
    }

    fs::path root = find_repo_root(fs::current_path());
    std::vector<fs::path> packs;
    const std::optional<fs::path> single_model_tar = resolve_single_model_tar(argc, argv);
    if (single_model_tar.has_value()) {
      const fs::path tar = fs::absolute(*single_model_tar);
      require(fs::exists(tar), "single model tar not found: " + tar.string());
      packs.push_back(tar);
    } else {
      fs::path variants_dir = resolve_variants_dir(root, argc, argv);
      packs = collect_model_packs(variants_dir);
      if (packs.empty()) {
        variants_dir = sima_yolov8_test::ensure_yolov8n_drive_variants(root);
        packs = collect_model_packs(variants_dir);
      }
      require(!packs.empty(), "no model packs found in " + variants_dir.string());
    }

    const cv::Mat img_bgr = sima_yolov8_test::load_people_image_or_skip(root);
    int failures = 0;
    std::size_t executed = 0;

    for (const auto& tar : packs) {
      const std::string model_name = tar.filename().string();
      try {
        const ProbeResult probe = probe_model(tar);
        (void)probe;

        simaai::neat::Model model(tar.string(), canonical_model_options(BoxDecodeRunMode::NoModel));
        (void)model.info();
        (void)model.input_specs();
        (void)model.inference();

        const auto tensor_io_before = simaai::neat::pipeline_internal::snapshot_tensor_io_stats();
        simaai::neat::Model::RouteOptions route_opt;
        const simaai::neat::Sample model_sample =
            run_canonical_model_sample(img_bgr, model, route_opt, 1);
        const simaai::neat::Sample graph_sample = run_graph_add_model_sample(img_bgr, model);
        const auto tensor_io_after = simaai::neat::pipeline_internal::snapshot_tensor_io_stats();
        const auto tensor_io = tensor_io_delta(tensor_io_before, tensor_io_after);

        require_tensor_contract_equivalent(model_sample, graph_sample,
                                           "Model::build vs Graph::add(model) " + model_name);
        const std::string model_sig = sample_output_signature_local(model_sample);
        const std::string graph_sig = sample_output_signature_local(graph_sample);
        require(model_sig == graph_sig,
                "signature mismatch model='" + model_sig + "' graph='" + graph_sig + "'");

        require_preprocess_meta_on_output_local(graph_sample, img_bgr.cols, img_bgr.rows,
                                                "graph_add_model_output");
        const AccuracyResult model_acc = run_framework_boxdecode_accuracy(
            model_sample, model, img_bgr, BoxDecodeRunMode::NoModel);
        require(model_acc.ok, "Model::build accuracy failed: " + model_acc.note);
        const AccuracyResult graph_acc = run_framework_boxdecode_accuracy(
            graph_sample, model, img_bgr, BoxDecodeRunMode::NoModel);
        require(graph_acc.ok, "Graph::add(model) accuracy failed: " + graph_acc.note);

        std::cout << "GRAPH_MODEL model=" << model_name << " case=add_model_tensor_or_contract"
                  << " signature=\"" << graph_sig << "\" accuracy=\"" << graph_acc.note
                  << "\" tensor_io=\"" << tensor_io_stats_string(tensor_io) << "\"\n";
        std::cout << "GRAPH_MODEL_PARITY model=" << model_name << " status=OK signature=\""
                  << graph_sig << "\" model_accuracy=\"" << model_acc.note << "\" graph_accuracy=\""
                  << graph_acc.note << "\"\n";

        if (should_run_stress_cases(tar, packs.size())) {
          simaai::neat::Model lifetime_verify_model(
              tar.string(), canonical_model_options(BoxDecodeRunMode::NoModel));
          const simaai::neat::Sample lifetime_sample = run_graph_lifetime_sample(img_bgr, tar);
          const AccuracyResult lifetime_acc = run_framework_boxdecode_accuracy(
              lifetime_sample, lifetime_verify_model, img_bgr, BoxDecodeRunMode::NoModel);
          require(lifetime_acc.ok, "Graph lifetime accuracy failed: " + lifetime_acc.note);
          std::cout << "GRAPH_MODEL_LIFETIME model=" << model_name << " status=OK accuracy=\""
                    << lifetime_acc.note << "\"\n";

          run_repeated_build_check(img_bgr, model, model_name);

          const simaai::neat::Sample connected_sample =
              run_graph_connected_model_sample(img_bgr, model);
          const AccuracyResult connected_acc = run_framework_boxdecode_accuracy(
              connected_sample, model, img_bgr, BoxDecodeRunMode::NoModel);
          require(connected_acc.ok, "Connected Graph accuracy failed: " + connected_acc.note);
          std::cout << "GRAPH_MODEL_CONNECTED model=" << model_name << " status=OK accuracy=\""
                    << connected_acc.note << "\"\n";

          simaai::neat::Model image_model(tar.string(), graph_image_model_options());
          const simaai::neat::Sample image_sample =
              run_graph_image_add_model_sample(img_bgr, image_model);
          const AccuracyResult image_acc = run_framework_boxdecode_accuracy(
              image_sample, image_model, img_bgr, BoxDecodeRunMode::NoModel);
          require(image_acc.ok, "Image Graph::add(model) accuracy failed: " + image_acc.note);
          std::cout << "GRAPH_MODEL_IMAGE model=" << model_name << " status=OK accuracy=\""
                    << image_acc.note << "\"\n";

          simaai::neat::Model stage_model(tar.string(), graph_image_model_options());
          const simaai::neat::Sample stage_sample =
              run_graph_explicit_stage_fragments_sample(img_bgr, stage_model);
          const AccuracyResult stage_acc = run_framework_boxdecode_accuracy(
              stage_sample, stage_model, img_bgr, BoxDecodeRunMode::NoModel);
          require(stage_acc.ok, "Stage-fragment Graph accuracy failed: " + stage_acc.note);
          std::cout << "GRAPH_MODEL_STAGE_FRAGMENTS model=" << model_name
                    << " status=OK accuracy=\"" << stage_acc.note << "\"\n";
        }

        executed += 1;
      } catch (const std::exception& ex) {
        failures += 1;
        std::cerr << "[FAIL] model=" << model_name << " err=" << ex.what() << "\n";
      }
    }

    require(executed > 0, "no model packs executed");
    require(failures == 0, "Graph model composition failed for one or more models");
    std::cout << "[OK] graph_migration_unified_yolov8_graph_model_composition_test models="
              << executed << "\n";
    return 0;
  } catch (const SkipTest& e) {
    std::cout << "[SKIP] " << e.what() << "\n";
    return skip_long_test(e.what());
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
#endif
}
