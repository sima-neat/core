#include "yolov8_variant_route_matrix_common.inc"

int main(int argc, char** argv) {
#if !defined(SIMA_WITH_OPENCV)
  (void)argc;
  (void)argv;
  return skip_long_test(
      "OpenCV required for graph_migration_unified_yolov8_variant_route_matrix_test");
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
    const std::string processcvu_run_target = resolve_processcvu_run_target(argc, argv);
    const std::string processcvu_placement = resolve_processcvu_placement(argc, argv);
    const BoxDecodeRunMode boxdecode_mode = resolve_boxdecode_run_mode(argc, argv);
    const AsyncSelection async_selection = resolve_async_selection(argc, argv);
    const int async_queue_depth = resolve_async_queue_depth(argc, argv);
    const bool processmla_defer_output_invalidate =
        resolve_processmla_defer_output_invalidate(argc, argv);
    const std::string prepared_runner_mode = resolve_prepared_runner_mode(argc, argv);
    const int prepared_runner_ring_depth = resolve_prepared_runner_ring_depth(argc, argv);
    const bool prepared_runner_profile = has_flag(argc, argv, "--prepared-runner-profile");
    const std::string prepared_runner_dequant_flags =
        resolve_prepared_runner_dequant_flags(argc, argv);
    const int frames = resolve_frames(argc, argv);
    if (single_model_tar.has_value()) {
      const fs::path tar = fs::absolute(*single_model_tar);
      require(fs::exists(tar), "single model tar not found: " + tar.string());
      packs.push_back(tar);
    } else {
      fs::path variants_dir = resolve_variants_dir(root, argc, argv);
      packs = collect_model_packs(variants_dir);
      if (packs.empty()) {
        // Mandatory test — download the six yolov8n variants on demand via
        // the shared helper (uses sima-cli for the OAuth-gated
        // docs.sima.ai endpoint) and rescan.
        variants_dir = sima_yolov8_test::ensure_yolov8n_drive_variants(root);
        packs = collect_model_packs(variants_dir);
      }
      require(!packs.empty(), "no model packs found in " + variants_dir.string() +
                                  " even after download attempt; check sima-cli login and "
                                  "SIMA_YOLOV8N_VARIANTS_BASE_URL");
    }

    const cv::Mat img_bgr = sima_yolov8_test::load_people_image_or_skip(root);
    int failures = 0;

    for (const auto& tar : packs) {
      try {
        auto model_options = canonical_model_options(boxdecode_mode);
        if (boxdecode_mode == BoxDecodeRunMode::Model) {
          model_options.boxdecode_original_width = img_bgr.cols;
          model_options.boxdecode_original_height = img_bgr.rows;
        }
        apply_processcvu_placement(processcvu_placement, &model_options.processcvu);
        apply_async_options(async_selection, async_queue_depth, &model_options, nullptr);
        apply_processmla_options(processmla_defer_output_invalidate, &model_options, nullptr);
        apply_prepared_runner_options(prepared_runner_mode, prepared_runner_ring_depth,
                                      prepared_runner_profile, prepared_runner_dequant_flags,
                                      &model_options, nullptr);
        simaai::neat::Model model(tar.string(), model_options);
        simaai::neat::Model::RouteOptions route_opt;
        route_opt.processcvu_requested_run_target = processcvu_run_target;
        apply_processcvu_placement(processcvu_placement, &route_opt.processcvu);
        apply_async_options(async_selection, async_queue_depth, nullptr, &route_opt);
        apply_processmla_options(processmla_defer_output_invalidate, nullptr, &route_opt);
        apply_prepared_runner_options(prepared_runner_mode, prepared_runner_ring_depth,
                                      prepared_runner_profile, prepared_runner_dequant_flags,
                                      nullptr, &route_opt);
        (void)model.info();
        (void)model.input_spec();
        (void)model.inference();
        std::cout << "MODEL_INIT_OK model=" << tar.filename().string()
                  << " backend=" << processcvu_run_target << " placement="
                  << (processcvu_placement.empty() ? "default" : processcvu_placement)
                  << " async_mode=" << async_selection_name(async_selection)
                  << " processcvu_async=" << (async_selection.processcvu ? 1 : 0)
                  << " processmla_async=" << (async_selection.processmla ? 1 : 0)
                  << " async_queue_depth=" << async_queue_depth << " prepared_runner_mode="
                  << (prepared_runner_mode.empty() ? "off" : prepared_runner_mode)
                  << " prepared_runner_dequant_flags="
                  << (prepared_runner_dequant_flags.empty() ? "default"
                                                            : prepared_runner_dequant_flags)
                  << " frames=" << frames
                  << " boxdecode_mode=" << boxdecode_run_mode_name(boxdecode_mode) << "\n";

        simaai::neat::pipeline_internal::reset_tensor_io_stats();
        const auto tensor_io_before = simaai::neat::pipeline_internal::snapshot_tensor_io_stats();
        const auto run_t0 = std::chrono::steady_clock::now();
        const simaai::neat::Sample infer_sample =
            run_canonical_model_sample(img_bgr, model, route_opt, frames);
        const auto run_t1 = std::chrono::steady_clock::now();
        const double run_ms = std::chrono::duration<double, std::milli>(run_t1 - run_t0).count();
        const double fps =
            run_ms > 0.0 ? (static_cast<double>(std::max(frames, 1)) * 1000.0 / run_ms) : 0.0;
        std::cout << "FPS model=" << tar.filename().string() << " backend=" << processcvu_run_target
                  << " placement="
                  << (processcvu_placement.empty() ? "default" : processcvu_placement)
                  << " async_mode=" << async_selection_name(async_selection)
                  << " processcvu_async=" << (async_selection.processcvu ? 1 : 0)
                  << " processmla_async=" << (async_selection.processmla ? 1 : 0)
                  << " async_queue_depth=" << async_queue_depth << " prepared_runner_mode="
                  << (prepared_runner_mode.empty() ? "off" : prepared_runner_mode)
                  << " prepared_runner_dequant_flags="
                  << (prepared_runner_dequant_flags.empty() ? "default"
                                                            : prepared_runner_dequant_flags)
                  << " frames=" << std::max(frames, 1) << " run_ms=" << run_ms << " fps=" << fps
                  << "\n";
        const auto tensor_io_after = simaai::neat::pipeline_internal::snapshot_tensor_io_stats();
        const auto tensor_io = tensor_io_delta(tensor_io_before, tensor_io_after);

        if (boxdecode_mode == BoxDecodeRunMode::NoModel) {
          require_preprocess_meta_on_output_local(infer_sample, img_bgr.cols, img_bgr.rows,
                                                  "canonical_e2e_output");
        }
        const AccuracyResult acc =
            run_framework_boxdecode_accuracy(infer_sample, model, img_bgr, boxdecode_mode);
        require(acc.ok, "accuracy check failed: " + acc.note);

        std::cout << "E2E model=" << tar.filename().string() << " backend=" << processcvu_run_target
                  << " placement="
                  << (processcvu_placement.empty() ? "default" : processcvu_placement)
                  << " boxdecode_mode=" << boxdecode_run_mode_name(boxdecode_mode)
                  << " signature=\"" << sample_output_signature_local(infer_sample) << "\""
                  << " accuracy=\"" << acc.note << "\" tensor_io=\""
                  << tensor_io_stats_string(tensor_io) << "\"\n";
      } catch (const std::exception& ex) {
        failures += 1;
        std::cerr << "[FAIL] model=" << tar.filename().string() << " err=" << ex.what() << "\n";
      }
    }

    require(!packs.empty(), "no model packs executed");
    require(failures == 0, "canonical yolov8 e2e failed for one or more models");
    std::cout << "[OK] graph_migration_unified_yolov8_variant_route_matrix_test models="
              << packs.size() << "\n";
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
