#!/usr/bin/env python3
"""Guard the Core -> Scheduler -> Runner cold numerical boundary."""
from pathlib import Path
import argparse
import py_compile
import tempfile
import unittest
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]

# Source lint is a host/build-tree gate, not an installed runtime test. These
# checks preserve the unique layering rules from the retired C++ source scanner.
GUARDED_FILES = (
    'src/model/Model.cpp',
    'src/model/ModelPack.cpp',
    'src/pipeline/internal/contract/ContractCompiler.cpp',
    'src/nodes/sima/Preproc.cpp',
    'src/nodes/sima/Quant.cpp',
    'src/nodes/sima/Tess.cpp',
    'src/nodes/sima/QuantTess.cpp',
    'src/nodes/sima/Dequant.cpp',
    'src/nodes/sima/Detess.cpp',
    'src/nodes/sima/DetessDequant.cpp',
)
FORBIDDEN_TOKENS = (
    'LogicalInputStaticSpec ',
    'LogicalTensorStaticSpec ',
    'InputBindingStaticSpec ',
    'PhysicalBufferStaticSpec ',
    'StageOutputRoute{',
)
PROCESSCVU_HELPER_GUARDED_FILES = (
    'src/model/Model.cpp',
    'src/model/ModelPack.cpp',
    'src/pipeline/internal/contract/ContractCompiler.cpp',
    'src/nodes/sima/Preproc.cpp',
    'src/nodes/sima/Dequant.cpp',
    'src/nodes/sima/Detess.cpp',
    'src/nodes/sima/DetessDequant.cpp',
)
PROCESSCVU_HELPER_FORBIDDEN_TOKENS = (
    'ProcessCvuSingleOutputFactsSpec ',
    'ProcessCvuPackedRouteEntry ',
    'CompiledProcessCvuRuntimeConfig ',
    'model/internal/ProcessCvuFamily.h',
)
WRAPPER_ADAPTER_GUARDED_FILES = (
    'src/nodes/sima/Preproc.cpp',
    'src/nodes/sima/Quant.cpp',
    'src/nodes/sima/Tess.cpp',
    'src/nodes/sima/QuantTess.cpp',
)
WRAPPER_ADAPTER_FORBIDDEN_TOKENS = (
    'build_processcvu_payload_from_options(',
    'build_processcvu_facts_from_options(',
    'build_processcvu_payload_from_runtime_config(',
    'build_processcvu_facts_from_runtime_config(',
)
DELETED_WRAPPER_TOKENS = (
    'build_preproc_contract_from_mpk_contract(',
    'build_preadapter_contract_from_mpk_contract(',
    'build_detess_contract_from_mpk_contract(',
    'build_dequant_contract_from_mpk_contract(',
    'build_detessdequant_contract_from_mpk_contract(',
    'build_preproc_compiled_contract(',
    'build_processcvu_compiled_contract_from_mpk_stage(',
    'build_processcvu_compiled_contract_from_runtime(',
    'build_processcvu_node_contract_from_runtime_config(',
    'build_preproc_runtime_config_from_options(',
    'build_preproc_single_handoff_facts_from_payload(',
    'build_mla_stage_payload_from_runtime_properties(',
    'build_boxdecode_payload(',
    'build_processcvu_contract_from_mpk_preproc(',
    'build_processcvu_contract_from_mpk_preadapter(',
    'build_processcvu_contract_from_mpk_detess(',
    'build_processcvu_contract_from_mpk_dequant(',
    'build_processcvu_contract_from_mpk_detessdequant(',
    'build_model_managed_preadapter_contract(',
    'build_model_managed_preproc_contract(',
    'build_model_managed_detess_contract(',
    'build_model_managed_dequant_contract(',
    'build_model_managed_detessdequant_contract(',
    'build_model_managed_dequant_processcvu_contract(',
    'build_transport_compiled_contract(',
)
ADAPTER_HEADER_ALLOWED_PATHS = (
    'src/nodes/sima/Quant.cpp',
    'src/nodes/sima/Tess.cpp',
    'src/nodes/sima/QuantTess.cpp',
    'src/nodes/sima/VisualFrontend.cpp',
    'src/pipeline/internal/sima/stagesemantics/ProcessCvuRuntimeConfigAdapter.cpp',
    'tests/unit_testing/unit_contract_compiler_processcvu_test.cpp',
    'tests/unit_testing/unit_dequant_processcvu_contract_equivalence_test.cpp',
    'tests/unit_testing/unit_dequant_node_fragment_test.cpp',
)
MODELPACK_PROCESSCVU_FORBIDDEN_TOKENS = (
    'build_processcvu_mpk_preproc_compile_inputs(',
    'build_processcvu_mpk_preadapter_compile_inputs(',
    'build_processcvu_mpk_detess_compile_inputs(',
    'build_processcvu_mpk_dequant_compile_inputs(',
    'build_processcvu_mpk_detessdequant_compile_inputs(',
)


def validate_source_root(root):
    required = ("CMakeLists.txt", "tests/CMakeLists.txt", "src/gst/GstInit.cpp",
                *GUARDED_FILES, *PROCESSCVU_HELPER_GUARDED_FILES,
                *WRAPPER_ADAPTER_GUARDED_FILES)
    for relative in required:
        if not (root / relative).is_file():
            raise ValueError(f"architecture source root is incomplete: {root / relative}")
    for relative in ("src", "include", "tests"):
        if not (root / relative).is_dir():
            raise ValueError(f"architecture source directory is missing: {root / relative}")


def reject_tokens(text, tokens, path):
    for token in tokens:
        if token in text:
            raise ValueError(f"{path}: forbidden architecture dependency {token}")


def validate_client_boundary(root):
    # This supplements actual installed compile/dependency-file isolation; it
    # cannot replace that gate. Match preprocessing directives, not comments.
    forbidden = re.compile(r'^\s*#\s*include\s*[<"](?:simaai/(?:kernels/|codecs/|runtime/(?!gstreamer/))|neat/video_frame_layout\.h)', re.MULTILINE)
    for directory in ("src", "include"):
        for path in (root / directory).rglob("*"):
            if path.suffix in (".h", ".cpp", ".inc") and forbidden.search(path.read_text()):
                raise ValueError(f"{path}: lower development header bypasses Scheduler client")


def validate_layering(root):
    validate_source_root(root)
    validate_client_boundary(root)
    for files, forbidden in (
        (GUARDED_FILES, FORBIDDEN_TOKENS),
        (PROCESSCVU_HELPER_GUARDED_FILES, PROCESSCVU_HELPER_FORBIDDEN_TOKENS),
        (WRAPPER_ADAPTER_GUARDED_FILES, WRAPPER_ADAPTER_FORBIDDEN_TOKENS),
        (("src/model/ModelPack.cpp",), MODELPACK_PROCESSCVU_FORBIDDEN_TOKENS),
    ):
        for relative in files:
            reject_tokens((root / relative).read_text(), forbidden, relative)
    restricted_header = '#include "pipeline/internal/sima/stagesemantics/ProcessCvuRuntimeConfigAdapter.h"'
    for directory in ("src", "include", "tests"):
        for path in (root / directory).rglob("*"):
            if not path.is_file() or path.relative_to(root).as_posix() == \
                    "tests/tools/check_kernel_runner_default.py":
                continue
            # Bytecode is generated, not source, and can contain this checker's
            # own forbidden-token constants. Keep checking real source even in
            # cache directories; do not exclude entire trees or ignored files.
            if path.suffix in (".pyc", ".pyo"):
                continue
            # Preserve the former guard's complete tree coverage, including .inc
            # fragments and extensionless support files. Decode fixture bytes
            # permissively: an unrelated binary is not a source decoding failure.
            text = path.read_bytes().decode("utf-8", errors="replace")
            relative = path.relative_to(root).as_posix()
            reject_tokens(text, DELETED_WRAPPER_TOKENS, relative)
            if restricted_header in text and relative not in ADAPTER_HEADER_ALLOWED_PATHS:
                raise ValueError(f"{relative}: restricted runtime-config adapter include")


def validate_renderer_startup(text):
    """Source contract only: device execution is qualified by the real fixture."""
    if '#include "gst/GstInit.h"' not in text:
        raise ValueError("renderer fixture must include the public initialization header")
    if re.search(r"\bgst_init\s*\(", text) or "SIMA_ALLOW_GST_INIT" in text:
        raise ValueError("renderer fixture must not bypass Core initialization")
    main = text[text.index("int main("):]
    init = main.index("neat::gst_init_once();")
    if not main.index("core_render_test::require_synchronous_teardown_environment();") < init:
        raise ValueError("synchronous teardown qualification must be checked before initialization")
    if not init < main.index("provider_capture_barrier();") < main.index("native_scope_started = true;"):
        raise ValueError("Core initialization must precede provider capture and native ownership")


class KernelRunnerDefaultContract(unittest.TestCase):
    def test_unique_layering_boundaries(self):
        validate_layering(ROOT)

    def test_client_boundary_rejects_each_lower_header(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "src").mkdir()
            (root / "include").mkdir()
            source = root / "src/probe.cpp"
            for header in ("simaai/kernels/types.h", "simaai/runtime/executor.h",
                           "simaai/runtime/memory/segment_types.h", "simaai/codecs/decoder.h",
                           "neat/video_frame_layout.h"):
                source.write_text('#include <' + header + '>\n')
                with self.subTest(header=header), self.assertRaisesRegex(ValueError, "lower development header"):
                    validate_client_boundary(root)
            source.write_text('#include <simaai/scheduler/client/kernel.h>\n')
            validate_client_boundary(root)

    def test_explicit_incomplete_source_root_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "source root is incomplete"):
                validate_source_root(Path(temporary))

    def test_incomplete_root_command_fails_not_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            completed = subprocess.run(
                [sys.executable, __file__, "--source-root", temporary],
                capture_output=True, text=True, check=False)
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("architecture source root is incomplete", completed.stderr)
        self.assertNotIn("OK", completed.stdout)

    def test_restricted_header_allowlist_is_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in ("CMakeLists.txt", "tests/CMakeLists.txt", "src/gst/GstInit.cpp",
                             *GUARDED_FILES, *PROCESSCVU_HELPER_GUARDED_FILES,
                             *WRAPPER_ADAPTER_GUARDED_FILES):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            (root / "include").mkdir()
            header = '#include "pipeline/internal/sima/stagesemantics/ProcessCvuRuntimeConfigAdapter.h"'
            allowed = root / ADAPTER_HEADER_ALLOWED_PATHS[0]
            allowed.write_text(header)
            validate_layering(root)
            misplaced = root / "src/unowned.cpp"
            misplaced.write_text(header)
            with self.assertRaisesRegex(ValueError, "restricted runtime-config adapter include"):
                validate_layering(root)
            misplaced.unlink()
            for relative in ("src/injected.inc", "tests/extensionless-support"):
                fragment = root / relative
                fragment.write_text(DELETED_WRAPPER_TOKENS[0])
                with self.assertRaisesRegex(ValueError, "forbidden architecture dependency"):
                    validate_layering(root)
                fragment.unlink()

    def test_layering_rejects_each_removed_dependency(self):
        # Each rule has an independent mutation control, not only a clean-tree pass.
        for token in (*FORBIDDEN_TOKENS, *PROCESSCVU_HELPER_FORBIDDEN_TOKENS,
                      *WRAPPER_ADAPTER_FORBIDDEN_TOKENS, *DELETED_WRAPPER_TOKENS,
                      *MODELPACK_PROCESSCVU_FORBIDDEN_TOKENS):
            with self.subTest(token=token):
                with self.assertRaisesRegex(ValueError, "forbidden architecture dependency"):
                    reject_tokens("before " + token + " after", (token,), "mutated.cpp")

    def test_layering_ignores_bytecode_but_checks_source_in_cache_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for relative in ("CMakeLists.txt", "tests/CMakeLists.txt", "src/gst/GstInit.cpp",
                             *GUARDED_FILES, *PROCESSCVU_HELPER_GUARDED_FILES,
                             *WRAPPER_ADAPTER_GUARDED_FILES):
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            (root / "include").mkdir()
            checker = root / "tests/tools/check_kernel_runner_default.py"
            checker.parent.mkdir(parents=True)
            checker.write_text("forbidden = " + repr(DELETED_WRAPPER_TOKENS) + "\n")
            bytecode = Path(py_compile.compile(str(checker), doraise=True)).read_bytes()
            self.assertIn(DELETED_WRAPPER_TOKENS[0].encode(), bytecode)
            for directory in ("src", "include", "tests"):
                cache = root / directory / "__pycache__"
                cache.mkdir(exist_ok=True)
                for relative in ("__pycache__/probe.pyc", "probe.pyc", "probe.pyo"):
                    (root / directory / relative).write_bytes(bytecode)
                validate_layering(root)
                for name in ("probe.py", "probe.cpp", "probe.inc", "extensionless-support"):
                    source = cache / name
                    source.write_text(DELETED_WRAPPER_TOKENS[0])
                    with self.subTest(directory=directory, source=name), \
                            self.assertRaisesRegex(ValueError, "forbidden architecture dependency"):
                        validate_layering(root)
                    source.unlink()

    def test_renderer_uses_public_cold_initialization(self):
        text = (ROOT / "tests/e2e_pipelines/obj_detection/"
                "core_scheduler_yolov8_render_integration_test.cpp").read_text()
        validate_renderer_startup(text)
        for mutation in (
            text.replace('neat::gst_init_once();', 'gst_init(nullptr, nullptr);'),
            text.replace('neat::gst_init_once();',
                         'setenv("SIMA_ALLOW_GST_INIT", "1", 1); neat::gst_init_once();'),
            text.replace('neat::gst_init_once();', '')
                .replace('provider_capture_barrier();',
                         'provider_capture_barrier(); neat::gst_init_once();'),
            text.replace('#include "gst/GstInit.h"', ''),
            text.replace('core_render_test::require_synchronous_teardown_environment();', ''),
            text.replace('core_render_test::require_synchronous_teardown_environment();', '')
                .replace('neat::gst_init_once();',
                         'neat::gst_init_once(); '
                         'core_render_test::require_synchronous_teardown_environment();'),
        ):
            with self.subTest(mutation=mutation[-1800:]):
                with self.assertRaises((ValueError, IndexError)):
                    validate_renderer_startup(mutation)

    def test_renderer_observes_plain_and_driver_streams(self):
        text = (ROOT / "tests/support/core_render_retirement.h").read_text()
        self.assertIn('inputstream_pipeline_teardown_policy(compiled_pipeline)', text)
        self.assertIn('driver && policy != Policy::MustReachNull', text)
        capture = text[text.index('void capture_core('):text.index('static void check_outer_stop(')]
        self.assertIn('owners_.push_back(core);', capture)
        self.assertIn('core->pipeline.stream.diag_ctx()', capture)
        self.assertIn('is_driver_stream(diag->pipeline_string,', capture)
        self.assertLess(capture.index('owners_.push_back(core);'), capture.index('is_driver_stream('))
        self.assertIn('++streams_;\n        if (driver)\n          ++driver_streams_;', capture)
        for factory in ('neatkernelrunner', 'neatprocessmla', 'neatdecoder', 'neatencoder'):
            self.assertNotIn('"' + factory + '"', text)
        fixture = (ROOT / 'tests/e2e_pipelines/obj_detection/'
                   'core_scheduler_yolov8_render_integration_test.cpp').read_text()
        self.assertIn('" total_streams=" << retirement.stream_count()', fixture)
        self.assertIn('" driver_streams=" << retirement.driver_stream_count()', fixture)
        cmake = (ROOT / 'tests/CMakeLists.txt').read_text()
        registration = cmake[cmake.index('# Real Core detections enter'):]
        registration = registration[:registration.index('endif()')]
        self.assertIn('"SIMA_GST_TEARDOWN_DEFER_NO_FLUSH=0"', registration)
        self.assertIn('"SIMA_GST_TEARDOWN_ASYNC=0"', registration)

    def test_synchronous_transport_fallback_is_always_diagnostic(self):
        # Source-bound qualification control, not a substitute for actual
        # target stop traces. Both explicit zero settings are necessary: either
        # early-defer or async-only otherwise permits silent reaper ownership.
        text = (ROOT / 'src/pipeline/gst/GstDiagnosticsUtil.cpp').read_text()
        body = text[text.index('void stop_and_unref_no_flush('):]
        self.assertIn('!must_reach_null && env_bool("SIMA_GST_TEARDOWN_DEFER_NO_FLUSH", '
                      '!prefer_synchronous)', body)
        self.assertIn('!prefer_synchronous && env_bool("SIMA_GST_TEARDOWN_ASYNC", false)', body)
        sync = body[body.index('if (!async_only)'):]
        self.assertIn('if (result.status == TeardownStatus::Complete)\n      return;', sync)
        self.assertLess(sync.index('warn_incomplete_teardown(local, result, timeout_ms, '
                                   '!must_reach_null);'),
                        sync.index('enqueue_teardown(local, /*flush=*/false);'))
        diagnostic = text[text.index('void warn_incomplete_teardown('):text.index('void reaper_main()')]
        self.assertIn('std::cerr << "[WARN] stop_and_unref(): "', diagnostic)
        self.assertIn('deferring to reaper.', diagnostic)

    def test_public_initialization_preserves_manual_init_diagnostic(self):
        text = (ROOT / "src/gst/GstInit.cpp").read_text()
        body = text[text.index("void gst_init_once()") :]
        self.assertIn('env_bool("SIMA_ALLOW_GST_INIT", false)', body)
        self.assertIn('if (already_initialized && !allow_manual)', body)
        diagnostic = body[body.index('if (already_initialized && !allow_manual)'):
                          body.index('if (env_bool("SIMA_GST_SUPPRESS_DEVICE_LOGS"')]
        self.assertIn('This bypasses plugin path setup and can load the wrong plugins.', diagnostic)
        self.assertIn('throw std::runtime_error(msg);', diagnostic)
        self.assertLess(body.index('throw std::runtime_error(msg);'),
                        body.index('gst_init(&argc, &argv);'))

    def test_public_initialization_selects_plugins_before_gstreamer(self):
        text = (ROOT / "src/gst/GstInit.cpp").read_text()
        body = text[text.index("void gst_init_once()") :]
        self.assertIn('g_getenv("SIMA_GST_PLUGIN_DIR")', body)
        self.assertLess(body.index('g_setenv("GST_PLUGIN_PATH_1_0", third_party.c_str(), TRUE)'),
                        body.index('preload_third_party_allocator(third_party);'))
        self.assertLess(body.index('enforce_third_party_allocator(third_party);'),
                        body.index('gst_init(&argc, &argv);'))

    def test_no_execution_feature_switch(self):
        removed = ("SIMANEAT_" + "KERNEL_RUNNER_INTEGRATION", "SIMA_CORE_" + "KERNEL_RUNNER")
        files = [ROOT / "CMakeLists.txt", ROOT / "build.sh", ROOT / "tests/CMakeLists.txt"]
        files.extend((ROOT / "src").rglob("*.cpp"))
        files.extend((ROOT / "src").rglob("*.h"))
        for path in files:
            text = path.read_text()
            for token in removed:
                with self.subTest(path=str(path.relative_to(ROOT)), token=token):
                    self.assertFalse(token in text, f"Removed execution switch {token} remains in {path.relative_to(ROOT)}")
        self.assertNotIn("--kernel-runner", (ROOT / "build.sh").read_text())

    def test_installed_scheduler_contract_is_required(self):
        text = (ROOT / "CMakeLists.txt").read_text()
        config = (ROOT / "cmake/SimaNeatConfig.cmake.in").read_text()
        self.assertIn("find_package(SimaScheduler CONFIG REQUIRED COMPONENTS frontend)", text)
        self.assertIn("find_dependency(SimaScheduler CONFIG COMPONENTS frontend)", config)
        self.assertIn("NOT TARGET SimaScheduler::frontend", config)
        for contents in (text, config):
            self.assertNotRegex(contents, r"find_(?:package|dependency)\((?:SimaRunner|SimaKernels|SimaCodecs|NeatInternals)\b")


    def test_core_uses_generic_cold_preparation(self):
        # Client tensor/error types preserve the generic vocabulary, while every
        # lower provider header and numerical implementation remains outside Core.
        forbidden = (r"simaai/kernels/", r"simaai/runtime/(?!gstreamer/)",
                     r"simaai/codecs/", r"neat/video_frame_layout\.h",
                     r"kernels::ops::", r"::make_executable\(", r"::with_image_size\(")
        for directory in (ROOT / "src", ROOT / "include"):
            for path in directory.rglob("*"):
                if path.suffix not in (".cpp", ".h"):
                    continue
                with self.subTest(path=str(path.relative_to(ROOT))):
                    for pattern in forbidden:
                        self.assertNotRegex(path.read_text(), pattern)
        argmax = (ROOT / "src/nodes/sima/SimaArgMax.cpp").read_text()
        self.assertIn("scheduler::check_kernel(", argmax)
        self.assertIn("scheduler::inspect_kernel(", argmax)
        recipe = (ROOT / "src/pipeline/internal/sima/BoxDecodeKernelRecipe.cpp").read_text()
        self.assertIn("scheduler::check_kernel(", recipe)
        self.assertIn("scheduler::inspect_kernel(", recipe)
        for operation in ('"detess"', '"boxdecode"', '"superpoint"'):
            self.assertIn(operation, recipe)

    def test_no_legacy_prepared_numerical_abi(self):
        forbidden = ("PreparedProcessCvuTypedConfig", "ProcessCvuPreparedStage",
                     "GraphProcessCvuStageRequest", "GraphProcessCvuPreparedBridgeAbi",
                     "runtime/compatibility/processcvu", "prepare_kernel_stages(")
        for directory in (ROOT / "src", ROOT / "include"):
            for path in directory.rglob("*"):
                if path.suffix not in (".h", ".cpp"):
                    continue
                for token in forbidden:
                    with self.subTest(path=str(path.relative_to(ROOT)), token=token):
                        self.assertFalse(token in path.read_text(), f"Removed prepared ABI {token} remains in {path}")
        integration = (ROOT / "src/pipeline/internal/sima/KernelRunnerIntegration.cpp").read_text()
        # Supplemental dependency lint only. Linked integration tests verify mode
        # forwarding and transactional attachment through this versioned facade.
        self.assertIn("gs::prepare_and_attach_tensor_stages(", integration)
        self.assertNotIn("std::visit(", integration)
        serializer = (ROOT / "src/pipeline/internal/sima/SimaPluginStaticManifest.cpp").read_text()
        # Canonical transport is normalized before recipes are dropped; the
        # accessor owns metadata backing, never checked execution recipes.
        snapshot = serializer[serializer.index("SimaPluginStaticManifest metadata_snapshot("):
                              serializer.index("class ManifestAccessorRegistry")]
        self.assertIn("if (stage.native_kernel)", snapshot)
        self.assertLess(snapshot.index("stage.payload_kind = StagePayloadKind::None"),
                        snapshot.index("stage.native_kernel.reset()"))
        self.assertIn("manifest_(metadata_snapshot(std::move(manifest)))", serializer)
        for name in ("CanonicalTransformLowering.cpp", "CanonicalVisualLowering.cpp", "PreprocFrameAdapter.cpp"):
            text = (ROOT / "src/pipeline/internal/sima" / name).read_text()
            self.assertTrue(re.search(r"(?:scheduler::|scheduler::client::)check_kernel(?:\(|;)", text), f"{name} bypasses Scheduler preparation")

    def test_factory_link_ownership_is_explicit(self):
        text = (ROOT / "CMakeLists.txt").read_text()
        self.assertIn("find_package(SimaScheduler CONFIG REQUIRED COMPONENTS frontend)", text)
        links = text[text.index("foreach(_target sima_neat_objects sima_neat sima_neat_static)"):]
        links = links[:links.index("endforeach()")]
        self.assertIn("SimaScheduler::frontend", links)
        self.assertNotIn("SimaKernels::", links)
        self.assertIn("find_dependency(SimaScheduler CONFIG COMPONENTS frontend)",
                      (ROOT / "cmake/SimaNeatConfig.cmake.in").read_text())
        internal_links = re.search(
            r"target_link_libraries\(sima_neat_internal_headers\s+INTERFACE([^)]*)\)", text)
        self.assertIsNotNone(internal_links)
        self.assertIn("SimaScheduler::frontend", internal_links.group(1))
        # Static Core clients still use the one shared Scheduler provider; they
        # do not acquire lower development archives through Core usage.

    def test_geometry_resolves_capability_before_streaming(self):
        text = (ROOT / "src/pipeline/internal/sima/BoxDecodeGeometry.cpp").read_text()
        self.assertLess(text.index("scheduler::bind_kernel_variant("),
                        text.index("recipe.resolver_factories.push_back"))
        self.assertLess(text.index("for (const auto& entry : cache->variants)"),
                        text.index("preparer.prepare("))
        self.assertIn("std::array<Entry, 4> variants", text)
        self.assertIn("preparer.prepare(", text)
        self.assertNotIn("rt::Executor", text)
        self.assertNotIn("scheduler::check_kernel(", text)

    def test_public_wire_remains_core_owned_host_work(self):
        text = (ROOT / "src/pipeline/internal/sima/BoxDecodePublication.cpp").read_text()
        self.assertIn("<simaai/scheduler/client/host_task.h>", text)
        self.assertIn("k::make_host_task(request)", text)
        self.assertIn("request.payload = std::shared_ptr<const WirePlan>", text)
        self.assertIn("request.callback = +[]", text)
        self.assertIn("return publish(*static_cast<const WirePlan*>(payload), buffers)", text)

    def test_runner_registration_precedes_startup_validation(self):
        text = (ROOT / "src/gst/GstInit.cpp").read_text()
        body = text[text.index("void gst_init_once()") :]
        self.assertLess(body.index("simaai::scheduler::client::register_kernel_runner()"),
                        body.index("validate_neat_startup_contract("))
        contract = text[text.index("void validate_neat_startup_contract(") :
                        text.index("} // namespace", text.index("void validate_neat_startup_contract("))]
        self.assertIn('validate_neat_factory_loaded("neatkernelrunner", nullptr)', contract)
        self.assertNotIn('"neatprocesscvu"', contract)
        # MLA remains a separate engine; numerical Nodes share the runner.
        self.assertIn('"neatprocessmla"', contract)
        self.assertNotIn('"neatobjectdecode"', contract)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--source-root", type=Path, default=ROOT)
    args, unittest_args = parser.parse_known_args()
    ROOT = args.source_root.resolve()
    try:
        validate_source_root(ROOT)
    except ValueError as error:
        parser.error(str(error))
    unittest.main(argv=[__file__, *unittest_args])
