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
    if not init < main.index("provider_capture_barrier();") < main.index("qualify(model, image, native_scope_started);"):
        raise ValueError("Core initialization must precede provider capture and native ownership")
    qualification = text[text.index("void qualify("):text.index("int main(")]
    if not qualification.index("ConfigFiles files(") < qualification.index("native_scope_started = true;") < qualification.index("model_owner = std::make_unique<neat::Model>"):
        raise ValueError("Fixture preparation must precede native ownership containment")


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
