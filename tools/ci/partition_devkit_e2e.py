#!/usr/bin/env python3
"""Split the canonical CTest selection, then run the unchanged coverage gate."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
# CTest names that read SIMA_RESNET50_TAR through sima_test::resolve_resnet50_tar().
RESNET50_TESTS = {
    'graph_migration_phase3_lazy_dynamic_spec_test',
    'h265_decode_resnet50_e2e_test',
    'h265_rtsp_resnet50_e2e_test',
    'model_archive_load_e2e_test',
    'model_resnet50_multi_test',
    'model_unique_names_test',
    'resnet50_modelpack_direct',
    'resnet50_modelpack_jpeg',
    'resnet50_modelpack_rtsp',
    'resnet50_neatargmax_test',
}
# CTest names that read SIMA_YOLO_TAR through sima_test::resolve_yolov8s_tar().
YOLOV8S_TESTS = {
    'core_scheduler_yolov8_render_integration_test',
    'graphpipes_yolo_metadata_sender_16stream_test',
    'model_execution_plan_test',
    'model_seeded_input_capacity_defaults_test',
    'model_seeded_input_capacity_oversize_height_test',
    'model_seeded_input_capacity_oversize_width_test',
    'model_seeded_input_capacity_sample_test',
    'model_seeded_input_capacity_tensor_test',
    'model_seeded_input_capacity_test',
    'model_seeded_input_capacity_unseeded_test',
    'preproc_roi_boxdecode_moving_test',
    'seedless_yolov8_preproc_input_max_smoke_test',
    'stage_chain_cache_admission_test',
    'sync_yolov8_test',
    'yolov8_boxdecode_model_benchmark_test',
    'yolov8_dynamic_boxdecode_geometry_async_test',
    'yolov8_dynamic_boxdecode_geometry_sync_test',
    'yolov8_model_benchmark_metrics_test',
    'yolov8_model_mla_stats_test',
}
# CTest names that read SIMA_YOLOV8_VARIANTS_DIR. The two phase3 tests open named MPKs in it;
# the others load every *.tar.gz.
YOLOV8N_VARIANT_TESTS = {
    'graph_migration_legacy_preproc_yolov8_matrix_test',
    'graph_migration_legacy_yolov8_variant_route_matrix_test',
    'graph_migration_phase3_connected_yolo_route_test',
    'graph_migration_phase3_yolov8_bgr_bf16_preproc_test',
    'graph_migration_unified_preproc_yolov8_matrix_test',
    'graph_migration_unified_yolov8_graph_model_composition_test',
    'graph_migration_unified_yolov8_variant_route_matrix_test',
    'preproc_yolov8_matrix_test',
    'yolov8_variant_route_matrix_test',
}
# CTest name -> (Model Browser id, variant) for model_browser_contract_suite_test. Each test reads
# <id>_<variant>_mpk.tar.gz from SIMA_MODEL_BROWSER_DIR.
MODEL_BROWSER_SUITE = {
    'model_browser_contract_inception_v3': ('inception_v3', 'w8a8'),
    'model_browser_contract_yolox_s': ('yolox_s', 'w8a8'),
}


def partition(names, weights):
    if len(names) != len(set(names)) or len(names) < 2:
        raise ValueError('E2E sharding requires at least two unique tests')
    bins, costs = [[], []], [0, 0]
    for name in sorted(names, key=lambda n: (-weights.get(n, 10), n)):
        i = min(range(2), key=lambda i: (costs[i], i))
        bins[i].append(name)
        costs[i] += weights.get(name, 10)
    return [sorted(b) for b in bins]


# (variable, Model Browser id, variant, CTest names that read it). A package with components maps
# each component to its own variable; one download prepares all of them.
FIXTURES = (
    ('SIMA_TEST_MLA_ONLY_MODEL', 'yolo26n', 'w8a8', {'model_execution_plan_test'}),
    ('SIMA_RESNET50_TAR', 'resnet_50', 'w8a8', RESNET50_TESTS),
    ('SIMA_RESNET18_TAR', 'resnet_18', 'w8a8', {'resnet18_async_consumer_visibility_test'}),
    ('SIMA_YOLO_TAR', 'yolov8s', 'w8a8', YOLOV8S_TESTS),
    ('SIMA_DETR_RESNET50_TAR', 'detr_resnet_50', 'w8a8_mixed', {'detr_resnet50_multi_output_test'}),
    ({'backbone': 'SIMA_RF_DETR_BACKBONE_TAR', 'decoder': 'SIMA_RF_DETR_DECODER_TAR'},
     'rf_detr_small', 'w8a16_w16a16', {'rf_detr_small_pipeline_test'}),
)


def prepare_fixture(model_id, variant, output, components=()):
    argv = [sys.executable, str(ROOT / 'scripts/ci/prepare_model_browser_fixture.py'),
            '--id', model_id, '--variant', variant, '--output', str(output)]
    for component in components:
        argv += ['--component', component]
    out = subprocess.check_output(argv, text=True, timeout=900).strip()
    return out.splitlines() if components else out


def prepare_selected_fixtures(selected, env, output):
    """Prepare each fixture a selected test reads, unless its variables are already set."""
    for target, model_id, variant, consumers in FIXTURES:
        if not consumers & set(selected):
            continue
        if isinstance(target, dict):
            if all(env.get(var) for var in target.values()):
                continue
            paths = prepare_fixture(model_id, variant, output, components=tuple(target))
            for var, path in zip(target.values(), paths):
                if not env.get(var):
                    env[var] = path
        elif not env.get(target):
            env[target] = prepare_fixture(model_id, variant, output)


def prepare_model_browser_suite(selected, env, tmp):
    """Prepare only the suite packages this partition runs; a preset directory wins."""
    wanted = [MODEL_BROWSER_SUITE[n] for n in sorted(MODEL_BROWSER_SUITE) if n in selected]
    if not wanted or env.get('SIMA_MODEL_BROWSER_DIR'):
        return
    suite_dir = Path(tmp, 'model-browser-suite')
    for model_id, variant in wanted:
        prepare_fixture(model_id, variant, suite_dir)
    env['SIMA_MODEL_BROWSER_DIR'] = str(suite_dir)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=int, choices=(0, 1), required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    if not command or command[0] != 'ctest' or '-R' in command or '--tests-regex' in command:
        raise ValueError('Expected canonical CTest argv without a name regex')
    inventory = json.loads(subprocess.check_output(command + ['--show-only=json-v1'], text=True))
    names = sorted(t['name'] for t in inventory['tests'])
    weights = json.loads((ROOT / 'tests/devkit-test-durations.json').read_text())['test_seconds']
    selected = partition(names, weights)[args.index]
    coverage = Path(os.environ['COVERAGE_DIR'])
    regex = '^(' + '|'.join(re.escape(n) for n in selected) + ')$'
    # Required zero-skip accounting still discovers and validates the exact
    # partial selection and executes CTest with its original timeout/jobs/labels.
    env = os.environ.copy()
    with tempfile.TemporaryDirectory(prefix='model-browser-', dir=env.get('RUNNER_TEMP')) as tmp:
        prepare_selected_fixtures(selected, env, tmp)
        # The tests that load every MPK need a directory of their own for YOLOv8n. They take
        # SIMA_YOLOV8_SINGLE_MODEL first when it is set; the phase3 tests read only the directory,
        # so it is prepared either way.
        variants_dir = Path(tmp, 'yolov8n-variants')
        if YOLOV8N_VARIANT_TESTS & set(selected) and not env.get('SIMA_YOLOV8_VARIANTS_DIR'):
            for variant in ('w8a8', 'w8a16', 'w16a16'):
                prepare_fixture('yolov8n', variant, variants_dir)
            env['SIMA_YOLOV8_VARIANTS_DIR'] = str(variants_dir)
        prepare_model_browser_suite(set(selected), env, tmp)
        result = subprocess.run([sys.executable, str(ROOT / 'tools/ci/check_test_coverage.py'),
                    '--profile', env.get('COVERAGE_PROFILE', 'strict'),
                    '--output-dir', str(coverage), '--', *command, '-R', regex], env=env)
    coverage.mkdir(parents=True, exist_ok=True)
    (coverage / 'e2e-partition.json').write_text(json.dumps(
        {'index': args.index, 'canonical': names, 'selected': selected}, indent=2) + '\n')
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
