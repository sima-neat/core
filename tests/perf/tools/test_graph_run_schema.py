#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import json
import sys
import unittest

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import graph_run_schema as schema


def valid_payload() -> dict[str, object]:
    return {
        "schema": "sima.neat.graph_run",
        "schema_version": 1,
        "producer": {"name": "neat"},
        "label": "unit",
        "metadata": {"test": "schema"},
        "graph": {
            "mode": "connected",
            "public_graph_id": 1,
            "public_graph_version": 2,
            "named_inputs": [
                {"name": "image", "kind": "pipeline_input", "node": "n0", "port": None, "segment": 0}
            ],
            "named_outputs": [
                {"name": "classes", "kind": "pipeline_output", "node": "n0", "port": None, "segment": 0}
            ],
            "public_view": {
                "nodes": [
                    {
                        "id": "p0",
                        "index": 0,
                        "kind": "Input",
                        "label": "image",
                        "endpoint_name": "image",
                        "input_endpoint": True,
                        "output_endpoint": False,
                        "runtime_node": "n0",
                        "source": {
                            "kind": "app_push",
                            "uri": None,
                            "endpoint": "image",
                            "details": {"media_type": "video/x-raw"},
                        },
                    },
                    {
                        "id": "p1",
                        "index": 1,
                        "kind": "Output",
                        "label": "classes",
                        "endpoint_name": "classes",
                        "input_endpoint": False,
                        "output_endpoint": True,
                        "runtime_node": "n0",
                        "sink": {
                            "kind": "appsink",
                            "uri": None,
                            "endpoint": "classes",
                            "details": {"max_buffers": 4},
                        },
                    },
                ],
                "edges": [
                    {
                        "id": "pe0",
                        "index": 0,
                        "from": "p0",
                        "to": "p1",
                        "kind": "public_endpoint",
                        "from_endpoint": "image",
                        "to_endpoint": "classes",
                        "runtime_from": "n0",
                        "runtime_to": "n0",
                        "runtime_edges": [],
                    }
                ],
            },
            "lowered_view": {
                "nodes": [
                    {
                        "id": "n0",
                        "stable_id": "segment_0.n0",
                        "backend": "pipeline",
                        "kind": "Input",
                        "compiler_generated": False,
                    }
                ],
                "edges": [
                    {"id": "pe0", "kind": "public_endpoint", "from": "n0", "to": "n0", "from_port": "image", "to_port": "classes"}
                ],
                "pipeline_segments": [],
            },
            "nodes": [
                {
                    "id": "n0",
                    "stable_id": "segment_0.n0",
                    "backend": "pipeline",
                    "kind": "Input",
                    "compiler_generated": False,
                }
            ],
            "edges": [
                {"id": "pe0", "kind": "public_endpoint", "from": "n0", "to": "n0", "from_port": "image", "to_port": "classes"}
            ],
        },
        "run": {
            "identity": {
                "uuid": "00000000-0000-4000-8000-000000000000",
                "created_at": "2026-05-20T00:00:00Z",
                "closed_at": None,
                "hostname": "unit",
                "pid": 1,
                "argv": ["unit"],
            },
            "stats": {"outputs_pulled": 1},
            "elapsed_seconds": 0.1,
            "throughput_fps": 10.0,
            "input_names": ["image"],
            "output_names": ["classes"],
            "last_error": "",
            "graph_metrics": {
                "measurement_scope": "run_lifetime",
                "aggregation": "run_lifetime",
                "latency_semantics": "sum_element_residency",
                "throughput_counting": "all_pulled_outputs",
                "throughput": {
                    "unit": "output_pulls_per_second",
                    "scope": "run_lifetime",
                    "semantics": "public_output_pulls_per_second",
                    "numerator_counter": "outputs_pulled",
                    "numerator_value": 1,
                    "denominator": "elapsed_seconds",
                    "denominator_seconds": 0.1,
                    "outputs_pulled": 1,
                    "elapsed_seconds": 0.1,
                    "outputs_per_s": 10.0,
                    "batches_per_s": 10.0,
                    "throughput_batches_per_s": 10.0,
                    "logical_batch_size": 1,
                    "logical_inferences_per_s": 10.0,
                    "multi_output_semantics": "each successful public pull increments outputs_pulled once",
                    "available": True,
                    "status": "collected",
                    "warnings": [],
                },
                "elapsed_seconds": 0.1,
                "outputs_pulled": 1,
                "throughput_fps": 10.0,
                "counters": {
                    "inputs_enqueued": 1,
                    "inputs_dropped": 0,
                    "inputs_pushed": 1,
                    "outputs_ready": 1,
                    "outputs_pulled": 1,
                    "outputs_dropped": 0,
                },
                "power": {
                    "enabled": True,
                    "samples": 0,
                    "duration_seconds": 0.0,
                    "total_avg_watts": 0.0,
                    "total_min_watts": 0.0,
                    "total_max_watts": 0.0,
                    "energy_joules": 0.0,
                    "rails": [],
                },
            },
            "graph_e2e_latency_ms": {
                "available": True,
                "status": "collected",
                "correlation_reliable": True,
                "survivor_biased": False,
                "unit": "milliseconds",
                "scope": "graph_application",
                "source": "runtime_graph_sample_timing",
                "semantics": "queue_inclusive_public_graph_push_to_public_output_pull",
                "interpretation": "single-flight approximates app-visible latency",
                "count": 1,
                "avg_ms": 0.4,
                "p50_ms": 0.4,
                "p90_ms": 0.4,
                "p95_ms": 0.4,
                "p99_ms": 0.4,
                "max_ms": 0.4,
                "warnings": [],
            },
            "output_materialization": {
                "available": True,
                "status": "metadata_only",
                "claim_status": "requested_mode_only_materialization_timing_unavailable",
                "claim_scope": "requested_output_memory_mode_only",
                "output_memory_mode": "owned",
                "semantics": "owned_output_copy",
                "timing_available": False,
                "runtime_resolved_mode_available": False,
                "copy_map_timing_available": False,
                "prepare_output_cpu_visible": False,
                "note": "Output memory/materialization mode is reported.",
                "warnings": [
                    "Output materialization reports requested mode only; copy timing unavailable."
                ],
            },
            "node_metrics": [
                {
                    "node_id": "n0",
                    "runtime_node": "n0",
                    "runtime_node_id": 0,
                    "public_node_ids": ["p0", "p1"],
                    "pipeline_segment_id": 0,
                    "kind": "Input",
                    "label": "image",
                    "element_names": ["n0_identity"],
                    "latency_semantics": "sum_element_residency",
                    "aggregation": "run_lifetime",
                    "latency_ms": {
                        "samples": 1,
                        "total_ms": 0.2,
                        "avg_ms": 0.2,
                        "min_ms": 0.2,
                        "max_ms": 0.2,
                        "min_max_available": True,
                    },
                    "elements": [
                        {
                            "name": "n0_identity",
                            "latency_ms": {
                                "samples": 1,
                                "total_ms": 0.2,
                                "avg_ms": 0.2,
                                "min_ms": 0.2,
                                "max_ms": 0.2,
                                "min_max_available": True,
                            },
                        }
                    ],
                    "plugins": [
                        {
                            "name": "A65:n0_identity",
                            "backend": "A65",
                            "phase": "Run",
                            "kernel_name": "identity",
                            "stage_name": "n0_identity",
                            "gst_element_name": "n0_identity",
                            "run_id_hash": None,
                            "pipeline_segment_id": 0,
                            "runtime_node_id": 0,
                            "public_node_id": None,
                            "public_node_ids": ["p0", "p1"],
                            "physical_input_index": -1,
                            "output_slot": -1,
                            "calls": 1,
                            "latency_ms": {
                                "samples": 1,
                                "total_ms": 0.1,
                                "avg_ms": 0.1,
                                "min_ms": 0.1,
                                "max_ms": 0.1,
                                "min_max_available": True,
                            },
                        }
                    ],
                }
            ],
            "plugin_metrics_unattributed": [
                {
                    "name": "MLA:unknown",
                    "backend": "MLA",
                    "phase": "Run",
                    "kernel_name": "unknown",
                    "stage_name": "unknown_stage",
                    "gst_element_name": None,
                    "run_id_hash": None,
                    "pipeline_segment_id": None,
                    "runtime_node_id": None,
                    "public_node_id": None,
                    "public_node_ids": [],
                    "physical_input_index": 0,
                    "output_slot": 0,
                    "calls": 1,
                    "latency_ms": {
                        "samples": 1,
                        "total_ms": 0.3,
                        "avg_ms": 0.3,
                        "min_ms": 0.3,
                        "max_ms": 0.3,
                        "min_max_available": True,
                    },
                    "mapping_error": "unattributed: unit test",
                }
            ],
        },
    }


class GraphRunSchemaTest(unittest.TestCase):
    def test_valid_payload(self) -> None:
        schema.validate_graph_run(valid_payload())

    def test_missing_public_view_node_field_fails(self) -> None:
        bad = valid_payload()
        del bad["graph"]["public_view"]["nodes"][0]["runtime_node"]  # type: ignore[index]
        with self.assertRaises(schema.SchemaError):
            schema.validate_graph_run(bad)

    def test_bad_runtime_edge_id_fails(self) -> None:
        bad = valid_payload()
        bad["graph"]["public_view"]["edges"][0]["runtime_edges"] = ["not_edge"]  # type: ignore[index]
        with self.assertRaises(schema.SchemaError):
            schema.validate_graph_run(bad)

    def test_node_metrics_must_not_contain_power(self) -> None:
        bad = valid_payload()
        bad["run"]["node_metrics"][0]["power"] = {"enabled": True}  # type: ignore[index]
        with self.assertRaises(schema.SchemaError):
            schema.validate_graph_run(bad)

    def test_plugin_metrics_must_not_contain_power(self) -> None:
        bad = valid_payload()
        bad["run"]["node_metrics"][0]["plugins"][0]["power"] = {"enabled": True}  # type: ignore[index]
        with self.assertRaises(schema.SchemaError):
            schema.validate_graph_run(bad)

    def test_bad_graph_metric_counter_fails(self) -> None:
        bad = valid_payload()
        bad["run"]["graph_metrics"]["outputs_pulled"] = "one"  # type: ignore[index]
        with self.assertRaises(schema.SchemaError):
            schema.validate_graph_run(bad)

    def test_graph_e2e_latency_requires_units_and_semantics(self) -> None:
        bad = valid_payload()
        del bad["run"]["graph_e2e_latency_ms"]["unit"]  # type: ignore[index]
        with self.assertRaises(schema.SchemaError):
            schema.validate_graph_run(bad)

    def test_graph_e2e_latency_requires_quality_flags(self) -> None:
        bad = valid_payload()
        del bad["run"]["graph_e2e_latency_ms"]["correlation_reliable"]  # type: ignore[index]
        with self.assertRaises(schema.SchemaError):
            schema.validate_graph_run(bad)

    def test_throughput_summary_requires_numerator(self) -> None:
        bad = valid_payload()
        del bad["run"]["graph_metrics"]["throughput"]["numerator_value"]  # type: ignore[index]
        with self.assertRaises(schema.SchemaError):
            schema.validate_graph_run(bad)

    def test_output_materialization_requires_semantics(self) -> None:
        bad = valid_payload()
        del bad["run"]["output_materialization"]["semantics"]  # type: ignore[index]
        with self.assertRaises(schema.SchemaError):
            schema.validate_graph_run(bad)

    def test_output_materialization_requires_zero_copy_caveat(self) -> None:
        bad = valid_payload()
        del bad["run"]["output_materialization"]["copy_map_timing_available"]  # type: ignore[index]
        with self.assertRaises(schema.SchemaError):
            schema.validate_graph_run(bad)

    def test_load_graph_run(self) -> None:
        path = Path("/tmp/graph_run_schema_test.json")
        path.write_text(json.dumps(valid_payload()))
        try:
            loaded = schema.load_graph_run(path)
            self.assertEqual(loaded["schema"], "sima.neat.graph_run")
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
