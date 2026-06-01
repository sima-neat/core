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
