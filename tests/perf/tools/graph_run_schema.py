#!/usr/bin/env python3
"""Small dependency-free validator for NEAT graph-run export JSON."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping
import json
import re


class SchemaError(ValueError):
    """Raised when a graph-run JSON payload violates the expected v1 shape."""


def _raise(context: str, message: str) -> None:
    raise SchemaError(f"{context}: {message}")


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _raise(context, "expected object")
    return value


def _array(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        _raise(context, "expected array")
    return value


def _string(value: Any, context: str) -> str:
    if not isinstance(value, str):
        _raise(context, "expected string")
    return value


def _bool(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        _raise(context, "expected boolean")
    return value


def _number(value: Any, context: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        _raise(context, "expected number")
    return float(value)


def _integer(value: Any, context: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        _raise(context, "expected integer")
    return value


def _require(data: Mapping[str, Any], keys: tuple[str, ...], context: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        _raise(context, "missing required fields: " + ", ".join(missing))


def _match(value: str, pattern: str, context: str) -> None:
    if re.match(pattern, value) is None:
        _raise(context, f"value {value!r} does not match {pattern}")


def validate_named_endpoint(value: Any, context: str) -> None:
    data = _mapping(value, context)
    _require(data, ("name", "kind", "node", "port"), context)
    _string(data["name"], f"{context}.name")
    _string(data["kind"], f"{context}.kind")
    _string(data["node"], f"{context}.node")
    if data["port"] is not None:
        _string(data["port"], f"{context}.port")


def validate_public_node(value: Any, context: str) -> None:
    data = _mapping(value, context)
    _require(
        data,
        ("id", "index", "kind", "label", "input_endpoint", "output_endpoint", "runtime_node"),
        context,
    )
    _match(_string(data["id"], f"{context}.id"), r"^p[0-9]+$", f"{context}.id")
    _number(data["index"], f"{context}.index")
    _string(data["kind"], f"{context}.kind")
    _string(data["label"], f"{context}.label")
    if data.get("endpoint_name") is not None:
        _string(data["endpoint_name"], f"{context}.endpoint_name")
    _bool(data["input_endpoint"], f"{context}.input_endpoint")
    _bool(data["output_endpoint"], f"{context}.output_endpoint")
    _string(data["runtime_node"], f"{context}.runtime_node")
    if "source" in data:
        validate_io_info(data["source"], f"{context}.source")
    if "sink" in data:
        validate_io_info(data["sink"], f"{context}.sink")
    if "model" in data:
        validate_model_info(data["model"], f"{context}.model")


def validate_public_edge(value: Any, context: str) -> None:
    data = _mapping(value, context)
    _require(
        data,
        ("id", "index", "from", "to", "kind", "runtime_from", "runtime_to", "runtime_edges"),
        context,
    )
    _match(_string(data["id"], f"{context}.id"), r"^pe[0-9]+$", f"{context}.id")
    _number(data["index"], f"{context}.index")
    _match(_string(data["from"], f"{context}.from"), r"^p[0-9]+$", f"{context}.from")
    _match(_string(data["to"], f"{context}.to"), r"^p[0-9]+$", f"{context}.to")
    _string(data["kind"], f"{context}.kind")
    if data.get("from_endpoint") is not None:
        _string(data["from_endpoint"], f"{context}.from_endpoint")
    if data.get("to_endpoint") is not None:
        _string(data["to_endpoint"], f"{context}.to_endpoint")
    _string(data["runtime_from"], f"{context}.runtime_from")
    _string(data["runtime_to"], f"{context}.runtime_to")
    for i, edge_id in enumerate(_array(data["runtime_edges"], f"{context}.runtime_edges")):
        _match(_string(edge_id, f"{context}.runtime_edges[{i}]"), r"^e[0-9]+$", f"{context}.runtime_edges[{i}]")


def validate_lowered_node(value: Any, context: str) -> None:
    data = _mapping(value, context)
    _require(data, ("id", "stable_id", "backend", "kind", "compiler_generated"), context)
    _match(_string(data["id"], f"{context}.id"), r"^n[0-9]+$", f"{context}.id")
    _string(data["stable_id"], f"{context}.stable_id")
    _string(data["backend"], f"{context}.backend")
    _string(data["kind"], f"{context}.kind")
    _bool(data["compiler_generated"], f"{context}.compiler_generated")
    if "source" in data:
        validate_io_info(data["source"], f"{context}.source")
    if "sink" in data:
        validate_io_info(data["sink"], f"{context}.sink")
    if "model" in data:
        validate_model_info(data["model"], f"{context}.model")


def validate_lowered_edge(value: Any, context: str) -> None:
    data = _mapping(value, context)
    _require(data, ("id", "from", "to", "from_port", "to_port"), context)
    _match(_string(data["id"], f"{context}.id"), r"^(e|pe)[0-9]+$", f"{context}.id")
    _string(data["from"], f"{context}.from")
    _string(data["to"], f"{context}.to")
    if data["from_port"] is not None:
        _string(data["from_port"], f"{context}.from_port")
    if data["to_port"] is not None:
        _string(data["to_port"], f"{context}.to_port")


def validate_io_info(value: Any, context: str) -> None:
    data = _mapping(value, context)
    _require(data, ("kind", "uri"), context)
    _string(data["kind"], f"{context}.kind")
    if data["uri"] is not None:
        _string(data["uri"], f"{context}.uri")
    if "endpoint" in data and data["endpoint"] is not None:
        _string(data["endpoint"], f"{context}.endpoint")
    if "details" in data:
        _mapping(data["details"], f"{context}.details")


def validate_model_info(value: Any, context: str) -> None:
    data = _mapping(value, context)
    _require(data, ("id", "stage_role"), context)
    _string(data["id"], f"{context}.id")
    _string(data["stage_role"], f"{context}.stage_role")
    if data.get("source_path") is not None:
        _string(data["source_path"], f"{context}.source_path")
    if data.get("name") is not None:
        _string(data["name"], f"{context}.name")


def validate_latency_summary(value: Any, context: str) -> None:
    data = _mapping(value, context)
    if "samples" in data:
        _integer(data["samples"], f"{context}.samples")
    for key in ("total_ms", "avg_ms", "min_ms", "max_ms"):
        if key in data:
            _number(data[key], f"{context}.{key}")


def validate_power_summary(value: Any, context: str) -> None:
    data = _mapping(value, context)
    if "enabled" in data:
        _bool(data["enabled"], f"{context}.enabled")
    if "samples" in data:
        _integer(data["samples"], f"{context}.samples")
    for key in ("duration_seconds", "total_avg_watts", "total_min_watts", "total_max_watts", "energy_joules"):
        if key in data:
            _number(data[key], f"{context}.{key}")
    if "rails" in data:
        _array(data["rails"], f"{context}.rails")


def validate_graph_metrics(value: Any, context: str) -> None:
    data = _mapping(value, context)
    _require(data, ("measurement_scope", "throughput_counting", "elapsed_seconds", "throughput_fps"), context)
    _string(data["measurement_scope"], f"{context}.measurement_scope")
    if "aggregation" in data:
        _string(data["aggregation"], f"{context}.aggregation")
    if "latency_semantics" in data:
        _string(data["latency_semantics"], f"{context}.latency_semantics")
    _string(data["throughput_counting"], f"{context}.throughput_counting")
    _number(data["elapsed_seconds"], f"{context}.elapsed_seconds")
    if "outputs_pulled" in data:
        _integer(data["outputs_pulled"], f"{context}.outputs_pulled")
    _number(data["throughput_fps"], f"{context}.throughput_fps")
    if "counters" in data:
        counters = _mapping(data["counters"], f"{context}.counters")
        for key in (
            "inputs_enqueued",
            "inputs_dropped",
            "inputs_pushed",
            "outputs_ready",
            "outputs_pulled",
            "outputs_dropped",
        ):
            if key in counters:
                _integer(counters[key], f"{context}.counters.{key}")
    if "power" in data:
        validate_power_summary(data["power"], f"{context}.power")


def validate_element_metric(value: Any, context: str) -> None:
    data = _mapping(value, context)
    _require(data, ("name", "latency_ms"), context)
    _string(data["name"], f"{context}.name")
    validate_latency_summary(data["latency_ms"], f"{context}.latency_ms")


def validate_node_metric(value: Any, context: str) -> None:
    data = _mapping(value, context)
    if "power" in data:
        _raise(context, "node metrics must not contain power")
    if data.get("node_id") is not None:
        _string(data["node_id"], f"{context}.node_id")
    if data.get("runtime_node") is not None:
        _string(data["runtime_node"], f"{context}.runtime_node")
    if data.get("runtime_node_id") is not None:
        _integer(data["runtime_node_id"], f"{context}.runtime_node_id")
    if "public_node_ids" in data:
        for i, node_id in enumerate(_array(data["public_node_ids"], f"{context}.public_node_ids")):
            _string(node_id, f"{context}.public_node_ids[{i}]")
    if data.get("pipeline_segment_id") is not None:
        _integer(data["pipeline_segment_id"], f"{context}.pipeline_segment_id")
    if data.get("kind") is not None:
        _string(data["kind"], f"{context}.kind")
    if data.get("label") is not None:
        _string(data["label"], f"{context}.label")
    if "element_names" in data:
        for i, name in enumerate(_array(data["element_names"], f"{context}.element_names")):
            _string(name, f"{context}.element_names[{i}]")
    if "latency_semantics" in data:
        _string(data["latency_semantics"], f"{context}.latency_semantics")
    if "aggregation" in data:
        _string(data["aggregation"], f"{context}.aggregation")
    if "latency_ms" in data:
        validate_latency_summary(data["latency_ms"], f"{context}.latency_ms")
    if "elements" in data:
        for i, element in enumerate(_array(data["elements"], f"{context}.elements")):
            validate_element_metric(element, f"{context}.elements[{i}]")


def validate_plugin_metric(value: Any, context: str) -> None:
    data = _mapping(value, context)
    if "power" in data:
        _raise(context, "plugin metrics must not contain power")
    for key in ("backend", "phase", "kernel_name", "stage_name", "mapping_error"):
        if key in data:
            _string(data[key], f"{context}.{key}")
    for key in ("pipeline_segment_id", "runtime_node_id"):
        if data.get(key) is not None:
            _integer(data[key], f"{context}.{key}")
    if "public_node_ids" in data:
        for i, node_id in enumerate(_array(data["public_node_ids"], f"{context}.public_node_ids")):
            _string(node_id, f"{context}.public_node_ids[{i}]")
    for key in ("physical_input_index", "output_slot", "calls"):
        if key in data:
            _integer(data[key], f"{context}.{key}")
    if "latency_ms" in data:
        validate_latency_summary(data["latency_ms"], f"{context}.latency_ms")


def validate_graph_run(data: Mapping[str, Any]) -> None:
    _require(data, ("schema", "schema_version", "producer", "label", "metadata", "graph", "run"), "$")
    if data["schema"] != "sima.neat.graph_run":
        _raise("$.schema", "expected sima.neat.graph_run")
    if data["schema_version"] != 1:
        _raise("$.schema_version", "expected 1")
    _string(_mapping(data["producer"], "$.producer").get("name"), "$.producer.name")
    _string(data["label"], "$.label")
    _mapping(data["metadata"], "$.metadata")

    graph = _mapping(data["graph"], "$.graph")
    _require(graph, ("mode", "nodes", "edges", "named_inputs", "named_outputs"), "$.graph")
    mode = _string(graph["mode"], "$.graph.mode")
    if mode not in ("linear", "connected"):
        _raise("$.graph.mode", "expected linear or connected")

    for i, endpoint in enumerate(_array(graph["named_inputs"], "$.graph.named_inputs")):
        validate_named_endpoint(endpoint, f"$.graph.named_inputs[{i}]")
    for i, endpoint in enumerate(_array(graph["named_outputs"], "$.graph.named_outputs")):
        validate_named_endpoint(endpoint, f"$.graph.named_outputs[{i}]")
    for i, node in enumerate(_array(graph["nodes"], "$.graph.nodes")):
        validate_lowered_node(node, f"$.graph.nodes[{i}]")
    for i, edge in enumerate(_array(graph["edges"], "$.graph.edges")):
        validate_lowered_edge(edge, f"$.graph.edges[{i}]")

    if "public_view" in graph:
        public = _mapping(graph["public_view"], "$.graph.public_view")
        _require(public, ("nodes", "edges"), "$.graph.public_view")
        for i, node in enumerate(_array(public["nodes"], "$.graph.public_view.nodes")):
            validate_public_node(node, f"$.graph.public_view.nodes[{i}]")
        for i, edge in enumerate(_array(public["edges"], "$.graph.public_view.edges")):
            validate_public_edge(edge, f"$.graph.public_view.edges[{i}]")

    if "lowered_view" in graph:
        lowered = _mapping(graph["lowered_view"], "$.graph.lowered_view")
        _require(lowered, ("nodes", "edges", "pipeline_segments"), "$.graph.lowered_view")
        for i, node in enumerate(_array(lowered["nodes"], "$.graph.lowered_view.nodes")):
            validate_lowered_node(node, f"$.graph.lowered_view.nodes[{i}]")
        for i, edge in enumerate(_array(lowered["edges"], "$.graph.lowered_view.edges")):
            validate_lowered_edge(edge, f"$.graph.lowered_view.edges[{i}]")

    run = _mapping(data["run"], "$.run")
    if "identity" in run:
        identity = _mapping(run["identity"], "$.run.identity")
        _require(identity, ("uuid", "created_at", "closed_at", "hostname", "pid", "argv"), "$.run.identity")
        _string(identity["uuid"], "$.run.identity.uuid")
        if identity["created_at"] is not None:
            _string(identity["created_at"], "$.run.identity.created_at")
        if identity["closed_at"] is not None:
            _string(identity["closed_at"], "$.run.identity.closed_at")
        _string(identity["hostname"], "$.run.identity.hostname")
        _number(identity["pid"], "$.run.identity.pid")
        for i, arg in enumerate(_array(identity["argv"], "$.run.identity.argv")):
            _string(arg, f"$.run.identity.argv[{i}]")
    if "elapsed_seconds" in run:
        _number(run["elapsed_seconds"], "$.run.elapsed_seconds")
    if "throughput_fps" in run:
        _number(run["throughput_fps"], "$.run.throughput_fps")
    if "input_names" in run:
        for i, name in enumerate(_array(run["input_names"], "$.run.input_names")):
            _string(name, f"$.run.input_names[{i}]")
    if "output_names" in run:
        for i, name in enumerate(_array(run["output_names"], "$.run.output_names")):
            _string(name, f"$.run.output_names[{i}]")
    if "graph_metrics" in run:
        validate_graph_metrics(run["graph_metrics"], "$.run.graph_metrics")
    if "node_metrics" in run:
        for i, metric in enumerate(_array(run["node_metrics"], "$.run.node_metrics")):
            validate_node_metric(metric, f"$.run.node_metrics[{i}]")
    if "plugin_metrics_unattributed" in run:
        for i, metric in enumerate(
            _array(run["plugin_metrics_unattributed"], "$.run.plugin_metrics_unattributed")
        ):
            validate_plugin_metric(metric, f"$.run.plugin_metrics_unattributed[{i}]")


def load_graph_run(path: str | Path) -> Mapping[str, Any]:
    data = json.loads(Path(path).read_text())
    validate_graph_run(_mapping(data, "$"))
    return data
