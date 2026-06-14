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


def _string_array(value: Any, context: str) -> None:
    for i, item in enumerate(_array(value, context)):
        _string(item, f"{context}[{i}]")


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
    if "min_max_available" in data:
        _bool(data["min_max_available"], f"{context}.min_max_available")
    if "percentiles_available" in data:
        _bool(data["percentiles_available"], f"{context}.percentiles_available")
    if "max_semantics" in data:
        _string(data["max_semantics"], f"{context}.max_semantics")


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


def validate_throughput_summary(value: Any, context: str) -> None:
    data = _mapping(value, context)
    _require(
        data,
        (
            "unit",
            "scope",
            "semantics",
            "numerator_counter",
            "numerator_value",
            "denominator",
            "denominator_seconds",
            "outputs_per_s",
            "batches_per_s",
            "logical_batch_size",
            "logical_inferences_per_s",
            "available",
            "status",
            "warnings",
        ),
        context,
    )
    for key in ("unit", "scope", "semantics", "numerator_counter", "denominator", "status"):
        _string(data[key], f"{context}.{key}")
    _integer(data["numerator_value"], f"{context}.numerator_value")
    _number(data["denominator_seconds"], f"{context}.denominator_seconds")
    for key in (
        "outputs_per_s",
        "batches_per_s",
        "throughput_batches_per_s",
        "logical_inferences_per_s",
        "elapsed_seconds",
    ):
        if key in data:
            _number(data[key], f"{context}.{key}")
    if "outputs_pulled" in data:
        _integer(data["outputs_pulled"], f"{context}.outputs_pulled")
    _integer(data["logical_batch_size"], f"{context}.logical_batch_size")
    _bool(data["available"], f"{context}.available")
    if "multi_output_semantics" in data:
        _string(data["multi_output_semantics"], f"{context}.multi_output_semantics")
    _string_array(data["warnings"], f"{context}.warnings")


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
    if "throughput" in data:
        validate_throughput_summary(data["throughput"], f"{context}.throughput")
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


def validate_graph_e2e_latency(value: Any, context: str) -> None:
    data = _mapping(value, context)
    _require(
        data,
        (
            "available",
            "status",
            "correlation_reliable",
            "survivor_biased",
            "unit",
            "semantics",
            "count",
            "warnings",
        ),
        context,
    )
    _bool(data["available"], f"{context}.available")
    _string(data["status"], f"{context}.status")
    _bool(data["correlation_reliable"], f"{context}.correlation_reliable")
    _bool(data["survivor_biased"], f"{context}.survivor_biased")
    _string(data["unit"], f"{context}.unit")
    _string(data["semantics"], f"{context}.semantics")
    _integer(data["count"], f"{context}.count")
    _string_array(data["warnings"], f"{context}.warnings")
    for key in ("scope", "source", "interpretation"):
        if key in data and data[key] is not None:
            _string(data[key], f"{context}.{key}")
    for key in ("avg_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms", "max_ms"):
        if key in data:
            _number(data[key], f"{context}.{key}")


def validate_output_materialization(value: Any, context: str) -> None:
    data = _mapping(value, context)
    _require(
        data,
        (
            "available",
            "status",
            "claim_status",
            "claim_scope",
            "output_memory_mode",
            "semantics",
            "timing_available",
            "runtime_resolved_mode_available",
            "copy_map_timing_available",
            "prepare_output_cpu_visible",
            "note",
            "warnings",
        ),
        context,
    )
    _bool(data["available"], f"{context}.available")
    _string(data["status"], f"{context}.status")
    _string(data["claim_status"], f"{context}.claim_status")
    _string(data["claim_scope"], f"{context}.claim_scope")
    _string(data["output_memory_mode"], f"{context}.output_memory_mode")
    _string(data["semantics"], f"{context}.semantics")
    _bool(data["timing_available"], f"{context}.timing_available")
    _bool(data["runtime_resolved_mode_available"], f"{context}.runtime_resolved_mode_available")
    _bool(data["copy_map_timing_available"], f"{context}.copy_map_timing_available")
    _bool(data["prepare_output_cpu_visible"], f"{context}.prepare_output_cpu_visible")
    _string(data["note"], f"{context}.note")
    _string_array(data["warnings"], f"{context}.warnings")


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
    if "plugins" in data:
        for i, plugin in enumerate(_array(data["plugins"], f"{context}.plugins")):
            validate_plugin_metric(plugin, f"{context}.plugins[{i}]")


def validate_plugin_metric(value: Any, context: str) -> None:
    data = _mapping(value, context)
    if "power" in data:
        _raise(context, "plugin metrics must not contain power")
    if "name" in data and data["name"] is not None:
        _string(data["name"], f"{context}.name")
    for key in (
        "backend",
        "phase",
        "kernel_name",
        "stage_name",
        "mapping_error",
        "stream_id",
        "plugin_instance_id",
        "source",
        "attribution_source",
    ):
        if key in data:
            if data[key] is not None:
                _string(data[key], f"{context}.{key}")
    if "gst_element_name" in data and data["gst_element_name"] is not None:
        _string(data["gst_element_name"], f"{context}.gst_element_name")
    for key in ("run_id_hash", "pipeline_segment_id", "runtime_node_id", "public_node_id"):
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
    if "reliable" in data:
        _bool(data["reliable"], f"{context}.reliable")


def validate_edge_metric(value: Any, context: str) -> None:
    data = _mapping(value, context)
    if "power" in data:
        _raise(context, "edge metrics must not contain power")
    for key in (
        "edge_id",
        "lowered_edge_id",
        "mapping_status",
        "name",
        "from_node",
        "to_node",
        "from_element_name",
        "to_element_name",
        "from_plugin_instance_id",
        "to_plugin_instance_id",
        "stream_id",
        "source",
        "timing_semantics",
        "attribution_source",
        "mapping_error",
    ):
        if key in data and data[key] is not None:
            _string(data[key], f"{context}.{key}")
    for key in ("from_runtime_node_id", "to_runtime_node_id", "samples"):
        if data.get(key) is not None:
            _integer(data[key], f"{context}.{key}")
    if "latency_ms" in data:
        latency = _mapping(data["latency_ms"], f"{context}.latency_ms")
        for key in ("samples",):
            if key in latency:
                _integer(latency[key], f"{context}.latency_ms.{key}")
        for key in ("total_ms", "avg_ms", "min_ms", "max_ms", "p50_ms", "p95_ms"):
            if key in latency:
                _number(latency[key], f"{context}.latency_ms.{key}")
    if "non_additive" in data:
        _bool(data["non_additive"], f"{context}.non_additive")
    if "reliable" in data:
        _bool(data["reliable"], f"{context}.reliable")


def validate_customer_view(value: Any, context: str, *, allow_private: bool = False) -> None:
    data = _mapping(value, context)
    _require(data, ("nodes", "edges"), context)
    if not allow_private and data.get("mapping_status") == "fallback":
        _raise(context, "customer view is fallback")
    node_ids: set[str] = set()
    for i, node in enumerate(_array(data["nodes"], f"{context}.nodes")):
        n = _mapping(node, f"{context}.nodes[{i}]")
        node_id = _string(n.get("id"), f"{context}.nodes[{i}].id")
        node_ids.add(node_id)
        if "lowered_node_ids" in n:
            for j, lowered in enumerate(_array(n["lowered_node_ids"], f"{context}.nodes[{i}].lowered_node_ids")):
                _string(lowered, f"{context}.nodes[{i}].lowered_node_ids[{j}]")
    for i, edge in enumerate(_array(data["edges"], f"{context}.edges")):
        e = _mapping(edge, f"{context}.edges[{i}]")
        _string(e.get("id"), f"{context}.edges[{i}].id")
        source = _string(e.get("from"), f"{context}.edges[{i}].from")
        target = _string(e.get("to"), f"{context}.edges[{i}].to")
        if source not in node_ids or target not in node_ids:
            _raise(f"{context}.edges[{i}]", "edge endpoint missing from customer nodes")
        if "lowered_edge_ids" in e:
            for j, lowered in enumerate(_array(e["lowered_edge_ids"], f"{context}.edges[{i}].lowered_edge_ids")):
                _match(_string(lowered, f"{context}.edges[{i}].lowered_edge_ids[{j}]"), r"^e[0-9]+$", f"{context}.edges[{i}].lowered_edge_ids[{j}]")


def validate_path_timing(value: Any, context: str) -> None:
    data = _mapping(value, context)
    if "available" in data:
        _bool(data["available"], f"{context}.available")
    for key in ("status", "source", "reason", "aggregation"):
        if key in data and data[key] is not None:
            _string(data[key], f"{context}.{key}")
    for array_key in ("node_arrival", "inter_plugin_gap", "inter_plugin_gap_ms", "output_tail"):
        if array_key in data:
            _array(data[array_key], f"{context}.{array_key}")


def validate_referential_integrity(data: Mapping[str, Any]) -> None:
    graph = _mapping(data.get("graph"), "$.graph")
    lowered = _mapping(graph.get("lowered_view", {"nodes": [], "edges": [], "pipeline_segments": []}), "$.graph.lowered_view")
    lowered_edges = {str(edge.get("id")) for edge in _array(lowered.get("edges", []), "$.graph.lowered_view.edges") if isinstance(edge, Mapping)}
    run = _mapping(data.get("run"), "$.run")
    for i, metric in enumerate(_array(run.get("edge_metrics", []), "$.run.edge_metrics")):
        if not isinstance(metric, Mapping):
            continue
        lowered_id = metric.get("lowered_edge_id")
        if lowered_id is not None:
            _match(_string(lowered_id, f"$.run.edge_metrics[{i}].lowered_edge_id"), r"^e[0-9]+$", f"$.run.edge_metrics[{i}].lowered_edge_id")
            if lowered_edges and lowered_id not in lowered_edges:
                _raise(f"$.run.edge_metrics[{i}].lowered_edge_id", "does not reference lowered_view edge")


def validate_graph_run(data: Mapping[str, Any], *, strict: bool = False, customer: bool = False, allow_private: bool = False) -> None:
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
    if "customer_view" in graph:
        validate_customer_view(graph["customer_view"], "$.graph.customer_view", allow_private=allow_private)

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
    if "graph_e2e_latency_ms" in run:
        validate_graph_e2e_latency(run["graph_e2e_latency_ms"], "$.run.graph_e2e_latency_ms")
    if "output_materialization" in run:
        validate_output_materialization(run["output_materialization"], "$.run.output_materialization")
    if "node_metrics" in run:
        for i, metric in enumerate(_array(run["node_metrics"], "$.run.node_metrics")):
            validate_node_metric(metric, f"$.run.node_metrics[{i}]")
    if "plugin_metrics_unattributed" in run:
        for i, metric in enumerate(
            _array(run["plugin_metrics_unattributed"], "$.run.plugin_metrics_unattributed")
        ):
            validate_plugin_metric(metric, f"$.run.plugin_metrics_unattributed[{i}]")
    if "edge_metrics" in run:
        for i, metric in enumerate(_array(run["edge_metrics"], "$.run.edge_metrics")):
            validate_edge_metric(metric, f"$.run.edge_metrics[{i}]")
    if "edge_metrics_unattributed" in run:
        for i, metric in enumerate(
            _array(run["edge_metrics_unattributed"], "$.run.edge_metrics_unattributed")
        ):
            validate_edge_metric(metric, f"$.run.edge_metrics_unattributed[{i}]")
    if "path_timing" in run:
        validate_path_timing(run["path_timing"], "$.run.path_timing")
    if strict:
        validate_referential_integrity(data)
    if customer and "customer_view" not in graph:
        _raise("$.graph.customer_view", "missing customer view")


def load_graph_run(path: str | Path) -> Mapping[str, Any]:
    data = json.loads(Path(path).read_text())
    validate_graph_run(_mapping(data, "$"))
    return data


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate NEAT graph-run JSON")
    parser.add_argument("input", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--customer", action="store_true")
    parser.add_argument("--allow-private", action="store_true")
    args = parser.parse_args()

    data = json.loads(args.input.read_text())
    validate_graph_run(_mapping(data, "$"), strict=args.strict, customer=args.customer,
                       allow_private=args.allow_private)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
