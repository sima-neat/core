#!/usr/bin/env python3
"""Render a NEAT graph-run JSON snapshot to a self-contained offline HTML file.

The renderer intentionally has no CDN or JavaScript dependency. It draws a simple
left-to-right SVG for either the public endpoint graph or the lowered runtime graph
and embeds the original JSON for inspection.
"""

from __future__ import annotations

import argparse
import copy
import html
import json
import re
from pathlib import Path
from typing import Any, Mapping


def _as_map(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _fmt(value: Any, suffix: str = "", precision: int = 3) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.{precision}f}{suffix}"
    return "—"


def _metric_lookup(payload: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    run = _as_map(payload.get("run"))
    by_lowered: dict[str, Mapping[str, Any]] = {}
    by_public: dict[str, Mapping[str, Any]] = {}
    for metric in _as_list(run.get("node_metrics")):
        if not isinstance(metric, Mapping):
            continue
        node_id = metric.get("node_id")
        if isinstance(node_id, str) and node_id:
            by_lowered[node_id] = metric
        for public_id in _as_list(metric.get("public_node_ids")):
            if isinstance(public_id, str) and public_id:
                by_public[public_id] = metric
    return by_lowered, by_public


def _node_metric(node: Mapping[str, Any], lookup: tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]) -> Mapping[str, Any]:
    by_lowered, by_public = lookup
    node_id = str(node.get("id", ""))
    if node_id in by_public:
        return by_public[node_id]
    if node_id in by_lowered:
        return by_lowered[node_id]
    runtime_node = node.get("runtime_node")
    if isinstance(runtime_node, str) and runtime_node in by_lowered:
        return by_lowered[runtime_node]
    return {}


def _canonical_lowered_edge_id(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return ""
    if re.fullmatch(r"e[0-9]+", value):
        return value
    if re.fullmatch(r"[0-9]+", value):
        return f"e{value}"
    return ""


def _avg_ms(metric: Mapping[str, Any]) -> float | None:
    latency = _as_map(metric.get("latency_ms"))
    value = latency.get("avg_ms")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _edge_metric_lookup(payload: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    run = _as_map(payload.get("run"))
    labels: dict[str, list[Mapping[str, Any]]] = {}
    for metric in _as_list(run.get("edge_metrics")):
        if not isinstance(metric, Mapping):
            continue
        edge_id = _canonical_lowered_edge_id(metric.get("lowered_edge_id") or metric.get("edge_id"))
        if edge_id:
            labels.setdefault(edge_id, []).append(metric)
    return labels


def _path_gap_lookup(payload: Mapping[str, Any]) -> dict[str, list[Mapping[str, Any]]]:
    timing = _as_map(_as_map(payload.get("run")).get("path_timing"))
    out: dict[str, list[Mapping[str, Any]]] = {}
    rows = _as_list(timing.get("inter_plugin_gap_ms")) or _as_list(timing.get("inter_plugin_gap"))
    seen: set[tuple[str, str, str, str, str]] = set()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        edge_id = _canonical_lowered_edge_id(row.get("lowered_edge_id"))
        if edge_id:
            latency = _as_map(row.get("latency_ms"))
            key = (
                edge_id,
                str(row.get("stream_id") or ""),
                str(row.get("semantics") or ""),
                str(latency.get("samples") or ""),
                str(latency.get("avg_ms") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            out.setdefault(edge_id, []).append(row)
    return out


def _edge_overlay_label(
    edge: Mapping[str, Any],
    edge_lookup: Mapping[str, list[Mapping[str, Any]]],
    gap_lookup: Mapping[str, list[Mapping[str, Any]]],
) -> str:
    lowered_ids = [_canonical_lowered_edge_id(x) for x in _as_list(edge.get("lowered_edge_ids"))]
    own_id = _canonical_lowered_edge_id(edge.get("id"))
    if own_id and own_id not in lowered_ids:
        lowered_ids.append(own_id)

    gap_rows: list[Mapping[str, Any]] = []
    for edge_id in lowered_ids:
        gap_rows.extend(gap_lookup.get(edge_id, []))
    if gap_rows:
        preferred = [
            row
            for row in gap_rows
            if str(row.get("semantics") or "") in ("edge_transport", "queue_residence")
        ]
        if preferred:
            gap_rows = preferred
    gap_values = [_avg_ms(row) for row in gap_rows if _avg_ms(row) is not None]
    if gap_values:
        semantics = sorted({str(row.get("semantics") or "gap") for row in gap_rows})
        label = semantics[0] if len(semantics) == 1 else "path"
        return f"{label} avg {_fmt(max(gap_values), ' ms')} / {len(gap_rows)} rows"

    metrics: list[Mapping[str, Any]] = []
    for edge_id in lowered_ids:
        metrics.extend(edge_lookup.get(edge_id, []))
    if not metrics:
        return "handoff N/A" if edge.get("mapping_status") not in (None, "mapped") else ""
    values = [_avg_ms(m) for m in metrics if _avg_ms(m) is not None]
    streams = sorted({str(m.get("stream_id")) for m in metrics if m.get("stream_id")})
    if not values:
        return f"handoff N/A / {len(metrics)} rows / {len(streams)} streams"
    return f"handoff max avg {_fmt(max(values), ' ms')} / {len(metrics)} rows / {len(streams)} streams"


def _best_plugin_label(metric: Mapping[str, Any]) -> str:
    plugins = [p for p in _as_list(metric.get("plugins")) if isinstance(p, Mapping)]
    if not plugins:
        return ""
    def total_ms(plugin: Mapping[str, Any]) -> float:
        latency = _as_map(plugin.get("latency_ms"))
        value = latency.get("total_ms")
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0.0

    top = max(plugins, key=total_ms)
    latency = _as_map(top.get("latency_ms"))
    backend = str(top.get("backend") or "")
    phase = str(top.get("phase") or "")
    avg = _fmt(latency.get("avg_ms"), " ms")
    return f"{backend} {phase} avg {avg}".strip()


def _metric_label(metric: Mapping[str, Any]) -> str:
    latency = _as_map(metric.get("latency_ms"))
    samples = latency.get("samples")
    parts = []
    if samples == 0 and float(latency.get("total_ms") or 0.0) == 0.0:
        plugin_text = _best_plugin_label(metric)
        if plugin_text:
            parts.append(plugin_text)
        else:
            parts.append("latency N/A")
    elif "avg_ms" in latency:
        parts.append(f"avg {_fmt(latency.get('avg_ms'), ' ms')}")
    if not (samples == 0 and float(latency.get("total_ms") or 0.0) == 0.0) and "total_ms" in latency:
        parts.append(f"total {_fmt(latency.get('total_ms'), ' ms')}")
    if isinstance(samples, int):
        parts.append(f"n={samples}")
    return " / ".join(parts)


def _label(node: Mapping[str, Any], metric: Mapping[str, Any] | None = None) -> str:
    pieces = [str(node.get("id", "node"))]
    kind = "" if node.get("kind") is None else str(node.get("kind", ""))
    label = "" if node.get("label") is None else str(node.get("label", ""))
    endpoint = node.get("endpoint_name")
    if kind:
        pieces.append(kind)
    if label and label not in pieces:
        pieces.append(label)
    if endpoint:
        pieces.append(f"({endpoint})")
    if node.get("compiler_generated"):
        pieces.append("[generated]")
    if metric:
        metric_text = _metric_label(metric)
        if metric_text:
            pieces.append(metric_text)
    return "\n".join(pieces)


def _view_nodes_edges(view: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    nodes = [n for n in _as_list(view.get("nodes")) if isinstance(n, Mapping)]
    edges = [e for e in _as_list(view.get("edges")) if isinstance(e, Mapping)]
    return nodes, edges


def _node_id_from_metric(metric: Mapping[str, Any], index: int) -> str:
    for key in ("node_id", "runtime_node"):
        value = metric.get(key)
        if isinstance(value, str) and value:
            return value
    value = metric.get("runtime_node_id")
    if isinstance(value, int) and value >= 0:
        return f"n{value}"
    return f"n{index}"


def _fallback_view_from_node_metrics(payload: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    run = _as_map(payload.get("run"))
    metrics = [m for m in _as_list(run.get("node_metrics")) if isinstance(m, Mapping)]
    nodes: list[Mapping[str, Any]] = []
    edges: list[Mapping[str, Any]] = []
    previous = ""
    for index, metric in enumerate(metrics):
        node_id = _node_id_from_metric(metric, index)
        nodes.append(
            {
                "id": node_id,
                "kind": str(metric.get("kind") or "Node"),
                "label": str(metric.get("label") or ""),
                "runtime_node": node_id,
                "topology_source": "node_metrics_fallback",
            }
        )
        if previous:
            edges.append(
                {
                    "id": f"e{index - 1}",
                    "from": previous,
                    "to": node_id,
                    "from_port": None,
                    "to_port": None,
                    "kind": "metrics_fallback_linear",
                }
            )
        previous = node_id
    return nodes, edges


def _normalize_view(payload: Mapping[str, Any], view: str) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], str]:
    graph = _as_map(payload.get("graph"))
    if view == "customer":
        selected = _as_map(graph.get("customer_view"))
        nodes, edges = _view_nodes_edges(selected)
        source = str(selected.get("topology_source") or "customer_view")
    elif view == "public":
        selected = _as_map(graph.get("public_view"))
        nodes, edges = _view_nodes_edges(selected)
        source = str(selected.get("topology_source") or "public_view")
    elif view == "lowered":
        selected = _as_map(graph.get("lowered_view"))
        nodes, edges = _view_nodes_edges(selected)
        source = str(selected.get("topology_source") or "lowered_view")
    else:
        selected = _as_map(graph.get("customer_view"))
        nodes, edges = _view_nodes_edges(selected)
        source = str(selected.get("topology_source") or "customer_view")
        if not nodes:
            selected = _as_map(graph.get("public_view"))
            nodes, edges = _view_nodes_edges(selected)
            source = str(selected.get("topology_source") or "public_view")
        if not nodes:
            selected = _as_map(graph.get("lowered_view"))
            nodes, edges = _view_nodes_edges(selected)
            source = str(selected.get("topology_source") or "lowered_view")
        if not nodes:
            nodes, edges = _view_nodes_edges(graph)
            source = "graph.nodes"
    if not nodes:
        nodes, edges = _fallback_view_from_node_metrics(payload)
        source = "node_metrics_fallback"
    return nodes, edges, source


def _edge_endpoints(edge: Mapping[str, Any]) -> tuple[str, str, str]:
    source = str(edge.get("from", ""))
    target = str(edge.get("to", ""))
    label = ""
    from_ep = edge.get("from_endpoint") or edge.get("from_port")
    to_ep = edge.get("to_endpoint") or edge.get("to_port")
    if from_ep or to_ep:
        label = f"{from_ep or ''} → {to_ep or ''}"
    elif edge.get("kind") and edge.get("kind") != "metrics_fallback_linear":
        label = str(edge.get("kind"))
    return source, target, label


def _parallel_pair_layout(
    nodes: list[Mapping[str, Any]],
    edges: list[Mapping[str, Any]],
    node_w: int,
    node_h: int,
    x_gap: int,
    y_gap: int,
    margin: int,
) -> tuple[dict[str, tuple[int, int]], int, int] | None:
    """Return a compact lane layout for disjoint source->sink stream pairs.

    Multi-stream public graphs are often exported as a set of independent
    endpoint pairs (camera_0->detections_0, camera_1->detections_1, ...).  A
    single left-to-right row is technically valid, but it becomes very wide and
    reads like one long chain.  When the topology is exactly a set of disjoint
    pairs, stack those pairs as parallel lanes instead.
    """

    node_order = {str(node.get("id", f"node{i}")): i for i, node in enumerate(nodes)}
    lanes: list[tuple[str, str]] = []
    sources: set[str] = set()
    targets: set[str] = set()

    if len(edges) < 2:
        return None

    for edge in edges:
        source, target, _ = _edge_endpoints(edge)
        if not source or not target or source == target:
            return None
        if source not in node_order or target not in node_order:
            return None
        if source in sources or target in targets:
            return None
        sources.add(source)
        targets.add(target)
        lanes.append((source, target))

    if sources & targets:
        return None
    if len(sources) + len(targets) != 2 * len(edges):
        return None

    lanes.sort(key=lambda pair: (min(node_order[pair[0]], node_order[pair[1]]), node_order[pair[0]]))
    positions: dict[str, tuple[int, int]] = {}
    for row, (source, target) in enumerate(lanes):
        y = margin + row * (node_h + y_gap)
        positions[source] = (margin, y)
        positions[target] = (margin + node_w + x_gap, y)

    # Keep any isolated nodes visible below the stream lanes.
    isolated = [node_id for node_id in node_order if node_id not in positions]
    for offset, node_id in enumerate(isolated):
        row = len(lanes) + offset // 2
        col = offset % 2
        positions[node_id] = (margin + col * (node_w + x_gap), margin + row * (node_h + y_gap))

    cols = 2 if lanes else max(1, min(2, len(nodes)))
    rows = len(lanes) + ((len(isolated) + 1) // 2)
    width = max(480, margin * 2 + cols * node_w + max(0, cols - 1) * x_gap)
    height = max(220, margin * 2 + max(1, rows) * node_h + max(0, rows - 1) * y_gap)
    return positions, width, height


def _default_layout(
    nodes: list[Mapping[str, Any]],
    edges: list[Mapping[str, Any]],
    node_w: int,
    node_h: int,
    x_gap: int,
    y_gap: int,
    margin: int,
) -> tuple[dict[str, tuple[int, int]], int, int]:
    pair_layout = _parallel_pair_layout(nodes, edges, node_w, node_h, x_gap, y_gap, margin)
    if pair_layout is not None:
        return pair_layout

    positions: dict[str, tuple[int, int]] = {}
    one_row = len(nodes) <= 8
    for i, node in enumerate(nodes):
        row = 0 if one_row else i % 4
        col = i if one_row else i // 4
        node_id = str(node.get("id", f"node{i}"))
        positions[node_id] = (margin + col * (node_w + x_gap), margin + row * (node_h + y_gap))

    cols = len(nodes) if one_row else (len(nodes) + 3) // 4
    width = max(480, margin * 2 + max(1, cols) * node_w + max(0, cols - 1) * x_gap)
    rows = 1 if one_row else min(4, len(nodes))
    height = max(220, margin * 2 + rows * node_h + max(0, rows - 1) * y_gap)
    return positions, width, height


def _layout_dag(
    nodes: list[Mapping[str, Any]],
    edges: list[Mapping[str, Any]],
    node_w: int,
    node_h: int,
    x_gap: int,
    y_gap: int,
    margin: int,
) -> tuple[dict[str, tuple[int, int]], int, int]:
    node_ids = [str(n.get("id", f"node{i}")) for i, n in enumerate(nodes)]
    incoming: dict[str, set[str]] = {nid: set() for nid in node_ids}
    outgoing: dict[str, set[str]] = {nid: set() for nid in node_ids}
    for edge in edges:
        source, target, _ = _edge_endpoints(edge)
        if source in incoming and target in incoming:
            outgoing[source].add(target)
            incoming[target].add(source)
    ranks: dict[str, int] = {nid: 0 for nid in node_ids}
    changed = True
    for _ in range(max(1, len(node_ids))):
        if not changed:
            break
        changed = False
        for source in node_ids:
            for target in outgoing[source]:
                if ranks[target] < ranks[source] + 1:
                    ranks[target] = ranks[source] + 1
                    changed = True
    by_rank: dict[int, list[str]] = {}
    for nid in node_ids:
        by_rank.setdefault(ranks[nid], []).append(nid)
    positions: dict[str, tuple[int, int]] = {}
    for rank, ids in by_rank.items():
        for row, nid in enumerate(ids):
            positions[nid] = (margin + rank * (node_w + x_gap), margin + row * (node_h + y_gap))
    cols = max(by_rank.keys(), default=0) + 1
    rows = max((len(v) for v in by_rank.values()), default=1)
    width = max(480, margin * 2 + cols * node_w + max(0, cols - 1) * x_gap)
    height = max(220, margin * 2 + rows * node_h + max(0, rows - 1) * y_gap)
    return positions, width, height


def _render_svg(
    nodes: list[Mapping[str, Any]],
    edges: list[Mapping[str, Any]],
    lookup: tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]],
    edge_lookup: Mapping[str, list[Mapping[str, Any]]],
    gap_lookup: Mapping[str, list[Mapping[str, Any]]],
    view_source: str,
) -> str:
    if not nodes:
        return (
            '<p class="empty">No graph topology was exported and no node metrics were available '
            'to build a fallback view.</p>'
        )

    node_w = 190
    node_h = 78
    x_gap = 64
    y_gap = 44
    margin = 40
    positions, width, height = _layout_dag(nodes, edges, node_w, node_h, x_gap, y_gap, margin)

    out: list[str] = [
        f'<div class="graph-source">Topology view: {html.escape(view_source)}</div>',
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="NEAT graph">',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#64748b" /></marker></defs>',
    ]

    for edge in edges:
        source, target, label = _edge_endpoints(edge)
        edge_metric_label = _edge_overlay_label(edge, edge_lookup, gap_lookup)
        if edge_metric_label:
            label = edge_metric_label if not label else f"{label} / {edge_metric_label}"
        if source not in positions or target not in positions:
            continue
        sx, sy = positions[source]
        tx, ty = positions[target]
        x1, y1 = sx + node_w, sy + node_h // 2
        x2, y2 = tx, ty + node_h // 2
        if source == target:
            x1, y1 = sx + node_w // 2, sy
            x2, y2 = sx + node_w // 2, sy + node_h
            path = f"M{x1},{y1} C{x1 + 40},{y1 - 40} {x2 + 40},{y2 + 40} {x2},{y2}"
        else:
            mid = (x1 + x2) // 2
            path = f"M{x1},{y1} C{mid},{y1} {mid},{y2} {x2},{y2}"
        out.append(f'<path class="edge" d="{path}" marker-end="url(#arrow)" />')
        if label:
            lx, ly = (x1 + x2) // 2, (y1 + y2) // 2 - 6
            out.append(f'<text class="edge-label" x="{lx}" y="{ly}">{html.escape(label)}</text>')

    for node in nodes:
        node_id = str(node.get("id", "node"))
        x, y = positions[node_id]
        generated = bool(node.get("compiler_generated"))
        cls = "node generated" if generated else "node"
        metric = _node_metric(node, lookup)
        if metric:
            cls += " with-metrics"
        out.append(f'<g class="{cls}">')
        out.append(f'<rect x="{x}" y="{y}" width="{node_w}" height="{node_h}" rx="12" />')
        lines = _label(node, metric).split("\n")[:4]
        for i, line in enumerate(lines):
            out.append(
                f'<text x="{x + node_w / 2:.1f}" y="{y + 22 + i * 15}" text-anchor="middle">{html.escape(line)}</text>'
            )
        out.append("</g>")

    out.append("</svg>")
    return "\n".join(out)


def _render_metric_cards(payload: Mapping[str, Any]) -> str:
    run = _as_map(payload.get("run"))
    graph_metrics = _as_map(run.get("graph_metrics"))
    throughput_summary = _as_map(graph_metrics.get("throughput"))
    measurement = _as_map(run.get("measurement"))
    power = _as_map(graph_metrics.get("power")) or _as_map(run.get("power"))
    graph_e2e = _as_map(run.get("graph_e2e_latency_ms"))
    end_to_end = graph_e2e or _as_map(measurement.get("end_to_end"))

    def latency_card(summary: Mapping[str, Any], key: str) -> str:
        if summary.get("count") == 0:
            return "N/A"
        return _fmt(summary.get(key), " ms")

    warmup = measurement.get("warmup_iterations")
    output_pull_tput = (
        throughput_summary.get("outputs_per_s")
        or graph_metrics.get("outputs_per_s")
        or graph_metrics.get("throughput_fps")
        or run.get("throughput_fps")
    )
    logical_tput = (
        throughput_summary.get("logical_inferences_per_s")
        or measurement.get("throughput_inferences_per_s")
    )
    cards = [
        ("Output pulls/s", _fmt(output_pull_tput, "")),
        ("Logical inf/s", _fmt(logical_tput, "")),
        ("Elapsed", _fmt(graph_metrics.get("elapsed_seconds", run.get("elapsed_seconds")), " s")),
        ("Queue-inclusive E2E p50", latency_card(end_to_end, "p50_ms")),
        ("Queue-inclusive E2E p95", latency_card(end_to_end, "p95_ms")),
        ("Warmup", str(warmup if isinstance(warmup, int) else "—")),
        ("Scope", str(graph_metrics.get("measurement_scope") or graph_metrics.get("aggregation") or "—")),
    ]
    if graph_e2e:
        reliability = "reliable" if graph_e2e.get("correlation_reliable") else str(graph_e2e.get("status") or "unreliable")
        if graph_e2e.get("survivor_biased"):
            reliability += " / survivor-biased"
        cards.append(("E2E correlation", reliability))
    materialization = _as_map(run.get("output_materialization"))
    if materialization:
        timing = (
            "timing unavailable"
            if materialization.get("copy_map_timing_available") is False
            else "timing available"
        )
        cards.append((
            "Output memory",
            f"{materialization.get('output_memory_mode') or '—'} / {materialization.get('semantics') or '—'}",
        ))
        cards.append((
            "Materialization",
            f"{materialization.get('claim_status') or materialization.get('status') or '—'}; {timing}",
        ))
    if power:
        if power.get("enabled") and power.get("samples", 0):
            cards.extend(
                [
                    ("Board rail avg W", _fmt(power.get("total_avg_watts"), " W")),
                    ("Energy", _fmt(power.get("energy_joules"), " J")),
                ]
            )
        else:
            status = str(power.get("status") or "disabled/no samples")
            if status == "disabled_by_options":
                status = "disabled"
            elif status == "requested_no_samples":
                status = "requested/no samples"
            cards.append(("Power", status))
    plugin_latency = _as_map(run.get("plugin_latency"))
    if plugin_latency:
        cards.append(("Plugin latency", str(plugin_latency.get("status") or "—")))
    edge_latency = _as_map(run.get("edge_message_latency"))
    if edge_latency:
        cards.append(("Edge latency", str(edge_latency.get("status") or "—")))
    path_timing = _as_map(run.get("path_timing"))
    if path_timing:
        cards.append(("Path timing", str(path_timing.get("status") or "—")))
    body = "\n".join(
        f'<div class="metric-card"><div class="metric-title">{html.escape(title)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div></div>'
        for title, value in cards
    )
    return f'<section><h2>Throughput</h2><div class="metric-cards">{body}</div></section>'


def _latency_table_cell(latency: Mapping[str, Any], key: str, suffix: str = " ms", availability_key: str | None = None) -> str:
    samples = latency.get("samples")
    total = latency.get("total_ms")
    if samples == 0 and (not isinstance(total, (int, float)) or float(total) == 0.0):
        return "N/A"
    if availability_key and latency.get(availability_key) is False:
        return "N/A"
    return _fmt(latency.get(key), suffix)


def _render_node_metrics_table(payload: Mapping[str, Any]) -> str:
    rows: list[str] = []
    run = _as_map(payload.get("run"))
    for metric in _as_list(run.get("node_metrics")):
        if not isinstance(metric, Mapping):
            continue
        latency = _as_map(metric.get("latency_ms"))
        public_ids = ", ".join(str(x) for x in _as_list(metric.get("public_node_ids")))
        plugin_count = len([p for p in _as_list(metric.get("plugins")) if isinstance(p, Mapping)])
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(metric.get('node_id') or '—'))}</td>"
            f"<td>{html.escape(public_ids or '—')}</td>"
            f"<td>{html.escape(str(metric.get('pipeline_segment_id') if metric.get('pipeline_segment_id') is not None else '—'))}</td>"
            f"<td>{html.escape(str(metric.get('kind') or '—'))}</td>"
            f"<td>{html.escape(str(metric.get('label') or '—'))}</td>"
            f"<td>{html.escape(_latency_table_cell(latency, 'avg_ms'))}</td>"
            f"<td>{html.escape(_latency_table_cell(latency, 'total_ms'))}</td>"
            f"<td>{html.escape(str(latency.get('samples', '—')))}</td>"
            f"<td>{plugin_count}</td>"
            "</tr>"
        )
    if not rows:
        return '<section><h2>Node metrics</h2><p class="empty">No node metrics in this export.</p></section>'
    return (
        "<section><h2>Node metrics</h2>"
        '<p class="hint">N/A means this node has no element-residency samples in the measured window; '
        "check attributed plugin rows for backend/device timing.</p>"
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Lowered node</th><th>Public IDs</th><th>Segment</th><th>Kind</th><th>Label</th>"
        "<th>Avg latency</th><th>Total latency</th><th>Samples</th><th>Plugins</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table></div></section>"
    )


def _plugin_display_name(plugin: Mapping[str, Any]) -> str:
    name = plugin.get("name")
    if isinstance(name, str) and name:
        return name
    backend = str(plugin.get("backend") or "plugin")
    stage = str(plugin.get("stage_name") or plugin.get("gst_element_name") or plugin.get("kernel_name") or "unknown")
    return f"{backend}:{stage}"


def _plugin_table(rows: list[str], title: str, empty: str, warning: bool = False) -> str:
    cls = ' class="warning"' if warning else ""
    if not rows:
        return f'<section{cls}><h2>{html.escape(title)}</h2><p class="empty">{html.escape(empty)}</p></section>'
    return (
        f"<section{cls}><h2>{html.escape(title)}</h2>"
        '<p class="hint">Plugin rows are diagnostic and non-additive; rows can be nested or overlap.</p>'
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Node</th><th>Backend</th><th>Phase</th><th>Kernel</th><th>Stage/element</th>"
        "<th>Stream</th><th>Source</th><th>Calls</th><th>Avg latency</th><th>Total latency</th><th>Mapping</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table></div></section>"
    )


def _plugin_row(plugin: Mapping[str, Any], node_id: str, mapping: str = "") -> str:
    latency = _as_map(plugin.get("latency_ms"))
    stage = plugin.get("gst_element_name") or plugin.get("stage_name")
    return (
        "<tr>"
        f"<td>{html.escape(node_id or '—')}</td>"
        f"<td>{html.escape(str(plugin.get('backend') or '—'))}</td>"
        f"<td>{html.escape(str(plugin.get('phase') or '—'))}</td>"
        f"<td>{html.escape(str(plugin.get('kernel_name') or _plugin_display_name(plugin)))}</td>"
        f"<td>{html.escape(str(stage or '—'))}</td>"
        f"<td>{html.escape(str(plugin.get('stream_id') or '—'))}</td>"
        f"<td>{html.escape(str(plugin.get('source') or '—'))}</td>"
        f"<td>{html.escape(str(plugin.get('calls', '—')))}</td>"
        f"<td>{html.escape(_fmt(latency.get('avg_ms'), ' ms'))}</td>"
        f"<td>{html.escape(_fmt(latency.get('total_ms'), ' ms'))}</td>"
        f"<td>{html.escape(mapping or str(plugin.get('mapping_error') or plugin.get('attribution_source') or 'attributed'))}</td>"
        "</tr>"
    )


def _render_plugin_metrics(payload: Mapping[str, Any]) -> str:
    run = _as_map(payload.get("run"))
    attributed_rows: list[str] = []
    for metric in _as_list(run.get("node_metrics")):
        if not isinstance(metric, Mapping):
            continue
        node_id = str(metric.get("node_id") or metric.get("runtime_node") or "—")
        public_ids = [str(x) for x in _as_list(metric.get("public_node_ids")) if isinstance(x, str)]
        label = node_id if not public_ids else f"{node_id} ({', '.join(public_ids)})"
        for plugin in _as_list(metric.get("plugins")):
            if isinstance(plugin, Mapping):
                attributed_rows.append(_plugin_row(plugin, label, "attributed"))

    unattributed_rows: list[str] = []
    for plugin in _as_list(run.get("plugin_metrics_unattributed")):
        if isinstance(plugin, Mapping):
            unattributed_rows.append(_plugin_row(plugin, "—"))

    return (
        _plugin_table(
            attributed_rows,
            "Attributed plugin metrics",
            "No attributed plugin/kernel metrics in this export.",
        )
        + "\n"
        + _plugin_table(
            unattributed_rows,
            "Unattributed plugin metrics",
            "No unattributed plugin/kernel metrics.",
            warning=bool(unattributed_rows),
        )
    )


def _edge_row(edge: Mapping[str, Any]) -> str:
    latency = _as_map(edge.get("latency_ms"))
    endpoint = " → ".join(
        part for part in (str(edge.get("from_node") or ""), str(edge.get("to_node") or "")) if part
    )
    return (
        "<tr>"
        f"<td>{html.escape(str(edge.get('edge_id') or edge.get('name') or '—'))}</td>"
        f"<td>{html.escape(endpoint or '—')}</td>"
        f"<td>{html.escape(str(edge.get('stream_id') or '—'))}</td>"
        f"<td>{html.escape(str(edge.get('samples', '—')))}</td>"
        f"<td>{html.escape(_fmt(latency.get('avg_ms'), ' ms'))}</td>"
        f"<td>{html.escape(_latency_table_cell(latency, 'p50_ms', ' ms', 'percentiles_available'))}</td>"
        f"<td>{html.escape(_latency_table_cell(latency, 'p95_ms', ' ms', 'percentiles_available'))}</td>"
        f"<td>{html.escape(_latency_table_cell(latency, 'max_ms', ' ms', 'min_max_available'))}</td>"
        f"<td>{html.escape(str(edge.get('timing_semantics') or '—'))}</td>"
        f"<td>{html.escape(str(edge.get('source') or '—'))}</td>"
        f"<td>{html.escape(str(edge.get('mapping_error') or edge.get('attribution_source') or '—'))}</td>"
        "</tr>"
    )


def _edge_table(rows: list[str], title: str, empty: str, warning: bool = False) -> str:
    cls = ' class="warning"' if warning else ""
    if not rows:
        return f'<section{cls}><h2>{html.escape(title)}</h2><p class="empty">{html.escape(empty)}</p></section>'
    return (
        f"<section{cls}><h2>{html.escape(title)}</h2>"
        '<p class="hint">Measures handoff/queue/transport time between nodes/plugins. '
        'Do not add to plugin execution latency or graph latency.</p>'
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Edge</th><th>From → To</th><th>Stream</th><th>Samples</th>"
        "<th>Avg</th><th>P50</th><th>P95</th><th>Max</th><th>Semantics</th><th>Source</th><th>Mapping/warning</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table></div></section>"
    )


def _render_edge_metrics(payload: Mapping[str, Any]) -> str:
    run = _as_map(payload.get("run"))
    rows = [_edge_row(edge) for edge in _as_list(run.get("edge_metrics")) if isinstance(edge, Mapping)]
    unattributed = [
        _edge_row(edge)
        for edge in _as_list(run.get("edge_metrics_unattributed"))
        if isinstance(edge, Mapping)
    ]
    return (
        _edge_table(rows, "Inter-plugin message / edge latency", "No edge/message metrics in this export.")
        + "\n"
        + _edge_table(
            unattributed,
            "Unattributed edge/message metrics",
            "No unattributed edge/message metrics.",
            warning=bool(unattributed),
        )
    )


def _path_latency_cells(latency: Mapping[str, Any]) -> str:
    return (
        f"<td>{html.escape(str(latency.get('samples', '—')))}</td>"
        f"<td>{html.escape(_fmt(latency.get('avg_ms'), ' ms'))}</td>"
        f"<td>{html.escape(_fmt(latency.get('p50_ms'), ' ms'))}</td>"
        f"<td>{html.escape(_fmt(latency.get('p95_ms'), ' ms'))}</td>"
        f"<td>{html.escape(_fmt(latency.get('max_ms'), ' ms'))}</td>"
        f"<td>{html.escape('yes' if latency.get('reliable', True) else 'no')}</td>"
    )


def _path_table(title: str, rows: list[str], empty: str, hint: str) -> str:
    if not rows:
        return f'<section><h2>{html.escape(title)}</h2><p class="hint">{html.escape(hint)}</p><p class="empty">{html.escape(empty)}</p></section>'
    return (
        f"<section><h2>{html.escape(title)}</h2>"
        f'<p class="hint">{html.escape(hint)}</p>'
        '<div class="table-wrap"><table><thead><tr>'
        "<th>Item</th><th>Stream</th><th>Semantics</th><th>Samples</th><th>Avg</th>"
        "<th>P50</th><th>P95</th><th>Max</th><th>Reliable</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table></div></section>"
    )


def _render_path_timing(payload: Mapping[str, Any]) -> str:
    timing = _as_map(_as_map(payload.get("run")).get("path_timing"))
    if not timing:
        return '<section><h2>Path timing</h2><p class="empty">No path timing object in this export.</p></section>'

    status = str(timing.get("status") or "off")
    source = str(timing.get("source") or "none")
    reason = str(timing.get("reason") or "")
    header = (
        f'<section><h2>Path timing status</h2><p class="hint">'
        f'status={html.escape(status)} source={html.escape(source)}'
        + (f' reason={html.escape(reason)}' if reason else "")
        + "</p></section>"
    )

    arrival_rows: list[str] = []
    for row in _as_list(timing.get("node_arrival")):
        if not isinstance(row, Mapping):
            continue
        item = row.get("customer_node_id") or row.get("lowered_node_id") or row.get("runtime_node_id") or "—"
        arrival_rows.append(
            "<tr>"
            f"<td>{html.escape(str(item))}</td>"
            f"<td>{html.escape(str(row.get('stream_id') or '—'))}</td>"
            f"<td>{html.escape(str(row.get('semantics') or '—'))}</td>"
            + _path_latency_cells(_as_map(row.get("latency_ms")))
            + "</tr>"
        )

    gap_rows: list[str] = []
    for row in _as_list(timing.get("inter_plugin_gap_ms")) or _as_list(timing.get("inter_plugin_gap")):
        if not isinstance(row, Mapping):
            continue
        item = row.get("customer_edge_id") or row.get("lowered_edge_id") or "—"
        gap_rows.append(
            "<tr>"
            f"<td>{html.escape(str(item))}</td>"
            f"<td>{html.escape(str(row.get('stream_id') or '—'))}</td>"
            f"<td>{html.escape(str(row.get('semantics') or '—'))}</td>"
            + _path_latency_cells(_as_map(row.get("latency_ms")))
            + "</tr>"
        )

    tail_rows: list[str] = []
    for row in _as_list(timing.get("output_tail")):
        if not isinstance(row, Mapping):
            continue
        item = row.get("output_endpoint") or row.get("customer_output_node_id") or row.get("lowered_edge_id") or "—"
        tail_rows.append(
            "<tr>"
            f"<td>{html.escape(str(item))}</td>"
            f"<td>{html.escape(str(row.get('stream_id') or '—'))}</td>"
            f"<td>{html.escape(str(row.get('semantics') or '—'))}</td>"
            + _path_latency_cells(_as_map(row.get("latency_ms")))
            + "</tr>"
        )

    return (
        header
        + "\n"
        + _path_table(
            "Graph entry → node arrival",
            arrival_rows,
            "No graph-entry-to-node arrival rows.",
            "Requires core graph-entry LTTng events and plugin spans with matching sample identity.",
        )
        + "\n"
        + _path_table(
            "Between plugins / transport",
            gap_rows,
            "No inter-plugin path rows.",
            "Shows exact edge transport/queue rows when message tracing is available; plugin-end-to-plugin-start rows are a diagnostic fallback.",
        )
        + "\n"
        + _path_table(
            "Last work → output pull",
            tail_rows,
            "No output-tail rows.",
            "Requires core graph-output LTTng events and plugin spans with matching sample identity.",
        )
    )


PRIVATE_KEYS = {"argv", "hostname", "pid", "uri", "source_path", "path"}
PRIVATE_PATTERNS = ("/workspace", "/home/", "rtsp://", "file://", "udp://")


def _redact_payload(value: Any, mode: str) -> Any:
    if mode == "none":
        return value
    def redact(obj: Any) -> Any:
        if isinstance(obj, dict):
            out: dict[str, Any] = {}
            for key, val in obj.items():
                if key in PRIVATE_KEYS:
                    out[key] = "<redacted>"
                else:
                    out[key] = redact(val)
            return out
        if isinstance(obj, list):
            return [redact(x) for x in obj]
        if isinstance(obj, str) and any(pattern in obj for pattern in PRIVATE_PATTERNS):
            return "<redacted>"
        return obj
    return redact(copy.deepcopy(value))


def _payload_has_fallback(payload: Mapping[str, Any]) -> bool:
    customer = _as_map(_as_map(payload.get("graph")).get("customer_view"))
    return str(customer.get("mapping_status") or "").startswith("fallback")


def _payload_has_unattributed(payload: Mapping[str, Any]) -> bool:
    run = _as_map(payload.get("run"))
    return bool(_as_list(run.get("plugin_metrics_unattributed")) or _as_list(run.get("edge_metrics_unattributed")))


def render_html(payload: Mapping[str, Any], view: str) -> str:
    nodes, edges, view_source = _normalize_view(payload, view)
    title = str(payload.get("label") or _as_map(payload.get("graph")).get("mode") or "NEAT graph")
    escaped_json = html.escape(json.dumps(payload, indent=2, sort_keys=True))
    lookup = _metric_lookup(payload)
    edge_lookup = _edge_metric_lookup(payload)
    gap_lookup = _path_gap_lookup(payload)
    svg = _render_svg(nodes, edges, lookup, edge_lookup, gap_lookup, view_source)
    metric_cards = _render_metric_cards(payload)
    node_metric_table = _render_node_metrics_table(payload)
    plugin_metric_tables = _render_plugin_metrics(payload)
    edge_metric_tables = _render_edge_metrics(payload)
    path_timing_tables = _render_path_timing(payload)
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>{html.escape(title)} - NEAT graph run</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 0; color: #0f172a; background: #f8fafc; }}
header {{ padding: 18px 24px; background: #0f172a; color: white; }}
main {{ padding: 20px 24px; }}
svg {{ width: 100%; min-height: 320px; background: white; border: 1px solid #cbd5e1; border-radius: 12px; }}
.node rect {{ fill: #2563eb; stroke: #1e40af; stroke-width: 1.5; }}
.node.generated rect {{ fill: #7c3aed; stroke: #5b21b6; }}
.node.with-metrics rect {{ stroke: #fbbf24; stroke-width: 3; }}
.node text {{ fill: white; font-size: 12px; font-weight: 600; }}
.edge {{ fill: none; stroke: #64748b; stroke-width: 2; }}
.edge-label {{ fill: #334155; font-size: 11px; paint-order: stroke; stroke: white; stroke-width: 4px; }}
pre {{ background: #0b1020; color: #dbeafe; border-radius: 12px; padding: 16px; overflow: auto; }}
.empty {{ padding: 16px; background: white; border-radius: 12px; }}
.hint {{ color: #475569; font-size: 13px; margin: 4px 0 10px; }}
.graph-source {{ color: #475569; font-size: 13px; margin: 0 0 8px; }}
.legend {{ color: #475569; font-size: 13px; margin: 8px 0 16px; }}
.metric-cards {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 16px; }}
.metric-card {{ background: white; border: 1px solid #cbd5e1; border-radius: 12px; padding: 12px 14px; min-width: 140px; }}
.metric-title {{ color: #64748b; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
.metric-value {{ color: #0f172a; font-weight: 700; font-size: 18px; margin-top: 4px; }}
.table-wrap {{ overflow-x: auto; border-radius: 12px; margin: 12px 0 20px; }}
table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; }}
th, td {{ border-bottom: 1px solid #e2e8f0; padding: 8px 10px; text-align: left; font-size: 13px; }}
th {{ background: #e2e8f0; color: #334155; }}
td:nth-child(1), td:nth-child(4), td:nth-child(5) {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
tbody tr:nth-child(even) {{ background: #f8fafc; }}
.warning table, .warning .empty {{ border: 1px solid #f59e0b; }}
.warning h2::after {{ content: " ⚠"; color: #d97706; }}
details {{ margin-top: 18px; }}
summary {{ cursor: pointer; font-weight: 700; margin-bottom: 8px; }}
</style>
</head>
<body>
<header><h1>{html.escape(title)}</h1><div>schema={html.escape(str(payload.get('schema', '')))} version={html.escape(str(payload.get('schema_version', '')))} view={html.escape(view)}</div></header>
<main>
{metric_cards}
<section>{svg}</section>
<div class="legend">Legend: yellow border = node has metrics/plugin timing; purple = compiler-generated; fallback topology may be reconstructed from node metrics when exported graph topology is unavailable.</div>
{node_metric_table}
{plugin_metric_tables}
{path_timing_tables}
{edge_metric_tables}
<details>
<summary>Raw graph-run JSON</summary>
<pre>{escaped_json}</pre>
</details>
</main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Render NEAT graph-run JSON to offline HTML")
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--view", choices=("auto", "customer", "public", "lowered"), default="auto")
    parser.add_argument("--svg", type=Path)
    parser.add_argument("--topology-json", type=Path)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--redact", choices=("none", "customer"), default="none")
    parser.add_argument("--fail-on-fallback", action="store_true")
    parser.add_argument("--fail-on-unattributed", action="store_true")
    args = parser.parse_args()

    payload = _redact_payload(json.loads(args.input.read_text()), args.redact)
    if args.fail_on_fallback and _payload_has_fallback(payload):
        raise SystemExit("customer view is fallback")
    if args.fail_on_unattributed and _payload_has_unattributed(payload):
        raise SystemExit("unattributed metrics present")
    nodes, edges, view_source = _normalize_view(payload, args.view)
    if args.validate or args.strict:
        if args.strict:
            node_ids = {str(n.get("id")) for n in nodes if isinstance(n, Mapping)}
            for edge in edges:
                if not isinstance(edge, Mapping):
                    continue
                source, target, _ = _edge_endpoints(edge)
                if source not in node_ids or target not in node_ids:
                    raise SystemExit(f"edge references missing node: {source}->{target}")
    if args.topology_json:
        args.topology_json.write_text(json.dumps({"topology_source": view_source, "nodes": nodes, "edges": edges}, indent=2))
    if args.summary_json:
        run = _as_map(payload.get("run"))
        args.summary_json.write_text(json.dumps({"graph_metrics": _as_map(run.get("graph_metrics")), "measurement": _as_map(run.get("measurement"))}, indent=2))
    if args.summary:
        run = _as_map(payload.get("run"))
        print(json.dumps({"graph_metrics": _as_map(run.get("graph_metrics"))}, indent=2))
    if args.svg:
        args.svg.write_text(_render_svg(nodes, edges, _metric_lookup(payload), _edge_metric_lookup(payload), _path_gap_lookup(payload), view_source))
    output = args.output or args.input.with_suffix(".html")
    output.write_text(render_html(payload, args.view))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
