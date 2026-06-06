#!/usr/bin/env python3
"""Render a NEAT graph-run JSON snapshot to a self-contained offline HTML file.

The renderer intentionally has no CDN or JavaScript dependency. It draws a simple
left-to-right SVG for either the public endpoint graph or the lowered runtime graph
and embeds the original JSON for inspection.
"""

from __future__ import annotations

import argparse
import html
import json
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


def _metric_label(metric: Mapping[str, Any]) -> str:
    latency = _as_map(metric.get("latency_ms"))
    samples = latency.get("samples")
    parts = []
    if "avg_ms" in latency:
        parts.append(f"avg {_fmt(latency.get('avg_ms'), ' ms')}")
    if "total_ms" in latency:
        parts.append(f"total {_fmt(latency.get('total_ms'), ' ms')}")
    if isinstance(samples, int):
        parts.append(f"n={samples}")
    return " / ".join(parts)


def _label(node: Mapping[str, Any], metric: Mapping[str, Any] | None = None) -> str:
    pieces = [str(node.get("id", "node"))]
    kind = str(node.get("kind", ""))
    label = str(node.get("label", ""))
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


def _normalize_view(payload: Mapping[str, Any], view: str) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    graph = _as_map(payload.get("graph"))
    if view == "public":
        selected = _as_map(graph.get("public_view"))
    elif view == "lowered":
        selected = _as_map(graph.get("lowered_view"))
    else:
        selected = _as_map(graph.get("public_view")) or _as_map(graph.get("lowered_view"))
    nodes = [n for n in selected.get("nodes", []) if isinstance(n, Mapping)]
    edges = [e for e in selected.get("edges", []) if isinstance(e, Mapping)]
    return nodes, edges


def _edge_endpoints(edge: Mapping[str, Any]) -> tuple[str, str, str]:
    source = str(edge.get("from", ""))
    target = str(edge.get("to", ""))
    label = ""
    from_ep = edge.get("from_endpoint") or edge.get("from_port")
    to_ep = edge.get("to_endpoint") or edge.get("to_port")
    if from_ep or to_ep:
        label = f"{from_ep or ''} → {to_ep or ''}"
    elif edge.get("kind"):
        label = str(edge.get("kind"))
    return source, target, label


def _render_svg(
    nodes: list[Mapping[str, Any]],
    edges: list[Mapping[str, Any]],
    lookup: tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]],
) -> str:
    if not nodes:
        return '<p class="empty">No nodes in selected view.</p>'

    node_w = 180
    node_h = 72
    x_gap = 70
    y_gap = 44
    margin = 40
    positions: dict[str, tuple[int, int]] = {}
    for i, node in enumerate(nodes):
        row = i % 4
        col = i // 4
        node_id = str(node.get("id", f"node{i}"))
        positions[node_id] = (margin + col * (node_w + x_gap), margin + row * (node_h + y_gap))

    cols = (len(nodes) + 3) // 4
    width = max(480, margin * 2 + max(1, cols) * node_w + max(0, cols - 1) * x_gap)
    rows = min(4, len(nodes))
    height = max(220, margin * 2 + rows * node_h + max(0, rows - 1) * y_gap)

    out: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="NEAT graph">',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L9,3 z" fill="#64748b" /></marker></defs>',
    ]

    for edge in edges:
        source, target, label = _edge_endpoints(edge)
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
    power = _as_map(graph_metrics.get("power")) or _as_map(run.get("power"))
    cards = [
        ("Throughput", _fmt(graph_metrics.get("throughput_fps", run.get("throughput_fps")), " fps")),
        ("Elapsed", _fmt(graph_metrics.get("elapsed_seconds", run.get("elapsed_seconds")), " s")),
        ("Scope", str(graph_metrics.get("measurement_scope") or graph_metrics.get("aggregation") or "—")),
        ("Counting", str(graph_metrics.get("throughput_counting") or "—")),
    ]
    if power:
        cards.extend(
            [
                ("Avg watts", _fmt(power.get("total_avg_watts"), " W")),
                ("Energy", _fmt(power.get("energy_joules"), " J")),
            ]
        )
    body = "\n".join(
        f'<div class="metric-card"><div class="metric-title">{html.escape(title)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div></div>'
        for title, value in cards
    )
    return f'<section class="metric-cards">{body}</section>'


def _render_node_metrics_table(payload: Mapping[str, Any]) -> str:
    rows: list[str] = []
    run = _as_map(payload.get("run"))
    for metric in _as_list(run.get("node_metrics")):
        if not isinstance(metric, Mapping):
            continue
        latency = _as_map(metric.get("latency_ms"))
        public_ids = ", ".join(str(x) for x in _as_list(metric.get("public_node_ids")))
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(metric.get('node_id') or '—'))}</td>"
            f"<td>{html.escape(public_ids or '—')}</td>"
            f"<td>{html.escape(str(metric.get('pipeline_segment_id') if metric.get('pipeline_segment_id') is not None else '—'))}</td>"
            f"<td>{html.escape(str(metric.get('kind') or '—'))}</td>"
            f"<td>{html.escape(str(metric.get('label') or '—'))}</td>"
            f"<td>{html.escape(_fmt(latency.get('avg_ms'), ' ms'))}</td>"
            f"<td>{html.escape(_fmt(latency.get('total_ms'), ' ms'))}</td>"
            f"<td>{html.escape(str(latency.get('samples', '—')))}</td>"
            "</tr>"
        )
    if not rows:
        return '<section><h2>Node metrics</h2><p class="empty">No node metrics in this export.</p></section>'
    return (
        "<section><h2>Node metrics</h2><table><thead><tr>"
        "<th>Lowered node</th><th>Public IDs</th><th>Segment</th><th>Kind</th><th>Label</th>"
        "<th>Avg latency</th><th>Total latency</th><th>Samples</th>"
        "</tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table></section>"
    )


def render_html(payload: Mapping[str, Any], view: str) -> str:
    nodes, edges = _normalize_view(payload, view)
    title = str(payload.get("label") or _as_map(payload.get("graph")).get("mode") or "NEAT graph")
    escaped_json = html.escape(json.dumps(payload, indent=2, sort_keys=True))
    lookup = _metric_lookup(payload)
    svg = _render_svg(nodes, edges, lookup)
    metric_cards = _render_metric_cards(payload)
    node_metric_table = _render_node_metrics_table(payload)
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
.metric-cards {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 16px; }}
.metric-card {{ background: white; border: 1px solid #cbd5e1; border-radius: 12px; padding: 12px 14px; min-width: 140px; }}
.metric-title {{ color: #64748b; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
.metric-value {{ color: #0f172a; font-weight: 700; font-size: 18px; margin-top: 4px; }}
table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; margin: 12px 0 20px; }}
th, td {{ border-bottom: 1px solid #e2e8f0; padding: 8px 10px; text-align: left; font-size: 13px; }}
th {{ background: #e2e8f0; color: #334155; }}
</style>
</head>
<body>
<header><h1>{html.escape(title)}</h1><div>schema={html.escape(str(payload.get('schema', '')))} version={html.escape(str(payload.get('schema_version', '')))} view={html.escape(view)}</div></header>
<main>
{metric_cards}
<section>{svg}</section>
{node_metric_table}
<h2>Raw graph-run JSON</h2>
<pre>{escaped_json}</pre>
</main>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Render NEAT graph-run JSON to offline HTML")
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--view", choices=("auto", "public", "lowered"), default="auto")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text())
    output = args.output or args.input.with_suffix(".html")
    output.write_text(render_html(payload, args.view))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
