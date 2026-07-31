# Issue #458: Node Identity and Build-Time Launch Validation Plan

## Summary

Issue #458 describes two independent correctness problems:

1. A single `Node` object can be inserted as more than one logical Graph vertex.
2. Multiple GStreamer elements can receive the same explicit name in the same parent bin.

This plan fixes both problems without changing the public C++ or Python API. Existing `add()`,
`connect()`, `build()`, `run()`, `run_rtsp()`, and `validate()` calls keep the same signatures and
usage. The behavioral change is intentional: unsupported or ambiguous compositions fail earlier
with actionable diagnostics.

The mandatory checks run automatically during composition and build. Applications do not need to
call `Graph::validate()` to receive this protection.

## Goals

- Enforce one logical composition vertex per non-null `Node*` identity.
- Preserve legitimate reuse of an existing vertex through `connect()` and fan-out.
- Make every public composition mutation atomic for expected and unexpected exceptions.
- Detect duplicate explicit GStreamer element names within the same immediate parent bin.
- Correctly extract and transform every explicit name in multi-element Custom fragments.
- Rewrite actual named-pad references with their element declarations.
- Run mandatory correctness checks in every build and execution path.
- Preserve structured `NeatError` and `GraphReport` diagnostics through connected lazy builds.
- Keep `Graph::validate()` as an optional diagnostic and CI API, not a required correctness step.

## Non-goals

- Do not add or change public method signatures.
- Do not add fields to the public `Graph` object layout.
- Do not clone `Node` objects automatically; `Node` has no general resource-safe clone contract.
- Do not silently rename collisions after final rendering.
- Do not reject equal short names that belong to different GStreamer parent bins.
- Do not predict every name GStreamer may generate for unnamed elements.
- Do not turn `Graph::build()` into a call to the full `Graph::validate()` workflow.

## Compatibility contract

### Unchanged valid usage

```cpp
graph.add(nodes::VideoConvert());
graph.connect(source, processor);
graph.connect(processor, sink);
auto run = graph.build(inputs);
```

### Intentionally tightened behavior

- Calling `graph.add(node)` twice with the same non-null object throws synchronously.
- Importing a new fragment that aliases a `Node` already present in the destination throws.
- `connect()` reuses an existing vertex for the same `Node` object instead of inserting another
  vertex.
- Duplicate explicit sibling `name=` bindings fail with `misconfig.pipeline_shape` before
  GStreamer construction.
- Connected lazy-build failures retain their original structured report instead of becoming a
  generic pull/runtime error.

`Graph::add(nullptr)` retains its current deferred-validation behavior. `connect(nullptr, ...)`
continues to reject immediately.

## Invariant 1: one Node object, one logical vertex

Within one Graph composition:

```text
one non-null Node* identity -> at most one pipeline vertex
```

This invariant applies to pipeline Nodes only. Internal runtime-stage vertices are outside this
identity domain.

### Required behavior

- `graph.add(node)` twice: reject the second operation.
- A new Graph or Model fragment that overlaps the destination by `Node*`: reject the import.
- Repeated references to the same imported Graph ID and version: reuse its existing range.
- `graph.add(node)` followed by `graph.connect(node, sink)`: reuse the existing vertex.
- `connect(node, sink_a)` followed by `connect(node, sink_b)`: reuse one vertex and allow fan-out.
- Separately constructed Node instances of the same type: allow both.

### CompositionGraph APIs

Add private helpers in `src/pipeline/graph/GraphDetail.h` and implement them in
`GraphComposition.cpp`:

```cpp
std::optional<VertexId>
find_pipeline_vertex(const Node* node) const noexcept;

void preflight_new_pipeline_nodes(
    std::span<const NodePtr> incoming,
    const char* where) const;

void preflight_new_pipeline_vertices(
    std::span<const CompositionVertex> incoming,
    const char* where) const;

VertexId append_linear_pipeline_vertex(NodePtr node, const char* where);
VertexId append_unlinked_pipeline_vertex(NodePtr node, const char* where);
```

`find_pipeline_vertex()` uses raw object identity and a linear scan. Graphs are small, and avoiding
a persistent identity index removes synchronization risk during move, load, and fragment import.

The preflight helpers:

1. skip null and runtime vertices;
2. detect a collision with an existing destination vertex;
3. detect repeated identities inside the incoming range;
4. retain original incoming vertex indices in diagnostics.

The linear append primitive creates the existing implicit-linear edge. The unlinked primitive does
not create an edge and is used when topology will be copied or reconstructed separately.

### Atomic public mutations

Per-fragment preflight is not sufficient. A multi-operand `connect(a, b)` currently imports `a`
before it validates `b` or resolves their endpoints. If a later step fails, the destination Graph
can remain partially mutated without a version increment.

Add a private RAII composition mutation transaction. It snapshots:

- `composition_`, including imported-fragment and imported-model maps;
- `groups_`;
- `input_route_processor_`.

On failure, its destructor restores the snapshot by move or swap. On success, the public method
commits the transaction and calls `mark_composition_changed()` exactly once.

Apply the transaction to every public `add()` and `connect()` overload. This also fixes existing
partial-mutation behavior for endpoint-resolution and allocation failures, not only duplicate-node
failures.

Do not make repeated edge insertion idempotent as part of this work. Reusing a fragment or vertex
is supported; inserting an already-connected edge follows the existing topology rules.

### Insertion-path coverage

Route every pipeline-vertex insertion through the appropriate identity-enforcing primitive:

- `Graph::add(std::shared_ptr<Node>)`;
- linear Graph-fragment append;
- connected composition import;
- output-collection import;
- Node operands used by `connect()`;
- Model-fragment import;
- `append_pipeline_vertex_for_internal_graph_()`;
- connected Graph loading in `GraphIo.cpp`.

Graph load and disconnected fragment import must use the unlinked primitive. Using the existing
linear append function there would create edges that are not present in the serialized or imported
topology.

Remove `CompositionGraph::imported_nodes`. It is incomplete because direct `add()` and fragment
imports do not populate it. `find_pipeline_vertex()` becomes the single source of truth for
Node-operand reuse.

## Invariant 2: valid explicit names in each GStreamer parent scope

GStreamer requires sibling objects to have unique names. It permits the same short name in
different child bins. Neat therefore validates explicit names using:

```text
(immediate parent scope, canonical explicit name)
```

This is more precise than a global per-launch-string set and preserves valid nested Custom
fragments.

### GStreamer remains authoritative

The precheck can reason about explicit `name=` bindings, but it cannot predict all names GStreamer
generates for unnamed elements. For example, an unnamed `identity` can collide with an explicit
`identity0` depending on process-global counters.

The build sequence is therefore:

1. analyze and reject known invalid explicit bindings;
2. call `gst_parse_launch()`;
3. retain `misconfig.pipeline_shape` for a precheck failure;
4. retain `misconfig.parse_launch` for other GStreamer construction failures.

Longer-term cleanup may explicitly name all framework-generated top-level elements, but it is not
required to close issue #458.

### Internal syntax utility

Add the shared private implementation under `src/pipeline/internal/` because it is consumed by
both Graph code and `src/nodes/common/Caps.cpp`:

```text
src/pipeline/internal/GstLaunchSyntax.h
src/pipeline/internal/GstLaunchSyntax.cpp
```

Suggested API:

```cpp
namespace simaai::neat::pipeline_internal::gst_launch_syntax {

struct ByteSpan {
  std::size_t begin = 0;
  std::size_t end = 0;
};

struct Assignment {
  std::string key;
  std::string raw_value;
  std::string canonical_value;
  ByteSpan assignment_span;
  ByteSpan value_span;
};

struct ElementDecl {
  ElementId id;
  ScopeId parent_scope;
  ByteSpan span;
  std::vector<Assignment> assignments;
  std::optional<NameBinding> effective_name;
};

struct Document {
  std::vector<ElementDecl> elements;
  std::vector<Reference> references;
  std::vector<Diagnostic> diagnostics;
  bool complete = false;
};

Document analyze(std::string_view launch);

std::vector<NameCollision>
find_explicit_name_collisions(const Document& document);

RewriteResult rewrite_bindings(
    std::string_view launch,
    const Document& document,
    const std::unordered_map<std::string, std::string>& mapping);

} // namespace simaai::neat::pipeline_internal::gst_launch_syntax
```

The utility is independent of `Graph`, `NeatError`, and plugin availability so its core tests run
on the host.

### Required shallow grammar

Model the relevant token categories from GStreamer's launch parser rather than using substring
searches or a region-between-`!` heuristic:

- assignments with maximal-token matching;
- element references, pad references, and bin references;
- URLs;
- `!` and `:` links;
- inline caps, caps features, and semicolon-separated caps;
- nested bins and parent scopes;
- escaped whitespace and escaped operators;
- quoted assignment values;
- malformed-input diagnostics with linear-time behavior.

Canonicalization must follow GStreamer semantics. In particular, double-quoted and single-quoted
assignment values must not be treated as interchangeable without verifying their actual parser
behavior.

Group assignments by their owning element. A fragment such as:

```text
identity name=a name=b
```

contains one element with a repeated `name` property, not two element declarations. Reject this as
a separate pipeline-shape error.

### Scope-aware collision rules

Reject:

```text
identity name=dup ! identity name=dup
```

Allow equal short names in different child bins:

```text
bin.( identity name=dup ) bin.( identity name=dup )
```

Also detect duplicate bin names in the same parent. A bin's own name belongs to its parent scope;
the elements inside it belong to the bin's child scope.

### Name and reference rewriting

The generic GStreamer utility rewrites only syntax it understands as GStreamer bindings:

- effective `name` declarations;
- the element component of actual named-pad reference tokens such as `demux.bbox`.

It must not rewrite:

- dotted URLs or property values;
- caps;
- quoted configuration strings;
- implicit `.pad` references;
- bin factory/opening syntax.

Apply replacements from the end of the string toward the beginning so byte spans remain valid.

Keep SiMa-specific alias handling in `GraphNaming.cpp`, outside the generic syntax utility.
`stage-id` and `op-buff-name` can be rewritten through explicit alias rules when their value maps
to an element binding. Do not rewrite `next-element`; values such as `CVU` and `APU` are selector
semantics, not generic element references.

## Custom fragment corrections

`CustomNode::element_names()` currently extracts only the first `name=` binding. Replace that logic
with `GstLaunchSyntax::analyze()` and return all effective explicit names.

For automatic naming, inject a generated name only when analysis proves the fragment contains:

- exactly one ordinary element declaration;
- no effective explicit name;
- no bin or URI ambiguity;
- complete, non-malformed shallow syntax.

If analysis is incomplete, do not inject or rewrite. Let `gst_parse_launch()` produce the syntax
error.

Update `make_node_fragment()` to:

1. render `backend_fragment()` once;
2. analyze the rendered fragment;
3. combine Node-reported element names with explicit analyzed bindings for diagnostics;
4. build the requested name transform;
5. rewrite declarations and actual named-pad references together;
6. reanalyze the result and record its effective names.

Do not require exact equality between `Node::element_names()` and explicit `name=` declarations.
A Node may report dynamically created children. Explicit names found in the rendered fragment must,
however, never be omitted from transformation or diagnostics.

The shipped `UdpOutputGroupG` fragment is a required regression case. The transformed output must
keep these declarations and references synchronized:

```text
name=demux
name=render
demux.bbox
demux.image
render.sink_0
render.sink_1
```

## Fold mandatory validation into build

Most applications do not call `Graph::validate()`. Correctness must not depend on that optional
API.

Do not call the full `validate()` implementation from `build()`. That would duplicate rendering,
parsing, and PAUSED-state work. Instead, factor shared internal stages:

```cpp
validate_composition_or_throw(...);
validate_explicit_name_bindings_or_throw(...);
parse_pipeline_or_throw(...);
```

Every build and execution path invokes the mandatory stages automatically:

- source builds;
- seeded-input builds;
- one-shot `run()` paths;
- connected lazy segment builds;
- fused real-time builds;
- RTSP launch construction.

The shared `session_build_parse_pipeline_or_throw()` boundary is the sole ordinary
`gst_parse_launch()` path. Run explicit binding validation immediately before it invokes
GStreamer. Fused and connected segments then inherit the same behavior automatically.

RTSP is the special path because `gst_rtsp_media_factory_set_launch()` parses later. Store the final
launch in `last_pipeline_`, validate its explicit bindings synchronously, and only then allocate the
RTSP implementation, create the server handle, or start the thread. This preserves the failed
launch for diagnostics without creating background resources.

### Role of Graph::validate()

Keep `Graph::validate()` as an optional diagnostic and CI API. It can:

- return a report instead of throwing;
- perform additional probes or PAUSED-state checks;
- produce detailed reproduction information;
- validate before deployment.

It must reuse the same mandatory composition, launch-binding, and parse helpers as build so the two
paths do not drift.

`ValidateOptions::enforce_names` continues to control optional post-parse ownership enforcement.
Duplicate explicit sibling bindings are a mandatory pipeline-shape invariant and are not disabled
by that option.

Current non-linear `Graph::validate()` is topology-only and does not render every connected
segment. Either document that limitation or add a segment-render validation phase. Do not claim
that connected validation checks final names until such a phase exists; connected `build()` still
performs the mandatory checks when each segment materializes.

## Error handling and report propagation

### Composition errors

Continue using `std::runtime_error` for synchronous structural misuse so C++ behavior remains
consistent and Python continues to receive `RuntimeError` without binding changes.

Diagnostics should identify:

- Node kind and optional label;
- existing destination vertex;
- incoming vertex or position;
- the action required: construct another Node instance or reuse through `connect()`.

### Launch errors

Explicit binding failures use `error_codes::kPipelineShape` and report:

- canonical duplicate name;
- immediate parent scope;
- both declaration byte spans;
- a pipeline reproducer;
- a note that automatic repair is intentionally disabled.

Other GStreamer construction failures retain `error_codes::kParseLaunch`.

### Graph::validate()

When catching `NeatError`, preserve its original code and merge missing diagnostics from the
current `BuildResult`. Do not overwrite every failure with `kParseLaunch`.

### Connected lazy builds

`RunCore::ensure_graph_pipeline_built()` must catch `NeatError` separately and preserve its
`GraphReport`. Store the report in the relevant runtime or RunCore failure state and propagate it
through graph stop and pull. Do not reduce it to `e.what()` and later report it as a generic
`runtime.pull` failure.

## File-by-file implementation map

### Composition

- `include/pipeline/Graph.h`
  - Add only private forward declarations needed for the mutation transaction.
  - Update `add()` and `connect()` contract comments; no signature or layout change.
- `src/pipeline/graph/GraphDetail.h`
  - Add identity lookup/preflight and linear/unlinked insertion declarations.
  - Define private composition transaction state if required.
- `src/pipeline/graph/GraphComposition.cpp`
  - Implement identity lookup, preflight, and insertion primitives.
- `src/pipeline/graph/Graph.cpp`
  - Apply whole-operation transactions to all public mutations.
  - Route every pipeline insertion through the new primitives.
  - Replace and remove `imported_nodes` behavior.
- `src/pipeline/graph/GraphIo.cpp`
  - Use unlinked insertion during connected Graph load.

### Launch analysis and rewriting

- `src/pipeline/internal/GstLaunchSyntax.h`
- `src/pipeline/internal/GstLaunchSyntax.cpp`
  - Add the pure shallow syntax model, scope analysis, collision detection, and binding rewrite.
- `src/pipeline/graph/GraphNaming.cpp`
  - Replace ad hoc fragment rewriting.
  - Keep SiMa alias-property rules separate.
  - Add structured explicit-binding enforcement.
- `src/nodes/common/Caps.cpp`
  - Replace Custom first-name extraction and fragile auto-name detection.
- `src/nodes/groups/UdpOutputGroupG.cpp`
  - No behavior-specific workaround; the generic rewrite must handle its fragment correctly.

### Build and runtime integration

- `src/pipeline/graph/GraphBuildPipeline.cpp`
  - Run mandatory explicit-binding validation immediately before `gst_parse_launch()`.
- `src/pipeline/graph/GraphRtsp.cpp`
  - Validate the final launch synchronously before RTSP resource/thread creation.
- `src/pipeline/graph/GraphValidate.cpp`
  - Preserve structured error codes and merge diagnostic reports.
- `src/pipeline/runtime/RunCore.cpp`
- `src/pipeline/runtime/RunCoreGraphStart.cpp`
  - Preserve `NeatError::report()` through lazy pipeline construction and pull failures.

## Test plan

### Composition identity and atomicity

1. Adding the same Node object twice rejects the second operation.
2. Vertex, edge, group, import-map, version, and built-cache state remain unchanged after failure.
3. `add(Graph)` with a shared Node collision rejects atomically.
4. `connect(a, b)` where both fragments share a Node leaves the destination entirely unchanged.
5. A valid connect succeeds after the failed connect, proving no stale imported-fragment entry
   remains.
6. `add(node)` followed by `connect(node, sink)` reuses one vertex.
7. One node fans out to two distinct sinks through one vertex.
8. The same source Graph connects to two sinks and is imported once.
9. Separately constructed instances of one Node type remain valid.
10. Internal unlinked insertion rejects duplicate pipeline identity.
11. Runtime vertices are excluded.
12. `Graph::add(nullptr)` retains deferred validation.
13. Save/load preserves the serialized topology; repeated JSON records create separate objects.
14. Decide and test the existing DAG policy for `connect(node, node)` without making edge
    idempotence part of this issue.

### GStreamer syntax analysis

1. Duplicate sibling explicit names reject.
2. Equal short names in separate child bins succeed.
3. Duplicate bin names in one parent reject.
4. Repeated `name` assignments on one element reject.
5. Double-quoted, single-quoted, escaped, and unquoted values match GStreamer semantics.
6. `encoding-name`, other suffixed properties, URLs, and configuration strings do not count.
7. Inline caps, caps features, semicolon caps, `: CAPS :`, and `caps=...name=...` do not produce
   false declarations.
8. Escaped whitespace and escaped operators remain inside their token.
9. Malformed quoting completes in linear time and reports incomplete analysis.
10. Explicit-versus-auto-generated collision remains a GStreamer parse failure.
11. A differential corpus compares analyzer decisions with the supported GStreamer parser.

### Rewriting and Custom fragments

1. All `UdpOutputGroupG` declarations and pad references transform together.
2. Forward and backward named-pad references transform correctly.
3. URLs, caps, dotted values, and quoted configuration remain unchanged.
4. `stage-id` and `op-buff-name` follow explicit Neat alias rules.
5. `next-element=CVU` and `next-element=APU` never change.
6. Simple unnamed Custom elements receive deterministic names.
7. Complex, malformed, URI, and bin fragments do not receive unsafe injected names.

### Build, validate, connected, fused, and RTSP paths

1. A normal build rejects duplicate explicit sibling names without an explicit `validate()` call.
2. One-shot `run()` and seeded builds receive the same protection.
3. A collision with an inserted boundary, queue, tap, or fused mux name is detected when explicit.
4. Names in separately parsed connected segments do not collide with each other.
5. Connected lazy-build failures retain `kPipelineShape` and the original `GraphReport`.
6. Fused render-only tests cover generated launch names.
7. RTSP rejects explicit collisions before starting its thread and preserves `last_pipeline()`.
8. `Graph::validate()` returns `kPipelineShape` instead of rewriting it as `kParseLaunch`.
9. Existing parser failures continue to report `gst_parse_launch failed` and `kParseLaunch`.
10. Python duplicate insertion raises synchronously without a binding change.

Add any new unit executable to the explicit `UNIT_TESTS` list in `tests/CMakeLists.txt`.

## Documentation updates

- `include/builder/Node.h`: one Node object represents one logical vertex within a Graph.
- `include/pipeline/Graph.h`: document duplicate insertion and `connect()` reuse behavior.
- `Graph::custom()` documentation: explain explicit-name transformation and bin-scoped
  uniqueness.
- `ValidateOptions::enforce_names`: clarify that it does not disable mandatory duplicate-binding
  checks.
- `docs/reference/error-codes.md`: document duplicate bindings under `kPipelineShape`.
- Architecture documentation: describe automatic build-time invariant enforcement.
- Release notes: call out synchronous rejection of previously unsupported duplicate insertion.

No user workflow or example needs migration unless it relies on duplicate Node objects or invalid
explicit sibling names.

## Delivery sequence

The work can land as one pull request with four reviewable commits:

1. **Atomic composition identity**
   - transaction, preflight, insertion primitives, reuse semantics, and composition tests.
2. **Scope-aware explicit binding analysis**
   - shallow grammar, collision diagnostics, parse-boundary integration, and analyzer tests.
3. **Custom extraction and safe reference rewriting**
   - complete Custom names, UdpOutputGroupG references, and Neat alias rules.
4. **Runtime propagation, build coverage, and documentation**
   - connected reports, validation merging, RTSP coverage, public comments, and release notes.

## Definition of done

- No public C++ or Python signature changes.
- No public `Graph` object-layout change.
- Valid existing pipelines build and run without requiring `Graph::validate()`.
- Duplicate Node insertion fails atomically at composition time.
- Duplicate explicit sibling bindings fail automatically during build.
- Same short names in different child bins remain valid.
- Custom multi-element names and named-pad references remain synchronized after transformation.
- `gst_parse_launch()` remains authoritative for actual GStreamer construction.
- Connected and fused failures preserve structured diagnostics.
- RTSP fails synchronously for detectable explicit collisions without creating background resources.
- Relevant host and DevKit tests pass.
- Documentation and release notes describe the behavioral hardening.
