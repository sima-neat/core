# Issue #458: Revisited Implementation Plan

> **Integration note:** This plan was reviewed against the original issue baseline. Before final
> integration, `develop` gained structured terminal-error propagation through `PullError` and
> `GraphReport` in #665. The implementation reuses that established carrier rather than adding the
> parallel `RuntimeFailure` type proposed below. The launch-validation and composition invariants
> are unchanged; runtime references below record the original gap analysis.

## Executive decision

Keep the two-invariant design, but change its internal implementation in four important ways:

1. Use a complete private `Node* -> VertexId` identity index rather than repeated linear scans.
2. Use an `O(delta)` rollback journal rather than copying the full composition for every mutation.
3. Validate final launch strings with a narrow binding lexer **and** the GStreamer object tree; do
   not clone GStreamer's full parser and do not trust `gst_parse_launch()` alone.
4. Define name uniqueness per materialized pipeline segment, not per public `Graph`. Within one
   segment, short names are globally unique because Neat resolves them with recursive short-name
   lookups. Independently parsed connected segments may reuse the same names.

The checks are automatic. Users do not need to call `Graph::validate()`.

There is no public C++ or Python signature change and no public `Graph` layout change. There is an
intentional behavior tightening: compositions and launch fragments that were ambiguous or could
silently lose an element now fail with a useful error.

## Why the previous plan needed revision

The earlier plan had the correct product goal, but review of the current implementation and
GStreamer exposed these gaps:

- A linear identity scan on every `add()` makes incremental construction `O(N^2)`.
- Copying the entire composition for every public mutation has the same asymptotic problem.
- `connect()` can modify existing edge options and can replace the edge vector during endpoint
  promotion, so append-only rollback is insufficient unless those writes are journaled.
- GStreamer can silently discard a duplicate-named child while returning a non-null pipeline and
  no `GError`. `GST_PARSE_FLAG_FATAL_ERRORS` does not repair an error that GStreamer never reports.
- Neat uses `gst_bin_get_by_name()` in build, validation, pull, and trace paths. That lookup is
  recursive, so equal short names in different nested bins are legal GStreamer but ambiguous Neat.
- A full scope-aware clone of GStreamer's Flex/Bison grammar would be large, fragile, and subject
  to upstream syntax drift.
- A seedless connected graph can materialize input-dependent segments only on first push. The
  correct guarantee is automatic enforcement at materialization, not that every error is returned
  synchronously from the initial `build()` call.
- RTSP stores a launch string and parses it later. Dry-parsing it eagerly constructs plugins twice
  and can consume scarce target resources or advance generated-name counters.

## Current-source findings

The implementation must account for these concrete paths:

- `Graph::add()` appends without an identity check in `src/pipeline/graph/Graph.cpp`.
- `CompositionGraph::append_vertex()` adds the implicit linear edge in
  `src/pipeline/graph/GraphComposition.cpp`.
- Pipeline vertices are also inserted directly by fragment import, output collection import,
  internal graph construction, and connected graph loading in `GraphIo.cpp`.
- `CompositionGraph::imported_nodes` is incomplete: direct `add()` and graph imports do not
  populate it.
- Public `connect()` overloads import the first operand before validating the second operand and
  endpoint match. A later exception can leave partial vertices, metadata, or import-map entries
  without a version increment.
- `promote_endpoint_mode_or_throw()` replaces/prunes the edge vector.
- `connect_endpoint()` and `connect_runtime_port()` can update existing real-time fan-in edge
  options before appending the new edge.
- `add_output_tensor()` is four public `add()` calls and can currently commit a prefix.
- All ordinary production parses converge on the helper in
  `src/pipeline/graph/GraphBuildPipeline.cpp`. RTSP is the separate delayed-parse path.
- `RunCore::ensure_graph_pipeline_built()` catches `NeatError` as `std::exception` and loses its
  `GraphReport`.
- `CustomNode::element_names()` in `src/nodes/common/Caps.cpp` returns only the first `name=`
  occurrence.
- `UdpOutputGroupG` declares `demux` and `render`, then refers to `demux.bbox`, `demux.image`,
  `render.sink_0`, and `render.sink_1`. The current rewriter can rename a declaration without
  renaming those references.
- The current generic rewrite includes `next-element`, even though decoder values such as `CVU`
  and `APU` are selectors rather than GStreamer element references.

## Invariant A: one Node object, one composition vertex

Within one `Graph` composition:

```text
one non-null pipeline Node* identity -> exactly zero or one composition vertex
```

Runtime-stage vertices and null compatibility entries are outside this identity domain.

### Required behavior

- `graph.add(node)` twice rejects the second call.
- Importing a new graph or model fragment that aliases an existing `Node*` rejects atomically.
- Importing the same graph ID and version again reuses its existing vertex range.
- `graph.add(node)` followed by `graph.connect(node, sink)` reuses the existing vertex.
- `connect(node, sink_a)` followed by `connect(node, sink_b)` reuses one vertex and permits
  legitimate fan-out.
- Two separately constructed instances of the same Node type remain valid.
- The same Node object may exist in two independent Graphs; the collision matters only when both
  are composed into one destination.

### Complete private identity index

Replace `imported_nodes` with an index that covers every pipeline insertion:

```cpp
std::unordered_map<const Node*, VertexId> pipeline_vertex_by_identity;
```

Add private `CompositionGraph` APIs:

```cpp
std::optional<VertexId>
find_pipeline_vertex(const Node* node) const noexcept;

void preflight_pipeline_vertices(
    std::span<const CompositionVertex> incoming,
    std::string_view operation) const;

VertexId insert_linear_pipeline_vertex(
    NodePtr node,
    CompositionMutation& mutation,
    std::string_view operation);

VertexId insert_unlinked_pipeline_vertex(
    NodePtr node,
    CompositionMutation& mutation,
    std::string_view operation);

void verify_identity_index_or_throw() const;
```

Rules for the index:

1. Index only non-null `Kind::PipelineNode` vertices.
2. Use raw object identity, not shared-pointer control-block identity.
3. Reject duplicate insertion in the low-level primitive. Reuse must be an explicit higher-level
   decision in `import_or_reuse_*()`.
4. Reserve once before bulk import.
5. Detect duplicates within the incoming fragment with a temporary set before mutation.
6. Retain the incoming source vertex index in diagnostics; never compact mixed pipeline/runtime
   vertex arrays before rebasing edges.
7. Verify index/vector equivalence in debug tests and at the internal compile boundary in a cheap
   assertion-enabled build.

The private `CompositionGraph` already lives behind `std::unique_ptr`, so this field does not alter
the public `Graph` object layout. Move operations naturally move the vector and index together.

### Separate linear and unlinked insertion

Do not make one insertion primitive guess topology:

- `insert_linear_pipeline_vertex()` preserves the current implicit `tail -> new` edge.
- `insert_unlinked_pipeline_vertex()` inserts no edge because topology will be copied or loaded
  separately.

Use the unlinked form for connected fragment import, output collection import, internal graph
construction, and `GraphIo.cpp` load. Routing load through the linear path would add edges not
present in serialized topology.

Direct insertion into `vertices` should remain only for runtime vertices, with a comment stating
that pipeline vertices must use the indexed primitives.

## Atomic mutation without full-graph copies

Every public composition operation must provide the strong logical exception guarantee:

```text
success -> the complete operation is visible and the version increments once
failure -> vertices, edges, metadata, maps, version, caches, and ownership are unchanged
```

Capacity growth is not observable state and need not be rolled back.

### `CompositionMutation` rollback journal

Use a private RAII journal with size/scalar checkpoints and sparse before-images:

```cpp
class Graph::CompositionMutation final {
public:
  explicit CompositionMutation(Graph& owner);
  ~CompositionMutation() noexcept;

  void before_edge_write(std::size_t edge_index);
  void replace_edges(std::vector<CompositionEdge> replacement);
  void before_imported_fragment_insert(std::uint64_t key);
  void before_imported_model_insert(std::string_view key);
  void commit() noexcept;

private:
  Graph& owner_;
  bool committed_ = false;
  bool composition_was_null_ = false;

  std::size_t vertices_size_ = 0;
  std::size_t edges_size_ = 0;
  std::size_t fragments_size_ = 0;
  std::size_t named_fragments_size_ = 0;
  std::size_t groups_size_ = 0;
  VertexId tail_ = CompositionGraph::kInvalid;
  bool endpoint_mode_ = false;
  InputRouteProcessorPtr input_route_processor_;

  std::vector<std::pair<std::size_t, CompositionEdge>> changed_edges_;
  std::optional<std::vector<CompositionEdge>> replaced_edges_;
  std::vector<std::uint64_t> inserted_fragment_keys_;
  std::vector<std::string> inserted_model_keys_;
};
```

The exact types may follow existing aliases, but the behavior must be fixed.

Rollback order:

1. Swap back a replaced edge vector, or truncate appended edges and swap first-write
   before-images into modified existing edges.
2. Erase import-map keys inserted by this operation.
3. Erase identity-index entries for vertices appended after `vertices_size_`.
4. Truncate appended vertices, fragments, named fragments, and `groups_`.
5. Restore `tail`, `endpoint_mode`, and `input_route_processor_`.
6. Restore a null `composition_` if the operation began in that state.

Correctness constraints:

- Record an existing edge before its first write and store its index, never a reference that vector
  reallocation can invalidate.
- Make before-image allocation/copy complete before modifying the edge.
- Build endpoint-promotion `kept` edges completely, then install them through `replace_edges()`.
- Log an import-map key before `emplace()`; erasing a nonexistent key during rollback is harmless.
- Use only nothrow swaps, erases, truncation, shared-pointer restoration, and destruction in the
  rollback path. Add `static_assert` checks for the required nothrow swaps.
- Commit calls `mark_composition_changed()` exactly once. Until commit, `built_`, `run_cache_`,
  `nodes_version_`, `built_version_`, and `last_pipeline_` remain untouched.
- Do not support nested independent transactions. Composite public helpers must use private
  transaction-aware insertion functions rather than recursively calling public mutators.

Apply one transaction around every public `add()` and `connect()` overload, around
`add_output_tensor()` as a four-node batch, and around internal mutations when the Graph remains
observable after an exception. Local temporary graph builders still use the central insertion
primitives even if destruction already supplies their rollback.

### Complexity target

| Operation | Target complexity |
| --- | --- |
| `add(node)` | Average `O(1)` |
| Import K vertices/E edges | `O(K + E + metadata)` |
| Indexed Node reuse | Average `O(1)` |
| Ordinary connect | Existing endpoint/edge scan plus `O(delta)` |
| Endpoint promotion | `O(E)`, matching current work |
| Rollback | `O(delta + modified existing edges)` |

Add a construction benchmark or complexity regression for at least 1,000 incremental Nodes. This
is setup-path work, not frame-path work, but it should not become quadratic unnecessarily.

Graph mutation, build, and execution on the same object remain non-thread-safe. The atomic version
counter does not imply concurrent mutation support, and this change should not add a mutation
mutex.

## Invariant B: unique short names per materialized segment

The Neat contract should be:

```text
within one final parsed pipeline segment, each short GstObject element name is unique
```

This is intentionally stricter than GStreamer's sibling-only rule. Neat repeatedly calls
`gst_bin_get_by_name()`, which recursively finds a descendant by short name, and launch references
can cross bin boundaries. Allowing the same short name in nested bins would make lookup and
attribution order-dependent. Separate connected segments are separately parsed objects, so they may
reuse names safely.

If nested duplicate short names must be supported in the future, first replace recursive short-name
lookups with retained object handles or full object paths. That migration is outside issue #458.

### Why GStreamer parsing alone is insufficient

GStreamer checks name uniqueness when adding a child to a bin. Its launch grammar, however, ignores
the Boolean return from some `gst_bin_add()` calls. A duplicate child can be destroyed while the
parser returns a non-null, truncated pipeline without a `GError`. `GST_PARSE_FLAG_FATAL_ERRORS`
only makes an existing parse error fatal; it cannot promote an error that was never recorded.

Therefore the required design is:

```text
final string
  -> exact explicit-binding analysis
  -> segment-global duplicate precheck
  -> one native parse
  -> recursive object-tree inventory
  -> declaration/result reconciliation
  -> downstream configuration
```

Do not install a temporary GStreamer debug log callback. It is process-global, concurrency-sensitive,
message-text-dependent, and unsuitable for an embeddable library.

## Narrow reusable launch-binding layer

Add a private, clean-room lexical utility:

```text
src/pipeline/internal/GstLaunchBindings.h
src/pipeline/internal/GstLaunchBindings.cpp
```

Suggested interface:

```cpp
namespace simaai::neat::pipeline_internal::gst_launch {

struct ByteSpan {
  std::size_t begin = 0;
  std::size_t end = 0;
};

enum class TokenKind {
  Assignment,
  Reference,
  BinReference,
  Link,
  Url,
  Operator,
};

struct Assignment {
  std::string_view key;
  std::string canonical_value;
  ByteSpan token_span;
  ByteSpan value_span;
};

struct Reference {
  std::string canonical_element_name;
  ByteSpan element_span;
};

struct Analysis {
  std::vector<Assignment> assignments;
  std::vector<Reference> references;
  std::vector<Diagnostic> diagnostics;
  bool complete = true;
};

Analysis analyze(std::string_view launch);

RewriteResult rewrite(
    std::string_view launch,
    const Analysis& analysis,
    const NameMapping& mapping,
    std::span<const std::string_view> alias_properties);

} // namespace simaai::neat::pipeline_internal::gst_launch
```

This layer is not a GStreamer grammar or topology parser. It recognizes only the documented lexical
categories needed to find exact assignments and true named-element references. It must use
maximal-token matching so it can distinguish:

- `name` from `encoding-name`;
- a property assignment from a direct caps field;
- `caps=video/x-raw,name=fake` from a separate `name` property;
- URLs and quoted configuration containing `name=` from declarations;
- `element.pad` references from dotted URLs or property values;
- `! CAPS !` and `: CAPS :` link tokens, including features and semicolon-separated caps;
- escaped whitespace and operators.

Canonicalize values to match supported GStreamer behavior:

```text
name=dup       -> dup
name="dup"     -> dup
name='dup'     -> 'dup'
name=foo\ bar  -> foo bar
```

The API receives a launch string, not shell source. Do not apply shell-quote rules.

Keep raw byte spans and allocate canonical strings only for relevant values. The analyzer and
rewriter must be linear in input plus output size. Build rewritten output in one forward pass over
sorted, non-overlapping replacements; repeated in-place `std::string::replace()` can become
quadratic.

If analysis is incomplete because of malformed quoting or an unsupported lexical form, never guess
or partially rewrite. At the mandatory build boundary, a native parse still runs; a parser-success
with incomplete binding analysis fails closed with a pipeline-shape diagnostic because Neat cannot
prove its naming invariant.

Do not copy GStreamer's LGPL Flex source or generated scanner into this Apache-2.0 repository.
Implement the small lexer independently from public syntax documentation and lock behavior with
differential tests against every supported target GStreamer version.

## Native parse and post-parse audit

Use the existing neutral files:

```text
src/pipeline/internal/GstParseLaunch.h
src/gst/GstParseLaunch.cpp
```

They should remain independent of `Graph`, `NeatError`, and `GraphReport`. The wrapper should:

1. Call `gst_parse_launch_full(..., GST_PARSE_FLAG_FATAL_ERRORS, ...)`.
2. Return an RAII-owned root and structured native error details.
3. Recursively inventory all constructed descendant elements as short name, full object path,
   factory/type, and parent path.
4. Count actual occurrences of each short name.
5. Reconcile every explicit binding from the final-string analysis with the actual tree.

The Graph layer applies policy:

1. Before parsing, reject any canonical explicit name with more than one declaration in the final
   segment. Include both byte spans and available Node/generated provenance.
2. Parse exactly once and retain that root for the build.
3. Reject any actual short name occurring more than once anywhere in the parsed segment.
4. Require each explicit declaration to resolve to exactly one actual object. A missing binding
   indicates that construction dropped an element or that rendered metadata drifted.
5. Report native syntax, link, or plugin failures as `kParseLaunch`; report binding collisions,
   ambiguity, or declaration-survival failures as `kPipelineShape`.

This design avoids a second render and a second parse. It adds one `O(launch bytes)` lexical pass
and one `O(actual elements)` tree walk on build/materialization only.

### Render provenance and generated names

Extend private build metadata with a lightweight name-origin ledger:

```cpp
struct NameOrigin {
  enum class Kind { NodeFragment, Boundary, Queue, Tap, AppSrc, AppSink, Mux, RtspHelper };
  Kind kind;
  int node_index = -1;
  std::string node_kind;
  std::string role;
};
```

`NodeFragment` registers its declared names; framework code registers every generated queue,
boundary, tap, appsrc, appsink, mux, and RTSP helper. Analyze the final string after all fast-path,
sync, and fused transformations; the final scan is authoritative, while the ledger supplies
attribution. Reconcile by canonical name and occurrence order, and mark unmatched declarations as
`Unknown` rather than omitting them.

Give every Neat-owned element an explicit deterministic name where feasible. This removes
framework-owned explicit/generated collisions and stabilizes reports. Do not attempt to predict
process-global GStreamer auto-name counters.

Arbitrary unnamed elements inside `Custom` remain governed by GStreamer-generated names. The
strong Neat guarantee covers all explicit declarations, all Neat-owned generated elements, and the
actual tree returned by parsing; it cannot prove that GStreamer did not silently drop an unnamed
Custom element after an explicit/generated collision. Either require explicit names for every
multi-element Custom fragment in a future contract revision or fix/check this behavior upstream.
Do not hide this limitation in release claims.

## Custom fragments and safe transformation

Validation and rewriting share tokens but remain separate operations and commits.

Update `make_node_fragment()` to:

1. Call `backend_fragment(index)` once.
2. Analyze that exact string.
3. Read `element_names(index)` as attribution metadata, not as the final authority.
4. Build the stable union of Node-reported names and certain explicit bindings.
5. Apply the transform in one forward pass.
6. Rewrite exact `name` assignment values and actual named-element/pad reference tokens together.
7. Apply only explicitly registered SiMa alias rules.
8. Reanalyze the exact transformed result and populate all effective explicit names.

Generic rewriting is limited to:

- exact `name` assignments;
- the element component of true reference tokens such as `demux.bbox`.

Graph-specific alias rules may rewrite:

- `stage-id` when its complete value is an element alias;
- `op-buff-name` when its documented contract identifies an element/buffer alias.

Never generically rewrite `next-element`. Preserve RTSP `pay0`, `pay1`, and related server
semantics.

For `CustomNode`:

- return every certain explicit binding from `element_names()` in declaration order;
- inject an automatic name only when analysis proves a single ordinary unnamed element;
- do not inject into complex, bin, URI, malformed, or ambiguous fragments;
- reuse assignment spans for config JSON detection, path extraction, and path rewriting instead of
  separate substring scans or global `replace_all()` calls.

The mandatory regression is `UdpOutputGroupG`: both declarations and all four named-pad references
must receive the same mapping. Do not add a node-specific workaround; the generic editor should
solve it.

## Automatic build integration

Do not call the full `Graph::validate()` workflow from `build()`. That would render, parse, and
potentially preroll twice.

At the sole ordinary parse boundary in `GraphBuildPipeline.cpp`:

```text
final render
-> analyze exact bindings
-> enforce segment policy
-> parse once with the neutral wrapper
-> audit actual tree and binding survival
-> continue with the same GstElement root
```

This covers source, seeded-input, one-shot, fused real-time, and every connected segment when that
segment materializes.

Precise timing contract:

- Eager complete segments fail during `Graph::build()`.
- Input-dependent connected segments fail on first materialization, commonly the first `push()`.
- No path requires an explicit `Graph::validate()` call.

`Graph::validate()` reuses the same helper and keeps its optional PAUSED/preroll and ownership
checks. `ValidateOptions::enforce_names` controls only the optional ownership/attribution contract;
it cannot disable mandatory name integrity.

Current non-linear `Graph::validate()` is topology-only and does not render all connected segments.
Document that limitation rather than constructing speculative placeholder pipelines. The mandatory
check still runs when each real segment is built.

## RTSP integration

Set `last_pipeline_` before validation so failed launches remain diagnosable. Then run final-string
binding analysis and the segment-global explicit-name precheck before creating the implementation,
factory, handle, or thread and before `gst_rtsp_media_factory_set_launch()`.

Do **not** dry-parse the RTSP launch by default. It would construct elements twice, can retain scarce
encoder/allocator resources, and changes process-global generated-name counters. The first patch
should provide exact synchronous detection for explicit collisions and deterministic Neat-owned
names, while native RTSP construction errors remain at media creation.

If exact post-parse audit is required for RTSP, implement a small custom
`GstRTSPMediaFactoryClass::create_element` override that invokes the neutral parse/audit wrapper and
returns that one audited bin. Do not accept double instantiation as the permanent design. Keep this
as a separately reviewable RTSP phase because it changes GObject lifecycle integration.

## Structured errors and lazy connected builds

### Composition errors

Continue using `std::runtime_error` for synchronous structural misuse so Python still receives
`RuntimeError` without binding changes. Include Node kind/label, existing vertex, incoming source
index, and the remediation: construct a distinct Node or reuse it through `connect()`.

### Launch errors

Use `error_codes::kPipelineShape` for duplicate explicit bindings, ambiguous actual short names,
incomplete analyzable syntax accepted by GStreamer, and missing declaration after construction.
Include canonical name, byte spans, origins, parent/full paths when available, the final pipeline,
and an explicit note that automatic renaming is unsafe.

Use `error_codes::kParseLaunch` for native grammar, linking, missing-plugin, and element-construction
failures not classified by the binding audit.

### Runtime failure carrier

Add one internal failure object protected by the existing error mutex:

```cpp
struct RuntimeFailure {
  std::string message;
  std::string error_code;
  std::optional<GraphReport> report;
};
```

Store the first terminal failure rather than maintaining unrelated string and report fields.
`RunCore::ensure_graph_pipeline_built()` catches `NeatError` before `std::exception`, clears its
building flag, notifies waiters, and forwards the full report through `graph_request_stop()`.
Seeded prebuild, routers, background segment workers, named/unnamed pull, and throwing convenience
paths all consume the same carrier. Populate the existing public `PullError::report`; no API change
is required.

In `GraphValidate.cpp`, preserve a caught `NeatError::report()` and fill only fields missing from the
current `BuildResult`. Do not rewrite every structured failure as `kParseLaunch`.

## File-by-file implementation map

### Composition

- `include/pipeline/Graph.h`
  - Add private nested transaction declarations and update contract comments only.
- `src/pipeline/graph/GraphDetail.h`
  - Add the identity index, indexed insertion methods, verification, and transaction declarations.
- `src/pipeline/graph/GraphComposition.cpp`
  - Implement identity/preflight/insertion and transaction-aware edge mutation.
- `src/pipeline/graph/Graph.cpp`
  - Apply operation-level transactions, remove `imported_nodes`, centralize reuse, and batch
    `add_output_tensor()`.
- `src/pipeline/graph/GraphIo.cpp`
  - Load through unlinked indexed insertion without changing serialized topology.
- `src/genai/GraphFragments.cpp`
  - Keep internal pipeline construction on the same insertion primitives.

### Launch analysis, parse, and rewriting

- `src/pipeline/internal/GstLaunchBindings.h`
- `src/pipeline/internal/GstLaunchBindings.cpp`
  - Add the pure lexical analysis and lossless editor.
- `src/pipeline/internal/GstParseLaunch.h`
- `src/gst/GstParseLaunch.cpp`
  - Fill the existing neutral parse wrapper and recursive object inventory.
- `src/pipeline/graph/GraphNaming.cpp`
  - Replace ad hoc property rewriting and separate SiMa alias semantics.
- `src/nodes/common/Caps.cpp`
  - Replace first-name extraction and unsafe config substring/path rewrites.
- `src/nodes/groups/UdpOutputGroupG.cpp`
  - No special workaround; keep it as the regression fixture.

### Build, RTSP, and diagnostics

- `src/pipeline/graph/GraphBuildPipeline.cpp`
  - Integrate mandatory final-string analysis, one parse, and post-parse audit.
- `src/pipeline/graph/GraphBuild.cpp`
- `src/pipeline/graph/GraphBuildInput.cpp`
- `src/pipeline/graph/GraphBuildSource.cpp`
  - Register deterministic framework names and origin metadata after final transformations.
- `src/pipeline/graph/GraphRtsp.cpp`
  - Precheck before resources and optionally add the single-parse custom factory phase.
- `src/pipeline/graph/GraphValidate.cpp`
  - Preserve structured reports and clarify non-linear validation behavior.
- `src/pipeline/runtime/RunCore.h`
- `src/pipeline/runtime/RunCore.cpp`
- `src/pipeline/runtime/RunCoreGraphStart.cpp`
- `src/pipeline/runtime/RunPull.cpp`
  - Carry one structured terminal failure through lazy materialization and pull.

## Test strategy

### Composition and transaction tests

1. Duplicate `add(node)` rejects with no logical state or version change.
2. An overlapping `add(Graph)` rejects atomically.
3. `connect(a, b)` where the operands share a Node rolls back the entire operation.
4. A valid connect after that failure proves no stale import entry remains.
5. `add(node)` then `connect(node, sink)` reuses one vertex.
6. One existing Node fans out to distinct sinks through one vertex.
7. Repeated import of one Graph ID/version reuses its range.
8. Distinct Node instances of one type succeed.
9. Null compatibility and runtime vertices never enter the index.
10. Internal unlinked insertion rejects duplicate pipeline identity.
11. Move and save/load preserve index/vector consistency and exact topology.
12. Endpoint promotion failure restores pruned edges.
13. Realtime fan-in failure restores modified link options and stream IDs.
14. `add_output_tensor()` either adds all four Nodes or none.
15. Synthetic incremental construction demonstrates non-quadratic scaling.

### Binding lexer/editor tests

1. Exact `name`, whitespace around `=`, double quotes, literal single quotes, and escapes.
2. `encoding-name`, URL/config text, and caps fields do not count.
3. `! CAPS !`, `: CAPS :`, features, semicolon caps, and `caps=...name=...` remain opaque.
4. Real forward/backward named-pad references are identified.
5. Malformed input completes in linear time and never partially rewrites.
6. Canonicalization is differential-tested on the host and supported DevKit GStreamer versions.
7. Fuzz tests assert bounds safety, non-overlapping replacements, idempotence, and linear growth.

### Parse and policy tests

1. Linked explicit duplicate names reject.
2. Unlinked duplicates that GStreamer otherwise silently drops reject before acceptance.
3. Equal short names in nested bins reject under the documented Neat segment policy.
4. The same names in separately parsed connected segments succeed.
5. Repeated `name` properties cannot silently lose a declaration.
6. Actual object inventory is recursive and reports full paths.
7. Every explicit binding resolves to exactly one actual short name; the arbitrary unnamed Custom
   limitation is covered separately.
8. Explicit/generated interactions are covered for all Neat-owned elements by deterministic
   explicit naming.
9. Ordinary syntax/plugin/link failures remain `kParseLaunch`.

### Custom and integration tests

1. `UdpOutputGroupG` declarations and all named-pad references transform together.
2. URLs, caps, dotted property values, and quoted configuration do not change.
3. `stage-id` and `op-buff-name` follow explicit rules.
4. `next-element=CVU/APU` and RTSP `payN` semantics remain unchanged.
5. Normal, seeded, source, one-shot, fused, and eager connected builds reject without an explicit
   `validate()` call.
6. A lazy connected failure retains `kPipelineShape` and its original `GraphReport` at push/pull.
7. RTSP explicit collision fails before implementation/factory/thread creation and preserves
   `last_pipeline_`.
8. Python duplicate insertion raises synchronously without binding changes.

Add every new unit executable to the explicit `UNIT_TESTS` list in `tests/CMakeLists.txt`. Run host
tests locally when `file` reports x86-64. Run aarch64 binaries and target/plugin integration tests
through `tools/run-on-devkit`, `tools/ctest-on-devkit`, or `dk`; never stop at `Exec format error`.

## Delivery sequence

Land one pull request as independently reviewable commits:

1. **Indexed identity and delta transaction**
   - Complete identity map, central insertion, rollback journal, and composition tests.
2. **Binding lexer and editor**
   - Clean-room token layer, canonicalization, differential corpus, and fuzz tests.
3. **Custom extraction and safe rewrite**
   - Complete names, reference rewrite, explicit alias rules, config spans, and Udp regression.
4. **Mandatory parse-boundary policy**
   - Global segment precheck, one native parse, recursive audit, deterministic framework names.
5. **RTSP and connected error propagation**
   - Synchronous RTSP precheck, optional custom factory phase, structured runtime failure state.
6. **Validation, documentation, and compatibility gates**
   - Report merging, public comments, release notes, ABI/header check, Python regression.

Do not combine the lexer/editor rewrite with the identity commit. The core composition fix remains
easy to review and revert if launch transformation needs more iteration.

## Documentation and compatibility

Update:

- `include/builder/Node.h`: one Node object is one logical vertex per composition.
- `include/pipeline/Graph.h`: duplicate insertion, explicit reuse, atomic mutation, and lazy error
  timing.
- `Graph::custom()` documentation: segment-global explicit-name policy and conservative rewrite.
- `ValidateOptions::enforce_names`: optional ownership only; mandatory integrity is always on.
- `docs/doxygen/pipeline_concepts.md`: composition identity and materialized-segment validation.
- `docs/reference/error-codes.md`: duplicate/ambiguous/missing binding as `kPipelineShape`.
- `docs/release-notes/neat.md`: behavior hardening and no required `validate()` call.

No migration is needed for valid graphs with distinct Node objects and unambiguous names. Users who
relied on duplicate short names in nested bins must make them unique; supporting that pattern would
require path-based object ownership throughout Neat.

Add an ABI/header-surface check to CI. Do not rely only on source inspection when claiming no public
ABI change.

## Primary-source evidence

- GStreamer documents object-name uniqueness within a parent:
  [GstObject design](https://gstreamer.freedesktop.org/documentation/additional/design/gstobject.html).
- The parse API and `GST_PARSE_FLAG_FATAL_ERRORS` behavior are documented in
  [GstParse](https://gstreamer.freedesktop.org/documentation/gstreamer/gstparse.html).
- The official parser grammar adds elements to bins without handling every failed `gst_bin_add()`:
  [GStreamer grammar](https://github.com/GStreamer/gstreamer/blob/1.24/subprojects/gstreamer/gst/parse/grammar.y.in).
- `gst_bin_add()` checks child-name uniqueness and rejects duplicates:
  [GstBin source](https://github.com/GStreamer/gstreamer/blob/1.24/subprojects/gstreamer/gst/gstbin.c).
- Launch tokens, named references, caps links, quoting, and escaping are described by the
  [gst-launch manual](https://gstreamer.freedesktop.org/documentation/tools/gst-launch.html).
- `gst_bin_get_by_name()` recursively searches descendants:
  [GstBin API](https://gstreamer.freedesktop.org/documentation/gstreamer/gstbin.html).
- RTSP factory launch parsing is delayed until element creation:
  [RTSP media factory API](https://gstreamer.freedesktop.org/documentation/gst-rtsp-server/rtsp-media-factory.html)
  and [implementation](https://github.com/GStreamer/gstreamer/blob/1.24/subprojects/gst-rtsp-server/gst/rtsp-server/rtsp-media-factory.c).

The implementation must be tested against the exact GStreamer versions shipped on every supported
DevKit. Host behavior alone is not an acceptance criterion.

## Acceptance gates

### Required before merge

- No public C++/Python signature or public object-layout change.
- Identity index and vertex vector agree after add, import, connect, move, rollback, and load.
- Every public mutation is atomic under all tested failure points.
- Incremental composition is not quadratic.
- Final explicit bindings are analyzed after all string transformations.
- Ordinary builds parse once, retain that root, and audit it recursively.
- Duplicate or missing explicit declarations fail automatically without `validate()`.
- Separate parsed segments may reuse names; one segment may not.
- Custom declarations and named-pad references transform consistently.
- Lazy connected failures preserve their structured report.
- RTSP explicit collisions fail before background resources are created.
- Host and DevKit differential suites pass.
- ABI/header-surface and Python regressions pass.

### Explicitly deferred

- Path-based support for duplicate short names in nested bins.
- Automatic renaming of final collisions.
- Predicting GStreamer's process-global auto-generated names.
- A full Gst launch grammar or copied upstream lexer.
- Strong dropped-unnamed-element proof for arbitrary multi-element `Custom` fragments.
- A custom RTSP factory unless single-parse post-construction audit is required for this release.

## Definition of done

Issue #458 is closed when duplicate Node identity is impossible in a composition, explicit launch
bindings cannot collide with other explicit or Neat-owned bindings in any materialized ordinary
segment, Custom multi-name references remain coherent, all mandatory checks happen automatically,
and failures remain structured across eager, lazy, fused, and RTSP paths within the stated RTSP
and unnamed Custom limits.

## Implementation record

Status: implemented end to end in the working tree on July 31, 2026.

- The complete private identity index and delta rollback journal cover add, import, endpoint
  promotion, realtime fan-in, internal insertion, load, move, and the four-Node output helper.
- The private launch binding analyzer/editor, native one-parse tree audit, provenance ledger,
  deterministic framework names, RTSP precheck, and structured lazy failure carrier are wired into
  the production materialization paths. No explicit `validate()` call is required.
- The binding analyzer follows GStreamer's `_string` longest-match behavior: paired apostrophes may
  group a token but remain literal in the property value, while unmatched quotes fall back to the
  ordinary token alternative. Differential tests compare these edge cases with native parsing.
- Eager seeded connected builds retain the original typed `NeatError` and materialized launch
  string instead of flattening it through the generic graph-start error wrapper. The detail is
  returned by the current segment's build attempt, so an earlier segment's graph-global terminal
  error cannot be misattributed.
- Public C++ and Python signatures are unchanged. The ARM64 `Graph` layout remains pinned at 688
  bytes, and the public header-surface gate passes.
- The complete ARM64 build, including the Python extension, passes.
- The binding differential passes against host GStreamer 1.24.2 and DevKit GStreamer 1.22.0.
- The focused DevKit acceptance matrix passes 18 of 18 executables, including composition
  rollback, lexer/audit, seeded/source/one-shot/eager/lazy/fused/RTSP integration, structured
  failures, save/load, naming, API layout, and header surface.
- The DevKit Python API-surface suite passes 50 tests with 18 environment-dependent skips.
