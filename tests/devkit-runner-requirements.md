# Core DevKit runner requirements

This audit separates external-source reachability from use of the RTSP protocol.
All pools require the resolved platform group, exact firmware and Vulcan environment.

| Test scope | Runner requirement | Current canonical CI policy |
|---|---|---|
| Eight C++ registrations selected by `-L llima`: direct/graph LLM, VLM, ASR, HTTP server and reasoning server | Non-RTSP `modalix-som-16gig` | Required standard lane, serial isolated processes |
| `python/tests/test_genai_bindings.py` (10 current cases) | Non-RTSP `modalix-som-16gig` | Required standard lane, one fresh pytest process per case |
| `codec_stream_rtsp-*_{smoke,contract}` (six cases, two variants) | RTSP source-access pod | Weekly `all` stream profile |
| `codec_stream_http-mjpeg-*_{smoke,contract}` (two cases, two variants) | Source-access pod (RTSP label) | Weekly `http` or `all`; HTTP sources share the restricted network |
| `rtsp_encoded_boundary_caps_test`, `decoder_multirun_admission_boundary_test`, `h265_rtsp_resnet50_e2e_test` | External RTSP sources | Existing `core#890` exclusions remain; this change does not enable or count them as passed |
| C++ tutorial `tutorial_018_consume_rtsp_stream{,_h265}`, Python `test_rtsp_codec_tutorial[h264/h265]`, tutorial-018 runtime case | External RTSP sources | Existing `core#890` exclusions remain |
| `rtsp_pull_frame_test --local`, `resnet50_modelpack_rtsp`, `decoder_quality_compare_test`, `hybrid_graph_rtsp_resnet50_tput_test`, macro mixed-stream tests | Default regular DevKit | Start their own local RTSP servers; no external sources required |
| `graph_rtsp_lifecycle_regression_test`, `stress_graph_validate_rtsp_churn_test` | Default regular DevKit | Local server lifecycle; existing standard/weekly selectors retained |
| `codec_runtime_bad_url_diagnostics` | Default regular DevKit | Deliberately uses unreachable loopback `rtsp://127.0.0.1:1/...` |
| Python RTSP input/group contracts and tutorial FPS-probe unit tests | Default regular DevKit | Mocked/API-only coverage; no external connection |
| Other strict, E2E, performance, docs-example, Python binding and tutorial tests | Default regular DevKit | Existing canonical selectors unchanged; no RAM-size constraint |

Evidence is in `tests/CMakeLists.txt`, `codec_runtime_stream_test.cpp`, the three
external-source tests, the local-server test implementations, tutorial 018, and
`tests/tutorials/test_rtsp_codec_tutorial.py`. The canonical broad E2E lane excludes
`codec_stream`/`codec_aggregate` and the same three external-source test IDs as
before. Exact weekly stream cases run separately, never as a partial aggregate.

A standard run plans seven lanes across up to seven regular workers (five default
and two 16GB). A weekly source profile adds at most one source-pod worker. Limited
capacity combines lanes within a pool or queues a worker; it never drops tests.
CTest labels/resource locks named `rtsp` alone are not proof that external access
is required. Existing long/profile exclusions remain explicit coverage gaps.
