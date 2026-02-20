# Coverage Map (Examples and Tutorials)

This map links common tasks to concrete in-repo references.

## Core session and run patterns

- Sync build/run basics:
  - `tutorials/tutorial_0002_pipeline_session_build_run.cpp`
- Async push/pull:
  - `tutorials/tutorial_0006_async_push_pull.cpp`
  - `tutorials/tutorial_0018_async_queue_tuning.cpp`
- Sample/tensor output handling:
  - `tutorials/tutorial_0004_samples_and_neattensor.cpp`

## Caps and input control

- Caps negotiation and dynamic size behavior:
  - `tutorials/tutorial_0007_caps_negotiation.cpp`
- Advanced appsrc caps and buffer contracts:
  - `tutorials/tutorial_0022_inputappsrc_advanced_caps.cpp`

## Model workflows

- Basic model usage:
  - `tutorials/tutorial_0013_model_resnet50.cpp`
  - `examples/model_resnet50.cpp`
- End-to-end model options/stages/sync-async:
  - `tutorials/tutorial_0017_neatmodel_end_to_end.cpp`
- MPK pipeline composition:
  - `tutorials/tutorial_0011_mpk_yolov8_pipeline.cpp`
  - `examples/modelmpk_resnet50.cpp`

## Output policies and tensor conversion

- Output tensor options:
  - `tutorials/tutorial_0005_output_tensor_options.cpp`
- Appsink policy behavior:
  - `tutorials/tutorial_0021_output_appsink_policies.cpp`

## RTSP and source-driven flows

- RTSP server:
  - `tutorials/tutorial_0014_rtsp_server.cpp`
  - `examples/rtsp_server.cpp`
- RTSP-heavy applications:
  - `examples/decode_rtsp.cpp`
  - `examples/yolov8_multi_rtsp_demo.cpp`

## Hybrid graph and multistream

- Hybrid graph basics:
  - `tutorials/tutorial_0024_hybrid_graph.cpp`
- Hybrid multistream orchestration:
  - `tutorials/tutorial_0025_hybrid_multistream.cpp`
- Large graph production pattern:
  - `examples/GraphPipesYOLOOptiview.cpp`

## PCIe flows

- PCIe send/receive:
  - `tutorials/tutorial_0026_send_and_receive_via_pcie.cpp`
- PCIe model pipeline:
  - `tutorials/tutorial_0027_run_e2e_application_via_pcie.cpp`

## Diagnostics and validation

- Debugging and diagnostics workflow:
  - `tutorials/tutorial_0010_diagnostics_debug.cpp`
  - `tutorials/tutorial_0020_debugging_workflow.cpp`
- Builder graph/contracts:
  - `tutorials/tutorial_0023_builder_graph_and_contracts.cpp`
