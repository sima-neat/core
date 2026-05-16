# Coverage Map (Examples and Tutorials)

This map links common tasks to concrete in-repo references. Tutorials live under
`tutorials/NNN_<name>/` with paired C++ and Python sources (plus a README).

## Core session and run patterns

- First model run (synchronous): `tutorials/001_run_your_first_model/`
- Async push/pull: `tutorials/002_run_inference_async/`
- Build a Session by hand (no model): `tutorials/003_build_inference_pipeline/`
- Plug a model into a Session: `tutorials/007_plug_model_into_pipeline/`

## Model configuration

- Model options (format, bounds, preproc, postproc): `tutorials/004_configure_model_options/`
- Preprocessing (normalize, channel stats): `tutorials/005_preprocess_images/`

## Model workflows

- Object detection (YOLOv8): `tutorials/012_detect_objects_with_yolov8/`
- Image classification (ResNet-50): `tutorials/013_classify_images_with_resnet50/`
- Decoding detection boxes (`SimaBoxDecode`): `tutorials/006_read_detection_boxes/`

## Tensor and sample handling

- NumPy/PyTorch interop: `tutorials/008_pass_numpy_to_model/`
- Multi-input (bundle) samples: `tutorials/009_feed_multi_input_model/`
- Reading `Sample` output: `tutorials/010_interpret_model_output/`

## Live streaming input

- Consume a live RTSP stream: `tutorials/017_consume_rtsp_stream/`

## Graph APIs

- Build a custom data graph: `tutorials/012_build_a_custom_data_graph/`
- Embed a model inside a graph: `tutorials/013_embed_model_inside_graph/`
- Run multiple streams in one graph: `tutorials/014_run_multiple_streams/`

## Performance and production

- Tune throughput and queues: `tutorials/015_tune_throughput_and_queues/`
- Production-ready pipeline skeleton: `tutorials/016_build_production_pipeline/`

## Diagnostics

- Diagnose and profile a pipeline: `tutorials/011_diagnose_a_pipeline/`

## Reference apps (not tutorials)

Larger end-to-end examples live in the `sima-neat/apps` repository under
`apps/examples/`. Use those for production-scale integration patterns (VideoSender
and MetadataSender publishing, multistream detection, face detection, pose, tracking, depth, RTSP
publishing, etc.). Tutorials in this repo stay minimal and single-topic.
