# 004 Model Options Chapter

## Metadata
| Field | Value |
| --- | --- |
| Difficulty | Beginner |
| Estimated Read Time | 5 minutes |
| Labels | model-options, configuration, contracts |

## Concept
Model options define the runtime contract between your input data, model pipeline stages, and output decoding behavior.

This chapter focuses on the options most teams use first:
- `format`, `media_type`: declare incoming data type (for example raw RGB/BGR image input).
- `input_max_width`, `input_max_height`, `input_max_depth`: set dynamic input bounds for validation and runtime sizing.
- `preproc.*`: control preprocessing behavior (normalization, channel stats, image-type and resize policy overrides).
- `decode_type`, `score_threshold`, `nms_iou_threshold`, `top_k`: control detection-style postprocessing and filtering.
- `original_width`, `original_height`: provide original image geometry when postprocessing requires source-frame coordinates.
- `name_suffix`, `upstream_name`: stabilize/clarify generated stage naming when composing bigger pipelines.

Use-case guidance:
- Prototype classification quickly: set `format` + input max dimensions, keep postproc defaults minimal.
- Detection model bring-up (YOLO-style): set `decode_type` plus threshold/NMS/top-k options to shape final boxes.
- Mixed input sizes in one app: set `input_max_*` high enough for expected ranges to avoid runtime contract failures.
- Accuracy tuning after deployment: adjust `preproc.normalize`, `channel_mean`, `channel_stddev` to match model training assumptions.
- Multi-model or hybrid pipelines: use `name_suffix`/`upstream_name` to keep pipeline graph naming explicit and debuggable.

Reference:
- [Model Options](/reference/{lsa}/structs/simaai-neat-model-options)

## Learning Process
1. Build a `Model::Options` / `ModelOptions` config covering input, preproc, and postproc settings.
2. Instantiate a model with those options and inspect `input_spec()`, `output_spec()`, and metadata.
3. Run one deterministic inference path and observe how options influence runtime behavior.
4. Validate completion through `CHECK`, `SIGNATURE`, and `[OK]` markers.

## What To Observe
- `CHECK ...` lines should indicate contract and runtime validation outcomes.
- `SIGNATURE ...` output should summarize machine-parseable chapter behavior.
- `[OK] ...` indicates the chapter flow completed successfully.

## Run
```bash
./tutorial_v2_004_model_options_chapter
python3 tutorials/004_model_options_chapter/model_options_chapter.py
```

## Source Files
- C++: `tutorials/004_model_options_chapter/model_options_chapter.cpp`
- Python: `tutorials/004_model_options_chapter/model_options_chapter.py`
