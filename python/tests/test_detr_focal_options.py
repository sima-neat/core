"""Native focal-query API is additive; classic DETR remains a distinct profile."""

import pyneat


def test_detr_focal_options():
    assert pyneat.BoxDecodeType.DetrFocal != pyneat.BoxDecodeType.Detr
    options = pyneat.ModelOptions()
    options.decode_type = pyneat.BoxDecodeType.DetrFocal
    options.num_classes = 80
    options.top_k = 300
    options.score_threshold = 0.0
    options.nms_iou_threshold = 0.0
    options.preprocess.resize.mode = pyneat.ResizeMode.Stretch
    assert options.decode_type == pyneat.BoxDecodeType.DetrFocal
    assert options.num_classes == 80
    assert options.top_k == 300
    assert options.score_threshold == options.nms_iou_threshold == 0.0


def test_raw_head_constructor_preserves_explicit_storage_contract():
    node = pyneat.nodes.sima_box_decode(
        pyneat.BoxDecodeType.DetrFocal,
        original_width=1280,
        original_height=720,
        model_width=640,
        model_height=640,
        detection_threshold=0.0,
        nms_iou_threshold=0.0,
        top_k=300,
        decode_type_option=pyneat.BoxDecodeTypeOption.InterleavedByHeadLogit,
        source_storage=pyneat.BoxDecodeSourceStorage.DenseHwc,
        detess=False,
        dequant=False,
        resize_mode=pyneat.ResizeMode.Stretch,
    )
    assert node.kind() == "SimaBoxDecode"
