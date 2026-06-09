from __future__ import annotations

from . import _pyneat_core as core

TensorLike = core.Tensor
TensorList = list[core.Tensor]
SampleLike = core.Sample
RunLike = core.Run
GraphLike = core.Graph
ModelLike = core.Model
PreprocessRoiLike = getattr(core, "PreprocessRoi", object)
PreprocessRuntimeMetaLike = getattr(core, "PreprocessRuntimeMeta", object)
BoxDecodeTensorList = TensorList
DecodedBoxes = core.Tensor
DecodedBoxesList = list[core.Tensor]
PoseDecodeTensorsLike = getattr(core, "PoseDecodeTensors", object)
SegmentationDecodeTensorsLike = getattr(core, "SegmentationDecodeTensors", object)
PoseDecodeTensorList = list[PoseDecodeTensorsLike]
SegmentationDecodeTensorList = list[SegmentationDecodeTensorsLike]
