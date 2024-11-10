# Copyright (c) OpenMMLab. All rights reserved.
from .inference import (convert_SyncBN, inference_detector,
                        inference_mono_3d_detector,
                        inference_multi_modality_detector, inference_segmentor,
                        init_model)
from .inferencers import (Base3DInferencer, LidarDet3DInferencer,
                          LidarSeg3DInferencer, MonoDet3DInferencer,
                          MultiModalityDet3DInferencer)

# 当一个模块定义了__all__时，使用from module import *只会导入__all__列表中指定的名称。
# 这可以避免意外导入模块中的所有名称，从而提高代码的可读性和可维护性。
__all__ = [
    'inference_detector', 'init_model', 'inference_mono_3d_detector',
    'convert_SyncBN', 'inference_multi_modality_detector',
    'inference_segmentor', 'Base3DInferencer', 'MonoDet3DInferencer',
    'LidarDet3DInferencer', 'LidarSeg3DInferencer',
    'MultiModalityDet3DInferencer'
]
