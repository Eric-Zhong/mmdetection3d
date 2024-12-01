# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import read_base

with read_base():
    from .._base_.datasets.semantickitti import *
    from .._base_.default_runtime import *
    from .._base_.models.cylinder3d import *
    from .._base_.schedules.schedule_3x import *

from mmcv.transforms.wrappers import RandomChoice

from mmdet3d.datasets.transforms.transforms_3d import LaserMix, PolarMix

train_pipeline = [
    dict(type=LoadPointsFromFile, coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(
        type=LoadAnnotations3D,
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti'),
    dict(type=PointSegClassMapping),
    dict(
        type=RandomChoice,
        transforms=[
            [
                dict(
                    type=LaserMix,
                    num_areas=[3, 4, 5, 6],
                    pitch_angles=[-25, 3],
                    pre_transform=[
                        dict(
                            type=LoadPointsFromFile,
                            coord_type='LIDAR',
                            load_dim=4,
                            use_dim=4),
                        dict(
                            type=LoadAnnotations3D,
                            with_bbox_3d=False,
                            with_label_3d=False,
                            with_seg_3d=True,
                            seg_3d_dtype='np.int32',
                            seg_offset=2**16,
                            dataset_type='semantickitti'),
                        dict(type=PointSegClassMapping)
                    ],
                    prob=1)
            ],
            [
                dict(
                    type=PolarMix,
                    instance_classes=[0, 1, 2, 3, 4, 5, 6, 7],
                    swap_ratio=0.5,
                    rotate_paste_ratio=1.0,
                    pre_transform=[
                        dict(
                            type=LoadPointsFromFile,
                            coord_type='LIDAR',
                            load_dim=4,
                            use_dim=4),
                        dict(
                            type=LoadAnnotations3D,
                            with_bbox_3d=False,
                            with_label_3d=False,
                            with_seg_3d=True,
                            seg_3d_dtype='np.int32',
                            seg_offset=2**16,
                            dataset_type='semantickitti'),
                        dict(type=PointSegClassMapping)
                    ],
                    prob=1)
            ],
        ],
        prob=[0.5, 0.5]),
    dict(
        type=GlobalRotScaleTrans,
        rot_range=[0., 6.28318531],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
    ),
    dict(type=Pack3DDetInputs, keys=['points', 'pts_semantic_mask'])
]

train_dataloader.update(dict(dataset=dict(pipeline=train_pipeline)))

default_hooks.update(dict(checkpoint=dict(type=CheckpointHook, interval=1)))
