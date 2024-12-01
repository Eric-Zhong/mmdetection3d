# For ScanNet seg we usually do 20-class segmentation
class_names = (
'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table',
'door', 'window', 'bookshelf', 'picture', 'counter', 'desk',
'curtain', 'refrigerator', 'showercurtrain', 'toilet', 'sink',
'bathtub', 'otherfurniture'
)

class_names_cn = (
' 墙 ', ' 地板 ', ' 橱柜 ', ' 床 ', ' 椅子 ', ' 沙发 ', ' 桌子 ',
' 门 ', ' 窗户 ', ' 书架 ', ' 图画 ', ' 柜台 ', ' 书桌 ',
' 窗帘 ', ' 冰箱 ', ' 淋浴帘 ', ' 马桶 ', ' 水槽 ',
' 浴缸 ', ' 其他家具 '
)

metainfo = dict(classes=class_names)
dataset_type = 'ScanNetSegDataset'
data_root = 'data/scannet/'
input_modality = dict(use_lidar=True, use_camera=False)
data_prefix = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask')

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection3d/scannet/'

# Method 2: Use backend_args, file_client_args in versions before 1.1.0
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection3d/',
#          'data/': 's3://openmmlab/datasets/detection3d/'
#      }))
backend_args = None

num_points = 8192

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True,
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='IndoorPatchPointSample',
        num_points=num_points,
        block_size=1.5,
        ignore_index=len(class_names),
        use_normalized_coord=False,
        enlarge_size=0.2,
        min_unique_num=None),
    dict(type='NormalizePointsColor', color_mean=None),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True,
        backend_args=backend_args),
    dict(type='NormalizePointsColor', color_mean=None),
    dict(type='Pack3DDetInputs', keys=['points'])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
# we need to load gt seg_mask!
eval_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
        backend_args=backend_args),
    dict(type='NormalizePointsColor', color_mean=None),
    dict(type='Pack3DDetInputs', keys=['points'])
]

# Test - Time Augmentation 测试时数据增强
# 这种技术主要是在模型测试阶段，通过对测试数据进行一系列的数据增强操作（如翻转、旋转、缩放等），来增加测试样本的多样性，
# 从而提高模型的鲁棒性和泛化能力，使得模型在测试阶段能够更准确地评估和预测结果。
tta_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6, # 如果有 6 个 dimension，{X, Y, Z, R, G, B, Depth}
        use_dim=[0, 1, 2, 3, 4, 5], # 只使用 0 ~ 5 维度，{X, Y, Z, R, G, B}
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=False,
        with_seg_3d=True,
        backend_args=backend_args),
    dict(type='NormalizePointsColor', color_mean=None),
    dict(
        type='TestTimeAug',
        transforms=[[
            dict(
                type='RandomFlip3D',
                sync_2d=False,
                flip_ratio_bev_horizontal=0.,
                flip_ratio_bev_vertical=0.)
        ], [dict(type='Pack3DDetInputs', keys=['points'])]])
]

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='scannet_infos_train.pkl',
        metainfo=metainfo,
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        modality=input_modality,
        ignore_index=len(class_names),
        scene_idxs=data_root + 'seg_info/train_resampled_scene_idxs.npy',
        test_mode=False,
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='scannet_infos_val.pkl',
        metainfo=metainfo,
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        modality=input_modality,
        ignore_index=len(class_names),
        test_mode=True,
        backend_args=backend_args,
        # xuzhong add：开发用 mmdet3d.apis 加载模型时，缺这个 box_type_3d 参数
        # xuzhong: 增加之后，运行 demo/pcd_seg_demo.py 会报错
        # dataset=dict(
        #     box_type_3d='LiDAR'
        # )
    )
)
val_dataloader = test_dataloader

val_evaluator = dict(type='SegMetric')
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]

# 可视化参数
visualizer = dict(
    type='Det3DLocalVisualizer',    # 指定可视化器的类型，一般为Visualizer
    vis_backends=vis_backends,      # 定义了可视化结果的存储后端，可以同时指定多个后端
    name='visualizer'               # 为可视化器指定一个名称，方便在多个可视化器实例存在时进行区分
)

tta_model = dict(type='Seg3DTTAModel')
