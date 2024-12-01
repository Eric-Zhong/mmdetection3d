'''
在 OpenMMLab 中，PointSegClassMapping是与点云语义分割任务相关的一个模块或类，主要用于处理点云数据集中的类别映射关系。
以下是关于它的一些相关介绍：

作用与意义：
在点云语义分割中，数据集中的每个点都需要被分配一个语义类别标签。
PointSegClassMapping的作用就是帮助建立原始数据集中的类别标签与模型训练、评估等过程中所使用的类别标签之间的映射关系。
这对于正确处理和理解数据集中的类别信息非常重要，确保模型能够准确地学习和预测每个点的类别。

具体功能：
 * 类别标签转换：将数据集中原始的类别标签转换为模型训练所需要的格式。
   例如，原始数据集中可能使用特定的数字或字符串来表示不同的类别，
   而 PointSegClassMapping可以将这些标签转换为模型能够理解的整数编码，方便模型进行训练和预测。
 * 数据集适配：不同的点云语义分割数据集可能有不同的类别定义和标签表示。
   PointSegClassMapping可以根据具体的数据集特点，对类别标签进行适配和转换，
   使得不同的数据集能够在同一个模型框架下进行处理。
 * 支持多类别任务：对于包含多个类别的点云语义分割任务，它可以有效地管理和处理各种类别的映射关系，
   确保模型能够准确地对每个类别进行学习和预测。
使用场景：
 * 模型训练：在训练点云语义分割模型时，需要将数据集中的点及其对应的类别标签输入到模型中。
   PointSegClassMapping可以在数据预处理阶段对类别标签进行转换和映射，使得模型能够正确地接收和处理这些数据。
 * 模型评估：在评估模型的性能时，需要将模型预测的类别标签与真实的类别标签进行比较。
   PointSegClassMapping可以确保预测的标签和真实的标签在同一类别体系下，以便进行准确的评估。
 * 跨数据集实验：当需要在不同的点云语义分割数据集上进行实验时，由于不同数据集的类别定义可能不同，
   PointSegClassMapping可以帮助实现类别标签的统一和转换，使得实验结果具有可比性。
'''

# 训练流程
# S3DIS 上 3D 语义分割的一种典型数据载入流程如下所示：

# For S3DIS seg we usually do 13-class segmentation
class_names = ('ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door',
               'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter')
metainfo = dict(classes=class_names)
dataset_type = 'S3DISSegDataset'
data_root = 'data/s3dis/'
input_modality = dict(use_lidar=True, use_camera=False)
data_prefix = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask')

# Example to use different file client
# Method 1: simply set the data root and let the file I/O module
# automatically infer from prefix (not support LMDB and Memcache yet)

# data_root = 's3://openmmlab/datasets/detection3d/s3dis/'

# Method 2: Use backend_args, file_client_args in versions before 1.1.0
# backend_args = dict(
#     backend='petrel',
#     path_mapping=dict({
#         './data/': 's3://openmmlab/datasets/detection3d/',
#          'data/': 's3://openmmlab/datasets/detection3d/'
#      }))
backend_args = None

# dataset
'''
Area_1
Area_2
Area_3
Area_4
Area_5
Area_6
'''

num_points = 4096

# 划分训练集和测试集的文件索引
train_area = [1, 2, 3, 4, 6]
test_area = 5

# 训练管道
train_pipeline = [
    # step 1: 加载点云数据
    dict(
        type='LoadPointsFromFile',  # 表示这个工作节点的类型是从文件中加载点云数据。
        coord_type='DEPTH',         # 确定点云的坐标类型。
        shift_height=False,         # 如果为False，表示不进行高度偏移操作。
        use_color=True,             # 如果为True，表示在加载点云数据时会使用颜色信息。
        load_dim=6,                 # 设置为 6，可能表示加载点云数据时包括六个维度的信息
        use_dim=[0, 1, 2, 3, 4, 5], # 列出了要使用的维度索引。这里表示使用从 0 到 5 这六个维度的数据。
        backend_args=backend_args), # 通常用于指定后端的一些参数设置。
    # step 2: 加载标注
    dict(
        type='LoadAnnotations3D',   # 表示这个节点的类型是从三维数据中加载标注信息。
        with_bbox_3d=False,         # 如果为False，表示在加载标注信息时不加载三维边界框（3D bounding box）信息。
        with_label_3d=False,        # 如果为False，表示在加载标注信息时不加载三维标签信息。
        with_mask_3d=False,         # 如果为False，表示在加载标注信息时不加载三维掩码（3D mask）信息。
        with_seg_3d=True,           # 如果为True，表示在加载标注信息时加载三维分割（3D segmentation）信息。
        backend_args=backend_args), # 通常是用于指定后端处理的参数。
    # step 3: PointSegClassMapping
    #         在训练过程中，只有被使用的类别的序号会被映射到类似 [0, 13) 范围内的类别标签。
    #         其余的类别序号会被转换为 ignore_index 所制定的忽略标签，在本例中是 13。
    dict(type='PointSegClassMapping'),
    # step 4: IndoorPatchPointSample
    #         从输入点云中裁剪一个含有固定数量点的小块 (patch)。block_size 指定了裁剪块的边长，
    #         在 S3DIS 上这个数值一般设置为 1.0。
    dict(
        type='IndoorPatchPointSample',  # 表示这个工作节点的类型是进行室内点云的补丁点采样操作。
        num_points=num_points,          # 确定采样得到的点的数量。
        block_size=1.0,                 # 可能表示采样的块大小或空间尺度参数。
        ignore_index=len(class_names),  # 设定一个被忽略的索引值。
        use_normalized_coord=True,      # 如果为True，表示使用归一化的坐标。归一化坐标可以使不同尺度的点云数据在处理过程中更加方便和稳定。
        enlarge_size=0.2,               # 可能表示扩大采样区域的比例或大小。
        min_unique_num=None             # 可能表示最小独特点数量的阈值。
    ),
    # step 5: NormalizePointsColor
    dict(
        type='NormalizePointsColor',    # 将输入点的颜色信息归一化，通过将 RGB 值除以 255 来实现。
        color_mean=None                 # 如果设置为None，可能表示在归一化过程中不使用预先给定的固定颜色均值，
                                        # 而是根据输入的点云数据自动计算颜色均值进行归一化处理。
                                        # 具体的归一化方法可能是将每个点的颜色值减去计算得到的颜色均值，
                                        # 然后再进行进一步的缩放等操作，以使得颜色值分布在一个特定的范围内。
    ),
    # step 1: LoadPointsFromFile
    dict(
        type='Pack3DDetInputs',         # 表示这个工作节点的类型是 “打包 3D 检测输入”。
                                        # 它的作用可能是将特定的数据进行整理和组合，以便作为输入提供给后续的 3D 检测模块。
        keys=['points', 'pts_semantic_mask'] # 指定了要被打包的数据的键名
    )
]

# 测试管道
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

# 评估管道
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

tta_pipeline = [
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
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(
                    type='RandomFlip3D',
                    sync_2d=False,
                    flip_ratio_bev_horizontal=0.,
                    flip_ratio_bev_vertical=0.)
            ], [
                dict(type='Pack3DDetInputs', keys=['points'])
            ]
        ]
    )
]

# 训练集
# train on area 1, 2, 3, 4, 6
# test on area 5
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_files=[f's3dis_infos_Area_{i}.pkl' for i in train_area],
        metainfo=metainfo,
        data_prefix=data_prefix,
        pipeline=train_pipeline,
        modality=input_modality,
        ignore_index=len(class_names),
        scene_idxs=[
            f'seg_info/Area_{i}_resampled_scene_idxs.npy' for i in train_area
        ],
        test_mode=False,
        backend_args=backend_args
    )
)

# 测试集
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_files=f's3dis_infos_Area_{test_area}.pkl',
        metainfo=metainfo,
        data_prefix=data_prefix,
        pipeline=test_pipeline,
        modality=input_modality,
        ignore_index=len(class_names),
        scene_idxs=f'seg_info/Area_{test_area}_resampled_scene_idxs.npy',
        test_mode=True,
        backend_args=backend_args
    )
)

val_dataloader = test_dataloader

val_evaluator = dict(type='SegMetric')
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

tta_model = dict(type='Seg3DTTAModel')
