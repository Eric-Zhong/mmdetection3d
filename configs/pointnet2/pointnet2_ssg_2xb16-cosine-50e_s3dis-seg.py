_base_ = [
    # 数据集配置
    '../_base_/datasets/s3dis-seg.py', 
    # 模型配置
    '../_base_/models/pointnet2_ssg.py',
    # 优化器配置
    '../_base_/schedules/seg-cosine-50e.py', 
    # 默认运行时配置
    '../_base_/default_runtime.py'
]

# 模型运行参数配置
# model settings
model = dict(
    backbone = dict(in_channels=9),  # [xyz, rgb, normalized_xyz]
    decode_head = dict(
        num_classes = 13, 
        ignore_index = 13,
        loss_decode = dict(
            class_weight = None
        )
    ),  # S3DIS doesn't use class_weight
    test_cfg=dict(
        num_points=4096,
        block_size=1.0,
        sample_rate=0.5,
        use_normalized_coord=True,
        batch_size=24
    )
)

# 训练数据集加载配置
# data settings
train_dataloader = dict(batch_size=16)

# runtime settings
default_hooks = dict(
    checkpoint = dict(
        type='CheckpointHook', 
        interval=2
        )
    )

# 训练参数配置
train_cfg = dict(
    val_interval=2
)
