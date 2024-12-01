_base_ = [
    '../_base_/datasets/scannet-seg.py', 
    '../_base_/models/pointnet2_ssg.py',
    '../_base_/schedules/seg-cosine-200e.py', 
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    decode_head=dict(
        num_classes=20,
        ignore_index=20,
        # `class_weight` is generated in data pre-processing, saved in
        # `data/scannet/seg_info/train_label_weight.npy`
        # you can copy paste the values here, or input the file path as
        # `class_weight=data/scannet/seg_info/train_label_weight.npy`
        loss_decode=dict(class_weight=[
            2.389689, 2.7215734, 4.5944676, 4.8543367, 4.096086, 4.907941,
            4.690836, 4.512031, 4.623311, 4.9242644, 5.358117, 5.360071,
            5.019636, 4.967126, 5.3502126, 5.4023647, 5.4027233, 5.4169416,
            5.3954206, 4.6971426
        ])
    ),
    test_cfg=dict(
        num_points=8192,
        block_size=1.5,
        sample_rate=0.5,
        use_normalized_coord=False,
        batch_size=24
    )
)

# data settings
train_dataloader = dict(batch_size=16)

# runtime settings
default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=5))
train_cfg = dict(val_interval=5)

# 可视化参数
visualizer = dict(
    # 为可视化器指定一个名称，方便在多个可视化器实例存在时进行区分
    name='visualizer',
    # 指定可视化器的类型
    type='Det3DLocalVisualizer',
    # 定义了可视化结果的存储后端，可以同时指定多个后端
    vis_backends=[
        # 它主要通过接收由Visualizer传递过来的数据，
        # 将其转换为可存储的图像格式，
        # 然后保存到本地指定的目录中。
        dict(
            type='LocalVisBackend',
            save_dir='C:/Users/zhong/git/mmdetection3d/outputs/cicc'
        ),
    ]
)
