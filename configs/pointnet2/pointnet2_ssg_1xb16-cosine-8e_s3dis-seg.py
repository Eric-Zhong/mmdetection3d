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
model = dict(  
    backbone = dict(  
        in_channels = 9   # [xyz, rgb, normalized_xyz] 输入通道数  
    ),  
    decode_head = dict(  
        num_classes  = 13,  # 类别数  
        ignore_index = 13, # 忽略索引  
        loss_decode  = dict(  
            class_weight = None # 类别权重（S3DIS不使用）  
        )  
    ),  
    test_cfg = dict(  
        num_points  = 4096, # 测试时每个块中的点数  
        block_size  = 1.0,  # 块大小  
        sample_rate = 0.5,  # 采样率  
        use_normalized_coord = True, # 是否使用归一化坐标  
        batch_size  = 4      # 测试时的批次大小  
    )  
)  

# 训练数据集加载配置  
# data settings  
train_dataloader = dict(  
    # batch_size=16 # 原始批次大小  
    # batch_size = 8 # 我减少了一半，先看看效果 = 没啥效果（中文注释保留）  
    batch_size = 16 # 我翻倍，先看看效果（中文注释保留）  
)  

# 运行时设置  
default_hooks = dict(  
    checkpoint   = dict(  
        type     = 'CheckpointHook',  # 钩子类型  
        interval = 2                  # 间隔  
    )  
)  

# 训练参数配置  
train_cfg = dict(  
    by_epoch     = True,  # 是否按epoch进行训练  
    max_epochs   = 1,     # 最大epoch数 （自动练习训练时，设置成 1 次，加快处理速度）
    val_interval = 1      # 验证间隔  
)  

auto_scale_lr = dict(  
    enable          = False,  # 是否启用自动调整学习率  
    base_batch_size = 32      # 基础批次大小  
)