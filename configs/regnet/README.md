# Designing Network Design Spaces

> [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678)

<!-- [BACKBONE] -->

## Abstract

In this work, we present a new network design paradigm. Our goal is to help advance the understanding of network design and discover design principles that generalize across settings. Instead of focusing on designing individual network instances, we design network design spaces that parametrize populations of networks. The overall process is analogous to classic manual design of networks, but elevated to the design space level. Using our methodology we explore the structure aspect of network design and arrive at a low-dimensional design space consisting of simple, regular networks that we call RegNet. The core insight of the RegNet parametrization is surprisingly simple: widths and depths of good networks can be explained by a quantized linear function. We analyze the RegNet design space and arrive at interesting findings that do not match the current practice of network design. The RegNet design space provides simple and fast networks that work well across a wide range of flop regimes. Under comparable training settings and flops, the RegNet models outperform the popular EfficientNet models while being up to 5x faster on GPUs.

在这项工作中，我们提出了一种新的网络设计范式。我们的目标是帮助增进对网络设计的理解，并发现适用于各种场景的设计原则。我们不是专注于设计单个网络实例，而是设计对网络群体进行参数化的网络设计空间。整个过程类似于经典的网络手动设计，但提升到了设计空间层面。使用我们的方法，我们探索了网络设计的结构方面，并得出了一个由简单、规则的网络组成的低维设计空间，我们称之为 RegNet。RegNet 参数化的核心见解惊人地简单：良好网络的宽度和深度可以由一个量化线性函数来解释。我们分析了 RegNet 设计空间，并得出了与当前网络设计实践不相符的有趣发现。RegNet 设计空间提供了简单且快速的网络，在广泛的浮点运算量范围内都表现良好。在可比较的训练设置和浮点运算量下，RegNet 模型的性能优于流行的 EfficientNet 模型，同时在 GPU 上的速度快达 5 倍。


<div align=center>
<img src="https://user-images.githubusercontent.com/79644370/144025148-b73002cb-3c82-42e4-8da4-65df97aead9c.png" width="800"/>
</div>

## Introduction

We implement RegNetX models in 3D detection systems and provide their first results with PointPillars on nuScenes and Lyft dataset.

我们在 3D 检测系统中实现了 RegNetX 模型，并在 nuScenes 和 Lyft 数据集上首次给出了与 PointPillars 结合的结果。

The pre-trained modles are converted from [model zoo of pycls](https://github.com/facebookresearch/pycls/blob/master/MODEL_ZOO.md) and maintained in [mmcv](https://github.com/open-mmlab/mmcv).

预训练模型是从 [model zoo of pycls](https://github.com/facebookresearch/pycls/blob/master/MODEL_ZOO.md) 的模型库转换而来，并在 [mmcv](https://github.com/open-mmlab/mmcv) 中进行维护。

## Usage

To use a regnet model, there are two steps to do:

1. Convert the model to ResNet-style supported by MMDetection
2. Modify backbone and neck in config accordingly

要使用 RegNet 模型，有两个步骤：

1. 将模型转换为 MMDetection 支持的 ResNet 风格。
2. 相应地在配置文件中修改骨干网络（backbone）和颈部网络（neck）。

### Convert model

We already prepare models of FLOPs from 800M to 12G in our model zoo.

For more general usage, we also provide script `regnet2mmdet.py` in the tools directory to convert the key of models pretrained by [pycls](https://github.com/facebookresearch/pycls/) to
ResNet-style checkpoints used in MMDetection.

我们在我们的模型库中已经准备了浮点运算量从 800M 到 12G 的模型。

为了更广泛的使用，我们还在工具目录中提供了脚本 `regnet2mmdet.py`，用于将由 [pycls](https://github.com/facebookresearch/pycls/) 预训练的模型的键转换为 MMDetection 中使用的 ResNet 风格的检查点。

```bash
python -u tools/model_converters/regnet2mmdet.py ${PRETRAIN_PATH} ${STORE_PATH}
```

This script convert model from `PRETRAIN_PATH` and store the converted model in `STORE_PATH`.

此脚本将从 `PRETRAIN_PATH` 转换模型，并将转换后的模型存储在 `STORE_PATH` 中。

### Modify config

The users can modify the config's `depth` of backbone and corresponding keys in `arch` according to the configs in the [pycls model zoo](https://github.com/facebookresearch/pycls/blob/master/MODEL_ZOO.md).
The parameter `in_channels` in FPN can be found in the Figure 15 & 16 of the paper (`wi` in the legend).
This directory already provides some configs with their performance, using RegNetX from 800MF to 12GF level.
For other pre-trained models or self-implemented regnet models, the users are responsible to check these parameters by themselves.

**Note**: Although Fig. 15 & 16 also provide `w0`, `wa`, `wm`, `group_w`, and `bot_mul` for `arch`, they are quantized thus inaccurate, using them sometimes produces different backbone that does not match the key in the pre-trained model.

### 修改配置：

用户可以根据 pycls 模型库中的配置来修改配置文件中骨干网络的 `depth`（深度）以及 `arch` 中的相应关键参数。FPN 中的参数 `in_channels` 可以在论文的图 15 和图 16 中找到（在图例中为 `wi`）。
此目录已经提供了一些带有性能指标的配置，使用从 800MF 到 12GF 级别的 RegNetX。对于其他预训练模型或自行实现的 RegNet 模型，用户有责任自行检查这些参数。
注意：尽管图 15 和图 16 也提供了 `arch` 中的 `w0`、`wa`、`wm`、`group_w` 和 `bot_mul`，但它们是经过量化的，因此不准确，使用它们有时会产生与预训练模型中的键不匹配的不同骨干网络。

## Results and models

### nuScenes

|                                        Backbone                                         | Lr schd | Mem (GB) | Inf time (fps) |  mAP  | NDS  |                                                                                                                                                                                                                       Download                                                                                                                                                                                                                       |
| :-------------------------------------------------------------------------------------: | :-----: | :------: | :------------: | :---: | :--: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|       [SECFPN](../pointpillars/pointpillars_hv_secfpn_sbn-all_8xb4-2x_nus-3d.py)        |   2x    |   16.4   |                | 35.17 | 49.7 |                     [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20200620_230725-0817d270.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_secfpn_sbn-all_4x8_2x_nus-3d_20200620_230725.log.json)                     |
| [RegNetX-400MF-SECFPN](./pointpillars_hv_regnet-400mf_secfpn_sbn-all_8xb4-2x_nus-3d.py) |   2x    |   16.4   |                | 41.2  | 55.2 | [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/regnet/hv_pointpillars_regnet-400mf_secfpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_regnet-400mf_secfpn_sbn-all_4x8_2x_nus-3d_20200620_230334-53044f32.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/regnet/hv_pointpillars_regnet-400mf_secfpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_regnet-400mf_secfpn_sbn-all_4x8_2x_nus-3d_20200620_230334.log.json) |
|          [FPN](../pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py)           |   2x    |   17.1   |                | 40.0  | 53.3 |                           [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20200620_230405-2fa62f3d.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20200620_230405.log.json)                           |
|    [RegNetX-400MF-FPN](./pointpillars_hv_regnet-400mf_fpn_sbn-all_8xb4-2x_nus-3d.py)    |   2x    |   17.3   |                | 44.8  | 56.4 |       [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/regnet/hv_pointpillars_regnet-400mf_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_regnet-400mf_fpn_sbn-all_4x8_2x_nus-3d_20200620_230239-c694dce7.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/regnet/hv_pointpillars_regnet-400mf_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_regnet-400mf_fpn_sbn-all_4x8_2x_nus-3d_20200620_230239.log.json)       |
|    [RegNetX-1.6gF-FPN](./pointpillars_hv_regnet-1.6gf_fpn_sbn-all_8xb4-2x_nus-3d.py)    |   2x    |   24.0   |                | 48.2  | 59.3 |       [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/regnet/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d_20200629_050311-dcd4e090.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/regnet/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d/hv_pointpillars_regnet-1.6gf_fpn_sbn-all_4x8_2x_nus-3d_20200629_050311.log.json)       |

### Lyft

|                                        Backbone                                         | Lr schd | Mem (GB) | Inf time (fps) | Private Score | Public Score |                                                                                                                                                                                                                         Download                                                                                                                                                                                                                         |
| :-------------------------------------------------------------------------------------: | :-----: | :------: | :------------: | :-----------: | :----------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|       [SECFPN](../pointpillars/pointpillars_hv_secfpn_sbn-all_8xb2-2x_lyft-3d.py)       |   2x    |   12.2   |                |     13.9      |     14.1     |                     [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d_20210517_204807-2518e3de.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_secfpn_sbn-all_2x8_2x_lyft-3d_20210517_204807.log.json)                     |
| [RegNetX-400MF-SECFPN](./hv_pointpillars_regnet-400mf_secfpn_sbn-all_4x8_2x_lyft-3d.py) |   2x    |   15.9   |                |     14.9      |     15.1     | [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/regnet/hv_pointpillars_regnet-400mf_secfpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_regnet-400mf_secfpn_sbn-all_2x8_2x_lyft-3d_20210524_092151-42513826.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/regnet/hv_pointpillars_regnet-400mf_secfpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_regnet-400mf_secfpn_sbn-all_2x8_2x_lyft-3d_20210524_092151.log.json) |
|          [FPN](../pointpillars/pointpillars_hv_fpn_sbn-all_8xb2-2x_lyft-3d.py)          |   2x    |   9.2    |                |     14.9      |     15.1     |                           [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d_20210517_202818-fc6904c3.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/pointpillars/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_fpn_sbn-all_2x8_2x_lyft-3d_20210517_202818.log.json)                           |
|    [RegNetX-400MF-FPN](./hv_pointpillars_regnet-400mf_fpn_sbn-all_4x8_2x_lyft-3d.py)    |   2x    |   13.0   |                |     16.0      |     16.1     |       [model](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/regnet/hv_pointpillars_regnet-400mf_fpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_regnet-400mf_fpn_sbn-all_2x8_2x_lyft-3d_20210521_115618-823dcf18.pth) \| [log](https://download.openmmlab.com/mmdetection3d/v0.1.0_models/regnet/hv_pointpillars_regnet-400mf_fpn_sbn-all_2x8_2x_lyft-3d/hv_pointpillars_regnet-400mf_fpn_sbn-all_2x8_2x_lyft-3d_20210521_115618.log.json)       |

## Citation

```latex
@article{radosavovic2020designing,
    title={Designing Network Design Spaces},
    author={Ilija Radosavovic and Raj Prateek Kosaraju and Ross Girshick and Kaiming He and Piotr Dollár},
    year={2020},
    eprint={2003.13678},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
