# Underwater Fish Detection

面向复杂水下成像条件的轻量化鱼类目标检测研究项目。

本项目基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 进行二次开发，重点研究水下图像低对比度、边缘模糊、背景干扰和目标尺度变化条件下的目标检测问题。项目实现了 FGFC 频域全局特征校准模块，并进一步完成了受 SFS-Conv 启发的轻量化空间-频域选择模块 SFSConv。

> 本项目是一个可复现的研究原型。实验结果主要用于比较不同模块在统一短程训练协议下的相对增益，不代表完全收敛后的性能上限。

## 1. 项目概述

### 研究问题

水下图像通常受到光谱吸收、悬浮颗粒散射、光照不均和运动模糊影响，容易出现低对比度、鱼体边缘不清晰、密集目标漏检和复杂背景误检等问题。轻量化检测模型在严格 IoU 条件下还可能存在定位精度不足。

### 研究思路

```text
水下图像
    -> CLAHE 局部对比度优化
    -> YOLOv8n 特征提取
    -> Residual CBAM 特征筛选
    -> FGFC 或 SFSConv 特征校准
    -> PAN-FPN 多尺度融合
    -> 鱼类检测框
```

## 2. 主要模块

### FGFC：Frequency-Guided Feature Calibration

核心代码位于：

```text
ultralytics/nn/modules/block.py
```

`FGFC` 使用 Haar 小波将特征分解为 `LL`、`LH`、`HL`、`HH` 四个子带。低频分支通过轻量偏移场和 `grid_sample` 进行结构校准；高频分支通过深度卷积和门控机制处理鱼体轮廓、鱼鳍和纹理细节；最后使用逆 Haar 变换恢复特征尺寸。

FGFC 在模型配置中的插入位置为 YOLOv8n Backbone 的 P3、P4、P5 特征层：

```text
ultralytics/cfg/models/v8/yolov8-dfcn.yaml
```

### Residual CBAM

核心代码位于：

```text
ultralytics/nn/modules/conv.py
```

该模块在 CBAM 的通道注意力和空间注意力基础上加入残差混合系数 `gamma`，并采用零初始化，使模块刚接入预训练模型时接近恒等映射，降低破坏原始特征分布的风险。

### SFSConv：SFS-Conv 启发式空间-频域选择模块

核心代码同样位于 `ultralytics/nn/modules/block.py`。该模块受到 SFS-Conv 的 `shunt-perceive-select` 思路启发，但不是对原论文 Fractional Gabor Transformer 的完整复现。

SFSConv 的处理流程为：

1. 按通道拆分为空间分支和频率分支；
2. 空间分支使用 3x3 和 5x5 深度卷积提取多尺度上下文；
3. 频率分支使用高通残差和水平/垂直方向滤波提取边缘响应；
4. 使用拼接、1x1 卷积和近恒等残差完成空间-频域融合。

对应配置文件：

```text
ultralytics/cfg/models/v8/yolov8-sfsneck.yaml
ultralytics/cfg/models/v8/yolov8-dfcn-sfs.yaml
```

## 3. 实验数据

项目使用公开 Aquarium-qlnqy 数据集中的 `fish` 类，并转换为 YOLO 单类别格式。

| 数据划分 | 图像数量 | fish 标注框 | 空标签图像 |
|---|---:|---:|---:|
| Train | 448 | 1965 | 207 |
| Validation | 127 | 459 | 64 |
| Test | 63 | 249 | 33 |
| 合计 | 638 | 2673 | 304 |

原始数据集没有随代码仓库上传。数据处理脚本默认原始 COCO 数据位于 `datasets/aquarium-qlnqy/raw/`。

```text
tools/prepare_aquarium_fish.py
tools/prepare_clahe_dataset.py
tools/prepare_underwater_robustness_set.py
```

退化鲁棒性测试集由官方测试图像派生生成，并非独立采集的真实水下测试集。退化过程包括 3x3 高斯模糊、通道衰减和固定雾化混合，用于在相同条件下比较模型鲁棒性。

## 4. 实验结果

所有对照实验使用相同的随机种子、AdamW、输入尺寸 320x320、batch size 4 和 3 个 epoch。3 个 epoch 用于控制训练预算并比较模块的相对增益，不代表模型已经完全收敛。

### FGFC 主实验

| 模型 | 参数量 | 清晰集 mAP50 | 清晰集 mAP50-95 | 退化集 mAP50 | 退化集 mAP50-95 |
|---|---:|---:|---:|---:|---:|
| YOLOv8n | 3.011M | 0.2277 | 0.0917 | 0.2324 | 0.0995 |
| DFCN-YOLO + FGFC | 3.378M | 0.2325 | **0.1069** | 0.2353 | **0.1121** |

### SFSConv 迁移实验

| 模型 | 参数量 | 清晰集 mAP50 | 清晰集 mAP50-95 | 退化集 mAP50 | 退化集 mAP50-95 |
|---|---:|---:|---:|---:|---:|
| YOLOv8n | 3.011M | 0.2277 | 0.0917 | 0.2324 | 0.0995 |
| SFS-Neck-P3 | 3.021M | **0.2581** | **0.1089** | **0.2824** | **0.1226** |
| DFCN-YOLO + FGFC | 3.378M | 0.2325 | 0.1069 | 0.2353 | 0.1121 |
| DFCN-SFS | 3.284M | 0.2247 | 0.0926 | 0.2375 | 0.1009 |

SFS-Neck-P3 在融合后的 Neck P3 特征上插入单个 SFSConv，取得了较好的相对提升；将 FGFC 在 P3/P4/P5 三个尺度全部替换为 SFSConv 的 DFCN-SFS 没有超过 FGFC。这说明模块的插入位置和频域归纳偏置都很重要，不能简单认为一种频域模块在所有位置都优于另一种模块。

## 5. 如何运行

建议使用 Python 3.8 或更高版本，并根据设备安装对应版本的 PyTorch。

```powershell
pip install -e .
```

完成原始数据下载并放置到 `datasets/aquarium-qlnqy/raw/` 后运行：

```powershell
python tools/prepare_aquarium_fish.py
python tools/prepare_clahe_dataset.py
python tools/prepare_underwater_robustness_set.py
```

运行 FGFC 消融实验：

```powershell
python tools/run_multilevel_ablation.py --epochs 3 --imgsz 320 --batch 4 --tag reproduce
```

运行 SFSConv 启发式实验：

```powershell
python tools/run_sfs_inspired_experiments.py --epochs 3 --imgsz 320 --batch 4 --tag reproduce
```

只运行某一个 SFS 实验：

```powershell
python tools/run_sfs_inspired_experiments.py --only sfs_neck_p3
python tools/run_sfs_inspired_experiments.py --only dfcn_sfs
```

训练结果默认保存到 `runs/underwater-fish-complete/` 和 `runs/sfs-inspired-underwater-fish/`。

## 6. 项目结构

```text
ultralytics-main/
├── ultralytics/                 # YOLO 核心代码及自定义模块
│   ├── nn/modules/block.py       # FGFC、SFSConv
│   ├── nn/modules/conv.py        # ResidualCBAM
│   ├── nn/modules/__init__.py    # 模块导出
│   ├── nn/tasks.py               # YAML 模型解析注册
│   └── cfg/                      # 模型和数据集配置
├── tools/                       # 数据处理和训练脚本
├── reports/                     # 实验报告和效果图
├── datasets/                    # 本地数据
└── runs/                        # 本地训练结果
```

## 7. 个人完成内容

- 基于 Ultralytics 完成模型配置、模块注册和检测模型二次开发；
- 独立设计并实现 FGFC 的 Haar 分解、低频偏移校准、高频门控和逆变换；
- 实现零初始化 Residual CBAM；
- 根据 SFS-Conv 的空间-频域选择思想实现轻量 SFSConv 近似模块；
- 完成 COCO 到 YOLO 的单类别标注转换、CLAHE 数据生成和退化鲁棒性测试集构造；
- 设计 FGFC 与 SFSConv 的消融实验并完成训练、验证、推理和可视化；
- 完成实验数据整理、结果分析和科研报告撰写。

## 8. 研究边界

- 当前训练协议为统一的 3 epoch 短程微调，未用于证明完全收敛性能；
- 当前主要验证 Aquarium 单类别数据，尚未覆盖更多真实水域和鱼种；
- 退化测试集由已有测试图像派生，不能完全替代真实采集的退化水下视频；
- 当前任务是单帧目标检测，视频场景中的跨帧跟踪需要进一步接入 ByteTrack 或 BoT-SORT；
- 目前尚未完成完整的视频端到端 FPS 和嵌入式设备部署测试。

## 9. 参考资料与许可证

- Ultralytics YOLO: https://github.com/ultralytics/ultralytics
- Ultralytics Documentation: https://docs.ultralytics.com/
- SFS-Conv paper: https://openaccess.thecvf.com/content/CVPR2024/html/Li_Unleashing_Channel_Potential_Space-Frequency_Selection_Convolution_for_SAR_Object_Detection_CVPR_2024_paper.html
- SFS-Conv official repository: https://github.com/like413/SFS-Conv
- Aquarium-qlnqy dataset: https://huggingface.co/datasets/Francesco/aquarium-qlnqy
