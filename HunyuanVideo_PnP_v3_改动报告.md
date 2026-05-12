# PnP_v3 相对于原版 HunyuanVideo 的优化与调整报告

## 1. 报告目的

本文档用于总结当前 `PnP_v3` 代码相对于原版 `HunyuanVideo` 的主要优化与调整，重点说明：

1. 做了哪些真正已经接入并生效的功能增强。
2. 做了哪些偏工程化、实验化的脚本与配置调整。
3. 哪些改动目前更像预留接口，而不是已经落地的核心算法优化。

## 2. 对比基线

本次对比的原版代码目录为：

- `../HunyuanVideo`

该目录的 Git 远程指向：

- `https://github.com/Tencent-Hunyuan/HunyuanVideo`

本地基线版本为：

- `e260ed4` `Merge pull request #301 from TianQi-777/patch-5`

因此，本报告中的“原版 HunyuanVideo”指的是当前本地同级目录下的 `../HunyuanVideo`，而不是泛指所有历史版本。

## 3. 总体结论

整体来看，`PnP_v3` 并没有重写 HunyuanVideo 的底层主干网络，而是在原版基础上做了两类增强：

1. **采样阶段能力增强**：新增了 `Glyph Guidance`，用于在推理时通过 OCR/文字感知损失对 latent 进行修正，从而改善视频中字符、招牌、文本内容的可读性和一致性。
2. **实验与评测工程化**：新增了面向 `T2V-CompBench` 的批量生成脚本，并补充了对应的命令行参数、默认路径、离线运行和多卡配置，使模型更适合大规模 benchmark 运行。

换句话说，当前版本的主要变化不在“模型骨干结构”，而在“推理控制能力”和“批量实验工作流”。

## 4. 核心优化一：新增 Glyph Guidance 文本引导采样

### 4.1 改动概述

当前版本新增了一个独立模块：

- `hyvideo/guidance/ocr_guidance.py`

它实现了一种 plug-and-play 的 OCR 引导采样方法。核心思想是：在每一步去噪过程中，根据模型当前预测得到的 `x0_hat` 解码出像素帧，再通过 OCR 相关损失计算梯度，对当前 latent 做一次修正，使采样轨迹更靠近“文字清晰、字符结构正确”的流形。

这部分改动属于**推理阶段增强**，不依赖重新训练主模型。

### 4.2 工作机制

`Glyph Guidance` 的核心公式可以概括为：

- 先由 flow-matching 预测得到当前步的近似干净样本 `x0_hat`
- 将 `x0_hat` 解码到像素空间
- 计算图像与目标文本之间的 OCR/字符一致性损失
- 对 latent 求梯度并执行一次修正更新

这意味着当前版本在原版的常规 denoising loop 基础上，多了一条“文字感知的纠偏路径”，适合处理如下场景：

- 画面中出现招牌、海报、字幕、印刷文字
- prompt 中明确要求出现某些文本内容
- 需要提升字符清晰度和可辨识性

### 4.3 支持的 3 种引导后端

当前实现支持三种 backend：

1. `clip`
   - 使用 CLIP 的图文相似度作为引导损失
   - 优点是实现通用、依赖常见、推理速度相对较快

2. `trocr`
   - 使用 TrOCR 编码器特征进行图像与参考文字图之间的表征对齐
   - 优点是字符识别能力更强，适合更关注文字准确性的任务

3. `render`
   - 将目标文字渲染成参考图，直接与预测图像做像素级损失
   - 优点是不依赖额外大型识别模型，部署最简单

这种设计比原版更灵活，允许根据显存、速度和文字效果要求选择不同的引导方式。

### 4.4 已接入推理主流程

这项优化不是停留在单独文件里，而是已经被接入到主采样流程：

- `hyvideo/inference.py`
- `hyvideo/diffusion/pipelines/pipeline_hunyuan_video.py`

接入方式主要体现在以下几个层面：

1. 在 sampler 初始化时，根据命令行参数决定是否构建 `glyph_guidance` 实例。
2. 在推理前，从显式参数或 prompt 中提取目标文本。
3. 在 denoising loop 中，于 `CFG` 之后、`scheduler.step()` 之前，对 latent 执行文字引导修正。

这个插入位置是合理的，因为它让文字约束作用在已完成 CFG 融合后的预测上，同时又能在 scheduler 更新前影响下一步采样轨迹。

### 4.5 额外的稳定性与效率优化

当前实现还做了几项比较实用的工程优化：

1. **只在特定 sigma 区间启用引导**
   - 通过 `glyph_sigma_min` 和 `glyph_sigma_max` 控制引导启用区间
   - 太早的高噪声阶段图像结构尚不稳定，太晚的低噪声阶段收益较小，因此仅在中间区间施加引导更高效

2. **支持缩小解码分辨率**
   - 通过 `glyph_decode_resize` 在 guidance 时降低 VAE decode 的空间尺寸
   - 可以明显减少显存占用和额外计算开销

3. **只选部分帧做引导**
   - 默认使用首帧、中间帧、末帧
   - 避免每一帧都计算 OCR 损失，兼顾视频全局覆盖与计算成本

4. **梯度裁剪**
   - 通过 `glyph_grad_clip` 限制 guidance 梯度范数
   - 可避免引导过强带来的数值不稳定或画面破坏

这些措施说明当前版本不是简单“加一个 OCR loss”，而是考虑了视频采样场景下的速度、显存和稳定性平衡。

## 5. 核心调整二：增加了完整的 Glyph Guidance 配置接口

为了让上述新能力可配置、可实验，`PnP_v3` 在 `hyvideo/config.py` 中加入了一整套命令行参数，包括：

- `--glyph-guidance`
- `--glyph-ocr-backend`
- `--glyph-eta`
- `--glyph-sigma-min`
- `--glyph-sigma-max`
- `--glyph-target-text`
- `--glyph-decode-resize`
- `--glyph-clip-model`
- `--glyph-trocr-model`
- `--glyph-grad-clip`

相比原版，这一调整带来了两个好处：

1. **推理控制粒度更细**
   - 可以精细控制引导强度、启用阶段、目标文本来源和后端模型

2. **更适合做消融实验**
   - 便于对不同 backend、不同 guidance 强度、不同 sigma 区间进行系统比较

此外，`sample_video.py` 也同步支持了 `glyph_target_text` 的透传，因此用户既可以依赖 prompt 自动提取文本，也可以显式指定想要在视频中出现的字符串。

## 6. 工程化调整一：新增面向 T2V-CompBench 的批量评测脚本

当前版本新增了：

- `run_hunyuan_t2v_compbench.sh`

这不是原版 HunyuanVideo 自带的标准脚本，而是一个明显面向 benchmark 评测场景的批量运行入口。它的主要作用是把 HunyuanVideo 的单条 prompt 推理流程，改造成适合 `T2V-CompBench` 的批处理任务。

### 6.1 批量运行能力

该脚本支持按 benchmark 类别读取 prompt 文件并逐条生成视频，覆盖如下类别：

- `consistent_attr`
- `dynamic_attr`
- `spatial_relationship`
- `motion_binding`
- `action_binding`
- `interaction`
- `numeracy`

这使得当前代码能直接对接 `T2V-CompBench` 数据组织形式，而原版更偏向单次样例生成。

### 6.2 多卡与并行配置整合

脚本中整合了：

- `torchrun`
- `NUM_GPUS`
- `ULYSSES_DEGREE`
- `RING_DEGREE`
- `MASTER_PORT`

并显式校验：

- `NUM_GPUS = ULYSSES_DEGREE x RING_DEGREE`

这说明当前版本在实验运行层面，强调与原版已有并行能力配合使用，从而更适合长序列视频的大规模生成。

### 6.3 更适合大规模实验的细节处理

脚本还加入了很多对 benchmark 很有帮助的细节：

1. **跳过已生成结果**
   - 如果目标 `mp4` 已存在则自动跳过
   - 方便断点续跑

2. **临时目录生成与自动清理**
   - 每次生成先输出到临时目录，再统一移动到最终命名位置
   - 避免中间结果污染最终目录

3. **测试模式**
   - `TEST_NUM` 可限制每个类别只跑前若干条 prompt
   - 便于小规模 smoke test

4. **统一输出目录结构**
   - 自动按 benchmark 类别创建输出目录
   - 便于后续评估脚本读取

5. **默认离线运行**
   - `HF_HUB_OFFLINE=1`
   - `TRANSFORMERS_OFFLINE=1`
   - `DIFFUSERS_OFFLINE=1`
   - 更适合集群环境和已缓存模型的批处理任务

总体上，这部分调整提升的不是单次生成质量，而是实验复现性、批处理效率和 benchmark 集成度。

## 7. 工程化调整二：默认路径与本地目录结构适配

在 `hyvideo/config.py` 中，`--model-base` 的默认值从原版的：

- `ckpts`

改成了：

- `../HunyuanVideo/ckpts`

这说明当前项目目录结构并不是“所有内容都在一个独立仓库里”，而是默认依赖旁边已有的 `HunyuanVideo/ckpts`。这一调整的作用主要是：

1. 复用原版已下载的 checkpoint
2. 避免在 `PnP_v3` 中重复维护模型权重目录
3. 适配当前工作目录下多个相关项目并列放置的实验环境

因此，这更像是一种**本地开发与实验目录组织优化**。

## 8. 需要澄清的一点：ST Attention Cohesion 目前更像预留接口

在 `hyvideo/config.py` 和 `run_hunyuan_t2v_compbench.sh` 中，当前版本增加了一组新的参数：

- `--enable-st-attn-cohesion`
- `--st-attn-spatial-topk-ratio`
- `--st-attn-temporal-strength`
- `--st-attn-background-suppress`
- `--st-attn-score-momentum`
- `--st-attn-localize-chunk-size`

从命名上看，这组参数显然是为“时空注意力一致性增强”准备的，目标可能是：

- 提升时序一致性
- 更稳定地聚焦主体区域
- 减少背景噪声干扰

但是，经过当前代码检查，这组参数**目前只出现在配置层和脚本透传层**，尚未看到它们真正接入：

- attention 核心实现
- transformer forward
- denoising pipeline 的实际计算逻辑

因此，比较严谨的结论是：

- **ST Attention Cohesion 目前属于预留实验接口或待接入功能**
- **不能把它当作当前版本已经生效的核心优化来描述**

如果后续有新的 attention 分支或 patch 接入，这一项才会成为真实的模型级优化。

## 9. 哪些部分基本没有变化

从与原版目录的比对结果来看，以下部分没有体现出当前版本的核心新增优化：

1. **底层主干网络结构**
   - 当前没有看到对 Transformer backbone 的系统性重写

2. **attention 核心实现**
   - 当前没有看到相对原版的实质性逻辑改造

3. **VAE 主体实现**
   - 未发现明显结构层面的新修改

4. **FP8 与并行能力本身**
   - `use-fp8`、`Ulysses`、`Ring`、`xFuser` 等能力在原版中已存在
   - 当前版本主要是把它们整合进 benchmark 运行脚本，而不是首次引入

因此，如果要概括这次 fork 的定位，更准确的说法不是“重做了 HunyuanVideo”，而是：

- **在原版 HunyuanVideo 上增加了文本感知的推理控制能力，并增强了 benchmark 运行工作流。**

## 10. 总结

相对于原版 `HunyuanVideo`，当前 `PnP_v3` 的主要优化和调整可以概括为以下几点：

1. **新增 Glyph Guidance**
   - 在采样阶段引入 OCR/文字一致性引导
   - 目标是提升视频中字符内容的清晰度和可读性

2. **将 Glyph Guidance 真正接入主推理流程**
   - 包括参数解析、实例构建、文本提取、denoising loop 中的 latent 修正

3. **加入多种文字引导后端**
   - `clip`、`trocr`、`render`
   - 增强灵活性，支持不同速度与效果折中

4. **做了 guidance 侧的效率与稳定性优化**
   - sigma 区间控制
   - 降分辨率 decode
   - 代表帧抽样
   - 梯度裁剪

5. **新增 T2V-CompBench 批量生成脚本**
   - 支持多类别 prompt 批处理
   - 支持断点续跑、临时目录、测试模式、统一输出结构

6. **增强了本地实验环境适配**
   - 默认 checkpoint 路径、离线模式、多卡并行参数更适合集群实验

7. **增加了 ST Attention Cohesion 参数接口，但尚未真正落地**
   - 目前更适合描述为“预留实验项”，不应算作已完成的模型优化

## 11. 一句话结论

当前 `PnP_v3` 相对于原版 `HunyuanVideo`，最主要的实质性改动是：**在不改动底层主干模型的前提下，引入了面向文字生成质量的 Glyph Guidance，并将代码工程化为更适合 T2V-CompBench 批量评测的实验版本。**
