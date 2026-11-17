<!--
  审稿期版本 README。提交前请搜索 "TODO" 并逐一替换/删除。
  HTML 注释（像这一段）在 GitHub 预览里不会显示。
-->

<div align="center">

# PD-Adapter

### Physics-Distilled Learning of Command-Space Initialization for Underactuated Grasp Adaptation

**Under review at IEEE Robotics and Automation Letters (RA-L)**

[Paper (coming soon)]() · [Video](#-video) · [Code Release Plan](#-code-release-plan)

</div>

> **Note for reviewers**
> This repository accompanies our RA-L submission. During the review period it contains the core components needed to understand and check the method (see [What is included now](#-what-is-included-now)). The full training and evaluation pipeline will be released upon acceptance.

<p align="center">
  <img src="assets/fig1_underactuation_gap.jpg" width="85%">
</p>
<p align="center"><em>
  TODO: one-sentence caption of Fig. 1, e.g. what the "underactuation gap" between the initial grasp and the executable hand command is.
</em></p>

---

## 📌 Overview

<!-- 3–5 句话，和论文摘要保持一致，不要写新的结论。 -->

Given an **initial grasp** produced by an upstream grasp generator, an underactuated hand (with coupled joints) usually cannot reproduce it exactly. **PD-Adapter** performs **grasp adaptation** in the hand's **command space**: it learns, from a physics-based search teacher, to predict good command-space initializations, which are then optimized and checked by **simulation verification** before execution.

TODO: 1–2 sentences on the key results (copy the numbers exactly as in the paper).

Hands evaluated: **Inspire**, **Ability**, **SVH** (simulation) and TODO (real robot).

## 🎬 Video

<!--
  在 GitHub 网页编辑 README 时，直接把 mp4 拖进编辑框，会自动生成一个 user-attachments 链接，
  预览里可直接播放（单个文件 ≤ 10 MB 免费账户限制，超了就先压缩或剪短）。
-->

TODO: drag the supplementary video (.mp4) here.

## 📂 Repository Structure

```
PD-Adapter/
├── src/                          # TODO: PD-Adapter model / inference code
├── eval/                         # TODO: evaluation scripts (simulation verification, metrics)
├── GEN/                          # TODO: initial grasps from GenDexGrasp (baseline / input)
├── DRO/                          # TODO: initial grasps from D(R,O) Grasp (baseline / input)
├── assets/                       # Hand models, figures, videos
├── inspire_coupling.py           # Joint-coupling model of the Inspire hand
└── command_target_projection.py  # Bounded least-squares projection from joint targets to hand commands (Naive-Φ)
```

## ✅ What Is Included Now

| Component | Status |
|---|---|
| Inspire hand coupling model (`inspire_coupling.py`) | ✅ Released |
| Command-target projection / Naive-Φ (`command_target_projection.py`) | ✅ Released |
| PD-Adapter network & inference | TODO ✅ / ⏳ |
| Simulation verification & evaluation | TODO ✅ / ⏳ |
| Pretrained checkpoints | ⏳ Upon acceptance |
| Training code & dataset generation (physics teacher) | ⏳ Upon acceptance |
| Real-robot deployment code | ⏳ Upon acceptance |

## ⚙️ Installation

Tested on Ubuntu TODO, Python TODO, PyTorch TODO, CUDA TODO (NVIDIA RTX A5000).

```bash
git clone https://github.com/IRM-Lab-robot/PD-Adapter.git
cd PD-Adapter
conda create -n pd-adapter python=TODO -y
conda activate pd-adapter
pip install -r requirements.txt   # TODO: add requirements.txt
```

## 🚀 Quick Start

<!-- 只放审稿人能在几分钟内跑通的命令。每条命令都要自己在干净环境里跑一遍再写上来。 -->

**1. Inspire hand coupling**

```bash
python inspire_coupling.py   # TODO: arguments / expected output
```

**2. Projecting a joint-space target to the command space**

```bash
python command_target_projection.py   # TODO: arguments / expected output
```

**3. (Optional) Run evaluation on the provided initial grasps**

```bash
python eval/TODO.py --hand inspire --input GEN/TODO
```

## 📊 Results

<!-- 直接放论文 Table I 的截图或 markdown 表格，数字与论文完全一致。 -->

TODO: main benchmark table (Table I) and/or real-robot result figure.

## 🗓️ Code Release Plan

- [x] Core coupling model and command-space projection (review period)
- [ ] Pretrained PD-Adapter checkpoints
- [ ] Training code and physics-teacher data generation
- [ ] Full simulation evaluation pipeline
- [ ] Real-robot deployment code

## 📝 Citation

The paper is currently under review. Citation information will be added upon publication.

## 🙏 Acknowledgements

We build on the following open-source projects: [GenDexGrasp](TODO-link), [D(R,O) Grasp](TODO-link), TODO (simulator, hand models). We thank the authors for releasing their code.

## 📄 License

TODO: e.g. "All rights reserved during the review period. A license will be added upon publication." — or MIT / Apache-2.0 if your lab agrees to release now.

## 📧 Contact

TODO: name, email / open an issue.
