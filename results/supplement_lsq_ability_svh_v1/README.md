# Ability/SVH 四线：全部环节一致的 LSQ 目标投影实验

本目录独立于历史 `none` 搜索/教师/检查点。两手各独立重训一个 K=8 模型，DRO+GenDex 教师合并训练；物体划分与 Inspire 新六向实验相同：预固定 **train30 / val10 / test10**，教师源每个 train40 物体每生成器各 20 条。旧 teacher 标签不复用。教师生成、测试搜索期的 GPU 打分与 CPU-final、最终复验均指定 `validator_projection=lsq`；同一手使用一个模型适配两种源。

测试源来自 `new/{DRO,GEN}/{ability,svh}.pt`，四线每线10物体×64源目标。CEM-288 使用72×4、Sample-288使用288×1、PD-9使用默认+8学习分支各32，均288次GPU采样/源。所有搜索按物体×8源分块保存，输入/检查点/搜索配置和GPU+CPU-final模式逐层校验。测试最终验证为一次固定 batch32/no-shuffle、尾批填充（只计真实候选），每条存储的GPU初筛并已进入CPU-final的候选各复验一次；由同一组标签计算预先按GPU分数选的top-1和任一候选通过的多候选恢复率。短进程每次最多四批、每批独立落盘，失败后重跑命令续跑。

在 `/home/user/yangbo_program/new` 的 `three_papers` 环境，按步骤顺序运行：

```bash
python results/supplement_lsq_ability_svh_v1/run.py check

# 可选：仅验证两源教师脚本可运行；不写入正式教师目录
python results/supplement_lsq_ability_svh_v1/run.py teacher --hand ability --gen dro --object contactdb+apple --smoke

# 160份独立的正式教师搜索：2手×2生成器×40物体
python results/supplement_lsq_ability_svh_v1/run.py teacher

# 教师全部完成后，可先2epoch冒烟；随后两手各训练200epoch，选val10最佳checkpoint
python results/supplement_lsq_ability_svh_v1/run.py train --hand ability --smoke
python results/supplement_lsq_ability_svh_v1/run.py train

python results/supplement_lsq_ability_svh_v1/run.py warm

# 可选单块测试；随后按手型和来源依次执行完整搜索
python results/supplement_lsq_ability_svh_v1/run.py search --hand ability --gen dro --method cem288 --object contactdb+banana --part 0
python results/supplement_lsq_ability_svh_v1/run.py search

# 建议逐线、逐方法分开运行，脚本会按batch续跑
python results/supplement_lsq_ability_svh_v1/run.py recheck
python results/supplement_lsq_ability_svh_v1/run.py summarize
```

参数 `--hand ability|svh`、`--gen dro|gendex`、`--method cem288|sample288|pd9` 可减少单次任务长度。教师正例只在各自物体级 train30 进入反向传播；val10 仅用于选择检查点；test10 不能进入模型选择。训练仅一个种子，不能解释为训练间方差已被控制。结果目录 `teacher/`、`train/`、`warmstarts/`、`search/`、`recheck_cache/`、`revalidation_batches/`、`revalidation_once/` 和 `summary/<hand>_<gen>.json`，**不覆盖**任何旧 `none` 物理标签。

**单方法提前汇总**：不必等三种方法都完成。例如SVH–DRO的CEM十个对象复验完成后，可执行 `python results/supplement_lsq_ability_svh_v1/run.py summarize --hand svh --gen dro --method cem288`，只要求这一方法十个物体完成，输出独立的 `summary/svh_dro_cem288.json`，不覆盖以后完整的 `summary/svh_dro.json`。不加 `--method` 时，自动汇总该手/来源下**已经完成全部十物体**的方法并保存 `summary/<hand>_<gen>.json`，同时记录未完成方法；PD和某基线均完成时才报告二者配对提升。指定单方法时不做其它方法的配对比较。

这里的 LSQ 约束仅作用于预张开和闭合的**控制目标**；仿真中的实际关节状态在接触过程中不被逐时间步强制耦合。288只匹配GPU搜索采样预算，实际搜索期 CPU-final 候选数和一次部署的墙钟仍需另报。
