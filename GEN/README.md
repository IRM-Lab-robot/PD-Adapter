# GenDexGrasp：六线中的三只手，每物体追加 44 条源抓取

输出 `new/GEN/{inspire,ability,svh}.pt`，test10 每个物体包含 64 条全关节抓取。`0..19` 原样复用各手历史 GenDex test10 文件，`20..63` 从对应的 `GenDexGrasp/run_<hand>_mainset.py` 以固定新种子生成 44 条；旧文件及其 details 不覆盖。生成每个物体一份独立产物和日志，失败可逐物体续跑。配置保持原 `num_particles=32`、`max_iter=100`、`learning_rate=0.005`、`init_rand_scale=0.5` 和 `align_dist`，仅新样本的随机种子不同。

在 `/home/user/yangbo_program/new` 执行：

```bash
python GEN/generate_64.py check
python GEN/generate_64.py generate
python GEN/generate_64.py merge
```

默认每物体新增的 44 条分为 **4 份×11 条**；每份独立保存 `.pt`、日志和生成种子。可先运行 `python GEN/generate_64.py generate --hand inspire --objects contactdb+banana` 检查单物体生成，再执行全量 `generate`；已完成的分块会被校验并跳过。

或先跑一只手：`python GEN/generate_64.py all --hand inspire`。GenDex 优化仍逐样本进行，分块不会改变已冻结的模型和优化参数；中断后只需重跑受影响的一份。SVH 比 Ability 更慢；参照原来 20 条的记录，新增 44 条每物体约需其 2.2 倍时间，另有分块重新加载模型的开销（实际以 `_parts/<hand>/*.log` 为准）。如果从未开始某一手，也可为它选用 `--chunk-size 22`（两份）以减少加载；`generate` 和 `merge` 必须指定相同分块大小。生成日志、优化器详情和逐分块 JSON 均放在 `_parts/<hand>/`；合并文件旁的 JSON 清单保存旧数据、新种子及哈希。脚本会校验每个物体 64 条、原20条逐项不变、有限值和精确去重，不根据物理结果挑选源抓取。

两套生成脚本会争用同一块 A5000，建议不要同时运行。六份 `.pt` 仍是**源抓取**；必须另行按新 64 目标生成学习与随机暖启动、搜索和验证结果。旧的按20目标设置的每物体候选截断上限不可直接视为已对64目标公平。
