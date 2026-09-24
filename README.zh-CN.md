# 从 SMILES 到 MACE 聚合物密度

[English](README.md) · [集群运行手册](mace-density-smiles1691/docs/CLUSTER_RUNBOOK.md) · [完整命令](mace-density-smiles1691/README.md) · [环境配置](mace-density-smiles1691/docs/ENVIRONMENT.md)

这个仓库面向愿意提供算力的协作者，包含 **1,691 条原始聚合物 SMILES**、
结构准备和密度计算代码。使用已有的 **MACE-MH-1 / omol / float32** 模型，
不是训练新模型，也不包含模型权重。

## 计算的是哪一种密度？

1. 从原始 SMILES 建链、封端、装箱，分配 GAFF2_mod/RESP。
2. RadonPy 进行经典准备。首段 5 ns 的原始 QC 不通过但运行正常时，
   接续增加 5 ns；默认累计上限 50 ns。执行错误或非有限值立即停止。
3. **经典 QC 通过后**才进入 MACE：初始化、305 K 下初始 50 ps NPT transition，
   再进行 300 K 的平衡与采样。仅有效样本数不足时允许有限续采样，规则见下文。
4. 从 MACE 的 300 K 采样窗口计算密度，保留最终 QC、receipt 和原始证据。

经典准备得到的密度只用于准备阶段，**不能冒充最终 MACE 密度**。
当前协议是单个装箱体系的 PILOT/screening，`scientific_eligible=false`，
不能自动当成生产级材料性质或实验精度保证。

## 先检查代码，再使用算力

从仓库根目录开始，使用 Python 3.11+：

```bash
python3 -m venv .venv
source .venv/bin/activate
cd mace-density-smiles1691
python -m pip install -r requirements-test.txt
python -B tools/check_public_bundle.py
python -B -m polymer_batch.cli plan --task-index 1
python -B -m pytest -q -p no:cacheprovider tests
```

这里不会调用真实 MACE、LAMMPS、GPU 或调度器。完整计算还需另外安装
RadonPy/经典 LAMMPS、MACE/ML-IAP/CUDA/MPI 等环境，并取得约定 checkpoint。
请按[主运行指南](mace-density-smiles1691/README.md#configure-once)生成本地配置。

先在计算节点验证一条任务，再协调分片或 array。默认全目录运行是顺序处理，
不是同时启动 1,691 个作业。CRC launcher 不能直接当作 Slurm launcher。
只读检查任务可用：

```bash
python -B -m polymer_batch.cli status --task-index 1 \
  --work-root /absolute/path/to/private_results
```

`INCOMPLETE` 在运行中并不代表失败，应同时查看 scheduler 和阶段日志。
只有最终 `COMPLETE_QC_PASS` 且 `integrity=VERIFIED` 才算本次 PILOT 完整通过。
批量运行会保留失败任务；不会放宽 QC，也不会自动重交 scheduler 作业。

## 有限的 MACE 续采样

这是一项预先规定的新协议：只有 `minimum_effective_samples` 是唯一失败项时，
才可从已验证的末态接续 25 ps 动力学。305 K transition 初始为 50 ps，
后续每个 25 ps 窗口独立评估，不与之前失败的 transition 样本合并，累计上限
100 ps。300 K 保留 0.1 ps 降温、5 ps 平衡和初始 25 ps 采样；每次只新增
25 ps 动力学，对平衡后的累计 25、50、75、100 ps 样本重新计算 QC，采样
上限为 100 ps。阈值仍为 transition 10、300 K 20 个有效样本。
旧 QC 记录、失败状态和原始证据保持不变，不按新协议事后重标旧结果。

运行错误、OOM、非有限值、结构或温度异常，以及其他 QC 失败或混合失败，
均立即停止。`NEEDS_MORE_SAMPLING`、`SAMPLING_BUDGET_EXHAUSTED` 和
`SAMPLING_QC_FAILED` 都不表示通过。完整轨迹的安全检查继续适用；本次不做
事后删去初始数据的 burn-in 处理，也不更改 QC 阈值或安全检查规则。

已有失败任务需明确选择父 attempt，离线检查后再在已获准的计算资源内运行；
见[续采样命令](mace-density-smiles1691/README.md#bounded-mace-sampling-continuation)。
当前跨 attempt 的入口仅选择 transition 末态，不是任意 300 K 任务的恢复入口。
新 attempt 复制并验证准备文件、父任务和输入身份、代码快照及模型哈希，跳过
准备和 MACE 初始化，并保留父任务、每段样本、每次 QC、restart、历史和累计预算。
不会自动提交或重交作业。

## 集群协作的标准入口

优先使用 [HPC 手册](mace-density-smiles1691/docs/CLUSTER_RUNBOOK.md)和
`examples/cluster.slurm.json` 或 `examples/cluster.sge.json`。在仓库之外保存三份站点文件：

- `site.local.json`：解释器、模型与 launcher 的真实路径。
- `cluster.local.json`：分区/队列、账户、GPU/CPU/内存、walltime、任务范围和并发数。
- `environment.sh`：由集群使用者确认的模块与运行库配置。

从包目录运行：

```bash
python -B tools/cluster.py check --profile /shared/project/density-private/cluster.local.json
python -B tools/cluster.py render --profile /shared/project/density-private/cluster.local.json \
  --output-dir /shared/project/density-private/submissions/qualification-001
```

工具只检查并生成脚本，**不会自动提交**。协作者检查后执行生成的 `submit-command.txt`
中的完整命令，不能省略资源参数直接执行 `sbatch job.sh` / `qsub job.sh`。
每个数组元素只启动一份 Python 驱动，内部 launcher 再启动 MPI；不要外层重复套并行命令。
先用默认的单任务/并发 1 做真实验收，再按分配范围扩大，避免一上来申请 1,691 份算力。

目前统一的是提交和协作接口，不是打包所有集群的 CUDA/MPI 环境；Slurm 仍需要该站点
适配的 launcher。没有改模型、checkpoint、dtype 或科学 QC，也没有新增真实计算。

## 当前验证范围

这不是“1,691 种材料已全部验证”的发布。已记录一次单输入的经典续跑通过并
进入 MACE，但 transition 未满足有效样本数要求，尚无最终通过的密度。
本次有限续采样是工程修改，完整真实端到端验收仍待完成。CPU 自动测试只验证软件行为。
最新已记录阶段和限制见 [VALIDATION.md](mace-density-smiles1691/VALIDATION.md)。

运行时间取决于原子数、经典收敛和 GPU 条件，可能需要数天；不要承诺每种材料
几小时完成。先测一条任务的耗时和存储增长，再规划大规模预算。

## CRC 配置与 CPU／GPU 时间预算

仓库的旧版合并式 [SGE 示例](mace-density-smiles1691/examples/cluster.sge.json)申请
`gpu@@zabaras_rtx6k` 队列、4 张 GPU、`smp` 16 个 CPU slots，walltime 上限
144 小时。MACE launcher 使用 4 个 MPI ranks，每个 rank 4 个线程。
默认只选择一条任务、并发 1；实际节点的 GPU 型号与显存需要现场确认。
144 小时是调度时限，不是预计完成时间。

推荐使用的[分阶段 GPU 配置](mace-density-smiles1691/examples/cluster.density.sge.json)
改为申请 **192 小时（8 天）**，经典准备另用 CPU 作业。
提交前须确认所选队列允许该时长；这不是全部 polymer 都能按时完成的保证。

以历史上约 **3,600 原子、4 张 Quadro RTX 6000** 的体系作参考：

| 阶段 | 推荐分阶段资源 | 大致实际耗时／估算耗时 | 资源占用小时 |
| --- | --- | --- | --- |
| 经典准备 | 16 个 CPU slots，0 GPU | 约 **43 小时**，历史 CPU/GPU 节点实测 | 约 **690 CPU-slot-hours** |
| MACE 密度 | 4 GPU + 16 CPU slots；4 MPI × 4 线程 | 约 **59–150 小时**，由实测过渡速度外推 | 约 **236–600 GPU-hours**，另占 **940–2,400 CPU-slot-hours** |

MACE 下限对应各初始窗口通过，上限对应用尽当前有限采样预算，**不是已经成功
跑完的端到端耗时**：历史上只完成了约 37 小时的初始化／过渡，尚未进入最终
300 K 密度阶段。新的 CPU 队列速度可能不同；排队、额外 I/O 和分析开销另计，
用尽预算也不保证 QC 通过。这一例子的准备加 MACE 条件估算约为 **4.3–8 天**，
不能当作所有 polymer 的固定耗时。
完整配置、软件版本和计算口径见[CRC 配置与算力预算](mace-density-smiles1691/docs/CRC_RESOURCES_AND_COST.md)。

目前没有对全部 1,691 条输入校准过的自动 GPU-hours 估算器。预算应区分：

- **占用 GPU-hours = 分配 GPU 数 × 作业实际运行小时数**。
- **MACE 阶段 GPU-hours = GPU 数 × MACE 阶段小时数**。
- 追加采样估算：`GPU 数 × 同类体系实测小时/ps × 新增 ps`。

旧版合并方式将准备与 MACE 放在同一个 allocation 中顺序执行，CPU 经典准备期间也保留
GPU，因此资源占用不等于 GPU 实际繁忙时间，也不一定等于集群计费口径。
排队时间不计作计算时间；初始准备、初始化／平衡、失败尝试和存储需要单独计入。
不能用一个小体系的速度直接保证所有 polymer 的成本。

批量运行可改用[CPU 准备／GPU 密度分离入口](mace-density-smiles1691/docs/CPU_GPU_SPLIT.md)：
CPU array 不申请 GPU；完成后，仅为已验证 `PREPARED_QC_PASS` 的任务生成 GPU array。
旧的 `stage=all` 兼容入口仍会在准备期间占着 GPU。两阶段交接校验结构和收据，
GPU 阶段不重复经典准备；工具不自动提交下一阶段。准备成功不代表密度计算完成。

**规模化状态：具备 array／分片和有限续采样机制，但新版完整真实验收仍待通过。**
先完成单任务的 300 K 密度与完整性验收，再用少量有代表性的体系测量成功率、
显存和成本，之后扩展已分配范围。暂不应把 1,691 条目录当作已验证的生产任务。

## 隐私与协作

公开包只有代码、SMILES 和合成测试数据，不含真实实验值、私有计算结果、
结构、轨迹、账号或密钥。输出和 site 配置应保存在**整个仓库之外**。
结果通过约定的私有渠道回传，不要贴到公开 issue。

SMILES 保留原字符串，只做完全相同字符串去重，不预先筛除无法建模的化学结构。
任务 ID 与 SMILES 哈希共同确定身份，不能拿旧 polymer 编号直接拼接结果。

贡献代码或算力请看 [CONTRIBUTING.md](CONTRIBUTING.md)。仓库已有 [MIT 许可证](LICENSE)，
依赖软件和模型仍遵循各自的授权条款。
