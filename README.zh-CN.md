# 从 SMILES 到 MACE 聚合物密度

[English](README.md) · [集群运行手册](mace-density-smiles1691/docs/CLUSTER_RUNBOOK.md) · [完整命令](mace-density-smiles1691/README.md) · [环境配置](mace-density-smiles1691/docs/ENVIRONMENT.md)

这个仓库面向愿意提供算力的协作者，包含 **1,691 条原始聚合物 SMILES**、
结构准备和密度计算代码。使用已有的 **MACE-MH-1 / omol / float32** 模型，
不是训练新模型，也不包含模型权重。

## 计算的是哪一种密度？

1. 从原始 SMILES 建链、封端、装箱，分配 GAFF2_mod/RESP。
2. RadonPy 进行经典准备。首段 5 ns 的原始 QC 不通过但运行正常时，
   接续增加 5 ns；默认累计上限 50 ns。执行错误或非有限值立即停止。
3. **经典 QC 通过后**才进入 MACE：初始化、50 ps NPT transition，
   再进行 300 K 的平衡与采样。
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
进入 MACE，但完整真实端到端验收仍待完成。CPU 自动测试只验证软件行为。
最新已记录阶段和限制见 [VALIDATION.md](mace-density-smiles1691/VALIDATION.md)。

运行时间取决于原子数、经典收敛和 GPU 条件，可能需要数天；不要承诺每种材料
几小时完成。先测一条任务的耗时和存储增长，再规划大规模预算。

## 隐私与协作

公开包只有代码、SMILES 和合成测试数据，不含真实实验值、私有计算结果、
结构、轨迹、账号或密钥。输出和 site 配置应保存在**整个仓库之外**。
结果通过约定的私有渠道回传，不要贴到公开 issue。

SMILES 保留原字符串，只做完全相同字符串去重，不预先筛除无法建模的化学结构。
任务 ID 与 SMILES 哈希共同确定身份，不能拿旧 polymer 编号直接拼接结果。

贡献代码或算力请看 [CONTRIBUTING.md](CONTRIBUTING.md)。仓库已有 [MIT 许可证](LICENSE)，
依赖软件和模型仍遵循各自的授权条款。
