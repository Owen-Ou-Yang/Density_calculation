# CRC reference configuration and approximate compute budget

Planning reference, updated 2026-09-24. Use the [split CPU/GPU workflow](CPU_GPU_SPLIT.md):
classical preparation gets a CPU allocation with **zero GPUs**, and only verified
`PREPARED_QC_PASS` tasks enter a separate MACE GPU allocation. No helper submits
or chains jobs automatically.

The timings below summarize one historical, approximately **3,600-atom**
case. They are not catalog averages or a successful complete density benchmark.
Only coarse resource/performance information is published: no density values,
experimental labels, structures, raw logs, receipts, job identities or private
installation paths are included.

## 1. Reference resources

| Item | Classical preparation | MACE density |
| --- | --- | --- |
| Allocation | Separate approved CPU queue; site must choose the actual queue | CRC Grid Engine `gpu@@zabaras_rtx6k` |
| GPU request | **0 GPUs** | **4 GPUs**, `gpu_card=4` |
| Historical reference GPU | Not used by classical preparation | 4 x Quadro RTX 6000; inventory reported 23,040 MiB per GPU |
| CPU allocation | 16 slots, subject to CPU-queue policy | `-pe smp 16` |
| Parallel layout | Classical LAMMPS: 16 MPI ranks x 1 OpenMP thread; Psi4: 16 threads | 4 MPI ranks x 4 OpenMP threads, one allocated GPU per rank |
| Scientific stack | RadonPy 0.2.11 / GAFF2_mod classical preparation | MACE-MH-1 / omol / float32, ML-IAP |
| Reference GPU environment | Not required | Previously used CUDA 13.2.1 / Intel MPI 2021.17 and the qualified LAMMPS build |
| Current example walltime request | **144 hours** | **192 hours (8 days)** |
| Example concurrency | 1 task initially | 1 task initially; 4 GPUs / 16 CPU slots per active task |

The public templates do not set every site-specific CPU budget to these historical
values automatically. Configure the preparation `mpi`, `omp` and `psi4_omp` values
explicitly and keep each within the allocation. CPU model and total peak host RAM
have not been calibrated into a portable requirement. A historical Psi4 setting
of `memory_mb=8000` is **not** the total host-memory requirement or a scheduler
memory reservation. Set host memory with the cluster's approved resource syntax.

Keep the same agreed checkpoint, head, dtype and qualified launcher. Record the
checkpoint hash in private site configuration. Another GPU type, three-GPU layout,
CPU type, MPI/CUDA build or larger cell needs its own timing and memory check;
these estimates must not be scaled by GPU count alone. See [environment setup](ENVIRONMENT.md).

**192 hours is a requested limit, not an established CRC queue maximum or an
approved request.** Confirm the queue permits it before submission. The legacy
combined profile still requests 144 hours and holds GPU resources during CPU
preparation; it is not the recommended split configuration. Queue waiting is
additional and is not included in any time below.

## 2. What was actually measured

| Historical work on the reference case | Elapsed wall-clock time | Resource accounting |
| --- | ---: | ---: |
| Initial preparation through the first classical sampling segment | About 9.9 h | About 158 CPU-slot-hours at 16 slots |
| Subsequent classical continuation, without repeating the initial preparation | About 33.2 h | About 532 CPU-slot-hours at 16 slots |
| Total classical preparation through classical QC | **About 43.1 h** | **About 690 CPU-slot-hours** |
| MACE initialization (1 ps) and transition (50 ps) | **About 37.1 h** | **About 148 GPU-hours** at 4 GPUs |

These stages originally ran in combined CPU/GPU allocations. The CPU-slot-hours
above are allocated-slot accounting, **not measured CPU utilization**. Reusing
43 h as a forecast for a new CPU-only queue assumes comparable CPU performance
and preparation/convergence behavior. The new split removes GPU reservation
during preparation; it does not speed up the classical algorithm.

In that legacy combined mode, the roughly 43 h of preparation also reserved
about **172 GPU-hours**, despite classical preparation not using the GPUs.
Including the subsequent partial MACE stages, the historical allocation cost
was therefore approximately **321 GPU-hours**, not just the 148 MACE-stage
GPU-hours. The split avoids that preparation-time GPU reservation.

The historical run did **not** produce an accepted final 300 K MACE density:
transition execution finished, but its QC did not pass. The measured 37 h / 148
GPU-hours therefore describes a partial MACE path, not the cost of a completed
density result. The newer bounded continuation has not yet completed real
end-to-end qualification. Classical preparation density is never substituted
for the requested MACE density.

## 3. Conditional MACE estimates for a fresh prepared cell

The measured 50 ps transition took about 36.3 h of engine time, approximately
**0.726 wall-hours/ps** on the four reference GPUs. Applying that rate gives:

| Scenario | Simulated MACE time | Estimated GPU-job elapsed time | Estimated allocated GPU-hours |
| --- | ---: | ---: | ---: |
| Every initial QC assessment passes | 1 + 50 + 0.1 + 5 + 25 = **81.1 ps** | **About 59 h** | **About 236 GPU-hours** |
| All bounded sampling budgets are used | 1 + 100 + 0.1 + 5 + 100 = **206.1 ps** | **About 150 h** | **About 600 GPU-hours** |
| One additional sampling window | **25 ps** | **About 18 h** | **About 73 GPU-hours** |

These are **extrapolations**, not measured successful full runs. In particular,
300 K target-stage speed was not measured in the reference run. Allow extra
time for minimization, initialization, I/O, analysis and performance variation.
QC success is not guaranteed, even after using the maximum budget. Runtime or
hard-QC failures stop the task; reaching a time limit cannot create a passing result.

Combining the illustrative 43 h preparation with the estimated 59–150 h MACE
path gives roughly **102–193 h (4.3–8 days) of serial execution**, before queue
waits and additional overhead. That is a conditional example, not a promise for
one arbitrary SMILES. Classical convergence can require more or less sampling;
1691 different inputs do not inherit this case's convergence history.

The 192 h GPU request provides about 42 h of headroom over this 150 h estimate.
It does not guarantee that every system fits. If it still expires, do not assume
automatic 300 K checkpoint recovery: the current public continuation selector
only supports eligible completed transition endpoints. See [recovery limits](CLUSTER_RUNBOOK.md#7-interpret-progress-interruption-and-scientific-qc).

## 4. CPU time, GPU time and allocation ceilings

- **Wall-clock hours**: elapsed running time of an allocation, excluding queue waiting.
- **Allocated CPU-slot-hours**: allocated CPU slots x allocation wall-clock hours.
  These are often used as an approximate core-hour budget, but actual slot/core
  mapping is site-specific; they are not CPU utilization measurements.
- **Allocated GPU-hours**: allocated GPU count x GPU-allocation wall-clock hours.
  These are not GPU utilization measurements or a statement about CRC billing.

The MACE job also reserves 16 CPU slots. Its estimated 59–150 h adds approximately
**940–2,400 CPU-slot-hours**, separate from the roughly 690 CPU-slot-hours of
classical preparation. Do not omit those CPU resources when requesting capacity.

The requested limits give an allocation ceiling, not an expected cost:

| One split task allocation | Limit-based ceiling if it runs for the entire requested time |
| --- | ---: |
| CPU preparation: 16 slots x 144 h | 2,304 CPU-slot-hours; zero GPU-hours |
| GPU density: 4 GPUs x 192 h | 768 GPU-hours |
| Supporting CPUs during GPU density: 16 slots x 192 h | 3,072 CPU-slot-hours |

For an observed rate `s` in wall-hours/ps and `p` new ps, estimate additional
GPU-hours as `GPU_count * s * p`, keeping initialization and non-MD overhead
separate. Do not restart the sampling budget on each scheduler submission or
count overlapping frames twice. All scientific thresholds and total sampling
caps remain unchanged.

## 5. Scaling to multiple polymers

At GPU concurrency `C`, this layout needs `4*C` GPUs and `16*C` CPU slots,
plus the independent CPU-preparation pool. Concurrency changes throughput,
not one polymer's MD duration. Queue contention and shared-storage traffic can
increase elapsed time.

There is no calibrated catalog-wide runtime predictor or total compute budget.
Do not treat `1691 * one_case_cost` as a reliable forecast. First qualify a full
split task, then measure a small representative set by atom count, chemistry,
classical convergence and MACE sampling needs. Keep success/failure counts,
CPU/GPU hours, peak RAM/VRAM and disk growth per stage. Real runtime measurements
and scientific QC, not green software tests, determine expansion readiness.
