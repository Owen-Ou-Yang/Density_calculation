# Release validation and limits

Validated on 2026-09-19 using Python 3.11.15, NumPy 2.4.3,
pandas 3.0.2 and pytest 9.0.3.

## Local release tests

From this package directory:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m pytest -q -p no:cacheprovider tests
```

Result: **59 passed, 23 subtests passed**. This covers catalog selection and
complete shard coverage; real fake-child subprocess execution; per-item failure
isolation; explicit retries; verified completed-task skipping; artifact and
identity mismatch rejection; task locks; interruption and nested child cleanup;
preparation contract handoff into the actual density configuration resolver;
virtual-environment interpreter path preservation; and density result/QC handling.
Scientific backends are faked in these tests. Existing process-runner and input
rendering regressions are also included.

Separately, the private source-to-catalog exporter passed **7 tests and 2
subtests**. The exporter and the original property table are not distributed.
All 1,691 exported strings were compared directly with the source strings, in
first-appearance order after exact-string deduplication. No chemistry parsing,
canonicalization, whitespace trimming, stereo rewriting or applicability
filtering was used to select this catalog. Each row has its own original-string
SHA-256. The only public CSV columns are task ID, SMILES and SMILES hash.

The packaging step checks every included file, rejects unexpected structures,
weights, results, caches, symlinks and private path/credential patterns, and
verifies ZIP bytes against the source package. `PUBLIC_MANIFEST.json` records
the release file inventory. Run `python -B tools/check_public_bundle.py` on a
fresh unpacked copy before adding local configuration or results.

## Reused scientific core

At the initial release, nineteen reused thermal-core files remained
byte-identical to the preceding public export. Only its command entrypoint
and source-contract support were adapted for this catalog and explicitly
labeled `RADONPY_EQ21` preparations. The numerical execution code,
density/transition QC policies, LAMMPS templates and density protocol settings
were unchanged then;
the bounded sampling revision below changes continuation behavior. The previous
15-polymer package and private research files were not replaced or deleted.

## Not validated by this release

A subsequent single-input real acceptance test completed DFT/RESP, chain
construction, packing and 5 ns classical sampling, but failed the original
classical equilibrium QC. It stopped before MACE; it did not establish
end-to-end acceptance. No private structure, density value or execution log is
included in this package. The bounded-extension revision addresses the missing
continuation behavior without relaxing QC. A subsequent recorded milestone
established classical QC PASS and entry into MACE (below), but not completed
end-to-end acceptance. CRC and Slurm examples remain templates.

## Bounded-extension revision, 2026-09-20

Local tests: **74 passed, 29 subtests passed** using fake scientific backends.
New coverage includes the first QC rejection followed by successful Additional
sampling, nine-extension/default-50M cap, stop at first PASS, no repeated
DFT/build, final-file handoff, unchanged thresholds, saved parent continuation,
nonzero exits, nonfinite analysis/thermo, damaged structures and no overwrite.
These tests are engineering evidence, not a successful MD result.

No claim is made that every string is buildable, that the potential covers all
chemistries, or that a single-packing PILOT yields a converged material density.
Preparation assumptions are not automatically matched to experimental sample
conditions. Unsupported entries remain in the catalog and receive explicit
failure records if attempted. Experimental density values and prior calculated
densities are absent, so this package cannot itself establish experimental
accuracy. It must never relabel failed or QC-negative tasks as successful data.

## Real continuation milestone recorded 2026-09-21

One simple catalog input continued from its original classical segment. During
bounded continuation it passed the unchanged classical QC and produced
the prepared-file handoff. MACE then started and completed initialization;
transition sampling was still running at the recorded observation.

This is a **dated milestone, not a live job-status claim**. A final verified
MACE result and completed end-to-end acceptance have not been recorded in this
release. No experimental value, real computed density, structure, raw log,
private host/path or execution receipt is published here. Terminal verification
must still establish finite 300 K density, `COMPLETE_QC_PASS`, `integrity=VERIFIED`
and the native run's artifact identities. A single successful input would only
qualify that PILOT; it would not validate the entire catalog or production use.

## GitHub documentation/packaging revision, 2026-09-22

The initial GitHub upload retained package bytes except that its hidden
`.gitignore` was omitted, and shell-script executable modes were lost. Baseline
CPU tests passed (74 tests and 29 subtests), but the inventory checker correctly
rejected the missing file. This revision restores that file and shell modes,
adds packaging regressions and CPU-only CI, and aligns notices with the existing
repository MIT license. The manifest is refreshed for this explicitly labeled
revision; the original frozen local release remains unchanged.

The SMILES catalog, protocol/QC, preparation adapter, density adapter, batch
runtime, numerical core and shell-script **bytes** are unchanged. CI checks
software and packaging only and cannot upgrade scientific acceptance.

## Cluster interface revision, 2026-09-22

The cluster wrapper standardizes offline configuration checks and single-node
Slurm/SGE array rendering. Its tests use real temporary files, synthetic model
bytes, fake coordinator executables, and local Bash subprocesses. They do not
call a scheduler, scientific backend or GPU. Runtime compatibility and real
end-to-end scientific acceptance remain unqualified by these tests.

The existing preparation/density adapters, batch execution engine, 1,691-string
catalog, scientific protocol/QC, numerical core and CRC launcher bytes remain
unchanged. A collaborator must qualify the installed environment and one full
task before expanding to a large assigned range; scheduler standardization alone
does not validate a different MPI/CUDA build or model memory requirements.

## Bounded MACE sampling revision, 2026-09-23

The earlier single-input milestone subsequently reached a terminal transition
QC rejection for insufficient effective sampling. That outcome is evidence of
a completed but unaccepted transition, not a final density or a completed real
SMILES-to-MACE acceptance test. The failed record remains part of the private
evidence; this public account contains no real density, trajectory, structure,
job identity or private path.

The new density-PILOT-only `sampling_continuation` policy adds 25 ps of dynamics
only if `minimum_effective_samples` is the sole rejected QC check.
The initial 305 K transition remains 50 ps, capped at 100 ps cumulative
transition sampling. The 300 K branch keeps its 0.1 ps ramp, 5 ps equilibration
and initial 25 ps sampling, capped at 100 ps cumulative target sampling.
Thresholds stay at 10 transition / 20 target effective samples. Transition
extensions receive independent QC on fresh 25 ps windows without pooling the
previous failed transition samples. Target QC instead uses all cumulative
post-equilibration samples at 25, 50, 75 and 100 ps, adding only 25 ps of new
dynamics each time. This is a prospective protocol, not a reanalysis that
changes the status of prior evidence. Previous QC records remain immutable;
each target reassessment records its contributing segments and a new QC result.

The explicit single-task `--continue-density-from` interface creates a new
attempt from an eligible transition endpoint after checking saved provenance
and current model identity. It preserves the parent, copies verified prepared
inputs, skips preparation/initialization, and retains restart/window identities
and consumed budgets. The current cross-attempt selector does not restore a
terminal 300 K target branch. No scheduler submission or automatic resubmission
is part of this revision.

Runtime/OOM, nonfinite values, invalid structures, temperature violations and
other or mixed QC failures stop continuation. `NEEDS_MORE_SAMPLING`,
`SAMPLING_BUDGET_EXHAUSTED` and `SAMPLING_QC_FAILED` are not passing outcomes.
Full-trajectory safety checks remain in force: no post-hoc burn-in trimming or
threshold or safety-policy change is introduced. A future analysis-only burn-in
study would be a separate protocol, not evidence that the current trajectory passed.

Integrated engineering verification on 2026-09-23: **273 tests and 29 subtests
passed** in 66.42 seconds. Tests include real fake-adapter subprocess recovery,
unchanged parent bytes, no repeated preparation/initialization, 50+25+25 ps
transition budgeting, cumulative 25/50/75/100 ps target analysis, truncated or
duplicated sample rejection, runtime/hard-QC stopping, and malformed statistical
receipts. The public-package checker passed for 64 files and all 1,691 original
SMILES strings. Catalog, QC-policy, LAMMPS-template and CRC-launcher bytes match
the preceding revision; no real model or scheduler was run.

Synthetic fixtures and fake backends verify software behavior, not real CRC execution or physical
convergence. A new real end-to-end PASS has not been established by this code
change. Final acceptance still requires a finite 300 K density,
`COMPLETE_QC_PASS`, `integrity=VERIFIED` and verified native artifacts.
