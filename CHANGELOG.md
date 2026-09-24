# Changelog

## CRC resource and compute-budget documentation — 2026-09-24

- Publish the reference four-GPU / 16-slot layout and separate CPU preparation
  profile, with concise English/Chinese README tables and a detailed cost guide.
- Separate measured partial-run times from the conditional 59–150 h MACE
  estimate, and account for supporting CPU slots and idle reserved GPU-hours.
- Clarify that 192 hours is a requested limit, queue approval is unverified,
  and a single-case estimate is not a catalog-wide runtime or QC guarantee.
- Documentation/inventory only: no scientific settings, runtime, SMILES,
  launcher or scheduler examples changed. No private density values, raw
  evidence, credentials or model weights are published.

## Separate CPU preparation and GPU density allocations — 2026-09-24

- Add explicit `prepare` and `density` execution stages; preserve legacy `all`.
  CPU preparation requests zero GPUs, needs no MACE installation/checkpoint,
  and finishes with `PREPARED_QC_PASS`, not a passing density result.
- Freeze verified CPU-parent receipts, structures and QC evidence into GPU
  plans. Copy prepared inputs into a new GPU attempt without repeating DFT,
  building, packing or classical equilibration. Preserve old attempts.
- Add CPU/GPU SGE and Slurm profiles, stage-specific site generation and a
  two-allocation runbook. GPU arrays contain only verified ready tasks with
  explicit mapping to original catalog identities. No automatic submission,
  job chaining or retries.
- Split GPU profiles request 192 hours; CPU preparation stays at 144 hours.
  This is a resource-budget change only and still requires queue-policy approval.
- Scientific adapters, model policy, protocol/QC, numerical core, catalog and
  launcher bytes are unchanged. Local fake-process tests do not establish
  real cluster qualification or completed end-to-end MACE acceptance.
- Local verification: **353 tests and 29 subtests passed**, including a
  1,691-entry synthetic array mapping with per-task parent checks. The 73-file
  public-package check passed; no real model or scheduler was run.

## Publication notes — 2026-09-24

- Publish the bounded MACE sampling revision below after a fresh CPU test run:
  273 tests and 29 subtests passed; the 64-file public-package check passed.
- Clarify the CRC example resource request, allocated versus MACE-stage
  GPU-hours, and the absence of a catalog-wide calibrated runtime estimate.
- Preserve the real end-to-end qualification limitation. No scientific results,
  trajectories, model weights or private site configuration are published.

## Bounded density sampling continuation — 2026-09-23

- Add `sampling_continuation` for density PILOT only: 25 ps increments,
  a 100 ps transition cap and a separate 100 ps target-sampling cap.
  Continue only when `minimum_effective_samples` is the sole rejected check;
  preserve the transition/target thresholds of 10/20.
- Keep the initial 305 K transition at 50 ps and the 300 K protocol at
  0.1 ps ramp, 5 ps equilibration and 25 ps initial sampling. Transition uses
  fresh, independently checked 25 ps extension windows. At 300 K, accumulate
  post-equilibration samples for QC at 25/50/75/100 ps, adding only 25 ps of
  new dynamics per extension. This is a prospective policy: prior QC records
  and evidence remain immutable. No post-hoc trimming or full-trajectory
  safety-gate change.
- Add explicit single-task `--continue-density-from` planning and execution
  from an eligible transition endpoint. New attempts verify and preserve
  parent/input/request, snapshot and model identities, copy prepared inputs,
  skip preparation/initialization, and retain window/restart/budget history.
  The cross-attempt selector does not restore a terminal 300 K target branch.
- Preserve `NEEDS_MORE_SAMPLING`, `SAMPLING_BUDGET_EXHAUSTED` and
  `SAMPLING_QC_FAILED` as non-passing sampling outcomes. Runtime/OOM, nonfinite,
  invalid-structure, temperature and other/mixed QC failures stop immediately.
  No scheduler submission or resubmission is introduced.
- Local verification: **273 tests and 29 subtests passed**, including real
  fake-adapter subprocess recovery, sampling caps, cumulative-row integrity,
  malformed receipts and immutable parent history. The public-package
  integrity/disclosure check passed for 64 files and all 1,691 original strings.
  Engineering coverage does not establish real CRC acceptance, a passing
  density, experimental agreement or catalog-wide scientific qualification.

## Cluster collaboration interface — 2026-09-22

- Add a standard-library offline `tools/cluster.py check/render` entrypoint,
  single-node Slurm/SGE resource profiles and a site-environment template.
- Generate private, non-overwriting submission bundles with explicit resource
  arguments, distinct array logs, file hashes and one coordinator per task.
  No submission, retry or model execution is performed by the helper.
- Check CPU budgets, checkpoint identity, private paths, allocation counts,
  array bounds and CRC-vs-Slurm launcher selection before starting preparation.
- Add a cluster collaborator runbook covering one-task qualification, assigned
  ranges, concurrency/resource budgets, storage and private evidence return.
- Scientific inputs, algorithms, numerical protocol/QC and existing launcher
  bytes are unchanged. The helper does not qualify a new cluster's GPU stack.
- The original raw array examples remain available; profile-based rendering is
  now the recommended entrypoint. No CPU/GPU job splitting or node-local staging
  is introduced in this revision.
- Local verification: **206 tests and 29 subtests passed**, including 124
  cluster-wrapper tests. The package checker passed for 60 package files;
  shell syntax and the 71-file public-repository disclosure scan passed.
  These are local fake-backend checks, not real Slurm/SGE or GPU qualification.

## GitHub readiness update — 2026-09-22

- Add repository-level English/Chinese entrypoints, contribution and privacy
  guidance, issue/PR templates and CPU-only continuous integration.
- Restore the omitted package `.gitignore` and executable shell-script modes.
  The initial web upload passed its 74 code tests but failed the package inventory
  check because `.gitignore` was missing.
- Align package licensing documentation with the existing root MIT license.
- Record classical-continuation/handoff progress without claiming a completed
  real end-to-end MACE acceptance test.
- Add release-packaging regressions and troubleshooting; refresh the package
  manifest as a new documentation/packaging revision.
- Keep all 1,691 SMILES bytes, protocol/QC settings, preparation and density
  runtime source, scheduler resource examples and launcher bytes unchanged.
- No real model calculation or scheduler submission is part of this update.
- Local verification: **82 tests and 29 subtests passed**; public package checker
  passed for 54 package files / 1,691 strings. All 33 scientific-input/runtime
  files checked against the frozen bounded-v2 release are byte-identical.
  Repository CI configuration is provided; local tests alone do not mean the
  workflow has already run successfully on GitHub.

## Bounded classical continuation — 2026-09-20

- Continue normally completed but QC-negative classical equilibration in 5 ns
  segments, default 50 ns total budget, without relaxing the original QC.
- Preserve per-segment evidence, stop on execution/nonfinite/structure errors,
  and permit MACE only after classical QC PASS.
- Local engineering tests: 74 passed, 29 subtests passed; fake scientific backends.
