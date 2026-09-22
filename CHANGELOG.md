# Changelog

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
