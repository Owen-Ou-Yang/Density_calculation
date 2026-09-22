# Contributing code or compute

## Offer compute before starting a large batch

Use the [standard HPC runbook](mace-density-smiles1691/docs/CLUSTER_RUNBOOK.md)
and the profile-based `tools/cluster.py check/render` entrypoint. Keep local
site/resource/environment files and generated submission bundles private.

Open a compute-offer issue with GPU model/memory/count, CPU capacity, scheduler,
walltime/storage limits and the intended catalog range or shard. Do not include
login details, credentials or absolute private paths. Agree a non-overlapping
assignment with the maintainer before running; locks cannot coordinate separate
collaborators' filesystems.

Validate one input on your actual installation. Keep the original SMILES and
hash, agreed MACE-MH-1 omol checkpoint/float32 and protocol/QC unchanged.
Unsupported chemistry, OOM, failed preparation and failed QC must remain visible
as failures, not disappear from the denominator. A protocol change is a separate
experiment, not a repair to make a result pass.

## Submit a code change

1. Create a branch from the current repository default branch.
2. Make a focused change and add synthetic/fake-process regression tests.
3. Run the checks below from `mace-density-smiles1691/`.
4. Describe any scientific/runtime change explicitly. Documentation and packaging
   fixes should not change `inputs/`, `configs/`, numerical policies or engines.
5. Open a PR. Do not commit generated results, weights, site files or logs.

```bash
python -B tools/check_public_bundle.py
python -B -m pytest -q -p no:cacheprovider tests
for script in examples/crc_array.sh examples/slurm_array.sh launchers/crc_intelmpi_3or4gpu.sh; do
  bash -n "$script"
done
```

`PUBLIC_MANIFEST.json` seals the **package subdirectory**, not the entire Git
repository. After intentionally editing package files, update only the reviewed
entries' SHA-256/byte counts and explicit new file entries; label the new release
revision. Do not blindly rehash unknown files to silence a check. Preserve the
catalog bytes/identity, and review the Git diff before accepting new hashes.
Repository-level docs, GitHub templates and CI are reviewed through Git normally.
Runtime outputs and local configuration must be outside the repository before
running the release check.

CI runs on ordinary GitHub-hosted CPU runners. It must never need CRC credentials,
model weights, scheduler access or GPUs. Do not introduce `pull_request_target`
execution of contributor code or upload private simulation artifacts to Actions.

## Report a problem safely

Use the bug-report template. Share a minimal sanitized error excerpt and software
versions; full logs often contain private paths and numerical results. Distinguish
an engine error from ordinary QC rejection and from an attempt still running.
See [troubleshooting](mace-density-smiles1691/docs/TROUBLESHOOTING.md).

## Return results privately

Follow the [private handoff checklist](mace-density-smiles1691/docs/COLLABORATION.md#private-handoff).
Include the commit, catalog identity, protocol/site configuration, final receipts,
native run evidence, software versions and archive hashes. Keep work roots and
old attempts unchanged. Hash-verified transport is not a substitute for scientific
QC; failed tasks remain failed. Coordinate resource budgets, data access and credit
with the maintainer; this repository promises no authorship arrangement.
