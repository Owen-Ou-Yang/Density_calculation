# Compute collaboration

## Assignment

Coordinate catalog index ranges or shard assignments before running. All
participants use the same catalog hash and original strings. Publish neither
experimental values nor private computed results in public issues.

For a resource offer, describe GPU type and memory, available GPUs and CPUs,
scheduler, allowed walltime, available preparation/MACE software and proposed
task range. Do not share credentials or private host paths. This code does not
promise automatic compatibility with every accelerator or cluster.

## One task

Each task has an isolated directory and numbered attempts. The batch records
its exact original SMILES/hash, deterministic task seed, site configuration,
stage commands, logs, preparation files, result and final receipt. The provided
preparation adapter records actual construction/force-field choices; the MACE
adapter uses the existing numerical density workflow without tuning to labels.

The catalog is deliberately not prescreened. Unsupported repeat units, failed
force-field assignment, failed equilibration, memory errors and density QC
failures are legitimate outcomes. Report them; do not delete them from the
denominator or manufacture a density to fill the table.

## Continue safely

Re-run the same batch command to skip verified completed tasks and take
unstarted tasks. It does not restart a failed task automatically. Explicit
`--retry-failed` creates a new attempt for failed/incomplete tasks, preserving
old directories. It does not silently change parameters or loosen QC. Completed
QC-negative outcomes are preserved; a different scientific protocol needs a
separate coordinated campaign/work root and must not be pooled silently.

File locks prevent overlapping workers only when they share a filesystem with
working POSIX locks. Distinct collaborators' disks need non-overlapping assigned
ranges. A status command does not authorize another scheduler submission.

## Private handoff

Return `tools/export_summary.py` output plus complete per-task evidence:
attempt receipts, request/config, prepared structure and provenance, density
result, native run directory/manifests/QC/logs and environment versions. Include
archive SHA-256 and the original catalog SHA-256. Keep successful, failed and
unstarted tasks distinguishable. Preserve original paths in the evidence;
current verification expects the work root's paths, so an archive copied to a
different machine should first be checked for byte identity, not rewritten.

The summary reports the **MACE 300 K sampling** estimate only, never a classical
preparation density. A QC-negative estimate is a diagnostic, not a validated
material property. Experimental comparison remains with the coordinator, using
private labels and exact string identity; this package carries no benchmark
score or accuracy claim.

Agree licensing, resource budget, data transfer and credit/authorship directly.
Publishing this package alone does not settle those arrangements.
