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

Nineteen reused thermal-core files remain byte-identical to the preceding
public export. Only its command entrypoint and source-contract support were
adapted for this catalog and explicitly labeled `RADONPY_EQ21` preparations.
The numerical execution code, density/transition QC policies, LAMMPS templates
and density protocol settings are unchanged. The previous 15-polymer package
and private research files were not replaced or deleted.

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

One simple catalog input continued from its original classical segment. At
30 ns cumulative sampling it passed the unchanged classical QC and produced
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
