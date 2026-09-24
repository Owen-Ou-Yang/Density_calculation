# Public data boundary

The scientific inputs in this release are exactly the 1,691 original SMILES
strings in `inputs/smiles.csv`, with stable task IDs and per-string hashes.
Only exact duplicate strings were merged; no stereochemical normalization,
chemical filtering, or validity screening changed this list.

## Deliberately excluded

- Experimental density values, experimental source tables and comparison plots.
- Previous classical or MACE scientific numerical results, receipts and raw run histories.
- All former 15-polymer starting structures, any other structures or trajectories.
- Model weights, compiled binaries, environment archives, private site paths,
  credentials, SSH configuration or Git history.
- The old polymer-to-experiment mapping and experimental record multiplicities.

This is an input-catalog release, not a guarantee that source SMILES alone
determines molecular weight, tacticity, temperature history or experimental
conditions. Generated structures/results belong in the private work root.

The owner-authorized [CRC resource guide](CRC_RESOURCES_AND_COST.md) includes
coarse hardware, software-layout and elapsed-time summaries for compute planning.
These limited performance summaries are not a release of density values, raw
logs, structures, experimental labels or execution receipts. Estimated budgets
are labeled separately from measured partial-run costs.

## Before publication

1. Upload this new package only, never the enclosing research checkout.
2. Check the ZIP/file inventory with `tools/check_public_bundle.py`.
3. Retain the repository's MIT LICENSE and verify applicable third-party rights.
4. Never commit `site.local.json`, model weights or generated results.

`.gitignore` is a convenience, not protection against a forced upload.
`PUBLIC_MANIFEST.json` hashes the released files; the checker rejects changed,
missing or extra content, except normal Git/cache bookkeeping. If modifying the
package later, keep the original catalog and identity record intact and issue
a clearly labeled new code release instead of claiming the old hash still fits.

In the GitHub layout, keep generated work and summaries outside the **entire
repository**, including its root. The package checker covers this subdirectory;
root documentation, CI and contribution templates are reviewed separately in
the Git diff. A green checker is not a blanket confidentiality guarantee.
