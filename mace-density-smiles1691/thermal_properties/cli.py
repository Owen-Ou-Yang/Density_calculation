"""Compatibility entry point; only the public SMILES batch is exposed."""
if __name__ == "__main__":
    from polymer_batch.cli import main
    raise SystemExit(main())
