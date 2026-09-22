"""Orchestration tests only: no RadonPy, RDKit, Psi4, LAMMPS, or model execution."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

MODULE = Path(__file__).resolve().parents[1] / "adapters" / "radonpy_prepare.py"
SPEC = importlib.util.spec_from_file_location("radonpy_prepare_under_test", MODULE)
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)


class Atom:
    def __init__(self, symbol, mass, isotope=0):
        self.symbol, self.mass, self.isotope = symbol, mass, isotope

    def GetSymbol(self): return self.symbol
    def GetMass(self): return self.mass
    def GetIsotope(self): return self.isotope


class Mol:
    def __init__(self, atoms): self.atoms = atoms
    def GetNumAtoms(self): return len(self.atoms)
    def GetAtoms(self): return self.atoms
    def GetAtomWithIdx(self, index): return self.atoms[index]
    def HasSubstructMatch(self, query, useChirality=False): return True


DATA = """Fake data; no scientific calculation was performed

6 atoms
2 atom types

0 10 xlo xhi
0 10 ylo yhi
0 10 zlo zhi

Masses

1 12.011
2 1.008

Atoms # full

1 1 1 0 1 1 1
2 1 2 0 2 1 1
3 1 2 0 3 1 1
4 2 1 0 4 1 1
5 2 2 0 5 1 1
6 2 2 0 6 1 1
"""


class PrepareTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.binary = self.root / "fake_lammps"
        self.binary.write_text("not an executable scientific engine\n")
        self.binary.chmod(0o700)
        self.output = self.root / "prep"
        self.request = {"task_id": "S000001", "smiles": "*CC*", "seed": 123,
                        "smiles_sha256": hashlib.sha256(b"*CC*").hexdigest(),
                        "attempt_id": "a1", "attempt_root": str(self.root),
                        "schema_version": "polymer-smiles-task/v1",
                        "prepared_manifest_path": str(self.output / "prepared.json"),
                        "catalog_sha256": "supplemental shared-runner metadata"}
        self.site = {"preparation": {"lammps_exec": str(self.binary), "chains": 2}}
        carbon, hydrogen = Atom("C", 12.011), Atom("H", 1.008)
        monomer = Mol([carbon, Atom("H", 3.016, 3), Atom("H", 3.016, 3)])
        terminal = Mol([carbon, Atom("H", 3.016, 3)])
        chain = Mol([carbon, hydrogen, hydrogen])
        cell = Mol(chain.atoms * 2)
        self.cell = cell
        analysis = SimpleNamespace(get_all_prop=Mock(return_value={}), check_eq=Mock(return_value=True))
        self.preset = SimpleNamespace(last_data="eq3_last.data", analyze=Mock(return_value=analysis))
        self.engine_codes = [0, 0, 0]
        self.additional_presets = []
        self.additional_checks = []
        self.additional_codes = []
        self.additional_properties = []
        case = self
        class FakeLAMMPS:
            def exec(self, **kwargs):
                return SimpleNamespace(returncode=case.engine_codes.pop(0))
        self.engine_class = FakeLAMMPS

        def execute(**kwargs):
            (self.output / "radonpy" / "eq3_last.data").write_text(DATA)
            for _ in range(3):
                FakeLAMMPS().exec()
            return cell

        self.preset.exec = Mock(side_effect=execute)

        def additional(*args, **kwargs):
            # These objects are explicit orchestration fakes, not a shortened
            # RadonPy calculation. Each fake still writes a distinct data file
            # and goes through the adapter's audited LAMMPS-return-code wrapper.
            offset = len(self.additional_presets)
            check = self.additional_checks[offset] if offset < len(self.additional_checks) else True
            code = self.additional_codes[offset] if offset < len(self.additional_codes) else 0
            properties = self.additional_properties[offset] if offset < len(self.additional_properties) else {}
            extra_analysis = SimpleNamespace(
                get_all_prop=Mock(return_value=properties), check_eq=Mock(return_value=check))
            extra = SimpleNamespace(last_data=f"eq{offset + 4}_last.data",
                                    analyze=Mock(return_value=extra_analysis))

            def execute_additional(**exec_kwargs):
                contents = DATA.replace("0 10 xlo xhi", f"0 {11 + offset} xlo xhi")
                Path(kwargs["work_dir"]).mkdir(parents=True, exist_ok=True)
                (Path(kwargs["work_dir"]) / extra.last_data).write_text(contents)
                self.engine_codes.append(code)
                FakeLAMMPS().exec()
                return cell

            extra.exec = Mock(side_effect=execute_additional)
            self.additional_presets.append(extra)
            return extra

        self.ff = SimpleNamespace(ff_assign=Mock(return_value=True))
        self.api = SimpleNamespace(
            versions={"radonpy-pypi": "0.2.11", "testing": "FAKE_ONLY"},
            np=SimpleNamespace(random=SimpleNamespace(seed=Mock())),
            Chem=SimpleNamespace(MolToSmiles=Mock(return_value="[3H]CC[3H]"),
                                 MolFromSmiles=Mock(return_value=monomer), MolFromSmarts=Mock(return_value=monomer)),
            utils=SimpleNamespace(mol_from_smiles=Mock(side_effect=[monomer, terminal])),
            GAFF2_mod=Mock(return_value=self.ff),
            qm=SimpleNamespace(conformation_search=Mock(return_value=(monomer, [0])), assign_charges=Mock(return_value=True)),
            lammps=SimpleNamespace(LAMMPS=FakeLAMMPS),
            poly=SimpleNamespace(calc_n_from_num_atoms=Mock(return_value=2), polymerize_rw=Mock(return_value=chain),
                                 terminate_rw=Mock(return_value=chain), amorphous_cell=Mock(return_value=cell)),
            eq=SimpleNamespace(EQ21step=Mock(return_value=self.preset),
                               Additional=Mock(side_effect=additional)),
        )

    def run_prepare(self):
        return adapter.prepare(self.request, self.output, self.site, backend=self.api)

    def test_success_binds_original_identity_and_actual_output(self):
        result = self.run_prepare()
        self.assertEqual(result["mace_elements"], ["C", "H"])
        self.assertEqual(result["task_id"], "S000001")
        self.assertEqual(Path(result["input_data"]).name, "eq3_last.data")
        metadata = json.loads(Path(result["snapshot_metadata"]).read_text())
        self.assertEqual(metadata["source_method"], "RADONPY_EQ21")
        self.assertEqual(metadata["snapshot_class"], "CLASSICAL_EQ2")
        self.assertEqual(metadata["element_counts"], {"C": 2, "H": 4})
        self.assertEqual(metadata["snapshot_sha256"], hashlib.sha256(DATA.encode()).hexdigest())
        provenance = result["preparation_provenance"]
        self.assertEqual(provenance["original_smiles"], "*CC*")
        self.assertEqual(provenance["builder_monomer_smiles"], "[3H]CC[3H]")
        self.assertFalse(provenance["mace_equilibrium_claim"])
        self.assertEqual(json.loads((self.output / "request_identity.json").read_text()), self.request)
        self.assertFalse((self.output / "preparation_failure.json").exists())
        self.api.poly.amorphous_cell.assert_called_once_with(self.api.poly.terminate_rw.return_value, 2, density=0.05)

    def test_prepared_manifest_resolves_through_actual_density_adapter(self):
        # Exercise the public handoff, including serialized files, but never a
        # chemistry backend, model loader, MD launcher, or campaign execution.
        with patch.object(sys, "path", [str(MODULE.parents[1]), *sys.path]):
            from adapters import mace_density
            from thermal_properties import simulation
            from thermal_properties.snapshot_contract import load_snapshot_contract

        weights = self.root / "fake-weights.pt"
        weights.write_bytes(b"synthetic weights: must never be loaded")
        launcher = self.root / "fake-density-launcher"
        launcher.write_text("#!/bin/sh\nexit 99\n")
        launcher.chmod(0o700)
        site = {**self.site, "density": {
            "model_path": str(weights), "model_sha256": mace_density.sha(weights),
            "launcher_path": str(launcher), "runtime_dependencies": [],
        }}
        density_output = self.root / "density"
        config_path = self.root / "density-config.json"
        with patch("subprocess.Popen", side_effect=AssertionError("no processes allowed")), patch.object(
            adapter, "load_backend", side_effect=AssertionError("fake preparation only")
        ), patch.object(simulation, "execute_thermal_campaign", side_effect=AssertionError("no MD execution")):
            adapter.prepare(self.request, self.output, site, backend=self.api)
            prepared = json.loads(Path(self.request["prepared_manifest_path"]).read_text())
            config = mace_density.build_config(self.request, prepared, site, density_output)
            config_path.write_text(json.dumps(config))
            resolved = simulation.resolve_thermal_config(config_path)
            system = resolved["system"]
            contract = load_snapshot_contract(
                snapshot_path=system["input_data"], metadata_path=system["snapshot_metadata"],
                expected_class=system["snapshot_class"],
            )

        self.assertEqual(system["polymer_id"], self.request["task_id"])
        self.assertEqual(system["input_data"], prepared["input_data"])
        self.assertEqual(system["snapshot_class"], "CLASSICAL_EQ2")
        self.assertEqual(contract.snapshot_class.value, "CLASSICAL_EQ2")
        self.assertEqual(contract.source_method, "RADONPY_EQ21")
        self.assertEqual(system["mace_elements"], ["C", "H"])
        self.assertEqual(system["element_list"], ["C", "H"])
        self.assertEqual(contract.atom_count, 6)
        self.assertEqual(contract.atom_type_count, len(system["mace_elements"]))
        self.assertEqual(dict(contract.element_counts), {"C": 2, "H": 4})
        self.assertEqual(contract.snapshot_sha256, hashlib.sha256(DATA.encode()).hexdigest())
        self.assertEqual(resolved["replicas"][0]["snapshot_metadata"], prepared["snapshot_metadata"])
        self.assertEqual(prepared["smiles_sha256"], self.request["smiles_sha256"])
        self.assertEqual(prepared["preparation_provenance"]["original_smiles"], self.request["smiles"])
        self.assertIsNone(config["density"]["reference_density_g_cm3"])
        self.assertFalse(density_output.exists())

    def test_identity_mismatch_does_not_start_or_create_output(self):
        self.request["smiles"] = "*C*"
        with self.assertRaisesRegex(adapter.PreparationError, "hash mismatch"):
            self.run_prepare()
        self.assertFalse(self.output.exists())
        self.api.utils.mol_from_smiles.assert_not_called()

    def test_explicit_double_bond_stereo_is_not_filtered(self):
        self.request["smiles"] = "*/C=C/CC*"
        self.request["smiles_sha256"] = hashlib.sha256(self.request["smiles"].encode()).hexdigest()
        self.run_prepare()
        self.assertEqual(json.loads((self.output / "request_identity.json").read_text())["smiles"], self.request["smiles"])
        self.assertTrue((self.output / "prepared.json").exists())

    def test_explicit_chiral_unit_uses_no_inversion_mode(self):
        self.request["smiles"] = "*C[C@H](F)C*"
        self.request["smiles_sha256"] = hashlib.sha256(self.request["smiles"].encode()).hexdigest()
        result = self.run_prepare()
        stereo = result["preparation_provenance"]["stereochemistry_assumptions"]
        self.assertEqual(stereo["requested_tacticity"], "atactic")
        self.assertEqual(stereo["effective_tacticity"], "isotactic")
        self.assertEqual(self.api.poly.polymerize_rw.call_args.kwargs["tacticity"], "isotactic")

    def test_actual_stereo_mismatch_fails_individually(self):
        self.api.qm.conformation_search.return_value[0].HasSubstructMatch = Mock(return_value=False)
        with self.assertRaisesRegex(adapter.PreparationError, "does not preserve"):
            self.run_prepare()
        self.api.eq.EQ21step.assert_not_called()
        self.assertFalse((self.output / "prepared.json").exists())

    def test_ff_failure_never_starts_equilibration(self):
        self.ff.ff_assign.return_value = False
        with self.assertRaisesRegex(adapter.PreparationError, "parameter assignment"):
            self.run_prepare()
        self.api.eq.EQ21step.assert_not_called()
        failure = json.loads((self.output / "preparation_failure.json").read_text())
        self.assertEqual(failure["stage"], "forcefield_assignment")
        self.assertFalse((self.output / "prepared.json").exists())

    def test_equilibrium_failure_does_not_publish_contract(self):
        self.preset.analyze.return_value.check_eq.return_value = False
        self.site["preparation"]["max_eq_step"] = 10.0
        self.additional_checks = [False]
        with self.assertRaisesRegex(adapter.PreparationError, "equilibrium check failed at maximum sampling budget"):
            self.run_prepare()
        self.api.eq.Additional.assert_called_once()
        self.assertFalse((self.output / "prepared.json").exists())
        self.assertFalse((self.output / "input.snapshot.json").exists())

    def test_initial_qc_failure_continues_then_passes_actual_last_snapshot(self):
        self.preset.analyze.return_value.check_eq.return_value = False
        self.site["preparation"]["max_eq_step"] = 10.0
        result = self.run_prepare()
        self.api.eq.Additional.assert_called_once()
        last = self.additional_presets[0]
        last.exec.assert_called_once()
        self.assertEqual(self.api.eq.Additional.call_args.kwargs["idx"], 4)
        final_path = Path(result["input_data"])
        self.assertEqual(final_path.name, "eq4_last.data")
        self.assertNotEqual(final_path, self.output / "radonpy" / "eq3_last.data")
        self.assertIn("0 11 xlo xhi", final_path.read_text())
        self.assertEqual((self.output / "radonpy" / "eq3_last.data").read_text(), DATA)
        metadata = json.loads(Path(result["snapshot_metadata"]).read_text())
        self.assertEqual(metadata["snapshot_sha256"], hashlib.sha256(final_path.read_bytes()).hexdigest())
        self.assertNotEqual(metadata["snapshot_sha256"], hashlib.sha256(DATA.encode()).hexdigest())
        provenance = result["preparation_provenance"]
        self.assertEqual(provenance["total_sampling_steps"], 10_000_000)
        history = provenance["equilibration_history"]
        self.assertEqual(len(history), 2)
        self.assertEqual([item["classical_equilibrium_check"] for item in history], [False, True])
        self.assertEqual([item["lammps_returncodes"] for item in history], [[0, 0, 0], [0]])
        self.assertEqual([item["completed_sampling_steps"] for item in history], [5_000_000, 10_000_000])
        # The expensive fresh-structure work is not repeated for a continuation.
        self.api.qm.conformation_search.assert_called_once()
        self.api.poly.polymerize_rw.assert_called_once()
        self.api.poly.amorphous_cell.assert_called_once()

    def test_first_check_passes_without_any_continuation(self):
        result = self.run_prepare()
        self.api.eq.Additional.assert_not_called()
        self.assertEqual(result["preparation_provenance"]["total_sampling_steps"], 5_000_000)
        self.assertEqual(len(result["preparation_provenance"]["equilibration_history"]), 1)

    def test_continuation_stops_at_first_success_with_bounded_accounting(self):
        self.preset.analyze.return_value.check_eq.return_value = False
        self.additional_checks = [False, True]
        self.site["preparation"]["max_eq_step"] = 20.0
        result = self.run_prepare()
        self.assertEqual(self.api.eq.Additional.call_count, 2)
        self.assertEqual([call.kwargs["idx"] for call in self.api.eq.Additional.call_args_list], [4, 5])
        self.assertEqual(Path(result["input_data"]).name, "eq5_last.data")
        self.assertEqual(result["preparation_provenance"]["total_sampling_steps"], 15_000_000)
        self.assertEqual([item["completed_sampling_steps"] for item in
                          result["preparation_provenance"]["equilibration_history"]],
                         [5_000_000, 10_000_000, 15_000_000])

    def test_nonzero_additional_exit_stops_without_analysis_or_publication(self):
        self.preset.analyze.return_value.check_eq.return_value = False
        self.additional_codes = [7]
        self.site["preparation"]["max_eq_step"] = 20.0
        original_exec = self.engine_class.exec
        with self.assertRaisesRegex(adapter.PreparationError, "LAMMPS exited with status 7"):
            self.run_prepare()
        self.assertIs(self.engine_class.exec, original_exec)
        self.api.eq.Additional.assert_called_once()
        self.additional_presets[0].analyze.assert_not_called()
        self.assertFalse((self.output / "prepared.json").exists())
        self.assertFalse((self.output / "input.snapshot.json").exists())
        record = json.loads((self.output / "equilibration_stage_eq0004.json").read_text())
        self.assertEqual(record["status"], "FAILED")
        self.assertEqual(record["lammps_returncodes"], [7])
        self.assertIsNone(record["classical_equilibrium_check"])

    def test_nonfinite_analysis_aborts_without_continuation(self):
        self.preset.analyze.return_value.get_all_prop.return_value = {"density": float("nan")}
        self.preset.analyze.return_value.check_eq.return_value = False
        with self.assertRaisesRegex(adapter.PreparationError, "nonfinite"):
            self.run_prepare()
        self.api.eq.Additional.assert_not_called()
        self.preset.analyze.return_value.check_eq.assert_not_called()
        self.assertFalse((self.output / "prepared.json").exists())

    def test_nonfinite_additional_analysis_does_not_trigger_another_extension(self):
        self.preset.analyze.return_value.check_eq.return_value = False
        self.additional_properties = [{"density": float("inf")}]
        self.additional_checks = [False]
        with self.assertRaisesRegex(adapter.PreparationError, "nonfinite"):
            self.run_prepare()
        self.api.eq.Additional.assert_called_once()
        self.additional_presets[0].analyze.return_value.check_eq.assert_not_called()
        self.assertFalse((self.output / "prepared.json").exists())

    def test_nonfinite_raw_thermo_is_rejected_even_if_derived_properties_are_finite(self):
        self.preset.analyze.return_value.get_all_prop.return_value = {"density": 0.8}
        self.preset.analyze.return_value.dfs = [SimpleNamespace(
            to_numpy=lambda: SimpleNamespace(tolist=lambda: [[0.0, float("nan")]]))]
        with self.assertRaisesRegex(adapter.PreparationError, "nonfinite"):
            self.run_prepare()
        self.api.eq.Additional.assert_not_called()
        self.preset.analyze.return_value.check_eq.assert_not_called()
        self.assertFalse((self.output / "prepared.json").exists())

    def test_analysis_exception_is_not_reclassified_as_nonconvergence(self):
        self.preset.analyze.return_value.get_all_prop.side_effect = RuntimeError("fake analysis failure")
        with self.assertRaisesRegex(RuntimeError, "fake analysis failure"):
            self.run_prepare()
        self.api.eq.Additional.assert_not_called()
        self.preset.analyze.return_value.check_eq.assert_not_called()
        self.assertFalse((self.output / "prepared.json").exists())
        record = json.loads((self.output / "equilibration_stage_eq0003.json").read_text())
        self.assertEqual(record["status"], "FAILED")
        self.assertIsNone(record["classical_equilibrium_check"])

    def test_broken_additional_structure_does_not_run_qc_or_extend(self):
        self.preset.analyze.return_value.check_eq.return_value = False
        original_constructor = self.api.eq.Additional.side_effect

        def broken_constructor(*args, **kwargs):
            preset = original_constructor(*args, **kwargs)
            original_execute = preset.exec.side_effect
            def broken_execute(**exec_kwargs):
                cell = original_execute(**exec_kwargs)
                path = Path(kwargs["work_dir"]) / preset.last_data
                path.write_text(path.read_text().replace("1 12.011", "1 16.0"))
                return cell
            preset.exec.side_effect = broken_execute
            return preset

        self.api.eq.Additional.side_effect = broken_constructor
        with self.assertRaisesRegex(adapter.PreparationError, "mass differs"):
            self.run_prepare()
        self.api.eq.Additional.assert_called_once()
        self.additional_presets[0].analyze.assert_not_called()
        self.assertFalse((self.output / "prepared.json").exists())

    def test_verified_legacy_continuation_starts_at_additional_without_initial_stages(self):
        # This unit test exercises the shared bounded engine only. Authenticating
        # old disk evidence is the private acceptance runner's separate job;
        # this fake returned cell is not claimed as verified scientific input.
        output, params = adapter._validate(self.request, self.output, self.site)
        output.mkdir()
        work = output / "radonpy"
        work.mkdir()
        self.engine_codes = []  # No initial EQ21 engine invocations in this path.
        result = adapter.equilibrate_bounded(
            self.api, self.cell, params, work, output, completed_steps=5_000_000,
            start_index=4, parent_history=[{"index": 3, "classical_equilibrium_check": False}])
        self.api.eq.EQ21step.assert_not_called()
        self.api.qm.conformation_search.assert_not_called()
        self.api.poly.polymerize_rw.assert_not_called()
        self.api.poly.amorphous_cell.assert_not_called()
        self.api.eq.Additional.assert_called_once()
        self.assertIs(self.api.eq.Additional.call_args.args[0], self.cell)
        self.assertEqual(self.api.eq.Additional.call_args.kwargs["idx"], 4)
        self.assertEqual(result["completed_steps"], 10_000_000)
        self.assertEqual(result["history"][-1]["lammps_returncodes"], [0])
        self.assertEqual(result["final_path"].name, "eq4_last.data")

    def test_legacy_continuation_at_cap_starts_no_engine(self):
        output, params = adapter._validate(self.request, self.output, self.site)
        output.mkdir()
        work = output / "radonpy"
        work.mkdir()
        with self.assertRaisesRegex(adapter.PreparationError, "below cap"):
            adapter.equilibrate_bounded(
                self.api, self.cell, params, work, output, completed_steps=50_000_000)
        self.api.eq.EQ21step.assert_not_called()
        self.api.eq.Additional.assert_not_called()

    def test_default_fifty_million_step_cap_is_finite(self):
        self.preset.analyze.return_value.check_eq.return_value = False
        self.additional_checks = [False] * 9
        with self.assertRaisesRegex(adapter.PreparationError, "maximum sampling budget"):
            self.run_prepare()
        self.assertEqual(self.api.eq.Additional.call_count, 9)
        self.assertEqual([call.kwargs["idx"] for call in self.api.eq.Additional.call_args_list], list(range(4, 13)))
        self.assertFalse((self.output / "prepared.json").exists())

    def test_continuation_preserves_qc_defaults_and_thermodynamic_conditions(self):
        self.preset.analyze.return_value.check_eq.return_value = False
        self.run_prepare()
        for preset in [self.preset, *self.additional_presets]:
            preset.analyze.return_value.check_eq.assert_called_once_with()
        for preset in self.additional_presets:
            kwargs = preset.exec.call_args.kwargs
            self.assertEqual(kwargs["temp"], 300.0)
            self.assertEqual(kwargs["press"], 1.0)
            self.assertEqual(kwargs["eq_step"], 5.0)
            self.assertNotIn("time_step", kwargs)

    def test_qc_threshold_override_is_rejected_before_execution(self):
        self.site["preparation"]["density_sma_sd_crit"] = 0.1
        with self.assertRaisesRegex(adapter.PreparationError, "unknown preparation parameters"):
            self.run_prepare()
        self.api.eq.EQ21step.assert_not_called()
        self.api.eq.Additional.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_invalid_continuation_budget_is_rejected_before_execution(self):
        for maximum in [0.0, 4.0, 7.5, float("nan"), float("inf"), True]:
            with self.subTest(max_eq_step=maximum):
                self.site["preparation"]["max_eq_step"] = maximum
                with self.assertRaises(adapter.PreparationError):
                    self.run_prepare()
                self.api.eq.EQ21step.assert_not_called()
                self.assertFalse(self.output.exists())

    def test_snapshot_identity_checks_mass(self):
        original = self.preset.exec.side_effect
        def wrong_mass(**kwargs):
            cell = original(**kwargs)
            path = self.output / "radonpy" / "eq3_last.data"
            path.write_text(DATA.replace("1 12.011", "1 16.0"))
            return cell
        self.preset.exec.side_effect = wrong_mass
        with self.assertRaisesRegex(adapter.PreparationError, "mass differs"):
            self.run_prepare()
        self.assertFalse((self.output / "prepared.json").exists())

    def test_missing_final_structure_cannot_be_success(self):
        self.preset.last_data = "missing.data"
        with self.assertRaisesRegex(adapter.PreparationError, "final data file missing"):
            self.run_prepare()
        self.assertFalse((self.output / "prepared.json").exists())

    def test_nonzero_engine_exit_is_failure_even_with_complete_files(self):
        original_exec = self.engine_class.exec
        self.engine_codes = [0, 1, 0]
        with self.assertRaisesRegex(adapter.PreparationError, "LAMMPS exited with status 1"):
            self.run_prepare()
        self.assertIs(self.engine_class.exec, original_exec)
        self.preset.analyze.assert_not_called()
        self.assertFalse((self.output / "prepared.json").exists())

    def test_no_reuse_and_no_experimental_density(self):
        self.run_prepare()
        with self.assertRaisesRegex(adapter.PreparationError, "no resume or overwrite"):
            self.run_prepare()
        self.site["preparation"]["experimental_density"] = 1.2
        with self.assertRaisesRegex(adapter.PreparationError, "unknown preparation"):
            adapter.prepare(self.request, self.root / "prep2", self.site, backend=self.api)


if __name__ == "__main__":
    unittest.main()
