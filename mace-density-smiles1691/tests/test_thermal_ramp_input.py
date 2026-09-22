from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

from thermal_properties.simulation import (
    ThermalSimulationError,
    execute_npt_stage,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
INITIALIZE_INPUT = (
    REPOSITORY_ROOT
    / "thermal_properties"
    / "lammps"
    / "in.initialize_mace_mh1.lmp"
)
NPT_INPUT = (
    REPOSITORY_ROOT
    / "thermal_properties"
    / "lammps"
    / "in.npt_stage_mace_mh1.lmp"
)
MACE_SETUP = (
    REPOSITORY_ROOT
    / "thermal_properties"
    / "lammps"
    / "mace_mh1_thermal_setup.mod"
)
CSV_HEADER = (
    "step",
    "time_ps",
    "temp_K",
    "press_bar",
    "density_g_cm3",
    "volume_A3",
)


def _active_lammps_source(source: str) -> str:
    return "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )


def _write_samples(path: Path, steps: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        for step in steps:
            writer.writerow((step, step * 0.00025, 300.0, 1.0, 1.0, 1000.0))


def _write_process_logs(invocation) -> None:
    invocation.stdout_path.touch()
    invocation.stderr_path.touch()
    invocation.log_path.write_text("synthetic LAMMPS log\n", encoding="utf-8")


def _write_success_artifacts(
    invocation,
    *,
    equilibration_steps: list[int],
    production_steps: list[int],
) -> None:
    _write_process_logs(invocation)
    _write_samples(invocation.equilibration_segment_path, equilibration_steps)
    _write_samples(invocation.production_segment_path, production_steps)
    invocation.final_restart_path.with_name("restart.equilibration").write_bytes(
        b"synthetic constant-equilibration boundary restart"
    )
    invocation.final_restart_path.write_bytes(b"synthetic final restart")


def _ramped_stage_spec(predecessor: Path, working_directory: Path) -> dict[str, object]:
    return {
        "stage_id": "replica_001__ramp_then_hold_300K",
        "replica_id": "replica_001",
        "start_step": 0,
        "ramp_steps": 4,
        "constant_equilibration_steps": 3,
        "equilibration_steps": 7,
        "production_steps": 2,
        "sample_every_steps": 1,
        "temperature_ramp_start_k": "500.0",
        "temperature_ramp_end_k": "300.0",
        "initialize_velocities": False,
        "predecessor_restart": str(predecessor),
        "working_directory": str(working_directory),
    }


class ThermalRampLammpsContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = NPT_INPUT.read_text(encoding="utf-8")
        cls.active_source = _active_lammps_source(cls.source)

    def test_ramp_fix_is_removed_before_constant_fix(self) -> None:
        ramp_fix = self.active_source.index("fix thermal_ramp_npt all npt")
        ramp_run = self.active_source.index('"run ${thermal_ramp_steps}"')
        ramp_unfix = self.active_source.index('"unfix thermal_ramp_npt"')
        constant_fix = self.active_source.index("fix thermal_stage_npt all npt")

        self.assertLess(ramp_fix, ramp_run)
        self.assertLess(ramp_run, ramp_unfix)
        self.assertLess(ramp_unfix, constant_fix)
        self.assertEqual(self.active_source.count("fix thermal_ramp_npt all npt"), 1)
        self.assertEqual(self.active_source.count("unfix thermal_ramp_npt"), 1)

    def test_constant_fix_spans_equilibration_and_production(self) -> None:
        constant_fix = self.active_source.index("fix thermal_stage_npt all npt")
        equilibration = self.active_source.index(
            'print "THERMAL_STAGE_SEGMENT=equilibration"'
        )
        production = self.active_source.index(
            'print "THERMAL_STAGE_SEGMENT=production"'
        )
        constant_unfix = self.active_source.index("unfix thermal_stage_npt")

        self.assertLess(constant_fix, equilibration)
        self.assertLess(equilibration, production)
        self.assertLess(production, constant_unfix)
        self.assertNotIn(
            "unfix thermal_stage_npt",
            self.active_source[constant_fix:production],
        )
        self.assertEqual(self.active_source.count("fix thermal_stage_npt all npt"), 1)
        self.assertEqual(self.active_source.count("unfix thermal_stage_npt"), 1)

    def test_npt_input_never_creates_velocities(self) -> None:
        commands = [line.strip() for line in self.active_source.splitlines()]
        self.assertFalse(
            any(line == "velocity" or line.startswith("velocity ") for line in commands)
        )

    def test_mace_model_path_expands_outside_parser_level_quotes(self) -> None:
        source = _active_lammps_source(MACE_SETUP.read_text(encoding="utf-8"))
        self.assertIn(
            "pair_style mliap unified ${thermal_mace_model} 0",
            source,
        )
        self.assertNotIn(
            'pair_style mliap unified "${thermal_mace_model}" 0',
            source,
        )

    def test_runtime_paths_expand_outside_parser_level_quotes(self) -> None:
        expected_fragments = (
            "log ${thermal_lammps_log}",
            "read_restart ${thermal_predecessor_restart}",
            "file ${thermal_equil_csv}",
            "file ${thermal_prod_csv}",
            '"restart ${thermal_restart_every_steps} ${thermal_equil_restart_root}"',
            '"restart ${thermal_restart_every_steps} ${thermal_prod_restart_root}"',
        )
        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, self.active_source)

        ramp = self.active_source.index(
            'print "THERMAL_STAGE_SEGMENT=temperature_ramp"'
        )
        equil_restart_root = self.active_source.index(
            '"restart ${thermal_restart_every_steps} ${thermal_equil_restart_root}"'
        )
        production_restart_root = self.active_source.index(
            '"restart ${thermal_restart_every_steps} ${thermal_prod_restart_root}"'
        )
        self.assertLess(equil_restart_root, ramp)
        self.assertLess(ramp, production_restart_root)
        self.assertNotIn("write_restart", self.active_source)

    def test_initialize_source_branch_reads_paths_as_direct_commands(self) -> None:
        source = _active_lammps_source(
            INITIALIZE_INPUT.read_text(encoding="utf-8")
        )
        expected_fragments = (
            '"jump SELF thermal_initialize_from_checkpoint"',
            "read_data ${thermal_input_data} nocoeff",
            "jump SELF thermal_initialize_source_loaded",
            "label thermal_initialize_from_checkpoint",
            "read_restart ${thermal_predecessor_restart}",
            "label thermal_initialize_source_loaded",
            '"restart ${thermal_restart_every_steps} ${thermal_initialize_restart_root}"',
            "log ${thermal_lammps_log}",
            "file ${thermal_initialize_csv}",
        )
        for fragment in expected_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, source)

        forbidden_fragments = (
            '"read_restart \'${thermal_predecessor_restart}\'"',
            '"read_data \'${thermal_input_data}\' nocoeff"',
            "'read_restart \"${thermal_predecessor_restart}\"'",
            "'read_data \"${thermal_input_data}\" nocoeff'",
            '"restart ${thermal_restart_every_steps} \'${thermal_initialize_restart_root}\'"',
            'log "${thermal_lammps_log}"',
            'file "${thermal_initialize_csv}"',
            'write_restart "${thermal_initialize_final_restart}"',
        )
        for fragment in forbidden_fragments:
            with self.subTest(forbidden=fragment):
                self.assertNotIn(fragment, source)
        self.assertNotIn("write_restart", source)

        input_read = source.index("read_data ${thermal_input_data} nocoeff")
        skip_checkpoint = source.index("jump SELF thermal_initialize_source_loaded")
        checkpoint_label = source.index("label thermal_initialize_from_checkpoint")
        restart_read = source.index(
            "read_restart ${thermal_predecessor_restart}"
        )
        loaded_label = source.index("label thermal_initialize_source_loaded")
        self.assertLess(input_read, skip_checkpoint)
        self.assertLess(skip_checkpoint, checkpoint_label)
        self.assertLess(checkpoint_label, restart_read)
        self.assertLess(restart_read, loaded_label)


class ExecuteRampedNptStageTests(unittest.TestCase):
    def test_fresh_stage_exports_full_ramp_and_constant_equilibration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "thermal root with spaces 热性质"
            root.mkdir()
            predecessor = root / "restart source with spaces 初始.restart"
            predecessor.write_bytes(b"predecessor")
            working_directory = root / "working directory 输入"
            working_directory.mkdir()
            stage = root / "stage directory 升降温"
            spec = _ramped_stage_spec(predecessor, working_directory)
            captured = []

            def complete(invocation):
                captured.append(invocation)
                self.assertIsNone(invocation.resume_checkpoint)
                self.assertEqual(invocation.remaining_equilibration_steps, 7)
                self.assertEqual(invocation.remaining_production_steps, 2)
                environment = invocation.environment
                self.assertEqual(environment["THERMAL_RAMP_STEPS"], "4")
                self.assertEqual(environment["THERMAL_EQUIL_STEPS"], "3")
                self.assertEqual(environment["THERMAL_PROD_STEPS"], "2")
                self.assertEqual(
                    Decimal(environment["THERMAL_RAMP_START_TEMP_K"]),
                    Decimal("500.0"),
                )
                self.assertEqual(
                    Decimal(environment["THERMAL_RAMP_END_TEMP_K"]),
                    Decimal("300.0"),
                )
                self.assertEqual(
                    environment["THERMAL_PREDECESSOR_RESTART"],
                    str(predecessor.resolve()),
                )
                self.assertEqual(invocation.cwd, working_directory.resolve())
                self.assertEqual(
                    environment["THERMAL_EQUIL_RESTART_ROOT"],
                    str(
                        (
                            stage
                            / "checkpoints"
                            / "equilibration.checkpoint.*.restart"
                        ).resolve()
                    ),
                )
                self.assertEqual(
                    environment["THERMAL_PROD_RESTART_ROOT"],
                    str(
                        (
                            stage
                            / "checkpoints"
                            / "production.checkpoint.*.restart"
                        ).resolve()
                    ),
                )
                self.assertIn("thermal root with spaces 热性质", environment["THERMAL_EQUIL_CSV"])
                self.assertFalse(environment["THERMAL_EQUIL_CSV"].startswith(("'", '"')))
                _write_success_artifacts(
                    invocation,
                    equilibration_steps=list(range(1, 8)),
                    production_steps=[8, 9],
                )
                return 0

            result = execute_npt_stage(
                stage_dir=stage,
                stage_spec=spec,
                command=("synthetic-lammps", "-in", "input path with spaces.lmp"),
                process_runner=complete,
            )

            self.assertEqual(result.execution_status, "COMPLETE")
            self.assertFalse(result.skipped)
            self.assertEqual(len(captured), 1)

    def test_resume_inside_ramp_exports_remaining_work_and_interpolated_start(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "resume root with spaces 恢复"
            root.mkdir()
            predecessor = root / "predecessor 初始.restart"
            predecessor.write_bytes(b"predecessor")
            working_directory = root / "working directory 恢复"
            working_directory.mkdir()
            stage = root / "ramp stage 恢复"
            spec = _ramped_stage_spec(predecessor, working_directory)
            invocations = []

            def interrupt_during_ramp(invocation):
                invocations.append(invocation)
                self.assertIsNone(invocation.resume_checkpoint)
                _write_process_logs(invocation)
                _write_samples(invocation.equilibration_segment_path, [1, 2])
                _write_samples(invocation.production_segment_path, [])
                checkpoint = (
                    invocation.checkpoint_dir
                    / "equilibration.checkpoint.2.restart"
                )
                checkpoint.write_bytes(b"checkpoint after two of four ramp steps")
                return 9

            with self.assertRaises(ThermalSimulationError):
                execute_npt_stage(
                    stage_dir=stage,
                    stage_spec=spec,
                    command=("synthetic-lammps",),
                    process_runner=interrupt_during_ramp,
                )

            def finish_from_ramp_checkpoint(invocation):
                invocations.append(invocation)
                self.assertIsNotNone(invocation.resume_checkpoint)
                self.assertEqual(
                    invocation.resume_checkpoint.completed_equilibration_steps,
                    2,
                )
                self.assertEqual(
                    invocation.environment["THERMAL_PREDECESSOR_RESTART"],
                    str(invocation.resume_checkpoint.path.resolve()),
                )
                self.assertEqual(invocation.remaining_equilibration_steps, 5)
                self.assertEqual(invocation.remaining_production_steps, 2)
                self.assertEqual(invocation.environment["THERMAL_RAMP_STEPS"], "2")
                self.assertEqual(invocation.environment["THERMAL_EQUIL_STEPS"], "3")
                self.assertEqual(invocation.environment["THERMAL_PROD_STEPS"], "2")
                self.assertEqual(
                    Decimal(invocation.environment["THERMAL_RAMP_START_TEMP_K"]),
                    Decimal("400.0"),
                )
                self.assertEqual(
                    Decimal(invocation.environment["THERMAL_RAMP_END_TEMP_K"]),
                    Decimal("300.0"),
                )
                _write_success_artifacts(
                    invocation,
                    equilibration_steps=[3, 4, 5, 6, 7],
                    production_steps=[8, 9],
                )
                return 0

            result = execute_npt_stage(
                stage_dir=stage,
                stage_spec=spec,
                command=("synthetic-lammps",),
                process_runner=finish_from_ramp_checkpoint,
            )

            self.assertEqual(result.execution_status, "COMPLETE")
            self.assertEqual(len(invocations), 2)
            with (stage / "samples" / "equilibration.csv").open(
                "r", encoding="utf-8", newline=""
            ) as handle:
                canonical_steps = [
                    int(row["step"]) for row in csv.DictReader(handle)
                ]
            self.assertEqual(canonical_steps, list(range(1, 8)))
            self.assertEqual(len(canonical_steps), len(set(canonical_steps)))


if __name__ == "__main__":
    unittest.main()
