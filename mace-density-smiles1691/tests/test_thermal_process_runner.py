from __future__ import annotations

from pathlib import Path
import json
import os
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from thermal_properties.simulation import (
    StageInvocation,
    _FATAL_RUNTIME_RETURN_CODE,
    _default_process_runner,
    _runtime_process_group_exists,
)


class ThermalProcessRunnerTests(unittest.TestCase):
    @staticmethod
    def _invocation(root: Path, command: tuple[str, ...]) -> StageInvocation:
        return StageInvocation(
            command=command,
            cwd=root,
            environment={},
            stdout_path=root / "stage.stdout.log",
            stderr_path=root / "stage.stderr.log",
            log_path=root / "log.lammps",
            equilibration_segment_path=root / "equilibration.csv",
            production_segment_path=root / "production.csv",
            final_restart_path=root / "final.restart",
            checkpoint_dir=root / "checkpoints",
            resume_checkpoint=None,
            remaining_equilibration_steps=0,
            remaining_production_steps=0,
            segment_index=0,
        )

    def test_normal_child_completion_is_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            invocation = self._invocation(
                root,
                (
                    sys.executable,
                    "-c",
                    "import sys; print('ok'); print('clean', file=sys.stderr)",
                ),
            )
            return_code = _default_process_runner(invocation)

            self.assertEqual(return_code, 0)
            self.assertEqual(invocation.stdout_path.read_text().strip(), "ok")
            self.assertEqual(invocation.stderr_path.read_text().strip(), "clean")

    def test_cuda_oom_terminates_hung_process_group(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            invocation = self._invocation(
                root,
                (
                    sys.executable,
                    "-c",
                    (
                        "import sys,time; "
                        "print('RuntimeError: CUDA out of memory', "
                        "file=sys.stderr, flush=True); time.sleep(30)"
                    ),
                ),
            )
            started = time.monotonic()
            return_code = _default_process_runner(invocation)
            elapsed = time.monotonic() - started

            self.assertEqual(return_code, _FATAL_RUNTIME_RETURN_CODE)
            self.assertLess(elapsed, 5.0)
            stderr = invocation.stderr_path.read_text(encoding="utf-8")
            self.assertIn("RuntimeError: CUDA out of memory", stderr)
            self.assertIn("terminated_process_group=yes", stderr)

    def test_old_oom_text_is_not_reinterpreted_as_a_new_failure(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            stderr_path = root / "stage.stderr.log"
            stderr_path.write_text(
                "RuntimeError: CUDA out of memory\n", encoding="utf-8"
            )
            invocation = self._invocation(
                root,
                (sys.executable, "-c", "print('new attempt')"),
            )

            self.assertEqual(_default_process_runner(invocation), 0)
            self.assertNotIn(
                "THERMAL_FATAL_RUNTIME_DETECTED",
                stderr_path.read_text(encoding="utf-8"),
            )

    def _assert_oom_detected(self, child_code: str) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            invocation = self._invocation(
                root, (sys.executable, "-B", "-c", child_code)
            )
            self.assertEqual(
                _default_process_runner(invocation), _FATAL_RUNTIME_RETURN_CODE
            )
            self.assertIn(
                "THERMAL_FATAL_RUNTIME_DETECTED",
                invocation.stderr_path.read_text(encoding="utf-8"),
            )

    def test_stdout_oom_is_detected(self) -> None:
        self._assert_oom_detected(
            "import os,time; os.write(1,b'RuntimeError: CUDA out of memory\\n'); "
            "time.sleep(0.6)"
        )

    def test_oom_before_long_stderr_block_is_not_discarded(self) -> None:
        self._assert_oom_detected(
            "import os,time; "
            "os.write(2,b'RuntimeError: CUDA out of memory\\n'+b'x'*8192); "
            "time.sleep(0.6)"
        )

    def test_oom_across_read_chunk_boundary_is_detected(self) -> None:
        self._assert_oom_detected(
            "import os,time; "
            "os.write(2,b'x'*(65536-10)+b'RuntimeError: CUDA out of memory\\n'"
            "+b'x'*65536); time.sleep(0.6)"
        )

    def test_oom_split_across_poll_reads_is_detected(self) -> None:
        self._assert_oom_detected(
            "import os,time; os.write(1,b'RuntimeError: CUDA out'); "
            "time.sleep(0.25); os.write(1,b' of memory\\n'); time.sleep(0.6)"
        )

    def test_oom_on_final_write_is_detected_despite_zero_child_exit(self) -> None:
        self._assert_oom_detected(
            "import os; os.write(2,b'torch.OutOfMemoryError: CUDA out of memory\\n')"
        )

    def test_old_stdout_oom_is_not_a_new_failure(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            (root / "stage.stdout.log").write_text(
                "RuntimeError: CUDA out of memory\n", encoding="utf-8"
            )
            invocation = self._invocation(
                root, (sys.executable, "-B", "-c", "print('new attempt')")
            )
            self.assertEqual(_default_process_runner(invocation), 0)
            self.assertEqual(invocation.stderr_path.read_text(), "")

    def test_separate_stream_fragments_do_not_create_a_false_oom(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            invocation = self._invocation(
                Path(raw_tmp),
                (
                    sys.executable, "-B", "-c",
                    "import os; os.write(1,b'RuntimeError: CUDA out'); "
                    "os.write(2,b' of memory\\n')",
                ),
            )
            self.assertEqual(_default_process_runner(invocation), 0)
            self.assertNotIn(
                "THERMAL_FATAL_RUNTIME_DETECTED", invocation.stderr_path.read_text()
            )

    def test_unrelated_nonzero_exit_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            invocation = self._invocation(
                Path(raw_tmp), (sys.executable, "-B", "-c", "raise SystemExit(7)")
            )
            self.assertEqual(_default_process_runner(invocation), 7)

    def _assert_term_resistant_descendant_is_stopped(self, *, leader_exits: bool) -> None:
        # The surviving descendant stays in the launcher's new process group,
        # like a local MPI rank. A heartbeat proves it is stopped even on hosts
        # whose init temporarily leaves the orphan as a non-running zombie.
        descendant_code = """
import pathlib, signal, sys, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
root = pathlib.Path(sys.argv[1])
(root / 'ready').touch()
with (root / 'heartbeat').open('ab', buffering=0) as heartbeat:
    for _ in range(200):
        heartbeat.write(b'x')
        time.sleep(0.02)
"""
        leader_code = """
import json, os, pathlib, subprocess, sys, time
root = pathlib.Path(sys.argv[1])
child = subprocess.Popen([sys.executable, '-B', '-c', sys.argv[2], str(root)])
(root / 'processes.json').write_text(json.dumps({'group': os.getpid(), 'child': child.pid}))
deadline = time.monotonic() + 2
while not (root / 'heartbeat').exists():
    if time.monotonic() > deadline:
        raise RuntimeError('fake descendant failed to start')
    time.sleep(0.005)
os.write(2, b'RuntimeError: CUDA out of memory\\n')
if sys.argv[3] != 'exit':
    time.sleep(3)
"""
        with tempfile.TemporaryDirectory() as raw_tmp:
            root = Path(raw_tmp)
            invocation = self._invocation(
                root,
                (
                    sys.executable, "-B", "-c", leader_code, str(root),
                    descendant_code, "exit" if leader_exits else "wait",
                ),
            )
            try:
                with (
                    patch("thermal_properties.simulation._FATAL_RUNTIME_POLL_SECONDS", 0.02),
                    patch("thermal_properties.simulation._FATAL_RUNTIME_TERMINATION_GRACE_SECONDS", 0.25),
                    patch("thermal_properties.simulation._FATAL_RUNTIME_KILL_GRACE_SECONDS", 0.25),
                ):
                    started = time.monotonic()
                    returncode = _default_process_runner(invocation)
                self.assertEqual(returncode, _FATAL_RUNTIME_RETURN_CODE)
                self.assertLess(time.monotonic() - started, 2.0)
                size_after_return = (root / "heartbeat").stat().st_size
                time.sleep(0.15)
                self.assertEqual((root / "heartbeat").stat().st_size, size_after_return)
                processes = json.loads((root / "processes.json").read_text())
                stderr = invocation.stderr_path.read_text()
                if _runtime_process_group_exists(processes["group"]):
                    self.assertIn("terminated_process_group=no", stderr)
                    self.assertIn("process_group_cleanup=UNCONFIRMED", stderr)
                else:
                    self.assertTrue(
                        "process_group_cleanup=CONFIRMED" in stderr
                        or "process_group_cleanup=UNCONFIRMED" in stderr
                    )
            finally:
                # Cleanup is restricted to the group and PID created by this
                # test. The fake descendant also has a four-second lifetime.
                if (root / "processes.json").exists():
                    processes = json.loads((root / "processes.json").read_text())
                    try:
                        os.killpg(processes["group"], signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_term_resistant_descendant_is_stopped_when_leader_exits_first(self) -> None:
        self._assert_term_resistant_descendant_is_stopped(leader_exits=True)

    def test_term_resistant_descendant_is_stopped_after_leader_handles_term(self) -> None:
        self._assert_term_resistant_descendant_is_stopped(leader_exits=False)


if __name__ == "__main__":
    unittest.main()
