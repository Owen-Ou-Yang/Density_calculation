"""Deterministic expansion of a thermal protocol into per-replica steps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from .config import StatePointConfig, ThermalConfigError, ThermalProtocolConfig
from .provenance import AnalysisStatus, ExecutionStatus, QCStatus, canonical_sha256


class ThermalPlanError(ValueError):
    """Raised when a plan would violate restart or initialization invariants."""


@dataclass(frozen=True)
class PlanStep:
    replica_id: str
    sequence_index: int
    state: StatePointConfig
    steps: int
    predecessor_restart: str | None
    output_restart: str
    initialize_velocities: bool
    execution_status: ExecutionStatus = ExecutionStatus.PLANNED
    qc_status: QCStatus = QCStatus.NOT_EVALUATED
    analysis_status: AnalysisStatus = AnalysisStatus.NOT_STARTED

    def __post_init__(self) -> None:
        if not isinstance(self.sequence_index, int) or isinstance(self.sequence_index, bool):
            raise ThermalPlanError("sequence_index must be an integer")
        if self.sequence_index < 0:
            raise ThermalPlanError("sequence_index must be >= 0")
        if not isinstance(self.steps, int) or isinstance(self.steps, bool) or self.steps <= 0:
            raise ThermalPlanError("steps must be a positive integer")
        if type(self.initialize_velocities) is not bool:
            raise ThermalPlanError("initialize_velocities must be an explicit boolean")
        if self.initialize_velocities and self.predecessor_restart is not None:
            raise ThermalPlanError(
                "a velocity-initializing step cannot also consume a predecessor restart"
            )
        if not self.initialize_velocities and self.predecessor_restart is None:
            raise ThermalPlanError(
                "every noninitializing step must consume its predecessor restart"
            )
        if not self.output_restart:
            raise ThermalPlanError("output_restart must not be empty")
        try:
            object.__setattr__(self, "execution_status", ExecutionStatus(self.execution_status))
            object.__setattr__(self, "qc_status", QCStatus(self.qc_status))
            object.__setattr__(self, "analysis_status", AnalysisStatus(self.analysis_status))
        except ValueError as exc:
            raise ThermalPlanError(f"invalid step status: {exc}") from exc

    @property
    def state_id(self) -> str:
        return self.state.state_id

    @property
    def use_for_tg_fit(self) -> bool:
        return self.state.use_for_tg_fit

    @property
    def use_for_density(self) -> bool:
        return self.state.use_for_density

    def to_dict(self) -> dict[str, object]:
        return {
            "replica_id": self.replica_id,
            "sequence_index": self.sequence_index,
            "state": self.state.to_dict(),
            "steps": self.steps,
            "predecessor_restart": self.predecessor_restart,
            "output_restart": self.output_restart,
            "initialize_velocities": self.initialize_velocities,
            "execution_status": self.execution_status.value,
            "qc_status": self.qc_status.value,
            "analysis_status": self.analysis_status.value,
        }


@dataclass(frozen=True)
class ThermalPlan:
    protocol_sha256: str
    replica_ids: tuple[str, ...]
    steps: tuple[PlanStep, ...]
    execution_status: ExecutionStatus = ExecutionStatus.PLANNED
    qc_status: QCStatus = QCStatus.NOT_EVALUATED
    analysis_status: AnalysisStatus = AnalysisStatus.NOT_STARTED

    def __post_init__(self) -> None:
        object.__setattr__(self, "replica_ids", tuple(self.replica_ids))
        object.__setattr__(self, "steps", tuple(self.steps))
        try:
            object.__setattr__(self, "execution_status", ExecutionStatus(self.execution_status))
            object.__setattr__(self, "qc_status", QCStatus(self.qc_status))
            object.__setattr__(self, "analysis_status", AnalysisStatus(self.analysis_status))
        except ValueError as exc:
            raise ThermalPlanError(f"invalid plan status: {exc}") from exc
        validate_restart_chain(self)

    def steps_for_replica(self, replica_id: str) -> tuple[PlanStep, ...]:
        if replica_id not in self.replica_ids:
            raise KeyError(f"unknown replica_id: {replica_id}")
        return tuple(step for step in self.steps if step.replica_id == replica_id)

    def to_dict(self) -> dict[str, object]:
        return {
            "protocol_sha256": self.protocol_sha256,
            "replica_ids": list(self.replica_ids),
            "execution_status": self.execution_status.value,
            "qc_status": self.qc_status.value,
            "analysis_status": self.analysis_status.value,
            "steps": [step.to_dict() for step in self.steps],
        }


def _restart_path(restart_root: str | Path, replica_id: str, index: int, state_id: str) -> str:
    root = PurePosixPath(str(restart_root))
    return str(root / replica_id / f"state_{index:04d}_{state_id}.restart")


def build_thermal_plan(
    config: ThermalProtocolConfig,
    restart_root: str | Path = "restart",
) -> ThermalPlan:
    """Expand all state points for every replica without altering the schedule."""

    if not isinstance(config, ThermalProtocolConfig):
        raise ThermalConfigError("config must be ThermalProtocolConfig")
    plan_steps: list[PlanStep] = []
    for replica_id in config.replica_ids:
        previous_restart: str | None = None
        for index, state in enumerate(config.state_points):
            output_restart = _restart_path(
                restart_root, replica_id, index, state.state_id
            )
            plan_steps.append(
                PlanStep(
                    replica_id=replica_id,
                    sequence_index=index,
                    state=state,
                    steps=state.steps(config.timestep_ps),
                    predecessor_restart=previous_restart,
                    output_restart=output_restart,
                    initialize_velocities=index == 0,
                )
            )
            previous_restart = output_restart

    return ThermalPlan(
        protocol_sha256=canonical_sha256(config.identity_dict()),
        replica_ids=config.replica_ids,
        steps=tuple(plan_steps),
    )


def validate_restart_chain(plan: ThermalPlan) -> None:
    """Validate one velocity initialization and a strict predecessor chain per replica."""

    if not plan.replica_ids:
        raise ThermalPlanError("plan must contain at least one replica_id")
    if len(set(plan.replica_ids)) != len(plan.replica_ids):
        raise ThermalPlanError("plan replica_id values must be unique")

    seen_outputs: set[str] = set()
    for replica_id in plan.replica_ids:
        replica_steps = sorted(
            (step for step in plan.steps if step.replica_id == replica_id),
            key=lambda step: step.sequence_index,
        )
        if not replica_steps:
            raise ThermalPlanError(f"replica {replica_id} has no planned steps")
        expected_indices = list(range(len(replica_steps)))
        actual_indices = [step.sequence_index for step in replica_steps]
        if actual_indices != expected_indices:
            raise ThermalPlanError(
                f"replica {replica_id} sequence indices are not contiguous: {actual_indices}"
            )

        for index, step in enumerate(replica_steps):
            if step.output_restart in seen_outputs:
                raise ThermalPlanError(
                    f"output restart is not unique: {step.output_restart}"
                )
            seen_outputs.add(step.output_restart)
            if index == 0:
                if not step.initialize_velocities or step.predecessor_restart is not None:
                    raise ThermalPlanError(
                        f"replica {replica_id} must initialize velocities exactly at step 0"
                    )
            else:
                predecessor = replica_steps[index - 1].output_restart
                if step.initialize_velocities:
                    raise ThermalPlanError(
                        f"replica {replica_id} initializes velocities more than once"
                    )
                if step.predecessor_restart != predecessor:
                    raise ThermalPlanError(
                        f"replica {replica_id} step {index} must consume {predecessor}"
                    )

    unknown_replicas = sorted(
        {step.replica_id for step in plan.steps}.difference(plan.replica_ids)
    )
    if unknown_replicas:
        raise ThermalPlanError(f"steps reference undeclared replicas: {unknown_replicas}")
