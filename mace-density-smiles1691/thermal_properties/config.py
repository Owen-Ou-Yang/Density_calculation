"""Validated configuration models for the additive thermal workflow.

This module intentionally has no dependency on the modulus workflow.  A thermal
protocol is an ordered collection of explicit state points and replica IDs.  It
never edits a schedule to make it valid.  Density-only and optional
cooling-history density targets are distinct, explicitly labelled roles.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


class ThermalConfigError(ValueError):
    """Raised when a thermal protocol is incomplete or internally inconsistent."""


class StageKind(str, Enum):
    """The role of a state point in the protocol."""

    EQUILIBRATION = "equilibration"
    PRODUCTION = "production"


_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_FORBIDDEN_MELTING_KEYS = {
    "tm",
    "tm_k",
    "melting_temperature",
    "melting_temperature_k",
}


def _as_decimal(value: object, field_name: str) -> Decimal:
    if isinstance(value, bool):
        raise ThermalConfigError(f"{field_name} must be numeric, not bool")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as exc:
        raise ThermalConfigError(f"{field_name} must be a finite decimal number") from exc
    if not result.is_finite():
        raise ThermalConfigError(f"{field_name} must be finite")
    return result


def _positive_decimal(value: object, field_name: str) -> Decimal:
    result = _as_decimal(value, field_name)
    if result <= 0:
        raise ThermalConfigError(f"{field_name} must be > 0")
    return result


def _validate_identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ThermalConfigError(f"{field_name} must be a string")
    stripped = value.strip()
    if not _IDENTIFIER_RE.fullmatch(stripped):
        raise ThermalConfigError(
            f"{field_name} must match {_IDENTIFIER_RE.pattern!r}; got {value!r}"
        )
    return stripped


def _reject_melting_keys(mapping: Mapping[str, Any], context: str) -> None:
    present = sorted(_FORBIDDEN_MELTING_KEYS.intersection(key.lower() for key in mapping))
    if present:
        raise ThermalConfigError(
            f"{context} contains unsupported melting-temperature fields: {present}"
        )


def ps_to_steps(duration_ps: object, timestep_ps: object) -> int:
    """Convert picoseconds to an exact positive integer number of MD steps.

    Decimal values are constructed from their string representations.  This
    avoids binary floating-point surprises while still rejecting a duration
    that is not exactly divisible by the timestep.  Rounding is never applied.
    """

    duration = _positive_decimal(duration_ps, "duration_ps")
    timestep = _positive_decimal(timestep_ps, "timestep_ps")
    ratio = duration / timestep
    integral = ratio.to_integral_value()
    if ratio != integral:
        raise ThermalConfigError(
            "duration_ps must be exactly divisible by timestep_ps; "
            f"got duration_ps={duration} and timestep_ps={timestep}"
        )
    steps = int(integral)
    if steps <= 0:
        raise ThermalConfigError("duration_ps must yield at least one MD step")
    return steps


@dataclass(frozen=True)
class StatePointConfig:
    """One explicitly declared temperature/pressure segment.

    Start and end values are both stored so a ramp that merely crosses 300 K
    cannot be mistaken for the required constant production plateau.
    """

    state_id: str
    temperature_start_k: float
    temperature_end_k: float
    pressure_start_bar: float
    pressure_end_bar: float
    duration_ps: float
    stage_kind: StageKind
    use_for_tg_fit: bool
    use_for_density: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_id", _validate_identifier(self.state_id, "state_id"))

        temperature_start = _positive_decimal(
            self.temperature_start_k, "temperature_start_k"
        )
        temperature_end = _positive_decimal(self.temperature_end_k, "temperature_end_k")
        pressure_start = _as_decimal(self.pressure_start_bar, "pressure_start_bar")
        pressure_end = _as_decimal(self.pressure_end_bar, "pressure_end_bar")
        duration = _positive_decimal(self.duration_ps, "duration_ps")

        # Keep the exact decimal values for divisibility and setpoint checks.
        # Public numeric fields remain floats for convenient JSON/dataframe use.
        object.__setattr__(self, "_temperature_start_decimal", temperature_start)
        object.__setattr__(self, "_temperature_end_decimal", temperature_end)
        object.__setattr__(self, "_pressure_start_decimal", pressure_start)
        object.__setattr__(self, "_pressure_end_decimal", pressure_end)
        object.__setattr__(self, "_duration_ps_decimal", duration)

        object.__setattr__(self, "temperature_start_k", float(temperature_start))
        object.__setattr__(self, "temperature_end_k", float(temperature_end))
        object.__setattr__(self, "pressure_start_bar", float(pressure_start))
        object.__setattr__(self, "pressure_end_bar", float(pressure_end))
        object.__setattr__(self, "duration_ps", float(duration))

        try:
            stage_kind = StageKind(self.stage_kind)
        except ValueError as exc:
            allowed = ", ".join(member.value for member in StageKind)
            raise ThermalConfigError(f"stage_kind must be one of: {allowed}") from exc
        object.__setattr__(self, "stage_kind", stage_kind)

        for field_name in ("use_for_tg_fit", "use_for_density"):
            if type(getattr(self, field_name)) is not bool:
                raise ThermalConfigError(f"{field_name} must be an explicit boolean")

    @classmethod
    def constant(
        cls,
        state_id: str,
        temperature_k: object,
        pressure_bar: object,
        duration_ps: object,
        *,
        stage_kind: StageKind | str,
        use_for_tg_fit: bool,
        use_for_density: bool,
    ) -> "StatePointConfig":
        """Construct a constant-temperature, constant-pressure state point."""

        return cls(
            state_id=state_id,
            temperature_start_k=temperature_k,
            temperature_end_k=temperature_k,
            pressure_start_bar=pressure_bar,
            pressure_end_bar=pressure_bar,
            duration_ps=duration_ps,
            stage_kind=stage_kind,
            use_for_tg_fit=use_for_tg_fit,
            use_for_density=use_for_density,
        )

    @property
    def is_constant_temperature(self) -> bool:
        return self._temperature_start_decimal == self._temperature_end_decimal

    @property
    def is_constant_pressure(self) -> bool:
        return self._pressure_start_decimal == self._pressure_end_decimal

    def steps(self, timestep_ps: object) -> int:
        return ps_to_steps(self._duration_ps_decimal, timestep_ps)

    def is_exact_production_plateau(
        self, temperature_k: float, pressure_bar: float
    ) -> bool:
        target_temperature = _as_decimal(temperature_k, "temperature_k")
        target_pressure = _as_decimal(pressure_bar, "pressure_bar")
        return (
            self.stage_kind is StageKind.PRODUCTION
            and self.is_constant_temperature
            and self.is_constant_pressure
            and self._temperature_start_decimal == target_temperature
            and self._pressure_start_decimal == target_pressure
        )

    def identity_dict(self) -> dict[str, object]:
        """Return an exact-decimal representation for protocol fingerprinting."""

        return {
            "state_id": self.state_id,
            "temperature_start_k": str(self._temperature_start_decimal),
            "temperature_end_k": str(self._temperature_end_decimal),
            "pressure_start_bar": str(self._pressure_start_decimal),
            "pressure_end_bar": str(self._pressure_end_decimal),
            "duration_ps": str(self._duration_ps_decimal),
            "stage_kind": self.stage_kind.value,
            "use_for_tg_fit": self.use_for_tg_fit,
            "use_for_density": self.use_for_density,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "state_id": self.state_id,
            "temperature_start_k": self.temperature_start_k,
            "temperature_end_k": self.temperature_end_k,
            "pressure_start_bar": self.pressure_start_bar,
            "pressure_end_bar": self.pressure_end_bar,
            "duration_ps": self.duration_ps,
            "stage_kind": self.stage_kind.value,
            "use_for_tg_fit": self.use_for_tg_fit,
            "use_for_density": self.use_for_density,
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "StatePointConfig":
        _reject_melting_keys(mapping, "state point")
        values = dict(mapping)
        if "temperature_k" in values:
            values.setdefault("temperature_start_k", values["temperature_k"])
            values.setdefault("temperature_end_k", values["temperature_k"])
        if "pressure_bar" in values:
            values.setdefault("pressure_start_bar", values["pressure_bar"])
            values.setdefault("pressure_end_bar", values["pressure_bar"])

        required = {
            "state_id",
            "temperature_start_k",
            "temperature_end_k",
            "pressure_start_bar",
            "pressure_end_bar",
            "duration_ps",
            "stage_kind",
            "use_for_tg_fit",
            "use_for_density",
        }
        missing = sorted(required.difference(values))
        if missing:
            raise ThermalConfigError(f"state point is missing required fields: {missing}")
        return cls(**{key: values[key] for key in required})


@dataclass(frozen=True)
class ThermalProtocolConfig:
    """A complete, immutable Tg/density protocol configuration."""

    timestep_ps: float
    state_points: tuple[StatePointConfig, ...]
    replica_ids: tuple[str, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        timestep = _positive_decimal(self.timestep_ps, "timestep_ps")
        object.__setattr__(self, "_timestep_ps_decimal", timestep)
        object.__setattr__(self, "timestep_ps", float(timestep))

        if self.schema_version != 1:
            raise ThermalConfigError(
                f"unsupported thermal protocol schema_version={self.schema_version}"
            )

        state_points = tuple(self.state_points)
        if not state_points:
            raise ThermalConfigError("state_points must not be empty")
        if not all(isinstance(state, StatePointConfig) for state in state_points):
            raise ThermalConfigError("every state_points entry must be StatePointConfig")
        state_ids = [state.state_id for state in state_points]
        if len(set(state_ids)) != len(state_ids):
            raise ThermalConfigError("state_id values must be unique")
        object.__setattr__(self, "state_points", state_points)

        replica_ids = tuple(
            _validate_identifier(replica_id, "replica_id") for replica_id in self.replica_ids
        )
        if not replica_ids:
            raise ThermalConfigError("replica_ids must not be empty")
        if len(set(replica_ids)) != len(replica_ids):
            raise ThermalConfigError("replica_id values must be unique")
        object.__setattr__(self, "replica_ids", replica_ids)

        # Validate every conversion now, not after a calculation has started.
        for state in state_points:
            state.steps(timestep)

        # A Tg-only protocol may intentionally have no density target.  The
        # high-level density planner still requires exactly one explicit
        # density plateau; keeping the generic protocol neutral lets the two
        # property branches share MD machinery without conflating their roles.

    @property
    def density_production_plateaus(self) -> tuple[StatePointConfig, ...]:
        return tuple(
            state
            for state in self.state_points
            if state.use_for_density
            and state.stage_kind is StageKind.PRODUCTION
            and state.is_constant_temperature
            and state.is_constant_pressure
        )

    @property
    def production_plateaus_300k_1bar(self) -> tuple[StatePointConfig, ...]:
        return tuple(
            state
            for state in self.density_production_plateaus
            if state.is_exact_production_plateau(300.0, 1.0)
        )

    def identity_dict(self) -> dict[str, object]:
        """Return the exact representation used to identify this protocol."""

        return {
            "schema_version": self.schema_version,
            "timestep_ps": str(self._timestep_ps_decimal),
            "replica_ids": list(self.replica_ids),
            "state_points": [state.identity_dict() for state in self.state_points],
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "timestep_ps": self.timestep_ps,
            "replica_ids": list(self.replica_ids),
            "state_points": [state.to_dict() for state in self.state_points],
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "ThermalProtocolConfig":
        _reject_melting_keys(mapping, "thermal protocol")
        required = {"timestep_ps", "state_points", "replica_ids"}
        missing = sorted(required.difference(mapping))
        if missing:
            raise ThermalConfigError(f"thermal protocol is missing required fields: {missing}")

        raw_states = mapping["state_points"]
        if not isinstance(raw_states, Sequence) or isinstance(raw_states, (str, bytes)):
            raise ThermalConfigError("state_points must be a sequence")
        raw_replicas = mapping["replica_ids"]
        if not isinstance(raw_replicas, Sequence) or isinstance(raw_replicas, (str, bytes)):
            raise ThermalConfigError("replica_ids must be a sequence")

        return cls(
            timestep_ps=mapping["timestep_ps"],
            state_points=tuple(StatePointConfig.from_mapping(item) for item in raw_states),
            replica_ids=tuple(raw_replicas),
            schema_version=int(mapping.get("schema_version", 1)),
        )


def load_protocol_config(path: str | Path) -> ThermalProtocolConfig:
    """Load a JSON protocol without applying defaults or schedule mutations."""

    config_path = Path(path)
    if config_path.suffix.lower() != ".json":
        raise ThermalConfigError("thermal protocol files must be JSON")
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ThermalConfigError(f"failed to read thermal protocol: {config_path}") from exc
    if not isinstance(payload, Mapping):
        raise ThermalConfigError("thermal protocol JSON root must be an object")
    return ThermalProtocolConfig.from_mapping(payload)
