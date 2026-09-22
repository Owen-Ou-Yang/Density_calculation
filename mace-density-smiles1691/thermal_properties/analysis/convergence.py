"""Autocorrelation- and block-aware state-point statistics."""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Any, Mapping

import numpy as np
import pandas as pd


def integrated_autocorrelation_time(
    values: np.ndarray, *, max_lag_fraction: float = 0.25
) -> tuple[float, float]:
    """Return initial-positive-sequence tau_int and effective sample size."""

    series = np.asarray(values, dtype=float)
    n = int(series.size)
    if n < 2:
        return 1.0, float(n)
    centered = series - float(np.mean(series))
    variance = float(np.dot(centered, centered) / n)
    if not math.isfinite(variance) or variance <= 0.0:
        return 1.0, float(n)

    fft_size = 1 << (2 * n - 1).bit_length()
    transformed = np.fft.rfft(centered, fft_size)
    autocovariance = np.fft.irfft(transformed * np.conjugate(transformed), fft_size)[:n]
    autocovariance /= np.arange(n, 0, -1, dtype=float)
    autocorrelation = autocovariance / autocovariance[0]
    max_lag = max(1, min(n - 1, int(math.floor(n * max_lag_fraction))))
    positive_sum = 0.0
    for lag in range(1, max_lag + 1):
        rho = float(autocorrelation[lag])
        if not math.isfinite(rho) or rho <= 0.0:
            break
        positive_sum += rho
    tau_int = max(1.0, 1.0 + 2.0 * positive_sum)
    effective_n = min(float(n), max(1.0, n / tau_int))
    return tau_int, effective_n


def observable_statistics(
    values: np.ndarray,
    *,
    confidence_level: float,
    block_count: int,
    min_samples_per_block: int,
    max_lag_fraction: float,
) -> dict[str, Any]:
    series = np.asarray(values, dtype=float)
    n = int(series.size)
    mean = float(np.mean(series))
    standard_deviation = float(np.std(series, ddof=1)) if n > 1 else 0.0
    tau_int, effective_n = integrated_autocorrelation_time(
        series, max_lag_fraction=max_lag_fraction
    )
    autocorrelation_se = (
        standard_deviation / math.sqrt(effective_n) if effective_n > 0 else math.inf
    )

    usable_blocks = min(block_count, n // max(1, min_samples_per_block))
    block_means: list[float] = []
    if usable_blocks >= 2:
        block_means = [
            float(np.mean(block))
            for block in np.array_split(series, usable_blocks)
            if block.size
        ]
        block_se = float(np.std(block_means, ddof=1) / math.sqrt(len(block_means)))
    else:
        block_se = math.inf
    standard_error = max(autocorrelation_se, block_se)
    alpha = 1.0 - confidence_level
    z_value = NormalDist().inv_cdf(1.0 - alpha / 2.0)
    ci_half_width = z_value * standard_error
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "autocorrelation_standard_error": autocorrelation_se,
        "block_standard_error": block_se,
        "integrated_autocorrelation_time_samples": tau_int,
        "effective_sample_count": effective_n,
        "confidence_level": confidence_level,
        "confidence_interval": [mean - ci_half_width, mean + ci_half_width],
        "block_count": len(block_means),
        "block_means": block_means,
        "block_mean_spread": (
            max(block_means) - min(block_means) if block_means else math.inf
        ),
    }


def assess_density_convergence(
    production: pd.DataFrame,
    *,
    target_temperature_K: float,
    target_pressure_bar: float,
    policy: Mapping[str, Any],
    expected_atom_count: int | None = None,
) -> dict[str, Any]:
    """Compute density/volume statistics and all deterministic QC gates."""

    checks: list[dict[str, Any]] = []

    def record(name: str, passed: bool, actual: Any, criterion: Any) -> None:
        checks.append(
            {
                "name": name,
                "status": "PASS" if passed else "FAIL",
                "actual": actual,
                "criterion": criterion,
            }
        )

    required_columns = [
        "step",
        "time_ps",
        "temp_K",
        "press_bar",
        "volume_A3",
        "density_g_cm3",
        "specific_volume_cm3_g",
    ]
    structural_columns = [
        "pe_eV",
        "etotal_eV",
        "enthalpy_eV",
        "fmax_eV_A",
        "lx_A",
        "ly_A",
        "lz_A",
        "atom_count",
    ]
    require_structural = bool(policy.get("require_structural_columns", False))
    missing_structural = [
        column for column in structural_columns if column not in production.columns
    ]
    if require_structural:
        record(
            "structural_diagnostics_present",
            not missing_structural,
            missing_structural,
            [],
        )
    n = int(len(production))
    record(
        "minimum_production_samples",
        n >= int(policy["min_production_samples"]),
        n,
        {">=": int(policy["min_production_samples"])},
    )
    finite_columns = required_columns + (
        structural_columns if require_structural and not missing_structural else []
    )
    finite = n > 0 and all(
        np.isfinite(production[column].to_numpy(dtype=float)).all()
        for column in finite_columns
    )
    record("all_required_values_finite", finite, finite, True)
    if not finite or (require_structural and missing_structural):
        return {
            "status": "FAIL",
            "checks": checks,
            "sample_count": n,
            "failure_reason": (
                "STRUCTURAL_DIAGNOSTICS_MISSING"
                if missing_structural
                else "NONFINITE_SAMPLES"
            ),
        }

    steps = production["step"].to_numpy(dtype=float)
    times = production["time_ps"].to_numpy(dtype=float)
    temperatures = production["temp_K"].to_numpy(dtype=float)
    pressures = production["press_bar"].to_numpy(dtype=float)
    volumes = production["volume_A3"].to_numpy(dtype=float)
    densities = production["density_g_cm3"].to_numpy(dtype=float)
    specific_volumes = production["specific_volume_cm3_g"].to_numpy(dtype=float)

    positive = bool(np.all(volumes > 0.0) and np.all(densities > 0.0))
    record("positive_volume_and_density", positive, positive, True)
    if require_structural:
        expected_atom_count_valid = (
            isinstance(expected_atom_count, int)
            and not isinstance(expected_atom_count, bool)
            and expected_atom_count > 0
        )
        record(
            "snapshot_contract_atom_count_present",
            expected_atom_count_valid,
            expected_atom_count,
            "positive authenticated integer",
        )
        structural_policy = policy.get("structural_integrity")
        if not isinstance(structural_policy, Mapping):
            record(
                "structural_policy_present",
                False,
                structural_policy,
                "object",
            )
        else:
            forces = production["fmax_eV_A"].to_numpy(dtype=float)
            lengths = production[["lx_A", "ly_A", "lz_A"]].to_numpy(
                dtype=float
            )
            atom_counts = production["atom_count"].to_numpy(dtype=float)
            integer_atom_counts = bool(
                np.all(atom_counts > 0.0)
                and np.all(atom_counts == np.floor(atom_counts))
            )
            unique_atom_counts = sorted(set(atom_counts.tolist()))
            record(
                "constant_positive_integer_atom_count",
                integer_atom_counts and len(unique_atom_counts) == 1,
                unique_atom_counts,
                "one positive integer value",
            )
            if expected_atom_count_valid:
                record(
                    "atom_count_matches_snapshot_contract",
                    integer_atom_counts
                    and len(unique_atom_counts) == 1
                    and int(unique_atom_counts[0]) == int(expected_atom_count),
                    unique_atom_counts[0] if len(unique_atom_counts) == 1 else unique_atom_counts,
                    int(expected_atom_count),
                )
            minimum_length = float(np.min(lengths))
            maximum_length = float(np.max(lengths))
            minimum_allowed_length = float(structural_policy["min_box_length_A"])
            record(
                "minimum_box_length",
                minimum_length >= minimum_allowed_length,
                minimum_length,
                {">=": minimum_allowed_length},
            )
            per_frame_minimum = np.min(lengths, axis=1)
            per_frame_maximum = np.max(lengths, axis=1)
            aspect_ratio = float(np.max(per_frame_maximum / per_frame_minimum))
            maximum_aspect = float(structural_policy["max_box_aspect_ratio"])
            record(
                "box_aspect_ratio",
                math.isfinite(aspect_ratio) and aspect_ratio <= maximum_aspect,
                aspect_ratio,
                {"<=": maximum_aspect},
            )
            maximum_force = float(np.max(forces))
            force_limit = float(structural_policy["max_force_eV_A"])
            record(
                "maximum_atomic_force",
                maximum_force <= force_limit,
                maximum_force,
                {"<=": force_limit},
            )
            minimum_density = float(structural_policy["min_density_g_cm3"])
            maximum_density = float(structural_policy["max_density_g_cm3"])
            observed_density_range = [
                float(np.min(densities)),
                float(np.max(densities)),
            ]
            record(
                "physical_density_range",
                observed_density_range[0] >= minimum_density
                and observed_density_range[1] <= maximum_density,
                observed_density_range,
                {"min": minimum_density, "max": maximum_density},
            )
    increasing_steps = n < 2 or bool(np.all(np.diff(steps) > 0.0))
    increasing_times = n < 2 or bool(np.all(np.diff(times) > 0.0))
    record("strictly_increasing_steps", increasing_steps, increasing_steps, True)
    record("strictly_increasing_times", increasing_times, increasing_times, True)

    uniform_stride = True
    if n > 2:
        differences = np.diff(steps)
        uniform_stride = bool(np.allclose(differences, differences[0], rtol=0, atol=0))
    if bool(policy.get("require_uniform_step_stride", True)):
        record("uniform_step_stride", uniform_stride, uniform_stride, True)

    duration_ps = float(times[-1] - times[0]) if n > 1 else 0.0
    record(
        "minimum_production_time_ps",
        duration_ps >= float(policy["min_production_time_ps"]),
        duration_ps,
        {">=": float(policy["min_production_time_ps"])},
    )

    confidence_level = float(policy.get("confidence_level", 0.95))
    block_count = int(policy["block_count"])
    min_per_block = int(policy["min_samples_per_block"])
    max_lag_fraction = float(policy.get("autocorrelation_max_lag_fraction", 0.25))
    density_stats = observable_statistics(
        densities,
        confidence_level=confidence_level,
        block_count=block_count,
        min_samples_per_block=min_per_block,
        max_lag_fraction=max_lag_fraction,
    )
    volume_stats = observable_statistics(
        volumes,
        confidence_level=confidence_level,
        block_count=block_count,
        min_samples_per_block=min_per_block,
        max_lag_fraction=max_lag_fraction,
    )
    specific_volume_stats = observable_statistics(
        specific_volumes,
        confidence_level=confidence_level,
        block_count=block_count,
        min_samples_per_block=min_per_block,
        max_lag_fraction=max_lag_fraction,
    )
    record(
        "minimum_effective_samples",
        density_stats["effective_sample_count"]
        >= float(policy["min_effective_samples"]),
        density_stats["effective_sample_count"],
        {">=": float(policy["min_effective_samples"])},
    )
    record(
        "minimum_complete_blocks",
        density_stats["block_count"] >= block_count,
        density_stats["block_count"],
        {">=": block_count},
    )

    mean_temperature = float(np.mean(temperatures))
    mean_pressure = float(np.mean(pressures))
    temperature_error = abs(mean_temperature - target_temperature_K)
    pressure_error = abs(mean_pressure - target_pressure_bar)
    record(
        "mean_temperature_control",
        temperature_error <= float(policy["max_abs_mean_temp_error_K"]),
        temperature_error,
        {"<=": float(policy["max_abs_mean_temp_error_K"])},
    )
    record(
        "mean_pressure_control",
        pressure_error <= float(policy["max_abs_mean_pressure_error_bar"]),
        pressure_error,
        {"<=": float(policy["max_abs_mean_pressure_error_bar"])},
    )

    if n > 1 and duration_ps > 0.0:
        centered_time = times - float(np.mean(times))
        denominator = float(np.dot(centered_time, centered_time))
        density_slope = (
            float(np.dot(centered_time, densities - np.mean(densities)) / denominator)
            if denominator > 0.0
            else math.inf
        )
    else:
        density_slope = math.inf
    record(
        "density_drift",
        abs(density_slope)
        <= float(policy["max_abs_density_drift_g_cm3_per_ps"]),
        density_slope,
        {
            "abs<=": float(policy["max_abs_density_drift_g_cm3_per_ps"])
        },
    )
    record(
        "block_mean_spread",
        density_stats["block_mean_spread"]
        <= float(policy["max_block_mean_spread_g_cm3"]),
        density_stats["block_mean_spread"],
        {"<=": float(policy["max_block_mean_spread_g_cm3"])},
    )

    if "_recomputed_density_g_cm3" in production:
        stored = production["density_g_cm3"].to_numpy(dtype=float)
        recomputed = production["_recomputed_density_g_cm3"].to_numpy(dtype=float)
        max_difference = float(np.max(np.abs(stored - recomputed)))
        tolerance = float(policy.get("max_stored_recomputed_density_diff", 1e-6))
        record(
            "stored_density_matches_mass_volume",
            max_difference <= tolerance,
            max_difference,
            {"<=": tolerance},
        )
    else:
        max_difference = None

    passed = all(check["status"] == "PASS" for check in checks)
    return {
        "status": "PASS" if passed else "FAIL",
        "checks": checks,
        "sample_count": n,
        "production_time_ps": duration_ps,
        "target_temperature_K": target_temperature_K,
        "observed_mean_temperature_K": mean_temperature,
        "target_pressure_bar": target_pressure_bar,
        "observed_mean_pressure_bar": mean_pressure,
        "density_drift_g_cm3_per_ps": density_slope,
        "stored_recomputed_density_max_abs_diff": max_difference,
        "density": density_stats,
        "volume_A3": volume_stats,
        "specific_volume_cm3_g": specific_volume_stats,
    }
