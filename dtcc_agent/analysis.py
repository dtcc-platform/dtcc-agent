"""Analysis utilities: extract summary statistics from simulation results.

Both urban_heat_simulation and air_quality_field return dolfinx.fem.Function
objects. The field values live in `function.x.array` as a flat numpy array.
This module extracts human-readable statistics from those arrays so the LLM
can reason about results without needing to parse binary files.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def summarize_field(values: np.ndarray, field_name: str = "field") -> dict[str, Any]:
    """Compute summary statistics for a scalar field.

    Parameters
    ----------
    values : np.ndarray
        1-D array of field values (e.g. temperature, concentration).
    field_name : str
        Human-readable name for the field (e.g. "temperature", "NO2").

    Returns
    -------
    dict with keys: field_name, min, max, mean, std, median,
    percentile_5, percentile_95, num_values
    """
    v = np.asarray(values).ravel()
    valid = v[np.isfinite(v)]

    if len(valid) == 0:
        return {
            "field_name": field_name,
            "error": "All values are NaN or inf",
            "num_values": 0,
        }

    return {
        "field_name": field_name,
        "min": round(float(np.min(valid)), 4),
        "max": round(float(np.max(valid)), 4),
        "mean": round(float(np.mean(valid)), 4),
        "std": round(float(np.std(valid)), 4),
        "median": round(float(np.median(valid)), 4),
        "percentile_5": round(float(np.percentile(valid, 5)), 4),
        "percentile_95": round(float(np.percentile(valid, 95)), 4),
        "num_values": int(len(valid)),
    }


def compare_fields(
    values_a: np.ndarray,
    values_b: np.ndarray,
    label_a: str = "scenario_a",
    label_b: str = "scenario_b",
    field_name: str = "field",
) -> dict[str, Any]:
    """Compare two scalar fields and compute difference statistics.

    The fields must be defined on the same mesh (same number of DOFs).

    Parameters
    ----------
    values_a, values_b : np.ndarray
        Field values for each scenario.
    label_a, label_b : str
        Human-readable labels for each scenario.
    field_name : str
        Name of the field being compared.

    Returns
    -------
    dict with per-scenario summaries and a difference summary.
    """
    a = np.asarray(values_a).ravel()
    b = np.asarray(values_b).ravel()

    if len(a) != len(b):
        return {
            "error": (
                f"Field sizes differ: {label_a} has {len(a)} values, "
                f"{label_b} has {len(b)} values. "
                "Fields must be on the same mesh to compare."
            )
        }

    diff = b - a  # positive means B is higher

    return {
        label_a: summarize_field(a, field_name),
        label_b: summarize_field(b, field_name),
        "difference": {
            "description": f"{label_b} minus {label_a}",
            **summarize_field(diff, f"{field_name}_difference"),
        },
    }
