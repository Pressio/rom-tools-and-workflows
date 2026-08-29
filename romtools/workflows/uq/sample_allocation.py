"""Pilot-based sample allocation for two-level MFMC."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from romtools.workflows.uq.statistics import compute_paired_statistics


@dataclass
class PilotSampleAllocation:
    """Integer sample allocation selected from paired pilot data."""

    number_high_fidelity_samples: int
    number_low_fidelity_samples: int
    control_variate_coefficients: np.ndarray
    pilot_correlations: np.ndarray
    estimated_variance: float
    high_fidelity_equivalent_cost: float
    use_multifidelity_estimator: bool


def allocate_samples_from_pilot(
    high_fidelity_qois: np.ndarray,
    low_fidelity_qois: np.ndarray,
    high_fidelity_equivalent_budget: float,
    low_to_high_fidelity_cost_ratio: float,
    allocation_qoi_index: int = 0,
) -> PilotSampleAllocation:
    """Select integer high- and low-fidelity counts from paired pilot data.

    The search minimizes the standard two-level MFMC variance approximation
    for the component selected by ``allocation_qoi_index`` subject to
    ``N_H + cost_ratio * N_L <= budget`` and ``N_L >= N_H``. Pilot evaluations
    are included in the returned counts and cost. The result switches to an
    HF-only estimator when spending the remaining budget on high-fidelity
    samples has no larger predicted variance.

    Args:
        high_fidelity_qois: Paired pilot QoIs with shape ``(N_p, N_qoi)`` or
            ``(N_p,)``.
        low_fidelity_qois: Low-fidelity pilot QoIs with the same shape and
            ordering as ``high_fidelity_qois``.
        high_fidelity_equivalent_budget: Total budget in high-fidelity cost
            units.
        low_to_high_fidelity_cost_ratio: Positive cost ratio ``c_L / c_H``.
        allocation_qoi_index: QoI component used by the allocation objective.

    Returns:
        PilotSampleAllocation: Selected counts, frozen componentwise
        coefficients, pilot correlations, predicted variance and cost, and an
        indicator selecting MFMC or the HF-only fallback.

    Raises:
        ValueError: If the pilot arrays are incompatible, contain fewer than
            two samples, the selected component is invalid, or the pilot does
            not fit in the budget.
    """
    high = np.asarray(high_fidelity_qois, dtype=float)
    low = np.asarray(low_fidelity_qois, dtype=float)
    if high.ndim == 1:
        high = high[:, None]
    if low.ndim == 1:
        low = low[:, None]
    if high.shape != low.shape:
        raise ValueError("pilot high- and low-fidelity QoIs must have the same shape")
    if high.ndim != 2 or high.shape[0] < 2:
        raise ValueError("the pilot must contain at least two paired samples")
    if not np.isfinite(high_fidelity_equivalent_budget) or high_fidelity_equivalent_budget <= 0:
        raise ValueError("high_fidelity_equivalent_budget must be positive")
    if not np.isfinite(low_to_high_fidelity_cost_ratio) or low_to_high_fidelity_cost_ratio <= 0:
        raise ValueError("low_to_high_fidelity_cost_ratio must be positive")
    if (
        isinstance(allocation_qoi_index, bool)
        or not isinstance(allocation_qoi_index, Integral)
        or allocation_qoi_index < 0
        or allocation_qoi_index >= high.shape[1]
    ):
        raise ValueError("allocation_qoi_index is outside the QoI range")

    pilot_count = high.shape[0]
    pilot_cost = pilot_count * (1.0 + low_to_high_fidelity_cost_ratio)
    tolerance = np.finfo(float).eps * max(high_fidelity_equivalent_budget, 1.0) * 100.0
    if pilot_cost > high_fidelity_equivalent_budget + tolerance:
        raise ValueError("the paired pilot exceeds the high-fidelity-equivalent budget")

    high_variance, _, _, coefficients, correlations = compute_paired_statistics(high, low)
    variance = float(high_variance[allocation_qoi_index])
    rho_squared = float(correlations[allocation_qoi_index] ** 2)
    rho_squared = min(max(rho_squared, 0.0), 1.0)
    budget = float(high_fidelity_equivalent_budget)
    cost_ratio = float(low_to_high_fidelity_cost_ratio)

    # The MC alternative retains the already-paid low-fidelity pilot but does
    # not purchase more low-fidelity evaluations.
    mc_high_count = int(np.floor(budget - cost_ratio * pilot_count + tolerance))
    if mc_high_count < pilot_count:
        raise ValueError("budget does not permit completion of the pilot")
    mc_variance = variance / mc_high_count
    mc_cost = mc_high_count + cost_ratio * pilot_count

    best = None
    maximum_high_count = int(np.floor(budget / (1.0 + cost_ratio) + tolerance))
    for number_high in range(pilot_count, maximum_high_count + 1):
        number_low = int(np.floor((budget - number_high) / cost_ratio + tolerance))
        if number_low < number_high:
            continue
        estimated_variance = variance * (
            (1.0 - rho_squared) / number_high + rho_squared / number_low
        )
        equivalent_cost = number_high + cost_ratio * number_low
        candidate = (
            estimated_variance,
            equivalent_cost,
            -number_high,
            number_high,
            number_low,
        )
        if best is None or candidate[:3] < best[:3]:
            best = candidate

    variance_tolerance = (
        np.finfo(float).eps
        * max(abs(mc_variance), 0.0 if best is None else abs(best[0]), np.finfo(float).tiny)
        * 100.0
    )
    if best is None or mc_variance <= best[0] + variance_tolerance:
        return PilotSampleAllocation(
            number_high_fidelity_samples=mc_high_count,
            number_low_fidelity_samples=pilot_count,
            control_variate_coefficients=np.zeros_like(coefficients),
            pilot_correlations=correlations,
            estimated_variance=mc_variance,
            high_fidelity_equivalent_cost=mc_cost,
            use_multifidelity_estimator=False,
        )

    return PilotSampleAllocation(
        number_high_fidelity_samples=best[3],
        number_low_fidelity_samples=best[4],
        control_variate_coefficients=coefficients,
        pilot_correlations=correlations,
        estimated_variance=best[0],
        high_fidelity_equivalent_cost=best[1],
        use_multifidelity_estimator=True,
    )
