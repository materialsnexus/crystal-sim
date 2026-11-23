"""Helpers for property schemas."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from crystal.schemas.properties import ForcesLike, VoigtStressesLike, VoigtStressLike


def get_force_norms(forces: ForcesLike) -> NDArray[np.float64]:
    """Return per-atom force-vector norms."""
    return np.asarray(np.linalg.norm(forces, axis=1), dtype=np.float64)


def get_maximum_force_norm(forces: ForcesLike) -> float:
    """Return the maximum per-atom force-vector norm."""
    return float(get_force_norms(forces).max(initial=0.0))


def get_stress_norms(stresses: VoigtStressesLike) -> NDArray[np.float64]:
    """Return L2 norms for Voigt-6 stresses."""
    stress_array = np.asarray(stresses, dtype=np.float64)
    return np.asarray(np.linalg.norm(stress_array, axis=1), dtype=np.float64)


def get_stress_norm(stress: VoigtStressLike) -> float:
    """Return the L2 norm of one Voigt-6 stress."""
    return float(np.linalg.norm(np.asarray(stress, dtype=np.float64)))
