"""QET-based structure encoder for DIRECT sampling."""

from __future__ import annotations

from enum import StrEnum, auto
from typing import Any

import numpy as np
import pandas as pd
import torch
from maml.base import BaseDescriber, describer_type
from maml.utils import get_full_stats_and_funcs
from numpy.typing import NDArray
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.schemas.mlip import MATGL_MISSING_ERROR_MESSAGE
from crystal.schemas.mlip_model_ref import MlipModelRef


class QetStructureEncoderMode(StrEnum):  # noqa: D101
    mean = auto()
    site_stats = auto()


STAT_TYPES: list[str] = ["min", "max", "range", "mean", "mean_absolute_error", "mode"]


@describer_type("structure")
class QetStructureEncoder(BaseDescriber):
    """Encode structures into feature vectors using a QET model."""

    def __init__(
        self,
        model: MlipModelRef,
        mode: QetStructureEncoderMode = QetStructureEncoderMode.mean,
        **kwargs: Any,
    ) -> None:
        """Initialize the QET-based structure encoder.

        Args:
            model: QET model used for featurization.
            mode: Aggregation mode for node features. `mean` returns mean-pooled node
                features, and `site_stats` concatenates per-feature statistics.
            **kwargs: Additional keyword arguments passed to `BaseDescriber`.
        """
        try:
            import matgl  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(MATGL_MISSING_ERROR_MESSAGE) from exc

        self.mode = mode
        self.describer_model = matgl.load_model(model.model_path).model
        self.describer_model.return_features = True
        super().__init__(**kwargs)

    def _is_multi_output(self) -> bool:
        """Shim for compatibility with MaML."""
        return False

    @staticmethod
    def _mean(node_features: NDArray[np.float32]) -> NDArray[np.float32]:
        structure_feature = np.mean(node_features, axis=0)
        return np.asarray(structure_feature, dtype=np.float32)

    @staticmethod
    def _site_stats(node_features: NDArray[np.float32]) -> pd.DataFrame:
        full_stats, stats_func = get_full_stats_and_funcs(STAT_TYPES)
        features_transpose = list(zip(*node_features, strict=False))
        column_names = []
        final_features = []
        for i, f in enumerate(features_transpose):
            column_names.extend([f"{i}_{n}" for n in full_stats])
            final_features.extend([func(f) for func in stats_func])
        return pd.DataFrame([final_features], columns=column_names)

    def transform_one(
        self, structure: PMG_Structure
    ) -> NDArray[np.float32] | pd.DataFrame:
        """Transform a structure into features.

        Args:
            structure: Target structure to featurize.

        Returns:
            Encoded structure features as a mean-pooled vector or a statistics table,
            depending on `mode`.
        """
        node_features, _ = self.describer_model.predict_structure(
            structure, total_charge=torch.tensor([0.0], dtype=torch.float32)
        )
        node_features = node_features.detach().cpu().numpy()

        if self.mode == QetStructureEncoderMode.mean:
            return self._mean(node_features)

        assert self.mode == QetStructureEncoderMode.site_stats
        return self._site_stats(node_features)
