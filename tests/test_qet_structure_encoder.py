from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crystal.schemas.mlip_model_ref import MlipModelRef, QetModelRegistry
from crystal.structure.generators.sampling.qet_structure_encoder import (
    STAT_TYPES,
    QetStructureEncoder,
    QetStructureEncoderMode,
)
from tests.helpers.constants.structure import TEST_BCC_FE_STRUCTURE, TEST_SI_STRUCTURE


@pytest.mark.parametrize(
    "mode", [QetStructureEncoderMode.mean, QetStructureEncoderMode.site_stats]
)
def test_qet_structure_encoder_transform_one(mode: QetStructureEncoderMode) -> None:
    encoder = QetStructureEncoder(
        model=MlipModelRef(registry_model=QetModelRegistry.foundational_matpes_pbe),
        mode=mode,
    )
    feature_vector_length = (
        encoder.describer_model.units + 2
    )  # +2 for atomic charge and potential
    output_a = encoder.transform_one(TEST_SI_STRUCTURE)
    output_b = encoder.transform_one(TEST_BCC_FE_STRUCTURE)

    if mode == QetStructureEncoderMode.mean:
        assert isinstance(output_a, np.ndarray)
        assert isinstance(output_b, np.ndarray)
        assert output_a.shape == (feature_vector_length,)
        assert output_a.dtype == np.float32
        assert not np.allclose(output_a, output_b)
    else:
        assert isinstance(output_a, pd.DataFrame)
        assert isinstance(output_b, pd.DataFrame)
        assert output_a.shape[0] == 1
        assert output_a.shape[1] == feature_vector_length * len(STAT_TYPES)
        values_a = output_a.to_numpy(dtype=float)
        values_b = output_b.to_numpy(dtype=float)
        assert not np.allclose(values_a, values_b)
