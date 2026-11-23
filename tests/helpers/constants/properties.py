import copy
from copy import deepcopy

import numpy as np

from crystal.schemas.properties import RelaxProperties, SinglePointProperties
from tests.helpers.constants.structure import TEST_SI_STRUCTURE

MOCK_SCF_PROPERTIES = SinglePointProperties(
    step=0,
    structure=deepcopy(TEST_SI_STRUCTURE),
    forces=np.array([[0, 0, 0], [0, 0, 0]]),
    stress=np.array([0, 0, 0, 0, 0, 0]),
    energy=0.2,
)
MOCK_MLIP_SINGLE_POINT_PROPERTIES = copy.deepcopy(MOCK_SCF_PROPERTIES)
MOCK_RELAX_PROPERTIES = RelaxProperties(relax_steps=[MOCK_SCF_PROPERTIES])
MOCK_MLIP_RELAX_PROPERTIES = copy.deepcopy(MOCK_RELAX_PROPERTIES)
