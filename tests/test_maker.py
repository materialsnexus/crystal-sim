import pytest

from crystal.schemas.base import BaseMaker


def test_maker() -> None:
    with pytest.raises(ValueError, match="simulation settings cannot be None"):
        _ = BaseMaker()
