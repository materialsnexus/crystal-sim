"""Tests for random-seed fields and stable random-state derivation."""

from pydantic import BaseModel

from crystal.schemas.randomness import RandomSeed, random_seed_field
from crystal.structure.generators.randomness import derive_random_seed


class _RandomSeedFieldModel(BaseModel):
    random_seed: RandomSeed = random_seed_field()


def test_random_seed_defaults_to_generated_int() -> None:
    model = _RandomSeedFieldModel()
    assert isinstance(model.random_seed, int)
    assert 0 <= model.random_seed < 2**32


def test_random_seed_none_is_replaced_with_generated_int() -> None:
    model = _RandomSeedFieldModel(random_seed=None)
    assert isinstance(model.random_seed, int)
    assert 0 <= model.random_seed < 2**32


def test_seed_derivation_preserves_component_boundaries() -> None:
    assert derive_random_seed(42, "a:b") != derive_random_seed(42, "a", "b")
