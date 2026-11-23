"""Stable random-state derivation for structure generation."""

from __future__ import annotations

import hashlib

import numpy as np

_STRUCTURE_GENERATION_SEED_NAMESPACE = "crystal-structure-generation"


def derive_random_seed(random_seed: int, *components: int | str) -> int:
    """Derive a reproducible 32-bit seed from canonical component values."""
    encoded_components = (
        _STRUCTURE_GENERATION_SEED_NAMESPACE,
        str(random_seed),
        *(str(component) for component in components),
    )
    seed_material = b"".join(
        len(component.encode()).to_bytes(8, byteorder="big") + component.encode()
        for component in encoded_components
    )
    digest = hashlib.blake2b(seed_material, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big") % (2**32)


def spawn_random_seeds(random_seed: int, number_of_variants: int) -> list[int]:
    """Return deterministic child seeds in requested variant order."""
    if number_of_variants < 0:
        msg = "number_of_variants cannot be negative."
        raise ValueError(msg)
    return [
        int(seed_sequence.generate_state(1)[0])
        for seed_sequence in np.random.SeedSequence(random_seed).spawn(
            number_of_variants
        )
    ]
