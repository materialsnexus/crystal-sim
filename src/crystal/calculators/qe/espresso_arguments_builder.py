"""Translate Quantum ESPRESSO (QE) configs into ASE Espresso calculator kwargs."""

from __future__ import annotations

from typing import Any, cast

from pydantic import ConfigDict

from crystal.schemas.elements import ElementStr
from crystal.schemas.pseudopotential import PseudopotentialFileName
from crystal.schemas.qe import QeConfig, SerializableEspressoProfile
from crystal.schemas.serialization import MSONableBaseModel


class EspressoArgumentsBuilder(MSONableBaseModel):
    """Describe the ASE Espresso kwargs built from validated QE inputs."""

    config: QeConfig
    pseudopotentials: dict[ElementStr, PseudopotentialFileName]
    profile: SerializableEspressoProfile

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def calculator_kwargs(self) -> dict[str, Any]:
        """Return kwargs ready for ase.calculators.espresso.Espresso."""
        kwargs: dict[str, Any] = {
            "input_data": self.config.input_data.build_qe_input_data_namelists_with_cutoffs_in_rydberg(),
            "pseudopotentials": self.pseudopotentials,
            "profile": self.profile,
        }
        return kwargs | self._kpoints_kwargs

    def validate_pseudopotentials(self, required_elements: set[str]) -> None:
        """Ensure every required element has a pseudopotential mapping."""
        missing = {
            symbol
            for symbol in required_elements
            if symbol not in self.pseudopotentials
        }
        if missing:
            msg = f"Missing pseudopotentials for: {', '.join(sorted(missing))}"
            raise ValueError(msg)

    @property
    def _kpoints_kwargs(self) -> dict[str, Any]:
        spacing = self.config.kpoints.spacing_in_inverse_angstrom
        if spacing is not None:
            return {"kspacing": spacing}
        # Model validation guarantees mesh is set when spacing is None.
        mesh = cast(tuple[int, int, int], self.config.kpoints.mesh)  # noqa: TC006
        offsets = tuple(1 if v else 0 for v in self.config.kpoints.offset)
        return {"kpts": mesh, "koffset": offsets}
