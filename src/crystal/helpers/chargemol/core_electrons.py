"""Helpers for Chargemol core-electron mappings."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from crystal.helpers.qe.pseudopotentials.library import PseudopotentialLibrary
from crystal.schemas.pseudopotential import PseudopotentialConfig, PseudopotentialInfo


class ChargemolCoreElectronCountBuilder(BaseModel):
    """Compute core-electron counts from pseudopotential metadata."""

    model_config = ConfigDict(frozen=True)

    pseudopotential_infos: list[PseudopotentialInfo]

    @property
    def core_electrons_by_atomic_number(self) -> dict[int, int]:
        """Return atomic-number → core-electron count dictionary."""
        core_electrons_by_atomic_number: dict[int, int] = {}
        for pseudopotential_info in self.pseudopotential_infos:
            atomic_number = pseudopotential_info.atomic_number
            core_electrons = pseudopotential_info.core_electron_count
            core_electrons_by_atomic_number[atomic_number] = core_electrons
        return core_electrons_by_atomic_number


def build_chargemol_core_electrons_by_atomic_number_from_pseudopotential_files(
    pseudopotential_config: PseudopotentialConfig,
) -> dict[int, int]:
    """Return core-electron counts from a pseudopotential directory and map."""
    library = PseudopotentialLibrary(
        pseudo_dir=pseudopotential_config.pseudopotential_dir
    )
    metadata = library.load_metadata(pseudopotential_config.pseudopotentials_file_map)
    builder = ChargemolCoreElectronCountBuilder(
        pseudopotential_infos=list(metadata.values())
    )
    return builder.core_electrons_by_atomic_number
