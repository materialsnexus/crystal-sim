"""Utilities for locating and validating Quantum ESPRESSO pseudopotentials."""

from __future__ import annotations

from pathlib import Path

from crystal.helpers.qe.pseudopotentials.parser import UPFMetadataParser
from crystal.schemas.elements import ElementStr
from crystal.schemas.pseudopotential import (
    PseudopotentialConfig,
    PseudopotentialFileName,
    PseudopotentialInfo,
)
from crystal.schemas.serialization import MSONableBaseModel


class PseudopotentialLibrary(MSONableBaseModel):
    """Load QE pseudopotential files and expose validated metadata."""

    pseudo_dir: Path

    def load_metadata(
        self, pseudopotentials_file_map: dict[ElementStr, PseudopotentialFileName]
    ) -> dict[ElementStr, PseudopotentialInfo]:
        """Return element→PseudoInfo mapping by reading the provided files."""
        config = PseudopotentialConfig(
            pseudopotentials_file_map=pseudopotentials_file_map,
            pseudopotential_dir=self.pseudo_dir,
        )
        metadata: dict[ElementStr, PseudopotentialInfo] = {}
        for element, filename in config.pseudopotentials_file_map.items():
            filepath = self._pseudo_path(filename)
            metadata[element] = UPFMetadataParser(filepath).parse()
        return metadata

    def validate_required_elements(
        self,
        file_map: dict[ElementStr, PseudopotentialFileName],
        required_elements: set[ElementStr],
    ) -> None:
        """Ensure every required element has a pseudopotential mapping."""
        config = PseudopotentialConfig(
            pseudopotential_dir=self.pseudo_dir, pseudopotentials_file_map=file_map
        )
        config.validate_elements(required_elements)

    def _pseudo_path(self, filename: PseudopotentialFileName) -> Path:
        path = self.pseudo_dir / filename
        if not path.is_file():
            msg = f"Missing pseudopotential file '{filename}' in {self.pseudo_dir}"
            raise ValueError(msg)
        return path
