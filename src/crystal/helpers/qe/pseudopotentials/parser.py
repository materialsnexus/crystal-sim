"""Utilities for parsing Quantum ESPRESSO UPF pseudopotential metadata."""

from __future__ import annotations

import re
from pathlib import Path

from defusedxml import ElementTree
from pymatgen.core.periodic_table import Element

from crystal.schemas.pseudopotential import (
    Functional,
    PseudopotentialInfo,
    PseudoPotentialType,
)


class UPFMetadataParser:
    """Minimal parser for Quantum ESPRESSO UPF metadata."""

    _HEADER_ATTRIBUTE_PATTERN = re.compile(
        r"<PP_HEADER[^>]*/>", flags=re.IGNORECASE | re.DOTALL
    )
    _ELEMENT_PATTERN = re.compile(
        r"Element\s*:\s*(?P<value>[A-Za-z]{1,2})", flags=re.IGNORECASE
    )
    _FUNCTIONAL_PATTERN = re.compile(
        r"Functional\s*:\s*(?P<value>[^\r\n]+)", flags=re.IGNORECASE
    )
    _FUNCTIONAL_DFT_PATTERN = re.compile(
        r"dft\s*=\s*['\"](?P<value>[A-Za-z0-9_-]+)['\"]", flags=re.IGNORECASE
    )
    _TYPE_PATTERN = re.compile(
        r"Pseudopotential\s+type\s*:\s*(?P<value>[A-Za-z0-9_-]+)", flags=re.IGNORECASE
    )
    _Z_VALENCE_PATTERN = re.compile(
        r"(?P<value>[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+Z\s+valence",
        flags=re.IGNORECASE,
    )
    _TYPE_ALIASES = {  # noqa: RUF012
        "us": PseudoPotentialType.ultrasoft,
        "uspp": PseudoPotentialType.ultrasoft,
        "usp": PseudoPotentialType.ultrasoft,
        "ultrasoft": PseudoPotentialType.ultrasoft,
        "nc": PseudoPotentialType.norm_conserving,
        "ncpp": PseudoPotentialType.norm_conserving,
        "normconserving": PseudoPotentialType.norm_conserving,
        "norm-conserving": PseudoPotentialType.norm_conserving,
        "paw": PseudoPotentialType.paw,
    }
    _COMPOSITE_FUNCTIONAL_TOKENS_TO_FUNCTIONAL: dict[tuple[str, ...], Functional] = {  # noqa: RUF012
        ("sla", "pw", "pbx", "pbc"): Functional.pbe
    }

    def __init__(self, filepath: Path | str) -> None:  # noqa: D107
        self.filepath = Path(filepath)

    def parse(self) -> PseudopotentialInfo:
        """Parse metadata from a UPF file into :class:`PseudoInfo`."""
        text = self.filepath.read_text(encoding="utf-8")
        metadata = self._collect_metadata(text)

        element_raw = metadata.get("element")
        functional_raw = metadata.get("functional")
        pseudo_type_raw = metadata.get("pseudo_type")

        if element_raw is None:
            msg = f"Missing element in UPF header ({self.filepath})."
            raise ValueError(msg)
        if functional_raw is None:
            msg = f"Missing functional in UPF header ({self.filepath})."
            raise ValueError(msg)
        if pseudo_type_raw is None:
            msg = f"Missing pseudopotential type in UPF header ({self.filepath})."
            raise ValueError(msg)

        element = self._coerce_element_symbol(element_raw)
        functional = self._coerce_functional(functional_raw)
        pseudo_type = self._coerce_pseudo_type(pseudo_type_raw)
        z_valence = self._coerce_z_valence(metadata.get("z_valence"))

        return PseudopotentialInfo(
            element=element,
            functional=functional,
            type=pseudo_type,
            filename=self.filepath.name,
            z_valence=z_valence,
        )

    def _collect_metadata(self, text: str) -> dict[str, str]:
        metadata = self._parse_textual_metadata(text)
        metadata.update(self._parse_header_attributes(text))
        return metadata

    def _parse_header_attributes(self, text: str) -> dict[str, str]:
        match = self._HEADER_ATTRIBUTE_PATTERN.search(text)
        if not match:
            return {}
        snippet = match.group(0)
        try:
            element = ElementTree.fromstring(snippet)
        except ElementTree.ParseError:
            return {}
        attrs = {key.lower(): value for key, value in element.attrib.items()}
        renamed: dict[str, str] = {}
        if "element" in attrs:
            renamed["element"] = attrs["element"]
        if "functional" in attrs:
            renamed["functional"] = attrs["functional"]
        if "pseudo_type" in attrs:
            renamed["pseudo_type"] = attrs["pseudo_type"]
        if "z_valence" in attrs:
            renamed["z_valence"] = attrs["z_valence"]
        return renamed

    def _parse_textual_metadata(self, text: str) -> dict[str, str]:
        metadata: dict[str, str] = {}

        match = self._ELEMENT_PATTERN.search(text)
        if match:
            metadata["element"] = match.group("value")

        func_match = self._FUNCTIONAL_PATTERN.search(
            text
        ) or self._FUNCTIONAL_DFT_PATTERN.search(text)
        if func_match:
            metadata["functional"] = func_match.group("value")

        type_match = self._TYPE_PATTERN.search(text)
        if type_match:
            metadata["pseudo_type"] = type_match.group("value")

        z_match = self._Z_VALENCE_PATTERN.search(text)
        if z_match:
            metadata["z_valence"] = z_match.group("value")

        return metadata

    def _coerce_element_symbol(self, value: str) -> str:
        try:
            return Element(value.strip()).symbol
        except Exception as error:  # pragma: no cover
            msg = f"Invalid element '{value}' in {self.filepath}"
            raise ValueError(msg) from error

    def _coerce_functional(self, value: str) -> Functional:
        normalized = value.strip().lower()
        try:
            return Functional(normalized)
        except ValueError as error:
            composite_functional = self._coerce_composite_functional(normalized)
            if composite_functional is not None:
                return composite_functional
            msg = f"Unknown functional '{value}' in {self.filepath}"
            raise ValueError(msg) from error

    def _coerce_composite_functional(self, normalized: str) -> Functional | None:
        """Map composite functional tokens to a known functional."""
        tokens = tuple(token for token in re.split(r"[\s,+]+", normalized) if token)
        return self._COMPOSITE_FUNCTIONAL_TOKENS_TO_FUNCTIONAL.get(tokens)

    def _coerce_pseudo_type(self, value: str) -> PseudoPotentialType:
        normalized = value.strip().lower()
        if normalized in self._TYPE_ALIASES:
            return self._TYPE_ALIASES[normalized]
        try:
            return PseudoPotentialType(normalized)
        except ValueError as error:
            msg = f"Unknown pseudopotential type '{value}' in {self.filepath}"
            raise ValueError(msg) from error

    def _coerce_z_valence(self, value: str | None) -> float | None:
        if value is None:
            return None
        try:
            return float(value)
        except ValueError as error:
            msg = f"Invalid z_valence '{value}' in {self.filepath}"
            raise ValueError(msg) from error
