"""Unit tests for pseudopotential models."""

from pathlib import Path

import pytest

from crystal.helpers.qe.pseudopotentials.parser import UPFMetadataParser
from crystal.schemas.pseudopotential import (
    Functional,
    PseudopotentialConfig,
    PseudopotentialInfo,
    PseudoPotentialType,
)
from tests.helpers.test_paths import (
    TEST_BAD_FUNCTIONAL_USP_FILENAME,
    TEST_BAD_USP_FILENAME,
    TEST_FILE_HELPERS_DATA_DIRNAME,
    TEST_NA_PBE_USP_FILENAME,
    TEST_O_NC_PSEUDOPOTENTIAL_FILENAME,
    TEST_O_PBE_PSEUDOPOTENTIAL_FILENAME,
    TEST_PSEUDOPOTENTIALS_DIRNAME,
    TEST_PSEUDOS_DIRNAME,
    TEST_SI_NC_PSEUDOPOTENTIAL_FILENAME,
    TEST_SI_PBE_PAW_UPF_FILENAME,
    TEST_SI_PSEUDOPOTENTIAL_FILENAME,
)

DATA_DIR = Path(__file__).parent / TEST_FILE_HELPERS_DATA_DIRNAME
PRODUCTION_DATA_DIR = DATA_DIR / TEST_PSEUDOPOTENTIALS_DIRNAME


@pytest.fixture
def sample_config(tmp_path: Path) -> PseudopotentialConfig:
    """Pseudopotential mapping used by multiple tests."""

    return PseudopotentialConfig(
        pseudopotential_dir=tmp_path / TEST_PSEUDOS_DIRNAME,
        pseudopotentials_file_map={
            "Si": TEST_SI_NC_PSEUDOPOTENTIAL_FILENAME,
            "O": TEST_O_NC_PSEUDOPOTENTIAL_FILENAME,
        },
    )


@pytest.mark.parametrize("query", ["si", "Si", "sI", "SI"])
def test_pseudopotential_config_getitem_is_case_insensitive(
    sample_config: PseudopotentialConfig, query: str
) -> None:
    """__getitem__ should normalize element symbols via capitalize."""

    assert sample_config[query] == TEST_SI_NC_PSEUDOPOTENTIAL_FILENAME


def test_pseudopotential_config_getitem_missing_element(
    sample_config: PseudopotentialConfig,
) -> None:
    """Requesting a missing element should bubble up KeyError."""

    with pytest.raises(KeyError):
        _ = sample_config["Ti"]


def test_validate_elements_pass_when_all_elements_present(
    sample_config: PseudopotentialConfig,
) -> None:
    """validate_elements should succeed when every symbol is mapped."""

    sample_config.validate_elements({"si", "O"})


def test_validate_elements_raises_with_descriptive_message(tmp_path: Path) -> None:
    """Missing elements should raise ValueError with sorted list in message."""

    config = PseudopotentialConfig(
        pseudopotential_dir=tmp_path / TEST_PSEUDOS_DIRNAME,
        pseudopotentials_file_map={"Si": TEST_SI_NC_PSEUDOPOTENTIAL_FILENAME},
    )
    with pytest.raises(ValueError, match="Missing pseudopotentials for: O, Ti"):
        config.validate_elements({"o", "ti"})


def test_pseudopotential_config_requires_directory() -> None:
    """Config should require a pseudopotential directory."""
    with pytest.raises(ValueError, match="pseudopotential_dir"):
        PseudopotentialConfig.model_validate(
            {"pseudopotentials_file_map": {"Si": TEST_SI_NC_PSEUDOPOTENTIAL_FILENAME}}
        )


def test_pseudo_info_key_uses_standard_format() -> None:
    """PseudoInfo.key should concatenate element, functional and type values."""

    pseudo_info = PseudopotentialInfo(
        element="Si",
        functional=Functional.pbe,
        type=PseudoPotentialType.paw,
        filename=TEST_SI_PBE_PAW_UPF_FILENAME,
    )
    assert pseudo_info.key == "Si.pbe.paw"


def test_upf_parser_handles_modern_xml_header() -> None:
    """UPFMetadataParser should read attributes from QE v2 UPF files."""

    parser = UPFMetadataParser(filepath=DATA_DIR / TEST_SI_PSEUDOPOTENTIAL_FILENAME)
    info = parser.parse()
    assert info.element == "Si"
    assert info.functional is Functional.pbe
    assert info.type is PseudoPotentialType.ultrasoft
    assert info.z_valence == pytest.approx(4.0)


def test_upf_parser_handles_legacy_header_blocks() -> None:
    """Legacy UPF files (UPF V1) without a root tag."""

    parser = UPFMetadataParser(filepath=DATA_DIR / TEST_NA_PBE_USP_FILENAME)
    info = parser.parse()
    assert info.element == "Na"
    assert info.functional is Functional.pbe
    assert info.type is PseudoPotentialType.ultrasoft
    assert info.z_valence == pytest.approx(9.0)


def test_upf_parser_supports_pbe_composite_functional_tokens() -> None:
    """Composite functional tokens should map to PBE."""

    parser = UPFMetadataParser(
        filepath=PRODUCTION_DATA_DIR / TEST_O_PBE_PSEUDOPOTENTIAL_FILENAME
    )
    info = parser.parse()
    assert info.functional is Functional.pbe


def test_upf_parser_rejects_unknown_composite_functionals(tmp_path: Path) -> None:
    """Unknown composite functionals should raise a descriptive error."""

    fake_file = tmp_path / TEST_BAD_FUNCTIONAL_USP_FILENAME
    fake_file.write_text(
        """
<PP_HEADER element="H" functional="SLA PW PBC" pseudo_type="USPP" z_valence="1" />
        """.strip()
    )
    parser = UPFMetadataParser(filepath=fake_file)
    with pytest.raises(ValueError, match="Unknown functional"):
        parser.parse()


def test_upf_parser_validates_element_symbols(tmp_path: Path) -> None:
    """Invalid element symbols should raise a descriptive error."""

    fake_file = tmp_path / TEST_BAD_USP_FILENAME
    fake_file.write_text(
        """
<PP_HEADER element=\"Xx\" functional=\"PBE\" pseudo_type=\"USPP\" z_valence=\"2\" />
        """.strip()
    )
    parser = UPFMetadataParser(filepath=fake_file)
    with pytest.raises(ValueError, match="Invalid element"):
        parser.parse()
