"""Tests for QE pp.x charge density cube input generation."""

from pathlib import Path

import pytest

from crystal.schemas.qe_pp import (
    QePpChargeDensityCubeInputBuilder,
    QePpInputSettings,
    QePpOutputFormat,
    QePpPlotFlag,
    QePpPlotNumber,
    QePpPlotSettings,
)
from tests.helpers.constants.qe_pp import (
    QE_PP_CHARGE_DENSITY_FILE_SUFFIX,
    QE_PP_CUSTOM_FILEOUT,
    QE_PP_CUSTOM_PREFIX,
    QE_PP_DEFAULT_RUN_PREFIX,
    QE_PP_TEST_INPUT_PREFIX,
    TEST_QE_PP_FILEOUT_LINE_TEMPLATE,
    TEST_QE_PP_IFLAG_LINE,
    TEST_QE_PP_INPUT_SECTION_HEADER,
    TEST_QE_PP_INVALID_NFILE,
    TEST_QE_PP_INVALID_WEIGHT_FIRST_FILE,
    TEST_QE_PP_NFILE_ERROR_PATTERN,
    TEST_QE_PP_NFILE_LINE,
    TEST_QE_PP_OUTDIR_LINE_TEMPLATE,
    TEST_QE_PP_OUTPUT_FORMAT_LINE,
    TEST_QE_PP_PLOT_NUM_LINE,
    TEST_QE_PP_PLOT_SECTION_HEADER,
    TEST_QE_PP_PREFIX_LINE_TEMPLATE,
    TEST_QE_PP_REQUIRED_OUTDIR_FIELD,
    TEST_QE_PP_REQUIRED_PREFIX_FIELD,
    TEST_QE_PP_SECTION_TERMINATOR,
    TEST_QE_PP_WEIGHT_ERROR_PATTERN,
    TEST_QE_PP_WEIGHT_LINE,
)


@pytest.fixture
def qe_pp_outdir(tmp_path: Path) -> str:
    return tmp_path.as_posix()


def test_pp_defaults_for_charge_density_cube_settings(qe_pp_outdir: str) -> None:
    """Defaults should align with charge density cube output."""
    input_settings = QePpInputSettings(
        prefix=QE_PP_TEST_INPUT_PREFIX, outdir=qe_pp_outdir
    )
    plot_settings = QePpPlotSettings()
    assert input_settings.plot_num is QePpPlotNumber.charge_density
    assert plot_settings.iflag is QePpPlotFlag.three_dimensional
    assert plot_settings.output_format is QePpOutputFormat.gaussian_cube


def test_rendered_input_text_is_deterministic_and_ordered(qe_pp_outdir: str) -> None:
    """Rendered pp.x input should be deterministic and use fixed ordering."""
    input_settings = QePpInputSettings(
        prefix=QE_PP_TEST_INPUT_PREFIX, outdir=qe_pp_outdir
    )
    plot_settings = QePpPlotSettings()
    builder = QePpChargeDensityCubeInputBuilder(
        input_settings=input_settings, plot_settings=plot_settings
    )
    expected = "\n".join(
        [
            TEST_QE_PP_INPUT_SECTION_HEADER,
            TEST_QE_PP_PREFIX_LINE_TEMPLATE.format(prefix=QE_PP_TEST_INPUT_PREFIX),
            TEST_QE_PP_OUTDIR_LINE_TEMPLATE.format(outdir=qe_pp_outdir),
            TEST_QE_PP_PLOT_NUM_LINE,
            TEST_QE_PP_SECTION_TERMINATOR,
            TEST_QE_PP_PLOT_SECTION_HEADER,
            TEST_QE_PP_NFILE_LINE,
            TEST_QE_PP_FILEOUT_LINE_TEMPLATE.format(
                fileout=f"{QE_PP_TEST_INPUT_PREFIX}{QE_PP_CHARGE_DENSITY_FILE_SUFFIX}"
            ),
            TEST_QE_PP_WEIGHT_LINE,
            TEST_QE_PP_IFLAG_LINE,
            TEST_QE_PP_OUTPUT_FORMAT_LINE,
            TEST_QE_PP_SECTION_TERMINATOR,
        ]
    )
    assert builder.pp_charge_density_input_text == expected


def test_cube_filename_defaults_from_prefix_and_plot_num(qe_pp_outdir: str) -> None:
    """Cube filename should follow the default prefix + plot label convention."""
    input_settings = QePpInputSettings(
        prefix=QE_PP_TEST_INPUT_PREFIX, outdir=qe_pp_outdir
    )
    plot_settings = QePpPlotSettings()
    builder = QePpChargeDensityCubeInputBuilder(
        input_settings=input_settings, plot_settings=plot_settings
    )
    assert (
        builder.cube_filename
        == f"{QE_PP_TEST_INPUT_PREFIX}{QE_PP_CHARGE_DENSITY_FILE_SUFFIX}"
    )


def test_cube_filename_uses_explicit_fileout(qe_pp_outdir: str) -> None:
    """Cube filename should use fileout when explicitly provided."""
    input_settings = QePpInputSettings(
        prefix=QE_PP_TEST_INPUT_PREFIX, outdir=qe_pp_outdir
    )
    plot_settings = QePpPlotSettings(fileout=QE_PP_CUSTOM_FILEOUT)
    builder = QePpChargeDensityCubeInputBuilder(
        input_settings=input_settings, plot_settings=plot_settings
    )
    assert builder.cube_filename == QE_PP_CUSTOM_FILEOUT


def test_prefix_and_outdir_are_required_at_model_validation(qe_pp_outdir: str) -> None:
    """prefix and outdir should be required to locate pw.x outputs."""
    with pytest.raises(ValueError, match=TEST_QE_PP_REQUIRED_PREFIX_FIELD):
        QePpInputSettings(outdir=qe_pp_outdir)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match=TEST_QE_PP_REQUIRED_OUTDIR_FIELD):
        QePpInputSettings(prefix=QE_PP_TEST_INPUT_PREFIX)  # type: ignore[call-arg]


def test_single_file_constraints_are_enforced() -> None:
    """nfile and weight(1) must remain at the single-file defaults."""
    with pytest.raises(ValueError, match=TEST_QE_PP_NFILE_ERROR_PATTERN):
        QePpPlotSettings(nfile=TEST_QE_PP_INVALID_NFILE)
    with pytest.raises(ValueError, match=TEST_QE_PP_WEIGHT_ERROR_PATTERN):
        QePpPlotSettings(weight_first_file=TEST_QE_PP_INVALID_WEIGHT_FIRST_FILE)


def test_input_settings_defaults_prefix_and_outdir_for_charge_density_runs(
    qe_pp_outdir: str,
) -> None:
    input_settings = QePpInputSettings(
        prefix=QE_PP_DEFAULT_RUN_PREFIX, outdir=qe_pp_outdir
    )
    assert input_settings.prefix == QE_PP_DEFAULT_RUN_PREFIX
    assert input_settings.outdir == qe_pp_outdir


def test_input_settings_preserves_custom_prefix_for_charge_density_runs(
    qe_pp_outdir: str,
) -> None:
    input_settings = QePpInputSettings(prefix=QE_PP_CUSTOM_PREFIX, outdir=qe_pp_outdir)
    assert input_settings.prefix == QE_PP_CUSTOM_PREFIX
    assert input_settings.outdir == qe_pp_outdir
