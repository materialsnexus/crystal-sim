"""Pydantic models for Quantum ESPRESSO pp.x charge density cube inputs."""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Literal, Self

from pydantic import Field, model_validator

from crystal.schemas.backend import SinglePointBackend
from crystal.schemas.qe import QeScfSimulationSettings
from crystal.schemas.serialization import MSONableBaseModel
from crystal.schemas.settings import MakerSettings, SimulationSettings


class QePpPlotNumber(IntEnum):
    """Selects the quantity written by pp.x.

    `charge_density=0` writes the electron (pseudo-)charge density.
    https://www.quantum-espresso.org/Doc/INPUT_PP.html#id7
    """

    charge_density = 0


class QePpPlotFlag(IntEnum):
    """Selects the plot dimensionality for pp.x.

    `three_dimensional=3` selects a 3D plot.
    https://www.quantum-espresso.org/Doc/INPUT_PP.html#id32
    """

    three_dimensional = 3


class QePpOutputFormat(IntEnum):
    """Selects the plot file format for pp.x.

    `gaussian_cube=6` writes a Gaussian cube file.
    https://www.quantum-espresso.org/Doc/INPUT_PP.html#id33
    """

    gaussian_cube = 6


class QePpPlotLabel(StrEnum):
    """Filename labels for supported pp.x plot_num values."""

    charge_density = "charge_density"


PLOT_NUMBER_TO_LABEL: dict[QePpPlotNumber, QePpPlotLabel] = {
    QePpPlotNumber.charge_density: QePpPlotLabel.charge_density
}
CUBE_FILE_EXTENSION = ".cube"
DEFAULT_QE_PP_PREFIX = "pwscf"


class QeScfPpChargemolSimulationSettings(SimulationSettings):  # noqa: D101
    qe_scf_simulation_settings: QeScfSimulationSettings
    qe_pp_input_settings: QePpInputSettings | None = Field(default=None)
    qe_pp_plot_settings: QePpPlotSettings | None = Field(default=None)


class QeScfPpChargemolMakerSettings(MakerSettings):  # noqa: D101
    backend: Literal[SinglePointBackend.qe] = SinglePointBackend.qe
    simulation_settings: QeScfPpChargemolSimulationSettings


class QePpInputSettings(MSONableBaseModel):
    """pp.x &INPUTPP settings, with outdir expected from the pw.x run folder."""

    prefix: str
    outdir: str
    plot_num: QePpPlotNumber = Field(default=QePpPlotNumber.charge_density)


class QePpPlotSettings(MSONableBaseModel):
    """pp.x &PLOT settings for charge density cube output."""

    nfile: int = Field(default=1, ge=1)
    fileout: str | None = Field(default=None)
    weight_first_file: float = Field(default=1.0)
    iflag: QePpPlotFlag = Field(default=QePpPlotFlag.three_dimensional)
    output_format: QePpOutputFormat = Field(default=QePpOutputFormat.gaussian_cube)

    @model_validator(mode="after")
    def _validate_single_file_settings(self) -> Self:
        if self.nfile != 1:
            error_msg = "Only nfile=1 is supported for charge density cubes."
            raise ValueError(error_msg)
        if self.weight_first_file != 1.0:
            error_msg = "Only weight(1)=1.0 is supported for charge density cubes."
            raise ValueError(error_msg)
        return self


class QePpChargeDensityCubeInputBuilder(MSONableBaseModel):
    """Build a pp.x input text for charge density cubes."""

    input_settings: QePpInputSettings
    plot_settings: QePpPlotSettings

    @property
    def cube_filename(self) -> str:
        """Return the cube filename produced by pp.x."""
        plot_label = PLOT_NUMBER_TO_LABEL[self.input_settings.plot_num]
        return (
            self.plot_settings.fileout
            or f"{self.input_settings.prefix}_{plot_label}{CUBE_FILE_EXTENSION}"
        )

    @property
    def pp_charge_density_input_text(self) -> str:
        """Return the full pp.x input text for a charge density cube."""
        fileout = self.cube_filename
        pp_charge_density_input_lines = [
            "&INPUTPP",
            f"  prefix='{self.input_settings.prefix}'",
            f"  outdir='{self.input_settings.outdir}'",
            f"  plot_num={int(self.input_settings.plot_num)}",
            "/",
            "&PLOT",
            f"  nfile={self.plot_settings.nfile}",
            f"  fileout='{fileout}'",
            f"  weight(1)={self.plot_settings.weight_first_file}",
            f"  iflag={int(self.plot_settings.iflag)}",
            f"  output_format={int(self.plot_settings.output_format)}",
            "/",
        ]
        return "\n".join(pp_charge_density_input_lines)
