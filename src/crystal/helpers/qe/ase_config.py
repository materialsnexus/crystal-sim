"""Utilities for interacting with Quantum ESPRESSO (QE) calculations."""

import os
from pathlib import Path

from crystal.constants.qe import QE_PW_EXECUTABLE_NAME


def set_ase_qe_config(qe_bin_path: Path, pseudo_dir: Path, config_path: Path) -> None:  # noqa: D103
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_content = (
        "[espresso]\n"
        f"command = {qe_bin_path}/{QE_PW_EXECUTABLE_NAME} -in PREFIX.pwi > PREFIX.pwo\n"
        f"pseudo_dir = {pseudo_dir}\n"
    )
    config_path.write_text(config_content)
    os.environ["ASE_CALCULATOR_CONFIGURATION"] = str(config_path)
