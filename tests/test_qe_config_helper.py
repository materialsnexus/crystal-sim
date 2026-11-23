"""Tests for ASE QE configuration helper."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

from crystal.helpers.qe.ase_config import set_ase_qe_config
from tests.helpers.test_paths import (
    TEST_ASE_QE_CONFIG_FILENAME,
    TEST_PSEUDOS_DIRNAME,
    TEST_QE_BIN_DIR,
)


def test_set_ase_qe_config_creates_config_file(tmp_path: Path) -> None:
    with TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / TEST_ASE_QE_CONFIG_FILENAME
        pseudo_dir = tmp_path / TEST_PSEUDOS_DIRNAME
        pseudo_dir.mkdir()
        set_ase_qe_config(
            qe_bin_path=TEST_QE_BIN_DIR, pseudo_dir=pseudo_dir, config_path=config_path
        )
        assert config_path.is_file()
        content = config_path.read_text()
        assert "[espresso]" in content
        assert "pseudo_dir" in content
        assert os.environ.get("ASE_CALCULATOR_CONFIGURATION") == str(config_path)
