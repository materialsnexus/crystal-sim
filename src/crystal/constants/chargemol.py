"""Chargemol-related constants."""

from pathlib import Path

CHARGEMOL_EXECUTABLE_NAME = "chargemol"
CHARGEMOL_PROCESS_NAME = "Chargemol"
CHARGEMOL_BIN_SUBDIR = Path("bin")
CHARGEMOL_ATOMIC_DENSITIES_SUBDIR = Path("atomic_densities")
CHARGEMOL_EXECUTABLE_RELATIVE_PATH = CHARGEMOL_BIN_SUBDIR / CHARGEMOL_EXECUTABLE_NAME
CHARGEMOL_CONFIG_NOT_FOUND_MESSAGE = (
    "Chargemol configuration not found. Set programs.chargemol_dir in crystal.toml."
)
CHARGEMOL_ATOMIC_DENSITIES_DIRECTORY_PLACEHOLDER = (
    "__CRYSTAL_CHARGEMOL_ATOMIC_DENSITIES_DIRECTORY__"
)
CHARGEMOL_JOB_CONTROL_FILENAME = "job_control.txt"
CHARGEMOL_NET_ATOMIC_CHARGES_FILENAME = "DDEC6_even_tempered_net_atomic_charges.xyz"
CHARGEMOL_BOND_ORDERS_FILENAME = "DDEC6_even_tempered_bond_orders.xyz"
CHARGEMOL_OVERLAP_POPULATIONS_FILENAME = "overlap_populations.xyz"
CHARGEMOL_TOTAL_CUBE_ANALYSIS_OUTPUT_FILENAME = "total_cube_DDEC_analysis.output"
CHARGEMOL_MISSING_OUTPUT_FILES_MESSAGE_TEMPLATE = (
    "Chargemol completed but expected output files were missing: {missing_filenames}"
)
