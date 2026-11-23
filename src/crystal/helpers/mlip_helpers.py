# noqa: D100
from __future__ import annotations

import ase
from fireworks.utilities.fw_utilities import get_fw_logger, log_multi

from crystal.helpers.structure import DEFAULT_PERIODICITY

L_LOGGER = get_fw_logger("rocket.launcher")


def prepare_atoms_for_matgl(atoms: ase.Atoms) -> ase.Atoms:
    """Normalize a CRYSTAL MLIP input for MatGL's ASE graph converter.

    MatGL supports fully periodic and fully nonperiodic graphs, but its ASE
    converter does not represent mixed PBC as a partially periodic graph. The
    CRYSTAL bulk/slab workflow therefore normalizes slab/interface inputs to
    fully periodic boundary conditions before evaluation.
    """
    periodicity = tuple(bool(flag) for flag in atoms.pbc)
    if not periodicity[0] or not periodicity[1]:
        error_msg = (
            "CRYSTAL MLIP evaluation requires periodicity in X and Y; only Z "
            "may be nonperiodic for slab/interface inputs."
        )
        raise ValueError(error_msg)

    prepared = atoms.copy()  # type: ignore[no-untyped-call]
    assert isinstance(prepared, ase.Atoms)
    prepared.set_pbc(DEFAULT_PERIODICITY)  # type: ignore[no-untyped-call]
    if atoms.calc is not None:
        prepared.calc = atoms.calc
    return prepared


def configure_default_torch_device(requested_gpu_count: int) -> None:
    """Prefer CUDA when GPU resources were explicitly requested."""
    import torch  # noqa: PLC0415

    if requested_gpu_count == 0:
        torch.set_default_device("cpu")  # type: ignore[no-untyped-call]
        log_multi(L_LOGGER, "Torch default device set to cpu for MLIP run.")
        return
    cuda_is_available = torch.cuda.is_available()
    log_msg = f"Configuring torch default device for MLIP run requested_gpu_count={requested_gpu_count} cuda_available={cuda_is_available}"
    log_multi(L_LOGGER, log_msg)
    if not cuda_is_available:
        error_msg = (
            "GPU resources were requested for the MLIP run, but CUDA is not available."
        )
        raise RuntimeError(error_msg)
    torch.set_default_device("cuda")  # type: ignore[no-untyped-call]
    current_device = torch.cuda.current_device()
    set_device_log_msg = f"Torch default device set to cuda for MLIP run current_device={current_device} device_name={torch.cuda.get_device_name(current_device)}"
    log_multi(L_LOGGER, set_device_log_msg)
