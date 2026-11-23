"""Tests for the external GPAW isolated-atom entry point."""

from __future__ import annotations

import importlib.util
import json
import sys
from hashlib import sha256
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from ase import Atoms
from ase.io import write

EXTERNAL_RUNTIME_DIRECTORY = (
    Path(__file__).parents[1] / "src/crystal/calculators/gpaw/external_runtime"
)
ISOLATED_ATOM_SCRIPT_PATH = (
    EXTERNAL_RUNTIME_DIRECTORY / "gpaw_external_isolated_atom_script.py"
)
GPAW_VERSION = "26.7.0"
SILICON_SETUP_FILENAME = "Si.PBE.gz"
SILICON_SETUP_CONTENT = b"silicon PAW setup"
SILICON_ENERGY = -5.25
SILICON_MAGNETIC_MOMENT = 2.0
PRIMARY_EIGENSOLVER_NAME = "etdm-fdpw"
FALLBACK_EIGENSOLVER_NAME = "ppcg"
EMPTY_OCCUPIED_BANDS_ERROR_MESSAGE = "max() arg is an empty sequence"
FALLBACK_REUSED_PRIMARY_CALCULATOR_MESSAGE = "fallback reused the primary calculator"
UNRELATED_CALCULATION_FAILURE_MESSAGE = "unrelated calculation failure"
PPCG_DID_NOT_CONVERGE_MESSAGE = "PPCG did not converge"


class FakeAtoms:
    """Small atom stand-in for deterministic external-script tests."""

    def __init__(self) -> None:
        self.calc: object | None = None

    def get_potential_energy(self) -> float:
        return SILICON_ENERGY

    def get_magnetic_moment(self) -> float:
        return SILICON_MAGNETIC_MOMENT


def _load_isolated_atom_script(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    fake_gpaw_module = ModuleType("gpaw")
    fake_gpaw_module.__dict__.update(
        {"__version__": GPAW_VERSION, "__path__": [], "GPAW": object}
    )
    fake_gpaw_mpi_module = ModuleType("gpaw.mpi")
    fake_gpaw_mpi_module.__dict__["world"] = SimpleNamespace(
        rank=0, size=1, barrier=lambda: None
    )
    monkeypatch.setitem(sys.modules, "gpaw", fake_gpaw_module)
    monkeypatch.setitem(sys.modules, "gpaw.mpi", fake_gpaw_mpi_module)
    monkeypatch.syspath_prepend(EXTERNAL_RUNTIME_DIRECTORY.as_posix())
    module_spec = importlib.util.spec_from_file_location(
        "test_gpaw_external_isolated_atom_script", ISOLATED_ATOM_SCRIPT_PATH
    )
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError
    isolated_atom_script = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(isolated_atom_script)
    return isolated_atom_script


def test_run_isolated_atom_from_payload_returns_energy_magnetic_moment_solver_and_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    isolated_atom_script = _load_isolated_atom_script(monkeypatch)
    setup_path = tmp_path / SILICON_SETUP_FILENAME
    setup_path.write_bytes(SILICON_SETUP_CONTENT)
    calculator_kwargs = {"mode": {"ecut": 600.0}, "hund": True}
    received_calculator_kwargs: dict[str, Any] = {}
    calculator = SimpleNamespace(
        setups=(SimpleNamespace(symbol="Si", filename=setup_path.as_posix(), Nv=4.0),)
    )

    def gpaw_factory(**kwargs: Any) -> object:
        received_calculator_kwargs.update(kwargs)
        return calculator

    monkeypatch.setattr(isolated_atom_script, "GPAW", gpaw_factory)
    output_payload = isolated_atom_script.run_isolated_atom_from_payload(
        atoms=FakeAtoms(),
        script_input_payload={"calculator_kwargs": calculator_kwargs},
        gpaw_version=GPAW_VERSION,
    )

    assert received_calculator_kwargs == calculator_kwargs
    assert output_payload == {
        "energy": SILICON_ENERGY,
        "magnetic_moment": SILICON_MAGNETIC_MOMENT,
        "eigensolver": PRIMARY_EIGENSOLVER_NAME,
        "runtime_provenance": {
            "gpaw_version": GPAW_VERSION,
            "atomic_data": [
                {
                    "element_symbol": "Si",
                    "paw_setup_sha256_digest": sha256(
                        SILICON_SETUP_CONTENT
                    ).hexdigest(),
                    "paw_setup_filename": SILICON_SETUP_FILENAME,
                    "paw_setup_kind": "paw",
                    "valence_electron_count": 4,
                }
            ],
        },
    }


def test_run_isolated_atom_from_payload_retries_empty_occupied_bands_with_ppcg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    isolated_atom_script = _load_isolated_atom_script(monkeypatch)
    setup_path = tmp_path / SILICON_SETUP_FILENAME
    setup_path.write_bytes(SILICON_SETUP_CONTENT)

    class FailingFirstCalculationAtoms(FakeAtoms):
        def __init__(self) -> None:
            super().__init__()
            self.energy_call_count = 0
            self.attached_calculators: list[object | None] = []
            self.primary_calculator: object | None = None

        def get_potential_energy(self) -> float:
            self.energy_call_count += 1
            self.attached_calculators.append(self.calc)
            if self.energy_call_count == 1:
                raise ValueError(EMPTY_OCCUPIED_BANDS_ERROR_MESSAGE)
            if self.calc is self.primary_calculator:
                raise RuntimeError(FALLBACK_REUSED_PRIMARY_CALCULATOR_MESSAGE)
            return super().get_potential_energy()

    atoms = FailingFirstCalculationAtoms()
    primary_calculator_kwargs = {
        "mode": {"name": "pw", "ecut": 600.0},
        "mixer": {"backend": "no-mixing", "method": "fullspin"},
        "txt": "gpaw.txt",
        "eigensolver": {"name": PRIMARY_EIGENSOLVER_NAME, "tol": 0.25},
    }
    received_calculator_kwargs: list[dict[str, Any]] = []
    primary_calculator = SimpleNamespace(
        setups=(SimpleNamespace(symbol="Si", filename=setup_path.as_posix(), Nv=4.0),)
    )
    fallback_calculator = SimpleNamespace(
        setups=(SimpleNamespace(symbol="Si", filename=setup_path.as_posix(), Nv=4.0),)
    )
    atoms.primary_calculator = primary_calculator
    calculators = iter((primary_calculator, fallback_calculator))

    def gpaw_factory(**kwargs: Any) -> object:
        received_calculator_kwargs.append(kwargs)
        return next(calculators)

    monkeypatch.setattr(isolated_atom_script, "GPAW", gpaw_factory)

    output_payload = isolated_atom_script.run_isolated_atom_from_payload(
        atoms=atoms,
        script_input_payload={"calculator_kwargs": primary_calculator_kwargs},
        gpaw_version=GPAW_VERSION,
    )

    assert output_payload["energy"] == SILICON_ENERGY
    assert output_payload["eigensolver"] == FALLBACK_EIGENSOLVER_NAME
    assert received_calculator_kwargs == [
        primary_calculator_kwargs,
        {
            **primary_calculator_kwargs,
            "eigensolver": {"name": FALLBACK_EIGENSOLVER_NAME},
        },
    ]
    assert atoms.attached_calculators == [primary_calculator, fallback_calculator]
    assert atoms.attached_calculators[0] is not atoms.attached_calculators[1]


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(RuntimeError(UNRELATED_CALCULATION_FAILURE_MESSAGE), id="runtime"),
        pytest.param(ValueError("unrelated value failure"), id="value-error"),
    ],
)
def test_run_isolated_atom_from_payload_does_not_retry_unrelated_calculation_failure(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    isolated_atom_script = _load_isolated_atom_script(monkeypatch)
    received_calculator_kwargs: list[dict[str, Any]] = []

    class RuntimeFailingAtoms(FakeAtoms):
        def get_potential_energy(self) -> float:
            raise failure

    def gpaw_factory(**kwargs: Any) -> object:
        received_calculator_kwargs.append(kwargs)
        return SimpleNamespace(
            setups=(SimpleNamespace(symbol="Si", filename="Si.PBE.gz", Nv=4.0),)
        )

    monkeypatch.setattr(isolated_atom_script, "GPAW", gpaw_factory)

    with pytest.raises(type(failure), match=str(failure)):
        isolated_atom_script.run_isolated_atom_from_payload(
            atoms=RuntimeFailingAtoms(),
            script_input_payload={
                "calculator_kwargs": {"eigensolver": {"name": PRIMARY_EIGENSOLVER_NAME}}
            },
            gpaw_version=GPAW_VERSION,
        )

    assert len(received_calculator_kwargs) == 1


def test_run_isolated_atom_from_payload_preserves_second_solver_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolated_atom_script = _load_isolated_atom_script(monkeypatch)
    received_calculator_kwargs: list[dict[str, Any]] = []

    class AlwaysFailingAtoms(FakeAtoms):
        def __init__(self) -> None:
            super().__init__()
            self.energy_call_count = 0

        def get_potential_energy(self) -> float:
            self.energy_call_count += 1
            if self.energy_call_count == 1:
                raise ValueError(EMPTY_OCCUPIED_BANDS_ERROR_MESSAGE)
            raise ValueError(PPCG_DID_NOT_CONVERGE_MESSAGE)

    def gpaw_factory(**kwargs: Any) -> object:
        received_calculator_kwargs.append(kwargs)
        return SimpleNamespace(
            setups=(SimpleNamespace(symbol="Si", filename="Si.PBE.gz", Nv=4.0),)
        )

    monkeypatch.setattr(isolated_atom_script, "GPAW", gpaw_factory)

    with pytest.raises(ValueError, match=PPCG_DID_NOT_CONVERGE_MESSAGE):
        isolated_atom_script.run_isolated_atom_from_payload(
            atoms=AlwaysFailingAtoms(),
            script_input_payload={
                "calculator_kwargs": {
                    "mixer": {"backend": "no-mixing"},
                    "eigensolver": {"name": PRIMARY_EIGENSOLVER_NAME},
                }
            },
            gpaw_version=GPAW_VERSION,
        )


def test_run_isolated_atom_from_payload_does_not_retry_ppcg_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolated_atom_script = _load_isolated_atom_script(monkeypatch)
    received_calculator_kwargs: list[dict[str, Any]] = []

    class FailingPpcgAtoms(FakeAtoms):
        def get_potential_energy(self) -> float:
            raise ValueError(EMPTY_OCCUPIED_BANDS_ERROR_MESSAGE)

    def gpaw_factory(**kwargs: Any) -> object:
        received_calculator_kwargs.append(kwargs)
        return SimpleNamespace(
            setups=(SimpleNamespace(symbol="Si", filename="Si.PBE.gz", Nv=4.0),)
        )

    monkeypatch.setattr(isolated_atom_script, "GPAW", gpaw_factory)

    with pytest.raises(ValueError, match=r"max\(\) arg is an empty sequence"):
        isolated_atom_script.run_isolated_atom_from_payload(
            atoms=FailingPpcgAtoms(),
            script_input_payload={
                "calculator_kwargs": {
                    "eigensolver": {"name": FALLBACK_EIGENSOLVER_NAME}
                }
            },
            gpaw_version=GPAW_VERSION,
        )

    assert len(received_calculator_kwargs) == 1


def test_write_isolated_atom_output_payload_writes_only_on_rank_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    isolated_atom_script = _load_isolated_atom_script(monkeypatch)
    output_path = tmp_path / "isolated_atom_output.json"
    output_payload = {"energy": SILICON_ENERGY}

    isolated_atom_script.write_isolated_atom_output_payload(
        output_json_path=output_path, output_payload=output_payload, mpi_rank=1
    )

    assert not output_path.exists()

    isolated_atom_script.write_isolated_atom_output_payload(
        output_json_path=output_path, output_payload=output_payload, mpi_rank=0
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == output_payload


def test_load_isolated_atom_rejects_multi_atom_structure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    isolated_atom_script = _load_isolated_atom_script(monkeypatch)
    input_structure_path = tmp_path / "multi_atom.extxyz"
    write(
        input_structure_path.as_posix(),
        Atoms("Si2", positions=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]),
    )

    with pytest.raises(ValueError, match="requires exactly one atom"):
        isolated_atom_script.load_isolated_atom(input_structure_path)


def test_main_uses_public_gpaw_calculator(monkeypatch: pytest.MonkeyPatch) -> None:
    isolated_atom_script = _load_isolated_atom_script(monkeypatch)
    captured_run_kwargs: dict[str, object] = {}
    output_payload = {"energy": SILICON_ENERGY}
    run_isolated_atom_from_payload = MagicMock(
        side_effect=lambda **kwargs: (
            captured_run_kwargs.update(kwargs) or output_payload
        )
    )
    write_isolated_atom_output_payload = MagicMock()
    monkeypatch.setattr(
        isolated_atom_script, "load_isolated_atom", MagicMock(return_value=FakeAtoms())
    )
    monkeypatch.setattr(
        isolated_atom_script,
        "load_script_input_payload",
        MagicMock(return_value={"calculator_kwargs": {}}),
    )
    monkeypatch.setattr(
        isolated_atom_script,
        "run_isolated_atom_from_payload",
        run_isolated_atom_from_payload,
    )
    monkeypatch.setattr(
        isolated_atom_script,
        "write_isolated_atom_output_payload",
        write_isolated_atom_output_payload,
    )

    isolated_atom_script.main(
        ["script.py", "atom.extxyz", "kwargs.json", "output.json"]
    )

    assert captured_run_kwargs["gpaw_version"] == GPAW_VERSION
