"""Workflow maker for interface adhesion calculations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from jobflow.core.flow import Flow
from jobflow.core.job import Job, Response, job
from pymatgen.core.structure import Structure as PMG_Structure

from crystal.calculators.relax_maker_factory import RelaxMaker, build_relax_maker
from crystal.calculators.single_point_maker_factory import build_single_point_maker
from crystal.constants.adhesion import (
    INTERFACE_RELAX_NOT_CONVERGED_ERROR,
    SLAB_CHARGEMOL_NOT_CONVERGED_ERROR,
    SLAB_RELAX_NOT_CONVERGED_ERROR,
)
from crystal.flows.adhesion_workflow_helpers import (
    build_adhesion_energy_records_from_single_point_task_docs,
    build_slab_from_structure_with_slab_metadata,
    extract_relax_final_structure_or_raise,
    relax_properties_converged,
)
from crystal.helpers.structure import split_items_into_batches
from crystal.schemas.adhesion import (
    AdhesionSimulationSettings,
    AdhesionWorkflowInputs,
    AdhesionWorkflowResult,
)
from crystal.schemas.base import BaseMaker
from crystal.schemas.properties import RelaxProperties
from crystal.schemas.relaxation import RelaxMakerSettings
from crystal.schemas.serialization import MSONableBaseModel
from crystal.schemas.single_point import (
    SinglePointWithChargesMakerSettings,
    SinglePointWithChargesTaskDocument,
)
from crystal.structure.interfaces.assembler import InterfaceAssembler
from crystal.structure.interfaces.settings import InterfaceAssemblySettings
from crystal.structure.slabs.builder import Slab


@dataclass(frozen=True)
class _InterfaceRelaxBatchResult:
    flows: list[Flow]
    relax_properties: list[RelaxProperties | None]
    relax_dir_names: list[str]


class _AdhesionInterfaceRelaxFlowBuilderInputs(MSONableBaseModel):
    top_slab_input: Slab
    bottom_slab_input: Slab
    assembly_settings: InterfaceAssemblySettings

    @property
    def interface_jobs_per_batch(self) -> int | None:
        return self.assembly_settings.interface_jobs_per_batch


class _AdhesionInterfaceRelaxFlowCalculationMakerSettings(MSONableBaseModel):
    interface_relax_maker_settings: RelaxMakerSettings
    single_point_maker_settings: SinglePointWithChargesMakerSettings


class _AdhesionInterfaceRelaxFlowSlabChargemolOutputs(MSONableBaseModel):
    slab_relax_dir_names: list[str]
    top_slab_single_point_task_doc: SinglePointWithChargesTaskDocument
    bottom_slab_single_point_task_doc: SinglePointWithChargesTaskDocument

    @property
    def slab_single_point_with_charges_task_docs(
        self,
    ) -> list[SinglePointWithChargesTaskDocument]:
        return [
            self.top_slab_single_point_task_doc,
            self.bottom_slab_single_point_task_doc,
        ]

    @property
    def has_successful_slab_chargemol_outputs(self) -> bool:
        top_slab_task_doc = self.top_slab_single_point_task_doc
        bottom_slab_task_doc = self.bottom_slab_single_point_task_doc
        return (
            top_slab_task_doc.is_success
            and top_slab_task_doc.structure is not None
            and bottom_slab_task_doc.is_success
            and bottom_slab_task_doc.structure is not None
        )

    def slab_structures_from_chargemol_or_raise(
        self,
    ) -> tuple[PMG_Structure, PMG_Structure]:
        top_slab_structure_from_chargemol = (
            self.top_slab_single_point_task_doc.structure
        )
        bottom_slab_structure_from_chargemol = (
            self.bottom_slab_single_point_task_doc.structure
        )
        if top_slab_structure_from_chargemol is None:
            raise ValueError(SLAB_CHARGEMOL_NOT_CONVERGED_ERROR)
        if bottom_slab_structure_from_chargemol is None:
            raise ValueError(SLAB_CHARGEMOL_NOT_CONVERGED_ERROR)
        return top_slab_structure_from_chargemol, bottom_slab_structure_from_chargemol


@dataclass(frozen=True)
class _InterfaceRelaxFlowBuilder:
    builder_inputs: _AdhesionInterfaceRelaxFlowBuilderInputs
    slab_chargemol_outputs: _AdhesionInterfaceRelaxFlowSlabChargemolOutputs
    calculation_maker_settings: _AdhesionInterfaceRelaxFlowCalculationMakerSettings

    def _build_slabs_from_slab_chargemol_outputs(self) -> tuple[Slab, Slab]:
        (top_slab_structure_with_charges, bottom_slab_structure_with_charges) = (
            self.slab_chargemol_outputs.slab_structures_from_chargemol_or_raise()
        )
        top_slab_from_single_point = build_slab_from_structure_with_slab_metadata(
            self.builder_inputs.top_slab_input, top_slab_structure_with_charges
        )
        bottom_slab_from_single_point = build_slab_from_structure_with_slab_metadata(
            self.builder_inputs.bottom_slab_input, bottom_slab_structure_with_charges
        )
        return top_slab_from_single_point, bottom_slab_from_single_point

    def _build_interface_relax_batch(self) -> _InterfaceRelaxBatchResult:
        builder_inputs = self.builder_inputs
        calculation_maker_settings = self.calculation_maker_settings
        (top_slab_from_single_point, bottom_slab_from_single_point) = (
            self._build_slabs_from_slab_chargemol_outputs()
        )
        assembled_interfaces_from_charged_slabs = (
            assemble_interfaces_after_slab_chargemol(
                top_slab=top_slab_from_single_point,
                bottom_slab=bottom_slab_from_single_point,
                assembly_settings=builder_inputs.assembly_settings,
            )
        )
        return _build_interface_relax_flows_before_interface_chargemol(
            interfaces=assembled_interfaces_from_charged_slabs,
            interface_jobs_per_batch=builder_inputs.interface_jobs_per_batch,
            interface_relax_maker_settings=(
                calculation_maker_settings.interface_relax_maker_settings
            ),
        )

    def build_interface_relax_jobflow_response(self) -> Response[Flow]:
        """Build the interface-relax branch response from slab Chargemol outputs."""
        slab_chargemol_outputs = self.slab_chargemol_outputs
        calculation_maker_settings = self.calculation_maker_settings
        if not slab_chargemol_outputs.has_successful_slab_chargemol_outputs:
            return Response(stop_children=True)

        interface_relax_batch = self._build_interface_relax_batch()
        interface_chargemol_flow_gate_job = (
            build_interface_chargemol_flow_after_interface_relax(
                slab_relax_dir_names=slab_chargemol_outputs.slab_relax_dir_names,
                slab_single_point_task_docs=(
                    slab_chargemol_outputs.slab_single_point_with_charges_task_docs
                ),
                interface_relax_properties=interface_relax_batch.relax_properties,
                interface_relax_dir_names=interface_relax_batch.relax_dir_names,
                single_point_maker_settings=(
                    calculation_maker_settings.single_point_maker_settings
                ),
            )
        )
        interface_relax_then_chargemol_flow = Flow(
            jobs=[*interface_relax_batch.flows, interface_chargemol_flow_gate_job],
            output=interface_chargemol_flow_gate_job.output,
        )
        return Response(replace=interface_relax_then_chargemol_flow)


def assemble_interfaces_after_slab_chargemol(  # noqa: D103
    *, top_slab: Slab, bottom_slab: Slab, assembly_settings: InterfaceAssemblySettings
) -> list[PMG_Structure]:
    interface_assembler = InterfaceAssembler(top_slab=top_slab, bottom_slab=bottom_slab)
    return list(
        interface_assembler.generate_interfaces(
            offsets=assembly_settings.offsets,
            gaps=assembly_settings.gaps,
            gap_multipliers=assembly_settings.gap_multipliers,
            vacuum=assembly_settings.vacuum,
        )
    )


def _build_interface_relax_flows_before_interface_chargemol(
    *,
    interfaces: Sequence[PMG_Structure],
    interface_jobs_per_batch: int | None,
    interface_relax_maker_settings: RelaxMakerSettings,
) -> _InterfaceRelaxBatchResult:
    interface_job_batches = split_items_into_batches(
        interfaces, interface_jobs_per_batch
    )

    relax_maker = build_relax_maker(relax_maker_settings=interface_relax_maker_settings)
    interface_relax_flows: list[Flow] = []
    interface_relax_properties_outputs_for_chargemol: list[RelaxProperties | None] = []
    interface_relax_dir_names: list[str] = []
    for interface_batch in interface_job_batches:
        if not interface_batch:
            continue
        batch_relax_jobs = [
            relax_maker.make(structure=interface) for interface in interface_batch
        ]
        interface_relax_properties_outputs_for_chargemol.extend(
            [batch_relax_job.output.properties for batch_relax_job in batch_relax_jobs]
        )
        interface_relax_dir_names.extend(
            [batch_relax_job.output.dir_name for batch_relax_job in batch_relax_jobs]
        )
        interface_relax_flows.append(
            Flow(jobs=batch_relax_jobs, output=[job.output for job in batch_relax_jobs])
        )

    return _InterfaceRelaxBatchResult(
        flows=interface_relax_flows,
        relax_properties=interface_relax_properties_outputs_for_chargemol,
        relax_dir_names=interface_relax_dir_names,
    )


def extract_relaxed_slabs_after_slab_relax(  # noqa: D103
    slab_relax_properties: Sequence[RelaxProperties | None],
    slab_relax_not_converged_error: str = SLAB_RELAX_NOT_CONVERGED_ERROR,
) -> tuple[PMG_Structure, PMG_Structure]:
    top_slab_relaxed_structure = extract_relax_final_structure_or_raise(
        slab_relax_properties[0], slab_relax_not_converged_error
    )
    bottom_slab_relaxed_structure = extract_relax_final_structure_or_raise(
        slab_relax_properties[1], slab_relax_not_converged_error
    )
    return top_slab_relaxed_structure, bottom_slab_relaxed_structure


def build_slab_chargemol_jobs_before_interface_relax(  # noqa: D103
    *,
    top_slab_relaxed_structure: PMG_Structure,
    bottom_slab_relaxed_structure: PMG_Structure,
    single_point_maker_settings: SinglePointWithChargesMakerSettings,
) -> tuple[Job | Flow, Job | Flow]:
    single_point_maker = build_single_point_maker(
        single_point_maker_settings=single_point_maker_settings
    )
    top_slab_single_point_job = single_point_maker.make(
        structure=top_slab_relaxed_structure
    )
    bottom_slab_single_point_job = single_point_maker.make(
        structure=bottom_slab_relaxed_structure
    )
    return top_slab_single_point_job, bottom_slab_single_point_job


@job(output_schema=AdhesionWorkflowResult)
def build_adhesion_workflow_result(  # noqa: D103
    slab_relax_dir_names: list[str],
    interface_relax_dir_names: list[str],
    slab_single_point_task_docs: list[SinglePointWithChargesTaskDocument],
    interface_single_point_task_docs: list[SinglePointWithChargesTaskDocument],
) -> AdhesionWorkflowResult:
    return AdhesionWorkflowResult(
        slab_relax_dir_names=slab_relax_dir_names,
        interface_relax_dir_names=interface_relax_dir_names,
        slab_chargemol_dir_names=[doc.dir_name for doc in slab_single_point_task_docs],
        interface_chargemol_dir_names=[
            doc.dir_name for doc in interface_single_point_task_docs
        ],
        adhesion_energy_records=build_adhesion_energy_records_from_single_point_task_docs(
            slab_task_docs=slab_single_point_task_docs,
            interface_task_docs=interface_single_point_task_docs,
        ),
    )


@job
def build_interface_chargemol_flow_after_interface_relax(  # noqa: D103
    *,
    slab_relax_dir_names: list[str],
    slab_single_point_task_docs: list[SinglePointWithChargesTaskDocument],
    interface_relax_properties: list[RelaxProperties | None],
    interface_relax_dir_names: list[str],
    single_point_maker_settings: SinglePointWithChargesMakerSettings,
) -> Response[Flow]:
    if not relax_properties_converged(interface_relax_properties):
        return Response(stop_children=True)

    single_point_maker = build_single_point_maker(
        single_point_maker_settings=single_point_maker_settings
    )
    interface_single_point_jobs_after_relax: list[Job | Flow] = []
    interface_single_point_task_doc_outputs = []
    for relax_properties in interface_relax_properties:
        interface_relaxed_structure = extract_relax_final_structure_or_raise(
            relax_properties, INTERFACE_RELAX_NOT_CONVERGED_ERROR
        )
        interface_single_point_job = single_point_maker.make(
            structure=interface_relaxed_structure
        )
        interface_single_point_jobs_after_relax.append(interface_single_point_job)
        interface_single_point_task_doc_outputs.append(
            interface_single_point_job.output
        )

    adhesion_result_job_from_chargemol = build_adhesion_workflow_result(
        slab_relax_dir_names=slab_relax_dir_names,
        interface_relax_dir_names=interface_relax_dir_names,
        slab_single_point_task_docs=slab_single_point_task_docs,
        interface_single_point_task_docs=interface_single_point_task_doc_outputs,
    )
    interface_chargemol_then_adhesion_flow = Flow(
        jobs=[
            *interface_single_point_jobs_after_relax,
            adhesion_result_job_from_chargemol,
        ],
        output=adhesion_result_job_from_chargemol.output,
    )
    return Response(replace=interface_chargemol_then_adhesion_flow)


@job
def build_interface_relax_flow_after_slab_chargemol(  # noqa: D103
    *,
    slab_relax_dir_names: list[str],
    top_slab_single_point_task_doc: SinglePointWithChargesTaskDocument,
    bottom_slab_single_point_task_doc: SinglePointWithChargesTaskDocument,
    top_slab_input: Slab,
    bottom_slab_input: Slab,
    assembly_settings: InterfaceAssemblySettings,
    interface_relax_maker_settings: RelaxMakerSettings,
    single_point_maker_settings: SinglePointWithChargesMakerSettings,
) -> Response[Flow]:
    builder = _InterfaceRelaxFlowBuilder(
        builder_inputs=_AdhesionInterfaceRelaxFlowBuilderInputs(
            top_slab_input=top_slab_input,
            bottom_slab_input=bottom_slab_input,
            assembly_settings=assembly_settings,
        ),
        slab_chargemol_outputs=_AdhesionInterfaceRelaxFlowSlabChargemolOutputs(
            slab_relax_dir_names=slab_relax_dir_names,
            top_slab_single_point_task_doc=top_slab_single_point_task_doc,
            bottom_slab_single_point_task_doc=bottom_slab_single_point_task_doc,
        ),
        calculation_maker_settings=_AdhesionInterfaceRelaxFlowCalculationMakerSettings(
            interface_relax_maker_settings=interface_relax_maker_settings,
            single_point_maker_settings=single_point_maker_settings,
        ),
    )
    return builder.build_interface_relax_jobflow_response()


@job
def build_slab_chargemol_flow_after_slab_relax(  # noqa: D103
    *,
    slab_relax_properties: list[RelaxProperties | None],
    slab_relax_dir_names: list[str],
    top_slab_input: Slab,
    bottom_slab_input: Slab,
    assembly_settings: InterfaceAssemblySettings,
    interface_relax_maker_settings: RelaxMakerSettings,
    single_point_maker_settings: SinglePointWithChargesMakerSettings,
) -> Response[Flow]:
    if not relax_properties_converged(slab_relax_properties):
        return Response(stop_children=True)

    (top_slab_relaxed_structure, bottom_slab_relaxed_structure) = (
        extract_relaxed_slabs_after_slab_relax(slab_relax_properties)
    )
    (top_slab_single_point_job, bottom_slab_single_point_job) = (
        build_slab_chargemol_jobs_before_interface_relax(
            top_slab_relaxed_structure=top_slab_relaxed_structure,
            bottom_slab_relaxed_structure=bottom_slab_relaxed_structure,
            single_point_maker_settings=single_point_maker_settings,
        )
    )

    interface_relax_flow_after_slab_chargemol_gate_job = (
        build_interface_relax_flow_after_slab_chargemol(
            slab_relax_dir_names=slab_relax_dir_names,
            top_slab_single_point_task_doc=top_slab_single_point_job.output,
            bottom_slab_single_point_task_doc=bottom_slab_single_point_job.output,
            top_slab_input=top_slab_input,
            bottom_slab_input=bottom_slab_input,
            assembly_settings=assembly_settings,
            interface_relax_maker_settings=interface_relax_maker_settings,
            single_point_maker_settings=single_point_maker_settings,
        )
    )
    slab_chargemol_then_interface_relax_flow = Flow(
        jobs=[
            top_slab_single_point_job,
            bottom_slab_single_point_job,
            interface_relax_flow_after_slab_chargemol_gate_job,
        ],
        output=interface_relax_flow_after_slab_chargemol_gate_job.output,
    )
    return Response(replace=slab_chargemol_then_interface_relax_flow)


@dataclass(kw_only=True)
class AdhesionWorkflowMaker(BaseMaker):
    """Builds an adhesion workflow ``Flow`` from user inputs.

    Attributes:
        simulation_settings: Calculator and runtime settings for the workflow.
            ``slab_relax_maker_settings`` configures slab relax jobs.
            ``interface_relax_maker_settings`` configures interface relax jobs.
            ``single_point_maker_settings`` configures single-point-with-charges jobs.

    Notes:
        - ``make(inputs)`` returns a ``jobflow.Flow`` whose terminal output is
          ``AdhesionWorkflowResult``.
    """

    simulation_settings: AdhesionSimulationSettings

    @property
    def slab_relax_maker(self) -> RelaxMaker:
        """Return the configured maker used for the initial slab relax jobs."""
        return build_relax_maker(
            relax_maker_settings=self.simulation_settings.slab_relax_maker_settings
        )

    def make(self, inputs: AdhesionWorkflowInputs) -> Flow:  # noqa: D102
        slab_relax_maker = self.slab_relax_maker
        top_slab_relax_job = slab_relax_maker.make(structure=inputs.top_slab)
        bottom_slab_relax_job = slab_relax_maker.make(structure=inputs.bottom_slab)
        slab_relax_to_chargemol_flow_gate_job = (
            build_slab_chargemol_flow_after_slab_relax(
                slab_relax_properties=[
                    top_slab_relax_job.output.properties,
                    bottom_slab_relax_job.output.properties,
                ],
                slab_relax_dir_names=[
                    top_slab_relax_job.output.dir_name,
                    bottom_slab_relax_job.output.dir_name,
                ],
                top_slab_input=inputs.top_slab,
                bottom_slab_input=inputs.bottom_slab,
                assembly_settings=inputs.assembly_settings,
                interface_relax_maker_settings=(
                    self.simulation_settings.interface_relax_maker_settings
                ),
                single_point_maker_settings=(
                    self.simulation_settings.single_point_maker_settings
                ),
            )
        )
        return Flow(
            jobs=[
                top_slab_relax_job,
                bottom_slab_relax_job,
                slab_relax_to_chargemol_flow_gate_job,
            ],
            output=slab_relax_to_chargemol_flow_gate_job.output,
        )
