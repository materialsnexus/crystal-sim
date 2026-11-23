# noqa: D100
from enum import StrEnum, auto


class MakerName(StrEnum):  # noqa: D101
    base = auto()
    gpaw_relax = auto()
    gpaw_isolated_atom = auto()
    gpaw_scf_chargemol_integration = auto()
    gpaw_chargemol_integration = auto()
    qe_chargemol_integration = auto()
    qe_relax = auto()
    mlip_single_point = auto()
    mlip_single_point_batch = auto()
    mlip_relax = auto()
    mlip_relax_batch = auto()
    qet_finetuning = auto()
    mlip_md_equilibration = auto()
    md_adhesion_energy = auto()
    interface_geometry_optimisation = auto()
    initial_structure_generation = auto()
    direct_sampling = auto()
    structure_augmentation_and_filtering = auto()
    single_point_filter = auto()
    dataset_generation_workflow = auto()
    validation_dataset_md = auto()
    validation_dataset_workflow = auto()
    structure_labelling_workflow = auto()
