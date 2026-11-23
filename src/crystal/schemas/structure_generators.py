"""Pydantic models for structure-generation workflows."""

from __future__ import annotations

from enum import StrEnum, auto
from typing import Self

from pydantic import ConfigDict, Field, PositiveFloat, PositiveInt, model_validator

from crystal.schemas.base import BaseTaskDocument
from crystal.schemas.filtering import (
    CohesiveEnergyFilteringSettings,
    CohesiveEnergyFilteringTaskDocument,
    GeometricStructureFilteringSettings,
)
from crystal.schemas.filtering_statistics import CohesiveEnergyFilteringStatistics
from crystal.schemas.initial_structure_generation import (
    InitialStructureGenerationSettings,
)
from crystal.schemas.mlip import (
    MlipRelaxMakerSettings,
    MlipRelaxTaskDocument,
    MlipSinglePointMakerSettings,
)
from crystal.schemas.mlip_model_ref import MlipModelRef
from crystal.schemas.randomness import RandomSeed, random_seed_field
from crystal.schemas.settings import MakerSettings, SimulationSettings
from crystal.schemas.structure_augmentation import (
    StructureAugmentationAndFilteringStatistics,
)
from crystal.schemas.structure_labelling import StructureLabellingWorkflowSettings
from crystal.structure.generators.perturbation.rattle import (
    DEFAULT_ATOM_POSITION_NOISE_SCALING_FACTOR,
    RattleAtomDisplacementMode,
)
from crystal.structure.generators.perturbation.stretch import (
    DEFAULT_MAX_NORMAL_STRAIN_FRACTION,
    DEFAULT_MAX_SHEAR_STRAIN_FRACTION,
    MINIMUM_STRAIN_FRACTION,
)


class InitialStructureGenerationTaskDocument(BaseTaskDocument):  # noqa: D101
    pass


class RattleAtomPositionsSettings(SimulationSettings):
    """Settings used by the rattle atom-position perturbation runner."""

    random_seed: RandomSeed = random_seed_field()
    noise_scaling_factor: PositiveFloat = DEFAULT_ATOM_POSITION_NOISE_SCALING_FACTOR
    displacement_mode: RattleAtomDisplacementMode = RattleAtomDisplacementMode.normal


class StretchCellSettings(SimulationSettings):
    """Settings used by the stretch-cell perturbation runner."""

    random_seed: RandomSeed = random_seed_field()
    max_normal_strain_fraction: PositiveFloat = Field(
        default=DEFAULT_MAX_NORMAL_STRAIN_FRACTION, ge=MINIMUM_STRAIN_FRACTION
    )
    max_shear_strain_fraction: PositiveFloat = Field(
        default=DEFAULT_MAX_SHEAR_STRAIN_FRACTION, ge=MINIMUM_STRAIN_FRACTION
    )


class StructureAugmentationAndFilteringSettings(SimulationSettings):
    """Settings for structure augmentation and geometric filtering."""

    model_config = ConfigDict(extra="forbid")

    number_of_rattled_variants: PositiveInt = 4
    number_of_stretched_variants: PositiveInt = 8
    number_of_stretched_variants_after_rattle: PositiveInt = 1
    rattle_settings: RattleAtomPositionsSettings = Field(
        default_factory=RattleAtomPositionsSettings
    )
    stretch_after_rattle_settings: StretchCellSettings = Field(
        default_factory=StretchCellSettings
    )
    stretch_after_mlip_relax_settings: StretchCellSettings = Field(
        default_factory=StretchCellSettings
    )
    geometric_filtering_settings: GeometricStructureFilteringSettings = Field(
        default_factory=GeometricStructureFilteringSettings
    )
    cohesive_energy_filtering_settings: CohesiveEnergyFilteringSettings | None = None


class StructureAugmentationAndFilteringTaskDocument(BaseTaskDocument):
    """Task document for structure augmentation and geometric filtering."""

    warnings: str = Field(
        default="",
        description="Non-fatal warnings from per-root augmentation failures.",
    )


class StructureAugmentationAndFilteringMakerSettings(MakerSettings):  # noqa: D101
    simulation_settings: StructureAugmentationAndFilteringSettings = Field(
        default_factory=StructureAugmentationAndFilteringSettings
    )


class PostAugmentationCohesiveEnergyFilteringSettings(SimulationSettings):
    """Settings for post-augmentation cohesive-energy filtering.

    When isolated-atom energies are derived from an evaluator model, that model
    must also be the model used for the single-point total-energy evaluations.
    """

    cohesive_energy_filtering_settings: CohesiveEnergyFilteringSettings = Field(
        default_factory=CohesiveEnergyFilteringSettings
    )
    mlip_single_point_maker_settings: MlipSinglePointMakerSettings
    structures_per_job: PositiveInt | None = Field(
        default=500,
        description="Maximum number of structures evaluated in one MLIP job.",
    )

    @model_validator(mode="after")
    def _validate_cohesive_energy_references(self) -> Self:
        cohesive_energy_settings = self.cohesive_energy_filtering_settings
        if cohesive_energy_settings.maximum_cohesive_energy_per_atom is None:
            msg = (
                "Post-augmentation cohesive-energy filtering requires "
                "maximum_cohesive_energy_per_atom."
            )
            raise ValueError(msg)
        if cohesive_energy_settings.isolated_atom_energies is not None:
            return self
        evaluator_model = cohesive_energy_settings.evaluator_model
        single_point_model = (
            self.mlip_single_point_maker_settings.simulation_settings.model
        )
        if evaluator_model != single_point_model:
            msg = (
                "Post-augmentation cohesive-energy filtering evaluator_model must "
                "match the MLIP single-point model when isolated_atom_energies "
                "are not provided."
            )
            raise ValueError(msg)
        return self


class InitialStructureGenerationMakerSettings(MakerSettings):  # noqa: D101
    simulation_settings: InitialStructureGenerationSettings = Field(
        default_factory=InitialStructureGenerationSettings
    )


class EmbeddingNormalizationSettings(SimulationSettings):
    """Settings for sklearn StandardScaler for DIRECT sampling via MaML."""

    allow_copy: bool = Field(
        default=True,
        description=(
            "If False, try to avoid a copy and do inplace scaling instead. "
            "This is not guaranteed to always work inplace; e.g. if the data "
            "is not a NumPy array or scipy.sparse CSR matrix, a copy may "
            "still be returned."
        ),
    )
    translate_with_mean: bool = Field(
        default=True,
        description=(
            "If True, center the data before scaling. This does not work "
            "(and will raise an exception) when attempted on sparse matrices, "
            "because centering them entails building a dense matrix which in "
            "common use cases is likely to be too large to fit in memory."
        ),
    )
    scale_with_std: bool = Field(
        default=True,
        description=(
            "If True, scale the data to unit variance "
            "(or equivalently, unit standard deviation)."
        ),
    )


class BirchClusteringSettings(SimulationSettings):
    """Settings for MaML BirchClustering for DIRECT sampling."""

    number_of_clusters: PositiveInt | None = Field(
        default=None,
        description=(
            "Clustering the PCs into n clusters. When n is None, the number "
            "of clusters is dependent on threshold_init and other kwargs, and "
            "the final (global) clustering step is skipped. `n` in MaML."
        ),
    )
    clustering_threshold_radius: PositiveFloat = Field(
        default=0.5,
        description=(
            "The initial radius of the subcluster obtained by merging a new "
            "sample and the closest subcluster should be lesser than the "
            "threshold. Otherwise, a new subcluster is started. See details "
            "in: "
            "https://scikit-learn.org/stable/modules/generated/"
            "sklearn.cluster.Birch.html. Users may tune this value for "
            "desired performance of birch, while 0.5 is generally a good "
            "starting point, and some automatic tuning is done with our "
            "built-in codes to achieve n clusters if given. `threshold_init` "
            "in MaML"
        ),
    )


class ClusterSamplingMode(StrEnum):
    """An enum class holding all the available cluster sampling modes.
    ranked_from_closest_to_centroid: structures are ranked with respect to their distances to the centroid of the cluster they locate, then up to k structures with fixed ranking intervals are selected from each cluster, and when k=1, the structure with the smallest Euclidean distance to the centroid of each cluster is sampled.
    ranked_from_fewest_atoms: select the k structures with the least number of atoms in each cluster.
    random: k structures are randomly sampled, `allow_duplicate` in `ClusterSamplingSettings` controls if replacement is permitted.
    """

    ranked_from_closest_to_centroid = "center"
    ranked_from_fewest_atoms = "smallest"
    random = auto()


class ClusterSamplingSettings(SimulationSettings):
    """Wrapper for settings for MaML's SelectKFromClusters for DIRECT sampling."""

    num_samples_per_cluster: PositiveInt = Field(
        default=1,
        description=(
            "Number of samples to select from each cluster; higher values "
            "will yield more samples but with lower diversity. `k` in MaML."
        ),
    )
    allow_duplicate: bool = Field(
        default=False,
        description=("Whether structures are allowed to be selected more than once."),
    )
    sampling_mode: ClusterSamplingMode = Field(
        default=ClusterSamplingMode.ranked_from_closest_to_centroid,
        description=(
            "The criteria to do stratified sampling from each cluster. "
            "Supported criterion include "
            "'ranked_from_closest_to_centroid' ('center' in MaML, default), "
            "'ranked_from_fewest_atoms' ('smallest' in MaML), and `random`."
        ),
    )


class DirectSamplingSettings(SimulationSettings):
    """Settings for DIRECT sampling via MaML; note that only the MatGL structure encoder is supported.
    Attribute descriptions (included for nested settings) come directly from either MaML or scikit-learn, with minimal changes to account for any renamings.
    Attributes are set using the names defined here.
    """

    structure_encoder_model: MlipModelRef = Field(
        description=(
            "MatGL model defined by path to the model or QetModelRegistry for creating structure embeddings."
        )
    )
    embedding_normalization_settings: EmbeddingNormalizationSettings = Field(
        default_factory=EmbeddingNormalizationSettings,
        description=(
            "Settings for embedding normalization before principle component "
            "analysis. `StandardScaler` in MaML."
        ),
    )
    weight_principal_components_with_explained_variances: bool = Field(
        default=True,
        description=(
            "Whether to weight principle components with their explained "
            "variance. `weighting_PCs` in MaML."
        ),
    )
    birch_clustering_settings: BirchClusteringSettings = Field(
        default_factory=BirchClusteringSettings,
        description=("Settings for the Birch clustering stage."),
    )
    cluster_sampling_settings: ClusterSamplingSettings = Field(
        default_factory=ClusterSamplingSettings,
        description=("Settings for the sampling of structures from each cluster."),
    )


class DirectSamplingMakerSettings(MakerSettings):  # noqa: D101
    simulation_settings: DirectSamplingSettings


class DatasetGenerationWorkflowSettings(SimulationSettings):
    """Settings for the overall dataset generation workflow."""

    model_config = ConfigDict(extra="forbid")

    number_of_initial_structures: PositiveInt
    mlip_relax_structures_per_job: PositiveInt | None = Field(
        default=500,  # Vibes based currently, subject to change based on future testing and benchmarking
        description="Maximum number of structures relaxed in one MLIP job.",
    )
    initial_structure_generation_maker_settings: InitialStructureGenerationMakerSettings = Field(
        default_factory=InitialStructureGenerationMakerSettings
    )
    mlip_relax_maker_settings: MlipRelaxMakerSettings
    structure_augmentation_and_filtering_maker_settings: StructureAugmentationAndFilteringMakerSettings = Field(
        default_factory=StructureAugmentationAndFilteringMakerSettings
    )
    post_augmentation_cohesive_energy_filtering_settings: (
        PostAugmentationCohesiveEnergyFilteringSettings | None
    ) = None
    direct_sampling_maker_settings: DirectSamplingMakerSettings
    structure_labelling_workflow_settings: StructureLabellingWorkflowSettings | None = (
        None
    )


class DatasetGenerationWorkflowTaskDocument(BaseTaskDocument):  # noqa: D101
    warnings: str = Field(
        default="",
        description=(
            "Non-fatal workflow warnings (for example, partial upstream failures)."
        ),
    )
    augmentation_and_filtering_statistics: (
        StructureAugmentationAndFilteringStatistics | None
    ) = Field(
        default=None,
        description="Candidate counts from structure augmentation and filtering.",
    )
    post_augmentation_cohesive_energy_filtering_statistics: (
        CohesiveEnergyFilteringStatistics | None
    ) = Field(
        default=None,
        description=(
            "Candidate counts from post-augmentation cohesive-energy filtering."
        ),
    )


class DirectSamplingTaskDocument(BaseTaskDocument):  # noqa: D101
    warnings: str = Field(
        default="",
        description=(
            "Non-fatal workflow warnings (for example, partial upstream failures)."
        ),
    )


DatasetGenerationTaskDocUnion = (
    InitialStructureGenerationTaskDocument
    | MlipRelaxTaskDocument
    | StructureAugmentationAndFilteringTaskDocument
    | CohesiveEnergyFilteringTaskDocument
    | DirectSamplingTaskDocument
)
