"""Maker for direct sampling."""

import logging
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from jobflow.core.job import Job, job
from maml.sampling.clustering import BirchClustering
from maml.sampling.direct import DIRECTSampler, SelectKFromClusters
from pymatgen.core.structure import Structure as PMG_Structure
from sklearn.preprocessing import StandardScaler

from crystal.constants.structure_generators import STRUCTURE_BLOB_STORE_NAME
from crystal.helpers.exception import get_diagnostic_exception_message
from crystal.schemas.base import BaseMaker
from crystal.schemas.maker_names import MakerName
from crystal.schemas.properties import StructureGenerationProperties
from crystal.schemas.structure_generators import (
    DirectSamplingSettings,
    DirectSamplingTaskDocument,
)
from crystal.structure.generators.extxyz import write_generated_structures_extxyz
from crystal.structure.generators.sampling.qet_structure_encoder import (
    QetStructureEncoder,
)

logger = logging.getLogger(__name__)

PCA_TRANSFORMER_NAME = "PrincipalComponentAnalysis"
SELECTED_INDEXES_KEY = "selected_indexes"
EMPTY_STRUCTURE_INPUT_WARNING = (
    "DIRECT sampling received no structures; returning an empty selection."
)


@dataclass(kw_only=True)
class DirectSamplingMaker(BaseMaker):  # noqa: D101
    name: MakerName = MakerName.direct_sampling
    simulation_settings: DirectSamplingSettings

    @cached_property
    def _structure_encoder(self) -> QetStructureEncoder:
        return QetStructureEncoder(
            model=self.simulation_settings.structure_encoder_model
        )

    @cached_property
    def _embedding_normalization(self) -> StandardScaler:
        settings = self.simulation_settings.embedding_normalization_settings
        return StandardScaler(
            copy=settings.allow_copy,
            with_mean=settings.translate_with_mean,
            with_std=settings.scale_with_std,
        )

    @cached_property
    def _birch_clustering(self) -> BirchClustering:
        settings = self.simulation_settings.birch_clustering_settings
        return BirchClustering(
            n=settings.number_of_clusters,
            threshold_init=settings.clustering_threshold_radius,
        )

    def _build_cluster_sampling(
        self, number_of_atoms_per_structure: list[int]
    ) -> SelectKFromClusters:
        settings = self.simulation_settings.cluster_sampling_settings
        return SelectKFromClusters(
            k=settings.num_samples_per_cluster,
            allow_duplicate=settings.allow_duplicate,
            selection_criteria=settings.sampling_mode,
            n_sites=number_of_atoms_per_structure,
        )

    def _build_direct_sampler(
        self, cluster_sampling: SelectKFromClusters
    ) -> DIRECTSampler:
        return DIRECTSampler(
            structure_encoder=self._structure_encoder,
            scaler=self._embedding_normalization,
            pca=PCA_TRANSFORMER_NAME,
            weighting_PCs=self.simulation_settings.weight_principal_components_with_explained_variances,
            clustering=self._birch_clustering,
            select_k_from_clusters=cluster_sampling,
        )

    @job(
        output_schema=DirectSamplingTaskDocument,
        **{STRUCTURE_BLOB_STORE_NAME: [["output", "properties", "structures"]]},
    )  # type: ignore[call-overload]
    def _run(self, structures: list[PMG_Structure]) -> DirectSamplingTaskDocument:
        run_directory = Path.cwd()
        logger.info("Running DIRECT sampling from %s structures", len(structures))
        if not structures:
            logger.warning(EMPTY_STRUCTURE_INPUT_WARNING)
            return DirectSamplingTaskDocument(
                dir_name=run_directory.as_posix(),
                properties=StructureGenerationProperties(structures=[]),
                warnings=EMPTY_STRUCTURE_INPUT_WARNING,
            )

        try:
            number_of_atoms_per_structure = [len(structure) for structure in structures]
            cluster_sampling = self._build_cluster_sampling(
                number_of_atoms_per_structure
            )
            direct_sampler = self._build_direct_sampler(cluster_sampling)
            direct_selection = direct_sampler.fit_transform(structures)
            selected_indexes = direct_selection[SELECTED_INDEXES_KEY]
            selected_structures = [structures[i] for i in selected_indexes]
            properties = StructureGenerationProperties(structures=selected_structures)
            errors = ""
        except Exception as exc:
            properties = None
            errors = get_diagnostic_exception_message(exc)
            logger.exception("DIRECT sampling failed.")
        else:
            assert properties is not None
            write_generated_structures_extxyz(
                structures=properties.structures,
                run_directory=run_directory,
                failure_message="Extxyz export for DIRECT sampling failed.",
            )

        task_document = DirectSamplingTaskDocument(
            dir_name=run_directory.as_posix(), properties=properties, errors=errors
        )
        logger.info("Finished DIRECT sampling.")
        return task_document

    def make(self, structures: list[PMG_Structure]) -> Job:  # noqa: D102
        structure_job: Job = self._run(structures=structures)
        structure_job.update_metadata(
            self.slurm_settings.to_queueadapter_spec(), dynamic=False
        )
        return structure_job
