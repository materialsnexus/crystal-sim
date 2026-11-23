import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fireworks import LaunchPad
from fireworks.core.fworker import FWorker
from fireworks.core.rocket_launcher import launch_rocket, rapidfire
from fireworks.queue.queue_launcher import rapidfire as queue_rapidfire
from fireworks.user_objects.dupefinders.dupefinder_exact import DupeFinderExact
from fireworks.utilities.fw_serializers import load_object_from_file
from jobflow.core.flow import Flow
from jobflow.core.store import JobStore
from jobflow.managers.fireworks import flow_to_workflow, job_to_firework
from jobflow.managers.local import run_locally
from monty.json import MontyDecoder

from crystal import LOCAL_FWORKER_YAML, SLURM_QADAPTER_YAML
from crystal.calculations import (
    AddAndSaveFileMaker,
    AdditionMaker,
    AddThenCube,
    AddThenFilteredCubeMaker,
    CubeMaker,
    ExternalExecutableRunMaker,
    FileSchema,
    SubtractMaker,
    addition,
)
from crystal.firework_helpers import (
    run_rapidfire_launcher,
    workflow_reached_terminal_state,
)
from crystal.helpers.fireworks.dupe_finder import DEDUPE_KEY
from crystal.helpers.fireworks.manager import JobManager, assign_dedupe_values
from crystal.helpers.job import compute_dedupe
from crystal.helpers.temp import run_in_temp_dir
from tests.helpers.test_paths import (
    TEST_FILE_HELPERS_DATA_DIRNAME,
    TEST_FIREWORKS_SUBMIT_SCRIPT_FILENAME,
    TEST_VARS_SCRIPT_FILENAME,
)

TEST_BASH_FILE = (
    Path(__file__).parent.resolve()
    / TEST_FILE_HELPERS_DATA_DIRNAME
    / TEST_VARS_SCRIPT_FILENAME
)
TEST_FWORKER = FWorker.from_file(LOCAL_FWORKER_YAML.as_posix())


def test_addition() -> None:
    assert addition(1, 1) == 2


@run_in_temp_dir
def test_makers_with_jobflow() -> None:
    add_maker = AdditionMaker()
    add_job = add_maker.make(1, 2)

    cube_maker = CubeMaker()
    cube_job = cube_maker.make(add_job.output["sum"])

    subtract_maker = SubtractMaker()
    subtract_job = subtract_maker.make(add_job.output["sum"], cube_job.output)

    flow = Flow(jobs=[add_job, cube_job, subtract_job], output=subtract_job.output)
    run_flow = run_locally(flow)
    assert isinstance(run_flow, dict)
    assert run_flow[subtract_job.uuid][1].output == -24


@run_in_temp_dir
def test_flow_with_simple_makers() -> None:
    add_then_cube_maker = AddThenCube()
    flow = add_then_cube_maker.make(1, 2)
    run_flow = run_locally(flow)
    [*_, last_job_uuid] = list(run_flow.keys())
    assert run_flow[last_job_uuid][1].output == 27


@run_in_temp_dir
@pytest.mark.needs_mongodb
def test_flow_with_filtering(mock_jobstore: JobStore) -> None:
    maker = AddThenFilteredCubeMaker()
    flow = maker.make(5, 7, 8)
    run_flow = run_locally(flow, store=mock_jobstore)
    assert (
        len(run_flow) == 6
    )  # one flow to tie everything, one job for addition, and 4 jobs (9, 10, 11, 12) for cube


@pytest.mark.needs_mongodb
@run_in_temp_dir
def test_makers_with_fireworks_single_job(
    mock_launchpad: LaunchPad, mock_jobstore: JobStore
) -> None:
    addition_maker = AdditionMaker()
    add_job = addition_maker.make(1, 2)

    fw = job_to_firework(add_job, store=mock_jobstore)
    mock_launchpad.add_wf(fw)

    launch_rocket(launchpad=mock_launchpad, fworker=TEST_FWORKER)
    [fw_id] = mock_launchpad.get_fw_ids()
    assert fw_id == 1
    uuid = mock_launchpad.get_fw_dict_by_id(fw_id)["spec"]["_tasks"][0]["job"]["uuid"]
    [job_doc] = list(mock_jobstore.docs_store.query({"uuid": uuid}))
    assert job_doc["output"]["sum"] == 3


@pytest.mark.needs_mongodb
@run_in_temp_dir
def test_makers_with_fireworks_and_local_rapidfire(
    mock_launchpad: LaunchPad, mock_jobstore: JobStore
) -> None:
    # always reset the launchpad - can be a fixture for the tests
    maker = AddThenFilteredCubeMaker()
    flow = maker.make(5, 7, 8)
    fw = flow_to_workflow(flow, store=mock_jobstore)

    mock_launchpad.add_wf(fw)

    rapidfire(launchpad=mock_launchpad, fworker=TEST_FWORKER)

    fw_ids = sorted(mock_launchpad.get_fw_ids())
    statuses = [mock_launchpad.get_fw_dict_by_id(fw_id)["state"] for fw_id in fw_ids]
    assert len(statuses) == 7
    for status in statuses:
        assert status == "COMPLETED"


# Tests with SLURM are flaky since jobs do not always go to running in slurm queue
@pytest.mark.needs_mongodb
@pytest.mark.needs_slurm
@run_in_temp_dir
def test_makers_with_fireworks_and_queue_adapters(
    mock_launchpad: LaunchPad, mock_jobstore: JobStore
) -> None:
    maker = AddThenFilteredCubeMaker()
    flow = maker.make(5, 7, 8)
    fw = flow_to_workflow(flow, store=mock_jobstore)
    mock_launchpad.add_wf(fw)

    qadapter = load_object_from_file(SLURM_QADAPTER_YAML.as_posix())
    queue_rapidfire(launchpad=mock_launchpad, fworker=TEST_FWORKER, qadapter=qadapter)

    anchor_firework_id = fw.fws[0].fw_id
    while True:
        finished = workflow_reached_terminal_state(anchor_firework_id, mock_launchpad)
        if finished:
            break

    fw_ids = sorted(mock_launchpad.get_fw_ids())
    statuses = [mock_launchpad.get_fw_dict_by_id(fw_id)["state"] for fw_id in fw_ids]
    assert len(statuses) == 7
    for status in statuses:
        assert status == "COMPLETED"


@pytest.mark.needs_mongodb
@pytest.mark.needs_slurm
@run_in_temp_dir
def test_check_output_from_jobstore(
    mock_launchpad: LaunchPad, mock_jobstore: JobStore
) -> None:
    maker = AddThenFilteredCubeMaker()
    flow = maker.make(5, 7, 8)
    fw = flow_to_workflow(flow, store=mock_jobstore)

    mock_launchpad.add_wf(fw)
    qadapter = load_object_from_file(SLURM_QADAPTER_YAML.as_posix())
    queue_rapidfire(launchpad=mock_launchpad, fworker=TEST_FWORKER, qadapter=qadapter)

    fw_ids = sorted(mock_launchpad.get_fw_ids())

    fireworks = [mock_launchpad.get_fw_by_id(fw_id) for fw_id in fw_ids]
    uuids = [fw.spec["_tasks"][0]["job"].uuid for fw in fireworks]
    docs_from_jobstore = list(mock_jobstore.docs_store.query({"uuid": {"$in": uuids}}))
    all_results = [
        doc["output"]
        for doc in docs_from_jobstore
        if doc["output"] and not isinstance(doc["output"], list)
    ]
    [sum_result] = [r for r in all_results if isinstance(r, dict)]
    filtered_cube_result = [r for r in all_results if not isinstance(r, dict)]
    assert sum_result["sum"] == 12
    assert sorted(filtered_cube_result) == [729, 1000, 1331, 1728]


@pytest.mark.needs_mongodb
@pytest.mark.needs_slurm
@run_in_temp_dir
def test_saving_file_in_local_filestore(
    mock_launchpad: LaunchPad, mock_jobstore: JobStore
) -> None:
    maker = AddAndSaveFileMaker()
    flow = maker.make(5, 7)
    fw = flow_to_workflow(flow, store=mock_jobstore)

    mock_launchpad.add_wf(fw)
    qadapter = load_object_from_file(SLURM_QADAPTER_YAML.as_posix())
    queue_rapidfire(launchpad=mock_launchpad, fworker=TEST_FWORKER, qadapter=qadapter)

    fw_ids = sorted(mock_launchpad.get_fw_ids())
    fireworks = sorted(
        [mock_launchpad.get_fw_by_id(fw_id) for fw_id in fw_ids],
        key=lambda x: x.updated_on,
    )
    [_, uuid_save_file_job] = [fw.spec["_tasks"][0]["job"].uuid for fw in fireworks]
    file_output_as_dict = mock_jobstore.get_output(uuid_save_file_job, load=True)
    file_output = MontyDecoder().process_decoded(file_output_as_dict)
    assert isinstance(file_output, FileSchema)
    assert file_output.data == "12"


@pytest.mark.needs_mongodb
@pytest.mark.needs_slurm
@run_in_temp_dir
def test_environment_setup_and_external_executable_with_flow(
    mock_launchpad: LaunchPad, mock_jobstore: JobStore
) -> None:
    # we use qadapter, add a prerocket and check if the environment variable is set
    qadapter = load_object_from_file(SLURM_QADAPTER_YAML.as_posix())
    qadapter["pre_rocket"] = f"source {TEST_BASH_FILE.as_posix()}"

    external_executable_job = ExternalExecutableRunMaker().make(ntasks=4)

    fw = job_to_firework(external_executable_job, store=mock_jobstore)
    mock_launchpad.add_wf(fw)
    queue_rapidfire(
        launchpad=mock_launchpad, fworker=TEST_FWORKER, qadapter=qadapter, nlaunches=1
    )

    # look at the launch directory and read slurm script if it has a the appropriate command

    fw_ids = sorted(mock_launchpad.get_fw_ids())
    while True:
        workflow = mock_launchpad.get_wf_by_fw_id(fw_ids[0])
        if all(v == "COMPLETED" for v in list(workflow.fw_states.values())):
            break
        time.sleep(2)

    slurm_script = (
        Path(workflow.to_dict()["fws"][0]["launches"][0]["launch_dir"]).resolve().parent
        / TEST_FIREWORKS_SUBMIT_SCRIPT_FILENAME
    )
    assert qadapter["pre_rocket"] in slurm_script.read_text(encoding="utf-8")
    fireworks = [mock_launchpad.get_fw_by_id(fw_id) for fw_id in fw_ids]
    uuids = [fw.spec["_tasks"][0]["job"].uuid for fw in fireworks]
    [docs_from_jobstore] = list(
        mock_jobstore.docs_store.query({"uuid": {"$in": uuids}})
    )
    assert [
        rank_info in docs_from_jobstore["output"]["stdout"]
        for rank_info in ["rank=0", "rank=1", "rank=2", "rank=3"]
    ]


@patch("crystal.helpers.fireworks.manager.assign_dedupe_values")
@run_in_temp_dir
def test_duplicate_job_with_dupe_finder_exact(
    mock_assign_dedupe_values: MagicMock,
    mock_jobstore: JobStore,
    mock_launchpad: LaunchPad,
) -> None:
    exact_dupe_finder = DupeFinderExact()

    mock_assign_dedupe_values.return_value = None
    job = AdditionMaker().make(a=1, b=2)
    duplicate_job = AdditionMaker().make(a=1, b=2)
    manager = JobManager(
        launchpad=mock_launchpad, jobstore=mock_jobstore, fireworker=TEST_FWORKER
    )
    manager.add_to_launchpad(job, dupe_finder=exact_dupe_finder)
    manager.add_to_launchpad(duplicate_job, dupe_finder=exact_dupe_finder)
    run_rapidfire_launcher(mock_launchpad, TEST_FWORKER)
    fw_ids = mock_launchpad.get_fw_ids()
    num_launches = len(
        {mock_launchpad.get_fw_by_id(fw_id).launches[0].launch_dir for fw_id in fw_ids}
    )
    # The DupeFinderExact fails to identify two identical jobs as identical
    assert len(fw_ids) == 2
    assert num_launches == 2


def test_dedupe_in_jobs() -> None:
    job = AdditionMaker().make(a=1, b=2)
    duplicate_job = AdditionMaker().make(a=1, b=2)
    duplicate_job_with_different_maker_name = AdditionMaker(name="new_maker").make(
        a=1, b=2
    )
    assert compute_dedupe(job) == compute_dedupe(duplicate_job)
    assert compute_dedupe(job) != compute_dedupe(
        duplicate_job_with_different_maker_name
    )


@run_in_temp_dir
def test_duplicate_jobs_are_run_only_once(
    mock_jobstore: JobStore, mock_launchpad: LaunchPad
) -> None:
    job = AdditionMaker().make(1, 2)
    duplicate_job = AdditionMaker().make(a=1, b=2)
    manager = JobManager(
        launchpad=mock_launchpad, jobstore=mock_jobstore, fireworker=TEST_FWORKER
    )

    manager.add_to_launchpad(job)
    manager.add_to_launchpad(duplicate_job)

    run_rapidfire_launcher(mock_launchpad, TEST_FWORKER)
    fw_ids = mock_launchpad.get_fw_ids()
    num_launches = len(
        {mock_launchpad.get_fw_by_id(fw_id).launches[0].launch_dir for fw_id in fw_ids}
    )

    assert len(fw_ids) == 2
    assert num_launches == 1


def test_dedupe_in_flows() -> None:
    flow = AdditionMaker().make_two_addition_jobs(a=1, b=2)
    duplicate_flow = AdditionMaker().make_two_addition_jobs(a=1, b=2)
    assign_dedupe_values(flow)
    assign_dedupe_values(duplicate_flow)
    deduped_keys_in_flow = {
        job.config.manager_config[DEDUPE_KEY]
        for job, _ in flow.iterflow()  # type:ignore [no-untyped-call]
    }
    deduped_keys_in_duplicate_flow = {
        job.config.manager_config[DEDUPE_KEY]
        for job, _ in duplicate_flow.iterflow()  # type: ignore[no-untyped-call]
    }
    assert sorted(deduped_keys_in_flow) == sorted(deduped_keys_in_duplicate_flow)

    duplicate_flow_with_different_maker_name = AdditionMaker(
        name="new_maker"
    ).make_two_addition_jobs(a=1, b=2)
    assign_dedupe_values(duplicate_flow_with_different_maker_name)
    deduped_keys_in_duplicate_flow_with_different_maker_name = {
        job.config.manager_config[DEDUPE_KEY]
        for job, _ in duplicate_flow_with_different_maker_name.iterflow()  # type:ignore [no-untyped-call]
    }
    for dk_flow, dk_duplicate_flow_with_different_maker in zip(
        deduped_keys_in_flow, deduped_keys_in_duplicate_flow_with_different_maker_name
    ):
        assert dk_flow != dk_duplicate_flow_with_different_maker


@run_in_temp_dir
def test_duplicate_jobs_in_flows_are_run_only_once(
    mock_jobstore: JobStore, mock_launchpad: LaunchPad
) -> None:
    # both jobs in flow are identical
    flow = AdditionMaker().make_two_addition_jobs(a=1, b=2)
    duplicate_flow = AdditionMaker().make_two_addition_jobs(a=1, b=2)
    manager = JobManager(
        launchpad=mock_launchpad, jobstore=mock_jobstore, fireworker=TEST_FWORKER
    )
    manager.add_to_launchpad(flow)
    manager.add_to_launchpad(duplicate_flow)
    run_rapidfire_launcher(mock_launchpad, TEST_FWORKER)
    fw_ids = mock_launchpad.get_fw_ids()
    root_fw_ids = {
        mock_launchpad.get_wf_by_fw_id(fw_id).root_fw_ids[0] for fw_id in fw_ids
    }
    root_dedupe_keys = {
        mock_launchpad.get_fw_by_id(fw_id).spec[DEDUPE_KEY] for fw_id in root_fw_ids
    }
    assert len(root_dedupe_keys) == 1
    # same as above for the leaf jobs
    leaf_fw_ids = {
        mock_launchpad.get_wf_by_fw_id(fw_id).leaf_fw_ids[0] for fw_id in fw_ids
    }
    leaf_dedupe_keys = {
        mock_launchpad.get_fw_by_id(fw_id).spec[DEDUPE_KEY] for fw_id in leaf_fw_ids
    }
    assert len(leaf_dedupe_keys) == 1

    num_launches = len(
        {mock_launchpad.get_fw_by_id(fw_id).launches[0].launch_dir for fw_id in fw_ids}
    )
    assert num_launches == 2


@run_in_temp_dir
def test_duplicate_jobs_from_different_flows_are_not_considered_identical(
    mock_jobstore: JobStore, mock_launchpad: LaunchPad
) -> None:
    flow = AdditionMaker().make_two_addition_jobs(a=1, b=2)
    # The second job of the above is identical to first job in the flow below
    flow_with_one_duplicate_job = AdditionMaker().make_two_addition_jobs(a=3, b=3)
    manager = JobManager(
        launchpad=mock_launchpad, jobstore=mock_jobstore, fireworker=TEST_FWORKER
    )
    manager.add_to_launchpad(flow)
    manager.add_to_launchpad(flow_with_one_duplicate_job)
    run_rapidfire_launcher(mock_launchpad, TEST_FWORKER)
    fw_ids = mock_launchpad.get_fw_ids()
    assert len(fw_ids) == 4
    unique_dedupe_keys = {
        mock_launchpad.get_fw_by_id(fw_id).spec[DEDUPE_KEY] for fw_id in fw_ids
    }
    # Firework checks the duplication before the rocket launcher, whereas we add the dedupe key before the job is added to launchpad
    # At this time, the output references in jobs are not resolved, and a hash is created based on the attributes of the job and on the parent job's dedupe key
    # As a result, two jobs which might require identical parameters in the workflow will not result in the same hash
    assert len(unique_dedupe_keys) == 4
    num_launches = len(
        {mock_launchpad.get_fw_by_id(fw_id).launches[0].launch_dir for fw_id in fw_ids}
    )
    assert num_launches == 4


def test_dedupe_in_cube_jobs_with_defined_dupe_property() -> None:
    job = CubeMaker().make(a=1)
    duplicate_job = CubeMaker().make(a=1)
    duplicate_job_with_different_maker_name = CubeMaker(name="new_maker").make(a=1)
    assert (
        compute_dedupe(job)
        == compute_dedupe(duplicate_job)
        == compute_dedupe(duplicate_job_with_different_maker_name)
    )


@run_in_temp_dir
def test_duplicate_flows_with_dedupe_property(
    mock_jobstore: JobStore, mock_launchpad: LaunchPad
) -> None:
    flow = CubeMaker().make_two_cube_jobs(a=1)
    duplicate_flow = CubeMaker(name="other_cube_flow").make_two_cube_jobs(a=1)
    manager = JobManager(
        launchpad=mock_launchpad, jobstore=mock_jobstore, fireworker=TEST_FWORKER
    )
    manager.add_to_launchpad(flow)
    manager.add_to_launchpad(duplicate_flow)
    run_rapidfire_launcher(mock_launchpad, TEST_FWORKER)
    fw_ids = mock_launchpad.get_fw_ids()
    assert len(fw_ids) == 4

    unique_dedupe_keys = {
        mock_launchpad.get_fw_by_id(fw_id).spec[DEDUPE_KEY] for fw_id in fw_ids
    }
    assert len(unique_dedupe_keys) == 2
    num_launches = len(
        {mock_launchpad.get_fw_by_id(fw_id).launches[0].launch_dir for fw_id in fw_ids}
    )
    assert num_launches == 2


@run_in_temp_dir
def test_adding_more_jobs_via_response_does_not_dedupe(
    mock_jobstore: JobStore, mock_launchpad: LaunchPad
) -> None:
    flow = AddThenFilteredCubeMaker().make(
        number_1=3, number_2=4, filtering_threshold=5
    )

    manager = JobManager(
        launchpad=mock_launchpad, jobstore=mock_jobstore, fireworker=TEST_FWORKER
    )
    manager.add_to_launchpad(flow)
    run_rapidfire_launcher(mock_launchpad, TEST_FWORKER)
    fw_ids = mock_launchpad.get_fw_ids()
    num_launches = len(
        {mock_launchpad.get_fw_by_id(fw_id).launches[0].launch_dir for fw_id in fw_ids}
    )
    # 2 initial jobs, spawned 2 jobs, and 1 output job
    assert num_launches == 5
    duplicate_flow = AddThenFilteredCubeMaker().make(
        number_1=3, number_2=4, filtering_threshold=5
    )
    manager.add_to_launchpad(duplicate_flow)
    run_rapidfire_launcher(mock_launchpad, TEST_FWORKER)
    # first two will be identical, and spawned jobs will run again
    fw_ids = mock_launchpad.get_fw_ids()
    num_launches = len(
        {mock_launchpad.get_fw_by_id(fw_id).launches[0].launch_dir for fw_id in fw_ids}
    )
    assert num_launches == 5 + 3
