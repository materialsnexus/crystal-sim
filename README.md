# crystal-sim

crystal-sim provides materials-workflow tooling for 
- generating atomic structures and interfaces
- Calculating energy, force, stress and charge labels using density functional theory (DFT) and Chargemol
- Automated training data generation, labeling and machine-learning interatomic potential (MLIP) training workflows
- MLIP and MD workflows for relaxation and work of adhesion calculations

Jobflow runs workflows locally. FireWorks and SLURM manage queued workflows on a cluster.

## Setup

Run the commands from the repository root unless an instruction specifies another directory.

### Prerequisites

- Linux. Use Linux `x86_64` for the MLIP workflows.
- `uv` to manage Python and the project environment.
- `curl` for the installation commands below.
- **[OPTIONAL] GPAW and PAW datasets:** required for GPAW calculations.
- **[OPTIONAL] Chargemol:** required for GPAW charge labels.
- **[OPTIONAL] MPI:** required for GPAW execution through crystal-sim.

You do not need GPAW or Chargemol to run MLIP calculations or read saved results.

### Install uv and create the environment

Install `uv` if it is not available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version
```

Follow the installer instructions if `uv` is not on `PATH`.

Create the project environment:

```bash
uv sync --frozen --no-editable --group dev
```

`uv` creates `.venv` and selects a compatible Python version. It uses `pyproject.toml` and `uv.lock`.

**[OPTIONAL] Add notebook and MLIP support:**

```bash
uv sync --frozen --no-editable --group dev --group notebooks --group mlip
```

The `notebooks` group supplies Jupyter and the 3D viewer. The `mlip` group supplies the model tools.

### Set up crystal.toml

Copy the template without replacing an existing configuration:

```bash
cp -n sample_crystal.toml crystal.toml
```

Edit the existing `[programs]` table. Do not add a second table with the same name.

| Setting | Value or action |
| --- | --- |
| `mlip_models_dir` | Keep `"mlip_models"` to use the supplied model directories. |
| `isolated_atom_energies_dir` | Keep `"isolated_atom_energies"` for the supplied reference files. |
| `gpaw_env_dir`, `gpaw_setups_dir` | Set these paths only for GPAW. See the instructions below. |
| `chargemol_dir` | Set this path only for Chargemol. See the instructions below. |
| `quantum_espresso_dir`, `pseudopotentials_dir` | **[OPTIONAL]** Set executable and pseudopotential directories only for Quantum ESPRESSO workflows. |
| `mpi_environment_setup_script` | **[OPTIONAL]** Set a shell script path if your MPI environment needs one. |

Set unused optional program and script paths to `""`. The template paths are examples, not installed programs.

Use actual paths in TOML and not shell variables. crystal-sim expands `~`. It resolves relative paths against `crystal.toml`.

Make sure to use paths that compute nodes can access. 

### [OPTIONAL] Install GPAW and PAW datasets

Complete this section for running GPAW calculations.

Install a compatible MPI runtime and development tools first. Make `mpirun` and `mpicc` available on `PATH`. On a cluster, use the MPI modules supplied by the administrator.

Install GPAW with the notebook and MLIP dependencies:

```bash
uv sync --frozen --no-editable --group dev --group notebooks --group mlip --group gpaw
uv run gpaw info
```

Download the PAW datasets:

```bash
GPAW_DATA_DIR="$HOME/gpaw-setup"
uv run gpaw install-data --gpaw --no-register \
  --version https://wiki.fysik.dtu.dk/gpaw-files/gpaw-setups-24.11.0.tar.gz \
  "$GPAW_DATA_DIR"
```

Set these entries in the existing `[programs]` table:

```toml
gpaw_env_dir = ".venv"
gpaw_setups_dir = "~/gpaw-setup/gpaw-setups-24.11.0"
```

`gpaw_env_dir` must contain `bin/python` and `bin/gpaw`. `gpaw_setups_dir` must contain the setup files, such as `O.PBE.gz`. Select the extracted dataset directory, not its parent directory.

### [OPTIONAL] Install Chargemol

Complete this section for charge labels in Notebook 02.

The installer needs `curl`, `tar`, `cmake`, `make`, and `gfortran`. On Ubuntu or Debian, install the build tools with:

```bash
sudo apt install build-essential gfortran cmake curl
```

Choose a new installation directory:

```bash
CHARGEMOL_DIR="$HOME/chargemol"
./install_chargemol.sh "$CHARGEMOL_DIR"
```

The destination must not exist. Use `./install_chargemol.sh --help` to see the options.

Set this entry in the existing `[programs]` table:

```toml
chargemol_dir = "~/chargemol"
```

This directory must contain `bin/chargemol` and `atomic_densities/`. Do not select the executable or its `bin` directory.

### Run the test suite

Install the `dev` dependency group before running tests. Configure the programs and services needed by the selected tests.

Exclude tests with MongoDB, SLURM, external-executable, or slow markers:

```bash
uv run pytest tests -m "not needs_mongodb and not needs_slurm and not needs_third_party_executables and not slow"
```

Run the complete suite only after its external dependencies are available:

```bash
uv run pytest tests
```

## [OPTIONAL] HPC setup

Use this section for FireWorks execution on a SLURM cluster. MongoDB is also required for the local JobStore workflow for example notebooks (particularly Notebooks 02 and 03, see below). 

### Install and start MongoDB

Use an existing MongoDB service if one is available. Otherwise, install MongoDB Community Server with the [official Linux instructions](https://www.mongodb.com/docs/manual/administration/install-community/).

For a local service, create a data directory and start MongoDB:

```bash
mkdir -p "$HOME/.local/share/crystal-sim/mongodb"
mongod --dbpath "$HOME/.local/share/crystal-sim/mongodb" \
  --bind_ip 127.0.0.1 --port 27017
```

Leave this process running. Use another terminal for the launcher commands.

This command accepts local connections only. For HPC, use a MongoDB service that compute nodes can reach. Ask the administrator to configure private-network access and authentication. Do not expose an unauthenticated database to the network.

### Configure FireWorks and Jobflow

Select the files for your execution mode:

| Purpose | Local file | SLURM file |
| --- | --- | --- |
| FireWorks LaunchPad | `.fireworks/local/local_launchpad.yaml` | `.fireworks/slurm/slurm_launchpad.yaml` |
| FireWorks worker | `.fireworks/local/my_fworker.yaml` | `.fireworks/slurm/my_fworker.yaml` |
| Jobflow JobStore | `.jobstore/local/jobstore.yaml` | `.jobstore/slurm/jobstore.yaml` |
| SLURM queue adapter | Not used | `.fireworks/slurm/my_qadapter.yaml` |

Set the following values:

- **LaunchPad:** MongoDB `host`, `port`, database `name`, and credentials.
- **JobStore:** MongoDB `host`, `port`, `database`, and credentials in `docs_store` and every `additional_stores` entry.
- **Worker:** `name`, `category`, and `query` for the jobs that it must accept.
- **Queue adapter:** `queue`, `account`, `qos`, `walltime`, `ntasks`, and `mem` for your cluster.

The supplied connections use `localhost:27017`. On a compute node, `localhost` means that compute node. Replace it with the MongoDB server address for remote connections.

In `.fireworks/slurm/my_qadapter.yaml`, replace `/path/to/folder/crystal-sim` in all three `rocket_launch` paths:

1. The `.venv/bin/rlaunch` executable.
2. The worker file after `-w`.
3. The LaunchPad file after `-l`.

Set `logdir` to a writable directory. Ensure compute nodes can access the repository, `.venv`, program installations, datasets, and `crystal.toml`.

### Start MongoDB, submit workflows, then run qlaunch

MongoDB must be available before you submit workflows. Use `JobManager.add_to_launchpad(...)` with the selected configuration files.

**WARNING: `lpad reset` deletes existing FireWorks jobs, workflows, and launch history. Do not run it on a database you need to retain.**

For a new, empty SLURM LaunchPad, initialize it once:

```bash
uv run lpad -l .fireworks/slurm/slurm_launchpad.yaml reset
```

Confirm the interactive prompt only after you check the target database. For local FireWorks, use `.fireworks/local/local_launchpad.yaml` instead. Local Jobflow execution does not need a LaunchPad reset.

After you submit workflows, start the SLURM launcher:

```bash
uv run qlaunch -r \
  -w .fireworks/slurm/my_fworker.yaml \
  -l .fireworks/slurm/slurm_launchpad.yaml \
  -q .fireworks/slurm/my_qadapter.yaml \
  rapidfire --nlaunches infinite
```

`qlaunch` submits queued FireWorks to SLURM. It does not create the scientific workflow.

### Recommended order

1. Install `uv`.
2. Create the project environment.
3. Configure `crystal.toml`.
4. Install the optional calculation tools required by the workflow.
5. Configure the selected LaunchPad, worker, JobStore, and queue adapter.
6. Start MongoDB, or use the configured service.
7. Initialize the LaunchPad only if its database is new and empty.
8. Submit the workflow.
9. Run `qlaunch` for SLURM or `rlaunch` for local FireWorks.

## Examples

Start Jupyter from the repository root with `uv run jupyter lab`. Select the project Python environment as the notebook kernel. The examples demonstrate the workflow. They do not establish scientific convergence.

| Notebook | Objective |
| --- | --- |
| [01 · Generate structures](examples/01_generate_structures.ipynb) | Generate and select atomic structures for MLIP fine-tuning. |
| [02 · Generate DFT labels](examples/02_generate_labels_for_finetuning.ipynb) | Calculate energy, force, stress, and charge labels with GPAW/Chargemol. Store labels and isolated-atom references in a JobStore. |
| [03 · Fine-tune and evaluate](examples/03_finetune_mlip_and_evaluation.ipynb) | Fine-tune a QET model from stored GPAW labels and matching references. Evaluate its predictions. |
| [04 · Generate surfaces and interfaces](examples/04_generate_interfaces.ipynb) | Generate slabs, select terminations, and match surface lattices. Build and show LiNbO3/SiO2 interfaces. |
| [05 · Optimize interface geometry](examples/05_geometry_optimisation.ipynb) | Relax interface candidates with the supplied LiNbOSi-finetuned model. Select the candidate with the lowest energy. |
| [06 · Calculate MD work of adhesion](examples/06_mlip_work_of_adhesion.ipynb) | Estimate work of adhesion with MLIP molecular dynamics of an interface and its separate slabs. |

## Acknowledgements

This research was developed with funding from the Defense Advanced Research Projects Agency (DARPA). The views, opinions, and/or findings expressed are those of the author(s) and should not be interpreted as representing the official views or policies of the Department of Defense or the U.S. Government.
