# Dual-Framework Privacy Preservation for Medical Images

A B.Tech CS final-year research project on medical image privacy preservation using
Jaya-optimized reversible perturbation and irreversible privacy transformation.

## Project structure

- `data/raw/`: original dataset inputs
- `data/processed/`: preprocessed dataset outputs
- `data/recoverable/`: outputs from the reversible privacy framework
- `data/non_recoverable/`: outputs from the irreversible privacy framework
- `data/recovered/`: images recovered by the reversible framework
- `cache/`: reusable intermediate cache artifacts
- `metadata/`: dataset and experiment metadata
- `keys/`: key material used by experiments; do not commit secret keys
- `models/`: model checkpoints and related artifacts
- `results/images/`: image outputs and visual comparisons
- `results/metrics/`: evaluation results
- `results/logs/`: persistent run logs
- `src/`: project implementation, organized by responsibility
- `notebooks/`: exploratory and reporting notebooks
- `tests/`: automated checks

## Environment

The project targets Python 3.11 (`>=3.11,<3.12`) and uses PyTorch. Install the
pinned dependencies in a virtual environment with:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Logging

Call `configure_logging()` from `src.utils.logging_config` at the start of a run.
It creates a console handler and writes a persistent log under `results/logs/` by
default. The caller should log the start, progress, outputs, and failures of each
processing step.

# Docker Development Environment

Docker provides the same Python and pinned direct dependency versions on both
developers' machines. The project folder is mounted into the container, so source
edits appear immediately. Git/GitHub synchronizes those source edits between
machines; Docker does not synchronize files or results between computers.

### Prerequisites

- Docker Desktop with Docker Compose
- Git
- For GPU execution on Linux: a compatible NVIDIA GPU, current drivers, and the
  NVIDIA Container Toolkit configured for Docker
- For GPU execution on Windows: Docker Desktop using its WSL 2 backend, a current
  NVIDIA driver with WSL GPU support, and an up-to-date WSL 2 kernel

The default `dev` service uses CPU-only PyTorch, matching the current workstation.
The application does not require a GPU. To use a GPU, build the CUDA-enabled
PyTorch variant and start the `dev-gpu` service as described below.

### Build

```bash
docker compose build
```

This builds CPU-only PyTorch from its CPU wheel index. To build PyTorch 2.14.0
with its CUDA 12.6 wheels, run:

```bash
docker compose --profile gpu build --build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126 dev-gpu
```

This CUDA wheel includes its CUDA runtime libraries. The host still needs a
compatible NVIDIA driver and NVIDIA Container Toolkit; a separate CUDA toolkit
installation inside the container is not required. PyTorch's 2.14 release lists
CUDA 12.6 among its supported CUDA builds. See the [PyTorch 2.14 release notes](https://pytorch.org/blog/pytorch-2-14-release-blog/),
[Docker Compose GPU support](https://docs.docker.com/compose/how-tos/gpu-support/),
and NVIDIA's [Container Toolkit installation guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
for Linux host setup and driver compatibility details. On Windows, Docker Desktop
GPU access requires its WSL 2 backend and a WSL-capable NVIDIA driver; see [Docker Desktop GPU support](https://docs.docker.com/desktop/features/gpu/).

### Start

```bash
docker compose up
```

This starts the CPU `dev` service and keeps it available for shell commands. For
GPU work, after the CUDA build above:

```bash
docker compose --profile gpu up dev-gpu
```

### Open shell

```bash
docker compose run --rm dev bash
```

For the optional GPU service:

```bash
docker compose --profile gpu run --rm dev-gpu bash
```

### Verify environment

Run these inside the container shell:

```bash
python --version
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA runtime:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
python -c "import numpy, cv2, medmnist; print('NumPy:', numpy.__version__); print('OpenCV:', cv2.__version__); print('MedMNIST:', medmnist.__version__)"
```

The image uses Python 3.11. The CPU build reports `CUDA available: False`; the
GPU build reports `True` only when Docker can access a compatible NVIDIA GPU.

### Running tests

The existing tests use Python's `unittest` runner:

```bash
python -m unittest discover -s tests -v
```

The tests include dataset integration and pipeline tests. They write reports,
plots, processed data, and recovery artifacts beneath the project tree. Use a
disposable checkout or back up those files before running them if you need to
preserve an existing experiment snapshot.

### Git collaboration

Recommended workflow:

```text
git pull
git checkout -b feature/<name>
make changes
run tests
git add .
git commit
git push
```

The other developer gets the source changes with `git pull` (or by checking out
the pushed branch). Both developers build the environment from the same Docker
configuration. Docker does not copy code changes or results between computers.

For a new checkout, use the repository's GitHub URL:

```bash
git clone <GitHub repository URL>
cd privacy_preservation
docker compose build
docker compose up
```

The ignore rules exclude raw and processed datasets, generated data arrays,
runtime logs, local caches, model checkpoints, and cryptographic key files.
Small JSON metrics and image reports remain eligible for Git because they are
useful research outputs; review them before staging. Use an approved shared
storage or dataset download mechanism for large data. Never send `keys/secret_key.bin`
through Git; collaborators who need identical recovery runs must exchange that
key through an approved secure channel.

PathMNIST can be downloaded by the existing `load_pathmnist()` helper into
`data/raw/PathMNIST/`. Processed arrays are generated data and are ignored by Git;
share them through approved storage or regenerate them from the raw archive with
`prepare_processed_dataset()` before running steps that require them.

The current project folder does not contain Git repository metadata or a remote.
To use the collaboration commands above, initialize or clone the intended
GitHub repository and configure its remote before sharing this project.
