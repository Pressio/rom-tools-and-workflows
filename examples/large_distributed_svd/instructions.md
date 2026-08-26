# Running the distributed SVD on a cluster

This directory contains `run_large_distributed_svd.py`, a simple MPI program
for running `romtools.linalg.linalg._distributed_svd` on a large random matrix.
Each MPI rank creates and owns only its assigned rows. The complete matrix is
never created on one rank.

The commands below are a generic handoff for a SLURM cluster. Module names,
accounts, partitions, and resource limits vary by site, so replace values in
angle brackets with the cluster-specific values.

## 1. Log in and load the cluster software

On the cluster login node, we need to load Git, Python, and one MPI
implementation. The MPI module used to build `mpi4py` must also be loaded when
the program runs.
For example:

```bash
module purge
module load <compiler-module>
module load <mpi-module>
module load <python-module>

which python
which mpicc
which srun
```

## 2. Clone romtools

```bash
git clone --branch develop git@github.com:Pressio/rom-tools-and-workflows.git
cd rom-tools-and-workflows
```

If SSH access to GitHub is not configured, use:

```bash
git clone --branch develop \
    https://github.com/Pressio/rom-tools-and-workflows.git
cd rom-tools-and-workflows
```

## 3. Create the Python environment and build mpi4py

Create the environment on a filesystem visible from every compute node:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Build `mpi4py` from source against the currently loaded cluster MPI:

```bash
MPICC="$(command -v mpicc)" \
    python -m pip install --no-binary=mpi4py 'mpi4py>3.1.4'
```

Install romtools and its remaining dependencies from the cloned source:

```bash
python -m pip install -e .
```

romtools itself is a Python package, so there is no separate CMake or compiler
build step. The `pip install -e .` command is the build/install step. The part
that normally compiles native cluster-specific code is `mpi4py`.

Confirm which MPI library Python will use:

```bash
python -c "from mpi4py import MPI; print(MPI.Get_library_version())"
python -c "import romtools; print(romtools.__file__)"
```

## 4. Run a small test first

Start an interactive allocation using the site's normal command, then run a
small verified case. For example:

```bash
salloc --nodes=2 --ntasks-per-node=2 --time=00:10:00 \
    --account=<account> --partition=<partition>

module load <compiler-module> <mpi-module> <python-module>
source .venv/bin/activate
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

srun --ntasks=4 python -u \
    examples/large_distributed_svd/run_large_distributed_svd.py \
    --global-rows 100003 --columns 100 --verify
```

The run is successful when it ends with `RESULT: PASS`. The program also prints
the hosts, row distribution, input size, elapsed time, and leading singular
values.

If the cluster does not use SLURM, the equivalent launch is usually:

```bash
mpiexec -n 4 python -u \
    examples/large_distributed_svd/run_large_distributed_svd.py \
    --global-rows 100003 --columns 100 --verify
```

## 5. Submit a large batch run

Create a file named `distributed_svd.slurm` in the repository root. A starting
template is:

```bash
#!/bin/bash -l
#SBATCH --job-name=distributed-svd
#SBATCH --account=<account>
#SBATCH --partition=<partition>
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --time=02:00:00
#SBATCH --output=distributed-svd-%j.out

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
module purge
module load <compiler-module>
module load <mpi-module>
module load <python-module>
source .venv/bin/activate

# Prevent every MPI rank from independently spawning many BLAS threads.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

srun --ntasks=32 --cpu-bind=cores python -u \
    examples/large_distributed_svd/run_large_distributed_svd.py \
    --global-rows 10000003 \
    --columns 1000 \
    --dtype float64
```

Submit and monitor the job:

```bash
sbatch distributed_svd.slurm
squeue --me
tail -f distributed-svd-<job-id>.out
```

The example matrix has 10,000,003 rows and 1,000 columns. Its input alone is
about 74.5 GiB globally, or about 2.33 GiB per rank with 32 ranks. The odd row
count intentionally tests an uneven distribution.

## Choosing a safe large problem

For a matrix with `M` global rows and `N` columns, input storage is:

```text
M * N * 8 bytes for float64
M * N * 4 bytes for float32
```

This is only the input. The factor-producing run also stores the local part of
`U`, QR factors, and temporary arrays. Allow several times the per-rank input
size plus BLAS/MPI overhead. Start below the node memory limit and increase the
row count gradually. The script prints the input size before starting the SVD.

`N` strongly affects both runtime and temporary storage because QR/SVD work
scales much more steeply with the column count than with the row count. Increase
the row count first when the goal is simply to exercise a very large matrix.

Useful modes:

```bash
# Less output memory: compute singular values but not U and Vh.
srun --ntasks=32 python -u \
    examples/large_distributed_svd/run_large_distributed_svd.py \
    --global-rows 10000003 --columns 1000 --values-only

# Lower input/factor memory by using single precision.
srun --ntasks=32 python -u \
    examples/large_distributed_svd/run_large_distributed_svd.py \
    --global-rows 10000003 --columns 1000 --dtype float32
```

Use `--verify` for the initial small run. It performs additional matrix
multiplications and is intentionally omitted from the very large timing run.
