#!/bin/bash -l
# MACE jobscript template for young-ng (mace_jobscript): one job runs several MACE relaxations side
# by side on one standard node, which it has to itself (young-ng doesn't share CPU nodes). dwf.py
# replaces the command line at the end with one line per worker (mace_pack of them, each with
# mace_threads threads: 5 x 8 = 40 cores by default), then waits for them, and fills in the job
# name. The workers run the Python given by mace_python (default: the one running dwf.py), which
# needs mace-torch: load the module and environment that Python comes from. Build that
# environment on an -avx2 Python module, so that the same one also runs on the GPU nodes
# (young_ng_gpu.sh). The compute nodes see your home and Scratch; keep the work directory in
# Scratch. The MACE foundation model is downloaded the first time it is used: load it once on the
# login node, so that it is in ~/.cache/mace for the jobs. --time is how long one job may run: a
# job that runs out loses only the relaxations it hadn't finished, which go into a new job.
#SBATCH --job-name={JOBNAME}
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=180G
#SBATCH --exclusive

module purge
module load ucl-stack/2025-12
module load python                           # EDIT: the -avx2 Python your environment was built on
source $HOME/Scratch/dwf_venv/bin/activate   # EDIT: your environment with mace-torch

{COMMAND}
