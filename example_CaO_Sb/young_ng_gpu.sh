#!/bin/bash -l
# GPU jobscript template for young-ng's A100 nodes, for two kinds of job (one GPU each):
#  - the MACE relaxations (mace_jobscript with mace_device : cuda, e.g. in %block runner_scan):
#    dwf.py replaces the {COMMAND} line with mace_pack workers side by side, then waits. They all
#    use the job's one GPU (dwf.py doesn't give workers GPUs of their own, so ask for one GPU a
#    job) and take relaxations from the job's queue, tasks.json, biggest first, until none are
#    left. A few workers (mace_pack 4, mace_threads 2) keep one A100 busy with cells of a few
#    hundred atoms; each holds its own cell on the GPU, so watch nvidia-smi before raising it.
#  - F2's trainings and phonon calculations (model_jobscript with model_device : cuda): {COMMAND}
#    becomes the job's one command.
# dwf.py also fills in {JOBNAME} and {NCORES}: the CPU cores the job's workers use (workers x
# mace_threads; for F2, mace_threads). --time is the longest a job may run, unless the step's
# block gives max_walltime; a MACE job that runs out loses only the relaxations it hadn't
# finished, which go into a new job. The environment must have mace-torch with CUDA (pip's
# torch wheels bring their own CUDA runtime) and be built on the -avx2 Python module, as
# young_ng_mace.sh says, so that the same one runs on the CPU and GPU nodes; keep the work
# directory in Scratch. Check the lines marked EDIT against young-ng's documentation:
# sinfo -o "%P %G %c %m" lists the partitions with their GPUs, cores and memory.
# EDIT: the three lines after these comments: young-ng's way of asking for one A100 (--gres),
# the job's CPU memory (--mem; not the GPU's), and a partition, if GPU jobs need one (remove
# one of the two #s to use it).
#SBATCH --job-name={JOBNAME}
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={NCORES}
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
##SBATCH --partition=gpu

module purge
module load ucl-stack/2025-12
module load python                           # EDIT: the -avx2 Python your environment was built on
source $HOME/Scratch/dwf_venv/bin/activate   # EDIT: your environment with mace-torch

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader   # which GPU the job got

{COMMAND}
