#!/bin/bash -l
# Jobscript template for the CASTEP jobs on Young's new environment, young-ng (RHEL 9 and Slurm:
# run dwf.py on young-ng.rc.ucl.ac.uk, not on the old Grid Engine login nodes). The runner fills
# in the placeholders in braces (job name, nodes, tasks per node, CPUs per task, seed). young-ng
# never shares a CPU node between jobs, so every job takes whole standard nodes, 40 cores and
# 188 GB usable each (cores_per_node : 40 and charging : node in the project file). --mem is per
# node: 180G keeps a job on the standard nodes (asking for more sends it to the one or two
# high-memory nodes), and mem_per_core : 4.5 in the project file makes the runner plan to it,
# putting fewer processes on each node when a job needs more memory per process. There is no
# --partition: Slurm picks one from what the job asks for (only the HBM nodes need one, which a
# %block partitions line can give). --time is the longest any job may run (24 hours for Free
# jobs), unless max_walltime in the project file says otherwise. young-ng runs every job as Free,
# under the allusers account; once project allocations are enabled there, add an --account line
# for your project. The CASTEP module: check its name and version with module avail castep (you
# need to be in the ag-archpc-castep group; UKCP members are covered by its licence).
#SBATCH --job-name={JOBNAME}
#SBATCH --time=24:00:00
#SBATCH --nodes={NODES}
#SBATCH --ntasks-per-node={TASKS_PER_NODE}
#SBATCH --cpus-per-task={CPUS_PER_TASK}
#SBATCH --mem=180G
#SBATCH --exclusive

module purge
module load ucl-stack/2025-12
module load castep                 # EDIT if needed: the CASTEP module (module avail castep)

export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

srun --distribution=block:block --hint=nomultithread castep.mpi {SEED}
