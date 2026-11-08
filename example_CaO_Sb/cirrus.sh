#!/bin/bash
# Jobscript template for Cirrus (charging : core): the runner fills in the placeholders in braces
# (job name, nodes, tasks per node, CPUs per task, seed). Cirrus shares its nodes (288 cores)
# between jobs and charges for the cores a job asks for, so a job of up to one node asks for just
# its processes x CPUs per task. The memory comes with the cores (2.66 GB each on standard, 5.33
# GB on highmem; --mem can't be given), which is why a process may get more than one CPU. A job
# bigger than a node takes whole nodes: the runner adds #SBATCH --exclusive (Cirrus requires it
# for multi-node jobs, and charges for the whole nodes). --time is the longest any job may run,
# unless max_walltime in the project file says otherwise. Use your own budget code (--account).
#SBATCH --job-name={JOBNAME}
#SBATCH --time=24:00:00
#SBATCH --nodes={NODES}
#SBATCH --ntasks-per-node={TASKS_PER_NODE}
#SBATCH --cpus-per-task={CPUS_PER_TASK}
#SBATCH --account=<budget>
#SBATCH --partition=standard
#SBATCH --qos=standard

module load castep

export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

srun --distribution=block:block --hint=nomultithread castep.mpi {SEED}
