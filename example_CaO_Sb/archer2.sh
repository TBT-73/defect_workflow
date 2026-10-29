#!/bin/bash
# Jobscript template for ARCHER2: the runner fills in {JOBNAME}, {NODES}, {TASKS_PER_NODE},
# {CPUS_PER_TASK} and {SEED}. --time is the longest any job may run, unless max_walltime in the
# project file says otherwise (for every step, or in a runner block for some); jobs ask for less
# once finished ones show what they need. Use your own account (--account) and CASTEP module.

#SBATCH --job-name={JOBNAME}
#SBATCH --time=24:00:00
#SBATCH --nodes={NODES}
#SBATCH --ntasks-per-node={TASKS_PER_NODE}
#SBATCH --cpus-per-task={CPUS_PER_TASK}
#SBATCH --account=e89
#SBATCH --partition=standard
#SBATCH --qos=standard

export OMP_NUM_THREADS=1
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

module purge
module load castep/26.11

srun --distribution=block:block --hint=nomultithread castep.mpi {SEED}
