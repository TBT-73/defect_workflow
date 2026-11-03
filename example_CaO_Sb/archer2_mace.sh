#!/bin/bash
# MACE jobscript template for ARCHER2 (mace_jobscript): one job runs several MACE relaxations side
# by side on one node. dwf.py replaces the {COMMAND} line with one line per worker (mace_pack of
# them, each with mace_threads threads; 16 x 8 = 128 cores here), then waits for them, and fills
# in {JOBNAME}. The workers run the Python given by mace_python (default: the one running dwf.py),
# which needs mace-torch; on ARCHER2 it, dwf.py and the work directory must be on /work, because
# the compute nodes can't see /home. --time is how long one job may run: a job that runs out
# loses only the relaxations it hadn't finished, which go into a new job. Use your own account.
#SBATCH --job-name={JOBNAME}
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --account=e89-man-m
#SBATCH --partition=standard
#SBATCH --qos=standard

module load cray-python

{COMMAND}
