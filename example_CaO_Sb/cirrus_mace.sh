#!/bin/bash
# MACE jobscript template for Cirrus (mace_jobscript): one job runs several MACE relaxations side
# by side on one node. dwf.py replaces the {COMMAND} line with one line per worker (up to
# mace_pack of them, each with mace_threads threads), then waits for them, and fills in {JOBNAME}
# and {NCORES}: the cores those workers use (workers x mace_threads, at most 36 x 8 = 288 here),
# so a job asks for, and is charged for, just those. The workers run the Python given by
# mace_python (default: the one running dwf.py), which needs mace-torch; it, dwf.py and the work
# directory must be on /work, because the compute nodes can't see /home. --time is how long one
# job may run: a job that runs out loses only the relaxations it hadn't finished, which go into a
# new job. Use your own budget code (--account).
#SBATCH --job-name={JOBNAME}
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={NCORES}
#SBATCH --account=<budget>
#SBATCH --partition=standard
#SBATCH --qos=standard

module load cray-python

{COMMAND}
