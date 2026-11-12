#!/bin/bash
# Jobscript template for the model jobs (model_jobscript) on Cirrus EX4000, which has no GPU nodes:
# the trainings of the ground-state models (D3, D4's retraining) and of G2's PBE models, and G2's
# phonon calculations, one per job on CPU cores of a standard node (model_device : cpu). dwf.py
# replaces the {COMMAND} line with the job's command (run in the job's folder, its output in a log
# there) and fills in {JOBNAME} (and {NCORES}: mace_threads, if you want it). A training uses
# OMP_NUM_THREADS threads; G2's phonon jobs set their own (mace_threads). The training command,
# mace_run_train (model_command), must be on the PATH, so activate the environment that has
# mace-torch; it, dwf.py and the work directory must be on /work, which the compute nodes can
# see. A phonon job that runs out of time carries on from its saved force constants when it is
# submitted again (--resubmit-failed). Check the time and cores against the Cirrus documentation
# (the standard QoS allows up to 48 hours), and use your own budget code (--account).
#SBATCH --job-name={JOBNAME}
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --account=<budget>
#SBATCH --partition=standard
#SBATCH --qos=standard

module load cray-python
source /work/<project>/<project>/<you>/venv/bin/activate    # your environment with mace-torch
export OMP_NUM_THREADS=72

{COMMAND}
