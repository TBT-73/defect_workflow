#!/bin/bash
# Jobscript template for the model jobs (model_jobscript): the trainings of the ground-state
# models (D3, D4's retraining) and of G2's PBE models, and G2's phonon calculations, one per job
# on one GPU. dwf.py replaces the {COMMAND} line with the job's command (run in the job's folder,
# its output in a log there) and fills in {JOBNAME} (and {NCORES}: mace_threads, if you want
# CPU cores for it). The Python it runs needs mace-torch with CUDA (mace_python, default: the
# one running dwf.py); it, dwf.py and the work directory must be on /work, which the compute
# nodes can see. A phonon job that runs out of time carries on from its saved force constants
# when it is submitted again (--resubmit-failed). Check the partition, QoS and GPU request
# against the Cirrus documentation, and use your own budget code (--account).
#SBATCH --job-name={JOBNAME}
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=<budget>
#SBATCH --partition=gpu
#SBATCH --qos=gpu

# load the Python environment with mace-torch here, e.g.
# source /work/<project>/<project>/<you>/venv/bin/activate

{COMMAND}
