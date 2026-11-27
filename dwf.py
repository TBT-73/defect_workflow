#!/usr/bin/env python3
"""
dwf.py - the defect workflow (phases A to F of the flowchart) in one script

    python3 dwf.py --example CaO_Sb            # writes CaO_Sb.parameters and CaO_Sb.run
    nohup python3 dwf.py CaO_Sb > run.log 2>&1 &
    python3 dwf.py CaO_Sb --status             # which steps have finished

A project is two files with one name: NAME.parameters, the science (the host, the dopants,
every step's settings), and NAME.run, how its jobs run on your machine (the runner settings and
blocks, the MACE and model jobs); python3 dwf.py NAME reads both. An older dwf.py's project file
(NAME.dwf, everything in one) becomes the two with  python3 dwf.py --split NAME.dwf.

One command runs every step in order, A1 to F2, waiting for its CASTEP jobs. Run it again
after any stop (Ctrl-C, touch <workdir>/STOP, --no-wait, a crash) and it carries on where it
stopped. Give step names, labels or phase letters to run only those (python3 dwf.py CaO_Sb C),
or --until STEP to stop after a step or phase.

  A  host            A1 converge, A2.1 relax, A2.2 bands, A2.3 dielectric, A3.1 ref_phases,
                     A3.2 ref_limits (reference energies at the screening settings: the host,
                     and the pure elements of the host and the dopants)
  B  candidates      B1 generate, B2 scan (the supercell sizes, and MACE on every candidate),
                     B3 choose (the screening cell), B4 electronic (the electronic cell, from
                     C4 on: every band-edge valley folded onto Gamma)
  C  screening,      C1 prerelax (MACE), C2 screen, C3 filter, in the screening cell; then C4
     ground and      distort (ShakeNBreak's distortions relaxed with MACE, the lowest kept), C5
     excited states  drelax (CASTEP from those), C6 rank, C7 excite (triplets, localised
                     levels, ΔSCF), in the electronic cell; C8 validate (optional finite-size
                     test)
  D  production      D1 relaxations, D2 zpl (production ΔSCF), in the electronic cell
  E  thermodynamics  E1.1 phases, E1.2 limits (the competing phases of the host and of the
                     dopants D1 has, at the production settings), E2 corrections, E3 formation
                     (a Fermi level per dopant set), E4 interest
  F  lineshapes      F1 ccd (1D configuration-coordinate diagram), F2 lineshape (optional:
                     multi-mode, a PBE model's phonons in a big cell)

Files. Everything goes in the work directory, dwf_<name>/ next to the project's files:
  results.dwf        everything in one file, as a .castep is for a CASTEP job: every step's
                     status, the keywords the project's files give and every phase's summary
  state.json         which steps have finished, and with what inputs (the checkpoints)
  A_host/ ... F_lineshape/  one folder per phase, each with
    results.json     everything the later steps use from this phase
    summary.txt      the same results as tables, for you to read
    jobs/            its CASTEP jobs, one folder per step
    structures/      the structures it made, as .cell files
A step reads only the results.json of earlier phases (and the files named there).

Checkpoints. A step that finishes is recorded in state.json with a fingerprint of everything it
used (its settings, the files it read, the results of the steps before it) and is skipped next
time. Change a setting and the steps that use it, and the steps after them, run again; CASTEP
jobs whose inputs did not change are reused, and a job that already ran with other inputs
stops the script, so results from two sets of settings never mix (--redo STEP starts a step
again from scratch). A work directory made by a dwf.py from when phase D held the ground and
excited states (D1-D5's model, then D6 rank, D7 excite and D8 validate, now C6-C8; E, F and G
were production, thermodynamics and lineshapes) is brought up to date, its results kept, with
--migrate.

All CASTEP jobs go through the job runner (PART 1 of this file, run_batch), which runs the
structures this script builds. Its runner settings (jobscript,
partitions, cores_per_node, charging, max_cores or max_nodes, max_walltime, ...) go in NAME.run,
for every step; %block runner_host, runner_screen and runner_production change them for phase
A's steps and the competing phases (A3.1, E1.1), the defects' steps at the screening
settings, or those at the production settings, and %block runner_<step> for one step (e.g.
runner_converge, runner_excite). The MACE relaxations (B2, C1, C4) run where this script
runs, or, with mace_jobscript, as jobs on compute nodes (each job runs this script as a worker:
python3 dwf.py --mace-worker <tasks>.json); %block runner_scan, runner_prerelax and
runner_distort give B2's, C1's and C4's MACE jobs settings of their own. F2's model training
and phonons (python3 dwf.py --phonon-worker spec.json) run here, or as jobs with
model_jobscript.

The job runner also works on its own, as supercell_convergence.py did before it became part of
this file: python3 dwf.py --conv NAME runs the supercell size series of one defect in NAME.conv
(python3 dwf.py --conv-example writes an annotated one, with every runner keyword), and python3
dwf.py --batch FILE the CASTEP calculations a batch file lists (README_runner.md).
"""

import argparse
import copy
import difflib
import hashlib
import itertools
import json
import math
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ase import Atoms
from ase.build import make_supercell
from ase.cell import Cell
from ase.data import atomic_numbers, chemical_symbols
from ase.geometry import cellpar_to_cell, get_distances

# ##################################################################################################
# PART 1: THE JOB RUNNER (until 2026-11, supercell_convergence.py)
# ##################################################################################################
# Runs CASTEP calculations on a Slurm machine: inputs, dry runs, a plan for each job (cores,
# memory, walltime), submission, waiting, restarts. The workflow (PART 2) runs every CASTEP job
# through run_batch; on its own, it runs a supercell size series of one defect
# (python3 dwf.py --conv NAME) or a batch file of calculations (python3 dwf.py --batch FILE).
# Its manual (README_runner.md has more):
#
# The job runner
# ==============
# Supercell-size convergence of a defect formation energy with CASTEP, using ASE to
# build the supercells.
#
# For each supercell size n the script
#   1. builds the supercell n*T of the primitive cell (T = identity unless transform),
#   2. inserts the defect (substitutions and/or vacancies). Any components after the
#      first go on their nearest sites to it, with the same relative geometry in every
#      supercell, and the defect is put as close to the cell centre as it can be: the
#      mean position of its components nearest the centre (the size table shows how
#      far off it is). Defect cells are run without symmetry (symmetry_generate is
#      removed), so the relaxation is free to break the starting symmetry,
#   3. sets a Monkhorst-Pack grid with the same k-point spacing as the primitive cell,
#   4. runs a CASTEP dry run (as a small batch job on a compute node, by default) to
#      get the irreducible k-points and the memory estimate,
#   5. picks a core count suited to k-point parallelisation and (with %block
#      partitions) a partition with enough memory, writes a jobscript from your
#      template and submits a geometry optimisation of the defect cell (ions only;
#      the lattice is fixed at the bulk value with fix_all_cell),
#   6. once it has converged, runs a single point on the relaxed structure with the same
#      k-points and cores (singlepoint : false skips this and uses the final energy of the
#      relaxation itself) and, with bulk_mode : supercell, one on each pristine supercell,
#   7. reads the defect energy and computes
#          E_f*(n) = E_defect(n) - N_host(n) * e_bulk + E_corr(n)
#      i.e. the formation energy without the chemical-potential and Fermi-level terms
#      (identical for every supercell, so they don't affect convergence). E_corr is 0 for
#      a neutral defect; for a charged one it is the image-charge correction
#      (charge_correction : lz or mp, with dielectric), see "Charged defects" below,
# and stops once |E_f*(n) - E_f*(previous n)| < tol.
#
# Usage
# -----
# Everything is set in an input file, written like a CASTEP .param/.cell file:
# "keyword : value" lines and %block ... %endblock blocks, keywords in any case,
# # or ! to start a comment. Run it with the seedname, like CASTEP:
#
#     python3 dwf.py --conv CaO_SbCa                  # reads CaO_SbCa.conv
#     python3 dwf.py --conv CaO_SbCa --prepare-only   # first time: check inputs
#
#     python3 dwf.py --conv-example > CaO_SbCa.conv   # annotated example
#
# Run-control switches go on the command line (or in the file as e.g. no_wait : true):
#     --prepare-only     build the inputs and jobscripts, submit no calculations
#     --no-wait          submit what is needed, report and exit instead of waiting
#     --resubmit-failed  resubmit the existing jobscript of a failed calculation once
#     --force            reuse a work directory that was set up with different inputs
#
# Use the relaxed primitive cell and the energy per atom from that same calculation
# (same cut-off, functional and pseudopotentials).
#
# Charged defects
# ---------------
# Set the same charge (e.g. charge : -1) in the .param and the geom_param; the script
# removes it for the pristine cells, reads the charge each defect ran with from its
# .castep, and stops if the single point and its relaxation differ. A charged defect
# interacts with its own periodic images and the neutralising background, an error that
# shrinks only as 1/L, so E_f* gets a correction:
#     Makov-Payne (mp):  E1 = q^2 alpha_M / (2 eps L)
#     Lany-Zunger (lz):  [1 + c_sh (1 - 1/eps)] E1   (about 2/3 of E1)
# alpha_M/L is computed by an Ewald sum for the actual supercell, eps is dielectric
# (static, electronic + ionic) and c_sh is -0.369, -0.343, -0.342 for simple cubic,
# FCC and BCC supercells (about -1/3 otherwise). No potential alignment is included.
#
# Jobscript template
# ------------------
# Copy the jobscript you normally use for CASTEP and put {SEED} where the seedname
# goes and {NCORES} where the number of cores (MPI processes) goes. On machines that
# hand out whole nodes, set cores_per_node and use {NODES} and {TASKS_PER_NODE}:
#     #SBATCH --nodes={NODES}
#     #SBATCH --ntasks-per-node={TASKS_PER_NODE}
#     #SBATCH --cpus-per-task={CPUS_PER_TASK}
# With omp_threads, a job that uses fewer processes per node to fit the memory runs
# OpenMP threads on the cores left over: the script sets OMP_NUM_THREADS (and, if the
# template lacks them, OMP_PLACES, SRUN_CPUS_PER_TASK and --cpus-per-task).
# Other placeholders: {JOBNAME} {NKPTS} {PARTITION} {MEM_GB} {MEM_MB} {MEM_PER_NODE_GB}
# {MEM_PER_CORE_GB} {MEM_PER_CORE_MB} (estimated memory in total, per node and per
# process, including the safety factor). Give the template as a file
# (jobscript : job_template.sh) or inline in %block jobscript.
#
# Whole-node machines
# -------------------
# With cores_per_node, each job uses up to max_nodes nodes, with the process count
# chosen for k-point parallelisation (e.g. 126 of 128 cores per node for 14 k-points).
# A job that needs more memory per process than a full node gives goes to the next
# partition (e.g. highmem); only if none has enough are fewer processes put on each
# node ({CPUS_PER_TASK} then spreads them out).
#
# Machines that charge by the core (charging : core, e.g. Cirrus)
# ---------------------------------------------------------------
# Where nodes are shared and a job is charged only for the cores it asks for, set
# charging : core with cores_per_node, and use {NODES}, {TASKS_PER_NODE} and
# {CPUS_PER_TASK} in the template. Each job gets up to max_cores processes, chosen for
# k-point parallelisation as on core-based machines. A job that fits on one node asks
# for just its cores and shares the node. Memory comes with the cores (the partition's
# GB per core; no --mem), so a job goes on the first partition where each process fits
# in one core's memory; if none has enough, each process gets as many cores as its
# memory needs (--cpus-per-task, on the partition where that takes the fewest cores;
# omp_threads can run threads on them). A job bigger than a node takes whole nodes
# (--exclusive, charged in full, up to max_nodes), laid out as on whole-node machines.
# In batch mode, pack : N puts up to N such jobs (on the same partition, and no more
# cores than a node has) in one job, which asks for their cores in all, so they wait in
# the queue once; dryrun_pack : N does the same for dry runs.
#
# Partitions
# ----------
# With %block partitions, each job goes on the first partition listed whose memory
# per core is enough for it (e.g. multicore, and himem only when a job needs more).
# The script sets #SBATCH --partition (or fills {PARTITION}) and adds any extra #SBATCH
# options given for that partition, e.g. --qos=highmem, or --mem={MEM_GB}G where
# memory is requested separately.
#
# Memory
# ------
# The memory per process is estimated from the serial dry run as
#     (baseline + other + S * (r + (1 - r) / p)) * mem_safety
# with S the dry run's 'Model' + 'Electronic' memory, p the number of processes and
# r = mem_replicated, the fraction of S that every process keeps its own copy of. With
# mem_replicated : auto (the default) r starts at 0.25 and is then measured from the
# 'Peak Memory Use' CASTEP prints at the end of each run (a relaxation that stopped at
# run_time counts too); the value from the biggest cell measured so far is used for every
# calculation planned after that, including jobs that are resubmitted.
#
# Walltimes
# ---------
# The #SBATCH --time in the template (or a --time= on a partition line, or max_walltime,
# which replaces both) is the longest any job may run. With walltime : auto (the default) a
# job asks for less once finished
# jobs show what it needs: their time per SCF cycle is scaled to the new cell as
# k-points x electrons^a / processes (a fitted from the finished sizes, 3 until there
# are two) and multiplied by the SCF cycles of the biggest finished relaxation, then by
# walltime_safety (a relaxation also gets the margin its run_time stops it by, below).
# Until a job has finished there is nothing to scale from, so the first
# jobs get the full --time; set sp_walltime to give the first single points (e.g. the
# pristine cells) a shorter one, so they wait less in the queue (one that runs out is
# resubmitted with the full --time). Each relaxation gets CASTEP's run_time a little below
# its walltime, so it stops cleanly and writes a .check before the limit; the script then
# resubmits it with continuation : default and the longest walltime until it converges (at most
# max_restarts times). A job killed at the time limit is resubmitted the same way. If
# CASTEP can't read the .check (e.g. it was cut short while being written), the .check is
# put aside and the relaxation starts again from the structure of its last geometry step
# in the .castep. A resubmitted job is planned again first, so it gets the best layout for
# the memory measured so far. With delete_check : true, a calculation's .check file is
# deleted once it has finished successfully (it is only needed to carry on an unfinished
# run, and can be tens of GB), and one CASTEP can't read is deleted straight away.
#
# MACE pre-relaxation
# -------------------
# With prerelax : mace each defect supercell is first relaxed with a MACE foundation
# model (positions only, the cell fixed at the bulk lattice) by this script, on the
# machine it runs on, and CASTEP's relaxation starts from that structure. The final
# structure and energy are still CASTEP's: MACE only gives it a better starting point,
# so it needs fewer geometry steps. Needs the mace-torch package (Python 3.9+) in the
# Python that runs this script. The result is kept in prerelax.json and the optimiser's
# log in prerelax.log, in each relaxation's directory.
#
# Dry runs
# --------
# By default each dry run is submitted as its own job (dryrun.sh), made from the same
# template: {NCORES} becomes dryrun_cores (default 1) and the CASTEP line becomes
#     mpirun -n 1 castep.mpi --dryrun <seed>
# so CASTEP always runs as one process (the memory model expects a serial estimate).
# If the template's partition needs more cores (e.g. multicore needs 2+), send the dry
# runs elsewhere with dryrun_partition : serial, and shorten their walltime with
# dryrun_time : 0-1. For anything else give a separate template in dryrun_jobscript
# (use {SEED} and write --dryrun yourself). dryrun_on : login runs them where the
# script runs instead, using dryrun_cmd. In batch mode, dryrun_pack : N runs up to N dry
# runs side by side in one job (packs/dryrun_NNN/job.sh), each as one process started with
# srun --exact and dryrun_mem_gb of memory, so a step's dry runs don't wait in the queue
# one by one; the template's CASTEP line must use srun.
#
# The script waits for the jobs, so run it in tmux/screen or with nohup. It can be
# stopped and rerun with the same command at any time: finished and queued
# calculations are picked up rather than repeated. To stop it from another login node,
# create a file called STOP in the work directory (touch <workdir>/STOP); it says at the
# start which node it runs on.
#
# Batch mode
# ----------
# The same machinery can run any list of CASTEP calculations instead of a size series.
# It runs the structures it is given and builds no supercells of its own. The defect
# workflow (PART 2) uses it for every CASTEP step, and you can use it directly:
#
#     python3 dwf.py --batch jobs.batch
#
# The batch file holds the runner settings of a .conv file (jobscript, partitions,
# cores_per_node, max_nodes, walltime, max_restarts, dryrun_*, ...), the .param and .cell
# every job starts from (param, cell; the structure, task and k-points are set per job),
# kspacing, and one line per job in %block jobs:
#     name  structure-file  [kind=geom|sp] [task=...] [fix_cell=false] [symmetry=true]
#                           [kgrid=1,1,1] [offset=gamma] [any .param keyword=value]
# e.g.  host  CaO_prim.cell  kind=geom fix_cell=false symmetry=true   (a variable-cell
# relaxation) or  scr1  cand1.cell  kind=geom kgrid=1,1,1 charge=-1 spin=2. Each job gets
# its own directory, dry run, plan, walltime and restarts, as in a size series; jobs that
# would give the same dry run share one. max_jobs limits how many jobs are queued at once,
# and with tolerate_failures : true a job that fails is marked failed (in its job_info.json)
# while the others carry on; --resubmit-failed starts the failed ones again. With pack : N
# (and cores_per_node), up to N jobs run side by side on one node, in one job of their own
# (packs/pack_NNN/job.sh), each on cores_per_node/N cores with srun --exact and its share of
# the node's memory; a job that doesn't fit in its share, or needs a restart, runs on its own.
# With charging : core, each keeps its own plan instead, and the pack asks for their cores.
# The results are written to batch_results.json in the work directory (batch_<file name> by
# default). From Python: run_batch(workdir, jobs, args, cell_text, param_text, ...) (see its
# comment).

# ==========================================================================================
# HOW THE FUNCTIONS FIT TOGETHER
# ==========================================================================================
# series_main() runs everything (see the comment above it). In outline:
#
#   get_args -> read_input           read <seed>.conv into the settings, `args`
#   read_primitive, parse_defect,    the host cell, the defect (with translation_copies of its
#     target_spacing                   first site) and the k-point spacing
#   build_supercell, defect_vectors, a supercell and a defect cell for each size (anchor_site:
#     insert_defect                    the defect as close to the cell centre as it can be),
#                                      and a Calc for each calculation (relaxation, single
#                                      point, bulk)
#   calibrate, measure_times,        learn memory use and timings from jobs that have run;
#     remove_check                     delete the .check files of finished calculations
#   then, for each pass of sizes:
#     prepare_all                    inputs (write_inputs -> prerelaxed [MACE], write_cell,
#                                      make_param), dry runs (dryrun_script or local_dryrun,
#                                      dryrun_output) and a plan for each job (finalize ->
#                                      choose_resources, thread_count, set_threads,
#                                      plan_walltime, set_param)
#     submit                         sbatch job.sh and record the run in job_info.json
#     wait_for                       poll the queue (job_active); a job that stops early goes
#                                      to handle_failure -> restart, which resubmits it to carry
#                                      on (planned again with calibrate + finalize)
#     prepare_sp / relaxed_atoms     single points on the relaxed cells, or a check of them
#     results_row, read_energy,      E_f* for each size (with charge_correction for a
#       report                         charged defect), the results table and the
#                                      convergence test
#
# Batch mode (--batch, and the workflow) runs a list of calculations instead:
#   read_batch_file / batch_args     the runner settings (and, for --batch, the jobs)
#   run_batch                        a Calc per job (any task, fixed or variable cell, own
#                                      k-points and .param keywords: cell_options, dryrun_key,
#                                      job_spec), then the same prepare_all / submit / wait_for
#                                      / restart machinery, with at most max_jobs queued and,
#                                      with tolerate_failures, a failed job marked (mark_failed)
#                                      rather than stopping everything
#   job_result, final_atoms,         what each job gave: status, energy, spin, final structure
#     batch_energy
#
# Each calculation is a Calc with its own directory. Its job_info.json holds everything the
# script knows about it (plan, job ids, runs, timings, memory): Calc.info() reads it and
# Calc.save() writes it. log() and die() are used throughout. These four are left out of the
# "Calls" lists in the comments below.
# ==========================================================================================

BOHR = 0.529177210903
COULOMB = 14.399645                    # e^2/(4 pi eps0) in eV*Ang
# Lany-Zunger shape factors c_sh for supercells of these shapes (Lany & Zunger, Modelling Simul.
# Mater. Sci. Eng. 17, 084002 (2009), Table 1); other shapes use LZ_SHAPE_OTHER
LZ_SHAPE = {"CUB": -0.369, "FCC": -0.343, "BCC": -0.342}
LZ_SHAPE_OTHER = -1 / 3
UNITS = {"ang": 1.0, "bohr": BOHR, "a0": BOHR, "nm": 10.0}
JOBSCRIPT, DRYRUN_SCRIPT, INFO = "job.sh", "dryrun.sh", "job_info.json"
# the line of the jobscript template that runs CASTEP, and MPI process-count variables
CASTEP_RUN = re.compile(r"((?<![\w])castep[\w.\-]*\s+)\{SEED\}")
DRYRUN_RUN = re.compile(r"((?<![\w])castep[\w.\-]*\s+(?:--dryrun|-d)\s+)\{SEED\}")   # dry runs
NPROC_VARS = re.compile(r"\$\{?(SLURM_NTASKS|SLURM_NPROCS|NSLOTS|PBS_NP)\}?")
# a filled-in jobscript line that runs CASTEP (not a comment or a module command)
CASTEP_LINE = re.compile(r"^(?!\s*#)(?!\s*module\b).*(?<![\w/])castep[\w.\-]*\s+\S", re.M)
OMP_AUTO_MAX = 4                       # most OpenMP threads per process with omp_threads : auto
# pack : N (batch mode): the jobs of a pack get this fraction of the node's memory between them
# (each in proportion to its cores; the rest is left for the system), and their job scripts go in
# <workdir>/packs/
PACK_MEM, PACK_DIR = 0.75, "packs"
ARRAY_TASKS = "tasks.txt"      # a job array's tasks (array : N): directory <tab> seed, one a line
TIMINGS = [0]      # how often a job's timings were saved: estimate_walltime's list of timed jobs
# srun options a pack sets itself for each job, taken out of the template's CASTEP line
SRUN_OWN = re.compile(r"\s(--(?:nodes|ntasks|ntasks-per-node|cpus-per-task|mem|mem-per-cpu)"
                      r"(?:=|\s+)\S+|--(?:exact|exclusive)(?:=\S+)?|-[nNc]\s*\S+)(?=\s|$)")

# Regenerated for every supercell; everything else in the primitive .cell
# (species_pot, ...) is copied across unchanged.
STRUCT_BLOCKS = {"lattice_cart", "lattice_abc", "positions_frac", "positions_abs"}
KPOINT_KEYS = {"kpoint_mp_grid", "kpoints_mp_grid", "kpoint_mp_spacing",
               "kpoints_mp_spacing", "kpoint_mp_offset", "kpoints_mp_offset"}
SET_KEYS = KPOINT_KEYS | {"fix_all_cell"}       # written by the script instead
# Atom-indexed / primitive-cell-specific blocks that would be wrong in a supercell
# Keywords CASTEP accepts under two spellings: read as the one on the right, so that setting
# either replaces the other (two spellings of one keyword in a .param is an error).
KEY_SYNONYMS = {"spin_polarised": "spin_polarized"}
DROP_BLOCKS = {"kpoint_list", "kpoints_list", "symmetry_ops",
               "ionic_constraints", "ionic_velocities", "cell_constraints"}
# Also left out of the defect cells, which are run without symmetry
SYMMETRY_KEYS = {"symmetry_generate", "symmetry_tol", "snap_to_symmetry"}
MEM_FLAGS = ("--mem", "--mem-per-cpu", "--mem-per-gpu")     # only one may be given
SBATCH_SHORT = {"--partition": "-p", "--time": "-t", "--nodes": "-N", "--ntasks": "-n",
                "--qos": "-q", "--account": "-A", "--job-name": "-J", "--constraint": "-C",
                "--cpus-per-task": "-c"}
GEOM_DONE = "Geometry optimization completed successfully"
PEAK_MEMORY = re.compile(r"Peak Memory Use\s*=\s*([\d.]+)\s*([kKmMgG])i?B")
R_START = 0.25        # mem_replicated : auto uses this until a job has been measured
SCF_CYCLE = re.compile(r"^\s*\d+\s+-?\d[^\n]*<-- SCF\s*$", re.M)    # one SCF iteration
TOTAL_TIME = re.compile(r"^\s*Total time\s*=\s*([\d.]+)\s*s", re.M)
GEOM_STEP = re.compile(r"finished iteration\s+\d+")
OUT_OF_TIME = "Insufficient time for another iteration"   # CASTEP stopped at run_time
# How long before its walltime a relaxation's run_time stops CASTEP (s; doubled after a job is
# killed at the time limit anyway). As each geometry iteration starts, CASTEP checks it has time
# for one as long as the last ("Insufficient time for another iteration" if not), so it stops
# between geometry steps on its own; this only has to cover what that check doesn't: the job's
# start-up before CASTEP's clock starts, an iteration that runs longer than the last (a line
# search's extra steps), writing the .check, and shutting down
RUN_TIME_MARGIN = 300.0
# How often the count of a batch's finished jobs, or of a MACE step's relaxations, goes in the log
# (s), and the width of a progress.txt's first line, its count (padded, so it can be written over
# in place): write_progress
PROGRESS_EVERY = 600
PROGRESS_WIDTH = 79
# CASTEP failed to read the .check file it was asked to carry on from
CHECK_UNREADABLE = re.compile(r"model_continuation|model_read|wave_read|Error reading[^\n]*check",
                              re.I)
# a row of the 'Fractional coordinates of atoms' table in the .castep
ATOM_ROW = re.compile(r"^\s*x\s+([A-Z][a-z]?)\S*\s+\d+\s+(\S+)\s+(\S+)\s+(\S+)\s+x", re.M)
# the lattice CASTEP prints for each structure (a variable-cell relaxation changes it), and the
# total spin of the electrons (in hbar/2) after each SCF of a spin-polarised calculation
REAL_LATTICE = re.compile(r"Real Lattice\(A\)[^\n]*\n((?:[^\n]*\n){3})")
SPIN_DENSITY = re.compile(r"Integrated Spin Density\s*=\s*(\S+)")
BATCH_RESULTS = "batch_results.json"
WALLTIME_MIN = 1800                   # s, the shortest walltime asked for
A_DEFAULT, A_RANGE = 3.0, (2.0, 3.5)   # time per SCF cycle ~ electrons^a
PRERELAX_FILE, PRERELAX_LOG = "prerelax.json", "prerelax.log"

ENERGY_PATTERNS = {
    "final": r"Final energy(?:, E)?\s*=\s*(\S+)\s*eV",
    "free": r"Final free energy \(E-TS\)\s*=\s*(\S+)\s*eV",
    "0K": r"est\. 0K energy \(E-0\.5TS\)\s*=\s*(\S+)\s*eV",
}
NET_CHARGE = re.compile(r"net charge of system\s*:\s*(\S+)")
SCF_FAILED = ("has not reached the groundstate", "electronic minimisation did not converge")
FINISHED = re.compile(r"^\s*Total time\s*=", re.M)
END_STATES = ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
              "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE")


# Prints a progress message with the time in front, e.g. [14:02:31] ...
# Flushes straight away, so out.log stays up to date when the script runs under nohup. Used for
# progress messages, notes and warnings.
#   Called by: almost every function
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# Writes a progress.txt (a CASTEP batch's: run_batch; a MACE job's: mace_progress): head, the
# count, as its first line, written over in place (padded to PROGRESS_WIDTH), an empty line, and
# then lines, added to the end, so that tail -f shows each as it comes. Makes the file if it isn't
# there, and starts it again under head if its first line isn't one of these (keeping the lines).
# Anything that stops it leaves the file as it was: nothing reads it but you.
#   Called by: mace_progress(), run_batch()
def write_progress(f, head, lines=()):
    head = head[:PROGRESS_WIDTH].ljust(PROGRESS_WIDTH).encode()
    add = "".join(x + "\n" for x in lines).encode()
    try:
        try:
            fh = open(f, "r+b")
        except FileNotFoundError:
            Path(f).write_bytes(head + b"\n\n" + add)
            return
        with fh:
            if fh.read(PROGRESS_WIDTH + 1)[PROGRESS_WIDTH:] != b"\n":
                fh.seek(0)
                rest = (fh.read().split(b"\n", 2) + [b""])[2]
                if rest and not rest.endswith(b"\n"):
                    rest += b"\n"
                fh.seek(0)
                fh.truncate()
                fh.write(head + b"\n\n" + rest + add)
                return
            if add:
                fh.seek(-1, os.SEEK_END)
                if fh.read(1) != b"\n":            # a line cut short (a job killed writing it)
                    add = b"\n" + add
                fh.seek(0, os.SEEK_END)
                fh.write(add)
            fh.seek(0)
            fh.write(head)
    except OSError:
        pass


# Stops the script with 'ERROR: <message>'.
# Used for every problem the script can't fix itself: bad input, a failed job that can't be
# restarted, a work directory set up with other inputs, ... Jobs already submitted keep running;
# rerun the same command once the problem is fixed.
#   Called by: almost every function
def die(msg):
    sys.exit(f"\nERROR: {msg}")


# --------------------------------------------------------------------------
# CASTEP input files
# --------------------------------------------------------------------------
# Removes a # or ! comment from one line and trims the spaces.
# Used whenever a CASTEP-style file (.cell, .param, .conv) is read line by line.
#   Called by: block_rows(), parse_castep_input(), set_param()
def strip_comment(line):
    return re.split(r"[#!]", line, maxsplit=1)[0].strip()


# A keyword in lower case, with CASTEP's other spellings turned into one (KEY_SYNONYMS):
# spin_polarised -> spin_polarized.
#   Called by: parse_castep_input(), read_batch_file(), set_param(), the workflow
def canonical_key(key):
    key = key.lower()
    return KEY_SYNONYMS.get(key, key)


# Reads any CASTEP-style text into a list of items.
# Used for your .cell and .param(s), the .conv input file and CASTEP's -out.cell. Each item is a
# block {'type': 'block', 'name', 'body'}, a keyword {'type': 'kw', 'key', 'value', 'raw'} or any
# other line {'type': 'raw'}. The original lines are kept, so render() can write the items back
# exactly as they were. A block may hold other blocks (the workflow's %block runner_<step> can
# hold a %block partitions): they stay in its body, to be read by parsing the body again.
#   Called by: cell_options(), final_atoms(), last_geometry(), series_main(), param_charge(),
#              read_batch_file(), read_input(), read_structure(), relaxed_atoms(), run_batch()
#   Calls:     canonical_key(), strip_comment()
def parse_castep_input(text):
    """Split a .cell/.param file into blocks, keywords and other lines."""
    items, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        s = strip_comment(lines[i])
        m = re.match(r"%block\s+(\w+)", s, re.I)
        if m:
            name, body, inner = m.group(1).lower(), [], 0      # inner: blocks open inside it
            i += 1
            while i < len(lines):
                t = strip_comment(lines[i])
                if re.match(r"%endblock", t, re.I):
                    if not inner:
                        break
                    inner -= 1
                elif re.match(r"%block\s+\w", t, re.I):
                    inner += 1
                body.append(lines[i])
                i += 1
            if i == len(lines):
                die(f"unterminated %block {name}")
            items.append({"type": "block", "name": name, "body": body})
        else:
            m = re.match(r"(\w+)\s*[:=]?\s*(.*)", s)
            if m:
                items.append({"type": "kw", "key": canonical_key(m.group(1)),
                              "value": m.group(2).strip(), "raw": lines[i]})
            else:
                items.append({"type": "raw", "raw": lines[i]})
        i += 1
    return items


# The reverse of parse_castep_input: turns a list of items back into text.
# Used to copy the parts of your .cell (species_pot, ...) and .param that the script doesn't change
# into every supercell's input files.
#   Called by: make_param(), write_cell()
def render(items):
    out = []
    for it in items:
        if it["type"] == "block":
            out += [f"%block {it['name']}"] + it["body"] + [f"%endblock {it['name']}"]
        else:
            out.append(it["raw"])
    return "\n".join(out)


# The lines of a %block, with comments and empty lines removed, split into words.
# Used for the lattice and positions blocks of a .cell, species_pot, and the blocks of the .conv
# file (%block defect, %block partitions, ...).
#   Called by: convert_settings(), series_main(), read_batch_file(), read_primitive()
#   Calls:     strip_comment()
def block_rows(item):
    rows = [strip_comment(line).split() for line in item["body"]]
    return [r for r in rows if r]


# Handles the optional unit line at the top of a lattice or positions block.
# If the first row is a unit (ang, bohr, a0, nm) it is removed and its factor to Angstrom returned;
# otherwise 1. Stops on an unknown unit.
#   Called by: read_primitive()
def pop_unit(rows):
    if rows and len(rows[0]) == 1:
        unit = rows.pop(0)[0].lower()
        if unit not in UNITS:
            die(f"unknown length unit '{unit}' in the .cell file")
        return UNITS[unit]
    return 1.0


# Turns a species label into an element symbol.
# 'Ca', 'ca', 'Ca:1' (a CASTEP label) and '20' (an atomic number) all give 'Ca'. Stops on anything
# that isn't an element.
#   Called by: series_main(), parse_defect(), read_primitive()
def element(label):
    lab = label.split(":")[0]
    sym = chemical_symbols[int(lab)] if lab.isdigit() else lab[:1].upper() + lab[1:].lower()
    if sym not in chemical_symbols[1:]:
        die(f"unrecognised element '{label}'")
    return sym


# Builds an ASE Atoms object from a parsed .cell file.
# Lattice from lattice_cart or lattice_abc, positions from positions_frac or positions_abs (with
# their units), and any SPIN= values as initial magnetic moments. Used for your host cell at the
# start, which every supercell is built from, and to read CASTEP's <seed>-out.cell after a
# relaxation.
#   Called by: final_atoms(), last_geometry(), series_main(), read_structure(), relaxed_atoms()
#   Calls:     block_rows(), element(), pop_unit()
def read_primitive(items):
    blocks = {it["name"]: it for it in items if it["type"] == "block"}
    if "lattice_cart" in blocks:
        rows = block_rows(blocks["lattice_cart"])
        unit = pop_unit(rows)
        cell = np.array(rows[:3], float) * unit
    elif "lattice_abc" in blocks:
        rows = block_rows(blocks["lattice_abc"])
        unit = pop_unit(rows)
        cell = cellpar_to_cell([float(x) * unit for x in rows[0][:3]] +
                               [float(x) for x in rows[1][:3]])
    else:
        die("no lattice_cart or lattice_abc block in the .cell file")
    key = "positions_frac" if "positions_frac" in blocks else "positions_abs"
    if key not in blocks:
        die("no positions_frac or positions_abs block in the .cell file")
    rows = block_rows(blocks[key])
    unit = pop_unit(rows) if key == "positions_abs" else 1.0
    symbols, xyz, spins = [], [], []
    for r in rows:
        symbols.append(element(r[0]))
        xyz.append([float(x) for x in r[1:4]])
        m = re.search(r"spin\s*=\s*([-+.\deE]+)", " ".join(r[4:]), re.I)
        spins.append(float(m.group(1)) if m else 0.0)
    if key == "positions_frac":
        atoms = Atoms(symbols, scaled_positions=xyz, cell=cell, pbc=True)
    else:
        atoms = Atoms(symbols, positions=np.array(xyz) * unit, cell=cell, pbc=True)
    atoms.set_initial_magnetic_moments(spins)
    return atoms


# Writes the CASTEP .cell file of one calculation.
# Writes a comment line (title), lattice_cart, positions_frac in species order (written_order) with
# SPIN= where set, the k-point grid and fix_all_cell : true (the lattice stays at the bulk value),
# then everything kept from your .cell (keep_items: species_pot, ...; for defect cells without the
# symmetry keywords). Batch mode can also leave out fix_all_cell (fix_cell=False: a variable-cell
# relaxation), add a kpoint_mp_offset (offset) and add lines at the end (extra, e.g.
# %block ionic_constraints); cell_options works these out.
#   Called by: restart(), write_inputs()
#   Calls:     render(), written_order()
def write_cell(path, atoms, grid, keep_items, title, fix_cell=True, extra=None, offset=None):
    syms = atoms.get_chemical_symbols()
    frac = atoms.get_scaled_positions(wrap=True)
    frac[np.isclose(frac, 0.0, atol=1e-10) | np.isclose(frac, 1.0, atol=1e-10)] = 0.0
    mags = atoms.get_initial_magnetic_moments()
    lines = [f"# {title}", "", "%block lattice_cart", "ang"]
    lines += ["  " + "".join(f"{x:20.12f}" for x in v) for v in np.asarray(atoms.cell)]
    lines += ["%endblock lattice_cart", "", "%block positions_frac"]
    for i in written_order(atoms):
        line = f"  {syms[i]:<3s}" + "".join(f"{x:18.12f}" for x in frac[i])
        if abs(mags[i]) > 1e-8:
            line += f"  SPIN={mags[i]:.4f}"
        lines.append(line)
    lines += ["%endblock positions_frac", "", "kpoint_mp_grid : {} {} {}".format(*grid)]
    if offset is not None and any(abs(x) > 1e-12 for x in offset):
        lines.append("kpoint_mp_offset : " + " ".join(f"{x:.10f}" for x in offset))
    if fix_cell:
        lines.append("fix_all_cell : true")            # lattice stays at the bulk value
    lines += ["", render(keep_items)] + (["", extra.strip()] if extra else [])
    path.write_text(re.sub(r"\n\s*\n(\s*\n)+", "\n\n", "\n".join(lines)).rstrip() + "\n")


# The atom order write_cell uses: grouped by species, original order within each.
# (CASTEP reorders the species by atomic number in its output, but keeps this order within each
# species: castep_order.)
#   Called by: write_cell()
def written_order(atoms):
    """Atom order used by write_cell (grouped by species)."""
    syms = atoms.get_chemical_symbols()
    species = list(dict.fromkeys(syms))
    return sorted(range(len(atoms)), key=lambda i: (species.index(syms[i]), i))


# Where the atoms CASTEP lists (their element symbols, in its order) are in an Atoms object.
# CASTEP groups the atoms by species, in order of increasing atomic number (not the order of the
# .cell), and keeps the order of the .cell within each species, which is the order within each
# species of the Atoms object (write_cell). So the k-th atom of an element in CASTEP's list is its
# k-th atom in the Atoms object, whatever order CASTEP puts the elements in. Returns the indices
# into atoms in CASTEP's order, or None if the elements don't match.
#   Called by: final_atoms(), last_geometry()
def castep_order(atoms, symbols):
    """Indices into atoms of the atoms CASTEP lists as symbols (in its order), or None."""
    ours = atoms.get_chemical_symbols()
    if sorted(ours) != sorted(symbols):
        return None
    where, seen = {}, {}
    for i, s in enumerate(ours):
        where.setdefault(s, []).append(i)
    order = []
    for s in symbols:
        order.append(where[s][seen.get(s, 0)])
        seen[s] = seen.get(s, 0) + 1
    return order


# Writes the .param text for one kind of calculation from your .param.
# Removes task, continuation, reuse and write_cell_structure (and, for the pristine bulk cell, spin
# and charge: it is neutral and closed shell), then puts 'task : geometryoptimization' +
# 'write_cell_structure : true' at the top for a relaxation, or 'task : singlepoint' otherwise.
# In batch mode a job can give its own task (e.g. phonon+efield) instead.
# run_time and continuation are added later by set_param (from finalize and restart).
#   Called by: write_inputs()
#   Calls:     render()
def make_param(param_items, kind, task=None):
    drop = {"task", "continuation", "reuse", "write_cell_structure"}
    if kind == "bulk":                       # the pristine host is neutral, closed shell
        drop |= {"spin", "charge"}
    kept = [it for it in param_items if not (it["type"] == "kw" and it["key"] in drop)]
    if task:                                 # batch mode: any CASTEP task
        head = f"task : {task}\n" + ("write_cell_structure : true\n" if kind == "geom" else "")
    else:
        head = ("task : geometryoptimization\nwrite_cell_structure : true\n" if kind == "geom"
                else "task : singlepoint\n")
    return head + render(kept).rstrip() + "\n"


# --------------------------------------------------------------------------
# k-points
# --------------------------------------------------------------------------
# Lengths of the three reciprocal lattice vectors, in CASTEP's 2*pi units (1/Ang).
# For a cubic cell this is 1/a. k-point spacings are measured in these units, as in CASTEP's
# kpoint_mp_spacing.
#   Called by: kgrid(), target_spacing()
def recip_lengths(cell):
    """|b_i| in 1/Angstrom in 2*pi units (CASTEP's kpoint_mp_spacing convention)."""
    return np.linalg.norm(np.linalg.inv(np.asarray(cell, float)).T, axis=1)


# The Monkhorst-Pack grid for a cell at the target k-point spacing.
# In each direction, the smallest number of k-points whose spacing is no larger than the target
# (rounded up, so every supercell is sampled at least as densely as the primitive cell).
#   Called by: series_main(), write_inputs()
#   Calls:     recip_lengths()
def kgrid(cell, spacing):
    return [max(1, math.ceil(L / spacing - 1e-4)) for L in recip_lengths(cell)]


# Decides the k-point spacing used for every supercell.
# In order of priority: kspacing in the input file; kgrid in the input file; kpoint_mp_grid in your
# .cell; kpoint_mp_spacing in your .cell; CASTEP's default (0.1 1/Ang, with a warning). For a grid
# it takes the coarsest spacing of the three directions. Returns (spacing, where it came from). The
# spacing goes into ctx['spacing'] and the work directory's fingerprint (setup.json).
#   Called by: series_main()
#   Calls:     recip_lengths()
def target_spacing(items, prim, args):
    if args.kspacing:
        return args.kspacing, "kspacing in the input file"
    kws = {it["key"]: it["value"] for it in items if it["type"] == "kw"}
    if args.kgrid:
        grid, src = args.kgrid, "kgrid in the input file"
    else:
        if any(k in kws for k in ("kpoint_mp_offset", "kpoints_mp_offset")):
            log("WARNING: kpoint_mp_offset in the .cell is not carried over to the supercells")
        if any(it["type"] == "block" and it["name"] in ("kpoint_list", "kpoints_list")
               for it in items):
            die("the .cell uses an explicit k-point list; give the primitive MP grid with "
                "kgrid in the input file")
        grid = next(([int(x) for x in kws[k].split()[:3]]
                     for k in ("kpoint_mp_grid", "kpoints_mp_grid") if k in kws), None)
        src = "kpoint_mp_grid in the .cell"
        if grid is None:
            for k in ("kpoint_mp_spacing", "kpoints_mp_spacing"):
                if k in kws:
                    val = kws[k].split()
                    s = float(val[0])
                    return (s / BOHR if len(val) > 1 and "bohr" in val[1].lower() else s), k
            log("WARNING: no k-points in the .cell; using CASTEP's default spacing, 0.1 1/Ang")
            return 0.1, "CASTEP default"
    return float(max(recip_lengths(prim.cell) / np.array(grid))), f"{src} {grid}"


# --------------------------------------------------------------------------
# Supercells and defects
# --------------------------------------------------------------------------
# Reads the defect from %block defect, e.g. Sb_Ca + Vac_O.
# Returns one dict per component: the new element (None for a vacancy), the host element it
# replaces, and the primitive-cell site it goes on (prim_site; ':k' picks the k-th site of that
# element, and the first component defaults to site 1 so it is always the same site), with the
# sites a pure translation of the crystal makes identical to it (prim_sites: translation_copies,
# e.g. all four Ca of a conventional rocksalt cell), its starting spin ('@s', e.g. Sb_Ca@3; 0 if
# not given) and its name without the spin. Stops on bad tokens, an element not in the host, an
# ambiguous 'V_..', or a spin on a vacancy. The components are used by defect_vectors and
# insert_defect, and named by defect_label.
#   Called by: series_main()
#   Calls:     element(), translation_copies()
def parse_defect(tokens, prim):
    host_species = prim.get_chemical_symbols()
    comps = []
    for tok in tokens:
        m = re.fullmatch(r"([A-Za-z]+|\d+)_([A-Za-z]{1,2})(?::(\d+))?(?:@([-+]?\d+(?:\.\d*)?))?",
                         tok)
        if not m:
            die(f"bad defect component '{tok}': use e.g. Sb_Ca (Sb on a Ca site), Vac_O, or "
                "Sb_Ca@3 (Sb on a Ca site, starting with a spin of 3)")
        new, host, k, spin = m.groups()
        host = element(host)
        if host not in host_species:
            die(f"{tok}: there is no {host} in the host cell")
        if new.lower() in ("vac", "va"):
            new = None
        elif new in ("V", "v"):
            die(f"'{tok}' is ambiguous: write Vac_{host} for a vacancy, or 23_{host} for vanadium")
        else:
            new = element(new)
        if spin is not None and new is None:
            die(f"{tok}: a vacancy has no atom to put a starting spin on")
        n_sites = host_species.count(host)
        if k is not None and not 1 <= int(k) <= n_sites:
            die(f"{tok}: the primitive cell has {n_sites} {host} site(s)")
        comps.append({"tok": tok, "name": tok.split("@")[0], "new": new, "host": host,
                      "k": int(k) if k else None, "spin": float(spin) if spin else 0.0})
    if comps[0]["k"] is None:     # anchor: always the same primitive-cell site
        comps[0]["k"] = 1
    for c in comps:               # primitive index of the k-th site of that species
        idx = [i for i, s in enumerate(host_species) if s == c["host"]]
        c["prim_site"] = idx[c["k"] - 1] if c["k"] else None
        c["prim_sites"] = translation_copies(prim, c["prim_site"]) if c["k"] else None
    return comps


# The pure translations of a crystal (fractional vectors, the zero one first): the shifts that map
# every atom of the cell onto an atom of the same element, such as the face-centring vectors of a
# conventional FCC cell. A primitive cell has only the zero one. tol is in Angstrom.
#   Called by: translation_copies(), the workflow
def pure_translations(prim, tol=1e-3):
    frac, cell = prim.get_scaled_positions(), np.asarray(prim.cell)
    syms = np.array(prim.get_chemical_symbols())
    ref = min(range(len(syms)), key=lambda i: (np.sum(syms == syms[i]), i))  # rarest element
    out = []
    for j in np.flatnonzero(syms == syms[ref]):
        t = (frac[j] - frac[ref] + 0.5) % 1.0 - 0.5
        d = ((frac + t)[:, None, :] - frac[None, :, :] + 0.5) % 1.0 - 0.5
        dist = np.linalg.norm(d @ cell, axis=2)            # [atom moved, atom it lands on]
        k = dist.argmin(axis=1)
        if np.all(dist[np.arange(len(frac)), k] < tol) and np.all(syms[k] == syms):
            out.append(t)
    return sorted(out, key=lambda t: float(np.linalg.norm(t @ cell)))


# The sites of a cell that a pure translation of the crystal (pure_translations) maps site i onto,
# i included: a defect on any of them is the same defect, moved. Just [i] in a primitive cell.
#   Called by: parse_defect(), the workflow
#   Calls:     pure_translations()
def translation_copies(prim, i):
    frac, cell = prim.get_scaled_positions(), np.asarray(prim.cell)
    out = set()
    for t in pure_translations(prim):
        d = (frac - (frac[i] + t) + 0.5) % 1.0 - 0.5
        out.add(int(np.argmin(np.linalg.norm(d @ cell, axis=1))))
    return sorted(out)


# The defect's name, e.g. 'Sb_Ca-Vac_O' (':' becomes 's').
# Used for the default work directory, conv_<cell name>_<defect>. Any starting spin ('@3') is left
# out, so adding one doesn't change the directory.
#   Called by: series_main()
def defect_label(comps):
    return "-".join(c["name"].replace(":", "s") for c in comps)   # without any @spin


# Makes the supercell P x (primitive cell) with ASE's make_supercell.
# Each atom is first tagged with the index of the primitive-cell site it came from (array
# 'prim_index'), so candidates() can pick out e.g. 'copies of the 2nd O site'. Used for every size
# (P = n*T) and for the 3x3x3 reference cell in defect_vectors.
#   Called by: defect_vectors(), series_main()
def build_supercell(prim, P):
    p = prim.copy()
    p.set_array("prim_index", np.arange(len(p)))   # remember which primitive site each atom is
    return make_supercell(p, P)


# The atoms of a supercell where one defect component can go.
# Atoms of the host element; if the component names a primitive-cell site, only copies of that site
# and of the sites identical to it by a pure translation (prim_sites), using 'prim_index' from
# build_supercell.
#   Called by: anchor_site(), defect_vectors(), insert_defect(), series_main()
def candidates(sc, comp):
    mask = np.array(sc.get_chemical_symbols()) == comp["host"]
    if comp["prim_site"] is not None:
        mask &= np.isin(sc.arrays["prim_index"], comp.get("prim_sites") or [comp["prim_site"]])
    return np.flatnonzero(mask)


# Where the first defect component goes: the candidate site that puts the whole defect as close to
# the cell centre as it can be. The defect's centre is the mean position of its components (this
# site, and the others at vecs from it), so a complex is centred as a whole, not just its first
# component; of equally good sites, the one whose farthest component is nearest the centre wins
# (then the lowest index). Only the site changes, never the orientation of the complex, so it is
# the same complex in every size. With no vecs: the candidate nearest the centre.
#   Called by: defect_vectors(), insert_defect(), series_main(), the workflow
#   Calls:     candidates()
def anchor_site(sc, comp, vecs=()):
    cand = candidates(sc, comp)
    centre = np.asarray(sc.cell).sum(axis=0) / 2
    offsets = np.array([np.zeros(3)] + [np.asarray(v, float) for v in vecs])
    pos = sc.positions[cand]
    mid = np.linalg.norm(pos + offsets.mean(axis=0) - centre, axis=1)
    far = np.linalg.norm(pos[:, None, :] + offsets[None, :, :] - centre, axis=2).max(axis=1)
    best = min(range(len(cand)), key=lambda i: (round(mid[i], 4), round(far[i], 4), i))
    return int(cand[best])


# Works out the geometry of a defect complex once, in a 3x3x3 reference cell.
# The first component goes on anchor_site (on a copy of its own primitive-cell site) and each other
# component on the nearest free site of its kind; the vectors between them are returned.
# insert_defect places the components at these same vectors in every supercell, so the complex is
# identical in every size.
#   Called by: series_main()
#   Calls:     anchor_site(), build_supercell(), candidates()
def defect_vectors(prim, comps):
    """Vectors from the first component to the others, found in a 3x3x3 reference cell."""
    ref = build_supercell(prim, 3 * np.eye(3, dtype=int))
    a = anchor_site(ref, dict(comps[0], prim_sites=None))
    used, vecs = {a}, []
    for c in comps[1:]:
        cand = [i for i in candidates(ref, c) if i not in used]
        if not cand:
            die(f"no free {c['host']} site for {c['tok']}")
        D, L = get_distances(ref.positions[[a]], ref.positions[cand], cell=ref.cell, pbc=True)
        j = int(np.argmin(L[0]))
        used.add(cand[j])
        vecs.append(D[0, j])
    return vecs


# Puts the defect into one supercell.
# The first component goes on anchor_site (which centres the whole defect in the cell), or on the
# atom anchor if given, and the others at the vectors from defect_vectors. Stops if a site can't be
# found, or if the complex would overlap its own periodic images (cell too small). Substitutes
# elements, giving each new atom the starting spin from its component (0 unless '@s' was given;
# write_cell writes it as SPIN=), and deletes vacancy sites. Returns the defect Atoms, the
# distances between the components, how far the defect's centre (the mean position of its
# components) is from the cell centre (both shown in the size table) and the first component's
# atom index.
#   Called by: series_main()
#   Calls:     anchor_site(), candidates()
def insert_defect(sc, comps, vecs, label, anchor=None):
    a = anchor_site(sc, comps[0], vecs) if anchor is None else int(anchor)
    middle = sc.positions[a] + sum((np.asarray(v, float) for v in vecs), np.zeros(3)) / len(comps)
    offset = float(np.linalg.norm(middle - np.asarray(sc.cell).sum(axis=0) / 2))
    sites = [a]
    for c, v in zip(comps[1:], vecs):
        cand = candidates(sc, c)
        _, L = get_distances(sc.positions[[a]] + v, sc.positions[cand], cell=sc.cell, pbc=True)
        j = int(np.argmin(L[0]))
        if L[0, j] > 0.05:
            die(f"could not place {c['tok']} in the {label} supercell")
        if cand[j] in sites:
            die(f"the {label} supercell is too small for this defect complex (its periodic "
                "images overlap) - start from a larger size")
        sites.append(int(cand[j]))
    seps = [float(sc.get_distance(a, s, mic=True)) for s in sites[1:]]
    defect = sc.copy()
    syms = defect.get_chemical_symbols()
    mags = defect.get_initial_magnetic_moments()
    for c, s in zip(comps, sites):
        if c["new"]:
            syms[s], mags[s] = c["new"], c["spin"]     # written as SPIN= in the .cell
    defect.set_chemical_symbols(syms)
    defect.set_initial_magnetic_moments(mags)
    vacancies = sorted((s for c, s in zip(comps, sites) if c["new"] is None), reverse=True)
    if vacancies:
        del defect[vacancies]
    return defect, seps, offset, a


# The length of the shortest lattice vector.
# This is the distance from the defect to its nearest periodic image: L_min in the size table and
# the results.
#   Called by: series_main()
def min_image_distance(cell):
    """Shortest lattice vector = distance from the defect to its nearest periodic image."""
    r = range(-3, 4)
    combos = np.array([(i, j, k) for i in r for j in r for k in r if (i, j, k) != (0, 0, 0)])
    return float(np.linalg.norm(combos @ np.asarray(cell, float), axis=1).min())


# --------------------------------------------------------------------------
# Calculations: dry run, resources, jobscript, submission, results
# --------------------------------------------------------------------------
# One CASTEP calculation and the directory it lives in.
# kind is 'geom' (the defect relaxation), 'defect' (a single point on the relaxed defect cell) or
# 'bulk' (a single point on the pristine supercell). The name and seed are e.g.
# sc_3x3x3_defect_geom, and everything about it lives in <workdir>/<name>/. Its job_info.json, read
# with info() and written with save(), is the script's memory between runs: k-grid, the plan
# (cores, nodes, partition, memory, walltime), job ids, every run of the job (submit), timings
# (measure_times), measured memory (calibrate) and restart counts (restart). series_main() makes
# one Calc per calculation; most other functions take a Calc. In batch mode (run_batch) a Calc is
# one job of the batch: kind 'geom' (any relaxation) or 'sp' (anything else), with its own name and
# options.
class Calc:
    """kind: 'geom' (defect relaxation), 'defect' (single point on the relaxed defect
    cell) or 'bulk' (single point on the pristine supercell); in batch mode 'geom' or
    'sp'."""
    # Sets the calculation's name, directory and output file paths.
    # <seed>.castep is CASTEP's output; <seed>.dryrun.castep is where a finished dry run is moved
    # (dryrun_output). Nothing is written to disk here: write_inputs creates the directory.
    # The other arguments are for batch mode (run_batch); the size series uses the defaults:
    #   name           the directory and seedname (default <label>_defect_geom etc.)
    #   task           CASTEP task instead of geometryoptimization / singlepoint (make_param)
    #   overrides      {keyword: value} set in the .param after it is written (write_inputs)
    #   fixed_kgrid    an MP grid to use instead of the one from the k-point spacing
    #   kpoint_offset  'gamma' or three numbers (cell_options)
    #   fix_cell       False for a variable-cell relaxation (no fix_all_cell)
    #   symmetry       keep the symmetry keywords of your .cell (default: only the pristine 'bulk')
    #   cell_extra     text added to the .cell (cell_options)
    #   param_items    a .param of its own (parsed), instead of the batch's
    #   group          jobs are only timed from jobs of the same group (estimate_walltime)
    #   Called by: series_main(), run_batch()
    def __init__(self, workdir, label, kind, atoms, name=None, task=None, overrides=None,
                 fixed_kgrid=None, kpoint_offset=None, fix_cell=True, symmetry=None,
                 cell_extra=None, param_items=None, group=None):
        self.label, self.kind, self.atoms = label, kind, atoms
        self.name = self.seed = name or f"{label}_{'defect_geom' if kind == 'geom' else kind}"
        self.dir = workdir / self.name
        self.castep = self.dir / f"{self.seed}.castep"
        self.dryrun_file = self.dir / f"{self.seed}.dryrun.castep"
        self.task, self.overrides, self.fixed_kgrid = task, overrides or {}, fixed_kgrid
        self.kpoint_offset, self.fix_cell, self.cell_extra = kpoint_offset, fix_cell, cell_extra
        self.symmetry = kind == "bulk" if symmetry is None else symmetry
        self.param_items, self.group = param_items, group
        self._info, self._done = None, False       # job_info.json, read once; done() once true

    # Reads job_info.json (an empty dict if it doesn't exist yet): a copy the caller may change and
    # save back. The file is read once and then kept in memory: only save() writes it.
    #   Called by: almost every function that plans, submits or checks a calculation
    #   Calls:     Calc.peek()
    def info(self):
        return json.loads(json.dumps(self.peek()))

    # job_info.json itself, as kept in memory, for reading only: the checks that run over every job
    # of a batch on every poll (run_batch, wait_for, calibrate, measure_times, estimate_walltime),
    # which with thousands of jobs would otherwise read thousands of files each time.
    #   Called by: Calc.info(), calibrate(), estimate_walltime(), measure_times(), run_batch(),
    #              wait_for()
    def peek(self):
        if self._info is None:
            p = self.dir / INFO
            self._info = json.loads(p.read_text()) if p.exists() else {}
        return self._info

    # Writes job_info.json, and keeps it in memory for info() and peek(). Saving a job with timings
    # counts in TIMINGS, so that estimate_walltime makes its list of timed jobs again.
    # Functions read it with info(), change it and save it back.
    #   Called by: every function that records something in job_info.json
    def save(self, info):
        if "timing" in info or "timing" in (self._info or {}):
            TIMINGS[0] += 1
        (self.dir / INFO).write_text(json.dumps(info, indent=2))
        self._info = json.loads(json.dumps(info))

    # The whole .castep: every run of the job, since each restart appends to it.
    # Used to see whether the calculation is done, and to read energies, structures and the dry
    # run. Empty string if there is no .castep yet.
    #   Called by: batch_energy(), Calc.done(), Calc.finished(), castep_charge(), dryrun_output(),
    #              final_atoms(), handle_failure(), job_result(), last_geometry(), read_energy(),
    #              relaxed_atoms()
    def output(self):
        return self.castep.read_text(errors="replace") if self.castep.exists() else ""

    # Only the output of the latest job: the .castep from where that run started.
    # submit() records the size of the .castep when each job is submitted (info['runs']), and the
    # output after that point belongs to the latest run. Lets restart and handle_failure see how
    # the last job ended without being misled by earlier runs.
    #   Called by: handle_failure(), restart()
    def run_text(self):
        """Output of the latest run only (a continuation appends to the .castep)."""
        if not self.castep.exists():
            return ""
        raw = self.castep.read_bytes()
        start = (self.info().get("runs") or [{"start": 0}])[-1]["start"]
        return raw[start if start <= len(raw) else 0:].decode(errors="replace")

    # True if CASTEP printed 'Total time', i.e. reached the end of a run.
    # Not used by the current workflow, which uses done() instead (a relaxation also has to have
    # converged).
    #   Called by: nothing at present
    #   Calls:     Calc.output()
    def finished(self):
        return bool(FINISHED.search(self.output()))

    # True when the calculation is complete.
    # CASTEP printed 'Total time' and, for a relaxation, 'Geometry optimization completed
    # successfully'. This is what wait_for waits for; it is also used to skip calculations that are
    # already done. Once true it is remembered, so a finished job's .castep isn't read again.
    #   Called by: calibrate(), job_result(), measure_times(), prepare_all(), prepare_sp(),
    #              remove_check(), run_batch(), series_main(), wait_for()
    #   Calls:     Calc.output()
    def done(self):
        """Finished, and for a geometry optimisation also converged."""
        if not self._done:                     # once done it stays done: not read again
            out = self.output()
            self._done = bool(FINISHED.search(out)) and (self.kind != "geom" or GEOM_DONE in out)
        return self._done

    # done() as far as it is known already, without reading the .castep: True once done() has
    # found the calculation done. For run_batch's progress.txt, which looks at every job on every
    # poll, after run_batch and wait_for have checked them.
    #   Called by: run_batch()
    def known_done(self):
        return self._done

    # CASTEP's error files (<seed>*.err), written when CASTEP aborts.
    # wait_for treats their appearance as a failure straight away (handle_failure); restart reads
    # them to recognise an unreadable .check and renames them before it resubmits, so the new job
    # doesn't look failed.
    #   Called by: handle_failure(), local_dryrun(), restart(), run_batch(), wait_for()
    def err_files(self):
        return sorted(self.dir.glob(f"{self.seed}*.err"))


# Replaces each placeholder ({SEED}, {NCORES}, {MEM_GB}, ...) in a template with its value.
# Used for job.sh, dryrun.sh, and the placeholders allowed in partition and dry-run #SBATCH
# options.
#   Called by: dryrun_sbatch(), dryrun_script(), finalize(), pack_script(), submit_dryrun_pack()
def fill(template, subs):
    for key, val in subs.items():
        template = template.replace(key, str(val))
    return template


# Sets one #SBATCH option in a jobscript, e.g. '--partition=highmem'.
# Replaces the option wherever it already appears, in either spelling (-p/--partition, -t/--time,
# ...: SBATCH_SHORT), removing duplicates; otherwise adds it after the last #SBATCH line. A memory
# request (--mem, --mem-per-cpu, ...) replaces any other memory request.
#   Called by: dryrun_sbatch(), finalize(), pack_script(), restart(), set_exclusive(),
#              set_threads(), submit_array(), submit_pack(), submit_pack_array()
def add_sbatch_option(text, option):
    """Set '#SBATCH <option>' (e.g. '--qos=highmem' or '-C icelake') in a jobscript,
    replacing that option if it is already there in either spelling (-p / --partition;
    for a memory request, any other memory request), else adding it after the last
    #SBATCH line."""
    flag = re.match(r"--?[\w-]+", option).group(0)
    flag = next((lng for lng, sht in SBATCH_SHORT.items() if sht == flag), flag)
    names = {flag, SBATCH_SHORT.get(flag, flag)} | (set(MEM_FLAGS) if flag in MEM_FLAGS else set())
    pat = re.compile(r"#SBATCH\s+(%s)(?=[=\s]|$)" % "|".join(map(re.escape, names)))
    lines = text.splitlines()
    hits = [i for i, ln in enumerate(lines) if pat.match(ln)]
    if hits:
        lines[hits[0]] = f"#SBATCH {option}"
        lines = [ln for i, ln in enumerate(lines) if i not in hits[1:]]
    else:
        last = max([i for i, ln in enumerate(lines) if ln.startswith("#SBATCH")], default=0)
        lines.insert(last + 1, f"#SBATCH {option}")
    return "\n".join(lines) + "\n"


# How many OpenMP threads each MPI process of a job gets (omp_threads).
# Each process already has cpus cores (--cpus-per-task, from choose_resources: more than one when
# a node is underpopulated to fit the memory). auto uses them all, up to OMP_AUTO_MAX; a number N
# uses up to N; 1 (the default) keeps one thread per process. Always 1 without cores_per_node.
#   Called by: finalize()
def thread_count(r, args):
    """OpenMP threads per MPI process for a job laid out as r (choose_resources)."""
    if not args.cores_per_node or args.omp_threads == 1:
        return 1
    cap = OMP_AUTO_MAX if args.omp_threads == "auto" else args.omp_threads
    return max(1, min(cap, r.cpus))


# Makes a jobscript run each MPI process with that many OpenMP threads, on cores the job already
# has but would otherwise leave idle.
# Sets OMP_NUM_THREADS (the template's export if it has one, else a new line before the CASTEP
# line), and adds OMP_PLACES=cores, SRUN_CPUS_PER_TASK (srun doesn't pass --cpus-per-task on by
# itself) and --cpus-per-task if the template has none of them. Returns the script unchanged for
# one thread.
#   Called by: finalize()
#   Calls:     add_sbatch_option()
def set_threads(script, threads, cpus):
    """Run each MPI process with `threads` OpenMP threads (each process has `cpus` cores)."""
    if threads <= 1:
        return script
    if not re.search(r"^#SBATCH\s+(--cpus-per-task\b|-c\b)", script, re.M):
        script = add_sbatch_option(script, f"--cpus-per-task={cpus}")
    add = []
    if re.search(r"^\s*export\s+OMP_NUM_THREADS=", script, re.M):
        script = re.sub(r"^(\s*export\s+OMP_NUM_THREADS=)\S*", rf"\g<1>{threads}", script,
                        flags=re.M)
    else:
        add.append(f"export OMP_NUM_THREADS={threads}")
    if not re.search(r"^\s*export\s+OMP_PLACES=", script, re.M):
        add.append("export OMP_PLACES=cores")
    if "SRUN_CPUS_PER_TASK" not in script:
        add.append("export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK")
    if add:
        lines = script.splitlines()
        at = next((k for k, ln in enumerate(lines) if CASTEP_LINE.match(ln)), len(lines))
        lines[at:at] = add
        script = "\n".join(lines) + "\n"
    return script


# With charging : core, makes a jobscript ask for whole nodes (#SBATCH --exclusive) or not: a job
# bigger than a node must have its nodes to itself (and is charged for all their cores), while a
# smaller one shares its node, and is charged only for its own cores. Takes out any --exclusive
# the template has, then adds it back for whole nodes.
#   Called by: dryrun_script(), finalize(), pack_script(), submit_dryrun_pack()
#   Calls:     add_sbatch_option()
def set_exclusive(script, whole):
    lines = [ln for ln in script.splitlines() if not re.match(r"#SBATCH\s+--exclusive\b", ln)]
    script = "\n".join(lines) + "\n"
    return add_sbatch_option(script, "--exclusive") if whole else script


# Reads the value of an #SBATCH option from a jobscript, in either spelling.
# Used to find a job's walltime (--time / -t).
#   Called by: plan_walltime(), restart(), submit()
def sbatch_value(text, flag):
    """The value of an #SBATCH option in a jobscript (either spelling), or None."""
    short = SBATCH_SHORT.get(flag)
    pat = re.escape(flag) + r"(?:=|\s+)" + (f"|{re.escape(short)}\\s*" if short else "")
    for line in text.splitlines():
        m = re.match(r"#SBATCH\s+(?:%s)(\S+)" % pat, line)
        if m:
            return m.group(1)
    return None


# Converts a Slurm time (min, min:s, h:min:s, d-h, d-h:min, d-h:min:s) to seconds.
# Returns None if the text isn't a Slurm time.
#   Called by: _slurm_time(), job_timing(), plan_walltime(), restart(), share_walltime(),
#              submit_array(), submit_pack(), submit_pack_array()
def slurm_seconds(spec):
    """A Slurm time (min, min:s, h:min:s, d-h, d-h:min, d-h:min:s) in seconds, or None."""
    m = re.fullmatch(r"(?:(\d+)-)?(\d+)(?::(\d+))?(?::(\d+))?", spec.strip())
    if not m:
        return None
    d, a, b, c = m.groups()
    if d is not None:
        return ((int(d) * 24 + int(a)) * 60 + int(b or 0)) * 60 + int(c or 0)
    if c is not None:
        return (int(a) * 60 + int(b)) * 60 + int(c)
    if b is not None:
        return int(a) * 60 + int(b)
    return int(a) * 60


# The reverse of slurm_seconds: seconds to Slurm's [d-]h:mm:ss.
#   Called by: plan_walltime(), submit_array(), submit_pack(), submit_pack_array()
def slurm_time(seconds):
    d, rest = divmod(int(math.ceil(seconds)), 86400)
    h, rest = divmod(rest, 3600)
    return (f"{d}-" if d else "") + f"{h}:{rest // 60:02d}:{rest % 60:02d}"


# Joins words into #SBATCH options so an option keeps its separate value.
# ['--mem=4G', '-C', 'icelake'] -> ['--mem=4G', '-C icelake']. Used for the extra options on a
# %block partitions line and for dryrun_options.
#   Called by: _options(), parse_partitions()
def group_options(tokens):
    """['--mem=4G', '-C', 'icelake'] -> ['--mem=4G', '-C icelake']"""
    options = []
    for tok in tokens:
        if tok.startswith("-") or not options:
            options.append(tok)
        else:
            options[-1] += " " + tok
    return options


# The template of the dry-run jobs: dryrun_jobscript if given; otherwise your main template, with
# the line that runs CASTEP changed to one process with --dryrun (CASTEP_RUN, NPROC_VARS).
#   Called by: batch_args(), dryrun_script(), submit_dryrun_pack()
def dryrun_template_text(job_template, dryrun_template):
    if dryrun_template:
        return dryrun_template
    lines = []
    for line in job_template.splitlines():
        if CASTEP_RUN.search(line) and not line.lstrip().startswith("#"):
            line = CASTEP_RUN.sub(r"\1--dryrun {SEED}", line).replace("{NCORES}", "1")
            line = NPROC_VARS.sub("1", line)
        lines.append(line)
    return "\n".join(lines) + "\n"


# The GB of memory each core of a dry-run job brings, with charging : core: that of
# dryrun_partition if %block partitions lists it, otherwise of the first partition listed (where
# the dry runs go), or mem_per_core. None if nothing says.
#   Called by: dryrun_cpus(), dryrun_lanes(), submit_dryrun_pack()
def dryrun_core_gb(args):
    parts = args.partitions or []
    name = args.dryrun_partition or (parts[0]["name"] if parts else None)
    return next((q["mem_per_core"] for q in parts if q["name"] == name),
                args.mem_per_core or min((q["mem_per_core"] for q in parts), default=None))


# How many cores each process of a dry-run job asks for: one, but with charging : core enough for
# the dry run's gb, as there the memory comes with the cores (dryrun_core_gb each).
#   Called by: check_runner(), dryrun_lanes(), dryrun_subs()
#   Calls:     dryrun_core_gb()
def dryrun_cpus(ncores, gb, args):
    if args.charging != "core":
        return 1
    per = dryrun_core_gb(args)
    return max(1, math.ceil(gb / (per * ncores))) if per else 1


# The placeholders of a dry-run job for ncores processes in all (several in a pack of dry runs).
#   Called by: dryrun_script(), submit_dryrun_pack()
#   Calls:     dryrun_cpus()
def dryrun_subs(seed, jobname, ncores, gb, args):
    return {"{SEED}": seed, "{JOBNAME}": jobname, "{NKPTS}": 0, "{NCORES}": ncores,
            "{NODES}": 1, "{TASKS_PER_NODE}": ncores,
            "{CPUS_PER_TASK}": dryrun_cpus(ncores, gb, args),
            "{MEM_GB}": math.ceil(gb), "{MEM_MB}": math.ceil(gb * 1024),
            "{MEM_PER_NODE_GB}": math.ceil(gb),
            "{MEM_PER_CORE_GB}": math.ceil(gb / ncores),
            "{MEM_PER_CORE_MB}": math.ceil(gb * 1024 / ncores),
            "{PARTITION}": args.dryrun_partition or (args.partitions[0]["name"]
                                                     if args.partitions else "")}


# Sets dryrun_partition, dryrun_time and dryrun_options in a dry-run job script.
#   Called by: dryrun_script(), submit_dryrun_pack()
#   Calls:     add_sbatch_option(), fill()
def dryrun_sbatch(script, subs, args):
    if args.dryrun_partition:
        script = add_sbatch_option(script, f"--partition={args.dryrun_partition}")
    if args.dryrun_time:
        script = add_sbatch_option(script, f"--time={args.dryrun_time}")
    for option in args.dryrun_options:
        script = add_sbatch_option(script, fill(option, subs))
    return script


# Writes the text of dryrun.sh, the batch job that runs CASTEP --dryrun.
# The dry-run template (dryrun_template_text) filled in for dryrun_cores processes, then
# dryrun_partition, dryrun_time and dryrun_options (dryrun_sbatch). With charging : core the dry run
# shares a node, so the template's --exclusive is taken out (set_exclusive). prepare_all writes the
# file and submits it, or runs it in a pack with other dry runs (dryrun_pack : N,
# submit_dryrun_pack).
#   Called by: prepare_all()
#   Calls:     dryrun_sbatch(), dryrun_subs(), dryrun_template_text(), fill(), set_exclusive()
def dryrun_script(calc, ctx, args):
    """Dry-run jobscript: dryrun_jobscript if given, otherwise the main template with
    dryrun_cores requested (on one node) and the CASTEP line run as one process with
    --dryrun. dryrun_partition, dryrun_time and dryrun_options then set those #SBATCH
    options."""
    subs = dryrun_subs(calc.seed, calc.name + "_dry", args.dryrun_cores, args.dryrun_mem_gb,
                       args)
    template = dryrun_template_text(ctx["job_template"], ctx["dryrun_template"])
    script = dryrun_sbatch(fill(template, subs), subs, args)
    return set_exclusive(script, False) if args.charging == "core" else script


# The text of a finished dry run, or None if it hasn't finished yet.
# The dry run writes <seed>.castep; once that contains CASTEP's memory estimate ('Approx. total
# storage required') it is renamed to <seed>.dryrun.castep, so the real job's .castep starts empty.
# wait_for also uses this to know when a dry-run job is done.
#   Called by: local_dryrun(), prepare_all(), wait_for()
#   Calls:     Calc.output()
def dryrun_output(calc):
    """Text of a completed dry run (moved to <seed>.dryrun.castep), or None."""
    if calc.dryrun_file.exists():
        return calc.dryrun_file.read_text(errors="replace")
    text = calc.output()
    if "Approx. total storage required" in text:
        calc.castep.rename(calc.dryrun_file)      # keep <seed>.castep free for the real run
        return text
    return None


# With dryrun_on : login, runs the dry run here instead of as a batch job.
# Runs dryrun_cmd (e.g. 'castep.serial --dryrun {seed}') in the calculation's directory and returns
# the output via dryrun_output. Stops with CASTEP's messages if it fails.
#   Called by: prepare_all()
#   Calls:     Calc.err_files(), dryrun_output()
def local_dryrun(calc, args):
    cmd = args.dryrun_cmd.format(seed=calc.seed)
    log(f"dry run here: {calc.name} ({len(calc.atoms)} atoms)")
    try:
        res = subprocess.run(cmd, shell=True, cwd=calc.dir, capture_output=True, text=True,
                             timeout=args.dryrun_timeout)
        out = res.stdout + res.stderr
    except subprocess.TimeoutExpired:
        out = "(timed out)"
    text = dryrun_output(calc)
    if text is None:
        errs = "".join(e.read_text(errors="replace") for e in calc.err_files())
        die(f"dry run failed in {calc.dir}\n  command: {cmd}\n{out[-1500:]}\n{errs[-1500:]}\n"
            "Is CASTEP loaded on this node? Change the command with dryrun_cmd.")
    return text


# Reads what the script needs from a dry run's output.
# The number of k-points CASTEP will use, the number of electrons, whether the calculation is
# spin-polarised, and the memory estimate (baseline, model, electronic and total, in MB). finalize
# plans the job from it; measured_replicated uses its memory figures.
#   Called by: finalize(), measured_replicated()
def parse_dryrun(text):
    # The last match of a pattern in the dry run, converted (CASTEP may print a value more than
    # once).
    def last(pattern, cast=float, default=None):
        found = re.findall(pattern, text)
        return cast(found[-1]) if found else default

    mem = {k: last(re.escape(label) + r"[^\n]*?([\d.]+)\s*MB", default=0.0) for k, label in
           (("baseline", "Baseline code"), ("model", "Model and support data"),
            ("electronic", "Electronic energy minimisation"),
            ("total", "Approx. total storage required"))}
    nk = re.search(r"Number of kpoints used\s*=\s*(\d+)", text)   # first = SCF k-points
    return {"nkpts": int(nk.group(1)) if nk else 1, "mem": mem,
            "nelec": last(r"number of\s+electrons\s*:\s*([\d.]+)"),
            "spin_polarized": "as spin-polarized" in text}


# The memory model: estimated memory per MPI process (MB) on ncores processes.
# (baseline + rest + S * (r + (1 - r) / p)) * mem_safety, with S = the dry run's model + electronic
# memory, p = processes and r = mem_replicated (the fraction of S every process keeps a full copy
# of). choose_resources uses it to decide which layouts and partitions fit. calibrate measures r by
# solving the same formula backwards (measured_replicated).
#   Called by: choose_resources(), core_resources()
def mem_per_core_mb(mem, ncores, args):
    """Heuristic per-process memory with the k-points (and G-vectors) shared over ncores.

    From a serial dry run: 'Model' + 'Electronic' are split across processes apart from
    a replicated fraction (mem_replicated); the baseline and the rest are per process.
    """
    base = mem["baseline"] if args.baseline_mb is None else args.baseline_mb
    shared = mem["model"] + mem["electronic"]
    rest = max(0.0, mem["total"] - mem["baseline"] - shared)
    r = args.mem_replicated
    return (base + rest + shared * (r + (1 - r) / ncores)) * args.mem_safety


# Splits the .castep into one piece per job run.
# Every restart appends to the same .castep; submit() records where each run starts (info['runs']).
# Output from before the first recorded run counts as a run too. Returns [(run, text)], oldest
# first, so timings and memory can be read run by run.
#   Called by: job_timing(), last_peak()
def run_segments(calc):
    """The .castep split into the job's runs, oldest first, as (run, text): every job
    appends its output to the .castep. Output from before the first recorded run counts
    as a run too."""
    raw = calc.castep.read_bytes() if calc.castep.exists() else b""
    runs = sorted((r for r in calc.info().get("runs", []) if r["start"] < len(raw)),
                  key=lambda r: r["start"])
    if not runs or runs[0]["start"] > 0:
        runs.insert(0, {"start": 0})
    bounds = [r["start"] for r in runs] + [len(raw)]
    return [(run, raw[a:b].decode(errors="replace"))
            for run, a, b in zip(runs, bounds, bounds[1:])]


# The 'Peak Memory Use' per process (MB) of the latest run CASTEP finished.
# Also returns that run's number of processes. Only runs that end cleanly print it; a relaxation
# that stopped at run_time counts.
#   Called by: measured_replicated()
#   Calls:     run_segments()
def last_peak(calc):
    """Peak memory per process (MB) of the latest run that CASTEP finished ('Peak Memory
    Use' at its end; a relaxation stopped at run_time counts) and the number of processes
    it ran on, or (None, None)."""
    for run, text in reversed(run_segments(calc)):
        found = PEAK_MEMORY.findall(text)
        if found:
            value, unit = found[-1]
            peak = float(value) * {"k": 1 / 1024, "m": 1.0, "g": 1024.0}[unit.lower()]
            return peak, run.get("procs") or calc.info().get("ncores")
    return None, None


# Turns a measured peak memory into a value of mem_replicated.
# Solves the memory model of mem_per_core_mb for r, using the dry-run memory saved in job_info.json
# and the run's number of processes (no safety factor; clipped to 0-1). Returns (r, peak,
# processes); r is None if it can't be worked out (no peak line, one process, no dry run).
#   Called by: calibrate()
#   Calls:     last_peak(), parse_dryrun()
def measured_replicated(calc, args):
    """The replicated fraction r that makes the memory model match the latest finished run
    of calc: peak = base + rest + S * (r + (1 - r) / p), solved for r (no safety factor).
    Returns (r or None, peak memory in MB or None, processes p)."""
    info = calc.info()
    peak, p = last_peak(calc)
    mem = info.get("dryrun_mem")
    if not mem and calc.dryrun_file.exists():      # planned by an older version
        mem = parse_dryrun(calc.dryrun_file.read_text(errors="replace"))["mem"]
    if peak is None or not mem or not p or p < 2:
        return None, peak, p
    base = mem["baseline"] if args.baseline_mb is None else args.baseline_mb
    shared = mem["model"] + mem["electronic"]
    rest = max(0.0, mem["total"] - mem["baseline"] - shared)
    if shared <= 0:
        return None, peak, p
    r = (peak - base - rest - shared / p) / (shared * (1 - 1 / p))
    return min(1.0, max(0.0, r)), peak, p


# The 'auto' in mem_replicated : auto: measures the memory model from jobs that have run.
# Works out r for every calculation that has run (measured_replicated) and stores it in its
# job_info.json. Then sets args.mem_replicated to the largest r among the biggest supercell size
# measured so far (rounded up to 3 decimals), logging any change. Every job planned afterwards uses
# it (finalize -> choose_resources -> mem_per_core_mb). Notes once if a finished job printed no
# 'Peak Memory Use'. In batch mode calcs is keyed by (atoms, job number), and the size is the
# number of atoms. A job measured once it had finished is marked so (peak_final) and not measured
# again.
#   Called by: series_main(), restart(), run_batch()
#   Calls:     Calc.done(), Calc.peek(), measured_replicated()
def calibrate(calcs, sizes, args):
    """mem_replicated : auto. Work out r from the latest finished run of every job (a
    relaxation that stopped at run_time and is carrying on counts), then use the largest r
    among the jobs of the biggest size measured so far (the most like the cells to come)."""
    if not args.r_auto:
        return
    found = []
    for n in sizes:
        for c in calcs[n].values():
            info = c.peek()
            if not info.get("jobid"):
                continue
            if info.get("peak_final"):             # measured after it finished: final
                r, peak, p = (info.get(k) for k in ("r_measured", "peak_mb", "peak_procs"))
            else:
                r, peak, p = measured_replicated(c, args)
                final = c.done()
                if final or [info.get(k) for k in ("r_measured", "peak_mb",
                                                    "peak_procs")] != [r, peak, p]:
                    info = c.info()
                    info.update(r_measured=r, peak_mb=peak, peak_procs=p)
                    if final:
                        info["peak_final"] = True
                    c.save(info)
            if peak is None and c.done() and not args.peak_warned:
                args.peak_warned = True
                log(f"note: no 'Peak Memory Use' line in {c.castep}, so mem_replicated "
                    f"can't be measured from it (staying at {args.mem_replicated:g})")
            if r is not None:                   # (batch mode: n = (atoms, job number))
                found.append((n[0] if isinstance(n, tuple) else n, r, c, peak, p))
    if found:
        top = max(f[0] for f in found)
        _, r, c, peak, p = max((f for f in found if f[0] == top), key=lambda f: f[1])
        r = math.ceil(r * 1000) / 1000                      # round up to 3 decimals
        if r != args.mem_replicated:
            log(f"mem_replicated : {r:.3f} (measured from {c.name}: peak {peak / 1024:.2f} GB "
                f"per process on {p} processes)")
            args.mem_replicated = r


# How long a job took, over all its runs.
# Uses run_segments: total seconds, number of jobs, SCF cycles, wall time per SCF cycle (from the
# runs CASTEP finished, scaled to the job's current number of processes), number of SCFs, and the
# cycles of the first SCF. A run killed at the time limit adds its walltime to the total. None if
# CASTEP printed no timings.
#   Called by: measure_times()
#   Calls:     run_segments(), slurm_seconds()
def job_timing(calc):
    """Wall time per SCF cycle and the SCF work of a finished job, over all its runs (a
    continuation appends to the .castep). The time per cycle comes from the runs CASTEP
    finished; a run killed at the time limit counts with its walltime. None if CASTEP
    printed no timings."""
    segs = run_segments(calc)
    ncores = calc.info().get("ncores") or 1
    secs = timed = cycles = total = 0
    for run, seg in segs:
        n, t = len(SCF_CYCLE.findall(seg)), TOTAL_TIME.findall(seg)
        cycles += n
        if t and n:             # (scaled to the job's current number of processes)
            scale = (run.get("procs") or ncores) / ncores
            secs, timed, total = secs + float(t[-1]) * scale, timed + n, total + float(t[-1])
        elif run.get("time") and run.get("end", "killed") == "killed":
            total += slurm_seconds(run["time"]) or 0
    if not timed:
        return None
    text = "".join(seg for _, seg in segs)
    first = re.split(ENERGY_PATTERNS["final"], text, maxsplit=1)[0]
    return {"seconds": round(total), "jobs": len(segs), "cycles": cycles,
            "per_cycle": secs / timed, "scfs": len(re.findall(ENERGY_PATTERNS["final"], text)),
            "first_scf_cycles": len(SCF_CYCLE.findall(first))}


# For walltime : auto: records the timing of every newly finished job.
# Saves job_timing() in job_info.json ('timing') and logs how long the job took. estimate_walltime
# reads these timings to plan later jobs.
#   Called by: series_main(), run_batch()
#   Calls:     Calc.done(), Calc.peek(), job_timing()
def measure_times(calcs, sizes):
    """Record how long each newly finished job took (for walltime : auto)."""
    for n in sizes:
        for c in calcs[n].values():
            info = c.peek()
            if "timing" in info or not info.get("jobid") or not c.done():
                continue
            info = c.info()
            info["timing"] = t = job_timing(c)
            c.save(info)
            if t:
                log(f"{c.name}: took {t['seconds'] / 3600:.2f} h"
                    + (f" in {t['jobs']} jobs" if t["jobs"] > 1 else "")
                    + f" on {info.get('ncores')} processes ({t['cycles']} SCF cycles, "
                    f"{t['per_cycle']:.0f} s each)")


# Predicts how long a calculation will run, from jobs that have finished.
# Takes the time per SCF cycle of the biggest finished job of the same kind and scales it to this
# cell by k-points x electrons^a / processes (a fitted from the smallest and biggest finished
# cells, kept within 2-3.5; 3 until there are two different sizes), then multiplies by the SCF
# cycles that job needed. A single point on a relaxed cell uses that relaxation's own speed. Only
# jobs of the same group are used (batch mode: the same kind and task, so e.g. a phonon+efield job
# is not timed from single points; the size series has one group).
# Returns (seconds, seconds per SCF, the job it scaled from), or Nones if there is nothing to scale
# from. The list of timed jobs is kept in ctx and made again only when a job's timings were saved
# (TIMINGS), not for every job planned: a batch of thousands plans thousands of jobs.
#   Called by: plan_walltime()
def estimate_walltime(calc, nkpts, nelec, procs, ctx):
    """Run time (s) of calc, from the finished jobs: their time per SCF cycle scaled as
    k-points x electrons^a / processes (a fitted from the smallest and biggest finished
    cell, 3 until there are two), times the SCF cycles of the biggest finished job of the
    same kind (for a first single point, the first SCF of the biggest relaxation). A
    single point on a relaxed cell takes its time per cycle from that relaxation.
    Returns (seconds, seconds per SCF, the job it is scaled from) or Nones."""
    # Whether a calculation has the timings, electrons and processes needed to scale from.
    def ok(c):
        i = c.peek()
        return (i.get("timing") or {}).get("per_cycle") and i.get("nelec") and i.get("ncores")

    key = (TIMINGS[0], len(ctx["calcs"]))
    if ctx.get("timed", (None,))[0] != key:
        ctx["timed"] = (key, [c for c in ctx["calcs"] if ok(c)])
    timed = [c for c in ctx["timed"][1] if c.group == calc.group]
    pool = [c for c in timed if c.kind == calc.kind] or (
        [c for c in timed if c.kind == "geom"] if calc.kind != "geom" else [])
    if not pool or not nelec:
        return None, None, None
    own = [c for c in timed if c.kind == "geom" and c.label == calc.label]
    if calc.kind == "defect" and own:    # single point on a relaxed cell: same speed per cycle
        t = own[0].peek()["timing"]
        sps = [c for c in pool if c.kind == "defect"]
        cycles = max(c.peek()["timing"]["cycles"] for c in sps) if sps else t["first_scf_cycles"]
        return t["per_cycle"] * own[0].peek()["ncores"] / procs * cycles, None, own[0]
    info = {c.name: c.peek() for c in pool}

    # A finished job's time per SCF cycle per k-point on one process, so jobs with different
    # k-points and process counts can be compared.
    def norm(c):                          # s per SCF cycle and k-point, on one process
        i = info[c.name]
        return i["timing"]["per_cycle"] * i["ncores"] / max(1, i.get("nkpts") or 1)

    big = max(pool, key=lambda c: info[c.name]["nelec"])
    small = min(pool, key=lambda c: info[c.name]["nelec"])
    ib, a = info[big.name], A_DEFAULT
    if ib["nelec"] > 1.2 * info[small.name]["nelec"]:
        a = math.log(norm(big) / norm(small)) / math.log(ib["nelec"] / info[small.name]["nelec"])
        a = min(max(a, A_RANGE[0]), A_RANGE[1])
    per_cycle = norm(big) * nkpts / procs * (nelec / ib["nelec"]) ** a
    t = ib["timing"]
    cycles = t["cycles"] if big.kind == calc.kind else t["first_scf_cycles"]
    return per_cycle * cycles, per_cycle * t["cycles"] / max(1, t["scfs"]), big


# Chooses a job's walltime.
# The --time already in the jobscript (from the template, a partition line or max_walltime) is the
# longest allowed. With walltime : auto and finished jobs to learn from, it asks for
# estimate_walltime x walltime_safety instead - plus, for a relaxation, the margin by which its
# run_time stops it before the walltime (run_time_for), so that run_time covers the estimate -
# rounded up to 15 minutes (at least WALLTIME_MIN), if that is shorter.
# With no finished job to learn from, a single point asks for sp_walltime instead, if set and
# shorter (restart gives it the full --time if it runs out). Returns the plan: walltime, longest
# walltime, seconds, seconds per SCF and a note for the log. Notes once if the template has no
# --time.
#   Called by: finalize()
#   Calls:     estimate_walltime(), sbatch_value(), slurm_seconds(), slurm_time()
def plan_walltime(calc, script, dr, procs, ctx, args):
    """Walltime for a job. The jobscript's --time (from the template or a partition line)
    is the most it may run; with walltime : auto it asks for less once finished jobs show
    what it needs."""
    tmax = sbatch_value(script, "--time")
    smax = slurm_seconds(tmax) if tmax else None
    plan = SimpleNamespace(max=tmax, time=tmax, secs=smax, per_scf=None, note="")
    if not smax:
        if not ctx.get("time_noted"):
            ctx["time_noted"] = True
            log("note: no #SBATCH --time in the jobscript, so walltimes are not estimated and "
                "relaxations can't stop cleanly before the time limit - add one, or set "
                "max_walltime (the longest a job may run)")
        return plan
    if args.walltime != "auto":
        return plan
    est, plan.per_scf, ref = estimate_walltime(calc, dr["nkpts"], dr["nelec"], procs, ctx)
    if est is None:
        guess = slurm_seconds(args.sp_walltime) if args.sp_walltime else None
        if calc.kind != "geom" and guess and guess < smax:   # a first single point
            plan.time, plan.secs = slurm_time(guess), guess
            plan.note = " (sp_walltime: no finished job to estimate it from yet)"
        return plan
    need = est * args.walltime_safety
    if calc.kind == "geom" and not (args.max_restarts == 0
                                    and getattr(args, "tolerate_failures", False)):
        need += RUN_TIME_MARGIN                     # run_time stops it this much earlier
    secs = max(WALLTIME_MIN, math.ceil(need / 900) * 900)
    if secs < smax:
        plan.time, plan.secs, plan.note = slurm_time(secs), secs, f" (estimated from {ref.name})"
    else:
        plan.note = (f" (estimated {est * args.walltime_safety / 3600:.0f} h from {ref.name}"
                     + (", so it will continue in further jobs)" if calc.kind == "geom" else
                        " - more than the walltime allows: give it more "
                        + ("nodes)" if args.cores_per_node and args.charging != "core"
                           else "cores)")))
    return plan


# CASTEP's run_time for a job with a given walltime (s).
# The walltime minus RUN_TIME_MARGIN, so CASTEP stops cleanly (it won't start a geometry iteration
# that the last one's time says won't fit) and writes its .check before Slurm stops the job; the
# continuation picks up from there. The margin is multiplied by a factor that restart doubles
# after a job is killed anyway, and is never more than half the walltime.
#   Called by: finalize(), restart(), share_walltime()
def run_time_for(walltime, factor=1.0):
    """CASTEP run_time (s) for a job with this walltime (s): early enough to stop cleanly
    and write the .check before the scheduler stops the job."""
    margin = RUN_TIME_MARGIN * factor
    return int(walltime - min(margin, 0.5 * walltime))


# Sets keywords in a calculation's .param, e.g. run_time and continuation.
# Removes any existing line for each keyword and appends the new value (a value of None just
# removes the keyword). Everything else in the file is left as it is, so changes you make by hand
# to a calculation's .param survive restarts. Either spelling of a keyword with two
# (canonical_key: spin_polarised, spin_polarized) replaces both.
#   Called by: finalize(), restart(), share_walltime(), write_inputs()
#   Calls:     canonical_key(), strip_comment()
def set_param(calc, **values):
    """Set keywords in the calc's .param (None removes the keyword)."""
    values = {canonical_key(k): v for k, v in values.items()}
    path = calc.dir / f"{calc.seed}.param"
    lines = []
    for line in path.read_text().splitlines():
        m = re.match(r"\s*(\w+)", strip_comment(line))
        if not (m and canonical_key(m.group(1)) in values):
            lines.append(line)
    lines += [f"{key} : {val}" for key, val in values.items() if val is not None]
    path.write_text("\n".join(lines) + "\n")


# The number of cores, on machines without whole nodes (no cores_per_node).
# k-point parallelism first: with no more k-points than max_cores, all of them run in parallel and
# the rest is filled with G-vector parallelism (a multiple of the k-points); otherwise the fewest
# cores that give the minimum number of k-points per core. 'cores : N' forces N.
#   Called by: choose_resources(), core_resources()
def choose_cores(nkpts, max_cores, fixed=None):
    """k-point parallelisation first. If there are no more k-points than max_cores, run
    them all in parallel and fill up with G-vector parallelism (a multiple of nkpts).
    Otherwise use the fewest cores that give the minimum number of k-points per core."""
    if fixed:
        return fixed
    m = max_cores
    if nkpts <= m:
        return nkpts * (m // nkpts)
    return math.ceil(nkpts / math.ceil(nkpts / m))


# Ranks a whole-node layout (nodes x processes per node); lower is better.
# Fewest k-points per process, then a process count that divides evenly with the k-points, then
# more processes, then fewer nodes.
#   Called by: choose_resources(), core_resources(), node_layouts()
def layout_key(nk, nodes, tpn):
    """Rank of a whole-node layout (lower is better): the fewest k-points per process, then
    a process count that splits evenly with the k-points, then the most processes, then
    the fewest nodes."""
    n = nodes * tpn
    return (math.ceil(nk / n), n % nk != 0 and nk % n != 0, -n, nodes)


# Every layout of up to max_nodes nodes x cores_per_node processes, best first.
# Sorted by layout_key. With 'cores : N', only layouts with exactly N processes; with nodes, only
# those on exactly that many nodes.
#   Called by: choose_resources(), core_resources()
#   Calls:     layout_key()
def node_layouts(nk, cpn, max_nodes, fixed=None, nodes=None):
    """All (nodes, processes per node) with up to cpn processes per node, best first."""
    cands = [(k, t) for k in range(nodes or 1, (nodes or max_nodes) + 1)
             for t in range(1, cpn + 1) if not fixed or k * t == fixed]
    return sorted(cands, key=lambda c: layout_key(nk, *c))


# Decides how and where a job runs on a machine that shares its nodes and charges by the core
# (charging : core, e.g. Cirrus), from its dry run. The processes are chosen as on core-based
# machines (choose_cores: k-points first, up to max_cores or the partition's own limit; or exactly
# cores : N). Memory comes with the cores (the partition's GB per core), so the job goes on the
# first partition where each process fits in one core's memory, and asks for just its cores on
# one node, which it shares with other jobs. If none has enough, the job is planned on every
# partition in the way that asks for the fewest cores (ties: the better layout, layout_key, then
# the partition listed first): on one node, each process with as many cores as its memory needs
# (--cpus-per-task; omp_threads can run threads on them), or on whole nodes. A job with more
# processes than a node has cores takes whole nodes from the start. Whole nodes (--exclusive,
# finalize) are charged in full, so they are laid out as on whole-node machines: the best layout
# (node_layouts) on the fewest nodes that hold the processes, at most max_nodes, with fewer
# processes per node if the memory needs it. If nothing fits, the layout that comes closest
# (finalize then logs a warning). Returns the namespace of choose_resources, with exclusive (whole
# nodes) and charged (the cores the job is charged for); underpop marks a layout that the memory
# made sparser than planned. With pack (more than 1, in batch mode: pack : N), packed marks a job
# on part of a node, which run_batch may put in one job with others (submit_pack); its plan is the
# same either way.
#   Called by: choose_resources()
#   Calls:     choose_cores(), layout_key(), mem_per_core_mb(), node_layouts()
def core_resources(dr, args, pack=1):
    nk, cpn = dr["nkpts"], args.cores_per_node
    skipped, cheaper, over = [], [], []

    # Memory per process (MB) for this job on p processes (mem_per_core_mb).
    def mpc(n):
        return mem_per_core_mb(dr["mem"], n, args)

    # Packs a layout (nodes x processes per node, cores per process) into the namespace finalize
    # uses.
    def result(nodes, tpn, cpus, p, fits, whole, sparse=False):
        return SimpleNamespace(nodes=nodes, tpn=tpn, ncores=nodes * tpn, mpc=mpc(nodes * tpn),
                               part=p, fits=fits, underpop=sparse, cpus=cpus,
                               skipped=[q for q in skipped if q is not p],
                               packed=pack > 1 and not whole, share=None, exclusive=whole,
                               charged=nodes * cpn if whole else tpn * cpus)

    for order, p in enumerate(args.partitions or [None]):
        core_mb = (p["mem_per_core"] if p else args.mem_per_core) * 1024
        pmax = (p["max_cores"] if p else None) or math.inf
        if args.cores and args.cores > pmax:
            continue
        n = choose_cores(nk, min(args.max_cores, pmax), args.cores)
        room = min(cpn, pmax)                     # the most cores one job can have on one node
        if n <= cpn:                              # one node, shared with other jobs
            cpus = math.ceil(mpc(n) / core_mb)
            if cpus <= 1:
                return result(1, n, 1, p, True, False)
            if n * cpus <= room:
                cheaper.append(((n * cpus, layout_key(nk, 1, n), order),
                                (1, n, cpus, p, True, False, True)))
            elif room // n:
                over.append((mpc(n) / (core_mb * (room // n)), (1, n, room // n, p, False, False)))
        first = math.ceil(n / cpn)                # whole nodes
        last = min(args.max_nodes, int(pmax // cpn) if pmax < math.inf else args.max_nodes)
        for k in range(first, last + 1):
            layouts = node_layouts(nk, cpn, k, args.cores, nodes=k)
            if not layouts:
                continue
            fit = next((lay for lay in layouts if lay[1] * mpc(k * lay[1]) <= core_mb * cpn),
                       None)
            if fit is None:
                if k == last:                     # (the closest this partition comes)
                    least = min(layouts, key=lambda lay: lay[1] * mpc(k * lay[1]))
                    over.append((least[1] * mpc(k * least[1]) / (core_mb * cpn),
                                 (k, least[1], cpn // least[1], p, False, True)))
                continue
            if n > cpn and k == first and fit == layouts[0]:      # whole nodes, as planned
                return result(k, fit[1], cpn // fit[1], p, True, True)
            cheaper.append(((k * cpn, layout_key(nk, *fit), order),
                            (k, fit[1], cpn // fit[1], p, True, True, fit != layouts[0])))
            break
        skipped.append(p)
    if cheaper:
        return result(*min(cheaper, key=lambda c: c[0])[1])
    if not over:
        die(f"no partition can run a job with cores : {args.cores}" if args.cores else
            "no partition can run this job: check max_nodes and the partitions' max_cores")
    return result(*min(over, key=lambda o: o[0])[1])


# Decides how and where a job runs, from its dry run.
# Picks the number of processes (on whole-node machines: nodes x processes per node), the memory
# per process (mem_per_core_mb) and the partition. Partitions from %block partitions are tried in
# order and the first where the best layout fits in memory is used. If none fits with full nodes,
# it takes the partition where fewer processes per node give the best layout (the job is
# underpopulated); if nothing fits at all, the one that comes closest (finalize then logs a
# warning). With pack (more than 1: that many jobs share a node, pack : N), the job is first planned
# on its share of a node, cores_per_node // pack cores, on the first partition where the best
# layout there fits in its share of PACK_MEM of the node's memory; only if none does is it planned
# as a job of its own. Returns a namespace: nodes, tpn, ncores, mpc, part, fits, underpop, cpus,
# skipped, and packed and share (its cores) for a packed job. With charging : core, the job is
# planned by core_resources instead.
#   Called by: finalize()
#   Calls:     choose_cores(), core_resources(), layout_key(), mem_per_core_mb(), node_layouts()
def choose_resources(dr, args, pack=1):
    """Processes (and nodes), memory per process (MB) and partition for a calculation,
    from its dry run. Partitions are tried in the order given and the first where the job
    fits is used, otherwise the one that comes closest. With cores_per_node, if no
    partition fits with full nodes, the job runs with fewer processes per node, on the
    partition where that gives the best layout. With pack > 1 the job is planned on its
    share of a node if it fits there."""
    if args.charging == "core":
        return core_resources(dr, args, pack)
    nk, cpn = dr["nkpts"], args.cores_per_node
    skipped, under, over = [], [], []

    # Memory per process (MB) for this job on p processes (mem_per_core_mb).
    def mpc(n):
        return mem_per_core_mb(dr["mem"], n, args)

    # Packs a chosen layout and partition into the namespace finalize uses.
    def result(layout, p, fits, underpop=False):
        nodes, tpn = layout
        return SimpleNamespace(nodes=nodes, tpn=tpn, ncores=nodes * tpn, mpc=mpc(nodes * tpn),
                               part=p, fits=fits, underpop=underpop,
                               cpus=max(1, cpn // tpn) if cpn else 1,
                               skipped=[q for q in skipped if q is not p], packed=False,
                               share=None)

    if pack > 1 and cpn:                                          # a share of one node
        share = cpn // pack
        for p in args.partitions or [None]:
            gb = p["mem_per_core"] if p else args.mem_per_core
            cap = min(share, p["max_cores"]) if p and p["max_cores"] else share
            layouts = node_layouts(nk, cap, 1, args.cores)
            if layouts and (gb is None or layouts[0][1] * mpc(layouts[0][1])
                            <= gb * 1024 * PACK_MEM * share):
                n = layouts[0][1]
                return SimpleNamespace(nodes=1, tpn=n, ncores=n, mpc=mpc(n), part=p, fits=True,
                                       underpop=False, cpus=max(1, share // n), skipped=[],
                                       packed=True, share=share)

    for p in args.partitions or [None]:
        gb = p["mem_per_core"] if p else args.mem_per_core       # None: no memory limit
        pmax = p["max_cores"] if p else None
        if args.cores and pmax and args.cores > pmax:
            continue
        if cpn:                                                   # whole nodes
            caps = [c for c in (args.max_nodes * cpn, pmax) if c]
            layouts = node_layouts(nk, cpn, max(1, min(caps) // cpn), args.cores)
            if not layouts:
                continue
            per_node = gb * 1024 * cpn if gb else math.inf        # MB of memory per node
            fit = next((l for l in layouts if l[1] * mpc(l[0] * l[1]) <= per_node), None)
            if fit == layouts[0]:
                return result(fit, p, True)
            skipped.append(p)
            if fit:
                under.append((layout_key(nk, *fit), fit, p))
            else:
                least = min(layouts, key=lambda l: l[1] * mpc(l[0] * l[1]))
                over.append((least[1] * mpc(least[0] * least[1]) / per_node, least, p))
        else:                                                     # cores
            n = choose_cores(nk, min(pmax or args.max_cores, args.max_cores), args.cores)
            if not gb or mpc(n) <= gb * 1024:
                return result((1, n), p, True)
            skipped.append(p)
            over.append((mpc(n) / (gb * 1024), (1, n), p))
    if under:
        return result(*min(under, key=lambda u: u[0])[1:], True, True)
    if not over:
        die(f"no partition can run a job with cores : {args.cores}")
    return result(*min(over, key=lambda o: o[0])[1:], False)


# How far each atom moved between two structures of the same atoms (Ang, nearest image).
# Used to report the largest move of the MACE pre-relaxation and of the CASTEP relaxation.
#   Called by: prerelaxed(), relaxed_atoms()
def displacements(ref, atoms):
    """How far each atom of atoms is from the same atom of ref (A, nearest image)."""
    df = atoms.get_scaled_positions(wrap=False) - ref.get_scaled_positions(wrap=False)
    return np.linalg.norm((df - np.round(df)) @ np.asarray(ref.cell), axis=1)


# CASTEP's name for atom i, e.g. O107: the element and its number within that element.
# So the log names atoms the way the .castep does.
#   Called by: prerelaxed(), relaxed_atoms()
def atom_label(atoms, i):
    """CASTEP's name for atom i: element and its number within that element."""
    sym = atoms.get_chemical_symbols()
    return f"{sym[i]}{sym[:i + 1].count(sym[i])}"


# The MACE settings that affect the pre-relaxed structure, or None with prerelax : none.
# Model, fmax and rattle. Stored in job_info.json and in the work directory's fingerprint, so
# changing them rewrites the inputs of calculations not yet submitted (prepare_all) and makes the
# script refuse an existing work directory (main). Also part of the key prerelaxed uses to decide
# whether its saved result can be reused.
#   Called by: series_main(), prepare_all(), prerelaxed(), write_inputs()
def prerelax_settings(args):
    if args.prerelax != "mace":
        return None
    return {"model": args.prerelax_model, "fmax": args.prerelax_fmax,
            "rattle": args.prerelax_rattle}


# Imports torch and mace_mp, stopping with installation instructions if mace-torch is missing.
# series_main() calls it at the start with prerelax : mace, so a missing package is found straight
# away rather than halfway through.
#   Called by: mace_calculator(), series_main()
def import_mace():
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)   # torch.load notices
    try:
        import torch
        from mace.calculators import mace_mp
    except ImportError as err:
        die(f"prerelax : mace needs the mace-torch package (Python 3.9+) in the Python that "
            f"runs this script ({sys.executable}):\n    pip install mace-torch\n({err})")
    return torch, mace_mp


# Loads the MACE model as an ASE calculator, once.
# Uses prerelax_model on prerelax_device in float64, sets torch's number of threads to
# prerelax_threads, and keeps the calculator in args.mace for every later defect cell. Stops if the
# model can't be loaded (unknown name, no network for the download).
#   Called by: prerelaxed()
#   Calls:     import_mace()
def mace_calculator(args):
    """The MACE calculator for prerelax : mace, loaded the first time it is needed."""
    if getattr(args, "mace", None) is None:
        torch, mace_mp = import_mace()
        torch.set_num_threads(args.prerelax_threads)
        try:
            args.mace = mace_mp(model=args.prerelax_model, device=args.prerelax_device,
                                default_dtype="float64")
        except Exception as err:          # unknown model, no network for the download, ...
            die(f"could not load the MACE model '{args.prerelax_model}': {err}")
    return args.mace


# A fingerprint of a structure (species, fractional positions, cell): prerelaxed keeps it with its
# result, so a saved MACE relaxation is only reused for the structure it started from.
#   Called by: prerelaxed()
def structure_key(atoms):
    frac = atoms.get_scaled_positions(wrap=True)
    frac[np.isclose(frac, 1.0, atol=1e-7)] = 0.0
    data = [atoms.get_chemical_symbols(), (np.round(frac, 5) + 0.0).tolist(),
            (np.round(np.asarray(atoms.cell), 5) + 0.0).tolist()]
    return hashlib.md5(json.dumps(data).encode()).hexdigest()


# The MACE pre-relaxation of one defect cell; returns the relaxed Atoms.
# Copies calc.atoms, optionally rattles it (same random displacements every time), attaches the
# MACE calculator and runs ASE's LBFGS on the atomic positions only (the cell stays fixed) until
# the largest force is below prerelax_fmax or prerelax_steps is reached. Saves the result in
# prerelax.json and the optimiser's log in prerelax.log, logs the steps, energy change and largest
# move, and warns if it didn't converge or an atom moved more than 1 Ang. If prerelax.json already
# holds a result for the same settings and the same starting structure (structure_key) it is
# reused instead. write_inputs writes the returned positions into the CASTEP .cell, so CASTEP's
# relaxation starts from them; relaxed_atoms later reads prerelax.json to report how far CASTEP
# moved the atoms from the MACE structure.
#   Called by: write_inputs()
#   Calls:     atom_label(), displacements(), mace_calculator(), prerelax_settings(),
#              structure_key()
def prerelaxed(calc, args):
    """calc.atoms relaxed with MACE (positions only, the cell fixed), for CASTEP's
    relaxation to start from. Done once: the result is kept in <dir>/prerelax.json."""
    from ase.optimize import LBFGS
    path, atoms = calc.dir / PRERELAX_FILE, calc.atoms.copy()
    key, start = dict(prerelax_settings(args), steps=args.prerelax_steps), structure_key(atoms)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("settings") == key and saved.get("start") == start and \
                len(saved.get("scaled_positions", [])) == len(atoms):
            atoms.set_scaled_positions(saved["scaled_positions"])
            log(f"{calc.name}: using its earlier MACE pre-relaxation ({saved.get('steps', '?')} "
                f"steps, largest move {saved.get('max_displacement', 0.0):.3f} A, kept in "
                f"{PRERELAX_FILE})")
            return atoms
    log(f"{calc.name}: MACE pre-relaxation of {len(atoms)} atoms ({args.prerelax_model} on "
        f"{args.prerelax_device}, {args.prerelax_threads} threads)")
    begin = time.time()
    if args.prerelax_rattle:
        atoms.rattle(stdev=args.prerelax_rattle, seed=1)    # same displacements every time
    atoms.calc = mace_calculator(args)
    e0 = atoms.get_potential_energy()
    opt = LBFGS(atoms, logfile=str(calc.dir / PRERELAX_LOG))
    opt.run(fmax=args.prerelax_fmax, steps=args.prerelax_steps)
    fmax = float(np.sqrt((atoms.get_forces() ** 2).sum(axis=1)).max())
    de = float(atoms.get_potential_energy() - e0)
    atoms.calc = None
    moved = displacements(calc.atoms, atoms)
    i = int(moved.argmax())
    steps = getattr(opt, "nsteps", 0)
    path.write_text(json.dumps({
        "settings": key, "start": start, "steps": steps, "fmax": fmax, "energy_change": de,
        "max_displacement": float(moved.max()), "moved_most": atom_label(calc.atoms, i),
        "scaled_positions": atoms.get_scaled_positions(wrap=False).tolist()}, indent=1))
    log(f"{calc.name}: MACE pre-relaxed in {steps} steps ({(time.time() - begin) / 60:.1f} min): "
        f"|F|max {fmax:.3f} eV/A, energy {de:+.3f} eV, largest move {moved.max():.3f} A "
        f"({atom_label(calc.atoms, i)})")
    if fmax > args.prerelax_fmax:
        log(f"WARNING: the MACE pre-relaxation of {calc.name} did not reach prerelax_fmax in "
            f"{args.prerelax_steps} steps; CASTEP starts from where it got to")
    if moved.max() > 1.0:
        log(f"WARNING: MACE moved {atom_label(calc.atoms, i)} of {calc.name} by "
            f"{moved.max():.2f} A - check the starting structure in its .cell is sensible")
    return atoms


# Creates a calculation's directory and writes its CASTEP .cell and .param.
# The .cell comes from write_cell, with the k-grid from kgrid (or the job's own fixed grid in batch
# mode), the options from cell_options and, for a defect relaxation with prerelax : mace, the
# MACE-relaxed positions (prerelaxed). The .param comes from make_param (geom_param for
# relaxations, if given; in batch mode the job's own .param and task, if it has them), then any
# keywords the job sets (overrides, written with set_param). Records the number of atoms, the
# k-grid and the MACE settings in job_info.json, with shared dry runs the dry-run fingerprint
# (dryrun_key), and for a size-series relaxation the atom its defect's first component is on
# (anchor), which series_main() keeps if the script is run again.
#   Called by: prepare_all(), prepare_sp()
#   Calls:     cell_options(), dryrun_key(), kgrid(), make_param(), param_value(),
#              prerelax_settings(), prerelaxed(), set_param(), write_cell()
def write_inputs(calc, ctx, args):
    calc.dir.mkdir(parents=True, exist_ok=True)
    grid = list(calc.fixed_kgrid) if calc.fixed_kgrid else kgrid(calc.atoms.cell, ctx["spacing"])
    keep, opts = cell_options(calc, ctx, grid)
    pre = calc.kind == "geom" and args.prerelax == "mace"
    atoms = prerelaxed(calc, args) if pre else calc.atoms
    write_cell(calc.dir / f"{calc.seed}.cell", atoms, grid, keep,
               f"{calc.name}: {len(calc.atoms)} atoms, {calc.atoms.get_chemical_formula()}"
               + (f", pre-relaxed with MACE ({args.prerelax_model})" if pre else ""), **opts)
    params = calc.param_items or (ctx["geom_param_items"] if calc.kind == "geom"
                                  else ctx["param_items"])
    (calc.dir / f"{calc.seed}.param").write_text(make_param(params, calc.kind, calc.task))
    if calc.overrides:                     # batch mode: this job's own .param keywords
        set_param(calc, **{k.lower(): param_value(v) for k, v in calc.overrides.items()})
    info = calc.info()
    info.update(natoms=len(calc.atoms), kgrid=grid,
                prerelax=prerelax_settings(args) if pre else None)
    if getattr(calc, "anchor", None) is not None:   # where the size series put the defect
        info["anchor"] = calc.anchor
    if ctx.get("share_dryruns"):
        info["dryrun_key"] = dryrun_key(calc)
    calc.save(info)


# Gets a list of calculations ready to submit: inputs, dry runs and job plans.
# For each calculation not yet submitted: writes its inputs if needed (write_inputs; again if the
# MACE settings changed), gets its dry run - already finished, run here (local_dryrun) or submitted
# as a batch job (dryrun_script + submit) - and plans the job (finalize). Dry-run jobs are
# submitted together and waited for together (wait_for with key 'dryrun_jobid'); in batch mode
# with dryrun_pack : N, up to N of them go in one job (even_groups, submit_dryrun_pack). With
# shared dry runs (batch mode), a calculation whose dry run would be the same as another's
# (dryrun_key) gets a copy of that one instead of its own. show=False plans without logging each
# plan. A calculation held for a restart (restart_ready: restart planned it already) is left as it
# is. Returns False if --no-wait and dry runs are still queued.
#   Called by: series_main(), run_batch()
#   Calls:     Calc.done(), dryrun_output(), dryrun_script(), even_groups(), finalize(),
#              local_dryrun(), prerelax_settings(), submit(), submit_dryrun_pack(), wait_for(),
#              write_inputs()
def prepare_all(calcs, ctx, args, show=True):
    """Give every calc its inputs, a dry run and a jobscript. Dry runs are submitted to
    the queue together (or run here with dryrun_on : login). Calcs not yet submitted are
    planned again each time, so they use the current settings. Returns False if
    --no-wait and some dry runs are still queued."""
    share = ctx.get("share_dryruns")
    ready = {}                              # dry-run fingerprint -> a finished dry run
    for c in ctx["calcs"] if share else []:
        key = c.info().get("dryrun_key")
        if key and key not in ready and c.dryrun_file.exists():
            ready[key] = c.dryrun_file

    # A copy of the finished dry run with this calculation's fingerprint, if there is one.
    def shared(c):
        src = ready.get(c.info().get("dryrun_key")) if share else None
        if src is None or src == c.dryrun_file:
            return None
        text = src.read_text(errors="replace")
        c.dryrun_file.write_text(text)
        return text

    queued, leaders, send = [], {}, []
    for c in calcs:
        if c.done() or c.info().get("jobid") or c.info().get("restart_ready"):
            continue
        info = c.info()                     # (re)write the inputs if prerelax was changed
        if "kgrid" not in info or info.get("prerelax") != (
                prerelax_settings(args) if c.kind == "geom" else None):
            write_inputs(c, ctx, args)
        text = dryrun_output(c) or shared(c)
        if text is None and args.dryrun_on == "login":
            text = local_dryrun(c, args)
        if text is not None:
            if share:
                ready.setdefault(c.info().get("dryrun_key"), c.dryrun_file)
            finalize(c, text, ctx, args, show=show)
            continue
        key = c.info().get("dryrun_key") if share else None
        if not c.info().get("dryrun_jobid"):
            if key in leaders:              # the same dry run is already on its way
                queued.append(c)
                continue
            (c.dir / DRYRUN_SCRIPT).write_text(dryrun_script(c, ctx, args))
            send.append(c)
        if key:
            leaders.setdefault(key, c)
        queued.append(c)
    npack = args.dryrun_pack if ctx.get("batch") else 1
    for group in even_groups(send, npack):         # dryrun_pack : N - N to a job
        if len(group) > 1:
            submit_dryrun_pack(group, ctx, args)
        else:
            submit(group[0], args, DRYRUN_SCRIPT, "dryrun_jobid")
    jobs = [c for c in queued if c.info().get("dryrun_jobid")]
    if jobs and not wait_for(jobs, args, "dryrun_jobid"):
        return False
    for c in queued:
        text = dryrun_output(c)
        if text is None:                    # shares the dry run of another calculation
            ready[c.info()["dryrun_key"]] = leaders[c.info()["dryrun_key"]].dryrun_file
            text = shared(c)
        finalize(c, text, ctx, args, show=show)
    return True


# Turns a dry run into a job plan and writes job.sh.
# parse_dryrun gives the k-points, electrons and memory; choose_resources picks the layout, memory
# per process and partition; the template is filled in (fill), OpenMP threads are set for cores the
# layout leaves idle (thread_count, set_threads), and the partition, its #SBATCH options,
# max_walltime (the longest a job may run, replacing the template's and the partition's --time) and
# the walltime (plan_walltime) are set with add_sbatch_option. For a relaxation, run_time is set in
# the .param (run_time_for, set_param), so it can stop cleanly and be carried on - except in a batch
# with max_restarts : 0 and tolerate_failures (screening), where nothing would carry it on: there it
# may use its whole walltime. The plan is saved in job_info.json and logged (show), with warnings
# about memory and odd electron counts unless quiet. Returns a one-line summary of the layout. In
# batch mode with pack : N the job is planned on its share of a node (choose_resources), 1/N of it
# or, given pack, 1/pack (pack=1: a job of its own, as for a restart); run_batch then runs it with
# others in one job (submit_pack), and its own job.sh is only used if it is ever submitted on its
# own. With charging : core, a job on whole nodes gets #SBATCH --exclusive and one on part of a node
# loses any the template has (set_exclusive); with pack : N too, its plan doesn't change, but one on
# part of a node is marked as one that can go in a pack.
#   Called by: prepare_all(), prepare_sp(), restart(), run_batch()
#   Calls:     add_sbatch_option(), choose_resources(), fill(), parse_dryrun(), plan_walltime(),
#              run_time_for(), set_exclusive(), set_param(), set_threads(), thread_count()
def finalize(calc, text, ctx, args, quiet=False, show=True, pack=None):
    """Choose cores, memory and partition from the dry run and write the real jobscript.
    quiet: no warnings; show: log the plan; pack: how many jobs share its node (default: the
    pack setting, in batch mode)."""
    dr = parse_dryrun(text)
    grid = calc.info()["kgrid"]
    npack = (args.pack if pack is None else pack) if ctx.get("batch") else 1
    r = choose_resources(dr, args, npack)
    ncores, mpc, part = r.ncores, r.mpc, r.part
    core = args.charging == "core"
    if not quiet:
        if dr["mem"]["baseline"] > 2000 and args.baseline_mb is None and not ctx.get("noted"):
            ctx["noted"] = True
            log(f"NOTE: CASTEP's 'baseline' memory estimate is {dr['mem']['baseline']:.0f} MB "
                "per process. That is often mostly virtual memory reserved by MPI/BLAS rather "
                "than memory actually used; if the requests look far too big, set "
                "baseline_mb (e.g. 500) and check real usage with: "
                "sacct -j <jobid> --format=MaxRSS")
        if not r.fits and part:
            log(f"WARNING: {calc.name} needs ~{mpc / 1024:.1f} GB per process on {ncores} "
                f"processes, more than any partition in %block partitions has. Using "
                f"{part['name']} ({part['mem_per_core']:g} GB/core), where it may run out of "
                "memory: check the estimate (baseline_mb, mem_replicated) or add a partition "
                "with more" + (", or raise max_nodes." if core else "."))
        elif not r.fits:
            log(f"WARNING: {calc.name} needs ~{mpc / 1024:.1f} GB per core on {ncores} cores, "
                f"more than mem_per_core ({args.mem_per_core} GB). It may run out of memory: "
                + ("raise max_nodes" if core else "raise max_cores")
                + " or use high-memory nodes (see %block partitions).")
        if dr["nelec"] and round(dr["nelec"]) % 2 == 1 and not dr["spin_polarized"]:
            log(f"WARNING: {calc.name} has an odd number of electrons ({dr['nelec']:.0f}) but "
                "is not spin-polarised - consider spin_polarized : true in the .param")
    subs = {"{NCORES}": ncores, "{NODES}": r.nodes, "{TASKS_PER_NODE}": r.tpn,
            "{CPUS_PER_TASK}": r.cpus, "{SEED}": calc.seed, "{JOBNAME}": calc.name,
            "{NKPTS}": dr["nkpts"], "{MEM_MB}": math.ceil(ncores * mpc),
            "{MEM_GB}": math.ceil(ncores * mpc / 1024),
            "{MEM_PER_NODE_GB}": math.ceil(r.tpn * mpc / 1024),
            "{MEM_PER_CORE_MB}": math.ceil(mpc), "{MEM_PER_CORE_GB}": math.ceil(mpc / 1024),
            "{PARTITION}": part["name"] if part else ""}
    script = fill(ctx["job_template"], subs)
    threads = thread_count(r, args)             # omp_threads: use cores left idle
    script = set_threads(script, threads, r.cpus)
    if core:                                    # whole nodes, or a share of one
        script = set_exclusive(script, r.exclusive)
    if part:                                    # partition and its extra #SBATCH options
        script = add_sbatch_option(script, f"--partition={part['name']}")
        for option in part["options"]:
            script = add_sbatch_option(script, fill(option, subs))
    if args.max_walltime:                       # the longest these jobs may run
        script = add_sbatch_option(script, f"--time={args.max_walltime}")
    wall = plan_walltime(calc, script, dr, ncores, ctx, args)
    if wall.time != wall.max:
        script = add_sbatch_option(script, f"--time={wall.time}")
    (calc.dir / JOBSCRIPT).write_text(script)
    (calc.dir / JOBSCRIPT).chmod(0o755)
    if calc.kind == "geom" and wall.secs:       # stop cleanly before the walltime runs out
        if args.max_restarts == 0 and getattr(args, "tolerate_failures", False):
            set_param(calc, run_time=None)      # (screening: no restart would carry it on)
        else:
            set_param(calc, run_time=run_time_for(wall.secs))
    info = calc.info()
    info.update(nkpts=dr["nkpts"], ncores=ncores, mem_per_core_gb=round(mpc / 1024, 2),
                nelec=dr["nelec"], partition=part["name"] if part else None,
                nodes=r.nodes if args.cores_per_node else None, tasks_per_node=r.tpn,
                omp_threads=threads, dryrun_mem=dr["mem"],
                mem_replicated=round(args.mem_replicated, 4),
                walltime=wall.time, walltime_max=wall.max, scf_seconds=wall.per_scf,
                packed=r.packed, share=r.share, cpus_per_task=r.cpus,
                charged_cores=r.charged if core else None, exclusive=r.exclusive if core else None)
    calc.save(info)
    if r.packed and not core:
        size = (f"{r.tpn} processes" + (f" x {threads} threads" if threads > 1 else "")
                + f" ({npack} jobs to a node, {r.share} cores each)")
        mem = f"~{mpc / 1024:.1f} GB/process"
    elif core and not r.exclusive:              # part of a shared node
        size = (f"{r.tpn} processes" + (f" x {r.cpus} cores" if r.cpus > 1 else "")
                + f" ({r.charged} cores of a shared node)")
        mem = f"~{mpc / 1024:.1f} GB/process (~{ncores * mpc / 1024:.0f} GB in all)"
    elif args.cores_per_node:
        size = (f"{r.nodes} {'whole ' if core else ''}node{'s' if r.nodes > 1 else ''} x "
                f"{r.tpn} processes" + (f" x {threads} threads" if threads > 1 else ""))
        mem = f"~{mpc / 1024:.1f} GB/process (~{r.tpn * mpc / 1024:.0f} GB/node)"
    else:
        size = f"{ncores} cores"
        mem = f"~{mpc / 1024:.1f} GB/core (~{ncores * mpc / 1024:.0f} GB total)"
    if r.underpop and core and not r.exclusive:
        notes = [f"{r.cpus} cores per process, for the memory"
                 + (f" ({threads} OpenMP threads each)" if threads > 1 else "")]
    else:
        notes = ([f"{r.tpn} of {args.cores_per_node} cores per node used, to fit the memory"
                  + (" (the rest run OpenMP threads)" if threads > 1 else "")]
                 if r.underpop else [])
    if r.fits and r.skipped:
        notes.append("too much for " + ", ".join(f"{p['name']} ({p['mem_per_core']:g} GB/core)"
                                                 for p in r.skipped))
    if show:
        log(f"  {calc.name}: k-grid {'x'.join(map(str, grid))} -> {dr['nkpts']} k-points, {size}"
            + (f" on {part['name']}" if part else "") + f", {mem}"
            + (f", walltime {wall.time}{wall.note}" if wall.time else "")
            + "".join(f" - {note}" for note in notes))
    return f"{size}" + (f" on {part['name']}" if part else "") + f", {mem}"


# Reads the relaxed structure of a finished relaxation and reports on it.
# final_atoms reads it (from <seed>-out.cell, or the last 'Fractional coordinates' table in the
# .castep) in the order of geom.atoms; then the cell is checked not to have changed, and the
# largest displacement, how far CASTEP moved the atoms from the MACE start (prerelax.json) and the
# number of geometry steps are logged. Returns the relaxed Atoms.
#   Called by: series_main(), prepare_sp()
#   Calls:     Calc.output(), atom_label(), displacements(), final_atoms()
def relaxed_atoms(geom):
    """Relaxed defect cell: <seed>-out.cell if CASTEP wrote it, else the last fractional
    coordinates in the .castep. Checks the cell didn't change and reports the relaxation."""
    atoms, start = final_atoms(geom), geom.atoms
    if atoms is None:
        die(f"could not read the relaxed structure of {geom.name} from {geom.seed}-out.cell or "
            f"{geom.castep.name}: its atoms don't match those of {geom.seed}.cell")
    if np.abs(np.asarray(atoms.cell) - np.asarray(start.cell)).max() > 1e-4:
        die(f"the cell of {geom.name} changed during the relaxation; it should stay fixed")
    atoms.set_cell(start.cell, scale_atoms=False)
    atoms.set_initial_magnetic_moments(start.get_initial_magnetic_moments())
    moved = displacements(start, atoms)
    note, pre = "", geom.dir / PRERELAX_FILE
    if pre.exists():                       # how far CASTEP went from the MACE structure
        mace = geom.atoms.copy()
        mace.set_scaled_positions(json.loads(pre.read_text())["scaled_positions"])
        off = displacements(mace, atoms)
        note = f", {off.max():.3f} A from the MACE start ({atom_label(atoms, int(off.argmax()))})"
    steps = len(GEOM_STEP.findall(geom.output()))
    log(f"{geom.name}: relaxed, largest atomic displacement {moved.max():.3f} A "
        f"({atom_label(atoms, int(moved.argmax()))}){note}"
        + (f", {steps} geometry steps" if steps else ""))
    return atoms


# Sets up the single point on a relaxed defect cell (singlepoint : true, or with geom_param).
# Takes the relaxed structure (relaxed_atoms), writes the inputs (write_inputs) and plans the job
# from the relaxation's dry run (finalize), so it runs with the same k-points and layout. Does
# nothing if the single point is already submitted or done; series_main() then submits it.
#   Called by: series_main()
#   Calls:     Calc.done(), finalize(), relaxed_atoms(), write_inputs()
def prepare_sp(sp, geom, ctx, args):
    """Single point on the relaxed defect cell, with the relaxation's k-points and cores."""
    if sp.done() or sp.info().get("jobid"):
        return
    sp.atoms = relaxed_atoms(geom)
    write_inputs(sp, ctx, args)
    finalize(sp, geom.dryrun_file.read_text(errors="replace"), ctx, args, quiet=True)


# Submits a jobscript with submit_cmd, from its own directory, and returns the job id. Stops if the
# submission fails. The workflow submits its MACE jobs with it too.
#   Called by: submit(), submit_array(), submit_dryrun_pack(), submit_pack(),
#              submit_pack_array(), the workflow
def send_job(directory, script, args):
    res = subprocess.run(shlex.split(args.submit_cmd) + [script], cwd=directory,
                         capture_output=True, text=True)
    m = (re.search(r"Submitted batch job (\d+)", res.stdout) or
         re.search(r"Your job (\d+)", res.stdout) or re.search(r"(\d+)", res.stdout))
    if res.returncode != 0 or not m:
        die(f"could not submit {directory / script}:\n{res.stdout}{res.stderr}")
    return m.group(1)


# Records a job submitted for a calculation in its job_info.json: the job id and submission time.
# For a real job also a new run in info['runs'] - the job id, where its output will start in the
# .castep (start: the file's size before the job), its walltime and number of processes; this is
# what run_segments, job_timing, last_peak and Calc.run_text work from - and the pack or job array
# it runs in (pack : N, array : N), if any.
#   Called by: submit(), submit_array(), submit_dryrun_pack(), submit_pack(), submit_pack_array()
def record_job(calc, jobid, start, wall, key="jobid", pack=None):
    info = calc.info()
    info[key] = jobid
    info["submitted" if key == "jobid" else "dryrun_submitted"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if key == "jobid":
        info["runs"] = ([r for r in info.get("runs", []) if r["start"] < start]
                        + [{"jobid": jobid, "start": start, "time": wall,
                            "procs": info.get("ncores")}])
        if pack:
            info["pack"] = pack
        else:
            info.pop("pack", None)
    calc.save(info)


# What a planned job runs on, for the log: its nodes and processes (or its cores of a shared node),
# and its partition.
#   Called by: submit(), submit_array()
def job_size(info):
    nodes, tpn, cpus = info.get("nodes"), info.get("tasks_per_node"), info.get("cpus_per_task")
    part = f" ({info['partition']})" if info.get("partition") else ""
    if info.get("charged_cores") and not info.get("exclusive"):     # part of a shared node
        return (f"{tpn} processes" + (f" x {cpus} cores" if cpus and cpus > 1 else "")
                + f" ({info['charged_cores']} cores of a shared node"
                + (f", {info['partition']})" if info.get("partition") else ")"))
    if nodes:
        return (f"{nodes} {'whole ' if info.get('exclusive') else ''}"
                f"node{'s' if nodes > 1 else ''} x {tpn} processes{part}")
    return f"{info.get('ncores')} cores{part}"


# Submits job.sh (or dryrun.sh) with submit_cmd and records the job in job_info.json (record_job).
# Stops if the submission fails.
#   Called by: handle_failure(), series_main(), prepare_all(), restart(), run_batch()
#   Calls:     job_size(), record_job(), sbatch_value(), send_job()
def submit(calc, args, script=JOBSCRIPT, key="jobid"):
    start = calc.castep.stat().st_size if calc.castep.exists() else 0   # where its output begins
    jobid = send_job(calc.dir, script, args)
    wall = sbatch_value((calc.dir / script).read_text(), "--time")
    record_job(calc, jobid, start, wall, key)
    info = calc.info()
    if key == "jobid":
        log(f"submitted {calc.name}: job {info[key]} on {job_size(info)}"
            + (f", walltime {wall}" if wall else ""))
    else:
        log(f"submitted dry run for {calc.name} ({len(calc.atoms)} atoms): job {info[key]}")


# The line of a jobscript template that runs CASTEP, for pack : N (rx CASTEP_RUN) or dryrun_pack : N
# (the dry-run template, rx DRYRUN_RUN). There must be exactly one, and it must start CASTEP with
# srun: a pack starts each of its jobs with its own srun. Returns the line; stops with what to
# change otherwise.
#   Called by: batch_args(), run_batch(), submit_dryrun_pack(), submit_pack(), submit_pack_array()
def pack_castep_line(template, name="the runner settings", rx=CASTEP_RUN, what="pack"):
    runs = [line for line in template.splitlines()
            if rx.search(line) and not line.lstrip().startswith("#")]
    if len(runs) != 1 or not re.search(r"(?<![\w/.\-])srun\b",
                                       runs[0][:rx.search(runs[0]).start()]):
        die(f"{name}: {what} runs several CASTEP jobs side by side, each with its own srun, so "
            + ("the jobscript template" if rx is CASTEP_RUN else
               "the dry-run template (dryrun_jobscript, or else the jobscript template)")
            + " needs exactly one line that runs CASTEP with srun, e.g.\n"
            "    srun --distribution=block:block --hint=nomultithread castep.mpi "
            + ("{SEED}" if rx is CASTEP_RUN else "--dryrun {SEED}"))
    return runs[0]


# One line of a pack's job script: the template's line that runs CASTEP (run, matched by rx), made
# to run one calculation, seed, in its own directory, where (both shell words: a name, quoted, or a
# variable such as "$SEED"), in the background (or not: background=False, for dry runs that take
# their turn on a core) - with env set and srun given the calculation's own options (own) in place
# of the template's ones of that kind (SRUN_OWN) - with its output in out there: slurm-<job id>.out,
# or for a task of a job array slurm-<array job id>_<task>.out.
#   Called by: submit_dryrun_pack(), submit_pack(), submit_pack_array()
def pack_line(run, rx, seed, where, own, env, out="slurm-${SLURM_JOB_ID}.out", background=True):
    m = rx.search(run)
    head, exe, tail = run[:m.start()], m.group(1), run[m.end():]
    s = re.search(r"(?<![\w/.\-])srun\b", head)
    rest = SRUN_OWN.sub("", " " + head[s.end():]).split()
    cmd = (f"{head[:s.start()]}{env}srun {own}" + (" " + " ".join(rest) if rest else "")
           + f" {exe}{seed}{tail}").replace("{SEED}", seed).strip()
    return f"( cd {where} && {cmd} ) > {where}/{out} 2>&1" + (" &" if background else "")


# A new directory for a pack's job script: <workdir>/packs/<prefix>_NNN, numbered on from those
# already there.
#   Called by: submit_array(), submit_dryrun_pack(), submit_pack(), submit_pack_array()
def new_pack_dir(workdir, prefix):
    packs = workdir / PACK_DIR
    packs.mkdir(exist_ok=True)
    n = len(prefix) + 1
    k = 1 + max([int(d.name[n:]) for d in packs.glob(f"{prefix}_*") if d.name[n:].isdigit()],
                default=0)
    pdir = packs / f"{prefix}_{k:03d}"
    pdir.mkdir()
    return pdir


# Splits a list into as few groups of at most size as possible, as evenly as possible, in order.
#   Called by: prepare_all(), run_batch()
def even_groups(items, size):
    n = math.ceil(len(items) / size) if items else 0
    bounds = [round(i * len(items) / n) for i in range(n + 1)] if n else [0]
    return [items[a:b] for a, b in zip(bounds, bounds[1:])]


# How each job of a pack (pack : N) runs on its node, from its job_info (infos, one per job): the
# options its srun gets in place of the template's ones of that kind (SRUN_OWN) - its processes
# and cores per process, --exact, and its memory - the environment it starts with
# (OMP_NUM_THREADS, SRUN_CPUS_PER_TASK and, with threads, OMP_PLACES=cores) and the MB of memory
# that gives it. The memory is its share of PACK_MEM of the node's memory (--mem), or with
# charging : core what its cores bring, asked for with --mem-per-cpu: SLURM_MEM_PER_CPU, which
# Slurm sets in the job, so that the jobs' memory adds up to the pack's (or, if it isn't set, the
# partition's GB per core). Cirrus takes no --mem for a job, and a job step that asks for no
# memory may be given all of the job's, which would hold the others back. Returns ([(own options,
# env, MB), ...], the cores the pack asks for - a whole node's, or with charging : core the sum of
# its jobs' - and its partition: the %block partitions row, or None).
#   Called by: run_batch(), submit_pack(), submit_pack_array()
def pack_parts(infos, args):
    core = args.charging == "core"
    part = next((p for p in args.partitions or [] if p["name"] == infos[0].get("partition")), None)
    gb = part["mem_per_core"] if part else args.mem_per_core
    jobs = []
    for i in infos:
        n, threads = i["ncores"], i.get("omp_threads") or 1
        if core:                                # its cores, and the memory they bring
            mb = math.ceil(i["charged_cores"] * gb * 1024)
            cpus = i.get("cpus_per_task") or 1
            memory = f"--mem-per-cpu=${{SLURM_MEM_PER_CPU:-{math.floor(gb * 1024)}}}M"
        else:
            mb = (math.ceil(i["share"] * gb * 1024 * PACK_MEM) if gb else
                  math.ceil(i["ncores"] * i["mem_per_core_gb"] * 1024 * 1.25))
            cpus = max(1, i["share"] // n)
            memory = f"--mem={mb}M"
        env = (f"OMP_NUM_THREADS={threads} SRUN_CPUS_PER_TASK={cpus} "
               + ("OMP_PLACES=cores " if threads > 1 else ""))
        jobs.append((f"--nodes=1 --ntasks={n} --ntasks-per-node={n} --cpus-per-task={cpus} "
                     f"--exact {memory}", env, mb))
    ncores = sum(i["charged_cores"] for i in infos) if core else args.cores_per_node
    return jobs, ncores, part


# A pack's job script (pack : N) from the jobscript template: its #SBATCH lines for one whole node
# (with --exclusive, and without --ntasks-per-node, --ntasks or --cpus-per-task, which each job
# sets for itself) - with charging : core, for its jobs' cores (ncores) of a node it shares with
# other jobs (no --exclusive): the template's --ntasks-per-node (or --ntasks) is their sum, with
# one CPU per task - the partition (part) and its options, and the lines of body in place of the
# line that runs CASTEP (run). The placeholders are filled in for the pack: ncores, its memory
# (total, MB), the most k-points of its jobs (nkpts), seed and job name (name).
#   Called by: submit_pack(), submit_pack_array()
#   Calls:     add_sbatch_option(), fill(), set_exclusive()
def pack_script(template, run, body, part, ncores, total, nkpts, seed, name, args):
    core = args.charging == "core"
    subs = {"{NCORES}": ncores, "{NODES}": 1, "{TASKS_PER_NODE}": ncores, "{CPUS_PER_TASK}": 1,
            "{SEED}": seed, "{JOBNAME}": name, "{NKPTS}": nkpts, "{MEM_MB}": total,
            "{MEM_GB}": math.ceil(total / 1024), "{MEM_PER_NODE_GB}": math.ceil(total / 1024),
            "{MEM_PER_CORE_MB}": math.ceil(total / ncores),
            "{MEM_PER_CORE_GB}": math.ceil(total / ncores / 1024),
            "{PARTITION}": part["name"] if part else ""}
    lines = []
    for line in template.splitlines():
        if not core and re.match(r"#SBATCH\s+((--ntasks-per-node|--ntasks|--cpus-per-task)(=|\s)"
                                 r"|-[nc](\s|\d))", line.strip()):
            continue                            # each job of the pack sets these for itself
        lines += body if line == run else [line]
    script = fill("\n".join(lines) + "\n", subs)
    script = set_exclusive(script, False) if core else add_sbatch_option(script, "--exclusive")
    if part:                                    # the partition and its extra #SBATCH options
        script = add_sbatch_option(script, f"--partition={part['name']}")
        for option in part["options"]:
            script = add_sbatch_option(script, fill(option, subs))
    return script


# Gives the jobs of a pack or a job array the walltime they all run with (wall: the longest of
# theirs) in their job_info.json, and each relaxation a run_time for it in its .param
# (run_time_for), so that it stops cleanly before the end - except with max_restarts : 0 and
# tolerate_failures (screening), where nothing would carry it on (as in finalize).
#   Called by: submit_array(), submit_pack(), submit_pack_array()
#   Calls:     run_time_for(), set_param(), slurm_seconds()
def share_walltime(group, infos, wall, args):
    for c, i in zip(group, infos):              # each relaxation stops cleanly before the end
        c.save(dict(c.info(), walltime=wall))
        if c.kind == "geom" and wall and not (args.max_restarts == 0
                                              and getattr(args, "tolerate_failures", False)):
            set_param(c, run_time=run_time_for(slurm_seconds(wall)))


# Runs several planned calculations side by side on one node, as one job (pack : N).
# Writes <workdir>/packs/pack_NNN/job.sh from the jobscript template (pack_script): its #SBATCH
# lines for one whole node, the partition and its options, and the longest walltime the
# calculations were given. In place of the line that runs CASTEP comes one line per calculation
# (pack_line), which runs it in its own directory in the background - srun with its own processes
# and cores per process, --exact, and --mem for its share of PACK_MEM of the node's memory
# (pack_parts), with the output in slurm-<job id>.out there - and then wait. A relaxation's
# run_time is set for the pack's walltime (share_walltime). The job is submitted from the pack's
# directory (send_job) and recorded for each calculation (record_job), so waiting, timings and
# restarts work as for a job of its own (a restart runs on its own: restart plans it with
# finalize(..., pack=1)).
# With charging : core the pack asks for the cores of its jobs on one node, which it shares with
# other jobs (no --exclusive), and each job's srun asks for its own processes and cores per
# process, with --mem-per-cpu for the memory each core brings (pack_parts).
#   Called by: run_batch()
#   Calls:     add_sbatch_option(), new_pack_dir(), pack_castep_line(), pack_line(), pack_parts(),
#              pack_script(), record_job(), send_job(), share_walltime(), slurm_seconds(),
#              slurm_time()
def submit_pack(group, ctx, args, workdir):
    infos = [c.info() for c in group]
    jobs, ncores, part = pack_parts(infos, args)
    walls = [slurm_seconds(i["walltime"]) for i in infos if i.get("walltime")]
    wall = slurm_time(max(walls)) if walls else None
    pdir = new_pack_dir(workdir, "pack")
    run = pack_castep_line(ctx["job_template"])
    body = [pack_line(run, CASTEP_RUN, c.seed, shlex.quote(str(c.dir.resolve())), own, env)
            for c, (own, env, _) in zip(group, jobs)] + ["wait"]
    script = pack_script(ctx["job_template"], run, body, part, ncores, sum(j[2] for j in jobs),
                         max(i["nkpts"] for i in infos), pdir.name,
                         f"{group[0].name}+{len(group) - 1}", args)
    if wall:
        script = add_sbatch_option(script, f"--time={wall}")
    (pdir / JOBSCRIPT).write_text(script)
    (pdir / JOBSCRIPT).chmod(0o755)
    share_walltime(group, infos, wall, args)
    starts = [c.castep.stat().st_size if c.castep.exists() else 0 for c in group]
    jobid = send_job(pdir, JOBSCRIPT, args)
    rel = str(pdir.relative_to(workdir))
    for c, start in zip(group, starts):
        record_job(c, jobid, start, wall, pack=rel)
    where = f"{ncores} cores of a shared node" if args.charging == "core" else "1 node"
    log(f"submitted {rel}: job {jobid} on {where}" + (f" ({part['name']})" if part else "")
        + (f", walltime {wall}" if wall else "") + ": "
        + ", ".join(f"{c.name} ({i['ncores']} processes)" for c, i in zip(group, infos)))


# Runs packs of planned jobs as one Slurm job array (array : N with pack : N): each task is one
# pack - a node (with charging : core, its jobs' cores of a node shared with other jobs) running
# its jobs side by side, each with its own srun, as submit_pack's job does. run_batch gives it
# packs of one size whose jobs all run alike in a pack (pack_parts; if no mem_per_core or
# partition gives the node's memory, each job gets the most any of them needs), so one script
# serves every task: <workdir>/packs/pack_array_NNN/job.sh, made as a pack's is (pack_script),
# with --array=0-<packs-1> and the longest walltime of all the jobs. In place of the line that
# runs CASTEP comes a loop over the task's jobs - lines task x size + 1 to (task + 1) x size of
# tasks.txt beside it, which lists the packs' jobs in order (directory, a tab, seed) - that starts
# each in its own directory in the background (pack_line), with its output in
# slurm-<array job id>_<task>.out there, then waits for them all. A relaxation's run_time is set
# for the array's walltime (share_walltime). The array is submitted from its directory
# (send_job), and each job is recorded (record_job) with its task's job id, <array job id>_<task>
# as squeue and sacct give it, so waiting, timings and restarts work as for a job of its own (a
# restart runs on its own).
#   Called by: run_batch()
#   Calls:     add_sbatch_option(), new_pack_dir(), pack_castep_line(), pack_line(), pack_parts(),
#              pack_script(), record_job(), send_job(), share_walltime(), slurm_seconds(),
#              slurm_time()
def submit_pack_array(packs, ctx, args, workdir):
    group, size = [c for p in packs for c in p], len(packs[0])
    infos = [c.info() for c in group]
    _, ncores, part = pack_parts(infos[:size], args)                # one task's pack
    own, env, mb = max(pack_parts(infos, args)[0], key=lambda j: j[2])  # each job's srun
    walls = [slurm_seconds(i["walltime"]) for i in infos if i.get("walltime")]
    wall = slurm_time(max(walls)) if walls else None
    adir = new_pack_dir(workdir, "pack_array")
    tasks = adir / ARRAY_TASKS
    tasks.write_text("".join(f"{c.dir.resolve()}\t{c.seed}\n" for c in group))
    pick = ("awk -F '\\t' -v n=\"$LINE\" 'NR == n {print $%d}' "
            + shlex.quote(str(tasks.resolve())))
    run = pack_castep_line(ctx["job_template"])
    body = ["SLOT=0",
            f'while [ "$SLOT" -lt {size} ]; do',
            f"    LINE=$((SLURM_ARRAY_TASK_ID * {size} + SLOT + 1))",
            "    TASK_DIR=$(" + pick % 1 + ")",
            "    SEED=$(" + pick % 2 + ")",
            "    " + pack_line(run, CASTEP_RUN, '"$SEED"', '"$TASK_DIR"', own, env,
                               "slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"),
            "    SLOT=$((SLOT + 1))",
            "done",
            "wait"]
    script = pack_script(ctx["job_template"], run, body, part, ncores, size * mb,
                         max(i["nkpts"] for i in infos), adir.name,
                         f"{group[0].name}+{len(group) - 1}", args)
    if wall:
        script = add_sbatch_option(script, f"--time={wall}")
    script = add_sbatch_option(script, f"--array=0-{len(packs) - 1}")
    (adir / JOBSCRIPT).write_text(script)
    (adir / JOBSCRIPT).chmod(0o755)
    share_walltime(group, infos, wall, args)
    starts = [c.castep.stat().st_size if c.castep.exists() else 0 for c in group]
    jobid = send_job(adir, JOBSCRIPT, args)
    rel = str(adir.relative_to(workdir))
    for k, (c, start) in enumerate(zip(group, starts)):
        record_job(c, f"{jobid}_{k // size}", start, wall, pack=rel)
    where = f"{ncores} cores of a shared node" if args.charging == "core" else "1 node"
    log(f"submitted {rel}: job array {jobid} of {len(packs)} packs of {size} jobs "
        f"({group[0].name} ... {group[-1].name}), each pack on {where}"
        + (f" ({part['name']})" if part else "") + (f", walltime {wall}" if wall else "")
        + f"; each job {infos[0]['ncores']} processes")


# The line of a jobscript template that runs CASTEP on {SEED}, for array : N. There must be
# exactly one; unlike a pack's, it needn't use srun, since each task of a job array is a job of
# its own. Returns the line; stops with what to change otherwise.
#   Called by: run_batch(), submit_array()
def array_castep_line(template, name="the runner settings"):
    runs = [line for line in template.splitlines()
            if CASTEP_RUN.search(line) and not line.lstrip().startswith("#")]
    if len(runs) != 1:
        die(f"{name}: array runs each job as a task of a Slurm job array, so the jobscript "
            "template needs exactly one line that runs CASTEP on {SEED}, e.g.\n"
            "    srun --distribution=block:block --hint=nomultithread castep.mpi {SEED}")
    return runs[0]


# A planned job's job.sh as a key that is the same for jobs that can go in one job array: the
# script with the job's seed and name put back as {SEED} and {JOBNAME}, and without its --time and
# --job-name, which the array sets for all of its tasks.
#   Called by: run_batch(), submit_array()
def array_key(calc):
    lines = []
    for line in (calc.dir / JOBSCRIPT).read_text().splitlines():
        if re.match(r"#SBATCH\s+((--time|--job-name)(=|\s|$)|-[tJ])", line.strip()):
            continue
        lines.append(line.replace(calc.seed, "{SEED}").replace(calc.name, "{JOBNAME}"))
    return "\n".join(lines) + "\n"


# Runs planned jobs as one Slurm job array (array : N).
# Writes <workdir>/packs/array_NNN/job.sh: the jobs' own job.sh (array_key, the same for all of
# them), with --array=0-<n-1>, the longest of their walltimes and a job name for the group; in
# place of the line that runs CASTEP, lines that read the task's directory and seed from tasks.txt
# beside it (line SLURM_ARRAY_TASK_ID + 1: directory, a tab, seed), go there and run CASTEP on that
# seed, with its output in slurm-<array job id>_<task>.out there. A relaxation's run_time is set
# for the array's walltime (share_walltime). The array is submitted from its directory
# (send_job), and each job is recorded (record_job) with its task's job id, <array job id>_<task>
# as squeue and sacct give it, so waiting, timings and restarts work as for a job of its own (a
# restart runs on its own). With pack : N, jobs planned to share a node go as job arrays of packs
# instead (submit_pack_array).
#   Called by: run_batch()
#   Calls:     add_sbatch_option(), array_castep_line(), array_key(), job_size(), new_pack_dir(),
#              record_job(), send_job(), share_walltime(), slurm_seconds(), slurm_time()
def submit_array(group, ctx, args, workdir):
    infos = [c.info() for c in group]
    walls = [slurm_seconds(i["walltime"]) for i in infos if i.get("walltime")]
    wall = slurm_time(max(walls)) if walls else None
    adir = new_pack_dir(workdir, "array")
    tasks = adir / ARRAY_TASKS
    tasks.write_text("".join(f"{c.dir.resolve()}\t{c.seed}\n" for c in group))
    text = array_key(group[0])
    run = array_castep_line(text)
    m = CASTEP_RUN.search(run)
    cmd = (run[:m.start()] + m.group(1) + '"$SEED"' + run[m.end():]).rstrip()
    pick = ("awk -F '\\t' -v n=\"$SLURM_ARRAY_TASK_ID\" 'NR == n + 1 {print $%d}' "
            + shlex.quote(str(tasks.resolve())))
    lines = []
    for line in text.splitlines():
        if line != run:
            lines.append(line)
            continue
        lines += ["TASK_DIR=$(" + pick % 1 + ")", "SEED=$(" + pick % 2 + ")",
                  'cd "$TASK_DIR" || exit 1',
                  cmd + ' > "slurm-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2>&1']
    name = f"{group[0].name}+{len(group) - 1}"
    script = "\n".join(lines).replace("{SEED}", "$SEED").replace("{JOBNAME}", name) + "\n"
    script = add_sbatch_option(script, f"--job-name={name}")
    if wall:
        script = add_sbatch_option(script, f"--time={wall}")
    script = add_sbatch_option(script, f"--array=0-{len(group) - 1}")
    (adir / JOBSCRIPT).write_text(script)
    (adir / JOBSCRIPT).chmod(0o755)
    share_walltime(group, infos, wall, args)
    starts = [c.castep.stat().st_size if c.castep.exists() else 0 for c in group]
    jobid = send_job(adir, JOBSCRIPT, args)
    rel = str(adir.relative_to(workdir))
    for k, (c, start) in enumerate(zip(group, starts)):
        record_job(c, f"{jobid}_{k}", start, wall, pack=rel)
    log(f"submitted {rel}: job array {jobid} of {len(group)} jobs ({group[0].name} ... "
        f"{group[-1].name}), each on {job_size(infos[0])}" + (f", walltime {wall}" if wall else ""))


# How many of a pack's n dry runs (dryrun_pack : N) run at once: one a core of a node (with
# charging : core, as many as a node's cores hold at dryrun_cpus cores each), and on whole nodes
# no more than the node's memory holds at dryrun_mem_gb each, when its GB per core is known
# (dryrun_core_gb); all n without cores_per_node. submit_dryrun_pack runs the rest in turn.
#   Called by: submit_dryrun_pack()
#   Calls:     dryrun_core_gb(), dryrun_cpus()
def dryrun_lanes(n, args):
    cpn = args.cores_per_node
    if not cpn:
        return n
    lanes = cpn // dryrun_cpus(1, args.dryrun_mem_gb, args)
    per = dryrun_core_gb(args)
    if per and args.charging != "core":
        lanes = min(lanes, math.floor(cpn * per / args.dryrun_mem_gb))
    return max(1, min(n, lanes))


# Runs several dry runs side by side, as one job (dryrun_pack : N).
# Writes <workdir>/packs/dryrun_NNN/job.sh from the dry-run template (dryrun_template_text), filled
# in for one core per dry run (on one node), with dryrun_partition, dryrun_time and dryrun_options
# (dryrun_sbatch); in place of the line that runs CASTEP comes one line per calculation (pack_line):
# its dry run, in its own directory, in the background, on one process (srun --ntasks=1 --exact
# --mem=dryrun_mem_gb), then wait. Each calculation gets the job id as its dryrun_jobid
# (record_job), so wait_for and handle_failure treat it as its own dry-run job; each also keeps
# its own dryrun.sh, which --resubmit-failed uses. With charging : core each dry run has the cores
# that hold dryrun_mem_gb (dryrun_cpus), asks for their memory with --mem-per-cpu (the job's
# SLURM_MEM_PER_CPU, or else dryrun_core_gb), and the job shares its node (no --exclusive).
# A pack of more dry runs than a node holds at once (dryrun_lanes) runs that many lanes, each on
# its own core(s): the job is filled in for the lanes, and each lane is a group of the pack's dry
# runs, as even as can be, that runs in the background one after another, each going on even if
# the one before failed (|| true); dryrun_time must allow for a lane's dry runs in turn.
#   Called by: prepare_all()
#   Calls:     dryrun_core_gb(), dryrun_lanes(), dryrun_sbatch(), dryrun_subs(),
#              dryrun_template_text(), fill(), new_pack_dir(), pack_castep_line(), pack_line(),
#              record_job(), send_job(), set_exclusive()
def submit_dryrun_pack(group, ctx, args):
    workdir, n, core = group[0].dir.parent, len(group), args.charging == "core"
    lanes = dryrun_lanes(n, args)
    pdir = new_pack_dir(workdir, "dryrun")
    subs = dryrun_subs(pdir.name, f"{group[0].name}_dry+{n - 1}", lanes,
                       lanes * args.dryrun_mem_gb, args)
    template = dryrun_template_text(ctx["job_template"], ctx["dryrun_template"])
    run = pack_castep_line(template, rx=DRYRUN_RUN, what="dryrun_pack")
    lines = []
    for line in template.splitlines():
        if line != run:
            lines.append(line)
            continue
        cpus, per = subs["{CPUS_PER_TASK}"], dryrun_core_gb(args)
        mb = math.floor(per * 1024) if core and per else math.ceil(args.dryrun_mem_gb * 1024)
        memory = (f"--mem-per-cpu=${{SLURM_MEM_PER_CPU:-{mb}}}M" if core and per else
                  f"--mem={mb}M")
        own = (f"--nodes=1 --ntasks=1 --ntasks-per-node=1 --cpus-per-task={cpus} --exact "
               + memory)
        env = f"OMP_NUM_THREADS=1 SRUN_CPUS_PER_TASK={cpus} "
        where = {c.name: shlex.quote(str(c.dir.resolve())) for c in group}
        if lanes == n:                          # all at once
            lines += [pack_line(run, DRYRUN_RUN, c.seed, where[c.name], own, env) for c in group]
        else:                                   # lanes, each running its share in turn
            for k in range(lanes):
                lane = group[round(k * n / lanes):round((k + 1) * n / lanes)]
                lines += (["("] + ["    " + pack_line(run, DRYRUN_RUN, c.seed, where[c.name], own,
                                                      env, background=False) + " || true"
                                   for c in lane] + [") &"])
        lines.append("wait")
    script = dryrun_sbatch(fill("\n".join(lines) + "\n", subs), subs, args)
    if core:
        script = set_exclusive(script, False)
    (pdir / JOBSCRIPT).write_text(script)
    (pdir / JOBSCRIPT).chmod(0o755)
    jobid = send_job(pdir, JOBSCRIPT, args)
    for c in group:
        record_job(c, jobid, 0, None, key="dryrun_jobid")
    if lanes == n:
        log(f"submitted {n} dry runs in {pdir.relative_to(workdir)}: job {jobid} ("
            + ", ".join(c.name for c in group) + ")")
    else:
        log(f"submitted {n} dry runs in {pdir.relative_to(workdir)}: job {jobid}, {lanes} side "
            f"by side, each running up to {math.ceil(n / lanes)} in turn ({group[0].name} ... "
            f"{group[-1].name})")


# Asks the scheduler whether a job is still queued or running.
# Uses status_cmd, or squeue/qstat. A job that has left the queue, or shows a finished state
# (END_STATES), is not active; if the scheduler doesn't answer within 2 minutes the job is assumed
# to be still there. The workflow checks its MACE jobs with it too.
#   Called by: wait_for(), the workflow
def job_active(jobid, args):
    if args.status_cmd:
        cmd = args.status_cmd
    elif "sbatch" in args.submit_cmd:
        cmd = "squeue -h -j {jobid} -o '%i %T'"
    else:
        cmd = "qstat -j {jobid}" if os.environ.get("SGE_ROOT") else "qstat {jobid}"
    try:
        res = subprocess.run(cmd.format(jobid=jobid), shell=True, capture_output=True,
                             text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return True                          # scheduler slow: assume still there
    out = res.stdout
    return res.returncode == 0 and jobid in out and not any(s in out for s in END_STATES)


# The queue's state of all the user's jobs, from one squeue call with one line per array task
# (-r), for array : N, where a step may have hundreds of jobs and array tasks queued: {job id:
# state}, an array task's job id being <array job id>_<task>. None if squeue failed or didn't
# answer within 2 minutes, so that every job counts as still there.
#   Called by: wait_for()
def queue_states():
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or ""
    try:
        res = subprocess.run(["squeue", "-h", "-r", "-o", "%i %T"] + (["-u", user] if user else []),
                             capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    return {f[0]: f[1] for f in (line.split() for line in res.stdout.splitlines()) if len(f) > 1}


# A size in bytes as 'x.y GB' or 'n MB', for log messages.
#   Called by: remove_check(), restart()
def file_size(nbytes):
    return f"{nbytes / 2**30:.1f} GB" if nbytes >= 2**30 else f"{nbytes / 2**20:.0f} MB"


# For delete_check : true: deletes the .check files of a calculation that finished successfully.
# Deletes <seed>.check and any .check.bad put aside by restart once Calc.done() is true, and logs
# the space freed. Does nothing for unfinished calculations, so a relaxation keeps the .check it
# carries on from.
#   Called by: series_main(), run_batch(), wait_for()
#   Calls:     Calc.done(), file_size()
def remove_check(calc, args):
    """delete_check : true. Delete the .check files of a calculation that has finished
    successfully (<seed>.check, and a .check.bad put aside by a restart): CASTEP only
    needs them to carry on an unfinished run."""
    if not args.delete_check:
        return
    files = sorted(f for f in calc.dir.glob(f"{calc.seed}.check*") if f.is_file())
    if not files or not calc.done():
        return
    size = sum(f.stat().st_size for f in files)
    for f in files:
        f.unlink()
    log(f"{calc.name}: finished, so deleted {', '.join(f.name for f in files)} "
        f"({file_size(size)})")


# Whether the scheduler stopped the latest job at its time limit.
# sacct reports TIMEOUT, or Slurm's 'DUE TO TIME LIMIT' message is in one of the job's output files
# (any file with the job id in its name). Lets restart tell a job that was killed from one that
# crashed.
#   Called by: restart()
def killed_at_time_limit(calc, args):
    """Did the scheduler stop the job at its time limit? (sacct, or Slurm's message in the
    job's output files)"""
    jobid = str(calc.info().get("jobid"))
    if "sbatch" in args.submit_cmd:
        try:
            res = subprocess.run(["sacct", "-n", "-X", "-j", jobid, "-o", "State"],
                                 capture_output=True, text=True, timeout=60)
            if "TIMEOUT" in res.stdout:
                return True
        except (OSError, subprocess.SubprocessError):
            pass
    return any(jobid in f.name and f.is_file() and f.stat().st_size < 5e7
               and "DUE TO TIME LIMIT" in f.read_text(errors="replace")
               for f in calc.dir.iterdir())


# The structure to restart a relaxation from when it can't carry on from its .check.
# Takes the last 'Fractional coordinates' table printed before the last 'finished iteration' in the
# .castep (all runs) - the structure of the last geometry step CASTEP finished - or, if the .castep
# has none, <seed>-out.cell. Maps CASTEP's atom order back onto calc.atoms (castep_order). For a
# variable-cell relaxation (batch mode) the lattice of that step is taken too (last_lattice).
# Returns (Atoms, steps finished, the file it came from), or (None, steps, None).
#   Called by: restart()
#   Calls:     Calc.output(), castep_order(), last_lattice(), parse_castep_input(),
#              read_primitive()
def last_geometry(calc):
    """The structure of the last geometry step a relaxation finished, read from its .castep
    (all runs): calc.atoms with those positions, and the number of steps finished. CASTEP
    prints each structure before working out its energy, so it is the last one printed
    before the last 'finished iteration'. If the .castep has none, the <seed>-out.cell
    written at the end of the last run is used. Returns (atoms, steps, the file read), or
    (None, steps, None)."""
    text = calc.output()
    ends = [m.start() for m in GEOM_STEP.finditer(text)]
    if not ends:
        return None, 0, None
    head = text[:ends[-1]].rsplit("Fractional coordinates of atoms", 1)
    rows = (ATOM_ROW.findall(head[-1]) if len(head) > 1 else [])[:len(calc.atoms)]
    order = castep_order(calc.atoms, [r[0] for r in rows])     # CASTEP's order of the atoms
    frac, source = [[float(x) for x in r[1:]] for r in rows], calc.castep
    lattice = None if calc.fix_cell else last_lattice(text[:ends[-1]])
    if order is None:
        source = calc.dir / f"{calc.seed}-out.cell"
        try:
            out = read_primitive(parse_castep_input(source.read_text(errors="replace")))
        except (OSError, SystemExit, ValueError, IndexError):
            return None, len(ends), None
        order = castep_order(calc.atoms, out.get_chemical_symbols())
        if order is None:
            return None, len(ends), None
        frac = out.get_scaled_positions(wrap=False)
        lattice = None if calc.fix_cell else np.asarray(out.cell)
    scaled = calc.atoms.get_scaled_positions(wrap=False)
    for k, i in enumerate(order):
        scaled[i] = frac[k]
    atoms = calc.atoms.copy()
    if lattice is not None:                        # a variable-cell relaxation
        atoms.set_cell(lattice, scale_atoms=False)
    atoms.set_scaled_positions(scaled)
    return atoms, len(ends), source.name


# The dry-run output a calculation is planned from.
# Its own <seed>.dryrun.castep, or for a single point (kind 'defect') its relaxation's. None if
# there isn't one.
#   Called by: restart()
def planning_dryrun(calc, ctx):
    """Text of the dry run a calculation is planned from (a single point on a relaxed cell
    uses its relaxation's), or None."""
    src = calc
    if calc.kind == "defect":
        src = next((c for c in ctx["calcs"] if c.kind == "geom" and c.label == calc.label), calc)
    return src.dryrun_file.read_text(errors="replace") if src.dryrun_file.exists() else None


# Automatic recovery of a job that stopped before the end (max_restarts > 0).
# Works out how the latest run ended (Calc.run_text, the .err files, killed_at_time_limit): CASTEP
# stopped itself at run_time; the scheduler killed it at the time limit; CASTEP couldn't read the
# .check it was carrying on from (CHECK_UNREADABLE); or it stopped at run_time without finishing
# writing its .check. For anything else (a crash, geom_max_iter reached) it returns False and
# handle_failure reports the problem. Otherwise it:
# - stops if max_restarts is used up, or if the job made no progress and can't get more time or
#   more processes;
# - records how the run ended, measures the memory again (calibrate) and plans the job again
#   (planning_dryrun + finalize), which may change its layout or partition;
# - for a relaxation, carries on from the .check (continuation : default, which replaces any reuse,
#   e.g. of the ground state a ΔSCF excited state started from) when it can; otherwise
#   puts the .check aside (deletes it with delete_check : true) and writes the structure of the
#   last geometry step into the .cell (last_geometry + write_cell). After a second unreadable
#   .check it stops using .check files for that calculation;
# - for a single point, renames the .castep and starts again;
# - renames the .err files, asks for the longest walltime (add_sbatch_option), sets a new run_time
#   (run_time_for; the margin doubles if the job was killed), counts the restart in
#   job_info.json and resubmits the job (submit); with defer (batch mode: handle_failure) it
#   marks it restart_ready instead, without a job, and run_batch submits it when max_jobs leaves
#   a place (so a pack or array that ran out of time, whose jobs restart one by one, can't put
#   many more jobs in the queue than max_jobs).
#   Called by: handle_failure()
#   Calls:     Calc.err_files(), Calc.run_text(), add_sbatch_option(), calibrate(), cell_options(),
#              file_size(), finalize(), killed_at_time_limit(), last_geometry(), planning_dryrun(),
#              run_time_for(), sbatch_value(), set_param(), slurm_seconds(), submit(),
#              write_cell()
def restart(calc, args, defer=False):
    """Resubmit a job that stopped before the end, when it can carry on:
     - CASTEP stopped itself at run_time: a relaxation carries on from its .check file
       (continuation : default);
     - the scheduler stopped it at the time limit: a relaxation carries on from its .check
       file, a single point starts again;
     - a relaxation could not read its .check file, or stopped at run_time but did not
       finish writing it, or has none: the .check is put aside and the relaxation starts
       again (without continuation) from the structure of its last geometry step.
    The job is planned again first with the memory measured meanwhile (which can give it a
    better layout) and gets the longest walltime. Returns False if none of these apply."""
    seg, info = calc.run_text(), calc.info()
    errs = "".join(e.read_text(errors="replace") for e in calc.err_files())
    param = (calc.dir / f"{calc.seed}.param").read_text()
    geom, finished = calc.kind == "geom", bool(FINISHED.search(seg))
    unreadable = (geom and not finished and not SCF_CYCLE.search(seg)
                  and bool(re.search(r"^\s*continuation\b", param, re.I | re.M))
                  and bool(CHECK_UNREADABLE.search(seg + errs)))
    stopped = OUT_OF_TIME in seg and not unreadable          # CASTEP stopped at run_time
    killed = not finished and not unreadable and killed_at_time_limit(calc, args)
    if not (unreadable or stopped or killed):
        return False
    cut = stopped and not finished                 # ... but didn't finish writing the .check
    what, n = f"{calc.name} (job {info.get('jobid')})", info.get("restarts", 0)
    if unreadable:
        why = "could not read its .check file"
    elif cut:
        why = "stopped at run_time but did not finish writing its .check file" + (
            " before the time limit" if killed else "")
    else:
        why = "stopped itself before the time limit" if stopped else "was stopped at the time limit"
    more = ("Give it more nodes (max_nodes) or a longer walltime (max_walltime, or #SBATCH "
            "--time in the jobscript), then rerun.")
    if n >= args.max_restarts:
        die(f"{what} {why}, after {n} restarts already (max_restarts). "
            + ("Check its output files and the disk quota." if unreadable or cut else more))
    check = calc.dir / f"{calc.seed}.check"
    if unreadable:
        info["bad_checks"] = info.get("bad_checks", 0) + 1
        if info["bad_checks"] == 2:
            info["no_continuation"] = True
            log(f"note: that is the second .check file of {calc.name} CASTEP could not read, "
                "so from now on it carries on from its last geometry step instead")
    use_check = (geom and check.exists() and not (unreadable or cut)
                 and not info.get("no_continuation"))
    nscf = len(re.findall(ENERGY_PATTERNS["final"], seg))
    progress = unreadable or (geom and (bool(GEOM_STEP.search(seg)) or nscf > 1))
    old_procs = (info.get("runs") or [{}])[-1].get("procs") or info.get("ncores")
    old_layout = [info.get(k) for k in ("nodes", "tasks_per_node", "ncores", "cpus_per_task",
                                        "partition")]
    had, longest = (slurm_seconds(info.get(k) or "") for k in ("walltime", "walltime_max"))
    for run in info.get("runs", []):               # runs submitted by an older version
        run.setdefault("procs", info.get("ncores"))
    if info.get("runs"):                           # for job_timing: count its walltime?
        info["runs"][-1]["end"] = "killed" if killed else "unreadable" if unreadable else "stopped"
    calc.save(info)

    calibrate(*args.all, args)                     # plan it again, with what's known now
    text, desc = planning_dryrun(calc, args.ctx), None
    if text:                                       # (a job of a pack restarts on its own)
        desc = finalize(calc, text, args.ctx, args, quiet=True, show=False, pack=1)
    info = calc.info()
    new_procs = info.get("ncores")
    if not progress and (not had or not longest or had >= longest) and not (
            new_procs and old_procs and new_procs > old_procs):
        die(f"{what} ran out of time before it finished a single geometry step. {more}" if geom
            else f"{what} was stopped at the time limit, already with the longest walltime. "
            f"{more}")

    stamp = time.strftime("%Y%m%d-%H%M%S")
    if (unreadable or cut) and check.exists():     # put the .check aside (or delete it)
        size, bad = check.stat().st_size, check.with_name(check.name + ".bad")
        if args.delete_check:
            check.unlink()
        else:
            check.replace(bad)
        log(f"WARNING: CASTEP {'could not read' if unreadable else 'did not finish writing'} "
            f"{calc.name}/{check.name} ({file_size(size)}). This usually means the file was "
            "cut short while it was being written - is the disk quota full? "
            + ("It has been deleted (delete_check : true)." if args.delete_check else
               f"It is kept as {bad.name} (delete it to free the space)."))
    steps = 0
    if geom and not use_check:                     # start from its last geometry step
        atoms, steps, source = last_geometry(calc)
        if atoms is None:
            steps = 0
        else:
            cell = calc.dir / f"{calc.seed}.cell"
            start = cell.with_name(cell.name + ".start")
            if not start.exists():
                cell.rename(start)                 # the structure it first started from
            keep, opts = cell_options(calc, args.ctx, info["kgrid"])
            write_cell(cell, atoms, info["kgrid"], keep,
                       f"{calc.name}: {len(atoms)} atoms, {atoms.get_chemical_formula()}, the "
                       f"structure after {steps} geometry steps (from {source})", **opts)
    if not geom and calc.castep.exists():          # a single point starts again
        calc.castep.rename(calc.castep.with_name(f"{calc.castep.name}.timeout-{stamp}"))
    for f in calc.err_files():                     # (else it looks failed straight away)
        f.rename(f.with_name(f"{f.name}.{stamp}"))
    job = calc.dir / JOBSCRIPT
    script = job.read_text()
    tmax = info.get("walltime_max") or sbatch_value(script, "--time")
    smax = slurm_seconds(tmax) if tmax else None
    if tmax:
        job.write_text(add_sbatch_option(script, f"--time={tmax}"))
    margin = info.get("margin_factor", 1.0)
    if killed and re.search(r"^\s*run_time\b", param, re.I | re.M):
        margin *= 2                                # stop earlier next time
    if geom:
        new = {"continuation": "default" if use_check else None}
        if use_check:                              # CASTEP takes continuation or reuse, not both
            new["reuse"] = None
        if smax:
            new["run_time"] = run_time_for(smax, margin)
        set_param(calc, **new)
    info.update(restarts=n + 1, walltime=tmax, margin_factor=margin)
    calc.save(info)
    how = ("from its .check file" if use_check else
           f"from the structure of its last geometry step (step {steps})" if steps else
           "from the structure in its .cell file" if geom else "from the start")
    log(f"{what} {why} - resubmitting it {how}" + (f" with walltime {tmax}" if tmax else "")
        + (" when max_jobs leaves a place" if defer else ""))
    layout = [info.get(k) for k in ("nodes", "tasks_per_node", "ncores", "cpus_per_task",
                                    "partition")]
    if desc and layout != old_layout:
        log(f"  {calc.name} planned again (mem_replicated {args.mem_replicated:g}): {desc}")
    if defer:
        hold_restart(calc)
    else:
        submit(calc, args)
    return True


# Holds a job ready to go again (restart, or --resubmit-failed's resubmission of its job.sh) for
# run_batch to submit when max_jobs leaves a place: restart_ready in job_info.json, and no job
# id (so it is neither a job to wait for nor one to plan from its dry run again).
#   Called by: handle_failure(), restart()
def hold_restart(calc):
    info = calc.info()
    info.pop("jobid", None)
    info["restart_ready"] = True
    calc.save(info)


# Deals with a job or dry run that left the queue without finishing (or wrote a .err file).
# For a job, restart is tried first. Then, with --resubmit-failed, the existing jobscript is
# resubmitted once (the old .castep and .err files are renamed). In batch mode a job is not
# submitted from here but held for run_batch (restart's defer, hold_restart), which sends it
# within max_jobs. Otherwise the script stops with an
# explanation: a relaxation that ran out of time with max_restarts : 0, one that reached
# geom_max_iter, or anything else (with the end of the .castep and the .err files).
#   Called by: wait_for()
#   Calls:     Calc.err_files(), Calc.output(), Calc.run_text(), hold_restart(), restart(),
#              submit()
def handle_failure(calc, args, resubmitted, key="jobid"):
    script = JOBSCRIPT if key == "jobid" else DRYRUN_SCRIPT
    what = f"{calc.name}{'' if key == 'jobid' else ' dry run'} (job {calc.info().get(key)})"
    hold = key == "jobid" and bool((getattr(args, "ctx", None) or {}).get("batch"))
    if key == "jobid" and args.max_restarts and restart(calc, args, defer=hold):
        return
    if args.resubmit_failed and calc.name not in resubmitted:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for f in [calc.castep] + calc.err_files():
            if f.exists():
                f.rename(f.with_name(f"{f.name}.failed-{stamp}"))
        log(f"{what} failed - resubmitting the existing {script}"
            + (" when max_jobs leaves a place" if hold else ""))
        resubmitted.add(calc.name)
        if hold:
            hold_restart(calc)
        else:
            submit(calc, args, script, key)
        return
    out = calc.output()
    seg = calc.run_text() if key == "jobid" else out
    if calc.kind == "geom" and key == "jobid" and OUT_OF_TIME in seg:
        die(f"{what} ran out of time and stopped, writing a .check file. In "
            f"{calc.dir}/{calc.seed}.param add 'continuation : default', then rerun with "
            "--resubmit-failed (max_restarts above 0 does this for you).")
    if calc.kind == "geom" and key == "jobid" and FINISHED.search(seg):
        die(f"{what}: the geometry optimisation did not converge within geom_max_iter.\n"
            f"In {calc.dir}/{calc.seed}.param raise geom_max_iter and add "
            "'continuation : default' to carry on from the .check file, then rerun with "
            "--resubmit-failed.")
    tail = "\n".join(out.splitlines()[-12:])
    errs = "".join(e.read_text(errors="replace") for e in calc.err_files())
    die(f"{what} finished without completing.\n"
        f"--- end of {calc.castep.name} ---\n{tail}\n{errs}\n---\n"
        f"Check the scheduler output files in {calc.dir}. When fixed, either delete that "
        f"directory (it is rebuilt) or edit its {script} and rerun with --resubmit-failed.")


# Waits until every calculation in the list is done; where the script spends most of its time.
# Polls every `poll` seconds. key 'jobid' waits for the jobs (Calc.done); key 'dryrun_jobid' for
# the dry runs (dryrun_output). A job that writes a .err file, or has left the queue for 3 polls
# (job_active) without finishing, goes to handle_failure (and so restart). Finished jobs get
# remove_check (delete_check : true). Stops the script when <workdir>/STOP appears. With --no-wait,
# reports what is still queued and returns False instead of waiting. Batch mode: with any_done it
# returns as soon as one of the jobs has finished (so run_batch can submit the next), and with
# tolerate_failures a job handle_failure would stop the script for is marked failed instead
# (mark_failed) and counts as finished. With array : N the queue is read once a poll for all the
# jobs (queue_states) instead of a squeue call for each. progress (run_batch's) is called on every
# poll, once the jobs have been checked. A job handle_failure held for a restart (restart_ready)
# is no longer waited for, and wait_for returns, so run_batch can send it within max_jobs.
#   Called by: series_main(), prepare_all(), run_batch()
#   Calls:     Calc.done(), Calc.err_files(), dryrun_output(), handle_failure(), job_active(),
#              mark_failed(), queue_states(), remove_check()
def wait_for(calcs, args, key="jobid", any_done=False, progress=None):
    """Poll until every calc's job (key='jobid') or dry run (key='dryrun_jobid') is done
    (with any_done, until at least one is). Returns False if --no-wait and some are still
    queued or running."""
    if key == "jobid":          # (a job marked failed, with tolerate_failures, has finished)
        is_done, what = (lambda c: c.done() or bool(c.peek().get("failed"))), ""
    else:
        is_done, what = (lambda c: dryrun_output(c) is not None), " dry run"
    gone, resubmitted, announced, held = {c.name: 0 for c in calcs}, set(), False, False
    stop = getattr(args, "stop_file", None)
    while True:
        if stop and stop.exists():
            stop.unlink()
            log(f"found {stop.name} - stopping. Jobs already submitted keep running; rerun the "
                "same command to carry on.")
            sys.exit(0)
        waiting = []
        polled = getattr(args, "array", 1) > 1
        states = queue_states() if polled else None
        for c in calcs:
            if is_done(c):
                if key == "jobid":
                    remove_check(c, args)      # delete_check : true
                continue
            waiting.append(c)
            jobid = str(c.peek()[key])
            if polled:                         # (squeue didn't answer: still there)
                there = states is None or (jobid in states and
                                           not any(s in states[jobid] for s in END_STATES))
            else:
                there = job_active(jobid, args)
            if c.err_files() or not there:
                gone[c.name] += 1          # job left the queue: confirm over a few polls
                if c.err_files() or gone[c.name] >= 3:
                    try:
                        handle_failure(c, args, resubmitted, key)
                    except SystemExit as err:
                        if key != "jobid" or not getattr(args, "tolerate_failures", False):
                            raise
                        mark_failed(c, err)        # note it and carry on with the others
                        waiting.remove(c)
                    gone[c.name] = 0
                    if c.peek().get("restart_ready"):   # back to run_batch, to go within max_jobs
                        waiting.remove(c)
                        held = True
            else:
                gone[c.name] = 0
        if progress:
            progress()
        if not waiting or held:
            return True
        if any_done and len(waiting) < len(calcs):
            return True
        if args.no_wait and not any(gone.values()):
            log("still queued/running: " + ", ".join(
                f"{c.name}{what} (job {c.info().get(key)})" for c in waiting))
            log("rerun the same command later to carry on")
            return False
        if not announced:
            log(f"waiting for {', '.join(c.name + what for c in waiting)} "
                f"(polling every {args.poll} s)")
            announced = True
        time.sleep(20 if any(gone.values()) else args.poll)


# Reads the energy of a finished calculation from its .castep.
# The last 'Final energy', 'Final free energy' or 0K estimate, as chosen by 'energy' (auto: the 0K
# estimate if CASTEP printed one, else the final energy). Stops if the last SCF didn't converge;
# unconverged SCFs earlier in a relaxation are fine. Returns (energy, which one).
#   Called by: results_row()
#   Calls:     Calc.output()
def read_energy(calc, which):
    """Energy of the last SCF in the .castep: the single point, or the final structure of
    a relaxation (earlier steps of a relaxation may have unconverged SCFs; that's fine)."""
    text = calc.output()
    ends = [m.end() for m in re.finditer(ENERGY_PATTERNS["final"], text)]
    last_scf = text[ends[-2]:] if len(ends) > 1 else text
    if any(s in last_scf for s in SCF_FAILED):
        die(f"the SCF did not converge in {calc.castep} - the energy is unreliable. Raise "
            "max_scf_cycles or adjust the mixing, then delete the directory and rerun.")
    found = {}
    for key, pattern in ENERGY_PATTERNS.items():
        hits = re.findall(pattern, text)
        if hits:
            found[key] = float(hits[-1])
    if which == "auto":
        which = "0K" if "0K" in found else "final"
    if which not in found:
        die(f"no '{which}' energy found in {calc.castep}")
    return found[which], which


# --------------------------------------------------------------------------
# Batch mode: any list of calculations
# --------------------------------------------------------------------------
# The kpoint_mp_offset that puts Gamma in a Monkhorst-Pack grid: 1/(2n) along each direction with
# an even number of points n (an odd n already includes Gamma).
#   Called by: cell_options()
def gamma_offset(grid):
    return [0.5 / n if n % 2 == 0 else 0.0 for n in grid]


# What write_cell needs for one calculation beyond its structure and k-grid.
# The items kept from your .cell (with the symmetry keywords if the calculation uses symmetry, e.g.
# a pristine cell; without for a defect cell) and the batch-mode options: fix_cell=False for a
# variable-cell relaxation, extra lines for the .cell (a block or keyword in them replaces the one
# from your .cell, e.g. a species_pot with other pseudopotentials) and the k-point offset ('gamma'
# becomes gamma_offset). For the size series it gives just the kept items, as before.
# Returns (keep_items, options for write_cell).
#   Called by: restart(), write_inputs()
#   Calls:     gamma_offset(), parse_castep_input()
def cell_options(calc, ctx, grid):
    keep = ctx["keep_items"] if calc.symmetry else ctx["defect_keep_items"]
    opts = {}
    if not calc.fix_cell:
        opts["fix_cell"] = False
    if calc.cell_extra:
        names = {it.get("name") or it.get("key") for it in parse_castep_input(calc.cell_extra)
                 if it["type"] != "raw"}
        keep = [it for it in keep if (it.get("name") or it.get("key")) not in names]
        opts["extra"] = calc.cell_extra
    if calc.kpoint_offset is not None:
        opts["offset"] = (gamma_offset(grid) if calc.kpoint_offset == "gamma"
                          else [float(x) for x in calc.kpoint_offset])
    return keep, opts


# A value for a .param keyword set from Python: true/false for booleans, numbers as they are, a
# list as its items separated by spaces, None to remove the keyword (set_param).
#   Called by: job_spec(), write_inputs()
def param_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return " ".join(str(x) for x in v)
    return None if v is None else str(v)


# The fingerprint of a calculation's dry run, for sharing dry runs in batch mode.
# Calculations with the same fingerprint get the same k-points and memory estimate from CASTEP, so
# one dry run does for all of them: the .param (without run_time and continuation, which are set
# per job), the .cell's lattice, k-points and everything else, and the atoms - with their positions
# only when the calculation uses symmetry (they then change the irreducible k-points), otherwise
# just the species and any starting spins. write_inputs saves it in job_info.json.
#   Called by: write_inputs()
def dryrun_key(calc):
    cell = (calc.dir / f"{calc.seed}.cell").read_text().splitlines()[1:]   # without the title
    if not calc.symmetry and "%block positions_frac" in cell:
        a, b = cell.index("%block positions_frac"), cell.index("%endblock positions_frac")
        cell[a + 1:b] = sorted(" ".join(row.split()[:1] + row.split()[4:]) for row in cell[a + 1:b])
    param = [line for line in (calc.dir / f"{calc.seed}.param").read_text().splitlines()
             if not re.match(r"\s*(run_time|continuation)\b", line, re.I)]
    return hashlib.md5("\n".join(cell + ["--"] + param).encode()).hexdigest()


# The last lattice CASTEP printed ('Real Lattice(A)') in a piece of .castep output, as a 3x3 array,
# or None. A variable-cell relaxation prints the new lattice at every step.
#   Called by: final_atoms(), last_geometry()
def last_lattice(text):
    found = REAL_LATTICE.findall(text)
    try:
        rows = [[float(x) for x in row.split()[:3]] for row in found[-1].splitlines()]
    except (IndexError, ValueError):
        return None
    return np.array(rows) if len(rows) == 3 and all(len(r) == 3 for r in rows) else None


# The final structure of a calculation: calc.atoms (same atom order and starting spins) with the
# positions and, for a variable-cell relaxation, the lattice CASTEP ended with.
# From <seed>-out.cell if CASTEP wrote it, otherwise from the last lattice and 'Fractional
# coordinates' table in the .castep (for a single point that is its starting structure); CASTEP's
# atoms are matched to calc.atoms by element (castep_order). None if neither can be read. The
# workflow uses it to carry a relaxed structure on to the next step.
#   Called by: relaxed_atoms(), the workflow
#   Calls:     Calc.output(), castep_order(), last_lattice(), parse_castep_input(),
#              read_primitive()
def final_atoms(calc):
    out_cell, atoms, order = calc.dir / f"{calc.seed}-out.cell", None, None
    if out_cell.exists():
        try:
            atoms = read_primitive(parse_castep_input(out_cell.read_text(errors="replace")))
            order = castep_order(calc.atoms, atoms.get_chemical_symbols())
        except (SystemExit, ValueError, IndexError):
            atoms = None
    if order is None:
        text = calc.output()
        rows = ATOM_ROW.findall(text.rsplit("Fractional coordinates of atoms", 1)[-1])
        rows = rows[:len(calc.atoms)]
        order = castep_order(calc.atoms, [r[0] for r in rows])
        if order is None:
            return None
        lattice = None if calc.fix_cell else last_lattice(text)
        atoms = Atoms([r[0] for r in rows],
                      scaled_positions=[[float(v) for v in r[1:]] for r in rows],
                      cell=calc.atoms.cell if lattice is None else lattice, pbc=True)
    result = calc.atoms.copy()
    result.set_cell(atoms.cell, scale_atoms=False)
    frac, scaled = atoms.get_scaled_positions(wrap=False), result.get_scaled_positions(wrap=False)
    for k, i in enumerate(order):
        scaled[i] = frac[k]
    result.set_scaled_positions(scaled)
    return result


# The energy of a calculation in batch mode, without stopping on a problem (unlike read_energy).
# Returns (energy or None, which energy, whether the last SCF converged); which as for energy
# (auto: the 0K estimate if CASTEP printed one, else the final energy).
#   Called by: job_result()
#   Calls:     Calc.output()
def batch_energy(calc, which="auto"):
    text = calc.output()
    ends = [m.end() for m in re.finditer(ENERGY_PATTERNS["final"], text)]
    last_scf = text[ends[-2]:] if len(ends) > 1 else text
    found = {}
    for key, pattern in ENERGY_PATTERNS.items():
        hits = re.findall(pattern, text)
        try:
            found[key] = float(hits[-1])
        except (IndexError, ValueError):
            pass
    if which == "auto":
        which = "0K" if "0K" in found else "final"
    return found.get(which), which, bool(ends) and not any(s in last_scf for s in SCF_FAILED)


# A summary of one batch job, for batch_results.json and the workflow scripts.
# Its status ('done'; 'failed', with the reason; 'running' once submitted; 'pending' before), the
# energy (batch_energy) and whether the last SCF and a relaxation converged, the number of geometry
# steps, the net charge and final integrated spin (hbar/2) CASTEP reports, and the job's size,
# k-points, cores, job id and restarts.
#   Called by: run_batch()
#   Calls:     Calc.done(), Calc.info(), Calc.output(), batch_energy(), castep_charge()
def job_result(calc, which="auto"):
    info, text = calc.info(), calc.output()
    status = ("done" if calc.done() else "failed" if info.get("failed") else
              "running" if info.get("jobid") else "pending")
    energy, key, scf_ok = batch_energy(calc, which) if text else (None, which, False)
    spins = SPIN_DENSITY.findall(text)
    try:
        spin = float(spins[-1])
    except (IndexError, ValueError):
        spin = None
    return {"name": calc.name, "status": status, "reason": info.get("failed"),
            "energy": energy, "energy_kind": key, "scf_converged": scf_ok,
            "geom_converged": (GEOM_DONE in text) if calc.kind == "geom" else None,
            "geom_steps": len(GEOM_STEP.findall(text)), "charge": castep_charge(calc),
            "spin": spin, "natoms": len(calc.atoms), "kind": calc.kind, "task": calc.task,
            "kgrid": info.get("kgrid"), "nkpts": info.get("nkpts"), "ncores": info.get("ncores"),
            "jobid": info.get("jobid"), "restarts": info.get("restarts", 0),
            "dir": str(calc.dir)}


# With tolerate_failures : true, records a failed job instead of stopping the script.
# The reason (the message the script would have stopped with) goes into job_info.json as 'failed';
# wait_for, prepare_all and run_batch then treat the job as finished, and job_result reports it.
# Rerun with --resubmit-failed to start it again.
#   Called by: wait_for()
def mark_failed(calc, err):
    msg = str(err.code if isinstance(err, SystemExit) else err).strip()
    msg = msg[len("ERROR: "):] if msg.startswith("ERROR: ") else msg
    info = calc.info()
    info["failed"] = msg or "failed"
    calc.save(info)
    log(f"FAILED: {calc.name}: {info['failed'].splitlines()[0]} (tolerate_failures : true, so the "
        "others carry on)")


# The fingerprint of one batch job's inputs: its structure (species, positions, cell, starting
# spins), kind, task, k-points, cell options, .param keywords and the .param, .cell and k-point
# spacing it starts from (base). run_batch keeps it in job_info.json, so it can tell when a job's
# inputs have changed since it was set up.
#   Called by: run_batch()
#   Calls:     param_value()
def job_spec(job, base):
    a = job["atoms"]
    frac = a.get_scaled_positions(wrap=True)
    frac[np.isclose(frac, 1.0, atol=1e-7)] = 0.0
    data = {"symbols": a.get_chemical_symbols(), "frac": (np.round(frac, 6) + 0.0).tolist(),
            "cell": (np.round(np.asarray(a.cell), 6) + 0.0).tolist(),
            "spins": (np.round(a.get_initial_magnetic_moments(), 4) + 0.0).tolist(),
            "base": base, "overrides": {str(k).lower(): param_value(v)
                                        for k, v in (job.get("overrides") or {}).items()}}
    for key in ("kind", "task", "kgrid", "kpoint_offset", "fix_cell", "symmetry", "cell_extra",
                "param_text"):
        data[key] = job.get(key)
    return hashlib.md5(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


# Runs a list of CASTEP calculations of any kind and returns what they gave: batch mode, used by
# the workflow and by --batch.
# jobs is a list of dicts, one per calculation:
#   name           unique; its directory and seedname (letters, digits, - _ . only)
#   atoms          the structure (ASE Atoms; initial magnetic moments are written as SPIN=)
#   kind           'geom' (a relaxation: carried on from its .check when it runs out of time) or
#                  'sp' (anything else: a single point, phonon+efield, ...); default 'sp'
#   task           CASTEP's task, if not geometryoptimization (geom) or singlepoint (sp)
#   overrides      {keyword: value} to set in its .param (None removes it), e.g. charge, spin
#   kgrid          a fixed MP grid such as [1, 1, 1]; otherwise from spacing
#   kpoint_offset  'gamma' or three numbers
#   fix_cell       False for a variable-cell relaxation (default True: fix_all_cell)
#   symmetry       True to keep symmetry_generate etc. from the .cell (default False)
#   cell_extra     text added to the .cell (a block or keyword in it replaces your .cell's)
#   param_text     a .param of its own instead of param_text / geom_param_text
# The other arguments: workdir (one directory per job is made in it), args (the runner settings,
# from batch_args or read_input), cell_text (the .cell every job copies species_pot and the rest
# from), param_text and geom_param_text (the .param for 'sp' and 'geom' jobs; the task is set per
# job), spacing (k-point spacing, 1/Ang, for jobs without a kgrid), energy (as the energy keyword),
# share_dryruns (one dry run for all jobs that would give the same one: dryrun_key), stop_file
# (default <workdir>/STOP) and label (what the count in the log is of: default 'batch <workdir
# name>'). While it waits, <workdir>/progress.txt lists the jobs as they finish, under their count
# (write_progress), and the log has the count every PROGRESS_EVERY seconds.
# It works as the size series does: inputs, dry runs and plans first (prepare_all, at most max_jobs
# dry-run jobs queued at once; with dryrun_pack : N, N dry runs to a job), then the jobs, submitted
# at most max_jobs at a time, each planned again just before it goes (so it uses the timings and
# memory measured so far), waited for (wait_for), restarted as usual - but a restart (restart,
# from wait_for) is only prepared there and held (restart_ready), and goes from here, before any
# new job, as a job of its own when max_jobs leaves a place: a pack or array task that ran out of
# time restarts one job a calculation, and max_jobs holds for those too. With pack : N, jobs planned on
# a share of a node go N at a time (or fewer, as evenly as the jobs left allow, each then getting
# more of the node) in one job (submit_pack); one left on its own is planned again for whole nodes.
# With charging : core, up to N jobs on part of a node go in one job, the first in the queue and
# those after it on the same partition, as long as one node holds their cores. With array : N, the
# jobs waiting go as Slurm job arrays of up to N tasks (submit_array), those planned alike
# (array_key) in one array; max_jobs still counts each task as a job, and the queue is topped up
# once array_min places are free (default: a quarter of array, or of max_jobs if that is smaller;
# at most that array or max_jobs, so a step whose own runner block lowers max_jobs still tops up),
# or all the jobs left fit. With array and pack : N both, each task is a pack: jobs planned to
# share a node that run alike in a pack (pack_parts) go N to a task (with charging : core, as many
# as a node's cores hold) in job arrays of packs (submit_pack_array), each task counting once in
# max_jobs; those left over, fewer than N, go as a pack of their own, and a group too small for
# two full packs as packs (send_pack). With
# tolerate_failures, a job that fails is marked failed (mark_failed) and the others carry on;
# --resubmit-failed starts failed jobs again from the beginning. A job whose inputs changed since it
# was set up (job_spec) is set up again if it never ran; one that has run stops the script, unless
# --force (which keeps its results). Rerunning picks up where it left off. Returns {'complete':
# True/False, 'jobs': {name: job_result}, 'calcs': {name: Calc}}, and writes the results to
# <workdir>/batch_results.json; 'complete' is False after --prepare-only, or with --no-wait while
# jobs are still queued.
#   Called by: batch_main(), the workflow
#   Calls:     Calc(), Calc.done(), Calc.err_files(), array_castep_line(), array_key(), calibrate(),
#              check_templates(), even_groups(), finalize(), job_result(), job_spec(),
#              measure_times(), pack_castep_line(), pack_parts(), parse_castep_input(),
#              prepare_all(), remove_check(), submit(), submit_array(), submit_pack(),
#              submit_pack_array(), wait_for(), write_progress()
def run_batch(workdir, jobs, args, cell_text, param_text, geom_param_text=None, spacing=None,
              energy="auto", share_dryruns=True, stop_file=None, label=None):
    workdir = Path(workdir)
    names = [job["name"] for job in jobs]
    twice = sorted({n for n in names if names.count(n) > 1})
    if twice:
        die("batch jobs with the same name: " + ", ".join(twice))
    bad = [n for n in names if not re.fullmatch(r"[\w.\-]+", n)]
    if bad:
        die("batch job names may only have letters, digits and - _ . : " + ", ".join(bad))
    job_template, dryrun_template = check_templates(args)
    workdir.mkdir(parents=True, exist_ok=True)
    keep_items = [it for it in parse_castep_input(cell_text)
                  if not (it["type"] == "block" and it["name"] in STRUCT_BLOCKS | DROP_BLOCKS)
                  and not (it["type"] == "kw" and it["key"] in SET_KEYS)]
    ctx = {"spacing": spacing, "keep_items": keep_items,
           "defect_keep_items": [it for it in keep_items
                                 if not (it["type"] == "kw" and it["key"] in SYMMETRY_KEYS)],
           "job_template": job_template, "dryrun_template": dryrun_template,
           "param_items": parse_castep_input(param_text),
           "geom_param_items": parse_castep_input(geom_param_text or param_text),
           "calcs": [], "share_dryruns": share_dryruns, "batch": True}
    if args.pack > 1:
        pack_castep_line(job_template)
    if getattr(args, "array", 1) > 1:
        array_castep_line(job_template)
    base = "\n".join([cell_text, param_text, geom_param_text or "", repr(spacing)])
    calcs, keys = {}, []
    for i, job in enumerate(jobs):
        kind = job.get("kind", "sp")
        if kind not in ("geom", "sp"):
            die(f"batch job {job['name']}: kind must be geom or sp, not {kind}")
        if not job.get("kgrid") and not spacing:
            die(f"batch job {job['name']}: give it a kgrid, or the batch a k-point spacing")
        c = Calc(workdir, job["name"], kind, job["atoms"], name=job["name"], task=job.get("task"),
                 overrides=job.get("overrides"), fixed_kgrid=job.get("kgrid"),
                 kpoint_offset=job.get("kpoint_offset"), fix_cell=job.get("fix_cell", True),
                 symmetry=job.get("symmetry", False), cell_extra=job.get("cell_extra"),
                 param_items=parse_castep_input(job["param_text"]) if job.get("param_text")
                 else None, group=(kind, job.get("task")))
        spec, info = job_spec(job, base), c.info()
        if info and info.get("spec") != spec:
            if (info.get("jobid") or c.done()) and not info.get("failed"):
                if not args.force:
                    die(f"the inputs of batch job {c.name} have changed since it ran ({c.dir}). "
                        "Delete that directory to run it again with the new inputs, or rerun "
                        "with --force to keep its results.")
                log(f"WARNING: the inputs of {c.name} have changed since it ran; keeping its "
                    "results (--force)")
                spec = info.get("spec")
            else:                          # never ran, or failed: set it up again
                shutil.rmtree(c.dir)
                info = {}
        if info.get("spec") != spec:
            c.dir.mkdir(parents=True, exist_ok=True)
            info["spec"] = spec
            c.save(info)
        calcs[(len(c.atoms), i)] = {"job": c}
        keys.append((len(c.atoms), i))
        ctx["calcs"].append(c)
    keys.sort()
    todo = ctx["calcs"]
    args.ctx, args.all = ctx, (calcs, keys)        # for restarts, which plan jobs again
    args.stop_file = Path(stop_file) if stop_file else workdir / "STOP"
    if args.stop_file.exists():
        args.stop_file.unlink()                    # left over from an earlier run

    # Writes batch_results.json and returns the results.
    def results(complete):
        res = {c.name: job_result(c, energy) for c in todo}
        (workdir / BATCH_RESULTS).write_text(json.dumps({"complete": complete, "jobs": res},
                                                        indent=1))
        if complete:
            failed = sum(r["status"] == "failed" for r in res.values())
            log(f"batch {workdir.name}: {len(res) - failed} job(s) done"
                + (f", {failed} failed" if failed else ""))
        return {"complete": complete, "jobs": res, "calcs": {c.name: c for c in todo}}

    # Finished, or marked failed (tolerate_failures).
    def finished(c):
        return c.done() or bool(c.peek().get("failed"))

    # Keeps <workdir>/progress.txt up to date (write_progress): a line for each job that has
    # finished, or failed (tolerate_failures), since it was last written, e.g. '[14:02:11]
    # 1,201/4,980  d555_q-1: done', under the count; and while jobs are queued or running, the count
    # in the log every PROGRESS_EVERY seconds. It looks only at what has been read already
    # (Calc.known_done, job_info.json in memory), so it reads no .castep itself. A job the file
    # lists already (an earlier run's) isn't listed again, unless it has gone from failed to done.
    pfile, name = workdir / "progress.txt", label or f"batch {workdir.name}"
    listed, shown = {}, {"done": None, "time": 0.0}
    try:
        for line in pfile.read_text(errors="replace").splitlines()[2:]:
            m = re.match(r"\[[\d:]{8}\] [\d,]+/[\d,]+  (.+?): (done|FAILED)", line)
            if m:
                listed[m.group(1)] = m.group(2)
    except OSError:
        pass

    def progress():
        state = {c.name: "done" if c.known_done() else "FAILED" if c.peek().get("failed")
                 else None for c in todo}
        done = sum(s == "done" for s in state.values())
        failed = sum(s == "FAILED" for s in state.values())
        new = [c for c in todo if state[c.name] and listed.get(c.name) != state[c.name]]
        if new or not pfile.exists():
            now, lines = time.strftime("%H:%M:%S"), []
            count = done - sum(state[c.name] == "done" for c in new)
            for c in new:
                if state[c.name] == "done":
                    count += 1
                    r = c.peek().get("restarts", 0)
                    what = "done" + (f" ({r} restart{'s' if r > 1 else ''})" if r else "")
                else:
                    what = "FAILED: " + " ".join(str(c.peek()["failed"]).split())[:250]
                lines.append(f"[{now}] {count:,}/{len(todo):,}  {c.name}: {what}")
                listed[c.name] = state[c.name]
            write_progress(pfile, f"{done:,}/{len(todo):,} calculations done"
                           + (f" ({failed:,} failed)" if failed else "")
                           + (f", the last at {now}" if new else ""), lines)
        busy = sum(1 for c in todo if not state[c.name] and c.peek().get("jobid"))
        ready = sum(1 for c in todo if not state[c.name] and c.peek().get("restart_ready"))
        if busy and done != shown["done"] and time.time() - shown["time"] >= PROGRESS_EVERY:
            log(f"{name}: {done:,}/{len(todo):,} CASTEP calculations done "
                f"({100 * done / len(todo):.0f}%)" + (f", {failed:,} failed" if failed else "")
                + f"; {busy:,} queued or running"
                + (f", {ready:,} restarts waiting for a place (max_jobs)" if ready else "")
                + f" (each listed in {pfile.name} as it finishes)")
            shown.update(done=done, time=time.time())

    if args.resubmit_failed:                       # start failed jobs again from the beginning
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for c in todo:
            info = c.info()
            if info.get("failed") and not c.done():
                for f in [c.castep] + c.err_files():
                    if f.exists():
                        f.rename(f.with_name(f"{f.name}.failed-{stamp}"))
                for k in ("failed", "jobid", "submitted", "runs", "restarts", "timing",
                          "margin_factor", "no_continuation", "bad_checks"):
                    info.pop(k, None)
                c.save(info)
                log(f"{c.name} failed before - starting it again (--resubmit-failed)")
    calibrate(calcs, keys, args)                   # mem_replicated : auto, from finished jobs
    measure_times(calcs, keys)                     # walltime : auto, from finished jobs
    for c in todo:                                 # delete_check : true
        remove_check(c, args)
    unplanned = [c for c in todo if not finished(c) and not c.info().get("jobid")
                 and not c.info().get("restart_ready")]
    chunk = (args.max_jobs * args.dryrun_pack if args.max_jobs else len(unplanned)) or 1
    for k in range(0, len(unplanned), chunk):      # dry runs, at most max_jobs jobs at a time
        if not prepare_all(unplanned[k:k + chunk], ctx, args, show=False):
            return results(False)
    if args.prepare_only:
        log(f"inputs for {len(unplanned)} job(s) written under {workdir}/ - check them, then "
            "rerun without --prepare-only")
        return results(False)
    # Plans a calculation again with what is known now, on 1/pack of a node (default: the pack
    # setting); True if it was planned to share a node.
    def plan(c, pack=None):
        finalize(c, c.dryrun_file.read_text(errors="replace"), ctx, args, quiet=True, show=False,
                 pack=pack)
        return c.info().get("packed")

    # Sends one pack (pack : N) of jobs planned to share a node, as one job (submit_pack): with
    # fewer than N, each is planned again for more of the node (with charging : core, as they
    # are), unless that changes the partition. A job on its own goes as a job of its own, planned
    # again for the node to itself if it was planned to share one.
    def send_pack(group):
        if 1 < len(group) < args.pack and args.charging != "core":
            if not all(plan(d, len(group)) for d in group) or len(
                    {d.info()["partition"] for d in group}) > 1:
                for d in group:                    # (unless that changes the partition)
                    plan(d)
        if len(group) > 1:
            submit_pack(group, ctx, args, workdir)
        else:
            if group[0].info().get("packed"):      # nothing to share a node with: all of it
                plan(group[0], 1)
            submit(group[0], args)

    # How a job goes in a job array (array : N), planned now with what is known so far: (key, jobs
    # to a task); jobs with the same key go in one array. A job planned on whole nodes is a task
    # of its own, keyed by its job.sh (array_key). With pack : N a job planned to share a node
    # goes N to a task (with charging : core, as many as a node's cores hold, up to N; if that is
    # one, it is a task of its own), keyed by how it runs in a pack (pack_parts) and where - but
    # for its memory if no mem_per_core or partition gives the node's (submit_pack_array then
    # gives each job the most any of them needs).
    def array_slot(c):
        if plan(c):
            info = c.info()
            jobs, _, part = pack_parts([info], args)
            (own, env, mb), size = jobs[0], args.pack
            if args.charging == "core":
                size = min(args.pack, args.cores_per_node // info["charged_cores"])
            elif (part["mem_per_core"] if part else args.mem_per_core) is None:
                own, mb = own.rsplit(" --mem=", 1)[0], None
            if size > 1:
                return ("pack", info["partition"], own, env, mb), size
            plan(c, 1)
        return ("job", array_key(c)), 1

    # Sends jobs that array_slot keys alike: a job array of them (submit_array), or a job of its
    # own. With size (jobs to a task) above one, the full packs of size go as a job array of packs
    # (submit_pack_array) and the jobs left over as a pack of their own (send_pack); with fewer
    # than two full packs, all go as packs, as evenly as the jobs allow.
    def send_array(group, size):
        if size == 1:
            if len(group) > 1:
                submit_array(group, ctx, args, workdir)
            else:
                submit(group[0], args)
            return
        full = len(group) // size
        if full < 2:
            for pack in even_groups(group, size):
                send_pack(pack)
            return
        submit_pack_array([group[k:k + size] for k in range(0, full * size, size)], ctx, args,
                          workdir)
        if group[full * size:]:
            send_pack(group[full * size:])

    while True:
        calibrate(calcs, keys, args)
        measure_times(calcs, keys)
        active = [c for c in todo if c.peek().get("jobid") and not finished(c)]
        held = [c for c in todo if c.peek().get("restart_ready") and not finished(c)]
        queue = [c for c in todo if not c.peek().get("jobid") and not finished(c)
                 and not c.peek().get("restart_ready")]
        running = {c.peek()["jobid"] for c in active}          # jobs, not calculations
        while held and (not args.max_jobs or len(running) < args.max_jobs):
            c = held.pop(0)                    # restarts first: each a job of its own, planned
            info = c.info()                    # by restart (a pack's restart on its own)
            info.pop("restart_ready", None)
            c.save(info)
            submit(c, args)
            running.add(c.info()["jobid"])
            active.append(c)
        progress()
        if getattr(args, "array", 1) > 1:          # array : N - the queue as job arrays
            free = (args.max_jobs - len(running)) if args.max_jobs else len(queue)
            top = min(args.array, args.max_jobs or args.array)
            least = min(getattr(args, "array_min", None) or max(1, top // 4), top)
            while queue and free > 0 and (not active or free >= min(
                    math.ceil(len(queue) / args.pack), least)):   # (the tasks the queue needs)
                groups, tasks = {}, 0              # at most array tasks, and the places free
                while queue:
                    key, size = array_slot(queue[0])
                    group = groups.setdefault(key, (size, []))[1]
                    more = math.ceil((len(group) + 1) / size) - math.ceil(len(group) / size)
                    if tasks + more > min(free, args.array):
                        break
                    group.append(queue.pop(0))
                    tasks += more
                for size, group in groups.values():
                    send_array(group, size)
                    running.update(c.info()["jobid"] for c in group)
                    active += group
                free -= tasks
        while queue and getattr(args, "array", 1) == 1 and (not args.max_jobs
                                                           or len(running) < args.max_jobs):
            c = queue.pop(0)
            group = [c]
            if plan(c) and args.charging == "core":    # pack : N - others beside it, in its
                cores = c.info()["charged_cores"]          # job, as long as one node holds them
                for d in list(queue):
                    if len(group) == args.pack:
                        break
                    if (plan(d) and d.info()["partition"] == c.info()["partition"]
                            and cores + d.info()["charged_cores"] <= args.cores_per_node):
                        group.append(d)
                        queue.remove(d)
                        cores += d.info()["charged_cores"]
            elif c.info().get("packed"):           # pack : N - others to share its node, as
                left = 1 + len(queue)              # many in each of the packs still to come
                size = min(args.pack, math.ceil(left / math.ceil(left / args.pack)))
                for d in list(queue):
                    if len(group) == size:
                        break
                    if plan(d) and d.info()["partition"] == c.info()["partition"]:
                        group.append(d)
                        queue.remove(d)
            send_pack(group)                       # (fewer than pack: more of the node each)
            active += group
            running.add(c.info()["jobid"])
        if not active:
            return results(True)
        if not wait_for(active, args, any_done=bool(queue or held), progress=progress):
            return results(False)


# The runner settings for batch mode, from CASTEP-style items (parse_castep_input).
# Any of the .conv keywords that control how jobs run (jobscript, partitions, cores_per_node,
# max_nodes, walltime, max_restarts, delete_check, dryrun_*, submit_cmd, poll, max_jobs,
# tolerate_failures, pack, ...), converted and checked as read_input does (convert_settings,
# check_runner; with pack, the template's CASTEP line: pack_castep_line); the size-series keywords
# (defect, sizes, e_bulk, ...) aren't needed. File names are relative to base. Returns the
# settings namespace run_batch takes. With dryrun_pack, the dry-run template's CASTEP line is
# checked too.
#   Called by: read_batch_file(), the workflow
#   Calls:     check_runner(), convert_settings(), dryrun_template_text(), pack_castep_line()
def batch_args(items, base, name="batch settings"):
    vals, raw_vals, templates = convert_settings(items, name)
    args = SimpleNamespace(**{key: default for key, (_, default) in KEYWORDS.items()})
    args.__dict__.update(vals)
    args.input_path, args.raw = None, raw_vals
    if args.jobscript is None and "jobscript" not in templates:
        die(f"{name}: give the jobscript template, as 'jobscript : file' or %block jobscript")
    check_runner(args, raw_vals, name)

    # The path of a file named by a keyword, relative to base; stops if it doesn't exist.
    def find(key):
        p = Path(base) / getattr(args, key)
        if not p.is_file():
            die(f"{name}: the {key} file was not found: {p}")
        return p

    args.job_template = templates.get("jobscript") or find("jobscript").read_text()
    args.dryrun_template = templates.get("dryrun_jobscript") or (
        find("dryrun_jobscript").read_text() if args.dryrun_jobscript else None)
    if args.pack > 1:
        pack_castep_line(args.job_template, name)
    if args.dryrun_pack > 1 and args.dryrun_on == "compute":
        pack_castep_line(dryrun_template_text(args.job_template, args.dryrun_template), name,
                         DRYRUN_RUN, "dryrun_pack")
    return args


# Reads a structure file for a batch job: a CASTEP .cell with read_primitive (so SPIN= values are
# kept as starting spins), anything else with ASE.
#   Called by: read_batch_file()
#   Calls:     parse_castep_input(), read_primitive()
def read_structure(path):
    if path.suffix.lower() == ".cell":
        return read_primitive(parse_castep_input(path.read_text(errors="replace")))
    import ase.io
    try:
        return ase.io.read(str(path))
    except Exception as err:               # any format problem
        die(f"could not read the structure in {path}: {err}")


# Reads a --batch file: the runner settings (batch_args), param (and geom_param for the 'geom'
# jobs), cell (the .cell every job copies species_pot etc. from), kspacing, energy and workdir, and
# %block jobs with one job per line:  name  structure-file  [option=value ...]. Options:
# kind=geom|sp, task=..., fix_cell=true|false, symmetry=true|false, kgrid=a,b,c,
# offset=gamma|a,b,c; anything else is a .param keyword for that job (charge=-1, spin=2, ...).
# File names are relative to the batch file. Returns the settings with the jobs in
# args.batch_jobs.
#   Called by: get_args()
#   Calls:     batch_args(), block_rows(), _bool(), _choice(), canonical_key(), _ints(),
#              parse_castep_input(), read_structure()
def read_batch_file(path):
    name, base = path.name, path.parent
    rows, items = None, []
    for it in parse_castep_input(path.read_text()):
        if it["type"] == "block" and it["name"] == "jobs":
            rows = block_rows(it)
        else:
            items.append(it)
    if not rows:
        die(f"{name}: list the jobs in %block jobs, one per line:  name  structure-file  [options]")
    args = batch_args(items, base, name)
    for key in ("cell", "param"):
        if getattr(args, key) is None:
            die(f"{name}: '{key}' is required")
    for key in ("cell", "param", "geom_param"):
        if getattr(args, key):
            setattr(args, key, base / getattr(args, key))
            if not getattr(args, key).is_file():
                die(f"{name}: the {key} file was not found: {getattr(args, key)}")
    args.workdir = base / args.workdir if args.workdir else None
    args.input_path, jobs = path, []
    for row in rows:
        if len(row) < 2:
            die(f"{name}: each line of %block jobs is  name  structure-file  [options]")
        job = {"name": row[0], "atoms": read_structure(base / row[1]), "overrides": {}}
        for opt in row[2:]:
            key, eq, val = opt.partition("=")
            key = key.lower()
            try:
                if not eq or not val:
                    raise ValueError("expected option=value")
                if key == "kind":
                    job["kind"] = _choice("geom", "sp")(val)
                elif key == "task":
                    job["task"] = val
                elif key in ("fix_cell", "symmetry"):
                    job[key] = _bool(val)
                elif key == "kgrid":
                    job["kgrid"] = _ints(3)(val)
                elif key == "offset":
                    job["kpoint_offset"] = ("gamma" if val.lower() == "gamma"
                                            else [float(x) for x in val.split(",")])
                else:
                    job["overrides"][canonical_key(key)] = val
            except ValueError as err:
                die(f"{name}: job {row[0]}: bad option '{opt}': {err}")
        jobs.append(job)
    args.batch_jobs = jobs
    return args


# --batch: runs the jobs of a batch file with run_batch and prints a table of the results (also
# in batch_results.json in the work directory, batch_<file name> unless workdir is set).
#   Called by: series_main()
#   Calls:     run_batch()
def batch_main(args):
    print(f"\nSettings from {args.input_path} (everything else at its default):")
    for key, raw in args.raw.items():
        print(f"  {key:<17}: {raw}")
    workdir = args.workdir or args.input_path.parent / f"batch_{args.input_path.stem}"
    print(f"\n{len(args.batch_jobs)} job(s), in {workdir}/\n")
    if not (args.prepare_only or args.no_wait):
        log(f"running on {socket.gethostname()} as process {os.getpid()}; to stop it from any "
            f"login node: touch {(workdir / 'STOP').absolute()}")
    try:
        res = run_batch(workdir, args.batch_jobs, args, args.cell.read_text(),
                        args.param.read_text(),
                        args.geom_param.read_text() if args.geom_param else None,
                        spacing=args.kspacing, energy=args.energy)
    except KeyboardInterrupt:
        print("\nInterrupted. Submitted jobs keep running - rerun the same command to carry on.")
        return
    print(f"\n{'job':<24} {'status':<8} {'energy/eV':>16} {'SCF':>4} {'steps':>6} {'spin':>7} "
          f"{'charge':>7}")
    for r in res["jobs"].values():
        e = f"{r['energy']:16.6f}" if r["energy"] is not None else f"{'-':>16}"
        print(f"{r['name']:<24} {r['status']:<8} {e} {'ok' if r['scf_converged'] else 'no':>4} "
              f"{r['geom_steps']:>6} "
              + (f"{r['spin']:7.3f}" if r["spin"] is not None else f"{'-':>7}")
              + (f" {r['charge']:7g}" if r["charge"] is not None else f" {'-':>7}")
              + (f"   {r['reason'].splitlines()[0][:60]}" if r["reason"] else ""))
    print(f"\n(also in {workdir / BATCH_RESULTS})")


# --------------------------------------------------------------------------
# Input file
# --------------------------------------------------------------------------
INPUT_EXT = ".conv"

EXAMPLE_INPUT = """\
# a .conv file of dwf.py's job runner     run:  python3 dwf.py --conv <seedname>
#
# Written like a CASTEP .param/.cell file: "keyword : value" (or "=", or a space),
# %block <name> ... %endblock <name>, keywords in any case, # or ! starts a comment.
# Lines below that start with ! are optional settings shown with an example value:
# delete the ! to use one. Anything left out takes its default. File names are
# relative to this file.

# ------------------------------------------------------------------ input files
cell        : CaO_prim.cell       # relaxed primitive cell of the host
param       : CaO.param           # used for both stages; the script sets task
!geom_param : CaO_geom.param      # separate .param for the relaxations
jobscript   : job_template.sh     # template with {NCORES} and {SEED} (or %block jobscript)

# ------------------------------------------------------------------ defect
# X_Y = element X on a Y site, Vac_Y = Y vacancy; add :k for the k-th Y site of the
# primitive cell (e.g. Vac_O:2). The other components sit on their nearest sites to the
# first, and the defect as a whole as close to the cell centre as it can be. Add @s to
# give a substituted atom a starting spin of s (in hbar/2, i.e. unpaired electrons), e.g.
# Sb_Ca@3: it is written as SPIN=3 on that atom in the .cell. It is a starting guess
# only, not used for the pristine cells. Check "net spin of system" in the .castep: if
# CASTEP kept its own default, also set spin in the .param (used for the defect runs only).
%block defect
Sb_Ca
Vac_O
%endblock defect

# ------------------------------------------------------------------ convergence
e_bulk    : -715.123456 eV        # bulk energy per atom (same settings as the .param)
tol       : 0.05 eV               # stop when E_f changes by less than this
sizes     : 2 3 4 5               # supercell sizes n to try (smallest first)
bulk_mode : peratom               # peratom: E_host = N*e_bulk; supercell: also run
                                  #   each pristine supercell (e_bulk then optional)
energy    : auto                  # auto (0K estimate if printed) | final | free | 0K
singlepoint : false               # false: energy from the end of each relaxation (fine as
                                  #   the cell is fixed); true: also run a single point on
                                  #   it (always done when geom_param is set)
!workdir  : conv_CaO_SbCa         # default: conv_<cell name>_<defect>

# Charged defects: set charge in the .param AND the geom_param (it is removed for the
# pristine cells). E_f* then includes an image-charge correction so it converges with
# size: lz = Lany-Zunger (default), mp = first-order Makov-Payne, none = off.
!charge_correction : lz
!dielectric        : 11.95        # static dielectric constant (electronic + ionic, e.g.
                                  #   from DFPT); needed for a charged defect

# Supercell shape: size n uses n*T. Leave it out for n x n x n copies of the
# primitive cell; this T gives cubic supercells of an FCC primitive cell.
!%block transform
!  -1  1  1
!   1 -1  1
!   1  1 -1
!%endblock transform

# k-points: the spacing comes from kpoint_mp_grid in the .cell unless set here
!kgrid    : 8 8 8                 # primitive-cell MP grid to take the spacing from
!kspacing : 0.05 1/ang            # or the spacing itself (CASTEP's 2*pi units)

# ------------------------------------------------------------------ resources
max_cores      : 32               # most cores per job (a partition's own limit applies too)
!cores         : 16               # use exactly this many cores for every job instead
mem_safety     : 1.05             # safety factor on the memory estimate
mem_replicated : auto             # fraction of CASTEP's memory copied on every process:
                                  #   auto = measured from finished jobs (0.25 until then)
!baseline_mb   : 500 MB           # replace CASTEP's 'baseline' memory estimate

# Machines that hand out whole nodes (e.g. ARCHER2): give the cores per node and use
# {NODES} {TASKS_PER_NODE} {CPUS_PER_TASK} in the jobscript instead of {NCORES}. The
# memory per core of a partition is then that of a full node (node memory / cores).
!cores_per_node : 128
!max_nodes      : 2               # most nodes per job (instead of max_cores)
!omp_threads    : auto            # OpenMP threads per MPI process, on the cores a job leaves
                                  #   idle when it uses fewer processes per node to fit the
                                  #   memory: auto = all of them (up to 4), N = up to N,
                                  #   1 = off (default). Sets OMP_NUM_THREADS in the jobscript
# Machines that share their nodes and charge by the core (e.g. Cirrus, 288 cores per
# node): add charging : core, and {CPUS_PER_TASK} in the jobscript. Jobs then get up to
# max_cores processes (as above) and ask for just the cores they need on one node, which
# they share. The memory comes with the cores (a partition's GB per core), so a process
# that needs more memory gets more cores. A job bigger than a node takes whole nodes
# (--exclusive, charged in full), up to max_nodes (default: as many as max_cores needs).
!charging       : core            # node (default with cores_per_node) | core

# Partitions in order of preference: each job goes on the first one with enough
# memory per core, so here himem is only used when 8 GB/core on multicore is too
# little. Each line:  name  GB_per_core  [max_cores]  [extra #SBATCH options]
# These are the CSF3 values; on himem memory is requested with --mem, up to 2 TB
# per node (62 GB/core on 32 cores). Leave the block out to keep the template's -p.
%block partitions
multicore   8    168
himem       62   32    --mem={MEM_GB}G
%endblock partitions
!mem_per_core  : 8 GB             # without %block partitions: warn if a job won't fit

# Walltime: the template's #SBATCH --time is the longest a job may run (a partition line
# can give its own, e.g. --time=48:00:00, and max_walltime replaces both). auto: ask for
# less once finished jobs show how long the next ones need; fixed: always the longest. A
# relaxation that runs out of time stops cleanly and is resubmitted to carry on from its
# .check file.
!max_walltime   : 12:00:00        # the longest a job may run, instead of the template's --time
walltime        : auto
walltime_safety : 1.5             # factor on the estimated time
!sp_walltime    : 4:00:00         # walltime for a single point (e.g. a pristine cell) that
                                  #   no finished job can estimate yet, instead of the full
                                  #   --time; if it runs out, it is resubmitted with the full
                                  #   --time (this counts towards max_restarts)
max_restarts    : 10              # resubmissions of a job that runs out of time (0: none)
delete_check    : false           # true: delete each calculation's .check file once it has
                                  #   finished successfully (they can be tens of GB), and
                                  #   any CASTEP can't read

# ------------------------------------------------------------------ MACE pre-relaxation
# Relax each defect cell with a MACE foundation model (positions only, cell fixed) before
# CASTEP relaxes it, so CASTEP starts close to the minimum and needs fewer steps. Runs
# here, where this script runs; needs  pip install mace-torch  (Python 3.9+).
prerelax          : none          # none | mace
!prerelax_model   : medium-mpa-0  # a mace_mp model (e.g. matpes-pbe), or a .model file
!prerelax_fmax    : 0.05 eV/ang   # stop when the largest force is below this
!prerelax_steps   : 500           # most optimiser steps
!prerelax_rattle  : 0.02 ang      # random displacements first, to break the symmetry
!prerelax_device  : cpu           # cpu | cuda
!prerelax_threads : 8             # CPU threads (be kind on a shared login node)

# ------------------------------------------------------------------ dry runs
dryrun_on        : compute        # compute (submitted as jobs) | login (run here)
dryrun_partition : serial         # replaces #SBATCH -p in the dry-run jobs
dryrun_time      : 0-1            # replaces #SBATCH -t in the dry-run jobs
dryrun_cores     : 1              # cores requested; CASTEP always runs as 1 process
dryrun_mem_gb    : 4 GB
!dryrun_options  : --qos=serial --mem={MEM_GB}G   # any other #SBATCH settings for dry runs
!dryrun_jobscript : dryrun_template.sh   # own template for dry runs (or %block)
!dryrun_cmd      : castep.mpi --dryrun {seed}   # the command for dryrun_on : login
!dryrun_timeout  : 900            # seconds, dryrun_on : login only
!dryrun_pack     : 32             # batch mode only: up to 32 dry runs side by side in one job

# ------------------------------------------------------------------ scheduler
submit_cmd  : sbatch
!status_cmd : squeue -h -j {jobid} -o '%i %T'   # default: chosen from submit_cmd
poll        : 120                 # seconds between queue checks
!max_jobs          : 16           # batch mode only: most jobs in the queue at once
!tolerate_failures : true         # batch mode only: a failed job is marked, the rest carry on
!pack              : 4            # batch mode only: up to 4 jobs side by side on one node

# Run-control switches are normally given on the command line (--prepare-only,
# --no-wait, --resubmit-failed, --force) but can also be set here:
!no_wait : true

# The jobscript template can go in here instead of a separate file. Lines inside
# the block are copied exactly as written, #SBATCH lines included:
!%block jobscript
!#!/bin/bash --login
!#SBATCH -p multicore
!#SBATCH -n {NCORES}
!#SBATCH -J {JOBNAME}
!#SBATCH -t 2-0
!module load <your castep module>
!mpirun -n $SLURM_NTASKS castep.mpi {SEED}
!%endblock jobscript
"""


# KEYWORDS converter for a non-empty text value (file names, model names, commands).
# Like every converter it returns the value or raises ValueError, which read_input turns into an
# error message naming the keyword.
#   Called by: convert_settings (through the KEYWORDS table)
def _text(v):
    if not v:
        raise ValueError("expected a value")
    return v


# KEYWORDS converter for true/false (also yes/no, on/off, t/f, 1/0).
# A keyword given with no value means true.
#   Called by: convert_settings (through the KEYWORDS table), read_batch_file()
def _bool(v):
    v = v.lower()
    if v in ("", "true", "t", "yes", "on", "1"):        # a bare keyword means true
        return True
    if v in ("false", "f", "no", "off", "0"):
        return False
    raise ValueError("expected true or false")


# Makes a KEYWORDS converter for a number, optionally followed by a unit, e.g. '0.05 eV'.
# The unit is only checked, not converted. cast=int for whole numbers.
#   Called by: the KEYWORDS table; convert_settings calls the converter it returns
def _num(*units, cast=float):
    """A number, optionally followed by one of the given units (e.g. 'eV')."""
    # The converter itself: checks the unit and converts the number.
    def conv(v):
        parts = v.split()
        try:
            if len(parts) == 2 and parts[1].lower() in [u.lower() for u in units]:
                parts = parts[:1]
            if len(parts) != 1:
                raise ValueError
            return cast(parts[0])
        except ValueError:
            raise ValueError(("expected a whole number" if cast is int else "expected a number")
                             + (f" (in {units[0]})" if units else "")) from None
    return conv


# Makes a KEYWORDS converter for a list of whole numbers (sizes, kgrid, transform).
# Optionally exactly n of them. Commas or spaces between them.
#   Called by: the KEYWORDS table; convert_settings calls the converter it returns,
#              read_batch_file()
def _ints(n=None):
    # The converter itself.
    def conv(v):
        try:
            vals = [int(x) for x in re.split(r"[\s,]+", v) if x]
        except ValueError:
            vals = []
        if not vals or (n and len(vals) != n):
            raise ValueError(f"expected {n or 'one or more'} whole numbers")
        return vals
    return conv


# KEYWORDS converter for a list of words (the defect tokens).
#   Called by: convert_settings (through the KEYWORDS table)
def _words(v):
    vals = [x for x in re.split(r"[\s,]+", v) if x]
    if not vals:
        raise ValueError("expected at least one entry")
    return vals


# KEYWORDS converter for omp_threads: 'auto' or a whole number of threads, at least 1.
#   Called by: convert_settings (through the KEYWORDS table)
def _omp_threads(v):
    if v.lower() == "auto":
        return "auto"
    try:
        n = int(v)
    except ValueError:
        n = 0
    if n < 1:
        raise ValueError("expected auto or a whole number of threads, at least 1")
    return n


# KEYWORDS converter for a Slurm time such as 4:00:00 (sp_walltime, max_walltime).
# Checks it with slurm_seconds and keeps the text.
#   Called by: convert_settings (through the KEYWORDS table)
#   Calls:     slurm_seconds()
def _slurm_time(v):
    if slurm_seconds(v) is None:
        raise ValueError("expected a Slurm time, e.g. 4:00:00 or 1-00:00:00")
    return v.strip()


# Makes a KEYWORDS converter that accepts one of a fixed set of words, in any case.
# Returns the word as written in the KEYWORDS table, e.g. 'auto'.
#   Called by: the KEYWORDS table; convert_settings calls the converter it returns,
#              read_batch_file()
def _choice(*options):
    # The converter itself.
    def conv(v):
        for o in options:
            if v.lower() == o.lower():
                return o
        raise ValueError("expected one of " + ", ".join(options))
    return conv


# KEYWORDS converter for mem_replicated: 'auto' or a number from 0 to 1.
#   Called by: convert_settings (through the KEYWORDS table)
def _replicated(v):
    if v.lower() == "auto":
        return "auto"
    try:
        r = float(v)
    except ValueError:
        r = -1.0
    if not 0 <= r <= 1:
        raise ValueError("expected auto or a number between 0 and 1")
    return r


# KEYWORDS converter for dryrun_options: #SBATCH options, grouped by group_options.
#   Called by: convert_settings (through the KEYWORDS table)
#   Calls:     group_options()
def _options(v):
    tokens = v.split()
    if not tokens or not tokens[0].startswith("-"):
        raise ValueError("expected #SBATCH options, e.g. --qos=serial --mem=4G")
    return group_options(tokens)


# keyword: (converter, default)
KEYWORDS = {
    "cell": (_text, None), "param": (_text, None), "geom_param": (_text, None),
    "jobscript": (_text, None), "defect": (_words, None),
    "e_bulk": (_num("eV", "eV/atom"), None), "tol": (_num("eV"), 0.05),
    "charge_correction": (_choice("lz", "mp", "none"), "lz"), "dielectric": (_num(), None),
    "sizes": (_ints(), [2, 3, 4, 5]), "transform": (_ints(9), None),
    "kgrid": (_ints(3), None), "kspacing": (_num("1/ang"), None),
    "energy": (_choice("auto", "final", "free", "0K"), "auto"),
    "bulk_mode": (_choice("peratom", "supercell"), "peratom"), "workdir": (_text, None),
    "singlepoint": (_bool, True),
    "max_cores": (_num(cast=int), 32), "cores": (_num(cast=int), None),
    "cores_per_node": (_num(cast=int), None), "max_nodes": (_num(cast=int), 1),
    "charging": (_choice("node", "core"), "node"),
    "mem_per_core": (_num("GB"), None), "mem_safety": (_num(), 1.05),
    "mem_replicated": (_replicated, "auto"), "baseline_mb": (_num("MB"), None),
    "omp_threads": (_omp_threads, 1),
    "walltime": (_choice("auto", "fixed"), "auto"), "walltime_safety": (_num(), 1.5),
    "sp_walltime": (_slurm_time, None), "max_walltime": (_slurm_time, None),
    "max_restarts": (_num(cast=int), 10), "delete_check": (_bool, False),
    "prerelax": (_choice("none", "mace"), "none"), "prerelax_model": (_text, "medium-mpa-0"),
    "prerelax_fmax": (_num("eV/ang", "eV/A"), 0.05), "prerelax_steps": (_num(cast=int), 500),
    "prerelax_rattle": (_num("ang", "A"), 0.0), "prerelax_device": (_text, "cpu"),
    "prerelax_threads": (_num(cast=int), 8),
    "dryrun_on": (_choice("compute", "login"), "compute"), "dryrun_jobscript": (_text, None),
    "dryrun_partition": (_text, None), "dryrun_time": (_text, None),
    "dryrun_cores": (_num(cast=int), 1), "dryrun_mem_gb": (_num("GB"), 4.0),
    "dryrun_options": (_options, []),
    "dryrun_cmd": (_text, "castep.mpi --dryrun {seed}"),
    "dryrun_timeout": (_num("s", cast=int), 900),
    "submit_cmd": (_text, "sbatch"), "status_cmd": (_text, None),
    "poll": (_num("s", cast=int), 120),
    "max_jobs": (_num(cast=int), None), "tolerate_failures": (_bool, False),    # batch mode
    "pack": (_num(cast=int), 1), "dryrun_pack": (_num(cast=int), 1),           # batch mode
    "array": (_num(cast=int), 1), "array_min": (_num(cast=int), None),          # batch mode
    "prepare_only": (_bool, False), "no_wait": (_bool, False),
    "resubmit_failed": (_bool, False), "force": (_bool, False),
    "partitions": (None, None),                               # %block only
}
LIST_BLOCKS = {"defect", "sizes", "transform", "kgrid"}      # block or keyword
TEMPLATE_BLOCKS = {"jobscript", "dryrun_jobscript"}         # file name, or inline block


# Reads %block partitions into the list of partitions choose_resources tries in order.
# One partition per line: name, GB per core, optional maximum cores, and optional extra #SBATCH
# options (grouped by group_options; they may use the template placeholders). Checks the values and
# that no partition is listed twice.
#   Called by: convert_settings()
#   Calls:     group_options()
def parse_partitions(rows, name):
    """%block partitions, one line per partition:
         name  GB_per_core  [max_cores]  [extra #SBATCH options for that partition]"""
    usage = (f"{name}: each line of %block partitions is  name  GB_per_core  [max_cores]  "
             "[extra #SBATCH options], e.g.  himem  62  32  --mem={MEM_GB}G")
    parts = []
    for row in rows:
        m = re.fullmatch(r"(\d+(?:\.\d*)?|\.\d+)(?:GB?)?", row[1], re.I) if len(row) > 1 else None
        if not m:
            die(usage)
        pname, mem, toks = row[0], float(m.group(1)), row[2:]
        if toks and toks[0].lower() in ("g", "gb"):
            toks = toks[1:]
        max_cores = int(toks.pop(0)) if toks and toks[0].isdigit() else None
        if toks and not toks[0].startswith("-"):
            die(f"{usage}\n  ('{toks[0]}' on the {pname} line is not an #SBATCH option)")
        if mem <= 0 or max_cores == 0:
            die(f"{name}: the {pname} partition needs memory and cores greater than 0")
        parts.append({"name": pname, "mem_per_core": mem, "max_cores": max_cores,
                      "options": group_options(toks)})
    if not parts:
        die(f"{name}: %block partitions is empty")
    if len({p["name"] for p in parts}) < len(parts):
        die(f"{name}: a partition is listed twice in %block partitions")
    return parts


# Converts the items of a CASTEP-style settings file (parse_castep_input) into values.
# Checks every keyword against KEYWORDS (suggesting the closest name for a typo) and converts each
# value with its converter; reads the blocks (defect, sizes, transform, kgrid, partitions, and an
# inline jobscript). Returns (values, the text each was given as, inline templates). name is the
# file name, for error messages.
#   Called by: batch_args(), read_input()
#   Calls:     block_rows(), parse_partitions()
def convert_settings(items, name):
    vals, raw_vals, templates = {}, {}, {}
    for it in items:
        if it["type"] == "raw":
            continue
        key = it["name"] if it["type"] == "block" else it["key"]
        if key not in KEYWORDS:
            close = difflib.get_close_matches(key, list(KEYWORDS), n=1)
            die(f"{name}: unknown keyword '{key}'"
                + (f" - did you mean '{close[0]}'?" if close else "")
                + "  (python3 dwf.py --conv-example lists them all)")
        if key in raw_vals:
            die(f"{name}: '{key}' is given more than once")
        if it["type"] == "block":
            if key in TEMPLATE_BLOCKS:
                templates[key] = textwrap.dedent("\n".join(it["body"])).strip("\n") + "\n"
                raw_vals[key] = "(%block in the input file)"
                continue
            if key == "partitions":
                vals[key] = parse_partitions(block_rows(it), name)
                raw_vals[key] = " / ".join(" ".join(row) for row in block_rows(it))
                continue
            if key not in LIST_BLOCKS:
                die(f"{name}: {key} is a keyword, not a block - write '{key} : value'")
            raw = " ".join(" ".join(row) for row in block_rows(it))
        else:
            if key == "partitions":
                die(f"{name}: partitions is a block - write %block partitions, one line per "
                    "partition, then %endblock partitions")
            raw = it["value"]
            if len(raw) > 1 and raw[0] == raw[-1] and raw[0] in "'\"":
                raw = raw[1:-1]
        raw_vals[key] = raw
        try:
            vals[key] = KEYWORDS[key][0](raw)
        except ValueError as err:
            die(f"{name}: bad value for {key} ('{raw}'): {err}")
    return vals, raw_vals, templates


# Checks the settings that control how jobs run, for both the size series and batch mode.
# Numbers that must be at least 1, greater than 0 or not negative; cores_per_node with max_nodes
# (not max_cores, unless charging : core) and a cores value that splits evenly over the nodes;
# charging : core with cores_per_node, the memory per core, a dry run that fits on a node (a
# dryrun_pack of more than a node holds runs them in turn), and max_nodes (by default the nodes
# max_cores or cores need) that holds the job; %block partitions without mem_per_core, and with a
# partition that can take cores; array (Slurm job arrays, of packs with pack too) with sbatch.
# Starts mem_replicated : auto at R_START.
#   Called by: batch_args(), read_input()
def check_runner(args, raw_vals, name):
    for key in ("max_cores", "cores", "dryrun_cores", "poll", "dryrun_timeout",
                "cores_per_node", "max_nodes", "prerelax_steps", "prerelax_threads", "max_jobs",
                "pack", "dryrun_pack", "array", "array_min"):
        if getattr(args, key) is not None and getattr(args, key) < 1:
            die(f"{name}: {key} must be at least 1")
    if args.charging == "core":              # nodes shared out, charged by the core
        if not args.cores_per_node:
            die(f"{name}: charging : core needs cores_per_node, the cores in a node (a job up "
                "to that size shares a node; a bigger one takes whole nodes)")
        if args.partitions is None and args.mem_per_core is None:
            die(f"{name}: with charging : core the memory comes with the cores, so give how "
                "much each core brings: mem_per_core, or a %block partitions")
        cpus = dryrun_cpus(1, args.dryrun_mem_gb, args)       # each dry run's cores
        if cpus > args.cores_per_node:
            die(f"{name}: a dry run with dryrun_mem_gb : {args.dryrun_mem_gb:g} needs {cpus} "
                f"cores for its memory, more than a node's {args.cores_per_node}")
        want = args.cores or args.max_cores
        nodes = math.ceil(want / args.cores_per_node)
        if "max_nodes" not in raw_vals:
            args.max_nodes = nodes
        elif args.max_nodes < nodes:
            die(f"{name}: {'cores' if args.cores else 'max_cores'} : {want} needs {nodes} nodes "
                f"of {args.cores_per_node} cores, more than max_nodes : {args.max_nodes}")
    if args.pack > 1:                        # several jobs side by side on one node
        if not args.cores_per_node:
            die(f"{name}: pack shares out whole nodes, so it needs cores_per_node")
        if args.pack > args.cores_per_node:
            die(f"{name}: pack : {args.pack} is more jobs than a node has cores "
                f"({args.cores_per_node})")
        if (args.cores and args.charging != "core"
                and args.cores > args.cores_per_node // args.pack):
            die(f"{name}: with pack : {args.pack} each job has {args.cores_per_node // args.pack} "
                f"cores, so cores : {args.cores} is too many")
    if args.array > 1 and "sbatch" not in args.submit_cmd:     # Slurm job arrays (of packs,
        die(f"{name}: array : {args.array} runs jobs as Slurm job arrays (sbatch --array), "
            f"but submit_cmd is '{args.submit_cmd}'")             # with pack too)
    if args.cores_per_node:
        if "max_cores" in raw_vals and args.charging != "core":
            die(f"{name}: with cores_per_node the job size is set by max_nodes - remove max_cores "
                "(or set charging : core, on a machine that charges by the core)")
        if args.cores and not any(args.cores % k == 0 and args.cores // k <= args.cores_per_node
                                  for k in range(1, args.max_nodes + 1)):
            die(f"{name}: cores : {args.cores} can't be split evenly over at most "
                f"{args.max_nodes} node(s) of {args.cores_per_node} cores")
    elif "max_nodes" in raw_vals:
        die(f"{name}: max_nodes needs cores_per_node (the number of cores in a node)")
    for key in ("tol", "kspacing", "mem_per_core", "mem_safety", "dryrun_mem_gb",
                "walltime_safety", "prerelax_fmax", "dielectric"):
        if getattr(args, key) is not None and getattr(args, key) <= 0:
            die(f"{name}: {key} must be greater than 0")
    for key in ("max_restarts", "prerelax_rattle"):
        if getattr(args, key) < 0:
            die(f"{name}: {key} can't be negative")
    args.r_auto, args.peak_warned = args.mem_replicated == "auto", False
    if args.r_auto:
        args.mem_replicated = R_START
    if args.partitions:
        if args.mem_per_core is not None:
            die(f"{name}: with %block partitions the memory per core is given for each "
                "partition - remove mem_per_core")
        if args.cores and all(p["max_cores"] and args.cores > p["max_cores"]
                              for p in args.partitions):
            die(f"{name}: cores : {args.cores} is more than any partition allows")


# Reads the .conv input file into the settings namespace (args).
# Parses it like a CASTEP file (parse_castep_input) and converts it (convert_settings), fills in
# defaults, checks what the size series needs (cell, param, defect, e_bulk for bulk_mode : peratom,
# sizes) and the runner settings (check_runner), and resolves file names relative to the input file.
#   Called by: get_args()
#   Calls:     check_runner(), convert_settings(), parse_castep_input()
def read_input(path):
    """Read the CASTEP-style input file into a namespace of settings."""
    name = path.name
    vals, raw_vals, templates = convert_settings(parse_castep_input(path.read_text()), name)
    args = SimpleNamespace(**{key: default for key, (_, default) in KEYWORDS.items()})
    args.__dict__.update(vals)
    args.input_path, args.raw = path, raw_vals
    base = path.parent

    # The path of a file named by a keyword, relative to the input file; stops if it doesn't exist.
    def find(key):
        p = base / getattr(args, key)
        if not p.is_file():
            die(f"{name}: the {key} file was not found: {p}")
        return p

    for key in ("cell", "param", "defect"):
        if getattr(args, key) is None:
            die(f"{name}: '{key}' is required")
    if args.jobscript is None and "jobscript" not in templates:
        die(f"{name}: give the jobscript template, as 'jobscript : file' or %block jobscript")
    if args.bulk_mode == "peratom" and args.e_bulk is None:
        die(f"{name}: e_bulk is required (or set bulk_mode : supercell)")
    if min(args.sizes) < 1:
        die(f"{name}: sizes must be positive whole numbers")
    check_runner(args, raw_vals, name)
    if args.kgrid and args.kspacing:
        die(f"{name}: give kgrid or kspacing, not both")
    args.cell, args.param = find("cell"), find("param")
    args.geom_param = find("geom_param") if args.geom_param else None
    args.job_template = templates.get("jobscript") or find("jobscript").read_text()
    args.dryrun_template = templates.get("dryrun_jobscript") or (
        find("dryrun_jobscript").read_text() if args.dryrun_jobscript else None)
    args.workdir = base / args.workdir if args.workdir else None
    return args


# The job runner's settings when it runs on its own, from dwf.py's command line (cl): --conv
# NAME, a size series (reads NAME.conv, or the file itself: read_input), or --batch FILE, a batch
# file (read_batch_file). --prepare-only, --no-wait, --resubmit-failed and --force override the
# same settings in the file. Returns the settings namespace that every other function of the
# runner receives as `args`.
#   Called by: main()
#   Calls:     read_batch_file(), read_input()
def runner_args(cl):
    if cl.batch:
        if not Path(cl.batch).is_file():
            die(f"no batch file {cl.batch}")
        args = read_batch_file(Path(cl.batch))
    else:
        path = Path(cl.conv)
        if not path.is_file() and Path(cl.conv + INPUT_EXT).is_file():
            path = Path(cl.conv + INPUT_EXT)
        if not path.is_file():
            die(f"no input file {cl.conv} or {cl.conv}{INPUT_EXT}")
        args = read_input(path)
    for flag in ("prepare_only", "no_wait", "resubmit_failed", "force"):
        if getattr(cl, flag):
            setattr(args, flag, True)
    return args


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
# The Madelung (Ewald) energy of one unit point charge per cell in a neutralising background,
# in units of e^2/(4 pi eps0) per Ang (so x COULOMB gives eV). It is the spurious interaction of a
# charged defect with its own periodic images in vacuum: -alpha_M / (2 L) with L = V^(1/3) (alpha_M
# = 2.8373 for a simple cubic cell). Ewald sum: real-space images + reciprocal-space terms - the
# self term - the background term, each converged to 1e-12; the split parameter eta only changes
# how the work is shared, not the result. Works for any cell shape.
#   Called by: charge_correction()
def madelung_energy(cell):
    """Ewald energy of a unit point charge per cell with a neutralising background
    (e^2/(4 pi eps0) = 1, lengths in Ang); negative."""
    A = np.asarray(cell, float)
    V = abs(np.linalg.det(A))
    B = 2 * np.pi * np.linalg.inv(A).T                      # reciprocal lattice (rows)
    eta = math.sqrt(math.pi) / V ** (1 / 3)
    lim = math.sqrt(-math.log(1e-12))
    rcut, gcut = lim / eta, 2 * eta * lim

    # The squared lengths of the lattice vectors (rows of vecs) shorter than cut, without the
    # origin; plane_gaps (the spacing of the lattice planes) sets how many cells to search.
    def points(vecs, plane_gaps, cut):
        n = [int(math.ceil(cut / g)) for g in plane_gaps]
        idx = np.stack(np.meshgrid(*[np.arange(-k, k + 1) for k in n], indexing="ij"), -1)
        pts = idx.reshape(-1, 3) @ vecs
        d2 = (pts ** 2).sum(1)
        return d2[(d2 > 1e-20) & (d2 < cut * cut)]

    r = np.sqrt(points(A, 2 * np.pi / np.linalg.norm(B, axis=1), rcut))
    g2 = points(B, 2 * np.pi / np.linalg.norm(A, axis=1), gcut)
    real = 0.5 * sum(math.erfc(eta * x) / x for x in r)
    recip = (2 * np.pi / V) * float(np.sum(np.exp(-g2 / (4 * eta * eta)) / g2))
    return real + recip - eta / math.sqrt(math.pi) - math.pi / (2 * V * eta * eta)


# The shape factor c_sh of the Lany-Zunger correction for a supercell, and the shape's name.
# Simple cubic, FCC and BCC supercells (ASE's get_bravais_lattice) get their values from LZ_SHAPE;
# any other shape gets about -1/3, which Lany and Zunger give for roughly isotropic cells.
#   Called by: charge_correction()
def lz_shape_factor(cell):
    """(c_sh, shape name) for the Lany-Zunger correction."""
    try:
        name = Cell(np.asarray(cell)).get_bravais_lattice(eps=1e-3).name
    except (ValueError, RuntimeError):
        name = "other"
    return LZ_SHAPE.get(name, LZ_SHAPE_OTHER), name


# The finite-size correction (eV) for a defect of charge q in a supercell, added to its energy.
# First-order Makov-Payne, E1 = q^2 alpha_M / (2 eps L) = -q^2 x madelung_energy x COULOMB / eps:
# the interaction of a point charge q with its periodic images and the neutralising background,
# screened by the static dielectric constant eps (electronic + ionic). charge_correction : mp uses
# E1; lz (Lany-Zunger) uses [1 + c_sh (1 - 1/eps)] E1, which also accounts for the defect charge
# not being a point (about 2/3 of E1 for large eps); none gives 0. No potential alignment and no
# dipole or quadrupole terms. Returns (correction, a description for the log).
#   Called by: results_row()
#   Calls:     lz_shape_factor(), madelung_energy()
def charge_correction(q, cell, args):
    """Image-charge correction (eV) for charge q in this supercell, and a note."""
    if not q or args.charge_correction == "none":
        return 0.0, ""
    e1 = float(-q * q * madelung_energy(cell) * COULOMB / args.dielectric)
    if args.charge_correction == "mp":
        return e1, f"Makov-Payne {e1:.4f} eV"
    c_sh, shape = lz_shape_factor(cell)
    e = (1 + c_sh * (1 - 1 / args.dielectric)) * e1
    return e, f"Lany-Zunger {e:.4f} eV (Makov-Payne {e1:.4f} eV, {shape} cell, c_sh {c_sh:.3f})"


# The net charge of a calculation, from the 'net charge of system' line CASTEP prints in its
# header (the last one, for a .castep with several runs); None if there is none.
#   Called by: job_result(), results_row()
#   Calls:     Calc.output()
def castep_charge(calc):
    """Net charge CASTEP used (from the .castep), or None."""
    hits = NET_CHARGE.findall(calc.output())
    return float(hits[-1]) if hits else None


# The 'charge' set in a .param file's text (0 if none). series_main() uses it to check, before
# anything is submitted, that the relaxations and the single points have the same charge, and that a
# charged defect has what its correction needs.
#   Called by: series_main()
#   Calls:     parse_castep_input()
def param_charge(text):
    """The charge keyword of a .param (0 if not set)."""
    for it in parse_castep_input(text):
        if it["type"] == "kw" and it["key"] == "charge":
            try:
                return float(it["value"].split()[0])
            except (ValueError, IndexError):
                die(f"can't read 'charge : {it['value']}' in the .param")
    return 0.0


# One row of the results table for supercell size n, with E_f* = E_defect - E_host + E_corr.
# E_defect comes from the single point, or from the end of the relaxation (read_energy). E_host
# comes from the pristine supercell with bulk_mode : supercell, otherwise it is N_host x e_bulk.
# For a charged defect (the net charge in its .castep, castep_charge) E_corr is the image-charge
# correction (charge_correction), logged once per size; it stops if the single point and the
# relaxation ran with different charges. The row also records the cell size, k-points, cores,
# charge, the correction and the uncorrected E_f*.
#   Called by: series_main()
#   Calls:     castep_charge(), charge_correction(), read_energy()
def results_row(n, calcs, structs, args):
    calc = calcs[n].get("defect") or calcs[n]["geom"]   # single point, or end of relaxation
    E_def, key = read_energy(calc, args.energy)
    defect, host = structs[n]["defect"], structs[n]["host"]
    info = calc.info()
    row = {"size": structs[n]["label"], "atoms": len(defect), "L_min": structs[n]["L"],
           "kgrid": "x".join(map(str, info.get("kgrid", ["?"]))), "nkpts": info.get("nkpts", "?"),
           "cores": info.get("ncores", "?"), "E_defect": E_def,
           "eV_per_atom": E_def / len(defect), "energy": key}
    if args.bulk_mode == "supercell":
        row["E_host"] = read_energy(calcs[n]["bulk"], args.energy)[0]
        if args.e_bulk is not None:
            row["host_check"] = row["E_host"] - len(host) * args.e_bulk
    else:
        row["E_host"] = len(host) * args.e_bulk
    q = castep_charge(calc)
    if calc.kind == "defect":                   # the single point must match its relaxation
        q_geom = castep_charge(calcs[n]["geom"])
        if q is not None and q_geom is not None and abs(q - q_geom) > 1e-6:
            die(f"{calc.name} ran with charge {q:g} but its relaxation with {q_geom:g}: set the "
                "same 'charge' in the .param and the geom_param, then delete "
                f"{calc.dir} and rerun")
    row["charge"] = q or 0.0
    row["E_corr"], note = charge_correction(row["charge"], defect.cell, args)
    if note and not args.ctx.get(f"corr_noted_{n}"):
        args.ctx[f"corr_noted_{n}"] = True
        log(f"{structs[n]['label']}: charge {row['charge']:g}, correction {note}")
    row["E_f_raw"] = E_def - row["E_host"]
    row["E_f"] = row["E_f_raw"] + row["E_corr"]  # what is compared with tol
    return row


# Prints the results table and writes convergence.txt and convergence.json.
# Adds dE_f*, the change in E_f* from the previous size; series_main() compares the last one with
# tol.
# For a charged defect the table also shows the charge q and the correction E_corr, and E_f* is
# the corrected value (the uncorrected one is E_f_raw in convergence.json).
#   Called by: series_main()
def report(rows, workdir):
    head = (f"{'size':>9} {'atoms':>6} {'L_min/A':>8} {'k-grid':>8} {'nk':>4} {'cores':>5} "
            f"{'E_defect/eV':>16} {'eV/atom':>12} {'E_host/eV':>16} {'E_f*/eV':>10} "
            f"{'dE_f*/eV':>9}")
    extra = "host_check" in rows[0]
    charged = any(r.get("charge") for r in rows)
    if charged:                                  # E_f* is then the corrected value
        head = head.replace(f"{'E_f*/eV':>10}", f"{'q':>4} {'E_corr/eV':>9} {'E_f*/eV':>10}")
    lines = [head + ("  E_host-N*e_bulk" if extra else ""), "-" * (len(head) + 17 * extra)]
    for i, r in enumerate(rows):
        r["dE_f"] = r["E_f"] - rows[i - 1]["E_f"] if i else None
        d = f"{r['dE_f']:9.4f}" if i else f"{'':>9}"
        lines.append(f"{r['size']:>9} {r['atoms']:>6} {r['L_min']:8.2f} {r['kgrid']:>8} "
                     f"{r['nkpts']:>4} {r['cores']:>5} {r['E_defect']:16.6f} "
                     f"{r['eV_per_atom']:12.6f} {r['E_host']:16.6f} "
                     + (f"{r['charge']:4g} {r['E_corr']:9.4f} " if charged else "")
                     + f"{r['E_f']:10.4f} {d}"
                     + (f"  {r['host_check']:15.4f}" if extra else ""))
    print("\n" + "\n".join(lines) + "\n", flush=True)
    (workdir / "convergence.txt").write_text("\n".join(lines) + "\n")
    (workdir / "convergence.json").write_text(json.dumps(rows, indent=2))


# Checks the jobscript templates before anything is submitted, and returns them ready to use.
# Adds a #!/bin/bash first line if a template has none (sbatch needs one), and stops if the main
# template lacks a placeholder it needs ({SEED} and {NCORES}, or {NODES} and {TASKS_PER_NODE} with
# cores_per_node, and {CPUS_PER_TASK} too with charging : core, where a process may need more
# than one core for its memory), if the dry-run template has no {SEED}, if there is no line that
# runs CASTEP to turn into a dry run, or if {PARTITION} is used without %block partitions (or
# dryrun_partition).
# Returns (job_template, dryrun_template).
#   Called by: series_main(), run_batch()
def check_templates(args):
    job_template, dryrun_template = args.job_template, args.dryrun_template
    job_template = job_template.lstrip()            # sbatch needs #! on the first line
    if not job_template.startswith("#!"):
        job_template = "#!/bin/bash\n" + job_template
        if not getattr(args, "shebang_noted", False):          # (once per run in batch mode)
            log("note: the jobscript template has no #! first line, so #!/bin/bash is added")
            args.shebang_noted = True
    if dryrun_template is not None:
        dryrun_template = dryrun_template.lstrip()
        if not dryrun_template.startswith("#!"):
            dryrun_template = "#!/bin/bash\n" + dryrun_template
    if args.cores_per_node:
        need = ["{SEED}", "{NODES}"] + ([] if "{NCORES}" in job_template else ["{TASKS_PER_NODE}"])
        if args.charging == "core":         # the cores per process bring their memory
            need.append("{CPUS_PER_TASK}")
        example = ("    #SBATCH --nodes={NODES}\n    #SBATCH --ntasks-per-node={TASKS_PER_NODE}\n"
                   + ("    #SBATCH --cpus-per-task={CPUS_PER_TASK}\n"
                      if args.charging == "core" else "")
                   + "    srun castep.mpi {SEED}")
    else:
        need = ["{SEED}", "{NCORES}"]
        example = "    #SBATCH -n {NCORES}\n    mpirun -n {NCORES} castep.mpi {SEED}"
    missing = [p for p in need if p not in job_template]
    if missing:
        die(f"the jobscript template needs {' and '.join(missing)}, e.g.\n{example}")
    if args.dryrun_on == "compute":
        if dryrun_template is not None and "{SEED}" not in dryrun_template:
            die("the dryrun_jobscript template needs {SEED}, e.g.\n"
                "    mpirun -n 1 castep.mpi --dryrun {SEED}")
        if dryrun_template is None and not any(
                CASTEP_RUN.search(line) and not line.lstrip().startswith("#")
                for line in job_template.splitlines()):
            die("could not find the line that runs CASTEP (castep... {SEED}) in the jobscript "
                "template, so can't turn it into a dry run. Give a dry-run template in "
                "dryrun_jobscript, or set dryrun_on : login.")
    if "{PARTITION}" in job_template and not args.partitions:
        die("the jobscript template uses {PARTITION}: list the partitions in "
            "%block partitions, or write the partition name into the template")
    if ("{PARTITION}" in (dryrun_template or "") and not args.partitions
            and not args.dryrun_partition):
        die("the dryrun_jobscript template uses {PARTITION}: set dryrun_partition")
    return job_template, dryrun_template


# Runs the whole workflow (or, with --batch, batch_main).
# 1. Reads the settings (get_args) and checks the jobscript templates (check_templates).
# 2. Reads the host cell (parse_castep_input, read_primitive), the defect (parse_defect) and the
#    k-point spacing (target_spacing), and decides what is copied into every supercell's .cell
#    (keep_items; the defect cells also lose the symmetry keywords).
# 3. Sets up the work directory: a copy of the input file, the fingerprint setup.json (refuses a
#    directory made with other inputs unless --force) and the STOP file.
# 4. Builds every supercell and defect cell (build_supercell, defect_vectors, insert_defect,
#    min_image_distance), makes a Calc for each calculation and prints the size table. A size
#    whose relaxation was set up before keeps the defect site recorded in its job_info.json (or,
#    set up by a version that recorded none, the site that version chose: anchor_site without
#    the centring), so a rerun never moves the defect of a calculation that exists.
# 5. Learns from jobs that already ran (calibrate, measure_times, remove_check).
# 6. Passes: the two smallest sizes first, then one more size each time. Each pass prepares and
#    submits the relaxations and pristine cells (prepare_all, submit), waits for the relaxations
#    (wait_for), then either sets up the single points (prepare_sp) or checks each relaxation
#    (relaxed_atoms), waits for those and the pristine cells, and builds the results (results_row,
#    report). It stops once |dE_f*| < tol, or after the last size.
# Ctrl-C (or the STOP file) leaves submitted jobs running; rerun to carry on.
#   Called by: the last lines of the file, when the script is run
#   Calls:     Calc(), Calc.done(), Calc.info(), anchor_site(), batch_main(), block_rows(),
#              build_supercell(), calibrate(), candidates(), check_templates(), defect_label(),
#              defect_vectors(), element(), get_args(),
#              import_mace(), insert_defect(), kgrid(), measure_times(), min_image_distance(),
#              param_charge(), parse_castep_input(), parse_defect(),
#              prepare_all(), prepare_sp(), prerelax_settings(), read_primitive(), relaxed_atoms(),
#              remove_check(), report(), results_row(), submit(), target_spacing(), wait_for()
def series_main(args):
    print(f"\nSettings from {args.input_path} (everything else at its default):")
    for key, raw in args.raw.items():
        print(f"  {key:<17}: {raw}")
    for key in ("max_jobs", "tolerate_failures", "pack", "dryrun_pack", "array", "array_min"):
        if key in args.raw:
            log(f"note: {key} is for batch mode only, so it is ignored here")
    args.max_jobs, args.tolerate_failures, args.pack, args.dryrun_pack = None, False, 1, 1
    args.array, args.array_min = 1, None
    cell_text, param_text = args.cell.read_text(), args.param.read_text()
    geom_param_text = args.geom_param.read_text() if args.geom_param else param_text
    job_template, dryrun_template = check_templates(args)
    cell_items = parse_castep_input(cell_text)
    prim = read_primitive(cell_items)
    comps = parse_defect(args.defect, prim)
    T = np.array(args.transform).reshape(3, 3) if args.transform else np.eye(3, dtype=int)
    if round(np.linalg.det(T)) <= 0:
        die("transform must have a positive determinant")
    spacing, k_src = target_spacing(cell_items, prim, args)
    keep_items = [it for it in cell_items
                  if not (it["type"] == "block" and it["name"] in STRUCT_BLOCKS | DROP_BLOCKS)
                  and not (it["type"] == "kw" and it["key"] in SET_KEYS)]
    defect_keep_items = [it for it in keep_items         # defect cells: no symmetry
                         if not (it["type"] == "kw" and it["key"] in SYMMETRY_KEYS)]
    for it in cell_items:
        if it["type"] == "block" and it["name"] in DROP_BLOCKS - {"kpoint_list", "kpoints_list"}:
            log(f"note: %block {it['name']} is not copied to the supercells")
    pots = [it for it in cell_items if it["type"] == "block" and it["name"] == "species_pot"]
    if pots:
        listed = {element(r[0]) for r in block_rows(pots[0]) if len(r) > 1}
        for c in comps:
            if c["new"] and c["new"] not in listed:
                log(f"WARNING: species_pot has no entry for {c['new']}; CASTEP will use its "
                    "default pseudopotential for it")
    has_sym = any(it["type"] == "kw" and it["key"] == "symmetry_generate" for it in cell_items)
    if has_sym:
        log("note: symmetry_generate is left out of the defect cells, so they run without "
            "symmetry" + ("; the pristine supercells keep it" if args.bulk_mode == "supercell"
                          else ""))
    elif args.bulk_mode == "supercell":
        log("note: no symmetry_generate in the .cell, so CASTEP will not use symmetry to "
            "reduce the k-points of the pristine supercells (slower)")
    run_sp = args.singlepoint or args.geom_param is not None
    q_param, q_geom = param_charge(param_text), param_charge(geom_param_text)
    if run_sp and q_param != q_geom:
        die(f"the relaxations ({args.geom_param.name}) have charge : {q_geom:g} but the single "
            f"points ({args.param.name}) have {q_param:g}. The defect energy comes from the "
            "single point, so set the same charge in both (it is removed for the pristine "
            "cells).")
    if q_geom and args.charge_correction != "none" and args.dielectric is None:
        die(f"the defect has charge {q_geom:g}: set dielectric (the static dielectric constant, "
            "electronic + ionic) for its correction, or charge_correction : none")
    if q_geom and args.charge_correction == "none":
        log(f"note: charge {q_geom:g} with charge_correction : none - E_f* of a charged defect "
            "changes slowly (about as 1/L) with cell size, so it may never get within tol")
    if args.geom_param and not args.singlepoint:
        log("note: geom_param is set, so each relaxation is still followed by a single point "
            "with the main .param (singlepoint : false is ignored)")
    elif not run_sp:
        log("note: singlepoint : false - each defect energy is the final one of its relaxation")
    if args.prerelax == "mace":
        import_mace()                       # stop now, not halfway, if it isn't installed
        log(f"note: prerelax : mace - each defect cell is relaxed with MACE "
            f"({args.prerelax_model}) to |F| < {args.prerelax_fmax:g} eV/A before CASTEP")

    label_d = defect_label(comps)
    workdir = args.workdir or args.input_path.parent / f"conv_{args.cell.stem}_{label_d}"
    workdir.mkdir(exist_ok=True)
    copy = workdir / args.input_path.name          # keep a record of the settings used
    if copy.resolve() != args.input_path.resolve():
        copy.write_text(args.input_path.read_text())
    fingerprint = {"cell": hashlib.md5(cell_text.encode()).hexdigest(),
                   "param": hashlib.md5(param_text.encode()).hexdigest(),
                   "geom_param": hashlib.md5(geom_param_text.encode()).hexdigest(),
                   "workflow": "relax+singlepoint", "defect": args.defect,
                   "transform": T.tolist(), "kspacing": round(spacing, 8),
                   "defect_symmetry": False, "prerelax": prerelax_settings(args)}
    fp_file = workdir / "setup.json"
    if fp_file.exists() and not args.force:
        old = json.loads(fp_file.read_text())
        old.setdefault("defect_symmetry", has_sym)     # earlier versions kept it
        changed = [k for k in fingerprint if old.get(k) != fingerprint[k]]
        if changed:
            why = (" (it was made by an earlier version of this script, which kept "
                   "symmetry_generate in the defect cells)" if "defect_symmetry" in changed
                   else "")
            die(f"{workdir} was set up with a different {', '.join(changed)}{why}. Set another "
                "workdir, or rerun with --force if you really want to mix them.")
    fp_file.write_text(json.dumps(fingerprint, indent=2))
    args.stop_file = workdir / "STOP"
    if args.stop_file.exists():
        args.stop_file.unlink()                     # left over from an earlier run
    if not (args.prepare_only or args.no_wait):
        log(f"running on {socket.gethostname()} as process {os.getpid()}; to stop it from any "
            f"login node: touch {args.stop_file.absolute()}")

    # ---- build every supercell up front (cheap) and show the plan ----
    vecs = defect_vectors(prim, comps)
    sizes = sorted(set(args.sizes))
    structs, calcs = {}, {}
    print(f"\nHost {prim.get_chemical_formula()} ({len(prim)} atoms/primitive cell), defect "
          f"{' + '.join(c['tok'] for c in comps)}")
    print(f"k-point spacing {spacing:.5f} 1/Ang (2*pi units) from {k_src}")
    print(f"\n{'size':>9} {'host atoms':>11} {'defect atoms':>13} {'L_min/A':>8} {'k-grid':>8}"
          f" {'off-centre/A':>12}" + ("   defect separations/A" if len(comps) > 1 else ""))
    kept = []
    for n in sizes:
        label = f"sc_{n}x{n}x{n}" if args.transform is None else f"sc_n{n}"
        host = build_supercell(prim, n * T)
        geom = Calc(workdir, label, "geom", None)
        anchor = geom.info().get("anchor")            # a size already set up keeps its site
        if anchor is None and "kgrid" in geom.info():  # (set up by an earlier version)
            anchor = anchor_site(host, dict(comps[0], prim_sites=None))
        if anchor is not None and anchor not in candidates(host, comps[0]):
            anchor = None
        defect, seps, off, a = insert_defect(host, comps, vecs, label, anchor)
        if anchor is not None and a != anchor_site(host, comps[0], vecs):
            kept.append(label)
        geom.atoms, geom.anchor = defect, a
        L = min_image_distance(host.cell)
        structs[n] = {"label": label, "host": host, "defect": defect, "L": L}
        calcs[n] = {"geom": geom}
        if run_sp:                  # single point: its atoms are set once relaxed
            calcs[n]["defect"] = Calc(workdir, label, "defect", defect)
        if args.bulk_mode == "supercell":
            calcs[n]["bulk"] = Calc(workdir, label, "bulk", host)
        print(f"{label:>9} {len(host):>11} {len(defect):>13} {L:8.2f} "
              f"{'x'.join(map(str, kgrid(host.cell, spacing))):>8} {off:12.2f}   "
              + " ".join(f"{s:.3f}" for s in seps))
    if kept:
        print(f"note: {', '.join(kept)}: the defect stays on the site an earlier version of this "
              "script set it up on, so results already run stay valid (the same defect, only "
              "moved). Delete a size's directories, or use a new workdir, to have it centred.")
    print()
    ctx = {"spacing": spacing, "keep_items": keep_items, "defect_keep_items": defect_keep_items,
           "job_template": job_template, "dryrun_template": dryrun_template,
           "param_items": parse_castep_input(param_text),
           "geom_param_items": parse_castep_input(geom_param_text),
           "calcs": [c for n in sizes for c in calcs[n].values()]}
    args.ctx, args.all = ctx, (calcs, sizes)       # for restarts, which plan jobs again

    # The calculations that can start straight away for the given sizes: the defect relaxations and
    # pristine supercells (single points wait for their relaxation).
    def first_stage(ns):        # defect relaxations (+ pristine single points)
        return [c for n in ns for kind, c in calcs[n].items() if kind != "defect"]

    try:
        calibrate(calcs, sizes, args)       # mem_replicated : auto, from finished jobs
        measure_times(calcs, sizes)         # walltime : auto, from finished jobs
        for c in ctx["calcs"]:              # delete_check : true
            remove_check(c, args)
        if args.prepare_only:
            if prepare_all(first_stage(sizes), ctx, args):
                log(f"inputs written under {workdir}/ - check them, then rerun without "
                    "--prepare-only." + (" (The single-point inputs are written once each "
                                         "relaxation has finished.)" if run_sp else ""))
            return
        first = 1 if len(sizes) > 1 else 0  # a single size is just run, with no check
        for k in range(first, len(sizes)):
            active = sizes[:k + 1]          # the first pass submits the first two sizes
            stage1 = first_stage(active)
            calibrate(calcs, sizes, args)
            measure_times(calcs, sizes)
            if not prepare_all(stage1, ctx, args):
                return
            for c in stage1:
                if not c.done() and not c.info().get("jobid"):
                    submit(c, args)
            geoms = [calcs[n]["geom"] for n in active]
            if not wait_for(geoms, args):
                return
            calibrate(calcs, sizes, args)
            measure_times(calcs, sizes)
            sps = []
            for n in active:            # single point on each relaxed defect cell
                sp = calcs[n].get("defect")
                if sp is None:          # singlepoint : false - just check the relaxation
                    geom, info = calcs[n]["geom"], calcs[n]["geom"].info()
                    if not info.get("checked"):
                        relaxed_atoms(geom)
                        info["checked"] = True
                        geom.save(info)
                    continue
                prepare_sp(sp, calcs[n]["geom"], ctx, args)
                if not sp.done() and not sp.info().get("jobid"):
                    submit(sp, args)
                sps.append(sp)
            if not wait_for(sps + [c for c in stage1 if c.kind == "bulk"], args):
                return
            rows = [results_row(n, calcs, structs, args) for n in active]
            report(rows, workdir)
            if len(rows) == 1:
                print(f"E_f* = {rows[0]['E_f']:.4f} eV for {rows[0]['size']} (only one size, so "
                      "no convergence check)")
                return
            if abs(rows[-1]["dE_f"]) < args.tol:
                prev, last = rows[-2], rows[-1]
                terms = " ".join(f"+ mu_{c['host']}" + (f" - mu_{c['new']}" if c["new"] else "")
                                 for c in comps)
                print(f"CONVERGED: E_f* changed by {last['dE_f']:+.4f} eV (|dE| < {args.tol} eV) "
                      f"from {prev['size']} to {last['size']}.\n"
                      f"  {prev['size']} ({prev['atoms']} atoms) is converged to within "
                      f"{args.tol} eV; best value E_f* = {last['E_f']:.4f} eV ({last['size']}).\n"
                      f"  Formation energy: E_f = E_f* {terms}   ({rows[-1]['energy']} energies)")
                return
        print(f"NOT CONVERGED to {args.tol} eV within sizes {sizes}: add larger sizes to the "
              "input file and rerun (finished calculations are reused).")
    except KeyboardInterrupt:
        print("\nInterrupted. Submitted jobs keep running - rerun the same command to carry on.")



# ##################################################################################################
# PART 2: THE DEFECT WORKFLOW
# ##################################################################################################

# ==========================================================================================
# HOW THE FUNCTIONS FIT TOGETHER
# ==========================================================================================
# The steps are listed near the end (STEPS: name, label, title, the function that does the
# work, and the settings, files and earlier steps its results depend on). main() runs them:
#
#   main                 the command line; Project(...) -> read_project reads the project file
#     for each step:
#       inputs_hash      fingerprint of its settings, files and the results of the steps it
#                        needs (skipped if state.json has it done with the same fingerprint)
#       step_<name>      does the work - CASTEP jobs through Project.batch -> supercell_
#                        convergence.run_batch - and returns its results (None: not finished
#                        yet, e.g. --no-wait), which Project.save_results puts in its phase's
#                        results.json; Project.out(step) gives them to the later steps
#       outputs_hash     fingerprint of the results and the files they name, for later steps
#       write_summary    the phase's summary.txt, from the notes and tables of its steps
#
# Helpers the steps share: structures (save_cell, load_cell, build_defect, host_supercell,
# symmetry_ops, equivalent_sites, electron_parity, spin_states), supercell sizes (min_image,
# size_matrix, size_ladder) and moving a defect between them (carry_over, same_basin), CASTEP
# output (read_bands, occupied_bands, band_edges, gap_levels, internal_pair, read_permittivity,
# read_trajectory, read_pdos_weights, participation), the charge correction (correction), MACE
# (mace_relaxations, which runs mace_relax here or mace_jobs on compute nodes, whose workers are
# mace_worker; train_command and run_training for F2's model), ΔSCF (dscf_job,
# dscf_result), lineshapes (pl_spectrum, phonon_worker) and tables (format_table).
# Project.out() and Project.phase_dir() are used by almost every step, so they are left out of
# the "Calls" lists, as are log() and die().
# ==========================================================================================

HARTREE = 27.211386245988                     # eV
STATE, RESULTS, SUMMARY = "state.json", "results.json", "summary.txt"
# the work directory's file of everything (every step's status, the project file's keywords and
# every phase's summary), rewritten as the steps finish: the workflow's .castep
OVERVIEW = "results.dwf"
PHASE_DIRS = {"A": "A_host", "B": "B_candidates", "C": "C_screen", "D": "D_production",
              "E": "E_thermo", "F": "F_lineshape"}
PHASE_TITLES = {"A": "host", "B": "candidate defects, the screening and electronic cells",
                "C": "screening, ground and excited states",
                "D": "production ground and excited states", "E": "thermodynamics",
                "F": "optical lineshapes"}
# Work directories made while phase D held the ground and excited states (D1-D5's ground-state
# model, D6 rank, D7 excite and D8 validate, now C6-C8; production, thermodynamics and lineshapes
# were phases E, F and G): the phase folders that were renamed, the folder that held D6-D8, what
# they kept there, and the steps that are gone (--migrate, migrate_workdir)
OLD_PHASE_DIRS = {"E_production": "D_production", "F_thermo": "E_thermo",
                  "G_lineshape": "F_lineshape"}
OLD_D_DIR = "D_states"
OLD_D_PATHS = ("jobs/share_*", "jobs/excite_relax", "jobs/excite_gs", "jobs/excite",
               "jobs/validate", "jobs/validate_gamma", "structures/share",
               "structures/excite_relax", "structures/excite", "structures/validate")
OLD_STEPS = ("dataset", "hybrid", "train", "check", "search")
# Steps removed since that a work directory of the current layout may have run (C4
# dopant_phases, the dopants' compounds, 19-23 Nov 2026): their records leave state.json and
# their results their phase's results.json when the project is opened; their folders stay.
GONE_STEPS = {"dopant_phases": "C"}
HB2M = 3.80998                                # hbar^2 / (2 m_e), eV Ang^2
GPA_A3 = 6.241509e-3                          # eV in 1 GPa Ang^3
# B2: the atoms at a supercell's edge, whose MACE displacements show whether a defect's strain
# field fits in it, are those further than this fraction of the shortest image distance from it
EDGE_FRACTION = 0.35
# B2 (scan_test : line): how many bigger sizes must lie on the R = R_inf - c/N line with a size for
# a candidate to count as converged there
SCAN_CONFIRM = 3
# How much deeper (eV) a relaxation from another size's minimum must go than the one from the ideal
# structure to take its place (B2's re-seeding, C1): more than two relaxations into one minimum
# differ by
SEED_DEEPER = 0.005
# C2: a Gamma-point gap of the screening cell this much above A2.2's means band edges that don't
# fold onto Gamma (eV)
FOLD_GAP_TOL = 0.05
# C2-C6: a screening run with at least this many electrons in the conduction band (or holes in the
# valence band) left them there instead of in defect levels: a band state, not a bound one
BAND_CARRIER = 0.5
# C4 (distort_around : auto): a complex whose components are all within this many host bond
# lengths of each other is distorted around its centre; one more spread out, around each component
PAIR_BOND_FACTOR = 1.25
# C4: an atom this close to a complex's centre (Ang) is the centre the distortions are made around
CENTRE_ATOM_TOL = 0.5
# C7: two relaxations of a defect (in different cells, or before and after C7 relaxes it) are in the
# same basin if no atom near the defect is displaced from its site by more than this differently
# (Ang)
BASIN_TOL = 0.2
# C5's ranking: two relaxations are compared for being the same structure only when their energies
# are within this (eV); further apart, they are different minima
RANK_E_TOL = 0.05
# C4's table: each defect's lowest this many distinct MACE minima are listed (more if
# distort_starts keeps more)
MACE_TABLE_MINIMA = 5
# C4 without MACE (mace : false): the distortions C5 relaxes in CASTEP, each in its own charge
# state, when the project gives none. ShakeNBreak's full set, the default with MACE (which relaxes
# them for little), would cost three times as much there; these four found the lowest structure
# of every charge state of CaO:Sb that the full set found
CASTEP_DISTORTIONS = (-0.4, -0.2, 0.2, 0.4)
# F2: the structures left out of its model's training data (more than MODEL_E_WINDOW eV above the
# lowest of their charge state, atoms closer than MODEL_MIN_DIST Ang, a force above MODEL_MAX_FORCE
# eV/Ang), the share kept for testing, the random seed; the total charge's embedding (classes from
# -CHARGE_OFFSET)
MODEL_E_WINDOW, MODEL_MIN_DIST, MODEL_MAX_FORCE = 3.0, 1.5, 10.0
MODEL_TEST_FRACTION, MODEL_SEED = 0.2, 7
CHARGE_OFFSET, CHARGE_CLASSES = 10, 21
# The settings a MACE relaxation task may carry of its own (a model other than mace_model, one of
# a multi-head model's heads, a total charge, fmax, steps, the rattle of its start: rattle_spec),
# in this order (mace_key's, mace_relax's)
EXTRA_TASK_KEYS = ("model", "head", "charge", "fmax", "steps", "rattle")
# The rattle of a MACE relaxation from a defect's ideal structure (B2, C1; rattle_spec): the atoms
# within this of one of the defect's components are moved (Ang), and the random displacements are
# the same every time (this seed)
MACE_RATTLE_RADIUS, MACE_RATTLE_SEED = 4.0, 11
# C7, D2: the smallest Kohn-Sham step taken as an excitation (it skips a degenerate partner level),
# eV; and a mode's energy from a curvature, hbar sqrt(eV / (amu Ang^2)), in eV
KS_STEP_MIN = 0.1
HW_UNIT = 0.0646541
# B4: how far beyond the ladder (in steps of ELECTRONIC_STEP Ang, at most ELECTRONIC_TRIES of them)
# to look for a cell that folds every band-edge k-point onto Gamma
ELECTRONIC_STEP, ELECTRONIC_TRIES = 0.5, 30
# F2: the model's relaxation of the big cell before its force constants (eV/Ang, steps), the
# lowest mode energy counted (eV; below it, the cell's translations) and how often (s) a phonon job
# saves the force constants it has so far
LINESHAPE_FMAX, LINESHAPE_STEPS = 0.002, 3000
LINESHAPE_HW_MIN = 0.001
LINESHAPE_SAVE = 300
MACE_ATTEMPTS = 3                             # a MACE relaxation's jobs before giving up on it


# --------------------------------------------------------------------------
# Project file
# --------------------------------------------------------------------------
# A list of numbers (commas or spaces), optionally exactly n of them, as a converter for the
# project keywords (like the runner's _ints, for decimals).
def _floats(n=None):
    # The converter itself.
    def conv(v):
        try:
            vals = [float(x) for x in re.split(r"[\s,]+", v) if x and x.lower() not in
                    ("ev", "ang", "a", "1/ang", "ev/atom", "k")]
        except ValueError:
            vals = []
        if not vals or (n and len(vals) != n):
            raise ValueError(f"expected {n or 'one or more'} numbers")
        return vals
    return conv


# A supercell size, as a converter for the supercell keyword: a whole number n (n x transform, or
# n x n x n primitive cells) or a length with its unit (15 ang: the most compact supercell with
# every periodic image at least that far away). Returns ['n', n] or ['length', L].
def _size(v):
    parts = v.split()
    try:
        if len(parts) == 2 and parts[1].lower() in ("ang", "a", "angstrom"):
            if float(parts[0]) > 0:
                return ["length", float(parts[0])]
        elif len(parts) == 1 and int(parts[0]) >= 1:
            return ["n", int(parts[0])]
    except ValueError:
        pass
    raise ValueError("expected a whole number n (n x transform) or a length with its unit, "
                     "e.g. 15 ang")


# The electronic_cell keyword: 'screening' (the screening cell itself), or a size as _size takes.
def _cellsize(v):
    if v.strip().lower() == "screening":
        return ["screening", None]
    return _size(v)


# The excite_spin keyword: a spin (unpaired electrons, as CASTEP counts them: 2 for a triplet), or
# 'any' (None: every spin).
def _spin_or_any(v):
    if v.strip().lower() in ("any", "none"):
        return None
    x = float(v)
    if x < 0:
        raise ValueError("expected a spin of at least 0, or 'any'")
    return x


# The dopant_pairs keyword: true (pairs of any two dopant defects, of the same element or not),
# false (none) or 'same' (only pairs of two defects of the same dopant, e.g. Sb_Ca + Sb_O: each
# dopant on its own with the host). true and false stay booleans, so a project file that gave
# them before keeps its B1.
def _dopant_pairs(v):
    if v.strip().lower() == "same":
        return "same"
    try:
        return _bool(v)
    except ValueError:
        raise ValueError("expected true, false or same") from None


_t, _b, _w = _text, _bool, _words
# keyword: (converter, default). Blocks of whole lines ('lines') keep their rows; 'kv' blocks
# hold "keyword : value" lines (.param keywords to set); a 'template' is a file name or a block
# holding the text itself (like the runner's jobscript).
PROJECT_KEYWORDS = {
    # general
    "name": (_t, None), "workdir": (_t, None), "host_cell": (_t, None), "param": (_t, None),
    "transform": (_ints(9), None), "energy": (_choice("auto", "final", "free", "0K"), "auto"),
    "mace": (_b, True), "mace_model": (_t, "medium-mpa-0"), "mace_device": (_t, "cpu"),
    "mace_threads": (_num(cast=int), 8), "mace_fmax": (_num("eV/ang", "eV/A"), 0.05),
    "mace_steps": (_num(cast=int), 500), "mace_rattle": (_num("ang", "A"), 0.02),
    "mace_jobscript": ("template", None), "mace_pack": (_num(cast=int), None),
    "mace_max_jobs": (_num(cast=int), 4), "mace_python": (_t, None),
    # A: host
    "converge": (_b, True), "converge_cutoffs": (_floats(), [300, 400, 500, 600, 700, 800]),
    "converge_kgrid_min": (_ints(), [4]), "converge_kgrid_max": (_ints(), [12]),
    "converge_kgrid_step": (_ints(), [2]),
    "converge_tol": (_num("eV/atom", "eV"), 0.001), "kspacing": (_num("1/ang"), None),
    "converge_kcutoff": (_num("eV"), None),
    "bands_kspacing": (_num("1/ang"), None),
    "dielectric": (_num(), None), "dielectric_inf": (_num(), None),
    "dielectric_task": (_t, "phonon+efield"), "dielectric_param": (_t, None),
    "dielectric_cell": (_t, None),
    "phases": ("lines", None), "phases_from": (_choice("none", "doped"), "none"),
    "mp_api_key": (_t, None), "phases_e_above_hull": (_num("eV/atom", "eV"), 0.05),
    "phases_max_steps": (_num(cast=int), 100),
    "phases_molecule_box": (_num("ang", "A"), 15.0), "chemical_potentials": ("lines", None),
    "ref_relax": (_choice("none", "positions", "cell"), "none"),
    "ref_kspacing_factor": (_num(), 1.0),
    "ref_polymorph_window": (_num("eV/atom", "eV"), 0.02),
    # B: candidates and the screening cell
    "dopants": (_w, []), "defects": ("lines", None), "vacancies": (_b, True),
    "antisites": (_b, False),
    "interstitials": (_choice("none", "dopants", "all"), "none"),
    "pairs": (_b, True), "pair_max_distance": (_num("ang", "A"), 5.0),
    "native_pairs": (_b, False), "dopant_pairs": (_dopant_pairs, False),
    "scan": (_b, True), "scan_lengths": (_floats(), [8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]),
    "scan_tol": (_num("eV"), 0.02), "scan_edge_tol": (_num("ang", "A"), 0.05),
    "scan_test": (_choice("line", "flat"), "line"),
    "scan_unconverged": (_choice("drop", "largest"), "drop"),
    "supercell": (_size, None), "screen_min_image": (_num("ang", "A"), None),
    "electronic_cell": (_cellsize, None), "electronic_depth": (_num("eV"), 0.25),
    "decay_length_multiple": (_num(), 4.0),
    # C: screening, ground and excited states
    "screen_charges": (_ints(), [-1, 0, 1]),
    "screen_spins": (_choice("triplet", "low", "both"), "triplet"),
    "screen_cutoff": (_num("eV"), None), "screen_max_steps": (_num(cast=int), 50),
    "screen_overrides": ("kv", {}),
    "configuration_margin": (_num("eV"), None),
    "hull_margin": (_num("eV"), 0.2), "level_margin": (_num("eV"), 0.1),
    "site_margin": (_num("eV"), 0.5),
    "require_bound": (_b, False), "require_spin": (_num(), None),
    "require_gap_states": (_choice("none", "occupied", "empty", "both"), "none"),
    "distortions": (_floats(), [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5,
                                0.6]),
    "distort_starts": (_num(cast=int), 2),
    "distort_charges": (_choice("relevant", "stable", "screened"), "relevant"),
    "charge_margin": (_num("eV"), 0.5), "carrier_reach": (_num("eV"), 1.0),
    "carrier_fraction": (_num(), 0.2),
    "distort_around": (_choice("auto", "centre", "each", "anchor", "vacancy"), "auto"),
    "distort_cell": (_choice("electronic", "screening"), "electronic"),
    "rattle_stdev": (_num("ang", "A"), None), "rattle_dmin": (_num("ang", "A"), None),
    "oxidation_states": ("lines", None),
    "distortion_min_gain": (_num("eV"), 0.05), "share_rounds": (_num(cast=int), 2),
    "keep_structures": (_num(cast=int), 1),
    "excite": (_b, True), "excite_window": (_floats(2), None),
    "excite_min": (_num("eV"), 0.4), "excite_relax": (_b, True),
    "excite_overrides": ("kv", {}), "excite_spin": (_spin_or_any, 2.0),
    "spin_tol": (_num(), 0.3), "local_fraction": (_num(), 0.2),
    "excite_reach": (_num("eV"), 1.0), "excite_min_dw": (_num(), None),
    "validate": (_b, False), "validate_tol": (_num("eV"), 0.05),
    "validate_defects": (_w, []), "validate_sizes": (_num(cast=int), 3),
    "validate_gamma": (_b, True),
    "elastic_constants": ("lines", None),
    # D: production
    "production_overrides": ("kv", {}), "freeze_radius": (_num("ang", "A"), 0.0),
    "band_spread_tol": (_num("eV"), 0.05),
    # E: thermodynamics
    "charge_correction": (_choice("lz", "mp", "none"), "lz"), "gap": (_num("eV"), None),
    "fermi_temperature": (_num("K"), None), "quench_temperature": (_num("K"), None),
    "deep_level_margin": (_num("eV"), 0.5),
    # F: lineshapes (F2's model: model_*)
    "model_jobscript": ("template", None), "model_command": (_t, "mace_run_train"),
    "model_device": (_t, None), "model_epochs": (_num(cast=int), 300),
    "model_r_max": (_num("ang", "A"), 5.0), "model_channels": (_num(cast=int), 64),
    "model_max_L": (_num(cast=int), 1), "model_batch": (_num(cast=int), 4),
    "model_train_args": (_t, ""),
    "ccd_points": (_floats(), [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]),
    "lineshape_temperature": (_num("K"), 10.0), "lineshape_sigma": (_num("eV"), 0.006),
    "lineshape_gamma": (_num("eV"), 0.002), "lineshape": (_b, False),
    "lineshape_displacements": (_num(cast=int), 30),
    "lineshape_amplitude": (_num("ang", "A"), 0.02),
    "lineshape_host": (_num(cast=int), 5), "lineshape_length": (_num("ang", "A"), 30.0),
    "lineshape_method": (_choice("forces", "displacements"), "forces"),
    "lineshape_delta": (_num("ang", "A"), 0.01),
}
LIST_KEYS = {"transform", "converge_cutoffs", "dopants", "scan_lengths", "screen_charges",
             "distortions", "excite_window", "ccd_points"}  # or block
# The project keywords that say how jobs run rather than what they compute: they go in NAME.run,
# with the runner keywords (RUNNER_KEYS) and blocks, not in NAME.parameters: where the work goes,
# the MACE jobs and F2's model jobs
RUN_PROJECT_KEYS = {"workdir", "mace_jobscript", "mace_pack", "mace_max_jobs", "mace_python",
                    "mace_threads", "mace_device", "model_jobscript", "model_command",
                    "model_device"}
# The steps whose MACE relaxations may have settings of their own, in %block runner_scan (B2),
# runner_prerelax (C1) or runner_distort (C4) in NAME.run, and the keywords such a block may hold:
# the MACE job settings and max_walltime, the jobs' --time (Project.mace_settings)
MACE_STEPS = ("scan", "prerelax", "distort")
MACE_BLOCK_KEYS = ("mace_jobscript", "mace_pack", "mace_max_jobs", "mace_threads", "mace_python",
                   "mace_device", "max_walltime")
# The job runner's keywords that control how jobs run (allowed in the project file, for every
# step, and in %block runner_<group> or runner_<step>)
RUNNER_KEYS = {"jobscript", "dryrun_jobscript", "partitions", "max_cores", "cores",
               "cores_per_node", "max_nodes", "charging", "mem_per_core", "mem_safety",
               "mem_replicated", "baseline_mb", "omp_threads", "walltime", "walltime_safety",
               "sp_walltime", "max_walltime", "max_restarts", "delete_check", "dryrun_on",
               "dryrun_partition", "dryrun_time", "dryrun_cores", "dryrun_mem_gb",
               "dryrun_options", "dryrun_cmd", "dryrun_timeout", "submit_cmd", "status_cmd",
               "poll", "max_jobs", "tolerate_failures", "pack", "dryrun_pack", "array",
               "array_min"}
# The steps that run CASTEP jobs, and the group of steps each belongs to. The runner settings of a
# step: the project's, changed by its group's defaults (GROUP_DEFAULTS), then by %block
# runner_<group>, then by %block runner_<step> (Project.runner_items). host is for phase A's
# steps and the competing phases wherever they run (A3.1, E1.1: small cells, which stop the
# step if one fails). screen is both a step and a group: runner_screen is for every defect step
# at the screening settings (C2, C3's single points, C5, C6, C7), and runner_filter,
# runner_drelax, runner_rank or runner_excite changes it for one of them; runner_production is
# for every step at the production settings (C8, D1, D2, F1, F2's single points).
RUNNER_STEPS = {"converge": "host", "relax": "host", "bands": "host", "dielectric": "host",
                "ref_phases": "host", "phases": "host",
                "screen": "screen", "filter": "screen", "drelax": "screen",
                "rank": "screen", "excite": "screen", "validate": "production",
                "relaxations": "production", "zpl": "production", "ccd": "production",
                "lineshape": "production"}
RUNNER_GROUPS = ("host", "screen", "production")
RUNNER_BLOCKS = RUNNER_GROUPS + tuple(s for s in RUNNER_STEPS if s not in RUNNER_GROUPS)
# what each group of steps uses unless the .run file's runner blocks say otherwise
GROUP_DEFAULTS = {"host": "",
                  "screen": "max_restarts : 0\ntolerate_failures : true",
                  "production": "tolerate_failures : true"}


# The k-point grids of A1's scan, coarse to fine: from converge_kgrid_min to converge_kgrid_max
# in steps of converge_kgrid_step. Each is one number (the same in all three directions: n x n x n)
# or three (one per reciprocal-lattice direction, for a non-cubic grid such as 4 4 2 to 12 12 6 in
# steps of 2 2 1); a step of 0 keeps that direction fixed (e.g. 1 for a slab). Every direction
# that moves must take the same number of steps. Stops with what is wrong otherwise.
#   Called by: read_project()
def kgrid_list(name, lo, hi, step):
    three = [v * 3 if len(v) == 1 else v for v in (lo, hi, step)]
    if any(len(v) != 3 for v in three):
        die(f"{name}: converge_kgrid_min, _max and _step take one number or three")
    lo, hi, step = three
    counts = set()
    for a, b, d in zip(lo, hi, step):
        if a < 1 or b < a or d < 0 or (d == 0 and a != b) or (d and (b - a) % d):
            die(f"{name}: converge_kgrid_min {' '.join(map(str, lo))} to _max "
                f"{' '.join(map(str, hi))} in steps of {' '.join(map(str, step))} does not work: "
                "each direction needs 1 <= min <= max, with max - min a multiple of its step "
                "(a step of 0 needs min = max)")
        if d:
            counts.add((b - a) // d)
    if len(counts) > 1:
        die(f"{name}: converge_kgrid_min/_max/_step take different numbers of steps in different "
            f"directions ({' '.join(map(str, lo))} to {' '.join(map(str, hi))} in steps of "
            f"{' '.join(map(str, step))}): make them the same")
    k = counts.pop() if counts else 0
    return [[a + j * d for a, d in zip(lo, step)] for j in range(k + 1)]


# The keyword, or the block's name, of an item of a CASTEP-style file (parse_castep_input).
#   Called by: Project.runner(), Project.runner_items(), mace_block(), read_project(),
#              write_results()
def item_key(x):
    return x["name"] if x["type"] == "block" else x["key"]


# Keywords that have been renamed or removed, with what to do instead.
#   Used by: read_project()
_NO_C2B = ("has been removed, with step C2b (the Gamma-only check of the screening cell): the "
           "screening goes on with its Gamma-point results, as ADAQ does")
_NO_C2C = "has been removed, with step C2c (the size check of the screening cell)"
_NO_SERIES = ("has been removed with the CASTEP size series of one defect (the old B3): B2 now "
              "scans every candidate with MACE, and the survivors run in the electronic cell "
              "(B4) from C4 on")
_NO_E0 = ("has been removed with E0's size series: every defect now runs in the electronic cell "
          "(B4), and C8 (validate : true) tests its finite-size error on a few defects")
_NO_E2 = ("has been removed with the old E2: F2 (lineshape : true) trains its own model on the "
          "production runs")
_NO_MODEL = ("has been removed, with D1-D5 (the ground-state model): C6 ranks C5's "
             "relaxations, with sharing between charge states")
RETIRED = {"series_tol": _NO_E0, "runner_cells": _NO_E0, "finetune": _NO_E2,
           "max_formation_energy": "has been removed: C3 no longer drops a defect on its absolute "
                                   "formation energy, which could drop every defect of a dopant "
                                   "(site_margin still compares a dopant's defects across "
                                   "their sites)",
           "finetune_args": _NO_E2, "runner_finetune": _NO_E2,
           "triage": "has been renamed: use configuration_margin (in eV; giving it switches the "
                     "filter on)",
           "triage_window": "has been renamed: use configuration_margin (in eV; giving it switches "
                            "the filter on)",
           "zpl_window": "has been removed, with step C3b and its Kohn-Sham ZPL estimate: C7 "
                         "filters on the ΔSCF ZPL instead (excite : true, excite_window)",
           "series_defect": _NO_SERIES,
           "series_charge": _NO_SERIES + " (in every charge state the defect reaches D in)",
           "series_sizes": _NO_SERIES + "; the sizes are now lengths: scan_lengths (Ang)",
           "series_choice": _NO_SERIES + ", taking the smaller of the first two sizes that agree "
                            "within series_tol",
           "runner_series": _NO_SERIES + ": the production runs use runner_production",
           "scan_sizes": "has been replaced by scan_lengths: the sizes are lengths in Ang, each "
                         "the most compact supercell with every periodic image at least that "
                         "far away",
           "finetuned_model": "has been removed: B2 and C1 use mace_model (a foundation model or "
                              "your own .model file); F2 (lineshape : true) trains its own model "
                              "on the production runs",
           "dispersion_check": _NO_C2B, "dispersion_defect": _NO_C2B, "dispersion_tol": _NO_C2B,
           "runner_dispersion": _NO_C2B,
           "size_check": _NO_C2C, "size_check_window": _NO_C2C, "size_check_max": _NO_C2C,
           "size_check_defects": _NO_C2C, "size_check_cell": _NO_C2C, "runner_sizecheck": _NO_C2C,
           "band_state_cell": "has been replaced by distort_cell: C4 and C5 work in the electronic "
                              "cell for every charge state of every survivor (distort_cell : "
                              "electronic, the default), band states included",
           "model_hybrid_fraction": _NO_MODEL, "runner_hybrid": _NO_MODEL,
           "runner_dataset": _NO_MODEL, "runner_check": _NO_MODEL,
           **{k: _NO_MODEL for k in ("model", "model_hybrid", "hybrid_cell", "hybrid_param",
                                     "model_hybrid_points", "model_rattles",
                                     "model_rattle_stdev", "model_force_tol", "model_rounds",
                                     "search_distortions", "search_rattles", "search_fmax",
                                     "search_steps", "search_keep")}}


# The settings of a MACE step's own block (%block runner_scan, runner_prerelax or runner_distort,
# it: the parsed
# block; name: the file, for messages): the MACE job keywords it gives (MACE_BLOCK_KEYS), each
# converted as the .run file's own, and max_walltime. Returns {keyword: value}.
#   Called by: read_project()
#   Calls:     item_key(), parse_castep_input()
def mace_block(it, name):
    key, out = item_key(it), {}
    for x in parse_castep_input("\n".join(it["body"])):
        if x["type"] == "raw":
            continue
        k = item_key(x)
        if k not in MACE_BLOCK_KEYS:
            die(f"{name}: %block {key} holds the settings of {STEP_INDEX[key[7:]].label}'s MACE "
                "jobs (" + ", ".join(MACE_BLOCK_KEYS) + f"), not '{k}'")
        if k in out:
            die(f"{name}: '{k}' is given more than once in %block {key}")
        if k == "mace_jobscript":
            out[k] = (["text", "\n".join(x["body"]).strip("\n") + "\n"] if x["type"] == "block"
                      else ["file", x["value"]])
            continue
        conv = _slurm_time if k == "max_walltime" else PROJECT_KEYWORDS[k][0]
        try:
            out[k] = conv(x.get("value", ""))
        except ValueError as err:
            die(f"{name}: bad value for {k} in %block {key} ('{x.get('value')}'): {err}")
        if k in ("mace_pack", "mace_max_jobs", "mace_threads") and out[k] < 1:
            die(f"{name}: {k} must be at least 1 (in %block {key})")
    return out


# Reads a project's two files into a namespace of settings: path, NAME.parameters (the project
# keywords, PROJECT_KEYWORDS, but those of RUN_PROJECT_KEYS), and run_path, NAME.run (how the jobs
# run: the runner keywords, RUNNER_KEYS, passed to batch_args, the
# RUN_PROJECT_KEYS and the runner blocks). %block runner_<group> or runner_<step> (RUNNER_BLOCKS)
# holds runner keywords, and may hold a %block partitions, for a group of steps or one step; %block
# runner_scan, runner_prerelax or runner_distort (MACE_STEPS) holds B2's, C1's or C4's MACE job
# settings (mace_block). A keyword in the wrong file stops with where it goes; a typo gets the
# closest keyword suggested, and a retired one (RETIRED) what to do. Without MACE (mace : false),
# distortions not given are CASTEP_DISTORTIONS. Returns (settings, the text each setting was given
# as, the runner items, {group or step: runner items}, {MACE step: its block's settings}).
#   Called by: Project.__init__()
#   Calls:     compliance(), item_key(), kgrid_list(), mace_block(), block_rows(),
#              parse_castep_input()
def read_project(path, run_path):
    vals, raw, runner, blocks, mace_blocks = {}, {}, [], {}, {}
    known = (set(PROJECT_KEYWORDS) | RUNNER_KEYS
             | {f"runner_{b}" for b in RUNNER_BLOCKS + MACE_STEPS})
    items = [(src, it) for src in (path, run_path) for it in parse_castep_input(src.read_text())
             if it["type"] != "raw"]
    for src, it in items:
        name = src.name
        key = item_key(it)
        if key in RETIRED:
            die(f"{name}: '{key}' {RETIRED[key]}")
        if key.startswith("runner_") and key not in known:
            die(f"{name}: there is no {key}: the runner blocks are runner_host (all of phase A's "
                "CASTEP steps, and the competing phases of E1.1), runner_screen (C2, C3, "
                "C5, C6, C7: the screening settings), "
                "runner_production (C8, D1, D2, F1, F2: the production settings), for one step "
                + ", ".join(f"runner_{s}" for s in RUNNER_STEPS if s not in RUNNER_GROUPS)
                + ", and for the MACE jobs of B2, C1 or C4 runner_scan, runner_prerelax or "
                "runner_distort")
        if key not in known:
            close = difflib.get_close_matches(key, sorted(known), n=1)
            die(f"{name}: unknown keyword '{key}'" + (f" - did you mean '{close[0]}'?" if close
                                                       else "")
                + "  (--example NAME writes a pair of files with them all)")
        running = key in RUNNER_KEYS or key in RUN_PROJECT_KEYS or key.startswith("runner_")
        if running and src != run_path:
            die(f"{name}: '{key}' says how jobs run: it goes in {run_path.name}")
        if not running and src == run_path:
            die(f"{name}: '{key}' is not a setting for running jobs: it goes in {path.name}")
        if key in raw:
            die(f"{name}: '{key}' is given more than once")
        if key in RUNNER_KEYS:
            runner.append(it)
            raw[key] = it.get("value", "(%block)")
            continue
        if key[len("runner_"):] in MACE_STEPS:
            if it["type"] != "block":
                die(f"{name}: {key} is a block of MACE job settings (%block {key} ... "
                    f"%endblock {key})")
            mace_blocks[key[len("runner_"):]] = mace_block(it, name)
            raw[key] = "(%block)"
            continue
        if key.startswith("runner_"):
            if it["type"] != "block":
                die(f"{name}: {key} is a block of runner settings (%block {key} ... "
                    f"%endblock {key})")
            body = [x for x in parse_castep_input("\n".join(it["body"])) if x["type"] != "raw"]
            seen = set()
            for k in map(item_key, body):
                if k not in RUNNER_KEYS:
                    die(f"{name}: %block {key} can only hold runner settings, not '{k}'")
                if k in seen:
                    die(f"{name}: '{k}' is given more than once in %block {key}")
                seen.add(k)
            blocks[key[len("runner_"):]] = body
            raw[key] = "(%block)"
            continue
        conv, _ = PROJECT_KEYWORDS[key]
        if conv == "template":                     # a file name, or the text in a block
            if it["type"] == "block":
                vals[key] = ["text", "\n".join(it["body"]).strip("\n") + "\n"]
                raw[key] = "(%block)"
            else:
                vals[key] = ["file", it["value"]]
                raw[key] = it["value"]
            continue
        if conv in ("lines", "kv"):
            if it["type"] != "block":
                die(f"{name}: {key} is a block - write %block {key} ... %endblock {key}")
            rows = block_rows(it)
            if conv == "lines":
                vals[key] = rows
                raw[key] = " / ".join(" ".join(r) for r in rows)
            else:
                kv = {x["key"]: x["value"] for x in
                      parse_castep_input("\n".join(it["body"])) if x["type"] == "kw"}
                vals[key] = kv
                raw[key] = " / ".join(f"{k} : {v}" for k, v in kv.items())
            continue
        if it["type"] == "block":
            if key not in LIST_KEYS:
                die(f"{name}: {key} is a keyword, not a block - write '{key} : value'")
            text = " ".join(" ".join(r) for r in block_rows(it))
        else:
            text = it["value"]
            if len(text) > 1 and text[0] == text[-1] and text[0] in "'\"":
                text = text[1:-1]
        raw[key] = text
        try:
            vals[key] = conv(text)
        except ValueError as err:
            die(f"{name}: bad value for {key} ('{text}'): {err}")
    s = SimpleNamespace(**{k: (v if not isinstance(v, (list, dict)) else type(v)(v))
                           for k, (_, v) in PROJECT_KEYWORDS.items()})
    s.__dict__.update(vals)
    name = path.name
    for key in ("host_cell", "param"):
        if getattr(s, key) is None:
            die(f"{name}: '{key}' is required")
    if not any(item_key(x) == "jobscript" for x in runner):
        die(f"{run_path.name}: give the jobscript template, as 'jobscript : file' or "
            "%block jobscript")
    if s.transform is not None and round(np.linalg.det(np.reshape(s.transform, (3, 3)))) <= 0:
        die(f"{name}: transform must have a positive determinant")
    for key in ("pair_max_distance", "scan_tol", "scan_edge_tol", "band_spread_tol",
                "configuration_margin", "converge_tol", "mace_fmax", "excite_min",
                "validate_tol", "lineshape_sigma", "lineshape_gamma", "lineshape_amplitude",
                "lineshape_delta", "decay_length_multiple", "ref_kspacing_factor"):
        if getattr(s, key) is not None and getattr(s, key) <= 0:
            die(f"{name}: {key} must be greater than 0")
    for key in ("mace_threads", "mace_steps", "mace_pack", "mace_max_jobs", "validate_sizes",
                "model_epochs", "model_channels", "model_batch"):
        if getattr(s, key) is not None and getattr(s, key) < 1:
            die(f"{run_path.name if key in RUN_PROJECT_KEYS else name}: {key} must be at least 1")
    if min(s.scan_lengths) <= 0:
        die(f"{name}: scan_lengths are lengths in Ang, greater than 0")
    for key in ("charge_margin", "carrier_reach", "mace_rattle"):
        if getattr(s, key) < 0:
            die(f"{name}: {key} can't be negative")
    if s.distort_starts < 1:
        die(f"{name}: distort_starts is how many MACE minima C5 starts from: 1 or more")
    if not s.mace and "distortions" not in vals:   # C5 relaxes each one in CASTEP
        s.distortions = list(CASTEP_DISTORTIONS)
    if s.phases_molecule_box <= 0:
        die(f"{name}: phases_molecule_box, the side of the molecules' box, must be above 0")
    if s.ref_polymorph_window < 0:
        die(f"{name}: ref_polymorph_window can't be negative (0: always the lowest polymorph)")
    if not 0 < s.carrier_fraction <= 1:
        die(f"{name}: carrier_fraction is a fraction of the cell's atoms, greater than 0 and at "
            "most 1")
    if s.screen_min_image is not None and s.screen_min_image < 0:
        die(f"{name}: screen_min_image can't be negative (0 switches the Gamma-only floor off)")
    s.converge_kgrids = kgrid_list(name, s.converge_kgrid_min, s.converge_kgrid_max,
                                   s.converge_kgrid_step)
    if s.excite_window and s.excite_window[0] >= s.excite_window[1]:
        die(f"{name}: excite_window is  lowest highest  (eV)")
    if len(set(s.ccd_points)) < 3:
        die(f"{name}: ccd_points needs at least three points (0: the ground state's structure, "
            "1: the excited state's)")
    if s.lineshape_temperature < 0 or s.lineshape_displacements < 0 or s.lineshape_host < 0:
        die(f"{name}: lineshape_temperature, lineshape_displacements and lineshape_host can't be "
            "negative")
    compliance(s.elastic_constants)                # a bad block stops now, not at C8
    return s, raw, runner, blocks, mace_blocks


# A project: its settings, files, directories, checkpoints (state.json), results (results.json
# of each phase) and job runner. main() makes one from the project file and the command-line
# switches, and passes it to the steps.
class Project:
    # Reads the project's two files (path: NAME.parameters; NAME.run beside it) and the state of
    # earlier runs, and checks the runner settings of every step now (rather than when the step
    # starts, perhaps days later), and each MACE step's job template, and that its workers fit on
    # a node. Stops if the work directory has the phase folders of an older dwf.py (old_layout),
    # unless this is --migrate; drops the records of steps since removed (GONE_STEPS).
    #   Called by: main()
    #   Calls:     Project.mace_runner(), Project.mace_settings(), Project.mace_template(),
    #              Project.runner(), Project.save_state(), old_layout(), read_project(),
    #              write_json()
    def __init__(self, path, cl):
        self.path, self.base, self.cl = path, path.parent, cl
        self.run_path = path.with_name(path.stem + ".run")
        (self.s, self.raw, self.runner_project, self.runner_blocks,
         self.mace_blocks) = read_project(path, self.run_path)
        self._runners, self._mace, self._model_ids, self._results = {}, {}, {}, {}
        self._mace_args = None
        for step in RUNNER_STEPS:
            self.runner(step)
        for step in MACE_STEPS if self.s.mace else ():
            m = self.mace_settings(step)
            if m.mace_jobscript:
                self.mace_template(step)
                cpn, n = self.mace_runner().cores_per_node, m.mace_pack or 1
                if cpn and n * m.mace_threads > cpn:
                    die(f"{self.run_path.name}: mace_pack x mace_threads ({n} x "
                        f"{m.mace_threads}) of {STEP_INDEX[step].label}'s MACE jobs is more cores "
                        f"than a node has ({cpn})")
        self.name = self.s.name or path.stem
        self.dir = self.base / (self.s.workdir or f"dwf_{self.name}")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.dir / STATE
        self.state = (json.loads(self.state_file.read_text()) if self.state_file.exists()
                      else {"steps": {}})
        gone = [s for s in GONE_STEPS if s in self.state["steps"]]
        for s in gone:                       # steps this dwf.py no longer has
            del self.state["steps"][s]
            res = self.dir / PHASE_DIRS[GONE_STEPS[s]] / RESULTS
            if res.is_file():
                saved = json.loads(res.read_text())
                if saved.pop(s, None) is not None:
                    write_json(res, saved)
        if gone:
            self.save_state()
            log("dropped the record of " + ", ".join(gone) + " (no longer a step: C4 is now the "
                "distortions, searched with MACE; its folders are left as they are)")
        if old_layout(self.dir) and not getattr(cl, "migrate", False):
            die(f"{self.dir} was made by an older dwf.py, from when phase D held the ground and "
                "excited states (D1-D5's model, D6 rank, D7 excite, D8 validate: now C6-C8) and "
                "production, thermodynamics and lineshapes were phases E, F and G. Once none of "
                f"its jobs from D6 on are queued or running, run  python3 dwf.py {path.stem} "
                "--migrate  to move its folders to the new layout (C3 to C6 run again, after the "
                "steps added since); or finish it with the older dwf.py")

    # A file named by a project keyword (relative to the project file); stops if it is missing.
    #   Called by: Project.cell_text(), Project.production_param(), inputs_hash(), step_converge(),
    #              step_dielectric(), step_relax()
    def file(self, key):
        value = getattr(self.s, key)
        p = self.base / value
        if not p.is_file():
            die(f"{self.path.name}: the {key} file was not found: {p}")
        return p

    # The directory of a phase (A_host, ...), made if needed.
    #   Called by: almost every step
    def phase_dir(self, phase):
        d = self.dir / PHASE_DIRS[phase]
        d.mkdir(parents=True, exist_ok=True)
        return d

    # Writes state.json (through a temporary file, so it is never left half written).
    #   Called by: Project.__init__(), main(), migrate_workdir(), move_old_phases(), reset_step()
    #   Calls:     write_json()
    def save_state(self):
        write_json(self.state_file, self.state)

    # The results of a phase's steps: its results.json, read once ({} if there is none yet).
    #   Called by: Project.out(), Project.save_results(), main(), migrate_workdir()
    def results(self, phase):
        if phase not in self._results:
            f = self.dir / PHASE_DIRS[phase] / RESULTS
            self._results[phase] = json.loads(f.read_text()) if f.exists() else {}
        return self._results[phase]

    # Puts a step's results in its phase's results.json, in the order of the steps (outputs None:
    # takes them out, for --redo). The later steps get them as they will read them back from the
    # file (keys as text, tuples as lists).
    #   Called by: main(), reset_step()
    #   Calls:     Project.results(), write_json()
    def save_results(self, step, outputs):
        phase = STEP_INDEX[step].phase
        res = self.results(phase)
        res.pop(step, None)
        if outputs is not None:
            res[step] = json.loads(json.dumps(outputs, default=float))
        self._results[phase] = {s.name: res[s.name] for s in STEPS
                                if s.name in res and s.phase == phase}
        write_json(self.phase_dir(phase) / RESULTS, self._results[phase])

    # The results of a finished step (from its phase's results.json); stops with what to run if
    # it hasn't finished.
    #   Called by: almost every step
    #   Calls:     Project.results()
    def out(self, step):
        st = STEP_INDEX[step]
        rec = self.state["steps"].get(step)
        if not rec or rec.get("status") != "done":
            die(f"step {st.label} {step} has not finished yet: run  python3 dwf.py "
                f"{self.path.name}  (without step names) to run the steps before this one")
        res = self.results(st.phase)
        if step not in res:
            die(f"{PHASE_DIRS[st.phase]}/{RESULTS} has no results for step {st.label} {step}: "
                f"run it again with  --redo {step}")
        return res[step]

    # Whether a step has finished (with any inputs).
    #   Called by: Project.kspacing(), Project.production_param(), main(), mu_limits()
    def finished(self, step):
        return self.state["steps"].get(step, {}).get("status") == "done"

    # The runner items of a step that runs CASTEP jobs (one of RUNNER_STEPS): the project's runner
    # keywords, changed by its group's defaults (GROUP_DEFAULTS), then by %block runner_<group>,
    # then by %block runner_<step>; each replaces the keywords (and %block partitions) it gives.
    #   Called by: Project.runner()
    #   Calls:     item_key(), parse_castep_input()
    def runner_items(self, step):
        group = RUNNER_STEPS[step]
        layers = [[x for x in parse_castep_input(GROUP_DEFAULTS[group]) if x["type"] != "raw"],
                  self.runner_blocks.get(group, [])]
        if step != group:
            layers.append(self.runner_blocks.get(step, []))
        items = list(self.runner_project)
        for layer in layers:
            given = {item_key(x) for x in layer}
            items = [x for x in items if item_key(x) not in given] + layer
        return items

    # The runner settings of a step (batch_args of its runner_items), with
    # the command-line switches. Steps with the same settings share them, as the steps of a group
    # did before there were step blocks (so e.g. the memory model measured in C2 is used for C5's
    # first jobs when they run in one go).
    #   Called by: Project.__init__(), Project.batch()
    #   Calls:     Project.runner_items(), item_key(), batch_args()
    def runner(self, step):
        items = self.runner_items(step)
        sig = json.dumps([[item_key(x), x.get("value"), x.get("body")] for x in items])
        if sig not in self._runners:
            args = batch_args(items, self.base, f"{self.run_path.name} (runner settings of "
                                                    f"step {STEP_INDEX[step].label} {step})")
            for flag in ("no_wait", "prepare_only", "force", "resubmit_failed"):
                if getattr(self.cl, flag, False):
                    setattr(args, flag, True)
            self._runners[sig] = args
        return self._runners[sig]

    # Runs a list of CASTEP jobs with run_batch in <workdir>/<subdir>, with
    # the runner settings of the step (its name, e.g. 'relax') and the project's STOP file;
    # keep_check : True keeps the jobs' .check files even with delete_check : true (for jobs that
    # later jobs start from). Returns run_batch's result ({'complete', 'jobs', 'calcs'}).
    #   Called by: run_screen_jobs(), step_bands(), step_ccd(), step_converge(), step_dielectric(),
    #              step_excite(), step_filter(), step_lineshape(), step_phases(), step_ref_phases(),
    #              step_relax(), step_relaxations(), step_screen(), step_validate(), step_zpl(),
    #              validate_gamma()
    #   Calls:     Project.cell_text(), Project.runner(), run_batch()
    def batch(self, subdir, jobs, step, param_text, cell_text=None, spacing=None,
              keep_check=False):
        args = self.runner(step)
        if keep_check and args.delete_check:
            args = copy.copy(args)
            args.delete_check = False
        return run_batch(self.dir / subdir, jobs, args,
                            cell_text or self.cell_text(), param_text, spacing=spacing,
                            energy=self.s.energy, stop_file=self.dir / "STOP",
                            label=STEP_INDEX[step].label)

    # The host .cell as given (species_pot, symmetry_generate and anything else every job copies).
    #   Called by: Project.batch(), Project.kspacing(), step_dielectric(), step_relax()
    #   Calls:     Project.file()
    def cell_text(self):
        return self.file("host_cell").read_text()

    # The .param of the production settings: A1's (with the converged cut-off) once it has run,
    # else the project's .param as it is.
    #   Called by: run_screen_jobs(), step_bands(), step_ccd(), step_dielectric(), step_excite(),
    #              step_filter(), step_lineshape(), step_phases(), step_ref_phases(), step_relax(),
    #              step_relaxations(), step_screen(), step_validate(), step_zpl(), validate_gamma()
    #   Calls:     Project.file(), Project.finished()
    def production_param(self):
        if self.finished("converge"):
            return (self.dir / self.out("converge")["production_param"]).read_text()
        return self.file("param").read_text()

    # The production k-point spacing (1/Ang): A1's, else kspacing, else from the host .cell's grid.
    #   Called by: step_bands(), step_ccd(), step_converge(), step_dielectric(), step_lineshape(),
    #              step_phases(), step_ref_phases(), step_relax(), step_relaxations(),
    #              step_validate(), step_zpl()
    #   Calls:     Project.cell_text(), Project.finished(), parse_castep_input(),
    #              read_primitive(), target_spacing()
    def kspacing(self):
        if self.finished("converge"):
            return self.out("converge")["kspacing"]
        if self.s.kspacing:
            return self.s.kspacing
        items = parse_castep_input(self.cell_text())
        prim = read_primitive(items)
        ns = SimpleNamespace(kspacing=None, kgrid=None)
        return target_spacing(items, prim, ns)[0]

    # The relaxed host primitive cell from A2.1 (ASE Atoms, from A_host/host_relaxed.cell).
    #   Called by: d1_dopants(), doped_phases(), run_screen_jobs(), screen_setup(), step_bands(),
    #              step_ccd(), step_choose(), step_corrections(), step_dielectric(),
    #              step_electronic(), step_excite(), step_formation(), step_generate(),
    #              step_interest(), step_limits(), step_lineshape(), step_phases(),
    #              step_ref_phases(), step_relaxations(), step_scan(), step_validate(), step_zpl()
    #   Calls:     load_cell()
    def host(self):
        return load_cell(self, self.out("relax")["cell"])

    # A MACE model as mace-torch takes it: the path of a model file (mace_model, or the model
    # given, relative to the project file), or else a foundation model's name.
    #   Called by: Project.mace(), Project.mace_model_file(), Project.model_id(), mace_jobs()
    def model_path(self, model=None):
        model = str(model or self.s.mace_model)
        for path in (self.base / model, Path(model)):
            if path.is_file():
                return str(path.resolve())
        return model

    # The MACE calculator for a model (default mace_model: a foundation model's name, or a model
    # file), loaded once for each device (mace-torch, on mace_device), for the relaxations run
    # here, with mace_threads threads: the .run file's, or with run (a MACE step's mace_settings)
    # that step's. head: one of a multi-head model's heads, loaded as a MACECalculator.
    #   Called by: mace_relax()
    #   Calls:     Project.model_path()
    def mace(self, model=None, head=None, run=None):
        model = self.model_path(model)
        device = (run and run.mace_device) or self.s.mace_device
        threads = (run and run.mace_threads) or self.s.mace_threads
        if head is not None:
            key = f"{model}#{head}@{device}"
            if key not in self._mace:
                import warnings
                warnings.filterwarnings("ignore")
                try:
                    from mace.calculators import MACECalculator
                except ImportError as err:
                    die(f"a multi-head MACE model needs mace-torch in this Python "
                        f"({sys.executable}): pip install mace-torch ({err})")
                self._mace[key] = MACECalculator(model_paths=model, head=head, device=device,
                                                 default_dtype="float64")
            import torch
            torch.set_num_threads(threads)
            return self._mace[key]
        key = f"{model}@{device}"
        if key not in self._mace:
            import warnings
            warnings.filterwarnings("ignore")
            try:
                import torch
                from mace.calculators import mace_mp
            except ImportError as err:
                die(f"mace : true needs the mace-torch package in this Python ({sys.executable})"
                    f":\n    pip install mace-torch\nor set mace : false ({err})")
            try:
                self._mace[key] = mace_mp(model=model, device=device, default_dtype="float64")
            except Exception as err:          # unknown model, no network for the download
                die(f"could not load the MACE model '{model}': {err}")
        import torch
        torch.set_num_threads(threads)
        return self._mace[key]

    # What identifies a MACE model in the caches: its name, or for a model file its contents
    # (MD5), so a model trained again under the same name isn't mistaken for the old one.
    #   Called by: mace_key()
    #   Calls:     Project.model_path()
    def model_id(self, model=None):
        model = self.model_path(model)
        if model not in self._model_ids:
            path = Path(model)
            self._model_ids[model] = (hashlib.md5(path.read_bytes()).hexdigest()
                                      if path.is_file() else model)
        return self._model_ids[model]

    # The MACE model as a file the MACE jobs can read: mace_model if it is a file; else the
    # foundation model of that name, which mace-torch downloads here (compute nodes may have no
    # network, as on ARCHER2) into its cache in your home directory, copied into
    # <workdir>/mace_models/ (compute nodes may not see your home directory, as on ARCHER2 and
    # Cirrus).
    #   Called by: mace_jobs()
    #   Calls:     Project.model_path()
    def mace_model_file(self):
        model = self.model_path()
        if Path(model).is_file():
            return model
        import contextlib
        import io
        try:
            import warnings
            warnings.filterwarnings("ignore")
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                from mace.calculators import foundations_models as fm
        except ImportError as err:
            die(f"the MACE jobs need the foundation model '{model}' as a file: run dwf.py with a "
                f"Python that has mace-torch (pip install mace-torch), which downloads it, or set "
                f"mace_model to a .model file ({err})")
        names = getattr(fm, "mace_mp_names", None)
        if not hasattr(fm, "download_mace_mp_checkpoint"):
            die("this mace-torch can't fetch foundation models for the MACE jobs: update it (pip "
                "install -U mace-torch), or set mace_model to a .model file")
        if names is not None and model not in names:
            die(f"mace_model '{model}' is neither a file nor a MACE foundation model ("
                + ", ".join(sorted(str(x) for x in names if x)) + ")")
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                src = Path(fm.download_mace_mp_checkpoint(model))
        except Exception as err:              # no network, ...
            die(f"could not download the MACE model '{model}': {err}")
        dest = self.dir / "mace_models" / (re.sub(r"[^\w.-]", "_", model) + ".model")
        if not dest.exists() or dest.stat().st_size != src.stat().st_size:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            log(f"copied the MACE model {model} to {dest} for the MACE jobs")
        return str(dest.resolve())

    # A job template keyword's text (mace_jobscript, model_jobscript: a file relative to the
    # project file, or the text of its %block), checked for its one {COMMAND} line (comment lines
    # don't count), where the command goes.
    #   Called by: run_training()
    def template(self, key):
        kind, value = getattr(self.s, key)
        if kind == "file":
            f = self.base / value
            if not f.is_file():
                die(f"{self.path.name}: the {key} file was not found: {f}")
            value = f.read_text()
        if sum("{COMMAND}" in line and not line.lstrip().startswith("#")
               for line in value.splitlines()) != 1:
            die(f"{self.path.name}: the {key} template needs one line with {{COMMAND}}, where "
                "the command goes")
        return value

    # The settings of a MACE step's jobs (step: one of MACE_STEPS, B2 scan, C1 prerelax or C4
    # distort): the .run file's mace_jobscript, mace_pack, mace_max_jobs, mace_threads, mace_python
    # and mace_device, each replaced by the step's own block (%block runner_scan, runner_prerelax or
    # runner_distort) where it gives one, and its max_walltime (the jobs' --time; without it, the
    # template's). Returns a namespace of MACE_BLOCK_KEYS.
    #   Called by: Project.__init__(), Project.mace_template(), mace_jobs(), mace_python(),
    #              mace_relaxations()
    def mace_settings(self, step):
        own = self.mace_blocks.get(step, {})
        return SimpleNamespace(**{k: own.get(k, getattr(self.s, k, None))
                                  for k in MACE_BLOCK_KEYS})

    # A MACE step's job template (its mace_jobscript, mace_settings: a file relative to the
    # project files, or the text of %block mace_jobscript), checked for its one {COMMAND} line
    # (comment lines don't count), which the job's workers go in place of.
    #   Called by: Project.__init__(), mace_jobs()
    def mace_template(self, step):
        kind, value = self.mace_settings(step).mace_jobscript
        if kind == "file":
            f = self.base / value
            if not f.is_file():
                die(f"{self.run_path.name}: the mace_jobscript file was not found: {f}")
            value = f.read_text()
        if sum("{COMMAND}" in line and not line.lstrip().startswith("#")
               for line in value.splitlines()) != 1:
            die(f"{self.run_path.name}: the mace_jobscript template needs one line with "
                "{COMMAND} where the MACE workers go, e.g.\n    {COMMAND}\n(or with srun in "
                "front, e.g. srun --exact --ntasks=1 --cpus-per-task=8 {COMMAND}, to start each "
                "one with srun)")
        return value

    # The settings the MACE jobs share with the CASTEP jobs: submit_cmd, status_cmd and poll (and
    # cores_per_node, for the number of workers), from the project's runner keywords
    # (batch_args).
    #   Called by: Project.__init__(), mace_jobs(), run_training()
    #   Calls:     batch_args()
    def mace_runner(self):
        if self._mace_args is None:
            self._mace_args = batch_args(self.runner_project, self.base,
                                            f"{self.path.name} (runner settings)")
        return self._mace_args


# --------------------------------------------------------------------------
# Steps, checkpoints and the command line
# --------------------------------------------------------------------------
# One step of the workflow.
# name (e.g. 'relax'), label (e.g. 'A2.1'; its letter is the phase), title, the function that does
# the work (it takes the Project and returns the step's results as a dict, or None if it is not
# finished yet, e.g. jobs still queued with --no-wait; '_note', a one-line summary, and '_table',
# a table for summary.txt, are taken out of the results), and what its results depend on: project
# keywords (keys), files named by project keywords (files) and earlier steps (needs). owns lists
# the paths under the work directory that belong to the step (moved aside by --redo).
class Step:
    # Records the step and its phase. version goes up when a change to dwf.py changes what the step
    # gives from the same jobs, so that work directories made before it run the step again
    # (reusing its finished CASTEP jobs).
    def __init__(self, name, label, title, func, keys=(), files=(), needs=(), owns=(), version=0):
        self.name, self.label, self.title, self.func = name, label, title, func
        self.keys, self.files, self.needs, self.owns = keys, files, needs, owns
        self.phase, self.version = label[0], version


# The fingerprint of what a step's results depend on: its settings (as written in the project
# file), the contents of its files (and, for a step that uses %block phases, of the structure files
# listed there), the result fingerprints of the steps it needs and the step's version (if not 0).
# Stops if one of those steps hasn't finished.
#   Called by: main(), migrate_workdir()
#   Calls:     Project.file()
def inputs_hash(p, step):
    data = {"keys": {k: p.raw.get(k) for k in step.keys}, "files": {}, "needs": {}}
    if step.version:
        data["version"] = step.version
    for f in step.files:
        if getattr(p.s, f):
            data["files"][f] = hashlib.md5(p.file(f).read_bytes()).hexdigest()
    if "phases" in step.keys:
        for row in p.s.phases or []:
            if len(row) > 1 and (p.base / row[1]).is_file():
                data["files"][row[1]] = hashlib.md5((p.base / row[1]).read_bytes()).hexdigest()
    for n in step.needs:
        p.out(n)                                   # stops if it hasn't finished
        data["needs"][n] = p.state["steps"][n]["out_hash"]
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


# The fingerprint of a step's results, which the steps after it depend on: the results themselves
# (as results.json has them, so the fingerprint can be worked out again from it) and the contents
# of every file under the work directory they name (e.g. the structures), so a step that is run
# again with different results makes the later ones run again.
#   Called by: main(), migrate_workdir()
def outputs_hash(p, outputs):
    files = {}

    # Collects the files named anywhere in the results.
    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, str) and len(v) < 300:
            try:
                if (p.dir / v).is_file():
                    files[v] = hashlib.md5((p.dir / v).read_bytes()).hexdigest()
            except OSError:               # text that can't be a file name (e.g. a long message)
                pass

    walk(outputs)
    return hashlib.md5(json.dumps([outputs, files], sort_keys=True,
                                  default=str).encode()).hexdigest()


# A phase's summary, as lines: for each of its steps that has run, a heading, when it finished
# and its one-line note, then its table (main() keeps the notes and tables in state.json); [] if
# none has run.
#   Called by: write_results(), write_summary()
def phase_text(p, phase):
    lines = []
    for s in STEPS:
        rec = p.state["steps"].get(s.name)
        if s.phase != phase or not rec:
            continue
        head = f"{s.label} {s.name}: {s.title}"
        lines += [head, "-" * len(head)]
        if rec.get("status") == "done":
            lines.append(f"finished {rec.get('time', '')}"
                         + (f": {rec['note']}" if rec.get("note") else ""))
            if rec.get("table"):
                lines += ["", rec["table"].rstrip()]
        else:
            lines.append(f"not finished yet ({rec.get('time', '')}): run dwf.py again to carry on")
        lines.append("")
    return lines


# Writes a phase's summary.txt (phase_text), and results.dwf with it.
#   Called by: main(), migrate_workdir(), move_old_phases(), reset_step()
#   Calls:     phase_text(), write_results()
def write_summary(p, phase):
    lines = [f"Project {p.name}, phase {phase}: {PHASE_TITLES[phase]}",
             f"(written by dwf.py; the numbers the later steps use are in {RESULTS})", ""]
    (p.phase_dir(phase) / SUMMARY).write_text("\n".join(lines + phase_text(p, phase)))
    write_results(p)


# Writes results.dwf in the work directory: everything in one file, as a .castep is for a CASTEP
# job. It is written again whenever a phase's summary.txt is (each step that finishes or stops),
# at the end of every run and by --status: the project and its two files, every step's status
# (status_lines), the keywords those files give (as written; the rest are at their defaults), then
# the summary of each phase that has run (phase_text), A to F. No step reads it.
#   Called by: main(), print_status(), write_summary()
#   Calls:     item_key(), phase_text(), block_rows(), status_lines()
def write_results(p):
    title = f"dwf.py results: project {p.name}"
    lines = [title, "=" * len(title), "",
             f"project files   {p.path.resolve()}"
             + (f", {p.run_path.name}" if getattr(p, "run_path", None) else ""),
             f"work directory  {p.dir.resolve()}/",
             f"written         {time.strftime('%Y-%m-%d %H:%M')}, "
             "and again whenever a step finishes",
             f"phase folders   each one's {SUMMARY} has its part of this, "
             f"its {RESULTS} the numbers", "",
             "Steps", "-----"] + status_lines(p)

    # A keyword's value as the project file gives it: a block of settings as its lines (without
    # their comments) joined with ' / ', a job-script template written in a block as its length.
    def given(k, v):
        if v != "(%block)":
            return v
        blocks = getattr(p, "runner_blocks", {})
        if k.startswith("runner_") and k[len("runner_"):] in blocks:
            return " / ".join(f"{item_key(x)} : {x.get('value', '')}"
                              for x in blocks[k[len("runner_"):]])
        it = next((x for x in getattr(p, "runner_project", []) if item_key(x) == k
                   and x["type"] == "block"), None)
        if it is None or "jobscript" in k:
            return v if it is None else f"(%block, {len(it['body'])} lines)"
        return " / ".join(" ".join(r) for r in block_rows(it))

    head = "Settings: the keywords the project's files give (the rest are at their defaults)"
    width = max((len(k) for k in p.raw), default=0)
    lines += ["", head, "-" * len(head)] + [f"  {k:<{width}} : {given(k, v)}"
                                            for k, v in p.raw.items()]
    for phase in PHASE_DIRS:
        body = phase_text(p, phase)
        if body:
            head = f"Phase {phase}: {PHASE_TITLES[phase]} ({PHASE_DIRS[phase]}/)"
            lines += ["", "=" * 100, head, "=" * 100, ""] + body
    (p.dir / OVERVIEW).write_text("\n".join(lines).rstrip() + "\n")


# Every step, in order under its phase, with its status: when it finished and its note, still
# running (jobs queued), or not run yet ('-').
#   Called by: print_status(), write_results()
def status_lines(p):
    lines, phase, w = [], None, max(len(s.name) for s in STEPS)
    for s in STEPS:
        if s.phase != phase:
            phase = s.phase
            lines.append(f"  {phase}: {PHASE_TITLES[phase]}  ({PHASE_DIRS[phase]}/)")
        rec = p.state["steps"].get(s.name, {})
        lines.append(f"    {s.label:<5} {s.name:<{w}} {rec.get('status', '-'):<8} "
                     f"{rec.get('time', ''):<16}  {rec.get('note', '')}".rstrip())
    return lines


# Prints every step's status (status_lines), and brings results.dwf up to date.
#   Called by: main()
#   Calls:     status_lines(), write_results()
def print_status(p):
    print(f"\nProject {p.name}: {p.path} -> {p.dir}/\n")
    print("\n".join(status_lines(p)) + "\n")
    write_results(p)
    print(f"(results.dwf: {p.dir / OVERVIEW})\n")


# Moves a step's files aside (<path>.old-<time>) and forgets that it ran, and its results: --redo.
# An owned path with a * in it (e.g. C_screen/jobs/share_*) moves every match, except earlier
# .old- copies.
#   Called by: main()
#   Calls:     Project.save_results(), Project.save_state(), write_summary()
def reset_step(p, step):
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for rel in step.owns:
        paths = ([x for x in sorted(p.dir.glob(rel)) if ".old-" not in x.name] if "*" in rel
                 else [p.dir / rel])
        for path in paths:
            if path.exists():
                path.rename(path.with_name(f"{path.name}.old-{stamp}"))
                log(f"{step.label} {step.name}: moved {path.relative_to(p.dir)} to "
                    f"{path.name}.old-{stamp} (delete it when you no longer need it)")
    p.state["steps"].pop(step.name, None)
    p.save_state()
    p.save_results(step.name, None)
    write_summary(p, step.phase)


# Whether a work directory was made by a dwf.py from when phase D held the ground and excited
# states (D1-D5, D6 rank, D7 excite, D8 validate) and hasn't been migrated: it has one of the
# phase folders that have been renamed since (OLD_PHASE_DIRS), state.json has a record of one of
# D1-D5 (OLD_STEPS), or D_states/results.json has D6's, D7's or D8's results. (D_states/ itself
# stays after --migrate, with what D1-D5 made.)
#   Called by: Project.__init__(), migrate_workdir()
def old_layout(d):
    if any((d / old).is_dir() for old in OLD_PHASE_DIRS):
        return True
    state, res = d / STATE, d / OLD_D_DIR / RESULTS
    steps = json.loads(state.read_text()).get("steps", {}) if state.is_file() else {}
    moved = json.loads(res.read_text()) if res.is_file() else {}
    return bool(set(OLD_STEPS) & set(steps) or {"rank", "excite", "validate"} & set(moved))


# --migrate: brings a work directory made while phase D held the ground and excited states
# (old_layout) up to date, unless a step from C6 on (or one of D1-D5) is still running (its jobs
# may be queued: it should finish with the dwf.py that started it). Moves C6's, C7's and C8's jobs
# and structures (OLD_D_PATHS) from D_states/ to C_screen/, and their results from
# D_states/results.json to C_screen/results.json; renames E_production/ to D_production/,
# F_thermo/ to E_thermo/ and G_lineshape/ to F_lineshape/ (OLD_PHASE_DIRS). The paths in every
# results.json and in state.json are changed to match (and E4's for_G becomes for_F), the records
# of the steps that are gone (OLD_STEPS: D1-D5) are dropped from state.json, the others' labels
# updated, and every phase's summary.txt is written again. D_states/ keeps what D1-D5 made, which
# nothing reads any more. The competing phases at the production settings move from A_host/ to
# E_thermo/ (move_old_phases). The checkpoints are brought up to date with the new paths. What
# then runs again: the steps added since (A3.1, A3.2: the references at the screening
# settings), C3 (it cuts against them) and so C4 to C6 (C4 is now a search with MACE, whose
# minima C5 relaxes too; C6 no longer waits for D5), and E1.1 and E1.2 (now only for the dopants
# that reach D). The steps after C6 run again only if its results change.
#   Called by: main()
#   Calls:     Project.results(), Project.save_state(), inputs_hash(), move_old_phases(),
#              old_layout(), outputs_hash(), write_json(), write_summary()
def migrate_workdir(p):
    d = p.dir
    if not old_layout(d):
        if not move_old_phases(p):
            log(f"{d}: already in the layout of this dwf.py - nothing to migrate")
        return
    first = STEPS.index(STEP_INDEX["rank"])
    busy = [f"{s.label} {s.name}" for s in STEPS[first:]
            if p.state["steps"].get(s.name, {}).get("status") == "running"]
    busy += [name for name in OLD_STEPS
             if p.state["steps"].get(name, {}).get("status") == "running"]
    if busy:
        die(f"{', '.join(busy)} still running (its jobs may be queued): let it finish with the "
            "dwf.py that started it, then --migrate")
    for old, new in OLD_PHASE_DIRS.items():
        if (d / old).is_dir() and (d / new).exists():
            die(f"{d} has both {old}/ and {new}/: move one of them aside first")
    for old, new in OLD_PHASE_DIRS.items():
        if (d / old).is_dir():
            (d / old).rename(d / new)
            log(f"moved {old}/ to {new}/")
    src, dst = d / OLD_D_DIR, d / PHASE_DIRS["C"]
    for pattern in OLD_D_PATHS:
        for x in sorted(src.glob(pattern)):
            target = dst / x.relative_to(src)
            if target.exists():
                die(f"{d}: both {x.relative_to(d)} and {target.relative_to(d)} exist: move one "
                    "of them aside first")
            target.parent.mkdir(parents=True, exist_ok=True)
            x.rename(target)
            log(f"moved {x.relative_to(d)} to {target.relative_to(d)}")
    subs = ([(f"{old}/", f"{new}/") for old, new in OLD_PHASE_DIRS.items()]
            + [(f"{OLD_D_DIR}/{x.rstrip('*')}", f"{PHASE_DIRS['C']}/{x.rstrip('*')}")
               for x in OLD_D_PATHS] + [('"for_G"', '"for_F"')])
    for f in [*sorted(d.glob(f"*/{RESULTS}")), d / STATE]:
        if f.is_file():
            text = f.read_text()
            for a, b in subs:
                text = text.replace(a, b)
            f.write_text(text)
    res_c = (json.loads((dst / RESULTS).read_text()) if (dst / RESULTS).is_file() else {})
    old_f = src / RESULTS
    res_d = json.loads(old_f.read_text()) if old_f.is_file() else {}
    for name in ("rank", "excite", "validate"):
        if name in res_d:
            res_c[name] = res_d.pop(name)
    dst.mkdir(parents=True, exist_ok=True)
    write_json(dst / RESULTS, {s.name: res_c[s.name] for s in STEPS if s.name in res_c})
    if old_f.is_file():
        write_json(old_f, res_d)
    p.state = json.loads(p.state_file.read_text()) if p.state_file.is_file() else {"steps": {}}
    for name in OLD_STEPS:
        p.state["steps"].pop(name, None)
    for name, rec in p.state["steps"].items():
        if name in STEP_INDEX:
            rec["label"] = STEP_INDEX[name].label
    p._results = {}
    move_old_phases(p)

    # The checkpoints of C6 and the steps after it, whose results now name the new folders: which
    # were up to date (their inputs fingerprint, worked out as this dwf.py does with the earlier
    # steps' results fingerprints as they were), then each one's results fingerprint again from
    # its results as they are now, and the inputs fingerprint again of each that was up to date,
    # so it stays so. C6 isn't (it no longer needs D5, and comes after C3 to C5, which run again):
    # it runs again, and if it gives the same results (from its finished jobs) nothing after it
    # does.
    done = [s for s in STEPS[first:] if p.state["steps"].get(s.name, {}).get("status") == "done"]
    fresh = []
    for s in done:
        try:
            if inputs_hash(p, s) == p.state["steps"][s.name].get("inputs"):
                fresh.append(s)
        except SystemExit:                 # one of the steps it needs hasn't finished
            pass
    for s in done:
        res = p.results(s.phase)
        if s.name in res:
            p.state["steps"][s.name]["out_hash"] = outputs_hash(p, res[s.name])
    for s in fresh:
        p.state["steps"][s.name]["inputs"] = inputs_hash(p, s)
    p.save_state()
    for phase in PHASE_DIRS:
        if (d / PHASE_DIRS[phase]).is_dir():
            write_summary(p, phase)
    log(f"{d}: migrated to the phases of this dwf.py (C screening, ground and excited states; "
        "D production; E thermodynamics; F lineshapes); D_states/ keeps what D1-D5 made (no "
        "longer used) - run it as usual")


# The competing phases at the production settings were A3.1 and A3.2 (phases, limits) before they
# became E1.1 and E1.2: a work directory from then has their jobs in A_host/jobs/phases and their
# results in A_host/results.json. Moves both to E_thermo/, where E1.1 reuses the finished jobs (the
# step itself runs again: it now relaxes only the phases of the dopants that reach D), and writes
# the two phases' summaries again. If the old A3.1 was still running, its record is dropped and
# any of its jobs still queued should be cancelled (they would start in the folder it has left;
# E1.1 runs again those it needs). Returns whether it moved anything.
#   Called by: main(), migrate_workdir()
#   Calls:     Project.save_state(), write_json(), write_summary()
def move_old_phases(p):
    a, e = p.dir / PHASE_DIRS["A"], p.dir / PHASE_DIRS["E"]
    jobs, target = a / "jobs" / "phases", e / "jobs" / "phases"
    moved = False
    if jobs.is_dir() and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        jobs.rename(target)
        moved = True
        log(f"moved {jobs.relative_to(p.dir)}/ to {target.relative_to(p.dir)}/ (the competing "
            "phases at the production settings are E1.1's now)")
        if p.state["steps"].get("phases", {}).get("status") == "running":
            p.state["steps"].pop("phases")
            p.save_state()
            log("WARNING: the old A3.1 phases was still running: cancel any of its jobs that are "
                "still in the queue (this dwf.py relaxes only the phases of the dopants that "
                "reach D, at E1.1; if it then finds one of them failed, rerun with "
                "--resubmit-failed)")
    f = a / RESULTS
    res = json.loads(f.read_text()) if f.is_file() else {}
    old = {k: res.pop(k) for k in ("phases", "limits") if k in res}
    if old:
        write_json(f, res)
        ef = e / RESULTS
        er = json.loads(ef.read_text()) if ef.is_file() else {}
        for k, v in old.items():
            er.setdefault(k, v)
        e.mkdir(parents=True, exist_ok=True)
        write_json(ef, {st.name: er[st.name] for st in STEPS if st.name in er})
        p._results = {}
        moved = True
        log(f"moved the results of phases and limits from {f.relative_to(p.dir)} to "
            f"{ef.relative_to(p.dir)}")
    if moved:
        for phase in "AE":
            write_summary(p, phase)
    return moved


# --------------------------------------------------------------------------
# A project's two files: --example, and --split for an older dwf.py's one
# --------------------------------------------------------------------------
# What an older dwf.py's step labels are now, for the comments --split keeps: C4 and C5 are
# C4 and C5 still (distort, drelax), D6-D8 became C6-C8, and E, F and G became D, E and F
# (their phase folders too), with E1-E3 now E2-E4 (E1 is the competing phases at the production
# settings); D1-D5 were removed, and their mentions become 'old D1' and so on.
OLD_LABELS = {"D6": "C6", "D7": "C7", "D8": "C8", "E1": "D1", "E2": "D2", "F1": "E2",
              "F2": "E3", "F3": "E4", "G1": "F1", "G2": "F2"}
OLD_PHASES = {"D": "C", "E": "D", "F": "E", "G": "F"}
OLD_FOLDERS = {"D_states": "C_screen", **OLD_PHASE_DIRS}


# A comment of an older project file with its step labels, phases and phase folders as they are
# now (OLD_LABELS, OLD_PHASES, OLD_FOLDERS), and a section heading's phase letter (# ===== E:
# production becomes D: production). A phase is a letter after 'phase' or 'phases', or alone
# after 'to', 'for', 'in', 'into', 'from' or 'of' (goes on to E: D now); other single letters
# are left, as they may be something else (its E as it is: an energy). Only the comment of a
# line changes: all of a line starting with # or !, else what follows its first ' #' or ' !'.
#   Called by: split_text()
def relabel_comment(line):
    m = re.match(r"(#\s*={3,}\s+)([D-G])(:.*)", line)
    if m:
        return m.group(1) + OLD_PHASES[m.group(2)] + m.group(3)
    s = line.lstrip()
    at = 0 if s[:1] in "#!" else min((k for k in (line.find(" #"), line.find(" !")) if k >= 0),
                                     default=-1)
    if at < 0:
        return line

    # One match: a range of removed steps, a step label, a phase or a folder.
    def sub(m):
        x = m.group(0)
        if m.group("range"):
            return f"old {x}"
        if m.group("label"):
            return OLD_LABELS.get(x, f"old {x}" if x[0] == "D" and x[1] in "12345" else x)
        if m.group("phase"):
            return m.group("word") + OLD_PHASES[m.group("letter")]
        if m.group("bare"):
            return m.group("pre") + OLD_PHASES[m.group("alone")]
        return OLD_FOLDERS[x]

    pat = (r"(?P<range>\bD[1-5]\s*-\s*D[1-5]\b)|(?P<label>(?<![\w.])[C-G][0-9](?!\w))"
           r"|(?P<phase>(?P<word>\bphases? )(?P<letter>[D-G])\b)"
           r"|(?P<bare>(?P<pre>\b(?:to|for|in|into|from|of) )(?P<alone>[D-G])(?![\w-]))|"
           + "|".join(re.escape(k) for k in OLD_FOLDERS))
    return line[:at] + re.sub(pat, sub, line[at:]).replace("phases A to G", "phases A to F")


# Splits the text of an older dwf.py's project file (everything in one, NAME.dwf) into the two a
# project now has: (NAME.parameters text, NAME.run text, the keywords left out). Each keyword or
# block - also one commented out with ! (an optional setting; a line starting with # is prose) -
# goes with the comment lines just above it and the indented comment lines just below it: to the
# .run text if it says how jobs run (RUNNER_KEYS, RUN_PROJECT_KEYS, the runner blocks), else to
# the .parameters text; comment lines just above a section heading (# ===...) stay with the
# keyword before them. One this dwf.py no longer has (RETIRED, e.g. D1-D5's) is left out, any
# section heading above it going to the next keyword of the .parameters text. Comment lines after
# the last keyword go to the .parameters text. Comments get the step labels of this dwf.py
# (relabel_comment).
#   Called by: split_project()
#   Calls:     relabel_comment()
def split_text(text):
    run_keys = RUNNER_KEYS | RUN_PROJECT_KEYS
    known = set(PROJECT_KEYWORDS) | run_keys | set(RETIRED)
    lines = text.splitlines()
    units, pre, i = [], [], 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        commented = s[:1] in "#!"
        bare = s[1:].strip() if s[:1] == "!" else s
        mb = re.match(r"%block\s+(\w+)", bare, re.I)
        mk = re.match(r"([A-Za-z_]\w*)\s*[:=]" if commented else r"([A-Za-z_]\w*)", bare)
        if not s or (commented and line[:1].isspace()):
            (pre if pre or not units or not s else units[-1]["lines"]).append(line)
            i += 1
            continue
        name = (mb or mk).group(1).lower() if (mb or mk) else ""
        if mb and (not commented or (s[:1] == "!" and (name in known
                                                        or name.startswith("runner_")))):
            key, body = mb.group(1).lower(), [line]
            i += 1
            while i < len(lines):
                body.append(lines[i])
                i += 1
                if re.match(r"[#!]?\s*%endblock\b", body[-1].strip(), re.I):
                    break
        elif mk and (not commented or (s[:1] == "!" and mk.group(1).lower() in known)):
            key, body = mk.group(1).lower(), [line]
            i += 1
        else:
            pre.append(line)
            i += 1
            continue
        while i < len(lines) and lines[i][:1].isspace() and lines[i].strip()[:1] in "#!":
            body.append(lines[i])
            i += 1
        head = next((k for k, ln in enumerate(pre) if re.match(r"#\s*={3,}", ln.strip())), None)
        if head and units:                 # what comes before a section heading ends the last one
            units[-1]["lines"] += pre[:head]
            pre = pre[head:]
        units.append({"key": key, "lines": pre + body})
        pre = []
    params, run, dropped, carry = [], [], [], []
    for u in units:
        if u["key"] in RETIRED:
            dropped.append(u["key"])
            carry += [ln for ln in u["lines"] if re.match(r"#\s*={3,}", ln.strip())]
        elif u["key"] in run_keys or u["key"].startswith("runner_"):
            run.extend(u["lines"])
        else:
            params.extend(([""] if carry else []) + carry + u["lines"])
            carry = []
    params += carry + pre

    # The lines as text: comments relabelled, no more than one blank line in a row.
    def text_of(out):
        out = [relabel_comment(ln) for ln in out]
        keep = [ln for k, ln in enumerate(out) if ln.strip() or (k and out[k - 1].strip())]
        while keep and not keep[0].strip():
            keep.pop(0)
        return "\n".join(keep).rstrip() + "\n"

    return text_of(params), text_of(run), dropped


# --split OLD.dwf: writes OLD.parameters and OLD.run beside it from an older dwf.py's project file
# (split_text), leaving it as it is; stops if either exists already. Each starts with a note of
# where it came from and what was left out.
#   Called by: main()
#   Calls:     split_text()
def split_project(path):
    if not path.is_file():
        die(f"no file {path}")
    outs = [path.with_suffix(s) for s in (".parameters", ".run")]
    for f in outs:
        if f.exists():
            die(f"{f} exists already: move it aside first (nothing was written)")
    params, run, dropped = split_text(path.read_text())
    stem = path.stem
    for out, other, body in ((outs[0], outs[1], params), (outs[1], outs[0], run)):
        note = (f"Made by dwf.py --split from {path.name} ({time.strftime('%Y-%m-%d')}), whose "
                f"other half is {other.name}. The comments' step labels are this dwf.py's: "
                "D6-D8 are now C6-C8, phases E, F and G are now D, E and F, and E's steps are "
                "E2-E4. D1-D5 (the ground-state model) were removed: the comments call them old "
                "D1 to old D5.")
        if dropped:
            note += (" Left out, as this dwf.py doesn't have them: " + ", ".join(dropped)
                     + f" (see {path.name}).")
        lines = textwrap.wrap(note, 98, initial_indent="# ", subsequent_indent="# ",
                              break_on_hyphens=False)
        out.write_text("\n".join(lines) + "\n\n" + body)
    log(f"wrote {outs[0].name} (the science) and {outs[1].name} (how the jobs run) from "
        f"{path.name}" + (f"; left out: {', '.join(dropped)}" if dropped else "")
        + f". Check them, then run  python3 dwf.py {stem}")


# --example NAME: writes NAME.parameters and NAME.run, the annotated example project (CaO:Sb on
# Cirrus: EXAMPLE_PARAMETERS, EXAMPLE_RUN); stops if either exists already.
#   Called by: main()
def write_example(stem):
    outs = [Path(stem + s) for s in (".parameters", ".run")]
    for f in outs:
        if f.exists():
            die(f"{f} exists already: give another name (nothing was written)")
    outs[0].write_text(EXAMPLE_PARAMETERS)
    outs[1].write_text(EXAMPLE_RUN)
    log(f"wrote {outs[0]} and {outs[1]}: the example project (CaO:Sb on Cirrus). Edit them, "
        f"then run  python3 dwf.py {stem}")


# The .parameters file of the project a command-line word names: NAME, NAME.parameters or
# NAME.run, with its NAME.run beside it. Stops if either is missing; an older dwf.py's project
# file (NAME.dwf) with how to split it. NAME.dwf named when the pair is there: a note that the
# pair is what is read.
#   Called by: main()
#   Calls:     log()
def project_path(word):
    base = Path(word)
    if base.suffix in (".parameters", ".run", ".dwf"):
        base = base.with_suffix("")
    params, run, old = (base.with_name(base.name + s) for s in (".parameters", ".run", ".dwf"))
    if not params.is_file() and not run.is_file():
        if old.is_file():
            die(f"{old} is a project file of an older dwf.py: a project is now two files, "
                f"{params.name} (the science) and {run.name} (how the jobs run). Make them from "
                f"it with  python3 dwf.py --split {old}")
        die(f"no project {word}: dwf.py reads {params} and {run} (--example NAME writes a pair to "
            "start from)")
    for f in (params, run):
        if not f.is_file():
            die(f"no {f}: a project is two files, {params.name} and {run.name}")
    if Path(word).suffix == ".dwf":
        log(f"note: reading {params.name} and {run.name}; {old.name}, an older dwf.py's project "
            "file, is not read (edit the pair)")
    return params


# The command line, and the work it does: runs the steps in order (all of them, or those named by
# name, label or phase letter, up to --until), skipping those that have finished with the same
# inputs, and stops at the first that isn't finished (jobs still queued with --no-wait). A step
# that finishes has its results put in its phase's results.json, its checkpoint (with its note
# and table) in state.json, and its phase's summary.txt written again. A project is named by
# NAME (its NAME.parameters and NAME.run: project_path). --status prints the checkpoints; --redo
# starts steps again from scratch; --example NAME writes an annotated project (write_example);
# --split OLD.dwf makes the two files from an older dwf.py's one (split_project); --migrate
# updates a work directory made by an older dwf.py (migrate_workdir); --mace-worker and
# --phonon-worker are what a MACE or F2 phonon job runs (mace_worker, phonon_worker), not a project.
# --conv NAME and --batch FILE run the job runner (PART 1) on its own (runner_args: series_main,
# batch_main), and --conv-example prints its annotated .conv file.
#   Called by: the main block
#   Calls:     Project.__init__(), Project.finished(), Project.results(), Project.save_results(),
#              Project.save_state(), batch_main(), inputs_hash(), mace_worker(), migrate_workdir(),
#              move_old_phases(), outputs_hash(), phonon_worker(), print_status(), project_path(),
#              reset_step(), runner_args(), series_main(), split_project(), write_example(),
#              write_results(), write_summary()
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project", nargs="?", help="the project's name: NAME, for its two files "
                    "NAME.parameters and NAME.run")
    ap.add_argument("steps", nargs="*", help="run only these steps: names (relax), labels (A2.1) "
                    "or phases (A to F); default: all of them")
    ap.add_argument("--until", metavar="STEP", help="stop after this step or phase (e.g. B)")
    ap.add_argument("--redo", action="append", default=[], metavar="STEP",
                    help="start this step again from scratch (can be given more than once)")
    ap.add_argument("--status", action="store_true",
                    help="show which steps have finished (and bring results.dwf up to date)")
    ap.add_argument("--example", nargs="?", const="example", metavar="NAME",
                    help="write an annotated example project, NAME.parameters and NAME.run")
    ap.add_argument("--split", metavar="OLD.dwf",
                    help="make OLD.parameters and OLD.run from an older dwf.py's project file")
    ap.add_argument("--prepare-only", action="store_true",
                    help="write the inputs of the next CASTEP jobs, submit nothing")
    ap.add_argument("--no-wait", action="store_true",
                    help="submit what is needed and exit instead of waiting; rerun later")
    ap.add_argument("--resubmit-failed", action="store_true",
                    help="start failed CASTEP jobs again, and MACE relaxations whose jobs "
                    "gave up")
    ap.add_argument("--force", action="store_true",
                    help="keep the results of jobs whose inputs have changed (--conv: reuse "
                    "a work directory set up with other inputs)")
    ap.add_argument("--migrate", action="store_true",
                    help="move the folders of a work directory made by an older dwf.py (when "
                    "phase D held D1-D8) to this version's layout")
    ap.add_argument("--conv", metavar="NAME",
                    help="the job runner on its own: the supercell size series of one defect "
                    "in NAME.conv (or the file itself)")
    ap.add_argument("--conv-example", action="store_true",
                    help="print an annotated example .conv file, with every runner keyword")
    ap.add_argument("--batch", metavar="FILE",
                    help="the job runner on its own: run the CASTEP calculations a batch file "
                    "lists")
    ap.add_argument("--mace-worker", metavar="TASKS",
                    help="(what a MACE job runs) relax the structures in this task file")
    ap.add_argument("--phonon-worker", metavar="SPEC",
                    help="(what an F2 job runs) a defect's force constants with its model")
    cl = ap.parse_args()
    if cl.conv_example:
        print(EXAMPLE_INPUT, end="")
        return
    if cl.conv or cl.batch:
        if cl.project or (cl.conv and cl.batch):
            ap.error("--conv NAME and --batch FILE run the job runner on its own: give one of "
                     "them, without a project")
        args = runner_args(cl)
        return batch_main(args) if cl.batch else series_main(args)
    if cl.mace_worker:
        mace_worker(cl.mace_worker)
        return
    if cl.phonon_worker:
        phonon_worker(cl.phonon_worker)
        return
    if cl.example:
        write_example(cl.example)
        return
    if cl.split:
        split_project(Path(cl.split))
        return
    if not cl.project:
        ap.error("give the project's name, NAME for NAME.parameters and NAME.run (or --example "
                 "NAME for an annotated pair)")
    p = Project(project_path(cl.project), cl)
    if cl.migrate:
        migrate_workdir(p)
        return
    move_old_phases(p)
    if cl.status:
        print_status(p)
        return

    # The steps a word stands for: a step's name or label, or a phase letter (all its steps).
    def find(word):
        w = word.lower()
        hits = [s for s in STEPS if w in (s.name, s.label.lower(), s.phase.lower())]
        if not hits:
            die(f"no step or phase '{word}': the steps are "
                + ", ".join(f"{s.label} {s.name}" for s in STEPS) + ", the phases A to F")
        return hits

    chosen = {s.name for w in cl.steps for s in find(w)} or {s.name for s in STEPS}
    last = max(STEPS.index(s) for s in find(cl.until)) + 1 if cl.until else len(STEPS)
    todo = [s for s in STEPS[:last] if s.name in chosen]
    redo = {s.name for w in cl.redo for s in find(w)}
    if not todo:
        die("no steps to run (check the step names and --until)")
    for name in sorted(redo - {s.name for s in todo}):
        die(f"--redo {name}: that step is not among the steps being run")
    print(f"\ndwf.py: the defect workflow\nProject {p.name}: {p.path} -> {p.dir}/")
    if not (cl.prepare_only or cl.no_wait):
        log(f"running on {socket.gethostname()} as process {os.getpid()}; to stop it from "
            f"any login node: touch {(p.dir / 'STOP').absolute()}")
    phase = None
    try:
        for step in todo:
            if step.phase != phase:
                phase = step.phase
                log(f"==== phase {phase}: {PHASE_TITLES[phase]} ({PHASE_DIRS[phase]}/)")
            if step.name in redo:
                reset_step(p, step)
            h = inputs_hash(p, step)
            rec = p.state["steps"].get(step.name, {})
            if rec.get("status") == "done" and rec.get("inputs") == h:
                log(f"{step.label} {step.name}: done already ({rec.get('note', '')})")
                continue
            if rec.get("status") == "done":
                log(f"{step.label} {step.name}: its settings or inputs have changed since it ran, "
                    "so it runs again (CASTEP jobs whose own inputs are unchanged are reused)")
            log(f"---- {step.label} {step.name}: {step.title}")
            try:
                outputs = step.func(p)
            except SystemExit as err:
                if "have changed since it ran" in str(err.code):
                    print(f"(to start step {step.name} again from scratch with the current "
                          f"settings: --redo {step.name})", file=sys.stderr)
                raise
            if outputs is None:
                p.state["steps"][step.name] = dict(rec, status="running", label=step.label,
                                                   time=time.strftime("%Y-%m-%d %H:%M"))
                p.save_state()
                write_summary(p, step.phase)
                log(f"{step.label} {step.name} is not finished yet - rerun the same command to "
                    "carry on")
                return
            note, table = outputs.pop("_note", ""), outputs.pop("_table", None)
            if table:
                print(table)
            p.save_results(step.name, outputs)
            p.state["steps"][step.name] = {
                "status": "done", "label": step.label, "inputs": h,
                "out_hash": outputs_hash(p, p.results(step.phase)[step.name]),
                "time": time.strftime("%Y-%m-%d %H:%M"), "note": note, "table": table}
            p.save_state()
            write_summary(p, step.phase)
            log(f"{step.label} {step.name}: finished" + (f" - {note}" if note else ""))
    except KeyboardInterrupt:
        print("\nInterrupted. Submitted jobs keep running - rerun the same command to carry on.")
        return
    write_results(p)
    if all(p.finished(s.name) for s in todo):
        log((f"step {todo[0].label}" if len(todo) == 1 else
             f"steps {todo[0].label} to {todo[-1].label}")
            + f" done: {p.dir / OVERVIEW} has the results (each phase's summary.txt its part)")


# --------------------------------------------------------------------------
# Structures
# --------------------------------------------------------------------------
# An ASE Atoms object as plain data (species, cell, fractional positions, starting spins), for the
# MACE caches.
#   Called by: mace_jobs(), mace_key(), phonon_worker(), save_mace_result(), step_lineshape()
def atoms_to_dict(atoms):
    return {"symbols": atoms.get_chemical_symbols(),
            "cell": np.asarray(atoms.cell).round(10).tolist(),
            "frac": atoms.get_scaled_positions(wrap=False).round(10).tolist(),
            "magmoms": atoms.get_initial_magnetic_moments().round(6).tolist()}


# The reverse of atoms_to_dict.
#   Called by: mace_cached(), mace_worker(), phonon_worker()
def atoms_from_dict(d):
    a = Atoms(d["symbols"], scaled_positions=d["frac"], cell=d["cell"], pbc=True)
    a.set_initial_magnetic_moments(d.get("magmoms") or [0.0] * len(a))
    return a


# Saves a structure as a CASTEP .cell file (a comment line, lattice_cart, positions_frac with
# SPIN= where a starting spin is set), the atoms in their own order so that reading it back
# (load_cell) gives the same atom numbers. Only the structure: each job adds its k-points,
# species_pot and the rest. Returns the path relative to the work directory, as the results give
# it.
#   Called by: dscf_result(), job_summary(), production_summary(), step_distort(), step_prerelax()
def save_cell(p, path, atoms, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    frac = atoms.get_scaled_positions(wrap=True)
    frac[np.isclose(frac, 0.0, atol=1e-10) | np.isclose(frac, 1.0, atol=1e-10)] = 0.0
    lines = [f"# {title}", "", "%block lattice_cart", "ang"]
    lines += ["  " + "".join(f"{x:20.12f}" for x in v) for v in np.asarray(atoms.cell)]
    lines += ["%endblock lattice_cart", "", "%block positions_frac"]
    for s, f, m in zip(atoms.get_chemical_symbols(), frac, atoms.get_initial_magnetic_moments()):
        lines.append(f"  {s:<3s}" + "".join(f"{x:18.12f}" for x in f)
                     + (f"  SPIN={m:.4f}" if abs(m) > 1e-8 else ""))
    path.write_text("\n".join(lines + ["%endblock positions_frac"]) + "\n")
    return str(path.relative_to(p.dir))


# Reads a structure saved by save_cell (rel: relative to the work directory), as ASE Atoms with
# the starting spins as initial magnetic moments.
#   Called by: Project.host(), distinct_states(), drelax_ranking(), duplicates(), into_cell(),
#              lineshape_vector(), run_screen_jobs(), step_ccd(), step_distort(), step_excite(),
#              step_filter(), step_lineshape(), step_rank(), step_screen(), step_validate(),
#              step_zpl(), validate_gamma()
#   Calls:     read_structure()
def load_cell(p, rel):
    return read_structure(p.dir / rel)


# Reads a structure file of yours: a CASTEP .cell (keeping SPIN= values) or anything ASE reads.
#   Called by: phase_list(), step_converge(), step_relax()
#   Calls:     read_structure()
def read_structure_file(path):
    return read_structure(Path(path))


# 0 or 1: whether a structure with charge q has an even or odd number of electrons. Core shells
# are full, so the parity of the valence electrons (whatever the pseudopotentials) is that of the
# sum of the atomic numbers, minus the charge.
#   Called by: spin_states(), start_spin()
def electron_parity(atoms, q):
    return int(sum(atomic_numbers[s] for s in atoms.get_chemical_symbols()) - q) % 2


# The starting spins (unpaired electrons, CASTEP's spin) of the screening jobs of a structure with
# charge q (screen_spins): triplet, one run from a triplet (2), or a doublet (1) if the electron
# count is odd; low, one run from the lowest spin the electron count allows (0 or 1); both, two
# runs, that and that + 2. A starting value only: the spin is never held fixed.
#   Called by: step_screen()
#   Calls:     electron_parity()
def spin_states(atoms, q, which):
    low = electron_parity(atoms, q)
    if which == "triplet":
        return [1 if low else 2]
    return [low] if which == "low" else [low, low + 2]


# The symmetry operations of a cell (spglib), as (rotations, translations) in fractional
# coordinates.
#   Called by: band_masses(), candidate_points(), equivalent_sites(), site_rotations()
def symmetry_ops(atoms, symprec=1e-3):
    try:
        import spglib
    except ImportError:
        die("the defect generator needs spglib (pip install spglib; it comes with pymatgen)")
    data = spglib.get_symmetry((np.asarray(atoms.cell), atoms.get_scaled_positions(),
                                atoms.get_atomic_numbers()), symprec=symprec)
    return data["rotations"], data["translations"]


# Which sites of a cell are equivalent by symmetry: a list giving each site's class (the index of
# the first site equivalent to it).
#   Called by: candidate_points(), comp_name(), parse_token(), single_defects()
#   Calls:     symmetry_ops()
def equivalent_sites(atoms):
    rots, trans = symmetry_ops(atoms)
    frac = atoms.get_scaled_positions()
    cls = list(range(len(atoms)))
    for i in range(len(atoms)):
        for R, t in zip(rots, trans):
            f = (R @ frac[i] + t) % 1.0
            d = (frac - f + 0.5) % 1.0 - 0.5
            j = int(np.argmin(np.linalg.norm(d @ np.asarray(atoms.cell), axis=1)))
            if j < i and cls[j] == j:
                cls[i] = min(cls[i], j)
    return cls


# --------------------------------------------------------------------------
# Defects
# --------------------------------------------------------------------------
# A defect component's name: Sb_Ca (substitution), Vac_O (vacancy), Sb_i (interstitial), with the
# primitive-cell site added (Vac_O:2, the 2nd O of the primitive cell) when its element has
# inequivalent sites.
#   Called by: build_defect(), defect_name()
#   Calls:     equivalent_sites()
def comp_name(c, prim=None):
    if c["host"] is None:
        return f"{c['new']}_i" + (f"{c['site_label']}" if c.get("site_label") else "")
    base = f"{c['new'] or 'Vac'}_{c['host']}"
    if prim is not None:
        same = [i for i, s in enumerate(prim.get_chemical_symbols()) if s == c["host"]]
        cls = equivalent_sites(prim) if len(same) > 1 else []
        if len({cls[i] for i in same} if cls else ()) > 1:
            base += f":{same.index(c['prim_site']) + 1}"
    return base


# The name of a defect: its components joined by '+'.
#   Called by: candidate()
#   Calls:     comp_name()
def defect_name(comps, prim=None):
    return "+".join(comp_name(c, prim) for c in comps)


# The change in composition a defect makes, e.g. {'Sb': 1, 'Ca': -1, 'O': -1}, and its text
# (+Sb-Ca-O) used to group arrangements that can be compared directly (the defect hull).
#   Called by: candidate()
def composition_change(comps):
    d = {}
    for c in comps:
        if c["new"]:
            d[c["new"]] = d.get(c["new"], 0) + 1
        if c["host"]:
            d[c["host"]] = d.get(c["host"], 0) - 1
    d = {k: v for k, v in d.items() if v}
    text = "".join(f"{'+' if v > 0 else '-'}{abs(v) if abs(v) > 1 else ''}{k}"
                   for k, v in sorted(d.items(), key=lambda kv: (-kv[1], kv[0])))
    return d, text or "none"


# Builds a defect supercell: the supercell P x (primitive cell) with the defect's components.
# Every component after the first (the anchor) goes at its vector from the anchor (vec, Ang), onto
# the host atom of its element there (within 0.1 Ang; an interstitial: at that point), and the
# anchor goes where it puts the whole defect as close to the cell centre as it can be: the mean
# position of the components nearest the centre. The anchor's choices are the copies of its
# primitive-cell site and of the sites a pure translation makes identical to it
# (translation_copies, anchor_site), or for an interstitial the images of its position, so
# the complex is only moved, never turned. Stops if a site can't be found or the complex overlaps
# its own periodic images. Substituted and interstitial atoms get their component's starting
# spin. Returns (Atoms, info): info has the defect's centre (the mean position of its components,
# cartesian), each component's position, the anchor's atom index (None for a vacancy) and the
# positions of the vacancies.
#   Called by: big_cell(), carry_over(), into_cell(), match_ideal(), run_screen_jobs(), step_ccd(),
#              step_lineshape(), step_prerelax(), step_scan(), step_validate(), step_zpl()
#   Calls:     comp_name(), anchor_site(), build_supercell(), get_distances(),
#              pure_translations(), translation_copies()
def build_defect(prim, P, comps, label="defect"):
    host = build_supercell(prim, P)
    cell = np.asarray(host.cell)
    centre = cell.sum(axis=0) / 2
    syms = np.array(host.get_chemical_symbols())

    # Host atoms a component after the anchor can go on: that element (its vector says where).
    def sites_of(c):
        return np.flatnonzero(syms == c["host"])

    # The minimum-image distances from a point to some atoms of the host supercell.
    def dists(point, idx):
        _, L = get_distances(np.atleast_2d(point), host.positions[idx], cell=host.cell,
                                pbc=True)
        return L[0]

    # How far the defect is from the centre with its anchor at r: (its mean position, its
    # farthest component), for choosing where an interstitial anchor goes.
    def off(r):
        return (round(float(np.linalg.norm(r + offsets.mean(axis=0) - centre)), 4),
                round(float(np.linalg.norm(r + offsets - centre, axis=1).max()), 4))

    a = comps[0]
    offsets = np.array([np.zeros(3)] + [np.asarray(c["vec"], float) for c in comps[1:]])
    if a["host"]:
        copies = translation_copies(prim, a["prim_site"])
        sites = [anchor_site(host, dict(a, prim_sites=copies), offsets[1:])]
        r0 = host.positions[sites[0]].copy()
    else:
        pc = np.asarray(prim.cell)
        tries = []
        for t in pure_translations(prim):
            rp = (np.asarray(a["frac"]) + t) @ pc
            f0 = np.round(np.linalg.solve(pc.T, centre - offsets.mean(axis=0) - rp))
            tries += [rp + (f0 + np.array(d)) @ pc for d in itertools.product((-1, 0, 1),
                                                                             repeat=3)]
        r0 = min(tries, key=off)
        sites = [None]
    points = [r0]
    for c in comps[1:]:
        target = r0 + np.asarray(c["vec"])
        if c["host"]:
            cand = sites_of(c)
            L = dists(target, cand)
            j = int(np.argmin(L))
            if L[j] > 0.1:
                die(f"could not place {comp_name(c)} of {label} in the supercell")
            if int(cand[j]) in sites:
                die(f"the supercell is too small for {label}: its components overlap their "
                    "periodic images")
            sites.append(int(cand[j]))
            points.append(host.positions[int(cand[j])].copy())
        else:
            sites.append(None)
            points.append(target)
    for i in range(len(comps)):                    # the complex must not touch its images
        for j in range(i + 1, len(comps)):
            want = np.linalg.norm(offsets[j] - offsets[i])
            _, L = get_distances(points[i][None], points[j][None], cell=host.cell, pbc=True)
            if L[0, 0] < want - 0.1:
                die(f"the supercell is too small for {label}: the {want:.2f} A between two "
                    "components is longer than the way round through the periodic images")
    atoms = host.copy()
    del atoms.arrays["prim_index"]
    s, mags = atoms.get_chemical_symbols(), atoms.get_initial_magnetic_moments()
    for c, site in zip(comps, sites):
        if c["host"] and c["new"]:
            s[site], mags[site] = c["new"], c.get("spin", 0.0)
    atoms.set_chemical_symbols(s)
    atoms.set_initial_magnetic_moments(mags)
    anchor = sites[0]
    for k, (c, pt) in enumerate(zip(comps, points)):
        if c["host"] is None:                      # an interstitial: a new atom at the end
            atoms.append(c["new"])
            atoms.positions[-1] = pt
            m = atoms.get_initial_magnetic_moments()
            m[-1] = c.get("spin", 0.0)
            atoms.set_initial_magnetic_moments(m)
            if k == 0:
                anchor = len(atoms) - 1
    vac = sorted((site for c, site in zip(comps, sites) if c["host"] and not c["new"]),
                 reverse=True)
    if anchor is not None and comps[0]["host"] and not comps[0]["new"]:
        anchor = None                              # the anchor is a vacancy
    if anchor is not None:
        anchor -= sum(1 for v in vac if v < anchor)
    if vac:
        del atoms[vac]
    atoms.wrap()
    info = {"centre": (r0 + offsets.mean(axis=0)).tolist(),
            "points": [pt.tolist() for pt in points], "anchor_index": anchor,
            "vacancies": [pt.tolist() for c, pt in zip(comps, points)
                          if c["host"] and not c["new"]]}
    return atoms, info


# The pristine supercell P x (primitive cell), without the site labels build_supercell adds.
#   Called by: run_screen_jobs(), step_distort(), step_excite(), step_filter(), step_lineshape(),
#              step_relaxations(), step_screen(), step_validate(), step_zpl(), validate_gamma()
#   Calls:     build_supercell()
def host_supercell(prim, P):
    atoms = build_supercell(prim, P)
    del atoms.arrays["prim_index"]
    return atoms


# The shortest distance between a point and its periodic images (the shortest lattice vector),
# from the Minkowski-reduced cell, so that it is right for skewed cells too.
#   Called by: carry_over(), cell_record(), lineshape_vector(), same_basin(), size_matrix()
#   Calls:     min_image_distance()
def min_image(cell):
    return min_image_distance(Cell(np.asarray(cell, float)).minkowski_reduce()[0])


# Whether a candidate can be built in a supercell whose images are L apart: a pair's components
# must be less than half of that apart (with 0.05 Ang to spare), or the nearest copy of one of them
# would be a periodic image of the other.
#   Called by: step_prerelax(), step_scan(), step_validate()
def fits(cand, L):
    return not (cand["separation"] and cand["separation"] > L / 2 - 0.05)


# The vectors from each of the points a to each of the points b (N x M x 3, Ang), each the nearest
# periodic image in fractional coordinates: right for the short distances they are used for in a
# Minkowski-reduced cell, and lighter than ASE's general search for big cells.
#   Called by: match_ideal()
def frac_mic(a, b, cell):
    cell = np.asarray(cell, float)
    df = (np.asarray(b)[None, :, :] - np.asarray(a)[:, None, :]) @ np.linalg.inv(cell)
    return (df - np.round(df)) @ cell


# A defect's relaxed structure matched atom by atom to the defect built in the same supercell
# (build_defect: the ideal structure, every atom on its perfect-crystal site): each ideal atom to
# one of the relaxed structure's of the same element, by the assignment that moves them least in
# total (scipy's linear_sum_assignment on the distances through the periodic images), so a matched
# pair is the same atom even when it has moved a long way, whatever order the relaxed structure's
# atoms are in. Returns (the ideal Atoms, its info, for each ideal atom the index of its atom in
# the relaxed structure, its displacement (N x 3, Ang) and its position relative to the defect's
# centre (N x 3, the shortest way through the images)).
#   Called by: big_cell(), displacement_field(), lineshape_vector()
#   Calls:     build_defect(), frac_mic()
def match_ideal(prim, comps, P, relaxed, label="defect"):
    from ase.geometry import find_mic
    from scipy.optimize import linear_sum_assignment
    ideal, info = build_defect(prim, P, comps, label)
    syms, rsyms = np.array(ideal.get_chemical_symbols()), np.array(relaxed.get_chemical_symbols())
    if sorted(syms) != sorted(rsyms) or not np.allclose(ideal.cell, relaxed.cell, atol=1e-4):
        die(f"{label}: the relaxed structure is not of the defect built in this cell")
    u, idx = np.zeros((len(ideal), 3)), np.zeros(len(ideal), int)
    for el in sorted(set(syms)):
        i, r = np.flatnonzero(syms == el), np.flatnonzero(rsyms == el)
        D = frac_mic(ideal.positions[i], relaxed.positions[r], ideal.cell)
        rows, cols = linear_sum_assignment(np.linalg.norm(D, axis=2))
        u[i[rows]] = D[rows, cols]
        idx[i[rows]] = r[cols]
    v, _ = find_mic(ideal.positions - np.asarray(info["centre"]), ideal.cell, pbc=True)
    return ideal, info, idx, u, v


# The displacement field of a defect's relaxed structure (match_ideal without the matching): the
# ideal Atoms, its info, each ideal atom's displacement and position relative to the centre.
#   Called by: carry_over(), same_basin(), step_scan()
#   Calls:     match_ideal()
def displacement_field(prim, comps, P, relaxed, label="defect"):
    ideal, info, _, u, v = match_ideal(prim, comps, P, relaxed, label)
    return ideal, info, u, v


# The atoms of two cells that are the same atom of a defect: of the same element, at the same
# position relative to the defect's centre (within 0.1 Ang), both within radius of it. Takes the
# positions relative to the centre (v1, v2) and the elements of each. Returns (indices in 1,
# indices in 2).
#   Called by: carry_over(), lineshape_vector(), same_basin()
def common_atoms(v1, syms1, v2, syms2, radius):
    from scipy.spatial import cKDTree
    syms1, syms2 = np.asarray(syms1), np.asarray(syms2)
    a, b = [], []
    for el in sorted(set(syms1) & set(syms2)):
        i = np.flatnonzero((syms1 == el) & (np.linalg.norm(v1, axis=1) <= radius))
        j = np.flatnonzero((syms2 == el) & (np.linalg.norm(v2, axis=1) <= radius))
        if not len(i) or not len(j):
            continue
        dist, k = cKDTree(v1[i]).query(v2[j], distance_upper_bound=0.1)
        hit = np.isfinite(dist)
        a += i[k[hit]].tolist()
        b += j[hit].tolist()
    return np.array(a, int), np.array(b, int)


# A defect's relaxed structure in one supercell (src_P, src_atoms) carried into another (dst_P):
# the defect is built in the new cell (build_defect: the same components, at the same place in the
# crystal) and each atom within radius of its centre is moved by the displacement the same atom has
# in the relaxed structure (displacement_field, common_atoms); the atoms further out stay on their
# perfect-crystal sites. radius defaults to just under half the smaller cell's shortest image
# distance, the region the two cells describe in the same way. Works for any host and any pair of
# supercells of it. Returns (Atoms, info) as build_defect does. field: the source's
# displacement_field, when it is known already (B2 carries one structure into many sizes).
#   Called by: big_cell(), into_cell(), step_distort(), step_prerelax(), step_scan(),
#              step_validate()
#   Calls:     build_defect(), common_atoms(), displacement_field(), min_image()
def carry_over(prim, comps, src_P, src_atoms, dst_P, radius=None, label="defect", field=None):
    ideal, _, u, v = field or displacement_field(prim, comps, src_P, src_atoms, label)
    new, info = build_defect(prim, dst_P, comps, label)
    from ase.geometry import find_mic
    w, _ = find_mic(new.positions - np.asarray(info["centre"]), new.cell, pbc=True)
    if radius is None:
        radius = 0.5 * min(min_image(ideal.cell), min_image(new.cell)) - 0.1
    a, b = common_atoms(v, ideal.get_chemical_symbols(), w, new.get_chemical_symbols(), radius)
    new.positions[b] += u[a]
    new.wrap()
    return new, info


# Whether two relaxed structures of a defect in different supercells are in the same basin: the
# displacements of the atoms near the defect (within EDGE_FRACTION of the smaller cell's shortest
# image distance of its centre; displacement_field, common_atoms) differ by no more than BASIN_TOL.
# Returns (same, the largest difference in Ang).
#   Called by: step_excite()
#   Calls:     common_atoms(), displacement_field(), min_image()
def same_basin(prim, comps, P1, atoms1, P2, atoms2, label="defect"):
    i1, _, u1, v1 = displacement_field(prim, comps, P1, atoms1, label)
    i2, _, u2, v2 = displacement_field(prim, comps, P2, atoms2, label)
    radius = EDGE_FRACTION * min(min_image(i1.cell), min_image(i2.cell))
    a, b = common_atoms(v1, i1.get_chemical_symbols(), v2, i2.get_chemical_symbols(), radius)
    worst = float(np.linalg.norm(u1[a] - u2[b], axis=1).max()) if len(a) else 0.0
    return worst <= BASIN_TOL, worst


# --------------------------------------------------------------------------
# CASTEP output
# --------------------------------------------------------------------------
# Reads a CASTEP .bands file: k-points, weights, eigenvalues (eV, [spin][k-point][band]), the
# electrons in each spin channel (one number, the total, when the calculation is not spin-
# polarised) and the Fermi energies (eV).
#   Called by: band_state_test(), job_summary(), run_screen_jobs(), step_bands(), step_choose(),
#              step_excite(), step_formation(), step_relaxations(), step_screen(), step_zpl()
def read_bands(path):
    lines = Path(path).read_text().splitlines()

    # The numbers on a header line.
    def nums(line):
        return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?", line)]

    nk, nspin = int(nums(lines[0])[-1]), int(nums(lines[1])[-1])
    nelec, nbands = nums(lines[2]), [int(x) for x in nums(lines[3])]
    efermi = [x * HARTREE for x in nums(lines[4])]
    nbands = (nbands * nspin)[:nspin] if len(nbands) < nspin else nbands
    eig = np.full((nspin, nk, max(nbands)), np.nan)
    kpts, weights = np.zeros((nk, 3)), np.zeros(nk)
    i, ik = 0, None
    while i < len(lines):
        line = lines[i]
        if line.startswith("K-point"):
            v = line.split()
            ik = int(v[1]) - 1
            kpts[ik], weights[ik] = [float(x) for x in v[2:5]], float(v[5])
        elif line.startswith("Spin component") and ik is not None:
            s = int(line.split()[-1]) - 1
            eig[s, ik, :nbands[s]] = [float(x) * HARTREE for x in lines[i + 1:i + 1 + nbands[s]]]
            i += nbands[s]
        i += 1
    return {"nspin": nspin, "nelec": nelec, "efermi": efermi, "kpts": kpts, "weights": weights,
            "eig": eig}


# Reads a CASTEP .pdos_weights file (pdos_calculate_weights : true): the weight of each band on each
# atomic orbital, as OptaDOS reads it. Fortran unformatted records, big- or little-endian (found
# from the first record's length): the numbers of k-points, spins, orbitals and bands (one record
# each), each orbital's species number, rank within its species and angular momentum (three
# records), then for each k-point (its number and fractional coordinates) and each spin (its number,
# then how many bands) one record of orbital weights per band. Returns {'species', 'rank', 'l'
# (per orbital), 'kpts', 'weights': [k][spin] -> array (bands x orbitals)}, or None if the file is
# missing or not in this form.
#   Called by: band_state_test(), dscf_result(), step_excite(), step_zpl()
def read_pdos_weights(path):
    path = Path(path)
    if not path.exists():
        return None
    raw = path.read_bytes()
    for end in (">", "<"):
        if len(raw) >= 4 and int(np.frombuffer(raw[:4], dtype=end + "i4")[0]) == 4:
            break
    else:
        return None
    pos, recs = 0, []
    while pos + 4 <= len(raw):
        n = int(np.frombuffer(raw[pos:pos + 4], dtype=end + "i4")[0])
        if n < 0 or pos + 8 + n > len(raw):
            return None
        recs.append(raw[pos + 4:pos + 4 + n])
        pos += 8 + n
    try:
        ints = [int(np.frombuffer(r, dtype=end + "i4")[0]) for r in recs[:4]]
        nk, ns, norb, _ = ints
        spec, rank, l_ = (np.frombuffer(r, dtype=end + "i4").copy() for r in recs[4:7])
        k, kpts, weights = 7, [], []
        for _ in range(nk):
            kpts.append(np.frombuffer(recs[k][4:28], dtype=end + "f8").tolist())
            k += 1
            per_spin = []
            for _ in range(ns):
                nb = int(np.frombuffer(recs[k + 1], dtype=end + "i4")[0])
                k += 2
                per_spin.append(np.array([np.frombuffer(recs[k + j], dtype=end + "f8")[:norb]
                                          for j in range(nb)]))
                k += nb
            weights.append(per_spin)
    except (IndexError, ValueError):
        return None
    return {"species": spec, "rank": rank, "l": l_, "kpts": kpts, "weights": weights}


# Each band's participation number at one k-point of a .pdos_weights (read_pdos_weights): how many
# atoms its weight is spread over, (sum of its atoms' weights)^2 / (sum of their squares), with an
# atom's weight the sum over its orbitals (an atom is a species number and rank). About 1 to 10 for
# a state on a defect and its neighbours; about the number of atoms of the elements a band is made
# of (half the cell for a rock-salt band) for a host state. Returns [spin] -> array (bands).
#   Called by: band_state_test(), dscf_result(), step_excite(), step_zpl(), zpl_bands_ok()
def participation(pw, k=0):
    keys = np.array(list(zip(pw["species"], pw["rank"])))
    _, atom = np.unique(keys, axis=0, return_inverse=True)
    atom = np.asarray(atom).ravel()
    out = []
    for w in pw["weights"][k]:
        per_atom = np.zeros((w.shape[0], atom.max() + 1))
        np.add.at(per_atom.T, atom, np.abs(w).T)
        tot = per_atom.sum(axis=1)
        sq = (per_atom ** 2).sum(axis=1)
        out.append(np.where(sq > 0, tot ** 2 / np.where(sq > 0, sq, 1), np.inf))
    return out


# The number of occupied bands in each spin channel of a .bands (read_bands). From the electron
# counts when CASTEP gives one per spin (or the total, not spin-polarised); otherwise the bands
# below the Fermi energy at the first k-point.
#   Called by: band_edges(), band_masses(), carrier_bands(), carriers(), defect_levels(),
#              gap_levels()
def occupied_bands(b):
    if b["nspin"] == 1:
        return [int(round(b["nelec"][0] / 2))]
    if len(b["nelec"]) == 2:
        return [int(round(x)) for x in b["nelec"]]
    ef = b["efermi"][0]
    return [int(np.sum(b["eig"][s, 0] < ef)) for s in range(2)]


# The valence band maximum and conduction band minimum (eV) of an insulator's .bands: the highest
# occupied and lowest empty eigenvalue over all k-points and spins.
#   Called by: run_screen_jobs(), step_bands(), step_excite(), step_relaxations(), step_screen()
#   Calls:     occupied_bands()
def band_edges(b):
    occ = occupied_bands(b)
    vbm = max(np.nanmax(b["eig"][s, :, occ[s] - 1]) for s in range(b["nspin"]))
    cbm = min(np.nanmin(b["eig"][s, :, occ[s]]) for s in range(b["nspin"]))
    return float(vbm), float(cbm)


# The defect levels in the band gap of a defect calculation (at its first k-point, Gamma for the
# screening): each eigenvalue more than margin inside the host's band edges (vbm, cbm, from a
# calculation with the same settings), with its spin channel, its energy above the VBM and
# whether it is occupied. Eigenvalues are compared as CASTEP gives them (both calculations have
# the average electrostatic potential at zero), without potential alignment: good enough to sort
# candidates, not for production levels.
#   Called by: job_summary()
#   Calls:     occupied_bands()
def gap_levels(b, vbm, cbm, margin):
    occ, out = occupied_bands(b), []
    for s in range(b["nspin"]):
        for n, e in enumerate(b["eig"][s, 0]):
            if np.isfinite(e) and vbm + margin < e < cbm - margin:
                out.append({"spin": s, "e": round(float(e - vbm), 4), "occupied": n < occ[s]})
    return out


# The electrons a defect calculation left in the conduction band and the holes it left in the
# valence band (at its first k-point, Gamma for the screening), against the host's band edges
# (vbm, cbm) with the same margin as gap_levels, so that every state is either a gap level or in a
# band: the electrons are those beyond what the states up to margin below the CBM can hold, the
# holes the states up to margin above the VBM that the electrons can't fill. Counted over both spin
# channels together, as one Fermi level fills them when the spin is free (so a channel's electron
# count, which CASTEP may give as the starting one, doesn't matter). Fractional with smearing.
# Returns {"electrons": ..., "holes": ...}.
#   Called by: job_summary()
def band_carriers(b, vbm, cbm, margin):
    per = 2 if b["nspin"] == 1 else 1                # electrons a band holds
    total = float(sum(b["nelec"]))
    eig = b["eig"][:, 0, :]
    below = sum(int(np.sum(np.isfinite(e) & (e < cbm - margin))) for e in eig) * per
    low = sum(int(np.sum(np.isfinite(e) & (e <= vbm + margin))) for e in eig) * per
    return {"electrons": round(max(0.0, total - below), 3),
            "holes": round(max(0.0, low - total), 3)}


# Whether a screening-type run is a band state: it left at least BAND_CARRIER electrons or holes in
# the bands (band_carriers). A run without a .bands file counts as bound.
#   Called by: step_excite(), step_filter(), step_validate()
def in_band(run):
    c = run.get("band")
    return bool(c) and max(c["electrons"], c["holes"]) >= BAND_CARRIER


# The band carriers of a band-state run as text, e.g. "1.00 e" or "0.98 h".
#   Called by: step_excite(), step_filter()
def carriers_text(run):
    c = run["band"]
    return f"{c['electrons']:.2f} e" if c["electrons"] >= c["holes"] else f"{c['holes']:.2f} h"


# Whether gap levels (gap_levels) include an occupied and an empty level in the same spin
# channel, the empty one above (the levels of an internal transition): require_gap_states : both.
#   Called by: step_filter()
def internal_pair(levels):
    for s in (0, 1):
        occ = [x["e"] for x in levels if x["spin"] == s and x["occupied"]]
        emp = [x["e"] for x in levels if x["spin"] == s and not x["occupied"]]
        if occ and emp and min(emp) > max(occ):
            return True
    return False


# A row of CASTEP's Forces table: species, number, and the three components (eV/A).
FORCE_ROW = re.compile(r"^\s*\*\s+([A-Z][a-z]?)\S*\s+(\d+)\s+(-?\d+\.\d+)\S*\s+(-?\d+\.\d+)\S*"
                       r"\s+(-?\d+\.\d+)\S*\s+\*", re.M)


# Every structure of a .castep with its energy and forces (all runs of a job that was restarted):
# for each 'Final energy', the last lattice and 'Fractional coordinates' table printed before it
# and the first Forces table after it. Atoms are in CASTEP's order (grouped by species). which:
# the energy to give ('final', or 'free' for the free energy E-TS printed after it, the one the
# forces belong to with smearing; the final energy where it isn't printed). Returns
# [(Atoms, energy, forces)], leaving out structures without forces.
#   Called by: castep_frames(), lineshape_vector()
#   Calls:     last_lattice()
def read_trajectory(text, which="final"):
    out = []
    ends = [m for m in re.finditer(ENERGY_PATTERNS["final"], text)]
    for k, m in enumerate(ends):
        head = text[:m.start()]
        part = head.rsplit("Fractional coordinates of atoms", 1)
        if len(part) < 2:
            continue
        rows = ATOM_ROW.findall(part[1])
        lattice = last_lattice(head)
        tail = text[m.end():ends[k + 1].start() if k + 1 < len(ends) else len(text)]
        fm = re.search(r"\*+ [\w ]*Forces \*+", tail)
        if not rows or lattice is None or not fm:
            continue
        forces = FORCE_ROW.findall(tail[fm.end():])[:len(rows)]
        if len(forces) != len(rows):
            continue
        atoms = Atoms([r[0] for r in rows], scaled_positions=[[float(x) for x in r[1:]]
                                                              for r in rows],
                      cell=lattice, pbc=True)
        alt = re.search(ENERGY_PATTERNS[which], tail) if which != "final" else None
        out.append((atoms, float(alt.group(1) if alt else m.group(1)),
                    np.array([[float(x) for x in f[2:]] for f in forces])))
    return out


# The permittivity tensors CASTEP prints for an efield or phonon+efield task: (optical, i.e.
# electronic, epsilon_inf; static epsilon_0 including the ions, or None if it printed only the
# optical one), each 3x3, from the last such table in the output.
#   Called by: step_dielectric()
def read_permittivity(text):
    found = list(re.finditer(r"Optical Permittivity \(f->infinity\)[^\n]*\n", text))
    if not found:
        return None, None
    rows = []
    for line in text[found[-1].end():].splitlines():
        vals = re.findall(r"[-+]?\d+\.\d+", line)
        if len(vals) >= 3:
            rows.append([float(v) for v in vals])
        if len(rows) == 3:
            break
    if len(rows) < 3:
        return None, None
    eps_inf = [r[:3] for r in rows]
    eps0 = [r[3:6] for r in rows] if all(len(r) >= 6 for r in rows) else None
    return eps_inf, eps0


# A row of the stress tensor CASTEP prints with calculate_stress: the axis, three numbers.
STRESS_ROW = re.compile(r"^\s*\*\s+([xyz])\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+\*\s*$",
                        re.M)
# The units CASTEP can print it in (pressure_unit; GPa by default), in GPa.
PRESSURE_UNITS = {"gpa": 1.0, "mpa": 1e-3, "pa": 1e-9, "bar": 1e-4, "atm": 1.01325e-4}


# The last stress tensor (3x3, GPa) and pressure (GPa) a .castep printed (calculate_stress : true;
# the symmetrised one when symmetry is on), in CASTEP's sign convention, whose pressure is minus
# a third of the trace: positive when the cell pushes outward. (None, None) if there is none, or
# if it is in a unit not in PRESSURE_UNITS.
#   Called by: step_validate()
def read_stress(text):
    i = text.rfind("Stress Tensor")
    if i < 0:
        return None, None
    block = text[i:i + 2500]
    u = re.search(r"components\s*\(([^)]+)\)", block)
    unit = PRESSURE_UNITS.get(u.group(1).strip().lower()) if u else 1.0
    rows = STRESS_ROW.findall(block)[:3]
    if unit is None or [r[0] for r in rows] != ["x", "y", "z"]:
        return None, None
    m = re.search(r"Pressure:\s*(-?\d+\.\d+)", block)
    return (np.array([[float(v) for v in r[1:]] for r in rows]) * unit,
            float(m.group(1)) * unit if m else None)


# The host's elastic compliance S (6x6, Voigt, 1/GPa: stress to engineering strain) from the rows
# of %block elastic_constants (GPa, in host_cell's Cartesian axes): one line C11 C12 C44 for a
# cubic host, or the full 6x6 stiffness in six lines. None without the block.
#   Called by: read_project(), step_validate()
def compliance(block):
    if not block:
        return None
    rows = []
    for r in block:
        nums = []
        for x in r:
            try:
                nums.append(float(x))
            except ValueError:
                if x.lower() != "gpa":
                    die(f"elastic_constants: '{x}' is not a number (GPa)")
        if nums:
            rows.append(nums)
    if len(rows) == 1 and len(rows[0]) == 3:
        c11, c12, c44 = rows[0]
        C = np.zeros((6, 6))
        C[:3, :3] = c12
        C[[0, 1, 2], [0, 1, 2]] = c11
        C[[3, 4, 5], [3, 4, 5]] = c44
    elif len(rows) == 6 and all(len(r) == 6 for r in rows):
        C = np.array(rows)
    else:
        die("elastic_constants: give one line C11 C12 C44 (a cubic host) or six lines of six "
            "(the Voigt stiffness), in GPa")
    if np.min(np.linalg.eigvalsh(0.5 * (C + C.T))) <= 0:
        die("elastic_constants: the stiffness matrix isn't positive definite")
    return np.linalg.inv(C)


# The energy (eV) a fixed cell of volume V (Ang^3) stores when it holds the stress s (3x3, GPa; the
# defect cell's less the host's in the same cell): 1/2 V s:S:s, S the host's compliance. Quadratic,
# so the same for a push (a positive pressure) as for a pull.
#   Called by: step_validate()
def elastic_energy(s, S, volume):
    v = np.array([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]], float)
    return 0.5 * volume * float(v @ S @ v) * GPA_A3


# The finite-size correction (eV) for charge q in a supercell (the runner's
# charge_correction: Lany-Zunger, Makov-Payne or none), with the static dielectric constant eps.
#   Called by: relative_formation(), step_corrections(), step_filter(), step_validate()
#   Calls:     charge_correction()
def correction(q, cell, eps, method):
    if not q or method == "none":
        return 0.0
    if not eps:
        die("a charged defect needs the dielectric constant: run the dielectric step (A2) or "
            "set dielectric, or set charge_correction : none")
    return charge_correction(q, cell, SimpleNamespace(charge_correction=method,
                                                         dielectric=eps))[0]


# --------------------------------------------------------------------------
# MACE
# --------------------------------------------------------------------------
# What identifies one MACE relaxation in the caches: the structure, the model (Project.model_id)
# and the optimiser's settings (mace_fmax, mace_steps, or a relaxation's own fmax and steps), and
# for a task with its own head, total charge or rattle (EXTRA_TASK_KEYS) those too.
#   Called by: mace_relax(), mace_relaxations(), step_prerelax()
#   Calls:     Project.model_id(), atoms_to_dict()
def mace_key(p, atoms, model=None, head=None, charge=None, fmax=None, steps=None, rattle=None):
    data = {"atoms": atoms_to_dict(atoms), "model": p.model_id(model),
            "fmax": fmax or p.s.mace_fmax, "steps": steps or p.s.mace_steps}
    if head is not None or charge is not None:            # (keeps the older caches valid)
        data.update(head=head, charge=charge)
    if rattle:
        data["rattle"] = rattle
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


# The cache file of a candidate's MACE relaxation in one supercell (tag: the size's tag, e.g.
# n144). B2 and C1 share them, so C1 reuses the scan's relaxations in the screening cell.
#   Called by: step_prerelax(), step_scan()
def mace_cache(p, cid, tag):
    return p.phase_dir("B") / "mace" / f"{cid}_{tag}.json"


# The cache file of a candidate's re-seeded MACE relaxation in one supercell: from the deepest
# minimum B2 found for it in another size, carried into this one (step_scan's second pass; C1's
# in the screening cell, the same file when that is a size of the ladder).
#   Called by: step_prerelax(), step_scan()
def mace_seed_cache(p, cid, tag):
    return p.phase_dir("B") / "mace" / f"{cid}_{tag}_seeded.json"


# A MACE relaxation's result from its cache file: (relaxed Atoms, energy before, energy after,
# steps), or None if the file doesn't hold this relaxation (key: mace_key). One that failed in a
# MACE job (the error is in the file) stops the script with its error.
#   Called by: mace_jobs(), mace_relax(), mace_relaxations()
#   Calls:     atoms_from_dict()
def mace_cached(cache, key):
    cache = Path(cache)
    if not cache.exists():
        return None
    try:
        saved = json.loads(cache.read_text())
    except ValueError:                            # being written: not there yet
        return None
    if saved.get("key") != key:
        return None
    if saved.get("error"):
        die(f"the MACE relaxation in {cache} failed: {saved['error']} (the workers' logs are in "
            "the jobs/mace_<step> folder of its phase; delete the file to try it again)")
    return atoms_from_dict(saved["atoms"]), saved["e0"], saved["e1"], saved["steps"]


# Writes a MACE relaxation's result (relaxed Atoms, energy before, energy after, steps), or the
# error that stopped it, to its cache file, through a temporary file (so a job killed while writing
# leaves no half-written file).
#   Called by: mace_relax(), mace_worker()
#   Calls:     atoms_to_dict()
def save_mace_result(cache, key, result=None, error=None):
    data = {"key": key}
    if error:
        data["error"] = error
    else:
        atoms, e0, e1, steps = result
        data.update(atoms=atoms_to_dict(atoms), e0=e0, e1=e1, steps=steps)
    cache = Path(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.name + ".tmp")
    tmp.write_text(json.dumps(data))
    tmp.replace(cache)


# The rattle of a relaxation from a defect's ideal structure (B2, C1), for its task: the atoms within
# MACE_RATTLE_RADIUS of one of the defect's components (info: build_defect's) moved at random,
# mace_rattle (Ang) the standard deviation of each coordinate, the same displacements every time
# (apply_rattle). A defect on a symmetric site (e.g. an interstitial in a void) feels no force off
# it there, its forces cancelling by symmetry, so a relaxation from the ideal structure would keep
# the symmetry even where breaking it goes lower; the rattle lets it go. None with
# mace_rattle : 0.
#   Called by: step_prerelax(), step_scan()
def rattle_spec(p, info):
    if not p.s.mace_rattle:
        return None
    return {"stdev": p.s.mace_rattle, "radius": MACE_RATTLE_RADIUS,
            "points": [[round(float(x), 6) for x in pt] for pt in info["points"]]}


# Moves a structure's atoms by a rattle_spec, in place: those within its radius of one of its points
# (through the periodic images), by normal random displacements (MACE_RATTLE_SEED: the same every
# time for the same structure).
#   Called by: mace_run()
def apply_rattle(atoms, spec):
    _, d = get_distances(np.asarray(spec["points"], float), atoms.positions, cell=atoms.cell,
                         pbc=True)
    near = d.min(axis=0) < spec["radius"]
    u = np.random.RandomState(MACE_RATTLE_SEED).normal(scale=spec["stdev"], size=(len(atoms), 3))
    pos = atoms.get_positions()
    pos[near] += u[near]
    atoms.set_positions(pos)


# Relaxes the positions of a structure with a MACE calculator, the cell fixed (ASE's LBFGS, to
# fmax, in at most steps steps), in total charge charge for a model that takes it; with rattle (a
# rattle_spec), from the structure rattled (apply_rattle), the energy before still the structure's
# as given. Returns (relaxed Atoms, energy before, energy after, steps); the starting spins are kept
# (MACE ignores them). The same here (mace_relax) and in the MACE jobs (mace_worker).
#   Called by: mace_relax(), mace_worker(), phonon_worker()
#   Calls:     apply_rattle()
def mace_run(calc, atoms, fmax, steps, charge=None, rattle=None):
    from ase.optimize import LBFGS
    work = atoms.copy()
    if charge is not None:
        work.info["charge"] = charge               # the model's total-charge embedding reads it
    work.calc = calc
    e0 = float(work.get_potential_energy())
    if rattle:
        apply_rattle(work, rattle)
    opt = LBFGS(work, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    e1 = float(work.get_potential_energy())
    work.calc = None
    work.set_initial_magnetic_moments(atoms.get_initial_magnetic_moments())
    return work, e0, e1, int(getattr(opt, "nsteps", 0))


# Relaxes a structure with MACE here, in this process (model: default mace_model; a task's own
# head, charge, fmax, steps and rattle, EXTRA_TASK_KEYS), with a MACE step's threads and device (run:
# its mace_settings), unless its cache file already holds the result (mace_cached). Returns
# (relaxed Atoms, energy before, energy after, steps).
#   Called by: mace_relaxations()
#   Calls:     Project.mace(), mace_cached(), mace_key(), mace_run(), save_mace_result()
def mace_relax(p, atoms, cache, model=None, head=None, charge=None, fmax=None, steps=None,
               rattle=None, run=None):
    key = mace_key(p, atoms, model, head, charge, fmax, steps, rattle)
    hit = mace_cached(cache, key)
    if hit:
        return hit
    result = mace_run(p.mace(model, head, run), atoms, fmax or p.s.mace_fmax,
                      steps or p.s.mace_steps, charge, rattle)
    save_mace_result(cache, key, result)
    return result


# A step's MACE relaxations (mace_model; step: B2 scan, C1 prerelax or C4 distort, MACE_STEPS):
# {cache file: (relaxed Atoms, energy before, energy after, steps)}, or None while MACE jobs are
# still queued (--no-wait) or after --prepare-only. tasks: [{'atoms', 'cache', 'label'}, and
# optionally a task's own 'model', 'head', 'charge', 'fmax', 'steps', 'rattle' (EXTRA_TASK_KEYS)]. A
# relaxation whose cache file already holds it (the same structure, model and settings: mace_key) is
# not run again. The others run as jobs on compute nodes with the step's mace_jobscript
# (mace_settings: the .run file's, or its own block's; mace_jobs), else here, one after the other
# (mace_relax).
#   Called by: step_distort(), step_prerelax(), step_scan()
#   Calls:     Project.mace_settings(), mace_cached(), mace_jobs(), mace_key(), mace_relax()
def mace_relaxations(p, step, tasks):
    out, todo = {}, []
    for t in tasks:
        t = dict(t, key=mace_key(p, t["atoms"], *(t.get(k) for k in EXTRA_TASK_KEYS)))
        hit = mace_cached(t["cache"], t["key"])
        if hit:
            out[str(t["cache"])] = hit
        else:
            todo.append(t)
    run = p.mace_settings(step)
    if todo and run.mace_jobscript:
        if not mace_jobs(p, step, todo, len(tasks) - len(todo)):
            return None
        for t in todo:
            out[str(t["cache"])] = mace_cached(t["cache"], t["key"])
    for k, t in enumerate([] if run.mace_jobscript else todo):
        log(f"{STEP_INDEX[step].label}: {t['label']}: MACE relaxation ({len(t['atoms'])} atoms; "
            f"{k + 1} of {len(todo)})")
        out[str(t["cache"])] = mace_relax(p, t["atoms"], t["cache"],
                                          *(t.get(k) for k in EXTRA_TASK_KEYS), run=run)
    return out


# The Python a step's MACE jobs run: its mace_python (mace_settings; step None: the .run file's,
# for F2's model jobs), or else the one running dwf.py, with any linked folder
# on its path followed to where it really is (e.g. a ~/shared that links to /work/...), as the
# compute nodes may not see the link's side (on Cirrus and ARCHER2, /home). Only the folders: a
# venv's bin/python is itself a link to the Python the venv was made from, and following that
# would leave the venv. A bare command (e.g. python3) is left to the job's PATH.
#   Called by: mace_jobs(), step_lineshape()
def mace_python(p, step=None):
    py = (p.mace_settings(step).mace_python if step else p.s.mace_python) or sys.executable
    if "/" not in py:
        return py
    path = Path(py).expanduser()
    return str(path.parent.resolve() / path.name)


# The MACE relaxations a step still needs (mace_relaxations), run as batch jobs on compute nodes, in
# <phase>/jobs/mace_<step>/job_NNN/, with the step's MACE settings (mace_settings: the .run file's,
# or its own block's, %block runner_scan or runner_prerelax). At most mace_max_jobs jobs are queued
# or running at once; the relaxations are shared out between as many new jobs as there are places
# free, but no more than one per mace_pack relaxations (write_jobs), biggest first, onto whichever
# job has the fewest atoms so far. Each job is its mace_jobscript template with its {COMMAND} line made into up to mace_pack
# workers, each running  <python> dwf.py --mace-worker tasks.json  (the python from mace_python())
# in the background with mace_threads threads (whatever is in front of {COMMAND} on its line, e.g.
# srun ..., goes in front of each), then wait; a {NCORES} in the template becomes the cores those
# workers use (workers x mace_threads), so on a machine that charges by the core (charging : core)
# a job asks for just those, and the block's max_walltime becomes its --time. The job's workers
# share its one queue, tasks.json, biggest first: each takes the next relaxation no other has
# taken (mace_worker, mace_claim), so none sits idle while another still has a list to work
# through. (A job holds its node until its last worker ends. With a list fixed for each worker
# beforehand, that was the slowest list, as a relaxation's cost goes with its steps, which nothing
# knows in advance; with the shared queue it is only the last relaxation or two.) The jobs go in
# jobs.json there; waiting polls Slurm's queue (status_cmd) every poll seconds, like the CASTEP
# jobs, and the project's STOP file stops it; every PROGRESS_EVERY seconds the log says how
# many of the step's relaxations are done (done_before: those done before this call, which aren't
# in todo), and each job's progress.txt how many of its own are (mace_progress). The relaxations
# a job leaves (e.g. it ran out of time) wait for a place to come free and go into the next jobs,
# packed into as few as there are places. One that a worker had started (its claim file:
# claim_file) but not finished counts an attempt when its job ends; every one the job left counts,
# if it started none (a job that can't run at all). At MACE_ATTEMPTS the run stops, with how the
# last of its jobs ended (why_ended); --resubmit-failed gives those relaxations another
# MACE_ATTEMPTS. (A job written by an older dwf.py counted an attempt for each of its relaxations
# when it was submitted, and is left at that.) Returns True when every
# relaxation is done, False while jobs are queued (--no-wait) or after --prepare-only (the job
# scripts are then in prepared/).
#   Called by: mace_relaxations()
#   Calls:     Project.mace_model_file(), Project.mace_runner(), Project.mace_settings(),
#              Project.mace_template(), Project.model_path(), atoms_to_dict(), claim_file(),
#              mace_cached(), mace_python(), add_sbatch_option(), job_active(), send_job(),
#              write_json()
def mace_jobs(p, step, todo, done_before=0):
    args = p.mace_runner()
    d = p.phase_dir(STEP_INDEX[step].phase) / "jobs" / f"mace_{step}"
    d.mkdir(parents=True, exist_ok=True)
    state_file = d / "jobs.json"
    state = (json.loads(state_file.read_text()) if state_file.exists()
             else {"jobs": [], "tries": {}})
    run = p.mace_settings(step)
    template, stop = p.mace_template(step), p.dir / "STOP"
    pack = run.mace_pack or max(1, (args.cores_per_node or run.mace_threads) // run.mace_threads)
    label = STEP_INDEX[step].label
    model = p.mace_model_file() if any(not t.get("model") for t in todo) else None
    announced, last_done, last_time = False, None, 0.0
    total = done_before + len(todo)                   # the step's relaxations, all of them
    worn = [t for t in todo if state["tries"].get(str(t["cache"]), 0) >= MACE_ATTEMPTS]
    if worn and getattr(p.cl, "resubmit_failed", False):       # the cause fixed: try again
        for t in worn:
            state["tries"][str(t["cache"])] = 0
        log(f"{label}: --resubmit-failed: {len(worn)} MACE relaxations whose jobs gave up get "
            "another go")

    # Whether a job is over: not in the queue, and either its end mark is there or it was
    # submitted long enough ago that the queue must know it (so a slow queue isn't taken for a
    # job that has ended).
    def ended(job):
        if job_active(job["jobid"], args):
            return False
        return (Path(job["dir"]) / "finished").exists() or time.time() - job["time"] > 120

    # How a job that left relaxations undone ended, from its folder: stopped by Slurm at its time
    # limit (its Slurm output says so), or else the last line of the first of its workers' logs,
    # then of its Slurm output, with anything in it (e.g. a Python error, or a Python the compute
    # node couldn't find).
    def why_ended(jdir):
        outs = sorted(jdir.glob("slurm-*.out"))
        if any("DUE TO TIME LIMIT" in f.read_text(errors="replace") for f in outs):
            return ("Slurm stopped it at its time limit (give the jobs longer: max_walltime "
                    f"in %block runner_{step}, or a longer --time in the template)")
        for f in sorted(jdir.glob("worker_*.log")) + outs:
            lines = [x.strip() for x in f.read_text(errors="replace").splitlines() if x.strip()]
            if lines:
                return f"{f.name} ends with: {lines[-1][:300]}"
        return "its workers' logs and Slurm output are empty"

    # Writes the scripts of new jobs for some relaxations into a folder each (job.sh and the job's
    # queue, tasks.json, biggest first, which its workers share): at most places jobs, and no more
    # than one per mace_pack relaxations, so a few left over don't become many small jobs; returns
    # [(folder, [cache files])].
    def write_jobs(tasks, where, places):
        n_jobs = max(1, min(places, math.ceil(len(tasks) / pack)))
        shares = [{"tasks": [], "atoms": 0} for _ in range(n_jobs)]
        for t in sorted(tasks, key=lambda t: -len(t["atoms"])):
            s = min(shares, key=lambda s: s["atoms"])
            s["tasks"].append(t)
            s["atoms"] += len(t["atoms"])
        made = []
        for j, s in enumerate(shares):
            jdir = where(j)
            jdir.mkdir(parents=True, exist_ok=True)
            f = jdir / "tasks.json"
            f.write_text(json.dumps({
                "model": model, "device": run.mace_device, "threads": run.mace_threads,
                "fmax": p.s.mace_fmax, "steps": p.s.mace_steps,
                "tasks": [dict({"cache": str(Path(t["cache"]).resolve()), "key": t["key"],
                                "label": t["label"], "atoms": atoms_to_dict(t["atoms"])},
                               **{k: (str(Path(p.model_path(t[k])).resolve())
                                      if k == "model" else t[k])
                                  for k in EXTRA_TASK_KEYS if t.get(k) is not None})
                          for t in s["tasks"]]}))
            caches = [str(t["cache"]) for t in s["tasks"]]
            workers = min(pack, len(s["tasks"]))
            lines = template.splitlines()
            i = next(k for k, line in enumerate(lines)
                     if "{COMMAND}" in line and not line.lstrip().startswith("#"))
            pre, post = lines[i].split("{COMMAND}", 1)
            py, me = mace_python(p, step), str(Path(__file__).resolve())
            cmds = [f"{pre}env OMP_NUM_THREADS={run.mace_threads} {shlex.quote(py)} "
                   f"{shlex.quote(me)} --mace-worker {shlex.quote(str(f.resolve()))}{post} > "
                   f"{shlex.quote(str((jdir / f'worker_{k + 1:02d}.log').resolve()))} 2>&1 &"
                   for k in range(workers)]
            cmds += ["wait", f"echo finished > {shlex.quote(str((jdir / 'finished').resolve()))}"]
            text = "\n".join(lines[:i] + cmds + lines[i + 1:]) + "\n"
            if run.max_walltime:                           # the step's own block: the jobs' time
                text = add_sbatch_option(text, f"--time={run.max_walltime}")
            name = f"dwf_mace_{step}_{jdir.name}"
            cores = workers * run.mace_threads             # (for a template that asks for cores)
            (jdir / "job.sh").write_text(text.replace("{JOBNAME}", name)
                                         .replace("{NCORES}", str(cores)))
            made.append((jdir, caches))
        return made

    while True:
        left = [t for t in todo if mace_cached(t["cache"], t["key"]) is None]
        if not left:
            if announced:
                log(f"{label}: the MACE jobs have finished ({total:,}/{total:,} relaxations done)")
            return True
        missing = {str(t["cache"]): t["key"] for t in left}
        for job in state["jobs"]:
            if not job.get("ended") and ended(job):
                job["ended"] = True
                if job.get("count") == "end":          # (an older dwf.py's counted at submission)
                    claims = Path(job["dir"]) / "claims"
                    gone = [c for c in job["caches"]   # (read again: it may have finished since)
                            if c in missing and mace_cached(c, missing[c]) is None]
                    started = claims.is_dir() and any(claims.iterdir())
                    for c in gone:
                        if not started or claim_file(claims, Path(c).resolve()).exists():
                            state["tries"][c] = state["tries"].get(c, 0) + 1
        active = [j for j in state["jobs"] if not j.get("ended")]
        busy = {c for j in active for c in j["caches"]}
        new = [t for t in left if str(t["cache"]) not in busy]
        places = run.mace_max_jobs - len(active)       # (mace_max_jobs queued or running at most)
        if new:
            worn = [t for t in new if state["tries"].get(str(t["cache"]), 0) >= MACE_ATTEMPTS]
            if worn:
                names, caches = [t["label"] for t in worn], {str(t["cache"]) for t in worn}
                last = next((Path(j["dir"]) for j in reversed(state["jobs"])
                             if caches & set(j["caches"])), None)
                write_json(state_file, state)          # (the attempts, for --resubmit-failed)
                die(f"{label}: the MACE jobs ended {MACE_ATTEMPTS} times without these "
                    f"relaxations: {', '.join(names[:8])}"
                    + (f" and {len(names) - 8} more" if len(names) > 8 else "")
                    + (f". The last of their jobs, {last.name}: {why_ended(last)}" if last else "")
                    + f". The workers' logs and Slurm output are in {d}/job_*/; once the cause is "
                    "fixed, rerun with --resubmit-failed")
        if new and p.cl.prepare_only:
            shutil.rmtree(d / "prepared", ignore_errors=True)
            write_jobs(new, lambda j: d / "prepared" / f"job_{j + 1:03d}", run.mace_max_jobs)
            log(f"{label}: the scripts of the MACE jobs for {len(new)} relaxations are in "
                f"{d / 'prepared'} - check them, then rerun without --prepare-only")
            return False
        if new and places > 0:
            first = 1 + max([int(x.name[4:]) for x in d.glob("job_*") if x.name[4:].isdigit()],
                            default=0)
            for jdir, caches in write_jobs(new, lambda j: d / f"job_{first + j:03d}", places):
                jobid = send_job(jdir, "job.sh", args)
                state["jobs"].append({"jobid": jobid, "dir": str(jdir), "caches": caches,
                                      "time": time.time(), "count": "end"})
                log(f"{label}: submitted MACE job {jobid} ({jdir.name}: {len(caches)} "
                    f"relaxation{'s' if len(caches) > 1 else ''})")
            write_json(state_file, state)
            continue
        waiting = len(new)                             # (no place free for them yet)
        write_json(state_file, state)
        if p.cl.no_wait:
            log(f"{label}: {len(left)} MACE relaxations still queued or running (jobs "
                + ", ".join(j["jobid"] for j in active) + ")"
                + (f", {waiting} of them waiting for a place (mace_max_jobs "
                   f"{run.mace_max_jobs})" if waiting else "") + "; rerun the same command later")
            return False
        if stop.exists():
            stop.unlink()
            log(f"found {stop.name} - stopping. Jobs already submitted keep running; rerun the "
                "same command to carry on.")
            sys.exit(0)
        if not announced:
            log(f"{label}: waiting for {len(left)} MACE relaxations (jobs "
                + ", ".join(j["jobid"] for j in active) + f"; polling every {args.poll} s; "
                "each job's progress.txt counts its own)")
            announced = True
        done = total - len(left)
        if done != last_done and time.time() - last_time >= PROGRESS_EVERY:
            log(f"{label}: {done:,}/{total:,} MACE relaxations done ({100 * done / total:.0f}%); "
                f"{len(active)} job{'s' if len(active) != 1 else ''} queued or running"
                + (f", {waiting:,} relaxations waiting for a place" if waiting else ""))
            last_done, last_time = done, time.time()
        time.sleep(args.poll)


# A relaxation's claim file in a MACE job's claims/ folder (cache: its cache file, as the job's
# tasks.json names it, resolved).
#   Called by: mace_claim(), mace_jobs()
def claim_file(folder, cache):
    return Path(folder) / hashlib.md5(str(cache).encode()).hexdigest()[:16]


# Takes a relaxation of a MACE job's queue for this worker: creates its claim file in the job's
# claims/ folder (claim_file), which only one worker can do (O_CREAT | O_EXCL is atomic, Lustre's
# too). Returns whether this worker got it.
#   Called by: mace_worker()
#   Calls:     claim_file()
def mace_claim(folder, cache):
    f = claim_file(folder, cache)
    try:
        fd = os.open(f, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as fh:
        fh.write(f"{socket.gethostname()} {os.getpid()} {time.strftime('%H:%M:%S')} {cache}\n")
    return True


# Adds a relaxation a worker has finished (or that failed) to its MACE job's progress.txt
# (write_progress): a line at the end, e.g. '[14:02:11] 201/398  d555 ...: 432 atoms, 11 steps,
# 326 s' (the count so far: tail -f follows it), and the count as the first line, e.g. '201/398
# relaxations done (1 failed), the last at 14:02:11'. The job's workers take turns (a lock file,
# made as a claim is: O_CREAT | O_EXCL; one older than a minute was left by a worker killed holding
# it). A few small file operations per relaxation, which takes minutes. Anything that stops it
# costs only this file: the result is already in its cache file.
#   Called by: mace_worker()
#   Calls:     write_progress()
def mace_progress(folder, total, entry):
    lock, f = folder / "progress.lock", folder / "progress.txt"
    entry = " ".join(str(entry).split())[:300]    # one line, however long the error
    for _ in range(600):                          # about a minute at most
        try:
            os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > 60:
                    lock.unlink()
                    continue
            except OSError:
                pass
            time.sleep(0.1)
        except OSError:
            return
    else:
        return
    try:
        try:
            old = [x for x in f.read_text(errors="replace").splitlines()[2:] if x.strip()]
        except OSError:
            old = []
        failed = sum(": FAILED: " in x for x in old + [entry])
        done, now = len(old) + 1 - failed, time.strftime("%H:%M:%S")
        write_progress(f, f"{done:,}/{total:,} relaxations done"
                       + (f" ({failed:,} failed)" if failed else "") + f", the last at {now}",
                       [f"[{now}] {done:,}/{total:,}  {entry}"])
    finally:
        try:
            lock.unlink()
        except OSError:
            pass


# What a MACE job's worker does (python3 dwf.py --mace-worker tasks.json, written by mace_jobs):
# works through the job's queue, biggest first, with the job's other workers: each relaxation
# that its cache file doesn't hold yet and no other worker has taken (mace_claim) it relaxes
# (mace_run, with the model file and settings given there, or a task's own model, head, charge,
# fmax, steps and rattle), and writes the result to its cache file as soon as it has it
# (save_mace_result); an error goes into the cache file instead, for the workflow to report. Each
# one it finishes, or that fails, it then adds to the job's progress.txt (mace_progress). Ends
# when nothing is left to take, saying how many it did. (A job written by an older dwf.py gives
# each worker a file of its own, tasks_KK.json, which it works through the same way, without
# progress.txt.)
#   Called by: main()
#   Calls:     atoms_from_dict(), mace_claim(), mace_progress(), mace_run(), save_mace_result()
def mace_worker(path):
    path = Path(path)
    spec = json.loads(path.read_text())
    claims = path.parent / "claims"
    claims.mkdir(exist_ok=True)
    calcs, mine, began = {}, 0, time.time()
    log(f"MACE worker: {len(spec['tasks'])} relaxations in {path.name}, shared with the job's "
        f"other workers; model {spec['model']}, {spec['threads']} threads")
    for t in spec["tasks"]:
        cache = Path(t["cache"])
        try:
            if json.loads(cache.read_text()).get("key") == t["key"]:
                continue                          # done already (an earlier job)
        except (OSError, ValueError):
            pass
        if not mace_claim(claims, cache):
            continue                              # another worker has it
        mine += 1
        start = time.time()
        try:
            model, head = t.get("model") or spec["model"], t.get("head")
            if (model, head) not in calcs:
                import warnings
                warnings.filterwarnings("ignore")
                import torch
                from mace.calculators import MACECalculator, mace_mp
                torch.set_num_threads(spec["threads"])
                calcs[(model, head)] = (
                    MACECalculator(model_paths=model, head=head, device=spec["device"],
                                   default_dtype="float64") if head is not None else
                    mace_mp(model=model, device=spec["device"], default_dtype="float64"))
            res = mace_run(calcs[(model, head)], atoms_from_dict(t["atoms"]),
                           t.get("fmax") or spec["fmax"], t.get("steps") or spec["steps"],
                           t.get("charge"), t.get("rattle"))
            save_mace_result(cache, t["key"], res)
            what = f"{len(res[0])} atoms, {res[3]} steps, {time.time() - start:.0f} s"
        except Exception as err:                  # anything MACE or torch raises
            save_mace_result(cache, t["key"], error=f"{type(err).__name__}: {err}")
            what = f"FAILED: {type(err).__name__}: {err}"
        log(f"{t['label']}: {what}")
        if path.name == "tasks.json":             # the job's shared queue
            mace_progress(path.parent, len(spec["tasks"]), f"{t['label']}: {what}")
    log(f"MACE worker: nothing left to take; {mine} relaxations here in "
        f"{time.time() - began:.0f} s")


# --------------------------------------------------------------------------
# Odds and ends
# --------------------------------------------------------------------------
# Writes a JSON file, made readable, through a temporary file so that it is never left half
# written (state.json, results.json).
#   Called by: Project.__init__(), Project.save_results(), Project.save_state(), mace_jobs(),
#              migrate_workdir(), move_old_phases(), phonon_worker(), run_training(), size_ladder()
def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=float))
    tmp.replace(path)


# A text table for summary.txt and the log: a title, a header row, a rule and the rows, in columns
# as wide as needed (numbers to the right, text to the left). Floats are shown with 4 decimals,
# None as blank.
#   Called by: drelax_ranking(), ladder_table(), limits_table(), problem_table(), step_ccd(),
#              step_choose(), step_converge(), step_corrections(), step_distort(),
#              step_electronic(), step_excite(), step_filter(), step_formation(), step_generate(),
#              step_interest(), step_limits(), step_lineshape(), step_phases(), step_prerelax(),
#              step_rank(), step_ref_phases(), step_relaxations(), step_scan(), step_validate(),
#              step_zpl()
def format_table(header, rows, title=""):
    cells = [[("" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v)))
              for v in r] for r in rows]
    width = [max([len(h)] + [len(r[i]) for r in cells]) for i, h in enumerate(header)]

    # Whether a column holds only numbers (or blanks): those are aligned to the right.
    def numeric(i):
        try:
            [float(r[i]) for r in cells if r[i]]
            return True
        except ValueError:
            return False

    right = [numeric(i) for i in range(len(header))]
    fmt = lambda r: "  ".join(c.rjust(w) if k else c.ljust(w)                 # noqa: E731
                              for c, w, k in zip(r, width, right)).rstrip()
    lines = ([title, ""] if title else []) + [fmt(header), "-" * len(fmt(header))]
    return "\n".join(lines + [fmt(r) for r in cells]) + "\n"


# The species counts of a structure, e.g. {'Ca': 32, 'O': 31, 'Sb': 1}.
#   Called by: phase_results(), reduced(), step_limits(), step_ref_phases()
def counts(atoms):
    syms = atoms.get_chemical_symbols()
    return {s: syms.count(s) for s in sorted(set(syms))}


# The tag of a charge state in job names (which can't hold + or -): q0, qp1, qm1, ...
#   Called by: step_ccd(), step_distort(), step_excite(), step_filter(), step_lineshape(),
#              step_rank(), step_relaxations(), step_screen(), step_validate(), step_zpl()
def qtag(q):
    return "q0" if q == 0 else f"q{'p' if q > 0 else 'm'}{abs(q)}"


# ==========================================================================================
# PHASE A: THE HOST (flowchart A1-A3)
# ==========================================================================================
# The .param keywords every host and competing-phase job sets: no charge, and no starting spin
# left over from a defect calculation (a phase that needs one gives it on its line in %block
# phases).
HOST_OVERRIDES = {"charge": None, "spin": None}


# The pseudopotential entries of a .cell that don't look norm-conserving (DFPT needs NCP): a file
# that isn't a .recpot or named NCP, or an on-the-fly string without NCP in it; no species_pot
# block at all means CASTEP's default on-the-fly ultrasoft potentials.
#   Called by: step_dielectric()
#   Calls:     block_rows(), parse_castep_input()
def ultrasoft_entries(cell_text):
    pots = [it for it in parse_castep_input(cell_text)
            if it["type"] == "block" and it["name"] == "species_pot"]
    if not pots:
        return ["(no species_pot: CASTEP's default ultrasoft potentials)"]
    bad = []
    for row in block_rows(pots[0]):
        entry = " ".join(row[1:]) if len(row) > 1 else row[0]
        if not (entry.lower().endswith(".recpot") or "ncp" in entry.lower()):
            bad.append(" ".join(row))
    return bad


# --------------------------------------------------------------------------
# A1: cut-off and k-points
# --------------------------------------------------------------------------
# The production .param: the project's .param without charge or spin (every job sets its own)
# and, given a cut-off, with cut_off_energy set to it (any cut_off_energy or basis_precision in it
# removed). Returns (text, the keywords removed).
#   Called by: step_converge()
#   Calls:     strip_comment()
def production_param_text(text, cutoff=None):
    drop = ("charge", "spin") + (("cut_off_energy", "basis_precision") if cutoff else ())
    keep, gone = [], []
    for line in text.splitlines():
        m = re.match(r"\s*(\w+)\b", strip_comment(line))
        if m and m.group(1).lower() in drop:
            gone.append(m.group(1).lower())
        else:
            keep.append(line)
    out = "\n".join(keep).rstrip() + "\n" + (f"cut_off_energy : {cutoff:g} eV\n" if cutoff else "")
    return out, [k for k in gone if k in ("charge", "spin")]


# The k-point spacing (1/Ang, CASTEP's 2*pi units) that a Monkhorst-Pack grid of a cell gives: the
# finest of its three directions, so that every later cell (kgrid) is sampled at least as
# densely as this grid in each direction (and exactly as densely for a cell like the host's).
#   Called by: step_converge()
#   Calls:     recip_lengths()
def grid_spacing(cell, grid):
    return float(min(L / n for L, n in zip(recip_lengths(cell), grid)))


# The smallest setting whose energy per atom, and that of every better setting, is within tol of
# the best one (the last of values, which must be ordered from worst to best). None if only the
# best one is (the list does not show convergence).
#   Called by: step_converge()
def converged_value(values, energies, tol):
    best = energies[-1]
    for i in range(len(values) - 1):
        if all(abs(e - best) < tol for e in energies[i:]):
            return values[i]
    return None


# A1: converges the k-point grid, then the plane-wave cut-off, with single points of the host
# primitive cell as given. First every Monkhorst-Pack grid of kgrid_list at the cheap cut-off
# converge_kcutoff (default the lowest of converge_cutoffs), in A_host/jobs/converge/kpoints; then
# every cut-off in converge_cutoffs on the converged grid, in A_host/jobs/converge/cutoff. Each
# picks the coarsest grid (smallest cut-off) whose energy per atom, and that of every finer value,
# is within converge_tol of the finest. The production k-point spacing, which every later cell
# uses, is the one the converged grid has (grid_spacing). With kspacing set, the k-point scan is
# skipped and the cut-offs run on the grid that spacing gives.
# Writes A_host/production.param (any charge or spin in the .param left out: every job sets its
# own). With converge : false, the .param and k-points are used as given.
#   Called by: main() (step A1)
#   Calls:     Project.batch(), Project.file(), Project.kspacing(), converged_value(),
#              format_table(), grid_spacing(), production_param_text(), read_structure_file(),
#              kgrid()
def step_converge(p):
    d = p.phase_dir("A")
    text = p.file("param").read_text()
    if production_param_text(text)[1]:
        log(f"note: {p.file('param').name} sets " + " and ".join(production_param_text(text)[1])
            + "; production.param leaves them out (every job sets its own)")
    if not p.s.converge:
        m = re.search(r"^\s*cut_off_energy\s*[:=\s]\s*([\d.]+)", text, re.I | re.M)
        (d / "production.param").write_text(production_param_text(text)[0])
        return {"cutoff": float(m.group(1)) if m else None, "kspacing": p.kspacing(),
                "production_param": "A_host/production.param", "_note": "converge : false - "
                "the project's .param and k-points are used as they are"}
    prim = read_structure_file(p.file("host_cell"))
    n, tol = len(prim), p.s.converge_tol
    cuts = sorted(p.s.converge_cutoffs)
    kcut = p.s.converge_kcutoff or cuts[0]
    rows, energies = [], {}

    # Runs single points of the primitive cell (one per (name, grid, cut-off)) in
    # A_host/jobs/converge/<sub>; their energies per atom, or None while jobs are queued.
    def run(sub, todo):
        jobs = [{"name": name, "atoms": prim, "kind": "sp", "symmetry": True, "kgrid": list(g),
                 "overrides": dict(HOST_OVERRIDES, cut_off_energy=f"{c:g} eV")}
                for name, g, c in todo]
        res = p.batch(f"A_host/jobs/converge/{sub}", jobs, "converge", text)
        if not res["complete"]:
            return None
        bad = [k for k, v in res["jobs"].items() if v["energy"] is None or not v["scf_converged"]]
        if bad:
            die("the convergence single points " + ", ".join(bad) + " have no converged energy "
                "- check their .castep files in " + str(p.dir / "A_host/jobs/converge" / sub))
        out = {name: res["jobs"][name]["energy"] / n for name, _, _ in todo}
        energies.update(out)
        return out

    # The table so far, printed before stopping when a scan has not converged.
    def table():
        how = (f"cut-offs at kspacing {p.s.kspacing:g}" if p.s.kspacing else
               f"k-points at {kcut:g} eV, then cut-offs on the converged grid")
        return format_table(["cut-off", "k-grid", "E/atom (eV)", "vs best"], rows,
                            f"Energy per atom: {how} (tolerance {tol:g} eV/atom)")

    kspacing = p.s.kspacing
    if kspacing:
        grid = tuple(kgrid(prim.cell, kspacing))
    else:
        grids = [tuple(g) for g in p.s.converge_kgrids]                      # coarse to fine
        names = {g: "k_" + "x".join(map(str, g)) for g in grids}
        e = run("kpoints", [(names[g], g, kcut) for g in grids])
        if e is None:
            return None
        e_k = [e[names[g]] for g in grids]
        grid = converged_value(grids, e_k, tol)
        rows += [[f"{kcut:g} eV", "x".join(map(str, g)), x, x - e_k[-1]]
                 for g, x in zip(grids, e_k)]
        if grid is None:
            print(table())
            die(f"the k-points have not converged: only the finest grid is within {tol:g} "
                "eV/atom of itself (table above). Raise converge_kgrid_max and rerun")
        kspacing = grid_spacing(prim.cell, grid)
    e = run("cutoff", [(f"cut_{c:g}", grid, c) for c in cuts])
    if e is None:
        return None
    e_cut = [e[f"cut_{c:g}"] for c in cuts]
    cutoff = converged_value(cuts, e_cut, tol)
    rows += [[f"{c:g} eV", "x".join(map(str, grid)), x, x - e_cut[-1]]
             for c, x in zip(cuts, e_cut)]
    if cutoff is None:
        print(table())
        die(f"the cut-off has not converged: only the highest is within {tol:g} eV/atom of "
            "itself (table above). Add higher cut-offs to converge_cutoffs and rerun")
    (d / "production.param").write_text(production_param_text(text, cutoff)[0])
    return {"cutoff": cutoff, "kgrid": list(grid), "kspacing": kspacing,
            "production_param": "A_host/production.param", "energies": energies,
            "_table": table(), "_note": f"k-point grid {'x'.join(map(str, grid))} (spacing "
                                        f"{kspacing:.4f} 1/A), cut-off {cutoff:g} eV"}


# --------------------------------------------------------------------------
# A2: bulk properties
# --------------------------------------------------------------------------
# A2.1: relaxes the host primitive cell (lattice and positions, with symmetry) at the production
# settings. Saves the relaxed cell as A_host/host_relaxed.cell (with your .cell's species_pot and
# the rest, a complete .cell to run by hand too), and the energy per atom, e_bulk.
#   Called by: main() (step A2.1)
#   Calls:     Project.batch(), Project.cell_text(), Project.file(), Project.kspacing(),
#              Project.production_param(), read_structure_file(), final_atoms(), kgrid(),
#              parse_castep_input(), write_cell()
def step_relax(p):
    d = p.phase_dir("A")
    prim = read_structure_file(p.file("host_cell"))
    res = p.batch("A_host/jobs/relax", [{"name": "host", "atoms": prim, "kind": "geom",
                                         "fix_cell": False, "symmetry": True,
                                         "overrides": HOST_OVERRIDES}],
                  "relax", p.production_param(), spacing=p.kspacing())
    if not res["complete"]:
        return None
    r = res["jobs"]["host"]
    if r["status"] != "done" or r["energy"] is None:
        die(f"the host relaxation did not finish: {r['reason'] or 'see ' + r['dir']}")
    atoms = final_atoms(res["calcs"]["host"])
    if atoms is None:
        die(f"could not read the relaxed host from host-out.cell or host.castep in {r['dir']}: "
            "its atoms don't match those of the host cell")
    keep = [it for it in parse_castep_input(p.cell_text())
            if not (it["type"] == "block" and it["name"] in STRUCT_BLOCKS | DROP_BLOCKS)
            and not (it["type"] == "kw" and it["key"] in SET_KEYS)]
    write_cell(d / "host_relaxed.cell", atoms, kgrid(atoms.cell, p.kspacing()), keep,
                  f"host relaxed in A2.1 ({atoms.get_chemical_formula()})")
    a, b, c, al, be, ga = atoms.cell.cellpar()
    e_bulk = r["energy"] / len(atoms)
    return {"cell": "A_host/host_relaxed.cell", "energy": r["energy"], "e_bulk": e_bulk,
            "natoms": len(atoms), "lattice": {"a": a, "b": b, "c": c, "alpha": al, "beta": be,
                                              "gamma": ga},
            "_note": f"a b c = {a:.4f} {b:.4f} {c:.4f} A, e_bulk = {e_bulk:.6f} eV/atom"}


# A2.2: band edges of the relaxed host from a single point on a Gamma-centred grid at
# bands_kspacing (default half the production spacing): VBM, CBM and gap from the .bands file,
# which the Fermi-level analysis (E3) also uses.
#   Called by: main() (step A2.2)
#   Calls:     Project.batch(), Project.host(), Project.kspacing(), Project.production_param(),
#              band_edges(), read_bands(), kgrid()
def step_bands(p):
    host = p.host()
    spacing = p.s.bands_kspacing or p.kspacing() / 2
    res = p.batch("A_host/jobs/bands", [{"name": "bands", "atoms": host, "kind": "sp",
                                         "symmetry": True, "kgrid": kgrid(host.cell, spacing),
                                         "kpoint_offset": "gamma", "overrides": HOST_OVERRIDES}],
                  "bands", p.production_param())
    if not res["complete"]:
        return None
    calc = res["calcs"]["bands"]
    path = calc.dir / f"{calc.seed}.bands"
    if res["jobs"]["bands"]["status"] != "done" or not path.exists():
        die(f"the band-edge single point did not finish or wrote no .bands ({calc.dir})")
    b = read_bands(path)
    vbm, cbm = band_edges(b)
    return {"vbm": vbm, "cbm": cbm, "gap": cbm - vbm, "bands_file": str(path.relative_to(p.dir)),
            "kgrid": res["jobs"]["bands"]["kgrid"],
            "_note": f"VBM {vbm:.4f} eV, CBM {cbm:.4f} eV, gap {cbm - vbm:.3f} eV"}


# A2.3: the dielectric constant for the charge corrections. A DFPT run (dielectric_task, default
# phonon+efield, at Gamma) on the relaxed host, with dielectric_param and dielectric_cell if given
# (CASTEP's DFPT needs norm-conserving pseudopotentials, fixed occupancies and no spin
# polarisation: without dielectric_param the production .param is used with fix_occupancy : true
# and spin_polarized : false, and a species_pot that doesn't look norm-conserving is warned about);
# the static value (electronic + ionic) is the average of the diagonal of CASTEP's DC permittivity.
# With dielectric set in the project file, that is used instead and nothing runs.
#   Called by: main() (step A2.3)
#   Calls:     Project.batch(), Project.cell_text(), Project.file(), Project.host(),
#              Project.kspacing(), Project.production_param(), read_permittivity(),
#              ultrasoft_entries()
def step_dielectric(p):
    if p.s.dielectric:
        return {"eps0": p.s.dielectric, "eps_inf": p.s.dielectric_inf, "source": "project file",
                "_note": f"dielectric {p.s.dielectric:g} from the project file"}
    host = p.host()
    task = p.s.dielectric_task
    extra = ("%block phonon_kpoint_list\n  0.0 0.0 0.0 1.0\n%endblock phonon_kpoint_list"
             if "phonon" in task.lower() else None)
    param = p.file("dielectric_param").read_text() if p.s.dielectric_param else p.production_param()
    over = ({} if p.s.dielectric_param else
            dict(HOST_OVERRIDES, fix_occupancy="true", spin_polarized="false"))
    cell = p.file("dielectric_cell").read_text() if p.s.dielectric_cell else None
    usp = ultrasoft_entries(cell or p.cell_text())
    if usp:
        log("WARNING: CASTEP's DFPT needs norm-conserving pseudopotentials, but these don't look "
            "norm-conserving: " + "; ".join(usp) + ". Give dielectric_cell (a .cell with an NCP "
            "species_pot) and dielectric_param, or set dielectric, if this job fails.")
    res = p.batch("A_host/jobs/dielectric", [{"name": "dielectric", "atoms": host, "kind": "sp",
                                              "task": task, "symmetry": True, "cell_extra": extra,
                                              "overrides": over}],
                  "dielectric", param, cell_text=cell, spacing=p.kspacing())
    if not res["complete"]:
        return None
    eps_inf, eps0 = read_permittivity(res["calcs"]["dielectric"].output())
    if eps_inf is None:
        die("no permittivity table in the dielectric run's .castep "
            f"({res['calcs']['dielectric'].castep}); set dielectric in the project file instead")
    if eps0 is None:
        die(f"task {task} gave only the optical permittivity; the charge correction needs the "
            "static one (electronic + ionic): use dielectric_task : phonon+efield, or set "
            "dielectric")
    e0, ei = float(np.trace(eps0) / 3), float(np.trace(eps_inf) / 3)
    aniso = float(np.ptp(np.diag(eps0)))
    if aniso > 0.1 * e0:
        log(f"WARNING: the dielectric tensor is anisotropic (diagonal {np.diag(eps0).round(3)}); "
            "the charge corrections use its average")
    return {"eps0": e0, "eps_inf": ei, "eps0_tensor": eps0, "eps_inf_tensor": eps_inf,
            "source": "DFPT", "_note": f"epsilon_0 {e0:.3f}, epsilon_inf {ei:.3f}"}


# --------------------------------------------------------------------------
# A3: competing phases and chemical potentials (A3 at the screening settings for C3; the same
# functions give E1's at the production settings)
# --------------------------------------------------------------------------
# The competing phases, from %block phases (name, structure file, options: molecule, metal
# (fix_occupancy : false unless the line says otherwise), kspacing=..., anything else a .param
# keyword) or from doped (phases_from : doped, with the Materials Project: every phase it
# chooses for the host and each dopant, each with eah, its energy above the hull; its molecules
# in boxes of phases_molecule_box, molecule_in_box; every one tidied as tidy_phase tidies it: the
# moments' noise gone, the solids as symmetrised primitive cells). A3.1 and E1.1 each take their
# part of it (phases_for). Returns a list of (name, Atoms, options dict).
#   Called by: step_phases(), step_ref_phases()
#   Calls:     doped_phases(), molecule_in_box(), read_structure_file(), canonical_key(),
#              tidy_phase()
def phase_list(p):
    if p.s.phases:
        out = []
        for row in p.s.phases:
            if len(row) < 2:
                die("each line of %block phases is  name  structure-file  [options]")
            opts = {"overrides": {}}
            for o in row[2:]:
                k, eq, v = o.partition("=")
                if not eq:
                    opts[k.lower()] = True
                elif k.lower() == "kspacing":
                    opts["kspacing"] = float(v)
                else:
                    opts["overrides"][canonical_key(k)] = v
            if opts.get("metal"):
                opts["overrides"].setdefault("fix_occupancy", "false")
            out.append((row[0], read_structure_file(p.base / row[1]), opts))
        return out
    if p.s.phases_from == "doped":
        out, size = doped_phases(p), p.s.phases_molecule_box
        mols = [name for name, _, opts in out if opts.get("molecule")]
        if mols:
            log(f"doped's molecules ({', '.join(mols)}) in boxes of {size:g} A "
                "(phases_molecule_box)")
        return [tidy_phase(name, molecule_in_box(name, atoms, size) if opts.get("molecule")
                           else atoms, opts) for name, atoms, opts in out]
    return []


# tidy_phase: moments smaller than this (muB) are the Materials Project's numerical noise, and
# spglib's tolerance (Ang) when a solid is made symmetric
MAGMOM_MIN = 0.1
PHASE_SYMPREC = 0.01


# A phase from doped as it is best run. The Materials Project's structures keep the magnetic
# moments its own runs ended with, which AseAtomsAdaptor makes initial moments and write_cell
# SPIN= lines: noise of a thousandth of a muB on one atom of a non-magnetic metal is enough for
# CASTEP to take that atom for another species and lose most of the symmetry (Na in alpha-Mn's
# structure kept 6 of its 24 operations). So the moments are rounded to 0.1 muB and those below
# MAGMOM_MIN dropped; a phase with none left runs without spin polarisation (spin_polarized :
# false, unless its options set it, as O2's do). A solid then becomes its primitive cell, its
# positions and lattice made exactly symmetric (pymatgen's SpacegroupAnalyzer, PHASE_SYMPREC,
# get_primitive_standard_structure), so CASTEP's symmetry_generate finds every operation; a
# Materials Project structure needn't be primitive. Its options get the space group and the atoms
# the Materials Project's cell had. Without pymatgen, or if it fails, the structure stays as it
# is. Molecules keep their boxes (they run without symmetry).
#   Called by: phase_list()
def tidy_phase(name, atoms, opts):
    out, opts = atoms.copy(), copy.deepcopy(opts)
    m = np.round(out.get_initial_magnetic_moments(), 1)
    m[np.abs(m) < MAGMOM_MIN] = 0.0
    out.set_initial_magnetic_moments(m)
    if not m.any():
        opts["overrides"].setdefault("spin_polarized", "false")
    if opts.get("molecule"):
        return name, out, opts
    try:
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    except ImportError:
        return name, out, opts
    props = {"magmom": m.tolist()} if m.any() else None
    try:
        s = Structure(np.asarray(out.cell), out.get_chemical_symbols(),
                      out.get_scaled_positions(), site_properties=props)
        sga = SpacegroupAnalyzer(s, symprec=PHASE_SYMPREC)
        prim = sga.get_primitive_standard_structure(keep_site_properties=bool(props))
        group = sga.get_space_group_symbol()
    except Exception as err:                       # (an older pymatgen, a structure spglib
        log(f"note: {name} could not be symmetrised ({err}), so it runs as the Materials "
            "Project gave it")                     #  can't handle)
        return name, out, opts
    new = Atoms([str(sp) for sp in prim.species], cell=prim.lattice.matrix,
                scaled_positions=prim.frac_coords, pbc=True)
    if props:
        new.set_initial_magnetic_moments(np.asarray(prim.site_properties["magmom"], float))
    opts.update(spacegroup=group, given_atoms=len(atoms))
    return name, new, opts


# The least vacuum (A) molecule_in_box leaves between a molecule and its periodic images.
MOLECULE_VACUUM = 5.0


# A molecule doped made (H2, N2, O2, F2 or Cl2, each in a 30 A box) moved into a box of side size
# (phases_molecule_box, 15 A by default). In a plane-wave code the vacuum costs as much as the
# atoms: a 30 A box has eight times the volume of a 15 A one, so about eight times the plane
# waves, memory and time, which a small neutral molecule doesn't need. The box has doped's shape
# (size + 0.01, size and size - 0.01 A along x, y and z, no two sides equal), and the molecule
# keeps the geometry doped gave it (its atoms taken together across the old box's edges), centred.
# A cell that is already that box is left as it is, so phases_molecule_box : 30 ang keeps
# doped's own structures. Stops if the box would leave less than MOLECULE_VACUUM between the
# molecule and its images.
#   Called by: phase_list()
def molecule_in_box(name, atoms, size):
    lengths = np.array([size + 0.01, size, size - 0.01])
    cell = np.asarray(atoms.cell)
    if np.allclose(cell, np.diag(lengths), atol=1e-6):
        return atoms
    d = atoms.get_positions() - atoms.get_positions()[0]       # from the first atom
    frac = np.linalg.solve(cell.T, d.T).T
    d = (frac - np.round(frac)) @ cell                          # nearest images
    lo, hi = d.min(axis=0), d.max(axis=0)
    gap = float(np.min(lengths - (hi - lo)))
    if gap < MOLECULE_VACUUM:
        die(f"phases_molecule_box : {size:g} ang leaves {gap:.1f} A between {name} and its "
            f"images (at least {MOLECULE_VACUUM:g} A is needed)")
    out = atoms.copy()
    out.set_cell(np.diag(lengths))
    out.set_positions(d - (lo + hi) / 2 + lengths / 2)
    return out


# How far beyond phases_e_above_hull (eV/atom) doped_phases still downloads Materials Project
# entries for doped: doped works out each entry's energy above the hull again, which can differ a
# little from the Materials Project's own. Entries further up can't be chosen, and don't move the
# hull.
MP_HULL_MARGIN = 0.1


# One Materials Project API query (route e.g. "materials/thermo", params as the API takes them),
# made as the Materials Project's tips for large downloads ask: a list in one request, only the
# fields asked for, and pages of 1000 (_limit and _skip, as the mp-api client pages). Logs what
# came (what: a label). Raises pymatgen's MPRestError in pymatgen's words, which doped and
# doped_phases() read.
#   Called by: lean_mp_entries()
def mp_query(mpr, route, params, what):
    from pymatgen.ext.matproj import MPRestError
    data, size, start = [], 0, time.time()
    while True:
        q = dict(params, _limit=1000, **({"_skip": len(data)} if data else {}))
        try:
            r = mpr.session.get(f"{mpr.preamble.rstrip('/')}/{route}/", params=q, timeout=300)
        except Exception as err:                      # no connection, a timeout, ...
            raise MPRestError(str(err)) from err
        if r.status_code != 200:
            raise MPRestError(f"REST query returned with error status code {r.status_code}. "
                              f"Content: {r.content[:1000]}")
        try:
            reply = json.loads(r.text)
            page = reply["data"]
        except (ValueError, KeyError, TypeError):
            raise MPRestError(f"no data in the reply: {r.text[:1000]}") from None
        data, size = data + page, size + len(r.content)
        total = (reply.get("meta") or {}).get("total_doc")
        if len(page) < 1000 or (total is not None and len(data) >= total):
            break
    log(f"  Materials Project: {len(data)} {what}, {size / 1e6:.2f} MB in "
        f"{time.time() - start:.1f} s")
    return data


# A stand-in for pymatgen's MPRester.get_entries while doped runs (doped_phases). pymatgen asks
# for every material of a chemical system on every hull the Materials Project has (GGA/GGA+U,
# r2SCAN and their mix), structures and all, and decodes it all. For chemical systems, this asks
# only for the GGA/GGA+U hull's documents (thermo_types: the entries
# MaterialsProject2020Compatibility keeps), only up to eah_max above that hull, and only the
# fields doped wants; its first query also takes the systems in expect (those doped asks for
# next: the host's and every dopant's), all in one request, and then the summaries of all their
# materials, a few hundred to a request. What comes is kept in folder (thermo_GGA_GGA+U.json,
# summary.json), so no system or material is asked for twice, in this run or a later one. The
# entries are then treated as pymatgen treats them: decoded, entry ids as text,
# MaterialsProject2020Compatibility applied, property_data in data, the summary's fields in
# data["summary"]. Any other query (doped's key check) goes to original, pymatgen's own, once a
# run; so does a chemical system the Materials Project won't take this way (400, 422).
#   Called by: doped_phases()
#   Calls:     mp_query()
def lean_mp_entries(folder, eah_max, original, expect=()):
    import warnings
    from monty.json import MontyDecoder
    from pymatgen.core import Composition
    from pymatgen.ext.matproj import MPRestError
    decode = MontyDecoder().process_decoded
    files = {"thermo": folder / "thermo_GGA_GGA+U.json", "summary": folder / "summary.json"}
    kept, answered = {}, {}           # the files' items (by kind); pymatgen's own answers

    def held(kind, **meta):           # what is kept of a kind, if it was kept with these settings
        if kind not in kept:
            try:
                got = json.loads(files[kind].read_text())
                kept[kind] = got["items"] if all(got.get(k) == v for k, v in meta.items()) else {}
            except (OSError, ValueError, KeyError, TypeError):
                kept[kind] = {}
        return kept[kind]

    def keep(kind, **meta):
        folder.mkdir(parents=True, exist_ok=True)
        part = files[kind].with_suffix(".part")
        part.write_text(json.dumps(dict(meta, items=kept[kind])))
        part.replace(files[kind])

    def system(doc):
        cs = doc.get("chemsys")
        if isinstance(cs, str) and cs:
            return "-".join(sorted(cs.split("-")))
        return "-".join(sorted(el.symbol for el in Composition(doc["formula_pretty"])))

    def get_entries(mpr, criteria, *args, compatible_only=True, property_data=None,
                    summary_data=None, **kw):
        items = criteria.split(",") if isinstance(criteria, str) else list(criteria)
        own = dict(compatible_only=compatible_only, property_data=property_data,
                   summary_data=summary_data, **kw)
        if args or not any("-" in c for c in items) or not all(
                re.fullmatch(r"[A-Z][a-z]?(-[A-Z][a-z]?)*", c) for c in items):
            tag = json.dumps([items, [str(a) for a in args], sorted(map(str, own.items()))])
            if tag not in answered:
                answered[tag] = original(mpr, criteria, *args, **own)
            return copy.deepcopy(answered[tag])
        systems = list(dict.fromkeys("-".join(sorted(c.split("-"))) for c in items))
        fields = sorted({"material_id", "chemsys", "entries", *(property_data or [])})
        meta = {"thermo_types": "GGA_GGA+U", "eah_max": round(eah_max, 4), "fields": fields}
        docs_of = held("thermo", **meta)
        missing = [s for s in dict.fromkeys([*systems, *expect]) if s not in docs_of]
        if missing:
            try:
                docs = mp_query(mpr, "materials/thermo", {
                    "chemsys": ",".join(missing), "thermo_types": "GGA_GGA+U",
                    "energy_above_hull_max": meta["eah_max"], "_fields": ",".join(fields),
                    "_all_fields": False},
                    f"thermo documents for {len(missing)} chemical systems")
            except MPRestError as err:
                if "status code 400" not in str(err) and "status code 422" not in str(err):
                    raise
                log(f"  note: the Materials Project wouldn't take the short query "
                    f"({str(err)[:200]}), so pymatgen asks for everything")
                return original(mpr, criteria, *args, **own)
            for s in missing:
                docs_of[s] = []
            for d in docs:
                docs_of.setdefault(system(d), []).append(d)
            keep("thermo", **meta)
        entries = []
        for s in systems:
            for doc in docs_of.get(s, []):
                for raw in (doc.get("entries") or {}).values():
                    e = decode(copy.deepcopy(raw))
                    if isinstance(e.entry_id, dict):  # as pymatgen does from 2026.9.23
                        i = e.entry_id
                        e.entry_id = str(i.get("identifier", "")) + (
                            f"{i.get('separator', '-')}{i['suffix']}" if i.get("suffix") else "")
                    e.data.setdefault("material_id", doc.get("material_id"))
                    for prop in property_data or []:
                        e.data[prop] = decode(copy.deepcopy(doc.get(prop)))
                    entries.append(e)
        if compatible_only:
            from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Failed to guess oxidation states.*")
                entries = MaterialsProject2020Compatibility().process_entries(entries,
                                                                              clean=True)
        if summary_data and entries:
            sfields = sorted({*summary_data, "material_id"})
            sums = held("summary", fields=sfields)
            ids = [str(e.data["material_id"]) for e in entries]
            if any(i not in sums for i in ids):           # first time: every kept material
                every = [i for i in dict.fromkeys(
                    [*ids, *(str(d.get("material_id")) for ds in docs_of.values() for d in ds)])
                    if i not in sums]
                for k in range(0, len(every), 250):
                    for d in mp_query(mpr, "materials/summary", {
                            "material_ids": ",".join(every[k:k + 250]),
                            "_fields": ",".join(sfields), "_all_fields": False},
                            "summary documents"):
                        sums[str(d["material_id"])] = {f: v for f, v in d.items()
                                                       if f != "material_id"}
                for i in every:
                    sums.setdefault(i, {})
                keep("summary", fields=sfields)
            for e in entries:
                e.data["summary"] = {f: decode(copy.deepcopy(v))
                                     for f, v in sums.get(str(e.data["material_id"]), {}).items()}
        return list(dict.fromkeys(entries))
    return get_entries


# Competing phases from doped's CompetingPhases (Materials Project entries within
# phases_e_above_hull of the hull, elemental references and diatomic molecules in a 30 A box,
# which phase_list makes smaller: molecule_in_box). Needs doped and an MP API key (mp_api_key,
# or doped's own configuration). Works with doped 3.1 and later
# (CompetingPhases(energy_above_hull=..., extrinsic=...)) and with doped 2.x and 3.0
# (CompetingPhases(e_above_hull=...), or ExtrinsicCompetingPhases with dopants, whose
# intrinsic_entries are the host's phases): the keywords doped's CompetingPhases takes tell them
# apart. Each keeps an entry's band gap in its own place, so the phases are grouped here as
# doped 3 groups them: molecules, metals (band gap 0 on the Materials Project), then the rest;
# and named as doped 3 names them (formula_spacegroup_EaH_e), each name once. When doped fails,
# the error adds what went wrong underneath (doped 3 calls any failed test query an invalid key)
# and where the key came from, its length and first four characters, never the whole key.
# doped is given one dopant a call: given them all, it ends by building one phase diagram of
# every element together (for reference only), whose convex hull takes hours from about 25
# elements on (a host of two with 33 dopants has 35). It chooses each dopant's phases on their
# own anyway, so one at a time gives the same phases. doped's Materials Project queries go
# through lean_mp_entries() (only what doped can use, every system it will ask for in one
# request, all kept in A_host/materials_project/), and the phases doped chose are kept in
# A_host/phases_doped.json: a run after that (the same formula, dopants and phases_e_above_hull)
# takes them from there and asks nothing.
#   Called by: phase_list()
#   Calls:     Project.host(), lean_mp_entries()
def doped_phases(p):
    import inspect
    import warnings
    from importlib import metadata
    from ase.io import jsonio
    formula = p.host().get_chemical_formula(empirical=True)
    dopants, eah, key = list(p.s.dopants), p.s.phases_e_above_hull, p.s.mp_api_key
    kept_file = p.dir / "A_host" / "phases_doped.json"
    asked = {"formula": formula, "dopants": sorted(dopants), "phases_e_above_hull": eah,
             "version": 2}                                # 2: each phase's eah in its options
    try:
        kept = json.loads(kept_file.read_text())
    except (OSError, ValueError):
        kept = {}
    if kept.get("for") == asked:
        log(f"the competing phases doped chose before ({kept_file.relative_to(p.dir)}; "
            "delete it to ask the Materials Project again)")
        return [(ph["name"], jsonio.decode(json.dumps(ph["atoms"])), ph["opts"])
                for ph in kept["phases"]]
    with warnings.catch_warnings(record=True) as said:     # doped and pymatgen say a lot
        warnings.simplefilter("always")
        try:
            from doped import chemical_potentials as dcp
            from pymatgen.io.ase import AseAtomsAdaptor
        except ImportError as err:
            die(f"phases_from : doped needs doped (pip install doped): {err}")
        try:
            have = metadata.version("doped")
        except metadata.PackageNotFoundError:            # doped from a source folder
            have = "?"
        try:
            from pymatgen.ext import matproj
            lean = getattr(dcp, "MPRester", None) is matproj.MPRester
        except ImportError:
            lean = False
        own = sorted(set(p.host().get_chemical_symbols()))     # the systems doped will ask
        expect = sorted({"-".join(sub) for group in [own, *(sorted({*own, x}) for x in dopants)]
                         for n in range(1, len(group) + 1)
                         for sub in itertools.combinations(group, n)})
        if lean:                                         # doped's queries, the lean way
            original = matproj.MPRester.get_entries
            matproj.MPRester.get_entries = lean_mp_entries(
                p.dir / "A_host" / "materials_project", eah + MP_HULL_MARGIN, original, expect)
        log(f"asking the Materials Project, through doped {have}, for the competing "
            f"phases of {formula}" + (f" with {', '.join(dopants)}" if dopants else ""))
        new = "energy_above_hull" in inspect.signature(dcp.CompetingPhases).parameters
        entries, doing = [], ""
        try:
            for n, x in enumerate(dopants or [None]):    # one dopant a call: see above
                doing = f" (for {x})" if x else ""
                if new and x:                            # doped 3.1 on
                    cp = dcp.CompetingPhases(formula, energy_above_hull=eah, extrinsic=[x],
                                             api_key=key)
                    host, extra = cp.intrinsic_entries, cp.extrinsic_entries
                elif new:
                    host, extra = dcp.CompetingPhases(formula, energy_above_hull=eah,
                                                      api_key=key).entries, []
                elif x:                                  # doped 2.x and 3.0
                    cp = dcp.ExtrinsicCompetingPhases(formula, extrinsic_species=[x],
                                                      e_above_hull=eah, api_key=key)
                    host, extra = cp.intrinsic_entries, cp.entries
                else:
                    host, extra = dcp.CompetingPhases(formula, e_above_hull=eah,
                                                      api_key=key).entries, []
                entries += (host if n == 0 else []) + extra
                if len(dopants) > 1:
                    log(f"  doped: {x} ({n + 1} of {len(dopants)}), {len(extra)} phases")
        except Exception as err:                         # network, API key, ...
            tips = dict.fromkeys(str(w.message) for w in said if "install" in str(w.message))
            msg = (f"doped {have} could not get the competing phases{doing} from the Materials "
                   f"Project: {err}" + "".join(f"\n  doped warned: {t}" for t in tips))
            why = err.__cause__ or err.__context__       # what went wrong underneath: doped 3
            if why is not None and str(why) not in msg:  # calls any failed test query a bad key
                msg += f"\n  the query's own error: {str(why)[:500]}"
            if have.startswith("2."):
                msg += ("\n  doped 2.x refuses the current Materials Project keys: it uses only "
                        "the legacy API, which the Materials Project was closing in September "
                        "2025. Use doped 3 (pip install -U doped, which needs Python 3.10 or "
                        "later), or give the phases yourself in %block phases")
            from pymatgen.core import SETTINGS
            mpkey = str(key or SETTINGS.get("PMG_MAPI_KEY") or "")    # as doped picks it
            said_why = str(why).lower()
            if mpkey and not have.startswith("2."):
                came = f"mp_api_key in {p.path.name}" if key else "PMG_MAPI_KEY in ~/.pmgrc.yaml"
                msg += f"\n  the key ({came}) starts {mpkey[:4]} and has {len(mpkey)} characters"
                if len(mpkey) != 32:
                    msg += "; the Materials Project's keys have 32 (no quotes or spaces)"
                elif any(s in said_why for s in ("max retries", "timed out", "proxyerror")):
                    msg += ("; it was never checked: this machine can't reach the Materials "
                            "Project, so run dwf.py where there is internet (a login node)")
                elif "no module named 'pymatgen" in said_why:   # the reply names a newer one
                    msg += (f"; it works (the Materials Project answered), but pymatgen "
                            f"{metadata.version('pymatgen')} can't read the answer: that needs "
                            "pymatgen 2026.9.23 or later, which needs Python 3.11 or later")
                elif "status code 401" in said_why or "status code 403" in said_why:
                    msg += ("; the Materials Project refused it: compare it with the key on your "
                            "dashboard (materialsproject.org/dashboard; a new key replaces the "
                            "old one, and mp_api_key comes before ~/.pmgrc.yaml)")
            for k in {str(k) for k in (key, SETTINGS.get("PMG_MAPI_KEY"),
                                       os.environ.get("MP_API_KEY")) if k}:
                msg = msg.replace(k, k[:4] + "...")      # doped prints the whole key
            die(msg)
        finally:
            if lean:
                matproj.MPRester.get_entries = original
        kinds = {"molecule": [], "metal": [], "insulator": []}
        for e in entries:
            gap = e.data.get("band_gap")                 # doped 2.x and 3.0 keep it here,
            if gap is None:                              # 3.1 on in the summary
                gap = (e.data.get("summary") or {}).get("band_gap")
            kinds["molecule" if e.data.get("molecule") else "metal" if gap == 0
                  else "insulator"].append(e)
        out, used = [], set()
        for kind, group in kinds.items():
            for e in group:
                above = next((e.data[k] for k in ("e_above_hull", "energy_above_hull")
                              if e.data.get(k) is not None), 0)
                name = e.data.get("doped_name")          # doped 3 names its entries
                if not name:                             # doped 2.x doesn't: name it as 3 would
                    sg = "mmm" if kind == "molecule" else e.structure.get_space_group_info()[0]
                    name = f"{e.name}_{sg}_EaH_{round(above, 3) or 0}"
                name = base = re.sub(r"[^\w.\-]", "_", name)
                n = 1
                while name in used:                      # two polymorphs named alike
                    n += 1
                    name = f"{base}_{n}"
                used.add(name)
                atoms = AseAtomsAdaptor.get_atoms(e.structure)
                opts = {"overrides": {}, "eah": round(float(above), 4)}  # above the hull, eV/atom
                if kind == "molecule":
                    opts["molecule"] = True
                    if e.composition.reduced_formula == "O2":
                        opts["overrides"].update(spin="2", spin_polarized="true")
                if kind == "metal":
                    opts["metal"] = True
                    opts["overrides"].update(fix_occupancy="false")
                out.append((name, atoms, opts))
    if not out:
        die("doped found no competing phases")
    try:
        text = json.dumps({"for": asked, "doped": have, "phases": [
            {"name": n, "atoms": json.loads(jsonio.encode(a)), "opts": o} for n, a, o in out]},
            indent=1)
    except (TypeError, ValueError) as err:          # a structure JSON can't hold
        log(f"note: the phases doped chose can't be kept ({err}), so a later run asks again")
    else:
        kept_file.parent.mkdir(parents=True, exist_ok=True)
        kept_file.write_text(text)
    log(f"doped chose {len(out)} competing phases: " + ", ".join(n for n, _, _ in out))
    return out


# A structure's composition, reduced: ((element, n), ...) in the order of the elements.
#   Called by: phases_for(), step_phases()
#   Calls:     counts()
def reduced(atoms):
    c = counts(atoms)
    g = math.gcd(*c.values())
    return tuple(sorted((el, n // g) for el, n in c.items()))


# Of the competing phases (phase_list's), those made of the host's elements (host_el) and at most
# one of the dopants given: a phase with two dopants limits neither on its own, and
# chemical_limits leaves it out. lowest: each composition only once: of its phases within window
# (eV/atom) of the lowest above the hull (doped's eah), the one with the fewest atoms in its cell
# (then the lowest). Polymorphs can lie within meV/atom of each other, where which comes lowest
# is noise, and the lowest can be a far bigger cell (doped's lowest Na was alpha-Mn's structure:
# 29 atoms, against hcp's 2); window 0 takes the lowest. A phase without an eah
# (%block phases) is always kept.
#   Called by: step_phases(), step_ref_phases()
#   Calls:     reduced()
def phases_for(phases, host_el, dopants, lowest=False, window=0.0):
    keep = [(n, a, o) for n, a, o in phases
            if len(set(a.get_chemical_symbols()) - host_el) <= 1
            and set(a.get_chemical_symbols()) - host_el <= set(dopants)]
    if not lowest:
        return keep
    groups = {}
    for x in keep:
        if x[2].get("eah") is not None:
            groups.setdefault(reduced(x[1]), []).append(x)
    chosen = set()
    for group in groups.values():
        low = min(o["eah"] for _, _, o in group)
        near = [x for x in group if x[2]["eah"] <= low + window + 1e-9]
        chosen.add(min(near, key=lambda x: (len(x[1]), x[2]["eah"]))[0])
    return [x for x in keep if x[2].get("eah") is None or x[0] in chosen]


# The .param keywords of a bulk run at the screening settings (A3.1): screen_cutoff and
# %block screen_overrides, but not screen_max_steps (it is for the defects' relaxations). (With
# ref_relax : cell, a cell relaxed at a cut-off well short of converged can feel a Pulay stress big
# enough to run away, a soft metal's most of all; the default single points relax nothing.)
#   Called by: step_ref_phases()
#   Calls:     screen_overrides()
def screen_bulk(p):
    return {k: v for k, v in screen_overrides(p).items() if k != "geom_max_iter"}


# The jobs of the competing phases: solids with symmetry, at the batch's k-point spacing (or the
# phase's own kspacing, times kfactor: A3.1's ref_kspacing_factor), as relax says: cell (E1.1's,
# and ref_relax : cell) relaxes lattice and positions, positions keeps the cell, none is a single
# point of the structure as given (A3.1's default); molecules ('molecule') relax their positions in
# their fixed box, at Gamma, without symmetry. Each relaxation gets steps geometry steps
# (phases_max_steps: CASTEP's own 30 can be too few for a soft metal's cell). extra: .param
# keywords for every one (the screening settings); a phase's own (%block phases, doped's:
# fix_occupancy : false for a metal, the spin of O2, tidy_phase's spin_polarized) win.
#   Called by: step_phases(), step_ref_phases()
#   Calls:     kgrid()
def phase_jobs(phases, steps, extra=None, kfactor=1.0, relax="cell"):
    jobs = []
    for name, atoms, o in phases:
        mol = bool(o.get("molecule"))
        sp = relax == "none" and not mol
        job = {"name": name, "atoms": atoms, "kind": "sp" if sp else "geom",
               "fix_cell": mol or relax != "cell", "symmetry": not mol,
               "overrides": dict(HOST_OVERRIDES, **({} if sp else {"geom_max_iter": steps}),
                                 **(extra or {}), **o["overrides"])}
        if mol:
            job["kgrid"] = [1, 1, 1]
        elif o.get("kspacing"):
            job["kgrid"] = kgrid(atoms.cell, o["kspacing"] * kfactor)
        jobs.append(job)
    return jobs


# The relaxed competing phases from a finished batch: {name: {energy, counts, natoms, molecule}}
# and their table rows; stops if one did not finish.
#   Called by: step_phases(), step_ref_phases()
#   Calls:     counts()
def phase_results(res, phases):
    out, rows = {}, []
    for name, atoms, o in phases:
        r = res["jobs"][name]
        if r["status"] != "done" or r["energy"] is None:
            die(f"competing phase {name} did not finish: {r['reason'] or 'see ' + r['dir']}")
        out[name] = {"energy": r["energy"], "counts": counts(atoms), "natoms": len(atoms),
                     "molecule": bool(o.get("molecule"))}
        rows.append([name, atoms.get_chemical_formula(), len(atoms), r["energy"],
                     r["energy"] / len(atoms)])
    return out, rows


# A3.1: the reference energies C3 uses, at the screening settings (screen_bulk), so they match
# C2's screening runs: the host itself, a single point of A2.1's relaxed cell (the cell the defect
# supercells are built from), and the defects' constituents: the pure elements of the host and of
# each dopant, each only once (phases_for, lowest: the smallest cell within ref_polymorph_window
# of the lowest). C3 compares formation energies across a dopant's sites over the host's region
# against its elements (the dopant's chemical potential cancels: of these only the host's
# elements decide anything) and gives each defect's least formation energy with the dopants at
# their elements (a bound from below, shown in its table only). They are needed to tens of
# meV/atom, not to A1's 1 meV/atom, so by default (ref_relax : none) each solid is a single point
# of its structure as given: doped's are the Materials Project's, already relaxed with PBE, as
# symmetrised primitive cells (tidy_phase). Relaxing the cell at the low screening cut-off can run
# away besides (the Pulay stress: Na's grew by two thirds as its energy rose). ref_relax :
# positions relaxes the positions in that cell, cell lattice and positions (molecules always
# relax in their box). The k-point spacing is A1's (or a phase's own kspacing) times
# ref_kspacing_factor (2: an eighth of the k-points), the host's too; molecules stay at Gamma. The
# compounds (the host's, e.g. CaO2 for CaO, and the dopants') are run only in E1.1, at the
# production settings. From %block phases, or doped (phases_from : doped); with neither, nothing
# runs and C3 compares within composition groups only. The production energies E needs are
# E1.1's. The table gives each phase's grid, and which polymorphs were taken over a lower one.
#   Called by: main() (step A3.1)
#   Calls:     Project.batch(), Project.host(), Project.kspacing(), Project.production_param(),
#              counts(), format_table(), phase_jobs(), phase_list(), phase_results(), phases_for(),
#              reduced(), screen_bulk()
def step_ref_phases(p):
    if not (p.s.phases or p.s.phases_from == "doped"):
        return {"host": None, "phases": {}, "_note": "no competing phases (%block phases or "
                "phases_from), so C3 compares only within composition groups"}
    host = p.host()
    own = set(host.get_chemical_symbols())
    every = phase_list(p)
    window = p.s.ref_polymorph_window
    phases = [x for x in phases_for(every, own, [], lowest=True, window=window)
              if len(set(x[1].get_chemical_symbols())) == 1]
    phases += [x for x in phases_for(every, own, p.s.dopants, lowest=True, window=window)
               if len(set(x[1].get_chemical_symbols())) == 1
               and not set(x[1].get_chemical_symbols()) <= own]
    if any(n == "host" for n, _, _ in phases):
        die("a competing phase can't be called 'host' (A3.1 runs the host under that name)")
    taken = []                     # polymorphs taken over a lower one, for the table
    for n, a, o in phases:
        alike = [x for x in every if x[2].get("eah") is not None and reduced(x[1]) == reduced(a)]
        if o.get("eah") is not None and alike:
            low = min(alike, key=lambda x: x[2]["eah"])
            if low[0] != n:
                taken.append(f"{n} ({len(a)} atoms) rather than {low[0]} ({len(low[1])} atoms, "
                             f"{1000 * (o['eah'] - low[2]['eah']):.0f} meV/atom lower)")
    extra, factor = screen_bulk(p), p.s.ref_kspacing_factor
    spacing = p.kspacing() * factor
    jobs = [{"name": "host", "atoms": host, "kind": "sp", "symmetry": True,
             "overrides": dict(HOST_OVERRIDES, **extra)}]
    jobs += phase_jobs(phases, p.s.phases_max_steps, extra, factor, p.s.ref_relax)
    res = p.batch("A_host/jobs/ref_phases", jobs, "ref_phases", p.production_param(),
                  spacing=spacing)
    if not res["complete"]:
        return None
    r = res["jobs"]["host"]
    if r["status"] != "done" or r["energy"] is None:
        die(f"A3.1's host single point did not finish: {r['reason'] or 'see ' + r['dir']}")

    def grid(name):                # the k-point grid a job ran with
        g = res["calcs"][name].info().get("kgrid")
        return "x".join(str(k) for k in g) if g else ""

    out, rows = phase_results(res, phases)
    rows = [row[:3] + [grid(n)] + row[3:] for row, (n, _, _) in zip(rows, phases)]
    rows.insert(0, ["host (A2.1's cell)", host.get_chemical_formula(), len(host), grid("host"),
                    r["energy"], r["energy"] / len(host)])
    cut = f"{p.s.screen_cutoff:g} eV" if p.s.screen_cutoff else "the production cut-off"
    how = {"none": "single points", "positions": "positions relaxed",
           "cell": "lattice and positions relaxed"}[p.s.ref_relax]
    table = format_table(["phase", "formula", "atoms", "k-points", "E/eV", "E/atom (eV)"], rows,
                         f"Reference energies at the screening settings ({cut}; solids: {how}; "
                         f"k-point spacing {spacing:.4f} 1/A, {factor:g} x A1's)")
    if taken:
        table += ("\nThe smallest cell within ref_polymorph_window "
                  f"({1000 * window:g} meV/atom) of an element's lowest polymorph: "
                  + "; ".join(taken) + "\n")
    return {"host": {"energy": r["energy"], "counts": counts(host), "natoms": len(host)},
            "phases": out, "kspacing": spacing, "relax": p.s.ref_relax,
            "_note": f"the host and {len(out)} pure elements at the screening settings ({how})",
            "_table": table}


# The chemical-potential limits from the energies of the host and its competing phases.
# The host's own elements: the corners of its stability region on a pymatgen phase diagram of the
# phases made of only those elements (named after the phases there, e.g. CaO-O2). Each dopant: at
# each corner, the highest value its phases allow, mu_X = min over phases P containing X of
# (E_P - sum of the other elements' n mu) / n_X, with the limiting phase. Stops if the host is not
# stable against its competing phases.
#   Called by: step_limits(), step_ref_limits()
def chemical_limits(host_counts, host_energy, phases, dopants):
    try:
        from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
        from pymatgen.core import Composition
    except ImportError as err:
        die(f"the chemical-potential limits need pymatgen (pip install pymatgen): {err}")
    own = set(host_counts)
    entries = [PDEntry(Composition(host_counts), host_energy,
                       name=Composition(host_counts).reduced_formula)]
    for name, ph in phases.items():
        if set(ph["counts"]) <= own:
            entries.append(PDEntry(Composition(ph["counts"]), ph["energy"], name=name))
    missing = [el for el in own if not any(set(e.composition.as_dict()) == {el}
                                           for e in entries)]
    if missing:
        die("the chemical-potential limits need a phase of each pure element of the host; "
            "missing: " + ", ".join(missing) + " (add it to %block phases)")
    pd = PhaseDiagram(entries)
    ehull = pd.get_e_above_hull(entries[0])
    if ehull > 1e-6:
        decomp = pd.get_decomp_and_e_above_hull(entries[0])[0]
        die(f"the host is {ehull * 1000:.1f} meV/atom above the hull of its competing phases "
            "(it would decompose into " + ", ".join(e.name for e in decomp) + "), so it has no "
            "stable chemical potentials. Check those phases and their settings.")
    limits = {}
    for facet, mus in pd.get_all_chempots(Composition(host_counts)).items():
        mu = {str(el): float(v) for el, v in mus.items()}
        name = facet
        limiting = {}
        for x in dopants:
            best = None
            for pname, ph in phases.items():
                c = ph["counts"]
                if x not in c or not set(c) <= own | {x}:
                    continue
                val = (ph["energy"] - sum(n * mu[el] for el, n in c.items() if el != x)) / c[x]
                if best is None or val < best[0]:
                    best = (val, pname)
            if best is None:
                die(f"no competing phase contains the dopant {x} (add its element at least)")
            mu[x], limiting[x] = best
        limits[name] = {"mu": mu, "limiting": limiting}
    return limits


# The table of chemical-potential limits: each limit as Delta mu against the pure elements (where
# those phases were run; each limit gets them as delta_mu), and the phases that limit each dopant.
# Returns (the pure elements' energies per atom, the table).
#   Called by: step_limits(), step_ref_limits()
#   Calls:     format_table()
def limits_table(limits, phases, title):
    ref = {}
    for ph in phases.values():
        if len(ph["counts"]) == 1:
            el, n = next(iter(ph["counts"].items()))
            ref[el] = min(ref.get(el, ph["energy"] / n), ph["energy"] / n)
    rows = []
    for name, lim in limits.items():
        lim["delta_mu"] = {el: (v - ref[el]) if el in ref else None for el, v in lim["mu"].items()}
        rows.append([name] + [lim["delta_mu"].get(el) for el in sorted(lim["mu"])]
                    + [", ".join(f"{k}: {v}" for k, v in lim["limiting"].items())])
    els = sorted(next(iter(limits.values()))["mu"])
    return ref, format_table(["limit"] + [f"dmu_{el}" for el in els] + ["dopants limited by"],
                             rows, title)


# A3.2: the chemical-potential limits at the screening settings, from A3.1's energies
# (chemical_limits): the corners of the host's stability region against its elements alone (the
# widest it can be: a compound of the host's elements, such as CaO2 next to CaO, would narrow it,
# and is only run in E1.1), and each dopant at its pure element, the highest its chemical
# potential can be. C3's cut across a dopant's sites with these limits can only keep too much,
# never too little. A dopant without its element among the phases gets none, and C3 can't bound
# its defects' formation energies from below (the least E_f in its table).
#   Called by: main() (step A3.2)
#   Calls:     chemical_limits(), limits_table()
def step_ref_limits(p):
    ref = p.out("ref_phases")
    if not ref.get("host"):
        return {"limits": {}, "dopants": [], "_note": "no reference phases, so no limits: C3 "
                "compares only within composition groups"}
    phases = ref["phases"]
    have = [x for x in p.s.dopants if any(set(ph["counts"]) == {x} for ph in phases.values())]
    limits = chemical_limits(ref["host"]["counts"], ref["host"]["energy"], phases, have)
    refs, table = limits_table(limits, phases, "Chemical-potential limits at the screening "
                               "settings, against the elements alone (eV per atom, against the "
                               "pure elements; each dopant at its element)")
    missing = [x for x in p.s.dopants if x not in have]
    return {"limits": limits, "dopants": have, "references": refs, "_table": table,
            "_note": f"{len(limits)} limits at the screening settings"
                     + (f"; no element of {', '.join(missing)}" if missing else "")}


# ==========================================================================================
# PHASE B: CANDIDATE DEFECTS AND THE SCREENING CELL (flowchart B1-B3)
# ==========================================================================================
# --------------------------------------------------------------------------
# B1: candidates
# --------------------------------------------------------------------------
# The single defects of the host primitive cell, one per symmetry-inequivalent site: vacancies
# (vacancies : true), substitutions by each dopant, antisites (antisites : true), and Voronoi
# interstitials of the dopants or of every element (interstitials : dopants / all; needs
# pymatgen-analysis-defects). Each is a component dict (new, host, prim_site or frac) with
# 'sites': how many such sites there are per primitive cell.
#   Called by: generate()
#   Calls:     equivalent_sites(), interstitial_sites()
def single_defects(p, prim):
    syms, cls = prim.get_chemical_symbols(), equivalent_sites(prim)
    out = []
    for i in sorted(set(cls)):
        host, n = syms[i], cls.count(i)
        news = ([None] if p.s.vacancies else []) + list(p.s.dopants)
        if p.s.antisites:
            news += [x for x in sorted(set(syms)) if x != host]
        for new in news:
            out.append({"new": new, "host": host, "prim_site": i, "sites": n})
    if p.s.interstitials != "none":
        out += interstitial_sites(p, prim)
    return out


# The Voronoi interstitial sites of the host primitive cell (pymatgen's
# VoronoiInterstitialGenerator), for the dopants or every element. Labelled a, b, c, ... in the
# order pymatgen gives them.
#   Called by: single_defects()
def interstitial_sites(p, prim):
    try:
        from pymatgen.analysis.defects.generators import VoronoiInterstitialGenerator
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError as err:
        die(f"interstitials : {p.s.interstitials} needs pymatgen-analysis-defects: {err}")
    import warnings
    warnings.filterwarnings("ignore")
    species = list(p.s.dopants) + (sorted(set(prim.get_chemical_symbols()))
                                   if p.s.interstitials == "all" else [])
    out, labels = [], {}
    for d in VoronoiInterstitialGenerator().generate(AseAtomsAdaptor.get_structure(prim),
                                                     insert_species=species):
        el = d.site.species_string
        labels[el] = labels.get(el, 0) + 1
        out.append({"new": el, "host": None, "frac": d.site.frac_coords.tolist(),
                    "site_label": "abcdefghijklmnopqrstuvwxyz"[(labels[el] - 1) % 26],
                    "sites": int(getattr(d, "multiplicity", 1))})
    return out


# The position (cartesian, in the primitive cell) of a single defect.
#   Called by: pair_arrangements()
def comp_position(prim, c):
    if c["host"] is None:
        return np.asarray(c["frac"]) @ np.asarray(prim.cell)
    return prim.positions[c["prim_site"]].copy()


# Every place a component of type b can sit in the crystal: the images, within reach, of the
# primitive-cell sites equivalent to b's (or of b's interstitial site and its symmetry copies).
#   Called by: pair_arrangements()
#   Calls:     equivalent_sites(), symmetry_ops()
def candidate_points(prim, b, reach):
    cell = np.asarray(prim.cell)
    if b["host"] is None:
        rots, trans = symmetry_ops(prim)
        f0 = np.asarray(b["frac"])
        fracs = []
        for R, t in zip(rots, trans):
            f = (R @ f0 + t) % 1.0
            if not any(np.linalg.norm(((f - g + 0.5) % 1.0 - 0.5) @ cell) < 1e-3 for g in fracs):
                fracs.append(f)
        base = [f @ cell for f in fracs]
    else:
        cls = equivalent_sites(prim)
        base = [prim.positions[j] for j in range(len(prim)) if cls[j] == cls[b["prim_site"]]]
    n = [int(math.ceil(reach / g)) + 1 for g in 1 / np.linalg.norm(np.linalg.inv(cell).T, axis=1)]
    shifts = np.array(list(itertools.product(*[range(-k, k + 1) for k in n]))) @ cell
    return np.array([r + s for r in base for s in shifts])


# The cartesian rotations of the symmetry operations that leave a point of the primitive cell where
# it is (up to a lattice vector): the site symmetry used to tell arrangements apart.
#   Called by: pair_arrangements()
#   Calls:     symmetry_ops()
def site_rotations(prim, point):
    cell = np.asarray(prim.cell)
    f0 = np.linalg.solve(cell.T, point)
    rots, trans = symmetry_ops(prim)
    out = []
    for R, t in zip(rots, trans):
        d = (R @ f0 + t - f0 + 0.5) % 1.0 - 0.5
        if np.linalg.norm(d @ cell) < 1e-3:
            out.append(cell.T @ R @ np.linalg.inv(cell.T))
    return out


# The symmetry-distinct arrangements of a pair: b at every allowed point within pair_max_distance
# of a, grouped by the site symmetry of a (two vectors are the same arrangement if a symmetry
# operation that keeps a in place turns one into the other; for two identical components, also if
# it turns one into the other reversed). Each arrangement's vector is one that starts from a's own
# site (a reversed vector may only fit the other end, e.g. in wurtzite). Filters: interstitials at
# least 0.5 A from anything and 1 A from a vacancy. Returns [(vector, separation, how many
# equivalent orientations)], nearest first.
#   Called by: generate(), user_defects()
#   Calls:     candidate_points(), comp_position(), site_rotations()
def pair_arrangements(prim, a, b, dmax):
    ra = comp_position(prim, a)
    pts = candidate_points(prim, b, dmax)
    vecs = pts - ra
    dist = np.linalg.norm(vecs, axis=1)
    lo = 1.0 if (a["host"] is None) != (b["host"] is None) and not (a["new"] and b["new"]) else 0.5
    keep = (dist > max(0.1, lo if (a["host"] is None or b["host"] is None) else 0.1)) & \
           (dist <= dmax + 1e-6)
    rots = site_rotations(prim, ra)
    same = (a["new"], a["host"], a.get("prim_site"), tuple(a.get("frac") or ())) == \
           (b["new"], b["host"], b.get("prim_site"), tuple(b.get("frac") or ()))
    # A vector as a sortable tuple.
    def tup(x):
        return tuple(np.round(x, 3) + 0.0)

    groups = {}
    for v in vecs[keep]:
        imgs = [R @ v for R in rots]
        key = min(tup(x) for x in imgs + ([-x for x in imgs] if same else []))
        groups.setdefault(key, []).append(min(imgs, key=tup))     # (not reversed: fits a's site)
    out = []
    for key, reps in groups.items():
        rep = np.array(min(tup(x) for x in reps))
        out.append((rep, float(np.linalg.norm(rep)), len(reps)))
    return sorted(out, key=lambda x: (round(x[1], 3), x[0].tolist()))


# Turns a component token (Sb_Ca, Vac_O, Vac_O:2, Sb_Ca@2) into a component dict; 'sites' is how
# many symmetry-equivalent sites of that kind the primitive cell has.
#   Called by: user_defects()
#   Calls:     equivalent_sites(), parse_defect()
def parse_token(tok, prim):
    comps = parse_defect([tok], prim)
    c = comps[0]
    cls = equivalent_sites(prim)
    i = c["prim_site"]
    return {"new": c["new"], "host": c["host"], "prim_site": i, "spin": c["spin"],
            "sites": cls.count(cls[i])}


# The candidates from %block defects (B1 'yes'): one line per defect, components joined by '+'
# (e.g. Sb_Ca + Vac_O). A pair with 'all' on its line gives every arrangement up to
# pair_max_distance; otherwise each further component goes on the nearest site of its kind (as in
# the runner's size series, --conv). q=... on a line (q=-1, or q=+1,0) gives the charge
# states of that line's candidates, used instead of screen_charges.
#   Called by: step_generate()
#   Calls:     candidate(), pair_arrangements(), parse_token(), defect_vectors(),
#              parse_defect()
def user_defects(p, prim):
    out = []
    for row in p.s.defects:
        every = any(w.lower() == "all" for w in row)
        qwords = [w for w in row if w.lower().startswith("q=")]
        words = [w for w in row if w.lower() != "all" and w not in qwords]
        try:
            charges = sorted({int(x) for w in qwords for x in w[2:].split(",") if x})
        except ValueError:
            die(f"%block defects: bad charge in '{' '.join(row)}' (write e.g. q=-1 or q=+1,0)")
        if qwords and not charges:
            die(f"%block defects: no charge after q= in '{' '.join(row)}'")
        toks = [t for t in " ".join(words).replace("+", " ").split() if t]
        comps = [parse_token(t, prim) for t in toks]
        found = []
        if len(comps) == 1:
            found.append(candidate(prim, comps, None, 1))
        elif every and len(comps) == 2:
            for v, d, m in pair_arrangements(prim, comps[0], comps[1], p.s.pair_max_distance):
                found.append(candidate(prim, [comps[0], dict(comps[1], vec=v.tolist())], d, m))
        elif every:
            die(f"%block defects: 'all' works for pairs only ({' + '.join(toks)})")
        else:
            vecs = defect_vectors(prim, parse_defect(toks, prim))
            full = [comps[0]] + [dict(c, vec=np.asarray(v).tolist())
                                 for c, v in zip(comps[1:], vecs)]
            sep = float(np.linalg.norm(vecs[0])) if vecs else None
            found.append(candidate(prim, full, sep, 1))
        for c in found:
            if charges:
                c["charges"] = charges
        out += found
    return out


# A candidate record: its components, name, composition change and group (composition_change),
# the separation of a pair, its orientational multiplicity and the number of anchor sites per
# primitive cell (both for concentrations in E).
#   Called by: generate(), user_defects()
#   Calls:     composition_change(), defect_name()
def candidate(prim, comps, sep, mult):
    delta, group = composition_change(comps)
    comps = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()}
             for c in comps]
    return {"name": defect_name(comps, prim), "comps": comps, "group": group,
            "delta": delta, "separation": sep, "multiplicity": mult,
            "sites": comps[0].get("sites", 1)}


# The generated candidates (B1 'no'): the single defects (single_defects), then the pairs
# (pairs : true) of a dopant defect with a native one, and with native_pairs / dopant_pairs also
# native + native and dopant + dopant pairs (dopant_pairs : same: only two of the same dopant,
# so each dopant meets the host on its own); never two interstitials. Each pair type gives its
# symmetry-distinct arrangements up to pair_max_distance (pair_arrangements).
#   Called by: step_generate()
#   Calls:     candidate(), pair_arrangements(), single_defects()
def generate(p, prim):
    singles = single_defects(p, prim)
    out = [candidate(prim, [s], None, 1) for s in singles]
    if not p.s.pairs:
        return out
    dop = set(p.s.dopants)

    # Whether a single defect involves a dopant.
    def is_dopant(s):
        return s["new"] in dop

    for i, a in enumerate(singles):
        for j, b in enumerate(singles):
            if j < i or (a["host"] is None and b["host"] is None):
                continue
            kinds = (is_dopant(a), is_dopant(b))
            both = p.s.dopant_pairs is True or (p.s.dopant_pairs == "same" and
                                                a["new"] == b["new"])
            ok = ((kinds in ((True, False), (False, True))) or
                  (kinds == (False, False) and p.s.native_pairs) or
                  (kinds == (True, True) and both))
            if not ok:
                continue
            first, second = (b, a) if (not kinds[0] and kinds[1]) else (a, b)
            for v, d, m in pair_arrangements(prim, first, second, p.s.pair_max_distance):
                out.append(candidate(prim, [first, dict(second, vec=v.tolist())], d, m))
    return out


# B1: lists the candidate defects (user_defects or generate), with ids d001, d002, ... sorted by
# composition group and separation, and a table of them.
#   Called by: main() (step B1)
#   Calls:     Project.host(), format_table(), generate(), user_defects()
def step_generate(p):
    prim = p.host()
    cands = user_defects(p, prim) if p.s.defects else generate(p, prim)
    if not cands:
        die("no candidate defects: name some dopants or list defects in %block defects")
    cands.sort(key=lambda c: (len(c["comps"]), c["group"], c["separation"] or 0, c["name"]))
    for k, c in enumerate(cands):
        c["id"] = f"d{k + 1:03d}"
    groups = sorted({c["group"] for c in cands})
    own = any(c.get("charges") for c in cands)       # q= in %block defects: show them
    rows = [[c["id"], c["name"], c["group"],
             f"{c['separation']:.3f}" if c["separation"] is not None else "",
             c["multiplicity"], c["sites"]]
            + ([" ".join(f"{q:+d}" for q in c["charges"]) if c.get("charges")
                else "screen_charges"] if own else []) for c in cands]
    table = format_table(["id", "defect", "composition", "separation/A", "orientations",
                          "sites/cell"] + (["charges"] if own else []), rows,
                         f"{len(cands)} candidates in {len(groups)} composition groups")
    return {"candidates": {c["id"]: c for c in cands}, "groups": groups,
            "_note": f"{len(cands)} candidates in {len(groups)} groups", "_table": table}


# --------------------------------------------------------------------------
# B2, B3: supercell sizes and the screening cell
# --------------------------------------------------------------------------
# The supercell matrix P (the supercell is P x the primitive cell) for a size: ['n', n] is n x
# transform (n x n x n primitive cells without it); ['length', L] the most compact supercell whose
# periodic images are all at least L apart: n x transform for the smallest n that gives it, or
# without transform doped's search for the most compact cell (get_ideal_supercell_matrix), or,
# without doped, n x n x n primitive cells. The matrix is Minkowski-reduced (the same supercell
# with its shortest vectors as lattice vectors) and right-handed.
#   Called by: size_ladder(), step_choose(), step_electronic(), step_lineshape()
#   Calls:     min_image()
def size_matrix(p, prim, size):
    kind, value = size
    cell = np.asarray(prim.cell, float)
    T = (np.reshape(p.s.transform, (3, 3)) if p.s.transform else np.eye(3)).astype(int)
    if kind == "n":
        P = value * T
    elif p.s.transform:
        n = 1
        while min_image(n * T @ cell) < value - 1e-6:
            n += 1
        P = n * T
    else:
        P = None
        try:
            import warnings
            warnings.filterwarnings("ignore")
            from doped.generation import get_ideal_supercell_matrix
            from pymatgen.io.ase import AseAtomsAdaptor
            P = get_ideal_supercell_matrix(AseAtomsAdaptor.get_structure(prim),
                                           min_image_distance=value, min_atoms=0,
                                           ideal_threshold=0)
        except ImportError:
            log("note: without doped (pip install doped) the supercells are n x n x n primitive "
                "cells; doped finds more compact ones, or give a transform")
        if P is None:
            n = 1
            while min_image(n * cell) < value - 1e-6:
                n += 1
            P = n * np.eye(3, dtype=int)
        P = np.rint(P).astype(int)
    _, op = Cell(P @ cell).minkowski_reduce()
    P = np.rint(op @ P).astype(int)
    return (-P if np.linalg.det(P) < 0 else P).tolist()


# A size of the ladder: its supercell matrix, the atoms of the pristine supercell, its shortest
# image distance (L, Ang) and the length it was made for (None for a size given as n).
#   Called by: size_ladder(), step_choose(), step_electronic()
#   Calls:     min_image()
def cell_record(prim, P, length=None):
    P = np.asarray(P, int)
    return {"matrix": P.tolist(), "natoms": int(round(abs(np.linalg.det(P)))) * len(prim),
            "L": round(min_image(P @ np.asarray(prim.cell, float)), 4), "length": length}


# Puts a size into a ladder (a list of cell_record dicts, smallest image distance first) unless the
# same supercell is there already (the same lattice: one matrix is an integer recombination of the
# other), giving it a tag for job names, n<atoms> (with b, c, ... for another cell with as many
# atoms). Returns the ladder's record of that supercell.
#   Called by: size_ladder(), step_choose(), step_electronic()
def add_cell(ladder, rec):
    P = np.array(rec["matrix"], float)
    for c in ladder:
        M = np.array(c["matrix"], float) @ np.linalg.inv(P)
        if np.allclose(M, np.rint(M), atol=1e-6) and abs(abs(np.linalg.det(M)) - 1) < 1e-6:
            return c
    tags = {c["tag"] for c in ladder}
    tag = next(f"n{rec['natoms']}{s}" for s in ("",) + tuple("bcdefghijk")
               if f"n{rec['natoms']}{s}" not in tags)
    rec = dict(rec, tag=tag)
    ladder.append(rec)
    ladder.sort(key=lambda c: (c["L"], c["natoms"]))
    return rec


# The size ladder: the supercell for each of scan_lengths (size_matrix), each supercell once,
# smallest first. B2 scans it, B3 picks the screening cell and B4 the electronic cell from it. doped
# takes seconds to find each cell, so the ladder is kept in B_candidates/ladder.json for the same
# host cell, lengths and transform (and doped there or not).
#   Called by: step_scan()
#   Calls:     add_cell(), cell_record(), size_matrix(), write_json()
def size_ladder(p, prim):
    import importlib.util
    key = hashlib.md5(json.dumps([np.asarray(prim.cell).round(6).tolist(),
                                  prim.get_chemical_symbols(), sorted(p.s.scan_lengths),
                                  p.s.transform, importlib.util.find_spec("doped") is not None]
                                 ).encode()).hexdigest()
    f = p.phase_dir("B") / "ladder.json"
    if f.exists():
        saved = json.loads(f.read_text())
        if saved.get("key") == key:
            return saved["ladder"]
    ladder = []
    for L in sorted(set(p.s.scan_lengths)):
        log(f"B2: the most compact supercell with its images at least {L:g} A apart")
        add_cell(ladder, cell_record(prim, size_matrix(p, prim, ["length", L]), L))
    write_json(f, {"key": key, "ladder": ladder})
    return ladder


# The sizes of a ladder as a table: tag, shortest image distance, atoms, the length asked for and
# the supercell matrix.
#   Called by: step_choose(), step_scan()
#   Calls:     format_table()
def ladder_table(ladder, title, mark=None):
    return format_table(["size", "L/A", "atoms", "for", "matrix"],
                        [[c["tag"] + (" <-" if c["tag"] == mark else ""), c["L"], c["natoms"],
                          f"{c['length']:g} A" if c.get("length") else "",
                          " / ".join(" ".join(map(str, r)) for r in c["matrix"])]
                         for c in ladder], title)


# The line R = R_inf - c/N through some of a candidate's B2 relaxation energies (least squares;
# N, the atoms of each size's host supercell). They take that form once the cell holds the
# defect's relaxation and what still changes with size is only the energy of the stress the fixed
# cell holds, which falls off as 1/volume (as N). Returns (R_inf, c, the largest distance in eV
# of any of the points from the line).
#   Called by: step_scan()
def inverse_n_fit(points):
    x = np.array([1.0 / q["N"] for q in points])
    y = np.array([q["E_relax"] for q in points])
    A = np.vstack([np.ones_like(x), -x]).T
    (r_inf, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(r_inf), float(c), float(np.abs(A @ [r_inf, c] - y).max())


# B2: the size ladder (size_ladder: the supercell for each of scan_lengths) and, with mace : true
# and scan : true, the MACE scan of every candidate. Each candidate is built in every size it fits
# (fits) and relaxed with MACE (positions only, the host's lattice vectors kept, as in C and D;
# mace_model; mace_relaxations, on compute nodes with mace_jobscript), from the ideal structure
# rattled near the defect (mace_rattle, rattle_spec), so that no relaxation is held at a
# symmetric start by its own symmetry; R is still measured from the ideal structure. Its relaxation energy R
# (unrelaxed - relaxed) shows whether the cell holds the relaxation. With scan_test : line (the
# default) a candidate is converged at the smallest size from which R lies on a line
# R_inf - c/N: that size and every bigger one (at least SCAN_CONFIRM of them) within scan_tol of
# the line fitted through them (inverse_n_fit). A defect that strains the crystal leaves the
# fixed cell holding a stress, whose energy, c/N, falls off only as 1/N; once R follows the line,
# that is all that still changes with size. The table gives R_inf, c, and c/N in the converged
# size: the stress energy still in that cell, which cancels between arrangements of one
# composition but not between charge states. With scan_test : flat it is converged at the
# smallest size whose R is within scan_tol of that of every bigger size, with the atoms at the
# cell's edge (further than EDGE_FRACTION of the shortest image distance from the defect's centre)
# moved less than scan_edge_tol; the edge displacement is shown with either test. A candidate
# with fewer sizes than the test needs (a pair too far apart for the smaller ones) can't
# converge: a warning before the relaxations says how far apart a pair this ladder can test may be.
# One that fails the test although it has the sizes has usually reached different minima in
# different sizes (R jumps between neighbouring sizes) rather than not converged with size: it is
# relaxed again in its other sizes from the deepest minimum it reached (the size whose R is
# furthest above the trend with size, its structure carried into the others, not rattled again:
# carry_over), each
# size keeps the deeper of its two relaxations (mace_seed_cache), and the test is made again. One
# still not converged is dropped in B3 (scan_unconverged : drop, the default), or counted as needing
# the largest size (largest). Relaxations that stopped at mace_steps are named in the log.
#   Called by: main() (step B2)
#   Calls:     Project.host(), build_defect(), carry_over(), displacement_field(), fits(),
#              format_table(), inverse_n_fit(), ladder_table(), mace_cache(), mace_relaxations(),
#              mace_seed_cache(), rattle_spec(), atoms_from_dict(), atoms_to_dict(),
#              displacements(), get_distances(), size_ladder()
def step_scan(p):
    prim = p.host()
    ladder = size_ladder(p, prim)
    sizes = ladder_table(ladder, "The size ladder (scan_lengths): the most compact supercell for "
                                 "each length")
    if not (p.s.mace and p.s.scan):
        return {"ladder": ladder, "candidates": {}, "_table": sizes,
                "_note": f"{len(ladder)} sizes; no MACE scan (mace or scan is false)"}
    cands = p.out("generate")["candidates"]
    line, drop = p.s.scan_test == "line", p.s.scan_unconverged == "drop"
    need = SCAN_CONFIRM + 1 if line else 2         # the sizes a candidate needs for the test
    by_tag = {cell["tag"]: cell for cell in ladder}
    tasks, plan = [], {}
    for cid, c in cands.items():
        for cell in ladder:
            if not fits(c, cell["L"]):
                continue
            atoms, info = build_defect(prim, np.array(cell["matrix"]), c["comps"],
                                       f"{cid} ({c['name']})")
            cache = mace_cache(p, cid, cell["tag"])
            tasks.append({"atoms": atoms, "cache": cache, "rattle": rattle_spec(p, info),
                          "label": f"{cid} {c['name']} in {cell['tag']}"})
            plan[(cid, cell["tag"])] = (atoms, info, cache)
    few = sorted(cid for cid in cands if sum((cid, t) in plan for t in by_tag) < need)
    if few:
        seps = [cands[cid]["separation"] for cid in few if cands[cid]["separation"]]
        reach = ladder[-need]["L"] / 2 - 0.05 if len(ladder) >= need else None
        nth = {2: "2nd", 3: "3rd"}.get(need, f"{need}th")
        log(f"WARNING: B2: {len(few)} candidate{'s fit' if len(few) > 1 else ' fits'} fewer than "
            f"{need} sizes of the ladder, so the size test can't pass for "
            + ("them" if len(few) > 1 else "it")
            + (f" (pairs {min(seps):.2f}-{max(seps):.2f} A apart)" if seps else "") + "; B3 "
            + ("drops them (scan_unconverged : drop)" if drop else
               "counts them as needing the largest size (scan_unconverged : largest)") + ". "
            + (f"This ladder can test pairs up to {reach:.2f} A apart (half its {nth}-largest "
               "size's image distance, less 0.05 A): a pair_max_distance no bigger, or "
               if reach is not None else "")
            + f"scan_lengths with {need - 1} more sizes above where the farthest pair first fits, "
              "would test every one")
    res = mace_relaxations(p, "scan", tasks)
    if res is None:
        return None

    # A relaxation's numbers in its size: R, the energy before (e0: the ideal structure's, for a
    # re-seeded one too) less after, the energy after, and the largest move of the atoms at the
    # cell's edge; seeded: the size it was re-seeded from.
    def measure(cid, cell, relaxed, e0, e1, steps, seeded=None):
        atoms, info, _ = plan[(cid, cell["tag"])]
        moved = displacements(atoms, relaxed)
        _, r = get_distances(np.array(info["centre"])[None], atoms.positions,
                             cell=atoms.cell, pbc=True)
        edge = moved[r[0] > EDGE_FRACTION * cell["L"]]
        x = {"tag": cell["tag"], "L": cell["L"], "N": cell["natoms"],
             "E_relax": round(e0 - e1, 5), "E": round(e1, 5),
             "edge": round(float(edge.max()) if edge.size else 0.0, 4), "steps": steps}
        return dict(x, seeded=seeded) if seeded else x

    # The test, on a candidate's sizes (smallest first): (the size it has converged at, or None;
    # the line through it and the bigger sizes, with scan_test : line).
    def test(seen):
        if len(seen) < need:
            return None, None
        if line:                            # the first size on a line with the bigger ones
            for i in range(len(seen) - SCAN_CONFIRM):
                r_inf, slope, worst = inverse_n_fit(seen[i:])
                if worst < p.s.scan_tol:
                    return seen[i]["tag"], {"R_inf": round(r_inf, 4), "c": round(slope, 3),
                                            "worst": round(worst, 4), "sizes": len(seen) - i}
            return None, None
        return next((x["tag"] for i, x in enumerate(seen[:-1])
                     if x["edge"] < p.s.scan_edge_tol and
                     all(abs(y["E_relax"] - x["E_relax"]) < p.s.scan_tol
                         for y in seen[i + 1:])), None), None

    seen = {cid: [measure(cid, cell, *res[str(plan[(cid, cell["tag"])][2])])
                  for cell in ladder if (cid, cell["tag"]) in plan] for cid in cands}
    first = {cid: test(seen[cid]) for cid in cands}

    # The second pass. A candidate the test fails although it has the sizes has usually reached
    # different minima in different sizes (R jumps between neighbouring sizes) rather than not
    # converged with size. It is relaxed again in each of its other sizes from the deepest minimum
    # it reached: the relaxed structure of the size whose R is furthest above the trend with size
    # (the line through all its sizes), carried into the others (carry_over). Each size keeps
    # whichever of its two relaxations went deeper, and the test is made again.
    seeds, seed_plan, seed_tasks = {}, {}, []
    for cid in cands:
        if first[cid][0] or len(seen[cid]) < need:
            continue
        c, xs = cands[cid], seen[cid]
        r_inf, slope, _ = inverse_n_fit(xs)
        src = max(xs, key=lambda x: x["E_relax"] - (r_inf - slope / x["N"]))["tag"]
        src_P = np.array(by_tag[src]["matrix"])
        # (as its cache file holds it, so that the starts, and their keys, are the same on a rerun)
        relaxed = atoms_from_dict(atoms_to_dict(res[str(plan[(cid, src)][2])][0]))
        label = f"{cid} ({c['name']})"
        try:
            field = displacement_field(prim, c["comps"], src_P, relaxed, label)
            starts = {x["tag"]: carry_over(prim, c["comps"], src_P, relaxed,
                                           np.array(by_tag[x["tag"]]["matrix"]), label=label,
                                           field=field)[0]
                      for x in xs if x["tag"] != src}
        except Exception as err:            # (a structure the matching can't follow)
            log(f"note: B2: {cid} {c['name']} can't be re-seeded from {src}: {err}")
            continue
        seeds[cid] = {"from": src, "matrix": by_tag[src]["matrix"]}
        for tag, atoms in starts.items():
            seed_plan[(cid, tag)] = mace_seed_cache(p, cid, tag)
            seed_tasks.append({"atoms": atoms, "cache": seed_plan[(cid, tag)],
                               "label": f"{cid} {c['name']} in {tag}, from {src}"})
    if seed_tasks:
        log(f"B2: {len(seeds)} candidate{'s' if len(seeds) > 1 else ''} not converged: each "
            "relaxed again in its other sizes from the deepest minimum it reached "
            f"({len(seed_tasks)} MACE relaxations)")
        res2 = mace_relaxations(p, "scan", seed_tasks)
        if res2 is None:
            return None
        for (cid, tag), cache in seed_plan.items():
            relaxed, _, e1, steps = res2[str(cache)]
            k = next(i for i, x in enumerate(seen[cid]) if x["tag"] == tag)
            x = seen[cid][k]
            if e1 < x["E"] - SEED_DEEPER:   # deeper: this size's relaxation from now on
                seen[cid][k] = measure(cid, by_tag[tag], relaxed, x["E_relax"] + x["E"], e1,
                                       steps, seeded=seeds[cid]["from"])

    out, rows, capped = {}, [], []
    for cid, c in cands.items():
        conv, fit = test(seen[cid]) if cid in seeds else first[cid]
        out[cid] = {"sizes": seen[cid], "converged": conv}
        if fit:
            out[cid]["fit"] = fit
        if cid in seeds:
            out[cid]["seed"] = dict(seeds[cid], deeper=[x["tag"] for x in seen[cid]
                                                        if x.get("seeded")])
        if len(seen[cid]) < need:
            out[cid]["few"] = True
        capped += [f"{cid} in {x['tag']}" for x in seen[cid] if x["steps"] >= p.s.mace_steps]
        by = {x["tag"]: x for x in seen[cid]}
        here = by.get(conv)
        row = ([cid, c["name"]]
               + [f"{by[t]['E_relax']:.3f}" + ("^" if by[t].get("seeded") else "")
                  + ("*" if t == conv else "") if t in by else "" for t in by_tag]
               + [f"{here['L']:.2f}" if here else
                  "too few sizes" if len(seen[cid]) < need else "not converged"])
        if line:
            row += ([f"{fit['R_inf']:.3f}", f"{fit['c']:.1f}", f"{fit['c'] / here['N']:.3f}"]
                    if fit else ["", "", ""])
        rows.append(row + [here["edge"] if here else None,
                           seeds[cid]["from"] if cid in seeds else ""])
    seeded = (f"; ^ where the relaxation from the deepest minimum (re-seeded from that size, for "
              "a candidate the test failed) went deeper than the one from the ideal structure")
    if line:
        heads = ["converged at L/A", "R_inf/eV", "c/eV", "c/N there/eV", "edge move/A"]
        title = ("MACE relaxation energies R (eV) of each candidate in each size, the cell fixed; "
                 f"* where it has converged: from that size up (it and at least {SCAN_CONFIRM} "
                 f"bigger sizes) R is within scan_tol {p.s.scan_tol:g} eV of a line R_inf - c/N, "
                 "N the cell's atoms. c/N there: the energy of the stress the fixed cell still "
                 "holds in the converged size; the edge move is shown, not tested" + seeded)
    else:
        heads = ["converged at L/A", "edge move/A"]
        title = ("MACE relaxation energies (eV) of each candidate in each size; * where it is "
                 f"converged (within scan_tol {p.s.scan_tol:g} eV of every bigger size, edge "
                 f"atoms moved less than scan_edge_tol {p.s.scan_edge_tol:g} A)" + seeded)
    table = format_table(["id", "defect"] + list(by_tag) + heads + ["re-seeded from"], rows,
                         title)
    if capped:
        log(f"note: B2: {len(capped)} MACE relaxations stopped at mace_steps "
            f"({p.s.mace_steps}) before their forces were below mace_fmax: "
            + ", ".join(capped[:8]) + (f" and {len(capped) - 8} more" if len(capped) > 8 else "")
            + " (raise mace_steps)")
    bad = sorted(cid for cid, x in out.items() if not x["converged"])
    nfew = sum(1 for cid in bad if out[cid].get("few"))
    fixed = sum(1 for cid in seeds if out[cid]["converged"])
    if seeds:
        log(f"B2: re-seeding from the deepest minimum converged {fixed} of the {len(seeds)} "
            "candidates it was tried on")
    if bad:
        log(f"note: B2: {len(bad)} candidates did not converge within the ladder (up to "
            f"{ladder[-1]['L']:.1f} A)" + (f", {nfew} of them with too few sizes for the test"
                                           if nfew else "")
            + ": " + ", ".join(bad[:20]) + (f" and {len(bad) - 20} more" if len(bad) > 20 else "")
            + "; B3 " + ("drops them (scan_unconverged : drop)" if drop else
                         "counts them as needing the largest size (scan_unconverged : largest)"))
    return {"ladder": ladder, "candidates": out, "test": p.s.scan_test,
            "_table": sizes + "\n" + table,
            "_note": f"{len(ladder)} sizes, {len(out) - len(bad)} of {len(out)} candidates "
                     "converged" + (f" ({fixed} after re-seeding)" if fixed else "")
                     + (f"; {len(bad) - nfew} not converged" if len(bad) > nfew else "")
                     + (f", {nfew} with too few sizes to test" if nfew else "")}


# The band edges of a .bands of the host primitive cell (read_bands; A2.2's) and their effective
# masses. For the VBM and the CBM: the k-points where it is (the star of the k-point CASTEP gives,
# fractional) and the lightest mass (in m_e) of the bands that meet there, from the nearest
# k-points around it, E = E_edge -+ hbar^2 dk^2 / 2m*: each nearest k-point gives a mass, and the
# lightest (the direction and band whose defect levels reach furthest) is taken. The .bands'
# k-points are unfolded with the host's symmetry operations and time reversal, so the irreducible
# wedge CASTEP writes is enough. Returns {'vbm': {'k', 'mass', 'energy'}, 'cbm': {...}}; a mass is
# None when the k-points don't show the band curving away from the edge (a grid too coarse, or an
# edge between grid points).
#   Called by: step_choose()
#   Calls:     occupied_bands(), symmetry_ops()
def band_masses(b, prim):
    occ = occupied_bands(b)
    eig, kf = b["eig"], np.asarray(b["kpts"], float)
    rots, _ = symmetry_ops(prim)
    recip = 2 * np.pi * np.linalg.inv(np.asarray(prim.cell, float)).T
    imgs = np.concatenate([s * kf @ R for R in rots for s in (1, -1)])
    src = np.tile(np.arange(len(kf)), 2 * len(rots))
    shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
    out = {}
    for edge, sign in (("vbm", -1), ("cbm", 1)):
        top = [(sign * eig[s, k, occ[s] - (sign < 0)], s, k) for s in range(b["nspin"])
               for k in range(len(kf)) if np.isfinite(eig[s, k, occ[s] - (sign < 0)])]
        _, s0, k0 = min(top)
        e0 = float(eig[s0, k0, occ[s0] - (sign < 0)])
        dk = imgs - kf[k0]
        dk -= np.round(dk)
        dist = np.linalg.norm((dk[:, None, :] + shifts[None]) @ recip, axis=2).min(axis=1)
        star = [kf[k0]]
        for x in (imgs[src == k0] % 1.0):
            if not any(np.allclose((x - y + 0.5) % 1.0 - 0.5, 0, atol=1e-5) for y in star):
                star.append(x)
        near = dist > 1e-6
        shell = near & (dist <= 1.05 * dist[near].min()) if near.any() else near
        masses = []
        for s in range(b["nspin"]):
            band_range = range(occ[s]) if sign < 0 else range(occ[s], eig.shape[2])
            for n in band_range:
                if not (np.isfinite(eig[s, k0, n]) and abs(eig[s, k0, n] - e0) < 0.05):
                    continue
                for j in np.flatnonzero(shell):
                    de = sign * (eig[s, src[j], n] - e0)
                    if de > 1e-4:
                        masses.append(HB2M * dist[j] ** 2 / de)
        out[edge] = {"k": [np.round(x, 6).tolist() for x in star], "energy": e0,
                     "mass": round(min(masses), 4) if masses else None}
    return out


# Whether a k-point of the primitive cell (fractional) folds onto Gamma of the supercell P x
# (primitive cell): its coordinates in the supercell's reciprocal lattice, k P^T, are whole
# numbers.
#   Called by: edge_folding(), step_choose()
def folds_to_gamma(k, P):
    f = np.asarray(k, float) @ np.asarray(P, float).T
    return bool(np.allclose(f, np.rint(f), atol=1e-4))


# The Gamma-only floor (B3, B4): the shortest image distance a supercell needs for Gamma-point
# runs to see a defect's levels as an isolated defect has them. A level E_b inside the gap
# decays away from the defect as exp(-r/xi), with 1/xi^2 = (2 m*/hbar^2) E_b (1 - E_b/E_g) in a
# two-band (Kane) model; neighbouring images overlap, and the level spreads into a band, unless
# they are several xi apart. The floor is decay_length_multiple xi (default 4) of the shallowest
# level that counts as deep (E_b = deep_level_margin), for the lighter of the hole and electron
# masses (band_masses) and A2.2's gap. At Gamma the overlap makes a level that deep look deeper
# than it is: by about 10% at 4 xi, 19% at 3.5 and 33% at 3 (an s-like level with no core, 12
# nearest images; more for a level with a broad core, less for a compact one), and deeper levels
# by much less. screen_min_image instead, if given (0: no floor). Returns {'L', 'source', ...}
# or None (no floor, or the masses couldn't be found: a warning says to set screen_min_image).
# depth: the level's depth E_b instead (B4's electronic_depth; screen_min_image is then ignored).
#   Called by: step_choose(), step_electronic()
def gamma_floor(p, masses, depth=None):
    if depth is None and p.s.screen_min_image is not None:
        return ({"L": p.s.screen_min_image, "source": "screen_min_image"}
                if p.s.screen_min_image > 0 else None)
    eb, eg = (p.s.deep_level_margin if depth is None else depth), p.out("bands")["gap"]
    if eb >= eg:
        log(f"note: no Gamma-only floor: a level {eb:g} eV deep is not inside the gap")
        return None
    best = None
    for edge, what in (("vbm", "holes"), ("cbm", "electrons")):
        m = masses[edge]["mass"]
        if m:
            xi = 1 / math.sqrt(m * eb * (1 - eb / eg) / HB2M)
            if best is None or xi > best["xi"]:
                best = {"L": round(p.s.decay_length_multiple * xi, 2), "xi": round(xi, 3),
                        "mass": m, "edge": edge, "carrier": what, "E_b": eb,
                        "gap": round(eg, 4), "source": "band masses"}
    if best is None:
        log("WARNING: the effective masses could not be found from A2.2's band edges (its "
            "k-point grid may be too coarse), so there is no Gamma-only floor: set "
            "screen_min_image (Ang; B3) or electronic_cell (B4) to give the size")
    return best


# configuration_margin's cut, made in B3 from B2's MACE energies so that the screening cell is
# sized for the candidates that survive it (a dropped arrangement that strains the crystal could
# otherwise set the size). Within each composition group, the arrangements are compared in the
# biggest size of the ladder that B2 relaxed all of them in (its energies the nearest to an
# isolated defect's); one more than configuration_margin above the group's lowest there is cut.
# Returns ({id: {'dE': above the lowest (eV), 'size': compared in, 'why': for C1's table}},
# {group: the size compared in}), or (None, {}) when there is nothing to cut with (no
# configuration_margin, mace : false, or no scan: then C1 makes the cut itself, in the screening
# cell). A group B2 has no common size for is left as it is.
#   Called by: step_choose()
def configuration_cut(p, cands, conv, ladder):
    if p.s.configuration_margin is None or not p.s.mace or not conv:
        return None, {}
    order = {c["tag"]: k for k, c in enumerate(ladder)}
    groups = {}
    for cid, r in conv.items():
        groups.setdefault(cands[cid]["group"], []).append(cid)
    cut, compared = {}, {}
    for g, ids in sorted(groups.items()):
        E = {cid: {s["tag"]: s["E"] for s in conv[cid]["sizes"] if s.get("E") is not None}
             for cid in ids}
        common = set.intersection(*(set(e) for e in E.values()))
        if not common:
            log(f"note: B3: the {g} arrangements share no size of B2's scan, so "
                "configuration_margin leaves them all in")
            continue
        tag = max(common, key=lambda t: order[t])
        compared[g] = tag
        low = min(E[cid][tag] for cid in ids)
        for cid in ids:
            dE = E[cid][tag] - low
            if dE > p.s.configuration_margin:
                cut[cid] = {"dE": round(dE, 4), "size": tag,
                            "why": f"MACE energy {dE:.2f} eV above the lowest {g} arrangement "
                                   f"in {tag} (B3, configuration_margin)"}
    return cut, compared


# B3: the screening cell, which every candidate in C1-C3 is built in.
# First configuration_margin's cut (configuration_cut, on B2's MACE energies): the cell is sized
# for the candidates that survive it, and C1 leaves the others out. Then supercell if given (n x
# transform, or the most compact cell for a length); otherwise the smallest size of the ladder at
# least as big as every surviving candidate's converged size (B2) and as the Gamma-only floor
# (gamma_floor), with what set it, and, when the cut made it smaller, the cell it would have been.
# A survivor B2's size scan didn't converge (re-seeded or not, or with too few sizes for the test)
# is dropped, with scan_unconverged : drop (the default): C1 leaves it out, as it does a cut one,
# and it doesn't size the cell; with largest it counts as needing the largest size. (With
# supercell, the scan is only reported and nothing is dropped.) A floor beyond the ladder gets a
# cell of its own, added to the ladder. Also says whether A2.2's band edges fold onto Gamma of
# this cell (folds_to_gamma): if not, the Gamma-point screening sees a wider gap (C2 measures
# it); and how many surviving pairs are too far apart for it (fits), which C1 leaves out.
#   Called by: main() (step B3)
#   Calls:     Project.host(), add_cell(), band_masses(), cell_record(), configuration_cut(),
#              fits(), folds_to_gamma(), format_table(), gamma_floor(), ladder_table(),
#              read_bands(), size_matrix()
def step_choose(p):
    prim = p.host()
    scan = p.out("scan")
    cands = p.out("generate")["candidates"]
    ladder = [dict(c) for c in scan["ladder"]]
    masses = band_masses(read_bands(p.dir / p.out("bands")["bands_file"]), prim)
    floor = gamma_floor(p, masses)
    conv, set_by = scan.get("candidates") or {}, None
    cut, compared = configuration_cut(p, cands, conv, ladder)
    keep = {cid: r for cid, r in conv.items() if cid not in (cut or {})}
    drop, dropped = p.s.scan_unconverged == "drop" and not p.s.supercell, {}
    if drop:                                 # scan_unconverged : drop
        dropped = {cid: "too few sizes of the ladder for B2's size test" if r.get("few") else
                        "not converged in B2's size scan"
                   for cid, r in keep.items() if not r["converged"]}
        keep = {cid: r for cid, r in keep.items() if cid not in dropped}
        if conv and not keep:
            die("B3: no candidate is left once configuration_margin's cut and the candidates "
                "B2's size scan didn't converge (scan_unconverged : drop) are out: see B2's table "
                "in B_candidates/summary.txt, or set scan_unconverged : largest")
    without = None                           # (the cell without the cut, what would have set it)
    if p.s.supercell:
        kind, value = p.s.supercell
        cell = add_cell(ladder, cell_record(prim, size_matrix(p, prim, p.s.supercell),
                                            value if kind == "length" else None))
        why = f"supercell : {p.raw['supercell']}"
        if floor and cell["L"] < floor["L"] - 1e-6:
            log(f"WARNING: B3: the supercell's images are {cell['L']:.2f} A apart, less than the "
                f"Gamma-only floor ({floor['L']:.2f} A): the Gamma-point screening may see its "
                "defect levels spread into bands")
    else:
        if not conv and not floor:
            die("B3 has nothing to choose the screening cell from: without the MACE scan (mace "
                "or scan false) give supercell (n, or a length such as 15 ang) or "
                "screen_min_image")
        L = {c["tag"]: c["L"] for c in ladder}

        # The size these candidates need (the largest of their converged sizes, or the floor's),
        # and what sets it.
        def needed(rs):
            need, by = 0.0, None
            for cid in sorted(rs):
                x = L[rs[cid]["converged"]] if rs[cid]["converged"] else ladder[-1]["L"]
                if x > need + 1e-6:
                    need, by = x, cid
            if floor and floor["L"] > need + 1e-6:
                need, by = floor["L"], "floor"
            return need, by

        need, set_by = needed(keep)
        if cut:
            need_all, by_all = needed({cid: r for cid, r in conv.items()
                                       if r["converged"] or not drop})
            if need_all > need + 1e-6:
                big = next((c for c in ladder if c["L"] >= need_all - 1e-6), None)
                without = (big["tag"] if big else f"{need_all:.2f} A", by_all)
        cell = next((c for c in ladder if c["L"] >= need - 1e-6), None)
        if cell is None:
            log(f"note: B3: the Gamma-only floor ({need:.2f} A) is beyond the ladder (up to "
                f"{ladder[-1]['L']:.2f} A): the screening cell is made for it")
            cell = add_cell(ladder, cell_record(prim, size_matrix(p, prim, ["length", need]),
                                                need))
        if set_by == "floor":
            why = f"the Gamma-only floor, {floor['L']:.2f} A"
        else:
            r = conv[set_by]
            why = (f"{set_by} {p.out('generate')['candidates'][set_by]['name']}, "
                   + (f"converged at {L[r['converged']]:.2f} A" if r["converged"] else
                      "not converged in the ladder"))
    fold = {e: any(folds_to_gamma(k, cell["matrix"]) for k in masses[e]["k"])
            for e in ("vbm", "cbm")}
    if not all(fold.values()):
        log("WARNING: B3: " + " and ".join(e.upper() for e, ok in fold.items() if not ok)
            + " of the host (A2.2) don't fold onto Gamma of the screening cell, so the Gamma-point "
            "screening sees a wider gap and may misjudge levels near the edges (C2 measures it)")
    rows = [[e.upper(), " ".join(f"{x:.3f}" for x in masses[e]["k"][0]), masses[e]["energy"],
             masses[e]["mass"], "yes" if fold[e] else "NO"] for e in ("vbm", "cbm")]
    table = (ladder_table(ladder, f"The screening cell: {cell['tag']} ({why})", cell["tag"])
             + "\n" + format_table(["edge", "k (fractional)", "E/eV", "m*/m_e", "folds to Gamma"],
                                   rows, "Band edges (A2.2) and their effective masses")
             + (f"\nGamma-only floor: {floor['L']:.2f} A ("
                + (f"decay_length_multiple {p.s.decay_length_multiple:g} x the decay length "
                   f"{floor['xi']:.2f} A of a level {floor['E_b']:g} eV from the band edge of the "
                   f"{floor['carrier']}, m* {floor['mass']:.3g} m_e, gap {floor['gap']:.2f} eV)"
                   if floor["source"] == "band masses" else "screen_min_image)") + "\n"
                if floor else "\nGamma-only floor: none\n"))
    if cut is not None:
        table += (f"\nconfiguration_margin ({p.s.configuration_margin:g} eV): {len(cut)} of "
                  f"{len(conv)} arrangements cut, each compared with the others of its "
                  "composition in the biggest size B2 relaxed them all in; C1 leaves them out, "
                  "and the screening cell is sized for the rest"
                  + (f". Without the cut it would have been {without[0]}, for {without[1]} "
                     f"{cands[without[1]]['name']}" if without else "") + "\n")
        if cut:
            table += format_table(
                ["id", "defect", "composition", "compared in", "above the lowest/eV"],
                [[cid, cands[cid]["name"], cands[cid]["group"], c["size"], c["dE"]]
                 for cid, c in sorted(cut.items())], "Arrangements cut (configuration_margin)")
    if dropped:
        nfew = sum(why.startswith("too few") for why in dropped.values())
        table += (f"\nscan_unconverged : drop: {len(dropped)} of the {len(keep) + len(dropped)} "
                  + ("candidates the cut kept" if cut else "candidates")
                  + " didn't converge in B2's size scan"
                  + (f" ({nfew} of them with too few sizes for the test)" if nfew else "")
                  + "; C1 leaves them out, and the screening cell is sized for the rest\n")
        table += format_table(["id", "defect", "composition", "why"],
                              [[cid, cands[cid]["name"], cands[cid]["group"], why]
                               for cid, why in sorted(dropped.items())],
                              "Candidates dropped (scan_unconverged : drop)")
    far = sorted(cid for cid in cands if cid not in (cut or {}) and cid not in dropped
                 and not fits(cands[cid], cell["L"]))
    if far:
        table += (f"\n{len(far)} pair{'s have their' if len(far) > 1 else ' has its'} components "
                  f"more than half this cell's image distance ({cell['L']:.2f} A) apart, so C1 "
                  "leaves them out: "
                  + ", ".join(far[:20]) + (f" and {len(far) - 20} more" if len(far) > 20 else "")
                  + "\n")
    idx = ladder.index(cell)
    return {"matrix": cell["matrix"], "natoms": cell["natoms"], "L_min": cell["L"],
            "tag": cell["tag"], "ladder": ladder, "index": idx, "floor": floor,
            "set_by": set_by, "masses": masses, "fold": fold, "cut": cut,
            "cut_sizes": compared, "dropped": dropped, "far_pairs": far, "_table": table,
            "_note": f"{cell['tag']}: {cell['natoms']} atoms, images {cell['L']:.2f} A apart "
                     f"({why})" + (f"; {len(cut)} arrangement{'s' if len(cut) > 1 else ''} cut "
                                   "by configuration_margin" if cut else "")
                     + (f"; {len(dropped)} dropped, not converged in B2 (scan_unconverged)"
                        if dropped else "")}


# How many of each band edge's symmetry-equivalent k-points (its star, band_masses) fold onto
# Gamma of the supercell P x (primitive cell): {'vbm': [folded, of], 'cbm': [folded, of]}.
#   Called by: step_electronic()
#   Calls:     folds_to_gamma()
def edge_folding(masses, P):
    return {e: [sum(folds_to_gamma(k, P) for k in masses[e]["k"]), len(masses[e]["k"])]
            for e in ("vbm", "cbm")}


# B4: the electronic cell, the supercell the survivors move into after C3 and every judgement of a
# defect's electronic structure is made in: C4-C6's distortions (distort_cell), C7's excited
# states, D's production runs, and so E and F (C1-C3, which screen every candidate, stay in the
# screening cell, where they cost least). The screening cell (B3) is sized for the relaxation
# (B2) and for levels at least deep_level_margin inside the gap; at the semilocal level many levels
# sit closer to the band edges than that, and a Gamma-point run only sees the host states whose
# k-points fold onto its Gamma. So the electronic cell is the smallest size of the ladder, from the
# screening cell up, that (1) folds EVERY symmetry-equivalent k-point of both band edges onto Gamma
# (edge_folding: each valley of an edge away from Gamma, e.g. the three X points of a rock-salt
# conduction band, so a shallow state can mix all of them), and (2) meets the Gamma-only floor for
# a level electronic_depth deep (gamma_floor: decay_length_multiple times its decay length, as in
# B3). If no size of the ladder does, the most compact cells for longer lengths (ELECTRONIC_STEP
# at a time) are tried. electronic_cell : screening uses the screening cell; electronic_cell : n
# or a length (e.g. 20 ang) gives the cell (with a warning if it doesn't fold every k-point). The
# table gives each cell's folding ("1 of 3" means only one valley of that edge is seen at Gamma)
# and the floor.
#   Called by: main() (step B4)
#   Calls:     Project.host(), add_cell(), cell_record(), edge_folding(), format_table(),
#              gamma_floor(), size_matrix()
def step_electronic(p):
    prim, choose = p.host(), p.out("choose")
    ladder, idx, masses = [dict(c) for c in choose["ladder"]], choose["index"], choose["masses"]

    # Whether a supercell folds every k-point of both band edges onto Gamma.
    def folds_all(P):
        return all(n == m for n, m in edge_folding(masses, P).values())

    floor = gamma_floor(p, masses, depth=p.s.electronic_depth)
    setting = p.s.electronic_cell
    if setting and setting[0] == "screening":
        cell, why = ladder[idx], "electronic_cell : screening"
    elif setting:
        cell = add_cell(ladder, cell_record(prim, size_matrix(p, prim, setting),
                                            setting[1] if setting[0] == "length" else None))
        why = f"electronic_cell : {p.raw['electronic_cell']}"
    else:
        need = max(ladder[idx]["L"], floor["L"] if floor else 0.0)
        cell = next((c for c in ladder[idx:] if c["L"] >= need - 1e-6
                     and folds_all(c["matrix"])), None)
        if cell is None:
            L = max(need, ladder[-1]["L"])
            for _ in range(ELECTRONIC_TRIES):
                L += ELECTRONIC_STEP
                rec = cell_record(prim, size_matrix(p, prim, ["length", L]), L)
                if rec["L"] >= need - 1e-6 and folds_all(rec["matrix"]):
                    cell = add_cell(ladder, rec)
                    break
            if cell is None:
                die(f"B4: no supercell up to {L:.1f} A folds every band-edge k-point onto Gamma; "
                    "give electronic_cell (a size, or 'screening')")
        why = ("the smallest that folds every band-edge k-point onto Gamma"
               + (f" and is at least the Gamma-only floor {floor['L']:.2f} A (a level "
                  f"{floor['E_b']:g} eV deep)" if floor else ""))
    fold = edge_folding(masses, cell["matrix"])
    fold_s = edge_folding(masses, ladder[idx]["matrix"])
    if not folds_all(cell["matrix"]):
        log("WARNING: B4: the electronic cell doesn't fold every band-edge k-point onto Gamma ("
            + ", ".join(f"{e.upper()} {n} of {m}" for e, (n, m) in fold.items()) + "), so "
            "Gamma-point runs miss some valleys of the band edges")
    if floor and cell["L"] < floor["L"] - 1e-6:
        log(f"WARNING: B4: the electronic cell's images are {cell['L']:.2f} A apart, less than "
            f"the Gamma-only floor for a level {floor['E_b']:g} eV deep ({floor['L']:.2f} A)")
    rows = [[name, c["tag"], c["natoms"], c["L"]]
            + [f"{n} of {m}" for n, m in edge_folding(masses, c["matrix"]).values()]
            for name, c in (("screening (B3)", ladder[idx]), ("electronic (B4)", cell))]
    table = format_table(["cell", "size", "atoms", "L/A", "VBM points at Gamma",
                          "CBM points at Gamma"], rows,
                         f"The electronic cell: {cell['tag']} ({why}); each band edge's "
                         "symmetry-equivalent k-points that fold onto Gamma")
    if floor:
        table += (f"\nGamma-only floor: {floor['L']:.2f} A (decay_length_multiple "
                  f"{p.s.decay_length_multiple:g} x the decay length {floor['xi']:.2f} A of a "
                  f"level {floor['E_b']:g} eV from the band edge of the {floor['carrier']}, m* "
                  f"{floor['mass']:.3g} m_e, gap {floor['gap']:.2f} eV)\n")
    return {"matrix": cell["matrix"], "natoms": cell["natoms"], "L": cell["L"],
            "tag": cell["tag"], "fold": fold, "fold_screening": fold_s, "floor": floor,
            "why": why, "same": cell["matrix"] == ladder[idx]["matrix"], "_table": table,
            "_note": f"{cell['tag']}: {cell['natoms']} atoms, images {cell['L']:.2f} A apart "
                     f"({why})"}


# ==========================================================================================
# PHASE C: SCREENING, GROUND AND EXCITED STATES (flowchart C1-C8)
# ==========================================================================================
# The candidates (B1) and the screening cell (B3): (candidates by id, primitive host, matrix).
#   Called by: step_distort(), step_filter(), step_prerelax(), step_screen()
#   Calls:     Project.host()
def screen_setup(p):
    return p.out("generate")["candidates"], p.host(), np.array(p.out("choose")["matrix"])


# The .param keywords of the screening runs: screen_cutoff, screen_max_steps and
# %block screen_overrides.
#   Called by: excite_job(), run_screen_jobs(), screen_bulk(), screen_job(), step_excite(),
#              step_screen()
def screen_overrides(p):
    o = {"geom_max_iter": p.s.screen_max_steps}
    if p.s.screen_cutoff:
        o["cut_off_energy"] = f"{p.s.screen_cutoff:g} eV"
    o.update(p.s.screen_overrides)
    return o


# A screening or distortion job for one structure in charge state q, starting with spin S
# (spin_polarized, spin : S, and SPIN=S on the anchor atom when there is one): a Gamma-point
# relaxation without symmetry, with the screening keywords.
#   Called by: excite_job(), run_screen_jobs(), step_excite(), step_filter(), step_screen()
#   Calls:     screen_overrides()
def screen_job(p, name, atoms, info, q, S):
    a = atoms.copy()
    mags = np.zeros(len(a))
    if S and info.get("anchor_index") is not None:
        mags[info["anchor_index"]] = S
    a.set_initial_magnetic_moments(mags)
    o = dict(screen_overrides(p), charge=q, spin=S, spin_polarized="true")
    return {"name": name, "atoms": a, "kind": "geom", "kgrid": [1, 1, 1], "symmetry": False,
            "overrides": o}


# What a finished screening-type job gave: energy (None unless the last SCF converged), whether
# its geometry converged, final spin, defect levels in the gap and the electrons or holes it left in
# the bands (both from its .bands against the host's band edges at the same settings), and its
# final structure, saved as <structdir>/<name>.cell.
#   Called by: run_screen_jobs(), step_excite(), step_screen()
#   Calls:     band_carriers(), gap_levels(), read_bands(), save_cell(), final_atoms()
def job_summary(p, res, name, edges, structdir):
    r, calc = res["jobs"][name], res["calcs"][name]
    out = {"status": r["status"], "reason": r["reason"], "geom_converged": r["geom_converged"],
           "energy": r["energy"] if r["scf_converged"] else None, "spin": r["spin"],
           "steps": r["geom_steps"], "levels": None, "band": None, "structure": None}
    bands = calc.dir / f"{calc.seed}.bands"
    if bands.exists() and edges:
        b = read_bands(bands)
        out["levels"] = gap_levels(b, edges[0], edges[1], p.s.level_margin)
        out["band"] = band_carriers(b, edges[0], edges[1], p.s.level_margin)
    if r["energy"] is not None:
        atoms = final_atoms(calc)
        if atoms is None:
            log(f"WARNING: could not read the final structure of {name} ({calc.dir})")
        else:
            out["structure"] = save_cell(p, structdir / f"{name}.cell", atoms,
                                         f"{name}: final structure of its screening-type run")
    return out


# A table of the runs without a usable energy (name, status, why) for summary.txt; None if there
# are none.
#   Called by: step_drelax(), step_screen()
#   Calls:     format_table()
def problem_table(runs):
    rows = [[name, r["status"], (r["reason"] or "no converged energy").splitlines()[0]]
            for name, r in runs.items() if r["energy"] is None]
    return format_table(["run", "status", "why"], rows, "Runs without a usable energy "
                        "(--resubmit-failed starts failed jobs again)") if rows else None


# --------------------------------------------------------------------------
# C1, C2
# --------------------------------------------------------------------------
# C1: builds every candidate in the screening cell (B3; build_defect) and, with mace : true,
# relaxes it with MACE (positions only, the cell fixed, mace_model, from the structure rattled
# near the defect: mace_rattle, rattle_spec): B2's relaxation in that cell
# when it ran there (the same cache file), else a new one (mace_relaxations, on compute nodes with
# mace_jobscript). A candidate B2 re-seeded (its size test failed at first) is also relaxed from
# the deepest minimum B2 found for it, carried into this cell (carry_over; B2's own relaxation in
# mace_seed_cache when this cell is a size of the ladder), and keeps the deeper of the two. A pair
# too far apart for this cell (fits) is left out, and so are an arrangement configuration_margin
# cut in B3 (configuration_cut, on B2's MACE energies) and a candidate B3 dropped as not converged
# in B2 (scan_unconverged : drop). Only when B3 had
# nothing to cut with (no scan) does C1 make that cut itself: arrangements whose configurational
# energy (MACE energy above the lowest arrangement of the same composition) here is more than
# configuration_margin are dropped.
# Each structure is saved as C_screen/structures/candidates/<id>.cell, and the results give where
# its defect is (its centre, each component's position, the anchor atom's index, the vacancies)
# and its MACE energy.
#   Called by: main() (step C1)
#   Calls:     build_defect(), carry_over(), fits(), format_table(), mace_cache(), mace_cached(),
#              mace_key(), mace_relaxations(), mace_seed_cache(), rattle_spec(), save_cell(),
#              screen_setup()
def step_prerelax(p):
    cands, prim, P = screen_setup(p)
    d = p.phase_dir("C")
    choose = p.out("choose")
    L, tag, cut = choose["L_min"], choose["tag"], choose.get("cut")
    dropped = choose.get("dropped") or {}
    built, skipped, energies, defects = {}, {}, {}, {}
    for cid, c in cands.items():
        if cut and cid in cut:
            skipped[cid] = cut[cid]["why"]
            continue
        if cid in dropped:
            skipped[cid] = f"{dropped[cid]} (B3, scan_unconverged : drop)"
            continue
        if not fits(c, L):
            skipped[cid] = f"components {c['separation']:.2f} A apart, too far for this cell"
            continue
        built[cid] = build_defect(prim, P, c["comps"], f"{cid} ({c['name']})")
    seeded, deeper = {}, []
    if p.s.mace:
        tasks = [{"atoms": atoms, "cache": mace_cache(p, cid, tag), "rattle": rattle_spec(p, info),
                  "label": f"{cid} {cands[cid]['name']} in {tag}"}
                 for cid, (atoms, info) in built.items()]
        scan = p.out("scan").get("candidates") or {}
        for cid in built:                    # B2's deepest minimum, carried into this cell
            seed = scan.get(cid, {}).get("seed")
            if not seed or seed["from"] == tag:
                continue
            c, src_P = cands[cid], np.array(seed["matrix"])
            label = f"{cid} ({c['name']})"
            ideal, src_info = build_defect(prim, src_P, c["comps"], label)
            hit = mace_cached(mace_cache(p, cid, seed["from"]),
                              mace_key(p, ideal, rattle=rattle_spec(p, src_info)))
            if hit is None:                  # (B2's file is gone: the relaxation from ideal only)
                continue
            start, _ = carry_over(prim, c["comps"], src_P, hit[0], P, label=label)
            seeded[cid] = mace_seed_cache(p, cid, tag)
            tasks.append({"atoms": start, "cache": seeded[cid],
                          "label": f"{cid} {c['name']} in {tag}, from {seed['from']}"})
        res = mace_relaxations(p, "prerelax", tasks)
        if res is None:
            return None
    for cid, (atoms, info) in built.items():
        e = None
        if p.s.mace:
            atoms, _, e, _ = res[str(mace_cache(p, cid, tag))]
            if cid in seeded and res[str(seeded[cid])][2] < e - SEED_DEEPER:
                atoms, _, e, _ = res[str(seeded[cid])]
                deeper.append(cid)
        title = f"{cid} {cands[cid]['name']} in the screening cell" + (
            ", MACE pre-relaxed" + (" from B2's deepest minimum" if cid in deeper else "")
            if p.s.mace else "")
        defects[cid] = dict(structure=save_cell(p, d / "structures" / "candidates" / f"{cid}.cell",
                                                atoms, title), mace_energy=e, **info)
        if cid in deeper:
            defects[cid]["reseeded"] = scan[cid]["seed"]["from"]
        energies[cid] = e
    if p.s.configuration_margin is not None and p.s.mace and cut is None:     # (B3 couldn't)
        groups = {}
        for cid in defects:
            groups.setdefault(cands[cid]["group"], []).append(cid)
        for g, ids in groups.items():
            low = min(energies[i] for i in ids)
            for i in ids:
                if energies[i] - low > p.s.configuration_margin:
                    skipped[i] = (f"MACE energy {energies[i] - low:.2f} eV above the lowest "
                                  f"{g} arrangement (configuration_margin)")
                    defects.pop(i)
    for cid, why in skipped.items():
        log(f"C1: {cid} {cands[cid]['name']} left out: {why}")
    if seeded:
        log(f"C1: {len(deeper)} of the {len(seeded)} candidates B2 re-seeded went deeper from "
            "B2's deepest minimum than from the ideal structure")
    table = format_table(["id", "defect", "MACE energy/eV", "left out because"],
                         [[cid, cands[cid]["name"], energies.get(cid), skipped.get(cid, "")]
                          for cid in cands]) if skipped or p.s.mace else None
    return {"defects": defects, "skipped": skipped,
            "_note": f"{len(defects)} candidates ready" + (f", {len(skipped)} left out"
                                                           if skipped else ""),
            "_table": table}


# C2: the screening pass. One CASTEP relaxation per candidate, charge state (screen_charges) and
# starting spin (spin_states: by default one, a triplet, or a doublet for an odd electron count;
# the spin is not held fixed), plus the pristine supercell at the same settings for the band edges
# and host energy. All at the Gamma point with the screening keywords; failures are flagged, not
# rescued (runner group 'screen': max_restarts 0, tolerate_failures). The results give each run's
# energy, final spin, gap levels and the electrons or holes it left in the bands (band_carriers:
# C3 doesn't judge a candidate on such a band state), and the pristine cell's band edges. The
# pristine cell's gap at Gamma is checked against A2.2's: one more than FOLD_GAP_TOL wider means
# band edges that don't fold onto Gamma of this cell (B3 names them), which the Gamma-point levels
# are measured against.
#   Called by: main() (step C2)
#   Calls:     Project.batch(), Project.production_param(), band_edges(), host_supercell(),
#              job_summary(), load_cell(), problem_table(), qtag(), read_bands(), screen_job(),
#              screen_overrides(), screen_setup(), spin_states()
def step_screen(p):
    cands, prim, P = screen_setup(p)
    d = p.phase_dir("C")
    defects = p.out("prerelax")["defects"]
    host = host_supercell(prim, P)
    jobs = [{"name": "host", "atoms": host, "kind": "sp", "kgrid": [1, 1, 1], "symmetry": True,
             "overrides": {k: v for k, v in screen_overrides(p).items() if k != "geom_max_iter"}}]
    plan = {}
    for cid, info in defects.items():
        atoms = load_cell(p, info["structure"])
        for q in cands[cid].get("charges") or p.s.screen_charges:
            for S in spin_states(atoms, q, p.s.screen_spins):
                name = f"{cid}_{qtag(q)}_s{S}"
                jobs.append(screen_job(p, name, atoms, info, q, S))
                plan[name] = {"cand": cid, "q": q, "spin_start": S}
    res = p.batch("C_screen/jobs/screen", jobs, "screen", p.production_param())
    if not res["complete"]:
        return None
    hc = res["calcs"]["host"]
    if res["jobs"]["host"]["status"] != "done" or not (hc.dir / f"{hc.seed}.bands").exists():
        die(f"the pristine screening cell did not finish ({hc.dir}); the band edges and host "
            "energy come from it")
    vbm, cbm = band_edges(read_bands(hc.dir / f"{hc.seed}.bands"))
    wider = (cbm - vbm) - p.out("bands")["gap"]
    host_out = {"energy": res["jobs"]["host"]["energy"], "vbm": vbm, "cbm": cbm, "gap": cbm - vbm,
                "gap_A22": p.out("bands")["gap"], "folded": wider <= FOLD_GAP_TOL}
    if wider > FOLD_GAP_TOL:
        log(f"WARNING: C2: the screening cell's gap at Gamma ({cbm - vbm:.3f} eV) is {wider:.3f} "
            f"eV wider than the host's (A2.2, {p.out('bands')['gap']:.3f} eV): its band edges "
            "don't fold onto Gamma of this cell (see B3), so levels near the edges are measured "
            "against edges that are too far out. A cell that folds them (another supercell, or "
            "transform) avoids it; a bigger level_margin makes it matter less")
    runs = {}
    for name, meta in plan.items():
        runs[name] = dict(meta, **job_summary(p, res, name, (vbm, cbm),
                                              d / "structures" / "screen"))
    bad = sum(r["energy"] is None for r in runs.values())
    return {"host": host_out, "runs": runs, "_table": problem_table(runs),
            "_note": f"{len(runs)} runs, gap {cbm - vbm:.2f} eV at Gamma"
                     + (f" ({wider:+.2f} eV against A2.2: band edges not at Gamma)"
                        if wider > FOLD_GAP_TOL else "")
                     + (f", {bad} without a usable energy" if bad else "")}


# --------------------------------------------------------------------------
# C3
# --------------------------------------------------------------------------
# The best screening result of each candidate in each charge state: the lowest energy over its
# starting spins, among runs with a converged last SCF (an unconverged geometry is kept, but
# flagged). Returns {cand: {q: run}}.
#   Called by: step_filter()
def best_states(runs):
    best = {}
    for name, r in runs.items():
        if r["energy"] is None:
            continue
        cur = best.setdefault(r["cand"], {}).get(r["q"])
        if cur is None or r["energy"] < cur["energy"]:
            best[r["cand"]][r["q"]] = dict(r, name=name)
    return best


# The formation energy of a candidate's charge states against the Fermi level, without the host
# energy and chemical potentials (they cancel within one composition): E + E_corr + q (VBM + E_F)
# on the grid ef. Returns {q: array}.
#   Called by: step_filter()
#   Calls:     correction()
def relative_formation(states, ef, vbm, cell, eps, method):
    return {q: s["energy"] + correction(q, cell, eps, method) + q * (vbm + ef)
            for q, s in states.items()}


# How far a defect stays above the lowest of a set of defects, at best, over the host's stability
# region (each point a convex combination w of its corners, the limits) and the Fermi levels e
# from 0 to top: for each of its charge states, the least t with f_q(w, e) - f_k(w, e) <= t for
# every charge state k of every defect in the set (its own among them), by linear programming;
# the least over its charge states (0 when it is the lowest somewhere). Every formation energy is
# linear in w and e, so this is exact. own and every: [(a, q)], a the formation energy at E_F = 0
# at each corner (an array), q the charge. Returns eV, or None if no problem could be solved.
#   Called by: step_filter()
def site_distance(own, every, top):
    from scipy.optimize import linprog
    m, out = len(own[0][0]), None
    for a, q in own:
        rows = [list(a - b) + [q - k, -1.0] for b, k in every]
        r = linprog([0.0] * (m + 1) + [1.0], A_ub=rows, b_ub=[0.0] * len(rows),
                    A_eq=[[1.0] * m + [0.0, 0.0]], b_eq=[1.0],
                    bounds=[(0, None)] * m + [(0.0, top), (None, None)], method="highs")
        if r.status == 0:
            out = r.fun if out is None else min(out, r.fun)
    return None if out is None else max(out, 0.0)


# The arrangements that relaxed into the same structure as another of their composition group: in
# every charge state both have, their best screened structures match (same_structures). Each is
# mapped to the one whose charge-corrected formation energy (env: its lowest over its charge
# states, relative_formation's, at each Fermi level) is lowest somewhere in the gap, which stays;
# raw energies of different charge states can't be compared. Returns {duplicate: the one it
# matches}.
#   Called by: step_filter()
#   Calls:     load_cell(), same_structures()
def duplicates(p, best, groups, env):
    dup = {}
    for ids in groups.values():
        ids = sorted(ids, key=lambda c: float(np.min(env[c])))
        for i, a in enumerate(ids):
            if a in dup:
                continue
            for b in ids[i + 1:]:
                common = sorted(set(best[a]) & set(best[b]))
                if b in dup or not common or not all(best[a][q]["structure"] and
                                                     best[b][q]["structure"] for q in common):
                    continue
                if all(len(same_structures([load_cell(p, best[x][q]["structure"])
                                            for x in (a, b)])) == 1 for q in common):
                    dup[b] = a
    return dup


# The bands that hold a band state's carriers, at its first k-point (Gamma for the screening),
# against the host's band edges (vbm, cbm) with the margin band_carriers counts them with: in each
# spin channel, the occupied bands from margin below the CBM up (electrons in the conduction band)
# and the empty ones from margin above the VBM down (holes in the valence band). Each: {spin, band
# (from 0, in its spin channel), kind ('e' or 'h'), depth (eV inside its band: above the CBM for
# an electron, below the VBM for a hole; a little negative within the margin)}.
#   Called by: band_state_test()
#   Calls:     occupied_bands()
def carrier_bands(b, vbm, cbm, margin):
    occ, out = occupied_bands(b), []
    for s in range(b["nspin"]):
        for n, e in enumerate(b["eig"][s, 0]):
            if not np.isfinite(e):
                continue
            if n < occ[s] and e >= cbm - margin:
                out.append({"spin": s, "band": n, "kind": "e", "depth": float(e - cbm)})
            elif n >= occ[s] and e <= vbm + margin:
                out.append({"spin": s, "band": n, "kind": "h", "depth": float(vbm - e)})
    return out


# Whether a band state is worth carrying on (distort_charges : relevant), from C3's single point of
# its screened structure with the band weights written (r: the job's result, calc: its Calc; nat:
# the cell's atoms): every band that holds one of its carriers (carrier_bands) must lie at most
# carrier_reach inside its band, and be spread over at most carrier_fraction of the cell's atoms
# (participation): a shallow carrier still on the defect, which a distortion in the electronic cell
# or a wider gap than the semilocal one may bind, rather than one spread through the cell or deep
# in a band. A single point that gave no usable energy, bands or weights can't judge it, and it is
# carried on; one whose carriers all went into gap levels there is bound at that structure, and is
# carried on too. Returns {ok, depth (eV, the deepest carrier), P (atoms, the most spread), limit
# (carrier_fraction x nat), why}.
#   Called by: step_filter()
#   Calls:     carrier_bands(), participation(), read_bands(), read_pdos_weights()
def band_state_test(p, r, calc, nat, host):
    out = {"ok": True, "depth": None, "P": None, "limit": round(p.s.carrier_fraction * nat, 1)}
    bands = calc.dir / f"{calc.seed}.bands"
    pw = read_pdos_weights(calc.dir / f"{calc.seed}.pdos_weights")
    if r["status"] != "done" or r["energy"] is None or not r["scf_converged"]:
        out["why"] = ("carried on: not judged, its single point failed ("
                      + (r["reason"] or "no converged energy").splitlines()[0] + ")")
        return out
    if not bands.exists() or pw is None:
        out["why"] = "carried on: not judged, its single point wrote no .bands or .pdos_weights"
        return out
    b = read_bands(bands)
    carriers = carrier_bands(b, host["vbm"], host["cbm"], p.s.level_margin)
    if not carriers:
        out["why"] = "carried on: its carriers are in gap levels in the single point"
        return out
    P = participation(pw)
    if any(x["spin"] >= len(P) or x["band"] >= len(P[x["spin"]]) for x in carriers):
        out["why"] = "carried on: not judged, its .pdos_weights has no weights for its carriers"
        return out
    depth = max(x["depth"] for x in carriers)
    spread = max(float(P[x["spin"]][x["band"]]) for x in carriers)
    out.update(depth=round(depth, 3), P=round(spread, 1))
    if depth > p.s.carrier_reach:
        out.update(ok=False, why=f"left out: {depth:.2f} eV inside the band (carrier_reach "
                                 f"{p.s.carrier_reach:g} eV)")
    elif spread > p.s.carrier_fraction * nat:
        out.update(ok=False, why=f"left out: spread over {spread:.0f} atoms (carrier_fraction "
                                 f"{p.s.carrier_fraction:g} of {nat}: {out['limit']:g})")
    else:
        out["why"] = "carried on"
    return out


# C3: the defect hull, the sites, the least formation energy, the property filters and the charge
# states each survivor carries on.
# A charge state whose best screening run is a band state (in_band: it left an electron in the
# conduction band or a hole in the valence band instead of in a defect level) isn't judged by the
# hull: its energy is that of a carrier in a band, not of the defect holding it, and it may yet
# bind in another structure (the distortions) or with a wider gap than the semilocal one.
# Everything below uses the bound charge states. For each composition group and Fermi level across
# the gap, the lowest formation energy of any arrangement and bound charge state
# (relative_formation, with each charge state's correction in the screening cell, so charge states
# can be compared); an arrangement is kept if its own lowest is within hull_margin of that
# somewhere in the gap (one with no bound charge state at all can't be judged by it). Across the
# groups, with A3.2's limits (the screening settings, so they match the screening runs): the
# formation energies E_f = E + E_corr - E_host - sum n_i mu_i + q (VBM + E_F), against C2's
# pristine screening cell, over the host's stability region (the convex combinations of its
# corners; A3.2's is against the host's elements alone, the widest it can be, so the cut can only
# keep too much) and the gap. A defect with dopant atoms is dropped if it stays more than
# site_margin above the lowest defect with the same dopant atoms everywhere there (site_distance:
# those atoms' chemical potentials cancel, so only the host's matter; native defects are not
# compared so, as they don't compete for the same atoms), so every dopant keeps its lowest
# defect. Each defect's least formation energy anywhere there, with the dopants at their pure
# elements (the highest their chemical potentials can be, so a bound from below), is worked out
# for the table only: nothing is dropped on it. Without A3.2's limits neither is. Its stable
# charge states are its bound ones that are its lowest somewhere in the gap. The gap is gap if the
# project gives it (an experimental or hybrid value), else the screening cell's own: a
# semilocal gap is too narrow, and a bound charge state that only becomes the lowest above its
# conduction band edge (the formation energies carried on as the straight lines they are in the
# Fermi level) still counts then; a band state can't, since its line is the carrier's band edge
# carried on. The gap levels are still those inside the screening cell's band edges. Then, if
# asked: gap levels (require_gap_states: occupied, empty, or both = an occupied and an empty level
# in one spin channel), the final spin (require_spin) and binding against its isolated components
# (require_bound: the components must have been screened too) - each met by at least one stable
# charge state. Arrangements that relaxed into the same structure as a lower one of their group in
# every charge state both have (duplicates) are dropped too. With excite : true, native defects
# (is_native) skip the property filters (C7 passes them on to D whatever their excited states, for
# E), but not the hull or the duplicates.
# The charge states each survivor carries on (distort_charges): relevant (the default) - its stable
# ones, its other bound ones whose line comes within charge_margin of its lowest somewhere in the
# gap (a distortion may lower one that much, C5's gain, and make it stable), and the band states
# one charge beyond the stable one at either band edge (one more electron than the state stable at
# the top of the gap, one more hole than the one at the bottom; every band state of a candidate
# with none bound) whose carrier is shallow and on the defect: band_state_test, from a single point
# of the screened structure with the band weights written (C_screen/jobs/filter, runner group
# screen, %block runner_filter). Only such a band state keeps a candidate that fails the property
# filters (pending: bound, it might have the property) or has no bound charge state. stable: its
# stable ones and every band state; screened: every charge state with a usable energy. Under these
# two, any band state keeps a candidate pending, and one with no bound charge state is kept.
# The results list the survivors, with the charge states to carry on (and the top of the Fermi
# levels scanned, fermi_top); the table gives every candidate's verdict, then the
# judgement of each band state (relevant), then the correction for each |q| in the screening
# cell. None while the single points run.
#   Called by: main() (step C3)
#   Calls:     Project.batch(), Project.production_param(), band_state_test(), best_states(),
#              carriers_text(), correction(), duplicates(), format_table(), host_supercell(),
#              in_band(), internal_pair(), is_native(), load_cell(), qtag(), relative_formation(),
#              screen_job(), screen_setup(), site_distance(), start_spin()
def step_filter(p):
    cands, prim, P = screen_setup(p)
    scr = p.out("screen")
    host, runs = scr["host"], scr["runs"]
    best = best_states(runs)
    eps = p.out("dielectric").get("eps0")
    cell = host_supercell(prim, P).cell
    top = p.s.gap or host["gap"]                 # the Fermi levels scanned, from the VBM
    ef = np.linspace(0.0, top, 401)
    form = {cid: relative_formation(st, ef, host["vbm"], cell, eps, p.s.charge_correction)
            for cid, st in best.items()}
    bound = {cid: sorted(q for q, s in st.items() if not in_band(s)) for cid, st in best.items()}
    banded = {cid: sorted(q for q, s in st.items() if in_band(s)) for cid, st in best.items()}
    env = {cid: np.min(np.array([form[cid][q] for q in bound[cid]]), axis=0)
           for cid in best if bound[cid]}
    groups = {}
    for cid in best:
        groups.setdefault(cands[cid]["group"], []).append(cid)
    hull = {g: np.min(np.array([env[c] for c in ids if c in env]), axis=0)
            for g, ids in groups.items() if any(c in env for c in ids)}
    host_el = set(prim.get_chemical_symbols())
    corners = [lim["mu"] for lim in p.out("ref_limits")["limits"].values()]  # screening settings
    base = {cid: {q: float(form[cid][q][0]) - host["energy"] for q in bound[cid]} for cid in best}
    lowest_f = {}                  # the least formation energy anywhere (dopants at their elements)
    for cid in best:
        delta = cands[cid]["delta"]
        if bound[cid] and corners and all(el in corners[0] for el in delta):
            lowest_f[cid] = min(base[cid][q] + q * e - sum(n * mu[el] for el, n in delta.items())
                                for mu in corners for q in bound[cid] for e in (0.0, top))
    sets = {}                      # the defects with the same dopant atoms (not the natives)
    for cid in best:
        key = tuple(sorted((el, n) for el, n in cands[cid]["delta"].items() if el not in host_el))
        if key and bound[cid] and corners:
            sets.setdefault(key, []).append(cid)
    site = {}                      # how far each stays above its set's lowest, at best
    for key, ids in sets.items():
        lines = {cid: [(np.array([base[cid][q] - sum(n * mu[el] for el, n in
                                                     cands[cid]["delta"].items() if el in host_el)
                                  for mu in corners]), q) for q in bound[cid]] for cid in ids}
        every = [x for cid in ids for x in lines[cid]]
        for cid in ids:
            site[cid] = 0.0 if len(ids) == 1 else site_distance(lines[cid], every, top)
    singles = {(c["comps"][0]["new"], c["comps"][0]["host"], c["comps"][0].get("prim_site"),
                tuple(c["comps"][0].get("frac") or ())): cid
               for cid, c in cands.items() if len(c["comps"]) == 1}
    order = {cid: env[cid] if cid in env else np.min(np.array(list(form[cid].values())), axis=0)
             for cid in best}                    # (all charge states, for one with none bound)
    dup = duplicates(p, best, groups, order)
    left_out = p.out("prerelax")["skipped"]
    relevant = p.s.distort_charges == "relevant"

    # Whether a charge state has the gap levels asked for.
    def levels_ok(s):
        lv = s.get("levels") or []
        want = p.s.require_gap_states
        return (want == "none" or (want == "occupied" and any(x["occupied"] for x in lv))
                or (want == "empty" and any(not x["occupied"] for x in lv))
                or (want == "both" and internal_pair(lv)))

    # Whether a charge state has the spin asked for.
    def spin_ok(s):
        return (p.s.require_spin is None or
                (s["spin"] is not None and abs(abs(s["spin"]) - p.s.require_spin) < 0.25))

    looks = {}                 # each candidate's hull, filters and band states to judge
    for cid in sorted(cands):
        c = cands[cid]
        if cid not in best:
            continue
        dist, stable, near = None, [], {}
        if cid in env:
            dist = float(np.min(env[cid] - hull[c["group"]]))
            lowest = np.argmin(np.array([form[cid][q] for q in bound[cid]]), axis=0)
            stable = sorted({bound[cid][i] for i in np.unique(lowest)})
            for q in bound[cid]:                 # how close each other one comes to the lowest
                gap_q = float(np.min(form[cid][q] - env[cid]))
                if q not in stable and gap_q <= p.s.charge_margin:
                    near[q] = round(gap_q, 4)
        why = [f"relaxed into the structure of {dup[cid]}"] if cid in dup else []
        if dist is not None and dist > p.s.hull_margin:
            why.append(f"{dist:.2f} eV above the hull")
        if site.get(cid) is not None and site[cid] > p.s.site_margin:
            why.append(f"{site[cid]:.2f} eV above the lowest defect with its dopant atoms")
        good = [q for q in stable if levels_ok(best[cid][q]) and spin_ok(best[cid][q])]
        failed, fixed = [], []      # the property filters it fails; those a band state can't help
        asked = [x for x, on in (("gap levels", p.s.require_gap_states != "none"),
                                 ("spin", p.s.require_spin is not None)) if on]
        if asked and not good:
            failed.append("no stable charge state with the " + " and ".join(asked)
                          + " asked for")
        bind = None
        if len(c["comps"]) > 1:
            parts = [singles.get((x["new"], x["host"], x.get("prim_site"),
                                  tuple(x.get("frac") or ()))) for x in c["comps"]]
            if cid in env and all(pt in env for pt in parts):
                eb = sum(env[pt] for pt in parts) - env[cid] - (len(parts) - 1) * host["energy"]
                bind = (float(eb.min()), float(eb.max()))
        if p.s.require_bound and len(c["comps"]) > 1:
            if bind is None and not all(pt in best for pt in parts):
                fixed.append("binding energy unknown (its components were not screened)")
            elif bind is None:
                failed.append("binding energy unknown (no bound charge state of it or a "
                              "component)")
            elif bind[1] <= 0:
                failed.append(f"not bound (E_b {bind[0]:.2f} to {bind[1]:.2f} eV)")
        tests = []                  # its band states to judge (relevant): one beyond each edge's
        if relevant and not why:
            if bound[cid]:
                ends = np.array([form[cid][q] for q in bound[cid]])
                for q, kind in ((bound[cid][int(np.argmin(ends[:, -1]))] - 1, "electrons"),
                                (bound[cid][int(np.argmin(ends[:, 0]))] + 1, "holes")):
                    if q in banded[cid] and best[cid][q]["band"][kind] >= BAND_CARRIER:
                        tests.append(q)
            else:
                tests = list(banded[cid])
        looks[cid] = {"dist": dist, "stable": stable, "near": near, "why": why, "good": good,
                      "failed": failed, "fixed": fixed, "bind": bind, "tests": tests,
                      "waived": bool(failed or fixed) and p.s.excite and is_native(c, host_el)}

    judged, jobs, which = {}, [], {}             # the band states' single points (relevant)
    for cid, look in looks.items():
        for q in look["tests"]:
            src = best[cid][q]
            if not src.get("structure"):
                judged[(cid, q)] = {"ok": True, "depth": None, "P": None, "limit": None,
                                    "why": "carried on: not judged, no screened structure"}
                continue
            atoms = load_cell(p, src["structure"])
            name = f"{cid}_{qtag(q)}_bw"
            job = screen_job(p, name, atoms, p.out("prerelax")["defects"][cid], q,
                             start_spin(atoms, q, src["spin"]))
            job.update(kind="sp", overrides=dict(job["overrides"], pdos_calculate_weights="true"))
            jobs.append(job)
            which[name] = (cid, q, len(atoms))
    if jobs:
        res = p.batch("C_screen/jobs/filter", jobs, "filter", p.production_param())
        if not res["complete"]:
            return None
        for name, (cid, q, nat) in which.items():
            judged[(cid, q)] = band_state_test(p, res["jobs"][name], res["calcs"][name], nat,
                                               host)

    survivors, rows, brows = [], [], []
    for cid in sorted(cands):
        c = cands[cid]
        if cid not in best:
            why = (f"left out in C1: {left_out[cid]}" if cid in left_out
                   else "dropped: no usable screening run")
            rows.append([cid, c["name"], c["group"], "", "", "", "", "", "", "", "", why])
            continue
        look = looks[cid]
        dist, stable, near, failed, fixed = (look[k] for k in ("dist", "stable", "near",
                                                               "failed", "fixed"))
        why, waived, bind = list(look["why"]), look["waived"], look["bind"]
        passing = [q for q in look["tests"] if judged[(cid, q)]["ok"]]
        helps = passing if relevant else banded[cid]    # the band states that keep it pending
        pending = bool(failed) and not fixed and bool(helps) and not waived
        if not (waived or pending):
            why += failed + fixed
        if relevant and dist is None and not passing and not why:
            why.append("no bound charge state, and no band state with a shallow carrier on the "
                       "defect")
        if relevant:
            charges = sorted(set(stable) | set(near) | set(passing))
        elif p.s.distort_charges == "stable":
            charges = sorted(set(stable) | set(banded[cid]))
        else:
            charges = sorted(best[cid])
        spins = ", ".join(f"{q:+d}:{best[cid][q]['spin']:.1f}" if best[cid][q]["spin"] is not None
                          else f"{q:+d}:-" for q in stable)
        notes = []
        if waived:
            notes.append("native, so the property filters are waived for E: "
                         + "; ".join(failed + fixed))
        if pending:
            notes.append("pending: " + "; ".join(failed) + ", but "
                         + " ".join(f"{q:+d}" for q in helps)
                         + (" is" if len(helps) == 1 else " are") + " in a band")
        if near and relevant:
            notes.append("within charge_margin of stable: "
                         + ", ".join(f"{q:+d} ({g:.2f} eV)" for q, g in sorted(near.items())))
        if dist is None:
            notes.append("no bound charge state, so the hull can't judge it")
        rows.append([cid, c["name"], c["group"], "" if dist is None else dist,
                     "" if site.get(cid) is None else site[cid], lowest_f.get(cid, ""),
                     " ".join(f"{q:+d}" for q in stable), spins,
                     ", ".join(f"{q:+d}: {carriers_text(best[cid][q])}" for q in banded[cid]),
                     f"{bind[0]:.2f}..{bind[1]:.2f}" if bind else "",
                     " ".join(f"{q:+d}" for q in charges) if not why else "",
                     ("kept" + (" (" + "; ".join(notes) + ")" if notes else ""))
                     if not why else "dropped: " + "; ".join(why)])
        if relevant and not look["why"]:
            for q in banded[cid]:
                t = judged.get((cid, q))
                brows.append([cid, f"{q:+d}", carriers_text(best[cid][q]),
                              "" if not t or t["depth"] is None else t["depth"],
                              "" if not t or t["P"] is None else f"{t['P']:g} / {t['limit']:g}",
                              t["why"] if t else "left out: not one charge beyond the state "
                                                 "stable at its band edge"])
        if not why:
            survivors.append({"cand": cid, "name": c["name"], "group": c["group"],
                              "hull_distance": dist, "site_distance": site.get(cid),
                              "lowest_formation": lowest_f.get(cid),
                              "stable": stable, "charges": charges,
                              "near": {str(q): g for q, g in near.items()},
                              "in_band": banded[cid], "pending": failed if pending else [],
                              "band_tests": {str(q): judged[(cid, q)] for q in look["tests"]},
                              "good": look["good"], "binding": bind,
                              "best": {str(q): best[cid][q] for q in best[cid]}})
    carry = {"relevant": f"the stable ones, bound ones within charge_margin "
                         f"({p.s.charge_margin:g} eV) of their defect's lowest somewhere in the "
                         "gap, and band states with a shallow carrier on the defect (below)",
             "stable": "the stable ones and every one in a band",
             "screened": "every one with a usable energy"}[p.s.distort_charges]
    table = format_table(["id", "defect", "composition", "above hull/eV", "above site/eV",
                          "least E_f/eV", "stable q", "spin (q:S)", "in a band (q: carriers)",
                          "E_b/eV", "carried q", "verdict"], rows,
                         f"Defect hull (margin {p.s.hull_margin:g} eV), sites (above the lowest "
                         f"defect with the same dopant atoms anywhere in the host's stability "
                         f"region against its elements, margin {p.s.site_margin:g} eV), the "
                         f"least formation energy (the dopants at their elements; shown only)"
                         + ("" if corners else " (no A3.2 limits: sites not judged)")
                         + f" and property filters, "
                         f"Fermi levels 0 to {top:.2f} eV above the VBM"
                         + (f" (gap; the screening cell's own gap is {host['gap']:.2f} eV)"
                            if p.s.gap else "")
                         + "; charge states in a band (e: electrons in the conduction band, "
                         "h: holes in the valence band) are not judged by the hull. Charge "
                         f"states carried on (distort_charges : {p.s.distort_charges}): {carry}")
    if brows:
        table += "\n" + format_table(
            ["id", "q", "carriers", "depth/eV", "spread/atoms (limit)", "verdict"], brows,
            "The band states: carried on if one charge beyond the state stable at that band "
            "edge, with each carrier at most carrier_reach "
            f"({p.s.carrier_reach:g} eV) inside its band and spread over at most carrier_fraction "
            f"({p.s.carrier_fraction:g}) of the cell's atoms, from a single point of the screened "
            "structure with the band weights (C_screen/jobs/filter)")
    charges = sorted({abs(q) for st in best.values() for q in st if q})
    if charges:                    # what the hull's corrections add, in this cell
        table += (f"\nCharge corrections in the screening cell ({p.s.charge_correction}"
                  + (f", eps0 {eps:.3f}" if eps else "") + "): "
                  + ", ".join(f"|q| = {a}: {correction(a, cell, eps, p.s.charge_correction):.3f} eV"
                              for a in charges) + "\n")
    ncharges = sum(len(s["charges"]) for s in survivors)
    return {"survivors": survivors, "fermi_top": top, "_table": table,
            "_note": f"{len(survivors)} of {len(cands)} candidates kept, {ncharges} charge "
                     "states carried on"}



# --------------------------------------------------------------------------
# C4, C5: distortions searched with MACE and relaxed in CASTEP, the start of the ground-state
# search that C6 carries on
# --------------------------------------------------------------------------
# The oxidation states for counting the neighbours to distort: %block oxidation_states, else
# pymatgen's guess for the host and, for each dopant, its common oxidation state closest to that
# of the element it replaces (or its most positive one).
#   Called by: step_distort()
#   Calls:     element()
def oxidation_states(p, prim, cands):
    if p.s.oxidation_states:
        return {element(r[0]): float(r[1]) for r in p.s.oxidation_states}
    try:
        from pymatgen.core import Composition, Element
    except ImportError:
        die("give %block oxidation_states (e.g. Ca +2 / O -2 / Sb +3): guessing them needs "
            "pymatgen")
    guess = Composition(prim.get_chemical_formula()).oxi_state_guesses()
    if not guess:
        die("pymatgen could not guess the host's oxidation states: give %block oxidation_states")
    ox = dict(guess[0])
    for dop in p.s.dopants:
        common = Element(dop).common_oxidation_states or (0,)
        sites = [x["host"] for c in cands.values() for x in c["comps"] if x["new"] == dop]
        target = ox.get(sites[0], 0) if sites and sites[0] else max(common)
        ox[dop] = min(common, key=lambda v: (abs(v - target), -v))
    return ox


# How many neighbours ShakeNBreak's rule distorts for a defect in charge state q: n = sum over the
# components of -(oxidation state put there - that of the site) + q, and |n| neighbours if |n| <= 4,
# else 8 - |n|.
#   Called by: step_distort()
def neighbours_to_distort(comps, q, ox):
    n = q
    for c in comps:
        new = ox.get(c["new"], 0) if c["new"] else 0
        site = ox.get(c["host"], 0) if c["host"] else 0
        n += -(new - site)
    n = int(round(n))
    return abs(n) if abs(n) <= 4 else abs(8 - abs(n))


# The final spin of a charge state's best screening run, as a starting spin with the right parity
# for its electron count.
#   Called by: step_ccd(), step_distort(), step_excite(), step_filter(), step_lineshape(),
#              step_rank(), step_relaxations(), step_zpl(), validate_gamma()
#   Calls:     electron_parity()
def start_spin(atoms, q, spin):
    par = electron_parity(atoms, q)
    if spin is None:
        return par
    s = int(round(abs(spin)))
    return s if s % 2 == par else s + 1


# The points C4 distorts a defect around (distort_around), each as (label, ShakeNBreak keyword: a
# site_index or frac_coords in atoms, the screened structure, which keeps the build's atom order):
# a component is its atom (the anchor's index, or the nearest atom of its element to where it was
# built) or its vacant site. anchor: the anchor atom (the first vacancy if the anchor is one);
# vacancy: the first vacancy (the anchor atom if there is none); centre: the mean of the
# components' positions (info's centre), or the atom within CENTRE_ATOM_TOL of it if there is one;
# each: every component in turn; auto: a single defect's own site, the centre of a complex whose
# components are all within PAIR_BOND_FACTOR host bond lengths (bond) of each other, else each
# component. A single defect has one point whatever the mode.
#   Called by: step_distort()
def distortion_centres(atoms, comps, info, mode, bond):
    from ase.geometry import find_mic
    cell = np.asarray(atoms.cell)
    syms = np.array(atoms.get_chemical_symbols())

    # The ShakeNBreak keyword for a point with no atom on it.
    def point(r):
        return {"frac_coords": np.linalg.solve(cell.T, np.asarray(r, float)) % 1.0}

    # The minimum-image distances from a point to the atoms of the screened structure.
    def dists(r, idx=None):
        idx = np.arange(len(atoms)) if idx is None else idx
        _, d = find_mic(atoms.positions[idx] - np.asarray(r, float), cell, pbc=True)
        return d

    pts = info.get("points") or [info["centre"]]
    own = []                                       # each component's (label, keyword)
    for k, c in enumerate(comps):
        if c["host"] and not c["new"]:
            own.append((f"V_{c['host']}", point(pts[k])))
        elif k == 0 and info.get("anchor_index") is not None:
            own.append((c["new"], {"site_index": int(info["anchor_index"])}))
        else:
            idx = np.flatnonzero(syms == c["new"])
            own.append((c["new"], {"site_index": int(idx[np.argmin(dists(pts[k], idx))])}))
    if len(comps) == 1:
        return own
    if mode == "anchor":
        return [own[0]]
    if mode == "vacancy":
        return [next((o for c, o in zip(comps, own) if c["host"] and not c["new"]), own[0])]
    if mode == "auto":
        r = np.asarray(pts, float)
        _, sep = find_mic((r[:, None, :] - r[None, :, :]).reshape(-1, 3), cell, pbc=True)
        mode = "centre" if np.max(sep) <= PAIR_BOND_FACTOR * bond else "each"
    if mode == "each":
        return own
    d = dists(info["centre"])
    i = int(np.argmin(d))
    return [("the centre", {"site_index": i} if d[i] < CENTRE_ATOM_TOL else point(info["centre"]))]


# The distinct minima of a defect's MACE relaxations (entries: dicts with 'atoms' and 'energy',
# MACE's, all of one defect in one cell, so on one energy surface whatever their charge state),
# lowest first: relaxations into the same structure (same_structures, compared only within
# RANK_E_TOL of each other, as C5's ranking does) merged under the lowest. Returns [(that entry,
# [the entries merged into it])].
#   Called by: step_distort()
#   Calls:     same_structures()
def mace_minima(entries):
    out = []
    for e in sorted(entries, key=lambda x: x["energy"]):
        hit = next((m for m in out if abs(m[0]["energy"] - e["energy"]) < RANK_E_TOL
                    and len(same_structures([m[0]["atoms"], e["atoms"]])) == 1), None)
        if hit is None:
            out.append((e, [e]))
        else:
            hit[1].append(e)
    return out


# C4: the distortions, searched with MACE, in the cell C4 and C5 work in (distort_cell): by
# default the electronic cell (B4), so that C1-C3, which screen every candidate, run in the small
# screening cell while the survivors' ground states are searched for where their electronic
# structure is judged. For each survivor and each of its charge states (charges from C3), its
# screened structure in that charge state (or the nearest screened one) is carried into that cell
# (carry_over: the defect rebuilt there, the atoms near it moved as in the screened structure, the
# rest on their perfect-crystal sites; <name>_<cell tag>), which C5 relaxes there as it is: the
# start every other structure has to beat. ShakeNBreak's distort_and_rattle then moves that charge
# state's neighbours_to_distort nearest neighbours of each point the defect is distorted around
# (distortion_centres: distort_around) in or out by each of distortions, and rattles the other
# atoms (rattle_stdev, default 10% of the host bond length; rattle_dmin, 80%); with no neighbours
# to distort it is only rattled. A defect distorted around more than one point gets c1, c2, ...
# (its components, in order) in its structures' names. With mace (the default), every one of
# these structures, of all the defect's charge states and the starts too, is relaxed with MACE,
# the cell fixed (mace_relaxations: as jobs with mace_jobscript, %block runner_distort; their
# files in C_screen/mace/). MACE ignores the charge, so they all lie on one energy surface: they
# are pooled, those that relaxed into the same structure merged (mace_minima), and the
# distort_starts lowest distinct ones kept (<cid>_m1, _m2, ...), which C5 relaxes in CASTEP in
# every charge state the defect carries (<cid>_<q>_m1, ...), beside its start. The charged
# states' distortions (other neighbour counts) thus widen the search for all of them, at MACE's
# cost. Without MACE (mace : false), C5 relaxes the distortions themselves, each in its own
# charge state. The jobs carry the cell's matrix and tag, which end their names. With
# distort_cell : screening, or when B4 chose the screening cell itself, the screened structures
# are distorted where they are, and the screened runs (C2) are the starts (also relaxed with MACE
# for the pool). The structures go to C_screen/structures/distortions/. The table lists each
# defect's lowest MACE minima. None while MACE jobs run.
#   Called by: main() (step C4)
#   Calls:     carry_over(), distortion_centres(), format_table(), host_supercell(), load_cell(),
#              mace_minima(), mace_relaxations(), neighbours_to_distort(), oxidation_states(),
#              qtag(), save_cell(), screen_setup(), start_spin()
def step_distort(p):
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        from shakenbreak.distortions import distort_and_rattle
    except ImportError as err:
        die(f"the distortion step needs ShakeNBreak (pip install shakenbreak): {err}")
    import warnings
    warnings.filterwarnings("ignore")
    cands, prim, P = screen_setup(p)
    d = p.phase_dir("C") / "structures" / "distortions"
    survivors = p.out("filter")["survivors"]
    defects = p.out("prerelax")["defects"]
    ox = oxidation_states(p, prim, cands)
    host = host_supercell(prim, P)
    bond = float(np.sort(host.get_all_distances(mic=True)[0])[1])
    stdev = p.s.rattle_stdev or 0.1 * bond
    dmin = p.s.rattle_dmin or 0.8 * bond
    el = None
    if p.s.distort_cell == "electronic":
        el = p.out("electronic")
        if el["same"]:
            el = None                       # (the electronic cell is the screening cell)
    end = f"_{el['tag']}" if el else ""
    extra = {"matrix": el["matrix"], "cell": el["tag"]} if el else {}
    where = f" in the electronic cell ({el['tag']})" if el else ""
    jobs, pool, spins = [], {}, {}       # C5's jobs; each survivor's structures for MACE
    for s in survivors:
        cid = s["cand"]
        comps, label = cands[cid]["comps"], f"{cid} ({cands[cid]['name']})"
        screened = {int(q): b for q, b in s["best"].items()}
        pool[cid] = []
        for q in s["charges"]:
            src = screened.get(q) or screened[min(screened, key=lambda x: (abs(x - q), x))]
            atoms = load_cell(p, src["structure"])
            S = spins[(cid, q)] = start_spin(atoms, q, src["spin"] if src["q"] == q else None)
            info = defects[cid]
            if el:
                atoms, info = carry_over(prim, comps, P, atoms, np.array(el["matrix"]),
                                         label=label)
                atoms.set_initial_magnetic_moments(np.zeros(len(atoms)))
                name = f"{cid}_{qtag(q)}{end}"
                rel = save_cell(p, d / f"{name}.cell", atoms,
                                f"{name}: {cands[cid]['name']}, q = {q:+d}, its screened "
                                f"structure{where} (C4)")
                jobs.append(dict({"name": name, "cand": cid, "q": q, "spin": S, "factor": None,
                                  "neighbours": None, "around": None, "structure": rel,
                                  "source": "screened, carried over"}, **extra))
                pool[cid].append({"name": name, "structure": rel, "what": f"{q:+d} start"})
            else:
                pool[cid].append({"name": f"{cid}_{qtag(q)}_screened",
                                  "structure": src["structure"], "what": f"{q:+d} screened"})
            nb = neighbours_to_distort(comps, q, ox)
            struct = AseAtomsAdaptor.get_structure(atoms)
            centres = distortion_centres(atoms, comps, info, p.s.distort_around, bond)
            factors = [f for f in p.s.distortions if abs(f) > 1e-9] if nb else ["rattled"]
            for k, (around, centre) in enumerate(centres):
                part = f"_c{k + 1}" if len(centres) > 1 else ""
                for f in factors:
                    tag = "rattled" if f == "rattled" else f"b{'m' if f < 0 else 'p'}{abs(f):.2f}"
                    out = distort_and_rattle(struct, 1.0 if f == "rattled" else 1.0 + f, nb,
                                             stdev=stdev, d_min=dmin, **centre)
                    new = AseAtomsAdaptor.get_atoms(out["distorted_structure"])
                    new.set_initial_magnetic_moments(atoms.get_initial_magnetic_moments())
                    name = f"{cid}_{qtag(q)}{part}_{tag}{end}"
                    rel = save_cell(p, d / f"{name}.cell", new,
                                    f"{name}: {cands[cid]['name']}, q = {q:+d}, "
                                    + ("rattled" if f == "rattled" else
                                       f"{nb} neighbours' bonds {f:+.2f}")
                                    + f" around {around}{where} (C4)")
                    pool[cid].append({"name": name, "structure": rel, "what": f"{q:+d} "
                                      + ("rattled" if f == "rattled" else f"{f:+.2f}")
                                      + (f" around {around}" if part else "")})
                    if not p.s.mace:
                        jobs.append(dict({"name": name, "cand": cid, "q": q, "spin": S,
                                          "factor": f, "neighbours": nb, "around": around,
                                          "structure": rel,
                                          "source": f"distortion around {around}" if part
                                          else "distortion"}, **extra))
    rows, nmin = [], 0
    if p.s.mace:
        files = p.phase_dir("C") / "mace"
        tasks = [{"atoms": load_cell(p, e["structure"]), "cache": files / f"{e['name']}.json",
                  "label": e["name"]} for es in pool.values() for e in es]
        res = mace_relaxations(p, "distort", tasks)
        if res is None:
            return None
        for s in survivors:
            cid = s["cand"]
            entries = []
            for e in pool[cid]:
                atoms, _, energy, steps = res[str(files / f"{e['name']}.json")]
                entries.append(dict(e, atoms=atoms, energy=energy, steps=steps))
            minima = mace_minima(entries)
            e_low = minima[0][0]["energy"]
            for k, (rep, members) in enumerate(minima):
                kept = k < p.s.distort_starts
                if kept:
                    nmin += 1
                    name = f"{cid}_m{k + 1}{end}"
                    rel = save_cell(p, d / f"{name}.cell", rep["atoms"],
                                    f"{name}: {cands[cid]['name']}, MACE minimum {k + 1} of its "
                                    f"pooled distortions{where} (C4)")
                    for q in s["charges"]:
                        jobs.append(dict({"name": f"{cid}_{qtag(q)}_m{k + 1}{end}", "cand": cid,
                                          "q": q, "spin": spins[(cid, q)], "factor": "mace",
                                          "neighbours": None, "around": None, "structure": rel,
                                          "source": f"MACE minimum {k + 1}",
                                          "mace": {"rank": k + 1, "energy": rep["energy"],
                                                   "from": [m["what"] for m in members]}},
                                         **extra))
                if k < max(p.s.distort_starts, MACE_TABLE_MINIMA):
                    whats = [m["what"] for m in members]
                    rows.append([cid, cands[cid]["name"], k + 1, rep["energy"] - e_low,
                                 rep["energy"], len(members),
                                 ", ".join(whats[:4]) + (f" (+{len(whats) - 4})"
                                                         if len(whats) > 4 else ""),
                                 "yes" if kept else ""])
            shown = max(p.s.distort_starts, MACE_TABLE_MINIMA)
            if len(minima) > shown:
                rows.append([cid, "", "", "", "", "", f"{len(minima) - shown} more distinct "
                             "minima", ""])
    cell = ({"tag": el["tag"], "matrix": el["matrix"]} if el else
            {"tag": p.out("choose")["tag"], "matrix": P.tolist()})
    nstruct = sum(len(es) for es in pool.values())
    table = format_table(
        ["id", "defect", "minimum", "dE/eV", "E (MACE)/eV", "structures", "from", "kept"], rows,
        f"MACE's search{where}: every charge state's distortions and starts, relaxed with MACE "
        "and pooled by defect (MACE ignores the charge), the distinct minima lowest first (dE: "
        f"above the lowest; from: charge state and distortion); the lowest {p.s.distort_starts} "
        "go on to C5 in every charge state") if rows else None
    return {"jobs": jobs, "oxidation_states": ox, "cell": cell, "electronic": el is not None,
            "mace": bool(p.s.mace), "_table": table,
            "_note": (f"{nstruct} structures relaxed with MACE for {len(survivors)} survivors, "
                      f"{nmin} minima kept: " if p.s.mace else
                      f"{nstruct} distortions for {len(survivors)} survivors (no MACE): ")
                     + f"{len(jobs)} CASTEP relaxations for C5, in the "
                     + (f"electronic cell ({el['tag']})" if el else
                        f"screening cell ({cell['tag']})")
                     + " (oxidation states "
                     + ", ".join(f"{k} {v:+g}" for k, v in ox.items()) + ")"}


# Runs a list of distortion-type jobs (name, cand, starting structure, q, spin) with the
# screening settings in <phase>/jobs/<subdir> (C_screen/ for drelax, C_screen/ for rank), with the
# runner settings of the step that runs them, and returns ({name: job_summary}, {cell tag: its
# band edges}), the final structures saved in <phase>/structures/<structdir>. A job in another
# cell than the screening cell (its 'matrix' and 'cell': C4's in the electronic cell) has its
# defect found in that cell (build_defect) and its levels and band carriers measured against that
# cell's own band edges at the same settings: those given for its cell in edges (C5's, for C6),
# else from a single point of its pristine supercell in the same batch (host_<cell tag>).
# (None, None) while the batch hasn't finished.
#   Called by: step_drelax(), step_rank()
#   Calls:     Project.batch(), Project.host(), Project.production_param(), band_edges(),
#              build_defect(), host_supercell(), job_summary(), load_cell(), read_bands(),
#              screen_job(), screen_overrides()
def run_screen_jobs(p, step, subdir, todo, structdir, edges=None):
    defects = p.out("prerelax")["defects"]
    cands, P = p.out("generate")["candidates"], np.array(p.out("choose")["matrix"])
    host = p.out("screen")["host"]
    known = {None: (host["vbm"], host["cbm"])}
    known.update({tag: tuple(e) for tag, e in (edges or {}).items()})
    jobs, cells, infos = [], {}, {}
    for t in todo:
        tag = None
        if t.get("matrix") is not None and not np.array_equal(np.array(t["matrix"]), P):
            tag = t["cell"]
            cells.setdefault(tag, t["matrix"])
        if (t["cand"], tag) not in infos:             # where the defect is, in that cell
            infos[(t["cand"], tag)] = defects[t["cand"]] if tag is None else build_defect(
                p.host(), np.array(t["matrix"]), cands[t["cand"]]["comps"], t["cand"])[1]
        jobs.append(screen_job(p, t["name"], load_cell(p, t["structure"]),
                               infos[(t["cand"], tag)], t["q"], t["spin"]))
    new = sorted(tag for tag in cells if tag not in known)
    for tag in new:
        jobs.append({"name": f"host_{tag}", "atoms": host_supercell(p.host(),
                                                                    np.array(cells[tag])),
                     "kind": "sp", "kgrid": [1, 1, 1], "symmetry": True,
                     "overrides": {k: v for k, v in screen_overrides(p).items()
                                   if k != "geom_max_iter"}})
    phase = STEP_INDEX[step].phase
    res = p.batch(f"{PHASE_DIRS[phase]}/jobs/{subdir}", jobs, step, p.production_param())
    if not res["complete"]:
        return None, None
    for tag in new:
        hc = res["calcs"][f"host_{tag}"]
        bands = hc.dir / f"{hc.seed}.bands"
        if res["jobs"][f"host_{tag}"]["status"] != "done" or not bands.exists():
            die(f"the pristine {tag} cell's single point did not finish ({hc.dir}): the runs in "
                "that cell are measured against its band edges")
        known[tag] = band_edges(read_bands(bands))
    out = {}
    for t in todo:
        tag = t.get("cell") if t.get("cell") in cells else None
        out[t["name"]] = dict(t, **job_summary(p, res, t["name"], known[tag],
                                               p.phase_dir(phase) / "structures" / structdir))
    return out, {tag: list(known[tag]) for tag in cells}


# The energy of every SCF of a .castep and whether it converged: [(energy, converged)] in order,
# the energy of the kind the workflow reads (which, the energy keyword; auto: the est. 0K one where
# CASTEP prints it, with smearing, else the final one), as the runs' own energies are. An SCF did
# not converge when CASTEP's warning (SCF_FAILED) comes between the previous energy and this
# one's forces; CASTEP still prints its energy and forces, and a relaxation carries on from it.
#   Called by: drelax_ranking()
#   Calls:     first_energy()
def step_energies(text, which="auto"):
    ends = list(re.finditer(ENERGY_PATTERNS["final"], text))
    out = []
    for k, m in enumerate(ends):
        stop = ends[k + 1].start() if k + 1 < len(ends) else len(text)
        tail = text[m.end():stop]
        fm = re.search(r"\*+ [\w ]*Forces \*+", tail)
        scf = text[ends[k - 1].end() if k else 0:m.end() + (fm.start() if fm else 0)]
        e = float(m.group(1)) if which == "final" else first_energy(m.group(0) + tail, which)
        out.append((float(m.group(1)) if e is None else e,
                    not any(s in scf for s in SCF_FAILED)))
    return out


# C5's ranking, for summary.txt, so that what C4's MACE minima (or, without MACE, its distortions)
# found can be looked at before C6 shares them: every relaxation of each survivor's charge states,
# lowest energy first, in the cell C4 worked in, against the start they had to beat (the screened
# structure relaxed there, C4's undistorted job; with distort_cell : screening, C2's screened run).
# Runs that relaxed into the same structure (same_structures, tried only for energies within
# RANK_E_TOL) share a letter, A the lowest. A charge state has a new minimum when its lowest
# structure is not the start's and is lower by at least distortion_min_gain (C6's test). Runs
# without a usable energy come last, with the lowest of their steps whose SCF converged
# (step_energies); a run with steps whose SCF did not converge says how many. Returns (the tables,
# the charge states with a new minimum, all charge states).
#   Called by: step_drelax()
#   Calls:     format_table(), load_cell(), same_structures(), step_energies()
def drelax_ranking(p, runs):
    survivors = p.out("filter")["survivors"]
    cands = p.out("generate")["candidates"]
    c4 = p.out("distort")
    cell = c4["cell"]["tag"] if c4.get("electronic") else None
    gmin = p.s.distortion_min_gain
    jobs = p.phase_dir("C") / "jobs" / "drelax"

    # A run's name in the tables: start, MACE and its minimum's rank, or (without MACE) its
    # distortion factor (with the point it was distorted around, when there are several) or rattled.
    def label(r):
        f = r.get("factor")
        if f is None:
            return "start" + ("" if cell else " (C2)")
        if f == "mace":
            return f"MACE {r['mace']['rank']}"
        if f == "rattled":
            return "rattled"
        return f"{f:+.2f}" + (f" around {r['around']}"
                              if str(r.get("source", "")).startswith("distortion around") else "")

    summary, rows, new = [], [], 0
    for s in survivors:
        cid = s["cand"]
        for q in s["charges"]:
            ok, failed = [], []
            if cell is None:                    # (the start is C2's screened run)
                b = s["best"].get(str(q))
                if b and b.get("energy") is not None and b.get("structure"):
                    ok.append(dict(b, factor=None, steps=None, geom_converged=None, bad=0))
            for name, r in sorted(runs.items()):
                if r["cand"] != cid or r["q"] != q:
                    continue
                f = jobs / name / f"{name}.castep"
                es = (step_energies(f.read_text(errors="replace"), p.s.energy) if f.exists()
                      else [])
                r = dict(r, name=name, bad=sum(not c for _, c in es),
                         best=min((e for e, c in es if c), default=None))
                (ok if r["energy"] is not None and r["structure"] else failed).append(r)
            ok.sort(key=lambda e: e["energy"])
            start = next((e for e in ok if e.get("factor") is None), None)
            e0 = start["energy"] if start else (ok[0]["energy"] if ok else None)
            letters, reps = [], []              # (each structure's first run, its letter)
            for e in ok:
                atoms = load_cell(p, e["structure"])
                hit = next((k for k, (r0, a0) in enumerate(reps)
                            if abs(r0["energy"] - e["energy"]) < RANK_E_TOL
                            and len(same_structures([a0, atoms])) == 1), None)
                if hit is None:
                    reps.append((e, atoms))
                    hit = len(reps) - 1
                letters.append(chr(ord("A") + hit) if hit < 26 else f"#{hit + 1}")
            for e, letter in zip(ok, letters):
                rows.append([cid, f"{q:+d}", label(e), e["energy"] - e0, e["energy"], e["spin"],
                             e["steps"], {True: "yes", False: "no"}.get(e["geom_converged"]),
                             letter, (f"SCF unconverged at {e['bad']} step"
                                      + ("s" if e["bad"] > 1 else "")) if e["bad"] else ""])
            for r in failed:
                why = (r["reason"] or "its last SCF did not converge").splitlines()[0]
                rows.append([cid, f"{q:+d}", label(r), None, None, r["spin"], r["steps"],
                             {True: "yes", False: "no"}.get(r["geom_converged"]), "",
                             f"no energy ({why}); " + (
                                 "no step's SCF converged" if r["best"] is None else
                                 f"lowest converged step {r['best']:.4f} eV" if e0 is None else
                                 f"lowest converged step {r['best'] - e0:+.4f} eV")])
            if not ok:
                verdict = "no run has an energy"
            elif start is None:
                verdict = "the start has no energy (dE against the lowest)"
            elif letters[0] == letters[ok.index(start)]:
                verdict = "the start is the lowest"
            elif e0 - ok[0]["energy"] >= gmin:
                verdict, new = "NEW MINIMUM", new + 1
            else:
                verdict = f"lower by less than {gmin:g} eV"
            summary.append([cid, cands[cid]["name"], f"{q:+d}", start["energy"] if start else None,
                            ok[0]["energy"] if ok else None,
                            e0 - ok[0]["energy"] if start and ok else None,
                            label(ok[0]) if ok else "", len(reps), len(failed), verdict])
    where = (f"the electronic cell ({cell})" if cell else
             f"the screening cell ({p.out('choose')['tag']})")
    table = (format_table(["id", "defect", "q", "start E/eV", "lowest E/eV", "gain/eV", "lowest",
                           "distinct", "failed", "verdict"], summary,
                          f"The distortions' search in {where}, by charge state (energies at the "
                          "screening settings; start: the screened structure"
                          + (", relaxed there" if cell else " (C2)") + "; a new minimum: a "
                          f"structure other than the start's, lower by distortion_min_gain "
                          f"({gmin:g} eV) or more)")
             + "\n" + format_table(["id", "q", "run", "dE/eV", "E/eV", "spin", "steps", "relaxed",
                                    "same", "note"], rows,
                                   "Every relaxation, lowest first in each charge state (dE: "
                                   "against the start; same letter: relaxed into the same "
                                   "structure, A the lowest; runs without a usable energy last)"))
    return table, new, len(summary)


# C5: relaxes C4's jobs with the screening settings (C2's, Gamma point, no restarts): each
# charge state from its start (the screened structure carried into C4's cell) and from each of
# C4's MACE minima (without MACE, from each of its distortions), in the cell C4 worked in, against
# that cell's band edges (run_screen_jobs). Its summary ranks every relaxation of each charge
# state against its start (drelax_ranking), so whether a MACE minimum led to a new minimum can be
# looked at before C6 (--until C5).
#   Called by: main() (step C5)
#   Calls:     drelax_ranking(), problem_table(), run_screen_jobs()
def step_drelax(p):
    out, edges = run_screen_jobs(p, "drelax", "drelax", p.out("distort")["jobs"], "drelax")
    if out is None:
        return None
    bad = sum(r["energy"] is None for r in out.values())
    ranking, new, states = drelax_ranking(p, out)
    problems = problem_table(out)
    return {"runs": out, "edges": edges,
            "_table": ranking + ("\n" + problems if problems else ""),
            "_note": f"{len(out)} relaxations" + (f", {bad} without a usable energy" if bad
                                                  else "")
                     + f"; a new minimum in {new} of {states} charge states"}


# Groups structures that are the same (pymatgen's StructureMatcher, which also finds a distortion
# repeated along another direction), as lists of indices; without pymatgen, by comparing atom by
# atom (all come from the same cell with the same atom order). The tolerances are tight (stol 0.05,
# i.e. about 0.1 A for a typical oxide): defect ground states can differ by a few tenths of an
# Angstrom around the defect, which pymatgen's defaults would merge.
#   Called by: distinct_states(), drelax_ranking(), duplicates(), mace_minima(), step_rank()
#   Calls:     displacements()
def same_structures(atoms_list):
    try:
        from pymatgen.analysis.structure_matcher import StructureMatcher
        from pymatgen.io.ase import AseAtomsAdaptor
        sm = StructureMatcher(ltol=0.05, stol=0.05, angle_tol=2, primitive_cell=False,
                              scale=False, attempt_supercell=False)
        structs = [AseAtomsAdaptor.get_structure(a) for a in atoms_list]

        # Whether two structures match.
        def match(i, j):
            return sm.fit(structs[i], structs[j])
    except ImportError:
        # Whether every atom of two structures is within 0.05 A of its counterpart.
        def match(i, j):
            return len(atoms_list[i]) == len(atoms_list[j]) and \
                displacements(atoms_list[i], atoms_list[j]).max() < 0.05
    groups = []
    for i in range(len(atoms_list)):
        for g in groups:
            if match(g[0], i):
                g.append(i)
                break
        else:
            groups.append([i])
    return groups


# --------------------------------------------------------------------------
# Training data and model jobs (F2's model of the production runs)
# --------------------------------------------------------------------------
# Every geometry step of a .castep as a training structure (read_trajectory, with the free energy
# E-TS that the forces belong to): ASE Atoms with info REF_energy, total_charge (q), run (the job),
# step, spin (the run's final spin) and arrays REF_forces.
#   Called by: step_lineshape()
#   Calls:     read_trajectory()
def castep_frames(path, q, spin, run):
    out = []
    for k, (atoms, e, forces) in enumerate(read_trajectory(Path(path).read_text(errors="replace"),
                                                           "free")):
        atoms.info.update(REF_energy=e, total_charge=q, run=run, step=k,
                          spin=0.0 if spin is None else float(spin))
        atoms.arrays["REF_forces"] = forces
        out.append(atoms)
    return out


# The filters of F2's training data, after Wang, Mosquera-Lois and Walsh (arXiv:2603.05238): a
# structure more than MODEL_E_WINDOW above the lowest of its charge state in its cell (its number
# of atoms), with two atoms closer than MODEL_MIN_DIST, or a force above MODEL_MAX_FORCE (the first
# steps of a wild distortion) is left out. Returns the structures kept.
#   Called by: step_lineshape()
def filter_frames(frames):

    # What a structure's energy is compared within: its charge and its number of atoms.
    def group(a):
        return a.info["total_charge"], len(a)

    low = {}
    for a in frames:
        low[group(a)] = min(low.get(group(a), np.inf), a.info["REF_energy"])
    out = []
    for a in frames:
        if a.info["REF_energy"] - low[group(a)] > MODEL_E_WINDOW:
            continue
        if np.abs(a.arrays["REF_forces"]).max() > MODEL_MAX_FORCE:
            continue
        d = a.get_all_distances(mic=True)
        if np.min(d + np.eye(len(a)) * 1e9) < MODEL_MIN_DIST:
            continue
        out.append(a)
    return out


# The mace_run_train command of a defect's model, run in its folder (the training files copied
# there): its heads (F2's model has one, 'pbe'), each with its train and test file,
# validated on 10 % of its training set, or on its own training structures ({head}_valid.xyz) when
# it has too few to hold some back (heads: [(head, has a test file, validated on its own)]), and
# E0s fitted to the data (average); the total charge as a categorical embedding (CHARGE_CLASSES
# classes from -CHARGE_OFFSET); trained from scratch (multiheads_finetuning off); model_r_max,
# model_channels, model_max_L, model_batch, model_epochs, model_device (default mace_device) and
# model_train_args added at the end.
#   Called by: step_lineshape()
def train_command(p, heads):
    hd = {h: {"train_file": f"{h}_train.xyz",
              **({"valid_file": f"{h}_valid.xyz"} if own else {"valid_fraction": 0.1}),
              "E0s": "average", **({"test_file": f"{h}_test.xyz"} if has_test else {})}
          for h, has_test, own in heads}
    emb = {"total_charge": {"type": "categorical", "per": "graph", "num_classes": CHARGE_CLASSES,
                            "offset": CHARGE_OFFSET, "emb_dim": 32}}
    return (f"{p.s.model_command} --name=model --work_dir=. --model=MACE "
            f"--heads={shlex.quote(repr(hd))} --multiheads_finetuning=False "
            f"--embedding_specs={shlex.quote(repr(emb))} --total_charge_key=total_charge "
            "--energy_key=REF_energy --forces_key=REF_forces --E0s=average "
            f"--r_max={p.s.model_r_max:g} --num_channels={p.s.model_channels} "
            f"--max_L={p.s.model_max_L} --batch_size={p.s.model_batch} "
            f"--valid_batch_size={p.s.model_batch} --max_num_epochs={p.s.model_epochs} "
            f"--default_dtype=float64 --seed={MODEL_SEED} "
            f"--device={p.s.model_device or p.s.mace_device} {p.s.model_train_args}").strip()


# The model file a training wrote in its folder (the second stage's, if there is one), or None.
#   Called by: run_training(), step_lineshape()
def trained_model(d):
    for name in ("model_stagetwo.model", "model.model"):
        if (Path(d) / name).exists():
            return Path(d) / name
    return None


# Runs model jobs until each has made what it makes: F2's trainings (done when its folder has a
# model, trained_model) and its phonon calculations (a task's own 'done' file). With
# model_jobscript as batch jobs (the template's {COMMAND} line becomes 'cd <folder> && <command> >
# <log> 2>&1', the task's 'log', default train.log; {JOBNAME} and {NCORES} filled in; submitted
# with the runner's submit_cmd, polled every poll s; --prepare-only writes them only, --no-wait
# returns while they run), else here, one after the other. One that ends without its result stops
# the run with the end of its log. A task: {'cid' (its name), 'dir', 'command', and optionally
# 'done', 'log', 'what' (for messages, default 'training') and 'job' (the job name's prefix)}.
# Returns True when every task is done.
#   Called by: step_lineshape()
#   Calls:     Project.mace_runner(), Project.template(), job_active(), send_job(),
#              trained_model(), write_json()
def run_training(p, label, tasks):

    # Whether a task has made what it makes.
    def finished(t):
        return Path(t["done"]).exists() if t.get("done") else trained_model(t["dir"]) is not None

    # The end of a task's log, for a message.
    def tail(t):
        f = Path(t["dir"]) / t.get("log", "train.log")
        lines = f.read_text(errors="replace").strip().splitlines() if f.exists() else []
        return lines[-1][:300] if lines else f"{f.name} is empty or missing"

    todo = [t for t in tasks if not finished(t)]
    if not todo:
        return True
    if not p.s.model_jobscript:
        import subprocess
        for t in todo:
            log(f"{label}: running the {t.get('what', 'training')} of {t['cid']} here "
                "(model_jobscript not given)")
            with open(t["dir"] / t.get("log", "train.log"), "w") as fh:
                subprocess.run(t["command"], shell=True, cwd=t["dir"], stdout=fh,
                               stderr=subprocess.STDOUT)
            if not finished(t):
                die(f"{label}: the {t.get('what', 'training')} of {t['cid']} did not finish "
                    f"({t['dir']}): {tail(t)}")
        return True
    args, template = p.mace_runner(), p.template("model_jobscript")
    for t in todo:
        state = t["dir"] / "job.json"
        info = json.loads(state.read_text()) if state.exists() else {}
        if info.get("jobid") and job_active(info["jobid"], args):
            continue
        if info.get("jobid"):
            if not getattr(p.cl, "resubmit_failed", False):
                die(f"{label}: the {t.get('what', 'training')} job of {t['cid']} "
                    f"({info['jobid']}) ended without its result: {tail(t)}; fix it, then rerun "
                    "with --resubmit-failed")
        lines = template.splitlines()
        i = next(k for k, line in enumerate(lines)
                 if "{COMMAND}" in line and not line.lstrip().startswith("#"))
        pre, post = lines[i].split("{COMMAND}", 1)
        run = f"cd {t['dir'].resolve()} && {t['command']} > {t.get('log', 'train.log')} 2>&1"
        lines[i] = f"{pre}bash -c {shlex.quote(run)}{post}"
        text = "\n".join(lines).replace("{JOBNAME}", f"dwf_{t.get('job', 'model')}_{t['cid']}")
        (t["dir"] / "job.sh").write_text(text.replace("{NCORES}", str(p.s.mace_threads)) + "\n")
        if p.cl.prepare_only:
            continue
        jobid = send_job(t["dir"], "job.sh", args)
        write_json(state, {"jobid": jobid, "time": time.time()})
        log(f"{label}: submitted the {t.get('what', 'training')} of {t['cid']} (job {jobid})")
    if p.cl.prepare_only:
        log(f"{label}: the job scripts are in " + ", ".join(str(t["dir"]) for t in todo))
        return False
    while True:
        left = [t for t in todo if not finished(t)]
        if not left:
            return True
        for t in left:
            info = json.loads((t["dir"] / "job.json").read_text())
            if not job_active(info["jobid"], args) and time.time() - info["time"] > 120:
                die(f"{label}: the {t.get('what', 'training')} job of {t['cid']} "
                    f"({info['jobid']}) ended without its result: {tail(t)}"
                    + (" (rerun with --resubmit-failed to carry on where it stopped)"
                       if t.get("done") else ""))
        if p.cl.no_wait:
            log(f"{label}: {len(left)} jobs still running; rerun the same command later")
            return False
        time.sleep(args.poll)


# --------------------------------------------------------------------------
# C6: ground states (C5's relaxations, distortions shared between charge states)
# --------------------------------------------------------------------------
# The distinct relaxed structures of each candidate and charge state in the cell C4 worked in (cell:
# its tag, None for the screening cell), lowest energy first: C5's relaxations there with a usable
# energy (in the screening cell, the screened one, C2, too), identical ones merged
# (same_structures, keeping the lowest). Returns {(cand, q): [entries]}.
#   Called by: step_rank()
#   Calls:     load_cell(), same_structures()
def distinct_states(p, survivors, results, cell):
    pool = {}
    for s in survivors:
        for q in s["charges"]:
            pool[(s["cand"], q)] = []
            b = s["best"].get(str(q))
            if cell is None and b and b["structure"]:
                pool[(s["cand"], q)].append(
                    {"name": b["name"], "energy": b["energy"], "spin": b["spin"],
                     "structure": b["structure"], "source": "screened", "factor": None,
                     "band": b.get("band")})
    for name, r in results.items():
        if r["energy"] is not None and r["structure"] and r.get("cell") == cell \
                and (r["cand"], r["q"]) in pool:
            pool[(r["cand"], r["q"])].append(
                {"name": name, "energy": r["energy"], "spin": r["spin"],
                 "structure": r["structure"], "source": r.get("source", "distortion"),
                 "factor": r.get("factor"), "band": r.get("band")})
    out = {}
    for key, entries in pool.items():
        if not entries:
            continue
        entries.sort(key=lambda e: e["energy"])
        atoms = [load_cell(p, e["structure"]) for e in entries]
        groups = same_structures(atoms)
        firsts = sorted(min(g) for g in groups)
        out[key] = [dict(entries[i], copies=len(next(g for g in groups if min(g) == i)))
                    for i in firsts]
    return out


# C6: ranks the relaxed structures of each survivor and charge state in the cell C4 worked in
# (distinct_states). A structure (from a MACE minimum, or without MACE a distortion) counts as found
# when it beats the start by distortion_min_gain: the screened structure (C2's run; in the
# electronic cell, its relaxation there, C4's undistorted job). A structure found in one charge
# state that no other charge state has is then relaxed in those charge states too, in the same cell
# (a batch per round, C_screen/jobs/share_<n>, measured against C5's band edges), for up to
# share_rounds rounds or until nothing new turns up. The lowest keep_structures distinct structures
# of each charge state go on to C7, and then to phase D, with the run each came from, the electrons
# or holes it left in the bands (C8 leaves out band states) and its cell ('matrix', 'cell').
#   Called by: main() (step C6)
#   Calls:     distinct_states(), format_table(), load_cell(), qtag(), run_screen_jobs(),
#              same_structures(), start_spin()
def step_rank(p):
    survivors = p.out("filter")["survivors"]
    results = dict(p.out("drelax")["runs"])
    choose, c4 = p.out("choose"), p.out("distort")
    cell = c4["cell"]["tag"] if c4.get("electronic") else None
    matrix = c4["cell"]["matrix"] if cell else choose["matrix"]

    # The energy of a charge state's start, which a distortion has to beat: its screened run (C2),
    # or in another cell its screened structure relaxed there (C4's undistorted job).
    def start(cid, q):
        if cell is None:
            b = next(s for s in survivors if s["cand"] == cid)["best"].get(str(q))
            return b["energy"] if b else None
        return min((r["energy"] for r in results.values() if r["cand"] == cid and r["q"] == q
                    and r.get("cell") == cell and r.get("factor") is None
                    and r["energy"] is not None), default=None)

    tried = set()
    for rnd in range(1, p.s.share_rounds + 1):
        states = distinct_states(p, survivors, results, cell)
        todo = []
        for (cid, q), entries in states.items():
            best = entries[0]
            if best["factor"] is None or best["name"] in tried:
                continue                        # (the start won, or it has been tried)
            e0 = start(cid, q)
            if e0 is not None and e0 - best["energy"] < p.s.distortion_min_gain:
                continue
            tried.add(best["name"])
            atoms = load_cell(p, best["structure"])
            for (c2, q2), ent2 in states.items():
                if c2 != cid or q2 == q:
                    continue
                others = [load_cell(p, e["structure"]) for e in ent2]
                if any(len(g) > 1 and 0 in g for g in same_structures([atoms] + others)):
                    continue                         # already has this structure
                spin = start_spin(atoms, q2, ent2[0]["spin"])
                todo.append(dict({"name": f"{best['name']}_to_{qtag(q2)}", "cand": cid, "q": q2,
                                  "spin": spin, "factor": best["factor"],
                                  "source": f"from {qtag(q)}", "structure": best["structure"]},
                                 **({"matrix": matrix, "cell": cell} if cell else {})))
        if not todo:
            break
        log(f"C6 round {rnd}: trying {len(todo)} structures found in one charge state in the "
            "others")
        out, _ = run_screen_jobs(p, "rank", f"share_{rnd}", todo, "share",
                                 edges=p.out("drelax").get("edges"))
        if out is None:
            return None
        results.update(out)
    states = distinct_states(p, survivors, results, cell)
    keep, rows = [], []
    for (cid, q), entries in sorted(states.items()):
        best = entries[0]
        e0 = start(cid, q)
        e0 = best["energy"] if e0 is None else e0
        gain = e0 - best["energy"]
        rows.append([cid, f"{q:+d}", e0, best["energy"], gain,
                     best["source"] + (f" {best['factor']}" if best["factor"] not in (None, "mace")
                                       else ""),
                     len(entries)])
        for k, e in enumerate(entries[:p.s.keep_structures]):
            atoms = load_cell(p, e["structure"])
            keep.append({"cand": cid, "q": q, "k": k, "energy": e["energy"],
                         "spin": start_spin(atoms, q, e["spin"]), "structure": e["structure"],
                         "source": e["source"], "factor": e["factor"], "name": e["name"],
                         "band": e.get("band"), "matrix": matrix,
                         "cell": cell or choose["tag"]})
    table = format_table(["id", "q", "start E/eV", "best E/eV", "gain/eV", "from", "distinct"],
                         rows, "Ground-state search in the "
                         + (f"electronic cell ({cell})" if cell else
                            f"screening cell ({choose['tag']})")
                         + " (energies at the screening settings; start: the screened structure"
                         + (", relaxed there)" if cell else ")"))
    found = sum(1 for r in rows if r[4] > p.s.distortion_min_gain)
    return {"structures": keep, "_table": table,
            "_note": f"{len(keep)} structures kept; a distortion won in {found} of "
                     f"{len(rows)} charge states"}


# --------------------------------------------------------------------------
# C7: excited states
# --------------------------------------------------------------------------
# Whether a candidate is a native defect: every element it adds or removes is one of the host's
# (vacancies, antisites, self-interstitials and complexes of them). C7 always passes these on to
# D, because E's formation energies and Fermi level need them, bright or not.
#   Called by: step_excite(), step_filter()
def is_native(cand, host_elements):
    return all(el in host_elements for el in cand["delta"])


# The defect levels of a ground state at one k-point (Gamma in C7): every band within reach (eV)
# of the host's band edges (vbm, cbm: a pristine run at the same settings) whose participation
# number (participation, P) is at most fraction x nat (the cell's atoms): localised on the defect,
# whether inside the gap or inside a band (a resonance, where a semilocal gap can push a level).
# Each: {spin, band (from 1, in its spin channel, as CASTEP counts), e (eV above the VBM),
# occupied, P, where ('gap', 'valence band' or 'conduction band')}.
#   Called by: step_excite(), step_zpl(), zpl_bands_ok()
#   Calls:     occupied_bands()
def defect_levels(b, P, nat, vbm, cbm, reach, fraction, k=0):
    occ, out = occupied_bands(b), []
    for s in range(b["nspin"]):
        for n, e in enumerate(b["eig"][s, k]):
            if not np.isfinite(e) or not vbm - reach <= e <= cbm + reach or n >= len(P[s]):
                continue
            if P[s][n] <= fraction * nat:
                out.append({"spin": s, "band": n + 1, "e": round(float(e - vbm), 4),
                            "occupied": n < occ[s], "P": round(float(P[s][n]), 2),
                            "where": "gap" if vbm < e < cbm else
                                     ("valence band" if e <= vbm else "conduction band")})
    return out


# The excitations C7 and D2 try, from a ground state's defect levels (defect_levels): in each spin
# channel (spin-conserving, so a triplet stays a triplet), from its highest occupied defect level to
# its lowest empty one at least KS_STEP_MIN higher (which skips a degenerate partner). The
# Kohn-Sham step only picks the bands: what counts is the ΔSCF energy. Each: {spin, from_band,
# to_band, e_from, e_to, step, P_from, P_to, where_to}.
#   Called by: step_excite(), step_zpl(), zpl_bands_ok()
def transitions(levels):
    out = []
    for s in sorted({lv["spin"] for lv in levels}):
        occ = [lv for lv in levels if lv["spin"] == s and lv["occupied"]]
        emp = [lv for lv in levels if lv["spin"] == s and not lv["occupied"]]
        if not occ or not emp:
            continue
        a = max(occ, key=lambda lv: lv["e"])
        up = [lv for lv in emp if lv["e"] - a["e"] >= KS_STEP_MIN]
        if not up:
            continue
        c = min(up, key=lambda lv: lv["e"])
        out.append({"spin": s, "from_band": a["band"], "to_band": c["band"], "e_from": a["e"],
                    "e_to": c["e"], "step": round(c["e"] - a["e"], 4), "P_from": a["P"],
                    "P_to": c["P"], "where_to": c["where"]})
    return out


# The %block deltascf_constraints of an excitation (transitions): its occupied band emptied and its
# empty band filled, in its spin channel (CASTEP's 1 = majority, 2 = minority), each looked for at
# its own band index only. In a calculation without spin polarisation an occupation of 1 is two
# electrons, so moving one electron is 0.5 in each band.
#   Called by: dscf_job()
def deltascf_block(ex, nspin):
    lo, hi = (0.5, 0.5) if nspin == 1 else (0.0, 1.0)
    sp = ex["spin"] + 1 if nspin == 2 else 1
    rows = [f"  {band:5d} {occ:8.4f} {sp:4d} {band:4d} {band:4d}"
            for band, occ in ((ex["from_band"], lo), (ex["to_band"], hi))]
    return ("%block deltascf_constraints\n" + "\n".join(rows)
            + "\n%endblock deltascf_constraints\n")


# A C7 job for one structure in charge state q, starting with spin S: a screening job (screen_job:
# Gamma point, no symmetry, the screening keywords) with metals_method : dm (CASTEP's excitation
# constraints need density mixing), %block excite_overrides, then extra. kind: 'sp' or 'geom'.
#   Called by: step_excite()
#   Calls:     screen_job(), screen_overrides()
def excite_job(p, name, atoms, info, q, S, kind, extra=None):
    job = screen_job(p, name, atoms, info, q, S)
    o = dict(screen_overrides(p), metals_method="dm")
    o.update(p.s.excite_overrides)
    o.update(charge=q, spin=S, spin_polarized="true")
    o.update(extra or {})
    job.update(kind=kind, overrides=o)
    return job


# The ΔSCF job of an excitation ex from a ground-state single point gs (its Calc, whose .check it
# reuses: reuse and deltascf_checkpoint, relative to the new job's folder jobdir/name), made from
# the ground-state job gs_job (its structure, settings and k-points) with CASTEP's simple ΔSCF and
# the band weights written (pdos_calculate_weights, for the excited state's localisation); relaxed
# (kind 'geom') or a single point. param: the .param it starts from, to which the constraints are
# added. nspin: the ground state's spin channels.
#   Called by: step_ccd(), step_excite(), step_zpl()
#   Calls:     deltascf_block()
def dscf_job(gs_job, gs, name, jobdir, ex, nspin, kind, param):
    check = os.path.relpath(gs.dir / f"{gs.seed}.check", jobdir / name)
    job = dict(gs_job, name=name, kind=kind,
               overrides=dict(gs_job["overrides"], calculate_deltascf="true",
                              deltascf_method="simple", reuse=check, deltascf_checkpoint=check,
                              pdos_calculate_weights="true"))
    job["param_text"] = param.rstrip() + "\n" + deltascf_block(ex, nspin)
    return job


# The settings every ΔSCF calculation of the production phases and its ground state share (D2,
# F1), as C7's (excite_job): metals_method : dm (CASTEP's excitation constraints need density
# mixing), then %block excite_overrides. Changes the job's overrides in place; returns the job.
#   Called by: step_ccd(), step_zpl()
def dscf_settings(p, job):
    job["overrides"]["metals_method"] = "dm"
    job["overrides"].update(p.s.excite_overrides)
    return job


# The mass-weighted distance between two structures of the same cell, in amu^1/2 Ang:
# ΔQ = sqrt(sum_i m_i |Δr_i|^2), each Δr_i the shortest through the periodic images.
#   Called by: dscf_result(), step_ccd()
def delta_q(a, b):
    from ase.geometry import find_mic
    d, _ = find_mic(b.positions - a.positions, a.cell, pbc=True)
    return float(np.sqrt(np.sum(a.get_masses()[:, None] * d * d)))


# The first energy of a .castep, of the kind the workflow reads (energy; auto: the 0 K estimate
# when printed, else the final energy): for a ΔSCF relaxation, the excited state at the ground
# state's structure, so (less the ground state's energy) the vertical absorption energy.
#   Called by: dscf_result(), step_energies()
def first_energy(text, which):
    for kind in (["0K", "final"] if which == "auto" else [which]):
        m = re.search(ENERGY_PATTERNS[kind], text)
        if m:
            return float(m.group(1))
    return None


# The 1D estimate of how strongly an excitation couples to the lattice, from a relaxed ΔSCF: the
# excited state's relaxation energy E_rel = E_abs - ZPL (E_abs: the vertical absorption, at the
# ground state's structure), the phonon energy of a harmonic well with that energy over the
# mass-weighted distance dQ, hw = HW_UNIT sqrt(2 E_rel) / dQ, the Huang-Rhys factor S = E_rel / hw
# and the Debye-Waller factor exp(-S) (the share of the emission in the ZPL). With the same
# harmonic vibrations in both states, this S can only be larger than the multi-mode one
# (Cauchy-Schwarz over the modes dQ projects onto), so exp(-S) is a lower bound.
# Returns {E_rel, hw, S, DW} (None where it can't be estimated).
#   Called by: dscf_result()
def coupling_1d(zpl, e_abs, dq):
    e_rel = None if zpl is None or e_abs is None else e_abs - zpl
    if e_rel is None or not dq or dq < 1e-3 or e_rel < 1e-5:
        return {"E_rel": None if e_rel is None else round(e_rel, 4), "hw": None, "S": None,
                "DW": None}
    hw = HW_UNIT * math.sqrt(2 * e_rel) / dq
    return {"E_rel": round(e_rel, 4), "hw": round(hw, 4), "S": round(e_rel / hw, 3),
            "DW": round(math.exp(-e_rel / hw), 4)}


# What an excitation's ΔSCF run (r, calc: run_batch's result and Calc) gave, against its ground
# state (energy e_gs, structure gs_atoms): ZPL (relaxed) or vertical energy, E_abs (its first
# energy), dQ and the 1D coupling (coupling_1d), whether the promoted electron stayed on the defect
# (its band's participation number in the excited state, at most fraction x nat), the relaxed
# excited structure (saved to structdir) and a verdict: pass, small (ZPL below emin), outside
# (outside window), ionises (the electron left the defect), low DW (below min_dw) or unknown.
#   Called by: step_excite(), step_zpl()
#   Calls:     coupling_1d(), delta_q(), first_energy(), participation(), read_pdos_weights(),
#              save_cell(), final_atoms()
def dscf_result(p, r, calc, ex, e_gs, gs_atoms, nat, relaxed, structdir, emin, window, min_dw):
    out = {"energy": None, "E_abs": None, "dq": None, "E_rel": None, "hw": None, "S": None,
           "DW": None, "P_to": None, "local": None, "structure": None, "verdict": "unknown",
           "why": ""}
    if r["status"] != "done" or r["energy"] is None or not r["scf_converged"]:
        out["why"] = ("excited state: " + (r["reason"] or "no converged energy")).splitlines()[0]
        return out
    if relaxed and not r["geom_converged"]:
        out["why"] = "excited state: the relaxation did not converge"
        return out
    text = calc.output()
    e1 = first_energy(text, p.s.energy)
    out["energy"] = round(r["energy"] - e_gs, 4)
    out["E_abs"] = round(e1 - e_gs, 4) if e1 is not None else None
    if relaxed:
        atoms = final_atoms(calc)
        if atoms is not None:
            out["dq"] = round(delta_q(gs_atoms, atoms), 4)
            out["structure"] = save_cell(p, structdir / f"{calc.name}.cell", atoms,
                                         f"{calc.name}: relaxed excited state (ΔSCF)")
        out.update(coupling_1d(out["energy"], out["E_abs"], out["dq"]))
    pw = read_pdos_weights(calc.dir / f"{calc.seed}.pdos_weights")
    if pw is not None:
        P = participation(pw)
        if ex["spin"] < len(P) and ex["to_band"] <= len(P[ex["spin"]]):
            out["P_to"] = round(float(P[ex["spin"]][ex["to_band"] - 1]), 2)
            out["local"] = out["P_to"] <= p.s.local_fraction * nat
    lo, hi = window or (None, None)
    e = out["energy"]
    if e < emin:
        out.update(verdict="small", why=f"{e:.3f} eV < excite_min {emin:g} eV")
    elif lo is not None and not lo <= e <= hi:
        out.update(verdict="outside", why=f"outside {lo:g}-{hi:g} eV")
    elif out["local"] is False:
        out.update(verdict="ionises", why=f"the promoted electron spread over {out['P_to']:.0f} "
                                          "atoms: it left the defect")
    elif min_dw and out["DW"] is not None and out["DW"] < min_dw:
        out.update(verdict="low DW", why=f"Debye-Waller {out['DW']:.3f} < {min_dw:g}")
    else:
        out["verdict"] = "pass"
    return out


# A C6 structure t as a structure of the electronic cell Pe (B4): as it is when it is already in
# that cell (its 'matrix'; without one, the screening cell's), else carried over (carry_over).
# Returns (Atoms, info: where its defect is, build_defect's).
#   Called by: step_excite(), step_relaxations()
#   Calls:     build_defect(), carry_over(), load_cell()
def into_cell(p, t, prim, cands, Pe):
    comps, label = cands[t["cand"]]["comps"], f"{t['cand']} ({cands[t['cand']]['name']})"
    src = np.array(t.get("matrix") or p.out("choose")["matrix"])
    atoms = load_cell(p, t["structure"])
    if np.array_equal(src, Pe):
        return atoms, build_defect(prim, Pe, comps, label)[1]
    return carry_over(prim, comps, src, atoms, Pe, label=label)


# C7: the screen of the excited states, in the electronic cell (B4) at the screening settings
# (Gamma, screen_cutoff), in three batches (C_screen/jobs/excite_relax, excite_gs, excite):
#  1. the pristine cell (its band edges), and every structure C6 kept, relaxed: the lowest that
#     relaxes is the ground state (C6's order may change); it is flagged if it slid more than
#     BASIN_TOL from where it started (same_basin). A band state (in_band) is 'pending': it can't
#     be judged here (a hybrid can). With excite_spin (2, a triplet; 'any' for every spin), a
#     ground state whose final spin is further than spin_tol from it goes no further ('spin').
#  2. a single point of each remaining ground state with the bands' weights on the atoms
#     (pdos_calculate_weights; its .check is kept): its defect levels are the bands localised on
#     the defect (defect_levels: a participation number of at most local_fraction of the atoms),
#     in the gap or up to excite_reach inside a band; in each spin channel, the highest occupied
#     one to the lowest empty one (transitions).
#  3. each excitation by CASTEP's simple ΔSCF from the ground state's .check, relaxed with
#     excite_relax : true: its ZPL, the vertical absorption, dQ and the 1D Huang-Rhys and
#     Debye-Waller estimates (coupling_1d), and whether the promoted electron stayed on the
#     defect (dscf_result).
# A charge state passes if one of its excitations does (dscf_result's verdicts; a failed run is
# 'unknown' and counts as a pass, with a warning). A defect goes on to D with all its charge
# states (their relaxed ground states here) if one passes, and every native defect (is_native)
# goes on anyway (its ground states, for E). With excite : false, every structure C6 kept as its
# lowest goes on to D as it is.
#   Called by: main() (step C7)
#   Calls:     Project.batch(), Project.host(), Project.production_param(), band_edges(),
#              carriers_text(), defect_levels(), dscf_job(), dscf_result(), excite_job(),
#              format_table(), host_supercell(), in_band(), into_cell(), is_native(), job_summary(),
#              load_cell(), participation(), qtag(), read_bands(), read_pdos_weights(),
#              same_basin(), screen_job(), screen_overrides(), start_spin(), transitions()
def step_excite(p):
    structures = p.out("rank")["structures"]
    if not p.s.excite:
        out = [dict(t) for t in structures if t["k"] == 0]
        return {"structures": out, "states": {}, "kept": sorted({t["cand"] for t in out}),
                "native": [], "_note": "excite is off: the lowest structure C6 kept of every "
                                       "charge state goes on to D"}
    d = p.phase_dir("C")
    cands, prim = p.out("generate")["candidates"], p.host()
    Pe = np.array(p.out("electronic")["matrix"])
    host = host_supercell(prim, Pe)
    nat = len(host)
    host_el = set(prim.get_chemical_symbols())
    jobs = [{"name": "host", "atoms": host, "kind": "sp", "kgrid": [1, 1, 1], "symmetry": True,
             "overrides": {k: v for k, v in screen_overrides(p).items()
                           if k != "geom_max_iter"}}]
    starts = {}
    for t in structures:
        name = f"{t['cand']}_{qtag(t['q'])}_k{t['k']}"
        atoms, info = into_cell(p, t, prim, cands, Pe)
        jobs.append(screen_job(p, name, atoms, info, t["q"], t["spin"]))
        starts[name] = (t, atoms, info)
    res = p.batch(f"{PHASE_DIRS['C']}/jobs/excite_relax", jobs, "excite", p.production_param())
    if not res["complete"]:
        return None
    hc = res["calcs"]["host"]
    if res["jobs"]["host"]["status"] != "done" or not (hc.dir / f"{hc.seed}.bands").exists():
        die(f"C7: the pristine electronic cell did not finish ({hc.dir}); the band edges come "
            "from it")
    vbm, cbm = band_edges(read_bands(hc.dir / f"{hc.seed}.bands"))
    states, infos = {}, {}
    for name, (t, atoms, info) in starts.items():
        s = job_summary(p, res, name, (vbm, cbm), d / "structures" / "excite_relax")
        moved = None
        if s["structure"]:
            moved = round(same_basin(prim, cands[t["cand"]]["comps"], Pe, atoms, Pe,
                                     load_cell(p, s["structure"]), name)[1], 3)
        states.setdefault((t["cand"], t["q"]), {"runs": []})["runs"].append(
            dict(s, job=name, k=t["k"], moved=moved, source=t.get("source")))
        infos[name] = info
    sp_jobs = {}
    for (cid, q), e in states.items():
        e.update(verdict="unknown", why="", transitions=[], levels=[], ground=None)
        ok = [r for r in e["runs"] if r["energy"] is not None and r["structure"]]
        if not ok:
            e["why"] = "no ground-state relaxation finished"
            continue
        g = min(ok, key=lambda r: r["energy"])
        e.update(ground=g["job"], structure=g["structure"], energy_relax=g["energy"],
                 spin=g["spin"], band=g["band"], moved=g["moved"], reordered=g["k"] != 0)
        if in_band(g):
            e.update(verdict="pending", why=f"a band state ({carriers_text(g)}): it needs a "
                                            "hybrid to judge")
            continue
        if p.s.excite_spin is not None and abs(abs(g["spin"] or 0.0) - p.s.excite_spin) \
                > p.s.spin_tol:
            e.update(verdict="spin", why=f"final spin {g['spin'] or 0:.2f}, not "
                                         f"{p.s.excite_spin:g}")
            continue
        atoms = load_cell(p, g["structure"])
        name = f"{cid}_{qtag(q)}_gs"
        sp_jobs[name] = (cid, q, excite_job(p, name, atoms, infos[g["job"]], q,
                                            start_spin(atoms, q, g["spin"]), "sp",
                                            {"pdos_calculate_weights": "true"}))
    ex_jobs, plan = [], {}
    if sp_jobs:
        res2 = p.batch(f"{PHASE_DIRS['C']}/jobs/excite_gs", [x[2] for x in sp_jobs.values()],
                       "excite", p.production_param(), keep_check=True)
        if not res2["complete"]:
            return None
        exdir = d / "jobs" / "excite"
        for name, (cid, q, job) in sp_jobs.items():
            e, r, calc = states[(cid, q)], res2["jobs"][name], res2["calcs"][name]
            bands = calc.dir / f"{calc.seed}.bands"
            pw = read_pdos_weights(calc.dir / f"{calc.seed}.pdos_weights")
            if r["status"] != "done" or r["energy"] is None or not r["scf_converged"]:
                e["why"] = ("ground state: " + (r["reason"] or "no converged energy")
                            ).splitlines()[0]
                continue
            if not bands.exists() or pw is None:
                e["why"] = "ground state: no .bands or .pdos_weights file"
                continue
            b = read_bands(bands)
            e["energy_gs"] = r["energy"]
            e["levels"] = defect_levels(b, participation(pw), nat, vbm, cbm, p.s.excite_reach,
                                        p.s.local_fraction)
            e["transitions"] = transitions(e["levels"])
            if not e["transitions"]:
                e.update(verdict="none", why="no occupied and empty defect level in one spin "
                                             "channel")
                continue
            for ex in e["transitions"]:
                xname = f"{cid}_{qtag(q)}_ex{ex['spin'] + 1}"
                ex_jobs.append(dscf_job(job, calc, xname, exdir, ex, b["nspin"],
                                        "geom" if p.s.excite_relax else "sp",
                                        p.production_param()))
                plan[xname] = (cid, q, ex, b["nspin"])
    if ex_jobs:
        res3 = p.batch(f"{PHASE_DIRS['C']}/jobs/excite", ex_jobs, "excite",
                       p.production_param())
        if not res3["complete"]:
            return None
        for xname, (cid, q, ex, _) in plan.items():
            e = states[(cid, q)]
            ex.update(job=xname, **dscf_result(
                p, res3["jobs"][xname], res3["calcs"][xname], ex, e["energy_gs"],
                load_cell(p, e["structure"]), nat, p.s.excite_relax,
                d / "structures" / "excite", p.s.excite_min, p.s.excite_window,
                p.s.excite_min_dw))
        for e in states.values():
            v = [ex["verdict"] for ex in e["transitions"] if "verdict" in ex]
            if v:
                e["verdict"] = ("pass" if "pass" in v else "unknown" if "unknown" in v else v[0])
                e["why"] = "" if e["verdict"] == "pass" else next(
                    ex["why"] for ex in e["transitions"] if ex.get("verdict") == e["verdict"])
    native = sorted({cid for cid, _ in states if is_native(cands[cid], host_el)})
    keep = sorted({cid for (cid, _), e in states.items()
                   if cid in native or e["verdict"] in ("pass", "unknown")})
    rows, out = [], []
    for (cid, q), e in sorted(states.items()):
        goes = ("yes (native)" if cid in native else "yes") if cid in keep else "no"
        if e["verdict"] == "unknown":
            log(f"WARNING: C7: {cid} {cands[cid]['name']} q {q:+d}: {e['why']} - kept")
        flags = ", ".join(x for x in (
            f"slid {e['moved']:.2f} A" if e.get("moved") and e["moved"] > BASIN_TOL else "",
            "reordered" if e.get("reordered") else "") if x)
        exs = e["transitions"] or [None]
        for i, ex in enumerate(exs):
            rows.append([cid if i == 0 else "", cands[cid]["name"] if i == 0 else "",
                         f"{q:+d}" if i == 0 else "",
                         (None if e.get("spin") is None else round(e["spin"], 2)) if i == 0
                         else None, flags if i == 0 else "",
                         f"{ex['spin'] + 1}: {ex['from_band']}->{ex['to_band']}" if ex else "",
                         ex["step"] if ex else None,
                         (f"{ex['P_from']:.0f}/{ex['P_to']:.0f}" + (
                             "" if ex["where_to"] == "gap" else " (in band)")) if ex else "",
                         ex.get("energy") if ex else None, ex.get("E_abs") if ex else None,
                         ex.get("dq") if ex else None, ex.get("S") if ex else None,
                         ex.get("DW") if ex else None,
                         ((ex.get("verdict") or "") + (f": {ex['why']}" if ex.get("why") else ""))
                         if ex else e["verdict"] + (f": {e['why']}" if e["why"] else ""),
                         goes if i == 0 else ""])
        if cid in keep and e.get("structure"):
            out.append({"cand": cid, "q": q, "k": 0, "structure": e["structure"],
                        "matrix": Pe.tolist(), "spin": e["spin"], "source": "C7",
                        "excite": [{k: ex[k] for k in ("spin", "from_band", "to_band")}
                                   for ex in e["transitions"] if ex.get("verdict") == "pass"]})
        elif cid in keep:                         # its relaxation failed: C6's structure
            t0 = next(t for t in structures if t["cand"] == cid and t["q"] == q and t["k"] == 0)
            out.append(dict(t0, excite=[]))
    what = "ZPL" if p.s.excite_relax else "vertical E"
    table = format_table(["id", "defect", "q", "spin", "flags", "spin: bands", "KS step/eV",
                          "P from/to", f"{what}/eV", "E_abs/eV", "dQ/amu^0.5 A", "S (1D)",
                          "DW (1D)", "verdict", "on to D"], rows,
                         f"Excited states in the electronic cell ({p.out('electronic')['tag']}) at "
                         "the screening settings: defect levels are bands on at most "
                         f"{p.s.local_fraction:g} of the {nat} atoms (P: the atoms each spreads "
                         "over); S and DW are 1D estimates (DW a lower bound)"
                         + (f"; window {p.s.excite_window[0]:g}-{p.s.excite_window[1]:g} eV"
                            if p.s.excite_window else ""))
    js = {cid: {} for cid, _ in states}
    for (cid, q), e in states.items():
        js[cid][str(q)] = {k: v for k, v in e.items() if k != "runs"}
        js[cid][str(q)]["runs"] = [{k: v for k, v in r.items() if k != "levels"}
                                   for r in e["runs"]]
    npass = sum(e["verdict"] == "pass" for e in states.values())
    return {"structures": out, "states": js, "kept": keep, "native": native,
            "pending": sorted(f"{cid} q {q:+d}" for (cid, q), e in states.items()
                              if e["verdict"] == "pending"),
            "relaxed": p.s.excite_relax, "edges": {"vbm": vbm, "cbm": cbm}, "_table": table,
            "_note": f"{npass} charge states pass; {len(keep)} defects go on to D "
                     f"({len(native)} native)"}


# --------------------------------------------------------------------------
# C8: the finite-size check (optional)
# --------------------------------------------------------------------------
# The line E = E_inf + b/N (least squares) through some sizes' E (dicts with N and E). Returns
# (E_inf, b, the largest distance of a point from the line), or for one size (its E, None, 0).
#   Called by: step_validate()
def tail_fit(points):
    if len(points) == 1:
        return points[0]["E"], None, 0.0
    x = np.array([1.0 / q["N"] for q in points])
    y = np.array([q["E"] for q in points])
    A = np.vstack([np.ones_like(x), x]).T
    (e_inf, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(e_inf), float(b), float(np.abs(A @ [e_inf, b] - y).max())


# C8 (optional, validate : true): whether the screening cell, with its corrections, gives the
# formation energies and transition levels of much bigger cells, so that a size series can be
# left out in later projects. For the lowest structure C6 kept of each defect (those in
# validate_defects, or every one C6 kept) in each charge state that isn't a band state: a
# relaxation at the production settings (production_job, but without freeze_radius, so the far
# field relaxes too) in the screening cell and in the next validate_sizes sizes of the ladder up
# that fit it (C6's structure carried over, carry_over), with the pristine cell's single point in
# each size; all with calculate_stress. In each size, E = E_D - E_host + E_corr (the charge
# correction, in that cell's shape); the reference E_inf is the line E_inf + b/N through the sizes
# above the screening cell (tail_fit; N, the host cell's atoms, as in B2), and b is the tail the
# sizes really show. Three estimates of E_inf from the screening cell alone are compared with it:
# E as it is; E - c/N with B2's c (MACE: one c per arrangement, blind to the charge); and E less
# the energy of the stress the fixed cell holds, 1/2 V s:S:s (elastic_energy; s, the defect cell's
# stress less the host's in the same size, so the Pulay stress of the basis cancels; S, the
# compliance from elastic_constants), each charge state's own. The transition levels between the
# charge states checked are compared the same way. A defect passes with an estimate when every
# error is within validate_tol. The jobs run in C_screen/jobs/validate with the runner settings of
# runner_production (or runner_validate); the relaxed structures go to C_screen/structures/validate.
# C6's structure is carried from the cell it is in (its 'matrix': the screening cell, or the
# electronic cell, distort_cell's). With validate_gamma (true), the same relaxed
# structures also get Gamma-only single points (validate_gamma), to show what Gamma-only sampling
# would change in each size. Nothing later uses the results: they are a test of the method.
#   Called by: main() (step C8)
#   Calls:     Project.batch(), Project.host(), Project.kspacing(), Project.production_param(),
#              build_defect(), carry_over(), compliance(), correction(), elastic_energy(), fits(),
#              format_table(), host_supercell(), in_band(), load_cell(), production_job(),
#              production_summary(), qtag(), read_stress(), tail_fit(), validate_gamma()
def step_validate(p):
    if not p.s.validate:
        return {"skipped": True, "_note": "validate : false"}
    d = p.phase_dir("C")
    cands, prim = p.out("generate")["candidates"], p.host()
    choose = p.out("choose")
    ladder, idx, P0 = choose["ladder"], choose["index"], np.array(choose["matrix"])
    eps = p.out("dielectric").get("eps0")
    S = compliance(p.s.elastic_constants)
    if S is None:
        log("C8: no %block elastic_constants, so the stress estimate is left out")
    mace_c = {cid: (x.get("fit") or {}).get("c")
              for cid, x in p.out("scan").get("candidates", {}).items()}
    ground, skipped, sizes = {}, [], {}
    for t in p.out("rank")["structures"]:
        if t["k"] != 0 or (p.s.validate_defects and t["cand"] not in p.s.validate_defects):
            continue
        if in_band(t):
            skipped.append(f"{t['cand']} q {t['q']:+d}")
        else:
            ground.setdefault(t["cand"], {})[t["q"]] = t
    for x in p.s.validate_defects:
        if x not in ground:
            log(f"WARNING: C8: {x} is in validate_defects but C6 kept no bound structure of it")
    if skipped:
        log("C8: left out (a band state, so there is no defect level to check): "
            + ", ".join(skipped))
    for cid in sorted(ground):
        up = [i for i in range(idx + 1, len(ladder)) if fits(cands[cid], ladder[i]["L"])]
        if not up:
            log(f"WARNING: C8: {cid} not checked: no size of the ladder above the screening cell "
                "fits it (longer scan_lengths add sizes)")
            del ground[cid]
        else:
            sizes[cid] = [idx] + up[:p.s.validate_sizes]
    if not ground:
        log("WARNING: C8: nothing to check (see above)")
        return {"defects": {}, "runs": {}, "hosts": {}, "skipped": skipped,
                "tol": p.s.validate_tol, "_note": "nothing to check"}
    extra = {"calculate_stress": "true"}
    jobs, meta, infos = [], {}, {}
    for cid, states in sorted(ground.items()):
        comps, label = cands[cid]["comps"], f"{cid} ({cands[cid]['name']})"
        for i in sizes[cid]:
            Q = np.array(ladder[i]["matrix"])
            for q, t in sorted(states.items()):
                name = f"{cid}_{qtag(q)}_{ladder[i]['tag']}"
                start = load_cell(p, t["structure"])          # C6's, in its own cell
                src = np.array(t.get("matrix") or P0)
                atoms, info = ((start, build_defect(prim, Q, comps, label)[1])
                               if np.array_equal(src, Q) else
                               carry_over(prim, comps, src, start, Q, label=label))
                job, _ = production_job(p, name, atoms, info, q, t["spin"])
                job.pop("cell_extra", None)          # no freeze_radius: the far field relaxes too
                job["overrides"].update(extra)
                jobs.append(job)
                meta[name] = (cid, q, i)
                infos[name] = info
    used = sorted({i for v in sizes.values() for i in v})
    for i in used:
        jobs.append({"name": f"host_{ladder[i]['tag']}",
                     "atoms": host_supercell(prim, np.array(ladder[i]["matrix"])), "kind": "sp",
                     "symmetry": True, "overrides": dict(p.s.production_overrides, **extra)})
    res = p.batch(f"{PHASE_DIRS['C']}/jobs/validate", jobs, "validate", p.production_param(),
                  spacing=p.kspacing())
    if not res["complete"]:
        return None
    hosts = {}
    for i in used:
        name = f"host_{ladder[i]['tag']}"
        r = res["jobs"][name]
        if r["status"] != "done" or r["energy"] is None:
            die(f"C8: the pristine {ladder[i]['tag']} single point did not finish ({r['dir']})")
        s, pr = read_stress(res["calcs"][name].output())
        hosts[i] = {"energy": r["energy"], "stress": None if s is None else s.tolist(),
                    "pressure": pr}
    runs, nostress = {}, []
    for name, (cid, q, i) in meta.items():
        r, _ = production_summary(p, res, name, d / "structures" / "validate")
        s, pr = read_stress(res["calcs"][name].output()) if r["energy"] is not None else (None,
                                                                                           None)
        if r["energy"] is not None and s is None:
            nostress.append(name)
        runs[name] = dict(r, cand=cid, q=q, cell=ladder[i]["tag"],
                          stress=None if s is None else s.tolist(), pressure=pr)
    if nostress:
        log(f"WARNING: C8: no stress tensor in {len(nostress)} runs (e.g. {nostress[0]}): does "
            "this CASTEP honour calculate_stress, or print it in a unit other than GPa, MPa, Pa, "
            "bar or atm (pressure_unit)? The stress estimate is left out there")
    gamma = validate_gamma(p, prim, ladder, used, meta, infos, runs, hosts) \
        if p.s.validate_gamma else {"rows": [], "worst": {}}
    if gamma is None:
        return None
    tol, kinds = p.s.validate_tol, ("as is", "MACE c/N", "stress")
    out, rows, errs_rows, verdict_rows = {}, [], [], []
    for cid, states in sorted(ground.items()):
        c_m = mace_c.get(cid)
        per_q, errors = {}, {}
        for q in sorted(states):
            pts = []
            for i in sizes[cid]:
                r = runs[f"{cid}_{qtag(q)}_{ladder[i]['tag']}"]
                if r["energy"] is None:
                    continue
                cell = np.array(ladder[i]["matrix"]) @ np.asarray(prim.cell, float)
                vol = abs(float(np.linalg.det(cell)))
                el = None
                if S is not None and r["stress"] is not None and hosts[i]["stress"] is not None:
                    el = elastic_energy(np.array(r["stress"]) - np.array(hosts[i]["stress"]), S,
                                        vol)
                dp = (r["pressure"] - hosts[i]["pressure"]
                      if r["pressure"] is not None and hosts[i]["pressure"] is not None else None)
                pts.append({"i": i, "tag": ladder[i]["tag"], "N": ladder[i]["natoms"],
                            "E": r["energy"] - hosts[i]["energy"]
                            + correction(q, cell, eps, p.s.charge_correction),
                            "elastic": el, "dp": dp})
            here = next((x for x in pts if x["i"] == idx), None)
            above = [x for x in pts if x["i"] > idx]
            if here is None or not above:
                why = ("its screening-cell run failed" if here is None else
                       "no bigger size to compare with (the runs failed, or none fits)")
                log(f"WARNING: C8: {cid} q {q:+d} not checked: {why}")
                per_q[q] = {"checked": False, "why": why}
                continue
            e_inf, b, worst = tail_fit(above)
            est = {"as is": here["E"],
                   "MACE c/N": here["E"] - c_m / here["N"] if c_m is not None else None,
                   "stress": here["E"] - here["elastic"] if here["elastic"] is not None
                   else None}
            errors[q] = {k: (v - e_inf if v is not None else None) for k, v in est.items()}
            per_q[q] = {"checked": True, "E_inf": round(e_inf, 5),
                        "b": None if b is None else round(b, 3), "fit_worst": round(worst, 5),
                        "c_stress": (None if here["elastic"] is None
                                     else round(here["elastic"] * here["N"], 3)),
                        "errors": {k: None if v is None else round(v, 4)
                                   for k, v in errors[q].items()},
                        "sizes": [{"cell": x["tag"], "N": x["N"], "E": round(x["E"], 5),
                                   "elastic": None if x["elastic"] is None
                                   else round(x["elastic"], 4), "dp": x["dp"]} for x in pts]}
            for x in pts:
                rows.append([cid, f"{q:+d}", x["tag"] + (" (screening)" if x["i"] == idx else ""),
                             x["N"], x["E"] - e_inf, x["elastic"],
                             c_m / x["N"] if c_m is not None else None, x["dp"]])
            errs_rows.append([cid, f"{q:+d}", len(above), b, per_q[q]["c_stress"], c_m, worst]
                             + [errors[q][k] for k in kinds])
        levels = []
        qs = sorted(errors)
        for a_, b_ in zip(qs, qs[1:]):
            lv = {k: ((errors[b_][k] - errors[a_][k]) / (a_ - b_)
                      if errors[a_][k] is not None and errors[b_][k] is not None else None)
                  for k in kinds}
            levels.append({"level": f"{b_:+d}/{a_:+d}",
                           **{k: None if v is None else round(v, 4) for k, v in lv.items()}})
            errs_rows.append([cid, f"({b_:+d}/{a_:+d})", "", "", "", "", ""]
                             + [lv[k] for k in kinds])
        verdict = {}
        for k in kinds:
            vals = [e[k] for e in errors.values()] + [x[k] for x in levels]
            if not errors or any(v is None for v in vals):
                verdict[k] = None
            else:
                verdict[k] = {"worst": round(max(abs(v) for v in vals), 4),
                              "pass": max(abs(v) for v in vals) <= tol}
        out[cid] = {"states": per_q, "levels": levels, "verdict": verdict,
                    "sizes": [ladder[i]["tag"] for i in sizes[cid]], "mace_c": c_m}
        verdict_rows.append([cid, cands[cid]["name"]]
                            + [("pass" if v["pass"] else "FAIL") + f" ({v['worst']:.3f})"
                               if v else "-" for v in (verdict[k] for k in kinds)])
    table = (format_table(["id", "q", "size", "atoms", "E - E_inf/eV", "stress E/eV",
                           "MACE c/N/eV", "dp/GPa"], rows,
                          "Each size at the production settings: E = E_D - E_host + E_corr "
                          f"({p.s.charge_correction}), against the line E_inf + b/N through the "
                          "sizes above the screening cell; stress E, the energy of the stress "
                          "the cell holds (1/2 V s:S:s); MACE c/N, B2's tail; dp, the defect "
                          "cell's pressure less the host's (positive: it pushes out)")
             + "\n" + format_table(["id", "q (level)", "fitted", "b/eV", "c stress/eV",
                                    "c MACE/eV", "fit worst/eV"]
                                   + [f"error {k}/eV" for k in kinds], errs_rows,
                                   "The screening cell's estimates against E_inf: as is, minus "
                                   "MACE's c/N, minus the stress energy; b is the tail the sizes "
                                   "show (E_inf + b/N), c stress the stress energy times N there; "
                                   "a level's error is that of the transition level")
             + "\n" + format_table(["id", "defect"] + list(kinds), verdict_rows,
                                   f"Verdict with each estimate: every error within validate_tol "
                                   f"{tol:g} eV (in brackets, the largest)"))
    counts = {k: sum(1 for x in out.values() if x["verdict"][k] and x["verdict"][k]["pass"])
              for k in kinds}
    if gamma["rows"]:
        table += "\n" + format_table(
            ["id", "q", "size", "atoms", "E (k-points)/eV", "E (Gamma)/eV", "Gamma - k/eV"],
            gamma["rows"], "Gamma-only against the production k-points, on the same relaxed "
            "structures (E = E_D - E_host; the charge correction is the same in both)")
    return {"defects": out, "runs": runs,
            "hosts": {ladder[i]["tag"]: h for i, h in hosts.items()}, "skipped": skipped,
            "tol": tol, "gamma": gamma, "_table": table,
            "_note": f"{len(out)} defects checked; pass: " + ", ".join(
                f"{counts[k]} {k}" for k in kinds) + f" (validate_tol {tol:g} eV)"
                     + (f"; Gamma-only differs by at most {max(gamma['worst'].values()):.3f} eV"
                        if gamma["worst"] else "")}


# C8's check of Gamma-only sampling (validate_gamma : true): a single point at the production
# cut-off with only the Gamma point, of every structure C8 relaxed at the production k-points and
# of the pristine cell in each size used (C_screen/jobs/validate_gamma). E = E_D - E_host with
# each, and the difference: how far Gamma-only would move a formation energy (the charge correction
# is the same in both, and cancels). Returns {'rows' (for the table), 'worst': {size tag: largest
# |difference|}, 'runs': {name: difference}}, or None while the jobs run.
#   Called by: step_validate()
#   Calls:     Project.batch(), Project.production_param(), host_supercell(), load_cell(),
#              production_job(), start_spin()
def validate_gamma(p, prim, ladder, used, meta, infos, runs, hosts):
    jobs, plan = [], {}
    for i in used:
        jobs.append({"name": f"host_{ladder[i]['tag']}_G", "kind": "sp", "kgrid": [1, 1, 1],
                     "atoms": host_supercell(prim, np.array(ladder[i]["matrix"])),
                     "symmetry": True, "overrides": dict(p.s.production_overrides)})
    for name, (cid, q, i) in meta.items():
        r = runs[name]
        if r["energy"] is None or not r["final"]:
            continue
        atoms = load_cell(p, r["final"])
        job, _ = production_job(p, f"{name}_G", atoms, infos[name], q,
                                start_spin(atoms, q, r["spin_final"]))
        job.pop("cell_extra", None)
        job.update(kind="sp", kgrid=[1, 1, 1])
        jobs.append(job)
        plan[f"{name}_G"] = name
    res = p.batch(f"{PHASE_DIRS['C']}/jobs/validate_gamma", jobs, "validate",
                  p.production_param())
    if not res["complete"]:
        return None
    hg = {}
    for i in used:
        r = res["jobs"][f"host_{ladder[i]['tag']}_G"]
        hg[i] = r["energy"] if r["status"] == "done" else None
    rows, worst, diffs = [], {}, {}
    for gname, name in sorted(plan.items()):
        cid, q, i = meta[name]
        rg = res["jobs"][gname]
        if rg["energy"] is None or hg[i] is None or not rg["scf_converged"]:
            continue
        ek = runs[name]["energy"] - hosts[i]["energy"]
        eg = rg["energy"] - hg[i]
        diffs[name] = round(eg - ek, 5)
        tag = ladder[i]["tag"]
        worst[tag] = max(worst.get(tag, 0.0), abs(eg - ek))
        rows.append([cid, f"{q:+d}", tag, ladder[i]["natoms"], ek, eg, eg - ek])
    return {"rows": rows, "worst": {k: round(v, 5) for k, v in worst.items()}, "runs": diffs}


SCREEN_KEYS = ("screen_cutoff", "screen_max_steps", "screen_overrides", "level_margin", "energy")
MODEL_KEYS = ("model_command", "model_device", "model_epochs", "model_r_max", "model_channels",
              "model_max_L", "model_batch", "model_train_args", "mace_device")


# ==========================================================================================
# PHASE D: PRODUCTION GROUND AND EXCITED STATES (flowchart D1-D2)
# ==========================================================================================
# The %block ionic_constraints that holds still every atom further than radius (Ang) from all of
# the defect's components (points: their positions; three constraints per atom: x, y and z).
# CASTEP numbers the atoms of each species in the order of the .cell, which keeps the order
# within each species (written_order).
#   Called by: production_job()
#   Calls:     get_distances()
def freeze_block(atoms, points, radius):
    _, dist = get_distances(np.atleast_2d(points), atoms.positions, cell=atoms.cell, pbc=True)
    dist = dist.min(axis=0)
    syms, lines, k = atoms.get_chemical_symbols(), [], 0
    for i, s in enumerate(syms):
        if dist[i] <= radius:
            continue
        n = syms[:i + 1].count(s)
        for v in ("1 0 0", "0 1 0", "0 0 1"):
            k += 1
            lines.append(f"  {k:6d}  {s:<3s} {n:6d}   {v}")
    if not lines:
        return None, 0
    return ("%block ionic_constraints\n" + "\n".join(lines) + "\n%endblock ionic_constraints",
            k // 3)


# A production relaxation of one structure in charge state q, starting with spin S
# (spin_polarized : true with spin S when S is not 0; a spin-0 state runs without spin
# polarisation unless %block production_overrides sets spin_polarized, and with SPIN=S on the
# anchor atom when there is one): the production .param with %block production_overrides, the
# production k-points, fixed cell, no symmetry; with freeze_radius, the far atoms fixed
# (freeze_block). info: where the defect is in this structure's cell (build_defect's). Returns
# (job, how many atoms are fixed).
#   Called by: step_ccd(), step_lineshape(), step_relaxations(), step_validate(), step_zpl(),
#              validate_gamma()
#   Calls:     freeze_block()
def production_job(p, name, atoms, info, q, S):
    a = atoms.copy()
    mags = np.zeros(len(a))
    if S and info.get("anchor_index") is not None:
        mags[info["anchor_index"]] = S
    a.set_initial_magnetic_moments(mags)
    o = dict(p.s.production_overrides, charge=q)
    if S:
        o.update(spin_polarized="true", spin=S)
    else:                                   # (production_overrides can ask for spin_polarized)
        o.setdefault("spin_polarized", "false")
        o["spin"] = None
    job = {"name": name, "atoms": a, "kind": "geom", "symmetry": False, "overrides": o}
    fixed = 0
    if p.s.freeze_radius:
        block, fixed = freeze_block(a, info.get("points") or [info["centre"]], p.s.freeze_radius)
        if block:
            job["cell_extra"] = block
    return job, fixed


# What a finished production relaxation gave (from run_batch's results): its energy (None unless
# it finished with a converged last SCF), final spin, status and why, geometry steps, restarts,
# how many atoms were fixed, and its final structure, saved as <structdir>/<name>.cell. The
# structure is also returned (None if there isn't one), for carrying on to another size.
#   Called by: step_relaxations(), step_validate()
#   Calls:     save_cell(), final_atoms()
def production_summary(p, res, name, structdir, frozen=0):
    r = res["jobs"][name]
    e = r["energy"] if r["scf_converged"] and r["status"] == "done" else None
    atoms = final_atoms(res["calcs"][name]) if e is not None else None
    if e is not None and atoms is None:
        log(f"WARNING: could not read the relaxed structure of {name}")
    path = (save_cell(p, structdir / f"{name}.cell", atoms,
                      f"{name}: relaxed at the production settings") if atoms is not None
            else None)
    return {"job": name, "energy": e, "spin_final": r["spin"], "status": r["status"],
            "reason": r["reason"], "steps": r["geom_steps"], "restarts": r["restarts"],
            "final": path, "frozen": frozen}, atoms


# D1: the production relaxations, in the electronic cell (B4): every charge state of each defect
# phase C passes on (C7's relaxed ground states there; with excite : false, the lowest structure C6
# kept of each charge state, carried in: into_cell), relaxed at the production settings
# (production_job: A1's cut-off and k-point spacing, %block production_overrides, freeze_radius),
# with the pristine electronic cell's single point, whose band edges D2 measures levels against
# (D_production/jobs/relaxations; the relaxed structures in D_production/structures/relaxations).
# Runner group 'production'. E takes its energies from here, D2 its ground states.
#   Called by: main() (step D1)
#   Calls:     Project.batch(), Project.host(), Project.kspacing(), Project.production_param(),
#              band_edges(), format_table(), host_supercell(), into_cell(), production_job(),
#              production_summary(), qtag(), read_bands(), start_spin()
def step_relaxations(p):
    d = p.phase_dir("D")
    todo = p.out("excite")["structures"]
    if not todo:
        die("phase C passed no structures on, so there is nothing to relax")
    cands, prim = p.out("generate")["candidates"], p.host()
    el = p.out("electronic")
    Pe = np.array(el["matrix"])
    jobs = [{"name": "host", "atoms": host_supercell(prim, Pe), "kind": "sp", "symmetry": True,
             "overrides": dict(p.s.production_overrides)}]
    meta, frozen = {}, {}
    for t in todo:
        name = f"{t['cand']}_{qtag(t['q'])}"
        atoms, info = into_cell(p, t, prim, cands, Pe)
        job, frozen[name] = production_job(p, name, atoms, info, t["q"],
                                           start_spin(atoms, t["q"], t["spin"]))
        jobs.append(job)
        meta[name] = t
    if any(frozen.values()):
        log(f"freeze_radius {p.s.freeze_radius:g} A holds {min(frozen.values())}-"
            f"{max(frozen.values())} atoms fixed in each relaxation (untested idea)")
    res = p.batch(f"{PHASE_DIRS['D']}/jobs/relaxations", jobs, "relaxations",
                  p.production_param(), spacing=p.kspacing())
    if not res["complete"]:
        return None
    hr, hc = res["jobs"]["host"], res["calcs"]["host"]
    if hr["status"] != "done" or hr["energy"] is None:
        die(f"D1: the pristine electronic cell's single point did not finish ({hr['dir']}): the "
            "formation energies need it")
    edges = None
    if (hc.dir / f"{hc.seed}.bands").exists():
        edges = band_edges(read_bands(hc.dir / f"{hc.seed}.bands"))
    runs, rows = {}, []
    for name, t in meta.items():
        r, _ = production_summary(p, res, name, d / "structures" / "relaxations", frozen[name])
        runs[name] = dict(r, cand=t["cand"], q=t["q"], excite=t.get("excite") or [])
        rows.append([t["cand"], cands[t["cand"]]["name"], f"{t['q']:+d}",
                     None if r["energy"] is None else r["energy"] - hr["energy"],
                     r["spin_final"], r["steps"], r["status"]
                     + (f": {r['reason'].splitlines()[0]}" if r["reason"] else "")])
    bad = sum(r["energy"] is None for r in runs.values())
    table = format_table(["id", "defect", "q", "E_D - E_host/eV", "spin", "steps", "status"],
                         rows, f"Production relaxations in the electronic cell ({el['tag']}, "
                               f"{el['natoms']} atoms)")
    return {"host_energy": hr["energy"], "edges": None if edges is None else
            {"vbm": edges[0], "cbm": edges[1]},
            "cell": {k: el[k] for k in ("matrix", "tag", "natoms", "L")}, "runs": runs,
            "_table": table, "_note": f"{len(runs)} relaxations in {el['tag']}"
                                      + (f", {bad} without a usable energy" if bad else "")}


# Whether an excitation (ex, from a ground state's first k-point) keeps its band numbers at every
# k-point of a production run, so CASTEP's constraints (a band number and spin, the same at every
# k-point) pick the same two orbitals everywhere: at each k-point the same excitation in the same
# spin channel must come out (defect_levels and transitions there), and each of its two bands must
# spread by less than band_spread_tol across the k-points. b: the .bands; pw: the .pdos_weights.
# Returns (ok, why not).
#   Called by: step_zpl()
#   Calls:     defect_levels(), participation(), transitions()
def zpl_bands_ok(p, b, pw, nat, vbm, cbm, ex):
    nk = len(b["kpts"])
    if pw is None or len(pw["weights"]) != nk:
        return False, "no band weights at every k-point"
    for k in range(nk):
        lv = defect_levels(b, participation(pw, k), nat, vbm, cbm, p.s.excite_reach,
                           p.s.local_fraction, k)
        tr = [x for x in transitions(lv) if x["spin"] == ex["spin"]]
        if not tr or (tr[0]["from_band"], tr[0]["to_band"]) != (ex["from_band"], ex["to_band"]):
            return False, f"the excitation's bands change at k-point {k + 1}"
    for band in (ex["from_band"], ex["to_band"]):
        e = b["eig"][ex["spin"], :, band - 1]
        if np.nanmax(e) - np.nanmin(e) > p.s.band_spread_tol:
            return False, (f"band {band} spreads by {np.nanmax(e) - np.nanmin(e):.3f} eV across "
                           "the k-points")
    return True, ""


# D2: the production ΔSCF of every excitation C7 passed, in D1's relaxed ground state:
#  1. a single point at the production settings with the bands' weights (pdos_calculate_weights;
#     its .check kept; metals_method : dm and %block excite_overrides, dscf_settings), against
#     D1's pristine band edges: the excitation is found again in the same spin channel
#     (defect_levels, transitions), and the band-number check (zpl_bands_ok);
#  2. where that fails, a Gamma-only single point at the production cut-off instead (the
#     excitation found again there);
#  3. the ΔSCF (dscf_job) from whichever single point, relaxed with excite_relax : true: ZPL, the
#     vertical absorption, dQ, the 1D Huang-Rhys and Debye-Waller estimates and whether the
#     promoted electron stayed on the defect (dscf_result; its verdict uses C7's thresholds).
# The relaxed excited structures go to D_production/structures/zpl, for F.
#   Called by: main() (step D2)
#   Calls:     Project.batch(), Project.host(), Project.kspacing(), Project.production_param(),
#              build_defect(), defect_levels(), dscf_job(), dscf_result(), dscf_settings(),
#              format_table(), host_supercell(), load_cell(), participation(), production_job(),
#              qtag(), read_bands(), read_pdos_weights(), start_spin(), transitions(),
#              zpl_bands_ok()
def step_zpl(p):
    d = p.phase_dir("D")
    e1 = p.out("relaxations")
    todo = [r for r in e1["runs"].values() if r["excite"] and r["energy"] is not None
            and r["final"]]
    if not todo:
        return {"states": {}, "_note": "no excitation passed C7, so there is no ΔSCF to run"}
    if not e1["edges"]:
        die("D2: D1's pristine single point wrote no .bands, so the defect levels can't be found")
    cands, prim = p.out("generate")["candidates"], p.host()
    Pe = np.array(e1["cell"]["matrix"])
    nat = len(host_supercell(prim, Pe))
    vbm, cbm = e1["edges"]["vbm"], e1["edges"]["cbm"]
    infos = {}

    # A single point of a relaxed ground state with the bands' weights (Gamma-only if gamma).
    def gs_job(r, gamma):
        atoms = load_cell(p, r["final"])
        name = f"{r['cand']}_{qtag(r['q'])}_gs" + ("_G" if gamma else "")
        info = build_defect(prim, Pe, cands[r["cand"]]["comps"], r["cand"])[1]
        job, _ = production_job(p, name, atoms, info, r["q"],
                                start_spin(atoms, r["q"], r["spin_final"]))
        job.pop("cell_extra", None)
        job["kind"] = "sp"
        job["overrides"]["pdos_calculate_weights"] = "true"
        dscf_settings(p, job)
        if gamma:
            job["kgrid"] = [1, 1, 1]
        infos[name] = r
        return job

    # The excitations of a ground-state single point in the spin channels C7 passed, at its first
    # k-point: [(ex, nspin)], with the .bands and .pdos_weights read (b, pw), or why there are none.
    def found(r, calc):
        bands = calc.dir / f"{calc.seed}.bands"
        pw = read_pdos_weights(calc.dir / f"{calc.seed}.pdos_weights")
        if not bands.exists() or pw is None:
            return [], None, None, "no .bands or .pdos_weights file"
        b = read_bands(bands)
        lv = defect_levels(b, participation(pw), nat, vbm, cbm, p.s.excite_reach,
                           p.s.local_fraction)
        want = {x["spin"] for x in r["excite"]}
        exs = [x for x in transitions(lv) if x["spin"] in want]
        return exs, b, pw, "" if exs else "the excitation isn't found at the production settings"

    first = [gs_job(r, False) for r in todo]
    jobs1 = {j["name"]: (r, j) for r, j in zip(todo, first)}
    res1 = p.batch(f"{PHASE_DIRS['D']}/jobs/zpl_gs", first, "zpl", p.production_param(),
                   spacing=p.kspacing(), keep_check=True)
    if not res1["complete"]:
        return None
    states, gamma_todo, plan = {}, [], []
    for name, (r, job) in jobs1.items():
        st = states.setdefault((r["cand"], r["q"]), {"cand": r["cand"], "q": r["q"],
                                                      "excitations": [], "why": ""})
        rr, calc = res1["jobs"][name], res1["calcs"][name]
        if rr["status"] != "done" or rr["energy"] is None or not rr["scf_converged"]:
            st["why"] = ("ground state: " + (rr["reason"] or "no converged energy")
                         ).splitlines()[0]
            continue
        exs, b, pw, why = found(r, calc)
        if not exs:
            st["why"] = why
            continue
        fails = [w for ok, w in (zpl_bands_ok(p, b, pw, nat, vbm, cbm, ex) for ex in exs)
                 if not ok]
        if fails:
            st["why_gamma"] = fails[0]
            gamma_todo.append(r)
        else:
            st.update(sampling="k-points", gs_energy=rr["energy"])
            plan += [(st, name, calc, ex, b["nspin"], job) for ex in exs]
    if gamma_todo:
        res2 = p.batch(f"{PHASE_DIRS['D']}/jobs/zpl_gs_gamma",
                       [gs_job(r, True) for r in gamma_todo], "zpl", p.production_param(),
                       keep_check=True)
        if not res2["complete"]:
            return None
        for r in gamma_todo:
            name = f"{r['cand']}_{qtag(r['q'])}_gs_G"
            st = states[(r["cand"], r["q"])]
            rr, calc = res2["jobs"][name], res2["calcs"][name]
            if rr["status"] != "done" or rr["energy"] is None or not rr["scf_converged"]:
                st["why"] = ("Gamma ground state: " + (rr["reason"] or "no converged energy")
                             ).splitlines()[0]
                continue
            exs, b, pw, why = found(r, calc)
            if not exs:
                st["why"] = "Gamma: " + why
                continue
            st.update(sampling="Gamma", gs_energy=rr["energy"])
            plan += [(st, name, calc, ex, b["nspin"], gs_job(r, True)) for ex in exs]
    if plan:
        exdir = d / "jobs" / "zpl"
        jobs3, keys = [], []
        for st, gname, calc, ex, nspin, gjob in plan:
            xname = f"{st['cand']}_{qtag(st['q'])}_ex{ex['spin'] + 1}"
            jobs3.append(dscf_job(gjob, calc, xname, exdir, ex, nspin,
                                  "geom" if p.s.excite_relax else "sp", p.production_param()))
            keys.append((st, xname, ex, gname))
        res3 = p.batch(f"{PHASE_DIRS['D']}/jobs/zpl", jobs3, "zpl", p.production_param(),
                       spacing=p.kspacing())
        if not res3["complete"]:
            return None
        for st, xname, ex, gname in keys:
            r = infos[gname]
            ex = dict(ex, job=xname, gs_job=gname, **dscf_result(
                p, res3["jobs"][xname], res3["calcs"][xname], ex, st["gs_energy"],
                load_cell(p, r["final"]), nat, p.s.excite_relax, d / "structures" / "zpl",
                p.s.excite_min, p.s.excite_window, p.s.excite_min_dw))
            st["excitations"].append(ex)
    rows = []
    for (cid, q), st in sorted(states.items()):
        for i, ex in enumerate(st["excitations"] or [None]):
            rows.append([cid if i == 0 else "", cands[cid]["name"] if i == 0 else "",
                         f"{q:+d}" if i == 0 else "", st.get("sampling", "") if i == 0 else "",
                         f"{ex['spin'] + 1}: {ex['from_band']}->{ex['to_band']}" if ex else "",
                         ex["energy"] if ex else None, ex["E_abs"] if ex else None,
                         ex["dq"] if ex else None, ex["S"] if ex else None,
                         ex["DW"] if ex else None,
                         (ex["verdict"] + (f": {ex['why']}" if ex["why"] else "")) if ex else
                         st["why"]])
    what = "ZPL" if p.s.excite_relax else "vertical E"
    table = format_table(["id", "defect", "q", "k-points", "spin: bands", f"{what}/eV",
                          "E_abs/eV", "dQ/amu^0.5 A", "S (1D)", "DW (1D)", "verdict"], rows,
                         "Production ΔSCF (A1's settings) in D1's ground states; k-points: the "
                         "production grid, or Gamma where the band-number check failed")
    js = {}
    for (cid, q), st in states.items():
        js.setdefault(cid, {})[str(q)] = st
    npass = sum(any(ex["verdict"] == "pass" for ex in st["excitations"])
                for st in states.values())
    return {"states": js, "relaxed": p.s.excite_relax, "_table": table,
            "_note": f"{npass} of {len(states)} charge states pass at the production settings"}


# ==========================================================================================
# PHASE E: THERMODYNAMICS (flowchart E1-E4)
# ==========================================================================================
KB = 8.617333262e-5                # eV/K
# chart colours (reference palette: light surface, categorical slots in their fixed order)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


# The dopants of the defects that reached phase D (D1's relaxations), whose competing phases
# E1.1 relaxes at the production settings.
#   Called by: step_limits(), step_phases()
#   Calls:     Project.host()
def d1_dopants(p):
    cands = p.out("generate")["candidates"]
    own = set(p.host().get_chemical_symbols())
    return sorted({el for r in p.out("relaxations")["runs"].values()
                   for el in cands[r["cand"]]["delta"] if el not in own})


# E1.1: the competing phases at the production settings, for E3's formation energies: the host's
# (made of its own elements) and those of each dopant with a defect in D1 (d1_dopants), all of
# them as phase_list gives them (phases_for), relaxed as phase_jobs relaxes them at the production
# k-point spacing (or a phase's own kspacing; give a metal e.g. smearing_width=0.1 on its line in
# %block phases). Phases of the host's own composition are left out, as in A3.1: A2.1's host is
# that phase (another copy of it, relaxed in another cell, could come out a little lower and stop
# E1.2 with the host off the hull). A3.1 ran the pure elements at the screening settings, for
# screening only. With %block chemical_potentials nothing runs.
#   Called by: main() (step E1.1)
#   Calls:     Project.batch(), Project.host(), Project.kspacing(), Project.production_param(),
#              d1_dopants(), format_table(), phase_jobs(), phase_list(), phase_results(),
#              phases_for(), reduced()
def step_phases(p):
    if p.s.chemical_potentials:
        return {"phases": {}, "_note": "chemical potentials from the project file"}
    host = p.host()
    phases = [x for x in phases_for(phase_list(p), set(host.get_chemical_symbols()), d1_dopants(p))
              if reduced(x[1]) != reduced(host)]
    if not phases:
        return {"phases": {}, "_note": "no competing phases given, so no chemical-potential "
                "limits (E3 then gives energies without the mu terms)"}
    res = p.batch("E_thermo/jobs/phases", phase_jobs(phases, p.s.phases_max_steps), "phases",
                  p.production_param(), spacing=p.kspacing())
    if not res["complete"]:
        return None
    out, rows = phase_results(res, phases)
    return {"phases": out, "_note": f"{len(out)} phases",
            "_table": format_table(["phase", "formula", "atoms", "E/eV", "E/atom (eV)"], rows)}


# E1.2: the chemical-potential limits (chemical_limits on E1.1's phases and A2.1's host, for the
# dopants of D1's defects), or the ones in %block chemical_potentials (name  El=value ..., eV per
# atom, absolute). Also gives each limit as Delta mu against the pure elements, where those
# phases were run.
#   Called by: main() (step E1.2)
#   Calls:     Project.host(), chemical_limits(), counts(), d1_dopants(), format_table(),
#              limits_table(), element()
def step_limits(p):
    if p.s.chemical_potentials:
        limits = {}
        for row in p.s.chemical_potentials:
            mu = {}
            for tok in row[1:]:
                k, _, v = tok.partition("=")
                try:
                    mu[element(k)] = float(v)
                except ValueError:
                    die(f"%block chemical_potentials: bad entry '{tok}' (write El=value)")
            limits[row[0]] = {"mu": mu, "limiting": {}}
        els = sorted({el for lim in limits.values() for el in lim["mu"]})
        table = format_table(["limit"] + [f"mu_{el}" for el in els],
                             [[name] + [lim["mu"].get(el) for el in els]
                              for name, lim in limits.items()],
                             "Chemical potentials from the project file (eV per atom)")
        return {"limits": limits, "_note": f"{len(limits)} limits from the project file",
                "_table": table}
    phases = p.out("phases")["phases"]
    if not phases:
        return {"limits": {}, "_note": "no competing phases, so no limits"}
    limits = chemical_limits(counts(p.host()), p.out("relax")["energy"], phases, d1_dopants(p))
    ref, table = limits_table(limits, phases, "Chemical-potential limits (eV per atom, against "
                              f"the pure elements; the absolute values are in {RESULTS})")
    return {"limits": limits, "references": ref, "_note": f"{len(limits)} limits",
            "_table": table}


# The production results (D1) of the runs that finished, by candidate and charge state (the run,
# with the electronic cell's tag and its pristine energy, host_energy).
#   Called by: step_corrections(), step_formation()
def best_production(p):
    e1, best = p.out("relaxations"), {}
    for r in e1["runs"].values():
        if r["energy"] is None:
            continue
        best.setdefault(r["cand"], {})[r["q"]] = dict(r, host_energy=e1["host_energy"],
                                                      cell=e1["cell"]["tag"])
    return best


# E2: the finite-size correction (charge_correction, with the static dielectric constant from
# A2.3) for every charge state that reached phase D, in the electronic cell (D1's). Returns
# {defect: {q: E_corr}}.
#   Called by: main() (step E2)
#   Calls:     Project.host(), best_production(), correction(), format_table()
def step_corrections(p):
    best = best_production(p)
    prim, c1 = p.host(), p.out("relaxations")["cell"]
    eps = p.out("dielectric").get("eps0")
    cell = np.array(c1["matrix"]) @ np.asarray(prim.cell)
    corr, rows = {}, []
    for cid, st in sorted(best.items()):
        corr[cid] = {str(q): correction(q, cell, eps, p.s.charge_correction) for q in sorted(st)}
        rows += [[cid, c1["tag"], c1["L"], f"{q:+d}", corr[cid][str(q)]] for q in sorted(st)]
    table = format_table(["id", "cell", "L/A", "q", "E_corr/eV"], rows,
                         f"{p.s.charge_correction} corrections (epsilon {eps}) in the electronic "
                         "cell")
    n = sum(1 for st in corr.values() for q in st if int(q))
    return {"corrections": corr, "method": p.s.charge_correction, "eps0": eps, "_table": table,
            "_note": f"{n} charged states of {len(corr)} defects"}


# The band gap used for the Fermi-level range: gap from the project file (e.g. experimental or
# hybrid) or the A2.2 one.
#   Called by: step_formation()
def band_gap(p):
    return p.s.gap or p.out("bands")["gap"]


# The chemical-potential limits (E1.2) as {name: {element: mu}}; without any, one entry 'mu=0' with
# no chemical-potential terms (energies are then only comparable within one composition).
#   Called by: step_formation()
#   Calls:     Project.finished()
def mu_limits(p):
    lim = p.out("limits")["limits"] if p.finished("limits") else {}
    return {k: v["mu"] for k, v in lim.items()} or {"mu=0": {}}


# The thermodynamic transition levels of one defect: walking up the Fermi level from 0 to the gap,
# each point where its lowest-energy charge state changes, as (q_before, q_after, level).
#   Called by: stable_windows(), step_formation()
def transition_levels(ef0, gap):
    qs = sorted(ef0)
    out, cur, x = [], None, 0.0
    while True:
        cur = min(qs, key=lambda q: ef0[q] + q * x)
        nxt = None
        for q in qs:
            if q < cur:                              # lower charges take over as E_F rises
                level = (ef0[q] - ef0[cur]) / (cur - q)
                if level > x + 1e-9 and (nxt is None or level < nxt[1]):
                    nxt = (q, level)
        if nxt is None or nxt[1] >= gap:
            return out
        out.append((cur, nxt[0], nxt[1]))
        x = nxt[1]


# The free carriers at Fermi level mu (absolute, eV) and temperature T, per cm^3, from the host's
# A2.2 band energies (k-point weights; a scissor shift of the conduction bands if gap is set):
# (electrons, holes).
#   Called by: fermi_level()
#   Calls:     occupied_bands()
def carriers(bands, mu, T, shift, volume_cm3):
    occ, eig, w = occupied_bands(bands), bands["eig"], bands["weights"]
    g = 2.0 if bands["nspin"] == 1 else 1.0
    n = h = 0.0
    for s in range(bands["nspin"]):
        cb, vb = eig[s, :, occ[s]:] + shift, eig[s, :, :occ[s]]
        n += g * np.nansum(w[:, None] / (1.0 + np.exp(np.clip((cb - mu) / (KB * T), -500, 500))))
        h += g * np.nansum(w[:, None] / (1.0 + np.exp(np.clip((mu - vb) / (KB * T), -500, 500))))
    return n / volume_cm3, h / volume_cm3


# The self-consistent Fermi level: the E_F (above the VBM) where the defect charges, free electrons
# and holes balance, found by bisection. Each defect's concentration per cm^3 in charge state q is
# sites x orientations x (spin degeneracy) / V x exp(-E_f / kT), or, with fixed totals (quenched),
# its total shared out between charge states by the same Boltzmann weights. Returns (E_F, n, p,
# {cand: {q: concentration}}).
#   Called by: step_formation()
#   Calls:     carriers()
def fermi_level(defects, bands, vbm, gap, T, shift, vol_cm3, totals=None):
    # The net charge per cm^3 at Fermi level x (eV above the VBM), and the concentrations.
    def charge(x):
        conc = {}
        for cid, d in defects.items():
            wts = {q: d["g"][q] * math.exp(-min(700.0, (d["ef0"][q] + q * x) / (KB * T)))
                   for q in d["ef0"]}
            if totals is None:
                conc[cid] = {q: d["sites"] / vol_cm3 * wt for q, wt in wts.items()}
            else:
                z = sum(wts.values()) or 1e-300
                conc[cid] = {q: totals[cid] * wt / z for q, wt in wts.items()}
        n, h = carriers(bands, vbm + x, T, shift, vol_cm3)
        return sum(q * c for cc in conc.values() for q, c in cc.items()) + h - n, n, h, conc

    lo, hi = -1.0, gap + 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if charge(mid)[0] > 0:
            lo = mid
        else:
            hi = mid
    x = 0.5 * (lo + hi)
    _, n, h, conc = charge(x)
    return x, n, h, conc


# Draws the formation energies against the Fermi level, one panel per chemical-potential limit
# (shared y-axis), each defect's lowest charge state as a line with its transition levels marked.
# At most eight defects (the palette's slots in order); the rest are in the tables.
#   Called by: step_formation()
def plot(path, curves, gap, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("note: matplotlib is not installed, so no plot")
        return None
    names = list(next(iter(curves.values())))[:8]
    fig, axes = plt.subplots(1, len(curves), figsize=(4.2 * len(curves) + 1.6, 4.4), sharey=True,
                             squeeze=False, facecolor=SURFACE)
    x = np.linspace(0, gap, 600)
    for ax, (lim, defs) in zip(axes[0], curves.items()):
        ax.set_facecolor(SURFACE)
        for i, name in enumerate(names):
            ef0, levels = defs[name]
            y = np.min([ef0[q] + q * x for q in ef0], axis=0)
            ax.plot(x, y, color=SERIES[i], lw=2, label=name, solid_capstyle="round")
            for _, _, lv in levels:
                ax.plot([lv], [min(ef0[q] + q * lv for q in ef0)], "o", ms=6, color=SERIES[i],
                        mec=SURFACE, mew=1.5, zorder=3)
        ax.set_xlim(0, gap)
        ax.set_title(lim, color=INK, fontsize=10)
        ax.set_xlabel("Fermi level above the VBM (eV)", color=INK2, fontsize=9)
        ax.grid(color=GRID, lw=0.8)
        ax.tick_params(colors=INK2, labelsize=8)
        for s in ax.spines.values():
            s.set_color(GRID)
    axes[0][0].set_ylabel("formation energy (eV)", color=INK2, fontsize=9)
    leg = axes[0][-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
                             fontsize=8)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.suptitle(title, color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


# The sets of defects E3 finds a Fermi level for, each as if only those dopants were in the crystal
# (so Sb's defects never set the Fermi level of a Bi-doped sample): 'native' (the native defects
# alone), each dopant X (the native defects and those that add only X), and each mix of dopants
# that one defect adds together, e.g. 'Bi+Sb' (with every defect whose dopants are among them).
# ids: the defects that reached phase D. Returns {name: [defect ids]}.
#   Called by: step_formation()
def dopant_sets(cands, ids, host_el):
    dop = {cid: frozenset(el for el, n in cands[cid]["delta"].items()
                          if el not in host_el and n > 0) for cid in ids}
    sets = {"native": sorted(c for c in ids if not dop[c])}
    for mix in sorted({x for x in dop.values() if x}, key=lambda x: (len(x), sorted(x))):
        sets["+".join(sorted(mix))] = sorted(c for c in ids if dop[c] <= mix)
    return sets


# E3: formation energies, transition levels and (with fermi_temperature) the Fermi level.
# For each defect that reached phase D, its energy in each charge state (D1, the electronic cell)
# gives E_f(q, E_F=0) = E_D - E_host - sum n_i mu_i + q E_VBM + E_corr at every limit, and its
# thermodynamic transition levels in the gap (transition_levels; the same at every limit); the
# plot goes to E_thermo/formation.png. With fermi_temperature, fermi_level finds the
# self-consistent Fermi level at each limit for each set of dopants on its own (dopant_sets), from
# their defects and the host's free carriers (quench_temperature: concentrations frozen at
# fermi_temperature, then charges redistributed).
#   Called by: main() (step E3)
#   Calls:     Project.host(), band_gap(), best_production(), dopant_sets(), fermi_level(),
#              format_table(), mu_limits(), plot(), read_bands(), transition_levels()
def step_formation(p):
    d = p.phase_dir("E")
    best = best_production(p)
    if not best:
        die("no production run finished, so there are no formation energies")
    cands = p.out("generate")["candidates"]
    corr = {cid: {int(q): v for q, v in c.items()}
            for cid, c in p.out("corrections")["corrections"].items()}
    vbm, gap = p.out("bands")["vbm"], band_gap(p)
    lims = mu_limits(p)
    table, curves, levels_rows, results = [], {}, [], {}
    for lim, mu in lims.items():
        curves[lim] = {}
        for cid in sorted(best):
            c = cands[cid]
            missing = [el for el in c["delta"] if mu and el not in mu]
            if missing:
                die(f"limit {lim} has no chemical potential for " + ", ".join(missing))
            dmu = sum(n * mu.get(el, 0.0) for el, n in c["delta"].items())
            ef0 = {q: r["energy"] - r["host_energy"] - dmu + q * vbm + corr[cid].get(q, 0.0)
                   for q, r in best[cid].items()}
            lv = transition_levels(ef0, gap)
            curves[lim][f"{cid} {c['name']}"] = (ef0, lv)
            results.setdefault(cid, {"name": c["name"], "levels": [list(x) for x in lv],
                                     "formation": {}})["formation"][lim] = {str(q): v
                                                                           for q, v in ef0.items()}
            for q in sorted(ef0):
                table.append([lim, cid, c["name"], f"{q:+d}", ef0[q], ef0[q] + q * gap,
                              best[cid][q]["job"], best[cid][q]["cell"]])
    for cid in sorted(best):
        lv = results[cid]["levels"]
        levels_rows.append([cid, cands[cid]["name"],
                            ", ".join(f"({a:+d}/{b:+d}) {e:.3f}" for a, b, e in lv) or
                            "one charge state across the gap"])
    tables = [format_table(["limit", "id", "defect", "q", "E_f(VBM)/eV", "E_f(CBM)/eV",
                            "from run", "cell"], table,
                           f"Formation energies (E_VBM {vbm:.4f} eV, gap {gap:.3f} eV)"),
              format_table(["id", "defect", "transition levels (eV above VBM)"], levels_rows,
                           "Thermodynamic transition levels in the gap")]
    png = plot(d / "formation.png", curves, gap, f"{p.name}: formation energies")
    fermi, sets = {}, dopant_sets(cands, sorted(best), set(p.host().get_chemical_symbols()))
    if p.s.fermi_temperature:
        bands = read_bands(p.dir / p.out("bands")["bands_file"])
        prim = p.host()
        vol = prim.get_volume() * 1e-24
        shift = gap - p.out("bands")["gap"]
        rows = []
        for name, ids in sets.items():
            if not ids:
                continue
            for lim in lims:
                defects = {}
                for cid in ids:
                    c = cands[cid]
                    # a pair of two identical components is counted from both ends: halve it
                    same = len(c["comps"]) == 2 and all(
                        c["comps"][0].get(k) == c["comps"][1].get(k)
                        for k in ("new", "host", "prim_site", "frac"))
                    sites = c["sites"] * c["multiplicity"] / (2 if same else 1)
                    g = {q: (abs(round(r["spin_final"] or 0)) + 1) for q, r in best[cid].items()}
                    defects[cid] = {"ef0": {int(q): float(v) for q, v in
                                            results[cid]["formation"][lim].items()},
                                    "g": g, "sites": sites}
                T = p.s.fermi_temperature
                x, n, h, conc = fermi_level(defects, bands, vbm, gap, T, shift, vol)
                out = {"T": T, "E_F": x, "electrons": n, "holes": h,
                       "concentrations": {cid: {str(q): v for q, v in cc.items()}
                                          for cid, cc in conc.items()}}
                rows.append([name, lim, f"{T:g} K", x, f"{n:.2e}", f"{h:.2e}"])
                if p.s.quench_temperature:
                    totals = {cid: sum(cc.values()) for cid, cc in conc.items()}
                    xq, nq, hq, cq = fermi_level(defects, bands, vbm, gap,
                                                 p.s.quench_temperature, shift, vol, totals)
                    out["quenched"] = {"T": p.s.quench_temperature, "E_F": xq, "electrons": nq,
                                       "holes": hq,
                                       "concentrations": {cid: {str(q): v for q, v in cc.items()}
                                                          for cid, cc in cq.items()}}
                    rows.append([name, lim, f"quenched {p.s.quench_temperature:g} K", xq,
                                 f"{nq:.2e}", f"{hq:.2e}"])
                fermi.setdefault(name, {})[lim] = out
        tables.append(format_table(["dopants", "limit", "T", "E_F above VBM/eV", "n/cm^-3",
                                    "p/cm^-3"], rows,
                                   "Self-consistent Fermi level for each set of dopants on its "
                                   "own (native: the native defects alone; X: the native "
                                   "defects and those of X alone; X+Y: also the defects with "
                                   "both), from the defects that reached phase D"))
        log("note: the Fermi level includes only the defects that reached phase D; defects "
            "left out (e.g. native defects dropped in C) may change it")
    return {"vbm": vbm, "gap": gap, "limits": lims, "defects": results, "fermi": fermi,
            "dopant_sets": sets,
            "plot": str(png.relative_to(p.dir)) if png else None, "_table": "\n".join(tables),
            "_note": f"{len(best)} defects at {len(lims)} limit(s)"}


# The Fermi-level range (eV above the VBM, within [0, gap]) where each charge state of a defect is
# its lowest (E3's formation energies at one limit, {q: E_f at E_F = 0}; the ranges are the same at
# every limit), from its transition levels (transition_levels). Returns {q: (lo, hi)} for the
# charge states that are lowest somewhere.
#   Called by: step_interest()
#   Calls:     transition_levels()
def stable_windows(ef0, gap):
    lv = transition_levels(ef0, gap)
    if not lv:
        return {min(ef0, key=lambda q: ef0[q] + q * gap / 2): (0.0, round(gap, 3))}
    out, x = {}, 0.0
    for cur, _, e in lv:
        out[cur] = (round(x, 3), round(e, 3))
        x = e
    out[lv[-1][1]] = (round(x, 3), round(gap, 3))
    return out


# E4: the decision before phase F. A charge state goes on to the lineshape if its excitation
# passed at the production settings (D2) and it is the defect's lowest charge state for some Fermi
# level inside the gap (stable_windows, from E3): an excitation of a charge state that never
# forms can't be seen. The table also gives, for each set of dopants E3 found a Fermi level for
# (dopant_sets: the defect's own), whether that Fermi level falls in the range at each limit, the
# deep levels (more than deep_level_margin from both band edges), and the band states C7 left
# pending (to characterise with a hybrid later).
#   Called by: main() (step E4)
#   Calls:     Project.host(), format_table(), stable_windows()
def step_interest(p):
    form, zpl, d7 = p.out("formation"), p.out("zpl"), p.out("excite")
    gap, m = form["gap"], p.s.deep_level_margin
    cands = p.out("generate")["candidates"]
    host_el = set(p.host().get_chemical_symbols())
    picks, rows = [], []
    for cid, dd in sorted(form["defects"].items()):
        lim0 = next(iter(dd["formation"]))
        win = stable_windows({int(q): v for q, v in dd["formation"][lim0].items()}, gap)
        deep = [f"({a:+d}/{b:+d}) {e:.2f}" for a, b, e in dd["levels"] if m <= e <= gap - m]
        mine = "+".join(sorted(el for el, n in cands[cid]["delta"].items()
                               if el not in host_el and n > 0)) or "native"
        for qs, st in sorted((zpl.get("states") or {}).get(cid, {}).items(), key=lambda kv:
                             int(kv[0])):
            q = int(qs)
            ok = [ex for ex in st["excitations"] if ex["verdict"] == "pass"]
            if not ok:
                continue
            ex = max(ok, key=lambda x: x["DW"] or 0.0)
            w = win.get(q)
            ef = {lim: v["E_F"] for lim, v in (form["fermi"].get(mine) or {}).items()}
            inside = [lim for lim, x in ef.items() if w and w[0] <= x <= w[1]]
            goes = w is not None
            rows.append([cid, dd["name"], f"{q:+d}", ex["energy"], ex["DW"],
                         f"{w[0]:.2f}-{w[1]:.2f}" if w else "never lowest",
                         (", ".join(inside) or "none") if ef else "", "; ".join(deep),
                         "yes" if goes else "no"])
            if goes:
                picks.append({"cand": cid, "name": dd["name"], "q": q, "spin": ex["spin"],
                              "from_band": ex["from_band"], "to_band": ex["to_band"],
                              "zpl": ex["energy"], "E_abs": ex["E_abs"], "dq": ex["dq"],
                              "S": ex["S"], "DW": ex["DW"], "window": list(w),
                              "excited": ex["structure"], "ex_job": ex["job"],
                              "gs_job": ex["gs_job"], "fermi_inside": inside})
    pending = d7.get("pending") or []
    table = format_table(["id", "defect", "q", "ZPL/eV", "DW (1D)", "lowest for E_F/eV",
                          "E_F there at", "deep levels/eV", "on to F"], rows,
                         "Excitations that passed D2, and whether their charge state forms "
                         f"(lowest for some Fermi level in the {gap:.2f} eV gap); E_F there at: "
                         "the limits whose self-consistent Fermi level (E3, the defect's own "
                         "dopants) falls in that range")
    if pending:
        table += ("\nBand states C7 left pending (to characterise with a hybrid): "
                  + ", ".join(pending) + "\n")
    return {"for_F": picks, "pending": pending, "_table": table,
            "_note": f"{len(picks)} excitation(s) for the lineshape phase (F)"}


# ==========================================================================================
# PHASE F: OPTICAL LINESHAPES (flowchart F1-F2)
# ==========================================================================================
# The excitations E4 sends on, at the production settings (PBE with A1's cut-off and k-points, or
# Gamma where D2 fell back to it). F1: the 1D configuration-coordinate diagram, always; F2
# (lineshape : true): the multi-mode lineshape from a PBE model's phonons in a big cell. Both give
# the luminescence lineshape by the generating-function method (pl_spectrum), in the equal-mode
# approximation (the excited state vibrates as the ground state does; F1 reports both states'
# curvatures, which shows how far that holds).

# The energy of a finished CASTEP job (run_batch's results), or None unless it finished with a
# converged SCF.
#   Called by: step_ccd(), step_lineshape()
def job_energy(res, name):
    r = res["jobs"].get(name)
    if not r or r["status"] != "done" or r["energy"] is None or not r["scf_converged"]:
        return None
    return r["energy"]


# A random number generator seeded by a job's name, so the job's structure is the same each time
# its step runs (the job is then reused, not found changed), whatever other jobs there are.
#   Called by: step_lineshape()
def named_rng(name):
    return np.random.default_rng(int(hashlib.md5(name.encode()).hexdigest()[:12], 16))


# A parabola through the points of an energy curve E(Q) (Q: the mass-weighted coordinate, amu^1/2
# Ang; E in eV) within half of centre (through all of them if fewer than three are that close):
# the mode energy from its curvature c, hw = HW_UNIT sqrt(2c) (E = c Q^2 = w^2 Q^2 / 2), where its
# minimum is (Q0, E0), and the largest misfit of the points it went through (eV; large: the curve
# is not a parabola there). None with fewer than three points, or a curvature that isn't positive.
#   Called by: ccd_fit()
def harmonic_fit(Q, E, centre, half):
    Q, E = np.asarray(Q, float), np.asarray(E, float)
    ok = np.isfinite(E)
    sel = ok & (np.abs(Q - centre) <= half + 1e-9)
    if sel.sum() < 3:
        sel = ok
    if sel.sum() < 3:
        return None
    c2, c1, c0 = np.polyfit(Q[sel], E[sel], 2)
    if c2 <= 0:
        return None
    misfit = float(np.abs(np.polyval([c2, c1, c0], Q[sel]) - E[sel]).max())
    return {"hw": round(HW_UNIT * math.sqrt(2 * c2), 5), "Q0": round(-c1 / (2 * c2), 4),
            "E0": c0 - c1 * c1 / (4 * c2), "misfit": round(misfit, 5),
            "coef": [float(c2), float(c1), float(c0)]}


# What F1's points give: xs, where they are along the path (0: the ground state's structure, 1:
# the excited state's, dq apart, amu^1/2 Ang); eg and ee, the ground and excited states' energies
# there (None where a job failed). Each state's parabola near its own minimum (harmonic_fit, the
# points within 0.6 dq), its mode energy hw and its Huang-Rhys factor in the 1D model,
# S = hw dQ^2 / (2 hbar^2) (for a parabola, its relaxation energy over hw); the ZPL (E_e(1) -
# E_g(0) when both are points, else between the parabolas' minima), the vertical absorption
# E_e(0) - E_g(0) and emission E_e(1) - E_g(1); the energies relative to the ground state's at 0
# (or its parabola's minimum).
#   Called by: step_ccd()
#   Calls:     harmonic_fit()
def ccd_fit(xs, eg, ee, dq):
    xs = np.asarray(xs, float)
    Q = xs * dq
    eg = np.array([np.nan if e is None else e for e in eg], float)
    ee = np.array([np.nan if e is None else e for e in ee], float)
    fg, fe = harmonic_fit(Q, eg, 0.0, 0.6 * dq), harmonic_fit(Q, ee, dq, 0.6 * dq)

    # The energy at x, if x is one of the points and its job finished.
    def at(arr, x):
        i = np.flatnonzero(np.isclose(xs, x))
        return float(arr[i[0]]) if len(i) and np.isfinite(arr[i[0]]) else None

    g0, g1, e0, e1 = at(eg, 0.0), at(eg, 1.0), at(ee, 0.0), at(ee, 1.0)
    ref = g0 if g0 is not None else (fg["E0"] if fg else float(np.nanmin(eg)))
    zpl = (e1 - g0 if e1 is not None and g0 is not None else
           fe["E0"] - fg["E0"] if fg and fe else None)
    out = {"Q": Q, "Eg": eg - ref, "Ee": ee - ref, "ref": ref, "fit_g": fg, "fit_e": fe,
           "zpl": None if zpl is None else round(zpl, 4),
           "E_abs": None if e0 is None or g0 is None else round(e0 - g0, 4),
           "E_em": None if e1 is None or g1 is None else round(e1 - g1, 4)}
    for s, f in (("g", fg), ("e", fe)):
        out[f"hw_{s}"] = f["hw"] if f else None
        out[f"S_{s}"] = round(f["hw"] * dq * dq / (2 * HW_UNIT ** 2), 3) if f else None
    return out


# The luminescence lineshape of an excitation from its vibrations (hw: each mode's energy, eV; S:
# its partial Huang-Rhys factor), by the generating-function method (Alkauskas et al., New J.
# Phys. 16, 073026 (2014); at a temperature T as in Razinkovas et al., Phys. Rev. B 104, 045303
# (2021)), on an energy grid (a quarter of the smaller of sigma and gamma, at least 0.1 meV):
#   S(hw) = sum_k S_k g(hw - hw_k), g a Gaussian of width sigma: the spectral function
#   G(t) = exp( integral of S(hw) [(n + 1)(exp(-iwt) - 1) + n (exp(iwt) - 1)] ), n the Bose
#          occupation of hw at T (absorbed phonons: the anti-Stokes side)
#   A(e) = 1/(2 pi) integral of G(t) exp(i e t - gamma |t|) dt, e the energy given to the lattice
#          (the ZPL at e = 0, a Lorentzian of half-width gamma)
#   L(E) = E^3 A(zpl - E), the photons' spectrum, scaled to a maximum of 1,
# both integrals by FFT. Returns {E, PL (on E), w, Sw (the spectral function on w), S (total),
# DW (exp(-S), at 0 K), DW_T (exp(-integral of S(hw) (2n + 1)), at T), hw_eff (S-weighted mean
# mode energy)}; DW is the share of the lineshape A in the ZPL.
#   Called by: step_ccd(), step_lineshape()
def pl_spectrum(hw, S, zpl, T, sigma, gamma):
    hw, S = np.asarray(hw, float), np.asarray(S, float)
    keep = (hw > 0) & (S > 0)
    hw, S = hw[keep], S[keep]
    stot = float(S.sum())
    de = max(min(sigma, gamma) / 4.0, 1e-4)
    wmax = (float(hw.max()) if len(hw) else 0.05) + 5 * sigma
    span = 2 * max(wmax * (stot + 6 * math.sqrt(stot) + 6), 0.5)
    n = int(2 ** math.ceil(math.log2(span / de)))
    if n > 2 ** 20:                                # (a very wide band: a coarser grid)
        n = 2 ** 20
        de = span / n
    eps = np.fft.fftfreq(n, 1.0 / (n * de))        # j de, the upper half negative
    t = np.fft.fftfreq(n, de / (2 * math.pi))      # the times that go with it (hbar / eV)
    w = eps[1:n // 2]
    Sw = np.zeros(len(w))
    for k in range(0, len(hw), 256):              # (in chunks: many modes, a fine grid)
        dw = w[None, :] - hw[k:k + 256, None]
        Sw += (S[k:k + 256, None] * np.exp(-dw * dw / (2 * sigma * sigma))).sum(axis=0)
    Sw /= sigma * math.sqrt(2 * math.pi)
    if Sw.sum() > 0:
        Sw *= stot / (Sw.sum() * de)              # (the part cut off below 0 put back)
    with np.errstate(over="ignore", divide="ignore"):
        nb = (np.zeros(len(w)) if T <= 0 else 1.0 / np.expm1(w / (KB * T)))
    nb = np.nan_to_num(nb, posinf=0.0)
    emit, absorb = np.zeros(n), np.zeros(n)
    emit[1:n // 2], absorb[1:n // 2] = Sw * (nb + 1) * de, Sw * nb * de
    X = (np.fft.fft(emit) - emit.sum()) + (n * np.fft.ifft(absorb) - absorb.sum())
    G = np.exp(X) * np.exp(-gamma * np.abs(t))
    A = (np.fft.ifft(G) / de).real                # (dt / 2 pi) n ifft, dt = 2 pi / (n de)
    E = zpl - eps
    order = np.argsort(E)
    E, A = E[order], A[order]
    sel = E > 0
    E, L = E[sel], np.clip(A[sel], 0.0, None) * E[sel] ** 3
    if L.max() > 0:
        L = L / L.max()
    return {"E": E, "PL": L, "w": w, "Sw": Sw, "S": round(stot, 4),
            "DW": round(math.exp(-stot), 4),
            "DW_T": round(math.exp(-float((Sw * (2 * nb + 1)).sum() * de)), 4),
            "hw_eff": round(float((S * hw).sum() / stot), 5) if stot > 0 else None}


# Writes columns of numbers to a text file with a header line (# ...).
#   Called by: step_ccd(), step_lineshape()
def write_columns(path, header, cols):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.column_stack(cols), fmt="%.8g", header=header)
    return path


# Draws a figure of side-by-side panels (F1, F2), each {'title', 'xlabel', 'ylabel', 'series':
# [{'x', 'y', 'label', 'kind': 'line' or 'points', 'colour': its slot}], 'mark': the x of a dashed
# line, or None}: each series in its own colour slot of the palette (the same state keeps its
# colour in every panel), a legend where a panel has more than one named series.
#   Called by: step_ccd(), step_lineshape()
def draw_panels(path, title, panels):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("note: matplotlib is not installed, so no plot")
        return None
    fig, axes = plt.subplots(1, len(panels), figsize=(4.8 * len(panels), 3.9), squeeze=False,
                             facecolor=SURFACE)
    for ax, pn in zip(axes[0], panels):
        ax.set_facecolor(SURFACE)
        for i, s in enumerate(pn["series"]):
            c = SERIES[s.get("colour", i) % len(SERIES)]
            if s.get("kind") == "points":
                ax.plot(s["x"], s["y"], "o", ms=6, color=c, mec=SURFACE, mew=1.5,
                        label=s.get("label"), zorder=3)
            else:
                ax.plot(s["x"], s["y"], color=c, lw=1.6, label=s.get("label"),
                        solid_capstyle="round")
        if pn.get("mark") is not None:
            ax.axvline(pn["mark"], color=INK2, lw=0.8, ls="--")
        ax.set_title(pn.get("title", ""), color=INK, fontsize=10)
        ax.set_xlabel(pn.get("xlabel", ""), color=INK2, fontsize=9)
        ax.set_ylabel(pn.get("ylabel", ""), color=INK2, fontsize=9)
        ax.grid(color=GRID, lw=0.8)
        ax.tick_params(colors=INK2, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        if sum(1 for s in pn["series"] if s.get("label")) > 1:
            leg = ax.legend(frameon=False, fontsize=8)
            for tx in leg.get_texts():
                tx.set_color(INK)
    fig.suptitle(title, color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


# F1: the 1D configuration-coordinate diagram of each excitation E4 sends on (F_lineshape/jobs/
# ccd_gs, ccd_ex): the structures along the straight line from D1's relaxed ground state (x = 0)
# to D2's relaxed excited state (x = 1), at ccd_points; at each, a ground-state single point (its
# .check kept) and, from it, a ΔSCF single point of the same excitation (dscf_job), both at D2's
# settings (the production ones with dscf_settings, Gamma where D2 used it); the parabolas, mode
# energies and 1D Huang-Rhys factors of both states (ccd_fit) and the 1D luminescence lineshape at
# lineshape_temperature (pl_spectrum: the ground state's mode, S_g, broadened by lineshape_sigma,
# the ZPL by lineshape_gamma). Files in F_lineshape/ccd/: <excitation>_ccd.dat (x, Q and both
# energies), <excitation>_pl_1d.dat and <excitation>_ccd.png. Runner group 'production'.
#   Called by: main() (step F1)
#   Calls:     Project.batch(), Project.host(), Project.kspacing(), Project.production_param(),
#              build_defect(), ccd_fit(), delta_q(), draw_panels(), dscf_job(), dscf_settings(),
#              format_table(), job_energy(), load_cell(), pl_spectrum(), production_job(), qtag(),
#              start_spin(), write_columns()
def step_ccd(p):
    from ase.geometry import find_mic
    picks = p.out("interest")["for_F"]
    if not picks:
        return {"lines": {}, "_note": "E4 sent no excitation on to F"}
    d = p.phase_dir("F")
    e1, e2 = p.out("relaxations"), p.out("zpl")
    cands, prim = p.out("generate")["candidates"], p.host()
    Pe = np.array(e1["cell"]["matrix"])
    xs = list(p.s.ccd_points)
    paths, gs_jobs, skipped = {}, [], {}
    for pk in picks:
        cid, q = pk["cand"], pk["q"]
        tag = f"{cid}_{qtag(q)}_ex{pk['spin'] + 1}"
        run = e1["runs"][f"{cid}_{qtag(q)}"]
        if not pk.get("excited"):
            skipped[tag] = "no relaxed excited state (excite_relax : false)"
            continue
        g, e = load_cell(p, run["final"]), load_cell(p, pk["excited"])
        if g.get_chemical_symbols() != e.get_chemical_symbols():
            skipped[tag] = "the ground and excited structures' atoms don't match"
            continue
        shift, _ = find_mic(e.positions - g.positions, g.cell, pbc=True)
        info = build_defect(prim, Pe, cands[cid]["comps"], cid)[1]
        S = start_spin(g, q, run["spin_final"])
        gamma = e2["states"][cid][str(q)].get("sampling") == "Gamma"
        jobs = []
        for i, x in enumerate(xs):
            a = g.copy()
            a.positions = g.positions + x * shift
            job, _ = production_job(p, f"{tag}_p{i:02d}_g", a, info, q, S)
            job.pop("cell_extra", None)
            job["kind"] = "sp"
            dscf_settings(p, job)
            if gamma:
                job["kgrid"] = [1, 1, 1]
            jobs.append(job)
        gs_jobs += jobs
        paths[tag] = {"pick": pk, "dq": delta_q(g, e), "gamma": gamma, "jobs": jobs}
    if not paths:
        return {"lines": {}, "skipped": skipped,
                "_note": "no excitation has a relaxed excited state to draw a path to"}
    res1 = p.batch(f"{PHASE_DIRS['F']}/jobs/ccd_gs", gs_jobs, "ccd", p.production_param(),
                   spacing=p.kspacing(), keep_check=True)
    if not res1["complete"]:
        return None
    ex_jobs = []
    for tag, pa in paths.items():
        pk = pa["pick"]
        ex = {"spin": pk["spin"], "from_band": pk["from_band"], "to_band": pk["to_band"]}
        for job in pa["jobs"]:
            if job_energy(res1, job["name"]) is None:
                continue
            nspin = 2 if job["overrides"].get("spin_polarized") == "true" else 1
            xj = dscf_job(job, res1["calcs"][job["name"]], job["name"][:-2] + "_e",
                          d / "jobs" / "ccd_ex", ex, nspin, "sp", p.production_param())
            xj["overrides"].pop("pdos_calculate_weights", None)
            ex_jobs.append(xj)
    res2 = {"complete": True, "jobs": {}, "calcs": {}}
    if ex_jobs:
        res2 = p.batch(f"{PHASE_DIRS['F']}/jobs/ccd_ex", ex_jobs, "ccd", p.production_param(),
                       spacing=p.kspacing())
        if not res2["complete"]:
            return None
    T, sig, gam = p.s.lineshape_temperature, p.s.lineshape_sigma, p.s.lineshape_gamma
    lines, rows = {}, []
    for tag, pa in paths.items():
        pk, dq = pa["pick"], pa["dq"]
        eg = [job_energy(res1, f"{tag}_p{i:02d}_g") for i in range(len(xs))]
        ee = [job_energy(res2, f"{tag}_p{i:02d}_e") for i in range(len(xs))]
        f = ccd_fit(xs, eg, ee, dq)
        misfits = [x["misfit"] for x in (f["fit_g"], f["fit_e"]) if x]
        line = {"cand": pk["cand"], "name": pk["name"], "q": pk["q"], "spin": pk["spin"],
                "dq": round(dq, 4), "sampling": "Gamma" if pa["gamma"] else "k-points",
                "points": sum(e is not None for e in eg),
                "excited_points": sum(e is not None for e in ee), "zpl_E2": pk["zpl"],
                **{k: f[k] for k in ("zpl", "E_abs", "E_em", "hw_g", "hw_e", "S_g", "S_e")},
                "misfit": max(misfits) if misfits else None,
                "DW": None, "DW_T": None, "files": {}}
        out = d / "ccd"
        line["files"]["ccd"] = str(write_columns(
            out / f"{tag}_ccd.dat", "x  Q/amu^0.5A  E_ground/eV  E_excited/eV (from the ground "
            "state at x = 0)", [xs, f["Q"], f["Eg"], f["Ee"]]).relative_to(p.dir))
        panels = [{"title": "configuration coordinate", "xlabel": "Q (amu$^{1/2}$ Å)",
                   "ylabel": "energy (eV)", "series": []}]
        qq = np.linspace(min(f["Q"]), max(f["Q"]), 200)
        for k, (lab, key, fit) in enumerate((("ground", "Eg", f["fit_g"]),
                                             ("excited", "Ee", f["fit_e"]))):
            panels[0]["series"].append({"x": f["Q"], "y": f[key], "label": lab,
                                        "kind": "points", "colour": k})
            if fit:
                panels[0]["series"].append({"x": qq, "y": np.polyval(fit["coef"], qq) - f["ref"],
                                            "colour": k})
        zpl = f["zpl"] if f["zpl"] is not None else pk["zpl"]
        if f["hw_g"] and f["S_g"]:
            sp = pl_spectrum([f["hw_g"]], [f["S_g"]], zpl, T, sig, gam)
            line.update(DW=sp["DW"], DW_T=sp["DW_T"])
            line["files"]["pl"] = str(write_columns(
                out / f"{tag}_pl_1d.dat", f"E/eV  PL (1D, {T:g} K, max 1)",
                [sp["E"], sp["PL"]]).relative_to(p.dir))
            win = (sp["E"] > zpl - 8 * f["S_g"] * f["hw_g"] - 0.3) & (sp["E"] < zpl + 0.15)
            panels.append({"title": f"luminescence, 1D ({T:g} K)", "xlabel": "energy (eV)",
                           "ylabel": "intensity (max 1)", "mark": zpl,
                           "series": [{"x": sp["E"][win], "y": sp["PL"][win], "colour": 0}]})
        png = draw_panels(out / f"{tag}_ccd.png", f"{tag}: {pk['name']} q {pk['q']:+d}", panels)
        if png:
            line["files"]["png"] = str(png.relative_to(p.dir))
        lines[tag] = line
        rows.append([tag, pk["name"], line["sampling"], line["zpl"], line["zpl_E2"],
                     line["E_abs"], line["E_em"], line["dq"], line["hw_g"], line["hw_e"],
                     line["S_g"], line["S_e"], line["DW"]] + ([line["DW_T"]] if T > 0 else [])
                    + [line["misfit"]])
    table = format_table(["excitation", "defect", "k-points", "ZPL/eV", "ZPL D2", "E_abs",
                          "E_em", "dQ", "hw_g/eV", "hw_e/eV", "S_g", "S_e", "DW 0K"]
                         + ([f"DW {T:g}K"] if T > 0 else []) + ["misfit/eV"], rows,
                         "1D configuration-coordinate diagrams (ground-state and ΔSCF single "
                         "points along the line between the relaxed structures; dQ in amu^0.5 A; "
                         "S = hw dQ^2/2hbar^2; DW from S_g)")
    if skipped:
        table += "\nNot drawn: " + "; ".join(f"{t} ({w})" for t, w in skipped.items()) + "\n"
    return {"points": xs, "lines": lines, "skipped": skipped, "_table": table,
            "_note": f"{len(lines)} configuration-coordinate diagram(s)"}


# A defect's relaxed structure in the electronic cell (Pe) as F2's big cell (Pbig): carried over
# (carry_over) or, when the big cell is the electronic cell, itself; either way with its atoms in
# the order of the defect built there (build_defect), and with each atom's ideal position relative
# to the defect's centre (for common_atoms). Returns (Atoms, those positions).
#   Called by: step_lineshape()
#   Calls:     build_defect(), carry_over(), match_ideal()
def big_cell(prim, comps, Pe, relaxed, Pbig, label):
    from ase.geometry import find_mic
    if np.array_equal(np.asarray(Pbig), np.asarray(Pe)):
        _, _, idx, _, v = match_ideal(prim, comps, Pe, relaxed, label)
        return relaxed[idx], v
    big, info = carry_over(prim, comps, Pe, relaxed, Pbig, label=label)
    ideal, _ = build_defect(prim, Pbig, comps, label)
    v, _ = find_mic(ideal.positions - np.asarray(info["centre"]), ideal.cell, pbc=True)
    return big, v


# The vector F2 projects onto the phonons, on the big cell's atoms (big_v: their ideal positions
# relative to the defect's centre; big_syms: their elements): with lineshape_method forces, ΔF,
# the excited state's forces at the ground state's structure (the first step of D2's ΔSCF) less
# the ground state's there (D2's ground-state single point; zero if it printed none: the structure
# is relaxed), eV/Ang; with displacements, ΔR, the displacement of D2's relaxed excited structure
# from D1's relaxed ground state, Ang. Each atom's value goes to the same atom of the big cell
# (match_ideal, common_atoms), out to just under half the electronic cell's shortest image
# distance (the region that cell describes); zero further out. Returns (N x 3 array or None, the
# atoms that carry it, why None).
#   Called by: step_lineshape()
#   Calls:     common_atoms(), load_cell(), match_ideal(), min_image(), read_trajectory()
def lineshape_vector(p, pk, run, prim, comps, Pe, big_v, big_syms, label):
    jobs = p.dir / PHASE_DIRS["D"] / "jobs"
    if p.s.lineshape_method == "forces":
        xf = jobs / "zpl" / pk["ex_job"] / f"{pk['ex_job']}.castep"
        sub = "zpl_gs_gamma" if pk["gs_job"].endswith("_G") else "zpl_gs"
        gf = jobs / sub / pk["gs_job"] / f"{pk['gs_job']}.castep"
        xs = read_trajectory(xf.read_text(errors="replace"), "free") if xf.exists() else []
        if not xs:
            return None, 0, f"no forces in D2's ΔSCF output ({xf.name})"
        atoms, _, fx = xs[0]
        gs = read_trajectory(gf.read_text(errors="replace"), "free") if gf.exists() else []
        fg = gs[0][2] if gs and gs[0][2].shape == fx.shape else np.zeros_like(fx)
        ideal, _, idx, _, v = match_ideal(prim, comps, Pe, atoms, label)
        vec = (fx - fg)[idx]
    else:
        if not pk.get("excited"):
            return None, 0, "no relaxed excited state (excite_relax : false)"
        ideal, _, _, ug, v = match_ideal(prim, comps, Pe, load_cell(p, run["final"]), label)
        _, _, _, ue, _ = match_ideal(prim, comps, Pe, load_cell(p, pk["excited"]), label)
        vec = ue - ug
    radius = 0.5 * min_image(ideal.cell) - 0.1
    a, b = common_atoms(v, ideal.get_chemical_symbols(), big_v, big_syms, radius)
    out = np.zeros((len(big_syms), 3))
    out[b] = vec[a]
    return out, len(a), ""


# Clears a training folder's model if the training it came from is not the one now in the folder:
# its data files and its command (cmd, less --device: where it runs doesn't change the model),
# fingerprinted in inputs.md5 (a folder without one keeps its model), so a changed training set or
# setting is never answered with the old model.
#   Called by: step_lineshape()
def check_training(d, cmd):
    h = hashlib.md5(re.sub(r"--device=\S+", "", cmd).encode())
    for f in sorted(d.glob("*_train.xyz")) + sorted(d.glob("*_test.xyz")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    stamp = d / "inputs.md5"
    old = stamp.read_text().strip() if stamp.exists() else None
    if old is not None and old != h.hexdigest():
        for name in ("model_stagetwo.model", "model.model", "job.json"):
            if (d / name).exists():
                (d / name).unlink()
        log(f"note: the training set in {d} has changed, so its model is trained again")
    stamp.write_text(h.hexdigest() + "\n")


# F2 (lineshape : true): the multi-mode luminescence lineshape of each excitation E4 sends on.
#  1. Single points at the production settings (F_lineshape/jobs/lineshape_data): lineshape_host
#     pristine electronic cells and lineshape_displacements copies of each charge state's D1
#     ground state, every atom displaced at random (normal, lineshape_amplitude Ang a component).
#  2. A PBE model of each defect (F_lineshape/model/<defect>: train_command's MACE, one 'pbe'
#     head, the total-charge embedding): trained on those, D1's relaxations of all its charge
#     states (filter_frames) and the pristine cells, a fifth kept for testing; with
#     model_jobscript, run_training's jobs.
#  3. The phonons of each excitation's ground state with its model (phonon_worker: a job with
#     model_jobscript, else here; F_lineshape/phonons/<excitation>) in a big cell, the smallest with
#     images at least lineshape_length apart (size_matrix; the electronic cell when that is longer
#     already), the relaxed structure carried into it (big_cell), its force constants by finite
#     differences (lineshape_delta), each mode's part in the excitation (lineshape_vector:
#     lineshape_method forces, ΔF, or displacements, ΔR) and partial Huang-Rhys factor.
#  4. The lineshape (pl_spectrum) at lineshape_temperature, with F1's 1D one for comparison:
#     F_lineshape/lineshape/<excitation>_S.dat (the spectral function), _pl.dat and .png.
# Runner group 'production' (the single points).
#   Called by: main() (step F2)
#   Calls:     Project.batch(), Project.host(), Project.kspacing(), Project.production_param(),
#              atoms_to_dict(), big_cell(), build_defect(), castep_frames(), check_training(),
#              draw_panels(), filter_frames(), format_table(), host_supercell(), job_energy(),
#              lineshape_vector(), load_cell(), mace_python(), named_rng(), pl_spectrum(),
#              production_job(), qtag(), run_training(), size_matrix(), start_spin(),
#              train_command(), trained_model(), write_columns()
def step_lineshape(p):
    if not p.s.lineshape:
        return {"skipped": True, "_note": "lineshape : false (F1's 1D lineshapes only)"}
    import ase.io
    picks = p.out("interest")["for_F"]
    if not picks:
        return {"lines": {}, "_note": "E4 sent no excitation on to F"}
    d = p.phase_dir("F")
    e1, ccd = p.out("relaxations"), p.out("ccd")
    cands, prim = p.out("generate")["candidates"], p.host()
    Pe = np.array(e1["cell"]["matrix"])
    amp = p.s.lineshape_amplitude
    jobs, meta = [], {}
    for k in range(p.s.lineshape_host):
        name = f"host_r{k + 1}"
        a = host_supercell(prim, Pe)
        a.positions += named_rng(name).normal(0.0, amp, a.positions.shape)
        o = dict(p.s.production_overrides, charge=0)
        o.setdefault("spin_polarized", "false")
        o["spin"] = None
        jobs.append({"name": name, "atoms": a, "kind": "sp", "symmetry": False, "overrides": o})
        meta[name] = (None, 0, 0)
    states = sorted({(pk["cand"], pk["q"]) for pk in picks})
    for cid, q in states:
        run = e1["runs"][f"{cid}_{qtag(q)}"]
        g = load_cell(p, run["final"])
        info = build_defect(prim, Pe, cands[cid]["comps"], cid)[1]
        S = start_spin(g, q, run["spin_final"])
        for k in range(p.s.lineshape_displacements):
            name = f"{cid}_{qtag(q)}_d{k + 1:03d}"
            a = g.copy()
            a.positions += named_rng(name).normal(0.0, amp, a.positions.shape)
            job, _ = production_job(p, name, a, info, q, S)
            job.pop("cell_extra", None)
            job["kind"] = "sp"
            jobs.append(job)
            meta[name] = (cid, q, S)
    res = p.batch(f"{PHASE_DIRS['F']}/jobs/lineshape_data", jobs, "lineshape",
                  p.production_param(), spacing=p.kspacing())
    if not res["complete"]:
        return None
    hosts = []
    for name, (cid, q, S) in meta.items():
        if cid is None and job_energy(res, name) is not None:
            hosts += castep_frames(res["calcs"][name].castep, 0, 0.0, name)
    tasks, data = [], {}
    for cid in sorted({c for c, _ in states}):
        frames = []
        for name, (c, q, S) in meta.items():
            if c == cid and job_energy(res, name) is not None:
                frames += castep_frames(res["calcs"][name].castep, q, S, name)
        for name, r in sorted(e1["runs"].items()):
            f = p.dir / PHASE_DIRS["D"] / "jobs" / "relaxations" / name / f"{name}.castep"
            if r["cand"] == cid and r["energy"] is not None and f.exists():
                frames += castep_frames(f, r["q"], r["spin_final"], name)
        kept = filter_frames(frames) + hosts
        order = np.random.default_rng(MODEL_SEED).permutation(len(kept))
        ntest = int(round(MODEL_TEST_FRACTION * len(kept)))
        folder = d / "model" / cid
        folder.mkdir(parents=True, exist_ok=True)
        split = {"train": [kept[i] for i in order[ntest:]],
                 "test": [kept[i] for i in order[:ntest]]}
        for s, sel in split.items():
            if sel:
                ase.io.write(str(folder / f"pbe_{s}.xyz"), sel, format="extxyz")
            elif (folder / f"pbe_{s}.xyz").exists():
                (folder / f"pbe_{s}.xyz").unlink()
        if not split["train"]:
            die(f"F2: no training data for the model of {cid} (did its single points finish?)")
        cmd = train_command(p, [("pbe", bool(split["test"]), False)])
        (folder / "train.sh").write_text(f"#!/bin/bash\n# The PBE model of {cid} for its "
                                         f"phonons (dwf.py F2)\n{cmd}\n")
        check_training(folder, cmd)
        tasks.append({"cid": cid, "dir": folder, "command": cmd, "heads": ["pbe"]})
        data[cid] = {"structures": len(kept), "host": len(hosts), "test": len(split["test"])}
    if not run_training(p, "F2", tasks):
        return None
    models = {t["cid"]: trained_model(t["dir"]) for t in tasks}
    Pbig = (Pe if p.s.lineshape_length <= e1["cell"]["L"] else
            np.array(size_matrix(p, prim, ("L", p.s.lineshape_length))))
    me = str(Path(__file__).resolve())
    ph, plan, skipped = [], {}, {}
    for pk in picks:
        cid, q = pk["cand"], pk["q"]
        tag = f"{cid}_{qtag(q)}_ex{pk['spin'] + 1}"
        comps, label = cands[cid]["comps"], f"{cid} ({cands[cid]['name']})"
        run = e1["runs"][f"{cid}_{qtag(q)}"]
        big, bv = big_cell(prim, comps, Pe, load_cell(p, run["final"]), Pbig, label)
        vec, n, why = lineshape_vector(p, pk, run, prim, comps, Pe, bv,
                                       big.get_chemical_symbols(), label)
        if vec is None:
            skipped[tag] = why
            continue
        folder = d / "phonons" / tag
        folder.mkdir(parents=True, exist_ok=True)
        test = models[cid].parent / "pbe_test.xyz"
        spec = {"model": str(models[cid].resolve()), "head": "pbe", "charge": q,
                "method": p.s.lineshape_method, "delta": p.s.lineshape_delta,
                "fmax": LINESHAPE_FMAX, "steps": LINESHAPE_STEPS,
                "model_md5": hashlib.md5(models[cid].read_bytes()).hexdigest(),
                "test_file": str(test.resolve()) if test.exists() else None,
                "atoms": atoms_to_dict(big), "vector": np.round(vec, 8).tolist()}
        spec["key"] = hashlib.md5(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        sf = folder / "spec.json"
        if sf.exists() and json.loads(sf.read_text()).get("key") != spec["key"]:
            for old in ("result.json", "hessian.npy", "hessian.json", "relaxed.json",
                        "job.json"):                    # a new calculation: start again
                if (folder / old).exists():
                    (folder / old).unlink()
        spec.update(device=p.s.model_device or p.s.mace_device, threads=p.s.mace_threads)
        sf.write_text(json.dumps(spec))
        cmd = (f"env OMP_NUM_THREADS={p.s.mace_threads} {shlex.quote(mace_python(p))} "
               f"{shlex.quote(me)} --phonon-worker spec.json")
        ph.append({"cid": tag, "dir": folder, "command": cmd, "done": folder / "result.json",
                   "what": "phonon calculation", "log": "phonons.log", "job": "phonons"})
        plan[tag] = (pk, n, len(big))
    if not run_training(p, "F2", ph):
        return None
    T, sig, gam = p.s.lineshape_temperature, p.s.lineshape_sigma, p.s.lineshape_gamma
    lines, rows = {}, []
    for tag, (pk, n, nat) in plan.items():
        r = json.loads((d / "phonons" / tag / "result.json").read_text())
        sp = pl_spectrum(r["hw"], r["S"], pk["zpl"], T, sig, gam)
        dq = math.sqrt(sum(x * x for x, hw in zip(r["dq"], r["hw"]) if hw > LINESHAPE_HW_MIN))
        one = ccd["lines"].get(tag) or {}
        out = d / "lineshape"
        files = {"S": write_columns(out / f"{tag}_S.dat", "hw/eV  S(hw)/(1/eV)",
                                    [sp["w"], sp["Sw"]]),
                 "pl": write_columns(out / f"{tag}_pl.dat",
                                     f"E/eV  PL (multi-mode, {T:g} K, max 1)", [sp["E"], sp["PL"]])}
        win = (sp["E"] > pk["zpl"] - 8 * max(sp["S"], 1) * (sp["hw_eff"] or 0.05) - 0.3) & (
            sp["E"] < pk["zpl"] + 0.15)
        series = [{"x": sp["E"][win], "y": sp["PL"][win], "label": "multi-mode", "colour": 0}]
        if one.get("hw_g") and one.get("S_g"):
            s1 = pl_spectrum([one["hw_g"]], [one["S_g"]], pk["zpl"], T, sig, gam)
            w1 = (s1["E"] >= sp["E"][win].min()) & (s1["E"] <= sp["E"][win].max())
            series.append({"x": s1["E"][w1], "y": s1["PL"][w1], "label": "1D (F1)",
                           "colour": 1})
        wsel = sp["w"] <= max(r["hw"] + [0.05]) + 5 * sig
        png = draw_panels(out / f"{tag}_lineshape.png", f"{tag}: {pk['name']} q {pk['q']:+d}",
                          [{"title": "spectral function", "xlabel": "phonon energy (eV)",
                            "ylabel": "S(ħω) (1/eV)",
                            "series": [{"x": sp["w"][wsel], "y": sp["Sw"][wsel], "colour": 0}]},
                           {"title": f"luminescence ({T:g} K)", "xlabel": "energy (eV)",
                            "ylabel": "intensity (max 1)", "mark": pk["zpl"],
                            "series": series}])
        if png:
            files["png"] = png
        lines[tag] = {"cand": pk["cand"], "name": pk["name"], "q": pk["q"], "atoms": nat,
                      "carried": n, "S": sp["S"], "DW": sp["DW"], "DW_T": sp["DW_T"],
                      "hw_eff": sp["hw_eff"], "dq": round(dq, 4), "dq_E2": pk["dq"],
                      "S_1D": one.get("S_g"), "imaginary": r["imaginary"],
                      "fmax": r["fmax"], "errors": r.get("errors"),
                      "files": {k: str(v.relative_to(p.dir)) for k, v in files.items()}}
        err = r.get("errors") or {}
        rows.append([tag, pk["name"], nat, err.get("F"), r["imaginary"], sp["S"], one.get("S_g"),
                     sp["hw_eff"], sp["DW"]] + ([sp["DW_T"]] if T > 0 else [])
                    + [round(dq, 3), pk["dq"]])
    table = format_table(["excitation", "defect", "atoms", "test F/meV/A", "imag.", "S",
                          "S (1D)", "hw_eff/eV", "DW 0K"] + ([f"DW {T:g}K"] if T > 0 else [])
                         + ["dQ", "dQ D2"], rows,
                         "Multi-mode lineshapes (the model's phonons in the big cell, "
                         f"lineshape_method {p.s.lineshape_method}; dQ in amu^0.5 A)")
    if skipped:
        table += "\nNot computed: " + "; ".join(f"{t} ({w})" for t, w in skipped.items()) + "\n"
    if any(x["imaginary"] for x in lines.values()):
        log("WARNING: F2: some big cells have imaginary modes (see the table): the model's "
            "minimum is not a minimum there; they are left out of the lineshape")
    return {"data": data, "big_cell": np.asarray(Pbig).tolist(), "lines": lines,
            "skipped": skipped, "_table": table,
            "_note": f"{len(lines)} multi-mode lineshape(s)"}


# What a F2 phonon job runs (python3 dwf.py --phonon-worker spec.json in its folder, written by
# step_lineshape), with the defect's model (MACECalculator, its 'pbe' head, the total charge in
# atoms.info): (1) its errors on the model's test structures (energy meV/atom, force components
# meV/Ang); (2) the big cell relaxed (mace_run, to LINESHAPE_FMAX; kept in relaxed.json); (3) its
# force constants by central differences (each atom and direction moved by +-delta in turn, saved
# to hessian.npy every LINESHAPE_SAVE s, so a job stopped at its time limit carries on where it
# was); (4) the modes: the dynamical matrix M^-1/2 Phi M^-1/2 diagonalised, each mode's energy
# hw_k (eV; negative: an imaginary mode) and how many atoms take part (1 / sum_i p_i^2, p_i its
# weight on atom i); (5) each mode's displacement between the two states, dQ_k = e_k.M^-1/2 dF /
# w_k^2 (method forces: the harmonic minimum dR = Phi^-1 dF) or e_k.M^1/2 dR (displacements),
# and its partial Huang-Rhys factor S_k = hw_k dQ_k^2 / (2 hbar^2) (0 for the modes below
# LINESHAPE_HW_MIN: the cell's translations, and the imaginary ones). Writes result.json.
#   Called by: main()
#   Calls:     atoms_from_dict(), atoms_to_dict(), mace_run(), write_json()
def phonon_worker(path):
    import warnings
    warnings.filterwarnings("ignore")
    import torch
    from mace.calculators import MACECalculator
    path = Path(path).resolve()
    folder, spec = path.parent, json.loads(path.read_text())
    torch.set_num_threads(spec["threads"])
    calc = MACECalculator(model_paths=spec["model"], head=spec.get("head"), device=spec["device"],
                          default_dtype="float64")
    out = {"errors": None}
    if spec.get("test_file") and Path(spec["test_file"]).exists():
        import ase.io
        de, df = [], []
        for a in ase.io.read(spec["test_file"], ":"):
            b = a.copy()
            b.info["charge"] = a.info.get("total_charge", 0)
            b.calc = calc
            de.append((b.get_potential_energy() - a.info["REF_energy"]) / len(a))
            df.append((b.get_forces() - a.arrays["REF_forces"]).ravel())
        out["errors"] = {"E": round(1000 * float(np.sqrt(np.mean(np.square(de)))), 2),
                         "F": round(1000 * float(np.sqrt(np.mean(np.square(np.concatenate(df))))),
                                    2)}
    rf = folder / "relaxed.json"
    if rf.exists():
        atoms = atoms_from_dict(json.loads(rf.read_text()))
    else:
        log(f"phonon worker: relaxing {len(spec['atoms']['symbols'])} atoms with the model")
        atoms = mace_run(calc, atoms_from_dict(spec["atoms"]), spec["fmax"], spec["steps"],
                         spec["charge"])[0]
        write_json(rf, atoms_to_dict(atoms))
    work = atoms.copy()
    work.info["charge"] = spec["charge"]
    work.calc = calc
    out["energy"] = float(work.get_potential_energy())
    out["fmax"] = round(float(np.linalg.norm(work.get_forces(), axis=1).max()), 5)
    n3, delta = 3 * len(work), spec["delta"]
    hf, hj = folder / "hessian.npy", folder / "hessian.json"
    done = json.loads(hj.read_text())["done"] if hj.exists() and hf.exists() else 0
    H = np.load(hf) if done else np.zeros((n3, n3))
    pos0, last = work.positions.copy(), time.time()
    log(f"phonon worker: force constants, {n3 - done} of {n3} columns to go")
    for c in range(done, n3):
        i, x = divmod(c, 3)
        work.positions[i, x] = pos0[i, x] + delta
        fp = work.get_forces().ravel()
        work.positions[i, x] = pos0[i, x] - delta
        fm = work.get_forces().ravel()
        work.positions[i, x] = pos0[i, x]
        H[:, c] = -(fp - fm) / (2 * delta)
        if time.time() - last > LINESHAPE_SAVE or c == n3 - 1:
            np.save(folder / "hessian.tmp.npy", H)
            (folder / "hessian.tmp.npy").replace(hf)
            write_json(hj, {"done": c + 1})
            last = time.time()
    H = 0.5 * (H + H.T)
    rm = 1.0 / np.sqrt(np.repeat(work.get_masses(), 3))
    lam, vec = np.linalg.eigh(H * rm[:, None] * rm[None, :])
    hw = HW_UNIT * np.sign(lam) * np.sqrt(np.abs(lam))
    v = np.asarray(spec["vector"], float).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        dq = (vec.T @ (rm * v)) / lam if spec["method"] == "forces" else vec.T @ (v / rm)
    keep = hw > LINESHAPE_HW_MIN
    dq = np.where(keep, dq, 0.0)
    S = np.where(keep, hw * dq * dq / (2 * HW_UNIT ** 2), 0.0)
    w2 = (vec * vec).reshape(len(work), 3, n3).sum(axis=1)
    out.update(natoms=len(work), imaginary=int((hw < -LINESHAPE_HW_MIN).sum()),
               hw=hw.round(6).tolist(), dq=dq.round(6).tolist(), S=S.round(8).tolist(),
               atoms_per_mode=(1.0 / (w2 * w2).sum(axis=0)).round(2).tolist())
    write_json(folder / "result.json", out)
    log(f"phonon worker: done; S = {S.sum():.3f}, {out['imaginary']} imaginary modes")


# ==========================================================================================
# ALL THE STEPS, IN ORDER
# ==========================================================================================
STEPS = [
    # A: the host
    Step("converge", "A1", "converge the k-point spacing, then the cut-off", step_converge,
         keys=("converge", "converge_cutoffs", "converge_kgrid_min", "converge_kgrid_max",
               "converge_kgrid_step", "converge_tol",
               "converge_kcutoff", "kspacing", "energy"), files=("host_cell", "param"),
         owns=("A_host/jobs/converge", "A_host/production.param")),
    Step("relax", "A2.1", "relax the host (cell and positions)", step_relax,
         keys=("energy",), files=("host_cell",), needs=("converge",),
         owns=("A_host/jobs/relax", "A_host/host_relaxed.cell")),
    Step("bands", "A2.2", "band edges on a Gamma-centred grid", step_bands,
         keys=("bands_kspacing",), needs=("relax",), owns=("A_host/jobs/bands",)),
    Step("dielectric", "A2.3", "dielectric constant (DFPT)", step_dielectric,
         keys=("dielectric", "dielectric_inf", "dielectric_task"),
         files=("dielectric_param", "dielectric_cell"), needs=("relax",),
         owns=("A_host/jobs/dielectric",)),
    Step("ref_phases", "A3.1", "reference energies at the screening settings (the host, and the "
         "pure elements of the host and the dopants)", step_ref_phases,
         keys=("phases", "phases_from", "phases_e_above_hull", "phases_max_steps",
               "phases_molecule_box", "dopants", "screen_cutoff", "screen_overrides", "energy",
               "ref_relax", "ref_kspacing_factor", "ref_polymorph_window"),
         needs=("relax",), owns=("A_host/jobs/ref_phases",),
         version=2),          # 1: the pure elements only; 2: doped's phases as symmetrised
                              # primitive cells without the moments' noise, the smallest
                              # polymorph within ref_polymorph_window, single points (ref_relax)
    Step("ref_limits", "A3.2", "chemical-potential limits at the screening settings",
         step_ref_limits, keys=("dopants",), needs=("ref_phases",)),
    # B: candidate defects and the screening cell
    Step("generate", "B1", "candidate defects", step_generate,
         keys=("dopants", "defects", "vacancies", "antisites", "interstitials", "pairs",
               "pair_max_distance", "native_pairs", "dopant_pairs"), needs=("relax",)),
    Step("scan", "B2", "supercell sizes, and the MACE scan of every candidate", step_scan,
         keys=("mace", "scan", "scan_lengths", "scan_tol", "scan_edge_tol", "scan_test",
               "mace_model", "mace_fmax", "mace_steps", "mace_rattle", "transform"),
         needs=("generate", "relax"),
         owns=("B_candidates/mace", "B_candidates/jobs/mace_scan", "B_candidates/ladder.json"),
         version=2),                         # 1: each size's relaxed MACE energy (B3's cut);
                                             # 2: the unconverged re-seeded from their deepest
                                             # minimum, and too few sizes told apart
    Step("choose", "B3", "the screening cell", step_choose,
         keys=("supercell", "transform", "screen_min_image", "deep_level_margin", "scan_lengths",
               "decay_length_multiple", "configuration_margin", "mace", "scan_unconverged"),
         needs=("scan", "relax", "bands", "generate"),
         version=2),                         # 1: configuration_margin's cut first;
                                             # 2: the unconverged dropped (scan_unconverged)
    Step("electronic", "B4", "the electronic cell (every band-edge valley at Gamma)",
         step_electronic, keys=("electronic_cell", "electronic_depth", "transform",
                                "decay_length_multiple"),
         needs=("choose", "bands", "relax")),
    # C: screening, ground and excited states
    Step("prerelax", "C1", "build the candidates and pre-relax them with MACE", step_prerelax,
         keys=("mace", "mace_model", "mace_fmax", "mace_steps", "mace_rattle",
               "configuration_margin"),
         needs=("generate", "scan", "choose", "relax"),
         owns=("C_screen/structures/candidates", "C_screen/jobs/mace_prerelax")),
    Step("screen", "C2", "cheap CASTEP screening of every candidate", step_screen,
         keys=SCREEN_KEYS + ("screen_charges", "screen_spins"),
         needs=("prerelax", "converge", "bands"),
         owns=("C_screen/jobs/screen", "C_screen/structures/screen"),
         version=1),                               # 1: the band carriers of each run
    Step("filter", "C3", "defect hull, sites and property filters", step_filter,
         keys=SCREEN_KEYS + ("hull_margin", "site_margin", "require_gap_states", "require_spin",
                             "require_bound", "distort_charges", "charge_correction", "excite",
                             "gap", "charge_margin", "carrier_reach", "carrier_fraction"),
         needs=("screen", "dielectric", "prerelax", "ref_limits"),
         owns=("C_screen/jobs/filter",),
         version=4),          # 1: the relevant charge states (band states judged); 2: the sites
                              # and least formation energies (A3.2's limits); 3: formation0 of
                              # every candidate (for a C4 since gone); 4: no cut on the
                              # formation energy
    Step("distort", "C4", "ShakeNBreak's distortions of every charge state, relaxed with MACE "
         "and pooled; the lowest distinct minima kept", step_distort,
         keys=("distortions", "distort_around", "rattle_stdev", "rattle_dmin",
               "oxidation_states", "distort_cell", "distort_starts", "mace", "mace_model",
               "mace_fmax", "mace_steps"),
         needs=("filter", "prerelax", "electronic"),
         owns=("C_screen/structures/distortions", "C_screen/mace", "C_screen/jobs/mace_distort"),
         version=3),                         # 2: in the electronic cell (distort_cell); 3: MACE
    Step("drelax", "C5", "Gamma-point CASTEP relaxations of each charge state from its screened "
         "structure and C4's MACE minima", step_drelax,
         keys=SCREEN_KEYS + ("distortion_min_gain",),
         needs=("distort", "filter", "generate"),
         owns=("C_screen/jobs/drelax", "C_screen/structures/drelax"),
         version=3),                         # 2: the ranking in summary.txt (the same results);
                                             # 3: C4's MACE minima
    Step("rank", "C6", "rank the ground states (C5's relaxations, with sharing)", step_rank,
         keys=SCREEN_KEYS + ("distortion_min_gain", "share_rounds", "keep_structures"),
         needs=("distort", "drelax", "choose", "filter"),
         owns=("C_screen/jobs/share_*", "C_screen/structures/share"),
         version=4),        # 1: each kept structure's run and band carriers; 2: its cell's
                            # matrix; 3: its cell's tag; 4: in C4's cell (distort_cell)
    Step("excite", "C7", "triplet ground states and their excitations (ΔSCF) in the electronic "
         "cell", step_excite,
         keys=SCREEN_KEYS + ("excite", "excite_window", "excite_min", "excite_relax",
                             "excite_overrides", "excite_spin", "spin_tol", "local_fraction",
                             "excite_reach", "excite_min_dw"),
         needs=("rank", "electronic", "choose", "generate", "relax", "converge"),
         owns=("C_screen/jobs/excite_relax", "C_screen/jobs/excite_gs", "C_screen/jobs/excite",
               "C_screen/structures/excite_relax", "C_screen/structures/excite"),
         version=1),                         # 1: the electronic cell, localised levels, triplets
    Step("validate", "C8", "finite-size check of the screening cell's corrections (optional)",
         step_validate, keys=("validate", "validate_tol", "validate_defects", "validate_sizes",
                              "validate_gamma", "elastic_constants", "production_overrides",
                              "charge_correction", "energy"),
         needs=("rank", "choose", "converge", "relax", "dielectric", "generate", "scan"),
         owns=("C_screen/jobs/validate", "C_screen/jobs/validate_gamma",
               "C_screen/structures/validate")),
    # D: production cells and relaxations
    Step("relaxations", "D1", "production relaxations of every charge state, electronic cell",
         step_relaxations, keys=("production_overrides", "freeze_radius", "energy"),
         needs=("excite", "electronic", "choose", "converge", "relax", "generate"),
         owns=("D_production/jobs/relaxations", "D_production/structures/relaxations")),
    Step("zpl", "D2", "production ΔSCF of the excitations C7 passed (ZPL, dQ)", step_zpl,
         keys=("production_overrides", "excite_overrides", "excite_relax", "excite_min",
               "excite_window", "excite_min_dw", "excite_reach", "local_fraction",
               "band_spread_tol", "energy"),
         needs=("relaxations", "generate", "relax", "converge"),
         owns=("D_production/jobs/zpl_gs", "D_production/jobs/zpl_gs_gamma",
               "D_production/jobs/zpl", "D_production/structures/zpl")),
    # E: thermodynamics
    Step("phases", "E1.1", "relax the competing phases at the production settings", step_phases,
         keys=("phases", "phases_from", "phases_e_above_hull", "phases_max_steps",
               "phases_molecule_box", "dopants", "chemical_potentials", "energy"),
         needs=("relaxations", "generate", "relax"), owns=("E_thermo/jobs/phases",)),
    Step("limits", "E1.2", "chemical-potential limits", step_limits,
         keys=("chemical_potentials", "dopants"),
         needs=("phases", "relaxations", "generate", "relax")),
    Step("corrections", "E2", "finite-size corrections", step_corrections,
         keys=("charge_correction",), needs=("relaxations", "dielectric")),
    Step("formation", "E3", "formation energies, levels, Fermi level per dopant set",
         step_formation, keys=("gap", "fermi_temperature", "quench_temperature", "dopants"),
         needs=("corrections", "relaxations", "bands", "limits", "generate"),
         owns=("E_thermo/formation.png",)),
    Step("interest", "E4", "which excitations go on to the lineshape (F)", step_interest,
         keys=("deep_level_margin",), needs=("formation", "zpl", "excite")),
    # F: optical lineshapes
    Step("ccd", "F1", "1D configuration-coordinate diagrams and lineshapes", step_ccd,
         keys=("ccd_points", "production_overrides", "excite_overrides", "lineshape_temperature",
               "lineshape_sigma", "lineshape_gamma", "energy"),
         needs=("interest", "zpl", "relaxations", "generate", "relax", "converge"),
         owns=("F_lineshape/jobs/ccd_gs", "F_lineshape/jobs/ccd_ex", "F_lineshape/ccd")),
    Step("lineshape", "F2", "multi-mode lineshapes (a PBE model's phonons in a big cell)",
         step_lineshape,
         keys=("lineshape", "lineshape_displacements", "lineshape_amplitude", "lineshape_host",
               "lineshape_length", "lineshape_method", "lineshape_delta",
               "lineshape_temperature", "lineshape_sigma", "lineshape_gamma",
               "production_overrides", "energy") + MODEL_KEYS,
         needs=("ccd", "interest", "zpl", "relaxations", "generate", "relax", "converge"),
         owns=("F_lineshape/jobs/lineshape_data", "F_lineshape/model", "F_lineshape/phonons",
               "F_lineshape/lineshape")),
]
STEP_INDEX = {s.name: s for s in STEPS}


EXAMPLE_PARAMETERS = """\
# Defect workflow project: the science of it (NAME.parameters). How its jobs run is in NAME.run,
# beside it, and dwf.py reads both:  nohup python3 dwf.py NAME > run.log 2>&1 &
# Written like a CASTEP .param/.cell file: "keyword : value", %block ... %endblock, keywords
# in any case, # or ! starts a comment. Lines starting with ! are optional settings with an
# example value: delete the ! to use one. File names are relative to this file: the example
# inputs (CaO:Sb on Cirrus) are in example_CaO_Sb/, these two files and every file they name.

# ======================================================================= general
host_cell : CaO_prim.cell      # host primitive cell: species_pot, symmetry_generate, k-points
param     : CaO.param          # production settings (functional, cut-off, ...); dwf.py
                               #   sets task, charge and spin, and A1 sets cut_off_energy
# Supercell sizes are lengths (scan_lengths, B2): each is the most compact supercell of the host
# with every periodic image at least that far away (found by doped). To use n x transform x
# (primitive cell) instead, the smallest n for each length, give a transform; this one gives
# cubic cells (8 n^3 atoms) from an FCC primitive cell:
!%block transform
!  -1  1  1
!   1 -1  1
!   1  1 -1
!%endblock transform
mace         : true            # MACE for the size scan (B2), the pre-relaxations (C1) and the
                               #   distortions' search (C4); false: C5 relaxes the distortions
!mace_model  : medium-mpa-0    # a mace_mp foundation model, or your own .model file
!mace_rattle : 0.02 ang        # B2, C1: the atoms within 4 A of the defect start moved at
                               #   random by about this, so that a symmetric start can relax

# ======================================================================= A: host
converge : true                # A1: converge the k-point grid, then the cut-off
converge_kgrid_min  : 4        # primitive-cell k-point grids from this to this in these steps
converge_kgrid_max  : 14       #   (n = n x n x n; three numbers each for a non-cubic grid,
converge_kgrid_step : 2        #   e.g. 4 4 2 / 12 12 6 / 2 2 1), run first ...
converge_kcutoff    : 400 eV   # ... at this cheap cut-off (default: the lowest cut-off below)
converge_cutoffs    : 400 500 600 700 800 900 eV   # then these, on the converged grid
converge_tol        : 0.001 eV/atom
!kspacing          : 0.04      # use this k-point spacing instead of converging it
!bands_kspacing    : 0.02      # A2.2 band edges (default: half the production spacing),
                               #   fine enough for B3's effective masses
# Dielectric constant for the charge corrections: DFPT phonon+efield, or give it yourself.
# CASTEP's DFPT needs norm-conserving potentials, fixed occupancies and no spin polarisation:
dielectric_param   : CaO_ncp.param   # its own .param (NCP cut-off, fix_occupancy : true)
dielectric_cell    : CaO_ncp.cell    # its own .cell (species_pot: NCP)
!dielectric        : 11.95     # static (electronic + ionic): skips the DFPT run
# Competing phases for the chemical potentials: A3.1 runs the host's and the dopants' elements at
# the screening settings (for C3), and E1.1 all of them, compounds too, for the dopants that
# reach D, at the production settings (for E3). Your own structure files...
%block phases
Ca      Ca.cell       metal  smearing_width=0.1  kspacing=0.03
O2      O2.cell       molecule  spin=2  spin_polarized=true
Sb      Sb.cell       metal  smearing_width=0.1  kspacing=0.03
Sb2O3   Sb2O3.cell
%endblock phases
# (add any other phase that could limit Sb, e.g. valentinite Sb2O3, Sb2O4, CaSb2O6,
#  Ca2Sb2O7 or a Ca-Sb compound: phases_from : doped would find them)
!phases_from : doped           # ...or from the Materials Project via doped (needs mp_api_key)
!mp_api_key  : <your key>
phases_max_steps : 100         # geometry steps for each phase's relaxation (CASTEP's own 30
                               #   can be too few for a soft metal's cell)
!phases_molecule_box : 15 ang  # the box for doped's molecules (H2, N2, O2, F2, Cl2): doped's
                               #   own 30 A has eight times the volume (default: 15 A)
# A3.1's pure elements are for C3's cuts only, which need them to tens of meV/atom:
!ref_relax : none              # none: single points of the structures as given (doped's: the
                               #   Materials Project's, PBE-relaxed); positions; cell
!ref_kspacing_factor : 2       # their k-point spacing, x A1's (2: an eighth of the k-points)
!ref_polymorph_window : 0.02 eV/atom   # of an element's polymorphs within this of the lowest
                               #   (doped's), the smallest cell
# ...or E's limits themselves (eV per atom, absolute: the same reference as your production
# runs; C3 then compares the defects within each composition only, unless phases are given)
!%block chemical_potentials
!O-rich   Ca=-1008.6  O=-432.6  Sb=-505.1
!O-poor   Ca=-1002.0  O=-439.2  Sb=-503.0
!%endblock chemical_potentials

# ======================================================================= B: candidates
dopants : Sb                   # extrinsic elements, as in the flowchart's B1
# A known defect (B1 'yes'): list it and generation is skipped. 'all' screens every
# arrangement within pair_max_distance, not just the nearest one; q= gives its charge states
# (e.g. q=-1, or q=-1,0), instead of screen_charges.
!%block defects
!Sb_Ca + Vac_O   all   q=-1
!%endblock defects
vacancies     : true           # native vacancies
antisites     : false
interstitials : none           # none | dopants | all (Voronoi sites, needs pymatgen)
pairs         : true           # dopant + native pairs ...
pair_max_distance : 5.0 ang    # ... up to this far apart
native_pairs  : false
dopant_pairs  : false          # true: any two dopant defects | same: two of the same dopant
# B2: the supercell sizes, as lengths (the shortest distance between periodic images), and
# the MACE scan of every candidate in each. A candidate is converged at the smallest size from
# which its MACE relaxation energy R lies on a line R_inf - c/N (N: the cell's atoms): that size
# and at least 3 bigger ones within scan_tol of it. c/N is the energy of the stress the fixed cell
# holds. B3 uses at most the fourth-largest (the line test needs 3 sizes above the one it
# picks), and drops a candidate that doesn't converge; B4 goes beyond the largest if it must.
scan          : true
scan_lengths  : 8 10 12 14 16 18 20 ang
scan_tol      : 0.02 eV
scan_test     : line           # line | flat: R within scan_tol of every bigger size's, and the
scan_edge_tol : 0.05 ang       #   atoms at the cell's edge moved less than this (flat only)
!scan_unconverged : drop       # drop | largest: what B3 does with a candidate B2 didn't converge
# B3: the screening cell is the smallest size that holds every converged candidate (B2), at least
# the Gamma-only floor: decay_length_multiple (below) times the decay length of the shallowest
# deep level (deep_level_margin, below), from the band edges' effective masses (A2.2).
!screen_min_image : 12 ang     # the floor yourself (0: none), instead of the estimate
!supercell : 14 ang            # fix the screening cell (a length, or n: n x transform)
# B4: the electronic cell, where everything from C4 on runs (but C8's test of the screening cell):
# the smallest size at least as big as the screening cell that folds every band-edge k-point
# (each valley's symmetry-equivalent copies too) onto Gamma, so a Gamma-point run sees the real
# gap, and is above the Gamma-only floor for a level electronic_depth from a band edge.
electronic_depth : 0.25 eV
!electronic_cell : 20 ang      # fix it: a length, n (n x transform) or screening
# B3, B4: the Gamma-only floor is this many decay lengths of the level it is for. At Gamma the
# images make a level that deep look deeper than it is: by about 10% at 4, 19% at 3.5 and 33%
# at 3 (more if the level's core is broad, less if it is compact), and deeper levels by much
# less. The cells' atoms go as its cube: 3 needs under half those of 4 (CaO's B4: n320, not n728).
decay_length_multiple : 4

# =============================================== C: screening, ground and excited states
screen_charges   : -1 0 1
screen_spins     : triplet     # the starting spin: triplet (a doublet for an odd electron
                               #   count) | low (0 or 1) | both (low, and low + 2)
!screen_cutoff   : 500 eV      # a lower cut-off for screening (default: production)
screen_max_steps : 50          # geometry steps per screening relaxation
%block screen_overrides        # any other .param keywords for the screening runs, and for
                               #   A3.1's phases
elec_energy_tol : 1e-5
!fine_gmax : 35 1/ang          # a finer density grid, for hard ultrasoft potentials at a low
                               #   screen_cutoff: a cell relaxation in A3.1 that keeps
                               #   reverting, as its energy jumps when the grid changes
%endblock screen_overrides
!configuration_margin : 1.0 eV # B3: drop arrangements this far above the lowest of their
                               # composition by MACE energy (B2's), before the screening cell
                               # is chosen (off unless given)
hull_margin   : 0.2 eV         # C3: keep arrangements within this of the defect hull
site_margin   : 0.5 eV         # C3: drop a defect this far above the lowest one with the
                               #   same dopant atoms at every growth condition and Fermi level
require_gap_states : none      # none | occupied | empty | both (filled and empty levels)
!require_spin : 2              # e.g. 2 = triplet ground state
require_bound : false          # bound against its isolated components
level_margin  : 0.1 eV         # gap levels lie further inside the band edges than this;
                               #   anything nearer or past them is in a band, and a run that
                               #   leaves an electron or hole there isn't judged in C3
distort_charges : relevant     # C3: the charge states carried on: relevant (the stable ones,
                               #   bound ones within charge_margin of stable, and band states
                               #   with a shallow carrier on the defect) | stable (the stable
                               #   ones, plus any in a band) | screened (every one)
charge_margin : 0.5 eV         #   a distortion may lower a charge state about this much
                               #   (CaO:Sb's C5: 0.34 eV for a bound one)
!carrier_reach    : 1.0 eV     #   a band state's carrier at most this far inside its band ...
!carrier_fraction : 0.2        #   ... and spread over at most this fraction of the cell's atoms
!distortions  : -0.4 -0.2 0.2 0.4  # C4: ShakeNBreak's bond distortions (default: its own,
                               #   -0.6 to 0.6 in steps of 0.1), each charge state's with its own
                               #   number of neighbours, all relaxed with MACE and pooled (with
                               #   mace : false, these four by default, each relaxed in C5)
distort_starts  : 2            # C5: the lowest distinct MACE minima each charge state is relaxed
                               #   from in CASTEP, beside its screened structure
distort_around  : auto         # auto: a single defect's site, a nearest-neighbour complex's
                               #   centre, else each component in turn | centre | each |
                               #   anchor (the first component, e.g. the dopant) | vacancy
distort_cell    : electronic   # where everything after C3 works: electronic (B4's cell;
                               #   C1-C3 screen every candidate in the smaller screening
                               #   cell) | screening (C4-C6 stay there)
!%block oxidation_states       # for the number of neighbours to distort
!Ca +2
!O  -2
!Sb +3
!%endblock oxidation_states
# C6: the ground states: C5's relaxations, with a winning structure tried in the other charge
# states.
distortion_min_gain : 0.05 eV  # a structure counts if it beats the screened one by this
share_rounds    : 2            # rounds of sharing winning structures between charge states
keep_structures : 1            # per charge state, on to C7
# C7: in the electronic cell, at the screening settings. Each charge state's ground state is
# relaxed: a triplet (spin within spin_tol of excite_spin) goes on, a band state is left
# 'pending' (PBE can't place it), any other only to E. Its defect levels are the bands localised
# on the defect (participation number at most local_fraction of the atoms), in the gap or up to
# excite_reach inside a band; in each spin channel the highest occupied one to the lowest empty
# one is excited (CASTEP's ΔSCF) and relaxed. It passes if its ZPL is at least excite_min (and
# inside excite_window), the promoted electron stays on the defect and the 1D Debye-Waller
# estimate is at least excite_min_dw. Defects that pass go on to D with all their charge states,
# and so does every native defect (for E). Test CASTEP's ΔSCF on one defect first.
excite         : true          # false: everything C6 kept goes on to D as it is
excite_spin    : 2             # CASTEP's spin: 2 = triplet; any: every spin state
!spin_tol      : 0.3
!local_fraction : 0.2
!excite_reach  : 1.0 eV
excite_min     : 0.4 eV
!excite_window : 0.3 3.5 eV    # pass only if the ZPL is in this window
!excite_min_dw : 0.05          # pass only if the 1D Debye-Waller estimate is at least this
excite_relax   : true          # relax the excited state (the ZPL); false: vertical energy
!%block excite_overrides       # any other .param keywords for C7's runs
!deltascf_smearing : 0.01
!%endblock excite_overrides
# C8 (optional): a test of the screening cell's finite-size error, which C3's hull rests on.
# Each defect named here is relaxed in each charge state C6 kept (its lowest structure; band
# states left out) at the production settings (production_overrides, with the stress on) in the
# screening cell and in the next validate_sizes sizes up. The line E_inf + b/N through the bigger
# sizes is the reference for three estimates from the screening cell: its E as it is, less
# MACE's c/N (B2), and less the energy of its own stress (needs elastic_constants). With
# validate_gamma, Gamma-point single points of the same structures show the Gamma-only error.
# Costly: the ladder's biggest cells, so name two or three defects.
!validate : true
!validate_defects : d001 d002  # B1's ids (default: every defect C6 kept)
!validate_sizes   : 3          # how many sizes above the screening cell
!validate_tol     : 0.05 eV    # how close each estimate must come (levels too)
!validate_gamma   : true
!%block elastic_constants      # the host's, GPa: C11 C12 C44 (cubic) or six lines of six;
!226.2 62.4 80.6               #   CaO at 298 K (Drago & Spain 1977); your functional's own
!%endblock elastic_constants   #   (e.g. PBE's) are better, as the stress comes from it

# ======================================================================= D: production
# D1: every charge state of each defect C7 passes (and of every native defect) relaxed at the
# production settings (A1's cut-off and k-points), in the electronic cell, with the pristine
# cell's single point. D2: the production ΔSCF of C7's excitations. A single point with the
# bands' weights first checks that each constrained band keeps its number at every k-point
# (inside the gap, spread less than band_spread_tol); where it doesn't, the ΔSCF runs at Gamma.
%block production_overrides
geom_energy_tol : 5e-6
elec_energy_tol : 5e-7
geom_max_iter   : 100
%endblock production_overrides
!band_spread_tol : 0.05 eV
freeze_radius : 0              # (untested idea) fix the atoms further than this (Ang) from
                               #   every component of the defect, in D1

# ======================================================================= E: thermodynamics
charge_correction : lz         # lz | mp | none
gap : 7.09 eV                  # band gap for the Fermi-level range of C3, E3 and E4 (default:
                               #   the screening cell's in C3, A2.2's in E): CaO's measured one
!fermi_temperature  : 1000 K   # self-consistent Fermi level at this temperature
!quench_temperature : 300 K    # ... with the defects frozen in, then at this one
deep_level_margin : 0.5 eV     # E4: a level this far from both band edges is "deep" (B3's
                               #   Gamma-only floor is for a level this deep)
# E3 finds the Fermi level of each set of dopants on its own (the native defects alone, with
# each dopant, and with each mix that one defect brings), and E4 sends on to F each excitation
# that passed D2 whose charge state is the lowest for some Fermi level in the gap.

# ======================================================================= F: lineshapes
# F1: the 1D configuration-coordinate diagram of each excitation E4 sends on: ground-state and
# ΔSCF single points along the line from the relaxed ground state (0) to the relaxed excited
# state (1), at D2's settings; parabolas give each state's phonon energy and Huang-Rhys factor,
# and the 1D luminescence lineshape.
!ccd_points : -0.4 -0.2 0 0.2 0.4 0.6 0.8 1 1.2 1.4
lineshape_temperature : 10 K   # the temperature of both lineshapes
!lineshape_sigma : 0.006 eV    # Gaussian width of each phonon line
!lineshape_gamma : 0.002 eV    # Lorentzian half-width of the ZPL
# F2 (optional): the multi-mode lineshape. A PBE model of each defect (one head, the charge
# embedded), trained on lineshape_displacements displaced copies of its ground state (normal,
# lineshape_amplitude a component), its D1 relaxations and lineshape_host rattled pristine
# cells, all at the production settings; its phonons in a cell lineshape_length across (finite
# differences, lineshape_delta), projected onto the excited state's forces at the ground
# state's structure (lineshape_method forces) or onto the displacement between the relaxed
# states (displacements). The trainings and phonons run with model_jobscript (in NAME.run),
# the model's size and training set here:
!lineshape : true
!lineshape_displacements : 30
!lineshape_amplitude : 0.02 ang
!lineshape_host   : 5
!lineshape_length : 30 ang     # CaO: about 1900 atoms, 11,500 model force evaluations
!lineshape_method : forces     # forces | displacements
!model_epochs   : 300
!model_channels : 64           # the model's size (with model_r_max 5 ang, model_max_L 1,
                               #   model_batch 4)
!model_train_args : --swa      # anything else for mace_run_train
"""

EXAMPLE_RUN = """\
# Defect workflow project: how its jobs run (NAME.run), here on Cirrus. The science is in
# NAME.parameters, beside it, and dwf.py reads both:  nohup python3 dwf.py NAME > run.log 2>&1 &
# Written like a CASTEP .param/.cell file: "keyword : value", %block ... %endblock, keywords
# in any case, # or ! starts a comment. Lines starting with ! are optional settings with an
# example value: delete the ! to use one. File names are relative to this file.

!workdir  : dwf_CaO_Sb         # where everything goes (default dwf_<name>, beside these files)

# ======================================================================= CASTEP jobs
# How the CASTEP jobs run: any runner keyword (python3 dwf.py --conv-example lists them).
# These are for Cirrus, whose nodes (288 cores) are shared between jobs, charged by the core:
jobscript      : cirrus.sh     # template with {SEED}, {NODES}, {TASKS_PER_NODE}, {CPUS_PER_TASK}
cores_per_node : 288
charging       : core          # a job asks for just the cores it needs (node: whole nodes)
max_cores      : 64            # most MPI processes per job (more or fewer for some steps: below)
!max_walltime  : 12:00:00      # the longest a job may run (default: the template's --time)
%block partitions              # memory comes with the cores (GB per core): each job goes on
standard   2.66                #   the first where one core holds a process, else gets more
highmem    5.33                #   cores per process where that takes the fewest
%endblock partitions
dryrun_time : 0:20:00          # dry runs: small jobs (one process, cores for 4 GB)
!dryrun_on  : login            # ... or run them where dwf.py runs (dryrun_cmd)
max_jobs  : 16                 # most jobs of one step queued at once
!array    : 500                # jobs planned alike as Slurm job arrays of up to this many
                               #   tasks (each still counts in max_jobs; with pack, a pack each)
max_restarts : 20              # a relaxation that runs out of time carries on this many times
poll      : 300
delete_check : true            # delete each .check once its calculation has finished
# ... and for some steps only, in a runner block: runner_host for all of phase A's CASTEP
# steps (A1 to A3.1) and the competing phases (E1.1), runner_screen for the defects at the
# screening settings (C2, C3's single points, C5, C6, C7) and runner_production for those at
# the production settings (C8, D1, D2, F1, F2's single points); for one step: runner_converge
# (A1), runner_relax (A2.1), runner_bands (A2.2), runner_dielectric (A2.3), runner_ref_phases
# (A3.1), runner_filter (C3), runner_drelax (C5), runner_rank (C6), runner_excite (C7),
# runner_validate (C8), runner_relaxations (D1), runner_zpl (D2), runner_phases (E1.1),
# runner_ccd (F1), runner_lineshape (F2); the MACE jobs have blocks of their own (below). A
# step's own block wins over its group's, and that wins over the settings above, for the
# keywords they give (a %block partitions in one replaces the whole list).
# Starting points for CaO:Sb on Cirrus - set your own:
%block runner_host             # A1 to A3.1, E1.1: the primitive cell and the competing
                               #   phases
max_cores    : 32
max_walltime : 4:00:00
%endblock runner_host
%block runner_converge         # A1: single points of the 2-atom primitive cell
max_cores    : 16
max_walltime : 0:20:00
%endblock runner_converge
%block runner_screen           # C2, C3, C5, C6, C7: Gamma-only screening jobs
max_restarts : 0               # (the default for screening: a job that fails is flagged)
%endblock runner_screen
%block runner_drelax           # C5: relaxations in the electronic cell, phase C's biggest
max_cores    : 288
max_walltime : 24:00:00
%endblock runner_drelax
%block runner_production       # C8, D1, D2, F1, F2: the production settings, electronic cell
max_cores    : 288             # up to a node (more: whole nodes, e.g. 576 for two)
%endblock runner_production
# On a machine that charges whole nodes, such as ARCHER2: jobscript : archer2.sh,
# cores_per_node : 128, charging : node (the default) with max_nodes (nodes per job) in place
# of max_cores, %block partitions with standard 2 --qos=standard and highmem 4 --qos=highmem,
# and dryrun_on : login with dryrun_cmd : castep.serial --dryrun {seed} (ARCHER2's serial
# partition rejects whole-node jobs); pack : N then runs N small jobs side by side on a node.

# ======================================================================= MACE jobs
mace_threads : 8               # threads per MACE relaxation
# The MACE relaxations run where dwf.py runs, unless mace_jobscript is given: then as jobs on
# compute nodes, each running up to mace_pack relaxations side by side (default
# cores_per_node / mace_threads, 36 here) and asking for their cores ({NCORES} in the
# template: 8 per relaxation), in at most mace_max_jobs jobs at a time.
mace_jobscript : cirrus_mace.sh    # a Slurm template with a {COMMAND} line
!mace_max_jobs : 4
!mace_python   : /work/<project>/<project>/<you>/venv/bin/python   # the Python the jobs run
                               #   (default: the one running dwf.py; it must be on /work,
                               #   as the compute nodes can't see /home)
# B2 relaxes every candidate in every size it fits, which for a big screen is thousands of
# relaxations, shared out over at most mace_max_jobs jobs: give it more jobs, and the time they
# need, in its own block (runner_prerelax and runner_distort do the same for C1 and C4). Such a
# block may give mace_jobscript, mace_pack, mace_max_jobs, mace_threads, mace_python,
# mace_device and max_walltime, each job's --time (default: the template's):
!%block runner_scan
!mace_max_jobs : 40
!max_walltime  : 12:00:00
!%endblock runner_scan

# ======================================================================= F2's model jobs
model_jobscript : cirrus_gpu.sh    # F2's training and phonons as GPU jobs; without
model_device    : cuda             #   it, they run where dwf.py runs
"""


if __name__ == "__main__":
    main()
