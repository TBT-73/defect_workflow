#!/usr/bin/env python3
"""
supercell_convergence.py
========================
Supercell-size convergence of a defect formation energy with CASTEP, using ASE to
build the supercells.

For each supercell size n the script
  1. builds the supercell n*T of the primitive cell (T = identity unless transform),
  2. inserts the defect (substitutions and/or vacancies). Any components after the
     first go on their nearest sites to it, with the same relative geometry in every
     supercell, and the defect is put as close to the cell centre as it can be: the
     mean position of its components nearest the centre (the size table shows how
     far off it is). Defect cells are run without symmetry (symmetry_generate is
     removed), so the relaxation is free to break the starting symmetry,
  3. sets a Monkhorst-Pack grid with the same k-point spacing as the primitive cell,
  4. runs a CASTEP dry run (as a small batch job on a compute node, by default) to
     get the irreducible k-points and the memory estimate,
  5. picks a core count suited to k-point parallelisation and (with %block
     partitions) a partition with enough memory, writes a jobscript from your
     template and submits a geometry optimisation of the defect cell (ions only;
     the lattice is fixed at the bulk value with fix_all_cell),
  6. once it has converged, runs a single point on the relaxed structure with the same
     k-points and cores (singlepoint : false skips this and uses the final energy of the
     relaxation itself) and, with bulk_mode : supercell, one on each pristine supercell,
  7. reads the defect energy and computes
         E_f*(n) = E_defect(n) - N_host(n) * e_bulk + E_corr(n)
     i.e. the formation energy without the chemical-potential and Fermi-level terms
     (identical for every supercell, so they don't affect convergence). E_corr is 0 for
     a neutral defect; for a charged one it is the image-charge correction
     (charge_correction : lz or mp, with dielectric), see "Charged defects" below,
and stops once |E_f*(n) - E_f*(previous n)| < tol.

Usage
-----
Everything is set in an input file, written like a CASTEP .param/.cell file:
"keyword : value" lines and %block ... %endblock blocks, keywords in any case,
# or ! to start a comment. Run it with the seedname, like CASTEP:

    python3 supercell_convergence.py CaO_SbCa                  # reads CaO_SbCa.conv
    python3 supercell_convergence.py CaO_SbCa --prepare-only   # first time: check inputs

    python3 supercell_convergence.py --example > CaO_SbCa.conv   # annotated example

Run-control switches go on the command line (or in the file as e.g. no_wait : true):
    --prepare-only     build the inputs and jobscripts, submit no calculations
    --no-wait          submit what is needed, report and exit instead of waiting
    --resubmit-failed  resubmit the existing jobscript of a failed calculation once
    --force            reuse a work directory that was set up with different inputs

Use the relaxed primitive cell and the energy per atom from that same calculation
(same cut-off, functional and pseudopotentials).

Charged defects
---------------
Set the same charge (e.g. charge : -1) in the .param and the geom_param; the script
removes it for the pristine cells, reads the charge each defect ran with from its
.castep, and stops if the single point and its relaxation differ. A charged defect
interacts with its own periodic images and the neutralising background, an error that
shrinks only as 1/L, so E_f* gets a correction:
    Makov-Payne (mp):  E1 = q^2 alpha_M / (2 eps L)
    Lany-Zunger (lz):  [1 + c_sh (1 - 1/eps)] E1   (about 2/3 of E1)
alpha_M/L is computed by an Ewald sum for the actual supercell, eps is dielectric
(static, electronic + ionic) and c_sh is -0.369, -0.343, -0.342 for simple cubic,
FCC and BCC supercells (about -1/3 otherwise). No potential alignment is included.

Jobscript template
------------------
Copy the jobscript you normally use for CASTEP and put {SEED} where the seedname
goes and {NCORES} where the number of cores (MPI processes) goes. On machines that
hand out whole nodes, set cores_per_node and use {NODES} and {TASKS_PER_NODE}:
    #SBATCH --nodes={NODES}
    #SBATCH --ntasks-per-node={TASKS_PER_NODE}
    #SBATCH --cpus-per-task={CPUS_PER_TASK}
With omp_threads, a job that uses fewer processes per node to fit the memory runs
OpenMP threads on the cores left over: the script sets OMP_NUM_THREADS (and, if the
template lacks them, OMP_PLACES, SRUN_CPUS_PER_TASK and --cpus-per-task).
Other placeholders: {JOBNAME} {NKPTS} {PARTITION} {MEM_GB} {MEM_MB} {MEM_PER_NODE_GB}
{MEM_PER_CORE_GB} {MEM_PER_CORE_MB} (estimated memory in total, per node and per
process, including the safety factor). Give the template as a file
(jobscript : job_template.sh) or inline in %block jobscript.

Whole-node machines
-------------------
With cores_per_node, each job uses up to max_nodes nodes, with the process count
chosen for k-point parallelisation (e.g. 126 of 128 cores per node for 14 k-points).
A job that needs more memory per process than a full node gives goes to the next
partition (e.g. highmem); only if none has enough are fewer processes put on each
node ({CPUS_PER_TASK} then spreads them out).

Partitions
----------
With %block partitions, each job goes on the first partition listed whose memory
per core is enough for it (e.g. multicore, and himem only when a job needs more).
The script sets #SBATCH --partition (or fills {PARTITION}) and adds any extra #SBATCH
options given for that partition, e.g. --qos=highmem, or --mem={MEM_GB}G where
memory is requested separately.

Memory
------
The memory per process is estimated from the serial dry run as
    (baseline + other + S * (r + (1 - r) / p)) * mem_safety
with S the dry run's 'Model' + 'Electronic' memory, p the number of processes and
r = mem_replicated, the fraction of S that every process keeps its own copy of. With
mem_replicated : auto (the default) r starts at 0.25 and is then measured from the
'Peak Memory Use' CASTEP prints at the end of each run (a relaxation that stopped at
run_time counts too); the value from the biggest cell measured so far is used for every
calculation planned after that, including jobs that are resubmitted.

Walltimes
---------
The #SBATCH --time in the template (or a --time= on a partition line) is the longest
any job may run. With walltime : auto (the default) a job asks for less once finished
jobs show what it needs: their time per SCF cycle is scaled to the new cell as
k-points x electrons^a / processes (a fitted from the finished sizes, 3 until there
are two) and multiplied by the SCF cycles of the biggest finished relaxation, then by
walltime_safety. Until a job has finished there is nothing to scale from, so the first
jobs get the full --time; set sp_walltime to give the first single points (e.g. the
pristine cells) a shorter one, so they wait less in the queue (one that runs out is
resubmitted with the full --time). Each relaxation gets CASTEP's run_time a little below
its walltime, so it stops cleanly and writes a .check before the limit; the script then
resubmits it with continuation : default and the longest walltime until it converges (at most
max_restarts times). A job killed at the time limit is resubmitted the same way. If
CASTEP can't read the .check (e.g. it was cut short while being written), the .check is
put aside and the relaxation starts again from the structure of its last geometry step
in the .castep. A resubmitted job is planned again first, so it gets the best layout for
the memory measured so far. With delete_check : true, a calculation's .check file is
deleted once it has finished successfully (it is only needed to carry on an unfinished
run, and can be tens of GB), and one CASTEP can't read is deleted straight away.

MACE pre-relaxation
-------------------
With prerelax : mace each defect supercell is first relaxed with a MACE foundation
model (positions only, the cell fixed at the bulk lattice) by this script, on the
machine it runs on, and CASTEP's relaxation starts from that structure. The final
structure and energy are still CASTEP's: MACE only gives it a better starting point,
so it needs fewer geometry steps. Needs the mace-torch package (Python 3.9+) in the
Python that runs this script. The result is kept in prerelax.json and the optimiser's
log in prerelax.log, in each relaxation's directory.

Dry runs
--------
By default each dry run is submitted as its own job (dryrun.sh), made from the same
template: {NCORES} becomes dryrun_cores (default 1) and the CASTEP line becomes
    mpirun -n 1 castep.mpi --dryrun <seed>
so CASTEP always runs as one process (the memory model expects a serial estimate).
If the template's partition needs more cores (e.g. multicore needs 2+), send the dry
runs elsewhere with dryrun_partition : serial, and shorten their walltime with
dryrun_time : 0-1. For anything else give a separate template in dryrun_jobscript
(use {SEED} and write --dryrun yourself). dryrun_on : login runs them where the
script runs instead, using dryrun_cmd.

The script waits for the jobs, so run it in tmux/screen or with nohup. It can be
stopped and rerun with the same command at any time: finished and queued
calculations are picked up rather than repeated. To stop it from another login node,
create a file called STOP in the work directory (touch <workdir>/STOP); it says at the
start which node it runs on.

Batch mode
----------
The same machinery can run any list of CASTEP calculations instead of a size series.
The defect-workflow scripts (dwf_*.py) use it for every CASTEP step, and you can use it
directly:

    python3 supercell_convergence.py --batch jobs.batch

The batch file holds the runner settings of a .conv file (jobscript, partitions,
cores_per_node, max_nodes, walltime, max_restarts, dryrun_*, ...), the .param and .cell
every job starts from (param, cell; the structure, task and k-points are set per job),
kspacing, and one line per job in %block jobs:
    name  structure-file  [kind=geom|sp] [task=...] [fix_cell=false] [symmetry=true]
                          [kgrid=1,1,1] [offset=gamma] [any .param keyword=value]
e.g.  host  CaO_prim.cell  kind=geom fix_cell=false symmetry=true   (a variable-cell
relaxation) or  scr1  cand1.cell  kind=geom kgrid=1,1,1 charge=-1 spin=2. Each job gets
its own directory, dry run, plan, walltime and restarts, as in a size series; jobs that
would give the same dry run share one. max_jobs limits how many jobs are queued at once,
and with tolerate_failures : true a job that fails is marked failed (in its job_info.json)
while the others carry on; --resubmit-failed starts the failed ones again. The results
are written to batch_results.json in the work directory (batch_<file name> by default).
From Python: run_batch(workdir, jobs, args, cell_text, param_text, ...) (see its comment).
"""

import argparse
import difflib
import hashlib
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
from ase.data import chemical_symbols
from ase.geometry import cellpar_to_cell, get_distances

# ==========================================================================================
# HOW THE FUNCTIONS FIT TOGETHER
# ==========================================================================================
# main() runs everything (see the comment above it). In outline:
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
# Batch mode (--batch, and the dwf_*.py workflow scripts) runs a list of calculations instead:
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

if sys.version_info < (3, 7):   # Python 3.6: subprocess.run has no capture_output/text
    _subprocess_run = subprocess.run

    # Stand-in for subprocess.run on Python 3.6.
    # Python 3.6's subprocess.run has no capture_output or text arguments. This translates them
    # (stdout/stderr = PIPE, universal_newlines) and replaces subprocess.run, so the script's
    # subprocess.run calls work on 3.6 as well. Only defined on Python < 3.7.
    #   Called by: every subprocess.run call (local_dryrun, submit, job_active,
    #              killed_at_time_limit)
    def _run(*args, capture_output=False, text=False, **kwargs):
        if capture_output:
            kwargs["stdout"] = kwargs["stderr"] = subprocess.PIPE
        return _subprocess_run(*args, universal_newlines=text, **kwargs)

    subprocess.run = _run

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
NPROC_VARS = re.compile(r"\$\{?(SLURM_NTASKS|SLURM_NPROCS|NSLOTS|PBS_NP)\}?")
# a filled-in jobscript line that runs CASTEP (not a comment or a module command)
CASTEP_LINE = re.compile(r"^(?!\s*#)(?!\s*module\b).*(?<![\w/])castep[\w.\-]*\s+\S", re.M)
OMP_AUTO_MAX = 4                       # most OpenMP threads per process with omp_threads : auto

# Regenerated for every supercell; everything else in the primitive .cell
# (species_pot, ...) is copied across unchanged.
STRUCT_BLOCKS = {"lattice_cart", "lattice_abc", "positions_frac", "positions_abs"}
KPOINT_KEYS = {"kpoint_mp_grid", "kpoints_mp_grid", "kpoint_mp_spacing",
               "kpoints_mp_spacing", "kpoint_mp_offset", "kpoints_mp_offset"}
SET_KEYS = KPOINT_KEYS | {"fix_all_cell"}       # written by the script instead
# Atom-indexed / primitive-cell-specific blocks that would be wrong in a supercell
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


# Reads any CASTEP-style text into a list of items.
# Used for your .cell and .param(s), the .conv input file and CASTEP's -out.cell. Each item is a
# block {'type': 'block', 'name', 'body'}, a keyword {'type': 'kw', 'key', 'value', 'raw'} or any
# other line {'type': 'raw'}. The original lines are kept, so render() can write the items back
# exactly as they were.
#   Called by: cell_options(), final_atoms(), last_geometry(), main(), param_charge(),
#              read_batch_file(), read_input(), read_structure(), relaxed_atoms(), run_batch()
#   Calls:     strip_comment()
def parse_castep_input(text):
    """Split a .cell/.param file into blocks, keywords and other lines."""
    items, lines, i = [], text.splitlines(), 0
    while i < len(lines):
        s = strip_comment(lines[i])
        m = re.match(r"%block\s+(\w+)", s, re.I)
        if m:
            name, body = m.group(1).lower(), []
            i += 1
            while i < len(lines) and not re.match(r"%endblock", strip_comment(lines[i]), re.I):
                body.append(lines[i])
                i += 1
            if i == len(lines):
                die(f"unterminated %block {name}")
            items.append({"type": "block", "name": name, "body": body})
        else:
            m = re.match(r"(\w+)\s*[:=]?\s*(.*)", s)
            if m:
                items.append({"type": "kw", "key": m.group(1).lower(),
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
#   Called by: convert_settings(), main(), read_batch_file(), read_primitive()
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
#   Called by: main(), parse_defect(), read_primitive()
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
#   Called by: final_atoms(), last_geometry(), main(), read_structure(), relaxed_atoms()
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
# CASTEP keeps this order in its output, so relaxed_atoms and last_geometry use it to match
# CASTEP's atoms back to the script's Atoms objects.
#   Called by: final_atoms(), last_geometry(), relaxed_atoms(), write_cell()
def written_order(atoms):
    """Atom order used by write_cell (grouped by species), which CASTEP keeps."""
    syms = atoms.get_chemical_symbols()
    species = list(dict.fromkeys(syms))
    return sorted(range(len(atoms)), key=lambda i: (species.index(syms[i]), i))


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
#   Called by: main(), write_inputs()
#   Calls:     recip_lengths()
def kgrid(cell, spacing):
    return [max(1, math.ceil(L / spacing - 1e-4)) for L in recip_lengths(cell)]


# Decides the k-point spacing used for every supercell.
# In order of priority: kspacing in the input file; kgrid in the input file; kpoint_mp_grid in your
# .cell; kpoint_mp_spacing in your .cell; CASTEP's default (0.1 1/Ang, with a warning). For a grid
# it takes the coarsest spacing of the three directions. Returns (spacing, where it came from). The
# spacing goes into ctx['spacing'] and the work directory's fingerprint (setup.json).
#   Called by: main()
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
#   Called by: main()
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
#   Called by: translation_copies(), the dwf_*.py scripts
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
#   Called by: parse_defect(), the dwf_*.py scripts
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
#   Called by: main()
def defect_label(comps):
    return "-".join(c["name"].replace(":", "s") for c in comps)   # without any @spin


# Makes the supercell P x (primitive cell) with ASE's make_supercell.
# Each atom is first tagged with the index of the primitive-cell site it came from (array
# 'prim_index'), so candidates() can pick out e.g. 'copies of the 2nd O site'. Used for every size
# (P = n*T) and for the 3x3x3 reference cell in defect_vectors.
#   Called by: defect_vectors(), main()
def build_supercell(prim, P):
    p = prim.copy()
    p.set_array("prim_index", np.arange(len(p)))   # remember which primitive site each atom is
    return make_supercell(p, P)


# The atoms of a supercell where one defect component can go.
# Atoms of the host element; if the component names a primitive-cell site, only copies of that site
# and of the sites identical to it by a pure translation (prim_sites), using 'prim_index' from
# build_supercell.
#   Called by: anchor_site(), defect_vectors(), insert_defect(), main()
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
#   Called by: defect_vectors(), insert_defect(), main(), the dwf_*.py scripts
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
#   Called by: main()
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
#   Called by: main()
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
#   Called by: main()
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
# (measure_times), measured memory (calibrate) and restart counts (restart). main() makes one Calc
# per calculation; most other functions take a Calc. In batch mode (run_batch) a Calc is one job of
# the batch: kind 'geom' (any relaxation) or 'sp' (anything else), with its own name and options.
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
    #   Called by: main(), run_batch()
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

    # Reads job_info.json (an empty dict if it doesn't exist yet).
    #   Called by: almost every function that plans, submits or checks a calculation
    def info(self):
        p = self.dir / INFO
        return json.loads(p.read_text()) if p.exists() else {}

    # Writes job_info.json.
    # Functions read it with info(), change it and save it back.
    #   Called by: every function that records something in job_info.json
    def save(self, info):
        (self.dir / INFO).write_text(json.dumps(info, indent=2))

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
    # already done.
    #   Called by: calibrate(), job_result(), main(), measure_times(), prepare_all(), prepare_sp(),
    #              remove_check(), run_batch(), wait_for()
    #   Calls:     Calc.output()
    def done(self):
        """Finished, and for a geometry optimisation also converged."""
        out = self.output()
        return bool(FINISHED.search(out)) and (self.kind != "geom" or GEOM_DONE in out)

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
#   Called by: dryrun_script(), finalize()
def fill(template, subs):
    for key, val in subs.items():
        template = template.replace(key, str(val))
    return template


# Sets one #SBATCH option in a jobscript, e.g. '--partition=highmem'.
# Replaces the option wherever it already appears, in either spelling (-p/--partition, -t/--time,
# ...: SBATCH_SHORT), removing duplicates; otherwise adds it after the last #SBATCH line. A memory
# request (--mem, --mem-per-cpu, ...) replaces any other memory request.
#   Called by: dryrun_script(), finalize(), restart(), set_threads()
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
#   Called by: _slurm_time(), job_timing(), plan_walltime(), restart()
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
#   Called by: plan_walltime()
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


# Writes the text of dryrun.sh, the batch job that runs CASTEP --dryrun.
# With dryrun_jobscript, that template is just filled in. Otherwise your main template is used,
# with the line that runs CASTEP changed to one process with --dryrun (CASTEP_RUN, NPROC_VARS) and
# dryrun_cores requested. dryrun_partition, dryrun_time and dryrun_options are then set with
# add_sbatch_option. prepare_all writes the file and submits it.
#   Called by: prepare_all()
#   Calls:     add_sbatch_option(), fill()
def dryrun_script(calc, ctx, args):
    """Dry-run jobscript: dryrun_jobscript if given, otherwise the main template with
    dryrun_cores requested (on one node) and the CASTEP line run as one process with
    --dryrun. dryrun_partition, dryrun_time and dryrun_options then set those #SBATCH
    options."""
    gb, n = args.dryrun_mem_gb, args.dryrun_cores
    subs = {"{SEED}": calc.seed, "{JOBNAME}": calc.name + "_dry", "{NKPTS}": 0, "{NCORES}": n,
            "{NODES}": 1, "{TASKS_PER_NODE}": n, "{CPUS_PER_TASK}": 1,
            "{MEM_GB}": math.ceil(gb), "{MEM_MB}": math.ceil(gb * 1024),
            "{MEM_PER_NODE_GB}": math.ceil(gb),
            "{MEM_PER_CORE_GB}": math.ceil(gb / n), "{MEM_PER_CORE_MB}": math.ceil(gb * 1024 / n),
            "{PARTITION}": args.dryrun_partition or (args.partitions[0]["name"]
                                                     if args.partitions else "")}
    if ctx["dryrun_template"]:
        script = fill(ctx["dryrun_template"], subs)
    else:
        lines = []
        for line in ctx["job_template"].splitlines():
            if CASTEP_RUN.search(line) and not line.lstrip().startswith("#"):
                line = CASTEP_RUN.sub(r"\1--dryrun {SEED}", line).replace("{NCORES}", "1")
                line = NPROC_VARS.sub("1", line)
            lines.append(line)
        script = fill("\n".join(lines) + "\n", subs)
    if args.dryrun_partition:
        script = add_sbatch_option(script, f"--partition={args.dryrun_partition}")
    if args.dryrun_time:
        script = add_sbatch_option(script, f"--time={args.dryrun_time}")
    for option in args.dryrun_options:
        script = add_sbatch_option(script, fill(option, subs))
    return script


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
#   Called by: choose_resources()
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
#   Called by: job_timing(), last_peak(), scf_seconds()
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
# number of atoms.
#   Called by: main(), restart(), run_batch()
#   Calls:     Calc.done(), measured_replicated()
def calibrate(calcs, sizes, args):
    """mem_replicated : auto. Work out r from the latest finished run of every job (a
    relaxation that stopped at run_time and is carrying on counts), then use the largest r
    among the jobs of the biggest size measured so far (the most like the cells to come)."""
    if not args.r_auto:
        return
    found = []
    for n in sizes:
        for c in calcs[n].values():
            info = c.info()
            if not info.get("jobid"):
                continue
            r, peak, p = measured_replicated(c, args)
            if [info.get(k) for k in ("r_measured", "peak_mb", "peak_procs")] != [r, peak, p]:
                info.update(r_measured=r, peak_mb=peak, peak_procs=p)
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
#   Called by: main(), run_batch()
#   Calls:     Calc.done(), job_timing()
def measure_times(calcs, sizes):
    """Record how long each newly finished job took (for walltime : auto)."""
    for n in sizes:
        for c in calcs[n].values():
            info = c.info()
            if "timing" in info or not info.get("jobid") or not c.done():
                continue
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
# from.
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
        i = c.info()
        return (i.get("timing") or {}).get("per_cycle") and i.get("nelec") and i.get("ncores")

    timed = [c for c in ctx["calcs"] if ok(c) and c.group == calc.group]
    pool = [c for c in timed if c.kind == calc.kind] or (
        [c for c in timed if c.kind == "geom"] if calc.kind != "geom" else [])
    if not pool or not nelec:
        return None, None, None
    own = [c for c in timed if c.kind == "geom" and c.label == calc.label]
    if calc.kind == "defect" and own:    # single point on a relaxed cell: same speed per cycle
        t = own[0].info()["timing"]
        sps = [c for c in pool if c.kind == "defect"]
        cycles = max(c.info()["timing"]["cycles"] for c in sps) if sps else t["first_scf_cycles"]
        return t["per_cycle"] * own[0].info()["ncores"] / procs * cycles, None, own[0]
    info = {c.name: c.info() for c in pool}

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
# The --time already in the jobscript (from the template or a partition line) is the longest
# allowed. With walltime : auto and finished jobs to learn from, it asks for estimate_walltime x
# walltime_safety instead, rounded up to 15 minutes (at least WALLTIME_MIN), if that is shorter.
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
                "relaxations can't stop cleanly before the time limit - add one (the longest "
                "a job may run)")
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
    secs = max(WALLTIME_MIN, math.ceil(est * args.walltime_safety / 900) * 900)
    if secs < smax:
        plan.time, plan.secs, plan.note = slurm_time(secs), secs, f" (estimated from {ref.name})"
    else:
        plan.note = (f" (estimated {est * args.walltime_safety / 3600:.0f} h from {ref.name}"
                     + (", so it will continue in further jobs)" if calc.kind == "geom" else
                        " - more than the walltime allows: give it more nodes)"))
    return plan


# CASTEP's run_time for a job with a given walltime (s).
# The walltime minus a margin, so CASTEP stops cleanly between geometry steps and writes its .check
# before Slurm stops the job. The margin is the larger of 900 s and 1.5 x the time of one SCF (5%
# of the walltime while the SCF time isn't known), times a factor that restart doubles after a job
# is killed anyway, and never more than half the walltime.
#   Called by: finalize(), restart()
def run_time_for(walltime, per_scf=None, factor=1.0):
    """CASTEP run_time (s) for a job with this walltime (s): early enough to finish the SCF
    under way and write the .check before the scheduler stops the job."""
    margin = max(900.0, 1.5 * per_scf if per_scf else 0.05 * walltime) * factor
    return int(walltime - min(margin, 0.5 * walltime))


# Sets keywords in a calculation's .param, e.g. run_time and continuation.
# Removes any existing line for each keyword and appends the new value (a value of None just
# removes the keyword). Everything else in the file is left as it is, so changes you make by hand
# to a calculation's .param survive restarts. Keywords must be given in lower case.
#   Called by: finalize(), restart(), write_inputs()
#   Calls:     strip_comment()
def set_param(calc, **values):
    """Set keywords in the calc's .param (None removes the keyword)."""
    path = calc.dir / f"{calc.seed}.param"
    lines = []
    for line in path.read_text().splitlines():
        m = re.match(r"\s*(\w+)", strip_comment(line))
        if not (m and m.group(1).lower() in values):
            lines.append(line)
    lines += [f"{key} : {val}" for key, val in values.items() if val is not None]
    path.write_text("\n".join(lines) + "\n")


# The number of cores, on machines without whole nodes (no cores_per_node).
# k-point parallelism first: with no more k-points than max_cores, all of them run in parallel and
# the rest is filled with G-vector parallelism (a multiple of the k-points); otherwise the fewest
# cores that give the minimum number of k-points per core. 'cores : N' forces N.
#   Called by: choose_resources()
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
#   Called by: choose_resources(), node_layouts()
def layout_key(nk, nodes, tpn):
    """Rank of a whole-node layout (lower is better): the fewest k-points per process, then
    a process count that splits evenly with the k-points, then the most processes, then
    the fewest nodes."""
    n = nodes * tpn
    return (math.ceil(nk / n), n % nk != 0 and nk % n != 0, -n, nodes)


# Every layout of up to max_nodes nodes x cores_per_node processes, best first.
# Sorted by layout_key. With 'cores : N', only layouts with exactly N processes.
#   Called by: choose_resources()
#   Calls:     layout_key()
def node_layouts(nk, cpn, max_nodes, fixed=None):
    """All (nodes, processes per node) with up to cpn processes per node, best first."""
    cands = [(nodes, t) for nodes in range(1, max_nodes + 1) for t in range(1, cpn + 1)
             if not fixed or nodes * t == fixed]
    return sorted(cands, key=lambda c: layout_key(nk, *c))


# Decides how and where a job runs, from its dry run.
# Picks the number of processes (on whole-node machines: nodes x processes per node), the memory
# per process (mem_per_core_mb) and the partition. Partitions from %block partitions are tried in
# order and the first where the best layout fits in memory is used. If none fits with full nodes,
# it takes the partition where fewer processes per node give the best layout (the job is
# underpopulated); if nothing fits at all, the one that comes closest (finalize then logs a
# warning). Returns a namespace: nodes, tpn, ncores, mpc, part, fits, underpop, cpus, skipped.
#   Called by: finalize()
#   Calls:     choose_cores(), layout_key(), mem_per_core_mb(), node_layouts()
def choose_resources(dr, args):
    """Processes (and nodes), memory per process (MB) and partition for a calculation,
    from its dry run. Partitions are tried in the order given and the first where the job
    fits is used, otherwise the one that comes closest. With cores_per_node, if no
    partition fits with full nodes, the job runs with fewer processes per node, on the
    partition where that gives the best layout."""
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
                               skipped=[q for q in skipped if q is not p])

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
#   Called by: main(), prepare_all(), prerelaxed(), write_inputs()
def prerelax_settings(args):
    if args.prerelax != "mace":
        return None
    return {"model": args.prerelax_model, "fmax": args.prerelax_fmax,
            "rattle": args.prerelax_rattle}


# Imports torch and mace_mp, stopping with installation instructions if mace-torch is missing.
# main() calls it at the start with prerelax : mace, so a missing package is found straight away
# rather than halfway through.
#   Called by: mace_calculator(), main()
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
# (anchor), which main() keeps if the script is run again.
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
# submitted together and waited for together (wait_for with key 'dryrun_jobid'). With shared dry
# runs (batch mode), a calculation whose dry run would be the same as another's (dryrun_key) gets a
# copy of that one instead of its own. show=False plans without logging each plan. Returns False if
# --no-wait and dry runs are still queued.
#   Called by: main(), run_batch()
#   Calls:     Calc.done(), dryrun_output(), dryrun_script(), finalize(), local_dryrun(),
#              prerelax_settings(), submit(), wait_for(), write_inputs()
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

    queued, leaders = [], {}
    for c in calcs:
        if c.done() or c.info().get("jobid"):
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
            submit(c, args, DRYRUN_SCRIPT, "dryrun_jobid")
        if key:
            leaders.setdefault(key, c)
        queued.append(c)
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
# per process and partition; the template is filled in (fill), OpenMP threads are set for cores
# the layout leaves idle (thread_count, set_threads), and the partition, its #SBATCH options and
# the walltime (plan_walltime) are set with add_sbatch_option. For a relaxation,
# run_time is set in the .param (run_time_for, set_param), so it can stop cleanly and be carried on
# - except in a batch with max_restarts : 0 and tolerate_failures (screening), where nothing would
# carry it on: there it may use its whole walltime. The plan is saved in job_info.json and
# logged (show), with warnings about memory and odd electron counts unless quiet. Returns a
# one-line summary of the layout.
#   Called by: prepare_all(), prepare_sp(), restart(), run_batch()
#   Calls:     add_sbatch_option(), choose_resources(), fill(), parse_dryrun(), plan_walltime(),
#              run_time_for(), set_param(), set_threads(), thread_count()
def finalize(calc, text, ctx, args, quiet=False, show=True):
    """Choose cores, memory and partition from the dry run and write the real jobscript.
    quiet: no warnings; show: log the plan."""
    dr = parse_dryrun(text)
    grid = calc.info()["kgrid"]
    r = choose_resources(dr, args)
    ncores, mpc, part = r.ncores, r.mpc, r.part
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
                "with more.")
        elif not r.fits:
            log(f"WARNING: {calc.name} needs ~{mpc / 1024:.1f} GB per core on {ncores} cores, "
                f"more than mem_per_core ({args.mem_per_core} GB). It may run out of memory: "
                "raise max_cores or use high-memory nodes (see %block partitions).")
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
    if part:                                    # partition and its extra #SBATCH options
        script = add_sbatch_option(script, f"--partition={part['name']}")
        for option in part["options"]:
            script = add_sbatch_option(script, fill(option, subs))
    wall = plan_walltime(calc, script, dr, ncores, ctx, args)
    if wall.time != wall.max:
        script = add_sbatch_option(script, f"--time={wall.time}")
    (calc.dir / JOBSCRIPT).write_text(script)
    (calc.dir / JOBSCRIPT).chmod(0o755)
    if calc.kind == "geom" and wall.secs:       # stop cleanly before the walltime runs out
        if args.max_restarts == 0 and getattr(args, "tolerate_failures", False):
            set_param(calc, run_time=None)      # (screening: no restart would carry it on)
        else:
            set_param(calc, run_time=run_time_for(wall.secs, wall.per_scf))
    info = calc.info()
    info.update(nkpts=dr["nkpts"], ncores=ncores, mem_per_core_gb=round(mpc / 1024, 2),
                nelec=dr["nelec"], partition=part["name"] if part else None,
                nodes=r.nodes if args.cores_per_node else None, tasks_per_node=r.tpn,
                omp_threads=threads, dryrun_mem=dr["mem"],
                mem_replicated=round(args.mem_replicated, 4),
                walltime=wall.time, walltime_max=wall.max, scf_seconds=wall.per_scf)
    calc.save(info)
    if args.cores_per_node:
        size = (f"{r.nodes} node{'s' if r.nodes > 1 else ''} x {r.tpn} processes"
                + (f" x {threads} threads" if threads > 1 else ""))
        mem = f"~{mpc / 1024:.1f} GB/process (~{r.tpn * mpc / 1024:.0f} GB/node)"
    else:
        size = f"{ncores} cores"
        mem = f"~{mpc / 1024:.1f} GB/core (~{ncores * mpc / 1024:.0f} GB total)"
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
# Uses <seed>-out.cell if CASTEP wrote it, otherwise the last 'Fractional coordinates' table in the
# .castep. Matches CASTEP's atom order to the script's (written_order), checks the cell didn't
# change, and logs the largest displacement, how far CASTEP moved the atoms from the MACE start
# (prerelax.json) and the number of geometry steps. Returns the relaxed Atoms.
#   Called by: main(), prepare_sp()
#   Calls:     Calc.output(), atom_label(), displacements(), parse_castep_input(),
#              read_primitive(), written_order()
def relaxed_atoms(geom):
    """Relaxed defect cell: <seed>-out.cell if CASTEP wrote it, else the last fractional
    coordinates in the .castep. Checks the cell didn't change and reports the relaxation."""
    out_cell = geom.dir / f"{geom.seed}-out.cell"
    order = written_order(geom.atoms)              # CASTEP keeps the order we wrote
    start = geom.atoms[order]
    if out_cell.exists():
        atoms = read_primitive(parse_castep_input(out_cell.read_text(errors="replace")))
    else:
        tail = geom.output().rsplit("Fractional coordinates of atoms", 1)[-1]
        rows = ATOM_ROW.findall(tail)[:len(start)]
        atoms = Atoms([r[0] for r in rows], scaled_positions=[[float(v) for v in r[1:]]
                      for r in rows], cell=start.cell, pbc=True)
    if atoms.get_chemical_symbols() != start.get_chemical_symbols():
        die(f"could not read the relaxed structure of {geom.name} (atoms don't match)")
    if np.abs(np.asarray(atoms.cell) - np.asarray(start.cell)).max() > 1e-4:
        die(f"the cell of {geom.name} changed during the relaxation; it should stay fixed")
    atoms.set_cell(start.cell, scale_atoms=False)
    atoms.set_initial_magnetic_moments(start.get_initial_magnetic_moments())
    moved = displacements(start, atoms)
    note, pre = "", geom.dir / PRERELAX_FILE
    if pre.exists():                       # how far CASTEP went from the MACE structure
        mace = geom.atoms.copy()
        mace.set_scaled_positions(json.loads(pre.read_text())["scaled_positions"])
        off = displacements(mace[order], atoms)
        note = f", {off.max():.3f} A from the MACE start ({atom_label(atoms, int(off.argmax()))})"
    steps = len(GEOM_STEP.findall(geom.output()))
    log(f"{geom.name}: relaxed, largest atomic displacement {moved.max():.3f} A "
        f"({atom_label(atoms, int(moved.argmax()))}){note}"
        + (f", {steps} geometry steps" if steps else ""))
    return atoms


# Sets up the single point on a relaxed defect cell (singlepoint : true, or with geom_param).
# Takes the relaxed structure (relaxed_atoms), writes the inputs (write_inputs) and plans the job
# from the relaxation's dry run (finalize), so it runs with the same k-points and layout. Does
# nothing if the single point is already submitted or done; main() then submits it.
#   Called by: main()
#   Calls:     Calc.done(), finalize(), relaxed_atoms(), write_inputs()
def prepare_sp(sp, geom, ctx, args):
    """Single point on the relaxed defect cell, with the relaxation's k-points and cores."""
    if sp.done() or sp.info().get("jobid"):
        return
    sp.atoms = relaxed_atoms(geom)
    write_inputs(sp, ctx, args)
    finalize(sp, geom.dryrun_file.read_text(errors="replace"), ctx, args, quiet=True)


# Submits job.sh (or dryrun.sh) with submit_cmd and records the job in job_info.json.
# Saves the job id and submission time. For a real job it also appends a run to info['runs']: the
# job id, where its output will start in the .castep (the file's current size), its walltime and
# number of processes - run_segments, job_timing, last_peak and Calc.run_text depend on this
# record. Stops if the submission fails.
#   Called by: handle_failure(), main(), prepare_all(), restart(), run_batch()
#   Calls:     sbatch_value()
def submit(calc, args, script=JOBSCRIPT, key="jobid"):
    start = calc.castep.stat().st_size if calc.castep.exists() else 0   # where its output begins
    res = subprocess.run(shlex.split(args.submit_cmd) + [script], cwd=calc.dir,
                         capture_output=True, text=True)
    m = (re.search(r"Submitted batch job (\d+)", res.stdout) or
         re.search(r"Your job (\d+)", res.stdout) or re.search(r"(\d+)", res.stdout))
    if res.returncode != 0 or not m:
        die(f"could not submit {calc.dir / script}:\n{res.stdout}{res.stderr}")
    info = calc.info()
    info[key] = m.group(1)
    info["submitted" if key == "jobid" else "dryrun_submitted"] = time.strftime("%Y-%m-%d %H:%M:%S")
    wall = sbatch_value((calc.dir / script).read_text(), "--time")
    if key == "jobid":
        info["runs"] = ([r for r in info.get("runs", []) if r["start"] < start]
                        + [{"jobid": m.group(1), "start": start, "time": wall,
                            "procs": info.get("ncores")}])
    calc.save(info)
    if key == "jobid":
        size = (f"{info['nodes']} node{'s' if info['nodes'] > 1 else ''} x "
                f"{info['tasks_per_node']} processes" if info.get("nodes")
                else f"{info.get('ncores')} cores")
        part = f" ({info['partition']})" if info.get("partition") else ""
        log(f"submitted {calc.name}: job {info[key]} on {size}{part}"
            + (f", walltime {wall}" if wall else ""))
    else:
        log(f"submitted dry run for {calc.name} ({len(calc.atoms)} atoms): job {info[key]}")


# Asks the scheduler whether a job is still queued or running.
# Uses status_cmd, or squeue/qstat. A job that has left the queue, or shows a finished state
# (END_STATES), is not active; if the scheduler doesn't answer within 2 minutes the job is assumed
# to be still there.
#   Called by: wait_for()
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


# A size in bytes as 'x.y GB' or 'n MB', for log messages.
#   Called by: remove_check(), restart()
def file_size(nbytes):
    return f"{nbytes / 2**30:.1f} GB" if nbytes >= 2**30 else f"{nbytes / 2**20:.0f} MB"


# For delete_check : true: deletes the .check files of a calculation that finished successfully.
# Deletes <seed>.check and any .check.bad put aside by restart once Calc.done() is true, and logs
# the space freed. Does nothing for unfinished calculations, so a relaxation keeps the .check it
# carries on from.
#   Called by: main(), run_batch(), wait_for()
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
# has none, <seed>-out.cell. Maps CASTEP's atom order back onto calc.atoms (written_order). For a
# variable-cell relaxation (batch mode) the lattice of that step is taken too (last_lattice).
# Returns (Atoms, steps finished, the file it came from), or (None, steps, None).
#   Called by: restart()
#   Calls:     Calc.output(), last_lattice(), parse_castep_input(), read_primitive(),
#              written_order()
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
    order = written_order(calc.atoms)              # CASTEP keeps the order we wrote
    syms = [calc.atoms.get_chemical_symbols()[i] for i in order]
    head = text[:ends[-1]].rsplit("Fractional coordinates of atoms", 1)
    rows = ATOM_ROW.findall(head[-1]) if len(head) > 1 else []
    frac, source = [[float(x) for x in r[1:]] for r in rows[:len(order)]], calc.castep
    lattice = None if calc.fix_cell else last_lattice(text[:ends[-1]])
    if [r[0] for r in rows[:len(order)]] != syms:
        source = calc.dir / f"{calc.seed}-out.cell"
        try:
            out = read_primitive(parse_castep_input(source.read_text(errors="replace")))
        except (OSError, SystemExit, ValueError, IndexError):
            return None, len(ends), None
        if out.get_chemical_symbols() != syms:
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


# Wall time per SCF in the latest run CASTEP finished, and that run's number of processes.
# Used to set the next run's run_time margin, scaled to its new number of processes.
#   Called by: restart()
#   Calls:     run_segments()
def scf_seconds(calc):
    """Wall time per SCF in the latest run that CASTEP finished, and the number of
    processes it ran on, or (None, None)."""
    for run, text in reversed(run_segments(calc)):
        tot, nscf = TOTAL_TIME.findall(text), len(re.findall(ENERGY_PATTERNS["final"], text))
        if tot and nscf:
            return float(tot[-1]) / nscf, run.get("procs") or calc.info().get("ncores")
    return None, None


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
# - for a relaxation, carries on from the .check (continuation : default) when it can; otherwise
#   puts the .check aside (deletes it with delete_check : true) and writes the structure of the
#   last geometry step into the .cell (last_geometry + write_cell). After a second unreadable
#   .check it stops using .check files for that calculation;
# - for a single point, renames the .castep and starts again;
# - renames the .err files, asks for the longest walltime (add_sbatch_option), sets a new run_time
#   (run_time_for, scf_seconds; the margin doubles if the job was killed), counts the restart in
#   job_info.json and resubmits the job (submit).
#   Called by: handle_failure()
#   Calls:     Calc.err_files(), Calc.run_text(), add_sbatch_option(), calibrate(), cell_options(),
#              file_size(), finalize(), killed_at_time_limit(), last_geometry(), planning_dryrun(),
#              run_time_for(), sbatch_value(), scf_seconds(), set_param(), slurm_seconds(),
#              submit(), write_cell()
def restart(calc, args):
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
    more = "Give it more nodes (max_nodes) or a longer walltime (#SBATCH --time), then rerun."
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
    per_scf, scf_procs = scf_seconds(calc)
    old_procs = (info.get("runs") or [{}])[-1].get("procs") or info.get("ncores")
    old_layout = [info.get(k) for k in ("nodes", "tasks_per_node", "ncores", "partition")]
    had, longest = (slurm_seconds(info.get(k) or "") for k in ("walltime", "walltime_max"))
    for run in info.get("runs", []):               # runs submitted by an older version
        run.setdefault("procs", info.get("ncores"))
    if info.get("runs"):                           # for job_timing: count its walltime?
        info["runs"][-1]["end"] = "killed" if killed else "unreadable" if unreadable else "stopped"
    calc.save(info)

    calibrate(*args.all, args)                     # plan it again, with what's known now
    text, desc = planning_dryrun(calc, args.ctx), None
    if text:
        desc = finalize(calc, text, args.ctx, args, quiet=True, show=False)
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
        if smax:
            if per_scf and scf_procs and new_procs:
                per_scf *= scf_procs / new_procs
            new["run_time"] = run_time_for(smax, per_scf or info.get("scf_seconds"), margin)
        set_param(calc, **new)
    info.update(restarts=n + 1, walltime=tmax, margin_factor=margin)
    calc.save(info)
    how = ("from its .check file" if use_check else
           f"from the structure of its last geometry step (step {steps})" if steps else
           "from the structure in its .cell file" if geom else "from the start")
    log(f"{what} {why} - resubmitting it {how}" + (f" with walltime {tmax}" if tmax else ""))
    layout = [info.get(k) for k in ("nodes", "tasks_per_node", "ncores", "partition")]
    if desc and layout != old_layout:
        log(f"  {calc.name} planned again (mem_replicated {args.mem_replicated:g}): {desc}")
    submit(calc, args)
    return True


# Deals with a job or dry run that left the queue without finishing (or wrote a .err file).
# For a job, restart is tried first. Then, with --resubmit-failed, the existing jobscript is
# resubmitted once (the old .castep and .err files are renamed). Otherwise the script stops with an
# explanation: a relaxation that ran out of time with max_restarts : 0, one that reached
# geom_max_iter, or anything else (with the end of the .castep and the .err files).
#   Called by: wait_for()
#   Calls:     Calc.err_files(), Calc.output(), Calc.run_text(), restart(), submit()
def handle_failure(calc, args, resubmitted, key="jobid"):
    script = JOBSCRIPT if key == "jobid" else DRYRUN_SCRIPT
    what = f"{calc.name}{'' if key == 'jobid' else ' dry run'} (job {calc.info().get(key)})"
    if key == "jobid" and args.max_restarts and restart(calc, args):
        return
    if args.resubmit_failed and calc.name not in resubmitted:
        stamp = time.strftime("%Y%m%d-%H%M%S")
        for f in [calc.castep] + calc.err_files():
            if f.exists():
                f.rename(f.with_name(f"{f.name}.failed-{stamp}"))
        log(f"{what} failed - resubmitting the existing {script}")
        resubmitted.add(calc.name)
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
# (mark_failed) and counts as finished.
#   Called by: main(), prepare_all(), run_batch()
#   Calls:     Calc.done(), Calc.err_files(), dryrun_output(), handle_failure(), job_active(),
#              mark_failed(), remove_check()
def wait_for(calcs, args, key="jobid", any_done=False):
    """Poll until every calc's job (key='jobid') or dry run (key='dryrun_jobid') is done
    (with any_done, until at least one is). Returns False if --no-wait and some are still
    queued or running."""
    if key == "jobid":          # (a job marked failed, with tolerate_failures, has finished)
        is_done, what = (lambda c: c.done() or bool(c.info().get("failed"))), ""
    else:
        is_done, what = (lambda c: dryrun_output(c) is not None), " dry run"
    gone, resubmitted, announced = {c.name: 0 for c in calcs}, set(), False
    stop = getattr(args, "stop_file", None)
    while True:
        if stop and stop.exists():
            stop.unlink()
            log(f"found {stop.name} - stopping. Jobs already submitted keep running; rerun the "
                "same command to carry on.")
            sys.exit(0)
        waiting = []
        for c in calcs:
            if is_done(c):
                if key == "jobid":
                    remove_check(c, args)      # delete_check : true
                continue
            waiting.append(c)
            if c.err_files() or not job_active(c.info()[key], args):
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
            else:
                gone[c.name] = 0
        if not waiting:
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
# coordinates' table in the .castep (for a single point that is its starting structure). None if
# neither can be read. The workflow scripts use it to carry a relaxed structure on to the next step.
#   Called by: the dwf_*.py scripts
#   Calls:     Calc.output(), last_lattice(), parse_castep_input(), read_primitive(),
#              written_order()
def final_atoms(calc):
    order = written_order(calc.atoms)              # CASTEP keeps the order we wrote
    syms = [calc.atoms.get_chemical_symbols()[i] for i in order]
    out_cell, atoms = calc.dir / f"{calc.seed}-out.cell", None
    if out_cell.exists():
        try:
            atoms = read_primitive(parse_castep_input(out_cell.read_text(errors="replace")))
        except (SystemExit, ValueError, IndexError):
            atoms = None
    if atoms is None or atoms.get_chemical_symbols() != syms:
        text = calc.output()
        rows = ATOM_ROW.findall(text.rsplit("Fractional coordinates of atoms", 1)[-1])
        rows = rows[:len(order)]
        if [r[0] for r in rows] != syms:
            return None
        lattice = None if calc.fix_cell else last_lattice(text)
        atoms = Atoms(syms, scaled_positions=[[float(v) for v in r[1:]] for r in rows],
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
# the dwf_*.py workflow scripts and by --batch.
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
# share_dryruns (one dry run for all jobs that would give the same one: dryrun_key) and stop_file
# (default <workdir>/STOP).
# It works as the size series does: inputs, dry runs and plans first (prepare_all, at most max_jobs
# dry runs queued at once), then the jobs, submitted at most max_jobs at a time, each planned again
# just before it goes (so it uses the timings and memory measured so far), waited for (wait_for),
# restarted as usual. With tolerate_failures, a job that fails is marked failed (mark_failed) and
# the others carry on; --resubmit-failed starts failed jobs again from the beginning. A job whose
# inputs changed since it was set up (job_spec) is set up again if it never ran; one that has run
# stops the script, unless --force (which keeps its results). Rerunning picks up where it left off.
# Returns {'complete': True/False, 'jobs': {name: job_result}, 'calcs': {name: Calc}}, and writes
# the results to <workdir>/batch_results.json; 'complete' is False after --prepare-only, or with
# --no-wait while jobs are still queued.
#   Called by: batch_main(), the dwf_*.py scripts
#   Calls:     Calc(), Calc.done(), Calc.err_files(), calibrate(), check_templates(), finalize(),
#              job_result(), job_spec(), measure_times(), parse_castep_input(), prepare_all(),
#              remove_check(), submit(), wait_for()
def run_batch(workdir, jobs, args, cell_text, param_text, geom_param_text=None, spacing=None,
              energy="auto", share_dryruns=True, stop_file=None):
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
           "calcs": [], "share_dryruns": share_dryruns}
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
        return c.done() or bool(c.info().get("failed"))

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
    unplanned = [c for c in todo if not finished(c) and not c.info().get("jobid")]
    chunk = args.max_jobs or len(unplanned) or 1
    for k in range(0, len(unplanned), chunk):      # dry runs, at most max_jobs at a time
        if not prepare_all(unplanned[k:k + chunk], ctx, args, show=False):
            return results(False)
    if args.prepare_only:
        log(f"inputs for {len(unplanned)} job(s) written under {workdir}/ - check them, then "
            "rerun without --prepare-only")
        return results(False)
    while True:
        calibrate(calcs, keys, args)
        measure_times(calcs, keys)
        active = [c for c in todo if c.info().get("jobid") and not finished(c)]
        queue = [c for c in todo if not c.info().get("jobid") and not finished(c)]
        while queue and (not args.max_jobs or len(active) < args.max_jobs):
            c = queue.pop(0)                       # plan it again with what is known now
            finalize(c, c.dryrun_file.read_text(errors="replace"), ctx, args, quiet=True,
                     show=False)
            submit(c, args)
            active.append(c)
        if not active:
            return results(True)
        if not wait_for(active, args, any_done=bool(queue)):
            return results(False)


# The runner settings for batch mode, from CASTEP-style items (parse_castep_input).
# Any of the .conv keywords that control how jobs run (jobscript, partitions, cores_per_node,
# max_nodes, walltime, max_restarts, delete_check, dryrun_*, submit_cmd, poll, max_jobs,
# tolerate_failures, ...), converted and checked as read_input does (convert_settings,
# check_runner); the size-series keywords (defect, sizes, e_bulk, ...) aren't needed. File names
# are relative to base. Returns the settings namespace run_batch takes.
#   Called by: read_batch_file(), the dwf_*.py scripts
#   Calls:     check_runner(), convert_settings()
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
#   Calls:     batch_args(), block_rows(), _bool(), _choice(), _ints(), parse_castep_input(),
#              read_structure()
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
                    job["overrides"][key] = val
            except ValueError as err:
                die(f"{name}: job {row[0]}: bad option '{opt}': {err}")
        jobs.append(job)
    args.batch_jobs = jobs
    return args


# --batch: runs the jobs of a batch file with run_batch and prints a table of the results (also
# in batch_results.json in the work directory, batch_<file name> unless workdir is set).
#   Called by: main()
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
# supercell_convergence.py input file     run:  python3 supercell_convergence.py <seedname>
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
# can give its own, e.g. --time=48:00:00). auto: ask for less once finished jobs show
# how long the next ones need; fixed: always the template's. A relaxation that runs out
# of time stops cleanly and is resubmitted to carry on from its .check file.
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

# ------------------------------------------------------------------ scheduler
submit_cmd  : sbatch
!status_cmd : squeue -h -j {jobid} -o '%i %T'   # default: chosen from submit_cmd
poll        : 120                 # seconds between queue checks
!max_jobs          : 16           # batch mode only: most jobs in the queue at once
!tolerate_failures : true         # batch mode only: a failed job is marked, the rest carry on

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


# KEYWORDS converter for a Slurm time such as 4:00:00 (sp_walltime).
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
    "mem_per_core": (_num("GB"), None), "mem_safety": (_num(), 1.05),
    "mem_replicated": (_replicated, "auto"), "baseline_mb": (_num("MB"), None),
    "omp_threads": (_omp_threads, 1),
    "walltime": (_choice("auto", "fixed"), "auto"), "walltime_safety": (_num(), 1.5),
    "sp_walltime": (_slurm_time, None),
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
                + "  (python3 supercell_convergence.py --example lists them all)")
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
# (not max_cores) and a cores value that splits evenly over the nodes; %block partitions without
# mem_per_core, and with a partition that can take cores. Starts mem_replicated : auto at R_START.
#   Called by: batch_args(), read_input()
def check_runner(args, raw_vals, name):
    for key in ("max_cores", "cores", "dryrun_cores", "poll", "dryrun_timeout",
                "cores_per_node", "max_nodes", "prerelax_steps", "prerelax_threads", "max_jobs"):
        if getattr(args, key) is not None and getattr(args, key) < 1:
            die(f"{name}: {key} must be at least 1")
    if args.cores_per_node:
        if "max_cores" in raw_vals:
            die(f"{name}: with cores_per_node the job size is set by max_nodes - remove max_cores")
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


# Reads the command line and the input file.
# Takes the seedname (reads <seedname>.conv) or the input file, plus --example, --prepare-only,
# --no-wait, --resubmit-failed and --force. --example prints the annotated example input and exits.
# --batch FILE reads a batch file instead (read_batch_file). The command-line switches override the
# same settings in the file. Returns the settings namespace that every other function receives as
# `args`.
#   Called by: main()
#   Calls:     read_batch_file(), read_input()
def get_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?",
                    help=f"seedname (reads <seedname>{INPUT_EXT}) or the input file itself")
    ap.add_argument("--example", action="store_true",
                    help="print an annotated example input file and exit")
    ap.add_argument("--prepare-only", action="store_true",
                    help="build inputs and jobscripts, submit no calculations")
    ap.add_argument("--no-wait", action="store_true",
                    help="submit what is needed, report and exit instead of waiting")
    ap.add_argument("--resubmit-failed", action="store_true",
                    help="resubmit the existing jobscript of a failed calculation once")
    ap.add_argument("--force", action="store_true",
                    help="reuse a work directory that was set up with different inputs")
    ap.add_argument("--batch", metavar="FILE",
                    help="run the jobs listed in a batch file (see 'Batch mode' above)")
    cl = ap.parse_args()
    if cl.example:
        print(EXAMPLE_INPUT, end="")
        sys.exit(0)
    if cl.batch:
        if not Path(cl.batch).is_file():
            die(f"no batch file {cl.batch}")
        args = read_batch_file(Path(cl.batch))
    else:
        if not cl.input:
            ap.error("give the seedname or input file (or --example for a template input file)")
        path = Path(cl.input)
        if not path.is_file() and Path(cl.input + INPUT_EXT).is_file():
            path = Path(cl.input + INPUT_EXT)
        if not path.is_file():
            die(f"no input file {cl.input} or {cl.input}{INPUT_EXT}")
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


# The 'charge' set in a .param file's text (0 if none). main() uses it to check, before anything
# is submitted, that the relaxations and the single points have the same charge, and that a
# charged defect has what its correction needs.
#   Called by: main()
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
#   Called by: main()
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
# Adds dE_f*, the change in E_f* from the previous size; main() compares the last one with tol.
# For a charged defect the table also shows the charge q and the correction E_corr, and E_f* is
# the corrected value (the uncorrected one is E_f_raw in convergence.json).
#   Called by: main()
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
# cores_per_node), if the dry-run template has no {SEED}, if there is no line that runs CASTEP to
# turn into a dry run, or if {PARTITION} is used without %block partitions (or dryrun_partition).
# Returns (job_template, dryrun_template).
#   Called by: main(), run_batch()
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
        example = ("    #SBATCH --nodes={NODES}\n    #SBATCH --ntasks-per-node={TASKS_PER_NODE}\n"
                   "    srun castep.mpi {SEED}")
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
def main():
    args = get_args()
    if getattr(args, "batch_jobs", None) is not None:     # --batch
        return batch_main(args)
    print(f"\nSettings from {args.input_path} (everything else at its default):")
    for key, raw in args.raw.items():
        print(f"  {key:<17}: {raw}")
    for key in ("max_jobs", "tolerate_failures"):
        if key in args.raw:
            log(f"note: {key} is for batch mode only, so it is ignored here")
    args.max_jobs, args.tolerate_failures = None, False
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


# Runs main() when this file is run as a script (python3 supercell_convergence.py ...),
# but not when it is imported into other Python code.
if __name__ == "__main__":
    main()
