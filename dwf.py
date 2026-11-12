#!/usr/bin/env python3
"""
dwf.py - the defect workflow (phases A to G of the flowchart) in one script

    python3 dwf.py --example > CaO_Sb.dwf      # an annotated project file to start from
    nohup python3 dwf.py CaO_Sb.dwf > run.log 2>&1 &
    python3 dwf.py CaO_Sb.dwf --status         # which steps have finished

One command runs every step in order, A1 to G2, waiting for its CASTEP jobs. Run it again
after any stop (Ctrl-C, touch <workdir>/STOP, --no-wait, a crash) and it carries on where it
stopped. Give step names, labels or phase letters to run only those (python3 dwf.py CaO_Sb C),
or --until STEP to stop after a step or phase.

  A  host            A1 converge, A2.1 relax, A2.2 bands, A2.3 dielectric, A3.1 phases,
                     A3.2 limits
  B  candidates      B1 generate, B2 scan (the supercell sizes, and MACE on every candidate),
                     B3 choose (the screening cell), B4 electronic (the electronic cell, from
                     C4 on: every band-edge valley folded onto Gamma)
  C  screening       C1 prerelax (MACE), C2 screen, C3 filter, in the screening cell; then
                     C4 distort (ShakeNBreak), C5 drelax, in the electronic cell
  D  ground and      with model : true, D1 dataset, D2 hybrid, D3 train, D4 check, D5 search
     excited states  (a MACE model of each defect, PBE and hybrid heads, searches for its
                     ground states in the electronic cell); D6 rank, D7 excite (triplets,
                     localised levels, ΔSCF), D8 validate (optional finite-size test)
  E  production      E1 relaxations, E2 zpl (production ΔSCF), in the electronic cell
  F  thermodynamics  F1 corrections, F2 formation (a Fermi level per dopant set), F3 interest
  G  lineshapes      G1 ccd (1D configuration-coordinate diagram), G2 lineshape (optional:
                     multi-mode, a PBE model's phonons in a big cell)

Files. Everything goes in the work directory, dwf_<name>/ next to the project file:
  state.json         which steps have finished, and with what inputs (the checkpoints)
  A_host/ ... G_lineshape/  one folder per phase, each with
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
again from scratch). A work directory made by a dwf.py from before phase D was split out of
phase C (when D6 and D7 were C11 and C12, and E and F were D and E) is brought up to date, its
results kept, with --migrate.

All CASTEP jobs go through supercell_convergence.py's batch mode (it must be in the same
directory), which runs the structures this script builds. Its runner settings (jobscript,
partitions, cores_per_node, charging, max_cores or max_nodes, max_walltime, ...) go in the
project file, for every step; %block runner_host, runner_screen, runner_hybrid and
runner_production change them for the steps at the host, screening, hybrid or production
settings, and %block runner_<step> for one step (e.g. runner_converge, runner_excite). The MACE
relaxations (B2, C1, D4, D5) run where this script runs, or, with mace_jobscript, as jobs on
compute nodes (each job runs this script as a worker: python3 dwf.py --mace-worker
<tasks>.json); the model trainings (D3, D4, G2) and G2's phonons (python3 dwf.py
--phonon-worker spec.json) run here, or as jobs with model_jobscript.
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
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ase import Atoms
from ase.cell import Cell
from ase.data import atomic_numbers

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import supercell_convergence as sc            # noqa: E402  (the job runner)

log, die = sc.log, sc.die

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
# mace_worker; train_command, train_tasks and run_training for the models), ΔSCF (dscf_job,
# dscf_result), lineshapes (pl_spectrum, phonon_worker) and tables (format_table).
# Project.out() and Project.phase_dir() are used by almost every step, so they are left out of
# the "Calls" lists, as are log() and die().
# ==========================================================================================

HARTREE = 27.211386245988                     # eV
STATE, RESULTS, SUMMARY = "state.json", "results.json", "summary.txt"
PHASE_DIRS = {"A": "A_host", "B": "B_candidates", "C": "C_screen", "D": "D_states",
              "E": "E_production", "F": "F_thermo", "G": "G_lineshape"}
PHASE_TITLES = {"A": "host", "B": "candidate defects, the screening and electronic cells",
                "C": "screening", "D": "ground and excited states",
                "E": "production ground and excited states", "F": "thermodynamics",
                "G": "optical lineshapes"}
# Work directories made before phase D was split out of phase C (D6 rank and D7 excite were C11
# and C12, and production and thermodynamics were phases D and E): the phase folders that were
# renamed, and what D6 and D7 kept in C_screen/ (--migrate, migrate_workdir)
OLD_PHASE_DIRS = {"D_production": "E_production", "E_thermo": "F_thermo"}
OLD_C_PATHS = ("jobs/share_*", "jobs/excite_gs", "jobs/excite", "structures/share",
               "structures/excite")
HB2M = 3.80998                                # hbar^2 / (2 m_e), eV Ang^2
GPA_A3 = 6.241509e-3                          # eV in 1 GPa Ang^3
# B2: the atoms at a supercell's edge, whose MACE displacements show whether a defect's strain
# field fits in it, are those further than this fraction of the shortest image distance from it
EDGE_FRACTION = 0.35
# B2 (scan_test : line): how many bigger sizes must lie on the R = R_inf - c/N line with a size for
# a candidate to count as converged there
SCAN_CONFIRM = 3
# B3: the Gamma-only floor is this many decay lengths of the shallowest deep level
FLOOR_DECAYS = 4.0
# C2: a Gamma-point gap of the screening cell this much above A2.2's means band edges that don't
# fold onto Gamma (eV)
FOLD_GAP_TOL = 0.05
# C2-D6: a screening run with at least this many electrons in the conduction band (or holes in the
# valence band) left them there instead of in defect levels: a band state, not a bound one
BAND_CARRIER = 0.5
# C4 (distort_around : auto): a complex whose components are all within this many host bond
# lengths of each other is distorted around its centre; one more spread out, around each component
PAIR_BOND_FACTOR = 1.25
# C4: an atom this close to a complex's centre (Ang) is the centre the distortions are made around
CENTRE_ATOM_TOL = 0.5
# D7: two relaxations of a defect (in different cells, or from the model and from DFT) are in the
# same basin if no atom near the defect is displaced from its site by more than this differently
# (Ang)
BASIN_TOL = 0.2
# D1: the structures left out of a model's training data (more than MODEL_E_WINDOW eV above the
# lowest of their charge state, atoms closer than MODEL_MIN_DIST Ang, a force above MODEL_MAX_FORCE
# eV/Ang), the share kept for testing, the random seed; the total charge's embedding (classes from
# -CHARGE_OFFSET)
MODEL_E_WINDOW, MODEL_MIN_DIST, MODEL_MAX_FORCE = 3.0, 1.5, 10.0
MODEL_TEST_FRACTION, MODEL_SEED = 0.2, 7
CHARGE_OFFSET, CHARGE_CLASSES = 10, 21
# The settings a MACE relaxation task may carry for the ground-state model (D4, D5), in this order
# (mace_key's, mace_relax's)
EXTRA_TASK_KEYS = ("model", "head", "charge", "fmax", "steps")
# D7, E2: the smallest Kohn-Sham step taken as an excitation (it skips a degenerate partner level),
# eV; and a mode's energy from a curvature, hbar sqrt(eV / (amu Ang^2)), in eV
KS_STEP_MIN = 0.1
HW_UNIT = 0.0646541
# B4: how far beyond the ladder (in steps of ELECTRONIC_STEP Ang, at most ELECTRONIC_TRIES of them)
# to look for a cell that folds every band-edge k-point onto Gamma
ELECTRONIC_STEP, ELECTRONIC_TRIES = 0.5, 30
# G2: the model's relaxation of the big cell before its force constants (eV/Ang, steps), the
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
# project keywords (like supercell_convergence's _ints, for decimals).
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


_t, _b, _w = sc._text, sc._bool, sc._words
# keyword: (converter, default). Blocks of whole lines ('lines') keep their rows; 'kv' blocks
# hold "keyword : value" lines (.param keywords to set); a 'template' is a file name or a block
# holding the text itself (like supercell_convergence's jobscript).
PROJECT_KEYWORDS = {
    # general
    "name": (_t, None), "workdir": (_t, None), "host_cell": (_t, None), "param": (_t, None),
    "transform": (sc._ints(9), None), "energy": (sc._choice("auto", "final", "free", "0K"), "auto"),
    "mace": (_b, True), "mace_model": (_t, "medium-mpa-0"), "mace_device": (_t, "cpu"),
    "mace_threads": (sc._num(cast=int), 8), "mace_fmax": (sc._num("eV/ang", "eV/A"), 0.05),
    "mace_steps": (sc._num(cast=int), 500),
    "mace_jobscript": ("template", None), "mace_pack": (sc._num(cast=int), None),
    "mace_max_jobs": (sc._num(cast=int), 4), "mace_python": (_t, None),
    # A: host
    "converge": (_b, True), "converge_cutoffs": (_floats(), [300, 400, 500, 600, 700, 800]),
    "converge_kgrid_min": (sc._ints(), [4]), "converge_kgrid_max": (sc._ints(), [12]),
    "converge_kgrid_step": (sc._ints(), [2]),
    "converge_tol": (sc._num("eV/atom", "eV"), 0.001), "kspacing": (sc._num("1/ang"), None),
    "converge_kcutoff": (sc._num("eV"), None),
    "bands_kspacing": (sc._num("1/ang"), None),
    "dielectric": (sc._num(), None), "dielectric_inf": (sc._num(), None),
    "dielectric_task": (_t, "phonon+efield"), "dielectric_param": (_t, None),
    "dielectric_cell": (_t, None),
    "phases": ("lines", None), "phases_from": (sc._choice("none", "doped"), "none"),
    "mp_api_key": (_t, None), "phases_e_above_hull": (sc._num("eV/atom", "eV"), 0.05),
    "chemical_potentials": ("lines", None),
    # B: candidates and the screening cell
    "dopants": (_w, []), "defects": ("lines", None), "vacancies": (_b, True),
    "antisites": (_b, False),
    "interstitials": (sc._choice("none", "dopants", "all"), "none"),
    "pairs": (_b, True), "pair_max_distance": (sc._num("ang", "A"), 5.0),
    "native_pairs": (_b, False), "dopant_pairs": (_b, False),
    "scan": (_b, True), "scan_lengths": (_floats(), [8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0]),
    "scan_tol": (sc._num("eV"), 0.02), "scan_edge_tol": (sc._num("ang", "A"), 0.05),
    "scan_test": (sc._choice("line", "flat"), "line"),
    "supercell": (_size, None), "screen_min_image": (sc._num("ang", "A"), None),
    "electronic_cell": (_cellsize, None), "electronic_depth": (sc._num("eV"), 0.25),
    # C: screening
    "screen_charges": (sc._ints(), [-1, 0, 1]),
    "screen_spins": (sc._choice("triplet", "low", "both"), "triplet"),
    "screen_cutoff": (sc._num("eV"), None), "screen_max_steps": (sc._num(cast=int), 50),
    "screen_overrides": ("kv", {}),
    "configuration_margin": (sc._num("eV"), None),
    "hull_margin": (sc._num("eV"), 0.2), "level_margin": (sc._num("eV"), 0.1),
    "require_bound": (_b, False), "require_spin": (sc._num(), None),
    "require_gap_states": (sc._choice("none", "occupied", "empty", "both"), "none"),
    "distortions": (_floats(), [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5,
                                0.6]),
    "distort_charges": (sc._choice("stable", "screened"), "screened"),
    "distort_around": (sc._choice("auto", "centre", "each", "anchor", "vacancy"), "auto"),
    "distort_cell": (sc._choice("electronic", "screening"), "electronic"),
    "rattle_stdev": (sc._num("ang", "A"), None), "rattle_dmin": (sc._num("ang", "A"), None),
    "oxidation_states": ("lines", None),
    # D: ground and excited states
    "model": (_b, False), "model_hybrid": (_b, True), "hybrid_cell": (_t, None),
    "hybrid_param": (_t, None), "model_hybrid_fraction": (sc._num(), 0.1),
    "model_rattles": (sc._num(cast=int), 10), "model_rattle_stdev": (sc._num("ang", "A"), 0.05),
    "model_jobscript": ("template", None), "model_command": (_t, "mace_run_train"),
    "model_device": (_t, None), "model_epochs": (sc._num(cast=int), 300),
    "model_r_max": (sc._num("ang", "A"), 5.0), "model_channels": (sc._num(cast=int), 64),
    "model_max_L": (sc._num(cast=int), 1), "model_batch": (sc._num(cast=int), 4),
    "model_train_args": (_t, ""), "model_force_tol": (sc._num("eV/ang", "eV/A"), 0.1),
    "model_rounds": (sc._num(cast=int), 2),
    "search_distortions": (_floats(), [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3,
                                       0.4, 0.5, 0.6, 0.7]),
    "search_rattles": (sc._num(cast=int), 2), "search_fmax": (sc._num("eV/ang", "eV/A"), 0.01),
    "search_steps": (sc._num(cast=int), 1000), "search_keep": (sc._num(cast=int), 10),
    "distortion_min_gain": (sc._num("eV"), 0.05), "share_rounds": (sc._num(cast=int), 2),
    "keep_structures": (sc._num(cast=int), 1),
    "excite": (_b, True), "excite_window": (_floats(2), None),
    "excite_min": (sc._num("eV"), 0.4), "excite_relax": (_b, True),
    "excite_overrides": ("kv", {}), "excite_spin": (_spin_or_any, 2.0),
    "spin_tol": (sc._num(), 0.3), "local_fraction": (sc._num(), 0.2),
    "excite_reach": (sc._num("eV"), 1.0), "excite_min_dw": (sc._num(), None),
    "validate": (_b, False), "validate_tol": (sc._num("eV"), 0.05),
    "validate_defects": (_w, []), "validate_sizes": (sc._num(cast=int), 3),
    "validate_gamma": (_b, True),
    "elastic_constants": ("lines", None),
    # E: production
    "production_overrides": ("kv", {}), "freeze_radius": (sc._num("ang", "A"), 0.0),
    "band_spread_tol": (sc._num("eV"), 0.05),
    # F: thermodynamics
    "charge_correction": (sc._choice("lz", "mp", "none"), "lz"), "gap": (sc._num("eV"), None),
    "fermi_temperature": (sc._num("K"), None), "quench_temperature": (sc._num("K"), None),
    "deep_level_margin": (sc._num("eV"), 0.5),
    # G: lineshapes
    "ccd_points": (_floats(), [-0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]),
    "lineshape_temperature": (sc._num("K"), 10.0), "lineshape_sigma": (sc._num("eV"), 0.006),
    "lineshape_gamma": (sc._num("eV"), 0.002), "lineshape": (_b, False),
    "lineshape_displacements": (sc._num(cast=int), 30),
    "lineshape_amplitude": (sc._num("ang", "A"), 0.02),
    "lineshape_host": (sc._num(cast=int), 5), "lineshape_length": (sc._num("ang", "A"), 30.0),
    "lineshape_method": (sc._choice("forces", "displacements"), "forces"),
    "lineshape_delta": (sc._num("ang", "A"), 0.01),
}
LIST_KEYS = {"transform", "converge_cutoffs", "dopants", "scan_lengths", "screen_charges",
             "distortions", "excite_window", "search_distortions", "ccd_points"}  # or block
# supercell_convergence keywords that control how jobs run (allowed in the project file, for every
# step, and in %block runner_<group> or runner_<step>)
RUNNER_KEYS = {"jobscript", "dryrun_jobscript", "partitions", "max_cores", "cores",
               "cores_per_node", "max_nodes", "charging", "mem_per_core", "mem_safety",
               "mem_replicated", "baseline_mb", "omp_threads", "walltime", "walltime_safety",
               "sp_walltime", "max_walltime", "max_restarts", "delete_check", "dryrun_on",
               "dryrun_partition", "dryrun_time", "dryrun_cores", "dryrun_mem_gb",
               "dryrun_options", "dryrun_cmd", "dryrun_timeout", "submit_cmd", "status_cmd",
               "poll", "max_jobs", "tolerate_failures", "pack", "dryrun_pack"}
# The steps that run CASTEP jobs, and the group of steps each belongs to. The runner settings of a
# step: the project's, changed by its group's defaults (GROUP_DEFAULTS), then by %block
# runner_<group>, then by %block runner_<step> (Project.runner_items). screen is both a step and
# a group: runner_screen is for every step at the screening settings (C2, C5, D1's pristine cells,
# D6, D7), and runner_drelax, runner_rank, runner_excite or runner_dataset changes it for one of
# them; runner_production is for every step at the production settings (D8, E1, E2, G1, G2's
# single points); runner_hybrid for the hybrid single points (D2, D4).
RUNNER_STEPS = {"converge": "host", "relax": "host", "bands": "host", "dielectric": "host",
                "phases": "host", "screen": "screen", "drelax": "screen", "rank": "screen",
                "excite": "screen", "validate": "production", "relaxations": "production",
                "zpl": "production", "dataset": "screen", "hybrid": "hybrid", "check": "hybrid",
                "ccd": "production", "lineshape": "production"}
RUNNER_GROUPS = ("host", "screen", "production", "hybrid")
RUNNER_BLOCKS = RUNNER_GROUPS + tuple(s for s in RUNNER_STEPS if s not in RUNNER_GROUPS)
# what each group of steps uses unless the project file's runner blocks say otherwise
GROUP_DEFAULTS = {"host": "",
                  "screen": "max_restarts : 0\ntolerate_failures : true",
                  "production": "tolerate_failures : true",
                  "hybrid": "tolerate_failures : true"}


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


# The keyword, or the block's name, of an item of a CASTEP-style file (sc.parse_castep_input).
#   Called by: Project.runner(), Project.runner_items(), read_project()
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
          "(B4), and D8 (validate : true) tests its finite-size error on a few defects")
_NO_E2 = ("has been removed with the old E2: G2 (lineshape : true) trains its own model on the "
          "production runs")
RETIRED = {"series_tol": _NO_E0, "runner_cells": _NO_E0, "finetune": _NO_E2,
           "finetune_args": _NO_E2, "runner_finetune": _NO_E2,
           "triage": "has been renamed: use configuration_margin (in eV; giving it switches the "
                     "filter on)",
           "triage_window": "has been renamed: use configuration_margin (in eV; giving it switches "
                            "the filter on)",
           "zpl_window": "has been removed, with step C3b and its Kohn-Sham ZPL estimate: D7 "
                         "filters on the ΔSCF ZPL instead (excite : true, excite_window)",
           "series_defect": _NO_SERIES,
           "series_charge": _NO_SERIES + " (in every charge state the defect reaches E in)",
           "series_sizes": _NO_SERIES + "; the sizes are now lengths: scan_lengths (Ang)",
           "series_choice": _NO_SERIES + ", taking the smaller of the first two sizes that agree "
                            "within series_tol",
           "runner_series": _NO_SERIES + ": the production runs use runner_production",
           "scan_sizes": "has been replaced by scan_lengths: the sizes are lengths in Ang, each "
                         "the most compact supercell with every periodic image at least that "
                         "far away",
           "finetuned_model": "has been removed: B2 and C1 use mace_model (a foundation model or "
                              "your own .model file); G2 (lineshape : true) trains its own model "
                              "on the production runs",
           "dispersion_check": _NO_C2B, "dispersion_defect": _NO_C2B, "dispersion_tol": _NO_C2B,
           "runner_dispersion": _NO_C2B,
           "size_check": _NO_C2C, "size_check_window": _NO_C2C, "size_check_max": _NO_C2C,
           "size_check_defects": _NO_C2C, "size_check_cell": _NO_C2C, "runner_sizecheck": _NO_C2C,
           "band_state_cell": "has been replaced by distort_cell: C4 and C5 work in the electronic "
                              "cell for every charge state of every survivor (distort_cell : "
                              "electronic, the default), band states included"}


# Reads the project file into a namespace of settings.
# Every keyword must be a project keyword (PROJECT_KEYWORDS) or a runner keyword (RUNNER_KEYS,
# passed to supercell_convergence.batch_args); %block runner_<group> or runner_<step>
# (RUNNER_BLOCKS) holds runner keywords, and may hold a %block partitions, for a group of steps or
# one step. A typo gets the closest keyword suggested, and a retired one (RETIRED) what to do.
# Returns (settings, the text each setting was given as, the runner items, {group or step: runner
# items}).
#   Called by: Project.__init__()
#   Calls:     compliance(), item_key(), kgrid_list(), sc.block_rows(), sc.parse_castep_input()
def read_project(path):
    name = path.name
    vals, raw, runner, blocks = {}, {}, [], {}
    known = set(PROJECT_KEYWORDS) | RUNNER_KEYS | {f"runner_{b}" for b in RUNNER_BLOCKS}
    for it in sc.parse_castep_input(path.read_text()):
        if it["type"] == "raw":
            continue
        key = item_key(it)
        if key in RETIRED:
            die(f"{name}: '{key}' {RETIRED[key]}")
        if key.startswith("runner_") and key not in known:
            die(f"{name}: there is no {key}: the runner blocks are runner_host (all of phase A's "
                "CASTEP steps), runner_screen (C2, C5, D1, D6, D7: the screening settings), "
                "runner_production (D8, E1, E2, G1, G2: the production settings), runner_hybrid "
                "(D2, D4) and, for one step, "
                + ", ".join(f"runner_{s}" for s in RUNNER_STEPS if s not in RUNNER_GROUPS))
        if key not in known:
            close = difflib.get_close_matches(key, sorted(known), n=1)
            die(f"{name}: unknown keyword '{key}'" + (f" - did you mean '{close[0]}'?" if close
                                                       else "")
                + "  (--example lists them all)")
        if key in raw:
            die(f"{name}: '{key}' is given more than once")
        if key in RUNNER_KEYS:
            runner.append(it)
            raw[key] = it.get("value", "(%block)")
            continue
        if key.startswith("runner_"):
            if it["type"] != "block":
                die(f"{name}: {key} is a block of runner settings (%block {key} ... "
                    f"%endblock {key})")
            items = [x for x in sc.parse_castep_input("\n".join(it["body"])) if x["type"] != "raw"]
            seen = set()
            for k in map(item_key, items):
                if k not in RUNNER_KEYS:
                    die(f"{name}: %block {key} can only hold runner settings, not '{k}'")
                if k in seen:
                    die(f"{name}: '{k}' is given more than once in %block {key}")
                seen.add(k)
            blocks[key[len("runner_"):]] = items
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
            rows = sc.block_rows(it)
            if conv == "lines":
                vals[key] = rows
                raw[key] = " / ".join(" ".join(r) for r in rows)
            else:
                kv = {x["key"]: x["value"] for x in
                      sc.parse_castep_input("\n".join(it["body"])) if x["type"] == "kw"}
                vals[key] = kv
                raw[key] = " / ".join(f"{k} : {v}" for k, v in kv.items())
            continue
        if it["type"] == "block":
            if key not in LIST_KEYS:
                die(f"{name}: {key} is a keyword, not a block - write '{key} : value'")
            text = " ".join(" ".join(r) for r in sc.block_rows(it))
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
    for key in ("host_cell", "param"):
        if getattr(s, key) is None:
            die(f"{name}: '{key}' is required")
    if not any(item_key(x) == "jobscript" for x in runner):
        die(f"{name}: give the jobscript template, as 'jobscript : file' or %block jobscript")
    if s.transform is not None and round(np.linalg.det(np.reshape(s.transform, (3, 3)))) <= 0:
        die(f"{name}: transform must have a positive determinant")
    for key in ("pair_max_distance", "scan_tol", "scan_edge_tol", "band_spread_tol",
                "configuration_margin", "converge_tol", "mace_fmax", "excite_min",
                "validate_tol", "lineshape_sigma", "lineshape_gamma", "lineshape_amplitude",
                "lineshape_delta"):
        if getattr(s, key) is not None and getattr(s, key) <= 0:
            die(f"{name}: {key} must be greater than 0")
    for key in ("mace_threads", "mace_steps", "mace_pack", "mace_max_jobs", "validate_sizes",
                "model_epochs", "model_channels", "model_batch", "model_rounds", "search_steps",
                "search_keep"):
        if getattr(s, key) is not None and getattr(s, key) < 1:
            die(f"{name}: {key} must be at least 1")
    if min(s.scan_lengths) <= 0:
        die(f"{name}: scan_lengths are lengths in Ang, greater than 0")
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
    compliance(s.elastic_constants)                # a bad block stops now, not at D8
    if s.model and s.model_hybrid and not (s.hybrid_param and s.hybrid_cell):
        die(f"{name}: model_hybrid (the model's hybrid head, D2) needs hybrid_param (a .param with "
            "e.g. xc_functional : HSE06) and hybrid_cell (a .cell whose species_pot gives "
            "norm-conserving potentials); or set model_hybrid : false")
    return s, raw, runner, blocks


# A project: its settings, files, directories, checkpoints (state.json), results (results.json
# of each phase) and job runner. main() makes one from the project file and the command-line
# switches, and passes it to the steps.
class Project:
    # Reads the project file and the state of earlier runs, and checks the runner settings of
    # every step now (rather than when the step starts, perhaps days later), and the MACE jobs'
    # template, and that their workers fit on a node. Stops if the work directory has the phase
    # folders of an older dwf.py (old_layout), unless this is --migrate.
    #   Called by: main()
    #   Calls:     Project.mace_runner(), Project.mace_template(), Project.runner(), old_layout(),
    #              read_project()
    def __init__(self, path, cl):
        self.path, self.base, self.cl = path, path.parent, cl
        self.s, self.raw, self.runner_project, self.runner_blocks = read_project(path)
        self._runners, self._mace, self._model_ids, self._results = {}, {}, {}, {}
        self._mace_args = None
        for step in RUNNER_STEPS:
            self.runner(step)
        if self.s.mace and self.s.mace_jobscript:
            self.mace_template()
            cpn, n = self.mace_runner().cores_per_node, self.s.mace_pack or 1
            if cpn and n * self.s.mace_threads > cpn:
                die(f"{path.name}: mace_pack x mace_threads ({n} x {self.s.mace_threads}) is more "
                    f"cores than a node has ({cpn})")
        self.name = self.s.name or path.stem
        self.dir = self.base / (self.s.workdir or f"dwf_{self.name}")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.dir / STATE
        self.state = (json.loads(self.state_file.read_text()) if self.state_file.exists()
                      else {"steps": {}})
        if old_layout(self.dir) and not getattr(cl, "migrate", False):
            die(f"{self.dir} was made by an older dwf.py, from before phase D (ground and excited "
                "states: D6 rank and D7 excite, which were C11 and C12) was split out of phase C, "
                "when production and thermodynamics were phases D and E. Once none of its jobs "
                f"from C11 on are queued or running, run  python3 dwf.py {path.name} --migrate  "
                "to move its folders to the new layout (nothing runs again); or finish it with "
                "the older dwf.py")

    # A file named by a project keyword (relative to the project file); stops if it is missing.
    #   Called by: Project.cell_text(), Project.production_param(), inputs_hash(), step_check(),
    #              step_converge(), step_dielectric(), step_hybrid(), step_relax()
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
    #   Called by: main(), migrate_workdir(), reset_step()
    #   Calls:     write_json()
    def save_state(self):
        write_json(self.state_file, self.state)

    # The results of a phase's steps: its results.json, read once ({} if there is none yet).
    #   Called by: Project.out(), Project.save_results()
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
        self._results[phase] = {s.name: res[s.name] for s in STEPS if s.name in res}
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
    #   Calls:     item_key(), sc.parse_castep_input()
    def runner_items(self, step):
        group = RUNNER_STEPS[step]
        layers = [[x for x in sc.parse_castep_input(GROUP_DEFAULTS[group]) if x["type"] != "raw"],
                  self.runner_blocks.get(group, [])]
        if step != group:
            layers.append(self.runner_blocks.get(step, []))
        items = list(self.runner_project)
        for layer in layers:
            given = {item_key(x) for x in layer}
            items = [x for x in items if item_key(x) not in given] + layer
        return items

    # The runner settings of a step (supercell_convergence.batch_args of its runner_items), with
    # the command-line switches. Steps with the same settings share them, as the steps of a group
    # did before there were step blocks (so e.g. the memory model measured in C2 is used for C5's
    # first jobs when they run in one go).
    #   Called by: Project.__init__(), Project.batch()
    #   Calls:     Project.runner_items(), item_key(), sc.batch_args()
    def runner(self, step):
        items = self.runner_items(step)
        sig = json.dumps([[item_key(x), x.get("value"), x.get("body")] for x in items])
        if sig not in self._runners:
            args = sc.batch_args(items, self.base, f"{self.path.name} (runner settings of step "
                                                    f"{STEP_INDEX[step].label} {step})")
            for flag in ("no_wait", "prepare_only", "force", "resubmit_failed"):
                if getattr(self.cl, flag, False):
                    setattr(args, flag, True)
            self._runners[sig] = args
        return self._runners[sig]

    # Runs a list of CASTEP jobs with supercell_convergence.run_batch in <workdir>/<subdir>, with
    # the runner settings of the step (its name, e.g. 'relax') and the project's STOP file;
    # keep_check : True keeps the jobs' .check files even with delete_check : true (for jobs that
    # later jobs start from). Returns run_batch's result ({'complete', 'jobs', 'calcs'}).
    #   Called by: run_screen_jobs(), step_bands(), step_ccd(), step_check(), step_converge(),
    #              step_dataset(), step_dielectric(), step_excite(), step_hybrid(),
    #              step_lineshape(), step_phases(), step_relax(), step_relaxations(), step_screen(),
    #              step_validate(), step_zpl(), validate_gamma()
    #   Calls:     Project.cell_text(), Project.runner(), sc.run_batch()
    def batch(self, subdir, jobs, step, param_text, cell_text=None, spacing=None,
              keep_check=False):
        args = self.runner(step)
        if keep_check and args.delete_check:
            args = copy.copy(args)
            args.delete_check = False
        return sc.run_batch(self.dir / subdir, jobs, args,
                            cell_text or self.cell_text(), param_text, spacing=spacing,
                            energy=self.s.energy, stop_file=self.dir / "STOP")

    # The host .cell as given (species_pot, symmetry_generate and anything else every job copies).
    #   Called by: Project.batch(), Project.kspacing(), step_dielectric(), step_relax()
    #   Calls:     Project.file()
    def cell_text(self):
        return self.file("host_cell").read_text()

    # The .param of the production settings: A1's (with the converged cut-off) once it has run,
    # else the project's .param as it is.
    #   Called by: run_screen_jobs(), step_bands(), step_ccd(), step_check(), step_dataset(),
    #              step_dielectric(), step_excite(), step_lineshape(), step_phases(), step_relax(),
    #              step_relaxations(), step_screen(), step_validate(), step_zpl(), validate_gamma()
    #   Calls:     Project.file(), Project.finished()
    def production_param(self):
        if self.finished("converge"):
            return (self.dir / self.out("converge")["production_param"]).read_text()
        return self.file("param").read_text()

    # The production k-point spacing (1/Ang): A1's, else kspacing, else from the host .cell's grid.
    #   Called by: step_bands(), step_ccd(), step_converge(), step_dielectric(), step_lineshape(),
    #              step_phases(), step_relax(), step_relaxations(), step_validate(), step_zpl()
    #   Calls:     Project.cell_text(), Project.finished(), sc.parse_castep_input(),
    #              sc.read_primitive(), sc.target_spacing()
    def kspacing(self):
        if self.finished("converge"):
            return self.out("converge")["kspacing"]
        if self.s.kspacing:
            return self.s.kspacing
        items = sc.parse_castep_input(self.cell_text())
        prim = sc.read_primitive(items)
        ns = SimpleNamespace(kspacing=None, kgrid=None)
        return sc.target_spacing(items, prim, ns)[0]

    # The relaxed host primitive cell from A2.1 (ASE Atoms, from A_host/host_relaxed.cell).
    #   Called by: doped_phases(), run_screen_jobs(), screen_setup(), step_bands(), step_ccd(),
    #              step_check(), step_choose(), step_corrections(), step_dielectric(),
    #              step_electronic(), step_excite(), step_formation(), step_generate(),
    #              step_interest(), step_limits(), step_lineshape(), step_relaxations(),
    #              step_scan(), step_validate(), step_zpl()
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
    # file), loaded once (mace-torch, on mace_device), for the relaxations run here. head: one of
    # a multi-head model's heads (D1-D5's 'pbe' or 'hse'), loaded as a MACECalculator.
    #   Called by: mace_relax(), model_errors()
    #   Calls:     Project.model_path()
    def mace(self, model=None, head=None):
        model = self.model_path(model)
        if head is not None:
            key = f"{model}#{head}"
            if key not in self._mace:
                import warnings
                warnings.filterwarnings("ignore")
                try:
                    import torch
                    from mace.calculators import MACECalculator
                except ImportError as err:
                    die(f"the ground-state model needs mace-torch in this Python "
                        f"({sys.executable}): pip install mace-torch ({err})")
                torch.set_num_threads(self.s.mace_threads)
                self._mace[key] = MACECalculator(model_paths=model, head=head,
                                                 device=self.s.mace_device,
                                                 default_dtype="float64")
            return self._mace[key]
        if model not in self._mace:
            import warnings
            warnings.filterwarnings("ignore")
            try:
                import torch
                from mace.calculators import mace_mp
            except ImportError as err:
                die(f"mace : true needs the mace-torch package in this Python ({sys.executable})"
                    f":\n    pip install mace-torch\nor set mace : false ({err})")
            torch.set_num_threads(self.s.mace_threads)
            try:
                self._mace[model] = mace_mp(model=model, device=self.s.mace_device,
                                            default_dtype="float64")
            except Exception as err:          # unknown model, no network for the download
                die(f"could not load the MACE model '{model}': {err}")
        return self._mace[model]

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

    # The MACE jobs' template (mace_jobscript: a file relative to the project file, or the text of
    # %block mace_jobscript), checked for its one {COMMAND} line (comment lines don't count),
    # which the job's workers go in place of.
    #   Called by: Project.__init__(), mace_jobs()
    def mace_template(self):
        kind, value = self.s.mace_jobscript
        if kind == "file":
            f = self.base / value
            if not f.is_file():
                die(f"{self.path.name}: the mace_jobscript file was not found: {f}")
            value = f.read_text()
        if sum("{COMMAND}" in line and not line.lstrip().startswith("#")
               for line in value.splitlines()) != 1:
            die(f"{self.path.name}: the mace_jobscript template needs one line with {{COMMAND}} "
                "where the MACE workers go, e.g.\n    {COMMAND}\n(or with srun in front, e.g. "
                "srun --exact --ntasks=1 --cpus-per-task=8 {COMMAND}, to start each one with srun)")
        return value

    # The settings the MACE jobs share with the CASTEP jobs: submit_cmd, status_cmd and poll (and
    # cores_per_node, for the number of workers), from the project's runner keywords
    # (supercell_convergence.batch_args).
    #   Called by: Project.__init__(), mace_jobs(), run_training()
    #   Calls:     sc.batch_args()
    def mace_runner(self):
        if self._mace_args is None:
            self._mace_args = sc.batch_args(self.runner_project, self.base,
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
#   Called by: main()
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
# and the contents of every file under the work directory they name (e.g. the structures), so a
# step that is run again with different results makes the later ones run again.
#   Called by: main()
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


# Writes a phase's summary.txt: for each of its steps that has run, a heading, when it finished
# and its one-line note, then its table (main() keeps the notes and tables in state.json).
#   Called by: main(), migrate_workdir(), reset_step()
def write_summary(p, phase):
    lines = [f"Project {p.name}, phase {phase}: {PHASE_TITLES[phase]}",
             f"(written by dwf.py; the numbers the later steps use are in {RESULTS})", ""]
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
    (p.phase_dir(phase) / SUMMARY).write_text("\n".join(lines))


# Prints every step, in order, with its status: when it finished and its note, still running
# (jobs queued), or not run yet ('-').
#   Called by: main()
def print_status(p):
    print(f"\nProject {p.name}: {p.path} -> {p.dir}/")
    phase = None
    for s in STEPS:
        if s.phase != phase:
            phase = s.phase
            print(f"\n  {phase}: {PHASE_TITLES[phase]}  ({PHASE_DIRS[phase]}/)")
        rec = p.state["steps"].get(s.name, {})
        print(f"    {s.label:<5} {s.name:<12} {rec.get('status', '-'):<8} "
              f"{rec.get('time', ''):<16}  {rec.get('note', '')}".rstrip())
    print()


# Moves a step's files aside (<path>.old-<time>) and forgets that it ran, and its results: --redo.
# An owned path with a * in it (e.g. D_states/jobs/share_*) moves every match, except earlier
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


# Whether a work directory was made by a dwf.py from before phase D was split out of phase C: it
# has the old phase folders (D_production/, E_thermo/), or phase C's results.json holds the
# results of D6 or D7.
#   Called by: Project.__init__(), migrate_workdir()
def old_layout(d):
    if any((d / old).is_dir() for old in OLD_PHASE_DIRS):
        return True
    f = d / PHASE_DIRS["C"] / RESULTS
    return f.is_file() and bool({"rank", "excite"} & set(json.loads(f.read_text())))


# --migrate: brings a work directory made before phase D was split out of phase C (old_layout) up
# to date, unless a step from D6 on is still running (its jobs may be queued: it should finish
# with the dwf.py that started it). Renames D_production/ to E_production/ and E_thermo/ to
# F_thermo/ (OLD_PHASE_DIRS), and moves D6's and D7's jobs and structures from C_screen/ to
# D_states/ (OLD_C_PATHS) and their results from C_screen/results.json to D_states/results.json.
# The paths in every results.json and in state.json are changed to match (and F3's for_F becomes
# for_G), the steps' labels in state.json are updated, and every phase's summary.txt is written
# again. The checkpoints are kept, so nothing runs again.
#   Called by: main()
#   Calls:     Project.save_state(), old_layout(), write_json(), write_summary()
def migrate_workdir(p):
    d = p.dir
    if not old_layout(d):
        log(f"{d}: already in the layout of this dwf.py - nothing to migrate")
        return
    busy = [f"{s.label} {s.name}" for s in STEPS if s.phase in "DEF"
            and p.state["steps"].get(s.name, {}).get("status") == "running"]
    if busy:
        die(f"{', '.join(busy)} still running (its jobs may be queued): let it finish with the "
            "dwf.py that started it, then --migrate")
    for old, new in OLD_PHASE_DIRS.items():
        if (d / old).is_dir():
            if (d / new).exists():
                die(f"{d} has both {old}/ and {new}/: move one of them aside first")
            (d / old).rename(d / new)
            log(f"moved {old}/ to {new}/")
    src, dst = d / PHASE_DIRS["C"], d / PHASE_DIRS["D"]
    for pattern in OLD_C_PATHS:
        for x in sorted(src.glob(pattern)):
            target = dst / x.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            x.rename(target)
            log(f"moved {x.relative_to(d)} to {target.relative_to(d)}")
    subs = ([(f"{old}/", f"{new}/") for old, new in OLD_PHASE_DIRS.items()]
            + [(f"{PHASE_DIRS['C']}/{x.rstrip('*')}", f"{PHASE_DIRS['D']}/{x.rstrip('*')}")
               for x in OLD_C_PATHS] + [('"for_F"', '"for_G"')])
    for f in [*sorted(d.glob(f"*/{RESULTS}")), d / STATE]:
        if f.is_file():
            text = f.read_text()
            for a, b in subs:
                text = text.replace(a, b)
            f.write_text(text)
    res = {ph: json.loads((d / PHASE_DIRS[ph] / RESULTS).read_text())
           if (d / PHASE_DIRS[ph] / RESULTS).is_file() else {} for ph in "CD"}
    for name in ("rank", "excite"):
        if name in res["C"]:
            res["D"][name] = res["C"].pop(name)
    for ph in "CD":
        (d / PHASE_DIRS[ph]).mkdir(parents=True, exist_ok=True)
        write_json(d / PHASE_DIRS[ph] / RESULTS,
                   {s.name: res[ph][s.name] for s in STEPS if s.name in res[ph]})
    p.state = json.loads(p.state_file.read_text()) if p.state_file.is_file() else {"steps": {}}
    for name, rec in p.state["steps"].items():
        if name in STEP_INDEX:
            rec["label"] = STEP_INDEX[name].label
    p.save_state()
    p._results = {}
    for phase in PHASE_DIRS:
        if (d / PHASE_DIRS[phase]).is_dir():
            write_summary(p, phase)
    log(f"{d}: migrated to the phases of this dwf.py (C screening, D ground and excited states, "
        "E production, F thermodynamics) - run it as usual")


# The command line, and the work it does: runs the steps in order (all of them, or those named by
# name, label or phase letter, up to --until), skipping those that have finished with the same
# inputs, and stops at the first that isn't finished (jobs still queued with --no-wait). A step
# that finishes has its results put in its phase's results.json, its checkpoint (with its note
# and table) in state.json, and its phase's summary.txt written again. --status prints the
# checkpoints; --redo starts steps again from scratch; --example prints an annotated project file;
# --migrate updates a work directory made by an older dwf.py (migrate_workdir); --mace-worker and
# --phonon-worker are what a MACE or G2 phonon job runs (mace_worker, phonon_worker), not a project.
#   Called by: the main block
#   Calls:     Project.__init__(), Project.finished(), Project.save_results(), Project.save_state(),
#              inputs_hash(), mace_worker(), migrate_workdir(), outputs_hash(), phonon_worker(),
#              print_status(), reset_step(), write_summary()
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project", nargs="?", help="the project file (<name>.dwf) or its name")
    ap.add_argument("steps", nargs="*", help="run only these steps: names (relax), labels (A2.1) "
                    "or phases (A to G); default: all of them")
    ap.add_argument("--until", metavar="STEP", help="stop after this step or phase (e.g. B)")
    ap.add_argument("--redo", action="append", default=[], metavar="STEP",
                    help="start this step again from scratch (can be given more than once)")
    ap.add_argument("--status", action="store_true", help="show which steps have finished")
    ap.add_argument("--example", action="store_true", help="print an annotated project file")
    ap.add_argument("--prepare-only", action="store_true",
                    help="write the inputs of the next CASTEP jobs, submit nothing")
    ap.add_argument("--no-wait", action="store_true",
                    help="submit what is needed and exit instead of waiting; rerun later")
    ap.add_argument("--resubmit-failed", action="store_true",
                    help="start failed CASTEP jobs again, and MACE relaxations whose jobs "
                    "gave up")
    ap.add_argument("--force", action="store_true",
                    help="keep the results of jobs whose inputs have changed")
    ap.add_argument("--migrate", action="store_true",
                    help="move the folders of a work directory made by an older dwf.py (before "
                    "phase D was split out of C) to this version's layout")
    ap.add_argument("--mace-worker", metavar="TASKS",
                    help="(what a MACE job runs) relax the structures in this task file")
    ap.add_argument("--phonon-worker", metavar="SPEC",
                    help="(what a G2 job runs) a defect's force constants with its model")
    cl = ap.parse_args()
    if cl.mace_worker:
        mace_worker(cl.mace_worker)
        return
    if cl.phonon_worker:
        phonon_worker(cl.phonon_worker)
        return
    if cl.example:
        print(EXAMPLE_PROJECT, end="")
        return
    if not cl.project:
        ap.error("give the project file (or --example for an annotated one)")
    path = Path(cl.project)
    if not path.is_file() and Path(cl.project + ".dwf").is_file():
        path = Path(cl.project + ".dwf")
    if not path.is_file():
        die(f"no project file {cl.project} or {cl.project}.dwf")
    p = Project(path, cl)
    if cl.migrate:
        migrate_workdir(p)
        return
    if cl.status:
        print_status(p)
        return

    # The steps a word stands for: a step's name or label, or a phase letter (all its steps).
    def find(word):
        w = word.lower()
        hits = [s for s in STEPS if w in (s.name, s.label.lower(), s.phase.lower())]
        if not hits:
            die(f"no step or phase '{word}': the steps are "
                + ", ".join(f"{s.label} {s.name}" for s in STEPS) + ", the phases A to G")
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
                "out_hash": outputs_hash(p, outputs), "time": time.strftime("%Y-%m-%d %H:%M"),
                "note": note, "table": table}
            p.save_state()
            write_summary(p, step.phase)
            log(f"{step.label} {step.name}: finished" + (f" - {note}" if note else ""))
    except KeyboardInterrupt:
        print("\nInterrupted. Submitted jobs keep running - rerun the same command to carry on.")
        return
    if all(p.finished(s.name) for s in todo):
        log((f"step {todo[0].label}" if len(todo) == 1 else
             f"steps {todo[0].label} to {todo[-1].label}")
            + f" done: each phase's summary.txt has the results ({p.dir}/)")


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
#   Called by: dscf_result(), job_summary(), production_summary(), step_distort(), step_prerelax(),
#              step_search()
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
#   Called by: Project.host(), distinct_states(), duplicates(), into_cell(), lineshape_vector(),
#              run_screen_jobs(), step_ccd(), step_check(), step_distort(), step_excite(),
#              step_lineshape(), step_rank(), step_screen(), step_search(), step_validate(),
#              step_zpl(), validate_gamma()
#   Calls:     sc.read_structure()
def load_cell(p, rel):
    return sc.read_structure(p.dir / rel)


# Reads a structure file of yours: a CASTEP .cell (keeping SPIN= values) or anything ASE reads.
#   Called by: phase_list(), step_converge(), step_relax()
#   Calls:     sc.read_structure()
def read_structure_file(path):
    return sc.read_structure(Path(path))


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
# (sc.translation_copies, sc.anchor_site), or for an interstitial the images of its position, so
# the complex is only moved, never turned. Stops if a site can't be found or the complex overlaps
# its own periodic images. Substituted and interstitial atoms get their component's starting
# spin. Returns (Atoms, info): info has the defect's centre (the mean position of its components,
# cartesian), each component's position, the anchor's atom index (None for a vacancy) and the
# positions of the vacancies.
#   Called by: big_cell(), carry_over(), into_cell(), match_ideal(), run_screen_jobs(), step_ccd(),
#              step_check(), step_lineshape(), step_prerelax(), step_scan(), step_search(),
#              step_validate(), step_zpl()
#   Calls:     comp_name(), sc.anchor_site(), sc.build_supercell(), sc.get_distances(),
#              sc.pure_translations(), sc.translation_copies()
def build_defect(prim, P, comps, label="defect"):
    host = sc.build_supercell(prim, P)
    cell = np.asarray(host.cell)
    centre = cell.sum(axis=0) / 2
    syms = np.array(host.get_chemical_symbols())

    # Host atoms a component after the anchor can go on: that element (its vector says where).
    def sites_of(c):
        return np.flatnonzero(syms == c["host"])

    # The minimum-image distances from a point to some atoms of the host supercell.
    def dists(point, idx):
        _, L = sc.get_distances(np.atleast_2d(point), host.positions[idx], cell=host.cell,
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
        copies = sc.translation_copies(prim, a["prim_site"])
        sites = [sc.anchor_site(host, dict(a, prim_sites=copies), offsets[1:])]
        r0 = host.positions[sites[0]].copy()
    else:
        pc = np.asarray(prim.cell)
        tries = []
        for t in sc.pure_translations(prim):
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
            _, L = sc.get_distances(points[i][None], points[j][None], cell=host.cell, pbc=True)
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
#   Called by: run_screen_jobs(), step_dataset(), step_distort(), step_excite(), step_filter(),
#              step_lineshape(), step_relaxations(), step_screen(), step_search(), step_validate(),
#              step_zpl(), validate_gamma()
#   Calls:     sc.build_supercell()
def host_supercell(prim, P):
    atoms = sc.build_supercell(prim, P)
    del atoms.arrays["prim_index"]
    return atoms


# The shortest distance between a point and its periodic images (the shortest lattice vector),
# from the Minkowski-reduced cell, so that it is right for skewed cells too.
#   Called by: carry_over(), cell_record(), lineshape_vector(), same_basin(), size_matrix()
#   Calls:     sc.min_image_distance()
def min_image(cell):
    return sc.min_image_distance(Cell(np.asarray(cell, float)).minkowski_reduce()[0])


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
#   Called by: carry_over(), same_basin()
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
# supercells of it. Returns (Atoms, info) as build_defect does.
#   Called by: big_cell(), into_cell(), step_check(), step_distort(), step_search(), step_validate()
#   Calls:     build_defect(), common_atoms(), displacement_field(), min_image()
def carry_over(prim, comps, src_P, src_atoms, dst_P, radius=None, label="defect"):
    ideal, _, u, v = displacement_field(prim, comps, src_P, src_atoms, label)
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
#   Called by: job_summary(), run_screen_jobs(), step_bands(), step_choose(), step_excite(),
#              step_formation(), step_relaxations(), step_screen(), step_zpl()
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
#   Called by: dscf_result(), step_excite(), step_zpl()
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
#   Called by: dscf_result(), step_excite(), step_zpl(), zpl_bands_ok()
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
#   Called by: band_edges(), band_masses(), carriers(), defect_levels(), gap_levels()
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
#   Calls:     sc.last_lattice()
def read_trajectory(text, which="final"):
    out = []
    ends = [m for m in re.finditer(sc.ENERGY_PATTERNS["final"], text)]
    for k, m in enumerate(ends):
        head = text[:m.start()]
        part = head.rsplit("Fractional coordinates of atoms", 1)
        if len(part) < 2:
            continue
        rows = sc.ATOM_ROW.findall(part[1])
        lattice = sc.last_lattice(head)
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
        alt = re.search(sc.ENERGY_PATTERNS[which], tail) if which != "final" else None
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


# The finite-size correction (eV) for charge q in a supercell (supercell_convergence's
# charge_correction: Lany-Zunger, Makov-Payne or none), with the static dielectric constant eps.
#   Called by: relative_formation(), step_corrections(), step_filter(), step_validate()
#   Calls:     sc.charge_correction()
def correction(q, cell, eps, method):
    if not q or method == "none":
        return 0.0
    if not eps:
        die("a charged defect needs the dielectric constant: run the dielectric step (A2) or "
            "set dielectric, or set charge_correction : none")
    return sc.charge_correction(q, cell, SimpleNamespace(charge_correction=method,
                                                         dielectric=eps))[0]


# --------------------------------------------------------------------------
# MACE
# --------------------------------------------------------------------------
# What identifies one MACE relaxation in the caches: the structure, the model (Project.model_id)
# and the optimiser's settings (mace_fmax, mace_steps, or a relaxation's own fmax and steps), and
# for the ground-state model the head and the total charge.
#   Called by: mace_relax(), mace_relaxations()
#   Calls:     Project.model_id(), atoms_to_dict()
def mace_key(p, atoms, model=None, head=None, charge=None, fmax=None, steps=None):
    data = {"atoms": atoms_to_dict(atoms), "model": p.model_id(model),
            "fmax": fmax or p.s.mace_fmax, "steps": steps or p.s.mace_steps}
    if head is not None or charge is not None:            # (keeps the older caches valid)
        data.update(head=head, charge=charge)
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


# The cache file of a candidate's MACE relaxation in one supercell (tag: the size's tag, e.g.
# n144). B2 and C1 share them, so C1 reuses the scan's relaxations in the screening cell.
#   Called by: step_prerelax(), step_scan()
def mace_cache(p, cid, tag):
    return p.phase_dir("B") / "mace" / f"{cid}_{tag}.json"


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


# Relaxes the positions of a structure with a MACE calculator, the cell fixed (ASE's LBFGS, to
# fmax, in at most steps steps), in total charge charge for a model that takes it. Returns
# (relaxed Atoms, energy before, energy after, steps); the starting spins are kept (MACE ignores
# them). The same here (mace_relax) and in the MACE jobs (mace_worker).
#   Called by: mace_relax(), mace_worker(), phonon_worker()
def mace_run(calc, atoms, fmax, steps, charge=None):
    from ase.optimize import LBFGS
    work = atoms.copy()
    if charge is not None:
        work.info["charge"] = charge               # the model's total-charge embedding reads it
    work.calc = calc
    e0 = float(work.get_potential_energy())
    opt = LBFGS(work, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    e1 = float(work.get_potential_energy())
    work.calc = None
    work.set_initial_magnetic_moments(atoms.get_initial_magnetic_moments())
    return work, e0, e1, int(getattr(opt, "nsteps", 0))


# Relaxes a structure with MACE here, in this process (model: default mace_model; head, charge,
# fmax and steps for the ground-state model), unless its cache file already holds the result
# (mace_cached). Returns (relaxed Atoms, energy before, energy after, steps).
#   Called by: mace_relaxations()
#   Calls:     Project.mace(), mace_cached(), mace_key(), mace_run(), save_mace_result()
def mace_relax(p, atoms, cache, model=None, head=None, charge=None, fmax=None, steps=None):
    key = mace_key(p, atoms, model, head, charge, fmax, steps)
    hit = mace_cached(cache, key)
    if hit:
        return hit
    result = mace_run(p.mace(model, head), atoms, fmax or p.s.mace_fmax,
                      steps or p.s.mace_steps, charge)
    save_mace_result(cache, key, result)
    return result


# A step's MACE relaxations (mace_model): {cache file: (relaxed Atoms, energy before, energy
# after, steps)}, or None while MACE jobs are still queued (--no-wait) or after --prepare-only.
# tasks: [{'atoms', 'cache', 'label'}, and for the ground-state model 'model', 'head', 'charge',
# 'fmax', 'steps' (EXTRA_TASK_KEYS)]. A relaxation whose cache file already holds it (the same
# structure, model and settings: mace_key) is not run again. The others run as jobs on compute
# nodes with mace_jobscript (mace_jobs), else here, one after the other (mace_relax).
#   Called by: step_check(), step_prerelax(), step_scan(), step_search()
#   Calls:     mace_cached(), mace_jobs(), mace_key(), mace_relax()
def mace_relaxations(p, step, tasks):
    out, todo = {}, []
    for t in tasks:
        t = dict(t, key=mace_key(p, t["atoms"], *(t.get(k) for k in EXTRA_TASK_KEYS)))
        hit = mace_cached(t["cache"], t["key"])
        if hit:
            out[str(t["cache"])] = hit
        else:
            todo.append(t)
    if todo and p.s.mace_jobscript:
        if not mace_jobs(p, step, todo):
            return None
        for t in todo:
            out[str(t["cache"])] = mace_cached(t["cache"], t["key"])
    for k, t in enumerate([] if p.s.mace_jobscript else todo):
        log(f"{STEP_INDEX[step].label}: {t['label']}: MACE relaxation ({len(t['atoms'])} atoms; "
            f"{k + 1} of {len(todo)})")
        out[str(t["cache"])] = mace_relax(p, t["atoms"], t["cache"],
                                          *(t.get(k) for k in EXTRA_TASK_KEYS))
    return out


# The Python the MACE jobs run: mace_python, or else the one running dwf.py, with any linked folder
# on its path followed to where it really is (e.g. a ~/shared that links to /work/...), as the
# compute nodes may not see the link's side (on Cirrus and ARCHER2, /home). Only the folders: a
# venv's bin/python is itself a link to the Python the venv was made from, and following that
# would leave the venv. A bare command (e.g. python3) is left to the job's PATH.
#   Called by: mace_jobs(), step_lineshape()
def mace_python(p):
    py = p.s.mace_python or sys.executable
    if "/" not in py:
        return py
    path = Path(py).expanduser()
    return str(path.parent.resolve() / path.name)


# The MACE relaxations a step still needs (mace_relaxations), run as batch jobs on compute nodes, in
# <phase>/jobs/mace_<step>/job_NNN/. Each job is the mace_jobscript template with its {COMMAND} line
# made into mace_pack workers, each running  <python> dwf.py --mace-worker tasks_KK.json  (the
# python from mace_python()) in the background with mace_threads threads (whatever is in front of
# {COMMAND} on its line, e.g. srun ..., goes in front of each), then wait; a {NCORES} in the
# template becomes the cores those workers use (workers x mace_threads), so on a machine that
# charges by the core (charging : core) a job asks for just those. The relaxations are shared out
# between at most mace_max_jobs jobs, biggest first, onto whichever worker has the fewest atoms so
# far, so the workers finish together. The jobs go in jobs.json there; waiting polls the queue
# (status_cmd) every poll seconds, like the CASTEP jobs, and the project's STOP file stops it. A
# relaxation still missing once its job has ended (e.g. it ran out of time) goes into a new job, at
# most MACE_ATTEMPTS times; then the run stops, with how the last of its jobs ended (why_ended).
# --resubmit-failed gives those relaxations another MACE_ATTEMPTS jobs. Returns True when every
# relaxation is done, False while jobs are queued (--no-wait) or after --prepare-only (the job
# scripts are then in prepared/).
#   Called by: mace_relaxations()
#   Calls:     Project.mace_model_file(), Project.mace_runner(), Project.mace_template(),
#              Project.model_path(), atoms_to_dict(), mace_cached(), mace_python(), sc.job_active(),
#              sc.send_job(), write_json()
def mace_jobs(p, step, todo):
    args = p.mace_runner()
    d = p.phase_dir(STEP_INDEX[step].phase) / "jobs" / f"mace_{step}"
    d.mkdir(parents=True, exist_ok=True)
    state_file = d / "jobs.json"
    state = (json.loads(state_file.read_text()) if state_file.exists()
             else {"jobs": [], "tries": {}})
    template, stop = p.mace_template(), p.dir / "STOP"
    pack = p.s.mace_pack or max(1, (args.cores_per_node or p.s.mace_threads) // p.s.mace_threads)
    label = STEP_INDEX[step].label
    model = p.mace_model_file() if any(not t.get("model") for t in todo) else None
    announced = False
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
        if sc.job_active(job["jobid"], args):
            return False
        return (Path(job["dir"]) / "finished").exists() or time.time() - job["time"] > 120

    # How a job that left relaxations undone ended, from its folder: stopped by Slurm at its time
    # limit (its Slurm output says so), or else the last line of the first of its workers' logs,
    # then of its Slurm output, with anything in it (e.g. a Python error, or a Python the compute
    # node couldn't find).
    def why_ended(jdir):
        outs = sorted(jdir.glob("slurm-*.out"))
        if any("DUE TO TIME LIMIT" in f.read_text(errors="replace") for f in outs):
            return "Slurm stopped it at its time limit (give mace_jobscript a longer --time)"
        for f in sorted(jdir.glob("worker_*.log")) + outs:
            lines = [x.strip() for x in f.read_text(errors="replace").splitlines() if x.strip()]
            if lines:
                return f"{f.name} ends with: {lines[-1][:300]}"
        return "its workers' logs and Slurm output are empty"

    # Writes the scripts of new jobs for some relaxations into a folder each (its task files, one
    # per worker, and job.sh); returns [(folder, [cache files])].
    def write_jobs(tasks, where):
        n_jobs = min(p.s.mace_max_jobs, math.ceil(len(tasks) / pack))
        slots = [{"tasks": [], "atoms": 0} for _ in range(min(len(tasks), n_jobs * pack))]
        for t in sorted(tasks, key=lambda t: -len(t["atoms"])):
            s = min(slots, key=lambda s: s["atoms"])
            s["tasks"].append(t)
            s["atoms"] += len(t["atoms"])
        made = []
        for j in range(n_jobs):
            jdir = where(j)
            jdir.mkdir(parents=True, exist_ok=True)
            files, caches = [], []
            for k, s in enumerate(slots[j::n_jobs]):
                f = jdir / f"tasks_{k + 1:02d}.json"
                f.write_text(json.dumps({
                    "model": model, "device": p.s.mace_device, "threads": p.s.mace_threads,
                    "fmax": p.s.mace_fmax, "steps": p.s.mace_steps,
                    "tasks": [dict({"cache": str(Path(t["cache"]).resolve()), "key": t["key"],
                                    "label": t["label"], "atoms": atoms_to_dict(t["atoms"])},
                                   **{k: (str(Path(p.model_path(t[k])).resolve())
                                          if k == "model" else t[k])
                                      for k in EXTRA_TASK_KEYS if t.get(k) is not None})
                              for t in s["tasks"]]}))
                files.append(f)
                caches += [str(t["cache"]) for t in s["tasks"]]
            lines = template.splitlines()
            i = next(k for k, line in enumerate(lines)
                     if "{COMMAND}" in line and not line.lstrip().startswith("#"))
            pre, post = lines[i].split("{COMMAND}", 1)
            py, me = mace_python(p), str(Path(__file__).resolve())
            run = [f"{pre}env OMP_NUM_THREADS={p.s.mace_threads} {shlex.quote(py)} "
                   f"{shlex.quote(me)} --mace-worker {shlex.quote(str(f.resolve()))}{post} > "
                   f"{shlex.quote(str((jdir / f'worker_{k + 1:02d}.log').resolve()))} 2>&1 &"
                   for k, f in enumerate(files)]
            run += ["wait", f"echo finished > {shlex.quote(str((jdir / 'finished').resolve()))}"]
            text = "\n".join(lines[:i] + run + lines[i + 1:]) + "\n"
            name = f"dwf_mace_{step}_{jdir.name}"
            cores = len(files) * p.s.mace_threads          # (for a template that asks for cores)
            (jdir / "job.sh").write_text(text.replace("{JOBNAME}", name)
                                         .replace("{NCORES}", str(cores)))
            made.append((jdir, caches))
        return made

    while True:
        left = [t for t in todo if mace_cached(t["cache"], t["key"]) is None]
        if not left:
            if announced:
                log(f"{label}: the MACE jobs have finished")
            return True
        for job in state["jobs"]:
            if not job.get("ended") and ended(job):
                job["ended"] = True
        active = [j for j in state["jobs"] if not j.get("ended")]
        busy = {c for j in active for c in j["caches"]}
        new = [t for t in left if str(t["cache"]) not in busy]
        if new:
            worn = [t for t in new if state["tries"].get(str(t["cache"]), 0) >= MACE_ATTEMPTS]
            if worn:
                names, caches = [t["label"] for t in worn], {str(t["cache"]) for t in worn}
                last = next((Path(j["dir"]) for j in reversed(state["jobs"])
                             if caches & set(j["caches"])), None)
                die(f"{label}: the MACE jobs ended {MACE_ATTEMPTS} times without these "
                    f"relaxations: {', '.join(names[:8])}"
                    + (f" and {len(names) - 8} more" if len(names) > 8 else "")
                    + (f". The last of their jobs, {last.name}: {why_ended(last)}" if last else "")
                    + f". The workers' logs and Slurm output are in {d}/job_*/; once the cause is "
                    "fixed, rerun with --resubmit-failed")
            if p.cl.prepare_only:
                shutil.rmtree(d / "prepared", ignore_errors=True)
                write_jobs(new, lambda j: d / "prepared" / f"job_{j + 1:03d}")
                log(f"{label}: the scripts of the MACE jobs for {len(new)} relaxations are in "
                    f"{d / 'prepared'} - check them, then rerun without --prepare-only")
                return False
            first = 1 + max([int(x.name[4:]) for x in d.glob("job_*") if x.name[4:].isdigit()],
                            default=0)
            for jdir, caches in write_jobs(new, lambda j: d / f"job_{first + j:03d}"):
                jobid = sc.send_job(jdir, "job.sh", args)
                state["jobs"].append({"jobid": jobid, "dir": str(jdir), "caches": caches,
                                      "time": time.time()})
                for c in caches:
                    state["tries"][c] = state["tries"].get(c, 0) + 1
                log(f"{label}: submitted MACE job {jobid} ({jdir.name}: {len(caches)} "
                    f"relaxation{'s' if len(caches) > 1 else ''})")
            write_json(state_file, state)
            continue
        write_json(state_file, state)
        if p.cl.no_wait:
            log(f"{label}: {len(left)} MACE relaxations still queued or running (jobs "
                + ", ".join(j["jobid"] for j in active) + "); rerun the same command later")
            return False
        if stop.exists():
            stop.unlink()
            log(f"found {stop.name} - stopping. Jobs already submitted keep running; rerun the "
                "same command to carry on.")
            sys.exit(0)
        if not announced:
            log(f"{label}: waiting for {len(left)} MACE relaxations (jobs "
                + ", ".join(j["jobid"] for j in active) + f"; polling every {args.poll} s)")
            announced = True
        time.sleep(args.poll)


# What a MACE job's worker does (python3 dwf.py --mace-worker <tasks>.json, written by
# mace_jobs): relaxes each structure in the file (mace_run, with the model file and settings given
# there, or a task's own model, head, charge, fmax and steps) that its cache file doesn't hold
# yet, and writes each result to its cache file as soon as it has it (save_mace_result); an error
# goes into the cache file instead, for the workflow to report.
#   Called by: main()
#   Calls:     atoms_from_dict(), mace_run(), save_mace_result()
def mace_worker(path):
    spec = json.loads(Path(path).read_text())
    calcs = {}
    log(f"MACE worker: {len(spec['tasks'])} relaxations, model {spec['model']}, "
        f"{spec['threads']} threads")
    for t in spec["tasks"]:
        cache = Path(t["cache"])
        try:
            if json.loads(cache.read_text()).get("key") == t["key"]:
                continue                          # done already (an earlier job)
        except (OSError, ValueError):
            pass
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
                           t.get("charge"))
            save_mace_result(cache, t["key"], res)
            log(f"{t['label']}: {len(res[0])} atoms, {res[3]} steps, "
                f"{time.time() - start:.0f} s")
        except Exception as err:                  # anything MACE or torch raises
            save_mace_result(cache, t["key"], error=f"{type(err).__name__}: {err}")
            log(f"{t['label']}: FAILED: {type(err).__name__}: {err}")


# --------------------------------------------------------------------------
# Odds and ends
# --------------------------------------------------------------------------
# Writes a JSON file, made readable, through a temporary file so that it is never left half
# written (state.json, results.json).
#   Called by: Project.save_results(), Project.save_state(), mace_jobs(), migrate_workdir(),
#              phonon_worker(), run_training(), size_ladder()
def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=float))
    tmp.replace(path)


# A text table for summary.txt and the log: a title, a header row, a rule and the rows, in columns
# as wide as needed (numbers to the right, text to the left). Floats are shown with 4 decimals,
# None as blank.
#   Called by: ladder_table(), problem_table(), rank_model(), step_ccd(), step_check(),
#              step_choose(), step_converge(), step_corrections(), step_dataset(),
#              step_electronic(), step_excite(), step_filter(), step_formation(), step_generate(),
#              step_hybrid(), step_interest(), step_limits(), step_lineshape(), step_phases(),
#              step_prerelax(), step_rank(), step_relaxations(), step_scan(), step_search(),
#              step_train(), step_validate(), step_zpl()
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
#   Called by: step_limits(), step_phases()
def counts(atoms):
    syms = atoms.get_chemical_symbols()
    return {s: syms.count(s) for s in sorted(set(syms))}


# The tag of a charge state in job names (which can't hold + or -): q0, qp1, qm1, ...
#   Called by: step_ccd(), step_check(), step_distort(), step_excite(), step_hybrid(),
#              step_lineshape(), step_rank(), step_relaxations(), step_screen(), step_search(),
#              step_validate(), step_zpl()
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
#   Calls:     sc.block_rows(), sc.parse_castep_input()
def ultrasoft_entries(cell_text):
    pots = [it for it in sc.parse_castep_input(cell_text)
            if it["type"] == "block" and it["name"] == "species_pot"]
    if not pots:
        return ["(no species_pot: CASTEP's default ultrasoft potentials)"]
    bad = []
    for row in sc.block_rows(pots[0]):
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
#   Calls:     sc.strip_comment()
def production_param_text(text, cutoff=None):
    drop = ("charge", "spin") + (("cut_off_energy", "basis_precision") if cutoff else ())
    keep, gone = [], []
    for line in text.splitlines():
        m = sc.re.match(r"\s*(\w+)\b", sc.strip_comment(line))
        if m and m.group(1).lower() in drop:
            gone.append(m.group(1).lower())
        else:
            keep.append(line)
    out = "\n".join(keep).rstrip() + "\n" + (f"cut_off_energy : {cutoff:g} eV\n" if cutoff else "")
    return out, [k for k in gone if k in ("charge", "spin")]


# The k-point spacing (1/Ang, CASTEP's 2*pi units) that a Monkhorst-Pack grid of a cell gives: the
# finest of its three directions, so that every later cell (sc.kgrid) is sampled at least as
# densely as this grid in each direction (and exactly as densely for a cell like the host's).
#   Called by: step_converge()
#   Calls:     sc.recip_lengths()
def grid_spacing(cell, grid):
    return float(min(L / n for L, n in zip(sc.recip_lengths(cell), grid)))


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
#              sc.kgrid()
def step_converge(p):
    d = p.phase_dir("A")
    text = p.file("param").read_text()
    if production_param_text(text)[1]:
        log(f"note: {p.file('param').name} sets " + " and ".join(production_param_text(text)[1])
            + "; production.param leaves them out (every job sets its own)")
    if not p.s.converge:
        m = sc.re.search(r"^\s*cut_off_energy\s*[:=\s]\s*([\d.]+)", text, sc.re.I | sc.re.M)
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
        grid = tuple(sc.kgrid(prim.cell, kspacing))
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
#              Project.production_param(), read_structure_file(), sc.final_atoms(), sc.kgrid(),
#              sc.parse_castep_input(), sc.write_cell()
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
    atoms = sc.final_atoms(res["calcs"]["host"])
    if atoms is None:
        die(f"could not read the relaxed host from host-out.cell or host.castep in {r['dir']}: "
            "its atoms don't match those of the host cell")
    keep = [it for it in sc.parse_castep_input(p.cell_text())
            if not (it["type"] == "block" and it["name"] in sc.STRUCT_BLOCKS | sc.DROP_BLOCKS)
            and not (it["type"] == "kw" and it["key"] in sc.SET_KEYS)]
    sc.write_cell(d / "host_relaxed.cell", atoms, sc.kgrid(atoms.cell, p.kspacing()), keep,
                  f"host relaxed in A2.1 ({atoms.get_chemical_formula()})")
    a, b, c, al, be, ga = atoms.cell.cellpar()
    e_bulk = r["energy"] / len(atoms)
    return {"cell": "A_host/host_relaxed.cell", "energy": r["energy"], "e_bulk": e_bulk,
            "natoms": len(atoms), "lattice": {"a": a, "b": b, "c": c, "alpha": al, "beta": be,
                                              "gamma": ga},
            "_note": f"a b c = {a:.4f} {b:.4f} {c:.4f} A, e_bulk = {e_bulk:.6f} eV/atom"}


# A2.2: band edges of the relaxed host from a single point on a Gamma-centred grid at
# bands_kspacing (default half the production spacing): VBM, CBM and gap from the .bands file,
# which the Fermi-level analysis (F2) also uses.
#   Called by: main() (step A2.2)
#   Calls:     Project.batch(), Project.host(), Project.kspacing(), Project.production_param(),
#              band_edges(), read_bands(), sc.kgrid()
def step_bands(p):
    host = p.host()
    spacing = p.s.bands_kspacing or p.kspacing() / 2
    res = p.batch("A_host/jobs/bands", [{"name": "bands", "atoms": host, "kind": "sp",
                                         "symmetry": True, "kgrid": sc.kgrid(host.cell, spacing),
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
# A3: competing phases and chemical potentials
# --------------------------------------------------------------------------
# The competing phases to relax, from %block phases (name, structure file, options: molecule,
# metal (fix_occupancy : false unless the line says otherwise), kspacing=..., anything else a
# .param keyword) or from doped (phases_from : doped, with the Materials Project). Returns a list
# of (name, Atoms, options dict).
#   Called by: step_phases()
#   Calls:     doped_phases(), read_structure_file(), sc.canonical_key()
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
                    opts["overrides"][sc.canonical_key(k)] = v
            if opts.get("metal"):
                opts["overrides"].setdefault("fix_occupancy", "false")
            out.append((row[0], read_structure_file(p.base / row[1]), opts))
        return out
    if p.s.phases_from == "doped":
        return doped_phases(p)
    return []


# Competing phases from doped's CompetingPhases (Materials Project entries within
# phases_e_above_hull of the hull, elemental references and diatomic molecules in a box). Needs
# doped and an MP API key (mp_api_key, or doped's own configuration).
#   Called by: phase_list()
#   Calls:     Project.host()
def doped_phases(p):
    import warnings
    warnings.filterwarnings("ignore")
    try:
        from doped.chemical_potentials import CompetingPhases
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError as err:
        die(f"phases_from : doped needs doped (pip install doped): {err}")
    host = p.host()
    formula = host.get_chemical_formula(empirical=True)
    try:
        cp = CompetingPhases(formula, energy_above_hull=p.s.phases_e_above_hull,
                             extrinsic=p.s.dopants or None, api_key=p.s.mp_api_key)
    except Exception as err:                       # network, API key, ...
        die(f"doped could not get the competing phases from the Materials Project: {err}")
    out = []
    for kind, entries in (("molecule", cp.molecular_entries), ("metal", cp.metallic_entries),
                          ("insulator", cp.nonmetallic_entries)):
        for e in entries:
            name = e.data.get("doped_name") or e.name
            atoms = AseAtomsAdaptor.get_atoms(e.structure)
            opts = {"overrides": {}}
            if kind == "molecule":
                opts["molecule"] = True
                if e.composition.reduced_formula == "O2":
                    opts["overrides"].update(spin="2", spin_polarized="true")
            if kind == "metal":
                opts["metal"] = True
                opts["overrides"].update(fix_occupancy="false")
            out.append((sc.re.sub(r"[^\w.\-]", "_", name), atoms, opts))
    if not out:
        die("doped found no competing phases")
    return out


# A3.1: relaxes the competing phases at the production settings. Solids: lattice and positions,
# with symmetry, at the production k-point spacing (or the phase's own kspacing). Molecules
# ('molecule'): positions only in a fixed box, Gamma point, no symmetry. Metals ('metal') get
# fix_occupancy : false (give e.g. smearing_width=0.1 on their line too). With %block
# chemical_potentials nothing runs.
#   Called by: main() (step A3.1)
#   Calls:     Project.batch(), Project.kspacing(), Project.production_param(), counts(),
#              format_table(), phase_list(), sc.kgrid()
def step_phases(p):
    if p.s.chemical_potentials:
        return {"phases": {}, "_note": "chemical potentials from the project file"}
    phases = phase_list(p)
    if not phases:
        return {"phases": {}, "_note": "no competing phases given, so no chemical-potential "
                "limits (F2 then gives energies without the mu terms)"}
    jobs = []
    for name, atoms, o in phases:
        mol = bool(o.get("molecule"))
        job = {"name": name, "atoms": atoms, "kind": "geom", "fix_cell": mol,
               "symmetry": not mol, "overrides": dict(HOST_OVERRIDES, **o["overrides"])}
        if mol:
            job["kgrid"] = [1, 1, 1]
        elif o.get("kspacing"):
            job["kgrid"] = sc.kgrid(atoms.cell, o["kspacing"])
        jobs.append(job)
    res = p.batch("A_host/jobs/phases", jobs, "phases", p.production_param(),
                  spacing=p.kspacing())
    if not res["complete"]:
        return None
    out, rows = {}, []
    for name, atoms, o in phases:
        r = res["jobs"][name]
        if r["status"] != "done" or r["energy"] is None:
            die(f"competing phase {name} did not finish: {r['reason'] or 'see ' + r['dir']}")
        out[name] = {"energy": r["energy"], "counts": counts(atoms), "natoms": len(atoms),
                     "molecule": bool(o.get("molecule"))}
        rows.append([name, atoms.get_chemical_formula(), len(atoms), r["energy"],
                     r["energy"] / len(atoms)])
    return {"phases": out, "_note": f"{len(out)} phases",
            "_table": format_table(["phase", "formula", "atoms", "E/eV", "E/atom (eV)"], rows)}


# The chemical-potential limits from the energies of the host and its competing phases.
# The host's own elements: the corners of its stability region on a pymatgen phase diagram of the
# phases made of only those elements (named after the phases there, e.g. CaO-O2). Each dopant: at
# each corner, the highest value its phases allow, mu_X = min over phases P containing X of
# (E_P - sum of the other elements' n mu) / n_X, with the limiting phase. Stops if the host is not
# stable against its competing phases.
#   Called by: step_limits()
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


# A3.2: the chemical-potential limits (chemical_limits on the A3.1 phases and the A2.1 host), or
# the ones in %block chemical_potentials (name  El=value ..., eV per atom, absolute). Also gives
# each limit as Delta mu against the pure elements, where those phases were run.
#   Called by: main() (step A3.2)
#   Calls:     Project.host(), chemical_limits(), counts(), format_table(), sc.element()
def step_limits(p):
    if p.s.chemical_potentials:
        limits = {}
        for row in p.s.chemical_potentials:
            mu = {}
            for tok in row[1:]:
                k, _, v = tok.partition("=")
                try:
                    mu[sc.element(k)] = float(v)
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
    host = p.out("relax")
    hc = counts(p.host())
    limits = chemical_limits(hc, host["energy"], phases, p.s.dopants)
    ref = {}
    for name, ph in phases.items():
        if len(ph["counts"]) == 1:
            el, n = next(iter(ph["counts"].items()))
            e = ph["energy"] / n
            ref[el] = min(ref.get(el, e), e)
    rows = []
    for name, lim in limits.items():
        lim["delta_mu"] = {el: (v - ref[el]) if el in ref else None for el, v in lim["mu"].items()}
        rows.append([name] + [lim["delta_mu"].get(el) for el in sorted(lim["mu"])]
                    + [", ".join(f"{k}: {v}" for k, v in lim["limiting"].items())])
    els = sorted(next(iter(limits.values()))["mu"])
    table = format_table(["limit"] + [f"dmu_{el}" for el in els] + ["dopants limited by"], rows,
                         "Chemical-potential limits (eV per atom, against the pure elements; the "
                         f"absolute values are in {RESULTS})")
    return {"limits": limits, "references": ref, "_note": f"{len(limits)} limits",
            "_table": table}


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
#   Calls:     equivalent_sites(), sc.parse_defect()
def parse_token(tok, prim):
    comps = sc.parse_defect([tok], prim)
    c = comps[0]
    cls = equivalent_sites(prim)
    i = c["prim_site"]
    return {"new": c["new"], "host": c["host"], "prim_site": i, "spin": c["spin"],
            "sites": cls.count(cls[i])}


# The candidates from %block defects (B1 'yes'): one line per defect, components joined by '+'
# (e.g. Sb_Ca + Vac_O). A pair with 'all' on its line gives every arrangement up to
# pair_max_distance; otherwise each further component goes on the nearest site of its kind (as in
# supercell_convergence.py's size series). q=... on a line (q=-1, or q=+1,0) gives the charge
# states of that line's candidates, used instead of screen_charges.
#   Called by: step_generate()
#   Calls:     candidate(), pair_arrangements(), parse_token(), sc.defect_vectors(),
#              sc.parse_defect()
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
            vecs = sc.defect_vectors(prim, sc.parse_defect(toks, prim))
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
# primitive cell (both for concentrations in F).
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
# native + native and dopant + dopant pairs; never two interstitials. Each pair type gives its
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
            ok = ((kinds in ((True, False), (False, True))) or
                  (kinds == (False, False) and p.s.native_pairs) or
                  (kinds == (True, True) and p.s.dopant_pairs))
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
# (fits) and relaxed with MACE (positions only, the host's lattice vectors kept, as in C and E;
# mace_model; mace_relaxations, on compute nodes with mace_jobscript). Its relaxation energy R
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
# moved less than scan_edge_tol; the edge displacement is shown with either test. One that never
# converges (or, with line, hasn't SCAN_CONFIRM bigger sizes) counts in B3 as needing the largest
# size. Relaxations that stopped at mace_steps are named in the log.
#   Called by: main() (step B2)
#   Calls:     Project.host(), build_defect(), fits(), format_table(), inverse_n_fit(),
#              ladder_table(), mace_cache(), mace_relaxations(), sc.displacements(),
#              sc.get_distances(), size_ladder()
def step_scan(p):
    prim = p.host()
    ladder = size_ladder(p, prim)
    sizes = ladder_table(ladder, "The size ladder (scan_lengths): the most compact supercell for "
                                 "each length")
    if not (p.s.mace and p.s.scan):
        return {"ladder": ladder, "candidates": {}, "_table": sizes,
                "_note": f"{len(ladder)} sizes; no MACE scan (mace or scan is false)"}
    cands = p.out("generate")["candidates"]
    tasks, plan = [], {}
    for cid, c in cands.items():
        for cell in ladder:
            if not fits(c, cell["L"]):
                continue
            atoms, info = build_defect(prim, np.array(cell["matrix"]), c["comps"],
                                       f"{cid} ({c['name']})")
            cache = mace_cache(p, cid, cell["tag"])
            tasks.append({"atoms": atoms, "cache": cache,
                          "label": f"{cid} {c['name']} in {cell['tag']}"})
            plan[(cid, cell["tag"])] = (atoms, info, cache)
    res = mace_relaxations(p, "scan", tasks)
    if res is None:
        return None
    line = p.s.scan_test == "line"
    out, rows, capped = {}, [], []
    for cid, c in cands.items():
        seen = []
        for cell in ladder:
            if (cid, cell["tag"]) not in plan:
                continue
            atoms, info, cache = plan[(cid, cell["tag"])]
            relaxed, e0, e1, steps = res[str(cache)]
            moved = sc.displacements(atoms, relaxed)
            _, r = sc.get_distances(np.array(info["centre"])[None], atoms.positions,
                                    cell=atoms.cell, pbc=True)
            edge = moved[r[0] > EDGE_FRACTION * cell["L"]]
            seen.append({"tag": cell["tag"], "L": cell["L"], "N": cell["natoms"],
                         "E_relax": round(e0 - e1, 5),
                         "edge": round(float(edge.max()) if edge.size else 0.0, 4),
                         "steps": steps})
            if steps >= p.s.mace_steps:
                capped.append(f"{cid} in {cell['tag']}")
        conv, fit = None, None
        if line:                            # the first size on a line with the bigger ones
            for i in range(len(seen) - SCAN_CONFIRM):
                r_inf, slope, worst = inverse_n_fit(seen[i:])
                if worst < p.s.scan_tol:
                    conv = seen[i]["tag"]
                    fit = {"R_inf": round(r_inf, 4), "c": round(slope, 3),
                           "worst": round(worst, 4), "sizes": len(seen) - i}
                    break
        else:
            conv = next((x["tag"] for i, x in enumerate(seen[:-1])
                         if x["edge"] < p.s.scan_edge_tol and
                         all(abs(y["E_relax"] - x["E_relax"]) < p.s.scan_tol
                             for y in seen[i + 1:])), None)
        out[cid] = dict({"sizes": seen, "converged": conv}, **({"fit": fit} if fit else {}))
        by = {x["tag"]: x for x in seen}
        here = by.get(conv)
        row = ([cid, c["name"]]
               + [f"{by[t]['E_relax']:.3f}" + ("*" if t == conv else "") if t in by else ""
                  for t in (cell["tag"] for cell in ladder)]
               + [f"{here['L']:.2f}" if here else "not converged"])
        if line:
            row += ([f"{fit['R_inf']:.3f}", f"{fit['c']:.1f}", f"{fit['c'] / here['N']:.3f}"]
                    if fit else ["", "", ""])
        rows.append(row + [here["edge"] if here else None])
    if line:
        heads = ["converged at L/A", "R_inf/eV", "c/eV", "c/N there/eV", "edge move/A"]
        title = ("MACE relaxation energies R (eV) of each candidate in each size, the cell fixed; "
                 f"* where it has converged: from that size up (it and at least {SCAN_CONFIRM} "
                 f"bigger sizes) R is within scan_tol {p.s.scan_tol:g} eV of a line R_inf - c/N, "
                 "N the cell's atoms. c/N there: the energy of the stress the fixed cell still "
                 "holds in the converged size; the edge move is shown, not tested")
    else:
        heads = ["converged at L/A", "edge move/A"]
        title = ("MACE relaxation energies (eV) of each candidate in each size; * where it is "
                 f"converged (within scan_tol {p.s.scan_tol:g} eV of every bigger size, edge "
                 f"atoms moved less than scan_edge_tol {p.s.scan_edge_tol:g} A)")
    table = format_table(["id", "defect"] + [cell["tag"] for cell in ladder] + heads, rows, title)
    if capped:
        log(f"note: B2: {len(capped)} MACE relaxations stopped at mace_steps "
            f"({p.s.mace_steps}) before their forces were below mace_fmax: "
            + ", ".join(capped[:8]) + (f" and {len(capped) - 8} more" if len(capped) > 8 else "")
            + " (raise mace_steps)")
    bad = sorted(cid for cid, x in out.items() if not x["converged"])
    if bad:
        log(f"note: B2: {len(bad)} candidates did not converge within the ladder (up to "
            f"{ladder[-1]['L']:.1f} A): {', '.join(bad)}; B3 counts them as needing the largest "
            "size" + (f" (the line test needs a size and {SCAN_CONFIRM} bigger ones on the line: "
                      "longer scan_lengths add sizes)" if line else ""))
    return {"ladder": ladder, "candidates": out, "test": p.s.scan_test,
            "_table": sizes + "\n" + table,
            "_note": f"{len(ladder)} sizes, {len(out) - len(bad)} of {len(out)} candidates "
                     "converged" + (f" ({', '.join(bad)} not)" if bad else "")}


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


# The Gamma-only floor (B3): the shortest image distance a supercell needs for Gamma-point
# screening to see a defect's levels as an isolated defect has them. A level E_b inside the gap
# decays away from the defect as exp(-r/xi), with 1/xi^2 = (2 m*/hbar^2) E_b (1 - E_b/E_g) in a
# two-band (Kane) model; neighbouring images overlap, and the level spreads into a band, unless
# they are several xi apart. The floor is FLOOR_DECAYS xi of the shallowest level that counts as
# deep (E_b = deep_level_margin), for the lighter of the hole and electron masses (band_masses)
# and A2.2's gap. screen_min_image instead, if given (0: no floor). Returns {'L', 'source', ...}
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
                best = {"L": round(FLOOR_DECAYS * xi, 2), "xi": round(xi, 3), "mass": m,
                        "edge": edge, "carrier": what, "E_b": eb, "gap": round(eg, 4),
                        "source": "band masses"}
    if best is None:
        log("WARNING: the effective masses could not be found from A2.2's band edges (its "
            "k-point grid may be too coarse), so there is no Gamma-only floor: set "
            "screen_min_image (Ang; B3) or electronic_cell (B4) to give the size")
    return best


# B3: the screening cell, which every candidate in C (and D1-D4's data) is built in.
# supercell if given (n x transform, or the most compact cell for a length); otherwise the smallest
# size of the ladder at least as big as every candidate's converged size (B2; a candidate that never
# converged counts as needing the largest size) and as the Gamma-only floor (gamma_floor), with
# what set it. A floor beyond the ladder gets a cell of its own, added to the ladder. Also says
# whether A2.2's band edges fold onto Gamma of this cell (folds_to_gamma): if not, the
# Gamma-point screening sees a wider gap (C2 measures it).
#   Called by: main() (step B3)
#   Calls:     Project.host(), add_cell(), band_masses(), cell_record(), folds_to_gamma(),
#              format_table(), gamma_floor(), ladder_table(), read_bands(), size_matrix()
def step_choose(p):
    prim = p.host()
    scan = p.out("scan")
    ladder = [dict(c) for c in scan["ladder"]]
    masses = band_masses(read_bands(p.dir / p.out("bands")["bands_file"]), prim)
    floor = gamma_floor(p, masses)
    conv, set_by = scan.get("candidates") or {}, None
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
        need = 0.0
        L = {c["tag"]: c["L"] for c in ladder}
        for cid in sorted(conv):
            x = L[conv[cid]["converged"]] if conv[cid]["converged"] else ladder[-1]["L"]
            if x > need + 1e-6:
                need, set_by = x, cid
        if floor and floor["L"] > need + 1e-6:
            need, set_by = floor["L"], "floor"
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
                + (f"{FLOOR_DECAYS:g} x the decay length {floor['xi']:.2f} A of a level "
                   f"{floor['E_b']:g} eV from the band edge of the {floor['carrier']}, m* "
                   f"{floor['mass']:.3g} m_e, gap {floor['gap']:.2f} eV)"
                   if floor["source"] == "band masses" else "screen_min_image)") + "\n"
                if floor else "\nGamma-only floor: none\n"))
    idx = ladder.index(cell)
    return {"matrix": cell["matrix"], "natoms": cell["natoms"], "L_min": cell["L"],
            "tag": cell["tag"], "ladder": ladder, "index": idx, "floor": floor,
            "set_by": set_by, "masses": masses, "fold": fold, "_table": table,
            "_note": f"{cell['tag']}: {cell['natoms']} atoms, images {cell['L']:.2f} A apart "
                     f"({why})"}


# How many of each band edge's symmetry-equivalent k-points (its star, band_masses) fold onto
# Gamma of the supercell P x (primitive cell): {'vbm': [folded, of], 'cbm': [folded, of]}.
#   Called by: step_electronic()
#   Calls:     folds_to_gamma()
def edge_folding(masses, P):
    return {e: [sum(folds_to_gamma(k, P) for k in masses[e]["k"]), len(masses[e]["k"])]
            for e in ("vbm", "cbm")}


# B4: the electronic cell, the supercell the survivors move into after C3 and every judgement of a
# defect's electronic structure is made in: C4-C5's distortions (distort_cell), D5's search, D7's
# excited states, E's production runs, and so F and G (C1-C3, which screen every candidate, stay in
# the screening cell, where they cost least). The screening cell (B3) is sized for the relaxation
# (B2) and for levels at least deep_level_margin inside the gap; at the semilocal level many levels
# sit closer to the band edges than that, and a Gamma-point run only sees the host states whose
# k-points fold onto its Gamma. So the electronic cell is the smallest size of the ladder, from the
# screening cell up, that (1) folds EVERY symmetry-equivalent k-point of both band edges onto Gamma
# (edge_folding: each valley of an edge away from Gamma, e.g. the three X points of a rock-salt
# conduction band, so a shallow state can mix all of them), and (2) meets the Gamma-only floor for
# a level electronic_depth deep (gamma_floor). If no size of the ladder does, the most compact
# cells for longer lengths (ELECTRONIC_STEP at a time) are tried. electronic_cell : screening uses
# the screening cell; electronic_cell : n or a length (e.g. 20 ang) gives the cell (with a warning
# if it doesn't fold every k-point). The table gives each cell's folding ("1 of 3" means only one
# valley of that edge is seen at Gamma).
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
    return {"matrix": cell["matrix"], "natoms": cell["natoms"], "L": cell["L"],
            "tag": cell["tag"], "fold": fold, "fold_screening": fold_s, "floor": floor,
            "why": why, "same": cell["matrix"] == ladder[idx]["matrix"], "_table": table,
            "_note": f"{cell['tag']}: {cell['natoms']} atoms, images {cell['L']:.2f} A apart "
                     f"({why})"}


# ==========================================================================================
# PHASE C: SCREENING (flowchart C1-C5)
# ==========================================================================================
# The candidates (B1) and the screening cell (B3): (candidates by id, primitive host, matrix).
#   Called by: step_dataset(), step_distort(), step_filter(), step_prerelax(), step_screen(),
#              step_search()
#   Calls:     Project.host()
def screen_setup(p):
    return p.out("generate")["candidates"], p.host(), np.array(p.out("choose")["matrix"])


# The .param keywords of the screening runs: screen_cutoff, screen_max_steps and
# %block screen_overrides.
#   Called by: excite_job(), run_screen_jobs(), screen_job(), step_check(), step_dataset(),
#              step_excite(), step_screen()
def screen_overrides(p):
    o = {"geom_max_iter": p.s.screen_max_steps}
    if p.s.screen_cutoff:
        o["cut_off_energy"] = f"{p.s.screen_cutoff:g} eV"
    o.update(p.s.screen_overrides)
    return o


# A screening or distortion job for one structure in charge state q, starting with spin S
# (spin_polarized, spin : S, and SPIN=S on the anchor atom when there is one): a Gamma-point
# relaxation without symmetry, with the screening keywords.
#   Called by: excite_job(), run_screen_jobs(), step_excite(), step_screen()
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
#   Calls:     band_carriers(), gap_levels(), read_bands(), save_cell(), sc.final_atoms()
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
        atoms = sc.final_atoms(calc)
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
# relaxes it with MACE (positions only, the cell fixed, mace_model): B2's relaxation in that cell
# when it ran there (the same cache file), else a new one (mace_relaxations, on compute nodes with
# mace_jobscript). A pair too far apart for this cell (fits) is left out. With
# configuration_margin (and MACE), arrangements whose configurational energy (MACE energy above
# the lowest arrangement of the same composition) is more than configuration_margin are dropped.
# Each structure is saved as C_screen/structures/candidates/<id>.cell, and the results give where
# its defect is (its centre, each component's position, the anchor atom's index, the vacancies)
# and its MACE energy.
#   Called by: main() (step C1)
#   Calls:     build_defect(), fits(), format_table(), mace_cache(), mace_relaxations(),
#              save_cell(), screen_setup()
def step_prerelax(p):
    cands, prim, P = screen_setup(p)
    d = p.phase_dir("C")
    choose = p.out("choose")
    L, tag = choose["L_min"], choose["tag"]
    built, skipped, energies, defects = {}, {}, {}, {}
    for cid, c in cands.items():
        if not fits(c, L):
            skipped[cid] = f"components {c['separation']:.2f} A apart, too far for this cell"
            continue
        built[cid] = build_defect(prim, P, c["comps"], f"{cid} ({c['name']})")
    if p.s.mace:
        res = mace_relaxations(p, "prerelax",
                               [{"atoms": atoms, "cache": mace_cache(p, cid, tag),
                                 "label": f"{cid} {cands[cid]['name']} in {tag}"}
                                for cid, (atoms, _) in built.items()])
        if res is None:
            return None
    for cid, (atoms, info) in built.items():
        e = None
        if p.s.mace:
            atoms, _, e, _ = res[str(mace_cache(p, cid, tag))]
        title = f"{cid} {cands[cid]['name']} in the screening cell" + (", MACE pre-relaxed"
                                                                        if p.s.mace else "")
        defects[cid] = dict(structure=save_cell(p, d / "structures" / "candidates" / f"{cid}.cell",
                                                atoms, title), mace_energy=e, **info)
        energies[cid] = e
    if p.s.configuration_margin is not None and p.s.mace:
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


# C3: the defect hull and the property filters.
# A charge state whose best screening run is a band state (in_band: it left an electron in the
# conduction band or a hole in the valence band instead of in a defect level) isn't judged: its
# energy is that of a carrier in a band, not of the defect holding it, and it may yet bind in
# another structure (the distortions) or with a wider gap than the semilocal one. Everything below
# uses the bound charge states. For each composition group and Fermi level across the gap, the
# lowest formation energy of any arrangement and bound charge state (relative_formation, with each
# charge state's correction in the screening cell, so charge states can be compared); an
# arrangement is kept if its own lowest is within hull_margin of that somewhere in the gap (one with
# no bound charge state at all can't be judged, and is kept). Its stable charge states are its
# bound ones that are its lowest somewhere in the gap. The gap is gap if the project file gives it
# (an experimental or hybrid value), else the screening cell's own: a semilocal gap is too narrow,
# and a bound charge state that only becomes the lowest above its conduction band edge (the
# formation energies carried on as the straight lines they are in the Fermi level) still counts
# then; a band state can't, since its line is the carrier's band edge carried on. The gap levels are
# still those inside the screening cell's band edges. Then, if asked: gap levels
# (require_gap_states: occupied, empty, or both = an occupied and an empty level in one spin
# channel), the final spin (require_spin) and binding against its isolated components
# (require_bound: the components must have been screened too) - each met by at least one stable
# charge state. A candidate that fails them but has a band state is kept, pending: the bound
# state might have the property. Arrangements that relaxed into the same structure as a lower one
# of their group in every charge state both have (duplicates) are dropped too. With excite : true,
# native defects (is_native) skip the property filters (D7 passes them on to E whatever their
# excited states, for F), but not the hull or the duplicates. The results list the survivors, with
# the charge states to carry on (distort_charges; band states always are); the table gives every
# candidate's verdict, then the correction for each |q| in the screening cell.
#   Called by: main() (step C3)
#   Calls:     best_states(), carriers_text(), correction(), duplicates(), format_table(),
#              host_supercell(), in_band(), internal_pair(), is_native(), relative_formation(),
#              screen_setup()
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
    singles = {(c["comps"][0]["new"], c["comps"][0]["host"], c["comps"][0].get("prim_site"),
                tuple(c["comps"][0].get("frac") or ())): cid
               for cid, c in cands.items() if len(c["comps"]) == 1}
    order = {cid: env[cid] if cid in env else np.min(np.array(list(form[cid].values())), axis=0)
             for cid in best}                    # (all charge states, for one with none bound)
    dup = duplicates(p, best, groups, order)
    left_out = p.out("prerelax")["skipped"]
    host_el = set(prim.get_chemical_symbols())
    survivors, rows = [], []
    for cid in sorted(cands):
        c = cands[cid]
        if cid not in best:
            why = (f"left out in C1: {left_out[cid]}" if cid in left_out
                   else "dropped: no usable screening run")
            rows.append([cid, c["name"], c["group"], "", "", "", "", "", why])
            continue
        dist, stable = None, []
        if cid in env:
            dist = float(np.min(env[cid] - hull[c["group"]]))
            lowest = np.argmin(np.array([form[cid][q] for q in bound[cid]]), axis=0)
            stable = sorted({bound[cid][i] for i in np.unique(lowest)})
        why = [f"relaxed into the structure of {dup[cid]}"] if cid in dup else []
        if dist is not None and dist > p.s.hull_margin:
            why.append(f"{dist:.2f} eV above the hull")
        props = {q: best[cid][q] for q in stable}

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

        good = [q for q, s in props.items() if levels_ok(s) and spin_ok(s)]
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
        waived = bool(failed or fixed) and p.s.excite and is_native(c, host_el)
        pending = bool(failed) and not fixed and bool(banded[cid]) and not waived
        if not (waived or pending):
            why += failed + fixed
        charges = (sorted(set(stable) | set(banded[cid])) if p.s.distort_charges == "stable"
                   else sorted(best[cid]))
        spins = ", ".join(f"{q:+d}:{best[cid][q]['spin']:.1f}" if best[cid][q]["spin"] is not None
                          else f"{q:+d}:-" for q in stable)
        notes = []
        if waived:
            notes.append("native, so the property filters are waived for F: "
                         + "; ".join(failed + fixed))
        if pending:
            notes.append("pending: " + "; ".join(failed) + ", but "
                         + " ".join(f"{q:+d}" for q in banded[cid])
                         + (" is" if len(banded[cid]) == 1 else " are") + " in a band")
        if dist is None:
            notes.append("no bound charge state, so the hull can't judge it")
        rows.append([cid, c["name"], c["group"], "" if dist is None else dist,
                     " ".join(f"{q:+d}" for q in stable), spins,
                     ", ".join(f"{q:+d}: {carriers_text(best[cid][q])}" for q in banded[cid]),
                     f"{bind[0]:.2f}..{bind[1]:.2f}" if bind else "",
                     ("kept" + (" (" + "; ".join(notes) + ")" if notes else ""))
                     if not why else "dropped: " + "; ".join(why)])
        if not why:
            survivors.append({"cand": cid, "name": c["name"], "group": c["group"],
                              "hull_distance": dist, "stable": stable, "charges": charges,
                              "in_band": banded[cid], "pending": failed if pending else [],
                              "good": good, "binding": bind,
                              "best": {str(q): best[cid][q] for q in best[cid]}})
    table = format_table(["id", "defect", "composition", "above hull/eV", "stable q",
                          "spin (q:S)", "in a band (q: carriers)", "E_b/eV", "verdict"], rows,
                         f"Defect hull (margin {p.s.hull_margin:g} eV) and property filters, "
                         f"Fermi levels 0 to {top:.2f} eV above the VBM"
                         + (f" (gap; the screening cell's own gap is {host['gap']:.2f} eV)"
                            if p.s.gap else "")
                         + "; charge states in a band (e: electrons in the conduction band, "
                         "h: holes in the valence band) are carried on but not judged")
    charges = sorted({abs(q) for st in best.values() for q in st if q})
    if charges:                    # what the hull's corrections add, in this cell
        table += (f"\nCharge corrections in the screening cell ({p.s.charge_correction}"
                  + (f", eps0 {eps:.3f}" if eps else "") + "): "
                  + ", ".join(f"|q| = {a}: {correction(a, cell, eps, p.s.charge_correction):.3f} eV"
                              for a in charges) + "\n")
    return {"survivors": survivors, "_table": table,
            "_note": f"{len(survivors)} of {len(cands)} candidates kept"}



# --------------------------------------------------------------------------
# C4, C5: distortions, the start of the ground-state search that phase D carries on
# --------------------------------------------------------------------------
# The oxidation states for counting the neighbours to distort: %block oxidation_states, else
# pymatgen's guess for the host and, for each dopant, its common oxidation state closest to that
# of the element it replaces (or its most positive one).
#   Called by: step_distort(), step_search()
#   Calls:     sc.element()
def oxidation_states(p, prim, cands):
    if p.s.oxidation_states:
        return {sc.element(r[0]): float(r[1]) for r in p.s.oxidation_states}
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
#   Called by: step_distort(), step_search()
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
#   Called by: step_ccd(), step_check(), step_distort(), step_excite(), step_hybrid(),
#              step_lineshape(), step_rank(), step_relaxations(), step_search(), step_zpl(),
#              validate_gamma()
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
#   Called by: step_distort(), step_search()
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


# C4: the distortions, in the cell C4 and C5 work in (distort_cell): by default the electronic cell
# (B4), so that C1-C3, which screen every candidate, run in the small screening cell while the
# survivors' ground states are searched for where their electronic structure is judged. For each
# survivor and each of its charge states (charges from C3), its screened structure in that charge
# state (or the nearest screened one) is carried into that cell (carry_over: the defect rebuilt
# there, the atoms near it moved as in the screened structure, the rest on their perfect-crystal
# sites) and relaxed there as it is (<name>_<cell tag>), the start its distortions have to beat.
# ShakeNBreak's distort_and_rattle then moves the neighbours_to_distort nearest neighbours of each
# point the defect is distorted around (distortion_centres: distort_around) in or out by each of
# distortions, and rattles the other atoms (rattle_stdev, default 10% of the host bond length;
# rattle_dmin, 80%); with no neighbours to distort it is only rattled. A defect distorted around
# more than one point gets c1, c2, ... (its components, in order) in its structures' names. In the
# electronic cell a charge state that C2 left in a band can bind its carrier, and so find the
# distortion the carrier drives, which the screening cell doesn't allow. The jobs carry the cell's
# matrix and tag, which end their names. With distort_cell : screening, or when B4 chose the
# screening cell itself, the screened structures are distorted where they are, and the screened
# runs (C2) are the start. The structures go to C_screen/structures/distortions/.
#   Called by: main() (step C4)
#   Calls:     carry_over(), distortion_centres(), host_supercell(), load_cell(),
#              neighbours_to_distort(), oxidation_states(), qtag(), save_cell(), screen_setup(),
#              start_spin()
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
    jobs = []
    for s in survivors:
        cid = s["cand"]
        comps, label = cands[cid]["comps"], f"{cid} ({cands[cid]['name']})"
        screened = {int(q): b for q, b in s["best"].items()}
        for q in s["charges"]:
            src = screened.get(q) or screened[min(screened, key=lambda x: (abs(x - q), x))]
            atoms = load_cell(p, src["structure"])
            S = start_spin(atoms, q, src["spin"] if src["q"] == q else None)
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
                    jobs.append(dict({"name": name, "cand": cid, "q": q, "spin": S, "factor": f,
                                      "neighbours": nb, "around": around, "structure": rel,
                                      "source": f"distortion around {around}" if part
                                      else "distortion"}, **extra))
    cell = ({"tag": el["tag"], "matrix": el["matrix"]} if el else
            {"tag": p.out("choose")["tag"], "matrix": P.tolist()})
    return {"jobs": jobs, "oxidation_states": ox, "cell": cell, "electronic": el is not None,
            "_note": f"{len(jobs)} structures for {len(survivors)} survivors, in the "
                     + (f"electronic cell ({el['tag']})" if el else
                        f"screening cell ({cell['tag']})")
                     + " (oxidation states "
                     + ", ".join(f"{k} {v:+g}" for k, v in ox.items()) + ")"}


# Runs a list of distortion-type jobs (name, cand, starting structure, q, spin) with the
# screening settings in <phase>/jobs/<subdir> (C_screen/ for drelax, D_states/ for rank), with the
# runner settings of the step that runs them, and returns ({name: job_summary}, {cell tag: its
# band edges}), the final structures saved in <phase>/structures/<structdir>. A job in another
# cell than the screening cell (its 'matrix' and 'cell': C4's in the electronic cell) has its
# defect found in that cell (build_defect) and its levels and band carriers measured against that
# cell's own band edges at the same settings: those given for its cell in edges (C5's, for D6),
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


# C5: relaxes every structure C4 made with the screening settings (C2's, Gamma point, no
# restarts), in the cell C4 made it in, against that cell's band edges (run_screen_jobs). Each
# run's .castep keeps every geometry step (energies and forces): D1's training data.
#   Called by: main() (step C5)
#   Calls:     problem_table(), run_screen_jobs()
def step_drelax(p):
    out, edges = run_screen_jobs(p, "drelax", "drelax", p.out("distort")["jobs"], "drelax")
    if out is None:
        return None
    bad = sum(r["energy"] is None for r in out.values())
    return {"runs": out, "edges": edges, "_table": problem_table(out),
            "_note": f"{len(out)} relaxations" + (f", {bad} without a usable energy" if bad
                                                  else "")}


# Groups structures that are the same (pymatgen's StructureMatcher, which also finds a distortion
# repeated along another direction), as lists of indices; without pymatgen, by comparing atom by
# atom (all come from the same cell with the same atom order). The tolerances are tight (stol 0.05,
# i.e. about 0.1 A for a typical oxide): defect ground states can differ by a few tenths of an
# Angstrom around the defect, which pymatgen's defaults would merge.
#   Called by: distinct_states(), duplicates(), step_rank(), step_search()
#   Calls:     sc.displacements()
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
                sc.displacements(atoms_list[i], atoms_list[j]).max() < 0.05
    groups = []
    for i in range(len(atoms_list)):
        for g in groups:
            if match(g[0], i):
                g.append(i)
                break
        else:
            groups.append([i])
    return groups


# ==========================================================================================
# PHASE D: GROUND AND EXCITED STATES (flowchart D1-D8)
# ==========================================================================================
# --------------------------------------------------------------------------
# D1-D5: the ground-state model (model : true)
# --------------------------------------------------------------------------
# After Wang, Mosquera-Lois and Walsh (arXiv:2603.05238): a MACE model for each defect, trained on
# its own C5 relaxations (a 'pbe' head) and on hybrid single points of a tenth of them (an 'hse'
# head, on the same representation), with a learned embedding of the total charge, so one model
# covers every charge state. Its hybrid head then relaxes far more starting structures than C4-C5
# can (D5), at about the hybrid level. Like everything after C3, all of it happens in the cell the
# survivors work in (work_cell): the electronic cell by default.

# The cell the survivors work in after C3, C4's (distort_cell): the electronic cell by default.
# Returns (its matrix, its tag, the end of the names of jobs in it: '_<tag>', or '' in the
# screening cell).
#   Called by: step_check(), step_dataset(), step_hybrid()
def work_cell(p):
    c4, choose = p.out("distort"), p.out("choose")
    if c4.get("electronic"):
        return np.array(c4["cell"]["matrix"]), c4["cell"]["tag"], f"_{c4['cell']['tag']}"
    return np.array(choose["matrix"]), choose["tag"], ""


# The defects D1-D5 model: C3's survivors, each with the charge states it carries on.
#   Called by: step_dataset(), step_search()
def model_defects(p):
    return {s["cand"]: sorted(s["charges"]) for s in p.out("filter")["survivors"]}


# A survivor's (s, C3's) lowest relaxed structure in charge state q, in the cell C4 worked in (its
# 'cell'; energies compare only within one cell): the lowest of its C5 runs there, and, when that
# is the screening cell, of its screened one (C2) too; when none of those has an energy, its
# screened structure. Returns (the run, with 'step' C2 or C5, its supercell matrix), or
# (None, None).
#   Called by: step_check(), step_search()
def lowest_start(p, s, q):
    P0 = p.out("choose")["matrix"]
    M = (p.out("distort").get("cell") or {}).get("matrix") or P0
    runs = [dict(r, name=name, step="C5") for name, r in p.out("drelax")["runs"].items()
            if r["cand"] == s["cand"] and r["q"] == q and (r.get("matrix") or P0) == M]
    b = s["best"].get(str(q))
    screened = [dict(b, step="C2", matrix=P0)] if b else []
    for pool in (runs + (screened if M == P0 else []), screened):
        pool = [r for r in pool if r.get("energy") is not None and r.get("structure")]
        if pool:
            best = min(pool, key=lambda r: r["energy"])
            return best, best.get("matrix") or P0
    return None, None


# A folder of the model's files: D_states/model/<part>/<defect>.
#   Called by: step_dataset(), step_hybrid(), train_tasks()
def model_dir(p, part, cid):
    d = p.phase_dir("D") / "model" / part / cid
    d.mkdir(parents=True, exist_ok=True)
    return d


# Every geometry step of a .castep as a training structure (read_trajectory, with the free energy
# E-TS that the forces belong to): ASE Atoms with info REF_energy, total_charge (q), run (the job),
# step, spin (the run's final spin, a starting spin for a hybrid single point) and arrays
# REF_forces.
#   Called by: step_check(), step_dataset(), step_hybrid(), step_lineshape()
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


# D1's filters, as in the paper: a structure more than MODEL_E_WINDOW above the lowest of its
# charge state in its cell (its number of atoms: C5 also relaxes band states in the electronic
# cell), with two atoms closer than MODEL_MIN_DIST, or a force above MODEL_MAX_FORCE (the first
# steps of a wild distortion) is left out. Returns the structures kept.
#   Called by: step_dataset(), step_lineshape()
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


# D1: the training set of each defect C3 kept (model_defects), in the cell the survivors work in
# (work_cell: the electronic cell by default): every geometry step of its C5 relaxations in the
# charge states it carries on (castep_frames, at the screening settings: the 'pbe' head; with
# distort_cell : screening, C2's too, which are in the same cell), and model_rattles pristine
# supercells of that cell with random displacements (model_rattle_stdev; single points at the
# screening settings, D_states/jobs/dataset), filtered (filter_frames) and split
# MODEL_TEST_FRACTION for testing, the rest for training (a fixed seed; info 'split'), in
# D_states/model/data/<defect>/pbe_train.xyz and pbe_test.xyz. Each structure's info 'supercell'
# is its cell's tag.
#   Called by: main() (step D1)
#   Calls:     Project.batch(), Project.production_param(), castep_frames(), filter_frames(),
#              format_table(), host_supercell(), model_defects(), model_dir(), screen_overrides(),
#              screen_setup(), work_cell()
def step_dataset(p):
    if not p.s.model:
        return {"skipped": True, "_note": "model : false (D6 ranks C5's relaxations)"}
    import ase.io
    cands, prim, _ = screen_setup(p)
    M, tag, end = work_cell(p)
    defs = model_defects(p)
    host = host_supercell(prim, M)
    rng = np.random.default_rng(MODEL_SEED)
    jobs = []
    for k in range(p.s.model_rattles):
        a = host.copy()
        a.positions += rng.normal(0.0, p.s.model_rattle_stdev, a.positions.shape)
        jobs.append({"name": f"host_r{k + 1}{end}", "atoms": a, "kind": "sp", "kgrid": [1, 1, 1],
                     "symmetry": False, "overrides": {k: v for k, v in screen_overrides(p).items()
                                                      if k != "geom_max_iter"}})
    hosts = []
    if jobs:
        res = p.batch(f"{PHASE_DIRS['D']}/jobs/dataset", jobs, "dataset", p.production_param())
        if not res["complete"]:
            return None
        for j in jobs:
            c = res["calcs"][j["name"]]
            if res["jobs"][j["name"]]["energy"] is not None:
                for a in castep_frames(c.castep, 0, 0.0, j["name"]):
                    a.info["supercell"] = tag
                    hosts.append(a)
    sources = [(p.out("drelax")["runs"], "drelax")]
    if not end:                           # (C4 worked in the screening cell, where C2's runs are)
        sources.insert(0, (p.out("screen")["runs"], "screen"))
    tag0 = p.out("choose")["tag"]
    out, rows = {}, []
    for cid, qs in sorted(defs.items()):
        frames = []
        for runs, sub in sources:
            for name, r in sorted(runs.items()):
                f = p.dir / PHASE_DIRS["C"] / "jobs" / sub / name / f"{name}.castep"
                if r["cand"] == cid and r["q"] in qs and f.exists():
                    for a in castep_frames(f, r["q"], r.get("spin"), name):
                        a.info["supercell"] = r.get("cell") or tag0
                        frames.append(a)
        kept = filter_frames(frames) + hosts
        order = np.random.default_rng(MODEL_SEED).permutation(len(kept))
        ntest = int(round(MODEL_TEST_FRACTION * len(kept)))
        for n, i in enumerate(order):
            kept[i].info["split"] = "test" if n < ntest else "train"
        d = model_dir(p, "data", cid)
        files = {}
        for split in ("train", "test"):
            sel = [a for a in kept if a.info["split"] == split]
            if sel:
                ase.io.write(str(d / f"pbe_{split}.xyz"), sel, format="extxyz")
                files[split] = str((d / f"pbe_{split}.xyz").relative_to(p.dir))
        out[cid] = {"charges": qs, "steps": len(frames), "kept": len(kept) - len(hosts),
                    "host": len(hosts), "files": files}
        rows.append([cid, cands[cid]["name"], " ".join(f"{q:+d}" for q in qs), len(frames),
                     len(kept) - len(hosts), len(hosts),
                     sum(a.info["split"] == "train" for a in kept),
                     sum(a.info["split"] == "test" for a in kept)])
    table = format_table(["id", "defect", "charges", "steps", "kept", "host", "train", "test"],
                         rows, f"Training data in {tag} (the geometry steps of the runs there; "
                               f"kept: within {MODEL_E_WINDOW:g} eV of the lowest of their "
                               f"charge state and cell, no atoms closer than {MODEL_MIN_DIST:g} "
                               f"A, no force above {MODEL_MAX_FORCE:g} eV/A)")
    return {"defects": out, "cell": tag, "_table": table,
            "_note": f"{len(out)} defects, {sum(x['kept'] for x in out.values())} structures "
                     f"in {tag}"}


# The structures D2 recomputes with the hybrid: about fraction of them, as in the paper: the first
# and last step of each relaxation, then an even sample of the rest. Returns their indices.
#   Called by: step_hybrid()
def hybrid_picks(frames, fraction):
    runs = {}
    for i, a in enumerate(frames):
        runs.setdefault(a.info.get("run", ""), []).append(i)
    pick = set()
    for idx in runs.values():
        idx = sorted(idx, key=lambda i: frames[i].info.get("step", 0))
        pick.update({idx[0], idx[-1]})
    want = max(len(pick), int(round(fraction * len(frames))))
    rest = [i for i in range(len(frames)) if i not in pick]
    if want > len(pick) and rest:
        n = min(len(rest), want - len(pick))
        pick.update(rest[int(x)] for x in np.linspace(0, len(rest) - 1, n))
    return sorted(pick)


# D2 (model_hybrid : true): hybrid single points (hybrid_param, e.g. xc_functional : HSE06, and
# hybrid_cell, its species_pot with norm-conserving potentials, which CASTEP's non-local exchange
# needs) of about model_hybrid_fraction of each defect's D1 structures and rattled pristine cells
# (hybrid_picks), at Gamma, in their charge state with their run's spin to start, in the cell they
# are in: the one the survivors work in after C3 (work_cell; the electronic cell by default, where
# a hybrid costs many times what it does in the screening cell). They run in D_states/jobs/hybrid
# (runner group 'hybrid'): the 'hse' head's data, split as D1 split them, in
# D_states/model/hybrid/<defect>/hse_train.xyz and hse_test.xyz. With model_hybrid : false the
# model has the 'pbe' head only, and D5 searches with it.
#   Called by: main() (step D2)
#   Calls:     Project.batch(), Project.file(), castep_frames(), format_table(), hybrid_picks(),
#              model_dir(), qtag(), start_spin(), work_cell()
def step_hybrid(p):
    if not (p.s.model and p.s.model_hybrid):
        return {"skipped": True, "_note": "no hybrid head (model or model_hybrid false)"}
    import ase.io
    data = p.out("dataset")["defects"]
    param, cell = p.file("hybrid_param").read_text(), p.file("hybrid_cell").read_text()
    _, tag, end = work_cell(p)
    jobs, meta = [], {}
    for cid, x in sorted(data.items()):
        frames = []
        for split in ("train", "test"):
            if split in x["files"]:
                frames += ase.io.read(str(p.dir / x["files"][split]), ":")
        for n, i in enumerate(hybrid_picks(frames, p.s.model_hybrid_fraction)):
            a = frames[i]
            q = int(a.info["total_charge"])
            name = f"{cid}_{qtag(q)}_h{n + 1:04d}{end}"
            atoms = Atoms(a.get_chemical_symbols(), positions=a.positions, cell=a.cell, pbc=True)
            S = start_spin(atoms, q, a.info.get("spin"))
            jobs.append({"name": name, "atoms": atoms, "kind": "sp", "kgrid": [1, 1, 1],
                         "symmetry": False,
                         "overrides": {"charge": q, "spin": S, "spin_polarized": "true"}})
            meta[name] = (cid, q, a.info.get("split", "train"), a.info.get("run", ""), S)
    res = p.batch(f"{PHASE_DIRS['D']}/jobs/hybrid", jobs, "hybrid", param, cell_text=cell)
    if not res["complete"]:
        return None
    got, rows = {}, []
    for name, (cid, q, split, run, S) in meta.items():
        r = res["jobs"][name]
        if r["energy"] is None or not r["scf_converged"]:
            continue
        for a in castep_frames(res["calcs"][name].castep, q, S, run):
            a.info["split"] = split
            got.setdefault(cid, []).append(a)
    out = {}
    for cid in sorted(data):
        frames = got.get(cid, [])
        d = model_dir(p, "hybrid", cid)
        files = {}
        for split in ("train", "test"):
            sel = [a for a in frames if a.info["split"] == split]
            if sel:
                ase.io.write(str(d / f"hse_{split}.xyz"), sel, format="extxyz")
                files[split] = str((d / f"hse_{split}.xyz").relative_to(p.dir))
        asked = sum(1 for m in meta.values() if m[0] == cid)
        out[cid] = {"asked": asked, "done": len(frames), "files": files}
        rows.append([cid, asked, len(frames), sum(a.info["split"] == "train" for a in frames)])
        if not frames:
            log(f"WARNING: D2: no hybrid single point of {cid} finished (or printed forces): its "
                "model will have no 'hse' head data")
    table = format_table(["id", "asked", "finished", "train"], rows,
                         f"Hybrid single points for the 'hse' head, in {tag}")
    return {"defects": out, "_table": table,
            "_note": f"{sum(x['done'] for x in out.values())} hybrid single points"}


# The mace_run_train command of a defect's model, run in its folder (the training files copied
# there): heads 'pbe' (and 'hse' when there is hybrid data), each with its train and test file,
# 10 % of the training set for validation and E0s fitted to the data (average); the total charge
# as a categorical embedding (CHARGE_CLASSES classes from -CHARGE_OFFSET); trained from scratch
# (multiheads_finetuning off); model_r_max, model_channels, model_max_L, model_batch,
# model_epochs, model_device (default mace_device) and model_train_args added at the end.
#   Called by: step_lineshape(), train_tasks()
def train_command(p, heads):
    hd = {h: {"train_file": f"{h}_train.xyz", "valid_fraction": 0.1, "E0s": "average",
              **({"test_file": f"{h}_test.xyz"} if has_test else {})}
          for h, has_test in heads}
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


# The training of some defects' models: for each, a folder (part/<defect>) with its data (the
# D1 and D2 files, and extra structures added by D4: extra {defect: {head: [Atoms]}}) and
# train.sh (a changed training set clears the folder's old model: check_training). Returns
# [{'cid', 'dir', 'command', 'heads'}].
#   Called by: step_check(), step_train()
#   Calls:     check_training(), model_dir(), train_command()
def train_tasks(p, part, cids, extra=None, base=None):
    import ase.io
    tasks = []
    for cid in cids:
        d = model_dir(p, part, cid)
        heads = []
        for head, step in (("pbe", "dataset"), ("hse", "hybrid")):
            files = ((p.out(step).get("defects") or {}).get(cid) or {}).get("files") or {}
            frames = {s: (ase.io.read(str(p.dir / files[s]), ":") if s in files else [])
                      for s in ("train", "test")}
            if base:                               # the previous round's training sets
                prev = base / cid / f"{head}_train.xyz"
                frames["train"] = ase.io.read(str(prev), ":") if prev.exists() else []
            frames["train"] += (extra or {}).get(cid, {}).get(head, [])
            if not frames["train"]:
                continue
            for s, sel in frames.items():
                if sel:
                    ase.io.write(str(d / f"{head}_{s}.xyz"), sel, format="extxyz")
            heads.append((head, bool(frames["test"])))
        cmd = train_command(p, heads)
        (d / "train.sh").write_text("#!/bin/bash\n# The ground-state model of "
                                    f"{cid} (dwf.py)\n{cmd}\n")
        check_training(d, cmd)
        tasks.append({"cid": cid, "dir": d, "command": cmd, "heads": [h for h, _ in heads]})
    return tasks


# The model file a training wrote in its folder (the second stage's, if there is one), or None.
#   Called by: run_training(), step_check(), step_lineshape(), step_train()
def trained_model(d):
    for name in ("model_stagetwo.model", "model.model"):
        if (Path(d) / name).exists():
            return Path(d) / name
    return None


# Runs model jobs until each has made what it makes: trainings (train_tasks; done when its folder
# has a model, trained_model) and G2's phonon calculations (a task's own 'done' file). With
# model_jobscript as batch jobs (the template's {COMMAND} line becomes 'cd <folder> && <command> >
# <log> 2>&1', the task's 'log', default train.log; {JOBNAME} and {NCORES} filled in; submitted
# with the runner's submit_cmd, polled every poll s; --prepare-only writes them only, --no-wait
# returns while they run), else here, one after the other. One that ends without its result stops
# the run with the end of its log. A task: {'cid' (its name), 'dir', 'command', and optionally
# 'done', 'log', 'what' (for messages, default 'training') and 'job' (the job name's prefix)}.
# Returns True when every task is done.
#   Called by: step_check(), step_lineshape(), step_train()
#   Calls:     Project.mace_runner(), Project.template(), sc.job_active(), sc.send_job(),
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
        if info.get("jobid") and sc.job_active(info["jobid"], args):
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
        jobid = sc.send_job(t["dir"], "job.sh", args)
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
            if not sc.job_active(info["jobid"], args) and time.time() - info["time"] > 120:
                die(f"{label}: the {t.get('what', 'training')} job of {t['cid']} "
                    f"({info['jobid']}) ended without its result: {tail(t)}"
                    + (" (rerun with --resubmit-failed to carry on where it stopped)"
                       if t.get("done") else ""))
        if p.cl.no_wait:
            log(f"{label}: {len(left)} jobs still running; rerun the same command later")
            return False
        time.sleep(args.poll)


# D3: trains each defect's model (train_tasks, run_training) in D_states/model/train/<defect>:
# heads 'pbe' and, with hybrid data, 'hse', a total-charge embedding, from scratch.
#   Called by: main() (step D3)
#   Calls:     format_table(), run_training(), train_tasks(), trained_model()
def step_train(p):
    if not p.s.model:
        return {"skipped": True, "_note": "model : false"}
    tasks = train_tasks(p, "train", sorted(p.out("dataset")["defects"]))
    if not run_training(p, "D3", tasks):
        return None
    models = {t["cid"]: {"model": str(trained_model(t["dir"]).relative_to(p.dir)),
                         "heads": t["heads"]} for t in tasks}
    table = format_table(["id", "heads", "model"],
                         [[c, " ".join(m["heads"]), m["model"]] for c, m in sorted(models.items())],
                         "The ground-state models")
    return {"models": models, "_table": table, "_note": f"{len(models)} models trained"}


# A model's errors on structures with reference energies and forces (one head): the root mean
# square error of the energy per atom (meV/atom) and of the force components (meV/A), or None
# without structures.
#   Called by: step_check()
#   Calls:     Project.mace()
def model_errors(p, model, head, frames):
    if not frames:
        return None
    calc = p.mace(model, head)
    de, df = [], []
    for a in frames:
        b = a.copy()
        b.info["charge"] = a.info["total_charge"]
        b.calc = calc
        de.append((b.get_potential_energy() - a.info["REF_energy"]) / len(a))
        df.append((b.get_forces() - a.arrays["REF_forces"]).ravel())
    return {"E": round(1000 * float(np.sqrt(np.mean(np.square(de)))), 2),
            "F": round(1000 * float(np.sqrt(np.mean(np.square(np.concatenate(df))))), 2)}


# D4: checks each defect's model, in up to model_rounds rounds, in the cell the survivors work in
# (work_cell: the electronic cell by default). Each round: its errors on the test structures of
# each head (model_errors), then its minimum in each charge state (its lowest C2/C5 structure,
# lowest_start, relaxed with the search head, 'hse' when there is one, to search_fmax), and single
# points there with the hybrid (when the model has an 'hse' head) and at the screening settings
# (D_states/jobs/check_r<n>): the model's minimum is good if the DFT
# force on every atom is below model_force_tol (the search head's level). If not, those single
# points are added to the training data and the model is trained again
# (D_states/model/check/round_<n>/<defect>). The results give the model each defect's search
# uses (the last one trained).
#   Called by: main() (step D4)
#   Calls:     Project.batch(), Project.file(), Project.host(), Project.production_param(),
#              build_defect(), carry_over(), castep_frames(), format_table(), load_cell(),
#              lowest_start(), mace_relaxations(), model_errors(), qtag(), run_training(),
#              screen_overrides(), start_spin(), train_tasks(), trained_model(), work_cell()
def step_check(p):
    if not p.s.model:
        return {"skipped": True, "_note": "model : false"}
    import ase.io
    trained = p.out("train")["models"]
    hybrid = p.s.model_hybrid and any("hse" in m["heads"] for m in trained.values())
    head = "hse" if hybrid else "pbe"
    survivors = {s["cand"]: s for s in p.out("filter")["survivors"]}
    cands = p.out("generate")["candidates"]
    models = {cid: p.dir / m["model"] for cid, m in trained.items()}
    M, tag, end = work_cell(p)
    rounds, extra = {}, {}
    for rnd in range(1, p.s.model_rounds + 1):
        tasks, meta = [], {}
        for cid, model in sorted(models.items()):
            for q in survivors[cid]["charges"]:
                b, Mb = lowest_start(p, survivors[cid], q)
                if b is None:
                    continue
                name = f"{cid}_{qtag(q)}_r{rnd}{end}"
                atoms = load_cell(p, b["structure"])
                if not np.array_equal(np.array(Mb), M):        # (its screened structure, C2)
                    atoms = carry_over(p.host(), cands[cid]["comps"], np.array(Mb), atoms, M,
                                       label=f"{cid} ({cands[cid]['name']})")[0]
                tasks.append({"atoms": atoms, "label": name,
                              "cache": p.phase_dir("D") / "mace" / "check" / f"{name}.json",
                              "model": str(model), "head": head, "charge": q,
                              "fmax": p.s.search_fmax, "steps": p.s.search_steps})
                meta[name] = (cid, q, b.get("spin"))
        relaxed = mace_relaxations(p, "check", tasks)
        if relaxed is None:
            return None
        jobs = {"pbe": [], "hse": []}
        for t in tasks:
            cid, q, spin = meta[t["label"]]
            atoms = relaxed[str(t["cache"])][0]
            info = build_defect(p.host(), M, cands[cid]["comps"], cid)[1]
            S = start_spin(atoms, q, spin)
            mags = np.zeros(len(atoms))
            if S and info.get("anchor_index") is not None:
                mags[info["anchor_index"]] = S
            atoms.set_initial_magnetic_moments(mags)
            o = {"charge": q, "spin": S, "spin_polarized": "true"}
            jobs["pbe"].append({"name": t["label"], "atoms": atoms, "kind": "sp",
                                "kgrid": [1, 1, 1], "symmetry": False,
                                "overrides": dict({k: v for k, v in screen_overrides(p).items()
                                                   if k != "geom_max_iter"}, **o)})
            if hybrid:
                jobs["hse"].append(dict(jobs["pbe"][-1], overrides=dict(o)))
        res = {"pbe": p.batch(f"{PHASE_DIRS['D']}/jobs/check_r{rnd}", jobs["pbe"], "check",
                              p.production_param())}
        if hybrid:
            res["hse"] = p.batch(f"{PHASE_DIRS['D']}/jobs/check_r{rnd}_hse", jobs["hse"],
                                 "check", p.file("hybrid_param").read_text(),
                                 cell_text=p.file("hybrid_cell").read_text())
        if not all(r["complete"] for r in res.values()):
            return None
        rows, failed, new = [], set(), {}
        for t in tasks:
            cid, q, spin = meta[t["label"]]
            got = {}
            for h, r in res.items():
                rr = r["jobs"][t["label"]]
                if rr["energy"] is None:
                    continue
                fr = castep_frames(r["calcs"][t["label"]].castep, q, spin, t["label"])
                if fr:
                    got[h] = fr[-1]
                    new.setdefault(cid, {}).setdefault(h, []).append(fr[-1])
            ref = got.get(head)
            fmax = (None if ref is None else
                    float(np.linalg.norm(ref.arrays["REF_forces"], axis=1).max()))
            ok = fmax is not None and fmax <= p.s.model_force_tol
            if not ok:
                failed.add(cid)
            rows.append([rnd, cid, f"{q:+d}", fmax, "yes" if ok else "NO"])
        errs = {}
        for cid, model in sorted(models.items()):
            errs[cid] = {}
            for h, step in (("pbe", "dataset"), ("hse", "hybrid")):
                files = ((p.out(step).get("defects") or {}).get(cid) or {}).get("files") or {}
                if "test" in files and (h == "pbe" or hybrid):
                    errs[cid][h] = model_errors(p, str(model), h,
                                                ase.io.read(str(p.dir / files["test"]), ":"))
        rounds[rnd] = {"errors": errs, "forces": rows, "failed": sorted(failed)}
        if not failed or rnd == p.s.model_rounds:
            break
        for cid in failed:                         # add the new single points, train again
            for h, frames in new.get(cid, {}).items():
                extra.setdefault(cid, {}).setdefault(h, []).extend(frames)
        prev = (p.phase_dir("D") / "model" / "train" if rnd == 1 else
                p.phase_dir("D") / "model" / "check" / f"round_{rnd}")
        tasks2 = train_tasks(p, f"check/round_{rnd + 1}", sorted(failed),
                             {c: extra[c] for c in failed}, base=prev)
        if not run_training(p, f"D4 round {rnd + 1}", tasks2):
            return None
        for t in tasks2:
            models[t["cid"]] = trained_model(t["dir"])
    last = rounds[max(rounds)]
    rows = [r for rd in rounds.values() for r in rd["forces"]]
    erows = [[rnd, cid, h, e["E"] if e else None, e["F"] if e else None]
             for rnd, rd in rounds.items() for cid, per in rd["errors"].items()
             for h, e in per.items()]
    table = (format_table(["round", "id", "head", "RMSE E/meV per atom", "RMSE F/meV/A"], erows,
                          "Test errors of each model")
             + "\n" + format_table(["round", "id", "q", "max DFT force at the model's minimum/eV/A",
                                    f"below {p.s.model_force_tol:g}"], rows,
                                   f"Forces at the minima of the '{head}' head"))
    if last["failed"]:
        log(f"WARNING: D4: after {max(rounds)} rounds the model of "
            + ", ".join(last["failed"]) + " still leaves DFT forces above model_force_tol at its "
            "minima: D5's search uses it anyway, and D7 relaxes its structures with DFT")
    return {"models": {cid: str(Path(m).relative_to(p.dir)) for cid, m in models.items()},
            "head": head, "rounds": {str(k): v for k, v in rounds.items()},
            "failed": last["failed"], "_table": table,
            "_note": f"{len(models)} models, search head '{head}', {len(rounds)} round(s)"
                     + (f"; still above model_force_tol: {', '.join(last['failed'])}"
                        if last["failed"] else "")}


# D5: the ground-state search with each defect's model (D4's), in the electronic cell (B4), in
# every charge state it carries on: from its lowest C2/C5 structure (lowest_start: C5's, already
# in the electronic cell; with distort_cell : screening, the lowest of C2's and C5's, carried
# over), every distortion of search_distortions around each point C4 distorts around
# (distortion_centres, ShakeNBreak's distort_and_rattle, with search_rattles more rattled copies
# of each), relaxed with the search head (D4's) to search_fmax in at most search_steps
# (D_states/mace/search; on compute nodes with mace_jobscript); then every distinct minimum of one
# charge state is relaxed in the others. The lowest search_keep distinct minima of each charge
# state are saved (D_states/structures/search) for D6.
#   Called by: main() (step D5)
#   Calls:     build_defect(), carry_over(), distortion_centres(), format_table(), host_supercell(),
#              load_cell(), lowest_start(), mace_relaxations(), model_defects(),
#              neighbours_to_distort(), oxidation_states(), qtag(), same_structures(), save_cell(),
#              screen_setup(), start_spin()
def step_search(p):
    if not p.s.model:
        return {"skipped": True, "_note": "model : false"}
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        from shakenbreak.distortions import distort_and_rattle
    except ImportError as err:
        die(f"D5 needs ShakeNBreak (pip install shakenbreak): {err}")
    import warnings
    warnings.filterwarnings("ignore")
    cands, prim, P0 = screen_setup(p)
    Pe = np.array(p.out("electronic")["matrix"])
    chk = p.out("check")
    head = chk["head"]
    survivors = {s["cand"]: s for s in p.out("filter")["survivors"]}
    ox = oxidation_states(p, prim, cands)
    host = host_supercell(prim, Pe)
    bond = float(np.sort(host.get_all_distances(mic=True)[0])[1])
    stdev, dmin = p.s.rattle_stdev or 0.1 * bond, p.s.rattle_dmin or 0.8 * bond
    cache = p.phase_dir("D") / "mace" / "search"
    sdir = p.phase_dir("D") / "structures" / "search"

    # A relaxation task with the defect's model in charge state q.
    def task(cid, q, atoms, name, source):
        return {"atoms": atoms, "label": name, "cache": cache / f"{name}.json",
                "model": str(p.dir / chk["models"][cid]), "head": head, "charge": q,
                "fmax": p.s.search_fmax, "steps": p.s.search_steps, "source": source,
                "cid": cid, "q": q}

    tasks = []
    for cid, qs in sorted(model_defects(p).items()):
        comps, label = cands[cid]["comps"], f"{cid} ({cands[cid]['name']})"
        for q in qs:
            b, M = lowest_start(p, survivors[cid], q)
            if b is None:
                continue
            base = load_cell(p, b["structure"])
            if np.array_equal(np.array(M), Pe):
                info, how = build_defect(prim, Pe, comps, label)[1], ""
            else:
                base, info = carry_over(prim, comps, np.array(M), base, Pe, label=label)
                how = " (carried over)"
            tasks.append(task(cid, q, base, f"{cid}_{qtag(q)}_base",
                              f"{b['step']} {b['name']}{how}"))
            nb = neighbours_to_distort(comps, q, ox)
            struct = AseAtomsAdaptor.get_structure(base)
            centres = distortion_centres(base, comps, info, p.s.distort_around, bond)
            factors = [x for x in p.s.search_distortions if abs(x) > 1e-9] if nb else []
            for k, (where, centre) in enumerate(centres):
                for fct in factors + ["rattled"]:
                    for r in range(1 + p.s.search_rattles):
                        out = distort_and_rattle(struct, 1.0 if fct == "rattled" else 1.0 + fct,
                                                 nb if fct != "rattled" else 0, stdev=stdev,
                                                 d_min=dmin, seed=MODEL_SEED + r, **centre)
                        new = AseAtomsAdaptor.get_atoms(out["distorted_structure"])
                        tag = "rattled" if fct == "rattled" else \
                            f"b{'m' if fct < 0 else 'p'}{abs(fct):.2f}"
                        tasks.append(task(cid, q, new,
                                          f"{cid}_{qtag(q)}_c{k + 1}_{tag}_r{r}",
                                          f"{tag} around {where}"))
    res = mace_relaxations(p, "search", tasks)
    if res is None:
        return None

    # The distinct minima of a list of relaxed tasks, lowest first (same_structures).
    def distinct(items):
        items = sorted(items, key=lambda x: x["energy"])
        groups = same_structures([x["atoms"] for x in items])
        return [items[min(g)] for g in sorted(groups, key=min)]

    found = {}
    for t in tasks:
        a, _, e1, steps = res[str(t["cache"])]
        found.setdefault((t["cid"], t["q"]), []).append(
            {"name": t["label"], "atoms": a, "energy": e1, "source": t["source"],
             "steps": steps})
    share = []
    for (cid, q), items in found.items():
        for x in distinct(items)[:p.s.search_keep]:
            for (c2, q2) in found:
                if c2 == cid and q2 != q:
                    share.append(task(cid, q2, x["atoms"], f"{x['name']}_to_{qtag(q2)}",
                                      f"from {qtag(q)}: {x['source']}"))
    if share:
        res2 = mace_relaxations(p, "search", share)
        if res2 is None:
            return None
        for t in share:
            a, _, e1, steps = res2[str(t["cache"])]
            found[(t["cid"], t["q"])].append({"name": t["label"], "atoms": a, "energy": e1,
                                              "source": t["source"], "steps": steps})
    states, rows = {}, []
    for (cid, q), items in sorted(found.items()):
        keep = distinct(items)[:p.s.search_keep]
        spin = (survivors[cid]["best"].get(str(q)) or {}).get("spin")
        out = []
        for x in keep:
            rel = save_cell(p, sdir / f"{x['name']}.cell", x["atoms"],
                            f"{x['name']}: {x['source']}, the model's minimum ({head} head)")
            out.append({"name": x["name"], "energy": round(x["energy"], 5), "source": x["source"],
                        "structure": rel, "spin": start_spin(x["atoms"], q, spin)})
        states.setdefault(cid, {})[str(q)] = out
        base = next((x["energy"] for x in items if x["name"].endswith("_base")), None)
        rows.append([cid, f"{q:+d}", len(items), len(keep), base, out[0]["energy"],
                     None if base is None else base - out[0]["energy"], out[0]["source"]])
    table = format_table(["id", "q", "relaxed", "distinct kept", "start/eV", "lowest/eV",
                          "gain/eV", "lowest from"], rows,
                         f"Ground-state search with the model ('{head}' head) in the electronic "
                         f"cell ({p.out('electronic')['tag']})")
    return {"states": states, "head": head, "matrix": Pe.tolist(), "_table": table,
            "_note": f"{len(tasks) + len(share)} model relaxations, "
                     f"{sum(len(v) for s in states.values() for v in s.values())} distinct "
                     "minima kept"}


# --------------------------------------------------------------------------
# D6: ground states (with model : true, the minima of D5's search with each defect's model;
# with model : false, C5's relaxations, distortions shared between charge states)
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


# D6 with the ground-state model (D5 ran): keeps the lowest keep_structures distinct minima of the
# model's search in each charge state (D5's, already merged and sorted by the model's energy), in
# the electronic cell (their 'matrix', and 'cell' its tag), for D7 to relax with DFT.
#   Called by: step_rank()
#   Calls:     format_table()
def rank_model(p, srch):
    tag = p.out("electronic")["tag"]
    keep, rows = [], []
    for cid, per_q in sorted(srch["states"].items()):
        for qs, items in sorted(per_q.items(), key=lambda kv: int(kv[0])):
            q = int(qs)
            for k, e in enumerate(items[:p.s.keep_structures]):
                keep.append({"cand": cid, "q": q, "k": k, "energy": e["energy"],
                             "spin": e["spin"], "structure": e["structure"],
                             "source": e["source"], "factor": None, "name": e["name"],
                             "band": None, "matrix": srch["matrix"], "cell": tag})
            rows.append([cid, f"{q:+d}", items[0]["energy"], items[0]["source"], len(items)])
    table = format_table(["id", "q", "model E/eV", "lowest from", "distinct"], rows,
                         f"Ground states from the model's search ('{srch['head']}' head, D5)")
    return {"structures": keep, "model": True, "_table": table,
            "_note": f"{len(keep)} structures kept from the model's search"}


# D6: ranks the relaxed structures of each survivor and charge state in the cell C4 worked in
# (distinct_states). A distortion counts as found when it beats the start by distortion_min_gain:
# the screened structure (C2's run; in the electronic cell, its relaxation there, C4's undistorted
# job). A structure found in one charge state that no other charge state has is then relaxed in
# those charge states too, in the same cell (a batch per round, D_states/jobs/share_<n>, measured
# against C5's band edges), for up to share_rounds rounds or until nothing new turns up. The lowest
# keep_structures distinct structures of each charge state go on to D7, and then to phase E, with
# the run each came from, the electrons or holes it left in the bands (D8 leaves out band states)
# and its cell ('matrix', 'cell').
#   Called by: main() (step D6)
#   Calls:     distinct_states(), format_table(), load_cell(), qtag(), rank_model(),
#              run_screen_jobs(), same_structures(), start_spin()
def step_rank(p):
    srch = p.out("search")
    if not srch.get("skipped"):
        return rank_model(p, srch)
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
        log(f"D6 round {rnd}: trying {len(todo)} structures found in one charge state in the "
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
                     best["source"] + (f" {best['factor']}" if best["factor"] is not None else ""),
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
# D7: excited states
# --------------------------------------------------------------------------
# Whether a candidate is a native defect: every element it adds or removes is one of the host's
# (vacancies, antisites, self-interstitials and complexes of them). D7 always passes these on to
# E, because F's formation energies and Fermi level need them, bright or not.
#   Called by: step_excite(), step_filter()
def is_native(cand, host_elements):
    return all(el in host_elements for el in cand["delta"])


# The defect levels of a ground state at one k-point (Gamma in D7): every band within reach (eV)
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


# The excitations D7 and E2 try, from a ground state's defect levels (defect_levels): in each spin
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


# A D7 job for one structure in charge state q, starting with spin S: a screening job (screen_job:
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


# The settings every ΔSCF calculation of the production phases and its ground state share (E2,
# G1), as D7's (excite_job): metals_method : dm (CASTEP's excitation constraints need density
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
#   Called by: dscf_result()
def first_energy(text, which):
    for kind in (["0K", "final"] if which == "auto" else [which]):
        m = re.search(sc.ENERGY_PATTERNS[kind], text)
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
#              save_cell(), sc.final_atoms()
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
        atoms = sc.final_atoms(calc)
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


# A D6 structure t as a structure of the electronic cell Pe (B4): as it is when it is already in
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


# D7: the screen of the excited states, in the electronic cell (B4) at the screening settings
# (Gamma, screen_cutoff), in three batches (D_states/jobs/excite_relax, excite_gs, excite):
#  1. the pristine cell (its band edges), and every structure D6 kept, relaxed: the lowest that
#     relaxes is the ground state (D6's order may change); it is flagged if it slid more than
#     BASIN_TOL from where it started (same_basin: from the model's minimum, a structure the
#     semilocal functional doesn't keep). A band state (in_band) is 'pending': it can't be judged
#     here (a hybrid can). With excite_spin (2, a triplet; 'any' for every spin), a ground state
#     whose final spin is further than spin_tol from it goes no further ('spin').
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
# 'unknown' and counts as a pass, with a warning). A defect goes on to E with all its charge
# states (their relaxed ground states here) if one passes, and every native defect (is_native)
# goes on anyway (its ground states, for F). With excite : false, every structure D6 kept as its
# lowest goes on to E as it is.
#   Called by: main() (step D7)
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
                "native": [], "_note": "excite is off: the lowest structure D6 kept of every "
                                       "charge state goes on to E"}
    d = p.phase_dir("D")
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
    res = p.batch(f"{PHASE_DIRS['D']}/jobs/excite_relax", jobs, "excite", p.production_param())
    if not res["complete"]:
        return None
    hc = res["calcs"]["host"]
    if res["jobs"]["host"]["status"] != "done" or not (hc.dir / f"{hc.seed}.bands").exists():
        die(f"D7: the pristine electronic cell did not finish ({hc.dir}); the band edges come "
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
        res2 = p.batch(f"{PHASE_DIRS['D']}/jobs/excite_gs", [x[2] for x in sp_jobs.values()],
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
        res3 = p.batch(f"{PHASE_DIRS['D']}/jobs/excite", ex_jobs, "excite",
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
            log(f"WARNING: D7: {cid} {cands[cid]['name']} q {q:+d}: {e['why']} - kept")
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
                        "matrix": Pe.tolist(), "spin": e["spin"], "source": "D7",
                        "excite": [{k: ex[k] for k in ("spin", "from_band", "to_band")}
                                   for ex in e["transitions"] if ex.get("verdict") == "pass"]})
        elif cid in keep:                         # its relaxation failed: D6's structure
            t0 = next(t for t in structures if t["cand"] == cid and t["q"] == q and t["k"] == 0)
            out.append(dict(t0, excite=[]))
    what = "ZPL" if p.s.excite_relax else "vertical E"
    table = format_table(["id", "defect", "q", "spin", "flags", "spin: bands", "KS step/eV",
                          "P from/to", f"{what}/eV", "E_abs/eV", "dQ/amu^0.5 A", "S (1D)",
                          "DW (1D)", "verdict", "on to E"], rows,
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
            "_note": f"{npass} charge states pass; {len(keep)} defects go on to E "
                     f"({len(native)} native)"}


# --------------------------------------------------------------------------
# D8: the finite-size check (optional)
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


# D8 (optional, validate : true): whether the screening cell, with its corrections, gives the
# formation energies and transition levels of much bigger cells, so that phase E's size series can
# be left out in later projects. For the lowest structure D6 kept of each defect (those in
# validate_defects, or every one D6 kept) in each charge state that isn't a band state: a
# relaxation at the production settings (production_job, but without freeze_radius, so the far
# field relaxes too) in the screening cell and in the next validate_sizes sizes of the ladder up
# that fit it (D6's structure carried over, carry_over), with the pristine cell's single point in
# each size; all with calculate_stress. In each size, E = E_D - E_host + E_corr (the charge
# correction, in that cell's shape); the reference E_inf is the line E_inf + b/N through the sizes
# above the screening cell (tail_fit; N, the host cell's atoms, as in B2), and b is the tail the
# sizes really show. Three estimates of E_inf from the screening cell alone are compared with it:
# E as it is; E - c/N with B2's c (MACE: one c per arrangement, blind to the charge); and E less
# the energy of the stress the fixed cell holds, 1/2 V s:S:s (elastic_energy; s, the defect cell's
# stress less the host's in the same size, so the Pulay stress of the basis cancels; S, the
# compliance from elastic_constants), each charge state's own. The transition levels between the
# charge states checked are compared the same way. A defect passes with an estimate when every
# error is within validate_tol. The jobs run in D_states/jobs/validate with the runner settings of
# runner_production (or runner_validate); the relaxed structures go to D_states/structures/validate.
# D6's structure is carried from the cell it is in (its 'matrix': the screening cell, or the
# electronic cell when D5's model found it). With validate_gamma (true), the same relaxed
# structures also get Gamma-only single points (validate_gamma), to show what Gamma-only sampling
# would change in each size. Nothing later uses the results: they are a test of the method.
#   Called by: main() (step D8)
#   Calls:     Project.batch(), Project.host(), Project.kspacing(), Project.production_param(),
#              build_defect(), carry_over(), compliance(), correction(), elastic_energy(), fits(),
#              format_table(), host_supercell(), in_band(), load_cell(), production_job(),
#              production_summary(), qtag(), read_stress(), tail_fit(), validate_gamma()
def step_validate(p):
    if not p.s.validate:
        return {"skipped": True, "_note": "validate : false"}
    d = p.phase_dir("D")
    cands, prim = p.out("generate")["candidates"], p.host()
    choose = p.out("choose")
    ladder, idx, P0 = choose["ladder"], choose["index"], np.array(choose["matrix"])
    eps = p.out("dielectric").get("eps0")
    S = compliance(p.s.elastic_constants)
    if S is None:
        log("D8: no %block elastic_constants, so the stress estimate is left out")
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
            log(f"WARNING: D8: {x} is in validate_defects but D6 kept no bound structure of it")
    if skipped:
        log("D8: left out (a band state, so there is no defect level to check): "
            + ", ".join(skipped))
    for cid in sorted(ground):
        up = [i for i in range(idx + 1, len(ladder)) if fits(cands[cid], ladder[i]["L"])]
        if not up:
            log(f"WARNING: D8: {cid} not checked: no size of the ladder above the screening cell "
                "fits it (longer scan_lengths add sizes)")
            del ground[cid]
        else:
            sizes[cid] = [idx] + up[:p.s.validate_sizes]
    if not ground:
        log("WARNING: D8: nothing to check (see above)")
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
                start = load_cell(p, t["structure"])          # D6's, in its own cell
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
    res = p.batch(f"{PHASE_DIRS['D']}/jobs/validate", jobs, "validate", p.production_param(),
                  spacing=p.kspacing())
    if not res["complete"]:
        return None
    hosts = {}
    for i in used:
        name = f"host_{ladder[i]['tag']}"
        r = res["jobs"][name]
        if r["status"] != "done" or r["energy"] is None:
            die(f"D8: the pristine {ladder[i]['tag']} single point did not finish ({r['dir']})")
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
        log(f"WARNING: D8: no stress tensor in {len(nostress)} runs (e.g. {nostress[0]}): does "
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
                log(f"WARNING: D8: {cid} q {q:+d} not checked: {why}")
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


# D8's check of Gamma-only sampling (validate_gamma : true): a single point at the production
# cut-off with only the Gamma point, of every structure D8 relaxed at the production k-points and
# of the pristine cell in each size used (D_states/jobs/validate_gamma). E = E_D - E_host with
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
    res = p.batch(f"{PHASE_DIRS['D']}/jobs/validate_gamma", jobs, "validate",
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
MODEL_KEYS = ("model", "model_command", "model_device", "model_epochs", "model_r_max",
              "model_channels", "model_max_L", "model_batch", "model_train_args", "mace_device")


# ==========================================================================================
# PHASE E: PRODUCTION GROUND AND EXCITED STATES (flowchart E1-E2)
# ==========================================================================================
# The %block ionic_constraints that holds still every atom further than radius (Ang) from all of
# the defect's components (points: their positions; three constraints per atom: x, y and z).
# CASTEP numbers the atoms of each species in the order of the .cell, which keeps the order
# within each species (written_order).
#   Called by: production_job()
#   Calls:     sc.get_distances()
def freeze_block(atoms, points, radius):
    _, dist = sc.get_distances(np.atleast_2d(points), atoms.positions, cell=atoms.cell, pbc=True)
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
#   Calls:     save_cell(), sc.final_atoms()
def production_summary(p, res, name, structdir, frozen=0):
    r = res["jobs"][name]
    e = r["energy"] if r["scf_converged"] and r["status"] == "done" else None
    atoms = sc.final_atoms(res["calcs"][name]) if e is not None else None
    if e is not None and atoms is None:
        log(f"WARNING: could not read the relaxed structure of {name}")
    path = (save_cell(p, structdir / f"{name}.cell", atoms,
                      f"{name}: relaxed at the production settings") if atoms is not None
            else None)
    return {"job": name, "energy": e, "spin_final": r["spin"], "status": r["status"],
            "reason": r["reason"], "steps": r["geom_steps"], "restarts": r["restarts"],
            "final": path, "frozen": frozen}, atoms


# E1: the production relaxations, in the electronic cell (B4): every charge state of each defect
# phase D passes on (D7's relaxed ground states there; with excite : false, the lowest structure D6
# kept of each charge state, carried in: into_cell), relaxed at the production settings
# (production_job: A1's cut-off and k-point spacing, %block production_overrides, freeze_radius),
# with the pristine electronic cell's single point, whose band edges E2 measures levels against
# (E_production/jobs/relaxations; the relaxed structures in E_production/structures/relaxations).
# Runner group 'production'. F takes its energies from here, E2 its ground states.
#   Called by: main() (step E1)
#   Calls:     Project.batch(), Project.host(), Project.kspacing(), Project.production_param(),
#              band_edges(), format_table(), host_supercell(), into_cell(), production_job(),
#              production_summary(), qtag(), read_bands(), start_spin()
def step_relaxations(p):
    d = p.phase_dir("E")
    todo = p.out("excite")["structures"]
    if not todo:
        die("phase D passed no structures on, so there is nothing to relax")
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
    res = p.batch(f"{PHASE_DIRS['E']}/jobs/relaxations", jobs, "relaxations",
                  p.production_param(), spacing=p.kspacing())
    if not res["complete"]:
        return None
    hr, hc = res["jobs"]["host"], res["calcs"]["host"]
    if hr["status"] != "done" or hr["energy"] is None:
        die(f"E1: the pristine electronic cell's single point did not finish ({hr['dir']}): the "
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


# E2: the production ΔSCF of every excitation D7 passed, in E1's relaxed ground state:
#  1. a single point at the production settings with the bands' weights (pdos_calculate_weights;
#     its .check kept; metals_method : dm and %block excite_overrides, dscf_settings), against
#     E1's pristine band edges: the excitation is found again in the same spin channel
#     (defect_levels, transitions), and the band-number check (zpl_bands_ok);
#  2. where that fails, a Gamma-only single point at the production cut-off instead (the
#     excitation found again there);
#  3. the ΔSCF (dscf_job) from whichever single point, relaxed with excite_relax : true: ZPL, the
#     vertical absorption, dQ, the 1D Huang-Rhys and Debye-Waller estimates and whether the
#     promoted electron stayed on the defect (dscf_result; its verdict uses D7's thresholds).
# The relaxed excited structures go to E_production/structures/zpl, for G.
#   Called by: main() (step E2)
#   Calls:     Project.batch(), Project.host(), Project.kspacing(), Project.production_param(),
#              build_defect(), defect_levels(), dscf_job(), dscf_result(), dscf_settings(),
#              format_table(), host_supercell(), load_cell(), participation(), production_job(),
#              qtag(), read_bands(), read_pdos_weights(), start_spin(), transitions(),
#              zpl_bands_ok()
def step_zpl(p):
    d = p.phase_dir("E")
    e1 = p.out("relaxations")
    todo = [r for r in e1["runs"].values() if r["excite"] and r["energy"] is not None
            and r["final"]]
    if not todo:
        return {"states": {}, "_note": "no excitation passed D7, so there is no ΔSCF to run"}
    if not e1["edges"]:
        die("E2: E1's pristine single point wrote no .bands, so the defect levels can't be found")
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

    # The excitations of a ground-state single point in the spin channels D7 passed, at its first
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
    res1 = p.batch(f"{PHASE_DIRS['E']}/jobs/zpl_gs", first, "zpl", p.production_param(),
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
        res2 = p.batch(f"{PHASE_DIRS['E']}/jobs/zpl_gs_gamma",
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
        res3 = p.batch(f"{PHASE_DIRS['E']}/jobs/zpl", jobs3, "zpl", p.production_param(),
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
                         "Production ΔSCF (A1's settings) in E1's ground states; k-points: the "
                         "production grid, or Gamma where the band-number check failed")
    js = {}
    for (cid, q), st in states.items():
        js.setdefault(cid, {})[str(q)] = st
    npass = sum(any(ex["verdict"] == "pass" for ex in st["excitations"])
                for st in states.values())
    return {"states": js, "relaxed": p.s.excite_relax, "_table": table,
            "_note": f"{npass} of {len(states)} charge states pass at the production settings"}


# ==========================================================================================
# PHASE F: THERMODYNAMICS (flowchart F1-F3)
# ==========================================================================================
KB = 8.617333262e-5                # eV/K
# chart colours (reference palette: light surface, categorical slots in their fixed order)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


# The production results (E1) of the runs that finished, by candidate and charge state (the run,
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


# F1: the finite-size correction (charge_correction, with the static dielectric constant from
# A2.3) for every charge state that reached phase E, in the electronic cell (E1's). Returns
# {defect: {q: E_corr}}.
#   Called by: main() (step F1)
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


# The chemical-potential limits (A3.2) as {name: {element: mu}}; without any, one entry 'mu=0' with
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


# The sets of defects F2 finds a Fermi level for, each as if only those dopants were in the crystal
# (so Sb's defects never set the Fermi level of a Bi-doped sample): 'native' (the native defects
# alone), each dopant X (the native defects and those that add only X), and each mix of dopants
# that one defect adds together, e.g. 'Bi+Sb' (with every defect whose dopants are among them).
# ids: the defects that reached phase E. Returns {name: [defect ids]}.
#   Called by: step_formation()
def dopant_sets(cands, ids, host_el):
    dop = {cid: frozenset(el for el, n in cands[cid]["delta"].items()
                          if el not in host_el and n > 0) for cid in ids}
    sets = {"native": sorted(c for c in ids if not dop[c])}
    for mix in sorted({x for x in dop.values() if x}, key=lambda x: (len(x), sorted(x))):
        sets["+".join(sorted(mix))] = sorted(c for c in ids if dop[c] <= mix)
    return sets


# F2: formation energies, transition levels and (with fermi_temperature) the Fermi level.
# For each defect that reached phase E, its energy in each charge state (E1, the electronic cell)
# gives E_f(q, E_F=0) = E_D - E_host - sum n_i mu_i + q E_VBM + E_corr at every limit, and its
# thermodynamic transition levels in the gap (transition_levels; the same at every limit); the
# plot goes to F_thermo/formation.png. With fermi_temperature, fermi_level finds the
# self-consistent Fermi level at each limit for each set of dopants on its own (dopant_sets), from
# their defects and the host's free carriers (quench_temperature: concentrations frozen at
# fermi_temperature, then charges redistributed).
#   Called by: main() (step F2)
#   Calls:     Project.host(), band_gap(), best_production(), dopant_sets(), fermi_level(),
#              format_table(), mu_limits(), plot(), read_bands(), transition_levels()
def step_formation(p):
    d = p.phase_dir("F")
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
                                   "both), from the defects that reached phase E"))
        log("note: the Fermi level includes only the defects that reached phase E; defects "
            "left out (e.g. native defects dropped in C) may change it")
    return {"vbm": vbm, "gap": gap, "limits": lims, "defects": results, "fermi": fermi,
            "dopant_sets": sets,
            "plot": str(png.relative_to(p.dir)) if png else None, "_table": "\n".join(tables),
            "_note": f"{len(best)} defects at {len(lims)} limit(s)"}


# The Fermi-level range (eV above the VBM, within [0, gap]) where each charge state of a defect is
# its lowest (F2's formation energies at one limit, {q: E_f at E_F = 0}; the ranges are the same at
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


# F3: the decision before phase G. A charge state goes on to the lineshape if its excitation
# passed at the production settings (E2) and it is the defect's lowest charge state for some Fermi
# level inside the gap (stable_windows, from F2): an excitation of a charge state that never
# forms can't be seen. The table also gives, for each set of dopants F2 found a Fermi level for
# (dopant_sets: the defect's own), whether that Fermi level falls in the range at each limit, the
# deep levels (more than deep_level_margin from both band edges), and the band states D7 left
# pending (to characterise with a hybrid later).
#   Called by: main() (step F3)
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
                          "E_F there at", "deep levels/eV", "on to G"], rows,
                         "Excitations that passed E2, and whether their charge state forms "
                         f"(lowest for some Fermi level in the {gap:.2f} eV gap); E_F there at: "
                         "the limits whose self-consistent Fermi level (F2, the defect's own "
                         "dopants) falls in that range")
    if pending:
        table += ("\nBand states D7 left pending (to characterise with a hybrid): "
                  + ", ".join(pending) + "\n")
    return {"for_G": picks, "pending": pending, "_table": table,
            "_note": f"{len(picks)} excitation(s) for the lineshape phase (G)"}


# ==========================================================================================
# PHASE G: OPTICAL LINESHAPES (flowchart G1-G2)
# ==========================================================================================
# The excitations F3 sends on, at the production settings (PBE with A1's cut-off and k-points, or
# Gamma where E2 fell back to it). G1: the 1D configuration-coordinate diagram, always; G2
# (lineshape : true): the multi-mode lineshape from a PBE model's phonons in a big cell. Both give
# the luminescence lineshape by the generating-function method (pl_spectrum), in the equal-mode
# approximation (the excited state vibrates as the ground state does; G1 reports both states'
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


# What G1's points give: xs, where they are along the path (0: the ground state's structure, 1:
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


# Draws a figure of side-by-side panels (G1, G2), each {'title', 'xlabel', 'ylabel', 'series':
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


# G1: the 1D configuration-coordinate diagram of each excitation F3 sends on (G_lineshape/jobs/
# ccd_gs, ccd_ex): the structures along the straight line from E1's relaxed ground state (x = 0)
# to E2's relaxed excited state (x = 1), at ccd_points; at each, a ground-state single point (its
# .check kept) and, from it, a ΔSCF single point of the same excitation (dscf_job), both at E2's
# settings (the production ones with dscf_settings, Gamma where E2 used it); the parabolas, mode
# energies and 1D Huang-Rhys factors of both states (ccd_fit) and the 1D luminescence lineshape at
# lineshape_temperature (pl_spectrum: the ground state's mode, S_g, broadened by lineshape_sigma,
# the ZPL by lineshape_gamma). Files in G_lineshape/ccd/: <excitation>_ccd.dat (x, Q and both
# energies), <excitation>_pl_1d.dat and <excitation>_ccd.png. Runner group 'production'.
#   Called by: main() (step G1)
#   Calls:     Project.batch(), Project.host(), Project.kspacing(), Project.production_param(),
#              build_defect(), ccd_fit(), delta_q(), draw_panels(), dscf_job(), dscf_settings(),
#              format_table(), job_energy(), load_cell(), pl_spectrum(), production_job(), qtag(),
#              start_spin(), write_columns()
def step_ccd(p):
    from ase.geometry import find_mic
    picks = p.out("interest")["for_G"]
    if not picks:
        return {"lines": {}, "_note": "F3 sent no excitation on to G"}
    d = p.phase_dir("G")
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
    res1 = p.batch(f"{PHASE_DIRS['G']}/jobs/ccd_gs", gs_jobs, "ccd", p.production_param(),
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
        res2 = p.batch(f"{PHASE_DIRS['G']}/jobs/ccd_ex", ex_jobs, "ccd", p.production_param(),
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
    table = format_table(["excitation", "defect", "k-points", "ZPL/eV", "ZPL E2", "E_abs",
                          "E_em", "dQ", "hw_g/eV", "hw_e/eV", "S_g", "S_e", "DW 0K"]
                         + ([f"DW {T:g}K"] if T > 0 else []) + ["misfit/eV"], rows,
                         "1D configuration-coordinate diagrams (ground-state and ΔSCF single "
                         "points along the line between the relaxed structures; dQ in amu^0.5 A; "
                         "S = hw dQ^2/2hbar^2; DW from S_g)")
    if skipped:
        table += "\nNot drawn: " + "; ".join(f"{t} ({w})" for t, w in skipped.items()) + "\n"
    return {"points": xs, "lines": lines, "skipped": skipped, "_table": table,
            "_note": f"{len(lines)} configuration-coordinate diagram(s)"}


# A defect's relaxed structure in the electronic cell (Pe) as G2's big cell (Pbig): carried over
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


# The vector G2 projects onto the phonons, on the big cell's atoms (big_v: their ideal positions
# relative to the defect's centre; big_syms: their elements): with lineshape_method forces, ΔF,
# the excited state's forces at the ground state's structure (the first step of E2's ΔSCF) less
# the ground state's there (E2's ground-state single point; zero if it printed none: the structure
# is relaxed), eV/Ang; with displacements, ΔR, the displacement of E2's relaxed excited structure
# from E1's relaxed ground state, Ang. Each atom's value goes to the same atom of the big cell
# (match_ideal, common_atoms), out to just under half the electronic cell's shortest image
# distance (the region that cell describes); zero further out. Returns (N x 3 array or None, the
# atoms that carry it, why None).
#   Called by: step_lineshape()
#   Calls:     common_atoms(), load_cell(), match_ideal(), min_image(), read_trajectory()
def lineshape_vector(p, pk, run, prim, comps, Pe, big_v, big_syms, label):
    jobs = p.dir / PHASE_DIRS["E"] / "jobs"
    if p.s.lineshape_method == "forces":
        xf = jobs / "zpl" / pk["ex_job"] / f"{pk['ex_job']}.castep"
        sub = "zpl_gs_gamma" if pk["gs_job"].endswith("_G") else "zpl_gs"
        gf = jobs / sub / pk["gs_job"] / f"{pk['gs_job']}.castep"
        xs = read_trajectory(xf.read_text(errors="replace"), "free") if xf.exists() else []
        if not xs:
            return None, 0, f"no forces in E2's ΔSCF output ({xf.name})"
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
#   Called by: step_lineshape(), train_tasks()
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


# G2 (lineshape : true): the multi-mode luminescence lineshape of each excitation F3 sends on.
#  1. Single points at the production settings (G_lineshape/jobs/lineshape_data): lineshape_host
#     pristine electronic cells and lineshape_displacements copies of each charge state's E1
#     ground state, every atom displaced at random (normal, lineshape_amplitude Ang a component).
#  2. A PBE model of each defect (G_lineshape/model/<defect>: train_command's MACE, one 'pbe'
#     head, the total-charge embedding): trained on those, E1's relaxations of all its charge
#     states (filter_frames) and the pristine cells, a fifth kept for testing; with
#     model_jobscript, run_training's jobs.
#  3. The phonons of each excitation's ground state with its model (phonon_worker: a job with
#     model_jobscript, else here; G_lineshape/phonons/<excitation>) in a big cell, the smallest with
#     images at least lineshape_length apart (size_matrix; the electronic cell when that is longer
#     already), the relaxed structure carried into it (big_cell), its force constants by finite
#     differences (lineshape_delta), each mode's part in the excitation (lineshape_vector:
#     lineshape_method forces, ΔF, or displacements, ΔR) and partial Huang-Rhys factor.
#  4. The lineshape (pl_spectrum) at lineshape_temperature, with G1's 1D one for comparison:
#     G_lineshape/lineshape/<excitation>_S.dat (the spectral function), _pl.dat and .png.
# Runner group 'production' (the single points).
#   Called by: main() (step G2)
#   Calls:     Project.batch(), Project.host(), Project.kspacing(), Project.production_param(),
#              atoms_to_dict(), big_cell(), build_defect(), castep_frames(), check_training(),
#              draw_panels(), filter_frames(), format_table(), host_supercell(), job_energy(),
#              lineshape_vector(), load_cell(), mace_python(), named_rng(), pl_spectrum(),
#              production_job(), qtag(), run_training(), size_matrix(), start_spin(),
#              train_command(), trained_model(), write_columns()
def step_lineshape(p):
    if not p.s.lineshape:
        return {"skipped": True, "_note": "lineshape : false (G1's 1D lineshapes only)"}
    import ase.io
    picks = p.out("interest")["for_G"]
    if not picks:
        return {"lines": {}, "_note": "F3 sent no excitation on to G"}
    d = p.phase_dir("G")
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
    res = p.batch(f"{PHASE_DIRS['G']}/jobs/lineshape_data", jobs, "lineshape",
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
            f = p.dir / PHASE_DIRS["E"] / "jobs" / "relaxations" / name / f"{name}.castep"
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
            die(f"G2: no training data for the model of {cid} (did its single points finish?)")
        cmd = train_command(p, [("pbe", bool(split["test"]))])
        (folder / "train.sh").write_text(f"#!/bin/bash\n# The PBE model of {cid} for its "
                                         f"phonons (dwf.py G2)\n{cmd}\n")
        check_training(folder, cmd)
        tasks.append({"cid": cid, "dir": folder, "command": cmd, "heads": ["pbe"]})
        data[cid] = {"structures": len(kept), "host": len(hosts), "test": len(split["test"])}
    if not run_training(p, "G2", tasks):
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
    if not run_training(p, "G2", ph):
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
            series.append({"x": s1["E"][w1], "y": s1["PL"][w1], "label": "1D (G1)",
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
                         + ["dQ", "dQ E2"], rows,
                         "Multi-mode lineshapes (the model's phonons in the big cell, "
                         f"lineshape_method {p.s.lineshape_method}; dQ in amu^0.5 A)")
    if skipped:
        table += "\nNot computed: " + "; ".join(f"{t} ({w})" for t, w in skipped.items()) + "\n"
    if any(x["imaginary"] for x in lines.values()):
        log("WARNING: G2: some big cells have imaginary modes (see the table): the model's "
            "minimum is not a minimum there; they are left out of the lineshape")
    return {"data": data, "big_cell": np.asarray(Pbig).tolist(), "lines": lines,
            "skipped": skipped, "_table": table,
            "_note": f"{len(lines)} multi-mode lineshape(s)"}


# What a G2 phonon job runs (python3 dwf.py --phonon-worker spec.json in its folder, written by
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
    Step("phases", "A3.1", "relax the competing phases", step_phases,
         keys=("phases", "phases_from", "phases_e_above_hull", "dopants", "chemical_potentials",
               "energy"), needs=("relax",), owns=("A_host/jobs/phases",)),
    Step("limits", "A3.2", "chemical-potential limits", step_limits,
         keys=("chemical_potentials", "dopants"), needs=("phases", "relax")),
    # B: candidate defects and the screening cell
    Step("generate", "B1", "candidate defects", step_generate,
         keys=("dopants", "defects", "vacancies", "antisites", "interstitials", "pairs",
               "pair_max_distance", "native_pairs", "dopant_pairs"), needs=("relax",)),
    Step("scan", "B2", "supercell sizes, and the MACE scan of every candidate", step_scan,
         keys=("mace", "scan", "scan_lengths", "scan_tol", "scan_edge_tol", "scan_test",
               "mace_model", "mace_fmax", "mace_steps", "transform"), needs=("generate", "relax"),
         owns=("B_candidates/mace", "B_candidates/jobs/mace_scan", "B_candidates/ladder.json")),
    Step("choose", "B3", "the screening cell", step_choose,
         keys=("supercell", "transform", "screen_min_image", "deep_level_margin", "scan_lengths"),
         needs=("scan", "relax", "bands", "generate")),
    Step("electronic", "B4", "the electronic cell (every band-edge valley at Gamma)",
         step_electronic, keys=("electronic_cell", "electronic_depth", "transform"),
         needs=("choose", "bands", "relax")),
    # C: screening
    Step("prerelax", "C1", "build the candidates and pre-relax them with MACE", step_prerelax,
         keys=("mace", "mace_model", "mace_fmax", "mace_steps", "configuration_margin"),
         needs=("generate", "choose", "relax"),
         owns=("C_screen/structures/candidates", "C_screen/jobs/mace_prerelax")),
    Step("screen", "C2", "cheap CASTEP screening of every candidate", step_screen,
         keys=SCREEN_KEYS + ("screen_charges", "screen_spins"),
         needs=("prerelax", "converge", "bands"),
         owns=("C_screen/jobs/screen", "C_screen/structures/screen"),
         version=1),                               # 1: the band carriers of each run
    Step("filter", "C3", "defect hull and property filters", step_filter,
         keys=("hull_margin", "require_gap_states", "require_spin", "require_bound",
               "distort_charges", "charge_correction", "excite", "gap"),
         needs=("screen", "dielectric", "prerelax")),
    Step("distort", "C4", "ShakeNBreak distortions of the survivors", step_distort,
         keys=("distortions", "distort_around", "rattle_stdev", "rattle_dmin",
               "oxidation_states", "distort_cell"), needs=("filter", "prerelax", "electronic"),
         owns=("C_screen/structures/distortions",),
         version=2),                         # 2: in the electronic cell (distort_cell)
    Step("drelax", "C5", "Gamma-point CASTEP relaxations of the distortions", step_drelax,
         keys=SCREEN_KEYS, needs=("distort",),
         owns=("C_screen/jobs/drelax", "C_screen/structures/drelax"),
         version=1),                         # 1: in C4's cell, against its band edges
    # D: ground and excited states
    Step("dataset", "D1", "training data for the ground-state model (C2/C5 steps, rattled host)",
         step_dataset, keys=SCREEN_KEYS + ("model", "model_rattles", "model_rattle_stdev"),
         needs=("screen", "distort", "drelax", "filter", "choose", "generate", "relax",
                "converge"),
         owns=("D_states/jobs/dataset", "D_states/model/data"),
         version=1),                         # 1: in the cell the survivors work in
    Step("hybrid", "D2", "hybrid single points for the model's 'hse' head", step_hybrid,
         keys=("model", "model_hybrid", "model_hybrid_fraction"),
         files=("hybrid_cell", "hybrid_param"), needs=("dataset", "distort", "choose"),
         owns=("D_states/jobs/hybrid", "D_states/model/hybrid"),
         version=2),                         # 2: in the cell the survivors work in
    Step("train", "D3", "train each defect's model (MACE: pbe/hse heads, charge embedding)",
         step_train, keys=MODEL_KEYS, needs=("dataset", "hybrid"),
         owns=("D_states/model/train",)),
    Step("check", "D4", "check the models (test errors, DFT forces at their minima), retrain",
         step_check, keys=SCREEN_KEYS + MODEL_KEYS + ("model_hybrid", "model_force_tol",
                                                      "model_rounds", "search_fmax",
                                                      "search_steps"),
         files=("hybrid_cell", "hybrid_param"),
         needs=("train", "dataset", "hybrid", "filter", "distort", "drelax", "choose",
                "generate", "relax", "converge"),
         owns=("D_states/jobs/check_r*", "D_states/model/check", "D_states/mace/check"),
         version=1),                         # 1: in the cell the survivors work in
    Step("search", "D5", "the ground-state search with the model, in the electronic cell",
         step_search, keys=("model", "search_distortions", "search_rattles", "search_fmax",
                            "search_steps", "search_keep", "distort_around", "rattle_stdev",
                            "rattle_dmin", "oxidation_states"),
         needs=("check", "filter", "distort", "drelax", "electronic", "choose", "generate",
                "relax"),
         owns=("D_states/mace/search", "D_states/structures/search")),
    Step("rank", "D6", "rank the ground states (the model's, or C5's with sharing)", step_rank,
         keys=SCREEN_KEYS + ("distortion_min_gain", "share_rounds", "keep_structures"),
         needs=("distort", "drelax", "search", "choose"),
         owns=("D_states/jobs/share_*", "D_states/structures/share"),
         version=4),        # 1: each kept structure's run and band carriers; 2: its cell's
                            # matrix; 3: its cell's tag; 4: in C4's cell (distort_cell)
    Step("excite", "D7", "triplet ground states and their excitations (ΔSCF) in the electronic "
         "cell", step_excite,
         keys=SCREEN_KEYS + ("excite", "excite_window", "excite_min", "excite_relax",
                             "excite_overrides", "excite_spin", "spin_tol", "local_fraction",
                             "excite_reach", "excite_min_dw"),
         needs=("rank", "electronic", "choose", "generate", "relax", "converge"),
         owns=("D_states/jobs/excite_relax", "D_states/jobs/excite_gs", "D_states/jobs/excite",
               "D_states/structures/excite_relax", "D_states/structures/excite"),
         version=1),                         # 1: the electronic cell, localised levels, triplets
    Step("validate", "D8", "finite-size check of the screening cell's corrections (optional)",
         step_validate, keys=("validate", "validate_tol", "validate_defects", "validate_sizes",
                              "validate_gamma", "elastic_constants", "production_overrides",
                              "charge_correction", "energy"),
         needs=("rank", "choose", "converge", "relax", "dielectric", "generate", "scan"),
         owns=("D_states/jobs/validate", "D_states/jobs/validate_gamma",
               "D_states/structures/validate")),
    # E: production cells and relaxations
    Step("relaxations", "E1", "production relaxations of every charge state, electronic cell",
         step_relaxations, keys=("production_overrides", "freeze_radius", "energy"),
         needs=("excite", "electronic", "choose", "converge", "relax", "generate"),
         owns=("E_production/jobs/relaxations", "E_production/structures/relaxations")),
    Step("zpl", "E2", "production ΔSCF of the excitations D7 passed (ZPL, dQ)", step_zpl,
         keys=("production_overrides", "excite_overrides", "excite_relax", "excite_min",
               "excite_window", "excite_min_dw", "excite_reach", "local_fraction",
               "band_spread_tol", "energy"),
         needs=("relaxations", "generate", "relax", "converge"),
         owns=("E_production/jobs/zpl_gs", "E_production/jobs/zpl_gs_gamma",
               "E_production/jobs/zpl", "E_production/structures/zpl")),
    # F: thermodynamics
    Step("corrections", "F1", "finite-size corrections", step_corrections,
         keys=("charge_correction",), needs=("relaxations", "dielectric")),
    Step("formation", "F2", "formation energies, levels, Fermi level per dopant set",
         step_formation, keys=("gap", "fermi_temperature", "quench_temperature", "dopants"),
         needs=("corrections", "relaxations", "bands", "limits", "generate"),
         owns=("F_thermo/formation.png",)),
    Step("interest", "F3", "which excitations go on to the lineshape (G)", step_interest,
         keys=("deep_level_margin",), needs=("formation", "zpl", "excite")),
    # G: optical lineshapes
    Step("ccd", "G1", "1D configuration-coordinate diagrams and lineshapes", step_ccd,
         keys=("ccd_points", "production_overrides", "excite_overrides", "lineshape_temperature",
               "lineshape_sigma", "lineshape_gamma", "energy"),
         needs=("interest", "zpl", "relaxations", "generate", "relax", "converge"),
         owns=("G_lineshape/jobs/ccd_gs", "G_lineshape/jobs/ccd_ex", "G_lineshape/ccd")),
    Step("lineshape", "G2", "multi-mode lineshapes (a PBE model's phonons in a big cell)",
         step_lineshape,
         keys=("lineshape", "lineshape_displacements", "lineshape_amplitude", "lineshape_host",
               "lineshape_length", "lineshape_method", "lineshape_delta",
               "lineshape_temperature", "lineshape_sigma", "lineshape_gamma",
               "production_overrides", "energy") + tuple(k for k in MODEL_KEYS if k != "model"),
         needs=("ccd", "interest", "zpl", "relaxations", "generate", "relax", "converge"),
         owns=("G_lineshape/jobs/lineshape_data", "G_lineshape/model", "G_lineshape/phonons",
               "G_lineshape/lineshape")),
]
STEP_INDEX = {s.name: s for s in STEPS}


EXAMPLE_PROJECT = """\
# Defect workflow project file    run:  nohup python3 dwf.py <this file> > run.log 2>&1 &
# Written like a CASTEP .param/.cell file: "keyword : value", %block ... %endblock, keywords
# in any case, # or ! starts a comment. Lines starting with ! are optional settings with an
# example value: delete the ! to use one. File names are relative to this file: the example
# inputs (CaO:Sb on Cirrus) are in example_CaO_Sb/, this file and every file it names.

# ======================================================================= general
host_cell : CaO_prim.cell      # host primitive cell: species_pot, symmetry_generate, k-points
param     : CaO.param          # production settings (functional, cut-off, ...); dwf.py
                               #   sets task, charge and spin, and A1 sets cut_off_energy
!workdir  : dwf_CaO_Sb         # where everything goes (default dwf_<file name>)
# Supercell sizes are lengths (scan_lengths, B2): each is the most compact supercell of the host
# with every periodic image at least that far away (found by doped). To use n x transform x
# (primitive cell) instead, the smallest n for each length, give a transform; this one gives
# cubic cells (8 n^3 atoms) from an FCC primitive cell:
!%block transform
!  -1  1  1
!   1 -1  1
!   1  1 -1
!%endblock transform
mace         : true            # MACE for the size scan (B2) and the pre-relaxations (C1)
!mace_model  : medium-mpa-0    # a mace_mp foundation model, or your own .model file
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

# ======================================================================= runner settings
# How the CASTEP jobs run: any supercell_convergence.py runner keyword (see its --example).
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
max_restarts : 20              # a relaxation that runs out of time carries on this many times
poll      : 300
delete_check : true            # delete each .check once its calculation has finished
# ... and for some steps only, in a runner block: runner_host for all of phase A's CASTEP
# steps (A1 to A3.1), runner_screen for those at the screening settings (C2, C5, D1's pristine
# cells, D6, D7), runner_hybrid for the hybrid single points (D2, D4) and runner_production for
# those at the production settings (D8, E1, E2, G1, G2's single points); for one step:
# runner_converge (A1), runner_relax (A2.1), runner_bands (A2.2), runner_dielectric (A2.3),
# runner_phases (A3.1), runner_drelax (C5), runner_dataset (D1), runner_check (D4), runner_rank
# (D6), runner_excite (D7), runner_validate (D8), runner_relaxations (E1), runner_zpl (E2),
# runner_ccd (G1), runner_lineshape (G2). A step's own block wins over its group's, and that
# wins over the settings above, for the keywords they give (a %block partitions in one replaces
# the whole list).
# Starting points for CaO:Sb on Cirrus - set your own:
%block runner_host             # A1 to A3.1: the primitive cell and the competing phases
max_cores    : 32
max_walltime : 4:00:00
%endblock runner_host
%block runner_converge         # A1: single points of the 2-atom primitive cell
max_cores    : 16
max_walltime : 0:20:00
%endblock runner_converge
%block runner_screen           # C2, C5, D1, D6, D7: Gamma-only screening jobs
max_restarts : 0               # (the default for screening: a job that fails is flagged)
%endblock runner_screen
%block runner_drelax           # C5: relaxations in the electronic cell, phase C's biggest
max_cores    : 288
max_walltime : 24:00:00
%endblock runner_drelax
%block runner_hybrid           # D2, D4: HSE06 single points at Gamma in the electronic cell,
max_cores    : 576             #   by far the dearest jobs of the workflow
max_walltime : 24:00:00
%endblock runner_hybrid
%block runner_production       # D8, E1, E2, G1, G2: the production settings, electronic cell
max_cores    : 288             # up to a node (more: whole nodes, e.g. 576 for two)
%endblock runner_production
# On a machine that charges whole nodes, such as ARCHER2: jobscript : archer2.sh,
# cores_per_node : 128, charging : node (the default) with max_nodes (nodes per job) in place
# of max_cores, %block partitions with standard 2 --qos=standard and highmem 4 --qos=highmem,
# and dryrun_on : login with dryrun_cmd : castep.serial --dryrun {seed} (ARCHER2's serial
# partition rejects whole-node jobs); pack : N then runs N small jobs side by side on a node.

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
# Competing phases for the chemical-potential limits (A3): your own structure files...
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
# ...or the limits themselves (eV per atom, absolute: the same reference as your CASTEP runs)
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
dopant_pairs  : false
# B2: the supercell sizes, as lengths (the shortest distance between periodic images), and
# the MACE scan of every candidate in each. A candidate is converged at the smallest size from
# which its MACE relaxation energy R lies on a line R_inf - c/N (N: the cell's atoms): that size
# and at least 3 bigger ones within scan_tol of it. c/N is the energy of the stress the fixed cell
# holds. The largest length is the most B3 will use (and the line test needs 3 sizes above the
# one it picks); B4 goes beyond it if it must.
scan          : true
scan_lengths  : 8 10 12 14 16 18 20 ang
scan_tol      : 0.02 eV
scan_test     : line           # line | flat: R within scan_tol of every bigger size's, and the
scan_edge_tol : 0.05 ang       #   atoms at the cell's edge moved less than this (flat only)
# B3: the screening cell is the smallest size that holds every candidate (B2) and is at least
# the Gamma-only floor, estimated from the band edges' effective masses (A2.2) for the
# shallowest deep level (deep_level_margin, below).
!screen_min_image : 12 ang     # the floor yourself (0: none), instead of the estimate
!supercell : 14 ang            # fix the screening cell (a length, or n: n x transform)
# B4: the electronic cell, where the survivors go from C4 on (C4-C5, D5, D7, E, F, G): the
# smallest size at least as big as the screening cell that folds every band-edge k-point (each
# valley's symmetry-equivalent copies too) onto Gamma, so a Gamma-point run sees the real gap,
# and is above the Gamma-only floor for a level electronic_depth from a band edge.
electronic_depth : 0.25 eV
!electronic_cell : 20 ang      # fix it: a length, n (n x transform) or screening

# ======================================================================= C: screening
screen_charges   : -1 0 1
screen_spins     : triplet     # the starting spin: triplet (a doublet for an odd electron
                               #   count) | low (0 or 1) | both (low, and low + 2)
!screen_cutoff   : 500 eV      # a lower cut-off for screening (default: production)
screen_max_steps : 50          # geometry steps per screening relaxation
%block screen_overrides        # any other .param keywords for the screening runs
elec_energy_tol : 1e-5
%endblock screen_overrides
!configuration_margin : 1.0 eV # C1: drop arrangements this far above the lowest of their
                               # composition by MACE energy (off unless given)
hull_margin   : 0.2 eV         # C3: keep arrangements within this of the defect hull
require_gap_states : none      # none | occupied | empty | both (filled and empty levels)
!require_spin : 2              # e.g. 2 = triplet ground state
require_bound : false          # bound against its isolated components
level_margin  : 0.1 eV         # gap levels lie further inside the band edges than this;
                               #   anything nearer or past them is in a band, and a run that
                               #   leaves an electron or hole there isn't judged in C3
distortions   : -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0.1 0.2 0.3 0.4 0.5 0.6
distort_charges : screened     # screened (every charge state) | stable (in the gap for that
                               #   arrangement, plus any in a band)
distort_around  : auto         # auto: a single defect's site, a nearest-neighbour complex's
                               #   centre, else each component in turn | centre | each |
                               #   anchor (the first component, e.g. the dopant) | vacancy
distort_cell    : electronic   # where everything after C3 works: electronic (B4's cell;
                               #   C1-C3 screen every candidate in the smaller screening
                               #   cell) | screening (C4-C5 and D1-D4 stay there)
!%block oxidation_states       # for the number of neighbours to distort
!Ca +2
!O  -2
!Sb +3
!%endblock oxidation_states

# ================================================================ D: ground and excited states
# D1-D5 (model : true): a MACE model of each defect C3 keeps (after Wang, Mosquera-Lois and
# Walsh, arXiv:2603.05238): a 'pbe' head trained on its C5 runs (D1, with model_rattles rattled
# pristine cells) and an 'hse' head on hybrid single points of about model_hybrid_fraction of
# them (D2), the total charge embedded so that one model covers every charge state (D3). D4
# checks it: its test errors, and the hybrid's forces at the model's minima (at most
# model_force_tol, else those points join the data and it is trained again, up to model_rounds
# times). D5 searches with it: ShakeNBreak distortions (search_distortions) and rattles of every
# charge state, and each distinct minimum of one charge state tried in the others. All of it in
# the electronic cell, like everything after C3. With model : false, D6 ranks C5's relaxations
# instead.
model           : true
model_hybrid    : true         # false: the 'pbe' head only, no hybrid single points
hybrid_param    : CaO_hse.param    # D2, D4: xc_functional : HSE06 (single points at Gamma)
hybrid_cell     : CaO_ncp.cell     # its species_pot: CASTEP's hybrids need norm-conserving
model_jobscript : cirrus_gpu.sh    # the trainings (and G2's phonons) as GPU jobs; without
model_device    : cuda             #   it, they run where dwf.py runs
!model_hybrid_fraction : 0.1
!model_rattles  : 10
!model_epochs   : 300
!model_channels : 64           # the model's size (with model_r_max 5 ang, model_max_L 1,
                               #   model_batch 4)
!model_train_args : --swa      # anything else for mace_run_train
!model_force_tol : 0.1 eV/ang
!model_rounds   : 2
!search_distortions : -0.7 -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0.1 0.2 0.3 0.4 0.5 0.6 0.7
!search_rattles : 2            # rattled copies of each distortion
!search_keep    : 10           # distinct minima per charge state for D6
# D6: the ground states: D5's minima ranked by the model (model : true), or C5's relaxations,
# a winning distortion tried in the other charge states (model : false).
distortion_min_gain : 0.05 eV  # a distortion counts if it beats the unperturbed by this
share_rounds    : 2            # (model : false) rounds of sharing distortions
keep_structures : 1            # per charge state, on to D7
# D7: in the electronic cell, at the screening settings. Each charge state's ground state is
# relaxed: a triplet (spin within spin_tol of excite_spin) goes on, a band state is left
# 'pending' (PBE can't place it), any other only to F. Its defect levels are the bands localised
# on the defect (participation number at most local_fraction of the atoms), in the gap or up to
# excite_reach inside a band; in each spin channel the highest occupied one to the lowest empty
# one is excited (CASTEP's ΔSCF) and relaxed. It passes if its ZPL is at least excite_min (and
# inside excite_window), the promoted electron stays on the defect and the 1D Debye-Waller
# estimate is at least excite_min_dw. Defects that pass go on to E with all their charge states,
# and so does every native defect (for F). Test CASTEP's ΔSCF on one defect first.
excite         : true          # false: everything D6 kept goes on to E as it is
excite_spin    : 2             # CASTEP's spin: 2 = triplet; any: every spin state
!spin_tol      : 0.3
!local_fraction : 0.2
!excite_reach  : 1.0 eV
excite_min     : 0.4 eV
!excite_window : 0.3 3.5 eV    # pass only if the ZPL is in this window
!excite_min_dw : 0.05          # pass only if the 1D Debye-Waller estimate is at least this
excite_relax   : true          # relax the excited state (the ZPL); false: vertical energy
!%block excite_overrides       # any other .param keywords for D7's runs
!deltascf_smearing : 0.01
!%endblock excite_overrides
# D8 (optional): a test of the screening cell's finite-size error, which C3's hull rests on.
# Each defect named here is relaxed in each charge state D6 kept (its lowest structure; band
# states left out) at the production settings (production_overrides, with the stress on) in the
# screening cell and in the next validate_sizes sizes up. The line E_inf + b/N through the bigger
# sizes is the reference for three estimates from the screening cell: its E as it is, less
# MACE's c/N (B2), and less the energy of its own stress (needs elastic_constants). With
# validate_gamma, Gamma-point single points of the same structures show the Gamma-only error.
# Costly: the ladder's biggest cells, so name two or three defects.
!validate : true
!validate_defects : d001 d002  # B1's ids (default: every defect D6 kept)
!validate_sizes   : 3          # how many sizes above the screening cell
!validate_tol     : 0.05 eV    # how close each estimate must come (levels too)
!validate_gamma   : true
!%block elastic_constants      # the host's, GPa: C11 C12 C44 (cubic) or six lines of six;
!226.2 62.4 80.6               #   CaO at 298 K (Drago & Spain 1977); your functional's own
!%endblock elastic_constants   #   (e.g. PBE's) are better, as the stress comes from it

# ======================================================================= E: production
# E1: every charge state of each defect D7 passes (and of every native defect) relaxed at the
# production settings (A1's cut-off and k-points), in the electronic cell, with the pristine
# cell's single point. E2: the production ΔSCF of D7's excitations. A single point with the
# bands' weights first checks that each constrained band keeps its number at every k-point
# (inside the gap, spread less than band_spread_tol); where it doesn't, the ΔSCF runs at Gamma.
%block production_overrides
geom_energy_tol : 5e-6
elec_energy_tol : 5e-7
geom_max_iter   : 100
%endblock production_overrides
!band_spread_tol : 0.05 eV
freeze_radius : 0              # (untested idea) fix the atoms further than this (Ang) from
                               #   every component of the defect, in E1

# ======================================================================= F: thermodynamics
charge_correction : lz         # lz | mp | none
!gap : 7.09 eV                 # band gap for the Fermi-level range of C3, F2 and F3 (default:
                               #   the screening cell's in C3, A2.2's in F)
!fermi_temperature  : 1000 K   # self-consistent Fermi level at this temperature
!quench_temperature : 300 K    # ... with the defects frozen in, then at this one
deep_level_margin : 0.5 eV     # F3: a level this far from both band edges is "deep" (B3's
                               #   Gamma-only floor is for a level this deep)
# F2 finds the Fermi level of each set of dopants on its own (the native defects alone, with
# each dopant, and with each mix that one defect brings), and F3 sends on to G each excitation
# that passed E2 whose charge state is the lowest for some Fermi level in the gap.

# ======================================================================= G: lineshapes
# G1: the 1D configuration-coordinate diagram of each excitation F3 sends on: ground-state and
# ΔSCF single points along the line from the relaxed ground state (0) to the relaxed excited
# state (1), at E2's settings; parabolas give each state's phonon energy and Huang-Rhys factor,
# and the 1D luminescence lineshape.
!ccd_points : -0.4 -0.2 0 0.2 0.4 0.6 0.8 1 1.2 1.4
lineshape_temperature : 10 K   # the temperature of both lineshapes
!lineshape_sigma : 0.006 eV    # Gaussian width of each phonon line
!lineshape_gamma : 0.002 eV    # Lorentzian half-width of the ZPL
# G2 (optional): the multi-mode lineshape. A PBE model of each defect (one head, the charge
# embedded), trained on lineshape_displacements displaced copies of its ground state (normal,
# lineshape_amplitude a component), its E1 relaxations and lineshape_host rattled pristine
# cells, all at the production settings; its phonons in a cell lineshape_length across (finite
# differences, lineshape_delta), projected onto the excited state's forces at the ground
# state's structure (lineshape_method forces) or onto the displacement between the relaxed
# states (displacements). The trainings and phonons run with model_jobscript.
!lineshape : true
!lineshape_displacements : 30
!lineshape_amplitude : 0.02 ang
!lineshape_host   : 5
!lineshape_length : 30 ang     # CaO: about 1900 atoms, 11,500 model force evaluations
!lineshape_method : forces     # forces | displacements
"""


if __name__ == "__main__":
    main()
