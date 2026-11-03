#!/usr/bin/env python3
"""
dwf.py - the defect workflow (phases A to E of the flowchart) in one script

    python3 dwf.py --example > CaO_Sb.dwf      # an annotated project file to start from
    nohup python3 dwf.py CaO_Sb.dwf > run.log 2>&1 &
    python3 dwf.py CaO_Sb.dwf --status         # which steps have finished

One command runs every step in order, A1 to E3, waiting for its CASTEP jobs. Run it again
after any stop (Ctrl-C, touch <workdir>/STOP, --no-wait, a crash) and it carries on where it
stopped. Give step names, labels or phase letters to run only those (python3 dwf.py CaO_Sb C),
or --until STEP to stop after a step or phase.

  A  host            A1 converge, A2.1 relax, A2.2 bands, A2.3 dielectric, A3.1 phases,
                     A3.2 limits
  B  candidates      B1 generate, B2 scan (the supercell sizes, and MACE on every candidate),
                     B3 choose (the screening cell)
  C  screening       C1 prerelax (MACE), C2 screen, C2b dispersion (the Gamma-only check),
                     C3 filter, C4 distort (ShakeNBreak), C5 drelax, C6 rank, C7 excite (ΔSCF)
  D  production      D0 cells (each defect's production cell), D1 production, D2 finetune
  E  thermodynamics  E1 corrections, E2 formation, E3 interest

Files. Everything goes in the work directory, dwf_<name>/ next to the project file:
  state.json         which steps have finished, and with what inputs (the checkpoints)
  A_host/ ... E_thermo/  one folder per phase, each with
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
again from scratch).

All CASTEP jobs go through supercell_convergence.py's batch mode (it must be in the same
directory), which runs the structures this script builds. Its runner settings (jobscript,
partitions, cores_per_node, max_nodes, max_walltime, pack, ...) go in the project file, for
every step; %block runner_host, runner_screen and runner_production change them for all of
phase A's, C's or D's CASTEP steps, and %block runner_<step> for one step (e.g.
runner_converge, runner_excite). The MACE relaxations (B2, C1) run where this script runs, or,
with mace_jobscript, as jobs on compute nodes (each job runs this script as a worker:
python3 dwf.py --mace-worker <tasks>.json).
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
# read_trajectory), the charge correction (correction), MACE (mace_relaxations, which runs
# mace_relax here or mace_jobs on compute nodes, whose workers are mace_worker) and tables
# (format_table).
# Project.out() and Project.phase_dir() are used by almost every step, so they are left out of
# the "Calls" lists, as are log() and die().
# ==========================================================================================

HARTREE = 27.211386245988                     # eV
STATE, RESULTS, SUMMARY = "state.json", "results.json", "summary.txt"
PHASE_DIRS = {"A": "A_host", "B": "B_candidates", "C": "C_screen", "D": "D_production",
              "E": "E_thermo"}
PHASE_TITLES = {"A": "host", "B": "candidate defects and the screening cell",
                "C": "screening and ground states", "D": "production cells and relaxations",
                "E": "thermodynamics"}
HB2M = 3.80998                                # hbar^2 / (2 m_e), eV Ang^2
# B2: the atoms at a supercell's edge, whose MACE displacements show whether a defect's strain
# field fits in it, are those further than this fraction of the shortest image distance from it
EDGE_FRACTION = 0.35
# B3: the Gamma-only floor is this many decay lengths of the shallowest deep level
FLOOR_DECAYS = 4.0
# C2: a Gamma-point gap of the screening cell this much above A2.2's means band edges that don't
# fold onto Gamma (eV)
FOLD_GAP_TOL = 0.05
# D0: two relaxations of a defect in different cells are in the same basin if no atom near the
# defect is displaced from its site by more than this differently (Ang)
BASIN_TOL = 0.2
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
    "supercell": (_size, None), "screen_min_image": (sc._num("ang", "A"), None),
    # C: screening
    "screen_charges": (sc._ints(), [-1, 0, 1]),
    "screen_spins": (sc._choice("low", "both"), "both"),
    "screen_cutoff": (sc._num("eV"), None), "screen_max_steps": (sc._num(cast=int), 50),
    "screen_overrides": ("kv", {}),
    "dispersion_check": (sc._choice("warn", "stop", "off"), "warn"),
    "dispersion_defect": (_t, None), "dispersion_tol": (sc._num("eV"), 0.05),
    "configuration_margin": (sc._num("eV"), None),
    "hull_margin": (sc._num("eV"), 0.2), "level_margin": (sc._num("eV"), 0.1),
    "require_bound": (_b, False), "require_spin": (sc._num(), None),
    "require_gap_states": (sc._choice("none", "occupied", "empty", "both"), "none"),
    "distortions": (_floats(), [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5,
                                0.6]),
    "distort_charges": (sc._choice("stable", "screened"), "stable"),
    "distort_around": (sc._choice("anchor", "vacancy"), "anchor"),
    "rattle_stdev": (sc._num("ang", "A"), None), "rattle_dmin": (sc._num("ang", "A"), None),
    "oxidation_states": ("lines", None),
    "distortion_min_gain": (sc._num("eV"), 0.05), "share_rounds": (sc._num(cast=int), 2),
    "keep_structures": (sc._num(cast=int), 1),
    "excite": (_b, False), "excite_window": (_floats(2), None),
    "excite_min": (sc._num("eV"), 0.4), "excite_relax": (_b, True),
    "excite_overrides": ("kv", {}),
    # D: production
    "series_tol": (sc._num("eV"), 0.05),
    "production_overrides": ("kv", {}), "freeze_radius": (sc._num("ang", "A"), 0.0),
    "finetune": (_b, False), "finetune_args": (_t, ""),
    # E: thermodynamics
    "charge_correction": (sc._choice("lz", "mp", "none"), "lz"), "gap": (sc._num("eV"), None),
    "fermi_temperature": (sc._num("K"), None), "quench_temperature": (sc._num("K"), None),
    "deep_level_margin": (sc._num("eV"), 0.5),
}
LIST_KEYS = {"transform", "converge_cutoffs", "dopants", "scan_lengths", "screen_charges",
             "distortions", "excite_window"}                                  # block or keyword
# supercell_convergence keywords that control how jobs run (allowed in the project file, for every
# step, and in %block runner_<group> or runner_<step>)
RUNNER_KEYS = {"jobscript", "dryrun_jobscript", "partitions", "max_cores", "cores",
               "cores_per_node", "max_nodes", "mem_per_core", "mem_safety", "mem_replicated",
               "baseline_mb", "omp_threads", "walltime", "walltime_safety", "sp_walltime",
               "max_walltime", "max_restarts", "delete_check", "dryrun_on", "dryrun_partition",
               "dryrun_time", "dryrun_cores", "dryrun_mem_gb", "dryrun_options", "dryrun_cmd",
               "dryrun_timeout", "submit_cmd", "status_cmd", "poll", "max_jobs",
               "tolerate_failures", "pack", "dryrun_pack"}
# The steps that run CASTEP jobs, and the group of steps each belongs to. The runner settings of a
# step: the project's, changed by its group's defaults (GROUP_DEFAULTS), then by %block
# runner_<group>, then by %block runner_<step> (Project.runner_items). screen and production are
# both a step and a group: runner_screen is for all five screening steps (C2, C2b, C5, C6, C7),
# and runner_dispersion, runner_drelax, runner_rank and runner_excite change it for one of them;
# runner_production is for D0 and D1, and runner_cells for D0 only.
RUNNER_STEPS = {"converge": "host", "relax": "host", "bands": "host", "dielectric": "host",
                "phases": "host", "screen": "screen", "dispersion": "screen", "drelax": "screen",
                "rank": "screen", "excite": "screen", "cells": "production",
                "production": "production"}
RUNNER_GROUPS = ("host", "screen", "production")
RUNNER_BLOCKS = RUNNER_GROUPS + tuple(s for s in RUNNER_STEPS if s not in RUNNER_GROUPS)
# what each group of steps uses unless the project file's runner blocks say otherwise
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


# The keyword, or the block's name, of an item of a CASTEP-style file (sc.parse_castep_input).
#   Called by: Project.runner(), Project.runner_items(), read_project()
def item_key(x):
    return x["name"] if x["type"] == "block" else x["key"]


# Keywords that have been renamed or removed, with what to do instead.
#   Used by: read_project()
_NO_SERIES = ("has been removed with the CASTEP size series of one defect (the old B3): B2 now "
              "scans every candidate with MACE, and D0 converges each defect's own production "
              "cell")
RETIRED = {"triage": "has been renamed: use configuration_margin (in eV; giving it switches the "
                     "filter on)",
           "triage_window": "has been renamed: use configuration_margin (in eV; giving it switches "
                            "the filter on)",
           "zpl_window": "has been removed, with step C3b and its Kohn-Sham ZPL estimate: C7 "
                         "filters on the ΔSCF ZPL instead (excite : true, excite_window)",
           "series_defect": _NO_SERIES,
           "series_charge": _NO_SERIES + " (in every charge state the defect reaches D in)",
           "series_sizes": _NO_SERIES + "; the sizes are now lengths: scan_lengths (Ang), which "
                           "D0 steps through too",
           "series_choice": _NO_SERIES + ", taking the smaller of the first two sizes that agree "
                            "within series_tol",
           "runner_series": _NO_SERIES + ": D0's runs use runner_production (or runner_cells)",
           "scan_sizes": "has been replaced by scan_lengths: the sizes are lengths in Ang, each "
                         "the most compact supercell with every periodic image at least that "
                         "far away",
           "finetuned_model": "has been removed: B2 and C1 use mace_model (a foundation model or "
                              "your own .model file), and the fine-tuning data are now written "
                              "after D1 (D2, for phase F)"}


# Reads the project file into a namespace of settings.
# Every keyword must be a project keyword (PROJECT_KEYWORDS) or a runner keyword (RUNNER_KEYS,
# passed to supercell_convergence.batch_args); %block runner_<group> or runner_<step>
# (RUNNER_BLOCKS) holds runner keywords, and may hold a %block partitions, for a group of steps or
# one step. A typo gets the closest keyword suggested, and a retired one (RETIRED) what to do.
# Returns (settings, the text each setting was given as, the runner items, {group or step: runner
# items}).
#   Called by: Project.__init__()
#   Calls:     item_key(), kgrid_list(), sc.block_rows(), sc.parse_castep_input()
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
                "CASTEP steps), runner_screen (all of phase C's), runner_production (D's) and, "
                "for one step, " + ", ".join(f"runner_{s}" for s in RUNNER_STEPS
                                             if s not in RUNNER_GROUPS))
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
    for key in ("pair_max_distance", "scan_tol", "scan_edge_tol", "series_tol",
                "configuration_margin", "converge_tol", "mace_fmax", "excite_min",
                "dispersion_tol"):
        if getattr(s, key) is not None and getattr(s, key) <= 0:
            die(f"{name}: {key} must be greater than 0")
    for key in ("mace_threads", "mace_steps", "mace_pack", "mace_max_jobs"):
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
    return s, raw, runner, blocks


# A project: its settings, files, directories, checkpoints (state.json), results (results.json
# of each phase) and job runner. main() makes one from the project file and the command-line
# switches, and passes it to the steps.
class Project:
    # Reads the project file and the state of earlier runs, and checks the runner settings of
    # every step now (rather than when the step starts, perhaps days later), and the MACE jobs'
    # template.
    #   Called by: main()
    #   Calls:     Project.mace_template(), Project.runner(), read_project()
    def __init__(self, path, cl):
        self.path, self.base, self.cl = path, path.parent, cl
        self.s, self.raw, self.runner_project, self.runner_blocks = read_project(path)
        self._runners, self._mace, self._model_ids, self._results = {}, {}, {}, {}
        self._mace_args = None
        for step in RUNNER_STEPS:
            self.runner(step)
        if self.s.mace and self.s.mace_jobscript:
            self.mace_template()
        self.name = self.s.name or path.stem
        self.dir = self.base / (self.s.workdir or f"dwf_{self.name}")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.dir / STATE
        self.state = (json.loads(self.state_file.read_text()) if self.state_file.exists()
                      else {"steps": {}})

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
    #   Called by: main(), reset_step()
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
    #   Called by: run_screen_jobs(), step_bands(), step_cells(), step_converge(),
    #              step_dielectric(), step_dispersion(), step_excite(), step_phases(),
    #              step_production(), step_relax(), step_screen()
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
    #   Called by: run_screen_jobs(), step_bands(), step_cells(), step_dielectric(),
    #              step_dispersion(), step_excite(), step_phases(), step_production(), step_relax(),
    #              step_screen()
    #   Calls:     Project.file(), Project.finished()
    def production_param(self):
        if self.finished("converge"):
            return (self.dir / self.out("converge")["production_param"]).read_text()
        return self.file("param").read_text()

    # The production k-point spacing (1/Ang): A1's, else kspacing, else from the host .cell's grid.
    #   Called by: step_bands(), step_cells(), step_converge(), step_dielectric(), step_phases(),
    #              step_production(), step_relax()
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
    #   Called by: doped_phases(), screen_setup(), step_bands(), step_cells(), step_choose(),
    #              step_corrections(), step_dielectric(), step_excite(), step_formation(),
    #              step_generate(), step_limits(), step_production(), step_scan()
    #   Calls:     load_cell()
    def host(self):
        return load_cell(self, self.out("relax")["cell"])

    # A MACE model as mace-torch takes it: the path of a model file (mace_model, or the model
    # given, relative to the project file), or else a foundation model's name.
    #   Called by: Project.mace(), Project.mace_model_file(), Project.model_id(), step_finetune()
    def model_path(self, model=None):
        model = str(model or self.s.mace_model)
        for path in (self.base / model, Path(model)):
            if path.is_file():
                return str(path.resolve())
        return model

    # The MACE calculator for a model (default mace_model: a foundation model's name, or a model
    # file), loaded once (mace-torch, on mace_device), for the relaxations run here.
    #   Called by: mace_relax()
    #   Calls:     Project.model_path()
    def mace(self, model=None):
        model = self.model_path(model)
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
    # foundation model of that name, which mace-torch downloads here (ARCHER2's compute nodes have
    # no network) into its cache in your home directory, copied into <workdir>/mace_models/
    # (compute nodes may not see your home directory, as on ARCHER2).
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

    # The settings the MACE jobs share with the CASTEP jobs: submit_cmd, status_cmd and poll, from
    # the project's runner keywords (supercell_convergence.batch_args).
    #   Called by: mace_jobs()
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
    # Records the step and its phase.
    def __init__(self, name, label, title, func, keys=(), files=(), needs=(), owns=()):
        self.name, self.label, self.title, self.func = name, label, title, func
        self.keys, self.files, self.needs, self.owns = keys, files, needs, owns
        self.phase = label[0]


# The fingerprint of what a step's results depend on: its settings (as written in the project
# file), the contents of its files (and, for a step that uses %block phases, of the structure files
# listed there) and the result fingerprints of the steps it needs. Stops if one of those steps
# hasn't finished.
#   Called by: main()
#   Calls:     Project.file()
def inputs_hash(p, step):
    data = {"keys": {k: p.raw.get(k) for k in step.keys}, "files": {}, "needs": {}}
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
        elif isinstance(v, str) and len(v) < 300 and (p.dir / v).is_file():
            files[v] = hashlib.md5((p.dir / v).read_bytes()).hexdigest()

    walk(outputs)
    return hashlib.md5(json.dumps([outputs, files], sort_keys=True,
                                  default=str).encode()).hexdigest()


# Writes a phase's summary.txt: for each of its steps that has run, a heading, when it finished
# and its one-line note, then its table (main() keeps the notes and tables in state.json).
#   Called by: main(), reset_step()
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


# The command line, and the work it does: runs the steps in order (all of them, or those named by
# name, label or phase letter, up to --until), skipping those that have finished with the same
# inputs, and stops at the first that isn't finished (jobs still queued with --no-wait). A step
# that finishes has its results put in its phase's results.json, its checkpoint (with its note
# and table) in state.json, and its phase's summary.txt written again. --status prints the
# checkpoints; --redo starts steps again from scratch; --example prints an annotated project file;
# --mace-worker is what a MACE job runs (mace_worker), not a project.
#   Called by: the main block
#   Calls:     Project.__init__(), Project.finished(), Project.save_results(),
#              Project.save_state(), inputs_hash(), mace_worker(), outputs_hash(),
#              print_status(), reset_step(), write_summary()
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project", nargs="?", help="the project file (<name>.dwf) or its name")
    ap.add_argument("steps", nargs="*", help="run only these steps: names (relax), labels (A2.1) "
                    "or phases (A to E); default: all of them")
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
                    help="start failed CASTEP jobs again")
    ap.add_argument("--force", action="store_true",
                    help="keep the results of jobs whose inputs have changed")
    ap.add_argument("--mace-worker", metavar="TASKS",
                    help="(what a MACE job runs) relax the structures in this task file")
    cl = ap.parse_args()
    if cl.mace_worker:
        mace_worker(cl.mace_worker)
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
    if cl.status:
        print_status(p)
        return

    # The steps a word stands for: a step's name or label, or a phase letter (all its steps).
    def find(word):
        w = word.lower()
        hits = [s for s in STEPS if w in (s.name, s.label.lower(), s.phase.lower())]
        if not hits:
            die(f"no step or phase '{word}': the steps are "
                + ", ".join(f"{s.label} {s.name}" for s in STEPS) + ", the phases A to E")
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
#   Called by: mace_jobs(), mace_key(), save_mace_result()
def atoms_to_dict(atoms):
    return {"symbols": atoms.get_chemical_symbols(),
            "cell": np.asarray(atoms.cell).round(10).tolist(),
            "frac": atoms.get_scaled_positions(wrap=False).round(10).tolist(),
            "magmoms": atoms.get_initial_magnetic_moments().round(6).tolist()}


# The reverse of atoms_to_dict.
#   Called by: mace_cached(), mace_worker()
def atoms_from_dict(d):
    a = Atoms(d["symbols"], scaled_positions=d["frac"], cell=d["cell"], pbc=True)
    a.set_initial_magnetic_moments(d.get("magmoms") or [0.0] * len(a))
    return a


# Saves a structure as a CASTEP .cell file (a comment line, lattice_cart, positions_frac with
# SPIN= where a starting spin is set), the atoms in their own order so that reading it back
# (load_cell) gives the same atom numbers. Only the structure: each job adds its k-points,
# species_pot and the rest. Returns the path relative to the work directory, as the results give
# it.
#   Called by: job_summary(), production_summary(), step_distort(), step_excite(),
#              step_prerelax()
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
#   Called by: Project.host(), distinct_states(), duplicates(), run_screen_jobs(), step_cells(),
#              step_dispersion(), step_distort(), step_excite(), step_production(), step_rank(),
#              step_screen()
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


# The starting spins (unpaired electrons, hbar/2) a screening job tries for a structure with charge
# q: the lowest the electron count allows (0 or 1) and, with both, that +2.
#   Called by: step_screen()
#   Calls:     electron_parity()
def spin_states(atoms, q, which):
    low = electron_parity(atoms, q)
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
#   Called by: carry_over(), displacement_field(), step_cells(), step_prerelax(),
#              step_production(), step_scan()
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
#   Called by: step_cells(), step_dispersion(), step_distort(), step_filter(), step_screen()
#   Calls:     sc.build_supercell()
def host_supercell(prim, P):
    atoms = sc.build_supercell(prim, P)
    del atoms.arrays["prim_index"]
    return atoms


# The shortest distance between a point and its periodic images (the shortest lattice vector),
# from the Minkowski-reduced cell, so that it is right for skewed cells too.
#   Called by: carry_over(), cell_record(), same_basin(), size_matrix()
#   Calls:     sc.min_image_distance()
def min_image(cell):
    return sc.min_image_distance(Cell(np.asarray(cell, float)).minkowski_reduce()[0])


# Whether a candidate can be built in a supercell whose images are L apart: a pair's components
# must be less than half of that apart (with 0.05 Ang to spare), or the nearest copy of one of them
# would be a periodic image of the other.
#   Called by: step_cells(), step_dispersion(), step_prerelax(), step_scan()
def fits(cand, L):
    return not (cand["separation"] and cand["separation"] > L / 2 - 0.05)


# The vectors from each of the points a to each of the points b (N x M x 3, Ang), each the nearest
# periodic image in fractional coordinates: right for the short distances they are used for in a
# Minkowski-reduced cell, and lighter than ASE's general search for big cells.
#   Called by: displacement_field()
def frac_mic(a, b, cell):
    cell = np.asarray(cell, float)
    df = (np.asarray(b)[None, :, :] - np.asarray(a)[:, None, :]) @ np.linalg.inv(cell)
    return (df - np.round(df)) @ cell


# The displacement field of a defect's relaxed structure: the defect built in the same supercell
# (build_defect: the ideal structure, every atom on its perfect-crystal site), each of its atoms
# matched to one of the relaxed structure's of the same element by the assignment that moves them
# least in total (scipy's linear_sum_assignment on the distances through the periodic images), so
# a matched pair is the same atom even when it has moved a long way. Returns (the ideal Atoms, its
# info, each ideal atom's displacement (N x 3, Ang), each ideal atom's position relative to the
# defect's centre (N x 3, the shortest way through the images)).
#   Called by: carry_over(), same_basin()
#   Calls:     build_defect(), frac_mic()
def displacement_field(prim, comps, P, relaxed, label="defect"):
    from ase.geometry import find_mic
    from scipy.optimize import linear_sum_assignment
    ideal, info = build_defect(prim, P, comps, label)
    syms, rsyms = np.array(ideal.get_chemical_symbols()), np.array(relaxed.get_chemical_symbols())
    if sorted(syms) != sorted(rsyms) or not np.allclose(ideal.cell, relaxed.cell, atol=1e-4):
        die(f"{label}: the relaxed structure is not of the defect built in this cell")
    u = np.zeros((len(ideal), 3))
    for el in sorted(set(syms)):
        i, r = np.flatnonzero(syms == el), np.flatnonzero(rsyms == el)
        D = frac_mic(ideal.positions[i], relaxed.positions[r], ideal.cell)
        rows, cols = linear_sum_assignment(np.linalg.norm(D, axis=2))
        u[i[rows]] = D[rows, cols]
    v, _ = find_mic(ideal.positions - np.asarray(info["centre"]), ideal.cell, pbc=True)
    return ideal, info, u, v


# The atoms of two cells that are the same atom of a defect: of the same element, at the same
# position relative to the defect's centre (within 0.1 Ang), both within radius of it. Takes the
# positions relative to the centre (v1, v2) and the elements of each. Returns (indices in 1,
# indices in 2).
#   Called by: carry_over(), same_basin()
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
#   Called by: step_cells(), step_dispersion(), step_production()
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
#   Called by: step_cells()
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
#   Called by: job_summary(), step_bands(), step_choose(), step_dispersion(), step_excite(),
#              step_formation(), step_screen()
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


# The number of occupied bands in each spin channel of a .bands (read_bands). From the electron
# counts when CASTEP gives one per spin (or the total, not spin-polarised); otherwise the bands
# below the Fermi energy at the first k-point.
#   Called by: band_edges(), band_masses(), carriers(), excitation(), gap_levels(),
#              level_dispersion()
def occupied_bands(b):
    if b["nspin"] == 1:
        return [int(round(b["nelec"][0] / 2))]
    if len(b["nelec"]) == 2:
        return [int(round(x)) for x in b["nelec"]]
    ef = b["efermi"][0]
    return [int(np.sum(b["eig"][s, 0] < ef)) for s in range(2)]


# The valence band maximum and conduction band minimum (eV) of an insulator's .bands: the highest
# occupied and lowest empty eigenvalue over all k-points and spins.
#   Called by: step_bands(), step_screen()
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
# and the first Forces table after it. Atoms are in CASTEP's order (grouped by species). Returns
# [(Atoms, energy, forces)], leaving out structures without forces.
#   Called by: step_finetune()
#   Calls:     sc.last_lattice()
def read_trajectory(text):
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
        out.append((atoms, float(m.group(1)), np.array([[float(x) for x in f[2:]]
                                                        for f in forces])))
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


# The finite-size correction (eV) for charge q in a supercell (supercell_convergence's
# charge_correction: Lany-Zunger, Makov-Payne or none), with the static dielectric constant eps.
#   Called by: relative_formation(), step_cells(), step_corrections()
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
# and the optimiser's settings (mace_fmax, mace_steps).
#   Called by: mace_relax(), mace_relaxations()
#   Calls:     Project.model_id(), atoms_to_dict()
def mace_key(p, atoms, model=None):
    return hashlib.md5(json.dumps({"atoms": atoms_to_dict(atoms), "model": p.model_id(model),
                                   "fmax": p.s.mace_fmax, "steps": p.s.mace_steps},
                                  sort_keys=True).encode()).hexdigest()


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
# fmax, in at most steps steps). Returns (relaxed Atoms, energy before, energy after, steps); the
# starting spins are kept (MACE ignores them). The same here (mace_relax) and in the MACE jobs
# (mace_worker).
#   Called by: mace_relax(), mace_worker()
def mace_run(calc, atoms, fmax, steps):
    from ase.optimize import LBFGS
    work = atoms.copy()
    work.calc = calc
    e0 = float(work.get_potential_energy())
    opt = LBFGS(work, logfile=None)
    opt.run(fmax=fmax, steps=steps)
    e1 = float(work.get_potential_energy())
    work.calc = None
    work.set_initial_magnetic_moments(atoms.get_initial_magnetic_moments())
    return work, e0, e1, int(getattr(opt, "nsteps", 0))


# Relaxes a structure with MACE here, in this process (model: default mace_model), unless its
# cache file already holds the result (mace_cached). Returns (relaxed Atoms, energy before, energy
# after, steps).
#   Called by: mace_relaxations()
#   Calls:     Project.mace(), mace_cached(), mace_key(), mace_run(), save_mace_result()
def mace_relax(p, atoms, cache, model=None):
    key = mace_key(p, atoms, model)
    hit = mace_cached(cache, key)
    if hit:
        return hit
    result = mace_run(p.mace(model), atoms, p.s.mace_fmax, p.s.mace_steps)
    save_mace_result(cache, key, result)
    return result


# A step's MACE relaxations (mace_model): {cache file: (relaxed Atoms, energy before, energy
# after, steps)}, or None while MACE jobs are still queued (--no-wait) or after --prepare-only.
# tasks: [{'atoms', 'cache', 'label'}]. A relaxation whose cache file already holds it (the same
# structure, model and settings: mace_key) is not run again. The others run as jobs on compute
# nodes with mace_jobscript (mace_jobs), else here, one after the other (mace_relax).
#   Called by: step_prerelax(), step_scan()
#   Calls:     mace_cached(), mace_jobs(), mace_key(), mace_relax()
def mace_relaxations(p, step, tasks):
    out, todo = {}, []
    for t in tasks:
        t = dict(t, key=mace_key(p, t["atoms"]))
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
        out[str(t["cache"])] = mace_relax(p, t["atoms"], t["cache"])
    return out


# The MACE relaxations a step still needs (mace_relaxations), run as batch jobs on compute nodes,
# in <phase>/jobs/mace_<step>/job_NNN/. Each job is the mace_jobscript template with its {COMMAND}
# line made into mace_pack workers, each running  python3 dwf.py --mace-worker tasks_KK.json  in
# the background with mace_threads threads (whatever is in front of {COMMAND} on its line, e.g.
# srun ..., goes in front of each), then wait. The relaxations are shared out between at most
# mace_max_jobs jobs, biggest first, onto whichever worker has the fewest atoms so far, so the
# workers finish together. The jobs go in jobs.json there; waiting polls the queue (status_cmd)
# every poll seconds, like the CASTEP jobs, and the project's STOP file stops it. A relaxation still
# missing once its job has ended (e.g. it ran out of time) goes into a new job, at most
# MACE_ATTEMPTS times. Returns True when every relaxation is done, False while jobs are queued
# (--no-wait) or after --prepare-only (the job scripts are then in prepared/).
#   Called by: mace_relaxations()
#   Calls:     Project.mace_model_file(), Project.mace_runner(), Project.mace_template(),
#              atoms_to_dict(), mace_cached(), write_json(), sc.job_active(), sc.send_job()
def mace_jobs(p, step, todo):
    args = p.mace_runner()
    d = p.phase_dir(STEP_INDEX[step].phase) / "jobs" / f"mace_{step}"
    d.mkdir(parents=True, exist_ok=True)
    state_file = d / "jobs.json"
    state = (json.loads(state_file.read_text()) if state_file.exists()
             else {"jobs": [], "tries": {}})
    template, stop = p.mace_template(), p.dir / "STOP"
    pack = p.s.mace_pack or max(1, (args.cores_per_node or p.s.mace_threads) // p.s.mace_threads)
    label, model = STEP_INDEX[step].label, p.mace_model_file()
    announced = False

    # Whether a job is over: not in the queue, and either its end mark is there or it was
    # submitted long enough ago that the queue must know it (so a slow queue isn't taken for a
    # job that has ended).
    def ended(job):
        if sc.job_active(job["jobid"], args):
            return False
        return (Path(job["dir"]) / "finished").exists() or time.time() - job["time"] > 120

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
                    "tasks": [{"cache": str(Path(t["cache"]).resolve()), "key": t["key"],
                               "label": t["label"], "atoms": atoms_to_dict(t["atoms"])}
                              for t in s["tasks"]]}))
                files.append(f)
                caches += [str(t["cache"]) for t in s["tasks"]]
            lines = template.splitlines()
            i = next(k for k, line in enumerate(lines)
                     if "{COMMAND}" in line and not line.lstrip().startswith("#"))
            pre, post = lines[i].split("{COMMAND}", 1)
            py, me = p.s.mace_python or sys.executable, str(Path(__file__).resolve())
            run = [f"{pre}env OMP_NUM_THREADS={p.s.mace_threads} {shlex.quote(py)} "
                   f"{shlex.quote(me)} --mace-worker {shlex.quote(str(f.resolve()))}{post} > "
                   f"{shlex.quote(str((jdir / f'worker_{k + 1:02d}.log').resolve()))} 2>&1 &"
                   for k, f in enumerate(files)]
            run += ["wait", f"echo finished > {shlex.quote(str((jdir / 'finished').resolve()))}"]
            text = "\n".join(lines[:i] + run + lines[i + 1:]) + "\n"
            name = f"dwf_mace_{step}_{jdir.name}"
            (jdir / "job.sh").write_text(text.replace("{JOBNAME}", name))
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
            worn = [t["label"] for t in new if state["tries"].get(str(t["cache"]), 0)
                    >= MACE_ATTEMPTS]
            if worn:
                die(f"{label}: the MACE jobs ended {MACE_ATTEMPTS} times without these "
                    f"relaxations: {', '.join(worn[:8])}"
                    + (f" and {len(worn) - 8} more" if len(worn) > 8 else "")
                    + f". See the workers' logs and slurm output in {d}/job_*/ (e.g. a longer "
                    "--time in mace_jobscript)")
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
# there) that its cache file doesn't hold yet, and writes each result to its cache file as soon as
# it has it (save_mace_result); an error goes into the cache file instead, for the workflow to
# report.
#   Called by: main()
#   Calls:     atoms_from_dict(), mace_run(), save_mace_result()
def mace_worker(path):
    spec = json.loads(Path(path).read_text())
    calc = None
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
            if calc is None:
                import warnings
                warnings.filterwarnings("ignore")
                import torch
                from mace.calculators import mace_mp
                torch.set_num_threads(spec["threads"])
                calc = mace_mp(model=spec["model"], device=spec["device"],
                               default_dtype="float64")
            res = mace_run(calc, atoms_from_dict(t["atoms"]), spec["fmax"], spec["steps"])
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
#   Called by: Project.save_results(), Project.save_state(), mace_jobs(), size_ladder()
def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=float))
    tmp.replace(path)


# A text table for summary.txt and the log: a title, a header row, a rule and the rows, in columns
# as wide as needed (numbers to the right, text to the left). Floats are shown with 4 decimals,
# None as blank.
#   Called by: ladder_table(), problem_table(), step_cells(), step_choose(), step_converge(),
#              step_corrections(), step_dispersion(), step_excite(), step_filter(),
#              step_formation(), step_generate(), step_interest(), step_limits(), step_phases(),
#              step_prerelax(), step_production(), step_rank(), step_scan()
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
#   Called by: step_cells(), step_dispersion(), step_distort(), step_excite(),
#              step_production(), step_rank(), step_screen()
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
# which the Fermi-level analysis (E2) also uses.
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
                "limits (E2 then gives energies without the mu terms)"}
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
#   Called by: size_ladder(), step_choose()
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
#   Called by: size_ladder(), step_choose()
#   Calls:     min_image()
def cell_record(prim, P, length=None):
    P = np.asarray(P, int)
    return {"matrix": P.tolist(), "natoms": int(round(abs(np.linalg.det(P)))) * len(prim),
            "L": round(min_image(P @ np.asarray(prim.cell, float)), 4), "length": length}


# Puts a size into a ladder (a list of cell_record dicts, smallest image distance first) unless the
# same supercell is there already (the same lattice: one matrix is an integer recombination of the
# other), giving it a tag for job names, n<atoms> (with b, c, ... for another cell with as many
# atoms). Returns the ladder's record of that supercell.
#   Called by: size_ladder(), step_choose()
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
# smallest first. B2 scans it, B3 picks the screening cell from it and D0 steps through it. doped
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


# B2: the size ladder (size_ladder: the supercell for each of scan_lengths) and, with mace : true
# and scan : true, the MACE scan of every candidate. Each candidate is built in every size it fits
# (fits) and relaxed with MACE (positions only, mace_model; mace_relaxations, on compute nodes
# with mace_jobscript). Its relaxation energy (unrelaxed - relaxed) and the largest displacement
# of the atoms at the cell's edge (further than EDGE_FRACTION of the shortest image distance from
# the defect's centre) show whether the cell holds its relaxation: it is converged at the smallest
# size whose relaxation energy is within scan_tol of that of every bigger size, with its edge
# displacement below scan_edge_tol. One that never is (or fits only the largest size) is not
# converged, and B3 counts it as needing the largest size.
#   Called by: main() (step B2)
#   Calls:     Project.host(), build_defect(), fits(), format_table(), ladder_table(),
#              mace_cache(), mace_relaxations(), size_ladder(), sc.displacements(),
#              sc.get_distances()
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
    out, rows = {}, []
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
            seen.append({"tag": cell["tag"], "L": cell["L"], "E_relax": round(e0 - e1, 5),
                         "edge": round(float(edge.max()) if edge.size else 0.0, 4),
                         "steps": steps})
        conv = next((x["tag"] for i, x in enumerate(seen[:-1])
                     if x["edge"] < p.s.scan_edge_tol and
                     all(abs(y["E_relax"] - x["E_relax"]) < p.s.scan_tol for y in seen[i + 1:])),
                    None)
        out[cid] = {"sizes": seen, "converged": conv}
        by = {x["tag"]: x for x in seen}
        here = by.get(conv)
        rows.append([cid, c["name"]]
                    + [f"{by[t]['E_relax']:.3f}" + ("*" if t == conv else "") if t in by else ""
                       for t in (cell["tag"] for cell in ladder)]
                    + [f"{here['L']:.2f}" if here else "not converged",
                       here["edge"] if here else None])
    table = format_table(["id", "defect"] + [cell["tag"] for cell in ladder]
                         + ["converged at L/A", "edge move/A"], rows,
                         "MACE relaxation energies (eV) of each candidate in each size; * where "
                         f"it is converged (within scan_tol {p.s.scan_tol:g} eV of every bigger "
                         f"size, edge atoms moved less than scan_edge_tol {p.s.scan_edge_tol:g} "
                         "A)")
    bad = sorted(cid for cid, x in out.items() if not x["converged"])
    if bad:
        log(f"note: B2: {len(bad)} candidates did not converge within the ladder (up to "
            f"{ladder[-1]['L']:.1f} A): {', '.join(bad)}; B3 counts them as needing the largest "
            "size")
    return {"ladder": ladder, "candidates": out, "_table": sizes + "\n" + table,
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
#   Called by: step_choose()
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
#   Called by: step_choose()
def gamma_floor(p, masses):
    if p.s.screen_min_image is not None:
        return ({"L": p.s.screen_min_image, "source": "screen_min_image"}
                if p.s.screen_min_image > 0 else None)
    eb, eg = p.s.deep_level_margin, p.out("bands")["gap"]
    if eb >= eg:
        log(f"note: B3: no Gamma-only floor: deep_level_margin {eb:g} eV is not inside the gap")
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
        log("WARNING: B3: the effective masses could not be found from A2.2's band edges (its "
            "k-point grid may be too coarse), so there is no Gamma-only floor: set "
            "screen_min_image (Ang) to give one")
    return best


# B3: the screening cell, which every candidate in C is built in and D0's size series starts from.
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
                "defect levels spread into bands (C2b measures it)")
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


# ==========================================================================================
# PHASE C: SCREENING AND GROUND STATES (flowchart C1-C7)
# ==========================================================================================
# The candidates (B1) and the screening cell (B3): (candidates by id, primitive host, matrix).
#   Called by: step_dispersion(), step_distort(), step_filter(), step_prerelax(), step_screen()
#   Calls:     Project.host()
def screen_setup(p):
    return p.out("generate")["candidates"], p.host(), np.array(p.out("choose")["matrix"])


# The .param keywords of the screening runs: screen_cutoff, screen_max_steps and
# %block screen_overrides.
#   Called by: excite_job(), screen_job(), step_dispersion(), step_screen()
def screen_overrides(p):
    o = {"geom_max_iter": p.s.screen_max_steps}
    if p.s.screen_cutoff:
        o["cut_off_energy"] = f"{p.s.screen_cutoff:g} eV"
    o.update(p.s.screen_overrides)
    return o


# A screening or distortion job for one structure in charge state q, starting with spin S
# (spin_polarized, spin : S, and SPIN=S on the anchor atom when there is one): a Gamma-point
# relaxation without symmetry, with the screening keywords.
#   Called by: excite_job(), run_screen_jobs(), step_dispersion(), step_screen()
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
# its geometry converged, final spin, defect levels in the gap (from its .bands against the host's
# band edges at the same settings), and its final structure, saved as <structdir>/<name>.cell.
#   Called by: run_screen_jobs(), step_screen()
#   Calls:     gap_levels(), read_bands(), save_cell(), sc.final_atoms()
def job_summary(p, res, name, edges, structdir):
    r, calc = res["jobs"][name], res["calcs"][name]
    out = {"status": r["status"], "reason": r["reason"], "geom_converged": r["geom_converged"],
           "energy": r["energy"] if r["scf_converged"] else None, "spin": r["spin"],
           "steps": r["geom_steps"], "levels": None, "structure": None}
    bands = calc.dir / f"{calc.seed}.bands"
    if bands.exists() and edges:
        out["levels"] = gap_levels(read_bands(bands), edges[0], edges[1], p.s.level_margin)
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
# C1, C2, C2b
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
# starting spin (the lowest the electron count allows and, with screen_spins : both, that + 2;
# the spin is not held fixed), plus the pristine supercell at the same settings for the band edges
# and host energy. All at the Gamma point with the screening keywords; failures are flagged, not
# rescued (runner group 'screen': max_restarts 0, tolerate_failures). The results give each run's
# energy, final spin and gap levels, and the pristine cell's band edges. The pristine cell's gap
# at Gamma is checked against A2.2's: one more than FOLD_GAP_TOL wider means band edges that don't
# fold onto Gamma of this cell (B3 names them), which the Gamma-point levels are measured against.
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


# How the gap levels of one defect calculation on a k-point grid spread across its k-points (C2b),
# from its .bands and the pristine cell's at the same settings (both read_bands): the host's band
# edges over the grid; the defect's bands that are in the gap at Gamma (more than margin inside);
# for each, its width W (highest minus lowest energy over the k-points) and whether it crosses
# into a band at some k-point; and, in each spin channel, whether its highest occupied band at
# some k-point is above its lowest empty one at another (the occupations would change across the
# k-points, which a calculation with fixed occupations can't show). Returns {'W' (the largest),
# 'levels', 'crossing', 'overlap'}.
#   Called by: step_dispersion()
#   Calls:     occupied_bands()
def level_dispersion(bd, bh, margin):
    occ_h, occ = occupied_bands(bh), occupied_bands(bd)
    vbm = max(np.nanmax(bh["eig"][s, :, occ_h[s] - 1]) for s in range(bh["nspin"]))
    cbm = min(np.nanmin(bh["eig"][s, :, occ_h[s]]) for s in range(bh["nspin"]))
    g = int(np.argmin(np.abs(bd["kpts"] - np.round(bd["kpts"])).sum(axis=1)))
    levels, overlap = [], False
    for s in range(bd["nspin"]):
        e = bd["eig"][s]
        if np.nanmax(e[:, occ[s] - 1]) > np.nanmin(e[:, occ[s]]):
            overlap = True
        for n in range(e.shape[1]):
            if np.isfinite(e[g, n]) and vbm + margin < e[g, n] < cbm - margin:
                band = e[:, n]
                levels.append({"spin": s, "band": n + 1, "e": round(float(band[g] - vbm), 4),
                               "W": round(float(np.nanmax(band) - np.nanmin(band)), 4),
                               "occupied": n < occ[s],
                               "crossing": bool(np.nanmin(band) <= vbm + margin or
                                                np.nanmax(band) >= cbm - margin)})
    return {"W": max((x["W"] for x in levels), default=0.0), "levels": levels,
            "crossing": any(x["crossing"] for x in levels), "overlap": overlap}


# C2b: the Gamma-only check (dispersion_check : warn, the default, or stop; off skips it). One
# defect - dispersion_defect (an id, or id:q for one charge state), else the screened state whose
# gap level is nearest a band edge (the most spread-out level, the first to feel its images) - is
# run as a single point on a Gamma-centred 2x2x2 k-point grid in the screening cell and in the size
# below it (B3's ladder; its structure carried over, carry_over), with each pristine cell at the
# same settings for the band edges. How much its gap levels change across the k-points
# (level_dispersion: W) measures how much its images interact: W falls as exp(-d/xi) with the image
# distance d, so the two sizes give xi, xi = (d2 - d1) / ln(W1/W2), and the image distance at which
# W would be dispersion_tol. The screening cell passes if its W is below dispersion_tol and no level
# crosses into a band or changes occupation across the k-points. If not, a warning; with stop, C3
# stops before it filters (raise the floor with screen_min_image, or set supercell, and rerun).
#   Called by: main() (step C2b)
#   Calls:     Project.batch(), Project.production_param(), carry_over(), fits(),
#              format_table(), host_supercell(), level_dispersion(), load_cell(), qtag(),
#              read_bands(), screen_job(), screen_overrides(), screen_setup(), start_spin()
def step_dispersion(p):
    if p.s.dispersion_check == "off":
        return {"skipped": True, "ok": None, "_note": "dispersion_check : off"}
    cands, prim, P = screen_setup(p)
    choose, scr = p.out("choose"), p.out("screen")
    runs, host_s = scr["runs"], scr["host"]
    pick = None
    if p.s.dispersion_defect:
        cid, _, qs = p.s.dispersion_defect.partition(":")
        mine = [(r["energy"], n) for n, r in runs.items() if r["cand"] == cid and r["energy"]
                is not None and (not qs or r["q"] == int(qs))]
        if not mine:
            die(f"dispersion_defect {p.s.dispersion_defect}: no screened run of it with a usable "
                "energy (C2)")
        pick = min(mine)[1]
    else:
        near = [(min(min(x["e"], host_s["gap"] - x["e"]) for x in r["levels"]), n)
                for n, r in sorted(runs.items()) if r["energy"] is not None and r["levels"]]
        if near:
            pick = min(near)[1]
    if pick is None:
        return {"skipped": True, "ok": None,
                "_note": "no screened state has a level in the gap, so there is nothing to check"}
    r = runs[pick]
    c = cands[r["cand"]]
    ladder, idx = choose["ladder"], choose["index"]
    sizes = [ladder[i] for i in (idx - 1, idx) if i >= 0 and fits(c, ladder[i]["L"])]
    atoms = load_cell(p, r["structure"])
    info = p.out("prerelax")["defects"][r["cand"]]
    S = start_spin(atoms, r["q"], r["spin"])
    over = {k: v for k, v in screen_overrides(p).items() if k != "geom_max_iter"}
    jobs, names = [], {}
    for cell in sizes:
        Q = np.array(cell["matrix"])
        if cell["tag"] == choose["tag"]:
            a, inf = atoms, info
        else:
            a, inf = carry_over(prim, c["comps"], P, atoms, Q, label=f"{r['cand']} ({c['name']})")
        job = screen_job(p, f"{r['cand']}_{qtag(r['q'])}_{cell['tag']}", a, inf, r["q"], S)
        job["overrides"].pop("geom_max_iter", None)
        job.update(kind="sp", kgrid=[2, 2, 2], kpoint_offset="gamma")
        jobs += [job, {"name": f"host_{cell['tag']}", "atoms": host_supercell(prim, Q),
                       "kind": "sp", "kgrid": [2, 2, 2], "kpoint_offset": "gamma",
                       "symmetry": True, "overrides": over}]
        names[cell["tag"]] = (job["name"], f"host_{cell['tag']}")
    res = p.batch("C_screen/jobs/dispersion", jobs, "dispersion", p.production_param())
    if not res["complete"]:
        return None
    found, rows = [], []
    for cell in sizes:
        dn, hn = names[cell["tag"]]
        files = [res["calcs"][n].dir / f"{res['calcs'][n].seed}.bands" for n in (dn, hn)]
        if any(res["jobs"][n]["status"] != "done" for n in (dn, hn)) or \
                not all(f.exists() for f in files):
            log(f"WARNING: C2b: the dispersion runs in {cell['tag']} did not finish ("
                + "; ".join(f"{n}: {res['jobs'][n]['status']}" for n in (dn, hn)) + ")")
            continue
        lv = level_dispersion(read_bands(files[0]), read_bands(files[1]), p.s.level_margin)
        found.append(dict(lv, tag=cell["tag"], L=cell["L"]))
        rows.append([cell["tag"], cell["L"], len(lv["levels"]), lv["W"],
                     "yes" if lv["crossing"] else "no", "yes" if lv["overlap"] else "no"])
    out = {"run": pick, "cand": r["cand"], "q": r["q"], "sizes": found, "xi": None,
           "d_needed": None, "ok": None, "stop": None}
    if not found or found[-1]["tag"] != choose["tag"]:
        log("WARNING: C2b: the dispersion check did not run in the screening cell, so it says "
            "nothing (see the runs in C_screen/jobs/dispersion)")
        return dict(out, _note="no result in the screening cell (runs failed)",
                    _table=format_table(["size", "L/A", "gap levels", "W/eV", "crosses a band",
                                         "occupations change"], rows))
    top, tol = found[-1], p.s.dispersion_tol
    if len(found) == 2 and found[0]["W"] > top["W"] > 0:
        out["xi"] = round((top["L"] - found[0]["L"]) / math.log(found[0]["W"] / top["W"]), 3)
        out["d_needed"] = round(top["L"] + out["xi"] * math.log(top["W"] / tol), 2)
    out["ok"] = top["W"] < tol and not top["crossing"] and not top["overlap"]
    what = (f"{r['cand']} {c['name']} q {r['q']:+d}: W {top['W']:.3f} eV in the screening cell "
            f"({top['L']:.2f} A)" + (f", xi {out['xi']:.2f} A: W < {tol:g} eV needs images "
                                     f"{out['d_needed']:.1f} A apart" if out["xi"] else ""))
    if not out["ok"]:
        why = ([f"W above dispersion_tol {tol:g} eV"] if top["W"] >= tol else []) + \
              (["a gap level crosses into a band at some k-point"] if top["crossing"] else []) + \
              (["the occupations change across the k-points"] if top["overlap"] else [])
        msg = (f"C2b: the screening cell may be too small for Gamma-point levels ({what}; "
               + "; ".join(why) + "). Raise the floor with screen_min_image"
               + (f" : {out['d_needed']:.1f} ang" if out["d_needed"] else "")
               + " (or set supercell) and rerun with --redo C")
        log("WARNING: " + msg)
        if p.s.dispersion_check == "stop":
            out["stop"] = msg + ", or set dispersion_check : warn to carry on with this cell"
    floor = choose.get("floor") or {}
    table = format_table(["size", "L/A", "gap levels", "W/eV", "crosses a band",
                          "occupations change"], rows,
                         f"Gamma-only check: {r['cand']} {c['name']} q {r['q']:+d} ({pick}) on a "
                         f"Gamma-centred 2x2x2 grid (dispersion_tol {tol:g} eV)") + (
        f"\nxi measured {out['xi']:.2f} A" + (f", estimated in B3 {floor['xi']:.2f} A"
                                              if floor.get("xi") else "") + "\n"
        if out["xi"] else "")
    return dict(out, _table=table, _note=("ok: " if out["ok"] else "TOO SMALL? ") + what)


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
# mapped to the one with the lowest energy, which stays. Returns {duplicate: the one it matches}.
#   Called by: step_filter()
#   Calls:     load_cell(), same_structures()
def duplicates(p, best, groups):
    dup = {}
    for ids in groups.values():
        ids = sorted(ids, key=lambda c: min(s["energy"] for s in best[c].values()))
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
# For each composition group and Fermi level across the screening gap, the lowest formation energy
# of any arrangement and charge state (relative_formation); an arrangement is kept if its own
# lowest is within hull_margin of that somewhere in the gap. Its stable charge states are those
# lowest for it somewhere in the gap. Then, if asked: gap levels (require_gap_states: occupied,
# empty, or both = an occupied and an empty level in one spin channel), the final spin
# (require_spin) and binding against its isolated components (require_bound: the components must
# have been screened too) - each met by at least one stable charge state. Arrangements that relaxed
# into the same structure as a lower one of their group in every charge state both have
# (duplicates) are dropped too. With excite : true, native defects (is_native) skip the property
# filters (C7 passes them on to D whatever their excited states, for E), but not the hull or the
# duplicates. The results list the survivors, with the charge states to carry on
# (distort_charges); the table gives every candidate's verdict. With dispersion_check : stop, a
# screening cell C2b found too small stops the workflow here, before anything is filtered.
#   Called by: main() (step C3)
#   Calls:     best_states(), duplicates(), format_table(), host_supercell(), internal_pair(),
#              is_native(), relative_formation(), screen_setup()
def step_filter(p):
    if p.out("dispersion").get("stop"):
        die(p.out("dispersion")["stop"])
    cands, prim, P = screen_setup(p)
    scr = p.out("screen")
    host, runs = scr["host"], scr["runs"]
    best = best_states(runs)
    eps = p.out("dielectric").get("eps0")
    cell = host_supercell(prim, P).cell
    ef = np.linspace(0.0, host["gap"], 401)
    form = {cid: relative_formation(st, ef, host["vbm"], cell, eps, p.s.charge_correction)
            for cid, st in best.items()}
    env = {cid: np.min(np.array(list(f.values())), axis=0) for cid, f in form.items()}
    groups = {}
    for cid in best:
        groups.setdefault(cands[cid]["group"], []).append(cid)
    hull = {g: np.min(np.array([env[c] for c in ids]), axis=0) for g, ids in groups.items()}
    singles = {(c["comps"][0]["new"], c["comps"][0]["host"], c["comps"][0].get("prim_site"),
                tuple(c["comps"][0].get("frac") or ())): cid
               for cid, c in cands.items() if len(c["comps"]) == 1}
    dup = duplicates(p, best, groups)
    left_out = p.out("prerelax")["skipped"]
    host_el = set(prim.get_chemical_symbols())
    survivors, rows = [], []
    for cid in sorted(cands):
        c = cands[cid]
        if cid not in best:
            why = (f"left out in C1: {left_out[cid]}" if cid in left_out
                   else "dropped: no usable screening run")
            rows.append([cid, c["name"], c["group"], "", "", "", "", why])
            continue
        dist = float(np.min(env[cid] - hull[c["group"]]))
        lowest = np.argmin(np.array([form[cid][q] for q in sorted(form[cid])]), axis=0)
        stable = sorted({sorted(form[cid])[i] for i in np.unique(lowest)})
        why = [f"relaxed into the structure of {dup[cid]}"] if cid in dup else []
        if dist > p.s.hull_margin:
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
        failed = []                                  # the property filters it fails
        if not good:
            asked = [x for x, on in (("gap levels", p.s.require_gap_states != "none"),
                                     ("spin", p.s.require_spin is not None)) if on]
            failed.append("no stable charge state with the " + " and ".join(asked)
                          + " asked for")
        bind = None
        if len(c["comps"]) > 1:
            parts = [singles.get((x["new"], x["host"], x.get("prim_site"),
                                  tuple(x.get("frac") or ()))) for x in c["comps"]]
            if all(pt in env for pt in parts):
                eb = sum(env[pt] for pt in parts) - env[cid] - (len(parts) - 1) * host["energy"]
                bind = (float(eb.min()), float(eb.max()))
        if p.s.require_bound and len(c["comps"]) > 1:
            if bind is None:
                failed.append("binding energy unknown (its components were not screened)")
            elif bind[1] <= 0:
                failed.append(f"not bound (E_b {bind[0]:.2f} to {bind[1]:.2f} eV)")
        waived = bool(failed) and p.s.excite and is_native(c, host_el)
        if not waived:
            why += failed
        charges = stable if p.s.distort_charges == "stable" else sorted(best[cid])
        spins = ", ".join(f"{q:+d}:{best[cid][q]['spin']:.1f}" if best[cid][q]["spin"] is not None
                          else f"{q:+d}:-" for q in stable)
        rows.append([cid, c["name"], c["group"], dist, " ".join(f"{q:+d}" for q in stable), spins,
                     f"{bind[0]:.2f}..{bind[1]:.2f}" if bind else "",
                     ("kept" + (" (native, so the property filters are waived for E: "
                                + "; ".join(failed) + ")" if waived else ""))
                     if not why else "dropped: " + "; ".join(why)])
        if not why:
            survivors.append({"cand": cid, "name": c["name"], "group": c["group"],
                              "hull_distance": dist, "stable": stable, "charges": charges,
                              "good": good, "binding": bind,
                              "best": {str(q): best[cid][q] for q in best[cid]}})
    table = format_table(["id", "defect", "composition", "above hull/eV", "stable q",
                          "spin (q:S)", "E_b/eV", "verdict"], rows,
                         f"Defect hull (margin {p.s.hull_margin:g} eV) and property filters")
    return {"survivors": survivors, "_table": table,
            "_note": f"{len(survivors)} of {len(cands)} candidates kept"}



# --------------------------------------------------------------------------
# C4-C6: ground-state search
# --------------------------------------------------------------------------
# The oxidation states for counting the neighbours to distort: %block oxidation_states, else
# pymatgen's guess for the host and, for each dopant, its common oxidation state closest to that
# of the element it replaces (or its most positive one).
#   Called by: step_distort()
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
#   Called by: step_dispersion(), step_distort(), step_excite(), step_rank()
#   Calls:     electron_parity()
def start_spin(atoms, q, spin):
    par = electron_parity(atoms, q)
    if spin is None:
        return par
    s = int(round(abs(spin)))
    return s if s % 2 == par else s + 1


# C4: the distortions. For each survivor and each of its charge states (charges from C3), starting
# from its screened structure in that charge state (or the nearest screened one), ShakeNBreak's
# distort_and_rattle moves the neighbours_to_distort nearest neighbours of the defect (the anchor
# atom, or with distort_around : vacancy / a vacancy anchor, the first vacant site; a defect with
# no vacancy always uses its anchor atom) in or out by each of distortions, and rattles the other
# atoms (rattle_stdev, default 10% of the host bond length; rattle_dmin, 80%); with no neighbours
# to distort it is only rattled. The structures go to C_screen/structures/distortions/.
#   Called by: main() (step C4)
#   Calls:     host_supercell(), load_cell(), neighbours_to_distort(), oxidation_states(),
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
    jobs = []
    for s in survivors:
        cid = s["cand"]
        info = defects[cid]
        screened = {int(q): b for q, b in s["best"].items()}
        for q in s["charges"]:
            src = screened.get(q) or screened[min(screened, key=lambda x: (abs(x - q), x))]
            atoms = load_cell(p, src["structure"])
            S = start_spin(atoms, q, src["spin"] if src["q"] == q else None)
            nb = neighbours_to_distort(cands[cid]["comps"], q, ox)
            struct = AseAtomsAdaptor.get_structure(atoms)
            around = info.get("anchor_index")
            centre = {}
            vacancy = (p.s.distort_around == "vacancy" or around is None) and info.get("vacancies")
            if vacancy:                     # (a defect with no vacancy uses its anchor atom)
                centre["frac_coords"] = np.linalg.solve(np.asarray(atoms.cell).T,
                                                        np.asarray(info["vacancies"][0])) % 1.0
            elif around is not None:
                centre["site_index"] = around
            else:
                die(f"{cid}: no anchor atom and no vacancy to distort around")
            factors = [f for f in p.s.distortions if abs(f) > 1e-9] if nb else ["rattled"]
            for f in factors:
                tag = "rattled" if f == "rattled" else f"b{'m' if f < 0 else 'p'}{abs(f):.2f}"
                out = distort_and_rattle(struct, 1.0 if f == "rattled" else 1.0 + f, nb,
                                         stdev=stdev, d_min=dmin, **centre)
                new = AseAtomsAdaptor.get_atoms(out["distorted_structure"])
                new.set_initial_magnetic_moments(atoms.get_initial_magnetic_moments())
                name = f"{cid}_{qtag(q)}_{tag}"
                rel = save_cell(p, d / f"{name}.cell", new,
                                f"{name}: {cands[cid]['name']}, q = {q:+d}, "
                                + ("rattled" if f == "rattled" else
                                   f"{nb} neighbours' bonds {f:+.2f}") + " (C4)")
                jobs.append({"name": name, "cand": cid, "q": q, "spin": S, "factor": f,
                             "neighbours": nb, "structure": rel})
    return {"jobs": jobs, "oxidation_states": ox,
            "_note": f"{len(jobs)} distorted structures for {len(survivors)} survivors "
                     "(oxidation states " + ", ".join(f"{k} {v:+g}" for k, v in ox.items()) + ")"}


# Runs a list of distortion-type jobs (name, cand, starting structure, q, spin) with the
# screening settings in C_screen/jobs/<subdir>, with the runner settings of the step that runs them
# (drelax or rank), and returns {name: job_summary}, the final structures saved in
# C_screen/structures/<structdir>.
#   Called by: step_drelax(), step_rank()
#   Calls:     Project.batch(), Project.production_param(), job_summary(), load_cell(),
#              screen_job()
def run_screen_jobs(p, step, subdir, todo, structdir):
    defects = p.out("prerelax")["defects"]
    jobs = []
    for t in todo:
        jobs.append(screen_job(p, t["name"], load_cell(p, t["structure"]), defects[t["cand"]],
                               t["q"], t["spin"]))
    res = p.batch(f"C_screen/jobs/{subdir}", jobs, step, p.production_param())
    if not res["complete"]:
        return None
    edges = p.out("screen")["host"]
    out = {}
    for t in todo:
        out[t["name"]] = dict(t, **job_summary(p, res, t["name"], (edges["vbm"], edges["cbm"]),
                                               p.phase_dir("C") / "structures" / structdir))
    return out


# C5: relaxes every distorted structure with the screening settings (C2's, Gamma point, no
# restarts).
#   Called by: main() (step C5)
#   Calls:     problem_table(), run_screen_jobs()
def step_drelax(p):
    out = run_screen_jobs(p, "drelax", "drelax", p.out("distort")["jobs"], "drelax")
    if out is None:
        return None
    bad = sum(r["energy"] is None for r in out.values())
    return {"runs": out, "_table": problem_table(out),
            "_note": f"{len(out)} relaxations" + (f", {bad} without a usable energy" if bad
                                                  else "")}


# Groups structures that are the same (pymatgen's StructureMatcher, which also finds a distortion
# repeated along another direction), as lists of indices; without pymatgen, by comparing atom by
# atom (all come from the same cell with the same atom order). The tolerances are tight (stol 0.05,
# i.e. about 0.1 A for a typical oxide): defect ground states can differ by a few tenths of an
# Angstrom around the defect, which pymatgen's defaults would merge.
#   Called by: distinct_states(), duplicates(), step_rank()
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


# The distinct relaxed structures of each candidate and charge state, lowest energy first: the
# screened one (C2) and every distortion relaxation with a usable energy, identical ones merged
# (same_structures, keeping the lowest). Returns {(cand, q): [entries]}.
#   Called by: step_rank()
#   Calls:     load_cell(), same_structures()
def distinct_states(p, survivors, results):
    pool = {}
    for s in survivors:
        for q in s["charges"]:
            b = s["best"].get(str(q))
            if b and b["structure"]:
                pool.setdefault((s["cand"], q), []).append(
                    {"name": b["name"], "energy": b["energy"], "spin": b["spin"],
                     "structure": b["structure"], "source": "screened", "factor": None})
    for name, r in results.items():
        if r["energy"] is not None and r["structure"] and (r["cand"], r["q"]) in pool:
            pool[(r["cand"], r["q"])].append(
                {"name": name, "energy": r["energy"], "spin": r["spin"],
                 "structure": r["structure"], "source": r.get("source", "distortion"),
                 "factor": r.get("factor")})
    out = {}
    for key, entries in pool.items():
        entries.sort(key=lambda e: e["energy"])
        atoms = [load_cell(p, e["structure"]) for e in entries]
        groups = same_structures(atoms)
        firsts = sorted(min(g) for g in groups)
        out[key] = [dict(entries[i], copies=len(next(g for g in groups if min(g) == i)))
                    for i in firsts]
    return out


# C6: ranks the relaxed structures of each survivor and charge state (distinct_states). A
# distortion counts as found when it beats the screened structure by distortion_min_gain. A
# structure found in one charge state that no other charge state has is then relaxed in those
# charge states too (a batch per round, C_screen/jobs/share_<n>), for up to share_rounds rounds or
# until nothing new turns up. The lowest keep_structures distinct structures of each charge state
# go to phase D.
#   Called by: main() (step C6)
#   Calls:     distinct_states(), format_table(), load_cell(), qtag(), run_screen_jobs(),
#              same_structures(), start_spin()
def step_rank(p):
    survivors = p.out("filter")["survivors"]
    results = dict(p.out("drelax")["runs"])
    tried = set()
    for rnd in range(1, p.s.share_rounds + 1):
        states = distinct_states(p, survivors, results)
        todo = []
        for (cid, q), entries in states.items():
            best = entries[0]
            if best["source"] == "screened" or best["name"] in tried:
                continue
            e0 = next((b["energy"] for s in survivors if s["cand"] == cid
                       for qq, b in s["best"].items() if int(qq) == q), None)
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
                todo.append({"name": f"{best['name']}_to_{qtag(q2)}", "cand": cid, "q": q2,
                             "spin": spin, "factor": best["factor"], "source": f"from {qtag(q)}",
                             "structure": best["structure"]})
        if not todo:
            break
        log(f"C6 round {rnd}: trying {len(todo)} structures found in one charge state in the "
            "others")
        out = run_screen_jobs(p, "rank", f"share_{rnd}", todo, "share")
        if out is None:
            return None
        results.update(out)
    states = distinct_states(p, survivors, results)
    screened = {(s["cand"], int(q)): b["energy"] for s in survivors for q, b in s["best"].items()}
    keep, rows = [], []
    for (cid, q), entries in sorted(states.items()):
        best = entries[0]
        e0 = screened.get((cid, q), best["energy"])
        gain = e0 - best["energy"]
        rows.append([cid, f"{q:+d}", e0, best["energy"], gain,
                     best["source"] + (f" {best['factor']}" if best["factor"] is not None else ""),
                     len(entries)])
        for k, e in enumerate(entries[:p.s.keep_structures]):
            atoms = load_cell(p, e["structure"])
            keep.append({"cand": cid, "q": q, "k": k, "energy": e["energy"],
                         "spin": start_spin(atoms, q, e["spin"]), "structure": e["structure"],
                         "source": e["source"], "factor": e["factor"]})
    table = format_table(["id", "q", "screened E/eV", "best E/eV", "gain/eV", "from",
                          "distinct"], rows,
                         "Ground-state search (energies at the screening settings)")
    found = sum(1 for r in rows if r[4] > p.s.distortion_min_gain)
    return {"structures": keep, "_table": table,
            "_note": f"{len(keep)} structures for phase D; a distortion won in {found} of "
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


# The excitation C7 calculates for a ground state, chosen as ADAQ does: from its .bands
# (read_bands) at the screening settings, the smallest step of at least emin from an occupied to an
# empty level in the same spin channel, both more than margin inside the host's band edges (vbm,
# cbm: the levels gap_levels counts). Returns {spin (0 or 1), from_band, to_band (counted from 1
# in that spin channel, as CASTEP does), e_from, e_to (eV above the VBM), step (eV)}, or None.
#   Called by: step_excite()
#   Calls:     occupied_bands()
def excitation(b, vbm, cbm, margin, emin):
    occ, best, step = occupied_bands(b), None, None
    for s in range(b["nspin"]):
        levels = [(n, float(x)) for n, x in enumerate(b["eig"][s, 0])
                  if np.isfinite(x) and vbm + margin < x < cbm - margin]
        for i, ei in (lv for lv in levels if lv[0] < occ[s]):
            for j, ej in (lv for lv in levels if lv[0] >= occ[s]):
                if ej - ei >= emin and (step is None or ej - ei < step - 1e-6):
                    step = ej - ei
                    best = {"spin": s, "from_band": i + 1, "to_band": j + 1,
                            "e_from": round(ei - vbm, 4), "e_to": round(ej - vbm, 4),
                            "step": round(step, 4)}
    return best


# The %block deltascf_constraints of an excitation (excitation): its occupied band emptied and its
# empty band filled, in its spin channel (CASTEP's 1 = majority, 2 = minority), each looked for at
# its own band index only. In a calculation without spin polarisation an occupation of 1 is two
# electrons, so moving one electron is 0.5 in each band.
#   Called by: step_excite()
def deltascf_block(ex, nspin):
    lo, hi = (0.5, 0.5) if nspin == 1 else (0.0, 1.0)
    sp = ex["spin"] + 1 if nspin == 2 else 1
    rows = [f"  {band:5d} {occ:8.4f} {sp:4d} {band:4d} {band:4d}"
            for band, occ in ((ex["from_band"], lo), (ex["to_band"], hi))]
    return ("%block deltascf_constraints\n" + "\n".join(rows)
            + "\n%endblock deltascf_constraints\n")


# A C7 job for one structure in charge state q, starting with spin S: a screening job (screen_job:
# Gamma point, no symmetry, the screening keywords) with metals_method : dm (CASTEP's excitation
# constraints need density mixing), %block excite_overrides, then extra (the ΔSCF keywords of an
# excited state). kind: 'sp' (the ground state; the vertical excited state) or 'geom' (the relaxed
# excited state).
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


# The mass-weighted distance between two structures of the same cell, in amu^1/2 Ang:
# ΔQ = sqrt(sum_i m_i |Δr_i|^2), each Δr_i the shortest through the periodic images.
#   Called by: step_excite()
def delta_q(a, b):
    from ase.geometry import find_mic
    d, _ = find_mic(b.positions - a.positions, a.cell, pbc=True)
    return float(np.sqrt(np.sum(a.get_masses()[:, None] * d * d)))


# C7 (excite : true): the excited state of each defect's ground state in each charge state (the
# lowest structure C6 kept), at the screening settings, in two batches:
#  1. a single point of the ground state (C_screen/jobs/excite_gs; its .check is kept even with
#     delete_check : true), whose .bands gives the excitation (excitation);
#  2. the excited state (C_screen/jobs/excite): CASTEP's simple ΔSCF (calculate_deltascf, the two
#     bands' occupations in %block deltascf_constraints), starting from the ground state's .check
#     (reuse, deltascf_checkpoint), relaxed with excite_relax : true (energy = the ZPL, E of the
#     relaxed excited state - E of the ground state; and ΔQ, delta_q), else a single point (the
#     vertical energy).
# A charge state passes if its energy is in excite_window (any converged energy, without a window);
# one without an excitation in the gap does not. One whose runs failed or did not converge is
# 'unknown' and counts as a pass, with a warning. A defect goes on to D (with all the structures C6
# kept for it) if a charge state passes, and every native defect (is_native) goes on anyway. With
# excite : false, everything C6 kept goes on.
#   Called by: main() (step C7)
#   Calls:     Project.batch(), Project.host(), Project.production_param(), delta_q(),
#              deltascf_block(), excitation(), excite_job(), format_table(), is_native(),
#              load_cell(), qtag(), read_bands(), save_cell(), start_spin(), sc.final_atoms()
def step_excite(p):
    structures = p.out("rank")["structures"]
    if not p.s.excite:
        return {"structures": structures, "excited": {}, "relaxed": None,
                "_note": "excite is off, so every structure C6 kept goes on to D"}
    cands, defects = p.out("generate")["candidates"], p.out("prerelax")["defects"]
    host_el = set(p.host().get_chemical_symbols())
    edges = p.out("screen")["host"]
    ground = [dict(t, gs=f"{t['cand']}_{qtag(t['q'])}_gs") for t in structures if t["k"] == 0]
    jobs = [excite_job(p, t["gs"], load_cell(p, t["structure"]), defects[t["cand"]], t["q"],
                       t["spin"], "sp") for t in ground]
    res = p.batch("C_screen/jobs/excite_gs", jobs, "excite", p.production_param(),
                  keep_check=True)
    if not res["complete"]:
        return None
    exc, jobs = {}, []
    exdir = p.dir / "C_screen" / "jobs" / "excite"
    for t in ground:
        r, calc = res["jobs"][t["gs"]], res["calcs"][t["gs"]]
        e = {"structure": t["structure"], "gs_job": t["gs"], "energy_gs": None,
             "excitation": None, "job": None, "energy": None, "dq": None, "verdict": "unknown",
             "why": ""}
        exc.setdefault(t["cand"], {})[str(t["q"])] = e
        bands = calc.dir / f"{calc.seed}.bands"
        if r["status"] != "done" or r["energy"] is None or not r["scf_converged"]:
            e["why"] = ("ground state: " + (r["reason"] or "no converged energy")).splitlines()[0]
            continue
        if not bands.exists():
            e["why"] = "ground state: no .bands file"
            continue
        e["energy_gs"] = r["energy"]
        b = read_bands(bands)
        ex = excitation(b, edges["vbm"], edges["cbm"], p.s.level_margin, p.s.excite_min)
        if ex is None:
            e.update(verdict="none", why=f"no step >= {p.s.excite_min:g} eV in the gap")
            continue
        name = f"{t['cand']}_{qtag(t['q'])}_ex"
        check = os.path.relpath(calc.dir / f"{calc.seed}.check", exdir / name)
        atoms = load_cell(p, t["structure"])
        job = excite_job(p, name, atoms, defects[t["cand"]], t["q"],
                         start_spin(atoms, t["q"], r["spin"]),
                         "geom" if p.s.excite_relax else "sp",
                         {"calculate_deltascf": "true", "deltascf_method": "simple",
                          "reuse": check, "deltascf_checkpoint": check})
        job["param_text"] = p.production_param().rstrip() + "\n" + deltascf_block(ex, b["nspin"])
        jobs.append(job)
        e.update(excitation=ex, job=name)
    if jobs:
        res = p.batch("C_screen/jobs/excite", jobs, "excite", p.production_param())
        if not res["complete"]:
            return None
    lo, hi = p.s.excite_window or (None, None)
    for per_q in exc.values():
        for e in per_q.values():
            if not e["job"]:
                continue
            r = res["jobs"][e["job"]]
            if r["status"] != "done" or r["energy"] is None or not r["scf_converged"]:
                e["why"] = ("excited state: " + (r["reason"] or "no converged energy")
                            ).splitlines()[0]
                continue
            if p.s.excite_relax and not r["geom_converged"]:
                e["why"] = "excited state: the relaxation did not converge"
                continue
            e["energy"] = round(r["energy"] - e["energy_gs"], 4)
            if p.s.excite_relax:
                atoms = sc.final_atoms(res["calcs"][e["job"]])
                if atoms is not None:
                    e["dq"] = round(delta_q(load_cell(p, e["structure"]), atoms), 3)
                    e["structure_ex"] = save_cell(
                        p, p.phase_dir("C") / "structures" / "excite" / f"{e['job']}.cell", atoms,
                        f"{e['job']}: relaxed excited state (C7, screening settings)")
            inside = lo is None or lo <= e["energy"] <= hi
            e["verdict"] = "pass" if inside else "outside"
    native = sorted(cid for cid in exc if is_native(cands[cid], host_el))
    keep = sorted(cid for cid, per_q in exc.items() if cid in native
                  or any(e["verdict"] in ("pass", "unknown") for e in per_q.values()))
    rows = []
    for cid, per_q in sorted(exc.items()):
        goes = ("yes (native)" if cid in native else "yes") if cid in keep else "no"
        for qs, e in sorted(per_q.items(), key=lambda kv: int(kv[0])):
            ex = e["excitation"]
            if e["verdict"] == "unknown":
                log(f"WARNING: C7: {cid} {cands[cid]['name']} q {int(qs):+d}: {e['why']} - kept")
            rows.append([cid, cands[cid]["name"], f"{int(qs):+d}",
                         f"{ex['spin'] + 1}: {ex['from_band']}->{ex['to_band']}" if ex else "",
                         ex["step"] if ex else None, e["energy"], e["dq"],
                         e["verdict"] + (f": {e['why']}" if e["why"] else ""), goes])
    what = "ZPL" if p.s.excite_relax else "vertical E"
    table = format_table(["id", "defect", "q", "spin: bands", "KS step/eV", f"{what}/eV",
                          "dQ/amu^0.5 A", "verdict", "on to D"], rows,
                         "Excited states (ΔSCF at the screening settings)"
                         + (f", window {lo:g}-{hi:g} eV" if lo is not None else ""))
    out = [t for t in structures if t["cand"] in keep]
    return {"structures": out, "excited": exc, "relaxed": p.s.excite_relax, "native": native,
            "kept": keep, "_table": table,
            "_note": f"{len(keep)} of {len(exc)} defects go on to D ({len(native)} native); "
                     f"{len(out)} structures"}


SCREEN_KEYS = ("screen_cutoff", "screen_max_steps", "screen_overrides", "level_margin", "energy")


# ==========================================================================================
# PHASE D: PRODUCTION CELLS AND RELAXATIONS (flowchart D0-D2)
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
#   Called by: step_cells(), step_production()
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
#   Called by: step_cells(), step_production()
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


# D0: the production cell of every defect phase C passes on (C7's), converged at the production
# settings; all the charge states of a defect share it. Each charge state's ground state (the
# lowest structure C6 kept) is relaxed at the production settings (production_job) in the
# screening cell and in the size below it (B3's ladder; the structure carried over, carry_over),
# with the pristine supercell's single point in each (runner group 'production'). A defect is
# converged at the smaller of two neighbouring sizes when every charge state's E_D - E_host +
# E_corr (what goes into its formation energy; correction, in each cell's own shape) agrees within
# series_tol and both relaxed into the same local structure (same_basin); otherwise the next size
# up is run, carried over from the biggest one relaxed so far, and so on, up to the largest size of
# the ladder, where a defect that still hasn't converged keeps that size and is flagged. Sizes a
# pair doesn't fit (fits) are skipped; a charge state whose run failed is left out of the
# comparison (with a warning). The results give each defect's cell, and every run, which D1 reuses.
#   Called by: main() (step D0)
#   Calls:     Project.batch(), Project.host(), Project.kspacing(), Project.production_param(),
#              build_defect(), carry_over(), correction(), fits(), format_table(),
#              host_supercell(), load_cell(), production_job(), production_summary(), qtag(),
#              same_basin()
def step_cells(p):
    d = p.phase_dir("D")
    structures = p.out("excite")["structures"]
    if not structures:
        die("phase C passed no structures on, so there is nothing to relax")
    cands, prim = p.out("generate")["candidates"], p.host()
    choose = p.out("choose")
    ladder, idx, P0 = choose["ladder"], choose["index"], np.array(choose["matrix"])
    eps = p.out("dielectric").get("eps0")
    ground = {}
    for t in structures:
        if t["k"] == 0:
            ground.setdefault(t["cand"], {})[t["q"]] = t
    plan = {}
    for cid in ground:
        fit = [i for i in range(len(ladder)) if fits(cands[cid], ladder[i]["L"])]
        start = [i for i in fit if i >= idx - 1][:2]
        plan[cid] = {"sizes": start, "src": {i: None for i in start}, "cell": None,
                     "converged": False, "why": ""}
    done, rows, runs, hosts = {}, [], {}, {}
    while True:
        jobs, meta, frozen = [], {}, {}
        used = sorted({i for x in plan.values() for i in x["sizes"]})
        for i in used:
            c = ladder[i]
            jobs.append({"name": f"host_{c['tag']}", "atoms": host_supercell(prim,
                                                                          np.array(c["matrix"])),
                         "kind": "sp", "symmetry": True,
                         "overrides": dict(p.s.production_overrides)})
        for cid, x in plan.items():
            comps, label = cands[cid]["comps"], f"{cid} ({cands[cid]['name']})"
            for i in x["sizes"]:
                c = ladder[i]
                Q = np.array(c["matrix"])
                for q, t in sorted(ground[cid].items()):
                    name = f"{cid}_{qtag(q)}_{c['tag']}"
                    s = x["src"][i]                # the size it is carried from (None: C6's)
                    if i == idx:
                        atoms, info = load_cell(p, t["structure"]), build_defect(prim, Q, comps,
                                                                                 label)[1]
                    elif s is not None and (cid, q, s) in done:
                        atoms, info = carry_over(prim, comps, np.array(ladder[s]["matrix"]),
                                                 done[(cid, q, s)], Q, label=label)
                    else:
                        atoms, info = carry_over(prim, comps, P0, load_cell(p, t["structure"]), Q,
                                                 label=label)
                    job, frozen[name] = production_job(p, name, atoms, info, q, t["spin"])
                    jobs.append(job)
                    meta[name] = (cid, q, i)
        if frozen and any(frozen.values()):
            log(f"freeze_radius {p.s.freeze_radius:g} A holds {min(frozen.values())}-"
                f"{max(frozen.values())} atoms fixed in each relaxation (untested idea: check it "
                "against a full relaxation)")
        res = p.batch("D_production/jobs/cells", jobs, "cells", p.production_param(),
                      spacing=p.kspacing())
        if not res["complete"]:
            return None
        for i in used:
            hr = res["jobs"][f"host_{ladder[i]['tag']}"]
            if hr["status"] != "done" or hr["energy"] is None:
                die(f"the pristine {ladder[i]['tag']} single point did not finish ({hr['dir']}): "
                    "the formation energies need it")
            hosts[ladder[i]["tag"]] = hr["energy"]
        for name, (cid, q, i) in meta.items():
            if name not in runs:
                runs[name], atoms = production_summary(p, res, name,
                                                       d / "structures" / "cells", frozen[name])
                runs[name].update(cand=cid, q=q, cell=ladder[i]["tag"])
                if atoms is not None:
                    done[(cid, q, i)] = atoms
        more = False
        for cid, x in plan.items():
            if x["cell"] is not None:
                continue
            if len(x["sizes"]) < 2:
                x.update(cell=ladder[x["sizes"][0]]["tag"], why="only one size of the ladder "
                         "from the screening cell up fits it: not checked")
                continue
            a, b = x["sizes"][-2:]
            comps, label = cands[cid]["comps"], f"{cid} ({cands[cid]['name']})"
            diffs, basins, lost = {}, {}, []
            for q in sorted(ground[cid]):
                if (cid, q, a) not in done or (cid, q, b) not in done:
                    lost.append(q)
                    continue
                ef = []
                for i in (a, b):
                    c = ladder[i]
                    cell = np.array(c["matrix"]) @ np.asarray(prim.cell)
                    ef.append(runs[f"{cid}_{qtag(q)}_{c['tag']}"]["energy"] - hosts[c["tag"]]
                              + correction(q, cell, eps, p.s.charge_correction))
                diffs[q] = ef[1] - ef[0]
                basins[q] = same_basin(prim, comps, np.array(ladder[a]["matrix"]),
                                       done[(cid, q, a)], np.array(ladder[b]["matrix"]),
                                       done[(cid, q, b)], label)
                rows.append([cid, f"{q:+d}", f"{ladder[a]['tag']} -> {ladder[b]['tag']}",
                             f"{ladder[a]['L']:.2f} -> {ladder[b]['L']:.2f}", diffs[q],
                             basins[q][1]])
            if lost:
                log(f"WARNING: D0: {cid} q " + ", ".join(f"{q:+d}" for q in lost)
                    + f": a run in {ladder[a]['tag']} or {ladder[b]['tag']} failed, so it is "
                    "left out of the comparison")
            if not diffs:                    # the biggest size whose runs all worked, if any
                good = [i for i in x["sizes"] if all((cid, q, i) in done for q in ground[cid])]
                bad = [ladder[i]["tag"] for i in (a, b) if i not in good]
                x.update(cell=ladder[max(good) if good else idx]["tag"],
                         why="NOT CONVERGED: runs in " + " and ".join(bad) + " failed")
                continue
            ok = all(abs(v) < p.s.series_tol for v in diffs.values()) and \
                all(s for s, _ in basins.values())
            if ok:
                x.update(cell=ladder[a]["tag"], converged=True,
                         why=f"agrees with {ladder[b]['tag']} within series_tol")
            elif b + 1 < len(ladder):
                x["sizes"].append(b + 1)
                x["src"][b + 1] = b
                more = True
            else:
                x.update(cell=ladder[b]["tag"], why="NOT CONVERGED: the largest size of the ladder "
                         f"still differs from {ladder[a]['tag']}"
                         + (" (a different structure)" if not all(s for s, _ in basins.values())
                            else ""))
        if not more:
            break
    for cid, x in sorted(plan.items()):
        log(f"D0: {cid} {cands[cid]['name']}: {x['cell']} ({x['why']})")
    table = format_table(["id", "q", "sizes", "L/A", "change/eV", "basin/A"], rows,
                         f"Each defect's E_D - E_host + E_corr from one size to the next "
                         f"(series_tol {p.s.series_tol:g} eV; basin: the largest difference in "
                         f"how the atoms near it moved, at most {BASIN_TOL:g} A)") + "\n" + \
        format_table(["id", "defect", "cell", "L/A", "atoms", "converged", "why"],
                     [[cid, cands[cid]["name"], x["cell"],
                       next(c["L"] for c in ladder if c["tag"] == x["cell"]),
                       next(c["natoms"] for c in ladder if c["tag"] == x["cell"]),
                       "yes" if x["converged"] else "no", x["why"]]
                      for cid, x in sorted(plan.items())], "The production cell of each defect")
    bad = sorted(cid for cid, x in plan.items() if not x["converged"])
    cells = {c["tag"]: dict(c, host_energy=hosts.get(c["tag"])) for c in ladder}
    return {"defects": {cid: {"cell": x["cell"], "converged": x["converged"], "why": x["why"],
                              "sizes": [ladder[i]["tag"] for i in x["sizes"]]}
                        for cid, x in plan.items()},
            "cells": cells, "runs": runs, "_table": table,
            "_note": f"{len(plan) - len(bad)} of {len(plan)} defects converged"
                     + (f" (not: {', '.join(bad)})" if bad else "")}


# D1: the production relaxations, each in its defect's production cell (D0). Every structure phase
# C passes on (C7's) is relaxed at the production settings: the ground state of each charge state
# is D0's own relaxation in that cell; any other structure C6 kept (keep_structures > 1) is
# carried into the cell (carry_over) and relaxed here (production_job), in
# D_production/jobs/production (named with the cell, so a cell D0 changes its mind about gets new
# jobs). The pristine energy of each cell is D0's single point. Runner group
# 'production' (restarts on; a job that fails is flagged and left out of E). The relaxed
# structures go to D_production/structures/.
#   Called by: main() (step D1)
#   Calls:     Project.batch(), Project.host(), Project.kspacing(), Project.production_param(),
#              build_defect(), carry_over(), format_table(), load_cell(), production_job(),
#              production_summary(), qtag()
def step_production(p):
    d = p.phase_dir("D")
    todo = [dict(t) for t in p.out("excite")["structures"]]
    cells = p.out("cells")
    cands, prim = p.out("generate")["candidates"], p.host()
    P0 = np.array(p.out("choose")["matrix"])
    jobs, frozen = [], {}
    for t in todo:
        tag = cells["defects"][t["cand"]]["cell"]
        t["cell"] = tag
        if t["k"] == 0:
            t["job"] = f"{t['cand']}_{qtag(t['q'])}_{tag}"
            continue
        comps, label = cands[t["cand"]]["comps"], f"{t['cand']} ({cands[t['cand']]['name']})"
        Q = np.array(cells["cells"][tag]["matrix"])
        atoms = load_cell(p, t["structure"])
        if np.allclose(Q, P0):
            info = build_defect(prim, Q, comps, label)[1]
        else:
            atoms, info = carry_over(prim, comps, P0, atoms, Q, label=label)
        name = f"{t['cand']}_{qtag(t['q'])}_{tag}_k{t['k']}"
        job, frozen[name] = production_job(p, name, atoms, info, t["q"], t["spin"])
        jobs.append(job)
        t["job"] = name
    res = None
    if jobs:
        res = p.batch("D_production/jobs/production", jobs, "production", p.production_param(),
                      spacing=p.kspacing())
        if not res["complete"]:
            return None
    out, failed = [], []
    for t in todo:
        if t["k"] == 0:
            r = dict(t, **cells["runs"][t["job"]])
        else:
            r = dict(t, **production_summary(p, res, t["job"], d / "structures" / "production",
                                             frozen[t["job"]])[0])
        r["host_energy"] = cells["cells"][t["cell"]]["host_energy"]
        if r["energy"] is None:
            failed.append(f"{t['job']} ({(r['reason'] or 'no converged energy').splitlines()[0]})")
        out.append(r)
    for f in failed:
        log(f"WARNING: production run {f} - left out of the formation energies; fix it and "
            "rerun with --resubmit-failed")
    rows = [[o["job"], f"{o['q']:+d}", o["cell"], o["energy"], o["spin_final"], o["steps"],
             o["restarts"], o["status"] + (f": {o['reason'].splitlines()[0]}" if o["reason"]
                                           else "")]
            for o in out]
    used = sorted({o["cell"] for o in out})
    table = format_table(["run", "q", "cell", "E/eV", "final spin", "geometry steps", "restarts",
                          "status"], rows, "Production relaxations; pristine cells: "
                         + ", ".join(f"{t} E_host = {cells['cells'][t]['host_energy']:.6f} eV"
                                     for t in used))
    return {"host_energy": {t: cells["cells"][t]["host_energy"] for t in used}, "runs": out,
            "_table": table,
            "_note": f"{len(out) - len(failed)} of {len(out)} relaxations done"
                     + (f", {len(failed)} failed" if failed else "")}


# D2 (optional, finetune : true): MACE fine-tuning data for phase F from the production
# relaxations (D0's and D1's). Every geometry step with energy and forces of each relaxation
# (read_trajectory, all runs of the job; steps with a force above 2 eV/A left out) goes into
# D_production/finetune/train.xyz (REF_energy, REF_forces) if the defect is neutral, or
# charged.xyz (with its total_charge) if not: a MACE model that doesn't take the charge can't use
# those. finetune.sh runs mace_run_train from mace_model on train.xyz (finetune_args adds
# options); run it yourself (a GPU helps).
#   Called by: main() (step D2)
#   Calls:     Project.model_path(), read_trajectory()
def step_finetune(p):
    if not p.s.finetune:
        return {"skipped": True, "_note": "finetune : false"}
    import ase.io
    d = p.phase_dir("D") / "finetune"
    d.mkdir(parents=True, exist_ok=True)
    runs = list(p.out("cells")["runs"].values()) + [
        r for r in p.out("production")["runs"] if r["k"] > 0]
    found = {"neutral": [], "charged": []}
    for r in runs:
        sub = "cells" if r["job"] in p.out("cells")["runs"] else "production"
        f = p.dir / "D_production" / "jobs" / sub / r["job"] / f"{r['job']}.castep"
        if not f.exists():
            continue
        for a, e, forces in read_trajectory(f.read_text(errors="replace")):
            if np.abs(forces).max() > 2.0:                 # far from the minimum: leave out
                continue
            a.info["REF_energy"] = e
            a.info["total_charge"] = r["q"]
            a.arrays["REF_forces"] = forces
            found["charged" if r["q"] else "neutral"].append(a)
    for kind, name in (("neutral", "train.xyz"), ("charged", "charged.xyz")):
        if found[kind]:
            ase.io.write(str(d / name), found[kind], format="extxyz")
    if not found["neutral"]:
        log("WARNING: D2: no geometry steps with forces in the neutral production relaxations; "
            "nothing to fine-tune on")
    script = ("#!/bin/bash\n# Fine-tune the MACE model on the production relaxations' CASTEP data "
              "(written by dwf.py, step D2)\n"
              f"mace_run_train --name=dwf_finetuned --foundation_model={p.model_path()} "
              "--multiheads_finetuning=False --train_file=train.xyz --valid_fraction=0.1 "
              "--energy_key=REF_energy --forces_key=REF_forces --E0s=average --loss=weighted "
              "--energy_weight=1 --forces_weight=10 --max_num_epochs=50 --batch_size=2 "
              f"--default_dtype=float64 --device={p.s.mace_device} {p.s.finetune_args}\n"
              "# charged.xyz has the charged defects' steps (info total_charge), for a model "
              "that takes the charge\n")
    (d / "finetune.sh").write_text(script)
    (d / "finetune.sh").chmod(0o755)
    n, m = len(found["neutral"]), len(found["charged"])
    return {"images": n, "charged_images": m,
            "train": str((d / "train.xyz").relative_to(p.dir)) if n else None,
            "script": str((d / "finetune.sh").relative_to(p.dir)),
            "_note": f"{n} neutral structures in train.xyz, {m} charged in charged.xyz: to use "
                     "them, run finetune.sh (for phase F)"}


# ==========================================================================================
# PHASE E: THERMODYNAMICS (flowchart E1-E3)
# ==========================================================================================
KB = 8.617333262e-5                # eV/K
# chart colours (reference palette: light surface, categorical slots in their fixed order)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


# The production results (D1) of the runs that finished, by candidate and charge state, keeping
# the lowest energy of each (the run it came from, with its cell and that cell's pristine energy,
# host_energy).
#   Called by: step_corrections(), step_formation()
def best_production(p):
    best = {}
    for r in p.out("production")["runs"]:
        if r["energy"] is None:
            continue
        cur = best.setdefault(r["cand"], {}).get(r["q"])
        if cur is None or r["energy"] < cur["energy"]:
            best[r["cand"]][r["q"]] = r
    return best


# E1: the finite-size correction (charge_correction, with the static dielectric constant from
# A2.3) for every charge state that reached phase D, in its defect's own production cell (D0).
# Returns {defect: {q: E_corr}}.
#   Called by: main() (step E1)
#   Calls:     Project.host(), best_production(), correction(), format_table()
def step_corrections(p):
    best = best_production(p)
    prim, cells = p.host(), p.out("cells")["cells"]
    eps = p.out("dielectric").get("eps0")
    corr, rows = {}, []
    for cid, st in sorted(best.items()):
        tag = next(iter(st.values()))["cell"]
        cell = np.array(cells[tag]["matrix"]) @ np.asarray(prim.cell)
        corr[cid] = {str(q): correction(q, cell, eps, p.s.charge_correction) for q in sorted(st)}
        rows += [[cid, tag, cells[tag]["L"], f"{q:+d}", corr[cid][str(q)]] for q in sorted(st)]
    table = format_table(["id", "cell", "L/A", "q", "E_corr/eV"], rows,
                         f"{p.s.charge_correction} corrections (epsilon {eps}), each in its "
                         "defect's production cell (D0)")
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
#   Called by: step_formation()
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


# E2: formation energies, transition levels and (with fermi_temperature) the Fermi level.
# For each defect that reached phase D, its lowest energy in each charge state gives
# E_f(q, E_F=0) = E_D - E_host - sum n_i mu_i + q E_VBM + E_corr at every limit (E_host and E_corr
# for the defect's own production cell, D0; the table flags a cell D0 could not converge), and its
# thermodynamic transition levels in the gap (transition_levels; the same at every limit); the
# plot goes to E_thermo/formation.png. With fermi_temperature, fermi_level finds
# the self-consistent Fermi level at each limit from these defects and the host's free carriers
# (quench_temperature: concentrations frozen at fermi_temperature, then charges redistributed).
#   Called by: main() (step E2)
#   Calls:     Project.host(), band_gap(), best_production(), fermi_level(), format_table(),
#              mu_limits(), plot(), read_bands(), transition_levels()
def step_formation(p):
    d = p.phase_dir("E")
    best = best_production(p)
    if not best:
        die("no production run finished, so there are no formation energies")
    cands, d0 = p.out("generate")["candidates"], p.out("cells")["defects"]
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
                              best[cid][q]["job"], best[cid][q]["cell"]
                              + ("" if d0[cid]["converged"] else " (not converged)")])
    for cid in sorted(best):
        lv = results[cid]["levels"]
        levels_rows.append([cid, cands[cid]["name"],
                            ", ".join(f"({a:+d}/{b:+d}) {e:.3f}" for a, b, e in lv) or
                            "one charge state across the gap"])
    tables = [format_table(["limit", "id", "defect", "q", "E_f(VBM)/eV", "E_f(CBM)/eV",
                            "from run", "cell (D0)"], table,
                           f"Formation energies (E_VBM {vbm:.4f} eV, gap {gap:.3f} eV)"),
              format_table(["id", "defect", "transition levels (eV above VBM)"], levels_rows,
                           "Thermodynamic transition levels in the gap")]
    png = plot(d / "formation.png", curves, gap, f"{p.name}: formation energies")
    fermi = {}
    if p.s.fermi_temperature:
        bands = read_bands(p.dir / p.out("bands")["bands_file"])
        prim = p.host()
        vol = prim.get_volume() * 1e-24
        shift = gap - p.out("bands")["gap"]
        for lim in lims:
            defects = {}
            for cid in best:
                c = cands[cid]
                # a pair of two identical components is counted from both ends: halve it
                same = len(c["comps"]) == 2 and all(c["comps"][0].get(k) == c["comps"][1].get(k)
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
            if p.s.quench_temperature:
                totals = {cid: sum(cc.values()) for cid, cc in conc.items()}
                xq, nq, hq, cq = fermi_level(defects, bands, vbm, gap, p.s.quench_temperature,
                                             shift, vol, totals)
                out["quenched"] = {"T": p.s.quench_temperature, "E_F": xq, "electrons": nq,
                                   "holes": hq,
                                   "concentrations": {cid: {str(q): v for q, v in cc.items()}
                                                      for cid, cc in cq.items()}}
            fermi[lim] = out
        rows = []
        for lim, f in fermi.items():
            rows.append([lim, f"{f['T']:g} K", f["E_F"], f"{f['electrons']:.2e}",
                         f"{f['holes']:.2e}"])
            if "quenched" in f:
                qf = f["quenched"]
                rows.append([lim, f"quenched {qf['T']:g} K", qf["E_F"], f"{qf['electrons']:.2e}",
                             f"{qf['holes']:.2e}"])
        tables.append(format_table(["limit", "T", "E_F above VBM/eV", "n/cm^-3", "p/cm^-3"],
                                   rows, "Self-consistent Fermi level (only the defects that "
                                   "reached phase D; others left out may change it)"))
        log("note: the Fermi level includes only the defects that reached phase D; defects "
            "left out (e.g. native defects dropped in C) may change it")
    return {"vbm": vbm, "gap": gap, "limits": lims, "defects": results, "fermi": fermi,
            "plot": str(png.relative_to(p.dir)) if png else None, "_table": "\n".join(tables),
            "_note": f"{len(best)} defects at {len(lims)} limit(s)"}


# E3: the decision before phase F. A defect is interesting if one of its transition levels lies
# more than deep_level_margin from both band edges (a deep level), or, with excite : true, if one
# of its charge states passed C7 (an internal transition: its ΔSCF ZPL, or vertical energy, is
# given) or its C7 runs failed (to check by hand). Without excite, only deep levels count. The
# list (for_F in the results) is for phase F (optical lineshapes), which is not scripted yet.
#   Called by: main() (step E3)
#   Calls:     format_table()
def step_interest(p):
    form = p.out("formation")
    gap, m = form["gap"], p.s.deep_level_margin
    c7 = p.out("excite")
    what = "ZPL" if c7.get("relaxed") else "vertical energy"
    picks, rows = [], []
    for cid, dd in sorted(form["defects"].items()):
        why = [f"deep level ({a:+d}/{b:+d}) at {e:.2f} eV" for a, b, e in dd["levels"]
               if m <= e <= gap - m]
        if c7.get("excited"):
            per_q = c7["excited"].get(cid, {})
            ok = sorted((e["energy"], int(q)) for q, e in per_q.items() if e["verdict"] == "pass")
            if ok:
                why.append(f"internal transition, ΔSCF {what} {ok[0][0]:.2f} eV "
                           f"(q {ok[0][1]:+d})")
            if any(e["verdict"] == "unknown" for e in per_q.values()):
                why.append("C7 excited state unknown (its runs failed): check by hand")
        rows.append([cid, dd["name"], "yes: " + "; ".join(why) if why else "no"])
        if why:
            picks.append({"cand": cid, "name": dd["name"], "why": why})
    table = format_table(["id", "defect", "on to F?"], rows,
                         f"Deep levels (>= {m:g} eV from both edges)"
                         + (" or internal transitions (C7)" if c7.get("excited") else
                            " (internal transitions need excite : true)"))
    return {"for_F": picks, "_table": table,
            "_note": f"{len(picks)} defect(s) for the lineshape phase (F, not scripted yet)"}


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
         keys=("mace", "scan", "scan_lengths", "scan_tol", "scan_edge_tol", "mace_model",
               "mace_fmax", "mace_steps", "transform"), needs=("generate", "relax"),
         owns=("B_candidates/mace", "B_candidates/jobs/mace_scan", "B_candidates/ladder.json")),
    Step("choose", "B3", "the screening cell", step_choose,
         keys=("supercell", "transform", "screen_min_image", "deep_level_margin", "scan_lengths"),
         needs=("scan", "relax", "bands", "generate")),
    # C: screening and ground states
    Step("prerelax", "C1", "build the candidates and pre-relax them with MACE", step_prerelax,
         keys=("mace", "mace_model", "mace_fmax", "mace_steps", "configuration_margin"),
         needs=("generate", "choose", "relax"),
         owns=("C_screen/structures/candidates", "C_screen/jobs/mace_prerelax")),
    Step("screen", "C2", "cheap CASTEP screening of every candidate", step_screen,
         keys=SCREEN_KEYS + ("screen_charges", "screen_spins"),
         needs=("prerelax", "converge", "bands"),
         owns=("C_screen/jobs/screen", "C_screen/structures/screen")),
    Step("dispersion", "C2b", "the Gamma-only check: level dispersion in two sizes",
         step_dispersion, keys=SCREEN_KEYS + ("dispersion_check", "dispersion_defect",
                                              "dispersion_tol"),
         needs=("screen", "choose", "prerelax", "generate", "relax", "converge"),
         owns=("C_screen/jobs/dispersion",)),
    Step("filter", "C3", "defect hull and property filters", step_filter,
         keys=("hull_margin", "require_gap_states", "require_spin", "require_bound",
               "distort_charges", "charge_correction", "excite"),
         needs=("screen", "dielectric", "prerelax", "dispersion")),
    Step("distort", "C4", "ShakeNBreak distortions of the survivors", step_distort,
         keys=("distortions", "distort_around", "rattle_stdev", "rattle_dmin",
               "oxidation_states"), needs=("filter", "prerelax"),
         owns=("C_screen/structures/distortions",)),
    Step("drelax", "C5", "cheap CASTEP relaxations of the distortions", step_drelax,
         keys=SCREEN_KEYS, needs=("distort",),
         owns=("C_screen/jobs/drelax", "C_screen/structures/drelax")),
    Step("rank", "C6", "rank, merge, share distortions between charge states", step_rank,
         keys=SCREEN_KEYS + ("distortion_min_gain", "share_rounds", "keep_structures"),
         needs=("drelax",), owns=("C_screen/jobs/share_*", "C_screen/structures/share")),
    Step("excite", "C7", "excited states (ΔSCF); the defects that pass and the native ones go on",
         step_excite, keys=SCREEN_KEYS + ("excite", "excite_window", "excite_min", "excite_relax",
                                         "excite_overrides"),
         needs=("rank", "screen", "generate", "prerelax", "relax"),
         owns=("C_screen/jobs/excite_gs", "C_screen/jobs/excite", "C_screen/structures/excite")),
    # D: production cells and relaxations
    Step("cells", "D0", "converge each defect's production cell", step_cells,
         keys=("production_overrides", "freeze_radius", "series_tol", "charge_correction",
               "energy"),
         needs=("excite", "choose", "converge", "relax", "dielectric", "generate"),
         owns=("D_production/jobs/cells", "D_production/structures/cells")),
    Step("production", "D1", "production relaxations, each in its defect's cell", step_production,
         keys=("production_overrides", "freeze_radius", "energy"),
         needs=("cells", "excite", "choose", "converge", "relax", "generate"),
         owns=("D_production/jobs/production", "D_production/structures/production")),
    Step("finetune", "D2", "MACE fine-tuning data (for phase F)", step_finetune,
         keys=("finetune", "finetune_args", "mace_model", "mace_device"),
         needs=("production", "cells"), owns=("D_production/finetune",)),
    # E: thermodynamics
    Step("corrections", "E1", "finite-size corrections", step_corrections,
         keys=("charge_correction",), needs=("production", "dielectric", "cells")),
    Step("formation", "E2", "formation energies, levels, Fermi level", step_formation,
         keys=("gap", "fermi_temperature", "quench_temperature"),
         needs=("corrections", "production", "bands", "limits", "generate", "cells"),
         owns=("E_thermo/formation.png",)),
    Step("interest", "E3", "which defects go on to the lineshape (F)", step_interest,
         keys=("deep_level_margin",), needs=("formation", "excite")),
]
STEP_INDEX = {s.name: s for s in STEPS}


EXAMPLE_PROJECT = """\
# Defect workflow project file    run:  nohup python3 dwf.py <this file> > run.log 2>&1 &
# Written like a CASTEP .param/.cell file: "keyword : value", %block ... %endblock, keywords
# in any case, # or ! starts a comment. Lines starting with ! are optional settings with an
# example value: delete the ! to use one. File names are relative to this file: the example
# inputs (CaO:Sb on ARCHER2) are in example_CaO_Sb/, this file and every file it names.

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
# compute nodes (not ARCHER2's login nodes), each running mace_pack relaxations side by side
# (default cores_per_node / mace_threads, 16 here), in at most mace_max_jobs jobs at a time.
mace_jobscript : archer2_mace.sh   # a Slurm template with a {COMMAND} line
!mace_max_jobs : 4
!mace_python   : /work/e89/e89/<you>/venv/bin/python   # the Python the jobs run (default:
                               #   the one running dwf.py; on ARCHER2 it must be on /work)

# ======================================================================= runner settings
# How the CASTEP jobs run: any supercell_convergence.py runner keyword (see its --example).
jobscript      : archer2.sh    # template with {SEED} and {NODES}/{TASKS_PER_NODE} or {NCORES}
cores_per_node : 128
max_nodes      : 2             # most nodes per job (more or fewer for some steps: below)
!max_walltime  : 12:00:00      # the longest a job may run (default: the template's --time)
%block partitions              # each job goes on the first with enough memory per core
standard   2   --qos=standard
highmem    4   --qos=highmem
%endblock partitions
dryrun_on  : login             # dry runs where dwf.py runs (not ARCHER2's shared serial
dryrun_cmd : castep.serial --dryrun {seed}   #  partition, which rejects whole-node jobs)
!dryrun_pack : 32              # with dryrun_on : compute: up to 32 dry runs in one job
max_jobs  : 16                 # most jobs of one step queued at once
max_restarts : 20              # a relaxation that runs out of time carries on this many times
poll      : 300
delete_check : true            # delete each .check once its calculation has finished
# ... and for some steps only, in a runner block: runner_host for all of phase A's CASTEP
# steps (A1 to A3.1), runner_screen for all of phase C's (C2, C2b, C5, C6, C7) and
# runner_production for phase D's (D0, D1); for one step: runner_converge (A1), runner_relax
# (A2.1), runner_bands (A2.2), runner_dielectric (A2.3), runner_phases (A3.1),
# runner_dispersion (C2b), runner_drelax (C5), runner_rank (C6), runner_excite (C7),
# runner_cells (D0). A step's own block wins over its group's, and that wins over the settings
# above, for the keywords they give (a %block partitions in one replaces the whole list).
# pack : N runs N of a step's jobs side by side on one node, each on 1/N of it: for small
# jobs, as a node is charged whole however few cores a job uses.
# Starting points for CaO:Sb on ARCHER2 - set your own:
%block runner_host             # A1 to A3.1: the primitive cell and the competing phases
max_walltime : 4:00:00
%endblock runner_host
%block runner_converge         # A1: single points of the 2-atom primitive cell
max_nodes    : 1
max_walltime : 0:20:00
pack         : 4               # four at a time on one node, 32 cores each
%block partitions              # ARCHER2's short QoS: at most 20 minutes and 32 nodes
standard   2   --qos=short
%endblock partitions
%endblock runner_converge
%block runner_screen           # C2, C2b, C5, C6, C7: Gamma-only screening jobs are small
max_nodes    : 1
max_restarts : 0               # (the default for screening: a job that fails is flagged)
%endblock runner_screen
%block runner_dispersion       # C2b: two single points with 8 k-points
max_nodes    : 2
%endblock runner_dispersion
%block runner_production       # D0, D1: the production relaxations, up to the largest size
max_nodes    : 8
%endblock runner_production

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
# the MACE scan of every candidate in each. A candidate is converged at the smallest size whose
# MACE relaxation energy is within scan_tol of every bigger size's, with the atoms at the cell's
# edge moved less than scan_edge_tol. The largest length is the most B3 and D0 will use.
scan          : true
scan_lengths  : 8 10 12 14 16 18 20 ang
scan_tol      : 0.02 eV
scan_edge_tol : 0.05 ang
# B3: the screening cell is the smallest size that holds every candidate (B2) and is at least
# the Gamma-only floor, estimated from the band edges' effective masses (A2.2) for the
# shallowest deep level (deep_level_margin, below).
!screen_min_image : 12 ang     # the floor yourself (0: none), instead of the estimate
!supercell : 14 ang            # fix the screening cell (a length, or n: n x transform);
                               #   D0 still converges each defect's production cell

# ======================================================================= C: screening
screen_charges   : -1 0 1
screen_spins     : both        # low | both (also +2 unpaired electrons)
!screen_cutoff   : 500 eV      # a lower cut-off for screening (default: production)
screen_max_steps : 50          # geometry steps per screening relaxation
%block screen_overrides        # any other .param keywords for the screening runs
elec_energy_tol : 1e-5
%endblock screen_overrides
# C2b: the Gamma-only check: one defect (by default the one with a level nearest a band edge)
# on a 2x2x2 k-point grid in the screening cell and the size below
dispersion_check : warn        # warn | stop (before C3) | off
dispersion_tol   : 0.05 eV     # the most a gap level may spread across the k-points
!dispersion_defect : d012      # the defect to check (an id, or id:q)
!configuration_margin : 1.0 eV # C1: drop arrangements this far above the lowest of their
                               # composition by MACE energy (off unless given)
hull_margin   : 0.2 eV         # C3: keep arrangements within this of the defect hull
require_gap_states : none      # none | occupied | empty | both (filled and empty levels)
!require_spin : 2              # e.g. 2 = triplet ground state
require_bound : false          # bound against its isolated components
level_margin  : 0.1 eV
distortions   : -0.6 -0.5 -0.4 -0.3 -0.2 -0.1 0.1 0.2 0.3 0.4 0.5 0.6
distort_charges : stable       # stable (in the gap for that arrangement) | screened
distort_around  : anchor       # anchor (the first component, e.g. the dopant) | vacancy
                               # (the vacant site, for defects that have one)
!%block oxidation_states       # for the number of neighbours to distort
!Ca +2
!O  -2
!Sb +3
!%endblock oxidation_states
distortion_min_gain : 0.05 eV  # a distortion counts if it beats the unperturbed by this
share_rounds    : 2            # C6: try a winning distortion in the other charge states
keep_structures : 1            # per charge state, on to D
# C7: the excited state of each defect's ground state (CASTEP's ΔSCF, at the screening
# settings). Only the defects that pass, and every native defect, go on to D. Test it on one
# defect first: CASTEP's ΔSCF must work on your build (Sb_Ca-V_O, q -1: ZPL about 0.53 eV).
!excite        : true          # C7 on (default off: everything C6 kept goes on)
!excite_window : 0.3 3.5 eV    # pass if the ZPL is in this window (default: any converged one)
excite_min     : 0.4 eV        # the smallest Kohn-Sham step taken as an excitation
excite_relax   : true          # relax the excited state (the ZPL); false: vertical energy
!%block excite_overrides       # any other .param keywords for C7's runs
!deltascf_smearing : 0.01
!%endblock excite_overrides

# ======================================================================= D: production
# D0: each defect's production cell: its ground states relaxed one size below the screening
# cell and in it, then up the sizes of scan_lengths until two neighbouring sizes agree on
# E_D - E_host + E_corr in every charge state; D1 then relaxes everything in that cell.
series_tol : 0.05 eV
%block production_overrides
geom_energy_tol : 5e-6
elec_energy_tol : 5e-7
geom_max_iter   : 100
%endblock production_overrides
freeze_radius : 0              # (untested idea) fix the atoms further than this (Ang) from
                               #   every component of the defect, in D0 and D1
!finetune     : true           # D2: MACE fine-tuning data from D0's and D1's relaxations

# ======================================================================= E: thermodynamics
charge_correction : lz         # lz | mp | none
!gap : 7.09 eV                 # band gap for the Fermi-level range (default: from A2)
!fermi_temperature  : 1000 K   # self-consistent Fermi level at this temperature
!quench_temperature : 300 K    # ... with the defects frozen in, then at this one
deep_level_margin : 0.5 eV     # E3: a level this far from both band edges is "deep" (B3's
                               #   Gamma-only floor is for a level this deep)
"""


if __name__ == "__main__":
    main()
