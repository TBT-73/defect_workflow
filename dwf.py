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
  B  candidates      B1 generate, B2 scan (MACE), B3 series (CASTEP size series),
                     B4 finetune, B5 choose (the supercell)
  C  screening       C1 prerelax (MACE), C2 screen, C3 filter, C3b zpl, C4 distort
                     (ShakeNBreak), C5 drelax, C6 rank
  D  production      D1 production (D2: freeze_radius)
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

All CASTEP jobs go through supercell_convergence.py, which must be in the same directory: the
B3 size series in its usual mode, which builds its own cells, every other step in its batch
mode, which runs the structures this script builds. Its runner settings (jobscript,
partitions, cores_per_node, max_nodes, max_walltime, pack, ...) go in the project file, for
every step; %block runner_host and runner_screen change them for all of phase A's or phase C's
CASTEP steps, and %block runner_<step> for one step (e.g. runner_converge, runner_production).
"""

import argparse
import difflib
import hashlib
import itertools
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ase import Atoms
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
# symmetry_ops, equivalent_sites, electron_parity, spin_states), CASTEP output (read_bands,
# occupied_bands, band_edges, gap_levels, zpl_estimate, read_permittivity, read_trajectory), the
# charge correction (correction), MACE (Project.mace, mace_relax) and tables (format_table).
# Project.out() and Project.phase_dir() are used by almost every step, so they are left out of
# the "Calls" lists, as are log() and die().
# ==========================================================================================

HARTREE = 27.211386245988                     # eV
STATE, RESULTS, SUMMARY = "state.json", "results.json", "summary.txt"
PHASE_DIRS = {"A": "A_host", "B": "B_candidates", "C": "C_screen", "D": "D_production",
              "E": "E_thermo"}
PHASE_TITLES = {"A": "host", "B": "candidate defects and cell size",
                "C": "screening and ground states", "D": "production relaxations",
                "E": "thermodynamics"}


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


_t, _b, _w = sc._text, sc._bool, sc._words
# keyword: (converter, default). Blocks of whole lines ('lines') keep their rows; 'kv' blocks
# hold "keyword : value" lines (.param keywords to set).
PROJECT_KEYWORDS = {
    # general
    "name": (_t, None), "workdir": (_t, None), "host_cell": (_t, None), "param": (_t, None),
    "transform": (sc._ints(9), None), "energy": (sc._choice("auto", "final", "free", "0K"), "auto"),
    "mace": (_b, True), "mace_model": (_t, "medium-mpa-0"), "mace_device": (_t, "cpu"),
    "mace_threads": (sc._num(cast=int), 8), "mace_fmax": (sc._num("eV/ang", "eV/A"), 0.05),
    "mace_steps": (sc._num(cast=int), 500),
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
    # B: candidates and cell size
    "dopants": (_w, []), "defects": ("lines", None), "vacancies": (_b, True),
    "antisites": (_b, False),
    "interstitials": (sc._choice("none", "dopants", "all"), "none"),
    "pairs": (_b, True), "pair_max_distance": (sc._num("ang", "A"), 5.0),
    "native_pairs": (_b, False), "dopant_pairs": (_b, False),
    "scan": (_b, True), "scan_sizes": (sc._ints(), [2, 3, 4, 5]),
    "scan_tol": (sc._num("eV"), 0.02),
    "series_defect": (_t, None), "series_charge": (sc._num(cast=int), 0),
    "series_sizes": (sc._ints(), [2, 3, 4]), "series_tol": (sc._num("eV"), 0.05),
    "series_choice": (sc._choice("smaller", "larger"), "smaller"),
    "supercell": (sc._num(cast=int), None),
    "finetune": (_b, False), "finetune_args": (_t, ""), "finetuned_model": (_t, None),
    # C: screening
    "screen_charges": (sc._ints(), [-1, 0, 1]),
    "screen_spins": (sc._choice("low", "both"), "both"),
    "screen_cutoff": (sc._num("eV"), None), "screen_max_steps": (sc._num(cast=int), 50),
    "screen_overrides": ("kv", {}),
    "triage": (_b, False), "triage_window": (sc._num("eV"), 1.0),
    "hull_margin": (sc._num("eV"), 0.2), "level_margin": (sc._num("eV"), 0.1),
    "require_bound": (_b, False), "require_spin": (sc._num(), None),
    "require_gap_states": (sc._choice("none", "occupied", "empty", "both"), "none"),
    "zpl_window": (_floats(2), None),
    "distortions": (_floats(), [-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4, 0.5,
                                0.6]),
    "distort_charges": (sc._choice("stable", "screened"), "stable"),
    "distort_around": (sc._choice("anchor", "vacancy"), "anchor"),
    "rattle_stdev": (sc._num("ang", "A"), None), "rattle_dmin": (sc._num("ang", "A"), None),
    "oxidation_states": ("lines", None),
    "distortion_min_gain": (sc._num("eV"), 0.05), "share_rounds": (sc._num(cast=int), 2),
    "keep_structures": (sc._num(cast=int), 1),
    # D: production
    "production_overrides": ("kv", {}), "freeze_radius": (sc._num("ang", "A"), 0.0),
    # E: thermodynamics
    "charge_correction": (sc._choice("lz", "mp", "none"), "lz"), "gap": (sc._num("eV"), None),
    "fermi_temperature": (sc._num("K"), None), "quench_temperature": (sc._num("K"), None),
    "deep_level_margin": (sc._num("eV"), 0.5),
}
LIST_KEYS = {"transform", "converge_cutoffs", "dopants", "scan_sizes",
             "series_sizes", "screen_charges", "distortions", "zpl_window"}   # block or keyword
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
# runner_<group>, then by %block runner_<step> (Project.runner_items). series, screen and
# production are both a step and a group: runner_screen is for all three screening steps (C2, C5,
# C6), and runner_drelax and runner_rank change it for C5 or C6 only.
RUNNER_STEPS = {"converge": "host", "relax": "host", "bands": "host", "dielectric": "host",
                "phases": "host", "series": "series", "screen": "screen", "drelax": "screen",
                "rank": "screen", "production": "production"}
RUNNER_GROUPS = ("host", "series", "screen", "production")
RUNNER_BLOCKS = RUNNER_GROUPS + tuple(s for s in RUNNER_STEPS if s not in RUNNER_GROUPS)
# what each group of steps uses unless the project file's runner blocks say otherwise
GROUP_DEFAULTS = {"host": "", "series": "",
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
#   Called by: Project.runner(), Project.runner_items(), read_project(), runner_conv_lines()
def item_key(x):
    return x["name"] if x["type"] == "block" else x["key"]


# Reads the project file into a namespace of settings.
# Every keyword must be a project keyword (PROJECT_KEYWORDS) or a runner keyword (RUNNER_KEYS,
# passed to supercell_convergence.batch_args); %block runner_<group> or runner_<step>
# (RUNNER_BLOCKS) holds runner keywords, and may hold a %block partitions, for a group of steps or
# one step. A typo gets the closest keyword suggested. Returns (settings, the text each setting was
# given as, the runner items, {group or step: runner items}).
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
        if key.startswith("runner_") and key not in known:
            die(f"{name}: there is no {key}: the runner blocks are runner_host (all of phase A's "
                "CASTEP steps), runner_screen (all of phase C's) and, for one step, "
                + ", ".join(f"runner_{s}" for s in RUNNER_STEPS if s != "screen"))
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
            for k in sorted(seen & {"pack", "dryrun_pack"} if key == "runner_series" else ()):
                die(f"{name}: {k} can't be used for the B3 size series (it runs as a size series, "
                    f"not a batch) - take it out of %block runner_series")
            blocks[key[len("runner_"):]] = items
            raw[key] = "(%block)"
            continue
        conv, _ = PROJECT_KEYWORDS[key]
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
    for key in ("pair_max_distance", "scan_tol", "series_tol", "triage_window", "converge_tol",
                "mace_fmax"):
        if getattr(s, key) is not None and getattr(s, key) <= 0:
            die(f"{name}: {key} must be greater than 0")
    s.converge_kgrids = kgrid_list(name, s.converge_kgrid_min, s.converge_kgrid_max,
                                   s.converge_kgrid_step)
    if s.zpl_window and s.zpl_window[0] >= s.zpl_window[1]:
        die(f"{name}: zpl_window is  lowest highest  (eV)")
    return s, raw, runner, blocks


# A project: its settings, files, directories, checkpoints (state.json), results (results.json
# of each phase) and job runner. main() makes one from the project file and the command-line
# switches, and passes it to the steps.
class Project:
    # Reads the project file and the state of earlier runs, and checks the runner settings of
    # every step now (rather than when the step starts, perhaps days later).
    #   Called by: main()
    #   Calls:     Project.runner(), read_project()
    def __init__(self, path, cl):
        self.path, self.base, self.cl = path, path.parent, cl
        self.s, self.raw, self.runner_project, self.runner_blocks = read_project(path)
        self._runners, self._mace, self._model_ids, self._results = {}, {}, {}, {}
        for step in RUNNER_STEPS:
            self.runner(step)
        self.name = self.s.name or path.stem
        self.dir = self.base / (self.s.workdir or f"dwf_{self.name}")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.dir / STATE
        self.state = (json.loads(self.state_file.read_text()) if self.state_file.exists()
                      else {"steps": {}})

    # A file named by a project keyword (relative to the project file); stops if it is missing.
    #   Called by: Project.cell_text(), Project.production_param(), inputs_hash(), step_converge(),
    #              step_dielectric(), step_prerelax(), step_relax()
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
    #   Called by: Project.runner(), runner_conv_lines()
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
    # the runner settings of the step (its name, e.g. 'relax') and the project's STOP file. Returns
    # run_batch's result ({'complete', 'jobs', 'calcs'}).
    #   Called by: run_screen_jobs(), step_bands(), step_converge(), step_dielectric(),
    #              step_phases(), step_production(), step_relax(), step_screen()
    #   Calls:     Project.cell_text(), Project.runner(), sc.run_batch()
    def batch(self, subdir, jobs, step, param_text, cell_text=None, spacing=None):
        return sc.run_batch(self.dir / subdir, jobs, self.runner(step),
                            cell_text or self.cell_text(), param_text, spacing=spacing,
                            energy=self.s.energy, stop_file=self.dir / "STOP")

    # The host .cell as given (species_pot, symmetry_generate and anything else every job copies).
    #   Called by: Project.batch(), Project.kspacing(), step_dielectric(), step_relax()
    #   Calls:     Project.file()
    def cell_text(self):
        return self.file("host_cell").read_text()

    # The .param of the production settings: A1's (with the converged cut-off) once it has run,
    # else the project's .param as it is.
    #   Called by: run_screen_jobs(), step_bands(), step_dielectric(), step_phases(),
    #              step_production(), step_relax(), step_screen(), step_series()
    #   Calls:     Project.file(), Project.finished()
    def production_param(self):
        if self.finished("converge"):
            return (self.dir / self.out("converge")["production_param"]).read_text()
        return self.file("param").read_text()

    # The production k-point spacing (1/Ang): A1's, else kspacing, else from the host .cell's grid.
    #   Called by: step_bands(), step_converge(), step_dielectric(), step_phases(),
    #              step_production(), step_relax(), step_series()
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
    #   Called by: doped_phases(), screen_setup(), step_bands(), step_choose(), step_corrections(),
    #              step_dielectric(), step_formation(), step_generate(), step_limits(),
    #              step_production(), step_scan(), step_series()
    #   Calls:     load_cell()
    def host(self):
        return load_cell(self, self.out("relax")["cell"])

    # The supercell matrix for size n: n x transform (or n x identity).
    #   Called by: step_choose(), step_scan(), step_series()
    def supercell_matrix(self, n):
        T = np.reshape(self.s.transform, (3, 3)) if self.s.transform else np.eye(3, dtype=int)
        return (n * T).astype(int)

    # The MACE calculator for a model (default mace_model: a foundation model's name, or a model
    # file), loaded once (mace-torch, on mace_device).
    #   Called by: mace_relax()
    def mace(self, model=None):
        model = str(model or self.s.mace_model)
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
    #   Called by: mace_relax()
    def model_id(self, model=None):
        model = str(model or self.s.mace_model)
        if model not in self._model_ids:
            path = Path(model)
            self._model_ids[model] = (hashlib.md5(path.read_bytes()).hexdigest()
                                      if path.is_file() else model)
        return self._model_ids[model]


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
# checkpoints; --redo starts steps again from scratch; --example prints an annotated project file.
#   Called by: the main block
#   Calls:     Project.__init__(), Project.finished(), Project.save_results(),
#              Project.save_state(), inputs_hash(), outputs_hash(), print_status(),
#              reset_step(), write_summary()
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
    cl = ap.parse_args()
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
#   Called by: mace_relax()
def atoms_to_dict(atoms):
    return {"symbols": atoms.get_chemical_symbols(),
            "cell": np.asarray(atoms.cell).round(10).tolist(),
            "frac": atoms.get_scaled_positions(wrap=False).round(10).tolist(),
            "magmoms": atoms.get_initial_magnetic_moments().round(6).tolist()}


# The reverse of atoms_to_dict.
#   Called by: mace_relax()
def atoms_from_dict(d):
    a = Atoms(d["symbols"], scaled_positions=d["frac"], cell=d["cell"], pbc=True)
    a.set_initial_magnetic_moments(d.get("magmoms") or [0.0] * len(a))
    return a


# Saves a structure as a CASTEP .cell file (a comment line, lattice_cart, positions_frac with
# SPIN= where a starting spin is set), the atoms in their own order so that reading it back
# (load_cell) gives the same atom numbers. Only the structure: each job adds its k-points,
# species_pot and the rest. Returns the path relative to the work directory, as the results give
# it.
#   Called by: job_summary(), step_distort(), step_prerelax(), step_production()
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
#   Called by: Project.host(), distinct_states(), duplicates(), run_screen_jobs(),
#              step_distort(), step_production(), step_rank(), step_screen()
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
#   Called by: spin_states(), start_spin(), step_series()
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
#   Called by: candidate_points(), equivalent_sites(), site_rotations()
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
#   Called by: step_prerelax(), step_scan(), step_series()
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
#   Called by: step_choose(), step_corrections(), step_distort(), step_filter(),
#              step_production(), step_screen()
#   Calls:     sc.build_supercell()
def host_supercell(prim, P):
    atoms = sc.build_supercell(prim, P)
    del atoms.arrays["prim_index"]
    return atoms


# --------------------------------------------------------------------------
# CASTEP output
# --------------------------------------------------------------------------
# Reads a CASTEP .bands file: k-points, weights, eigenvalues (eV, [spin][k-point][band]), the
# electrons in each spin channel (one number, the total, when the calculation is not spin-
# polarised) and the Fermi energies (eV).
#   Called by: job_summary(), step_bands(), step_formation(), step_screen()
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
#   Called by: band_edges(), carriers(), gap_levels()
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


# A first estimate of the lowest internal transition (the zero-phonon line, eV) from the gap levels
# (gap_levels): the smallest step from an occupied to an empty level in the same spin channel.
# None if no spin channel has both.
#   Called by: job_summary(), step_filter()
def zpl_estimate(levels):
    best = None
    for s in (0, 1):
        occ = [x["e"] for x in levels if x["spin"] == s and x["occupied"]]
        emp = [x["e"] for x in levels if x["spin"] == s and not x["occupied"]]
        if occ and emp and min(emp) > max(occ):
            d = min(emp) - max(occ)
            best = d if best is None else min(best, d)
    return best


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
#   Called by: relative_formation(), step_corrections()
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
# Relaxes the positions of a structure with MACE (the cell fixed) with model (default
# mace_model), unless cache (a JSON file) already holds the result for the same structure, model
# and settings. Returns (relaxed Atoms, energy before, energy after, steps). The starting spins are
# kept (MACE ignores them).
#   Called by: step_prerelax(), step_scan()
#   Calls:     Project.mace(), Project.model_id(), atoms_from_dict(), atoms_to_dict()
def mace_relax(p, atoms, cache, model=None):
    key = hashlib.md5(json.dumps({"atoms": atoms_to_dict(atoms), "model": p.model_id(model),
                                  "fmax": p.s.mace_fmax, "steps": p.s.mace_steps},
                                 sort_keys=True).encode()).hexdigest()
    cache = Path(cache)
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved.get("key") == key:
            return atoms_from_dict(saved["atoms"]), saved["e0"], saved["e1"], saved["steps"]
    from ase.optimize import LBFGS
    work = atoms.copy()
    work.calc = p.mace(model)
    e0 = float(work.get_potential_energy())
    opt = LBFGS(work, logfile=None)
    opt.run(fmax=p.s.mace_fmax, steps=p.s.mace_steps)
    e1 = float(work.get_potential_energy())
    work.calc = None
    work.set_initial_magnetic_moments(atoms.get_initial_magnetic_moments())
    steps = int(getattr(opt, "nsteps", 0))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"key": key, "atoms": atoms_to_dict(work), "e0": e0, "e1": e1,
                                 "steps": steps}))
    return work, e0, e1, steps


# --------------------------------------------------------------------------
# Odds and ends
# --------------------------------------------------------------------------
# Writes a JSON file, made readable, through a temporary file so that it is never left half
# written (state.json, results.json).
#   Called by: Project.save_results(), Project.save_state()
def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=float))
    tmp.replace(path)


# A text table for summary.txt and the log: a title, a header row, a rule and the rows, in columns
# as wide as needed (numbers to the right, text to the left). Floats are shown with 4 decimals,
# None as blank.
#   Called by: problem_table(), step_converge(), step_corrections(), step_filter(),
#              step_formation(), step_generate(), step_interest(), step_limits(), step_phases(),
#              step_prerelax(), step_production(), step_rank(), step_scan(), step_series(),
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
#   Called by: step_limits(), step_phases()
def counts(atoms):
    syms = atoms.get_chemical_symbols()
    return {s: syms.count(s) for s in sorted(set(syms))}


# The tag of a charge state in job names (which can't hold + or -): q0, qp1, qm1, ...
#   Called by: step_distort(), step_production(), step_rank(), step_screen()
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
# the rest, so the size series can use it too), and the energy per atom, e_bulk.
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
# PHASE B: CANDIDATE DEFECTS AND CELL SIZE (flowchart B1-B5)
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
#   Called by: representative(), user_defects()
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
# the size series). q=... on a line (q=-1, or q=+1,0) gives the charge states of that line's
# candidates, used instead of screen_charges.
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
# B2-B5: cell size
# --------------------------------------------------------------------------
# The defect the size scan and series use: series_defect (tokens, e.g. Sb_Ca@1 + Vac_O) or the
# nearest arrangement of the first pair with a dopant (else the first candidate). Returns
# (component dicts for build_defect, tokens for supercell_convergence.py, its name, the charge
# states of its line in %block defects or None). A token names
# the primitive-cell site (:k) its component sits on when its element has more than one, so that
# supercell_convergence.py builds the same arrangement.
#   Called by: step_scan(), step_series()
#   Calls:     parse_token(), sc.defect_vectors(), sc.parse_defect()
def representative(p, prim):
    if p.s.series_defect:
        toks = p.s.series_defect.replace("+", " ").split()
        comps = sc.parse_defect(toks, prim)
        vecs = sc.defect_vectors(prim, comps)
        full = [parse_token(toks[0], prim)] + [
            dict(parse_token(t, prim), vec=np.asarray(v).tolist()) for t, v in zip(toks[1:], vecs)]
        return full, toks, " + ".join(toks), None
    cands = list(p.out("generate")["candidates"].values())
    pairs = [c for c in cands if len(c["comps"]) == 2 and any(x["new"] in p.s.dopants
                                                              for x in c["comps"])]
    c = min(pairs, key=lambda c: c["separation"]) if pairs else cands[0]
    if any(x["host"] is None for x in c["comps"]):
        die(f"the representative defect {c['name']} has an interstitial, which the size series "
            "can't build: set series_defect (e.g. Sb_Ca + Vac_O) or supercell")
    prim_syms, frac, cell = prim.get_chemical_symbols(), prim.get_scaled_positions(), prim.cell
    ra = prim.positions[c["comps"][0]["prim_site"]]
    toks = []
    for x in c["comps"]:
        site = x["prim_site"]
        if "vec" in x:                              # the site its vector lands on
            f = np.linalg.solve(np.asarray(cell).T, ra + np.asarray(x["vec"]))
            d = (frac - f + 0.5) % 1.0 - 0.5
            site = int(np.argmin(np.linalg.norm(d @ np.asarray(cell), axis=1)))
        same = [i for i, s in enumerate(prim_syms) if s == x["host"]]
        tok = f"{x['new'] or 'Vac'}_{x['host']}"
        if len(same) > 1:
            tok += f":{same.index(site) + 1}"
        toks.append(tok)
    return c["comps"], toks, c["name"], c.get("charges")


# B2: MACE size scan. Builds the representative defect in each of scan_sizes, relaxes it with
# MACE (positions only) and records its relaxation energy (unrelaxed - relaxed) and the largest
# displacement of the atoms in the outer part of the cell (beyond 35% of the shortest image
# distance from the defect). The CASTEP series goes no further than the first size whose
# relaxation energy is within scan_tol of the size before (the cap). Skipped with mace : false,
# scan : false or supercell (the cap is only for the series).
#   Called by: main() (step B2)
#   Calls:     Project.host(), Project.supercell_matrix(), build_defect(), format_table(),
#              mace_relax(), representative(), sc.displacements(), sc.get_distances(),
#              sc.min_image_distance()
def step_scan(p):
    if not (p.s.mace and p.s.scan) or p.s.supercell:
        return {"cap": None, "_note": "skipped (" + ("supercell given" if p.s.supercell else
                                                    "mace or scan is false") + ")"}
    d = p.phase_dir("B")
    prim = p.host()
    comps, _, name, _ = representative(p, prim)
    rows, out, prev, cap = [], [], None, None
    for n in sorted(p.s.scan_sizes):
        P = p.supercell_matrix(n)
        atoms, info = build_defect(prim, P, comps, name)
        L = sc.min_image_distance(atoms.cell)
        log(f"B2: {name} in size {n} ({len(atoms)} atoms): MACE relaxation")
        relaxed, e0, e1, steps = mace_relax(p, atoms, d / "mace" / f"n{n}.json")
        moved = sc.displacements(atoms, relaxed)
        _, r = sc.get_distances(np.array(info["centre"])[None], atoms.positions, cell=atoms.cell,
                                pbc=True)
        far = moved[r[0] > 0.35 * L]
        erel = e0 - e1
        row = {"n": n, "atoms": len(atoms), "L_min": L, "E_relax": erel, "steps": steps,
               "far_move": float(far.max()) if far.size else 0.0}
        row["change"] = None if prev is None else erel - prev
        out.append(row)
        rows.append([n, len(atoms), L, erel, row["change"], row["far_move"], steps])
        if prev is not None and abs(erel - prev) < p.s.scan_tol and cap is None:
            cap = n
            break
        prev = erel
    table = format_table(["n", "atoms", "L_min/A", "E_relax/eV", "change/eV",
                          "far-field move/A", "MACE steps"], rows,
                         f"MACE size scan of {name} (scan_tol {p.s.scan_tol:g} eV)")
    if cap is None:
        log(f"note: the MACE relaxation energy has not levelled off within scan_sizes "
            f"{p.s.scan_sizes}; the CASTEP series is not capped")
    return {"cap": cap, "sizes": out, "_table": table,
            "_note": f"levels off at size {cap}" if cap else "did not level off"}


# The runner settings of a step, as lines of a .conv file: its runner items (the project's runner
# keywords, changed by its runner blocks), with file names made absolute (the .conv is elsewhere).
# Batch-only keywords (max_jobs, tolerate_failures, pack, dryrun_pack) are left out.
#   Called by: step_series()
#   Calls:     Project.runner_items(), item_key()
def runner_conv_lines(p, step):
    lines = []
    for it in p.runner_items(step):
        k = item_key(it)
        if k in ("max_jobs", "tolerate_failures", "pack", "dryrun_pack"):
            continue
        if it["type"] == "block":
            lines += [f"%block {k}"] + it["body"] + [f"%endblock {k}"]
        elif k in ("jobscript", "dryrun_jobscript"):
            lines.append(f"{k} : {(p.base / it['value']).resolve()}")
        else:
            lines.append(f"{k} : {it['value']}")
    return lines


# B3: the CASTEP size series of the representative defect, run by supercell_convergence.py in
# B_candidates/series: its .conv is written from the project (relaxed host cell, production
# .param and k-points, e_bulk, transform, series_sizes up to the B2 cap, series_tol, MACE
# pre-relaxation with mace : true, the charge with its correction: series_charge if given, else
# the defect's own q= in %block defects (0 if it allows it), else 0), then the script runs in
# a subprocess (output to this terminal) and its convergence.json is read. A charged or
# odd-electron defect runs spin-polarised with a starting spin (the lowest its electron count
# allows, also put on the first component); otherwise the .param's own spin_polarized is kept.
# Not finished while jobs are queued (--no-wait).
#   Called by: main() (step B3)
#   Calls:     Project.host(), Project.kspacing(), Project.production_param(),
#              Project.supercell_matrix(), build_defect(), electron_parity(), format_table(),
#              representative(), runner_conv_lines(), sc.strip_comment()
def step_series(p):
    if p.s.supercell:
        return {"skipped": True, "_note": f"supercell : {p.s.supercell} given"}
    d = p.phase_dir("B") / "series"
    d.mkdir(parents=True, exist_ok=True)
    prim = p.host()
    comps, toks, name, qs = representative(p, prim)
    cap = p.out("scan").get("cap")
    sizes = [n for n in sorted(p.s.series_sizes) if cap is None or n <= cap]
    if len(sizes) < 2:
        sizes = sorted(p.s.series_sizes)[:2]
    q = p.s.series_charge
    if qs and "series_charge" not in p.raw:      # its own q= in %block defects: 0 if allowed
        q = 0 if 0 in qs else qs[0]
        log(f"B3: series in charge state {q:+d} (from q= in %block defects)")
    spin = electron_parity(build_defect(prim, p.supercell_matrix(sizes[0]), comps, name)[0], q)
    drop = (r"(charge|spin|spin_polari[sz]ed)" if spin or q       # set below (either spelling)
            else r"(charge|spin)")
    param = "\n".join(line for line in p.production_param().splitlines()
                      if not sc.re.match(rf"\s*{drop}\b", sc.strip_comment(line),
                                         sc.re.I)).rstrip() + "\n"
    if q:
        param += f"charge : {q}\n"
    if spin or q:
        param += f"spin_polarized : true\nspin : {spin}\n"
    (d / "series.param").write_text(param)
    host = p.out("relax")
    toks = [toks[0] + (f"@{spin}" if spin and "@" not in toks[0] and not
                       toks[0].lower().startswith("vac") else "")] + toks[1:]
    conv = [f"# size series of {name} (written by dwf.py, step B3)",
            f"cell : {(p.dir / host['cell']).resolve()}", "param : series.param",
            "%block defect"] + toks + ["%endblock defect",
                                       f"e_bulk : {host['e_bulk']:.10f} eV",
                                       f"sizes : {' '.join(map(str, sizes))}",
                                       f"kspacing : {p.kspacing():.8f}",
                                       f"tol : {p.s.series_tol:g} eV", "singlepoint : false",
                                       f"energy : {p.s.energy}", "workdir : run"]
    if p.s.transform:
        conv += ["%block transform"] + [" ".join(map(str, p.s.transform[i:i + 3]))
                                         for i in (0, 3, 6)] + ["%endblock transform"]
    if q:
        eps = p.out("dielectric")["eps0"]
        conv += [f"charge_correction : {p.s.charge_correction}", f"dielectric : {eps:.6f}"]
    if p.s.mace:
        conv += ["prerelax : mace", f"prerelax_model : {p.s.mace_model}",
                 f"prerelax_device : {p.s.mace_device}", f"prerelax_threads : {p.s.mace_threads}",
                 f"prerelax_fmax : {p.s.mace_fmax:g}", f"prerelax_steps : {p.s.mace_steps}"]
    conv += runner_conv_lines(p, "series")
    (d / "series.conv").write_text("\n".join(conv) + "\n")
    cmd = [sys.executable, str(HERE / "supercell_convergence.py"), "series.conv"]
    for flag in ("no_wait", "prepare_only", "force", "resubmit_failed"):
        if getattr(p.cl, flag, False):
            cmd.append("--" + flag.replace("_", "-"))
    log(f"B3: size series of {name}, sizes {sizes}, in {d}")
    sys.stdout.flush()
    ret = subprocess.run(cmd, cwd=d)
    if ret.returncode != 0:
        die(f"the size series stopped with an error (above); fix it and rerun ({d})")
    res = d / "run" / "convergence.json"
    if not res.exists():
        return None
    rows = json.loads(res.read_text())
    ns = [int(sc.re.findall(r"\d+", r["size"])[0]) for r in rows]
    converged = len(rows) > 1 and abs(rows[-1]["dE_f"]) < p.s.series_tol
    if not converged and ns[-1] < sizes[-1]:
        return None                                   # more sizes still to come
    series = [{"n": n, "atoms": r["atoms"], "E_f": r["E_f"], "dE_f": r.get("dE_f")}
              for n, r in zip(ns, rows)]
    table = format_table(["n", "atoms", "E_f*/eV", "change/eV"],
                         [[x["n"], x["atoms"], x["E_f"], x["dE_f"]] for x in series],
                         f"CASTEP size series of {name} (series_tol {p.s.series_tol:g} eV; "
                         "the jobs are in series/run/)")
    return {"converged": converged, "series": series,
            "choice": (ns[-2] if p.s.series_choice == "smaller" else ns[-1]) if converged else None,
            "_note": (f"converged: E_f* changed by {rows[-1]['dE_f']:+.4f} eV from size {ns[-2]} "
                      f"to {ns[-1]}" if converged else f"not converged within sizes {ns}"),
            "_table": table}


# B4 (optional): MACE fine-tuning data from the series' smallest-cell relaxation. Every geometry
# step with energy and forces (read_trajectory, all runs of the job; steps with a force above
# 2 eV/A left out) goes into B_candidates/finetune/train.xyz (REF_energy, REF_forces), and
# finetune.sh runs mace_run_train from mace_model on it (finetune_args adds options). Run it
# yourself (a GPU helps), then set finetuned_model to the .model it writes: C1 uses it (B2 and
# B3 keep mace_model, so they don't run again).
#   Called by: main() (step B4)
#   Calls:     read_trajectory()
def step_finetune(p):
    if not p.s.finetune:
        return {"skipped": True, "_note": "finetune : false"}
    import ase.io
    d = p.phase_dir("B") / "finetune"
    d.mkdir(parents=True, exist_ok=True)
    series = p.out("series")
    if series.get("skipped"):
        die("finetune : true needs the size series (B3), which supercell skipped")
    run = p.dir / "B_candidates" / "series" / "run"
    castep = sorted(run.glob("*/sc_*_defect_geom.castep"),
                    key=lambda f: len(f.read_text(errors="replace")))
    if not castep:
        die(f"no relaxation output found under {run}")
    first = min(castep, key=lambda f: int(sc.re.findall(r"\d+", f.parent.name)[0]))
    images = []
    for a, e, f in read_trajectory(first.read_text(errors="replace")):
        if np.abs(f).max() > 2.0:                     # far from the minimum: leave out
            continue
        a.info["REF_energy"] = e
        a.arrays["REF_forces"] = f
        images.append(a)
    if not images:
        log(f"WARNING: no geometry steps with forces in {first}; nothing to fine-tune on")
        return {"images": 0, "_note": "no training data"}
    ase.io.write(str(d / "train.xyz"), images, format="extxyz")
    script = ("#!/bin/bash\n# Fine-tune the MACE foundation model on the size series' CASTEP data "
              "(written by dwf.py, step B4)\n"
              f"mace_run_train --name=dwf_finetuned --foundation_model={p.s.mace_model} "
              "--multiheads_finetuning=False --train_file=train.xyz --valid_fraction=0.1 "
              "--energy_key=REF_energy --forces_key=REF_forces --E0s=average --loss=weighted "
              "--energy_weight=1 --forces_weight=10 --max_num_epochs=50 --batch_size=2 "
              f"--default_dtype=float64 --device={p.s.mace_device} {p.s.finetune_args}\n"
              f"# then set  finetuned_model : {(d / 'dwf_finetuned.model').resolve()}  in the "
              "project file (C1 uses it)\n")
    (d / "finetune.sh").write_text(script)
    (d / "finetune.sh").chmod(0o755)
    return {"images": len(images), "train": str((d / "train.xyz").relative_to(p.dir)),
            "script": str((d / "finetune.sh").relative_to(p.dir)),
            "_note": f"{len(images)} structures in train.xyz: to use them, run finetune.sh and "
                     "set finetuned_model (C1 uses it)"}


# B5: the supercell for C and D: supercell : n if given, else the size series' choice (the
# smaller of the two sizes that agreed within series_tol, or the larger with series_choice :
# larger). Stops if the series did not converge. Every arrangement of one composition uses this
# cell, so their energies can be compared directly.
#   Called by: main() (step B5)
#   Calls:     Project.host(), Project.supercell_matrix(), host_supercell(),
#              sc.min_image_distance()
def step_choose(p):
    if p.s.supercell:
        n, why = p.s.supercell, "supercell in the project file"
    else:
        s = p.out("series")
        if not s.get("converged"):
            die("the size series did not converge within sizes "
                f"{[x['n'] for x in s.get('series', [])]}: add bigger "
                "sizes to series_sizes (or raise series_tol), or choose the cell yourself with "
                "supercell : n")
        n, why = s["choice"], "size series"
    prim = p.host()
    P = p.supercell_matrix(n)
    host = host_supercell(prim, P)
    L = sc.min_image_distance(host.cell)
    return {"n": n, "matrix": P.tolist(), "natoms": len(host), "L_min": L,
            "_note": f"size {n}: {len(host)} atoms, images {L:.2f} A apart ({why})"}


# ==========================================================================================
# PHASE C: SCREENING AND GROUND STATES (flowchart C1-C6)
# ==========================================================================================
# The candidates (B1) and the chosen supercell (B5): (candidates by id, primitive host, matrix).
#   Called by: step_distort(), step_filter(), step_prerelax(), step_screen()
#   Calls:     Project.host()
def screen_setup(p):
    return p.out("generate")["candidates"], p.host(), np.array(p.out("choose")["matrix"])


# The .param keywords of the screening runs: screen_cutoff, screen_max_steps and
# %block screen_overrides.
#   Called by: screen_job(), step_screen()
def screen_overrides(p):
    o = {"geom_max_iter": p.s.screen_max_steps}
    if p.s.screen_cutoff:
        o["cut_off_energy"] = f"{p.s.screen_cutoff:g} eV"
    o.update(p.s.screen_overrides)
    return o


# A screening or distortion job for one structure in charge state q, starting with spin S
# (spin_polarized, spin : S, and SPIN=S on the anchor atom when there is one): a Gamma-point
# relaxation without symmetry, with the screening keywords.
#   Called by: run_screen_jobs(), step_screen()
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
# its geometry converged, final spin, defect levels in the gap and the first ZPL estimate (from its
# .bands against the host's band edges at the same settings), and its final structure, saved as
# <structdir>/<name>.cell.
#   Called by: run_screen_jobs(), step_screen()
#   Calls:     gap_levels(), read_bands(), save_cell(), zpl_estimate(), sc.final_atoms()
def job_summary(p, res, name, edges, structdir):
    r, calc = res["jobs"][name], res["calcs"][name]
    out = {"status": r["status"], "reason": r["reason"], "geom_converged": r["geom_converged"],
           "energy": r["energy"] if r["scf_converged"] else None, "spin": r["spin"],
           "steps": r["geom_steps"], "levels": None, "zpl": None, "structure": None}
    bands = calc.dir / f"{calc.seed}.bands"
    if bands.exists() and edges:
        lv = gap_levels(read_bands(bands), edges[0], edges[1], p.s.level_margin)
        out["levels"], out["zpl"] = lv, zpl_estimate(lv)
    if r["energy"] is not None:
        atoms = sc.final_atoms(calc)
        if atoms is not None:
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
# C1: builds every candidate in the B5 supercell (build_defect) and, with mace : true, relaxes
# it with MACE (positions only, the cell fixed; finetuned_model if given, else mace_model). A pair
# too far apart for this cell (more than half the shortest image distance) is left out. With
# triage : true (and MACE), arrangements more than triage_window above the lowest MACE energy of
# the same composition are dropped. Each structure is saved as
# C_screen/structures/candidates/<id>.cell, and the results give where its defect is (its centre,
# each component's position, the anchor atom's index, the vacancies) and its MACE energy.
#   Called by: main() (step C1)
#   Calls:     Project.file(), build_defect(), format_table(), mace_relax(), save_cell(),
#              screen_setup()
def step_prerelax(p):
    cands, prim, P = screen_setup(p)
    d = p.phase_dir("C")
    L = p.out("choose")["L_min"]
    defects, skipped, energies = {}, {}, {}
    model = p.file("finetuned_model") if p.s.finetuned_model else None
    for cid, c in cands.items():
        if c["separation"] and c["separation"] > L / 2 - 0.05:
            skipped[cid] = f"components {c['separation']:.2f} A apart, too far for this cell"
            continue
        atoms, info = build_defect(prim, P, c["comps"], f"{cid} ({c['name']})")
        e = None
        if p.s.mace:
            log(f"C1: {cid} {c['name']}: MACE pre-relaxation ({len(atoms)} atoms)")
            atoms, e0, e, steps = mace_relax(p, atoms, d / "mace" / f"{cid}.json", model)
        title = f"{cid} {c['name']} in the chosen supercell" + (", MACE pre-relaxed"
                                                                 if p.s.mace else "")
        defects[cid] = dict(structure=save_cell(p, d / "structures" / "candidates" / f"{cid}.cell",
                                                atoms, title), mace_energy=e, **info)
        energies[cid] = e
    if p.s.triage and p.s.mace:
        groups = {}
        for cid in defects:
            groups.setdefault(cands[cid]["group"], []).append(cid)
        for g, ids in groups.items():
            low = min(energies[i] for i in ids)
            for i in ids:
                if energies[i] - low > p.s.triage_window:
                    skipped[i] = (f"MACE energy {energies[i] - low:.2f} eV above the lowest "
                                  f"{g} arrangement (triage)")
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
# energy, final spin and gap levels, and the pristine cell's band edges.
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
    host_out = {"energy": res["jobs"]["host"]["energy"], "vbm": vbm, "cbm": cbm, "gap": cbm - vbm}
    runs = {}
    for name, meta in plan.items():
        runs[name] = dict(meta, **job_summary(p, res, name, (vbm, cbm),
                                              d / "structures" / "screen"))
    bad = sum(r["energy"] is None for r in runs.values())
    return {"host": host_out, "runs": runs, "_table": problem_table(runs),
            "_note": f"{len(runs)} runs, gap {cbm - vbm:.2f} eV at Gamma"
                     + (f", {bad} without a usable energy" if bad else "")}


# --------------------------------------------------------------------------
# C3, C3b
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
# (duplicates) are dropped too. The results list the survivors, with the charge states to carry
# on (distort_charges); the table gives every candidate's verdict.
#   Called by: main() (step C3)
#   Calls:     best_states(), duplicates(), format_table(), host_supercell(),
#              relative_formation(), screen_setup(), zpl_estimate()
def step_filter(p):
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
    survivors, rows = [], []
    for cid in sorted(cands):
        c = cands[cid]
        if cid not in best:
            why = (f"left out in C1: {left_out[cid]}" if cid in left_out
                   else "dropped: no usable screening run")
            rows.append([cid, c["name"], c["group"], "", "", "", "", "", why])
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
                    or (want == "both" and zpl_estimate(lv) is not None))

        # Whether a charge state has the spin asked for.
        def spin_ok(s):
            return (p.s.require_spin is None or
                    (s["spin"] is not None and abs(abs(s["spin"]) - p.s.require_spin) < 0.25))

        good = [q for q, s in props.items() if levels_ok(s) and spin_ok(s)]
        if not good:
            asked = [x for x, on in (("gap levels", p.s.require_gap_states != "none"),
                                     ("spin", p.s.require_spin is not None)) if on]
            why.append("no stable charge state with the " + " and ".join(asked)
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
                why.append("binding energy unknown (its components were not screened)")
            elif bind[1] <= 0:
                why.append(f"not bound (E_b {bind[0]:.2f} to {bind[1]:.2f} eV)")
        charges = stable if p.s.distort_charges == "stable" else sorted(best[cid])
        spins = ", ".join(f"{q:+d}:{best[cid][q]['spin']:.1f}" if best[cid][q]["spin"] is not None
                          else f"{q:+d}:-" for q in stable)
        zpls = [best[cid][q]["zpl"] for q in stable if best[cid][q]["zpl"] is not None]
        rows.append([cid, c["name"], c["group"], dist, " ".join(f"{q:+d}" for q in stable), spins,
                     f"{min(zpls):.2f}" if zpls else "",
                     f"{bind[0]:.2f}..{bind[1]:.2f}" if bind else "",
                     "kept" if not why else "dropped: " + "; ".join(why)])
        if not why:
            survivors.append({"cand": cid, "name": c["name"], "group": c["group"],
                              "hull_distance": dist, "stable": stable, "charges": charges,
                              "good": good, "binding": bind,
                              "best": {str(q): best[cid][q] for q in best[cid]}})
    table = format_table(["id", "defect", "composition", "above hull/eV", "stable q",
                          "spin (q:S)", "ZPL est./eV", "E_b/eV", "verdict"], rows,
                         f"Defect hull (margin {p.s.hull_margin:g} eV) and property filters")
    return {"survivors": survivors, "_table": table,
            "_note": f"{len(survivors)} of {len(cands)} candidates kept"}


# C3b (optional, zpl_window): keeps the survivors with a first ZPL estimate (zpl_estimate, from
# the screening run of a stable charge state that met the property filters) inside zpl_window; one
# with no occupied -> empty step in any spin channel has no internal transition and is dropped.
# Without zpl_window every survivor carries on.
#   Called by: main() (step C3b)
#   Calls:     format_table()
def step_zpl(p):
    survivors = p.out("filter")["survivors"]
    if not p.s.zpl_window:
        return {"survivors": survivors, "_note": "no zpl_window, so no ZPL filter"}
    lo, hi = p.s.zpl_window
    kept, rows = [], []
    for s in survivors:
        est = [s["best"][str(q)]["zpl"] for q in s["good"] if s["best"][str(q)]["zpl"] is not None]
        inside = [e for e in est if lo <= e <= hi]
        shown = ", ".join(f"{e:.2f}" for e in est)
        verdict = ("kept" if inside else "dropped: " + (f"estimates {shown} eV outside" if est
                                                        else "no internal transition"))
        rows.append([s["cand"], s["name"], ", ".join(f"{e:.2f}" for e in est), verdict])
        if inside:
            kept.append(dict(s, zpl=inside))
    table = format_table(["id", "defect", "ZPL estimates/eV", "verdict"], rows,
                         f"First ZPL estimates, window {lo:g}-{hi:g} eV")
    return {"survivors": kept, "_table": table,
            "_note": f"{len(kept)} of {len(survivors)} inside the window"}


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
#   Called by: step_distort(), step_rank()
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
    survivors = p.out("zpl")["survivors"]
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
    survivors = p.out("zpl")["survivors"]
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


SCREEN_KEYS = ("screen_cutoff", "screen_max_steps", "screen_overrides", "level_margin", "energy")


# ==========================================================================================
# PHASE D: PRODUCTION RELAXATIONS (flowchart D1, D2)
# ==========================================================================================
# The %block ionic_constraints that holds still every atom further than radius (Ang) from all of
# the defect's components (points: their positions; three constraints per atom: x, y and z).
# CASTEP numbers the atoms of each species in the order of the .cell, which keeps the order
# within each species (written_order).
#   Called by: step_production()
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


# D1: the production relaxations. One job per structure phase C kept (C6): its charge state,
# starting spin (spin_polarized : true with spin S when S is not 0; a spin-0 state runs without
# spin polarisation unless %block production_overrides sets spin_polarized), the production .param
# with %block production_overrides, the production k-points, fixed cell, no symmetry; with
# freeze_radius, the far atoms fixed (freeze_block). Plus the pristine supercell single point (with
# symmetry). Runner group 'production' (restarts on; a job that fails is flagged and left out of
# E). The relaxed structures go to D_production/structures/.
#   Called by: main() (step D1)
#   Calls:     Project.batch(), Project.host(), Project.kspacing(), Project.production_param(),
#              format_table(), freeze_block(), host_supercell(), load_cell(), qtag(),
#              save_cell(), sc.final_atoms()
def step_production(p):
    d = p.phase_dir("D")
    todo = [dict(t) for t in p.out("rank")["structures"]]
    if not todo:
        die("phase C kept no structures, so there is nothing to relax")
    prim, P = p.host(), np.array(p.out("choose")["matrix"])
    host = host_supercell(prim, P)
    defects = p.out("prerelax")["defects"]
    jobs = [{"name": "host", "atoms": host, "kind": "sp", "symmetry": True,
             "overrides": dict(p.s.production_overrides)}]
    frozen = {}
    for t in todo:
        atoms = load_cell(p, t["structure"])
        info = defects[t["cand"]]
        S = t["spin"]
        mags = np.zeros(len(atoms))
        if S and info.get("anchor_index") is not None:
            mags[info["anchor_index"]] = S
        atoms.set_initial_magnetic_moments(mags)
        o = dict(p.s.production_overrides, charge=t["q"])
        if S:
            o.update(spin_polarized="true", spin=S)
        else:                               # (production_overrides can ask for spin_polarized)
            o.setdefault("spin_polarized", "false")
            o["spin"] = None
        name = f"{t['cand']}_{qtag(t['q'])}_k{t['k']}"
        job = {"name": name, "atoms": atoms, "kind": "geom", "symmetry": False, "overrides": o}
        if p.s.freeze_radius:
            block, n = freeze_block(atoms, info.get("points") or [info["centre"]],
                                    p.s.freeze_radius)
            if block:
                job["cell_extra"] = block
                frozen[name] = n
        jobs.append(job)
        t["job"] = name
    if frozen:
        log(f"D2: freeze_radius {p.s.freeze_radius:g} A holds "
            f"{min(frozen.values())}-{max(frozen.values())} atoms fixed in each relaxation "
            "(untested idea: check it against a full relaxation)")
    res = p.batch("D_production/jobs/production", jobs, "production", p.production_param(),
                  spacing=p.kspacing())
    if not res["complete"]:
        return None
    hr = res["jobs"]["host"]
    if hr["status"] != "done" or hr["energy"] is None:
        die(f"the pristine supercell single point did not finish ({hr['dir']}): the formation "
            "energies need it")
    out, failed = [], []
    for t in todo:
        r = res["jobs"][t["job"]]
        e = r["energy"] if r["scf_converged"] else None
        if r["status"] != "done" or e is None:
            failed.append(f"{t['job']} ({(r['reason'] or 'no converged energy').splitlines()[0]})")
        path = None
        if e is not None:
            atoms = sc.final_atoms(res["calcs"][t["job"]])
            if atoms is not None:
                path = save_cell(p, d / "structures" / f"{t['job']}.cell", atoms,
                                 f"{t['job']}: relaxed at the production settings (D1)")
        out.append(dict(t, energy=e if r["status"] == "done" else None, energy_any=e,
                        spin_final=r["spin"], status=r["status"], reason=r["reason"],
                        steps=r["geom_steps"], restarts=r["restarts"], final=path,
                        frozen=frozen.get(t["job"], 0)))
    for f in failed:
        log(f"WARNING: production run {f} - left out of the formation energies; fix it and "
            "rerun with --resubmit-failed")
    rows = [[o["job"], f"{o['q']:+d}", o["energy"], o["spin_final"], o["steps"], o["restarts"],
             o["status"] + (f": {o['reason'].splitlines()[0]}" if o["reason"] else "")]
            for o in out]
    table = format_table(["run", "q", "E/eV", "final spin", "geometry steps", "restarts",
                          "status"], rows, f"Pristine supercell: E_host = {hr['energy']:.6f} eV")
    return {"host_energy": hr["energy"], "runs": out, "_table": table,
            "_note": f"{len(out) - len(failed)} of {len(out)} relaxations done"
                     + (f", {len(failed)} failed" if failed else "")}


# ==========================================================================================
# PHASE E: THERMODYNAMICS (flowchart E1-E3)
# ==========================================================================================
KB = 8.617333262e-5                # eV/K
# chart colours (reference palette: light surface, categorical slots in their fixed order)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


# The production results (D1) of the runs that finished, by candidate and charge state, keeping
# the lowest energy of each (with the run it came from).
#   Called by: step_corrections(), step_formation()
def best_production(p):
    prod = p.out("production")
    best = {}
    for r in prod["runs"]:
        if r["energy"] is None:
            continue
        cur = best.setdefault(r["cand"], {}).get(r["q"])
        if cur is None or r["energy"] < cur["energy"]:
            best[r["cand"]][r["q"]] = r
    return prod["host_energy"], best


# E1: the finite-size correction (charge_correction, with the static dielectric constant from
# A2.3) for every charge state that reached phase D.
#   Called by: main() (step E1)
#   Calls:     Project.host(), best_production(), correction(), format_table(), host_supercell()
def step_corrections(p):
    _, best = best_production(p)
    prim, P = p.host(), np.array(p.out("choose")["matrix"])
    cell = host_supercell(prim, P).cell
    eps = p.out("dielectric").get("eps0")
    qs = sorted({q for st in best.values() for q in st})
    corr = {str(q): correction(q, cell, eps, p.s.charge_correction) for q in qs}
    rows = [[q, corr[str(q)]] for q in qs]
    table = format_table(["q", "E_corr/eV"], rows,
                         f"{p.s.charge_correction} corrections (epsilon {eps})")
    return {"corrections": corr, "method": p.s.charge_correction, "eps0": eps, "_table": table,
            "_note": ", ".join(f"q {q:+d}: {corr[str(q)]:.3f} eV" for q in qs if q)}


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
# E_f(q, E_F=0) = E_D - E_host - sum n_i mu_i + q E_VBM + E_corr at every limit, and its
# thermodynamic transition levels in the gap (transition_levels; the same at every limit); the
# plot goes to E_thermo/formation.png. With fermi_temperature, fermi_level finds
# the self-consistent Fermi level at each limit from these defects and the host's free carriers
# (quench_temperature: concentrations frozen at fermi_temperature, then charges redistributed).
#   Called by: main() (step E2)
#   Calls:     Project.host(), band_gap(), best_production(), fermi_level(), format_table(),
#              mu_limits(), plot(), read_bands(), transition_levels()
def step_formation(p):
    d = p.phase_dir("E")
    host_e, best = best_production(p)
    if not best:
        die("no production run finished, so there are no formation energies")
    cands = p.out("generate")["candidates"]
    corr = {int(q): v for q, v in p.out("corrections")["corrections"].items()}
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
            ef0 = {q: r["energy"] - host_e - dmu + q * vbm + corr.get(q, 0.0)
                   for q, r in best[cid].items()}
            lv = transition_levels(ef0, gap)
            curves[lim][f"{cid} {c['name']}"] = (ef0, lv)
            results.setdefault(cid, {"name": c["name"], "levels": [list(x) for x in lv],
                                     "formation": {}})["formation"][lim] = {str(q): v
                                                                           for q, v in ef0.items()}
            for q in sorted(ef0):
                table.append([lim, cid, c["name"], f"{q:+d}", ef0[q], ef0[q] + q * gap,
                              best[cid][q]["job"]])
    for cid in sorted(best):
        lv = results[cid]["levels"]
        levels_rows.append([cid, cands[cid]["name"],
                            ", ".join(f"({a:+d}/{b:+d}) {e:.3f}" for a, b, e in lv) or
                            "one charge state across the gap"])
    tables = [format_table(["limit", "id", "defect", "q", "E_f(VBM)/eV", "E_f(CBM)/eV",
                            "from run"], table,
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
# more than deep_level_margin from both band edges (a deep level), or if phase C found an internal
# transition for it (a first ZPL estimate between occupied and empty gap levels). The list
# (for_F in the results) is for phase F (optical lineshapes), which is not scripted yet.
#   Called by: main() (step E3)
#   Calls:     format_table()
def step_interest(p):
    form = p.out("formation")
    gap, m = form["gap"], p.s.deep_level_margin
    surv = {s["cand"]: s for s in p.out("zpl")["survivors"]}
    picks, rows = [], []
    for cid, dd in sorted(form["defects"].items()):
        why = [f"deep level ({a:+d}/{b:+d}) at {e:.2f} eV" for a, b, e in dd["levels"]
               if m <= e <= gap - m]
        s = surv.get(cid)
        if s:
            zpl = [s["best"][str(q)]["zpl"] for q in s.get("good", [])
                   if s["best"].get(str(q), {}).get("zpl") is not None]
            if zpl:
                why.append(f"internal transition, first estimate {min(zpl):.2f} eV")
        rows.append([cid, dd["name"], "yes: " + "; ".join(why) if why else "no"])
        if why:
            picks.append({"cand": cid, "name": dd["name"], "why": why})
    table = format_table(["id", "defect", "on to F?"], rows,
                         f"Deep levels (>= {m:g} eV from both edges) or internal transitions")
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
    # B: candidate defects and cell size
    Step("generate", "B1", "candidate defects", step_generate,
         keys=("dopants", "defects", "vacancies", "antisites", "interstitials", "pairs",
               "pair_max_distance", "native_pairs", "dopant_pairs"), needs=("relax",)),
    Step("scan", "B2", "MACE size scan", step_scan,
         keys=("mace", "scan", "scan_sizes", "scan_tol", "mace_model", "mace_fmax", "mace_steps",
               "transform", "series_defect", "supercell"), needs=("generate",),
         owns=("B_candidates/mace",)),
    Step("series", "B3", "CASTEP size series", step_series,
         keys=("series_defect", "series_charge", "series_sizes", "series_tol", "series_choice",
               "supercell", "transform", "mace", "mace_model", "charge_correction", "energy"),
         needs=("generate", "scan", "relax", "converge", "dielectric"),
         owns=("B_candidates/series",)),
    Step("finetune", "B4", "MACE fine-tuning data", step_finetune,
         keys=("finetune", "finetune_args", "mace_model"), needs=("series",),
         owns=("B_candidates/finetune",)),
    Step("choose", "B5", "choose the supercell", step_choose,
         keys=("supercell", "transform"), needs=("series", "relax")),
    # C: screening and ground states
    Step("prerelax", "C1", "build the candidates and pre-relax them with MACE", step_prerelax,
         keys=("mace", "mace_model", "mace_fmax", "mace_steps", "triage", "triage_window"),
         files=("finetuned_model",), needs=("generate", "choose", "relax"),
         owns=("C_screen/structures/candidates", "C_screen/mace")),
    Step("screen", "C2", "cheap CASTEP screening of every candidate", step_screen,
         keys=SCREEN_KEYS + ("screen_charges", "screen_spins"), needs=("prerelax", "converge"),
         owns=("C_screen/jobs/screen", "C_screen/structures/screen")),
    Step("filter", "C3", "defect hull and property filters", step_filter,
         keys=("hull_margin", "require_gap_states", "require_spin", "require_bound",
               "distort_charges", "charge_correction"),
         needs=("screen", "dielectric", "prerelax")),
    Step("zpl", "C3b", "quick ZPL filter", step_zpl, keys=("zpl_window",), needs=("filter",)),
    Step("distort", "C4", "ShakeNBreak distortions of the survivors", step_distort,
         keys=("distortions", "distort_around", "rattle_stdev", "rattle_dmin",
               "oxidation_states"), needs=("zpl", "prerelax"),
         owns=("C_screen/structures/distortions",)),
    Step("drelax", "C5", "cheap CASTEP relaxations of the distortions", step_drelax,
         keys=SCREEN_KEYS, needs=("distort",),
         owns=("C_screen/jobs/drelax", "C_screen/structures/drelax")),
    Step("rank", "C6", "rank, merge, share distortions between charge states", step_rank,
         keys=SCREEN_KEYS + ("distortion_min_gain", "share_rounds", "keep_structures"),
         needs=("drelax",), owns=("C_screen/jobs/share_*", "C_screen/structures/share")),
    # D: production relaxations
    Step("production", "D1", "production relaxations", step_production,
         keys=("production_overrides", "freeze_radius", "energy"),
         needs=("rank", "choose", "converge", "relax"),
         owns=("D_production/jobs", "D_production/structures")),
    # E: thermodynamics
    Step("corrections", "E1", "finite-size corrections", step_corrections,
         keys=("charge_correction",), needs=("production", "dielectric", "choose")),
    Step("formation", "E2", "formation energies, levels, Fermi level", step_formation,
         keys=("gap", "fermi_temperature", "quench_temperature"),
         needs=("corrections", "production", "bands", "limits", "generate"),
         owns=("E_thermo/formation.png",)),
    Step("interest", "E3", "which defects go on to the lineshape (F)", step_interest,
         keys=("deep_level_margin",), needs=("formation", "zpl")),
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
# Supercell shape: size n uses n x transform x (primitive cell). This one gives cubic cells
# (8 n^3 atoms) from an FCC primitive cell; leave it out for n x n x n primitive cells.
%block transform
  -1  1  1
   1 -1  1
   1  1 -1
%endblock transform
mace        : true             # MACE for the size scan (B2) and pre-relaxations (B3, C1)
!mace_model : medium-mpa-0     # a mace_mp foundation model, or a .model file
!finetuned_model : dwf_CaO_Sb/B_candidates/finetune/dwf_finetuned.model   # C1 only: the
                               #   model fine-tuned from B4's data (finetune.sh)
!mace_threads : 8

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
# steps (A1 to A3.1) and runner_screen for all of phase C's (C2, C5, C6); for one step:
# runner_converge (A1), runner_relax (A2.1), runner_bands (A2.2), runner_dielectric (A2.3),
# runner_phases (A3.1), runner_series (B3), runner_drelax (C5), runner_rank (C6),
# runner_production (D1). A step's own block wins over runner_host or runner_screen, and
# they win over the settings above, for the keywords they give (a %block partitions in one
# replaces the whole list). pack : N runs N of a step's jobs side by side on one node, each
# on 1/N of it: for small jobs, as a node is charged whole however few cores a job uses.
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
%block runner_series           # B3: up to the 512-atom cell of size 4
max_nodes    : 12
%endblock runner_series
%block runner_screen           # C2, C5, C6: Gamma-only screening jobs are small
max_nodes    : 1
max_restarts : 0               # (the default for screening: a job that fails is flagged)
%endblock runner_screen
%block runner_production       # D1: the production relaxations
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
!bands_kspacing    : 0.02      # A2 band edges (default: half the production spacing)
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
scan          : true           # B2: MACE size scan
scan_sizes    : 2 3 4 5
scan_tol      : 0.02 eV
!series_defect : Sb_Ca@1 + Vac_O   # B3: the defect for the CASTEP size series (default:
                               #   the first dopant pair, nearest arrangement)
!series_charge : 0             # default: 0, or the defect's own q= in %block defects
series_sizes  : 2 3 4
series_tol    : 0.05 eV
!supercell    : 3              # skip B2-B5 and use this size
!finetune     : true           # B4: export MACE fine-tuning data from the series

# ======================================================================= C: screening
screen_charges   : -1 0 1
screen_spins     : both        # low | both (also +2 unpaired electrons)
!screen_cutoff   : 500 eV      # a lower cut-off for screening (default: production)
screen_max_steps : 50          # geometry steps per screening relaxation
%block screen_overrides        # any other .param keywords for the screening runs
elec_energy_tol : 1e-5
%endblock screen_overrides
triage        : false          # C1: drop arrangements far above the lowest by MACE energy
triage_window : 1.0 eV
hull_margin   : 0.2 eV         # C3: keep arrangements within this of the defect hull
require_gap_states : none      # none | occupied | empty | both (filled and empty levels)
!require_spin : 2              # e.g. 2 = triplet ground state
require_bound : false          # bound against its isolated components
level_margin  : 0.1 eV
!zpl_window   : 0.8 3.5 eV     # C3b: keep estimates in this window (wide and lopsided)
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

# ======================================================================= D: production
%block production_overrides
geom_energy_tol : 5e-6
elec_energy_tol : 5e-7
geom_max_iter   : 100
%endblock production_overrides
freeze_radius : 0              # D2 (untested idea): fix the atoms further than this (Ang)
                               #   from every component of the defect

# ======================================================================= E: thermodynamics
charge_correction : lz         # lz | mp | none
!gap : 7.09 eV                 # band gap for the Fermi-level range (default: from A2)
!fermi_temperature  : 1000 K   # self-consistent Fermi level at this temperature
!quench_temperature : 300 K    # ... with the defects frozen in, then at this one
deep_level_margin : 0.5 eV     # E3: a level this far from both band edges is "deep"
"""


if __name__ == "__main__":
    main()
