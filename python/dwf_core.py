#!/usr/bin/env python3
"""
dwf_core.py
===========
Shared code for the defect-workflow scripts. It is not run on its own: each phase of
the workflow has its own script, and they all use this module.

    dwf_host.py        A  host reference calculations (converge, relax, bands,
                          dielectric, competing phases, chemical-potential limits)
    dwf_candidates.py  B  candidate defects and supercell size (generate, MACE size scan,
                          CASTEP size series, MACE fine-tuning data, choice of cell)
    dwf_screen.py      C  screening funnel and ground-state search (MACE pre-relaxation,
                          cheap CASTEP screen, hull and property filters, ZPL filter,
                          ShakeNBreak distortions, ranking)
    dwf_production.py  D  production relaxations
    dwf_thermo.py      E  corrections, formation energies, transition levels, Fermi level

Every script reads the same project file, <name>.dwf, written like a CASTEP .param file
(keyword : value, %block ... %endblock). Print an annotated example with any of them:

    python3 dwf_host.py --example > CaO_Sb.dwf

and run the phases in order:

    python3 dwf_host.py CaO_Sb.dwf          # or just CaO_Sb
    python3 dwf_candidates.py CaO_Sb.dwf
    ...

Checkpoints
-----------
Each phase is a list of steps. A step that finishes is recorded in <workdir>/state.json
with a fingerprint of everything it used (its settings, the files it read and the
results of the steps before it), and is skipped when the script is run again. So a
script can be stopped at any point (Ctrl-C, the STOP file, --no-wait, a crash) and
rerun with the same command: finished steps are skipped and a step that was part-way
through carries on, because its CASTEP jobs are run by supercell_convergence.py, which
picks up submitted and finished jobs rather than repeating them.
If you change a setting, the steps that use it (and the steps after them) run again;
jobs whose inputs did not change are reused, and a job that already ran with other
inputs stops the script, so results from two sets of settings are never mixed. Start
a step again from scratch with --redo STEP (its old directory is kept, renamed).

All CASTEP jobs go through supercell_convergence.py (which must be in the same
directory as these scripts): the B3 size series is a normal size-series run, which builds
its own cells, and every other step uses its batch mode, which runs the structures the
scripts build (build_defect uses the same supercell and centring code, so the cells
match). Its runner settings (jobscript, partitions, cores_per_node, max_nodes, walltime,
max_restarts, ...) go in the project file too, and %block runner_<group> changes them for
one group of steps (host, series, screen, production).
"""

import argparse
import difflib
import hashlib
import itertools
import json
import os
import re
import socket
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
# Each phase script lists its steps (Step: name, label, title, the function that does the work,
# and the settings, files and earlier steps its results depend on) and hands them to run_phase:
#
#   run_phase            the command line; Project(...) -> read_project reads the project file
#     for each step:
#       inputs_hash      fingerprint of its settings, files and the outputs of the steps it needs
#                        (skipped if state.json has it done with the same fingerprint)
#       <step function>  does the work - CASTEP jobs through Project.batch -> supercell_convergence
#                        run_batch - and returns its outputs (None: not finished, e.g. --no-wait)
#       outputs_hash     fingerprint of the outputs and the files they name, for the later steps
#
# Helpers the steps share: structures (atoms_to_dict, save_structure, load_structure,
# build_defect, host_supercell, symmetry_ops, equivalent_sites, electron_parity, spin_states),
# CASTEP output (read_bands, occupied_bands, band_edges, gap_levels, zpl_estimate,
# read_permittivity, read_trajectory), the charge correction (correction) and MACE (Project.mace,
# mace_relax). Project.out(), Project.phase_dir(), read_json() and write_json() are used by
# almost every step, so they are left out of the "Calls" lists, as are log() and die().
# ==========================================================================================

HARTREE = 27.211386245988                     # eV
STATE = "state.json"
PHASE_DIRS = {"A": "A_host", "B": "B_candidates", "C": "C_screen", "D": "D_production",
              "E": "E_thermo"}
SCRIPTS = {"A": "dwf_host.py", "B": "dwf_candidates.py", "C": "dwf_screen.py",
           "D": "dwf_production.py", "E": "dwf_thermo.py"}


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
    "converge_kspacings": (_floats(), [0.1, 0.08, 0.06, 0.05, 0.04, 0.03]),
    "converge_tol": (sc._num("eV/atom", "eV"), 0.001), "kspacing": (sc._num("1/ang"), None),
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
LIST_KEYS = {"transform", "converge_cutoffs", "converge_kspacings", "dopants", "scan_sizes",
             "series_sizes", "screen_charges", "distortions", "zpl_window"}   # block or keyword
# supercell_convergence keywords that control how jobs run (allowed in the project file and in
# %block runner_<group>)
RUNNER_KEYS = {"jobscript", "dryrun_jobscript", "partitions", "max_cores", "cores",
               "cores_per_node", "max_nodes", "mem_per_core", "mem_safety", "mem_replicated",
               "baseline_mb", "omp_threads", "walltime", "walltime_safety", "sp_walltime",
               "max_restarts", "delete_check", "dryrun_on", "dryrun_partition", "dryrun_time",
               "dryrun_cores", "dryrun_mem_gb", "dryrun_options", "dryrun_cmd",
               "dryrun_timeout", "submit_cmd", "status_cmd", "poll", "max_jobs",
               "tolerate_failures"}
RUNNER_GROUPS = ("host", "series", "screen", "production")
# what each group of steps uses unless the project file's %block runner_<group> says otherwise
GROUP_DEFAULTS = {"host": "", "series": "",
                  "screen": "max_restarts : 0\ntolerate_failures : true",
                  "production": "tolerate_failures : true"}


# Reads the project file into a namespace of settings.
# Every keyword must be a project keyword (PROJECT_KEYWORDS) or a runner keyword (RUNNER_KEYS,
# passed to supercell_convergence.batch_args); %block runner_<group> holds runner keywords for one
# group of steps. A typo gets the closest keyword suggested. Returns (settings, the text each
# setting was given as, the runner items, {group: runner items}).
#   Called by: Project.__init__()
#   Calls:     sc.block_rows(), sc.parse_castep_input()
def read_project(path):
    name = path.name
    vals, raw, runner, groups = {}, {}, [], {}
    known = set(PROJECT_KEYWORDS) | RUNNER_KEYS | {f"runner_{g}" for g in RUNNER_GROUPS}
    for it in sc.parse_castep_input(path.read_text()):
        if it["type"] == "raw":
            continue
        key = it["name"] if it["type"] == "block" else it["key"]
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
            for x in items:
                k = x["name"] if x["type"] == "block" else x["key"]
                if k not in RUNNER_KEYS:
                    die(f"{name}: %block {key} can only hold runner settings, not '{k}'")
            groups[key[len("runner_"):]] = items
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
    if not any((x["name"] if x["type"] == "block" else x["key"]) == "jobscript" for x in runner):
        die(f"{name}: give the jobscript template, as 'jobscript : file' or %block jobscript")
    if s.transform is not None and round(np.linalg.det(np.reshape(s.transform, (3, 3)))) <= 0:
        die(f"{name}: transform must have a positive determinant")
    for key in ("pair_max_distance", "scan_tol", "series_tol", "triage_window", "converge_tol",
                "mace_fmax"):
        if getattr(s, key) is not None and getattr(s, key) <= 0:
            die(f"{name}: {key} must be greater than 0")
    if s.zpl_window and s.zpl_window[0] >= s.zpl_window[1]:
        die(f"{name}: zpl_window is  lowest highest  (eV)")
    return s, raw, runner, groups


# A project: its settings, files, directories, checkpoints (state.json) and job runner.
# Each phase script makes one from the project file and the command-line switches, and passes it
# to its steps.
class Project:
    # Reads the project file and the state of earlier runs.
    #   Called by: run_phase()
    #   Calls:     read_project()
    def __init__(self, path, cl):
        self.path, self.base, self.cl = path, path.parent, cl
        self.s, self.raw, self.runner_items, self.runner_groups = read_project(path)
        self.name = self.s.name or path.stem
        self.dir = self.base / (self.s.workdir or f"dwf_{self.name}")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.dir / STATE
        self.state = (json.loads(self.state_file.read_text()) if self.state_file.exists()
                      else {"steps": {}})
        self._runners, self._mace, self._model_ids = {}, {}, {}

    # A file named by a project keyword (relative to the project file); stops if it is missing.
    #   Called by: Project.cell_text(), Project.production_param(), inputs_hash(),
    #              dwf_host.step_converge(), dwf_host.step_dielectric(), dwf_host.step_relax(),
    #              dwf_screen.step_prerelax()
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
    #   Called by: reset_step(), run_phase()
    def save_state(self):
        tmp = self.state_file.with_name(STATE + ".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.state_file)

    # The outputs a finished step recorded; stops with what to run if it hasn't finished.
    #   Called by: almost every step
    def out(self, step):
        rec = self.state["steps"].get(step)
        if not rec or rec.get("status") != "done":
            owner = STEP_OWNER.get(step, "?")
            die(f"step '{step}' has not finished yet: run  python3 {SCRIPTS.get(owner, '?')} "
                f"{self.path.name}  first")
        return rec["outputs"]

    # Whether a step has finished (with any inputs).
    #   Called by: Project.kspacing(), Project.production_param(), run_phase(), dwf_thermo.limits()
    def finished(self, step):
        return self.state["steps"].get(step, {}).get("status") == "done"

    # The runner settings for a group of steps (supercell_convergence.batch_args): the project's
    # runner keywords, then the group's defaults (GROUP_DEFAULTS), then %block runner_<group>,
    # each replacing the one before; then the command-line switches.
    #   Called by: Project.batch(), dwf_candidates.runner_conv_lines()
    #   Calls:     sc.batch_args(), sc.parse_castep_input()
    def runner(self, group):
        if group not in self._runners:
            def key(x):
                return x["name"] if x["type"] == "block" else x["key"]
            own = self.runner_groups.get(group, [])
            dflt = [x for x in sc.parse_castep_input(GROUP_DEFAULTS.get(group, ""))
                    if x["type"] != "raw" and key(x) not in {key(y) for y in own}]
            later = {key(y) for y in own + dflt}
            items = [x for x in self.runner_items if key(x) not in later] + dflt + own
            args = sc.batch_args(items, self.base, f"{self.path.name} (runner settings, "
                                                    f"{group} steps)")
            for flag in ("no_wait", "prepare_only", "force", "resubmit_failed"):
                if getattr(self.cl, flag, False):
                    setattr(args, flag, True)
            self._runners[group] = args
        return self._runners[group]

    # Runs a list of CASTEP jobs with supercell_convergence.run_batch in <workdir>/<subdir>, with
    # the runner settings of a group of steps and the project's STOP file. Returns run_batch's
    # result ({'complete', 'jobs', 'calcs'}).
    #   Called by: dwf_host.step_bands(), dwf_host.step_converge(), dwf_host.step_dielectric(),
    #              dwf_host.step_phases(), dwf_host.step_relax(), dwf_production.step_production(),
    #              dwf_screen.run_screen_jobs(), dwf_screen.step_screen()
    #   Calls:     Project.cell_text(), Project.runner(), sc.run_batch()
    def batch(self, subdir, jobs, group, param_text, cell_text=None, spacing=None):
        return sc.run_batch(self.dir / subdir, jobs, self.runner(group),
                            cell_text or self.cell_text(), param_text, spacing=spacing,
                            energy=self.s.energy, stop_file=self.dir / "STOP")

    # The host .cell as given (species_pot, symmetry_generate and anything else every job copies).
    #   Called by: Project.batch(), Project.kspacing(), dwf_host.step_dielectric(),
    #              dwf_host.step_relax()
    #   Calls:     Project.file()
    def cell_text(self):
        return self.file("host_cell").read_text()

    # The .param of the production settings: A1's (with the converged cut-off) once it has run,
    # else the project's .param as it is.
    #   Called by: dwf_candidates.step_series(), dwf_host.step_bands(), dwf_host.step_dielectric(),
    #              dwf_host.step_phases(), dwf_host.step_relax(), dwf_production.step_production(),
    #              dwf_screen.run_screen_jobs(), dwf_screen.step_screen()
    #   Calls:     Project.file(), Project.finished()
    def production_param(self):
        if self.finished("converge"):
            return (self.dir / self.out("converge")["param_file"]).read_text()
        return self.file("param").read_text()

    # The production k-point spacing (1/Ang): A1's, else kspacing, else from the host .cell's grid.
    #   Called by: dwf_candidates.step_series(), dwf_host.step_bands(), dwf_host.step_converge(),
    #              dwf_host.step_dielectric(), dwf_host.step_phases(), dwf_host.step_relax(),
    #              dwf_production.step_production()
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

    # The relaxed host primitive cell from A2 (ASE Atoms).
    #   Called by: dwf_candidates.step_choose(), dwf_candidates.step_generate(),
    #              dwf_candidates.step_scan(), dwf_candidates.step_series(),
    #              dwf_host.doped_phases(), dwf_host.step_bands(), dwf_host.step_dielectric(),
    #              dwf_host.step_limits(), dwf_production.step_production(), dwf_screen.setup(),
    #              dwf_thermo.step_corrections(), dwf_thermo.step_formation()
    #   Calls:     load_structure()
    def host(self):
        return load_structure(self.dir / self.out("relax")["structure"])[0]

    # The supercell matrix for size n: n x transform (or n x identity).
    #   Called by: dwf_candidates.step_choose(), dwf_candidates.step_scan(),
    #              dwf_candidates.step_series()
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
# Steps and checkpoints
# --------------------------------------------------------------------------
# step name -> phase, so a script can say which script runs a step it needs
STEP_OWNER = {"converge": "A", "relax": "A", "bands": "A", "dielectric": "A", "phases": "A",
              "limits": "A", "generate": "B", "scan": "B", "series": "B", "finetune": "B",
              "choose": "B", "prerelax": "C", "screen": "C", "filter": "C", "zpl": "C",
              "distort": "C", "drelax": "C", "rank": "C", "production": "D",
              "corrections": "E", "formation": "E", "interest": "E"}


# One step of a phase.
# name (e.g. 'relax'), label (e.g. 'A2'), title, the function that does the work (it takes the
# Project and returns the step's outputs as a dict, or None if it is not finished yet, e.g. jobs
# still queued with --no-wait), and what its results depend on: project keywords (keys), files
# named by project keywords (files) and earlier steps (needs). owns lists the paths under the work
# directory that belong to the step (moved aside by --redo).
class Step:
    # Records the step and which phase it belongs to.
    def __init__(self, name, label, title, func, keys=(), files=(), needs=(), owns=()):
        self.name, self.label, self.title, self.func = name, label, title, func
        self.keys, self.files, self.needs, self.owns = keys, files, needs, owns


# The fingerprint of what a step's results depend on: its settings (as written in the project
# file), the contents of its files and the output fingerprints of the steps it needs. Stops if one
# of those steps hasn't finished.
#   Called by: run_phase()
#   Calls:     Project.file()
def inputs_hash(p, step):
    data = {"keys": {k: p.raw.get(k) for k in step.keys}, "files": {}, "needs": {}}
    for f in step.files:
        if getattr(p.s, f):
            data["files"][f] = hashlib.md5(p.file(f).read_bytes()).hexdigest()
    for n in step.needs:
        p.out(n)                                   # stops if it hasn't finished
        data["needs"][n] = p.state["steps"][n]["out_hash"]
    return hashlib.md5(json.dumps(data, sort_keys=True).encode()).hexdigest()


# The fingerprint of a step's outputs, which the steps after it depend on: the outputs themselves
# and the contents of every file under the work directory they name (e.g. the JSON files with the
# results), so a step that is run again with different results makes the later ones run again.
#   Called by: run_phase()
def outputs_hash(p, outputs):
    files = {}

    # Collects the files named anywhere in the outputs.
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


# Prints every step of every phase that has run, with its status and when it finished.
#   Called by: run_phase()
def print_status(p):
    print(f"\nProject {p.name} ({p.dir}):")
    steps = p.state["steps"]
    if not steps:
        print("  nothing has run yet")
    for name, rec in sorted(steps.items(), key=lambda kv: kv[1].get("label", "")):
        print(f"  {rec.get('label', '?'):<4} {name:<12} {rec.get('status', '?'):<9} "
              f"{rec.get('time', '')}  {rec.get('note', '')}")
    print()


# Moves a step's files aside (<path>.old-<time>) and forgets that it ran: --redo. An owned path
# with a * in it (e.g. C_screen/share_*) moves every match, except earlier .old- copies.
#   Called by: run_phase()
#   Calls:     Project.save_state()
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


# The command line every phase script takes, and the work it does: runs the phase's steps in order,
# skipping those that have finished with the same inputs, and stops at the first that isn't
# finished (jobs still queued with --no-wait). Named steps run only those (and need the steps
# before them to have finished). --status prints the checkpoints of the whole project; --redo
# starts steps again from scratch; --example prints an annotated project file.
#   Called by: the main block of each phase script
#   Calls:     Project.__init__(), Project.finished(), Project.save_state(), inputs_hash(),
#              outputs_hash(), print_status(), reset_step()
def run_phase(phase, steps, doc):
    ap = argparse.ArgumentParser(description=doc,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("project", nargs="?", help="the project file (<name>.dwf) or its name")
    ap.add_argument("steps", nargs="*", help="run only these steps (names or labels, e.g. relax "
                    "or A2); default: all of this phase")
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
    names = {s.name: s for s in steps}
    names.update({s.label.lower(): s for s in steps})
    for want in cl.steps + cl.redo:
        if want.lower() not in names:
            die(f"no step '{want}' in phase {phase}; its steps are "
                + ", ".join(f"{s.label} {s.name}" for s in steps))
    chosen = {names[w.lower()].name for w in cl.steps} or {s.name for s in steps}
    redo = {names[w.lower()].name for w in cl.redo}
    print(f"\n{doc.strip().splitlines()[0]}\nProject {p.name}: {p.path} -> {p.dir}/")
    if not (cl.prepare_only or cl.no_wait):
        log(f"running on {socket.gethostname()} as process {os.getpid()}; to stop it from "
            f"any login node: touch {(p.dir / 'STOP').absolute()}")
    try:
        for step in steps:
            if step.name not in chosen:
                continue
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
                log(f"{step.label} {step.name} is not finished yet - rerun the same command to "
                    "carry on")
                return
            note = outputs.pop("_note", "")
            p.state["steps"][step.name] = {
                "status": "done", "label": step.label, "inputs": h, "outputs": outputs,
                "out_hash": outputs_hash(p, outputs),
                "time": time.strftime("%Y-%m-%d %H:%M"), "note": note}
            p.save_state()
            log(f"{step.label} {step.name}: finished" + (f" - {note}" if note else ""))
    except KeyboardInterrupt:
        print("\nInterrupted. Submitted jobs keep running - rerun the same command to carry on.")
        return
    log(f"phase {phase} done" if all(p.finished(s.name) for s in steps if s.name in chosen)
        else f"phase {phase}: stopped")


# --------------------------------------------------------------------------
# Structures
# --------------------------------------------------------------------------
# An ASE Atoms object as plain data (species, cell, fractional positions, starting spins), for the
# JSON files that carry structures from one step to the next.
#   Called by: mace_relax(), save_structure()
def atoms_to_dict(atoms):
    return {"symbols": atoms.get_chemical_symbols(),
            "cell": np.asarray(atoms.cell).round(10).tolist(),
            "frac": atoms.get_scaled_positions(wrap=False).round(10).tolist(),
            "magmoms": atoms.get_initial_magnetic_moments().round(6).tolist()}


# The reverse of atoms_to_dict.
#   Called by: load_structure(), mace_relax()
def atoms_from_dict(d):
    a = Atoms(d["symbols"], scaled_positions=d["frac"], cell=d["cell"], pbc=True)
    a.set_initial_magnetic_moments(d.get("magmoms") or [0.0] * len(a))
    return a


# Saves a structure, with any extra information, as JSON.
#   Called by: dwf_host.step_relax(), dwf_production.step_production(), dwf_screen.job_summary(),
#              dwf_screen.step_distort(), dwf_screen.step_prerelax(), dwf_screen.step_rank()
#   Calls:     atoms_to_dict()
def save_structure(path, atoms, **meta):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(meta, atoms=atoms_to_dict(atoms)), indent=1, default=float))


# Reads a structure saved by save_structure: (Atoms, the extra information).
#   Called by: Project.host(), dwf_production.step_production(), dwf_screen.distinct_states(),
#              dwf_screen.duplicates(), dwf_screen.run_screen_jobs(), dwf_screen.step_distort(),
#              dwf_screen.step_rank(), dwf_screen.step_screen()
#   Calls:     atoms_from_dict()
def load_structure(path):
    d = json.loads(Path(path).read_text())
    return atoms_from_dict(d.pop("atoms")), d


# Reads a structure file: a CASTEP .cell (keeping SPIN= values) or anything ASE reads.
#   Called by: dwf_host.phase_list(), dwf_host.step_converge(), dwf_host.step_relax()
#   Calls:     sc.read_structure()
def read_structure_file(path):
    return sc.read_structure(Path(path))


# 0 or 1: whether a structure with charge q has an even or odd number of electrons. Core shells
# are full, so the parity of the valence electrons (whatever the pseudopotentials) is that of the
# sum of the atomic numbers, minus the charge.
#   Called by: spin_states(), dwf_candidates.step_series(), dwf_screen.start_spin()
def electron_parity(atoms, q):
    return int(sum(atomic_numbers[s] for s in atoms.get_chemical_symbols()) - q) % 2


# The starting spins (unpaired electrons, hbar/2) a screening job tries for a structure with charge
# q: the lowest the electron count allows (0 or 1) and, with both, that +2.
#   Called by: dwf_screen.step_screen()
#   Calls:     electron_parity()
def spin_states(atoms, q, which):
    low = electron_parity(atoms, q)
    return [low] if which == "low" else [low, low + 2]


# The symmetry operations of a cell (spglib), as (rotations, translations) in fractional
# coordinates.
#   Called by: equivalent_sites(), dwf_candidates.candidate_points(),
#              dwf_candidates.site_rotations()
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
#   Called by: comp_name(), dwf_candidates.candidate_points(), dwf_candidates.parse_token(),
#              dwf_candidates.single_defects()
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
#   Called by: dwf_candidates.candidate()
#   Calls:     comp_name()
def defect_name(comps, prim=None):
    return "+".join(comp_name(c, prim) for c in comps)


# The change in composition a defect makes, e.g. {'Sb': 1, 'Ca': -1, 'O': -1}, and its text
# (+Sb-Ca-O) used to group arrangements that can be compared directly (the defect hull).
#   Called by: dwf_candidates.candidate()
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
#   Called by: dwf_candidates.step_scan(), dwf_candidates.step_series(), dwf_screen.step_prerelax()
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
#   Called by: dwf_candidates.step_choose(), dwf_production.step_production(),
#              dwf_screen.step_distort(), dwf_screen.step_filter(), dwf_screen.step_screen(),
#              dwf_thermo.step_corrections()
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
#   Called by: dwf_host.step_bands(), dwf_screen.job_summary(), dwf_screen.step_screen(),
#              dwf_thermo.step_formation()
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
#   Called by: band_edges(), gap_levels(), dwf_thermo.carriers()
def occupied_bands(b):
    if b["nspin"] == 1:
        return [int(round(b["nelec"][0] / 2))]
    if len(b["nelec"]) == 2:
        return [int(round(x)) for x in b["nelec"]]
    ef = b["efermi"][0]
    return [int(np.sum(b["eig"][s, 0] < ef)) for s in range(2)]


# The valence band maximum and conduction band minimum (eV) of an insulator's .bands: the highest
# occupied and lowest empty eigenvalue over all k-points and spins.
#   Called by: dwf_host.step_bands(), dwf_screen.step_screen()
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
#   Called by: dwf_screen.job_summary()
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
#   Called by: dwf_screen.job_summary(), dwf_screen.step_filter()
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
#   Called by: dwf_candidates.step_finetune()
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
#   Called by: dwf_host.step_dielectric()
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
#   Called by: dwf_screen.relative_formation(), dwf_thermo.step_corrections()
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
#   Called by: dwf_candidates.step_scan(), dwf_screen.step_prerelax()
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
# Writes a JSON file (made readable) and returns its path relative to the work directory, which is
# how steps record their output files.
#   Called by: almost every step
def write_json(p, path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, default=float))
    return str(path.relative_to(p.dir))


# Reads a JSON file named relative to the work directory.
#   Called by: almost every step
def read_json(p, rel):
    return json.loads((p.dir / rel).read_text())


# Writes a text table (a title, a header row, a rule and the rows, in columns as wide as needed)
# and returns its text. Floats are shown with 4 decimals, None as blank.
#   Called by: dwf_candidates.step_generate(), dwf_candidates.step_scan(), dwf_host.step_converge(),
#              dwf_host.step_limits(), dwf_screen.step_filter(), dwf_screen.step_rank(),
#              dwf_screen.step_zpl(), dwf_thermo.step_corrections(), dwf_thermo.step_formation(),
#              dwf_thermo.step_interest()
def write_table(path, header, rows, title=""):
    cells = [[("" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v)))
              for v in r] for r in rows]
    width = [max([len(h)] + [len(r[i]) for r in cells]) for i, h in enumerate(header)]
    fmt = lambda r: "  ".join(c.rjust(w) for c, w in zip(r, width))      # noqa: E731
    lines = ([title, ""] if title else []) + [fmt(header), "-" * len(fmt(header))]
    text = "\n".join(lines + [fmt(r) for r in cells]) + "\n"
    Path(path).write_text(text)
    return text


# The species counts of a structure, e.g. {'Ca': 32, 'O': 31, 'Sb': 1}.
#   Called by: dwf_host.step_limits(), dwf_host.step_phases()
def counts(atoms):
    syms = atoms.get_chemical_symbols()
    return {s: syms.count(s) for s in sorted(set(syms))}


# The tag of a charge state in job names (which can't hold + or -): q0, qp1, qm1, ...
#   Called by: dwf_production.step_production(), dwf_screen.step_distort(), dwf_screen.step_rank(),
#              dwf_screen.step_screen()
def qtag(q):
    return "q0" if q == 0 else f"q{'p' if q > 0 else 'm'}{abs(q)}"


EXAMPLE_PROJECT = """\
# Defect workflow project file    run:  python3 dwf_host.py <name>, then dwf_candidates.py,
#                                       dwf_screen.py, dwf_production.py, dwf_thermo.py
# Written like a CASTEP .param/.cell file: "keyword : value", %block ... %endblock, keywords
# in any case, # or ! starts a comment. Lines starting with ! are optional settings with an
# example value: delete the ! to use one. File names are relative to this file: the example
# inputs (CaO:Sb on ARCHER2) are in example_CaO_Sb/, this file and every file it names.

# ======================================================================= general
host_cell : CaO_prim.cell      # host primitive cell: species_pot, symmetry_generate, k-points
param     : CaO.param          # production settings (functional, cut-off, ...); the scripts
                               #   set task, charge and spin, and A1 sets cut_off_energy
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
max_nodes      : 4
%block partitions
standard   1.9
highmem    3.9    --qos=highmem
%endblock partitions
dryrun_partition : serial
dryrun_time      : 0:20:0
max_jobs  : 16                 # most jobs queued at once (ARCHER2 standard QoS: 16 running)
poll      : 300
delete_check : true
# ... and for one group of steps (host, series, screen, production) only:
%block runner_screen
max_nodes    : 1               # Gamma-only screening jobs are small
max_restarts : 0               # (the default for screening: a job that fails is flagged)
%endblock runner_screen

# ======================================================================= A: host
converge : true                # A1: converge the cut-off and k-point spacing first
converge_cutoffs   : 400 500 600 700 800 900 eV
converge_kspacings : 0.08 0.06 0.05 0.04 0.03   # 1/Ang (CASTEP's 2*pi units)
converge_tol       : 0.001 eV/atom
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
# arrangement within pair_max_distance, not just the nearest one.
!%block defects
!Sb_Ca + Vac_O   all
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
series_charge : 0
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
