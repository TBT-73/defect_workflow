# Defect workflow scripts (dwf_*.py)

Five scripts run the defect screening pipeline from a relaxed host to formation energies and
transition levels, one script per phase of the flowchart (`defect_workflow_flowchart_A1.pdf`).
This README covers the whole workflow; each script also has its own (below), and
`example_CaO_Sb/` is a complete example project ([Quick start](#quick-start)).

| Phase | Script | What it does |
|---|---|---|
| A | `dwf_host.py` | converge cut-off and k-points, relax the host, band edges, dielectric constant, competing phases, chemical-potential limits |
| B | `dwf_candidates.py` | candidate defects (yours, or generated), MACE size scan, CASTEP size series, MACE fine-tuning data, choice of supercell |
| C | `dwf_screen.py` | MACE pre-relaxation, cheap CASTEP screen of every candidate, defect hull and property filters, ZPL filter, ShakeNBreak distortions, ranking |
| D | `dwf_production.py` | production relaxations of the survivors in each charge state |
| E | `dwf_thermo.py` | finite-size corrections, formation energies, transition levels, Fermi level, which defects go on to phase F |

Phase F (optical lineshapes) is not scripted yet.

Two more files go in the same directory:

- `dwf_core.py`: the code the five scripts share: the project file, checkpoints, the CASTEP job
  runner interface, structures, parsers, MACE. It is not run on its own.
- `supercell_convergence.py`: the job runner (dry runs, layouts, walltimes, restarts). Every
  CASTEP job goes through it: the B3 size series in its usual mode, which builds its own cells,
  and every other step in its batch mode, which runs the structures the scripts build. See
  [How the scripts use supercell_convergence.py](#how-the-scripts-use-supercell_convergencepy)
  and its own README, `README_supercell_convergence.md`.

Each script has its own README: `README_dwf_core.md`, `README_dwf_host.md`,
`README_dwf_candidates.md`, `README_dwf_screen.md`, `README_dwf_production.md`,
`README_dwf_thermo.md`, and `README_supercell_convergence.md` for the job runner.

---

## Contents

1. [Requirements](#requirements)
2. [Quick start](#quick-start)
3. [How the scripts use supercell_convergence.py](#how-the-scripts-use-supercell_convergencepy)
4. [The project file](#the-project-file)
5. [Checkpoints, stopping and rerunning](#checkpoints-stopping-and-rerunning)
6. [Command line](#command-line)
7. [The work directory](#the-work-directory)
8. [Flowchart steps and files](#flowchart-steps-and-files)
9. [Keyword glossary](#keyword-glossary)
10. [Testing so far, and what to check first](#testing-so-far-and-what-to-check-first)

---

## Requirements

- Python 3.9+ with `numpy` and `ase` (all phases);
- `spglib` (B1: symmetry), `pymatgen` (A3 limits, C3/C6 structure matching, the oxidation-state
  guess in C4), `shakenbreak` (C4), `matplotlib` (E2 plot);
- `mace-torch` for the MACE steps (B2, the B3 pre-relaxation, C1), or set `mace : false`;
- optional: `pymatgen-analysis-defects` for Voronoi interstitials, `doped` and a Materials
  Project API key for `phases_from : doped`;
- CASTEP and Slurm (other schedulers through `submit_cmd` / `status_cmd`, as in
  `supercell_convergence.py`).

On ARCHER2:

```bash
module load PrgEnv-gnu cray-python
python -m venv ~/dwfenv && source ~/dwfenv/bin/activate
pip install ase spglib pymatgen shakenbreak matplotlib mace-torch
```

All seven `.py` files go in one directory; run the scripts from anywhere.

---

## Quick start

`example_CaO_Sb/` is a complete example (CaO:Sb on ARCHER2): the project file `CaO_Sb.dwf` and
every file it names.

| File | What it is |
|---|---|
| `CaO_Sb.dwf` | the project file, annotated (the same as `python3 dwf_host.py --example`) |
| `CaO_prim.cell` | the host primitive cell (rocksalt, a = 4.8105 Å); A2.1 relaxes it |
| `CaO.param` | production settings (PBE; A1 sets the cut-off) |
| `archer2.sh` | jobscript template |
| `CaO_ncp.cell`, `CaO_ncp.param` | the DFPT dielectric run (A2.3): norm-conserving potentials, fixed occupancies |
| `Ca.cell`, `O2.cell`, `Sb.cell`, `Sb2O3.cell` | competing phases (A3): fcc Ca, O₂ in a box, A7 Sb, senarmontite Sb₂O₃, near their experimental structures (A3.1 relaxes them) |

Before running it, check: the account and CASTEP module in `archer2.sh`; the pseudopotentials
(`CaO_prim.cell` has no `species_pot`, so CASTEP uses its default on-the-fly ones); the cut-off
in `CaO_ncp.param`; and the competing phases (add any other that could limit Sb). Then, with
`DWF` the directory holding the seven `.py` files:

```bash
cd example_CaO_Sb
python3 $DWF/dwf_host.py CaO_Sb.dwf --prepare-only   # check the first inputs, submit nothing
nohup python3 $DWF/dwf_host.py CaO_Sb.dwf > A.log 2>&1 &
```

then, each when the one before has finished:

```bash
nohup python3 $DWF/dwf_candidates.py CaO_Sb.dwf > B.log 2>&1 &
nohup python3 $DWF/dwf_screen.py     CaO_Sb.dwf > C.log 2>&1 &
nohup python3 $DWF/dwf_production.py CaO_Sb.dwf > D.log 2>&1 &
python3 $DWF/dwf_thermo.py CaO_Sb.dwf                 # quick: no CASTEP jobs
```

or all in one go: `nohup sh -c "python3 $DWF/dwf_host.py CaO_Sb && python3
$DWF/dwf_candidates.py CaO_Sb && python3 $DWF/dwf_screen.py CaO_Sb && python3
$DWF/dwf_production.py CaO_Sb && python3 $DWF/dwf_thermo.py CaO_Sb" > run.log 2>&1 &`. Each
script exits with an error if the step it needs from an earlier phase hasn't finished, so the
chain stops at the first problem.

For a new project, `python3 $DWF/dwf_host.py --example > <name>.dwf` prints the annotated
project file to start from; it names the same input files.

The scripts wait for their jobs (polling every `poll` seconds), so run them with `nohup` or in
`tmux`/`screen`. MACE runs where the script runs (the login node): keep `mace_threads` modest.

---

## How the scripts use supercell_convergence.py

The workflow uses `supercell_convergence.py` in two ways, and only one of them builds supercells:

| | Batch mode | Size series (its usual mode) |
|---|---|---|
| Used by | every CASTEP step except B3: A1–A3, C2, C5, C6, D1 | B3 only |
| Builds the structures | the `dwf_*.py` scripts (`dwf_core.build_defect`, `host_supercell`); the runner only runs them | `supercell_convergence.py` itself, from the defect tokens B3 writes into `B_candidates/series/series.conv` |
| Used later | each job's energy, spin and final structure | the cell size (B5) and, with `finetune : true`, the steps of its smallest relaxation as MACE training data (B4) |

So every structure that is screened (C) or relaxed for production (D) comes from one builder,
`dwf_core.build_defect`. The size series is a check on the cell size: none of its relaxed
structures or energies is used in C or D.

The two builders make the same cells because they share the code: `dwf_core.build_defect` calls
the runner's own `build_supercell`, `anchor_site` and `translation_copies`. Both build
n × `transform` of the relaxed host (A2.1) and centre the defect the same way, and the tokens B3
writes name the primitive-cell site each component sits on, so the series has the same
arrangement as B2 and C1.

They can differ in one way: the orientation of a complex. The series chooses the direction from
the first component to each of the others by its own rule (the first nearest site of that kind
it finds), so its complex can be a rotated, symmetry-equivalent copy of the one B2 and C1 use.
That is the same calculation in an n × n × n supercell of the primitive cell, and with a
`transform` that keeps the host's symmetry (such as the cubic cells in the example). Only a
supercell shape that breaks the host's symmetry (an elongated `transform`, say) makes the two
orientations see their periodic images differently; B3's E_f*(n) can then differ slightly from
that of the same defect in the C and D cells.

On its own, outside the workflow, `supercell_convergence.py` is the size-series tool described in
`README_supercell_convergence.md`.

---

## The project file

One file, `<name>.dwf`, holds everything, written like a CASTEP `.param` file:
`keyword : value`, `%block name … %endblock name`, any case, `#` or `!` comments. File names
are relative to it. A misspelt keyword is rejected with the closest match suggested.

It has three kinds of setting:

1. **Project keywords** for each phase, e.g. `dopants`, `screen_charges`, `hull_margin`
   ([Keyword glossary](#keyword-glossary)).
2. **Runner keywords**: any `supercell_convergence.py` keyword that controls how jobs run
   (`jobscript`, `cores_per_node`, `max_nodes`, `%block partitions`, `walltime`,
   `max_restarts`, `delete_check`, `dryrun_*`, `poll`, `max_jobs`, `tolerate_failures`, ...).
3. **Runner overrides** for one group of steps, in `%block runner_<group>` with group `host`
   (phase A), `series` (B3), `screen` (C2, C5, C6) or `production` (D1). Inside, runner keywords
   only:

   ```
   %block runner_screen
   max_nodes : 1          # Gamma-only screening jobs are small
   %endblock runner_screen
   ```

   Screening defaults to `max_restarts : 0` and `tolerate_failures : true` (a run that fails is
   flagged, not rescued); production defaults to `tolerate_failures : true`. A runner block
   overrides those defaults.

Your `.param` holds the production settings (functional, cut-off, SCF settings, smearing,
`geom_preconditioner`, ...). The scripts set `task`, and per job `charge`, `spin`,
`spin_polarized` and the k-points; A1 sets `cut_off_energy`. Any `charge` or `spin` in it is
left out of the production `.param` (A1 says so). Your host `.cell` gives `species_pot` (list
the dopants' potentials too), `symmetry_generate` and anything else every job copies.

Every defect cell is built with the defect as close to the cell centre as it can be: the mean
position of its components nearest the centre, with the complex in the same orientation in every
cell (see `README_dwf_candidates.md`).

---

## Checkpoints, stopping and rerunning

Each phase is a list of steps. When a step finishes, `<workdir>/state.json` records it with a
fingerprint of everything it used: its settings (as written in the project file), the files it
read, and the outputs (and output files) of the steps it needs.

- **Rerun after any stop** (Ctrl-C, `touch <workdir>/STOP`, `--no-wait`, a crash, a lost
  login session): finished steps are skipped; a step that was part-way through carries on,
  because the job runner picks up submitted and finished jobs instead of repeating them.
- **Change a setting:** the steps that use it, and the steps after them, run again. CASTEP jobs
  whose own inputs did not change are reused, so e.g. raising `hull_margin` re-runs C3 onwards
  and only submits the jobs for the extra survivors. A job that already ran with different
  inputs stops the script, so results from two sets of settings never mix: start that step
  again with `--redo STEP` (or keep the old results with `--force`).
- **`--redo STEP`** moves the step's files aside (`<name>.old-<time>`, delete them when you no
  longer need them) and runs it from scratch.
- **`--status`** lists every step of every phase that has run, when, and its one-line result.
- **`--no-wait`** submits what is needed and exits; run it from cron (or by hand) until the
  phase reports done.

---

## Command line

The same for every phase script:

```
python3 dwf_<phase>.py <project>.dwf [STEP ...] [--redo STEP] [--status] [--example]
                       [--prepare-only] [--no-wait] [--resubmit-failed] [--force]
```

| Option | Effect |
|---|---|
| `STEP ...` | run only these steps, by name or label (`relax`, `A2.1`); default all of the phase |
| `--redo STEP` | start that step again from scratch (can be repeated) |
| `--status` | show which steps have finished |
| `--example` | print the annotated project file |
| `--prepare-only` | write the inputs of the next CASTEP jobs (with dry runs), submit nothing |
| `--no-wait` | submit what is needed and exit; rerun later |
| `--resubmit-failed` | start failed CASTEP jobs again |
| `--force` | keep the results of jobs whose inputs have changed |

The project can be given as `CaO_Sb.dwf` or `CaO_Sb`.

---

## The work directory

`dwf_<name>/` next to the project file (or `workdir`):

```
dwf_CaO_Sb/
  state.json              checkpoints
  STOP                    touch it to stop the running script
  A_host/                 converge/ relax/ bands/ dielectric/ phases/ (job directories),
                          production.param, host_relaxed.json/.cell, converge.txt, limits.txt
  B_candidates/           candidates.json/.txt, scan/, scan.txt, series/ (size series),
                          finetune/
  C_screen/               structures/, screen/, screen.json, filter.json/.txt, zpl.json/.txt,
                          distortions/, drelax/, drelax.json, share_<n>/, rank.json/.txt
  D_production/           relax/, production.json, structures/
  E_thermo/               corrections.txt, formation.txt/.json/.png, levels.txt, fermi.txt,
                          interest.txt, for_F.json
```

Every job directory (under `relax/`, `screen/`, ...) is a `supercell_convergence.py` batch job:
CASTEP inputs and outputs, `job.sh`, `dryrun.sh` and `job_info.json`, and each batch has a
`batch_results.json`.

---

## Flowchart steps and files

| Flowchart | Script, step | Output |
|---|---|---|
| A1 | `dwf_host.py converge` | `A_host/production.param`, `converge.txt` |
| A2 (cell, e_bulk) | `dwf_host.py relax` | `A_host/host_relaxed.json/.cell` |
| A2 (VBM, gap) | `dwf_host.py bands` | `A_host/bands/` |
| A2 (dielectric) | `dwf_host.py dielectric` | in `state.json` (`--status`) |
| A3 | `dwf_host.py phases`, `limits` | `A_host/limits.txt` |
| B1 | `dwf_candidates.py generate` | `B_candidates/candidates.txt` |
| B2 | `dwf_candidates.py scan` | `B_candidates/scan.txt` |
| B3 | `dwf_candidates.py series` | `B_candidates/series/run/convergence.txt` |
| B4 | `dwf_candidates.py finetune` | `B_candidates/finetune/` |
| B5 | `dwf_candidates.py choose` | in `state.json` |
| C1 | `dwf_screen.py prerelax` | `C_screen/structures/` |
| C2 | `dwf_screen.py screen` | `C_screen/screen.json` |
| C3 | `dwf_screen.py filter` | `C_screen/filter.txt` |
| C3b | `dwf_screen.py zpl` | `C_screen/zpl.txt` |
| C4 | `dwf_screen.py distort` | `C_screen/distortions/` |
| C5 | `dwf_screen.py drelax` | `C_screen/drelax.json` |
| C6 | `dwf_screen.py rank` | `C_screen/rank.txt` |
| D1, D2 | `dwf_production.py production` | `D_production/production.json` |
| E1 | `dwf_thermo.py corrections` | `E_thermo/corrections.txt` |
| E2 | `dwf_thermo.py formation` | `E_thermo/formation.txt`, `levels.txt`, `formation.png`, `fermi.txt` |
| E3 | `dwf_thermo.py interest` | `E_thermo/interest.txt`, `for_F.json` |

---

## Keyword glossary

Every keyword of every input file, sorted by file. The project file (section 1) is the only file
the workflow scripts are given; it names the others. For the CASTEP files (sections 2 to 4) the
tables list only the keywords the workflow sets, changes or removes; every other keyword is
passed on to the jobs as you wrote it.

| Input file | Named by | Written by | Section |
|---|---|---|---|
| Project file `<name>.dwf` | the command line | you (`--example` writes a template) | [1](#1-the-project-file-namedwf) |
| Production `.param` | `param` | you | [2](#2-the-production-param-param) |
| Host `.cell` | `host_cell` | you | [3](#3-the-host-cell-host_cell) |
| Dielectric `.param` and `.cell` (optional) | `dielectric_param`, `dielectric_cell` | you | [4](#4-the-dielectric-param-and-cell-dielectric_param-dielectric_cell) |
| Competing-phase structures | `%block phases` | you | [5](#5-competing-phase-structure-files-block-phases) |
| Jobscript template | `jobscript` | you | [6](#6-the-jobscript-template-jobscript) |
| `series.conv` | — | B3 (for reference: don't edit it) | [7](#7-seriesconv-written-by-b3) |
| `.conv` and batch files | the command line | you, when running `supercell_convergence.py` on its own | [8](#8-supercell_convergencepy-on-its-own-conv-and-batch-files) |

### 1. The project file (`<name>.dwf`)

The file you pass to every `dwf_*.py` script. Its keywords are grouped below by the phase that
uses them.

**Writing them.** `keyword : value` (or `=`, or a space), in any case, each keyword once; `#` or
`!` starts a comment. File names are relative to the project file.

- **Numbers** can be followed by the unit in the Default column: `eV`, `ang` (or `A`) for Å,
  `1/ang`, `eV/atom`, `eV/ang`, `K`. The unit is checked, not converted: give the value in that
  unit.
- **Lists** (marked *list*) go on one line, separated by spaces or commas, or in a block with
  any number of values per line (`%block distortions` … `%endblock distortions`).
- **Switches** take `true` or `false` (also `yes`/`no`, `on`/`off`); the keyword alone means
  `true`.
- **Blocks** that hold `.param` keywords (`screen_overrides`, `production_overrides`) take
  `keyword : value` lines; the others take the rows described below.

**Changing them.** The Steps column lists the steps that use a keyword. Change it and those steps
run again the next time you run their script; the steps after them run again only if their
results change. CASTEP jobs whose inputs have not changed are reused, not repeated.

**Alphabetical index** (the group in brackets):

- **A**: [`antisites`](#b-candidates-and-cell-size) (B)
- **B**: [`bands_kspacing`](#a-host) (A) · [`baseline_mb`](#runner-settings) (runner)
- **C**: [`charge_correction`](#e-thermodynamics) (E) · [`%block chemical_potentials`](#a-host) (A) · [`converge`](#a-host) (A) · [`converge_cutoffs`](#a-host) (A) · [`converge_kspacings`](#a-host) (A) · [`converge_tol`](#a-host) (A) · [`cores`](#runner-settings) (runner) · [`cores_per_node`](#runner-settings) (runner)
- **D**: [`deep_level_margin`](#e-thermodynamics) (E) · [`%block defects`](#b-candidates-and-cell-size) (B) · [`delete_check`](#runner-settings) (runner) · [`dielectric`](#a-host) (A) · [`dielectric_cell`](#a-host) (A) · [`dielectric_inf`](#a-host) (A) · [`dielectric_param`](#a-host) (A) · [`dielectric_task`](#a-host) (A) · [`distort_around`](#c-screening) (C) · [`distort_charges`](#c-screening) (C) · [`distortion_min_gain`](#c-screening) (C) · [`distortions`](#c-screening) (C) · [`dopant_pairs`](#b-candidates-and-cell-size) (B) · [`dopants`](#b-candidates-and-cell-size) (B) · [`dryrun_cmd`](#runner-settings) (runner) · [`dryrun_cores`](#runner-settings) (runner) · [`dryrun_jobscript`](#runner-settings) (runner) · [`dryrun_mem_gb`](#runner-settings) (runner) · [`dryrun_on`](#runner-settings) (runner) · [`dryrun_options`](#runner-settings) (runner) · [`dryrun_partition`](#runner-settings) (runner) · [`dryrun_time`](#runner-settings) (runner) · [`dryrun_timeout`](#runner-settings) (runner)
- **E**: [`energy`](#general) (general)
- **F**: [`fermi_temperature`](#e-thermodynamics) (E) · [`finetune`](#b-candidates-and-cell-size) (B) · [`finetune_args`](#b-candidates-and-cell-size) (B) · [`finetuned_model`](#mace) (MACE) · [`freeze_radius`](#d-production) (D)
- **G**: [`gap`](#e-thermodynamics) (E)
- **H**: [`host_cell`](#general) (general) · [`hull_margin`](#c-screening) (C)
- **I**: [`interstitials`](#b-candidates-and-cell-size) (B)
- **J**: [`jobscript`](#runner-settings) (runner)
- **K**: [`keep_structures`](#c-screening) (C) · [`kspacing`](#a-host) (A)
- **L**: [`level_margin`](#c-screening) (C)
- **M**: [`mace`](#mace) (MACE) · [`mace_device`](#mace) (MACE) · [`mace_fmax`](#mace) (MACE) · [`mace_model`](#mace) (MACE) · [`mace_steps`](#mace) (MACE) · [`mace_threads`](#mace) (MACE) · [`max_cores`](#runner-settings) (runner) · [`max_jobs`](#runner-settings) (runner) · [`max_nodes`](#runner-settings) (runner) · [`max_restarts`](#runner-settings) (runner) · [`mem_per_core`](#runner-settings) (runner) · [`mem_replicated`](#runner-settings) (runner) · [`mem_safety`](#runner-settings) (runner) · [`mp_api_key`](#a-host) (A)
- **N**: [`name`](#general) (general) · [`native_pairs`](#b-candidates-and-cell-size) (B)
- **O**: [`omp_threads`](#runner-settings) (runner) · [`%block oxidation_states`](#c-screening) (C)
- **P**: [`pair_max_distance`](#b-candidates-and-cell-size) (B) · [`pairs`](#b-candidates-and-cell-size) (B) · [`param`](#general) (general) · [`%block partitions`](#runner-settings) (runner) · [`%block phases`](#a-host) (A) · [`phases_e_above_hull`](#a-host) (A) · [`phases_from`](#a-host) (A) · [`poll`](#runner-settings) (runner) · [`%block production_overrides`](#d-production) (D)
- **Q**: [`quench_temperature`](#e-thermodynamics) (E)
- **R**: [`rattle_dmin`](#c-screening) (C) · [`rattle_stdev`](#c-screening) (C) · [`require_bound`](#c-screening) (C) · [`require_gap_states`](#c-screening) (C) · [`require_spin`](#c-screening) (C) · [`%block runner_<group>`](#runner-settings) (runner)
- **S**: [`scan`](#b-candidates-and-cell-size) (B) · [`scan_sizes`](#b-candidates-and-cell-size) (B) · [`scan_tol`](#b-candidates-and-cell-size) (B) · [`screen_charges`](#c-screening) (C) · [`screen_cutoff`](#c-screening) (C) · [`screen_max_steps`](#c-screening) (C) · [`%block screen_overrides`](#c-screening) (C) · [`screen_spins`](#c-screening) (C) · [`series_charge`](#b-candidates-and-cell-size) (B) · [`series_choice`](#b-candidates-and-cell-size) (B) · [`series_defect`](#b-candidates-and-cell-size) (B) · [`series_sizes`](#b-candidates-and-cell-size) (B) · [`series_tol`](#b-candidates-and-cell-size) (B) · [`share_rounds`](#c-screening) (C) · [`sp_walltime`](#runner-settings) (runner) · [`status_cmd`](#runner-settings) (runner) · [`submit_cmd`](#runner-settings) (runner) · [`supercell`](#b-candidates-and-cell-size) (B)
- **T**: [`tolerate_failures`](#runner-settings) (runner) · [`transform`](#general) (general) · [`triage`](#c-screening) (C) · [`triage_window`](#c-screening) (C)
- **V**: [`vacancies`](#b-candidates-and-cell-size) (B)
- **W**: [`walltime`](#runner-settings) (runner) · [`walltime_safety`](#runner-settings) (runner) · [`workdir`](#general) (general)
- **Z**: [`zpl_window`](#c-screening) (C)

Groups: [General](#general) · [MACE](#mace) · [A: host](#a-host) ·
[B: candidates and cell size](#b-candidates-and-cell-size) · [C: screening](#c-screening) ·
[D: production](#d-production) · [E: thermodynamics](#e-thermodynamics) ·
[Runner settings](#runner-settings)

#### General

| Keyword | Default | Steps | What it does |
|---|---|---|---|
| `host_cell` | required | A1, A2.1 | The host cell, normally the primitive cell (a conventional cell also works); A2.1 relaxes it. Everything in it apart from the structure and k-points (`species_pot`, `symmetry_generate`, any other `.cell` keyword) is copied into every job's `.cell`; defect cells lose the symmetry keywords. List the dopants in `species_pot` too, or leave the block out to use CASTEP's default on-the-fly potentials for every species. |
| `param` | required | A1 | The production settings: functional, SCF and geometry settings, smearing, and so on. A1 writes `production.param` from it, with the converged `cut_off_energy` and without any `charge` or `spin`, and every later job starts from that file. The scripts set `task`, and per job `charge`, `spin`, `spin_polarized` and the k-points. |
| `transform` | identity | B2, B3, B5 | The supercell shape: 9 integers, row by row, each row a supercell vector in terms of the primitive vectors (determinant > 0). Size n is n × `transform`. The example's matrix turns an FCC primitive cell into cubic cells of 8n³ atoms. Without it, size n is n × n × n primitive cells. Everything after B5 uses the chosen cell. |
| `energy` | `auto` | A1, A2.1, A3.1, B3, C2, C5, C6, D1 | Which CASTEP energy every step reads: `final` (Final energy), `free` (E − TS), `0K` (the 0 K estimate) or `auto` (the 0 K estimate when CASTEP prints one, else the final energy). They differ only with smearing. |
| `name` | the file name | — | The project's name: the default work directory is `dwf_<name>`, and it titles the E2 plot. |
| `workdir` | `dwf_<name>` | — | The work directory, relative to the project file. |

#### MACE

MACE runs on the machine where the script runs, usually a login node. It has no charge or spin:
it only prepares structures and never ranks charge states.

| Keyword | Default | Steps | What it does |
|---|---|---|---|
| `mace` | `true` | B2, B3, C1 | Use MACE for the size scan (B2), the pre-relaxation of each cell of the size series (B3), and the pre-relaxation and triage of the candidates (C1). With `false`, B2 is skipped, the series and C1 start from the unrelaxed structures, and `triage` does nothing. |
| `mace_model` | `medium-mpa-0` | B2, B3, B4, C1 | The model: the name of a `mace_mp` foundation model, or a model file. B4's `finetune.sh` starts from it. Changing it after B3 has run makes B3 run again, and `supercell_convergence.py` then stops because the series was set up with another model (`--redo series` starts the series again). To change only C1's model, use `finetuned_model`. |
| `finetuned_model` | — | C1 | A model file that C1 uses instead of `mace_model`, such as the one `finetune.sh` (B4) writes. A new file, or a new version of it, makes C1 run again. |
| `mace_fmax` | `0.05 eV/ang` | B2, C1 | A MACE relaxation stops when the largest force is below this. B3 passes it to the size series when the series is set up. |
| `mace_steps` | `500` | B2, C1 | ... or after this many optimiser (LBFGS) steps. B3 passes it on too. |
| `mace_device` | `cpu` | — | `cpu`, or `cuda` on a GPU. |
| `mace_threads` | `8` | — | Torch threads. Keep it modest on a shared login node. |

#### A: host

`dwf_host.py`; details in `README_dwf_host.md`.

| Keyword | Default | Steps | What it does |
|---|---|---|---|
| `converge` | `true` | A1 | Converge the plane-wave cut-off and k-point spacing first. With `false`, A1 runs nothing: `production.param` is your `.param` (without `charge` or `spin`), and the k-point spacing is `kspacing`, or that of the host `.cell`'s grid. |
| `converge_cutoffs` | `300 400 500 600 700 800 eV` (*list*) | A1 | The cut-offs tried: a single point of the host cell at each, on the finest k-grid (`kspacing`, or the smallest of `converge_kspacings`). |
| `converge_kspacings` | `0.1 0.08 0.06 0.05 0.04 0.03 1/ang` (*list*) | A1 | The k-point spacings tried, at the highest cut-off; spacings that give the same grid run once. Not used with `kspacing`. |
| `converge_tol` | `0.001 eV/atom` | A1 | A1 takes the smallest cut-off, and the largest spacing, whose energy per atom (and that of every finer value) is within this of the finest one. If only the finest one is, A1 stops and asks for higher cut-offs or smaller spacings; the finished single points are reused. |
| `kspacing` | from A1 | A1 | The production k-point spacing, in 1/Å as CASTEP's `kpoints_mp_spacing`. Every job without its own grid uses it. Setting it skips the k-point part of A1 (the cut-off is still converged). |
| `bands_kspacing` | half the production spacing | A2.2 | The spacing of the Γ-centred grid on which A2.2 finds the VBM, CBM and gap. Make it smaller if the band edges may lie between grid points. |
| `dielectric` | from A2.3 | A2.3 | The static dielectric constant (electronic + ionic), ε₀, for every charge correction (B3, C3, E1). Setting it skips the DFPT run. |
| `dielectric_inf` | — | A2.3 | The optical constant ε∞, recorded with `dielectric`. The corrections don't use it. |
| `dielectric_task` | `phonon+efield` | A2.3 | The CASTEP task of the DFPT run, at q = Γ. It must give the DC (static) permittivity: with `efield`, which gives only ε∞, A2.3 stops. |
| `dielectric_param` | — | A2.3 | A `.param` for the DFPT run instead of `production.param`. CASTEP's DFPT needs norm-conserving potentials, fixed occupancies and no spin polarisation. Without this keyword the production settings are used with `fix_occupancy : true` and `spin_polarized : false`; with it, put those in the file, with a cut-off suited to norm-conserving potentials. |
| `dielectric_cell` | — | A2.3 | A `.cell` whose `species_pot` (and other settings) the DFPT run uses instead of the host `.cell`'s, e.g. `NCP` for CASTEP's norm-conserving library. Its structure is ignored: the run uses the relaxed host. A2.3 warns before submitting if the potentials don't look norm-conserving. |
| `%block phases` | — | A3.1 | The competing phases, one per line: `name  structure-file  [options]`. Options: `molecule` (positions only, in its fixed box, at Γ, without symmetry); `metal` (`fix_occupancy : false` unless the line says otherwise; add `smearing_width=...`); `kspacing=0.03` (its own k-point spacing); and any other `keyword=value`, a `.param` keyword for that phase only (e.g. `spin=2 spin_polarized=true` for O₂). Solids relax lattice and positions, with symmetry, at the production settings. A3.2 needs a phase of each pure element of the host, and for each dopant at least one phase that contains it. |
| `phases_from` | `none` | A3.1 | `doped`: when there is no `%block phases`, take the competing phases from the Materials Project through doped (needs doped and `mp_api_key`). |
| `mp_api_key` | — | — | Materials Project API key for `phases_from : doped` (or set it in doped's configuration). |
| `phases_e_above_hull` | `0.05 eV/atom` | A3.1 | `phases_from : doped` takes the Materials Project entries up to this far above the hull. |
| `%block chemical_potentials` | — | A3.1, A3.2 | The limits themselves, one per line: `name  El=value ...`, in absolute eV per atom on the energy scale of your CASTEP runs, for every element of the host and the dopants. A3.1 then relaxes nothing, and A3.2 records them. It takes precedence over `%block phases` and `phases_from`. |

#### B: candidates and cell size

`dwf_candidates.py`; details in `README_dwf_candidates.md`.

| Keyword | Default | Steps | What it does |
|---|---|---|---|
| `dopants` | — (*list*) | A3.1, A3.2, B1 | The extrinsic elements. B1 puts each on every inequivalent host site, and on the interstitial sites with `interstitials`; A3 limits their chemical potentials; C4 guesses their oxidation states. |
| `%block defects` | — | B1 | Your own defects, one per line, components joined by `+`: `Sb_Ca` (Sb on a Ca site), `Vac_O`, `Vac_O:2` (the 2nd O site of the primitive cell), `Sb_Ca@2` (starting spin 2 on the Sb). Each later component goes on its nearest site of that kind to the first one; with `all` on the line (pairs only), every arrangement up to `pair_max_distance` becomes a candidate. With this block, nothing is generated. |
| `vacancies` | `true` | B1 | A vacancy on each inequivalent host site. |
| `antisites` | `false` | B1 | Each other host element on each inequivalent host site. |
| `interstitials` | `none` | B1 | `dopants`: the dopants on each Voronoi interstitial site (pymatgen-analysis-defects); `all`: every element there. |
| `pairs` | `true` | B1 | Pairs of a dopant defect with a native one, in every symmetry-distinct arrangement up to `pair_max_distance` apart. Never two interstitials. |
| `native_pairs` | `false` | B1 | Also pairs of two native defects. |
| `dopant_pairs` | `false` | B1 | Also pairs of two dopant defects. |
| `pair_max_distance` | `5.0 ang` | B1 | The largest separation of the pair arrangements, generated or from `all`. In C1, a pair farther apart than half the shortest image distance of the chosen cell is left out. |
| `scan` | `true` | B2 | Run the MACE size scan (also needs `mace : true`). |
| `scan_sizes` | `2 3 4 5` (*list*) | B2 | The sizes n the scan tries, smallest first. |
| `scan_tol` | `0.02 eV` | B2 | The scan stops at the first size whose MACE relaxation energy is within this of the size before. That size caps the CASTEP series; if the energy never levels off, there is no cap. |
| `series_defect` | see text | B2, B3 | The defect of the size scan and series, as tokens like those of `%block defects` (e.g. `Sb_Ca@1 + Vac_O`). By default, the nearest arrangement of the first pair that contains a dopant, else the first candidate. It can't contain an interstitial. |
| `series_charge` | `0` | B3 | The charge of the series defect. If it isn't 0, the series corrects E_f* with `charge_correction` and the A2.3 dielectric constant; a charged E_f* converges more slowly with size. |
| `series_sizes` | `2 3 4` (*list*) | B3 | The sizes of the CASTEP series: those up to the B2 cap, and at least two. |
| `series_tol` | `0.05 eV` | B3 | The series has converged when E_f* changes by less than this from one size to the next. If it doesn't converge, B5 stops: add sizes and rerun. |
| `series_choice` | `smaller` | B3 | Which of the two sizes that agreed becomes the cell from B5 on: `smaller` or `larger`. |
| `supercell` | — | B2, B3, B5 | Use size n from B5 on and skip the scan and the series. |
| `finetune` | `false` | B4 | Write MACE training data from the series' smallest-cell relaxation (`finetune/train.xyz`) and a script that fine-tunes `mace_model` on it (`finetune/finetune.sh`). Needs the series, so not with `supercell`. |
| `finetune_args` | — | B4 | Extra options added to the `mace_run_train` command in `finetune.sh`. |

#### C: screening

`dwf_screen.py`; details in `README_dwf_screen.md`. The screening runs are cheap on purpose:
Γ point only, possibly a lower cut-off, few geometry steps. Their energies rank candidates;
phase D redoes the survivors properly.

| Keyword | Default | Steps | What it does |
|---|---|---|---|
| `triage` | `false` | C1 | Before any CASTEP run, drop arrangements whose MACE energy is more than `triage_window` above the lowest of their composition group (needs `mace : true`). |
| `triage_window` | `1.0 eV` | C1 | How far above the group's lowest MACE energy an arrangement may be. MACE has no charge or spin, so keep it generous. |
| `screen_charges` | `-1 0 1` (*list*) | C2 | The charge states every candidate is screened in. |
| `screen_spins` | `both` | C2 | The starting spins: `low` is the lowest the electron count allows (0 or 1); `both` adds that + 2. They are starting values: the results keep the final spin. |
| `screen_cutoff` | the production cut-off | C2, C5, C6 | The cut-off of the screening runs. |
| `screen_max_steps` | `50` | C2, C5, C6 | `geom_max_iter` of the screening relaxations. A run that reaches it is logged as failed but keeps its last energy, marked as not converged. |
| `%block screen_overrides` | — | C2, C5, C6 | Any other `.param` keywords for the screening runs, e.g. a looser `elec_energy_tol`. |
| `level_margin` | `0.1 eV` | C2, C5, C6 | How far inside the host's band edges (from the pristine cell at Γ) an eigenvalue must lie to count as a defect level in the gap. |
| `hull_margin` | `0.2 eV` | C3 | Keep an arrangement if its lowest formation energy comes within this of its composition group's defect hull somewhere in the gap. Allow for the error of the screening settings. |
| `require_gap_states` | `none` | C3 | Keep only defects with `occupied` gap levels, `empty` ones, or `both` (an occupied and an empty level in the same spin channel: an internal transition), in a stable charge state. |
| `require_spin` | — | C3 | Keep only defects with this final spin (unpaired electrons, within 0.25) in a stable charge state, e.g. `2` for a triplet. |
| `require_bound` | `false` | C3 | Keep only pairs that are bound against their components somewhere in the gap: E_b = Σ E_f(parts) − E_f(pair) > 0. The components must be candidates too. |
| `distort_charges` | `stable` | C3 | The charge states that go on to the distortions: `stable` (lowest in energy for that defect somewhere in the gap) or `screened` (every one with a usable energy). |
| `zpl_window` | — | C3b | `lowest highest` in eV: keep only defects whose first ZPL estimate (the smallest step from an occupied to an empty gap level in one spin channel) lies inside. Without it, C3b keeps every survivor. |
| `distortions` | `-0.6 -0.5 … 0.5 0.6` (*list*) | C4 | ShakeNBreak bond distortions of the defect's nearest neighbours: `-0.4` brings them 40 % closer, `0.4` takes them 40 % further. One relaxation per factor, charge state and survivor. |
| `distort_around` | `anchor` | C4 | The centre of the distortion: `anchor` (the first component's atom) or `vacancy` (the vacant site, for defects that have one). |
| `%block oxidation_states` | guessed | C4 | One `El  value` per line, for ShakeNBreak's count of neighbours to distort: n = q − Σ (oxidation state put there − that of the site), and that many neighbours (the absolute value), or 8 minus it above 4. Without the block, pymatgen guesses the host's, and each dopant takes its common oxidation state closest to that of the element it replaces. |
| `rattle_stdev` | 10 % of the host bond | C4 | The standard deviation (Å) of the random displacements of the other atoms. |
| `rattle_dmin` | 80 % of the host bond | C4 | The shortest distance (Å) between atoms that rattling may produce. |
| `distortion_min_gain` | `0.05 eV` | C6 | A distortion counts as found when it beats the screened structure by this. |
| `share_rounds` | `2` | C6 | How many rounds of trying a winning distortion in the other charge states of its defect. |
| `keep_structures` | `1` | C6 | How many distinct structures, lowest first, of each defect and charge state go on to phase D. |

#### D: production

`dwf_production.py`; details in `README_dwf_production.md`.

| Keyword | Default | Steps | What it does |
|---|---|---|---|
| `%block production_overrides` | — | D1 | `.param` keywords for the production runs (the relaxations and the pristine single point), e.g. `geom_energy_tol`, `geom_force_tol`, `geom_max_iter`, `elec_energy_tol`. A relaxation that reaches `geom_max_iter` is flagged and left out of E, so raise it here. States whose spin is 0 run without spin polarisation unless this block sets `spin_polarized`. |
| `freeze_radius` | `0` (off) | D1 | D2, an untested idea: hold every atom further than this (Å) from all of the defect's components fixed (`%block ionic_constraints`). Compare with a full relaxation before relying on it. |

#### E: thermodynamics

`dwf_thermo.py`; details in `README_dwf_thermo.md`.

| Keyword | Default | Steps | What it does |
|---|---|---|---|
| `charge_correction` | `lz` | B3, C3, E1 | The image-charge correction for charged defects, with ε₀ from A2.3: `lz` (Lany–Zunger), `mp` (first-order Makov–Payne) or `none`. Used for a charged size series, the screening hull and the final formation energies. |
| `gap` | from A2.2 | E2 | The band gap for E2 and E3: the range of Fermi levels, the window in which transition levels are found, E_f at the CBM, and the free carriers (the conduction bands are shifted by the difference). Put an experimental or hybrid value here; the defect energies themselves are not corrected. |
| `fermi_temperature` | — | E2 | Find the self-consistent Fermi level, and the defect and carrier concentrations, at this temperature (K), at each chemical-potential limit. Only the defects that reached phase D are included. |
| `quench_temperature` | — | E2 | With `fermi_temperature`: freeze each defect's total concentration at `fermi_temperature`, then share it out between its charge states at this temperature (K). |
| `deep_level_margin` | `0.5 eV` | E3 | A transition level at least this far from both band edges counts as deep, and its defect goes on to phase F. |

#### Runner settings

These are `supercell_convergence.py`'s keywords for how jobs run (see
`README_supercell_convergence.md` for the details). In the project file they apply to every
group of steps; `%block runner_<group>` changes them for one group, holding runner keywords only:

| Group | Steps | Defaults that differ |
|---|---|---|
| `host` | A1–A3 | — |
| `series` | B3 | — (written into `series.conv`; `max_jobs` and `tolerate_failures` don't apply) |
| `screen` | C2, C5, C6 | `max_restarts : 0`, `tolerate_failures : true` |
| `production` | D1 | `tolerate_failures : true` |

These group defaults take precedence over the project-wide settings, and a
`%block runner_<group>` setting over both.

| Keyword | Default | What it does |
|---|---|---|
| `jobscript` | required | Your jobscript template, as a file or as `%block jobscript`. It needs `{SEED}` where the seedname goes, and `{NCORES}`, or on whole-node machines `{NODES}`, `{TASKS_PER_NODE}` and `{CPUS_PER_TASK}`. Other placeholders: `{JOBNAME}`, `{PARTITION}`, `{NKPTS}`, `{MEM_GB}`, `{MEM_MB}`, `{MEM_PER_NODE_GB}`, `{MEM_PER_CORE_GB}`, `{MEM_PER_CORE_MB}`. The runner sets the partition and `--time` itself; the template's `--time` is the longest any job may run. |
| `cores_per_node` | — | Cores per node on a machine that hands out whole nodes (128 on ARCHER2). Jobs are then laid out as nodes × processes per node, chosen for k-point parallelism. |
| `max_nodes` | `1` | The most nodes a job may use (with `cores_per_node`). |
| `max_cores` | `32` | The most cores a job may use on a machine that hands out cores (without `cores_per_node`). |
| `cores` | — | An exact core count for every job. |
| `%block partitions` | — | One partition per line: `name  GB_per_core  [max_cores]  [#SBATCH options]`, e.g. `highmem  4  --qos=highmem`. A job goes on the first partition where it fits in memory with full nodes; if none fits, on the best layout with fewer processes per node. |
| `mem_per_core` | — | Without `%block partitions`: the memory per core (GB) jobs must fit in; a job that can't is warned about. |
| `mem_safety` | `1.05` | Factor on the memory estimated from the dry run. |
| `mem_replicated` | `auto` | The fraction of the dry run's memory that every process keeps its own copy of. `auto` starts at 0.25 and is measured from each finished job. |
| `baseline_mb` | — | Replaces CASTEP's baseline memory estimate (MB). |
| `omp_threads` | `1` | `auto` or a number: OpenMP threads per process on cores left idle when nodes are underpopulated for memory. |
| `walltime` | `auto` | `auto`: once jobs have finished, later ones ask for what they need (scaled from the measured time per SCF cycle) instead of the template's `--time`; `fixed`: always the template's. |
| `walltime_safety` | `1.5` | Factor on the walltime estimate. |
| `sp_walltime` | — | The walltime (`H:MM:SS`) of the first single points, before there is anything to estimate from. One that runs out is resubmitted with the full time. |
| `max_restarts` | `10` (`0` for `screen`) | How many times a job that runs out of time, or stops in a way the runner can recover from, is resubmitted; a relaxation carries on from its `.check`. |
| `delete_check` | `false` | Delete each calculation's `.check` file once it has finished; they can be large. |
| `dryrun_on` | `compute` | Where the dry runs go (one per distinct input, for the k-points and memory): `compute`, as small batch jobs from your template; `login`, where the script runs, with `dryrun_cmd`. On ARCHER2, don't send them to the shared `serial` partition with a whole-node template, which it rejects: use `login`, or `dryrun_partition : standard`. |
| `dryrun_cmd` | `castep.mpi --dryrun {seed}` | The command for `dryrun_on : login`, e.g. `castep.serial --dryrun {seed}`. |
| `dryrun_timeout` | `900 s` | How long a login dry run may take. |
| `dryrun_jobscript` | the jobscript | A separate template for dry-run jobs. |
| `dryrun_partition`, `dryrun_time`, `dryrun_options` | — | The partition, time and other `#SBATCH` options of dry-run jobs. |
| `dryrun_cores`, `dryrun_mem_gb` | `1`, `4` | The cores and memory (GB) of a dry-run job. |
| `submit_cmd` | `sbatch` | The command that submits a job. |
| `status_cmd` | `squeue` or `qstat` | The command that lists queued and running jobs. |
| `poll` | `120 s` | Seconds between checks of the queue. |
| `max_jobs` | no limit | The most jobs, and dry runs, of one step in the queue at once. |
| `tolerate_failures` | `false` (`true` for `screen` and `production`) | A job that fails is marked as failed and the others carry on, instead of the script stopping. `--resubmit-failed` starts failed jobs again. |

`--prepare-only`, `--no-wait`, `--resubmit-failed` and `--force` are command-line switches
([Command line](#command-line)), not project-file keywords.

### 2. The production `.param` (`param`)

Your own CASTEP `.param`. A1 copies it into `A_host/production.param`, and every later job
starts from that file. Any keyword not in this table (`xc_functional`, `elec_energy_tol`,
`smearing_width`, mixing, geometry tolerances, ...) is passed on to every job unchanged.

| Keyword | What the workflow does with it |
|---|---|
| `task` | Set for every job: `geometryoptimization` for relaxations, `singlepoint` for single points, `dielectric_task` for the A2.3 run. Yours is ignored. |
| `cut_off_energy` | A1 sets it on each convergence run and writes the converged value into `production.param`, replacing yours. With `converge : false`, yours is kept. The screening runs (C2, C5, C6) use `screen_cutoff` if it is set. |
| `basis_precision` | Removed by A1 when it sets `cut_off_energy`. |
| `charge` | Set per job, to the charge state; removed from `production.param` and from the host, competing-phase and pristine-supercell jobs. Leave it out. |
| `spin` | The starting total spin (unpaired electrons), set per defect job: from `screen_spins` in C2, from the final spin of the screening run in C5, C6 and D1, and from the electron count in B3. Removed from `production.param` and from the host and phase jobs (unless a phase line sets it). A starting value only: with `fix_occupancy : false`, CASTEP holds it for `spin_fix` SCF cycles and then lets it relax. |
| `spin_polarized` | `true` on every screening run (C2, C5, C6), on D1 runs with a non-zero spin, and on charged or odd-electron size series (B3); `false` for D1's spin-0 states unless `%block production_overrides` sets it, and for the A2.3 run without `dielectric_param`. The host and phase jobs use yours. |
| `fix_occupancy` | `true` for the A2.3 run without `dielectric_param`, `false` for `metal` phases; otherwise yours. |
| `geom_max_iter` | `screen_max_steps` on the screening relaxations; yours, or `%block production_overrides`, elsewhere. A production relaxation that reaches it is flagged and left out of E. |
| `write_cell_structure` | Set to `true` on relaxations: their final structure is read from `<seed>-out.cell`. |
| `run_time` | Set on relaxations, a little below the job's walltime, so CASTEP stops cleanly and writes its `.check` before the time limit (not on screening runs, which are not restarted). |
| `continuation` | Yours is removed. Set to `default` when a relaxation is resubmitted to carry on from its `.check`. |
| `reuse` | Removed. |

`%block screen_overrides` and `%block production_overrides` (project file) set further keywords
for the screening and production runs, and options on a `%block phases` line set them for that
phase.

### 3. The host `.cell` (`host_cell`)

The structure is read once (A2.1 relaxes it); every job gets its own structure and k-points,
and everything else in the file is copied into every job's `.cell`.

| Keyword | What the workflow does with it |
|---|---|
| `%block lattice_cart`, `lattice_abc` | The host lattice, read once; replaced in every job. |
| `%block positions_frac`, `positions_abs` | The host atoms, read once; replaced in every job. |
| `kpoints_mp_grid`, `kpoints_mp_spacing` (or `kpoint_…`) | A fallback only: with `converge : false` and no `kspacing`, the production spacing is taken from them. Every job gets its own `kpoint_mp_grid`, and A2.2 a Γ-centring `kpoint_mp_offset`. |
| `%block species_pot` | Copied into every job, so list the dopants too; without it, CASTEP generates its default on-the-fly potentials for every species. `dielectric_cell`'s replaces it for A2.3. |
| `symmetry_generate`, `symmetry_tol`, `snap_to_symmetry` | Kept for the host, competing-phase solids and pristine supercells; removed for defect cells and molecules, which run without symmetry. |
| `fix_all_cell` | Set for every job: `true`, except in the variable-cell relaxations (A2.1 and the competing-phase solids). |
| `%block kpoints_list`, `symmetry_ops`, `ionic_constraints`, `ionic_velocities`, `cell_constraints` | Dropped: they refer to the primitive cell's k-points or atoms. (D2 writes its own `ionic_constraints`.) |
| `SPIN=` on an atom | Written per job on the atom that carries the starting spin (the defect's first component). |
| any other keyword or block | Copied unchanged into every job (e.g. `species_mass`, `hubbard_u`, `external_pressure`). |

### 4. The dielectric `.param` and `.cell` (`dielectric_param`, `dielectric_cell`)

For the A2.3 DFPT run only. CASTEP's DFPT needs norm-conserving pseudopotentials, fixed
occupancies and no spin polarisation.

| File | What it must hold | What the workflow does with it |
|---|---|---|
| `dielectric_param` | `fix_occupancy : true`; no `spin_polarized : true`; a `cut_off_energy` suited to norm-conserving potentials (check that ε has converged with it); no `charge` or `spin` | Used in full as the run's `.param`, with `task` set to `dielectric_task`. Nothing else is added or removed. |
| `dielectric_cell` | `%block species_pot` with norm-conserving potentials, e.g. the single line `NCP` (CASTEP's on-the-fly norm-conserving library) | Read like `host_cell`: its structure and k-points are ignored (the run uses the relaxed host and the production spacing), the rest is copied into the run's `.cell`. A2.3 adds `%block phonon_kpoint_list` (q = Γ) for phonon tasks. |

### 5. Competing-phase structure files (`%block phases`)

Each line of `%block phases` is `name  structure-file  [options]`. The file can be a `.cell`
(lattice and positions are read) or any format ASE reads (`.cif`, `POSCAR`, ...); only the
structure is used. The settings come from `production.param`, the host `.cell`'s `species_pot`
and other keywords, and the line's options:

| Option | Effect |
|---|---|
| `molecule` | Relax the positions only, in the file's own box, at the Γ point, without symmetry (make the box big enough and slightly asymmetric). |
| `metal` | `fix_occupancy : false` unless the line sets it; also give `smearing_width=...` and usually a finer `kspacing=`. |
| `kspacing=0.03` | This phase's own k-point spacing (1/Å) instead of the production one. |
| `keyword=value` | Any other `.param` keyword, for this phase only, e.g. `spin=2 spin_polarized=true` for O₂. |

Solids relax lattice and positions, with symmetry.

### 6. The jobscript template (`jobscript`)

Your usual CASTEP jobscript with placeholders. The runner fills them in for each job:

| Placeholder | Replaced by |
|---|---|
| `{SEED}` | the job's seedname (required) |
| `{NCORES}` | the number of MPI processes (core-based machines) |
| `{NODES}`, `{TASKS_PER_NODE}`, `{CPUS_PER_TASK}` | the layout on whole-node machines (with `cores_per_node`) |
| `{JOBNAME}` | the job's name |
| `{NKPTS}` | the number of k-points |
| `{PARTITION}` | the partition chosen |
| `{MEM_GB}`, `{MEM_MB}` | the estimated total memory |
| `{MEM_PER_NODE_GB}` | the estimated memory per node |
| `{MEM_PER_CORE_GB}`, `{MEM_PER_CORE_MB}` | the estimated memory per process |

The runner also sets `#SBATCH --partition` (with the partition line's extra options) and
`--time`, replacing the template's in either spelling; the template's `--time` is the longest
any job may run. It sets `OMP_NUM_THREADS`, and with threads adds `OMP_PLACES`,
`SRUN_CPUS_PER_TASK` and `--cpus-per-task` if they are missing. A dry-run job (`dryrun_on :
compute`) is made from the same template (or `dryrun_jobscript`), with the CASTEP line run as
one process with `--dryrun`.

### 7. `series.conv` (written by B3)

B3 writes `B_candidates/series/series.conv`, a `supercell_convergence.py` input, from the
project. Don't edit it: B3 writes it again on every run. It holds:

| Keyword | Value |
|---|---|
| `cell` | `A_host/host_relaxed.cell` (the relaxed host from A2.1) |
| `param` | `series.param`: `production.param` with the series' `charge`, `spin_polarized` and `spin` |
| `%block defect` | the representative defect's tokens (`series_defect`, or the default), with `@s` on the first |
| `e_bulk` | A2.1's energy per atom |
| `sizes` | `series_sizes`, up to the B2 cap |
| `kspacing` | the production spacing (A1) |
| `tol` | `series_tol` |
| `singlepoint` | `false` |
| `energy` | `energy` |
| `workdir` | `run` |
| `%block transform` | `transform` |
| `charge_correction`, `dielectric` | with a non-zero `series_charge`: `charge_correction` and A2.3's ε₀ |
| `prerelax`, `prerelax_model`, `prerelax_device`, `prerelax_threads`, `prerelax_fmax`, `prerelax_steps` | with `mace : true`: `mace` and the `mace_*` keywords |
| runner keywords | the project's, with `%block runner_series` (not `max_jobs` or `tolerate_failures`) |

### 8. `supercell_convergence.py` on its own: `.conv` and batch files

Outside the workflow, `supercell_convergence.py` reads its own input files; the full description
is in `README_supercell_convergence.md`. Its runner keywords are those under
[Runner settings](#runner-settings) above (`max_jobs` and `tolerate_failures` only in batch
mode). The others:

**Size-series input (`<seed>.conv`)**

| Keyword | Default | What it does |
|---|---|---|
| `cell` | required | The relaxed host cell (primitive or conventional): structure, `species_pot`, k-points and anything else every supercell copies. |
| `param` | required | The `.param` of the single points, and of the relaxations unless `geom_param` is set. The script sets `task`. For a charged defect, put `charge` in it (and in `geom_param`). |
| `geom_param` | — | A separate `.param` for the relaxations; a single point with `param` then always follows each relaxation. |
| `jobscript` | required | The jobscript template, or `%block jobscript`. |
| `%block defect` | required | The defect, one token per line or joined by `+`: `Sb_Ca`, `Vac_O`, `Vac_O:2`, `Sb_Ca@3`, `23_Ca` (V on a Ca site). The first sits as close to the cell centre as the lattice allows, the others on their nearest sites to it. |
| `e_bulk` | — (eV/atom) | The host's energy per atom, at the same settings; required with `bulk_mode : peratom`. |
| `tol` | `0.05 eV` | Converged when E_f* changes by less than this between consecutive sizes. |
| `sizes` | `2 3 4 5` | The sizes n tried, smallest first. |
| `%block transform` | identity | The supercell shape: size n is n × this 3 × 3 integer matrix. |
| `kgrid` | from `cell` | A primitive-cell Monkhorst–Pack grid to take the k-point spacing from. |
| `kspacing` | from `cell` | The k-point spacing itself (1/Å). |
| `energy` | `auto` | `auto`, `final`, `free` or `0K`, as in the project file. |
| `bulk_mode` | `peratom` | `peratom`: E_host = N × `e_bulk`; `supercell`: also run every pristine supercell and use its energy. |
| `singlepoint` | `true` | `true`: a single point on each relaxed cell gives the energy; `false`: the energy at the end of the relaxation. |
| `charge_correction` | `lz` | `lz`, `mp` or `none`, for a charged defect. |
| `dielectric` | — | The static dielectric constant; required for a charged defect unless `charge_correction : none`. |
| `workdir` | `conv_<cell>_<defect>` | The work directory. |
| `prerelax` | `none` | `mace`: relax each defect cell with MACE (positions only) before CASTEP. |
| `prerelax_model` | `medium-mpa-0` | The MACE model (name or file). |
| `prerelax_fmax`, `prerelax_steps` | `0.05 eV/ang`, `500` | When the MACE relaxation stops. |
| `prerelax_rattle` | `0` | Random displacements (Å) before the MACE relaxation, to break symmetry. |
| `prerelax_device`, `prerelax_threads` | `cpu`, `8` | Where MACE runs. |
| `prepare_only`, `no_wait`, `resubmit_failed`, `force` | `false` | The command-line switches, which can also go in the file. |

**Batch input (`--batch <file>`)**

| Keyword | Default | What it does |
|---|---|---|
| `cell` | required | The `.cell` every job copies `species_pot` and its other settings from (not its structure or k-points). |
| `param` | required | The `.param` every job starts from (the script sets `task`). |
| `geom_param` | — | A separate `.param` for the `kind=geom` jobs. |
| `kspacing` | — | The k-point spacing (1/Å) for jobs without `kgrid=`. |
| `energy`, `workdir` | `auto`, `batch_<file name>` | As above. |
| `%block jobs` | required | One job per line: `name  structure-file  [option=value ...]`. Options: `kind=geom` or `sp`; `task=...` (e.g. `phonon+efield`); `fix_cell=false` (variable cell); `symmetry=true`; `kgrid=1,1,1`; `offset=gamma` (or three numbers); and any other `keyword=value`, a `.param` keyword for that job (`charge=-1`, `spin=2`). |

---

## Testing so far, and what to check first

Everything has been run against a mock CASTEP and a mock Slurm, with the real MACE
(medium-mpa-0), ShakeNBreak 3.4.4, pymatgen and spglib, on CaO:Sb. The mock gives made-up
energies, spins and band structures, so these runs check the workflow's logic, not its physics.

| Test project | What it covered |
|---|---|
| `CaO_Sb` | A to E with MACE: cut-off and k-point convergence, DFPT, competing phases and limits, generated candidates, MACE scan, size series, fine-tuning data (B4), `require_gap_states`, `zpl_window`, `hull_margin`, distortions and sharing, Fermi level with a quench |
| `Final` | B to E again on the final code: the centred defect placement, `triage`, a MACE model file for C1 (`finetuned_model`) |
| `Opts2` | B to E with `mace : false`, antisites, native and dopant pairs, a charged size series, `series_choice : larger`, `screen_cutoff`, `%block screen_overrides`, `require_spin`, `require_bound`, `distort_charges : screened`, `distort_around : vacancy`, `%block oxidation_states`, `keep_structures : 2`, `%block production_overrides`, `freeze_radius`, `charge_correction : mp`, `gap`, `fermi_temperature` |
| `Opts` | phase A with `converge : false`, `kspacing`, `dielectric`, `%block chemical_potentials`; `supercell` |
| `User` | Voronoi interstitials (`interstitials : dopants`), through screening |
| `Async` | `--no-wait`, rerun until the steps finish |

Also tested: stopping (STOP file, Ctrl-C, a killed script) and rerunning; changed settings
(only the steps that use them run again, and finished jobs are reused); `--redo`; failed jobs
and `--resubmit-failed`; shared dry runs; `max_jobs`; `%block defects` with `all`, `:k` and
`@s` (the candidate list and defect cells); the example folder's phase A, with dry runs on the
login node; and the defect placement on CaO (primitive and conventional cells), wurtzite GaN
and a monoclinic cell, against a brute-force search for the most central position.
`supercell_convergence.py`'s own regression tests give the same results as before; the only
new output is the `off-centre/A` column of its size table.

Not tested: `phases_from : doped` (no Materials Project access here), a real fine-tuned model
(`finetune.sh` is written but was not run), and real CASTEP output for the DFPT permittivity
and the `.bands` of spin-polarised runs.

On the first real runs, check:

1. **A2.3 dielectric:** CASTEP's DFPT needs norm-conserving pseudopotentials, fixed occupancies
   and no spin polarisation. Give `dielectric_cell` (with an NCP `species_pot`) and
   `dielectric_param`, or set `dielectric` yourself. The script warns if the potentials don't
   look norm-conserving.
2. **The `.bands` reader** (A2.2, C2): the electron count per spin channel of a spin-polarised
   run is taken from the header when CASTEP gives one number per channel, otherwise from the
   Fermi energy. Compare the gap levels in `C_screen/screen.json` with the `.bands` of one run.
3. **The screening spins** (C2): each run starts from a total spin (`spin`) and SPIN= on the
   dopant. With `fix_occupancy : false` (smearing) CASTEP holds it only for its first `spin_fix`
   SCF cycles (10 by default) and then lets it relax, so both starting spins can end in the
   same state; the results record the final spin. With `fix_occupancy : true` the spin is not
   freed.
4. **Gap levels** are compared with the host's band edges at the same settings without
   potential alignment: fine for sorting candidates, not for production levels (those come
   from E2).
5. **The permittivity table** (A2.3) is read from the lines after "Optical Permittivity
   (f->infinity)"; check `--status` shows the values CASTEP printed.
