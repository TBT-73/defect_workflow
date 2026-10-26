# Defect workflow scripts (dwf_*.py)

Five scripts run the defect screening pipeline from a relaxed host to formation energies and
transition levels, one script per phase of the flowchart (`defect_workflow_flowchart_A1.pdf`):

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
- `supercell_convergence.py`: the job runner. Every CASTEP job of every phase goes through its
  batch mode (dry runs, layouts, walltimes, restarts), and phase B runs its size series. See its
  own `README.md`.

Each script has its own README: `README_dwf_core.md`, `README_dwf_host.md`,
`README_dwf_candidates.md`, `README_dwf_screen.md`, `README_dwf_production.md`,
`README_dwf_thermo.md`.

---

## Contents

1. [Requirements](#requirements)
2. [Quick start](#quick-start)
3. [The project file](#the-project-file)
4. [Checkpoints, stopping and rerunning](#checkpoints-stopping-and-rerunning)
5. [Command line](#command-line)
6. [The work directory](#the-work-directory)
7. [Flowchart steps and files](#flowchart-steps-and-files)
8. [Keyword reference](#keyword-reference)
9. [Testing so far, and what to check first](#testing-so-far-and-what-to-check-first)

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

```bash
python3 dwf_host.py --example > CaO_Sb.dwf      # annotated project file: edit it
python3 dwf_host.py CaO_Sb.dwf --prepare-only   # check the first inputs, submit nothing
nohup python3 dwf_host.py CaO_Sb.dwf > A.log 2>&1 &
```

then, each when the one before has finished:

```bash
nohup python3 dwf_candidates.py CaO_Sb.dwf > B.log 2>&1 &
nohup python3 dwf_screen.py     CaO_Sb.dwf > C.log 2>&1 &
nohup python3 dwf_production.py CaO_Sb.dwf > D.log 2>&1 &
python3 dwf_thermo.py CaO_Sb.dwf                 # quick: no CASTEP jobs
```

or all in one go: `nohup sh -c 'python3 dwf_host.py CaO_Sb && python3 dwf_candidates.py CaO_Sb
&& python3 dwf_screen.py CaO_Sb && python3 dwf_production.py CaO_Sb && python3 dwf_thermo.py
CaO_Sb' > run.log 2>&1 &`. Each script exits with an error if the step it needs from an
earlier phase hasn't finished, so the chain stops at the first problem.

The scripts wait for their jobs (polling every `poll` seconds), so run them with `nohup` or in
`tmux`/`screen`. MACE runs where the script runs (the login node): keep `mace_threads` modest.

---

## The project file

One file, `<name>.dwf`, holds everything, written like a CASTEP `.param` file:
`keyword : value`, `%block name … %endblock name`, any case, `#` or `!` comments. File names
are relative to it. A misspelt keyword is rejected with the closest match suggested.

It has three kinds of setting:

1. **Project keywords** for each phase, e.g. `dopants`, `screen_charges`, `hull_margin`
   ([Keyword reference](#keyword-reference)).
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

## Keyword reference

Defaults in brackets. Each script's README explains the keywords it uses.

**General**

- `host_cell`, `param`: required. `workdir` [dwf_<file name>], `name`.
- `%block transform`: supercell shape; size n uses n·T (9 integers) [identity].
- `energy` [auto | final | free | 0K].
- `mace` [true], `mace_model` [medium-mpa-0], `mace_device` [cpu], `mace_threads` [8],
  `mace_fmax` [0.05 eV/Å], `mace_steps` [500].

**A: host** (`dwf_host.py`)

- `converge` [true], `converge_cutoffs` [300 400 500 600 700 800 eV], `converge_kspacings`
  [0.1 0.08 0.06 0.05 0.04 0.03 1/Å], `converge_tol` [0.001 eV/atom], `kspacing`.
- `bands_kspacing` [half the production spacing].
- `dielectric`, `dielectric_inf`, `dielectric_task` [phonon+efield], `dielectric_param`,
  `dielectric_cell`.
- `%block phases`, `phases_from` [none | doped], `mp_api_key`, `phases_e_above_hull`
  [0.05 eV/atom], `%block chemical_potentials`.

**B: candidates** (`dwf_candidates.py`)

- `dopants`, `%block defects`, `vacancies` [true], `antisites` [false], `interstitials`
  [none | dopants | all], `pairs` [true], `pair_max_distance` [5.0 Å], `native_pairs` [false],
  `dopant_pairs` [false].
- `scan` [true], `scan_sizes` [2 3 4 5], `scan_tol` [0.02 eV].
- `series_defect`, `series_charge` [0], `series_sizes` [2 3 4], `series_tol` [0.05 eV],
  `series_choice` [smaller | larger], `supercell`.
- `finetune` [false], `finetune_args`, `finetuned_model` (the fine-tuned model C1 uses).

**C: screening** (`dwf_screen.py`)

- `screen_charges` [-1 0 1], `screen_spins` [both | low], `screen_cutoff`, `screen_max_steps`
  [50], `%block screen_overrides`.
- `triage` [false], `triage_window` [1.0 eV].
- `hull_margin` [0.2 eV], `require_gap_states` [none | occupied | empty | both],
  `require_spin`, `require_bound` [false], `level_margin` [0.1 eV], `zpl_window`.
- `distortions` [-0.6 … 0.6 in 0.1 steps], `distort_charges` [stable | screened],
  `distort_around` [anchor | vacancy: the vacant site for defects that have one],
  `rattle_stdev`, `rattle_dmin`,
  `%block oxidation_states`.
- `distortion_min_gain` [0.05 eV], `share_rounds` [2], `keep_structures` [1].

**D: production** (`dwf_production.py`)

- `%block production_overrides`, `freeze_radius` [0].

**E: thermodynamics** (`dwf_thermo.py`)

- `charge_correction` [lz | mp | none], `gap`, `fermi_temperature`, `quench_temperature`,
  `deep_level_margin` [0.5 eV].

**Runner** (any phase; see `supercell_convergence.py`'s README)

- `jobscript` (required), `cores_per_node`, `max_nodes`, `max_cores`, `cores`,
  `%block partitions`, `mem_*`, `baseline_mb`, `omp_threads`, `walltime`, `walltime_safety`,
  `sp_walltime`, `max_restarts`, `delete_check`, `dryrun_*`, `submit_cmd`, `status_cmd`,
  `poll`, `max_jobs`, `tolerate_failures`, and `%block runner_<host|series|screen|production>`.

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
`@s` (the candidate list and defect cells); and the defect placement on CaO (primitive and conventional cells), wurtzite GaN and a
monoclinic cell, against a brute-force search for the most central position.
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
