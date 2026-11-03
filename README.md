# Defect workflow (dwf.py)

`dwf.py` runs the defect workflow of the flowchart (`defect_workflow_flowchart_A1.pdf`), phases
A to E, with one command: from a host crystal and its dopants to formation energies, transition
levels and a short list of defects for the optical calculations (phase F, not scripted yet).

Two files go in one directory:

- `dwf.py`: the workflow;
- `supercell_convergence.py`: runs the CASTEP jobs (dry runs, node layouts, walltimes,
  restarts). It has its own README, `README_supercell_convergence.md`.

The MACE relaxations (B2, C1) run in `dwf.py` itself, or as jobs on compute nodes that run
`dwf.py` as a worker (`mace_jobscript`).

`example_CaO_Sb/` is a complete example project.

---

## Contents

1. [Quick start](#quick-start)
2. [Requirements](#requirements)
3. [How a run works](#how-a-run-works)
4. [Command line](#command-line)
5. [Files](#files)
6. [The phases](#the-phases)
7. [Keyword glossary](#keyword-glossary)
8. [Testing so far, and what to check first](#testing-so-far-and-what-to-check-first)

---

## Quick start

With `DWF` the directory that holds `dwf.py` and `supercell_convergence.py`:

```bash
cd example_CaO_Sb
python3 $DWF/dwf.py CaO_Sb.dwf --prepare-only     # write the first jobs' inputs, submit nothing
nohup python3 $DWF/dwf.py CaO_Sb.dwf > run.log 2>&1 &
python3 $DWF/dwf.py CaO_Sb.dwf --status           # which steps have finished
```

The run goes through every step, A1 to E3, and waits for its CASTEP jobs, so run it with
`nohup` (or in `tmux`/`screen`). If it stops for any reason, the same command carries on where
it stopped. Each phase's `summary.txt` shows its results as it goes, and
`E_thermo/summary.txt` has the formation energies and transition levels at the end.

The first time, stop after phase B (`--until B`) and read `B_candidates/summary.txt` before the
screening: the candidate list (each candidate costs about 2 × (number of `screen_charges`)
screening runs) and the screening cell B3 chose, with what set it.

The example folder holds the project file and every file it names:

| File | What it is |
|---|---|
| `CaO_Sb.dwf` | the project file, annotated (the same as `python3 dwf.py --example`) |
| `CaO_prim.cell` | the host primitive cell (rocksalt, a = 4.8105 Å); A2.1 relaxes it |
| `CaO.param` | production settings (PBE; A1 sets the cut-off) |
| `archer2.sh` | jobscript template for the CASTEP jobs |
| `archer2_mace.sh` | jobscript template for the MACE jobs (B2, C1) |
| `CaO_ncp.cell`, `CaO_ncp.param` | the DFPT dielectric run (A2.3): norm-conserving potentials, fixed occupancies |
| `Ca.cell`, `O2.cell`, `Sb.cell`, `Sb2O3.cell` | competing phases (A3): fcc Ca, O₂ in a box, A7 Sb, senarmontite Sb₂O₃, near their experimental structures |

Before running it, check: the account and CASTEP module in `archer2.sh`, and the account and
Python environment in `archer2_mace.sh`; the nodes and
walltimes of each step (the runner blocks in `CaO_Sb.dwf` are starting points); the
pseudopotentials (`CaO_prim.cell` has no `species_pot`, so CASTEP uses its default on-the-fly
ones); the cut-off in `CaO_ncp.param`; and the competing phases (add any other that could limit
Sb).

For a project of your own, `python3 dwf.py --example > <name>.dwf` prints the annotated project
file to start from.

---

## Requirements

- Python 3.9+ with `numpy`, `scipy` and `ase`;
- `spglib` (B1, B3: symmetry), `pymatgen` (A3 limits, C3/C6 structure matching, the
  oxidation-state guess in C4), `shakenbreak` (C4), `matplotlib` (E2 plot);
- `doped` (B2: the most compact supercell for each length; without it, n × n × n primitive
  cells, or give a `transform`);
- `mace-torch` for the MACE steps (B2, C1), or set `mace : false`: in the Python that runs
  `dwf.py` (it fetches a foundation model there) and in the one the MACE jobs run;
- optional: `pymatgen-analysis-defects` for Voronoi interstitials, a Materials Project API key
  for `phases_from : doped`;
- CASTEP and Slurm (other schedulers through `submit_cmd` / `status_cmd`).

On ARCHER2, put the environment on `/work` (the compute nodes, which run the MACE jobs, can't
see `/home`):

```bash
module load PrgEnv-gnu cray-python
python -m venv /work/e89/e89/$USER/dwfenv && source /work/e89/e89/$USER/dwfenv/bin/activate
pip install ase spglib pymatgen shakenbreak matplotlib mace-torch doped
```

Without `mace_jobscript` the MACE relaxations run where `dwf.py` runs: fine on a workstation,
but B2 relaxes every candidate in every size, so on ARCHER2 give `mace_jobscript` and let them
run on compute nodes.

---

## How a run works

**Order.** The steps run in order, A1 to E3 ([The phases](#the-phases) lists them). Each step
uses the results of the steps before it, so a step only starts once those have finished.

**Checkpoints.** When a step finishes, `state.json` in the work directory records it with a
fingerprint of everything it used: its settings (as written in the project file), the files it
read and the results of the steps it needs. A finished step with the same fingerprint is
skipped the next time.

**Stopping and carrying on.** Ctrl-C, `touch <workdir>/STOP`, `--no-wait`, a crash or a lost
login session: run the same command again. Finished steps are skipped and a step that was
part-way through carries on: submitted jobs keep running, and finished ones are picked up rather
than repeated.

**Changing a setting.** The steps that use it run again the next time, and the steps after them
too if their results change. CASTEP jobs whose own inputs have not changed are reused, so e.g.
raising `hull_margin` re-runs C3 onwards and submits only the jobs of the extra survivors. A job
that already ran with different inputs stops the run, so results from two sets of settings never
mix: start that step again with `--redo STEP` (its old folders are kept, renamed
`<name>.old-<time>`), or keep the old results with `--force`.

**The CASTEP jobs** all go through `supercell_convergence.py`, which plans each job (dry run,
nodes, memory, walltime), submits it, waits and restarts a relaxation that runs out of time.
Every step uses its batch mode, which runs the structures `dwf.py` builds.

**The MACE jobs.** With `mace_jobscript`, B2's and C1's MACE relaxations run as Slurm jobs made
from that template: its `{COMMAND}` line becomes `mace_pack` workers side by side (default
`cores_per_node` / `mace_threads`, 16 on ARCHER2), each running
`python3 dwf.py --mace-worker tasks_NN.json` with `mace_threads` threads, and the relaxations are
shared out between at most `mace_max_jobs` jobs, biggest first. Each result goes into its own
cache file as soon as it is done, so a job that runs out of time loses only the relaxations it
hadn't finished: they go into a new job (three times at most). The jobs wait in the queue, and
`--no-wait`, `--prepare-only` and the `STOP` file work, as for the CASTEP jobs. A foundation
model is downloaded where `dwf.py` runs and copied into `<workdir>/mace_models/`, where the
compute nodes can read it.

**Nodes and walltimes for each step.** The runner settings in the project file
([Runner settings](#runner-settings): nodes, walltime, partitions, ...) apply to every CASTEP
step. A runner block changes them for some steps only (the table below), and changes only the
keywords it gives. So the primitive-cell single points of A1 can run on one node for at most 20
minutes, while the supercell relaxations of D0 and D1 get 8 nodes and up to 24 hours:

```
max_nodes : 2                  # every step, unless a block says otherwise
%block runner_converge         # A1
max_nodes    : 1
max_walltime : 0:20:00
%endblock runner_converge
%block runner_production       # D0 and D1
max_nodes    : 8
max_walltime : 24:00:00
%endblock runner_production
```

Three blocks cover a whole phase, and the others one step each:

| Block | Applies to |
|---|---|
| `runner_host` | every CASTEP step of phase A: A1, A2.1, A2.2, A2.3 and A3.1 |
| `runner_converge` | A1 |
| `runner_relax` | A2.1 |
| `runner_bands` | A2.2 |
| `runner_dielectric` | A2.3 |
| `runner_phases` | A3.1 |
| `runner_screen` | every CASTEP step of phase C: C2, C2b, C5, C6 and C7 |
| `runner_dispersion` | C2b |
| `runner_drelax` | C5 |
| `runner_rank` | C6 |
| `runner_excite` | C7 (both of its batches) |
| `runner_production` | every CASTEP step of phase D: D0 and D1 |
| `runner_cells` | D0 |

You only need the blocks for the steps you want to change. A step uses the project-wide
settings, then `runner_host`, `runner_screen` or `runner_production` for its phase, then its own
block; the later one wins for any keyword both give. C2 and D1 have no blocks of their own: set
them with `runner_screen` and `runner_production`, and give the other steps their own blocks
where they should differ.

The screening steps (C2, C2b, C5, C6, C7) use `max_restarts : 0` and `tolerate_failures : true`
(a run that fails is flagged, not rescued), and D0 and D1 `tolerate_failures : true`, whatever
the project-wide settings say; a runner block can change them. The MACE jobs have settings of
their own (`mace_jobscript`, `mace_pack`, `mace_max_jobs`), and share only `submit_cmd`,
`status_cmd` and `poll` with the CASTEP jobs.

A block can also hold its own `%block partitions`, which replaces the whole list for those steps,
e.g. to send A1's jobs to a short-job QoS (`standard  2  --qos=short` on ARCHER2, for jobs of up
to 20 minutes). The settings of every step are checked when `dwf.py` starts. They don't change
any results, so changing them never makes a step run again: when you run `dwf.py` again, the
jobs it has not submitted yet, and the restarts of those already queued, use the new settings.

**Several small jobs on one node (`pack`).** ARCHER2 charges for whole nodes, however few cores
a job uses, and nodes are never shared between jobs. `pack : N` in a step's block runs N of that
step's CASTEP jobs side by side in one Slurm job on one node, each on 1/N of its cores: with
`pack : 4`, four jobs of 32 cores each. It suits steps with many small jobs, such as A1's single
points of the primitive cell:

```
%block runner_converge
max_nodes    : 1
max_walltime : 0:20:00
pack         : 4
%endblock runner_converge
```

- Each job is planned for its own share of the node as usual (processes for its k-points, memory
  from its dry run). Together the jobs get three-quarters of the node's memory, in proportion to
  their cores, which leaves the rest for the system. A job too big for its share runs on its own,
  as without `pack`.
- When fewer than N jobs are left, the last packs share them out evenly, and each job gets more
  of the node; a job left on its own gets whole nodes.
- The pack asks for the longest walltime of its jobs.
- A job that has to be restarted (a relaxation that ran out of time) runs on its own, on whole
  nodes. A job that fails doesn't stop the others in its pack from running.
- The pack's job script is `jobs/<step>/packs/pack_NNN/job.sh`. Each job still has its own
  folder, where it writes its output and `slurm-<job id>.out`; its `job_info.json` names the pack.
- It needs `cores_per_node` and a jobscript whose CASTEP line uses `srun` (each job of a pack is
  started with its own `srun --exact`).

The dry runs can be packed too, which matters with `dryrun_on : compute`, where each dry run is
otherwise a job of its own that waits in the queue and, on ARCHER2's standard partition, takes a
whole node. `dryrun_pack : N` (e.g. 32, for every step) runs up to N of a step's dry runs side
by side in one job, one process each with `dryrun_mem_gb` of memory, in
`jobs/<step>/packs/dryrun_NNN/job.sh`. With `dryrun_on : login` the dry runs don't go to the queue
at all, so it doesn't apply.

---

## Command line

```
python3 dwf.py <project>.dwf [STEP ...] [--until STEP] [--redo STEP] [--status] [--example]
                             [--prepare-only] [--no-wait] [--resubmit-failed] [--force]
```

| Option | Effect |
|---|---|
| `STEP ...` | run only these steps: names (`relax`), labels (`A2.1`) or phase letters (`C`); default all |
| `--until STEP` | stop after this step or phase (`--until B`) |
| `--redo STEP` | start this step again from scratch (can be repeated) |
| `--status` | list every step: finished (when, and its one-line result), still running, or not run |
| `--example` | print the annotated project file |
| `--prepare-only` | write the inputs of the next CASTEP jobs (with their dry runs), submit nothing |
| `--no-wait` | submit what is needed and exit; run it again later to carry on |
| `--resubmit-failed` | start failed CASTEP jobs again |
| `--force` | keep the results of jobs whose inputs have changed |

The project can be given as `CaO_Sb.dwf` or `CaO_Sb`. (`--mace-worker <tasks>.json` is what a
MACE job runs; you don't need it yourself.) For example:

```bash
python3 dwf.py CaO_Sb B1            # just the candidate list (quick), to check it
python3 dwf.py CaO_Sb --until B     # phases A and B, then stop
python3 dwf.py CaO_Sb E             # phase E again, e.g. after changing gap
python3 dwf.py CaO_Sb --redo filter # C3 from scratch, then whatever its results change
```

---

## Files

### What you give it

| File | Named by | Used for |
|---|---|---|
| project file `<name>.dwf` | the command line | every setting ([glossary](#1-the-project-file-namedwf)) |
| production `.param` | `param` | the settings of every CASTEP job ([2](#2-the-production-param-param)) |
| host `.cell` | `host_cell` | the host structure, and `species_pot` etc. for every job ([3](#3-the-host-cell-host_cell)) |
| dielectric `.param` and `.cell` | `dielectric_param`, `dielectric_cell` | the A2.3 DFPT run (optional, [4](#4-the-dielectric-param-and-cell-dielectric_param-dielectric_cell)) |
| competing-phase structures | `%block phases` | A3 ([5](#5-competing-phase-structure-files-block-phases)) |
| jobscript template | `jobscript` | every CASTEP job ([6](#6-the-jobscript-template-jobscript)) |
| MACE jobscript template | `mace_jobscript` | the MACE jobs of B2 and C1 (optional, [7](#7-the-mace-jobscript-template-mace_jobscript)) |

File names are relative to the project file.

### What it writes

Everything goes in the work directory, `dwf_<name>/` next to the project file (`workdir`
changes it):

```
dwf_CaO_Sb/
  state.json              which steps have finished, and with what inputs
  STOP                    touch it to stop a running dwf.py
  A_host/                 results.json, summary.txt,
                          production.param (your .param with the converged cut-off),
                          host_relaxed.cell (the relaxed host),
                          jobs/ (converge, relax, bands, dielectric, phases)
  B_candidates/           results.json, summary.txt, ladder.json (the sizes),
                          mace/ (every MACE relaxation, B2's and C1's: one file each),
                          jobs/mace_scan/ (B2's MACE jobs, with mace_jobscript)
  C_screen/               results.json, summary.txt,
                          structures/ (candidates, screen, distortions, drelax, share, excite),
                          jobs/ (screen, dispersion, drelax, share_1, ..., excite_gs, excite,
                          and mace_prerelax: C1's MACE jobs)
  D_production/           results.json, summary.txt,
                          structures/ (cells: D0's, production: D1's), jobs/ (cells,
                          production), finetune/ (D2)
  E_thermo/               results.json, summary.txt, formation.png
  mace_models/            the foundation model, for the MACE jobs
```

Each phase has one folder with the same layout:

- **`results.json`**: everything the later steps use from this phase, under the name of the
  step that worked it out. **A step reads only the `results.json` of earlier phases**, and the
  files named in them (the relaxed host, `production.param`, structures).
- **`summary.txt`**: the same results for you to read: for each step, when it finished, its
  one-line result and its table (the convergence test, the candidates, the filter verdicts, the
  ranking, the formation energies, ...). No step reads it, so editing it changes nothing.
- **`structures/`**: the structures the phase made, as CASTEP `.cell` files (lattice, positions,
  `SPIN=` where a starting spin is set): look at one with `ase gui`, or turn it into another
  format with `ase convert` (e.g. to `.cif`).
- **`jobs/<step>/<job>/`**: one CASTEP calculation each: `.cell`, `.param`, `.castep`,
  `-out.cell`, `.check` (unless `delete_check`), `job.sh`, `dryrun.sh` and `job_info.json`
  (what the runner decided and how it went). Each `jobs/<step>/` has a `batch_results.json`,
  and with `pack` or `dryrun_pack` a `packs/` folder with the job script of each pack.

What each phase's `results.json` holds:

| Phase | Results (by step) | Used by |
|---|---|---|
| A | `converge`: cut-off, k-point spacing, `production.param`; `relax`: `host_relaxed.cell`, `e_bulk`; `bands`: VBM, CBM, gap, the `.bands` file; `dielectric`: ε₀, ε∞; `phases`: their energies; `limits`: μ at each limit | every later phase |
| B | `generate`: the candidates (id, components, composition group, orientations, sites); `scan`: the size ladder (each size's matrix, atoms, shortest image distance and tag) and each candidate's MACE relaxation energy and edge displacement in every size, and where it converged; `choose`: the screening cell (matrix, atoms, shortest image distance, its place in the ladder), what set it, the Γ-only floor, the band edges' masses and whether they fold onto Γ | C, D, E |
| C | `prerelax`: each candidate's starting structure and where its defect is; `screen`: every run (energy, final spin, gap levels, structure) and the host's band edges at Γ; `dispersion`: the Γ-only check (W in each size, ξ, the image distance needed); `filter`: the survivors; `distort`: the distorted structures; `drelax`: their runs; `rank`: the ranked structures; `excite`: each ground state's excitation, ZPL (or vertical energy), ΔQ and verdict, and the structures for D | D, E |
| D | `cells`: each defect's production cell and whether it converged, every cell's E_host, every D0 run; `production`: every run's energy, final spin, status, relaxed structure, cell and that cell's E_host; `finetune`: `train.xyz`, `charged.xyz`, `finetune.sh` | E |
| E | `corrections`: E_corr per defect and charge, in its cell; `formation`: formation energies at each limit, transition levels, Fermi level; `interest`: the defects for phase F (`for_F`) | phase F |

Job names say what they are: `d001_qm1_s0` is candidate d001 in charge state −1 starting with
spin 0 (C2), `d001_qm1_n102` the same on C2b's 2 × 2 × 2 grid in size n102 (and `host_n102` the
pristine cell), `d001_qm1_bm0.40` the same with the bonds around the defect 40 % shorter (C4,
C5), `..._to_q0` a structure found in another charge state tried in q = 0 (C6), `d001_qm1_gs` and
`d001_qm1_ex` its ground and excited states (C7), `d001_qm1_n144` its ground state at the
production settings in size n144 (D0; D1 reuses the one in its cell), and `d001_qm1_n144_k1` its
second-lowest structure (D1). A size's tag is `n` and the atoms of its pristine supercell.

---

## The phases

Each phase's steps, what they do and what they decide. The settings are in the
[Keyword glossary](#keyword-glossary), grouped the same way.

### Phase A: the host

The reference calculations every later phase depends on.

| Label | Step | What it does | Stops if |
|---|---|---|---|
| A1 | `converge` | single points of the primitive cell: k-point grids at a cheap cut-off, then cut-offs on the converged grid; writes `production.param` and the k-point spacing | a single point has no converged energy; only the finest value is within tolerance |
| A2.1 | `relax` | variable-cell relaxation of the host | it doesn't finish |
| A2.2 | `bands` | single point on a denser Γ-centred grid: VBM, CBM, gap | no `.bands` file |
| A2.3 | `dielectric` | DFPT for the static and optical dielectric constants, or the value you give | no permittivity in the output, or only the optical one |
| A3.1 | `phases` | relaxes the competing phases | a phase doesn't finish |
| A3.2 | `limits` | chemical-potential limits | a pure element of the host is missing; the host is above the hull |

Every job copies the host `.cell`'s `species_pot` (and anything else in it apart from the
structure and k-points), so list the dopants' potentials there too.

**A1 converge.** Two scans, one after the other. First the k-points: Monkhorst–Pack grids of
the primitive cell from `converge_kgrid_min` to `converge_kgrid_max` in steps of
`converge_kgrid_step` (one number each for n × n × n grids, e.g. 4, 6, ..., 14; three each for a
non-cubic grid, e.g. `4 4 2` to `12 12 6` in steps of `2 2 1`), at the cheap cut-off
`converge_kcutoff` (default the lowest of `converge_cutoffs`), in `jobs/converge/kpoints/`. Then
the cut-off: every value in `converge_cutoffs` on the converged grid, in `jobs/converge/cutoff/`.
Each scan takes the coarsest grid (smallest cut-off) whose energy per atom, and that of every
finer value, is within `converge_tol` of the finest. If only the finest value qualifies, the step
stops: raise `converge_kgrid_max` (or add higher cut-offs) and rerun; the finished single points
are reused.

Every later cell (the relaxed host, the competing phases, the supercells) gets its k-points from
a spacing, so A1 turns the converged grid into one: the spacing halfway between that grid and the
next coarser one, along the reciprocal vector that needs the finest spacing. That gives the same
grid for the host even after A2.1 changes its lattice a little, the grid divided by m for an
m × m × m supercell, and never fewer points than the converged grid in any direction.
`summary.txt` gives both the grid and the spacing. With `kspacing` set, the k-point scan is
skipped and the cut-offs run on the grid that spacing gives. `A_host/production.param` is your
`.param` with the converged `cut_off_energy`, and without any `charge` or `spin` (every job sets
its own). With `converge : false`, nothing runs: your `.param` is used as it is, and the k-point
spacing is `kspacing`, or that of the host `.cell`'s grid.

**A2.1 relax** relaxes lattice and positions (with symmetry) at the production settings;
`host_relaxed.cell` is the cell every later phase builds on, and `e_bulk` its energy per atom.
**A2.2 bands** runs a single point of it on a Γ-centred grid at `bands_kspacing` (default half
the production spacing): VBM, CBM and gap are the highest occupied and lowest empty eigenvalues
over that grid. E2 uses its `.bands` file again for the free carriers.

**A2.3 dielectric.** The charge corrections need the static dielectric constant (electronic +
ionic). With `dielectric` set, nothing runs. Otherwise one DFPT job on the relaxed host
(`dielectric_task`, default `phonon+efield`, at q = Γ); ε₀ is the average of the diagonal of
CASTEP's DC permittivity (a warning if it varies by more than 10 %). CASTEP's DFPT needs
norm-conserving pseudopotentials, fixed occupancies and no spin polarisation: give
`dielectric_cell` (a `.cell` with an NCP `species_pot`) and `dielectric_param` (a `.param` with a
cut-off suited to it), or `dielectric`. Without them the production settings are used with
`fix_occupancy : true` and `spin_polarized : false`, and the step warns if the potentials don't
look norm-conserving.

**A3.1 phases.** The competing phases come from, in this order: `%block chemical_potentials`
(the limits themselves: nothing runs); `%block phases` (your structure files, one per line, with
options such as `molecule`, `metal`, `kspacing=`); `phases_from : doped` (the Materials Project,
through doped); or none (no limits: E2 then leaves out the μ terms, and only defects of one
composition can be compared). Solids relax lattice and positions at the production settings;
molecules relax in their fixed box at Γ.

**A3.2 limits.** A pymatgen phase diagram of the host and the phases made of its own elements
(it needs a phase of each pure element, and stops if the host lies above the hull). Each corner
of the host's stability region is a limit, named after the phases there (`CaO-Ca`, `CaO-O2`).
Each dopant X then gets, at every corner, the highest μ_X its phases allow,
μ_X = min over phases P containing X of (E_P − Σ_{i≠X} n_i μ_i) / n_X, with the limiting
phase recorded. `summary.txt` shows the limits as Δμ against the pure elements;
`results.json` has the absolute values (CASTEP's energy scale), which E2 uses.

### Phase B: candidate defects and the screening cell

| Label | Step | What it does | Decision |
|---|---|---|---|
| B1 | `generate` | the candidate defects | `%block defects` given (flowchart "yes"), or generate them ("no") |
| B2 | `scan` | the size ladder (a supercell for each of `scan_lengths`); with MACE, every candidate relaxed in every size | where each candidate's MACE relaxation has converged (`scan_tol`, `scan_edge_tol`) |
| B3 | `choose` | the screening cell for phase C | the smallest size that holds every candidate and is at least the Γ-only floor, or `supercell` |

**B1 generate.** Your defects, one per line of `%block defects`, components joined by `+`:
`Sb_Ca`, `Sb_Ca + Vac_O` (the partner on its nearest O site), `Sb_Ca + Vac_O  all` (every
arrangement up to `pair_max_distance`), `Vac_O:2` (the 2nd O site of the primitive cell),
`Sb_Ca@2` (a starting spin of 2 on the Sb), and `q=` for the charge states of that line's
defects (`Sb_Ca + Vac_O  all  q=-1`, `Vac_O  q=+2,+1`): those are screened instead of
`screen_charges`. Without the block, on each symmetry-inequivalent site
of the primitive cell: a vacancy (`vacancies`), a substitution by each dopant, antisites
(`antisites`), Voronoi interstitials (`interstitials`), and pairs: dopant + native (`pairs`),
native + native (`native_pairs`), dopant + dopant (`dopant_pairs`), never two interstitials. Each
pair type gives every symmetry-distinct arrangement up to `pair_max_distance` apart. Candidates
get ids `d001`, `d002`, ... and a composition group, e.g. `+Sb-Ca-O` for Sb_Ca + Vac_O: those with
the same group (Sb_Ca + Vac_O and Sb_O + Vac_Ca) are compared directly in C3. The number of
candidates grows quickly with each option and with `pair_max_distance`: check the list
(`python3 dwf.py <project> B1`) before screening.

**Where a defect goes in the supercell.** Every defect cell (B2, C1, C2b, D0) is built the
same way: the first component (the anchor) and the vectors from it to the others fix the
complex, and the complex is put as close to the cell centre as it can be (the mean position of
its components nearest the centre), only ever moved, never turned. It can't always be exactly at
the centre: a defect sits on lattice sites.

**Sizes are lengths.** A supercell's size is its shortest image distance L: how far the defect is
from its nearest periodic image. `scan_lengths` (default 8 to 20 Å in steps of 2) gives the size
ladder: for each length, the most compact supercell of the host whose images are all at least
that far apart, found by doped (the cell with the fewest atoms, whatever its shape: for a rock-salt
host, FCC-shaped cells, which reach a given L with about 30 % fewer atoms than cubic ones). With
a `transform`, each size is n × transform instead (the smallest n that reaches the length),
e.g. cubic cells; without doped, n × n × n primitive cells. The same cell for two lengths counts
once. Each size gets a tag, `n` and its number of atoms (`n144`), used in job names and tables.
B3 picks the screening cell from the ladder, and D0 steps through it, so the largest length is the
biggest cell either will use.

**B2 scan** (`mace`, `scan`): every candidate, built in every size it fits (a pair needs its
components less than half the image distance apart), relaxed with MACE (positions only,
`mace_model`), on compute nodes with `mace_jobscript`. Two numbers per relaxation: its
relaxation energy (unrelaxed − relaxed) and the largest displacement of the atoms at the cell's
edge (further than 35 % of the image distance from the defect). A candidate is converged at the
smallest size whose relaxation energy is within `scan_tol` of that of every bigger size, with its
edge displacement below `scan_edge_tol`: from there on, the cell holds its relaxation. One that
never converges (or fits only the largest size) counts as needing the largest size. MACE is cheap
next to CASTEP, so every candidate gets its own answer instead of one defect's standing in for
the rest. `summary.txt` has a table of every candidate's relaxation energy in every size. With
`mace : false` or `scan : false` only the ladder is made, and B3 goes by the floor (or
`supercell`).

**B3 choose: the screening cell.** Every candidate is built in this one cell in phase C, so the
energies of arrangements of one composition can be compared directly. It is the smallest size of
the ladder that is at least as big as

- every candidate's converged size (B2): the worst case sets it, and `summary.txt` names the
  candidate that did, and
- the **Γ-only floor**: the image distance a cell needs for Γ-point screening to see a defect's
  levels as an isolated defect has them. A level E_b inside the gap decays away from the defect
  as exp(−r/ξ), with 1/ξ² = (2m*/ħ²) E_b (1 − E_b/E_g) in a two-band (Kane) model, and
  neighbouring images' levels overlap and spread into a band unless they are several ξ apart. The
  floor is 4ξ for the shallowest level that counts as deep (E_b = `deep_level_margin`, E3's), with
  the lighter of the hole and electron effective masses, which B3 fits to A2.2's `.bands` around
  the VBM and CBM (each k-point next to the edge gives a mass, the lightest is taken; the
  irreducible k-points are unfolded with the host's symmetry), and A2.2's gap. For CaO-like
  masses (0.4 to 1 m_e), E_b = 0.5 eV and a PBE gap of 3.6 eV it is 12 to 19 Å.
  `screen_min_image` gives the floor yourself (0: none).

A floor beyond the ladder gets a cell of its own. `supercell` fixes the screening cell instead (a
length, `supercell : 14 ang`, or n × transform, `supercell : 3`); D0 still converges each defect's
production cell. B3 also checks whether A2.2's band edges fold onto Γ of the screening cell: a
k-point k folds if k Pᵀ is a whole-number vector (P the supercell matrix). One that doesn't (e.g.
an edge at X, and n × n × n copies of the FCC primitive cell with n odd) means the
Γ-point screening sees a wider gap, and its levels near that edge are measured against an edge
that is too far out: B3 warns, and C2 measures it. A cubic `transform` folds rock salt's X onto
Γ.

### Phase C: screening and ground states

Every candidate goes through cheap CASTEP relaxations; the ones that are competitive and have
the properties you ask for get a search for their ground-state structure. With `excite : true`,
C7 then calculates each one's excited state, and only the defects that pass, plus every native
defect, go on to the production runs.

| Label | Step | What it does | Decision |
|---|---|---|---|
| C1 | `prerelax` | builds every candidate in the screening cell and relaxes it with MACE (B2's relaxation there, if it ran) | pairs too far apart for the cell are left out; `configuration_margin` drops arrangements far above the lowest |
| C2 | `screen` | one cheap CASTEP relaxation per candidate, charge state and starting spin | — (warns if the band edges don't fold onto Γ) |
| C2b | `dispersion` | the Γ-only check: one defect's gap levels on a 2 × 2 × 2 grid, in the screening cell and the size below | warns (or with `dispersion_check : stop`, stops before C3) if they spread by more than `dispersion_tol` |
| C3 | `filter` | defect hull, duplicates, property filters | within `hull_margin` of the hull; the `require_*` filters |
| C4 | `distort` | ShakeNBreak distortions of the survivors | which charge states (`distort_charges`) |
| C5 | `drelax` | cheap relaxations of the distortions | — |
| C6 | `rank` | ranks the structures, shares winning distortions between charge states | a distortion counts if it beats the screened structure by `distortion_min_gain` |
| C7 | `excite` | optional (`excite : true`): the excited state of each ground state, by ΔSCF | the ZPL inside `excite_window`; native defects go on anyway |

The CASTEP jobs of C2, C2b, C5, C6 and C7 run with the runner settings of `%block runner_screen`
(and of `runner_dispersion`, `runner_drelax`, `runner_rank` or `runner_excite` for one of them
alone): no restarts, and a job that fails is flagged while the others carry on
(`--resubmit-failed` tries it again).

**C1 prerelax.** Each candidate is built in the screening cell; a pair whose components are more
than half the shortest image distance apart is left out. With `mace : true` it is relaxed with
MACE (positions only, `mace_model`): when the screening cell is one of B2's sizes, that is B2's
relaxation, read from its file in `B_candidates/mace/`; otherwise C1 runs it (on compute nodes
with `mace_jobscript`). With `configuration_margin`, arrangements whose configurational energy
(MACE energy above the lowest arrangement of the same composition) is more than the margin are
dropped before any CASTEP run (MACE knows nothing of charge or spin, so keep the margin
generous).

**C2 screen.** For each candidate, charge state (its own `q=` from `%block defects`, else
`screen_charges`) and starting spin, one CASTEP
relaxation at Γ, without symmetry, at `screen_cutoff` (default the production cut-off), with at
most `screen_max_steps` geometry steps and any `%block screen_overrides`. The starting spins
(`screen_spins`): the lowest the electron count allows (0 or 1) and, with `both`, that + 2, set
as `spin` and as `SPIN=` on the anchor atom. They are starting values: CASTEP may end elsewhere,
and the results keep the final spin. The pristine supercell gets a single point at the same
settings: its band edges at Γ are the reference for the gap levels. Each run records its energy
(only if the last SCF cycle converged), final spin, and the defect levels in the gap (more than
`level_margin` inside the band edges). C2 compares the pristine cell's gap at Γ with A2.2's: more
than 0.05 eV wider means band edges that don't fold onto Γ of this cell (B3 says which), and C2
warns.

**C2b dispersion: the Γ-only check** (`dispersion_check : warn`, the default, or `stop`; `off`
skips it). The floor in B3 is an estimate; C2b measures. It takes one defect: the screened state
whose gap level is nearest a band edge (the most spread-out level, the first to feel its images),
or `dispersion_defect` (an id, or `id:q`). Its structure from C2 is run as a single point on a
Γ-centred 2 × 2 × 2 k-point grid in the screening cell, and carried into the size below it (B3's
ladder) and run there too, each with its pristine cell at the same settings for the band edges
(`C_screen/jobs/dispersion/`). At Γ only, a level's dispersion is invisible; across the 2 × 2 × 2
grid it shows as W, the spread of the level's energy over the k-points. W falls as exp(−d/ξ)
with the image distance d, so the two sizes give ξ = (d₂ − d₁) / ln(W₁/W₂) and the image distance
at which W would be `dispersion_tol` (0.05 eV). It also flags a level that crosses into a band
at some k-point, and a spin channel whose highest occupied band at one k-point lies above its
lowest empty band at another (the occupations would change across the k-points). The screening
cell passes if its W is below `dispersion_tol` and neither flag is set; if not, C2b warns, with
the image distance needed (e.g. `screen_min_image : 13.2 ang`); with `stop`, C3 stops before it
filters anything. To use a bigger cell, set `screen_min_image` (or `supercell`) and rerun with
`--redo C` (the jobs of a new cell can't reuse the old cell's job folders). `summary.txt` gives
the measured ξ next to B3's estimate.

**C3 filter.**

1. *Best state:* for each candidate and charge state, the lowest energy over its starting spins.
2. *Defect hull:* within a composition group, the formation energy of every arrangement and
   charge state across the gap, E + E_corr + q (E_VBM + E_F), without the host energy and
   chemical potentials (the same for all of them). An arrangement is kept if its lowest comes
   within `hull_margin` of the group's lowest somewhere in the gap. Its *stable* charge states
   are those that are its lowest somewhere in the gap.
3. *Duplicates:* an arrangement that relaxed into the same structure as a lower one of its group
   is dropped.
4. *Property filters,* each met by a stable charge state: `require_gap_states` (occupied, empty
   or both kinds of gap level), `require_spin` (the final spin), `require_bound` (pairs bound
   against their isolated components somewhere in the gap). With `excite : true`, native
   defects skip these filters (but not the hull or the duplicates), so that they reach D.

The charge states carried on are the stable ones (`distort_charges : stable`) or every one with a
usable energy (`screened`). `summary.txt` gives every candidate's verdict and why.

**C4 distort, C5 drelax.** For each survivor and charge state, from its screened structure,
ShakeNBreak moves the nearest neighbours of the defect (ShakeNBreak's count, from the oxidation
states in `%block oxidation_states` or pymatgen's guess) in or out by each factor in
`distortions`, around the anchor atom or, with `distort_around : vacancy`, the vacant site, and
rattles the other atoms (`rattle_stdev`, `rattle_dmin`). C5 relaxes every distorted structure at
the C2 settings, starting from the screened run's final spin.

**C6 rank.** For each survivor and charge state, the screened structure and every distortion
relaxation are pooled, identical structures merged (pymatgen's StructureMatcher with tight
tolerances) and ranked by energy. A distortion that wins in one charge state (by
`distortion_min_gain`) is also relaxed in the other charge states of that defect, for up to
`share_rounds` rounds. The lowest `keep_structures` distinct structures of each charge state go
on to C7 (and so to phase D).

**C7 excite** (`excite : true`; off by default, when everything C6 kept goes on to D). It
calculates the excited state of each defect's ground state, in each charge state C6 kept (its
lowest structure), the way ADAQ's screening does, and filters on the result:

1. *Ground state:* a single point of the structure at the screening settings (Γ, `screen_cutoff`,
   `%block screen_overrides`), with `metals_method : dm` and `%block excite_overrides`
   (`C_screen/jobs/excite_gs/`). Its `.check` is kept, even with `delete_check : true`, since
   the excited state starts from it: once you are happy with C7's results you can delete
   `C_screen/jobs/excite_gs/*/*.check` to save space (a 216-atom cell's can be a few GB).
2. *Which excitation:* from its eigenvalues, the smallest step of at least `excite_min` (0.4 eV)
   from an occupied to an empty level in one spin channel, both more than `level_margin` inside
   the host's band edges. One excitation per charge state. None: the charge state has no internal
   transition in the gap.
3. *Excited state:* CASTEP's simple ΔSCF (`calculate_deltascf : true`, `deltascf_method :
   simple`, the two bands' occupations in `%block deltascf_constraints`), starting from the ground
   state's `.check` (`reuse` and `deltascf_checkpoint`) (`C_screen/jobs/excite/`). With
   `excite_relax : true` (the default) it is relaxed: its energy above the ground state is the ZPL,
   and ΔQ (the mass-weighted displacement, amu^½ Å) comes from the two structures
   (`C_screen/structures/excite/`). With `false`, a single point gives the vertical energy.
4. *Verdict per charge state:* `pass` (energy inside `excite_window`; any converged energy without
   a window), `outside`, `none` (no excitation in the gap), or `unknown` (a run failed or did not
   converge: kept, with a warning, rather than dropped on a failed calculation).
5. *On to D:* a defect with a charge state that passes (or is unknown) goes on with every
   structure C6 kept for it, in every charge state. Every native defect (one that only adds or
   removes host elements: vacancies, antisites, self-interstitials and their complexes) goes on
   whatever its result, because E's formation energies and Fermi level need it.

The ΔSCF occupations are only set in the excited-state runs; every ground-state run still lets
the spin relax as before. Test C7 on one defect before a full run: CASTEP's ΔSCF must work on
your build, and its documentation does not say whether excited-state geometry optimisation
honours the constraints (if it doesn't, use `excite_relax : false`). Sb_Ca–V_O in q = −1 is a
good test: ADAQ found its ZPL at about 0.53 eV (PBE 0.54, HSE 0.53). The ΔSCF energies are at
the PBE level, which in SiC and diamond comes out about 0.2 eV low; keep the window generous. A
constrained-occupation run can fail to converge (ADAQ had this for the P and As versions of
X_Ca–V_O); those charge states come out `unknown`. Like ADAQ, it only looks at excitations
between levels in the gap, so a defect whose level lies inside a band at the PBE level has none.

### Phase D: production cells and relaxations

| Label | Step | What it does | Decision |
|---|---|---|---|
| D0 | `cells` | each defect's production cell: its ground states at the production settings, from one size below the screening cell up | converged at the smaller of two neighbouring sizes that agree within `series_tol` in every charge state; flagged if the largest size is reached |
| D1 | `production` | every structure phase C passed on, relaxed in its defect's cell (the ground states are D0's runs) | — |
| D2 | `finetune` | optional: MACE fine-tuning data from D0's and D1's relaxations, for phase F | — |

The CASTEP jobs of D0 and D1 run with the runner settings of `%block runner_production` (and of
`runner_cells` for D0 alone): a relaxation that runs out of walltime is carried on from its
`.check`, and a job that fails is flagged and left out of phase E (fix the cause and use
`--resubmit-failed`).

**D0 cells.** The screening cell was chosen for screening (every candidate's MACE relaxation, the
Γ-only floor); a defect's formation energy at the production settings can need a different size,
and charged defects converge more slowly. So D0 converges it for every defect that goes on to D,
with all the charge states of a defect in one cell:

1. *Start:* each charge state's ground state (the lowest structure C6 kept) is relaxed at the
   production settings in the screening cell, and in the size below it in the ladder (B3's: the
   sizes of `scan_lengths`), with a single point of the pristine supercell in each
   (`D_production/jobs/cells/`). The relaxed structure is carried from one cell to the other: the
   defect is built in the new cell, and each atom within half the smaller cell's image distance
   of it is moved by the displacement the same atom has in the relaxed structure (the atom of
   the same element at the same position relative to the defect; the atoms are matched to their
   sites by an assignment that keeps each on its own site, so it works for any host and any two
   supercells of it). The others stay on their perfect-crystal sites.
2. *Compare:* for each charge state, E_D − E_host + E_corr (what goes into the formation energy,
   with each cell's own E_host and charge correction) in the two sizes, and whether both relaxed
   into the same local structure (the displacements of the atoms near the defect differ by no
   more than 0.2 Å). If every charge state agrees within `series_tol` and is in the same basin,
   the defect is converged at the smaller size, which is its production cell.
3. *Otherwise* the next size up is run, carried over from the biggest size relaxed so far, and
   compared with the one below it, and so on. A defect that reaches the largest size of the
   ladder without converging keeps that size and is flagged (`NOT CONVERGED` in `summary.txt`,
   and in E2's table). A size a pair doesn't fit in is skipped (it starts at the screening cell
   and the size above). A charge state whose run failed is left out of the comparison, with a
   warning; if every one failed, the defect keeps the biggest size whose runs all worked.

If the screening cell is the smallest size of the ladder, D0 starts at it and the size above.
Next to relaxing every ground state in the screening cell, D0 adds one relaxation per charge state
in the smaller cell, and when a defect converges there, D1's other structures run in that cheaper
cell.

**D1 production.** One relaxation per structure phase C passed on (C6's, or with `excite : true`
those of the defects that passed C7 and of every native defect), in its defect's production cell
from D0: the ground state of each charge state is D0's own run in that cell, so only the other
structures C6 kept (`keep_structures` > 1) run here, carried into that cell
(`D_production/jobs/production/`, named after the cell). The settings: A1's cut-off and k-point
spacing, `%block production_overrides`, its charge, and its starting spin from phase C
(`spin_polarized : true` with that spin; a spin-0 state runs without spin polarisation unless
`production_overrides` sets `spin_polarized`). Positions only (the cell stays at the host's
relaxed lattice), no symmetry. E_host for each cell is D0's single point.

**`freeze_radius`** (an untested idea, in D0 and D1): every atom further than r Å from all of
the defect's components is held fixed. That cuts the geometry steps of large cells but changes
the relaxation energy; compare it with full relaxations on one defect before relying on it.

**D2 finetune** (`finetune : true`): every geometry step, with its energy and forces, of D0's and
D1's relaxations (steps with a force above 2 eV/Å left out) goes into
`D_production/finetune/train.xyz` for the neutral defects and `charged.xyz` for the charged ones
(with their `total_charge`: a MACE model that doesn't take the charge can't use them), and
`finetune.sh` runs `mace_run_train` on `train.xyz` from `mace_model`. It is training data for the
optical calculations (phase F); B2 and C1 always use `mace_model`.

### Phase E: thermodynamics

No CASTEP jobs: it runs in seconds once phase D has finished.

**E1 corrections.** For each charge state that reached phase D, the image-charge correction of
its defect's production cell (D0) with ε₀ from A2.3: `lz` (Lany–Zunger, the default), `mp`
(first-order Makov–Payne) or `none`. No potential alignment.

**E2 formation.** For each defect, its lowest production energy in each charge state gives

  E_f(q, E_F) = E_D − E_host − Σᵢ nᵢ μᵢ + q (E_VBM + E_F) + E_corr

at every chemical-potential limit, with E_host the pristine energy of the defect's own cell
(D0), E_VBM from A2.2 and μᵢ from A3.2; the table names each defect's cell and flags one D0
could not converge. The transition levels
are the Fermi levels where the lowest-energy charge state changes, between the VBM and the gap
(`gap`, if set, else A2.2's). `formation.png` plots each defect's lowest charge state against E_F,
one panel per limit. With `fermi_temperature`, the self-consistent Fermi level at each limit: the
E_F where the charges of the defects and the free electrons and holes balance (concentrations
from the formation energies, the sites, orientations and the final spin's degeneracy; carriers
from A2.2's band energies). With `quench_temperature`, the concentrations are frozen at
`fermi_temperature` and shared out again between charge states at the lower temperature. Only
the defects that reached phase D take part, so native defects dropped earlier can move the real
Fermi level; with `excite : true`, every native defect on the hull reaches D.

**E3 interest.** A defect goes on to phase F if one of its transition levels lies at least
`deep_level_margin` from both band edges, or if phase C found an internal transition for it: with
`excite : true`, a charge state that passed C7 (its ΔSCF ZPL or vertical energy is given), or one
whose C7 runs failed (flagged to check by hand). Without `excite`, only deep levels count. E3 does not use the Fermi level: check that the charge state you take to phase F is
the one present at E2's Fermi level.

---

## Keyword glossary

Every keyword of every input file, sorted by file. The project file (section 1) is the only file
`dwf.py` is given; it names the others. For the CASTEP files (sections 2 to 4) the tables list
only the keywords the workflow sets, changes or removes; every other keyword is passed on to the
jobs as you wrote it.

Sections: [1. The project file](#1-the-project-file-namedwf) ·
[2. The production `.param`](#2-the-production-param-param) ·
[3. The host `.cell`](#3-the-host-cell-host_cell) ·
[4. The dielectric files](#4-the-dielectric-param-and-cell-dielectric_param-dielectric_cell) ·
[5. Competing phases](#5-competing-phase-structure-files-block-phases) ·
[6. The jobscript](#6-the-jobscript-template-jobscript) ·
[7. The MACE jobscript](#7-the-mace-jobscript-template-mace_jobscript) ·
[8. The runner's own files](#8-supercell_convergencepy-on-its-own-conv-and-batch-files)

### 1. The project file (`<name>.dwf`)

The file you give `dwf.py`. Its keywords are grouped below by the phase that uses them.

**Writing them.** `keyword : value` (or `=`, or a space), in any case, each keyword once; `#` or
`!` starts a comment. File names are relative to the project file.

- **Numbers** can be followed by the unit in the Default column: `eV`, `ang` (or `A`) for Å,
  `1/ang`, `eV/atom`, `eV/ang`, `K`. The unit is checked, not converted: give the value in that
  unit.
- **Lists** (marked *list*) go on one line, separated by spaces or commas, or in a block with
  any number of values per line (`%block distortions` … `%endblock distortions`).
- **Switches** take `true` or `false` (also `yes`/`no`, `on`/`off`); the keyword alone means
  `true`.
- **Blocks** that hold `.param` keywords (`screen_overrides`, `excite_overrides`,
  `production_overrides`) take
  `keyword : value` lines; the others take the rows described below.

**Changing them.** The Steps column lists the steps that use a keyword. Change it and those steps
run again the next time you run `dwf.py`; the steps after them run again only if their
results change. CASTEP jobs whose inputs have not changed are reused, not repeated.

**Alphabetical index** (the group in brackets):

- **A**: [`antisites`](#b-candidates-and-the-screening-cell) (B)
- **B**: [`bands_kspacing`](#a-host) (A) · [`baseline_mb`](#runner-settings) (runner)
- **C**: [`charge_correction`](#e-thermodynamics) (E) · [`%block chemical_potentials`](#a-host) (A) · [`configuration_margin`](#c-screening) (C) · [`converge`](#a-host) (A) · [`converge_cutoffs`](#a-host) (A) · [`converge_kcutoff`](#a-host) (A) · [`converge_kgrid_max`](#a-host) (A) · [`converge_kgrid_min`](#a-host) (A) · [`converge_kgrid_step`](#a-host) (A) · [`converge_tol`](#a-host) (A) · [`cores`](#runner-settings) (runner) · [`cores_per_node`](#runner-settings) (runner)
- **D**: [`deep_level_margin`](#e-thermodynamics) (E) · [`%block defects`](#b-candidates-and-the-screening-cell) (B) · [`delete_check`](#runner-settings) (runner) · [`dielectric`](#a-host) (A) · [`dielectric_cell`](#a-host) (A) · [`dielectric_inf`](#a-host) (A) · [`dielectric_param`](#a-host) (A) · [`dielectric_task`](#a-host) (A) · [`dispersion_check`](#c-screening) (C) · [`dispersion_defect`](#c-screening) (C) · [`dispersion_tol`](#c-screening) (C) · [`distort_around`](#c-screening) (C) · [`distort_charges`](#c-screening) (C) · [`distortion_min_gain`](#c-screening) (C) · [`distortions`](#c-screening) (C) · [`dopant_pairs`](#b-candidates-and-the-screening-cell) (B) · [`dopants`](#b-candidates-and-the-screening-cell) (B) · [`dryrun_cmd`](#runner-settings) (runner) · [`dryrun_cores`](#runner-settings) (runner) · [`dryrun_jobscript`](#runner-settings) (runner) · [`dryrun_mem_gb`](#runner-settings) (runner) · [`dryrun_on`](#runner-settings) (runner) · [`dryrun_options`](#runner-settings) (runner) · [`dryrun_pack`](#runner-settings) (runner) · [`dryrun_partition`](#runner-settings) (runner) · [`dryrun_time`](#runner-settings) (runner) · [`dryrun_timeout`](#runner-settings) (runner)
- **E**: [`energy`](#general) (general) · [`excite`](#c-screening) (C) · [`excite_min`](#c-screening) (C) · [`%block excite_overrides`](#c-screening) (C) · [`excite_relax`](#c-screening) (C) · [`excite_window`](#c-screening) (C)
- **F**: [`fermi_temperature`](#e-thermodynamics) (E) · [`finetune`](#d-production) (D) · [`finetune_args`](#d-production) (D) · [`freeze_radius`](#d-production) (D)
- **G**: [`gap`](#e-thermodynamics) (E)
- **H**: [`host_cell`](#general) (general) · [`hull_margin`](#c-screening) (C)
- **I**: [`interstitials`](#b-candidates-and-the-screening-cell) (B)
- **J**: [`jobscript`](#runner-settings) (runner)
- **K**: [`keep_structures`](#c-screening) (C) · [`kspacing`](#a-host) (A)
- **L**: [`level_margin`](#c-screening) (C)
- **M**: [`mace`](#mace) (MACE) · [`mace_device`](#mace) (MACE) · [`mace_fmax`](#mace) (MACE) · [`mace_jobscript`](#mace) (MACE) · [`mace_max_jobs`](#mace) (MACE) · [`mace_model`](#mace) (MACE) · [`mace_pack`](#mace) (MACE) · [`mace_python`](#mace) (MACE) · [`mace_steps`](#mace) (MACE) · [`mace_threads`](#mace) (MACE) · [`max_cores`](#runner-settings) (runner) · [`max_jobs`](#runner-settings) (runner) · [`max_nodes`](#runner-settings) (runner) · [`max_restarts`](#runner-settings) (runner) · [`max_walltime`](#runner-settings) (runner) · [`mem_per_core`](#runner-settings) (runner) · [`mem_replicated`](#runner-settings) (runner) · [`mem_safety`](#runner-settings) (runner) · [`mp_api_key`](#a-host) (A)
- **N**: [`name`](#general) (general) · [`native_pairs`](#b-candidates-and-the-screening-cell) (B)
- **O**: [`omp_threads`](#runner-settings) (runner) · [`%block oxidation_states`](#c-screening) (C)
- **P**: [`pack`](#runner-settings) (runner) · [`pair_max_distance`](#b-candidates-and-the-screening-cell) (B) · [`pairs`](#b-candidates-and-the-screening-cell) (B) · [`param`](#general) (general) · [`%block partitions`](#runner-settings) (runner) · [`%block phases`](#a-host) (A) · [`phases_e_above_hull`](#a-host) (A) · [`phases_from`](#a-host) (A) · [`poll`](#runner-settings) (runner) · [`%block production_overrides`](#d-production) (D)
- **Q**: [`quench_temperature`](#e-thermodynamics) (E)
- **R**: [`rattle_dmin`](#c-screening) (C) · [`rattle_stdev`](#c-screening) (C) · [`require_bound`](#c-screening) (C) · [`require_gap_states`](#c-screening) (C) · [`require_spin`](#c-screening) (C) · [`%block runner_host`, `runner_screen`, `runner_production`, `runner_<step>`](#runner-settings) (runner)
- **S**: [`scan`](#b-candidates-and-the-screening-cell) (B) · [`scan_edge_tol`](#b-candidates-and-the-screening-cell) (B) · [`scan_lengths`](#b-candidates-and-the-screening-cell) (B) · [`scan_tol`](#b-candidates-and-the-screening-cell) (B) · [`screen_charges`](#c-screening) (C) · [`screen_cutoff`](#c-screening) (C) · [`screen_max_steps`](#c-screening) (C) · [`screen_min_image`](#b-candidates-and-the-screening-cell) (B) · [`%block screen_overrides`](#c-screening) (C) · [`screen_spins`](#c-screening) (C) · [`series_tol`](#d-production) (D) · [`share_rounds`](#c-screening) (C) · [`sp_walltime`](#runner-settings) (runner) · [`status_cmd`](#runner-settings) (runner) · [`submit_cmd`](#runner-settings) (runner) · [`supercell`](#b-candidates-and-the-screening-cell) (B)
- **T**: [`tolerate_failures`](#runner-settings) (runner) · [`transform`](#general) (general)
- **V**: [`vacancies`](#b-candidates-and-the-screening-cell) (B)
- **W**: [`walltime`](#runner-settings) (runner) · [`walltime_safety`](#runner-settings) (runner) · [`workdir`](#general) (general)

Groups: [General](#general) · [MACE](#mace) · [A: host](#a-host) ·
[B: candidates and the screening cell](#b-candidates-and-the-screening-cell) · [C: screening](#c-screening) ·
[D: production](#d-production) · [E: thermodynamics](#e-thermodynamics) ·
[Runner settings](#runner-settings)

#### General

| Keyword | Default | Steps | What it does |
|---|---|---|---|
| `host_cell` | required | A1, A2.1 | The host cell, normally the primitive cell (a conventional cell also works); A2.1 relaxes it. Everything in it apart from the structure and k-points (`species_pot`, `symmetry_generate`, any other `.cell` keyword) is copied into every job's `.cell`; defect cells lose the symmetry keywords. List the dopants in `species_pot` too, or leave the block out to use CASTEP's default on-the-fly potentials for every species. |
| `param` | required | A1 | The production settings: functional, SCF and geometry settings, smearing, and so on. A1 writes `production.param` from it, with the converged `cut_off_energy` and without any `charge` or `spin`, and every later job starts from that file. `dwf.py` sets `task`, and per job `charge`, `spin`, `spin_polarized` and the k-points. |
| `transform` | — | B2, B3 | A supercell shape: 9 integers, row by row, each row a supercell vector in terms of the primitive vectors (determinant > 0). With it, each size of the ladder is n × `transform`, the smallest n that reaches the length, and `supercell : n` is n × `transform`; e.g. the example's (commented-out) matrix turns an FCC primitive cell into cubic cells of 8n³ atoms, which fold rock salt's X point onto Γ. Without it (the default), each size is the most compact cell doped finds. |
| `energy` | `auto` | A1, A2.1, A3.1, C2, C2b, C5, C6, C7, D0, D1 | Which CASTEP energy every step reads: `final` (Final energy), `free` (E − TS), `0K` (the 0 K estimate) or `auto` (the 0 K estimate when CASTEP prints one, else the final energy). They differ only with smearing. |
| `name` | the file name | — | The project's name: the default work directory is `dwf_<name>`, and it titles the E2 plot. |
| `workdir` | `dwf_<name>` | — | The work directory, relative to the project file. |

#### MACE

MACE has no charge or spin: it only prepares structures (B2, C1) and never ranks charge states.

| Keyword | Default | Steps | What it does |
|---|---|---|---|
| `mace` | `true` | B2, C1 | Use MACE for the scan of every candidate (B2) and the pre-relaxation and configurational-energy filter of the candidates (C1). With `false`, B2 makes only the size ladder, B3 goes by the Γ-only floor (or `supercell`), C1 builds unrelaxed structures, and `configuration_margin` does nothing. |
| `mace_model` | `medium-mpa-0` | B2, C1, D2 | The model: the name of a `mace_mp` foundation model, or a model file of your own. D2's `finetune.sh` starts from it. A new model makes B2 and C1 run again (their MACE relaxations are kept per model, so going back to an earlier one reuses them). |
| `mace_fmax` | `0.05 eV/ang` | B2, C1 | A MACE relaxation stops when the largest force is below this. |
| `mace_steps` | `500` | B2, C1 | ... or after this many optimiser (LBFGS) steps. |
| `mace_device` | `cpu` | — | `cpu`, or `cuda` on a GPU. |
| `mace_threads` | `8` | — | Torch threads per relaxation. |
| `mace_jobscript` | — | — | A Slurm template (a file, or `%block mace_jobscript`) for running the MACE relaxations as jobs on compute nodes. It needs one line with `{COMMAND}`, which becomes the workers; whatever is in front of it on that line (e.g. `srun --exact --ntasks=1 --cpus-per-task=8`) goes in front of each. `{JOBNAME}` is filled in too. Without it, the relaxations run where `dwf.py` runs. See [section 7](#7-the-mace-jobscript-template-mace_jobscript). |
| `mace_pack` | `cores_per_node` / `mace_threads` | — | Relaxations side by side in one MACE job (16 on ARCHER2 with 8 threads each). |
| `mace_max_jobs` | `4` | — | The most MACE jobs a step's relaxations are shared out between. |
| `mace_python` | the Python running `dwf.py` | — | The Python the MACE jobs run, with mace-torch. On ARCHER2 it must be on `/work`. |

#### A: host

Details: [Phase A](#phase-a-the-host).

| Keyword | Default | Steps | What it does |
|---|---|---|---|
| `converge` | `true` | A1 | Converge the k-point grid, then the plane-wave cut-off, first. With `false`, A1 runs nothing: `production.param` is your `.param` (without `charge` or `spin`), and the k-point spacing is `kspacing`, or that of the host `.cell`'s grid. |
| `converge_cutoffs` | `300 400 500 600 700 800 eV` (*list*) | A1 | The cut-offs tried, after the k-point scan: a single point of the host cell at each, on the converged grid (or the grid of `kspacing`). |
| `converge_kgrid_min`, `converge_kgrid_max`, `converge_kgrid_step` | `4`, `12`, `2` | A1 | The Monkhorst–Pack grids of the primitive cell tried first, at `converge_kcutoff`: from min to max in steps of step. One number each gives n × n × n grids; three each give a non-cubic grid, one number per reciprocal-lattice direction (e.g. `4 4 2`, `12 12 6`, `2 2 1`). A step of 0 keeps a direction fixed (min = max, e.g. 1 for a slab); every direction that moves must take the same number of steps. Not used with `kspacing`. |
| `converge_kcutoff` | the lowest of `converge_cutoffs` | A1 | The cheap cut-off of the k-point scan. The k-point convergence barely depends on the cut-off, so a low one is enough. |
| `converge_tol` | `0.001 eV/atom` | A1 | A1 takes the smallest cut-off, and the coarsest k-point grid, whose energy per atom (and that of every finer value) is within this of the finest one. If only the finest one is, A1 stops and asks for higher cut-offs or a higher `converge_kgrid_max`; the finished single points are reused. |
| `kspacing` | from A1 | A1 | The production k-point spacing, in 1/Å as CASTEP's `kpoints_mp_spacing`. Every job without its own grid uses it; A1 works it out from the converged grid. Setting it skips the k-point part of A1 (the cut-off is still converged). |
| `bands_kspacing` | half the production spacing | A2.2 | The spacing of the Γ-centred grid on which A2.2 finds the VBM, CBM and gap. Make it smaller if the band edges may lie between grid points. |
| `dielectric` | from A2.3 | A2.3 | The static dielectric constant (electronic + ionic), ε₀, for every charge correction (C3, D0, E1). Setting it skips the DFPT run. |
| `dielectric_inf` | — | A2.3 | The optical constant ε∞, recorded with `dielectric`. The corrections don't use it. |
| `dielectric_task` | `phonon+efield` | A2.3 | The CASTEP task of the DFPT run, at q = Γ. It must give the DC (static) permittivity: with `efield`, which gives only ε∞, A2.3 stops. |
| `dielectric_param` | — | A2.3 | A `.param` for the DFPT run instead of `production.param`. CASTEP's DFPT needs norm-conserving potentials, fixed occupancies and no spin polarisation. Without this keyword the production settings are used with `fix_occupancy : true` and `spin_polarized : false`; with it, put those in the file, with a cut-off suited to norm-conserving potentials. |
| `dielectric_cell` | — | A2.3 | A `.cell` whose `species_pot` (and other settings) the DFPT run uses instead of the host `.cell`'s, e.g. `NCP` for CASTEP's norm-conserving library. Its structure is ignored: the run uses the relaxed host. A2.3 warns before submitting if the potentials don't look norm-conserving. |
| `%block phases` | — | A3.1 | The competing phases, one per line: `name  structure-file  [options]`. Options: `molecule` (positions only, in its fixed box, at Γ, without symmetry); `metal` (`fix_occupancy : false` unless the line says otherwise; add `smearing_width=...`); `kspacing=0.03` (its own k-point spacing); and any other `keyword=value`, a `.param` keyword for that phase only (e.g. `spin=2 spin_polarized=true` for O₂). Solids relax lattice and positions, with symmetry, at the production settings. A3.2 needs a phase of each pure element of the host, and for each dopant at least one phase that contains it. |
| `phases_from` | `none` | A3.1 | `doped`: when there is no `%block phases`, take the competing phases from the Materials Project through doped (needs doped and `mp_api_key`). |
| `mp_api_key` | — | — | Materials Project API key for `phases_from : doped` (or set it in doped's configuration). |
| `phases_e_above_hull` | `0.05 eV/atom` | A3.1 | `phases_from : doped` takes the Materials Project entries up to this far above the hull. |
| `%block chemical_potentials` | — | A3.1, A3.2 | The limits themselves, one per line: `name  El=value ...`, in absolute eV per atom on the energy scale of your CASTEP runs, for every element of the host and the dopants. A3.1 then relaxes nothing, and A3.2 records them. It takes precedence over `%block phases` and `phases_from`. |

#### B: candidates and the screening cell

Details: [Phase B](#phase-b-candidate-defects-and-the-screening-cell).

| Keyword | Default | Steps | What it does |
|---|---|---|---|
| `dopants` | — (*list*) | A3.1, A3.2, B1 | The extrinsic elements. B1 puts each on every inequivalent host site, and on the interstitial sites with `interstitials`; A3 limits their chemical potentials; C4 guesses their oxidation states. |
| `%block defects` | — | B1 | Your own defects, one per line, components joined by `+`: `Sb_Ca` (Sb on a Ca site), `Vac_O`, `Vac_O:2` (the 2nd O site of the primitive cell), `Sb_Ca@2` (starting spin 2 on the Sb). Each later component goes on its nearest site of that kind to the first one; with `all` on the line (pairs only), every arrangement up to `pair_max_distance` becomes a candidate. `q=-1` (or `q=+1,0`) on a line gives its defects' charge states, used instead of `screen_charges` from C2 on. With this block, nothing is generated. |
| `vacancies` | `true` | B1 | A vacancy on each inequivalent host site. |
| `antisites` | `false` | B1 | Each other host element on each inequivalent host site. |
| `interstitials` | `none` | B1 | `dopants`: the dopants on each Voronoi interstitial site (pymatgen-analysis-defects); `all`: every element there. |
| `pairs` | `true` | B1 | Pairs of a dopant defect with a native one, in every symmetry-distinct arrangement up to `pair_max_distance` apart. Never two interstitials. |
| `native_pairs` | `false` | B1 | Also pairs of two native defects. |
| `dopant_pairs` | `false` | B1 | Also pairs of two dopant defects. |
| `pair_max_distance` | `5.0 ang` | B1 | The largest separation of the pair arrangements, generated or from `all`. A pair farther apart than half a cell's shortest image distance isn't built in it: B2 and D0 skip that size, and C1 leaves the pair out if it is the screening cell. |
| `scan` | `true` | B2 | Scan every candidate with MACE in every size (also needs `mace : true`). With `false`, B2 makes only the size ladder. |
| `scan_lengths` | `8 10 12 14 16 18 20 ang` (*list*) | B2, B3, D0 | The sizes, as lengths: for each, the most compact supercell whose periodic images are all at least that far apart (doped; n × `transform` with a transform). B2 scans them, B3 picks the screening cell from them, D0 steps through them; the largest is the biggest either uses. |
| `scan_tol` | `0.02 eV` | B2 | A candidate's MACE relaxation energy must be within this of that of every bigger size ... |
| `scan_edge_tol` | `0.05 ang` | B2 | ... and the atoms at the cell's edge (further than 35 % of the image distance from the defect) moved less than this, for it to be converged in that size. |
| `screen_min_image` | from the band masses | B3 | The Γ-only floor (Å) yourself: the screening cell's images are at least this far apart. `0`: no floor. Without it, B3 estimates the floor from A2.2's effective masses and `deep_level_margin`. |
| `supercell` | — | B3 | Fix the screening cell: a length (`supercell : 14 ang`, the most compact cell for it) or n (`supercell : 3`, n × `transform`). B2's scan is then only reported, and D0 still converges each defect's production cell. |

#### C: screening

Details: [Phase C](#phase-c-screening-and-ground-states). The screening runs are cheap on purpose:
Γ point only, possibly a lower cut-off, few geometry steps. Their energies rank candidates;
phase D redoes the survivors properly.

| Keyword | Default | Steps | What it does |
|---|---|---|---|
| `configuration_margin` | — | C1 | Before any CASTEP run, drop arrangements whose configurational energy (MACE energy above the lowest arrangement of the same composition) is more than this, e.g. `1.0 eV`. Off unless given; needs `mace : true`. MACE has no charge or spin, so keep it generous. Replaces `triage` and `triage_window`. |
| `screen_charges` | `-1 0 1` (*list*) | C2 | The charge states every candidate is screened in, except one with its own `q=` in `%block defects`. |
| `screen_spins` | `both` | C2 | The starting spins: `low` is the lowest the electron count allows (0 or 1); `both` adds that + 2. They are starting values: the results keep the final spin. |
| `screen_cutoff` | the production cut-off | C2, C2b, C5, C6, C7 | The cut-off of the screening runs. |
| `screen_max_steps` | `50` | C2, C5, C6, C7 | `geom_max_iter` of the screening relaxations. A run that reaches it is logged as failed but keeps its last energy, marked as not converged. |
| `%block screen_overrides` | — | C2, C2b, C5, C6, C7 | Any other `.param` keywords for the screening runs, e.g. a looser `elec_energy_tol`. |
| `level_margin` | `0.1 eV` | C2, C2b, C5, C6, C7 | How far inside the host's band edges (from the pristine cell at the same settings) an eigenvalue must lie to count as a defect level in the gap. |
| `dispersion_check` | `warn` | C2b, C3 | The Γ-only check: `warn` (a warning if the screening cell is too small), `stop` (and C3 stops before it filters) or `off` (not run). |
| `dispersion_tol` | `0.05 eV` | C2b | The most a gap level may spread across C2b's 2 × 2 × 2 k-points in the screening cell. |
| `dispersion_defect` | the level nearest a band edge | C2b | The defect C2b checks: an id (`d012`, its lowest screened state) or `id:q` (`d012:-1`). |
| `hull_margin` | `0.2 eV` | C3 | Keep an arrangement if its lowest formation energy comes within this of its composition group's defect hull somewhere in the gap. Allow for the error of the screening settings. |
| `require_gap_states` | `none` | C3 | Keep only defects with `occupied` gap levels, `empty` ones, or `both` (an occupied and an empty level in the same spin channel: an internal transition), in a stable charge state. |
| `require_spin` | — | C3 | Keep only defects with this final spin (unpaired electrons, within 0.25) in a stable charge state, e.g. `2` for a triplet. |
| `require_bound` | `false` | C3 | Keep only pairs that are bound against their components somewhere in the gap: E_b = Σ E_f(parts) − E_f(pair) > 0. The components must be candidates too. |
| `distort_charges` | `stable` | C3 | The charge states that go on to the distortions: `stable` (lowest in energy for that defect somewhere in the gap) or `screened` (every one with a usable energy). |
| `distortions` | `-0.6 -0.5 … 0.5 0.6` (*list*) | C4 | ShakeNBreak bond distortions of the defect's nearest neighbours: `-0.4` brings them 40 % closer, `0.4` takes them 40 % further. One relaxation per factor, charge state and survivor. |
| `distort_around` | `anchor` | C4 | The centre of the distortion: `anchor` (the first component's atom) or `vacancy` (the vacant site, for defects that have one). |
| `%block oxidation_states` | guessed | C4 | One `El  value` per line, for ShakeNBreak's count of neighbours to distort: n = q − Σ (oxidation state put there − that of the site), and that many neighbours (the absolute value), or 8 minus it above 4. Without the block, pymatgen guesses the host's, and each dopant takes its common oxidation state closest to that of the element it replaces. |
| `rattle_stdev` | 10 % of the host bond | C4 | The standard deviation (Å) of the random displacements of the other atoms. |
| `rattle_dmin` | 80 % of the host bond | C4 | The shortest distance (Å) between atoms that rattling may produce. |
| `distortion_min_gain` | `0.05 eV` | C6 | A distortion counts as found when it beats the screened structure by this. |
| `share_rounds` | `2` | C6 | How many rounds of trying a winning distortion in the other charge states of its defect. |
| `keep_structures` | `1` | C6 | How many distinct structures, lowest first, of each defect and charge state go on to phase D (C7 excites the lowest). |
| `excite` | `false` | C3, C7, E3 | Run C7: the excited state of each defect's ground state (ΔSCF at the screening settings). Only the defects that pass, and every native defect, go on to D; native defects also skip C3's property filters. E3 takes its internal transitions from C7. With `false`, everything C6 kept goes on, and E3 counts deep levels only. |
| `excite_window` | — | C7 | `lowest highest` in eV: a charge state passes if its ZPL (or vertical energy, with `excite_relax : false`) lies inside. Without it, any converged excited state passes. |
| `excite_min` | `0.4 eV` | C7 | The smallest step from an occupied to an empty gap level (one spin channel, Kohn–Sham) taken as the excitation: C7 calculates the smallest one at least this big. |
| `excite_relax` | `true` | C7 | Relax the excited state (its energy is then the ZPL, and ΔQ is given). `false`: a single point at the ground-state structure (the vertical energy, above the ZPL), for a CASTEP build whose ΔSCF has no excited-state forces. |
| `%block excite_overrides` | — | C7 | Any other `.param` keywords for C7's runs (ground and excited states), after `screen_overrides` and `metals_method : dm`, e.g. `deltascf_smearing : 0.01` for degenerate levels or more `nextra_bands`. |

#### D: production

Details: [Phase D](#phase-d-production-cells-and-relaxations).

| Keyword | Default | Steps | What it does |
|---|---|---|---|
| `series_tol` | `0.05 eV` | D0 | A defect's production cell is converged at the smaller of two neighbouring sizes when E_D − E_host + E_corr of every charge state changes by less than this between them. |
| `%block production_overrides` | — | D0, D1 | `.param` keywords for the production runs (the relaxations and the pristine single points), e.g. `geom_energy_tol`, `geom_force_tol`, `geom_max_iter`, `elec_energy_tol`. A relaxation that reaches `geom_max_iter` is flagged and left out of E, so raise it here. States whose spin is 0 run without spin polarisation unless this block sets `spin_polarized`. |
| `freeze_radius` | `0` (off) | D0, D1 | An untested idea: hold every atom further than this (Å) from all of the defect's components fixed (`%block ionic_constraints`). Compare with a full relaxation before relying on it. |
| `finetune` | `false` | D2 | Write MACE training data from D0's and D1's relaxations (`D_production/finetune/train.xyz`, and `charged.xyz` for the charged defects) and a script that fine-tunes `mace_model` on it (`finetune.sh`), for phase F. |
| `finetune_args` | — | D2 | Extra options added to the `mace_run_train` command in `finetune.sh`. |

#### E: thermodynamics

Details: [Phase E](#phase-e-thermodynamics).

| Keyword | Default | Steps | What it does |
|---|---|---|---|
| `charge_correction` | `lz` | C3, D0, E1 | The image-charge correction for charged defects, with ε₀ from A2.3: `lz` (Lany–Zunger), `mp` (first-order Makov–Payne) or `none`. Used for the screening hull, D0's comparison of sizes and the final formation energies. |
| `gap` | from A2.2 | E2 | The band gap for E2 and E3: the range of Fermi levels, the window in which transition levels are found, E_f at the CBM, and the free carriers (the conduction bands are shifted by the difference). Put an experimental or hybrid value here; the defect energies themselves are not corrected. |
| `fermi_temperature` | — | E2 | Find the self-consistent Fermi level, and the defect and carrier concentrations, at this temperature (K), at each chemical-potential limit. Only the defects that reached phase D are included. |
| `quench_temperature` | — | E2 | With `fermi_temperature`: freeze each defect's total concentration at `fermi_temperature`, then share it out between its charge states at this temperature (K). |
| `deep_level_margin` | `0.5 eV` | B3, E3 | A transition level at least this far from both band edges counts as deep, and its defect goes on to phase F. B3's Γ-only floor is for a level this far inside the gap. |

#### Runner settings

These are `supercell_convergence.py`'s keywords for how the CASTEP jobs run (see
`README_supercell_convergence.md` for the details). In the project file they apply to every
CASTEP step. A runner block changes them for some steps: `%block runner_host` for every CASTEP
step of phase A, `%block runner_screen` for every one of phase C, `%block runner_production` for
D0 and D1, and `%block runner_converge`, `runner_relax`, `runner_bands`, `runner_dielectric`,
`runner_phases`, `runner_dispersion`, `runner_drelax`, `runner_rank`, `runner_excite` or
`runner_cells` for one step ([How a run works](#how-a-run-works) has the table). They hold
runner keywords only, and may hold a `%block partitions`. The MACE jobs use only `submit_cmd`,
`status_cmd` and `poll` from here ([MACE](#mace) has their own settings).

| Keyword | Default | What it does |
|---|---|---|
| `jobscript` | required | Your jobscript template, as a file or as `%block jobscript`. It needs `{SEED}` where the seedname goes, and `{NCORES}`, or on whole-node machines `{NODES}`, `{TASKS_PER_NODE}` and `{CPUS_PER_TASK}`. Other placeholders: `{JOBNAME}`, `{PARTITION}`, `{NKPTS}`, `{MEM_GB}`, `{MEM_MB}`, `{MEM_PER_NODE_GB}`, `{MEM_PER_CORE_GB}`, `{MEM_PER_CORE_MB}`. The runner sets the partition and `--time` itself; the template's `--time` is the longest any job may run, unless `max_walltime` is set. |
| `cores_per_node` | — | Cores per node on a machine that hands out whole nodes (128 on ARCHER2). Jobs are then laid out as nodes × processes per node, chosen for k-point parallelism. |
| `max_nodes` | `1` | The most nodes a job may use (with `cores_per_node`). |
| `max_cores` | `32` | The most cores a job may use on a machine that hands out cores (without `cores_per_node`). |
| `cores` | — | An exact core count for every job. |
| `%block partitions` | — | One partition per line: `name  GB_per_core  [max_cores]  [#SBATCH options]`, e.g. `highmem  4  --qos=highmem`. A job goes on the first partition where it fits in memory with full nodes; if none fits, on the best layout with fewer processes per node. One inside a runner block replaces the whole list for those steps. |
| `mem_per_core` | — | Without `%block partitions`: the memory per core (GB) jobs must fit in; a job that can't is warned about. |
| `mem_safety` | `1.05` | Factor on the memory estimated from the dry run. |
| `mem_replicated` | `auto` | The fraction of the dry run's memory that every process keeps its own copy of. `auto` starts at 0.25 and is measured from each finished job. |
| `baseline_mb` | — | Replaces CASTEP's baseline memory estimate (MB). |
| `omp_threads` | `1` | `auto` or a number: OpenMP threads per process on cores left idle when nodes are underpopulated for memory. |
| `max_walltime` | the template's `--time` | The longest a job may run (`H:MM:SS`, or `D-HH:MM:SS`), instead of the template's `--time` (or a partition line's `--time=`). The first jobs of a step ask for this; with `walltime : auto` later ones ask for less once finished jobs show what they need. A relaxation stops a little before it and carries on in a new job; a single point can't, so give single points enough. |
| `walltime` | `auto` | `auto`: once jobs have finished, later ones ask for what they need (scaled from the measured time per SCF cycle) instead of the longest walltime; `fixed`: always the longest. |
| `walltime_safety` | `1.5` | Factor on the walltime estimate. |
| `sp_walltime` | — | The walltime (`H:MM:SS`) of the first single points, before there is anything to estimate from, if shorter than the longest. One that runs out is resubmitted with the longest. |
| `max_restarts` | `10` (`0` for C2, C2b, C5, C6, C7) | How many times a job that runs out of time, or stops in a way the runner can recover from, is resubmitted; a relaxation carries on from its `.check`. |
| `delete_check` | `false` | Delete each calculation's `.check` file once it has finished; they can be large. |
| `dryrun_on` | `compute` | Where the dry runs go (one per distinct input, for the k-points and memory): `compute`, as small batch jobs from your template; `login`, where `dwf.py` runs, with `dryrun_cmd`. On ARCHER2, don't send them to the shared `serial` partition with a whole-node template, which it rejects: use `login`, or `dryrun_partition : standard`. |
| `dryrun_cmd` | `castep.mpi --dryrun {seed}` | The command for `dryrun_on : login`, e.g. `castep.serial --dryrun {seed}`. |
| `dryrun_timeout` | `900 s` | How long a login dry run may take. |
| `dryrun_jobscript` | the jobscript | A separate template for dry-run jobs. |
| `dryrun_partition`, `dryrun_time`, `dryrun_options` | — | The partition, time and other `#SBATCH` options of dry-run jobs. |
| `dryrun_cores`, `dryrun_mem_gb` | `1`, `4` | The cores and memory (GB) of a dry-run job. |
| `dryrun_pack` | `1` | With `dryrun_on : compute`: run up to N of a step's dry runs side by side in one job, one process each ([How a run works](#how-a-run-works)). Needs a jobscript (or `dryrun_jobscript`) whose CASTEP line uses `srun`. |
| `submit_cmd` | `sbatch` | The command that submits a job. |
| `status_cmd` | `squeue` or `qstat` | The command that lists queued and running jobs. |
| `poll` | `120 s` | Seconds between checks of the queue. |
| `max_jobs` | no limit | The most jobs, and dry runs, of one step in the queue at once. |
| `tolerate_failures` | `false` (`true` for C2, C2b, C5, C6, C7, D0 and D1) | A job that fails is marked as failed and the others carry on, instead of the run stopping. `--resubmit-failed` starts failed jobs again. |
| `pack` | `1` | Run up to N of a step's jobs side by side on one node, in one job, each on `cores_per_node`/N cores and its share of the node's memory ([How a run works](#how-a-run-works)). Needs `cores_per_node` and a jobscript whose CASTEP line uses `srun`. |

`--prepare-only`, `--no-wait`, `--resubmit-failed` and `--force` are command-line switches
([Command line](#command-line)), not project-file keywords.

**Retired keywords.** A project file that still has one stops with what to do instead:
`series_defect`, `series_charge`, `series_sizes`, `series_choice` and `%block runner_series` went
with the CASTEP size series of one defect (B2 now scans every candidate, and D0 converges each
defect's production cell; `series_tol` is D0's tolerance now); `scan_sizes` is now
`scan_lengths`, in Å; `finetuned_model` went (B2 and C1 use `mace_model`, and the fine-tuning
data come after D1, in D2); `triage`, `triage_window` (now `configuration_margin`) and
`zpl_window` (C7 filters on the ΔSCF ZPL) went earlier.

### 2. The production `.param` (`param`)

Your own CASTEP `.param`. A1 copies it into `A_host/production.param`, and every later job
starts from that file. Any keyword not in this table (`xc_functional`, `elec_energy_tol`,
`smearing_width`, mixing, geometry tolerances, ...) is passed on to every job unchanged.

| Keyword | What the workflow does with it |
|---|---|
| `task` | Set for every job: `geometryoptimization` for relaxations, `singlepoint` for single points, `dielectric_task` for the A2.3 run. Yours is ignored. |
| `cut_off_energy` | A1 sets it on each convergence run and writes the converged value into `production.param`, replacing yours. With `converge : false`, yours is kept. The screening runs (C2, C2b, C5, C6, C7) use `screen_cutoff` if it is set. |
| `basis_precision` | Removed by A1 when it sets `cut_off_energy`. |
| `charge` | Set per job, to the charge state; removed from `production.param` and from the host, competing-phase and pristine-supercell jobs. Leave it out. |
| `spin` | The starting total spin (unpaired electrons), set per defect job: from `screen_spins` in C2, from the final spin of the screening run in C2b, C5, C6, C7, D0 and D1 (for C7's excited state, of its ground-state run). Removed from `production.param` and from the host and phase jobs (unless a phase line sets it). A starting value only: with `fix_occupancy : false`, CASTEP holds it for `spin_fix` SCF cycles and then lets it relax. |
| `spin_polarized` (or `spin_polarised`) | Either spelling works: a job that sets it replaces yours, whichever spelling you used. `true` on every screening run (C2, C2b, C5, C6, C7) and on D0 and D1 runs with a non-zero spin; `false` for D0's and D1's spin-0 states unless `%block production_overrides` sets it, and for the A2.3 run without `dielectric_param`. The host and phase jobs use yours. |
| `fix_occupancy` | `true` for the A2.3 run without `dielectric_param`, `false` for `metal` phases; otherwise yours. |
| `geom_max_iter` | `screen_max_steps` on the screening relaxations; yours, or `%block production_overrides`, elsewhere. A production relaxation that reaches it is flagged and left out of E. |
| `write_cell_structure` | Set to `true` on relaxations: their final structure is read from `<seed>-out.cell`. |
| `run_time` | Set on relaxations, a little below the job's walltime, so CASTEP stops cleanly and writes its `.check` before the time limit (not on screening runs, which are not restarted). |
| `continuation` | Yours is removed. Set to `default` when a relaxation is resubmitted to carry on from its `.check` (which removes any `reuse`: CASTEP takes one or the other). |
| `reuse` | Yours is removed. C7's excited-state runs set it to their ground state's `.check`. |
| `metals_method` | `dm` on C7's runs (CASTEP's ΔSCF constraints need density mixing), unless `%block excite_overrides` sets it; yours elsewhere. |
| `calculate_deltascf`, `deltascf_method`, `deltascf_checkpoint`, `%block deltascf_constraints` | Set on C7's excited-state runs only: `true`, `simple`, the ground state's `.check`, and the occupations of the two bands (the occupied one emptied, the empty one filled, in one spin channel). |

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
| `%block kpoints_list`, `symmetry_ops`, `ionic_constraints`, `ionic_velocities`, `cell_constraints` | Dropped: they refer to the primitive cell's k-points or atoms. (`freeze_radius` writes its own `ionic_constraints`.) |
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
any job may run, unless `max_walltime` is set. It sets `OMP_NUM_THREADS`, and with threads adds
`OMP_PLACES`, `SRUN_CPUS_PER_TASK` and `--cpus-per-task` if they are missing. A dry-run job
(`dryrun_on : compute`) is made from the same template (or `dryrun_jobscript`), with the CASTEP
line run as one process with `--dryrun`.

### 7. The MACE jobscript template (`mace_jobscript`)

A Slurm script for one MACE job: your usual header (account, partition, QoS, one node, a
`--time` long enough for the job's share of the relaxations), the modules and environment
mace-torch needs, and one line with `{COMMAND}`:

```bash
#!/bin/bash
#SBATCH --job-name={JOBNAME}
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH --account=e89
#SBATCH --partition=standard
#SBATCH --qos=standard
module load cray-python
{COMMAND}
```

| Placeholder | Replaced by |
|---|---|
| `{COMMAND}` | its line becomes one line per worker, `mace_pack` of them, each in the background, then `wait`: `env OMP_NUM_THREADS=<mace_threads> <mace_python> <dwf.py> --mace-worker tasks_NN.json > worker_NN.log 2>&1 &`. Anything in front of `{COMMAND}` on its line goes in front of each (e.g. `srun --exact --ntasks=1 --cpus-per-task=8 {COMMAND}` starts each worker with its own `srun`; without it the workers run on the job's node directly, which is enough for one node) |
| `{JOBNAME}` | `dwf_mace_<step>_job_NNN` |

The job's folder (`<phase>/jobs/mace_<step>/job_NNN/`) holds `job.sh`, the task files (the
structures and settings), each worker's log, `slurm-<id>.out` and, once every worker has
finished, `finished`. Each worker writes every relaxation's result to its file in
`B_candidates/mace/` as soon as it has it.

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

---

## Testing so far, and what to check first

Everything has been run against a mock CASTEP and a mock Slurm, with the real MACE
(medium-mpa-0), ShakeNBreak 3.4.4, pymatgen and spglib, on CaO:Sb. The mock gives made-up
energies, spins and band structures, so these runs check the workflow's logic, not its physics.

| Test project | What it covered |
|---|---|
| `M1`, `M2` | A to E with the MACE cell strategy: MACE jobs through the mock Slurm (`mace_jobscript`, `mace_pack`, `mace_max_jobs`), doped's compact cells (16 to 144 atoms), the scan of every candidate, the Γ-only floor from the band masses, C1 reusing B2's relaxations, C2b (warn and stop), D0 converging at the size below the screening cell (`M1`) and going a size up to the cap (`M2`, a tighter `series_tol`), `keep_structures : 2` carried into another cell, D2 |
| `M3`, `M4` | MACE jobs that end without their relaxations (resubmitted, then a stop after three tries); `--no-wait` with the jobs running in the background; a floor beyond the ladder |
| `Final` | A to E with MACE: convergence, DFPT, competing phases and limits, generated candidates, `scan : false` with `supercell : 2`, C1's MACE run here (no `mace_jobscript`), `configuration_margin`, `require_gap_states`, `hull_margin`, distortions and sharing, D2's fine-tuning data, Fermi level with a quench |
| `Opts2` | A to E with `mace : false`, a `transform` (cubic cells), antisites, native and dopant pairs, `screen_cutoff`, `%block screen_overrides`, `require_spin`, `require_bound`, `distort_charges : screened`, `distort_around : vacancy`, `%block oxidation_states`, `keep_structures : 2`, `%block production_overrides`, `freeze_radius`, `charge_correction : mp`, `gap`, `fermi_temperature` |
| `User` | A to E with Voronoi interstitials (`interstitials : dopants`), `supercell : 2` below the floor (warned, and C2b too), a quench |
| `Opts` | phase A with `converge : false`, `kspacing`, `dielectric`, `%block chemical_potentials` |
| `Async` | `--no-wait`, run again until the steps finish |
| `ExciteA` | `Charges` with C7: relaxed excited states, `excite_window`, `delete_check : true`, a restart of an excited state, `%block runner_excite` |
| `ExciteB` | `Charges` with C7: vertical energies (`excite_relax : false`), no window, a failed excited-state run |
| `ExciteC` | `Opts2` with C7: native defects past `require_spin` and `require_bound`, `keep_structures : 2` |

`dwf.py` replaces five scripts (one per phase) that passed results on through more files. The
`Final`, `Opts2` and `User` projects were run from scratch with both, and give the same results at
every step: the energies, spins and gap levels of every run, the survivors, distortions and
rankings, the production energies, the formation energies, levels and Fermi levels. Also tested
with `dwf.py`: carrying on after it was killed part-way through a step; a changed setting
(`require_spin`), which re-ran C3 onwards and reused the finished jobs; `--redo`; running single
steps and phases; `--until`; `--status`; and the example folder's phase A, with dry runs on the
login node.

Since then, A1 converges k-point grids (from `converge_kgrid_min` to `_max`) instead of spacings,
and `%block defects` takes `q=`. Run again, the test projects choose the same 8 × 8 × 8 grid and
cut-off as before, and phases B to D give the same results. The production spacing worked out
from the grid (about 0.045 1/Å) is finer than the 0.05 the old list of spacings gave, so A2.2's
grid and the competing phases' k-points are a little denser, which moves their energies (and the
chemical potentials) by a few meV. A project with `q=` on its lines (`Sb_Ca + Vac_O  all
q=-1`, `Vac_O  q=+2,+1`, `Sb_Ca` without) screened, relaxed and analysed each defect in only its
own charge states, and ran the size series in q = −1. A non-cubic grid list (`6 6 4` to
`16 16 14` in steps of `2 2 2`) was tested through A1 and A2.1, and the checks on the three
keywords on their own.

Runner blocks for single steps (`%block runner_<step>`) and `max_walltime` came last. Without
them, the test projects submit exactly the same jobs as before (every `job.sh`, layout and
walltime) and give the same results. `Final` was also run with a block for every step
(different `max_nodes`, `cores` and `max_walltime`, and for A1 its own partition line with
`--qos=short`): every job got its own step's nodes and longest walltime, the B3 size series too
(through `series.conv`), and the results were the same as without the blocks. A relaxation that
ran out of time under a 20-minute `max_walltime` was carried on in a new job with the same 20
minutes.

Then `pack`. `Charges`, `Opts2` and `User` were run with `pack : 4` for A1 and the screening
steps and `pack : 2` for A3.1 and D1, and gave the same results as without it at every step,
also with `--no-wait` and the jobs finishing in the background. The packs filled up as planned
(the last ones shared out evenly), a screening job that failed inside a pack was flagged while
the others carried on, and relaxations of a pack that stopped at their `run_time` were carried
on as jobs of their own. These runs also showed that a relaxation's walltime, worked out from
earlier jobs, left too little time once its `run_time` margin (15 minutes or more) was taken
off, so short relaxations were stopped and restarted needlessly; the margin is now added to the
estimate. Without `pack`, the test projects again submit the same jobs and give the same
results, except that some relaxations now ask for a little more time (e.g. 1:15:00 instead of
1:00:00).

Then `dryrun_pack`. With `dryrun_pack : 16` as well (their dry runs go to the compute nodes),
the same three projects ran their dry runs up to 16 to a job (e.g. `Opts2`'s 138 screening dry
runs in 9 jobs) and gave the same results, also with `--no-wait`. Without it, every dry-run
script (`dryrun.sh`) and job script is the same as before.

The first real run (CaO:Sb on ARCHER2) stopped after A2.1: CASTEP lists the atoms in its output
grouped by species in order of atomic number (O before Ca), keeping the order of the `.cell` only
within each species, and the scripts expected the order of the `.cell`, so they couldn't read
the relaxed structure. They now match CASTEP's atoms to their own by element, which works
whatever order CASTEP uses. The mock CASTEP now lists the atoms the same way: with it the old
scripts fail as they did on ARCHER2, and the new ones give the same results as before.

Then C7 (`excite`). The mock CASTEP now does a simple ΔSCF: it needs `metals_method : dm` and
the `.check` it reuses, stops if `reuse` and `continuation` are both set, raises the energy by
the gap between the two constrained bands (less 0.1 eV, or 0.25 eV and one atom moved for a
relaxation), and can stop a chosen relaxation at its `run_time`. `ExciteA` (window 1.0–1.5 eV):
the Sb_Ca–V_O complex (ZPL 1.95 eV) was dropped and did not reach D, Sb_Ca passed through its
q = +1 state (1.25 eV), and V_O went on as a native defect; the ground states' `.check` files
survived `delete_check : true`, and the excited state stopped at `run_time` carried on from its
own `.check`, with `reuse` removed. `ExciteB` gave the vertical energies, and its failed
excited-state run kept its defect as `unknown`, flagged again in E3. `ExciteC` took all 14
native defects past `require_spin : 2` and `require_bound` to D (22 defects instead of 8).
Without `excite`, `Charges`, `Opts2`, `User` and `Final` submit the same jobs as before and give
the same results at every step (`User` differs only in the mock's job ids in its failure
messages).

Then the first ZPL estimate went: C2's Kohn–Sham estimate, its column in C3's table, step C3b
and `zpl_window` (a project file that still has it stops with a note to use C7).
`require_gap_states : both` still checks for an occupied and an empty gap level in one spin
channel. The four test projects (`User` and `Final` without their `zpl_window`, which had kept
every survivor anyway) submit the same jobs and give the same results as before, apart from the
estimate itself and E3: without `excite`, E3 now counts deep levels only, so `Charges`'s
Sb_Ca–V_O complex, picked before only for its estimate, no longer goes to F (the other projects'
picks are the same, for their deep levels). `ExciteA` and `ExciteB` give the same results as
before.

Then the MACE cell strategy replaced the CASTEP size series of one defect (B3) and the single
supercell: B2 scans every candidate with MACE (as jobs on compute nodes with `mace_jobscript`),
B3 takes the worst case with the Γ-only floor, C2b checks the screening cell, and D0 converges
each defect's production cell. The mock CASTEP now treats any supercell of the rock-salt lattice
as a defect cell (not only cubic ones), has parabolic band edges (2.0 and 1.2 m_e) for B3's
masses, and spreads its defect levels across the k-points by 10 exp(−L/2.5 Å) eV for C2b. B3
found the masses (2.02 and 1.21 m_e, also from the irreducible k-points alone, and for a
made-up band edge at X, whose three points were found and folded onto Γ only by supercells that
should fold them), and C2b measured ξ = 2.50 Å. In `Charges`, `Opts2`, `User`, `Final`,
`ExciteA`, `ExciteB` and `ExciteC`, run with the transform's cubic cells and the floor (or
`supercell : 2`), the screening cell was the one the size series had chosen before, and phases
A and C give the same results as before at every step (in `ExciteB` apart from its excited-state
run, which failed on purpose before); D and E differ, as each defect now has its own cell. In
D0, relaxations of 512-atom cells that ran out of time were flagged and their defect kept the
biggest size whose runs had worked.

The real MACE (medium-mpa-0) on the mock's CaO showed something to watch for: its relaxation
energy of the neutral O vacancy grew with the cell, 0.48, 0.79 and 0.92 eV in 54-, 102- and
144-atom cells, with the six Ca around it pushed out by 0.12 to 0.17 Å, as for a charged
vacancy. A foundation model has no electrons to keep in the vacancy, so it relaxes the ions as if
there were none, and that relaxation is long-ranged. With `scan_tol : 0.02 eV` no candidate
converged within 14 Å, and the screening cell went to the largest size; the edge displacements
were small (0.01 to 0.03 Å). On a real host, look at B2's table before phase C: if vacancies
never converge for this reason, a larger `scan_tol` (or a cap with smaller `scan_lengths`) is
reasonable, since D0 checks the production cell with CASTEP anyway.

Not tested: `phases_from : doped` (no Materials Project access here), a real fine-tuned model
(`finetune.sh` is written but was not run), real CASTEP ΔSCF (C7), real Slurm for the MACE jobs,
and real CASTEP output for the DFPT permittivity and the `.bands` of spin-polarised runs.

On the first real runs, check:

0. **The MACE jobs** (B2): before the first run, `python3 dwf.py <project> B2 --prepare-only`
   writes the jobs' scripts to `B_candidates/jobs/mace_scan/prepared/`: check that `job.sh`
   loads what mace-torch needs and that `mace_python` and `dwf.py` are on `/work`. Then look
   at B2's table in `B_candidates/summary.txt` (see the O-vacancy note above) and B3's choice,
   floor and band-edge folding before phase C. CaO's conduction-band minimum is usually
   reported at X in PBE (an indirect Γ–X gap), which an odd multiple of the FCC primitive cell
   doesn't fold onto Γ: B3's table shows where A2.2 found it, and if B3 warns, consider a cubic
   `transform`, whose cells fold X.

1. **A2.3 dielectric:** CASTEP's DFPT needs norm-conserving pseudopotentials, fixed occupancies
   and no spin polarisation. Give `dielectric_cell` (with an NCP `species_pot`) and
   `dielectric_param`, or set `dielectric` yourself. `dwf.py` warns if the potentials don't
   look norm-conserving. Check that the `.castep` has both "Optical Permittivity
   (f->infinity)" and "DC Permittivity (f=0)", and that `A_host/summary.txt` shows the values
   printed there.
2. **The `.bands` reader** (A2.2, C2): the electron count per spin channel of a spin-polarised
   run is taken from the header when CASTEP gives one number per channel, otherwise from the
   Fermi energy. Compare the gap levels of one run (`screen` in `C_screen/results.json`) with
   its `.bands`.
3. **The screening spins** (C2): each run starts from a total spin (`spin`) and `SPIN=` on the
   dopant. With `fix_occupancy : false` (smearing) CASTEP holds it only for its first `spin_fix`
   SCF cycles (10 by default) and then lets it relax, so both starting spins can end in the
   same state; the results record the final spin. With `fix_occupancy : true` the spin is not
   freed.
4. **Gap levels** are compared with the host's band edges at the same settings without potential
   alignment: fine for sorting candidates, not for production levels (those come from E2).
5. **The band edges** come from A2.2's uniform Γ-centred grid, not a band-structure path: lower
   `bands_kspacing` if the edges may lie between grid points, or give `gap`.
6. **Packs** (`pack`, `dryrun_pack`): the mock Slurm doesn't check what a pack's job script asks
   for. On the first packed step, look at a `jobs/<step>/packs/pack_NNN/job.sh` (or
   `dryrun_NNN`), and check with
   `sacct -j <job id> --format=JobID,JobName,NTasks,AllocCPUS,Start,End,MaxRSS` that its job
   steps ran side by side, each with its own processes.
7. **C7 (`excite`)**, before relying on it: run one defect (e.g. `%block defects` with
   `Sb_Ca + Vac_O  all  q=-1`, and `--until C7`). In `C_screen/jobs/excite/`, check that the
   `.castep` shows CASTEP applying the constraints to the two bands named in the `.param`, that
   the excited state relaxed (or set `excite_relax : false` if CASTEP won't relax it), and that
   the ZPL in `C_screen/summary.txt` is near ADAQ's 0.53 eV for Sb_Ca–V_O⁻ (ADAQ used VASP with
   PAW potentials, so expect some difference). If the SCF won't converge with the constraints,
   try `deltascf_smearing : 0.01` (or more `nextra_bands`) in `%block excite_overrides`.
