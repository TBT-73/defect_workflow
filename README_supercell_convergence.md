# supercell_convergence.py

Supercell-size convergence of a defect formation energy with CASTEP, using ASE to build the
supercells. The script builds a series of defect supercells, sizes and submits the CASTEP jobs
for you (Slurm), relaxes each defect cell, and stops once the formation energy has converged
with cell size.

It computes

    E_f*(n) = E_defect(n) - E_host(n) + E_corr(n)

for each supercell size n: the formation energy without the chemical-potential and Fermi-level
terms. Those terms are the same for every size, so they don't affect convergence. E_corr is 0 for
a neutral defect and an image-charge correction for a charged one (see
[Charged defects](#charged-defects)). It stops when
|E_f*(n) - E_f*(previous n)| < `tol`, and prints the full expression for E_f.

The same machinery can also run any list of CASTEP calculations ([Batch mode](#batch-mode)):
the defect-workflow scripts (`dwf_*.py`) run all their CASTEP jobs that way except their size
series.

---

## Contents

1. [Requirements](#requirements)
2. [Quick start](#quick-start)
3. [What the script does](#what-the-script-does)
4. [The input file](#the-input-file)
   - [Charged defects](#charged-defects)
5. [The jobscript template](#the-jobscript-template)
6. [Resources: cores, nodes, partitions, memory](#resources-cores-nodes-partitions-memory)
7. [Walltimes and automatic restarts](#walltimes-and-automatic-restarts)
8. [MACE pre-relaxation](#mace-pre-relaxation)
9. [Dry runs](#dry-runs)
10. [Running, stopping and resuming](#running-stopping-and-resuming)
11. [Output files](#output-files)
12. [Batch mode](#batch-mode)
13. [Keyword reference](#keyword-reference)
14. [Troubleshooting](#troubleshooting)
15. [ARCHER2 example](#archer2-example)

---

## Requirements

- Python 3.6+ with `numpy` and `ase`.
- CASTEP, plus a batch system: Slurm by default. SGE/PBS work through `submit_cmd` and `status_cmd`.
- **Optional:** `mace-torch` (Python 3.9+) for the MACE pre-relaxation: `pip install mace-torch`.

On ARCHER2 a working environment is:

```bash
module load PrgEnv-gnu cray-python
python -m venv ~/myenv && source ~/myenv/bin/activate
pip install ase mace-torch        # mace-torch only if you use prerelax : mace
```

---

## Quick start

```bash
python3 supercell_convergence.py --example > CaO_SbCa.conv   # annotated template input
# edit CaO_SbCa.conv: cell, param, jobscript, defect, e_bulk, resources ...
python3 supercell_convergence.py CaO_SbCa --prepare-only     # build and check the inputs
nohup python3 supercell_convergence.py CaO_SbCa > out.log 2>&1 &   # run it
```

The script waits for its jobs, so run it with `nohup` (or in `tmux`/`screen`). You can stop it
and rerun the same command at any time: finished and queued calculations are picked up, not
repeated.

---

## What the script does

For each size *n* in `sizes`:

1. **Builds the supercell** n·T of your relaxed primitive (or conventional) cell. T is the identity unless you give `%block transform`.
2. **Inserts the defect** (substitutions and/or vacancies, e.g. `Sb_Ca` + `Vac_O`):
   - the components after the first go on their nearest sites to it, and the whole defect is put as close to the cell centre as it can be. The first component can go on any copy of its primitive-cell site, or of a site that a pure translation of the crystal makes identical to it (all four Ca of a conventional rocksalt cell, for example). The script picks the copy that puts the mean position of the components nearest the centre;
   - the relative geometry, and the orientation, are the same in every supercell: the complex is only moved, never turned;
   - the size table at the start shows how far the defect's centre is from the cell centre (`off-centre/A`). It isn't always zero, because a defect has to sit on lattice sites: when the cell centre is a lattice site, the midpoint of a nearest-neighbour Ca–O pair can get no closer to it than half a bond length;
   - defect cells are run **without symmetry** (`symmetry_generate` is removed), so the relaxation can break it. The pristine supercells keep it.
3. **Sets the k-points:** a Monkhorst–Pack grid with the same spacing as your primitive cell (or `kgrid`/`kspacing`).
4. **Optionally pre-relaxes** the defect cell with a MACE foundation model.
5. **Runs a CASTEP dry run** to get the number of k-points and the memory estimate.
6. **Plans and submits the relaxation:** it chooses cores/nodes, partition and walltime, writes `job.sh` from your template, and submits a geometry optimisation of the defect cell. Only the atoms move; the lattice is fixed at the bulk value (`fix_all_cell : true`).
7. **Keeps relaxations running:** a relaxation that runs out of time stops cleanly and is resubmitted to carry on.
8. **Gets the energy:** once a relaxation has converged, the script either takes the final energy of the relaxation (`singlepoint : false`) or runs a single point on the relaxed structure (`singlepoint : true`).
9. **Gets E_host:** with `bulk_mode : supercell` each pristine supercell is also run with identical settings and its energy used as E_host. With `bulk_mode : peratom`, E_host = N_host × `e_bulk`.

The first pass runs the two smallest sizes together. Each later pass adds the next size until
|ΔE_f*| < `tol` or the sizes run out.

---

## The input file

Settings go in `<seedname>.conv`, written like a CASTEP `.param`/`.cell` file:

- `keyword : value` lines (`=` or a space also work);
- `%block name … %endblock name` for blocks;
- keywords in any case, and `#` or `!` starting a comment.

File names are relative to the input file. Unknown or misspelt keywords are rejected with a
suggestion. `python3 supercell_convergence.py --example` prints a fully annotated template.

A minimal input:

```
cell      : CaO_convent.cell
param     : CaO.param
jobscript : job_template.sh

%block defect
Sb_Ca
Vac_O
%endblock defect

e_bulk      : -721.83 eV
tol         : 0.05 eV
sizes       : 2 3 4 5
singlepoint : false
```

### Defect notation

- `X_Y`: element X on a Y site (`Sb_Ca`).
- `Vac_Y`: a Y vacancy (`Vac_O`).
- `X_Y:k`: the k-th Y site of the primitive cell (`Vac_O:2`).
- `X_Y@s`: give the substituted atom a starting spin of s, in units of ħ/2 (unpaired electrons). For example, `Sb_Ca@3` writes `SPIN=3` on the Sb in the defect `.cell`. It's a starting guess only. It isn't used for the pristine cells, and it doesn't change the work directory name. Check the `net spin of system` line in the `.castep` header (or the dry run's): it should read 3.000. If it doesn't, CASTEP has kept its own default total spin; add `spin : 3` to the `.param` as well (the script uses it for the defect runs only). With `fix_occupancy : false`, CASTEP holds the total spin only for its first `spin_fix` SCF cycles (10 by default) and then lets it relax. With `fix_occupancy : true`, the occupancies, and so the spin, stay as set.
- Use the atomic number for vanadium (`23_Ca`), since `V_Ca` would be ambiguous.

### Important: settings must match `e_bulk`

Use the relaxed host cell, and the energy per atom from that same calculation: same cut-off,
functional and pseudopotentials as `param`. With `bulk_mode : supercell` this is checked for
you (the `E_host-N*e_bulk` column).

### Charged defects

Put the charge in **both** the `.param` and the `geom_param` (e.g. `charge : -1`). The relaxations
read `geom_param`, but the single points, whose energies are used, read `.param`. The script removes
`charge` (and `spin`) for the pristine cells. Before submitting anything it stops if the two files
disagree, and afterwards it reads the charge each defect actually ran with from its `.castep`.

Remember that the starting spin has to match the electron count: an even count needs an even spin
(in ħ/2), e.g. `Sb_Ca@2` for the triplet −1 state of Sb_Ca–V_O.

**Why a correction is needed.** A charged defect in a periodic cell interacts with its own images
and with the uniform neutralising background CASTEP adds. That spurious energy shrinks only as 1/L,
so the uncorrected E_f* creeps with cell size (by roughly 0.04–0.06 eV between the 2×2×2 and
3×3×3 CaO cells for q = −1) and may never get within `tol`. The correction removes most of it:

- **Makov–Payne, first order** (`charge_correction : mp`): `E1 = q² α_M / (2 ε L)`, the energy of a
  point charge q in a lattice of its images with a neutralising background, screened by the static
  dielectric constant ε. α_M/L is computed for the actual supercell by an Ewald sum, so any cell
  shape works (α_M = 2.837 for simple cubic, 2.888 for FCC and BCC, with L = V^(1/3)).
- **Lany–Zunger** (`charge_correction : lz`, the default): `[1 + c_sh (1 − 1/ε)] E1`. A real
  defect charge is spread out, not a point, and the screened interaction is smaller; Lany and
  Zunger found this scaling, with shape factor c_sh = −0.369 (simple cubic), −0.343 (FCC),
  −0.342 (BCC), about −1/3 otherwise. For large ε it is about 2/3 of E1.
- `charge_correction : none` turns it off (a note reminds you E_f* may not converge).

Set `dielectric` to the static dielectric constant (electronic + ionic, e.g. from DFPT); it is
required for a charged defect unless the correction is off. For CaO with q = −1 and ε = 11.95,
the Lany–Zunger correction is 0.117 eV for the 2×2×2 cell and 0.078 eV for the 3×3×3.

The results table then shows `q` and `E_corr`, and E_f* (the value compared with `tol`) is the
corrected one; `convergence.json` also has the uncorrected `E_f_raw`.

**Limits.** No potential alignment (it needs CASTEP's electrostatic potential), no dipole or
quadrupole terms, and ε is a single number (use an average for anisotropic hosts). These matter
less for convergence, which compares sizes, than for absolute formation energies. For final
formation energies, apply a full scheme (e.g. Freysoldt or Kumagai–Oba) in your analysis.

---

## The jobscript template

Copy the jobscript you normally use for CASTEP and put placeholders in it:

| Placeholder | Replaced by |
|---|---|
| `{SEED}` | the seedname (required) |
| `{NCORES}` | number of MPI processes |
| `{NODES}`, `{TASKS_PER_NODE}`, `{CPUS_PER_TASK}` | layout on whole-node machines (with `cores_per_node`) |
| `{JOBNAME}` | job name |
| `{NKPTS}` | number of k-points |
| `{PARTITION}` | chosen partition |
| `{MEM_GB}`, `{MEM_MB}` | estimated total memory |
| `{MEM_PER_NODE_GB}` | estimated memory per node |
| `{MEM_PER_CORE_GB}`, `{MEM_PER_CORE_MB}` | estimated memory per process |

Memory estimates include `mem_safety`. The script also sets `#SBATCH --partition`, the
partition's extra options and `--time` itself, replacing whatever the template has (in either
spelling, e.g. `-p` or `--partition`). If the template has no `#!` first line, `#!/bin/bash` is
added.

The `#SBATCH --time` in the template is the **longest any job may run**. See
[Walltimes](#walltimes-and-automatic-restarts).

The template can also go inline in the input file as `%block jobscript`.

---

## Resources: cores, nodes, partitions, memory

### Core-based machines

Each job gets up to `max_cores` cores, chosen for k-point parallelisation. When there are fewer
k-points than cores, all k-points run in parallel and the rest is filled with G-vector
parallelism. `cores` forces an exact count.

### Whole-node machines (e.g. ARCHER2)

Set `cores_per_node` and `max_nodes`, and use `{NODES}`/`{TASKS_PER_NODE}` in the template. The
script tries every nodes × processes-per-node layout and ranks them by:

1. the fewest k-points per process;
2. a process count that divides evenly with the k-points (e.g. 126 of 128 cores for 14 k-points);
3. the most processes;
4. the fewest nodes.

### OpenMP threads on underpopulated nodes (`omp_threads`)

When a job needs more memory per process than a full node allows, the script runs fewer MPI
processes per node, so some cores sit idle (e.g. 49 of 128). With `omp_threads : auto`, each process
runs OpenMP threads on those spare cores instead: as many as it has cores (`--cpus-per-task`), up to 4.
`omp_threads : N` uses up to N; `1` (the default) keeps one thread per process.

The script sets `OMP_NUM_THREADS` in each jobscript (replacing the template's value), and adds
`export OMP_PLACES=cores`, `export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK` and `--cpus-per-task`
if the template lacks them. The node and memory request don't change, so the threads cost nothing
extra. Expect a speed-up of tens of percent rather than a doubling, since not all of CASTEP is
threaded. Each thread adds a little memory, which `mem_safety` normally covers. The log shows the
layout, e.g. `12 nodes x 49 processes x 2 threads`, and the `.castep` header shows `OpenMP threads per
process`. Walltime estimates from earlier one-thread runs will be on the long side for threaded jobs,
which is safe.

### Partitions

```
%block partitions
standard   2   --qos=standard
highmem    4   --qos=highmem
%endblock partitions
```

Each line is `name GB_per_core [max_cores] [extra #SBATCH options]`. A job goes on the **first**
partition where it fits in memory with full nodes. Only if none fits are fewer processes put on
each node, on whichever partition gives the best layout; the log says so ("N of 128 cores per
node used, to fit the memory").

### Memory model

The memory per process is estimated from the serial dry run as

    (baseline + other + S × (r + (1 − r)/p)) × mem_safety

where S is the dry run's *Model* + *Electronic* memory, p the number of processes, and
r = `mem_replicated` (the fraction of S every process keeps its own copy of).

With `mem_replicated : auto` (the default):

- r starts at 0.25;
- it is then measured from the "Peak Memory Use" CASTEP prints at the end of each run, including a relaxation that stopped at run_time;
- the largest value from the biggest cell measured so far is used for every job planned afterwards, including resubmitted jobs.

The log shows each new value:

```
mem_replicated : 0.050 (measured from sc_3x3x3_defect_geom: peak 4.76 GB per process on 252 processes)
```

The 0.25 starting value is cautious. Until something has been measured, jobs may be placed on
underpopulated high-memory nodes. If you already know a good value (e.g. from an earlier run),
set it directly, e.g. `mem_replicated : 0.05`.

`baseline_mb` replaces CASTEP's "baseline" estimate, which is sometimes mostly reserved virtual
memory.

---

## Walltimes and automatic restarts

### Walltimes (`walltime : auto`)

- **The ceiling:** the template's `--time`, or a `--time=` on a partition line, is the most any job may ask for.
- **How the estimate works:** once jobs have finished, later ones ask for less. The script takes the measured time per SCF cycle and scales it to the new cell by k-points × electrons^a ÷ processes. The exponent a is fitted from the sizes done so far, and is 3 until there are two. That is multiplied by the SCF cycles the biggest finished relaxation needed, then by `walltime_safety` (1.5).
- **Rounding:** up to 15 minutes, with at least 30 minutes.
- **Single points** on a relaxed cell use that cell's own relaxation timing.
- **First single points** (`sp_walltime`): until a job has finished there is nothing to estimate from, so the first pristine-cell single points would ask for the full `--time` and wait longer in the queue. Set e.g. `sp_walltime : 4:00:00` to ask for that instead. If one runs out of time, it is resubmitted with the full `--time` (this counts towards `max_restarts`).
- `walltime : fixed` always uses the template's time.

### Clean stops

Each relaxation gets CASTEP's `run_time` a margin below its walltime: at least 15 minutes, or
1.5 × one SCF if longer. CASTEP then stops between geometry steps and writes its `.check`
before Slurm would kill it.

### Automatic restarts (`max_restarts`, default 10)

A job that doesn't finish is resubmitted automatically:

| What happened | What the script does |
|---|---|
| CASTEP stopped at run_time ("Insufficient time for another iteration") | carries on from the `.check` (`continuation : default`) |
| Slurm killed it at the time limit | relaxation: carries on from the `.check` if there is one, and uses a larger run_time margin next time; single point: starts again |
| CASTEP couldn't read the `.check`, or stopped without finishing writing it | puts the `.check` aside (or deletes it with `delete_check : true`) and restarts the relaxation **from the structure of its last geometry step**, read from the `.castep` (or `-out.cell`) |
| killed with no `.check` | restarts from its last geometry step rather than from scratch |
| a second unreadable `.check` for the same calculation | from then on that calculation always restarts from its last geometry step |

Every restart:

- uses the longest walltime;
- renames the old `.err` files;
- plans the job again with the memory measured so far, so a badly placed job can move to a fuller layout or another partition.

The script stops with an explanation if a job made no progress, already had the longest
walltime, and can't be given more processes.

A relaxation that reaches `geom_max_iter` without converging is **not** restarted
automatically. The script stops and tells you to raise `geom_max_iter` in the `.param`.

---

## MACE pre-relaxation

```
prerelax         : mace
prerelax_model   : medium-mpa-0
prerelax_fmax    : 0.05 eV/ang
prerelax_steps   : 500
prerelax_threads : 8
```

- **What happens:** each defect cell is first relaxed with a MACE foundation model, positions only with the cell fixed, using ASE's LBFGS. This runs on the machine the script runs on. CASTEP then starts from that structure, so it needs fewer geometry steps. The final structure and energy are still CASTEP's.
- **Where the results go:** `prerelax.json` holds the structure, step count, energy change and largest move; `prerelax.log` holds the optimiser log. A stored result is reused, and the log says so.
- **Changing settings:** a different model or tolerance rewrites the inputs of calculations not yet submitted.
- **Models:** `prerelax_model` takes any `mace_mp` model name (e.g. `matpes-pbe`) or a `.model` file.
- **Options:** `prerelax_rattle` adds small random displacements first, to break symmetry. `prerelax_device` can be `cuda`.
- **Reporting:** for each relaxation, the log reports how far CASTEP moved the atoms from the MACE structure.

Keep `prerelax_threads` modest on a shared login node.

---

## Dry runs

By default each dry run is a small batch job (`dryrun.sh`) made from your template. CASTEP is
run as one process with `--dryrun`, because the memory model expects a serial estimate. Use
these to send dry runs somewhere cheap:

```
dryrun_partition : standard
dryrun_time      : 0:20:0
dryrun_options   : --qos=standard
```

- `dryrun_jobscript` gives a separate template.
- `dryrun_on : login` runs them where the script runs, using `dryrun_cmd` (e.g. `castep.serial --dryrun {seed}`).
- **ARCHER2:** don't send dry runs to the shared `serial` partition with a whole-node template; it rejects exclusive jobs.

---

## Running, stopping and resuming

| Command / action | Effect |
|---|---|
| `--prepare-only` | build all inputs and jobscripts, submit nothing |
| `--no-wait` | submit what is needed, report, and exit |
| `--resubmit-failed` | resubmit the existing jobscript of a failed calculation once (after you've fixed it) |
| `--force` | reuse a work directory set up with different inputs (not recommended) |
| `--batch FILE` | run the jobs listed in a batch file instead of a size series ([Batch mode](#batch-mode)) |
| `touch <workdir>/STOP` | stop the script cleanly at its next poll, from any login node; submitted jobs keep running |
| rerun the same command | carries on where it left off |

At startup the script logs the node and process ID it runs on, and the path of the STOP file.

The work directory (`conv_<cell>_<defect>` by default) records the inputs in `setup.json`. If
you change the `.cell`, `.param`, defect, transform, k-point spacing or MACE settings, the script
refuses to mix results. Give it a new `workdir` instead.

A size that was already set up keeps the defect site it was set up on, even when an older version
of the script (which didn't centre the defect) set it up: its results stay valid, since it is the
same defect, only moved. The size table lists these sizes in a note. To have one centred, delete
its directories (or use a new `workdir`).

---

## Output files

In the work directory:

| File | Contents |
|---|---|
| `convergence.txt` / `convergence.json` | the results table: size, atoms, shortest image distance, k-grid, energies, E_f*, ΔE_f* (plus charge and E_corr for a charged defect) |
| `setup.json` | fingerprint of the inputs |
| `<seedname>.conv` | copy of the input file used |
| `sc_NxNxN_defect_geom/` | the relaxation |
| `sc_NxNxN_defect/` | the single point (with `singlepoint : true`) |
| `sc_NxNxN_bulk/` | the pristine supercell (with `bulk_mode : supercell`) |

Each calculation directory has the CASTEP inputs and outputs, `job.sh`, `dryrun.sh`,
`<seed>.dryrun.castep` and `job_info.json`. That last file records:

- the layout, memory and walltime planned;
- every job run (id, walltime, processes, how it ended);
- the measured timings and memory.

After restarts you may also see:

- `<seed>.cell.start`: the structure it first started from;
- `<seed>.check.bad`: an unreadable checkpoint;
- `*.err.<timestamp>`: old error files.

### Disk space: `delete_check : true`

`.check` files can be tens to hundreds of GB for the larger cells. With `delete_check : true`:

- each calculation's `.check` (and any `.check.bad`) is deleted once it has **finished successfully**, including ones that finished before the script started;
- an unreadable `.check` is deleted straight away;
- the `.check` of a relaxation that is still carrying on is kept until it converges.

Each deletion is logged with its size.

---

## Batch mode

The same machinery (dry runs, layouts, partitions, walltimes, restarts, STOP, resuming) can run
any list of CASTEP calculations instead of a size series. In batch mode the script builds no
supercells or defects: each job runs the structure it is given. The defect-workflow scripts
(`dwf_*.py`) run every CASTEP job this way except their size series (step B3), which is a normal
size-series run that builds its own cells. They build their structures with this script's own
`build_supercell`, `anchor_site` and `translation_copies`, so the cells match. You can also use
batch mode directly:

```bash
python3 supercell_convergence.py --batch jobs.batch
```

A batch file is written like a `.conv` file and holds:

- the runner settings: `jobscript`, `cores_per_node`, `max_nodes`, `%block partitions`,
  `walltime`, `max_restarts`, `delete_check`, `dryrun_*`, `poll`, ...;
- `param` (and `geom_param` for the relaxations): the `.param` every job starts from (the
  script sets `task`);
- `cell`: the `.cell` every job copies `species_pot` and its other settings from (not its
  structure or k-points);
- `kspacing` (for jobs without their own grid), `energy`, `workdir` (default
  `batch_<file name>`);
- `max_jobs` (the most jobs queued at once) and `tolerate_failures` (a job that fails is marked
  failed and the others carry on);
- `%block jobs`, one job per line: `name  structure-file  [option=value ...]`.

Job options: `kind=geom|sp` (a relaxation, carried on from its `.check` when it runs out of
time, or anything else), `task=...` (e.g. `phonon+efield`), `fix_cell=false` (a variable-cell
relaxation), `symmetry=true` (keep the `.cell`'s symmetry keywords), `kgrid=1,1,1`,
`offset=gamma`; anything else is a `.param` keyword for that job only (`charge=-1`, `spin=2`).
Structure files are read with ASE.

```
jobscript      : archer2.sh
cores_per_node : 128
max_nodes      : 2
param    : CaO.param
cell     : CaO_prim.cell
kspacing : 0.05
max_jobs : 8
tolerate_failures : true

%block jobs
host    CaO_prim.cell   kind=geom  fix_cell=false  symmetry=true
d1_q0   cand1.cell      kind=geom  kgrid=1,1,1  charge=0   spin=1  spin_polarized=true
d1_qm1  cand1.cell      kind=geom  kgrid=1,1,1  charge=-1  spin=0  spin_polarized=true
%endblock jobs
```

Each job gets its own directory, dry run, plan, walltime and restarts, as in a size series;
jobs that would give the same dry run share one. A job whose inputs change after it was set up
is set up again if it never ran; if it has run, the script stops (`--force` keeps its results).
`--resubmit-failed` starts failed jobs again from the beginning. The results are printed as a
table at the end and written to `batch_results.json` in the work directory. From Python, call
`run_batch(workdir, jobs, args, cell_text, param_text, ...)`: its comment in the script lists
the job options.

---

## Keyword reference

Defaults in brackets.

**Input files**

- `cell`: host cell, required.
- `param`: CASTEP `.param`, required. The script sets `task`.
- `geom_param`: a separate `.param` for the relaxations. With it set, a single point is always run with `param`.
- `jobscript`: template file, or `%block jobscript`.

**Defect and convergence**

- `%block defect`: required.
- `e_bulk`: eV/atom; required with `bulk_mode : peratom`.
- `tol` [0.05 eV].
- `charge_correction` [lz | mp | none], `dielectric`: for charged defects, see [Charged defects](#charged-defects).
- `sizes` [2 3 4 5].
- `%block transform`: supercell matrix T; size n uses n·T.
- `bulk_mode` [peratom | supercell].
- `energy` [auto | final | free | 0K]: auto uses the 0 K estimate if printed.
- `singlepoint` [true]: false takes the energy from the end of the relaxation.
- `workdir`.

**k-points**

- `kgrid`: the primitive MP grid to take the spacing from.
- `kspacing`: the spacing itself, in 1/Å (CASTEP's 2π units).

**Resources**

- `max_cores` [32], `cores`.
- `cores_per_node`, `max_nodes` [1].
- `omp_threads` [1]: auto | N, OpenMP threads per process on cores left idle by underpopulated nodes.
- `%block partitions`.
- `mem_per_core`: warning only, without partitions.
- `mem_safety` [1.05].
- `mem_replicated` [auto].
- `baseline_mb`.

**Walltime and restarts**

- `walltime` [auto | fixed].
- `walltime_safety` [1.5].
- `sp_walltime` [none]: walltime for a first single point with no estimate yet.
- `max_restarts` [10].
- `delete_check` [false].

**MACE**

- `prerelax` [none | mace].
- `prerelax_model` [medium-mpa-0].
- `prerelax_fmax` [0.05 eV/Å].
- `prerelax_steps` [500].
- `prerelax_rattle` [0].
- `prerelax_device` [cpu].
- `prerelax_threads` [8].

**Dry runs**

- `dryrun_on` [compute | login].
- `dryrun_partition`, `dryrun_time`, `dryrun_options`, `dryrun_jobscript`.
- `dryrun_cores` [1], `dryrun_mem_gb` [4].
- `dryrun_cmd` [castep.mpi --dryrun {seed}].
- `dryrun_timeout` [900 s].

**Scheduler**

- `submit_cmd` [sbatch].
- `status_cmd`: defaults to squeue or qstat.
- `poll` [120 s].

**Batch mode**

- `max_jobs` [no limit]: the most jobs queued at once.
- `tolerate_failures` [false]: a job that fails is marked failed and the others carry on.
- `%block jobs`, plus `param`, `geom_param`, `cell`, `kspacing`, `energy`, `workdir`: see [Batch mode](#batch-mode).

**Run control**

`prepare_only`, `no_wait`, `resubmit_failed`, `force`. These can also be set in the file.

---

## Troubleshooting

| Message / symptom | Cause and fix |
|---|---|
| `unknown keyword 'xyz'` | a typo, or a shell command pasted into the `.conv`; fix the line |
| `… was set up with a different param …` | inputs changed since the work directory was made; use a new `workdir` |
| jobs on highmem with most cores idle | `mem_replicated` is still at 0.25; it corrects itself after the first job finishes, or set it by hand |
| `could not read its .check file` warning | the `.check` was cut short, often because the **disk quota** is full. The relaxation carries on from its last geometry step. Check with `lfs quota -hu $USER /work/<project>` (or SAFE), and consider `delete_check : true` |
| `did not converge within geom_max_iter` | raise `geom_max_iter` in the `.param` (e.g. 100), add `continuation : default` to that calculation's `.param`, and rerun with `--resubmit-failed` |
| `the SCF did not converge` | raise `max_scf_cycles` or adjust the mixing, delete that calculation's directory and rerun |
| `ran out of time before it finished a single geometry step` | give it more nodes (`max_nodes`) or a longer `#SBATCH --time` |
| `Serial nodes are shared nodes` (ARCHER2) | dry runs went to `serial` with an exclusive template; use `dryrun_partition : standard` |
| script stopped when you logged out | it was started without `nohup`/`tmux`; rerun the same command |

**Known limitation:** if you delete a relaxation's `.check` **and** its `.err` files by hand
after a failed continuation, this version can't tell what happened. It stops with "finished
without completing". To get it going, remove `continuation : default` from that calculation's
`.param` and rerun with `--resubmit-failed`, or keep the `.err` files so it can restart on its
own.

---

## ARCHER2 example

`CaO_SbCa.conv`:

```
cell      : CaO_convent.cell
param     : CaO.param
jobscript : job_template.sh

%block defect
Sb_Ca
Vac_O
%endblock defect

e_bulk      : -721.83 eV
tol         : 0.05 eV
sizes       : 2 3 4 5
bulk_mode   : supercell
singlepoint : false

cores_per_node : 128
max_nodes      : 12
%block partitions
standard   2   --qos=standard
highmem    4   --qos=highmem
%endblock partitions

walltime     : auto
max_restarts : 20
delete_check : true

prerelax         : mace
prerelax_model   : medium-mpa-0
prerelax_threads : 8

dryrun_on        : compute
dryrun_partition : standard
dryrun_time      : 0:20:0
dryrun_options   : --qos=standard
poll             : 120
```

`job_template.sh`:

```bash
#!/bin/bash
#SBATCH --job-name={JOBNAME}
#SBATCH --time=24:00:00
#SBATCH --nodes={NODES}
#SBATCH --ntasks-per-node={TASKS_PER_NODE}
#SBATCH --cpus-per-task={CPUS_PER_TASK}
#SBATCH --account=e89
#SBATCH --partition=standard
#SBATCH --qos=standard

export OMP_NUM_THREADS=1
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

module load castep

srun --distribution=block:block --hint=nomultithread castep.mpi {SEED}
```

Run it:

```bash
module load PrgEnv-gnu cray-python
source ~/myenv/bin/activate
python3 supercell_convergence.py CaO_SbCa --prepare-only
nohup python3 supercell_convergence.py CaO_SbCa > out.log 2>&1 &
tail -f out.log
```

### Settings worth testing first (cost)

- **k-points:** the default keeps your bulk k-point density, which is very dense for large insulating supercells. Try `kspacing : 0.04` and compare E_f* for the 2x2x2 at both settings.
- **Cut-off:** formation energies converge with cut-off faster than total energies, and a fixed-cell relaxation needs no stress convergence.
- **Geometry tolerances:** `geom_energy_tol` is per atom, so it gets looser in total-energy terms as cells grow. Consider `geom_energy_tol : 5e-6 eV` with `elec_energy_tol : 5e-7 eV`, and `geom_max_iter : 100`.
