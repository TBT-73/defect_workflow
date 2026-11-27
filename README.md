# Defect workflow (dwf.py)

`dwf.py` runs the defect workflow of the flowchart (`defect_workflow_flowchart_A1.pdf`), phases
A to F, with one command: from a host crystal and its dopants to formation energies, transition
levels, the excited states of the defects that could be colour centres, and their luminescence
lineshapes. `dwf_workflow_report.pdf` explains every step, and why it is there, in more detail
than the flowchart can.

It is one file, `dwf.py`: the workflow (PART 2 of the file) and the job runner that runs its
CASTEP jobs (PART 1: dry runs, cores or nodes, memory, walltimes, restarts), on machines that
charge by the core (Cirrus) or by the node (ARCHER2). The runner, a script of its own until
November 2026 (`supercell_convergence.py`), also works on its own (`python3 dwf.py --conv NAME`,
`--batch FILE`), and has its own README, `README_runner.md`.

The MACE relaxations (B2, C1, C4) run in `dwf.py` itself, or as jobs on compute nodes that run
`dwf.py` as a worker (`mace_jobscript`, with blocks of their own for B2, C1 and C4); so do F2's
model trainings and phonon calculations (`model_jobscript`, best on a GPU).

A project is two files with one name: `NAME.parameters`, the science (the host, its dopants, the
cells, the thresholds of each step), and `NAME.run`, how its jobs run (job templates, cores,
walltimes, queues), which `python3 dwf.py NAME` reads together. `example_CaO_Sb/` is a complete
example project.

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

With `DWF` the directory that holds `dwf.py`:

```bash
cd example_CaO_Sb
python3 $DWF/dwf.py CaO_Sb --prepare-only     # write the first jobs' inputs, submit nothing
nohup python3 $DWF/dwf.py CaO_Sb > run.log 2>&1 &
python3 $DWF/dwf.py CaO_Sb --status           # which steps have finished (and results.dwf)
```

The run goes through every step, A1 to F2, and waits for its CASTEP jobs, so run it with
`nohup` (or in `tmux`/`screen`). If it stops for any reason, the same command carries on where
it stopped. `dwf_CaO_Sb/results.dwf` has everything in one file, as a `.castep` does for a CASTEP
job: every step's status, the keywords your project's files give, and each phase's results, A to
F, written again as each step finishes. Each phase's `summary.txt` has its part:
`E_thermo/summary.txt` the formation energies, transition levels and Fermi levels, and
`F_lineshape/summary.txt` the lineshapes of the excitations that made it through.

The first time, stop after phase B (`--until B`) and read `B_candidates/summary.txt` before the
screening: the candidate list (each candidate costs about (number of `screen_charges`) screening
runs in the screening cell, and each charge state a survivor carries on C4's MACE relaxations of its
distortions and three CASTEP relaxations in C5, in the bigger electronic cell: the carried-over
start and the two lowest MACE minima), and the screening and electronic cells B3 and B4 chose, with
what set them.

The example folder holds the project's two files and every file they name:

| File | What it is |
|---|---|
| `CaO_Sb.parameters` | the project's science, annotated: the host, dopant, cells and thresholds (the pair is what `python3 dwf.py --example CaO_Sb` writes) |
| `CaO_Sb.run` | how its jobs run on Cirrus, annotated: the CASTEP, MACE and model jobs' templates, cores, walltimes and queues |
| `CaO_prim.cell` | the host primitive cell (rocksalt, a = 4.8105 Å); A2.1 relaxes it |
| `CaO.param` | production settings (PBE; A1 sets the cut-off) |
| `cirrus.sh` | jobscript template for the CASTEP jobs on Cirrus |
| `cirrus_mace.sh` | jobscript template for the MACE jobs (B2, C1, C4) on Cirrus |
| `cirrus_gpu.sh` | jobscript template for the model jobs (F2's trainings and phonons) on a Cirrus GPU |
| `archer2.sh`, `archer2_mace.sh` | the same for ARCHER2 (`CaO_Sb.run` says what else to change) |
| `young_ng.sh`, `young_ng_mace.sh`, `young_ng_gpu.sh` | the same for UCL's Young (young-ng); `young_ng_gpu.sh` runs the MACE jobs on its A100s, and F2's model jobs |
| `CaO_ncp.cell`, `CaO_ncp.param` | the DFPT dielectric run (A2.3): norm-conserving potentials, fixed occupancies |
| `Ca.cell`, `O2.cell`, `Sb.cell`, `Sb2O3.cell` | competing phases (the elements in A3.1, all four in E1.1): fcc Ca, O₂ in a box, A7 Sb, senarmontite Sb₂O₃, near their experimental structures |

Before running it, check: the budget code (`--account`) and CASTEP module in `cirrus.sh`, and
the budget code and Python environment in `cirrus_mace.sh` and `cirrus_gpu.sh` (and its GPU
partition and QoS against the Cirrus documentation); the cores and walltimes of each step (the
runner blocks in `CaO_Sb.run` are starting points); the pseudopotentials (`CaO_prim.cell` has
no `species_pot`, so CASTEP uses its default on-the-fly ones); the cut-off in `CaO_ncp.param`;
and the competing phases (add any other that could limit Sb).

For a project of your own, `python3 dwf.py --example NAME` writes the annotated pair,
`NAME.parameters` and `NAME.run`, to start from. A project file of an older `dwf.py` (one file,
`NAME.dwf`) is split into the two by `python3 dwf.py --split NAME.dwf`, and a work directory it
made is brought up to date by `--migrate` ([Command line](#command-line)).

---

## Requirements

- Python 3.9+ with `numpy`, `scipy` and `ase`;
- `spglib` (B1, B3, B4: symmetry), `pymatgen` (the chemical-potential limits of A3.2 and E1.2,
  structure matching in C3 to C6, the oxidation-state guess in C4), `shakenbreak` (C4),
  `matplotlib` (the E3 and F plots);
- `doped` (B2: the most compact supercell for each length; without it, n × n × n primitive
  cells, or give a `transform`);
- `mace-torch` for the MACE steps (B2, C1, C4), or set `mace : false`: in the Python that runs
  `dwf.py` (it fetches a foundation model there) and in the one the MACE jobs run; version
  0.3.15 or later for F2's models (`mace_run_train` with the total-charge embedding), whose
  jobs want a GPU build (`model_device : cuda`);
- optional: `pymatgen-analysis-defects` for Voronoi interstitials, a Materials Project API key
  for `phases_from : doped`, which needs doped 3 and pymatgen 2026.9.23 or later, and so Python
  3.11 or later (older pymatgen can't read the Materials Project's replies, and doped 2.x takes
  only the legacy keys, whose API the Materials Project was closing in September 2025);
- CASTEP and Slurm (other schedulers through `submit_cmd` / `status_cmd`).

On Cirrus (as on ARCHER2), put the environment on `/work`: the compute nodes, which run the MACE
jobs, can't see `/home`. Reaching it through a link in your home folder is fine: a venv's `activate`
keeps the path it was made with, but `dwf.py` follows the link and gives the MACE jobs the `/work`
path. With your project code for `<project>`:

```bash
module load PrgEnv-gnu cray-python
python -m venv --system-site-packages /work/<project>/<project>/$USER/dwfenv
source /work/<project>/<project>/$USER/dwfenv/bin/activate
pip install ase spglib pymatgen shakenbreak matplotlib mace-torch doped
```

Without `mace_jobscript` the MACE relaxations run where `dwf.py` runs: fine on a workstation,
but B2 relaxes every candidate in every size, so on Cirrus or ARCHER2 give `mace_jobscript` and
let them run on compute nodes (and, for a big screen, B2 more of them in `%block runner_scan`:
[MACE](#mace)). Likewise give `model_jobscript` for F2's trainings and phonons.

---

## How a run works

**Order.** The steps run in order, A1 to F2 ([The phases](#the-phases) lists them). Each step
uses the results of the steps before it, so a step only starts once those have finished.

**Checkpoints.** When a step finishes, `state.json` in the work directory records it with a
fingerprint of everything it used: its settings (as written in the project's files), the files
it read and the results of the steps it needs. A finished step with the same fingerprint is
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

**The CASTEP jobs** all go through the job runner (`run_batch`), which plans each job (dry run,
cores or nodes, memory, walltime), submits it, waits and restarts a relaxation that runs out of
time, never with more than `max_jobs` jobs queued or running: a restart waits for a place, so a
pack (or array task) that ran out of time, whose calculations restart one job each, can't fill the
queue.
Every step uses its batch mode, which runs the structures `dwf.py` builds.

**The MACE jobs.** With `mace_jobscript`, the MACE relaxations (B2, C1, C4) run as Slurm jobs made
from that template: its `{COMMAND}` line becomes up to `mace_pack` workers side by side
(default `cores_per_node` / `mace_threads`: 36 on Cirrus, 16 on ARCHER2), each running
`python3 dwf.py --mace-worker tasks.json` with `mace_threads` threads. At most `mace_max_jobs` jobs
are queued or running at once, the relaxations are shared out between them, biggest first, and a
job's workers share its queue,
`tasks.json`: each takes the next relaxation no other worker has taken (a claim file in the job's
`claims/`), until none are left. A job holds its node until its last worker ends, so this keeps
the node busy to the end: with a list fixed for each worker beforehand, the node waited for the
slowest list, as a relaxation's cost goes with its steps, which nothing knows in advance. A `{NCORES}` in the template becomes
the cores the job's workers use (workers × `mace_threads`), so on Cirrus a job asks for, and is
charged for, just those. Each result goes into its own cache file as soon as it is done, so a job
that runs out of time loses only the relaxations it hadn't finished. They wait for a place to come
free and go into the next jobs, packed into as few as there are places (and no more than one job
per `mace_pack` relaxations), so a whole round of jobs that reach their time limit together
becomes a new round of as many jobs, not hundreds of small ones. A relaxation that a worker had
started but not finished (its claim file) counts an attempt; every one a job left counts, if it
started none (a job that can't run at all, e.g. a Python it can't find). After three, the run
stops and says how the last of those jobs ended (Slurm's time limit, or the last line of a
worker's log, e.g. a Python error); once that is fixed, `--resubmit-failed` gives them three more. Each job's `progress.txt` counts its relaxations
as they finish: its first line, e.g. `200/398 relaxations done, the last at 14:02:11`, then a line
for each with the count so far (`[14:02:11] 200/398  d555 ...: 432 atoms, 11 steps, 326 s`), so
`tail -f` follows it. While it waits, `dwf.py` logs how many of the step's are done every 10
minutes (`B2: 5,231/19,920 MACE relaxations done (26%); 50 jobs queued or running`). The jobs wait in the queue, and `--no-wait`,
`--prepare-only` and the `STOP` file work, as for the CASTEP jobs. A foundation model is downloaded
where `dwf.py` runs and copied into `<workdir>/mace_models/`, where the compute nodes can read it.
B2, C1 and C4 can each have settings of their own for these jobs, in `%block runner_scan`,
`%block runner_prerelax` and `%block runner_distort` in `NAME.run` ([MACE](#mace)): B2 relaxes every
candidate in every size it fits, by far the most relaxations of the three, so a big screen gives it
more jobs there, and the time they need (`max_walltime`, each job's `--time`; without it, the
template's); C4's are fewer, but in the electronic cell, the biggest structures MACE relaxes.

**The model jobs.** F2's trainings run `mace_run_train` (`model_command`) in each model's folder,
and F2's phonon calculations run `python3 dwf.py --phonon-worker spec.json` in theirs. With
`model_jobscript` each is a Slurm job made from that template (its `{COMMAND}` line becomes `cd
<folder> && <command> > <log> 2>&1`; `{JOBNAME}` and `{NCORES}` are filled in), waited for like the
CASTEP jobs; without it they run where `dwf.py` runs, one after the other. A training whose job ends
without a model, or a phonon job without its result, stops the run with the end of its log. A phonon
job saves its force constants every five minutes, so after a time limit `--resubmit-failed` carries
it on where it stopped. A changed training set or setting clears a folder's old model, which is then
trained again.

**Cores, nodes and walltimes for each step.** The runner settings in `NAME.run`
([Runner settings](#runner-settings): cores or nodes, walltime, partitions, ...) apply to every
CASTEP step. A runner block changes them for some steps only (the table below), and changes only
the keywords it gives. So on Cirrus the primitive-cell single points of A1 can run on 16 cores
for at most 20 minutes, while the production relaxations in the electronic cell (D1) get up to
a node and 24 hours:

```
max_cores : 64                 # every step, unless a block says otherwise
%block runner_converge         # A1
max_cores    : 16
max_walltime : 0:20:00
%endblock runner_converge
%block runner_production       # C8, D1, D2, F1 and F2
max_cores    : 288
max_walltime : 24:00:00
%endblock runner_production
```

On a machine that charges whole nodes (ARCHER2), the same blocks give `max_nodes` instead.

Three blocks cover a group of steps each, and the others one step each:

| Block | Applies to |
|---|---|
| `runner_host` | every CASTEP step of phase A, and the competing phases wherever they run (all small bulk cells): A1, A2.1, A2.2, A2.3, A3.1 and E1.1 |
| `runner_converge` | A1 |
| `runner_relax` | A2.1 |
| `runner_bands` | A2.2 |
| `runner_dielectric` | A2.3 |
| `runner_ref_phases` | A3.1 (the references at the screening settings) |
| `runner_phases` | E1.1 (the competing phases at the production settings) |
| `runner_screen` | every CASTEP step at the screening settings: C2, C3 (its single points), C5, C6 and C7 |
| `runner_filter` | C3 |
| `runner_drelax` | C5 |
| `runner_rank` | C6 |
| `runner_excite` | C7 (all three of its batches) |
| `runner_production` | every CASTEP step at the production settings: C8, D1, D2, F1 and F2 (its single points) |
| `runner_validate` | C8 |
| `runner_relaxations` | D1 |
| `runner_zpl` | D2 (all three of its batches) |
| `runner_ccd` | F1 |
| `runner_lineshape` | F2 |

You only need the blocks for the steps you want to change. A step uses the settings at the top
of `NAME.run`, then `runner_host`, `runner_screen` or `runner_production` for its group, then its
own block; the later one wins for any keyword both give. C2 has no block of its own (its name is
its group's): set it with `runner_screen`, and give the other steps their own blocks where they
should differ.

The screening steps (C2, C3, C5, C6, C7) use `max_restarts : 0` and `tolerate_failures : true` (a
run that fails is flagged, not rescued), and the production steps (C8, D1, D2, F1, F2)
`tolerate_failures : true`, whatever the settings at the top say; a runner block can change them.
The MACE and model jobs have settings of their own (`mace_jobscript`, `mace_pack`, `mace_max_jobs`,
with B2's, C1's and C4's own blocks, `runner_scan`, `runner_prerelax` and `runner_distort`;
`model_jobscript`), and share only `submit_cmd`, `status_cmd`, `poll` and `cores_per_node` (for the
default `mace_pack`) with the CASTEP jobs.

A block can also hold its own `%block partitions`, which replaces the whole list for those steps,
e.g. to send A1's jobs to a short-job QoS (`standard  2  --qos=short` on ARCHER2, for jobs of up
to 20 minutes). The settings of every step are checked when `dwf.py` starts. They don't change
any results, so changing them never makes a step run again: when you run `dwf.py` again, the
jobs it has not submitted yet, and the restarts of those already queued, use the new settings.

**Charged by the core (`charging : core`, Cirrus).** Cirrus shares its nodes (288 cores) between
jobs and charges for the cores a job asks for, so there a job should ask only for what it needs.
With `cores_per_node : 288` and `charging : core`, each job gets up to `max_cores` MPI processes,
chosen for k-point parallelisation (all the k-points side by side when there are no more of them
than `max_cores`, the rest G-vector parallelism), and asks for just those cores on one node. Its
memory comes with its cores (2.66 GB each on `standard`, 5.33 GB on `highmem`; Cirrus doesn't
take `--mem`), so it goes on the first partition where one core's memory holds a process; if
none does, each process gets as many cores as its memory needs (`--cpus-per-task`), on the
partition where that takes the fewest cores in all (`omp_threads : auto` runs OpenMP threads on
them). A job with more processes than a node has cores takes whole nodes with `--exclusive`
(Cirrus requires it, and charges for the whole nodes), up to `max_nodes` (by default as many as
`max_cores` needs), and uses all their cores. The dry runs ask for enough cores for
`dryrun_mem_gb` (4 GB: 2 cores on `standard`). The log shows what each job asks for, e.g.
`64 processes x 2 cores (128 cores of a shared node, standard)`. The jobscript needs `{NODES}`,
`{TASKS_PER_NODE}` and `{CPUS_PER_TASK}` (as in `cirrus.sh`).

**Fewer waits in the queue on Cirrus (`pack`, `dryrun_pack`).** Each job, and each dry run,
otherwise waits in the queue on its own. `pack : N` puts up to N of a step's jobs in one Slurm job,
which asks for all their cores on one node, as long as they fit (288 cores) and are on the same
partition: e.g. eight of A1's 16-core single points in one 128-core job. Each job keeps its own plan
and runs with its own `srun --exact`, asking with `--mem-per-cpu` for its cores' share of the pack's
memory (`SLURM_MEM_PER_CPU`, which Slurm sets in the job). `dryrun_pack : N` does the same for the
dry runs (2 cores each, so up to 144 at a time on a node; more take turns, as below):

```
dryrun_pack : 16               # every step: 16 dry runs to a job
%block runner_converge         # A1
max_cores    : 16
max_walltime : 0:20:00
pack         : 8               # eight single points to a job, 128 cores
%endblock runner_converge
```

A pack holds all its cores until its last job ends, and you are charged for them, so pack jobs
that take about as long as each other (dry runs, A1's single points, a step's screening runs)
rather than relaxations of very different sizes. The pack asks for the longest walltime of its
jobs; a job that has to be restarted runs on its own. Cirrus takes no `--mem` options for a job:
check on the first pack that it takes `--mem-per-cpu` on the job steps inside it (each job's
`slurm-<id>.out`, and `sacct`, below).

**Several small jobs on one node (`pack`, ARCHER2).** On a machine that charges for whole nodes,
however few cores a job uses, and never shares them between jobs (ARCHER2; `charging : node`, the
default with `cores_per_node`), `pack : N` in a step's block runs N of that step's CASTEP jobs
side by side in one Slurm job on one node, each on 1/N of its cores: with `pack : 4`, four jobs
of 32 cores each. It suits steps with many small jobs, such as A1's single points of the
primitive cell:

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

With `array` as well, the packs go as Slurm job arrays, one pack a task: each task is a node
running N jobs side by side, so a step of thousands of small jobs needs only a few submissions.
With `pack` at `cores_per_node` (40 on young-ng, 128 on ARCHER2), each job has one core. E.g.:

```
%block runner_screen           # C2, C3, C5, C6, C7: the screening jobs
max_nodes : 1
pack      : 4                  # four jobs side by side on a node (40 for one core each)
array     : 500                # the packs as job arrays of up to 500 tasks, one node each
max_jobs  : 500                # at most 500 tasks (nodes) queued at once: 2,000 jobs
%endblock runner_screen
```

- The jobs planned alike (the same processes, cores per process, memory and partition) go N to a
  task, in one array. A job too big for its share is planned on whole nodes, and goes in a plain
  array with the jobs planned like it.
- Every task is a full pack. The jobs left over, fewer than N, go as a pack of their own, each
  given more of the node, as above.
- `max_jobs` and `array_min` count tasks, each a node of N jobs.
- The array's job script is `jobs/<step>/packs/pack_array_NNN/job.sh`, with its jobs listed in
  `tasks.txt` beside it; each job's output, `slurm-<array job id>_<task>.out`, is in its own folder.
- With one core a job, each job's share of a node's memory is small (3.4 GB on young-ng), and a job
  takes up to N times as long as on N cores: it suits small cells, with `max_walltime` in mind.

The dry runs can be packed too, which matters with `dryrun_on : compute`, where each dry run is
otherwise a job of its own that waits in the queue and, on ARCHER2's standard partition, takes a
whole node. `dryrun_pack : N` (e.g. 32, for every step) runs up to N of a step's dry runs side
by side in one job, one process each with `dryrun_mem_gb` of memory, in
`jobs/<step>/packs/dryrun_NNN/job.sh`. With `dryrun_on : login` the dry runs don't go to the queue
at all, so it doesn't apply: where a serial CASTEP runs on the login nodes (`castep.serial` on
ARCHER2), that is the quickest way of all.

N can be more than a node holds at once: one dry run a core, and no more than the node's memory
holds at `dryrun_mem_gb` each. The job then runs that many side by side, and each core takes its
share of the rest one after another. Jobs that would give the same dry run share one (the same
cell, species, starting spins and `.param`; positions don't count without symmetry): in C2, one
for each composition group (B1), charge state and starting spin. ADAQ's CaO set (4,709
candidates in 511 groups, three charge states each) needs about 1,500 dry runs for its 14,127
relaxations, so four jobs on young-ng with:

```
dryrun_pack : 400              # up to 400 dry runs to a job: 40 at a time, 10 in turn on each core
dryrun_time : 1:00:00          # long enough for a core's dry runs one after another
```

A dry run usually takes well under a minute, but give `dryrun_time` room to spare: dry runs that
a job had no time for stop the step, and `--resubmit-failed` then sends each as a job of its own.

---

## Command line

```
python3 dwf.py NAME [STEP ...] [--until STEP] [--redo STEP] [--status] [--prepare-only]
                    [--no-wait] [--resubmit-failed] [--force] [--migrate]
python3 dwf.py --example [NAME]
python3 dwf.py --split OLD.dwf
```

| Option | Effect |
|---|---|
| `STEP ...` | run only these steps: names (`relax`), labels (`A2.1`) or phase letters (`C`); default all |
| `--until STEP` | stop after this step or phase (`--until B`) |
| `--redo STEP` | start this step again from scratch (can be repeated) |
| `--status` | list every step: finished (when, and its one-line result), still running, or not run; and write `results.dwf` again (nothing runs) |
| `--example [NAME]` | write the annotated example project, `NAME.parameters` and `NAME.run` (default `example`); stops if either is there already |
| `--split OLD.dwf` | split an older `dwf.py`'s project file (one file) into `OLD.parameters` and `OLD.run` beside it, leaving it as it is: each keyword in its file, the comments kept (their step labels as they are now), D1–D5's settings left out (each file lists them at the top); stops if either is there already |
| `--prepare-only` | write the inputs of the next CASTEP jobs (with their dry runs), submit nothing |
| `--no-wait` | submit what is needed and exit; run it again later to carry on |
| `--resubmit-failed` | start failed CASTEP jobs again, and MACE relaxations whose jobs gave up |
| `--force` | keep the results of jobs whose inputs have changed |
| `--migrate` | bring a work directory made by an older `dwf.py` (from when phase D held the ground and excited states) to this layout (below) |
| `--conv NAME` | the job runner on its own, without a project: the supercell size series of one defect in `NAME.conv` (with `--prepare-only`, `--no-wait`, `--resubmit-failed` or `--force`; [9](#9-the-job-runner-on-its-own-conv-and-batch-files), `README_runner.md`) |
| `--conv-example` | print an annotated `.conv` file, with every runner keyword |
| `--batch FILE` | the job runner on its own: the CASTEP calculations a batch file lists |

The project can be given as `CaO_Sb`, `CaO_Sb.parameters` or `CaO_Sb.run`: either way `dwf.py`
reads both. (`--mace-worker <tasks>.json` and `--phonon-worker spec.json` are what a MACE job and
an F2 phonon job run; you don't need them yourself.) For example:

```bash
python3 dwf.py CaO_Sb B1            # just the candidate list (quick), to check it
python3 dwf.py CaO_Sb --until B     # phases A and B, then stop
python3 dwf.py CaO_Sb E             # phase E again, e.g. after changing gap
python3 dwf.py CaO_Sb --redo filter # C3 from scratch, then whatever its results change
```

**Projects and work directories of an older `dwf.py`.** Until now a project was one file,
`NAME.dwf`, and phase D held the ground and excited states: D1–D5 trained a model of each
defect's ground states (with hybrid single points) and searched it, D6 rank, D7 excite and D8
validate were what are now C6–C8, in `D_states/`, and production, thermodynamics and lineshapes
were phases E, F and G, in `E_production/`, `F_thermo/` and `G_lineshape/`. `python3 dwf.py
--split NAME.dwf` writes the project's two files from the old one. `dwf.py` stops on a work
directory of the old layout and says so; once none of its jobs from D6 on are queued or running,
`python3 dwf.py NAME --migrate` moves C6–C8's jobs, structures and results from `D_states/` to
`C_screen/`, renames `E_production/` to `D_production/`, `F_thermo/` to `E_thermo/` and
`G_lineshape/` to `F_lineshape/`, changes the paths in every `results.json` and in `state.json`
to match, drops D1–D5 from `state.json` and writes each `summary.txt` again. `D_states/` keeps
what D1–D5 made, which nothing reads any more. (A step still `running` in `state.json` from D6
on stops it: let it finish with the older `dwf.py` first.)

The competing phases at the production settings were A3.1 and A3.2 (`phases`, `limits`) until
they became E1.1 and E1.2: in a work directory from then, `dwf.py` (any run, `--status` or
`--migrate`) moves their jobs from `A_host/jobs/phases/` to `E_thermo/jobs/phases/`, where E1.1
reuses those it still needs, and their results to `E_thermo/results.json`. If that A3.1 was still
running, its record is dropped, with a warning: cancel any of its jobs still in the queue (E1.1
relaxes only the phases of the dopants that reach D; one it needs that is then marked failed runs
again with `--resubmit-failed`). The `phases_doped.json` it kept is asked for again once (doped's
choice now records each phase's energy above the hull), from the kept Materials Project replies.

The checkpoints are brought up to date with the new paths. What then runs again: the steps added
since (A3.1 and A3.2, the references at the screening settings), C3 (its cut across a dopant's sites
uses them) and so C4 to C6 (C4 is now a search with MACE, whose minima C5 relaxes as well), and E1.1
and E1.2; the steps after C6 run again only where their inputs change. With `model : true` in the
old run, C6 now ranks C5's relaxations, with its rounds of sharing, in place of the model's minima.

A work directory of the `dwf.py` that ran the compounds at the screening settings as a step of their
own (C4 `dopant_phases`, with a cut on formation energy) needs no `--migrate`: that step's record is
dropped (its jobs stay in `C_screen/jobs/dopant_phases/`, unused), and C3, without the cut, and C4
onwards run again.

---

## Files

### What you give it

| File | Named by | Used for |
|---|---|---|
| `NAME.parameters` | the command line (`NAME`) | the science: every setting but how the jobs run ([glossary](#1-the-projects-two-files-nameparameters-and-namerun)) |
| `NAME.run` | the command line (`NAME`) | how the CASTEP, MACE and model jobs run ([glossary](#1-the-projects-two-files-nameparameters-and-namerun)) |
| production `.param` | `param` | the settings of every CASTEP job ([2](#2-the-production-param-param)) |
| host `.cell` | `host_cell` | the host structure, and `species_pot` etc. for every job ([3](#3-the-host-cell-host_cell)) |
| dielectric `.param` and `.cell` | `dielectric_param`, `dielectric_cell` | the A2.3 DFPT run (optional, [4](#4-the-dielectric-param-and-cell-dielectric_param-dielectric_cell)) |
| competing-phase structures | `%block phases` | A3.1 and E1.1 ([5](#5-competing-phase-structure-files-block-phases)) |
| jobscript template | `jobscript` | every CASTEP job ([6](#6-the-jobscript-template-jobscript)) |
| MACE jobscript template | `mace_jobscript` | the MACE jobs of B2, C1 and C4 (optional, [7](#7-the-mace-jobscript-template-mace_jobscript)) |
| model jobscript template | `model_jobscript` | F2's trainings and phonons (optional, [8](#8-the-model-jobscript-template-model_jobscript)) |

File names are relative to the project's files (the two are in one directory).

### What it writes

Everything goes in the work directory, `dwf_<name>/` next to the project's files (`workdir`
changes it):

```
dwf_CaO_Sb/
  results.dwf             everything in one file: every step's status, your settings and
                          each phase's summary (rewritten as the steps finish)
  state.json              which steps have finished, and with what inputs
  STOP                    touch it to stop a running dwf.py
  A_host/                 results.json, summary.txt,
                          production.param (your .param with the converged cut-off),
                          host_relaxed.cell (the relaxed host),
                          phases_doped.json and materials_project/ (phases_from : doped:
                          the phases doped chose, and the Materials Project's replies),
                          jobs/ (converge, relax, bands, dielectric, ref_phases)
  B_candidates/           results.json, summary.txt, ladder.json (the sizes),
                          mace/ (every MACE relaxation of B2 and C1: one file each;
                          <id>_<size>_seeded.json from the deepest minimum),
                          jobs/mace_scan/ (B2's MACE jobs, with mace_jobscript)
  C_screen/               results.json, summary.txt,
                          mace/ (C4's MACE relaxations: one file each),
                          structures/ (candidates, screen, distortions, drelax, share,
                          excite_relax, excite, validate),
                          jobs/ (screen, filter, drelax, share_1, ..., excite_relax,
                          excite_gs, excite, validate, validate_gamma, and mace_prerelax
                          and mace_distort: C1's and C4's MACE jobs)
  D_production/           results.json, summary.txt,
                          structures/ (relaxations: D1's, zpl: D2's excited states),
                          jobs/ (relaxations, zpl_gs, zpl_gs_gamma, zpl)
  E_thermo/               results.json, summary.txt, formation.png, jobs/phases (E1.1)
  F_lineshape/            results.json, summary.txt, ccd/ (F1's diagrams and 1D lineshapes),
                          lineshape/ (F2's spectral functions and lineshapes), model/ and
                          phonons/ (F2's models and phonon calculations),
                          jobs/ (ccd_gs, ccd_ex, lineshape_data)
  mace_models/            the foundation model, for the MACE jobs
```

`results.dwf` is the file to read: like a `.castep`, it has the whole run in one place. At the top,
every step with its status (as `--status` prints it: when it finished and its one-line result,
`running`, or `-` for not yet), then the keywords your project's files give, as you wrote them
(blocks on one line), then each phase that has run, A to F, with every step's table. It is written
again whenever a step finishes or stops, at the end of every run, and by `--status` (which runs
nothing, so it brings the file up to date for a work directory made before it existed). No step
reads it.

Each phase has one folder with the same layout:

- **`results.json`**: everything the later steps use from this phase, under the name of the
  step that worked it out. **A step reads only the `results.json` of earlier phases**, and the
  files named in them (the relaxed host, `production.param`, structures).
- **`summary.txt`**: the same results for you to read: for each step, when it finished, its
  one-line result and its table (the convergence test, the candidates, the filter verdicts, the
  ranking, the formation energies, ...); `results.dwf` has every phase's together. No step reads
  it, so editing it changes nothing.
- **`structures/`**: the structures the phase made, as CASTEP `.cell` files (lattice, positions,
  `SPIN=` where a starting spin is set): look at one with `ase gui`, or turn it into another
  format with `ase convert` (e.g. to `.cif`).
- **`jobs/<step>/<job>/`**: one CASTEP calculation each: `.cell`, `.param`, `.castep`,
  `-out.cell`, `.check` (unless `delete_check`), `job.sh`, `dryrun.sh` and `job_info.json`
  (what the runner decided and how it went). Each `jobs/<step>/` has a `batch_results.json`, a
  `progress.txt` (the count of its finished calculations as its first line, e.g.
  `1,201/4,980 calculations done (3 failed), the last at 14:02:11`, then a line for each as it
  finishes, with the count so far, e.g. `[14:02:11] 1,201/4,980  d555_q-1: done`, so
  `tail -f C_screen/jobs/screen/progress.txt` follows C2; the log has the count every 10 minutes),
  and with `pack` or `dryrun_pack` a `packs/` folder with the job script of each pack.

What each phase's `results.json` holds:

| Phase | Results (by step) | Used by |
|---|---|---|
| A | `converge`: cut-off, k-point spacing, `production.param`; `relax`: `host_relaxed.cell`, `e_bulk`; `bands`: VBM, CBM, gap, the `.bands` file; `dielectric`: ε₀, ε∞; `ref_phases`: the energies of the host and of the pure elements at the screening settings; `ref_limits`: μ at each limit at those settings, each dopant at its element | every later phase |
| B | `generate`: the candidates (id, components, composition group, orientations, sites); `scan`: the size ladder (each size's matrix, atoms, shortest image distance and tag) and each candidate's MACE relaxation energy and edge displacement in every size, and where it converged; `choose`: the screening cell (matrix, atoms, shortest image distance, its place in the ladder), what set it, the Γ-only floor, the band edges' masses and whether they fold onto Γ; `electronic`: the electronic cell, what set it, and how many of each band edge's valleys each size folds onto Γ | C to F |
| C | `prerelax`: each candidate's starting structure and where its defect is; `screen`: every run (energy, final spin, gap levels, structure) and the host's band edges at Γ; `filter`: the survivors (with how far each is above the lowest defect with its dopant atoms, its least formation energy and its charge states' formation energies at E_F = 0); `distort`: C5's jobs (each charge state's start and C4's lowest MACE minima, with the relaxations each minimum gathers; without MACE, the distortions) and the cell they are in (the electronic cell: its matrix and tag); `drelax`: their runs; `rank`: the ranked structures (with their cell); `excite`: each ground state relaxed in the electronic cell, its spin, defect levels and excitations (ZPL, E_abs, ΔQ, the 1D S and DW, the localisation, the verdict), the structures for phase D and the band states left pending; `validate`: C8's test (each size's E, stress and pressure, E_inf and b, each estimate's errors and verdict, Γ against k-points) | D, E (not `validate`: only you) |
| D | `relaxations`: every run's energy, final spin, status and relaxed structure, the electronic cell and its pristine energy and band edges; `zpl`: each excitation's sampling (k-points or Γ, and why), ZPL, E_abs, ΔQ, the 1D S and DW and its verdict | E, F |
| E | `phases`: the competing phases' energies at the production settings; `limits`: μ at each limit; `corrections`: E_corr per defect and charge, in the electronic cell; `formation`: formation energies at each limit, transition levels, the dopant sets and each one's Fermi level; `interest`: the excitations for phase F (`for_F`), and the band states C7 left pending | F |
| F | `ccd`: each excitation's 1D diagram (both states' ħω and S, DW, ZPL, E_abs, E_em, the files); `lineshape`: F2's multi-mode results (S, ħω_eff, DW, ΔQ, the model's test error, imaginary modes, the files) | only you |

Job names say what they are: `d001_qm1_s2` is candidate d001 in charge state −1 starting with spin
2, a triplet (C2), `d001_qm1_n432` its screened structure carried into the electronic cell n432 and
relaxed there (C5; `host_n432` the pristine cell's single point beside it), `d001_qm1_bm0.40_n432`
the same with the bonds around the defect 40 % shorter, relaxed with MACE (C4; with `mace : false`,
in CASTEP in C5), `d001_qm1_m1_n432` the lowest distinct minimum of all of d001's MACE relaxations,
relaxed in CASTEP in q = −1 (C5; with `distort_cell : screening`, `d001_qm1_m1` in the screening
cell), `..._to_q0` a structure found in another charge state tried in q = 0 (C6), `d001_qm1_k0` C6's
lowest structure relaxed again in the electronic cell (C7), `d001_qm1_gs` and `d001_qm1_ex2` its
ground state with the bands' weights and its excitation in spin channel 2 (C7, and D2, where `_gs_G`
is the Γ-only fallback), `d001_qm1` its production relaxation (D1), `d001_qm1_n216` a relaxation in
size n216 (C8), `d001_qm1_ex2_p03_g` and `_p03_e` F1's ground-state and ΔSCF single points at the
4th point of its path, and `d001_qm1_d001` a displaced single point of F2. A size's tag is `n` and
the atoms of its pristine supercell.

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
| A3.1 | `ref_phases` | the reference energies at the screening settings: single points of the host and of the pure elements of the host and each dopant (the smallest polymorph within `ref_polymorph_window`, at `ref_kspacing_factor` × A1's spacing; `ref_relax` to relax them; the compounds run only in E1.1) | a phase doesn't finish |
| A3.2 | `ref_limits` | the chemical-potential limits at the screening settings, against the elements alone, each dopant at its element | a pure element of the host is missing |

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
over that grid. E3 uses its `.bands` file again for the free carriers.

**A2.3 dielectric.** The charge corrections need the static dielectric constant (electronic +
ionic). With `dielectric` set, nothing runs. Otherwise one DFPT job on the relaxed host
(`dielectric_task`, default `phonon+efield`, at q = Γ); ε₀ is the average of the diagonal of
CASTEP's DC permittivity (a warning if it varies by more than 10 %). CASTEP's DFPT needs
norm-conserving pseudopotentials, fixed occupancies and no spin polarisation: give
`dielectric_cell` (a `.cell` with an NCP `species_pot`) and `dielectric_param` (a `.param` with a
cut-off suited to it), or `dielectric`. Without them the production settings are used with
`fix_occupancy : true` and `spin_polarized : false`, and the step warns if the potentials don't
look norm-conserving.

**The competing phases** are run in two places, each taking only the part of them it needs: A3.1, at
the screening settings, the pure elements of the host and the dopants (the defects' constituents),
for C3; E1.1, at the production settings, the compounds too, the host's (CaO₂ for CaO) and those of
the dopants that reach phase D, for E3's formation energies. With doped, a screen of CaO with 33
dopants had 287 competing phases, most of them compounds of dopants whose defects never reach phase
D. They come from `%block phases` (your structure files, one per line, with options such as
`molecule`, `metal`, `kspacing=`); `phases_from : doped` (the Materials Project, through doped); or
neither (no limits: C3 then compares defects only within their composition groups, and E3 leaves out
the μ terms). `%block chemical_potentials` gives E1.2 its limits itself (E1.1 runs nothing), on the
production settings' energy scale, which the screening runs don't share: A3 still runs the phases of
`%block phases` or doped, if given. In E1.1 solids relax lattice and positions (in A3.1 they are
single points, unless `ref_relax` says otherwise); molecules relax in their fixed box at Γ.
doped's phases are run as the Materials Project's structures are best run: each solid as its
primitive cell, made exactly symmetric (pymatgen's `SpacegroupAnalyzer`), so `symmetry_generate`
finds every operation; and without the moments the Materials Project's own runs ended with below
0.1 μB, which are noise (−0.001 μB on one atom of non-magnetic Na made CASTEP take that atom for
another species and keep 6 of its 24 symmetry operations). A phase with no moment left runs
without spin polarisation (`spin_polarized : false`), half the cost, which a non-magnetic phase
doesn't need; O₂ keeps its triplet. Your own structure files (`%block phases`) are run as they are. doped puts its molecules (H₂, N₂, O₂, F₂, Cl₂) in 30 Å boxes, and `dwf.py` moves
each into a box of `phases_molecule_box` (15 Å) instead: in a plane-wave code the vacuum costs as
much as the atoms, and a 30 Å box has eight times the volume, so about eight times the plane waves,
memory and time. The molecules of `%block phases` keep their files' boxes.

With `phases_from : doped`, each dopant gets its phases on its own: doped is given one dopant at a
time, so each works on the hull of the host's elements and that dopant (Ca-O-Sb, then Ca-O-Bi,
...), as an Sb compound has nothing to do with doping with Bi. (Given all the dopants at once,
doped ends by building one phase diagram of every element together, which with many dopants never
finishes: see the testing log.) `dwf.py` also makes doped's Materials Project queries itself, as
the Materials Project's tips for large downloads ask: every chemical system the dopants need in
one request, then the summaries (band gaps) of their materials a few hundred at a time; only the
GGA/GGA+U hull (the entries doped can use), only the materials up to `phases_e_above_hull` + 0.1
eV/atom above it (doped picks within `phases_e_above_hull`, and those further up don't move the
hull), and only the fields doped reads. What comes is kept in `A_host/materials_project/`, so a
dopant added later costs one request for its own systems; the phases doped chooses are kept in
`A_host/phases_doped.json`, with their structures, and a later run (the same host, `dopants` and
`phases_e_above_hull`) takes them from there and asks nothing. Delete `phases_doped.json` to have
doped choose again (from the kept replies), and the folder too to ask the Materials Project again.

**A3.1 ref_phases.** The energies C3 measures formation energies with, at the screening settings so
that they match C2's runs (`screen_cutoff` and `%block screen_overrides`, but not
`screen_max_steps`, which is for the defects; `A_host/jobs/ref_phases/`): a single point of the host
on A2.1's relaxed cell, the cell the defect supercells are built from, and the defects'
constituents, the pure elements of the host and of each dopant. A compound, even the host's own
(CaO₂ for CaO), is run only in E1.1, at the production settings. Of these only the host and its
own elements decide anything (C3's cut across a dopant's sites, where the dopant's chemical
potential cancels); the dopants' elements give the least formation energy C3's table shows. So
they are needed to tens of meV/atom, not to A1's 1 meV/atom, and A3.1 keeps them cheap:

- **One polymorph of each element, a small one.** Of doped's polymorphs within
  `ref_polymorph_window` (0.02 eV/atom) of the lowest above the hull, the one with the fewest
  atoms. Polymorphs can lie within meV/atom of each other, where which comes lowest is noise:
  doped's lowest Na was α-Mn's structure, 29 atoms in its primitive cell, against hcp's 2. The
  table says which were taken over a lower one.
- **Single points** (`ref_relax : none`, the default) of the structures as given: doped's are the
  Materials Project's, already relaxed with PBE. Relaxing the cell at a low `screen_cutoff` can run
  away: the Pulay stress of an unconverged basis is not the derivative of the energy being
  minimised, and a soft metal's cell follows it (Na's grew by two thirds while its energy rose
  0.18 eV/atom, a 2-hour job at a time). `ref_relax : positions` relaxes the positions in the given
  cell, `cell` lattice and positions, with `phases_max_steps` geometry steps each.
- **A coarser k-point spacing**: A1's (or a phase's own `kspacing=`) times `ref_kspacing_factor`;
  2 gives half the k-points in each direction, an eighth in all. The host's single point too;
  molecules stay at Γ. The table gives each phase's grid. The factor scales A1's spacing, which
  was converged for the host: if an insulating host converged on a coarse grid, check the metals'
  grids in the table.

At a low `screen_cutoff`, a hard ultrasoft potential (O, or Be with its 1s electrons in valence)
may need a finer density grid than CASTEP's default: its energy then jumps whenever the grid
changes as a cell relaxes, and the relaxation keeps "reverting to earlier configuration" until it
runs out of steps. `fine_gmax : 35 1/ang` in `%block screen_overrides` gives the screening runs and
these phases a finer grid alike (it was needed for relaxed cells; for single points it only costs
time). Without `%block phases` or `phases_from : doped` nothing runs, and C3 makes no cuts across
composition groups.

**A3.2 ref_limits.** A pymatgen phase diagram of A3.1's host and its pure elements (it needs one of
each). Each corner of the host's stability region is a limit, named after the phases there
(`CaO-Ca`, `CaO-O2`). Against the elements alone the region is the widest it can be (a compound of
the host's elements that is more stable, CaO₂ next to O₂, would narrow it), so C3's cut across a
dopant's sites with these limits can only keep too much, never too little. Each dopant is at its
pure element, the highest its chemical potential can be: C3's table gives each defect's least
formation energy with it (a bound from below), by which a dopant's defects can be compared, those
with other numbers of its atoms too; its cut doesn't need it, as the dopant's μ cancels between
defects with the same dopant atoms. A dopant without its element among the phases gets none, and C3
gives its defects no least formation energy. `summary.txt` shows the limits as Δμ against the pure
elements; `results.json` has the absolute values (CASTEP's energy scale at the screening settings),
which C3 uses. E1.2 works out the limits again from E1.1's production energies, with the compounds.

### Phase B: candidate defects, the screening and electronic cells

| Label | Step | What it does | Decision |
|---|---|---|---|
| B1 | `generate` | the candidate defects | `%block defects` given (flowchart "yes"), or generate them ("no") |
| B2 | `scan` | the size ladder (a supercell for each of `scan_lengths`); with MACE, every candidate relaxed in every size | where each candidate's MACE relaxation has converged (`scan_tol`, `scan_edge_tol`) |
| B3 | `choose` | the screening cell for phase C, after `configuration_margin`'s cut | the smallest size that holds every candidate the cut keeps and is at least the Γ-only floor, or `supercell` |
| B4 | `electronic` | the electronic cell, where the survivors go from C4 on | the smallest size from the screening cell up that folds every band-edge valley onto Γ and meets the floor for a level `electronic_depth` deep, or `electronic_cell` |

**B1 generate.** Your defects, one per line of `%block defects`, components joined by `+`:
`Sb_Ca`, `Sb_Ca + Vac_O` (the partner on its nearest O site), `Sb_Ca + Vac_O  all` (every
arrangement up to `pair_max_distance`), `Vac_O:2` (the 2nd O site of the primitive cell),
`Sb_Ca@2` (a starting spin of 2 on the Sb), and `q=` for the charge states of that line's
defects (`Sb_Ca + Vac_O  all  q=-1`, `Vac_O  q=+2,+1`): those are screened instead of
`screen_charges`. Without the block, on each symmetry-inequivalent site
of the primitive cell: a vacancy (`vacancies`), a substitution by each dopant, antisites
(`antisites`), Voronoi interstitials (`interstitials`), and pairs: dopant + native (`pairs`),
native + native (`native_pairs`), dopant + dopant (`dopant_pairs`: any two, or with `same` only
two of one dopant, such as Sb_Ca + Sb_O, so that each dopant meets the host on its own), never two
interstitials. Each pair type gives every symmetry-distinct arrangement up to `pair_max_distance`
apart. Candidates
get ids `d001`, `d002`, ... and a composition group, e.g. `+Sb-Ca-O` for Sb_Ca + Vac_O: those with
the same group (Sb_Ca + Vac_O and Sb_O + Vac_Ca) are compared directly in C3. The number of
candidates grows quickly with each option and with `pair_max_distance`: check the list
(`python3 dwf.py NAME B1`) before screening.

**Where a defect goes in the supercell.** Every defect cell (B2, C1, C7, C8) is built the
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
B3 picks the screening cell from the ladder and B4 the electronic cell, so the largest length is
the biggest cell B3 will use (B4 goes beyond it if it must).

**B2 scan** (`mace`, `scan`): every candidate, built in every size it fits (a pair needs its
components less than half the image distance apart), relaxed with MACE (positions only, in the
host's lattice, as in C and D; `mace_model`), on compute nodes with `mace_jobscript`. Each starts
from the ideal structure with the atoms within 4 Å of the defect's components moved at random
(`mace_rattle`, 0.02 Å by default; the same displacements every time): on a symmetric site (an
interstitial in a void, say) the forces on a defect cancel exactly, so a relaxation from the ideal
structure could never leave it, even where breaking the symmetry goes lower. Two numbers
per relaxation: its relaxation energy R (ideal − relaxed, the rattle left out) and the largest
displacement of the
atoms at the cell's edge (further than 35 % of the image distance from the defect). With
`scan_test : line` (the default) a candidate is converged at the smallest size from which R lies on
a line R∞ − c/N, N being the cell's number of atoms: that size and every bigger one, at least three
of them, all within `scan_tol` of the line fitted through them. A defect that strains the crystal
leaves the cell, whose lattice vectors are fixed, holding a stress, and the energy of that stress,
c/N, falls off only as 1/N. Once R follows the line, the cell holds the defect's relaxation and that
tail is all that still changes with size. The table gives R∞, c and c/N in the converged size: the
stress energy still in that cell, which cancels between arrangements of one composition but not
between charge states or against isolated components. With `scan_test : flat` a candidate is
converged at the smallest size whose R is within `scan_tol` of that of every bigger size, with its
edge displacement below `scan_edge_tol`. A defect with a big stress tail may never converge that way
(Sb<sub>Ca</sub> + V<sub>O</sub> in CaO gained about 57 eV / N, still 0.13 eV short at 432 atoms),
while on the line it converged at 144 atoms.

A candidate needs enough sizes for the test to pass at all: with `line`, a size and three bigger
ones. A pair fits only the sizes whose image distance is more than twice its separation (plus
0.05 Å), so `pair_max_distance` and `scan_lengths` have to agree: before the relaxations B2 warns
when some pairs fit too few sizes, and says how far apart a pair the ladder can test (half the
image distance of its fourth-largest size, less 0.05 Å; 6.99 Å for 8 10 12 14 16 18 20 Å).
`summary.txt` marks those "too few sizes", apart from the ones that were tested and failed.

A candidate that fails the test although it has the sizes has usually reached different minima in
different sizes, not converged slowly with size: its R jumps between neighbouring sizes (by more
than 0.3 eV for most of the failures of a CaO run, before the rattle), as each size is relaxed once,
from (nearly) the ideal structure, and a cell's shape and symmetry tip the relaxation into one basin
or another. So B2
relaxes it again in each of its other sizes from the deepest minimum it reached (the relaxed
structure of the size whose R is furthest above the trend with size, the line through all its
sizes, carried into each other cell around the defect, not rattled again), keeps in each size
whichever relaxation
went deeper (by more than 5 meV; `^` in `summary.txt`'s table) and tests it again; the log says how
many that converged. This is one more round of MACE jobs, for the failures only. It only puts every
size in the same minimum: a candidate in the same shallow minimum in every size passes as it is,
and finding lower minima is C4's job. One still not converged is dropped by B3
(`scan_unconverged : drop`, the default), or counted as needing the largest size (`largest`). MACE
is cheap next to CASTEP, so every candidate gets its own answer instead of one defect's standing in
for the rest. `summary.txt` has a table of every
candidate's relaxation energy in every size. With `mace : false` or `scan : false` only the ladder
is made, and B3 goes by the floor (or `supercell`).

**B3 choose: the screening cell.** Every candidate is built in this one cell in phase C, so the
energies of arrangements of one composition can be compared directly. First, with
`configuration_margin`, B3 cuts the arrangements whose MACE energy is more than the margin above
the lowest of their composition, each composition's arrangements compared in the biggest size B2
relaxed them all in (its energies the nearest to an isolated defect's); C1 leaves them out. The
cell is sized for the rest, so an arrangement that strains the crystal but is cut can't make
every C2 run pay for a bigger cell: `summary.txt` lists the cut and, when it made the cell
smaller, the cell it would have been. The screening cell is the smallest size of the ladder that
is at least as big as

- every surviving candidate's converged size (B2): the worst case sets it, and `summary.txt`
  names the candidate that did (one B2 didn't converge: below), and
- the **Γ-only floor**: the image distance a cell needs for Γ-point screening to see a defect's
  levels as an isolated defect has them. A level E_b inside the gap decays away from the defect as
  exp(−r/ξ), with 1/ξ² = (2m*/ħ²) E_b (1 − E_b/E_g) in a two-band (Kane) model, and neighbouring
  images' levels overlap and spread into a band unless they are several ξ apart. The floor is
  `decay_length_multiple` times ξ (4ξ by default) for the shallowest level that counts as deep
  (E_b = `deep_level_margin`, E4's), with the lighter of the hole and electron effective masses,
  which B3 fits to A2.2's `.bands` around the VBM and CBM (each k-point next to the edge gives a
  mass, the lightest is taken; the irreducible k-points are unfolded with the host's symmetry), and
  A2.2's gap. For CaO-like masses (0.4 to 1 m_e), E_b = 0.5 eV and a PBE gap of 3.6 eV, 4ξ is 12 to
  19 Å. At Γ the overlap makes an s-like level that deep look deeper than it is, by about 10 % at
  4ξ, 19 % at 3.5ξ and 33 % at 3ξ (no core, 12 nearest images; more for a level with a broad core,
  less for a compact one), and deeper levels by much less, as they decay faster. A cell's atoms go
  as the cube of the multiple, so 3 needs under half the atoms of 4. `screen_min_image` gives the
  floor yourself (0: none).

A survivor B2 didn't converge (with too few sizes, or still not after the re-seeding) is dropped
(`scan_unconverged : drop`, the default): C1 leaves it out, as it does a cut one, and it doesn't
size the cell; `summary.txt` lists them. So the sizes above the fourth-largest are only there for
the test, and the screening cell is at most the fourth-largest size (with `scan_test : line`;
or the floor). With
`scan_unconverged : largest` such a candidate counts as needing the largest size instead, and one
is enough to put every C2 run in the largest cell. With `supercell` nothing is dropped.

A floor beyond the ladder gets a cell of its own. `supercell` fixes the screening cell instead (a
length, `supercell : 14 ang`, or n × transform, `supercell : 3`). `summary.txt` also lists the
pairs whose components are more than half the chosen cell's image distance apart, which C1
leaves out: a smaller cell leaves out more of the far pairs (`pair_max_distance`). B3 also checks whether A2.2's
band edges fold onto Γ of the screening cell: a k-point k folds if k Pᵀ is a whole-number vector
(P the supercell matrix). One that doesn't (e.g. an edge at X, and n × n × n copies of the FCC
primitive cell with n odd) means the Γ-point screening sees a wider gap, and its levels near that
edge are measured against an edge that is too far out: B3 warns, and C2 measures it. That is
acceptable for screening, which compares defects in one cell, but not for judging a defect's
electronic structure, which is what B4's cell is for.

**B4 electronic: the electronic cell.** The survivors of C3 go into this cell for good: C4–C5's
distortions (`distort_cell`), C6's sharing, C7's excited states, D's production runs, and so E's
formation energies and F's lineshapes. C1–C3, which screen every candidate, stay in the screening
cell, the smallest that works for them. The screening cell is sized for the relaxations (B2) and for
levels at least `deep_level_margin` inside the gap; at the semilocal level many levels sit closer to
a band edge than that, and a Γ-point calculation only sees the host states whose k-points fold onto
its Γ. A band edge away from Γ usually comes as a *star* of symmetry-equivalent k-points, its
*valleys* (the X point of rock salt has three: X<sub>x</sub>, X<sub>y</sub>, X<sub>z</sub>), and a
shallow state near that edge is made of all of them. A cell that folds only some of them (CaO's
144-atom FCC-shaped cell folds one X of three) gives such a state the wrong make-up, and its Γ point
sees too wide a gap. So the electronic cell is the smallest size of the ladder, from the screening
cell up, that

- folds every valley of both band edges onto Γ (the table gives each size's folding, e.g.
  `cbm 1/3`), and
- meets the Γ-only floor (B3's estimate, from the band masses: `decay_length_multiple` times
  the decay length) for a level `electronic_depth` (0.25 eV) from a band edge.

If no size of the ladder does, B4 tries the most compact cells for longer lengths, 0.5 Å at a time.
For CaO, whose conduction-band minimum is at X in PBE, the FCC-shaped cells of 320 and 432 atoms
fold all three X points (and the 240- and 384-atom ones don't); with a cubic `transform` every size
folds them. With PBE's masses and gap, CaO's floor for a level 0.25 eV deep is 23.0 Å at the
default 4ξ, beyond the ladder, and B4 goes to a 728-atom cell; `decay_length_multiple : 3.5` gives
the 432-atom cell and `3` the 320-atom one. `electronic_cell` fixes the cell instead (a length, n ×
`transform`, or `screening` for the screening cell): B4 warns if it doesn't fold every valley. In a
cell that folds the band edges, a localised defect level is flat across the k-points, which D2's
band-number check relies on.

### Phase C: screening, ground and excited states

Every candidate goes through cheap CASTEP relaxations at the Γ point; the ones that are competitive
and have the properties you ask for are distorted (ShakeNBreak), the distortions relaxed with MACE,
and the lowest structures that search finds relaxed again in CASTEP, in every charge state; the
lowest structures of each charge state, with winning structures tried in the others, are their
ground states. The triplet ground states are then excited (ΔSCF): only the defects with an
excitation that passes, and every native defect, go on to the production runs of phase D. C1–C3 run
in the screening cell (B3), which keeps the cost of screening every candidate down; C3 also compares
the formation energies of defects of one dopant across sites, with A3's references. The survivors
then move into the electronic cell (B4) for C4 to C7, and stay there. C8, optional, tests the
finite-size error of the screening cell that C3 relied on.

| Label | Step | What it does | Decision |
|---|---|---|---|
| C1 | `prerelax` | builds every candidate in the screening cell and relaxes it with MACE (B2's relaxation there, if it ran) | pairs too far apart for the cell are left out, and the arrangements B3's `configuration_margin` cut (C1 makes the cut itself only without B2's scan) |
| C2 | `screen` | one cheap Γ-point CASTEP relaxation per candidate and charge state, from a triplet (a doublet for an odd electron count) | — (warns if the band edges don't fold onto Γ) |
| C3 | `filter` | defect hull, the formation energies across a dopant's sites, duplicates, property filters, and the charge states each survivor carries on, with a Γ single point of each band state it judges | within `hull_margin` of the hull; within `site_margin` of the lowest defect with its dopant atoms somewhere; the `require_*` filters; which charge states (`distort_charges`, `charge_margin`, `carrier_reach`, `carrier_fraction`) |
| C4 | `distort` | each survivor's screened structures carried into the electronic cell, and ShakeNBreak's distortions of every charge state there, all relaxed with MACE and pooled by defect | by how much (`distortions`), around which points (`distort_around`), in which cell (`distort_cell`); the `distort_starts` lowest distinct MACE minima go on |
| C5 | `drelax` | Γ-point CASTEP relaxations, in each charge state, of its carried-over structure and of C4's minima, against the electronic cell's band edges, every geometry step kept; `summary.txt` ranks them | — (it says where a minimum beat the start by `distortion_min_gain`) |
| C6 | `rank` | the ground states: C5's relaxations ranked, a winning structure tried in the other charge states | `keep_structures` per charge state |
| C7 | `excite` | in the electronic cell: the ground states relaxed, the triplet filter, the localised levels, ΔSCF | a triplet whose localised excitation has a ZPL of at least `excite_min` and stays on the defect; native defects go on anyway |
| C8 | `validate` | optional (`validate : true`): a finite-size test of the screening cell, and Γ against k-points | an estimate passes for a defect when every error is within `validate_tol`; nothing later uses it |

The CASTEP jobs of C2, C3 and C5 run with the runner settings of `%block runner_screen` (and of
`runner_filter` for C3's single points alone, `runner_drelax` for C5's jobs alone, which, in the
electronic cell, are the biggest of phase C), as do C6's and C7's: no restarts, and a job that fails
is flagged while the others carry on (`--resubmit-failed` tries it again). C4's MACE relaxations run
as B2's and C1's do, with `%block runner_distort` for settings of their own.

**C1 prerelax.** Each candidate is built in the screening cell; a pair whose components are more
than half the shortest image distance apart is left out. With `mace : true` it is relaxed with
MACE (positions only, `mace_model`, rattled as in B2): when the screening cell is one of B2's sizes,
that is B2's
relaxation, read from its file in `B_candidates/mace/`; otherwise C1 runs it (on compute nodes
with `mace_jobscript`). A candidate B2 re-seeded is also relaxed from the deepest minimum B2 found
for it, carried into the screening cell (B2's own relaxation there when the cell is one of its
sizes), and keeps the deeper of the two (`reseeded` in `results.json`). The arrangements B3's
`configuration_margin` cut, and the candidates it dropped as not converged, are left out, unbuilt.
Without B2's scan (`scan : false`) B3 has no energies to cut with, and C1 makes the cut itself:
arrangements whose configurational energy (MACE energy above the lowest arrangement of the same
composition) in the screening cell is more than the margin are dropped before any CASTEP run
(MACE knows nothing of charge or spin, so keep the margin generous).

**C2 screen.** For each candidate, charge state (its own `q=` from `%block defects`, else
`screen_charges`) and starting spin, one CASTEP relaxation at Γ, without symmetry, at
`screen_cutoff` (default the production cut-off), with at most `screen_max_steps` geometry steps and
any `%block screen_overrides`. The starting spin (`screen_spins`): by default a triplet (`spin : 2`,
two unpaired electrons, the ground state the screening looks for), or a doublet (`spin : 1`) when
the electron count is odd and a triplet can't be had; `low` starts from the lowest spin the electron
count allows (0 or 1), and `both` runs that and that + 2. It is set as `spin` and as `SPIN=` on the
anchor atom, as a starting value only: CASTEP may end elsewhere, and the results keep the final
spin. The pristine supercell gets a single point at the same settings: its band edges at Γ are the
reference for the gap levels. Each run records its energy (only if the last SCF cycle converged),
final spin, the defect levels in the gap (more than `level_margin` inside the band edges), and
the electrons it left in the conduction band and holes in the valence band: those the states
more than `level_margin` inside the gap can't hold, counted over both spin channels from its
`.bands`. A run with at least half an electron or hole there is a *band state*: its extra
electron (or hole) went into a band instead of a defect level, so its energy is that of the
defect with one charge less plus a free carrier, and C3 doesn't judge the candidate on it. This
happens when the run settles in the wrong structure (Sb<sub>Ca</sub> + V<sub>O</sub> in its −1
state, relaxed from the MACE structure with Sb on its site, puts its fourth electron in the Ca
3d<sub>xy</sub> conduction band; ADAQ's −1, with Sb moved toward the vacancy, binds it), or when
the defect's level lies above the PBE conduction band edge, where only a wider-gap functional
could bind it. C2 compares the pristine cell's gap at Γ with A2.2's: more than 0.05 eV wider
means band edges that don't fold onto Γ of this cell (B3 says which), and C2 warns. The screening
goes on with these Γ-point results, as ADAQ's does: there is no check of the levels' spread on a
k-point grid (the old C2b).

**C3 filter.**

1. *Best state:* for each candidate and charge state, its screening run (the lowest energy over
   its starting spins, with `screen_spins : both`). If that run is a band state (C2), the charge
   state is *in a band*: it takes no part in the hull or the filters, which use the candidate's
   *bound* charge states, and point 6 decides whether it goes on to the distortions.
2. *Defect hull:* within a composition group, the formation energy of every arrangement and
   bound charge state across the gap, E + E_corr + q (E_VBM + E_F), without the host energy and
   chemical potentials (the same for all of them). E_corr is the charge correction
   (`charge_correction`, with ε₀ from A2.3) in the screening cell: without it, charge states
   couldn't be compared. An arrangement is kept if its lowest comes within `hull_margin` of the
   group's lowest somewhere in the gap; one with no bound charge state can't be judged by the
   hull (point 6 decides whether it stays). Its *stable* charge states are its bound ones that are
   its lowest somewhere in the gap.
   The gap is `gap` if you give it (an experimental or hybrid value), else the screening cell's
   own. A semilocal gap is too narrow: CaO's PBE gap at Γ is 3.64 eV against a measured 7.1 eV,
   and a bound charge state that only becomes the lowest above the PBE conduction band edge would
   not count as stable without `gap`. The formation energies are the PBE ones carried on past
   that edge, as the straight lines they are in the Fermi level; a band state's line is the band
   edge carried on, which is why it can't count. The gap levels (`require_gap_states`) are still
   those inside the screening cell's band edges.
3. *Across a dopant's sites* (with A3.2's limits, at the screening settings): E_f = E + E_corr −
   E_host − Σᵢ nᵢ μᵢ + q (E_VBM + E_F), with E_host C2's pristine screening cell, over the host's
   stability region (every mix of A3.2's corners) and the gap of point 2. A3.2's region is against
   the host's elements alone, the widest it can be, so this cut can only keep too much. Defects with
   the same dopant atoms, whatever host atoms they add or remove (Sb<sub>Ca</sub>, Sb<sub>O</sub>,
   Sb<sub>i</sub> and Sb<sub>Ca</sub> + V<sub>O</sub> all have one Sb), compete for the same atoms,
   and the dopant's μ cancels between them, so only the host elements' μ matter: a defect is dropped
   if it stays more than `site_margin` (0.5 eV) above the lowest of them everywhere in that region
   and gap, so each such set keeps at least its lowest. The formation energies are straight lines in
   the mix of corners and in E_F, so linear programming finds where it comes closest, exactly (the
   table's `above site` column). Native defects aren't compared so (they compete for no dopant), and
   a defect alone in its set passes. The table also gives each defect's least formation energy, at
   the best corner and Fermi level, with the dopants at their pure elements (the `least E_f`
   column): the highest their μ can be, so a bound from below, by which a dopant's defects can be
   compared, those with other numbers of its atoms too. Nothing is dropped on it, and a defect of a
   dopant without its element among A3's phases gets none. Without A3's phases no cut is made here.
   The screening settings' energies (a lower cut-off, Γ only, few geometry steps) can be some tenths
   of an eV off: keep `site_margin` loose enough for that.
4. *Duplicates:* an arrangement that relaxed into the same structure as a lower one of its group
   (lower in charge-corrected formation energy) is dropped.
5. *Property filters,* each met by a stable charge state: `require_gap_states` (occupied, empty
   or both kinds of gap level), `require_spin` (the final spin), `require_bound` (pairs bound
   against their isolated components somewhere in the gap). A candidate that fails them but has
   a band state that point 6 carries on is kept, *pending*: bound, that state might have the
   property (Sb<sub>Ca</sub> + V<sub>O</sub>'s triplet is its −1). In a host where PBE leaves many
   charge states in the bands, these filters drop fewer candidates. With `excite : true` (the
   default), native defects skip these filters (but not the hull or the duplicates), so that
   they reach the production runs (D) for E3's Fermi level.
6. *The charge states carried on* (`distort_charges : relevant`, the default):
   - the survivor's stable charge states;
   - its other bound ones whose formation energy comes within `charge_margin` (0.5 eV) of its
     lowest somewhere in the gap: a distortion can lower a charge state by about that much
     (C5's gain; CaO:Sb's neutral state gained 0.34 eV) and so make it stable;
   - a band state one charge beyond a stable one at a band edge (one more electron than the
     state stable at the top of the gap, one more hole than the one stable at the bottom; every
     band state of a candidate with none bound), if its carrier is shallow and on the defect.
     A distortion can bind such a carrier, as one binds Sb<sub>Ca</sub> + V<sub>O</sub>'s fourth
     electron in its −1 state; one deep in a band or spread through the cell is a free carrier,
     which no distortion of the defect is likely to bind. This is the *band-state test*: a Γ
     single point of the screened structure at the screening settings, with the band weights
     written (`pdos_calculate_weights`), in `C_screen/jobs/filter`. Its carriers are the bands
     C2 counts them in (from `level_margin` inside a band edge onwards); the state goes on if
     the deepest lies at most `carrier_reach` (1 eV) inside its band and none is spread over
     more than `carrier_fraction` (0.2) of the cell's atoms (the participation number, as C7
     counts it). A single point that fails, or whose carriers settle in gap levels, carries the
     state on unjudged.

   Only a band state carried on keeps a candidate that fails the property filters (pending) or
   has no bound charge state; a candidate with no bound state and none carried on is dropped. A
   `gap` wider than the semilocal one (point 2) makes more of the bound charge states stable, so
   they go on by their energies. The earlier rules make no single points: `stable` carries on the
   stable charge states and every band state, `screened` every charge state with a usable
   energy; under both, any band state keeps a candidate pending, and one with no bound charge
   state is kept.

`summary.txt` gives every candidate's verdict and why, how far it is above the lowest defect with
its dopant atoms and its least formation energy, the charge states it carries on, its charge
states in a band (with the electrons, e, or holes, h, left there), each band state's test (how far
inside the band its carriers lie, the atoms they are spread over against the limit, and whether
it goes on), and the correction for each |q| in the screening cell.

C3 is the only screening step that compares different charge states, so it is the only one that
needs the correction: C2 (the starting spins), C5 and C6 (ranking and sharing) and C7 (excitations)
compare structures of one charge state in one cell, where the correction is the same for all of
them, and C4's MACE knows no charge at all.

**C4 distort: ShakeNBreak's distortions, searched with MACE.** For each survivor and charge state,
its screened structure is carried into the electronic cell (B4; `distort_cell : electronic`, the
default): the defect is rebuilt there, the atoms near it are moved as the same atoms are in the
screened structure, and the rest stay on their perfect-crystal sites (`d001_qm1_n432`, the *start*).
There, ShakeNBreak moves the nearest neighbours of a point in or out by each factor in `distortions`
(by default ShakeNBreak's own, −0.6 to 0.6 in steps of 0.1), and rattles the other atoms
(`rattle_stdev`, `rattle_dmin`). How many neighbours is ShakeNBreak's count for that charge state,
from the oxidation states in `%block oxidation_states` or pymatgen's guess: the electrons a charge
state adds or takes away change it, so each charge state has its own (Sb<sub>Ca</sub> +
V<sub>O</sub>: 4 in −1, 3 in 0, 2 in +1), and one with none to move is only rattled. The point is
set by `distort_around`. With `auto`, the default:

- a single defect is distorted around its own site;
- a complex whose components are all nearest neighbours (within 1.25 host bond lengths of each
  other) is distorted around its centre. For Sb<sub>Ca</sub> + V<sub>O</sub> that is the
  midpoint of the Ca and O sites: Sb is its nearest atom, so a compression moves it toward the
  vacancy (by 0.36 Å at −0.3), where ADAQ's −1 sits, and a stretch moves it away;
- a complex more spread out is distorted around each component in turn (the dopant, then the
  vacant site), since its centre is surrounded by atoms of neither: twice the structures, named
  `..._c1_...` and `..._c2_...`.

`centre` and `each` use one of these for every complex, `anchor` the first component's atom (or its
vacant site) and `vacancy` the vacant site (or the anchor atom, for a defect without one).

With `mace : true` (the default), all of these structures, of every charge state the defect carries,
and the starts too, are relaxed with MACE (positions only, the cell fixed; `mace_model`,
`mace_fmax`, `mace_steps`; on compute nodes with `mace_jobscript` and `%block runner_distort`), each
in its own file in `C_screen/mace/`. MACE knows nothing of charge or spin, so all of a defect's
relaxations lie on one energy surface, whichever charge state they came from, and they are *pooled*:
those that relaxed into the same structure (pymatgen's StructureMatcher, compared only within 0.05
eV of each other) are merged, and the `distort_starts` (2) lowest distinct minima are kept. A
structure that one charge state's distortions found is so tried in every charge state, and the other
charge states' neighbour counts widen the search for all of them, at MACE's cost rather than
CASTEP's. What MACE can't see is the charge: a distortion that only a bound carrier drives
(Sb<sub>Ca</sub> + V<sub>O</sub>'s −1) may not stand out in its search, and is left to C5's
relaxation of the start in that charge state, and to C6's sharing. Nor are a foundation model's
minima DFT's: on V<sub>Sb</sub> they lay 0.2–0.4 Å from DFT's, and DFT relaxations from them could
miss the DFT ground state (Wang, Mosquera-Lois and Walsh, PRX Energy 2026). So the MACE minima are
only starts, the carried-over start is always relaxed beside them, and `mace : false` gives back
CASTEP's relaxation of every distortion, to compare on a few defects. C4's table lists each defect's
lowest minima (five at least, then how many more there were): its energy above the lowest, MACE's
energy, how many relaxations ended there and which (charge state and distortion), and whether it was
kept. With `mace : false` there is no search: C5 relaxes every distortion in CASTEP, each in its own
charge state (`d001_qm1_bm0.40_n432`), and `distortions` is `-0.4 -0.2 0.2 0.4` unless you give it.

**C5 drelax.** For each survivor and charge state, CASTEP relaxations at the C2 settings (Γ point),
starting from the screened run's final spin, of its start (the structure every other has to beat)
and of C4's kept minima, each in this charge state (`d001_qm1_m1_n432`, `d001_qm1_m2_n432`): three
relaxations per charge state by default. A single point of the pristine electronic cell
(`host_n432`) is in the same batch: their levels and band carriers are measured against that cell's
band edges. Each run's `.castep` keeps every geometry step, with its energy and forces.

C5's part of `C_screen/summary.txt` ranks them, so that what the search found can be looked at
before C6 tries the winners in the other charge states (stop there with `--until C5`). The first
table has a line per survivor and charge state: the start's energy, the lowest, the gain, which run
gave it (`start`, `MACE 1`, `MACE 2`, ...), how many distinct structures and failed runs there were,
and a verdict: `NEW MINIMUM` when the lowest structure is not the start's and beats it by
`distortion_min_gain` (C6's test), else `the start is the lowest` or `lower by less than ...`. The
second lists every relaxation of each charge state, lowest first: dE against the start, the energy,
final spin, steps, whether it relaxed, and a letter shared by runs that relaxed into the same
structure (A the lowest; compared only within 0.05 eV of each other). Runs without a usable energy
come last, with why and the lowest of their steps whose SCF converged, which shows whether a failed
run was heading below the others. A run with steps whose SCF didn't converge says how many: CASTEP
prints an energy and forces for such a step and carries on.

*Why the cell changes after C3.* C2 runs every candidate in every charge state, so its cell sets
most of the cost of the screening, and the screening cell is the smallest that holds every
candidate's relaxation (B2) and deep levels (the Γ-only floor). That is enough to rank arrangements
(the hull) and find each one's charge states, but not to judge a survivor's electronic structure,
and the distortions are the start of that. In the screening cell a charge state whose level lies
near a band edge is often left in a band (its extra electron or hole spread through the band, so
nothing near the defect pulls on the atoms, and its distortions relax back), and the band-edge
valleys may not all fold onto its Γ (B3 says). The electronic cell folds them all and holds a level
`electronic_depth` (0.25 eV) from a band edge, so a carrier that can bind does so there, and the
distortion it drives can be found: in CaO:Sb, the −1 state of Sb<sub>Ca</sub> + V<sub>O</sub>, whose
fourth electron the 144-atom cell leaves in the conduction band. So everything after C3 works in the
electronic cell: C4–C5, the ranking and sharing (C6), the excited states (C7), and phases D, E and
F; only C8, which tests the screening cell, goes back to it. The price: C5's relaxations are among
the dearest runs of phase C (in CaO, three times the atoms, so roughly 10 to 30 times a screening
run each), three per charge state a survivor carries on (the start and the two MACE minima; with
`mace : false`, the start and every distortion). With `distort_cell : screening`, C4–C6 stay in the
screening cell (the screened run is then the start, and C7 carries the structures C6 keeps into the
electronic cell); when B4 chose the screening cell itself, the two are the same.

C6's and C7's jobs run at the screening settings, with the runner settings of `%block
runner_screen` (and of `runner_rank` or `runner_excite` for one of them alone); C8's at the
production settings, with those of `%block runner_production` (and `runner_validate`).

**C6 rank.** For each survivor and charge state, C5's relaxations (the carried-over screened
structure and C4's minima, or every distortion with `mace : false`, all in the electronic cell) are
pooled, identical structures merged (pymatgen's StructureMatcher with tight tolerances) and ranked
by energy; a structure that beats the carried-over one by `distortion_min_gain` is also relaxed in
the other charge states of that defect, in the same cell (against C5's band edges,
`C_screen/jobs/share_<n>/`), for up to `share_rounds` rounds or until nothing new turns up; the
lowest `keep_structures` go on, already in the electronic cell. With `distort_cell : screening` the
same happens in the screening cell, with the screened run as the start, and C7 carries the
structures into the electronic cell.

**C7 excite** (`excite : true`, the default). The excited states, in the electronic cell, at the
screening settings (Γ, `screen_cutoff`, `%block screen_overrides`, with `metals_method : dm` and
`%block excite_overrides`), in three batches:

1. *Ground states* (`C_screen/jobs/excite_relax/`): the pristine electronic cell (its band edges),
   and every structure C6 kept, carried into the cell (unless it is there already, as C5's are) and
   relaxed. The lowest that relaxes is the ground state of its charge state (C6's order may change);
   it is flagged `moved` if it slid more than 0.2 Å from where it started. A charge state that
   leaves an electron in the conduction band or a hole in the valence band (a *band state*, as in
   C2) is *pending*: PBE can't place its level, so it is left for a hybrid calculation later, and E4
   lists it. A ground state whose final spin is further than `spin_tol` (0.3) from `excite_spin` (2:
   a triplet, in CASTEP's unpaired electrons; `any` for every spin) is not excited; it still goes on
   to phase D as a ground state if its defect passes, for phase E.
2. *Defect levels* (`C_screen/jobs/excite_gs/`): a single point of each remaining ground state with
   `pdos_calculate_weights : true`, which writes each band's weight on every atom
   (`<seed>.pdos_weights`). A band's participation number P = (Σ w)² / Σ w² (over the atoms)
   counts the atoms it spreads over, and a band with P at most `local_fraction` (0.2) of the
   cell's atoms is localised on the defect. The defect levels are the localised bands in the gap
   or up to `excite_reach` (1 eV) inside a band: PBE's gap is far too narrow (CaO: 3.6 eV,
   against 7.1 eV measured), and pushes many empty defect levels into the conduction band, where
   they are still resonances localised on the defect. Choosing levels by their localisation
   rather than their place in the PBE gap keeps those, where a gap-only rule would throw them
   away (ADAQ, which faces the same problem, keeps excitations whose states are localised too).
   In each spin channel the excitation goes from the highest occupied defect level to the lowest
   empty one at least 0.1 eV higher (spin-conserving, so a triplet stays a triplet). The
   single point's `.check` is kept, even with `delete_check : true`, since the excited state
   starts from it; once you are happy with C7's results you can delete
   `C_screen/jobs/excite_gs/*/*.check` to save space.
3. *Excited states* (`C_screen/jobs/excite/`): CASTEP's simple ΔSCF (`calculate_deltascf :
   true`, `deltascf_method : simple`, the two bands' occupations in `%block
   deltascf_constraints`), from the ground state's `.check` (`reuse`, `deltascf_checkpoint`),
   relaxed with `excite_relax : true` (the default). From it: the ZPL (the relaxed excited state
   less the ground state), the vertical absorption E_abs (the excited state's first SCF energy,
   at the ground state's structure), ΔQ (the mass-weighted distance between the two structures,
   amu^½ Å), and the 1D estimates of how strongly the excitation couples to the lattice: the
   relaxation energy E_rel = E_abs − ZPL, the phonon energy of a parabola with that energy over
   ΔQ, ħω = ħ √(2 E_rel) / ΔQ, the Huang–Rhys factor S = E_rel / ħω and the Debye–Waller factor
   DW = e^−S (the share of the light in the ZPL; a lower bound, since one coordinate gathers the
   displacement of every mode). The promoted electron's band must still be localised in the
   excited state; if it isn't, the electron left the defect (`ionises`).

A charge state passes if one of its excitations does: a ZPL of at least `excite_min` (0.4 eV),
inside `excite_window` if one is given, the electron still on the defect, and DW at least
`excite_min_dw` if that is given. A failed run is `unknown` and counts as a pass, with a warning.
A defect with a passing charge state goes on to phase D with all its charge states (their relaxed
ground states here). Every native defect (one that only adds or removes host elements:
vacancies, antisites, self-interstitials and their complexes) goes on whatever its result,
because E3's Fermi level needs it. With `excite : false`, the lowest structure C6 kept of each
charge state goes on to phase D as it is.

The thresholds are loose on purpose. The energies are PBE's (in SiC and diamond PBE's ΔSCF ZPLs
come out about 0.2 eV low), so C7 only removes what can't be a colour centre (no triplet, no
localised excitation, a ZPL near zero, an electron that leaves the defect) and leaves the
ranking to the calculations after it. Test C7 on one defect before a full run: CASTEP's ΔSCF
must work on your build. Sb_Ca–V_O in q = −1 is a good test: ADAQ found its ZPL at about 0.53 eV
(PBE 0.54, HSE 0.53). A constrained-occupation run can fail to converge (ADAQ had this for the
P and As versions of X_Ca–V_O); try `deltascf_smearing : 0.01` or more `nextra_bands` in `%block
excite_overrides`.

**C8 validate** (`validate : true`; off by default). A test of the method rather than a step the
workflow needs: whether the screening cell, with its corrections, gives the formation energies
and transition levels of much bigger cells. C3's defect hull compares the candidates in the
screening cell, so its errors are what this measures; the production energies (D, E) come from
the electronic cell instead. For each defect in `validate_defects` (B1's ids, e.g. `d001 d002`;
with none, every defect C6 kept), in each charge state:

1. *Runs:* C6's lowest structure is relaxed at the production settings (A1's cut-off and k-point
   spacing, `%block production_overrides`, positions only, `calculate_stress : true`, and no
   `freeze_radius`, so the far field relaxes too) in the screening cell and in the next
   `validate_sizes` (3) sizes of the ladder up, its relaxed displacements carried into each
   (the atoms near the defect moved as the same atoms are in the relaxed structure), with a
   single point of the pristine supercell in each (`C_screen/jobs/validate/`; the relaxed
   structures in `C_screen/structures/validate/`). A charge state C6 left in a band is left out:
   it has no defect level to test.
2. *Reference:* in each size, E = E_D − E_host + E_corr (the formation energy less its
   chemical-potential and Fermi-level terms, with `charge_correction` in that cell's shape). The
   line E_inf + b/N through the sizes above the screening cell (N, the pristine supercell's
   atoms) gives E_inf, the isolated defect's value, and b, the tail the sizes really show. `fit
   worst` is the farthest any of those sizes lies from the line: if it isn't small, they aren't
   on 1/N yet and E_inf is uncertain (more or bigger sizes help). With only one size above, E_inf
   is that size's E.
3. *Three estimates from the screening cell alone,* each compared with E_inf:
   - *as is:* its E, with the charge correction only;
   - *MACE c/N:* E − c/N, with c from B2's line fit: one c per arrangement, the same for every
     charge state, since MACE has no charge;
   - *stress:* E less ½ V s:S:s, the energy of the stress the fixed cell holds (s, the defect
     cell's stress tensor less the pristine cell's in the same size, so the basis set's Pulay
     stress cancels; S, the host's elastic compliance from `%block elastic_constants`). Each
     charge state gets its own, from its own stress. Without the block, it is left out.
4. *Verdict:* the error of each estimate in every charge state, and in every transition level
   between them (the level (q/q′) moves by (error(q′) − error(q)) / (q − q′)). A defect passes
   with an estimate when all of them are within `validate_tol` (0.05 eV).
5. *Γ against k-points* (`validate_gamma : true`, the default): a Γ-only single point at the
   production cut-off of every structure C8 relaxed, and of the pristine cell in each size
   (`C_screen/jobs/validate_gamma/`). The table gives how far Γ-only sampling moves E_D − E_host
   in each size: the error of the Γ-point screening (C2 to C7) and of D2's Γ fallback.

`C_screen/summary.txt` has a table of every size (E − E_inf, its stress energy, MACE's c/N and its
pressure less the pristine cell's: positive when the defect pushes out); each estimate's errors,
with b, the stress c (the stress energy × N in the screening cell) and MACE's c side by side; the
verdicts; and the Γ-only differences. b is what the two elastic estimates try to predict. If the
stress c follows b in every charge state and MACE's c doesn't, the strain differs between charge
states, which a model without charge can't see, and the stress estimate is the one to use. If
neither follows it, something besides strain still changes with size: a defect state spread over
its images (a shallow level, or one close to a band edge), or charge the point-charge correction
handles badly (potential alignment, an anisotropic ε). The stress energy is that of the cell's
average strain only: in a strongly anisotropic host, or for a defect whose strain field isn't the
same in every direction, the images also interact through their strain fields, which it leaves
out, and the test shows how much that matters.

It is costly: its biggest cells are among the ladder's largest. Name two or three defects (ones
whose charge states strain the lattice differently are the best test), and give `%block
runner_production` (or `runner_validate`) the cores and walltime for them. Before relying on it,
check that your CASTEP prints the stress tensor in a fixed-cell relaxation with `calculate_stress :
true` (C8 warns if it found none, and leaves the stress estimate out there). The elastic
constants should be your functional's own, since the stress comes from it: CaO's measured ones
(298 K) are C11 = 226, C12 = 62 and C44 = 81 GPa, and a GGA such as PBE usually gives somewhat
smaller ones.

### Phase D: production ground and excited states

| Label | Step | What it does | Decision |
|---|---|---|---|
| D1 | `relaxations` | every charge state of each defect phase C passes on, relaxed at the production settings in the electronic cell, with the pristine cell's single point | — |
| D2 | `zpl` | the production ΔSCF of each excitation C7 passed: at the production k-points if its bands keep their numbers there, else at Γ | C7's verdicts, with its thresholds |

Both run with the runner settings of `%block runner_production` (and `runner_relaxations` or
`runner_zpl` for one of them alone): a relaxation that runs out of walltime is carried on from
its `.check`, and a job that fails is flagged and left out (fix the cause and use
`--resubmit-failed`).

**D1 relaxations.** Every charge state of each defect phase C passes on, from C7's relaxed ground
state (with `excite : false`, C6's lowest structure, carried into the electronic cell), relaxed at
the production settings: A1's cut-off and k-point spacing, `%block production_overrides`, its
charge, and its spin from C7 (`spin_polarized : true` with that spin; a spin-0 state runs without
spin polarisation unless `production_overrides` sets `spin_polarized`). Positions only (the cell
stays at the host's relaxed lattice), no symmetry (`D_production/jobs/relaxations/`; the relaxed
structures in `D_production/structures/relaxations/`). The pristine electronic cell's single
point gives E_host and the band edges D2 measures levels against. All of a defect's charge states
are in one cell, so its transition levels compare like with like, and the cell folds the band
edges onto Γ, so the production k-points add accuracy rather than a different band structure.

**D2 zpl.** For each excitation C7 passed, in D1's relaxed ground state:

1. *Ground state:* a single point at the production settings with the bands' weights (its `.check`
   kept; `metals_method : dm` and `%block excite_overrides`, as C7's runs). The excitation is
   found again in the same spin channel (the localised levels, as in C7), and then the band-number
   check: CASTEP's ΔSCF constraints name a band number and spin channel, the same at every
   k-point, so at every k-point the same excitation must come out between the same two band
   numbers, and each of the two bands must spread by less than `band_spread_tol` (0.05 eV) across
   the k-points. In a cell that folds the band edges, a localised level is flat, and it passes.
2. *Γ fallback:* where the check fails (a level that disperses, or crosses another band at some
   k-point), a Γ-only single point at the production cut-off instead
   (`D_production/jobs/zpl_gs_gamma/`), where the band numbers are unambiguous; the table says
   which excitations fell back, and why.
3. *Excited state:* the ΔSCF from whichever single point (`D_production/jobs/zpl/`), relaxed: the
   ZPL, E_abs, ΔQ, the 1D S and DW and whether the electron stayed on the defect, with C7's
   verdicts and thresholds. The relaxed excited structures go to `D_production/structures/zpl/`,
   for phase F.

These are still PBE energies: the plan is to characterise the finalists from phase F with better
methods afterwards (many-body perturbation theory, or a hybrid ΔSCF in a code that does it:
CASTEP's ΔSCF does not support hybrid functionals).

**`freeze_radius`** (an untested idea, in D1): every atom further than r Å from all of the defect's
components is held fixed. That cuts the geometry steps of large cells but changes the relaxation
energy; compare it with full relaxations on one defect before relying on it.

### Phase E: thermodynamics

| Label | Step | What it does | Stops if |
|---|---|---|---|
| E1.1 | `phases` | the competing phases of the host and of the dopants of D1's defects, relaxed at the production settings | a phase doesn't finish |
| E1.2 | `limits` | the chemical-potential limits (or those of `%block chemical_potentials`) | a pure element of the host is missing; a dopant has no phase; the host is above the hull |
| E2 | `corrections` | the charge corrections in the electronic cell | — |
| E3 | `formation` | formation energies, transition levels, Fermi levels, `formation.png` | — |
| E4 | `interest` | the excitations that go on to phase F | — |

E1.1's are the phase's only CASTEP jobs (small bulk cells, with the runner settings of `%block
runner_host` and `runner_phases`); the rest runs in seconds.

**E1.1 phases.** The competing phases again, at the production settings now, for the formation
energies: every phase of the host's elements (but not of its own composition: A2.1's host is that
phase) and of each dopant with a defect in D1, all of those `%block phases` or doped give rather
than the lowest of each composition, relaxed as A3.1 relaxes them (solids at the production
k-point spacing, or their own `kspacing=`; give a metal e.g. `smearing_width=0.1` on its line),
in `E_thermo/jobs/phases/`. With `%block chemical_potentials` nothing runs.

**E1.2 limits.** A pymatgen phase diagram of the host (A2.1's energy) and the phases made of its own
elements (it needs a phase of each pure element, and stops if the host lies above the hull). Each
corner of the host's stability region is a limit, named after the phases there (`CaO-Ca`,
`CaO-O2`). Each dopant X of D1's defects then gets, at every corner, the highest μ_X its phases
allow, μ_X = min over phases P containing X of (E_P − Σ_{i≠X} n_i μ_i) / n_X, with the limiting
phase recorded. `summary.txt` shows the limits as Δμ against the pure elements; `results.json` has
the absolute values (CASTEP's energy scale), which E3 uses. With `%block chemical_potentials`,
those limits (absolute eV per atom on the energy scale of your production runs).

**E2 corrections.** For each charge state that reached phase D, the image-charge correction of the
electronic cell with ε₀ from A2.3: `lz` (Lany–Zunger, the default), `mp` (first-order
Makov–Payne) or `none`. No potential alignment.

**E3 formation.** For each defect, its production energy in each charge state (D1) gives

  E_f(q, E_F) = E_D − E_host − Σᵢ nᵢ μᵢ + q (E_VBM + E_F) + E_corr

at every chemical-potential limit, with E_host the pristine electronic cell's energy (D1), E_VBM
from A2.2 and μᵢ from E1.2. The transition levels are the Fermi levels where the lowest-energy
charge state changes, between the VBM and the gap (`gap`, if set, else A2.2's). `formation.png`
plots each defect's lowest charge state against E_F, one panel per limit.

With `fermi_temperature`, the self-consistent Fermi level at each limit, for each *set of
dopants* on its own: a crystal doped with Sb alone has none of Bi's defects, so E3 finds a Fermi
level for the native defects alone (`native`), for each dopant with the native defects (`Sb`,
`Bi`), and for each mix of dopants that one defect brings in together (`Bi+Sb`, for a pair with
both, with every defect of either dopant). Each is the E_F where the charges of that set's
defects and the free electrons and holes balance (concentrations from the formation energies,
the sites, orientations and the final spin's degeneracy; carriers from A2.2's band energies).
With `quench_temperature`, the concentrations are frozen at `fermi_temperature` and shared out
again between charge states at the lower temperature. Only the defects that reached phase D take
part; every native defect on the hull does (C7 lets them through).

C3's defect hull already compares like with like: arrangements of the same composition change
(the same atoms added and removed), so Sb's and Bi's defects are never on one hull, and a pair
with both is compared only with other arrangements of that pair.

**E4 interest.** The decision before phase F: an excitation goes on to F if it passed D2 and its
charge state is its defect's lowest for some Fermi level inside the gap (an excitation of a
charge state that never forms can't be seen). The table also says, for each, whether the
self-consistent Fermi level of the defect's own dopant set (E3) falls in that range at each limit,
the defect's deep levels (more than `deep_level_margin` from both band edges), and the band
states C7 left pending, to characterise with a hybrid functional later.

### Phase F: optical lineshapes

For each excitation E4 sends on, at the production settings (Γ where D2 used it), the
luminescence lineshape: F1, always, in the 1D configuration-coordinate model, and F2
(`lineshape : true`) from all the phonons. Both use the equal-mode approximation (the excited
state vibrates as the ground state does) and the generating-function method of Alkauskas et al.
(New J. Phys. 16, 073026, 2014), at `lineshape_temperature` as in Razinkovas et al. (Phys. Rev.
B 104, 045303, 2021).

| Label | Step | What it does | Decision |
|---|---|---|---|
| F1 | `ccd` | ground-state and ΔSCF single points along the line between the relaxed ground and excited states; parabolas; the 1D lineshape | — |
| F2 | `lineshape` | optional: a PBE model of each defect, its phonons in a big cell, the partial Huang–Rhys factors, the multi-mode lineshape | — |

Their CASTEP jobs run with the runner settings of `%block runner_production` (and `runner_ccd` or
`runner_lineshape`); F2's trainings and phonons are model jobs (`model_jobscript`).

**F1 ccd.** The structures R(x) = R_g + x (R_e − R_g) at each x of `ccd_points` (−0.4 to 1.4 in
steps of 0.2: x = 0 is D1's relaxed ground state, x = 1 D2's relaxed excited state), at
Q = x ΔQ. At each, a ground-state single point (its `.check` kept) and, from it, a ΔSCF single
point of the same excitation, at D2's settings (`F_lineshape/jobs/ccd_gs/`, `ccd_ex/`). A
parabola through each state's points near its own minimum (within 0.6 ΔQ) gives its phonon
energy, ħω = ħ √(2c) for E = c Q², and its Huang–Rhys factor in the 1D model, S = ħω ΔQ² / 2ħ²;
the misfit column is how far those points are from the parabola (anharmonicity). Also the ZPL
(E_e(1) − E_g(0), which should match D2's), the vertical absorption E_e(0) − E_g(0) and emission
E_e(1) − E_g(1). The 1D luminescence takes the ground state's mode (ħω_g, S_g), each phonon line
broadened by a Gaussian of `lineshape_sigma` (6 meV) and the ZPL by a Lorentzian of
`lineshape_gamma` (2 meV); DW = e^−S_g at 0 K, and its value at the temperature. Files in
`F_lineshape/ccd/`: `<excitation>_ccd.dat` (x, Q and both energies), `_pl_1d.dat` and
`_ccd.png` (the diagram and the lineshape).

The 1D model puts the displacement of every mode into one effective mode, so it overestimates S
(and underestimates DW) when many modes take part; F2 does it properly. F1 is cheap (20 single
points per excitation) and shows directly whether both states are harmonic and alike (ħω_g
against ħω_e), which the equal-mode approximation assumes.

**F2 lineshape** (`lineshape : true`).

1. *Data* (`F_lineshape/jobs/lineshape_data/`): `lineshape_displacements` (30) copies of each
   charge state's D1 ground state with every atom displaced at random (normal,
   `lineshape_amplitude`, 0.02 Å, a component), and `lineshape_host` (5) such copies of the
   pristine electronic cell, single points at the production settings. Each copy's displacements
   depend only on its name, so a rerun reuses the jobs.
2. *Model* (`F_lineshape/model/<defect>/`): a PBE MACE model of each defect, trained on those
   single points, on every geometry step of D1's relaxations of all its charge states, and on the
   pristine copies, a fifth kept for testing. `mace_run_train` (`model_command`) trains it from
   scratch, one `pbe` head with the total charge as a categorical embedding (`--embedding_specs`,
   21 classes from −10), so that one model covers every charge state, and its reference energies
   fitted to its data; with `model_r_max`, `model_channels`, `model_max_L`, `model_batch`,
   `model_epochs`, `model_device` and `model_train_args`. `train.sh` in the folder is the command;
   with `model_jobscript` it runs as a job, best on a GPU.
3. *Phonons* (`F_lineshape/phonons/<excitation>/`, a model job): the ground state carried into a
   big cell, the smallest whose images are at least `lineshape_length` (30 Å) apart (the
   electronic cell if that is longer already), and relaxed with the model; its force constants by
   central finite differences (±`lineshape_delta`, 0.01 Å, each atom and direction: 6N force
   evaluations, saved every five minutes so a job stopped at its time limit carries on); the
   dynamical matrix diagonalised: each mode's ħω_k, and how many atoms take part in it.
4. *Coupling:* with `lineshape_method : forces` (the default, after Razinkovas et al.), ΔF, the
   excited state's forces at the ground state's structure (the first step of D2's ΔSCF) less the
   ground state's there, put onto the same atoms of the big cell out to just under half the
   electronic cell's image distance (it is localised, so it carries over cleanly; zero further
   out), and ΔQ_k = e_k · M^−½ ΔF / ω_k²; with `displacements`, ΔR between D2's relaxed excited and
   D1's ground state, ΔQ_k = e_k · M^½ ΔR (Alkauskas et al. 2014; ΔR is cut off at the electronic
   cell's edge). Each mode's partial Huang–Rhys factor S_k = ω_k ΔQ_k² / 2ħ, and the spectral
   function S(ħω) = Σ_k S_k g(ħω − ħω_k), g a Gaussian of width `lineshape_sigma`.
5. *Lineshape:* the generating function G(t) = exp(∫ S(ħω) [(n + 1)(e^−iωt − 1) + n (e^iωt − 1)]
   dω), n the Bose occupation at `lineshape_temperature`; its Fourier transform A, with the ZPL's
   Lorentzian (`lineshape_gamma`); and the photons' spectrum L(E) ∝ E³ A(ZPL − E). The table gives
   S, ħω_eff (the S-weighted mean phonon energy), DW at 0 K and at the temperature, the ΔQ the
   phonons imply (against D2's) and the model's test error and imaginary modes (an imaginary mode
   means the model's minimum is not a minimum there: it is left out, with a warning). Files in
   `F_lineshape/lineshape/`: `<excitation>_S.dat` (the spectral function), `_pl.dat` and
   `_lineshape.png` (with F1's 1D lineshape for comparison).

A 30 Å cell of CaO holds about 1900 atoms: some 11,500 model force evaluations, minutes on a GPU
and hours on a CPU node. The model gives the force constants of the whole cell; embedding a
smaller cell's force constants in the bulk's (Razinkovas et al.) would reach larger cells, and is
not done here.

---

## Keyword glossary

Every keyword of every input file, sorted by file. The project's two files (section 1) are the only
files `dwf.py` is given; they name the others. For the CASTEP files (sections 2 to 4) the tables
list only the keywords the workflow sets, changes or removes; every other keyword is passed on to
the jobs as you wrote it.

Sections: [1. The project's two files](#1-the-projects-two-files-nameparameters-and-namerun) ·
[2. The production `.param`](#2-the-production-param-param) ·
[3. The host `.cell`](#3-the-host-cell-host_cell) ·
[4. The dielectric files](#4-the-dielectric-param-and-cell-dielectric_param-dielectric_cell) ·
[5. Competing phases](#5-competing-phase-structure-files-block-phases) ·
[6. The jobscript](#6-the-jobscript-template-jobscript) ·
[7. The MACE jobscript](#7-the-mace-jobscript-template-mace_jobscript) ·
[8. The model jobscript](#8-the-model-jobscript-template-model_jobscript) ·
[9. The runner's own files](#9-the-job-runner-on-its-own-conv-and-batch-files)

### 1. The project's two files (`NAME.parameters` and `NAME.run`)

The two files you give `dwf.py`, with one name: `NAME.parameters`, the science, and `NAME.run`, how
the jobs run. `NAME.run` holds the `workdir`, how the [MACE](#mace) jobs run (`mace_jobscript`,
`mace_pack`, `mace_max_jobs`, `mace_threads`, `mace_python`, `mace_device`, and B2's, C1's and C4's
own blocks, `%block runner_scan`, `%block runner_prerelax` and `%block runner_distort`), how F2's
model jobs run (`model_jobscript`, `model_command`, `model_device`) and every [runner
setting](#runner-settings) of the CASTEP jobs, with their blocks; `NAME.parameters` holds everything
else. `dwf.py` stops on a keyword in the wrong one, and says which it goes in. The keywords are
grouped below by the phase that uses them; those of `NAME.run` say so.

**Writing them.** `keyword : value` (or `=`, or a space), in any case, each keyword once (in one
of the two files); `#` or `!` starts a comment. File names are relative to the project's files.

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

- **A**: [`antisites`](#b-candidates-the-screening-and-electronic-cells) (B) · [`array`](#runner-settings) (runner) · [`array_min`](#runner-settings) (runner)
- **B**: [`band_spread_tol`](#d-production) (D) · [`bands_kspacing`](#a-host) (A) · [`baseline_mb`](#runner-settings) (runner)
- **C**: [`carrier_fraction`](#c-screening-ground-and-excited-states) (C) · [`carrier_reach`](#c-screening-ground-and-excited-states) (C) · [`ccd_points`](#f-lineshapes) (F) · [`charge_correction`](#e-thermodynamics) (E) · [`charge_margin`](#c-screening-ground-and-excited-states) (C) · [`charging`](#runner-settings) (runner) · [`%block chemical_potentials`](#a-host) (A) · [`configuration_margin`](#c-screening-ground-and-excited-states) (C) · [`converge`](#a-host) (A) · [`converge_cutoffs`](#a-host) (A) · [`converge_kcutoff`](#a-host) (A) · [`converge_kgrid_max`](#a-host) (A) · [`converge_kgrid_min`](#a-host) (A) · [`converge_kgrid_step`](#a-host) (A) · [`converge_tol`](#a-host) (A) · [`cores`](#runner-settings) (runner) · [`cores_per_node`](#runner-settings) (runner)
- **D**: [`decay_length_multiple`](#b-candidates-the-screening-and-electronic-cells) (B) · [`deep_level_margin`](#e-thermodynamics) (E) · [`%block defects`](#b-candidates-the-screening-and-electronic-cells) (B) · [`delete_check`](#runner-settings) (runner) · [`dielectric`](#a-host) (A) · [`dielectric_cell`](#a-host) (A) · [`dielectric_inf`](#a-host) (A) · [`dielectric_param`](#a-host) (A) · [`dielectric_task`](#a-host) (A) · [`distort_around`](#c-screening-ground-and-excited-states) (C) · [`distort_cell`](#c-screening-ground-and-excited-states) (C) · [`distort_charges`](#c-screening-ground-and-excited-states) (C) · [`distort_starts`](#c-screening-ground-and-excited-states) (C) · [`distortion_min_gain`](#c-screening-ground-and-excited-states) (C) · [`distortions`](#c-screening-ground-and-excited-states) (C) · [`dopant_pairs`](#b-candidates-the-screening-and-electronic-cells) (B) · [`dopants`](#b-candidates-the-screening-and-electronic-cells) (B) · [`dryrun_cmd`](#runner-settings) (runner) · [`dryrun_cores`](#runner-settings) (runner) · [`dryrun_jobscript`](#runner-settings) (runner) · [`dryrun_mem_gb`](#runner-settings) (runner) · [`dryrun_on`](#runner-settings) (runner) · [`dryrun_options`](#runner-settings) (runner) · [`dryrun_pack`](#runner-settings) (runner) · [`dryrun_partition`](#runner-settings) (runner) · [`dryrun_time`](#runner-settings) (runner) · [`dryrun_timeout`](#runner-settings) (runner)
- **E**: [`%block elastic_constants`](#c-screening-ground-and-excited-states) (C) · [`electronic_cell`](#b-candidates-the-screening-and-electronic-cells) (B) · [`electronic_depth`](#b-candidates-the-screening-and-electronic-cells) (B) · [`energy`](#general) (general) · [`excite`](#c-screening-ground-and-excited-states) (C) · [`excite_min`](#c-screening-ground-and-excited-states) (C) · [`excite_min_dw`](#c-screening-ground-and-excited-states) (C) · [`%block excite_overrides`](#c-screening-ground-and-excited-states) (C) · [`excite_reach`](#c-screening-ground-and-excited-states) (C) · [`excite_relax`](#c-screening-ground-and-excited-states) (C) · [`excite_spin`](#c-screening-ground-and-excited-states) (C) · [`excite_window`](#c-screening-ground-and-excited-states) (C)
- **F**: [`fermi_temperature`](#e-thermodynamics) (E) · [`freeze_radius`](#d-production) (D)
- **G**: [`gap`](#e-thermodynamics) (E)
- **H**: [`host_cell`](#general) (general) · [`hull_margin`](#c-screening-ground-and-excited-states) (C)
- **I**: [`interstitials`](#b-candidates-the-screening-and-electronic-cells) (B)
- **J**: [`jobscript`](#runner-settings) (runner)
- **K**: [`keep_structures`](#c-screening-ground-and-excited-states) (C) · [`kspacing`](#a-host) (A)
- **L**: [`level_margin`](#c-screening-ground-and-excited-states) (C) · [`lineshape`](#f-lineshapes) (F) · [`lineshape_amplitude`](#f-lineshapes) (F) · [`lineshape_delta`](#f-lineshapes) (F) · [`lineshape_displacements`](#f-lineshapes) (F) · [`lineshape_gamma`](#f-lineshapes) (F) · [`lineshape_host`](#f-lineshapes) (F) · [`lineshape_length`](#f-lineshapes) (F) · [`lineshape_method`](#f-lineshapes) (F) · [`lineshape_sigma`](#f-lineshapes) (F) · [`lineshape_temperature`](#f-lineshapes) (F) · [`local_fraction`](#c-screening-ground-and-excited-states) (C)
- **M**: [`mace`](#mace) (MACE) · [`mace_device`](#mace) (MACE) · [`mace_fmax`](#mace) (MACE) · [`mace_jobscript`](#mace) (MACE) · [`mace_max_jobs`](#mace) (MACE) · [`mace_model`](#mace) (MACE) · [`mace_pack`](#mace) (MACE) · [`mace_python`](#mace) (MACE) · [`mace_rattle`](#mace) (MACE) · [`mace_steps`](#mace) (MACE) · [`mace_threads`](#mace) (MACE) · [`max_cores`](#runner-settings) (runner) · [`max_jobs`](#runner-settings) (runner) · [`max_nodes`](#runner-settings) (runner) · [`max_restarts`](#runner-settings) (runner) · [`max_walltime`](#runner-settings) (runner) · [`mem_per_core`](#runner-settings) (runner) · [`mem_replicated`](#runner-settings) (runner) · [`mem_safety`](#runner-settings) (runner) · [`model_batch`](#f-lineshapes) (F) · [`model_channels`](#f-lineshapes) (F) · [`model_command`](#f-lineshapes) (F) · [`model_device`](#f-lineshapes) (F) · [`model_epochs`](#f-lineshapes) (F) · [`model_jobscript`](#f-lineshapes) (F) · [`model_max_L`](#f-lineshapes) (F) · [`model_r_max`](#f-lineshapes) (F) · [`model_train_args`](#f-lineshapes) (F) · [`mp_api_key`](#a-host) (A)
- **N**: [`name`](#general) (general) · [`native_pairs`](#b-candidates-the-screening-and-electronic-cells) (B)
- **O**: [`omp_threads`](#runner-settings) (runner) · [`%block oxidation_states`](#c-screening-ground-and-excited-states) (C)
- **P**: [`pack`](#runner-settings) (runner) · [`pair_max_distance`](#b-candidates-the-screening-and-electronic-cells) (B) · [`pairs`](#b-candidates-the-screening-and-electronic-cells) (B) · [`param`](#general) (general) · [`%block partitions`](#runner-settings) (runner) · [`%block phases`](#a-host) (A) · [`phases_e_above_hull`](#a-host) (A) · [`phases_from`](#a-host) (A) · [`phases_max_steps`](#a-host) (A) · [`phases_molecule_box`](#a-host) (A) · [`poll`](#runner-settings) (runner) · [`%block production_overrides`](#d-production) (D)
- **Q**: [`quench_temperature`](#e-thermodynamics) (E)
- **R**: [`rattle_dmin`](#c-screening-ground-and-excited-states) (C) · [`rattle_stdev`](#c-screening-ground-and-excited-states) (C) · [`ref_kspacing_factor`](#a-host) (A) · [`ref_polymorph_window`](#a-host) (A) · [`ref_relax`](#a-host) (A) · [`require_bound`](#c-screening-ground-and-excited-states) (C) · [`require_gap_states`](#c-screening-ground-and-excited-states) (C) · [`require_spin`](#c-screening-ground-and-excited-states) (C) · [`%block runner_scan`, `runner_prerelax`, `runner_distort`](#mace) (MACE) · [`%block runner_host`, `runner_screen`, `runner_production`, `runner_<step>`](#runner-settings) (runner)
- **S**: [`scan`](#b-candidates-the-screening-and-electronic-cells) (B) · [`scan_edge_tol`](#b-candidates-the-screening-and-electronic-cells) (B) · [`scan_lengths`](#b-candidates-the-screening-and-electronic-cells) (B) · [`scan_test`](#b-candidates-the-screening-and-electronic-cells) (B) · [`scan_tol`](#b-candidates-the-screening-and-electronic-cells) (B) · [`scan_unconverged`](#b-candidates-the-screening-and-electronic-cells) (B) · [`screen_charges`](#c-screening-ground-and-excited-states) (C) · [`screen_cutoff`](#c-screening-ground-and-excited-states) (C) · [`screen_max_steps`](#c-screening-ground-and-excited-states) (C) · [`screen_min_image`](#b-candidates-the-screening-and-electronic-cells) (B) · [`%block screen_overrides`](#c-screening-ground-and-excited-states) (C) · [`screen_spins`](#c-screening-ground-and-excited-states) (C) · [`share_rounds`](#c-screening-ground-and-excited-states) (C) · [`site_margin`](#c-screening-ground-and-excited-states) (C) · [`sp_walltime`](#runner-settings) (runner) · [`spin_tol`](#c-screening-ground-and-excited-states) (C) · [`status_cmd`](#runner-settings) (runner) · [`submit_cmd`](#runner-settings) (runner) · [`supercell`](#b-candidates-the-screening-and-electronic-cells) (B)
- **T**: [`tolerate_failures`](#runner-settings) (runner) · [`transform`](#general) (general)
- **V**: [`vacancies`](#b-candidates-the-screening-and-electronic-cells) (B) · [`validate`](#c-screening-ground-and-excited-states) (C) · [`validate_defects`](#c-screening-ground-and-excited-states) (C) · [`validate_gamma`](#c-screening-ground-and-excited-states) (C) · [`validate_sizes`](#c-screening-ground-and-excited-states) (C) · [`validate_tol`](#c-screening-ground-and-excited-states) (C)
- **W**: [`walltime`](#runner-settings) (runner) · [`walltime_safety`](#runner-settings) (runner) · [`workdir`](#general) (general)

Groups: [General](#general) · [MACE](#mace) · [A: host](#a-host) ·
[B: candidates, the screening and electronic cells](#b-candidates-the-screening-and-electronic-cells)
· [C: screening, ground and excited states](#c-screening-ground-and-excited-states) ·
[D: production](#d-production) · [E: thermodynamics](#e-thermodynamics) ·
[F: lineshapes](#f-lineshapes) · [Runner settings](#runner-settings)

#### General

| Keyword | Default | Steps | What it does |
|---|---|---|---|
| `host_cell` | required | A1, A2.1 | The host cell, normally the primitive cell (a conventional cell also works); A2.1 relaxes it. Everything in it apart from the structure and k-points (`species_pot`, `symmetry_generate`, any other `.cell` keyword) is copied into every job's `.cell`; defect cells lose the symmetry keywords. List the dopants in `species_pot` too, or leave the block out to use CASTEP's default on-the-fly potentials for every species. |
| `param` | required | A1 | The production settings: functional, SCF and geometry settings, smearing, and so on. A1 writes `production.param` from it, with the converged `cut_off_energy` and without any `charge` or `spin`, and every later job starts from that file. `dwf.py` sets `task`, and per job `charge`, `spin`, `spin_polarized` and the k-points. |
| `transform` | — | B2, B3, B4 | A supercell shape: 9 integers, row by row, each row a supercell vector in terms of the primitive vectors (determinant > 0). With it, each size of the ladder is n × `transform`, the smallest n that reaches the length, and `supercell : n` is n × `transform`; e.g. the example's (commented-out) matrix turns an FCC primitive cell into cubic cells of 8n³ atoms, which fold rock salt's X point onto Γ. Without it (the default), each size is the most compact cell doped finds. |
| `energy` | `auto` | A1, A2.1, A3.1, C2, C3, C5, C6, C7, C8, D1, D2, E1.1, F1, F2 | Which CASTEP energy every step reads: `final` (Final energy), `free` (E − TS), `0K` (the 0 K estimate) or `auto` (the 0 K estimate when CASTEP prints one, else the final energy). They differ only with smearing. |
| `name` | the files' name | — | The project's name: the default work directory is `dwf_<name>`, and it titles the E3 plot. |
| `workdir` | `dwf_<name>` | — | In `NAME.run`. The work directory, relative to the project's files. |

#### MACE

The foundation model (`mace_model`) has no charge or spin: it prepares structures (B2, C1) and
searches the distortions (C4), whose minima CASTEP then relaxes in every charge state, and never
ranks charge states. F2's model of each defect, which has the charge, is set up under
[F](#f-lineshapes). `mace`, `mace_model`, `mace_fmax` and `mace_steps` go in `NAME.parameters`; the
rest, how the MACE jobs run, in `NAME.run`. Those settings are B2's, C1's and C4's alike, unless a
step's own block gives some of them: `%block runner_scan` for B2, whose relaxations of every
candidate in every size are by far the most, `%block runner_prerelax` for C1 and
`%block runner_distort` for C4.

| Keyword | Default | Steps | What it does |
|---|---|---|---|
| `mace` | `true` | B2, B3, C1, C4 | Use MACE for the scan of every candidate (B2), the configurational-energy cut (B3, on B2's energies), the pre-relaxation of the candidates (C1) and the search of the distortions (C4). With `false`, B2 makes only the size ladder, B3 goes by the Γ-only floor (or `supercell`), C1 builds unrelaxed structures, `configuration_margin` does nothing, and C5 relaxes every distortion in CASTEP (`distortions` then `-0.4 -0.2 0.2 0.4` unless you give it). |
| `mace_model` | `medium-mpa-0` | B2, C1, C4 | The model: the name of a `mace_mp` foundation model, or a model file of your own. A new model makes B2, C1 and C4 run again (their MACE relaxations are kept per model, so going back to an earlier one reuses them). |
| `mace_fmax` | `0.05 eV/ang` | B2, C1, C4 | A MACE relaxation stops when the largest force is below this. |
| `mace_steps` | `500` | B2, C1, C4 | ... or after this many optimiser (LBFGS) steps. |
| `mace_rattle` | `0.02 ang` | B2, C1 | The standard deviation of the random displacements of a relaxation's start: the atoms within 4 Å of the defect's components, moved the same way every time, so that a defect on a symmetric site isn't held there by its own symmetry. R is still measured from the ideal structure. `0`: no rattle. Part of each relaxation's cache key, so a new value (the first run with this keyword, too) runs B2's and C1's relaxations again; C4's distortions have ShakeNBreak's own rattle. |
| `mace_device` | `cpu` | — | In `NAME.run`. `cpu`, or `cuda` on a GPU; also F2's models' default (`model_device`). |
| `mace_threads` | `8` | — | In `NAME.run`. Torch threads per relaxation. |
| `mace_jobscript` | — | — | In `NAME.run`. A Slurm template (a file, or `%block mace_jobscript`) for running the MACE relaxations (B2, C1, C4) as jobs on compute nodes. It needs one line with `{COMMAND}`, which becomes the workers; whatever is in front of it on that line (e.g. `srun --exact --ntasks=1 --cpus-per-task=8`) goes in front of each. `{JOBNAME}` is filled in too. Without it, the relaxations run where `dwf.py` runs. See [section 7](#7-the-mace-jobscript-template-mace_jobscript). |
| `mace_pack` | `cores_per_node` / `mace_threads` | — | In `NAME.run`. Relaxations side by side in one MACE job (36 on Cirrus, 16 on ARCHER2, with 8 threads each). |
| `mace_max_jobs` | `4` | — | In `NAME.run`. The most MACE jobs a step has queued or running at once; its relaxations are shared out between them, biggest first, and what a job leaves (it ran out of time) goes into the next jobs as places come free. Give the jobs time for their share (`max_walltime` in the step's block): a round that runs out leaves most of its relaxations to the next. |
| `mace_python` | the Python running `dwf.py` | — | In `NAME.run`. The Python the MACE jobs run, with mace-torch. On Cirrus and ARCHER2 it must be on `/work`. A linked folder in its path (e.g. a `~/shared` that links to `/work/...`) is followed, so the jobs get the `/work` path. |
| `%block runner_scan` | — | B2 | In `NAME.run`. B2's own settings for its MACE jobs, over those above: any of `mace_jobscript`, `mace_pack`, `mace_max_jobs`, `mace_threads`, `mace_python` and `mace_device`, and `max_walltime`, each job's `--time` (without it, the template's). B2 relaxes every candidate in every size it fits, thousands of relaxations for a big screen, so e.g. `mace_max_jobs : 40` with `max_walltime : 12:00:00`. Its `mace_pack` × `mace_threads` must fit on a node (`cores_per_node`). |
| `%block runner_prerelax` | — | C1 | In `NAME.run`. The same for C1's MACE jobs (the relaxations C1 runs itself: all of them when the screening cell isn't one of B2's sizes). |
| `%block runner_distort` | — | C4 | In `NAME.run`. The same for C4's MACE jobs: every survivor's distortions, of all its charge states, in the electronic cell (432 atoms for CaO), so fewer relaxations than B2's but bigger ones. |

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
| `dielectric` | from A2.3 | A2.3 | The static dielectric constant (electronic + ionic), ε₀, for every charge correction (C3, C8, E2). Setting it skips the DFPT run. |
| `dielectric_inf` | — | A2.3 | The optical constant ε∞, recorded with `dielectric`. The corrections don't use it. |
| `dielectric_task` | `phonon+efield` | A2.3 | The CASTEP task of the DFPT run, at q = Γ. It must give the DC (static) permittivity: with `efield`, which gives only ε∞, A2.3 stops. |
| `dielectric_param` | — | A2.3 | A `.param` for the DFPT run instead of `production.param`. CASTEP's DFPT needs norm-conserving potentials, fixed occupancies and no spin polarisation. Without this keyword the production settings are used with `fix_occupancy : true` and `spin_polarized : false`; with it, put those in the file, with a cut-off suited to norm-conserving potentials. |
| `dielectric_cell` | — | A2.3 | A `.cell` whose `species_pot` (and other settings) the DFPT run uses instead of the host `.cell`'s, e.g. `NCP` for CASTEP's norm-conserving library. Its structure is ignored: the run uses the relaxed host. A2.3 warns before submitting if the potentials don't look norm-conserving. |
| `%block phases` | — | A3.1, E1.1 | The competing phases, one per line: `name  structure-file  [options]`. Options: `molecule` (positions only, in its fixed box, at Γ, without symmetry); `metal` (`fix_occupancy : false` unless the line says otherwise; add `smearing_width=...`); `kspacing=0.03` (its own k-point spacing); and any other `keyword=value`, a `.param` keyword for that phase only (e.g. `spin=2 spin_polarized=true` for O₂). Solids relax lattice and positions, with symmetry: at the screening settings in A3.1 (the pure elements, the host's and the dopants'), at the production settings in E1.1 (the host's, and those of the dopants that reach D, compounds too). A phase of the host's own composition is left out (the host is A2.1's). A3.2 and E1.2 need a phase of each pure element of the host; C3 gives a dopant's defects a least formation energy only with its element, and E1.2 needs at least one phase containing each dopant of D1's defects. |
| `phases_from` | `none` | A3.1, E1.1 | `doped`: when there is no `%block phases`, take the competing phases from the Materials Project through doped (needs doped and `mp_api_key`). The phases doped chooses are kept in `A_host/phases_doped.json` for later runs ([Phase A](#phase-a-the-host)). |
| `mp_api_key` | — | — | Materials Project API key for `phases_from : doped`: the 32 characters on your dashboard, without quotes. Given here, it is used rather than doped's own (`PMG_MAPI_KEY` in `~/.pmgrc.yaml`). It needs doped 3 and pymatgen 2026.9.23 or later (so Python 3.11 or later; older pymatgen can't read the Materials Project's replies): doped 2.x takes only the legacy keys, whose API the Materials Project was closing in September 2025; 3.0 takes the current ones but also needs `mp-api` and `emmet-core`, 3.1 and later nothing more. |
| `phases_e_above_hull` | `0.05 eV/atom` | A3.1, E1.1 | `phases_from : doped` takes the Materials Project entries up to this far above the hull. A3.1 relaxes only the lowest of each element, E1.1 every one. |
| `phases_max_steps` | `100` | E1.1, A3.1 | `geom_max_iter` of each competing phase's relaxation (CASTEP's own is 30, which can be too few for a soft metal's cell): E1.1's, A3.1's molecules, and A3.1's solids with `ref_relax : positions` or `cell`. A `%block phases` line's own `geom_max_iter=` wins for that phase. |
| `ref_relax` | `none` | A3.1 | How A3.1 runs its solids (the host and the pure elements): `none`, single points of the structures as given (doped's: the Materials Project's, PBE-relaxed, as symmetrised primitive cells); `positions`, the positions relaxed in the given cell; `cell`, lattice and positions (at a low `screen_cutoff` the Pulay stress can make a soft metal's cell run away). Molecules always relax in their box. |
| `ref_kspacing_factor` | `1` | A3.1 | A3.1's k-point spacing as a multiple of A1's (or of a phase's own `kspacing=`): 2 gives half the k-points in each direction, an eighth in all. C3's cuts need these energies to tens of meV/atom, not A1's 1 meV/atom. The factor scales A1's spacing, converged for the host; the table gives every phase's grid. |
| `ref_polymorph_window` | `0.02 eV/atom` | A3.1 | Of an element's polymorphs (doped's) within this of the lowest above the hull, A3.1 runs the one with the fewest atoms in its primitive cell (then the lowest); `0` takes the lowest. Polymorphs can lie within meV/atom of each other, and the lowest can be a far bigger cell. It moves that element's reference by at most this much. |
| `phases_molecule_box` | `15 ang` | A3.1, E1.1 | The side of the box each of `phases_from : doped`'s molecules (H₂, N₂, O₂, F₂, Cl₂) relaxes in: doped's shape, that +0.01, that and −0.01 Å along x, y and z, with the molecule as doped made it, centred. doped's own box is 30 Å, eight times the volume, so about eight times the plane waves, memory and time. `30 ang` keeps doped's structures as they are; a box that leaves less than 5 Å between a molecule and its images is refused. A `%block phases` molecule keeps its file's box. |
| `%block chemical_potentials` | — | E1.1, E1.2 | The limits themselves, one per line: `name  El=value ...`, in absolute eV per atom on the energy scale of your production runs, for every element of the host and the dopants. E1.1 then relaxes nothing, and E1.2 records them: it takes precedence over `%block phases` and `phases_from` there. They are not on the screening settings' scale, so A3 still runs `%block phases` or doped's phases, if given (without them, C3 makes no cut across a dopant's sites). |

#### B: candidates, the screening and electronic cells

Details: [Phase B](#phase-b-candidate-defects-the-screening-and-electronic-cells).

| Keyword | Default | Steps | What it does |
|---|---|---|---|
| `dopants` | — (*list*) | A3.1, A3.2, B1, E1.1, E1.2 | The extrinsic elements. B1 puts each on every inequivalent host site, and on the interstitial sites with `interstitials`; A3 puts each at its element for C3, and E1 holds their chemical potentials down with their compounds; C4 guesses their oxidation states. |
| `%block defects` | — | B1 | Your own defects, one per line, components joined by `+`: `Sb_Ca` (Sb on a Ca site), `Vac_O`, `Vac_O:2` (the 2nd O site of the primitive cell), `Sb_Ca@2` (starting spin 2 on the Sb). Each later component goes on its nearest site of that kind to the first one; with `all` on the line (pairs only), every arrangement up to `pair_max_distance` becomes a candidate. `q=-1` (or `q=+1,0`) on a line gives its defects' charge states, used instead of `screen_charges` from C2 on. With this block, nothing is generated. |
| `vacancies` | `true` | B1 | A vacancy on each inequivalent host site. |
| `antisites` | `false` | B1 | Each other host element on each inequivalent host site. |
| `interstitials` | `none` | B1 | `dopants`: the dopants on each Voronoi interstitial site (pymatgen-analysis-defects); `all`: every element there. |
| `pairs` | `true` | B1 | Pairs of a dopant defect with a native one, in every symmetry-distinct arrangement up to `pair_max_distance` apart. Never two interstitials. |
| `native_pairs` | `false` | B1 | Also pairs of two native defects. |
| `dopant_pairs` | `false` | B1 | Also pairs of two dopant defects: `true` any two (Sb_Ca + N_O as well as Sb_Ca + Sb_O), `same` only two of the same dopant, so that each dopant meets the host on its own. |
| `pair_max_distance` | `5.0 ang` | B1 | The largest separation of the pair arrangements, generated or from `all`. A pair farther apart than half a cell's shortest image distance isn't built in it: B2 skips that size, and C1 leaves the pair out if it is the screening cell. |
| `scan` | `true` | B2 | Scan every candidate with MACE in every size (also needs `mace : true`). With `false`, B2 makes only the size ladder. |
| `scan_lengths` | `8 10 12 14 16 18 20 ang` (*list*) | B2, B3, B4, C8 | The sizes, as lengths: for each, the most compact supercell whose periodic images are all at least that far apart (doped; n × `transform` with a transform). B2 scans them, B3 and B4 pick their cells from them (B4 goes beyond the largest if it must), and C8's sizes are the next ones above the screening cell. With `scan_test : line` a candidate converges at the fourth-largest size at most, and a pair can only be tested if it fits that size: B2 warns when `pair_max_distance` is too big for the ladder. |
| `scan_tol` | `0.02 eV` | B2 | With `scan_test : line`, the most any size's MACE relaxation energy may be off the line R∞ − c/N through it and the bigger sizes. With `flat`, a candidate's relaxation energy must be within this of that of every bigger size ... |
| `scan_edge_tol` | `0.05 ang` | B2 | (`scan_test : flat` only) ... and the atoms at the cell's edge (further than 35 % of the image distance from the defect) moved less than this, for it to be converged in that size. |
| `scan_unconverged` | `drop` | B3 | What B3 does with a candidate B2's size scan didn't converge (with too few sizes for the test, or still not after the re-seeding from its deepest minimum). `drop`: C1 leaves it out, and it doesn't size the screening cell. `largest`: it is screened, in a cell sized as if it needed the largest size. Not used with `supercell`. |
| `scan_test` | `line` | B2 | How B2 decides a candidate has converged. `line`: the smallest size from which its relaxation energy follows R∞ − c/N (N, the cell's atoms): it and at least three bigger sizes within `scan_tol` of the line through them; the table gives R∞, c and c/N there. `flat`: the smallest size whose relaxation energy is within `scan_tol` of every bigger size's, with the edge atoms moved less than `scan_edge_tol`. |
| `screen_min_image` | from the band masses | B3 | The Γ-only floor (Å) yourself: the screening cell's images are at least this far apart. `0`: no floor. Without it, B3 estimates the floor from A2.2's effective masses, `deep_level_margin` and `decay_length_multiple`. |
| `supercell` | — | B3 | Fix the screening cell: a length (`supercell : 14 ang`, the most compact cell for it) or n (`supercell : 3`, n × `transform`). B2's scan is then only reported. |
| `electronic_cell` | from B4's search | B4 | Fix the electronic cell: a length (`20 ang`), n (n × `transform`), or `screening` (the screening cell). B4 warns if it doesn't fold every valley of the band edges onto Γ. |
| `electronic_depth` | `0.25 eV` | B4 | The electronic cell must meet the Γ-only floor (B3's estimate, from the band masses: `decay_length_multiple` decay lengths) for a level this far from a band edge. |
| `decay_length_multiple` | `4` | B3, B4 | The Γ-only floor is this many decay lengths ξ of the level it is for: B3's of a level `deep_level_margin` deep (unless `screen_min_image` gives the floor), B4's of one `electronic_depth` deep. At Γ the images make a level that deep look deeper than it is, by about 10 % at 4, 19 % at 3.5 and 33 % at 3 (an s-like level with no core and 12 nearest images; more for a broad core, less for a compact one), and deeper levels by much less. A cell's atoms go as the cube of the multiple. Must be greater than 0. |

#### C: screening, ground and excited states

Details: [Phase C](#phase-c-screening-ground-and-excited-states). The screening runs are cheap on
purpose: Γ point only, possibly a lower cut-off, few geometry steps. Their energies rank candidates;
phase D redoes the survivors properly. The screening settings (`screen_cutoff`, `screen_max_steps`,
`%block screen_overrides`, `level_margin`) are C3's single points', C6's and C7's too, and
`screen_cutoff` and `%block screen_overrides` those of A3.1's bulk runs.

| Keyword | Default | Steps | What it does |
|---|---|---|---|
| `configuration_margin` | — | B3 (C1 without B2's scan) | Before any CASTEP run, and before B3 chooses the screening cell, drop arrangements whose configurational energy (MACE energy above the lowest arrangement of the same composition, compared in the biggest size B2 relaxed them all in) is more than this, e.g. `1.0 eV`; the screening cell is then sized for the rest. Without B2's scan, C1 makes the cut in the screening cell. Off unless given; needs `mace : true`. MACE has no charge or spin, so keep it generous. Replaces `triage` and `triage_window`. |
| `screen_charges` | `-1 0 1` (*list*) | C2 | The charge states every candidate is screened in, except one with its own `q=` in `%block defects`. |
| `screen_spins` | `triplet` | C2 | The starting spin (unpaired electrons): `triplet` runs once from 2, or from 1 (a doublet) when the electron count is odd; `low` runs once from the lowest the electron count allows (0 or 1); `both` runs from that and from that + 2. A starting value: the results keep the final spin. |
| `screen_cutoff` | the production cut-off | A3.1, C2, C3, C5, C6, C7 | The cut-off of the screening runs, and of A3.1's reference phases, so that their energies are on one scale. |
| `screen_max_steps` | `50` | C2, C5, C6, C7 | `geom_max_iter` of the screening relaxations. A run that reaches it is logged as failed but keeps its last energy, marked as not converged. |
| `%block screen_overrides` | — | A3.1, C2, C3, C5, C6, C7 | Any other `.param` keywords for the screening runs (and A3.1's phases), e.g. a looser `elec_energy_tol`, or `fine_gmax : 35 1/ang`, a finer density grid, for hard ultrasoft potentials at a low `screen_cutoff` (a phase's cell relaxation whose energy jumps as the grid changes, and never converges). |
| `level_margin` | `0.1 eV` | C2, C3, C5, C6, C7 | How far inside the host's band edges (from the pristine cell at the same settings) an eigenvalue must lie to count as a defect level in the gap. Anything closer to an edge, or past it, is in a band: the electrons and holes there make a run a band state (at least 0.5 of either), which C3 doesn't judge by its energy. The eigenvalues aren't aligned to the host's, so raise it if a defect cell's band edges sit visibly off the host's. |
| `hull_margin` | `0.2 eV` | C3 | Keep an arrangement if its lowest formation energy comes within this of its composition group's defect hull somewhere in the gap. Allow for the error of the screening settings. |
| `site_margin` | `0.5 eV` | C3 | Keep a defect with dopant atoms if its formation energy comes within this of the lowest of the defects with the same dopant atoms (on any site, with any host atoms added or removed) somewhere in the host's stability region and the gap: the dopant's chemical potential cancels between them. The region is A3.2's, against the host's elements alone; each such set keeps at least its lowest. Needs A3's phases. Native defects aren't compared so. |
| `require_gap_states` | `none` | C3 | Keep only defects with `occupied` gap levels, `empty` ones, or `both` (an occupied and an empty level in the same spin channel: an internal transition), in a stable charge state. A defect that fails but has a band state that C3 carries on is kept, pending (with `distort_charges : stable` or `screened`, any band state). |
| `require_spin` | — | C3 | Keep only defects with this final spin (unpaired electrons, within 0.25) in a stable charge state, e.g. `2` for a triplet. A defect that fails but has a band state that C3 carries on is kept, pending (with `distort_charges : stable` or `screened`, any band state). |
| `require_bound` | `false` | C3 | Keep only pairs that are bound against their components somewhere in the gap: E_b = Σ E_f(parts) − E_f(pair) > 0, from their bound charge states. The components must be candidates too. A pair that fails but has a band state that C3 carries on is kept, pending (with `distort_charges : stable` or `screened`, any band state). |
| `distort_charges` | `relevant` | C3 | The charge states each survivor carries on to the distortions. `relevant`: its stable ones (the lowest in formation energy somewhere in the gap), its other bound ones within `charge_margin` of its lowest somewhere in the gap, and a band state one charge beyond the one stable at a band edge (one more electron than at the top of the gap, one more hole than at the bottom; any band state of a defect with none bound) if a Γ single point shows its carrier shallow (`carrier_reach`) and on the defect (`carrier_fraction`). `stable`: the stable ones and every band state. `screened`: every one with a usable energy. The last two make no single points. |
| `charge_margin` | `0.5 eV` | C3 | With `relevant`, also carry on a bound charge state whose formation energy comes within this of the defect's lowest somewhere in the gap: a distortion may lower it that much and make it stable. Set it from what the distortions gain (C5's gain column; CaO:Sb's neutral state gained 0.34 eV); `0` carries on the stable ones alone. |
| `carrier_reach` | `1.0 eV` | C3 | With `relevant`, a band state is carried on only if its carriers lie at most this far inside their band in its single point: above the CBM for electrons, below the VBM for holes. Deeper, it is a free carrier that no distortion of the defect is likely to bind. |
| `carrier_fraction` | `0.2` | C3 | With `relevant`, a band state is carried on only if its carriers are spread over at most this fraction of the cell's atoms (their participation number, from the single point's band weights, as C7 counts it): on the defect, not through the cell. Greater than 0, at most 1. |
| `distortions` | ShakeNBreak's, −0.6 to 0.6 in steps of 0.1 (*list*); `-0.4 -0.2 0.2 0.4` with `mace : false` | C4, C5 | ShakeNBreak bond distortions of the nearest neighbours of the point the defect is distorted around: `-0.4` brings them 40 % closer, `0.4` takes them 40 % further. One MACE relaxation per factor, point, charge state and survivor (C4), cheap beside CASTEP's. Without MACE, one CASTEP relaxation each (C5): there the default's four found the lowest structure of every charge state of CaO:Sb that the full set found, and half the ±0.6 runs failed or didn't finish relaxing. |
| `distort_starts` | `2` | C4, C5 | How many of the lowest distinct minima of a defect's MACE relaxations (pooled over its charge states) C5 relaxes in CASTEP, in every charge state, beside the start: each costs one relaxation in the electronic cell per charge state. At least 1. |
| `distort_around` | `auto` | C4 | The point the distortions are made around: `auto` (a single defect's site; the centre of a complex whose components are nearest neighbours, within 1.25 host bond lengths of each other; each component in turn for one more spread out), `centre` (the mean of the components' positions, or an atom within 0.5 Å of it), `each` (every component in turn), `anchor` (the first component's atom, or its vacant site) or `vacancy` (the vacant site, or the anchor atom for a defect without one). A single defect always uses its own site. |
| `distort_cell` | `electronic` | C4 to C6 | The cell everything after C3 works in: `electronic` carries each survivor's screened structures into the electronic cell (B4) and does C4–C5's distortions and relaxations and C6's sharing there (C7 to F are there anyway), so a carrier that the screening cell leaves in a band can bind and drive a distortion; C1–C3 stay in the screening cell, where screening every candidate costs least. `screening` keeps C4–C6 in the screening cell. The same when B4 chose the screening cell itself. |
| `%block oxidation_states` | guessed | C4 | One `El  value` per line, for ShakeNBreak's count of neighbours to distort: n = q − Σ (oxidation state put there − that of the site), and that many neighbours (the absolute value), or 8 minus it above 4. Without the block, pymatgen guesses the host's, and each dopant takes its common oxidation state closest to that of the element it replaces. |
| `rattle_stdev` | 10 % of the host bond | C4 | The standard deviation (Å) of the random displacements of the other atoms. |
| `rattle_dmin` | 80 % of the host bond | C4 | The shortest distance (Å) between atoms that rattling may produce. |
| `distortion_min_gain` | `0.05 eV` | C5, C6 | A structure (a MACE minimum, or a distortion without MACE) counts as found when it beats the screened structure by this: in C5's ranking (its verdict), and in C6, which then tries it in the defect's other charge states. |
| `share_rounds` | `2` | C6 | How many rounds of trying a winning structure in the other charge states of its defect. |
| `keep_structures` | `1` | C6 | How many distinct structures, lowest first, of each defect and charge state go on to C7, which relaxes them all and takes the lowest as the ground state. |
| `excite` | `true` | C3, C7 | Run C7. Only the defects with an excitation that passes, and every native defect, go on to phase D; native defects also skip C3's property filters. With `false`, everything C6 kept goes on, and E4 has no excitations for phase F. |
| `excite_spin` | `2` | C7 | The ground-state spin C7 excites (CASTEP's spin, unpaired electrons: 2 is a triplet), or `any`. |
| `spin_tol` | `0.3` | C7 | How far the final spin may be from `excite_spin`. |
| `local_fraction` | `0.2` | C7, D2 | A band is localised on the defect when its participation number is at most this share of the cell's atoms. |
| `excite_reach` | `1.0 eV` | C7, D2 | How far inside a band a localised level still counts as a defect level (a resonance). |
| `excite_min` | `0.4 eV` | C7, D2 | The smallest ZPL (or vertical energy) that passes. |
| `excite_window` | — | C7, D2 | `lowest highest` in eV: the ZPL (or the vertical energy, with `excite_relax : false`) must lie inside. Without it, any converged excited state of at least `excite_min` passes. |
| `excite_min_dw` | — | C7, D2 | The smallest 1D Debye–Waller estimate that passes (e.g. `0.05`). |
| `excite_relax` | `true` | C7, D2 | Relax the excited state (the ZPL, ΔQ and the 1D estimates); `false`: the vertical energy only, for a CASTEP build whose ΔSCF has no excited-state forces (F1 then has no excited structure to draw a path to). |
| `%block excite_overrides` | — | C7, D2, F1 | Any other `.param` keywords for the ΔSCF runs and their ground states, after `screen_overrides` (C7) or `production_overrides` (D2, F1) and `metals_method : dm`, e.g. `deltascf_smearing : 0.01` for degenerate levels or more `nextra_bands`. |
| `validate` | `false` | C8 | Run C8: the finite-size test of the screening cell (its estimates, at the production settings, against the next sizes up). Nothing later uses its results. |
| `validate_defects` | every defect C6 kept | C8 | The defects to test, by B1's id (e.g. `d001 d002`). Two or three are enough: each needs relaxations in cells among the ladder's largest. |
| `validate_sizes` | `3` | C8 | How many sizes of the ladder above the screening cell to run; the reference line E_inf + b/N goes through them. |
| `validate_tol` | `0.05 eV` | C8 | An estimate passes for a defect when its error in every charge state's E_D − E_host + E_corr, and in every transition level between them, is within this. |
| `validate_gamma` | `true` | C8 | Also a Γ-only single point at the production cut-off of every structure C8 relaxed, against its k-point energy. |
| `%block elastic_constants` | — | C8 | The host's elastic constants, in GPa and `host_cell`'s Cartesian axes: one line `C11 C12 C44` (a cubic host) or six lines of six (the Voigt stiffness). C8's stress estimate needs it; your functional's own values are best. |

#### D: production

Details: [Phase D](#phase-d-production-ground-and-excited-states).

| Keyword | Default | Steps | What it does |
|---|---|---|---|
| `%block production_overrides` | — | C8, D1, D2, F1, F2 | `.param` keywords for the production runs (the relaxations, the single points, the ΔSCF runs), e.g. `geom_energy_tol`, `geom_force_tol`, `geom_max_iter`, `elec_energy_tol`. A relaxation that reaches `geom_max_iter` is flagged and left out of phase E, so raise it here. States whose spin is 0 run without spin polarisation unless this block sets `spin_polarized`. |
| `band_spread_tol` | `0.05 eV` | D2 | The most each band of an excitation may spread across the production k-points for its ΔSCF to run there; otherwise it runs at Γ. |
| `freeze_radius` | `0` (off) | D1 | An untested idea: hold every atom further than this (Å) from all of the defect's components fixed (`%block ionic_constraints`). Compare with a full relaxation before relying on it. |

#### E: thermodynamics

Details: [Phase E](#phase-e-thermodynamics).

| Keyword | Default | Steps | What it does |
|---|---|---|---|
| `charge_correction` | `lz` | C3, C8, E2 | The image-charge correction for charged defects, with ε₀ from A2.3: `lz` (Lany–Zunger), `mp` (first-order Makov–Payne) or `none`. Used for the screening hull, C8's sizes and the final formation energies. |
| `gap` | from A2.2 (C3: the screening cell's) | C3, E3, E4 | The range of Fermi levels C3 judges the hull and the stable charge states over, and so which charge states it carries on. The band gap for E3 and E4: the range of Fermi levels, the window in which transition levels are found, E_f at the CBM, and the free carriers (the conduction bands are shifted by the difference). Put an experimental or hybrid value here; the defect energies themselves are not corrected. |
| `fermi_temperature` | — | E3 | Find the self-consistent Fermi level, and the defect and carrier concentrations, at this temperature (K), at each chemical-potential limit, for each set of dopants on its own (the native defects alone, each dopant with them, and each mix of dopants one defect brings). Only the defects that reached phase D are included. |
| `quench_temperature` | — | E3 | With `fermi_temperature`: freeze each defect's total concentration at `fermi_temperature`, then share it out between its charge states at this temperature (K). |
| `deep_level_margin` | `0.5 eV` | B3, E4 | A transition level at least this far from both band edges counts as deep (E4 lists them). B3's Γ-only floor is for a level this far inside the gap. |

#### F: lineshapes

Details: [Phase F](#phase-f-optical-lineshapes). F2's model of each defect is trained with the
`model_*` settings; how its trainings and phonon calculations run (`model_jobscript`,
`model_command`, `model_device`) goes in `NAME.run`.

| Keyword | Default | Steps | What it does |
|---|---|---|---|
| `ccd_points` | `-0.4 -0.2 … 1.4` (*list*) | F1 | Where F1's single points go along the line from the relaxed ground state (0) to the relaxed excited state (1); at least three. |
| `lineshape_temperature` | `10 K` | F1, F2 | The lineshapes' temperature (the phonons' Bose occupation; 0 for none). |
| `lineshape_sigma` | `0.006 eV` | F1, F2 | The Gaussian width of each phonon line (the broadening of the spectral function). |
| `lineshape_gamma` | `0.002 eV` | F1, F2 | The Lorentzian half-width of the ZPL. |
| `lineshape` | `false` | F2 | Run F2, the multi-mode lineshape. |
| `lineshape_displacements` | `30` | F2 | Displaced copies of each charge state's ground state in the model's data (single points). |
| `lineshape_amplitude` | `0.02 ang` | F2 | Their displacements' standard deviation, per Cartesian component. |
| `lineshape_host` | `5` | F2 | Displaced copies of the pristine electronic cell in the data. |
| `lineshape_length` | `30 ang` | F2 | The big cell's shortest image distance (the electronic cell if that is longer). |
| `lineshape_method` | `forces` | F2 | `forces`: the phonons projected onto the excited state's forces at the ground state's structure; `displacements`: onto the displacement between the two relaxed states. |
| `lineshape_delta` | `0.01 ang` | F2 | The finite-difference step of the force constants. |
| `model_command` | `mace_run_train` | F2 | In `NAME.run`. The training command; the workflow adds the heads, the charge embedding and the settings below. |
| `model_jobscript` | — | F2 | In `NAME.run`. A Slurm template (a file, or `%block model_jobscript`) with a `{COMMAND}` line, for running each training (and each of F2's phonon calculations) as a job, best on a GPU ([section 9](#8-the-model-jobscript-template-model_jobscript)). Without it they run where `dwf.py` runs. |
| `model_device` | `mace_device` | F2 | In `NAME.run`. Where the trainings and F2's phonons run: `cpu` or `cuda`. |
| `model_epochs` | `300` | F2 | `mace_run_train`'s `--max_num_epochs`. |
| `model_r_max`, `model_channels`, `model_max_L`, `model_batch` | `5 ang`, `64`, `1`, `4` | F2 | The model's cut-off radius, channels, symmetry order and batch size. |
| `model_train_args` | — | F2 | Anything else for `mace_run_train` (e.g. `--swa`). |

#### Runner settings

These are the job runner's keywords for how the CASTEP jobs run (see `README_runner.md` for the
details). They go in `NAME.run`, where they apply to every
CASTEP step. A runner block changes them for some steps: `%block runner_host` for every CASTEP step
of phase A and the competing phases (A3.1, E1.1), `%block runner_screen` for C2, C3, C5, C6 and C7,
`%block runner_production` for C8, D1, D2, F1 and F2, and `%block runner_converge`, `runner_relax`,
`runner_bands`, `runner_dielectric`, `runner_ref_phases`, `runner_phases`, `runner_filter`,
`runner_drelax`, `runner_rank`, `runner_excite`, `runner_validate`, `runner_relaxations`,
`runner_zpl`, `runner_ccd` or `runner_lineshape` for one step ([How a run works](#how-a-run-works)
has the table). They hold runner keywords only, and may hold a `%block partitions`. The MACE and
model jobs use only `submit_cmd`, `status_cmd` and `poll` from here, and `cores_per_node` for the
default `mace_pack` ([MACE](#mace) has their own settings, and B2's, C1's and C4's blocks,
`runner_scan`, `runner_prerelax` and `runner_distort`, their own `max_walltime`).

| Keyword | Default | What it does |
|---|---|---|
| `jobscript` | required | Your jobscript template, as a file or as `%block jobscript`. It needs `{SEED}` where the seedname goes, and `{NCORES}`, or with `cores_per_node` `{NODES}`, `{TASKS_PER_NODE}` and `{CPUS_PER_TASK}` (with `charging : core`, `{CPUS_PER_TASK}` is required). Other placeholders: `{JOBNAME}`, `{PARTITION}`, `{NKPTS}`, `{MEM_GB}`, `{MEM_MB}`, `{MEM_PER_NODE_GB}`, `{MEM_PER_CORE_GB}`, `{MEM_PER_CORE_MB}`. The runner sets the partition and `--time` itself; the template's `--time` is the longest any job may run, unless `max_walltime` is set. |
| `cores_per_node` | — | Cores per node (288 on Cirrus, 128 on ARCHER2). Jobs are then laid out as nodes × processes per node, chosen for k-point parallelism. |
| `charging` | `node` | How the machine charges, with `cores_per_node`: `node` (whole nodes, never shared: ARCHER2), where every job fills its nodes, up to `max_nodes`; or `core` (shared nodes, charged by the core: Cirrus), where a job gets up to `max_cores` processes and asks for just the cores it needs on one node (more cores per process where the memory needs them), or whole nodes with `--exclusive` once it needs more than a node ([How a run works](#how-a-run-works)). |
| `max_nodes` | `1` | The most nodes a job may use (with `cores_per_node`). With `charging : core`, by default as many as `max_cores` needs. |
| `max_cores` | `32` | The most MPI processes a job may use, without `cores_per_node` or with `charging : core`. |
| `cores` | — | An exact core count for every job. |
| `%block partitions` | — | One partition per line: `name  GB_per_core  [max_cores]  [#SBATCH options]`, e.g. `highmem  5.33` on Cirrus or `highmem  4  --qos=highmem` on ARCHER2. A job goes on the first partition where it fits in memory with full nodes (with `charging : core`, with one core per process); if none fits, on the best layout with fewer processes per node (with `charging : core`, where more cores per process take the fewest cores). One inside a runner block replaces the whole list for those steps. |
| `mem_per_core` | — | Without `%block partitions`: the memory per core (GB) jobs must fit in; a job that can't is warned about (with `charging : core`, it gets more cores per process). |
| `mem_safety` | `1.05` | Factor on the memory estimated from the dry run. |
| `mem_replicated` | `auto` | The fraction of the dry run's memory that every process keeps its own copy of. `auto` starts at 0.25 and is measured from each finished job. |
| `baseline_mb` | — | Replaces CASTEP's baseline memory estimate (MB). |
| `omp_threads` | `1` | `auto` or a number: OpenMP threads per process on cores left idle when nodes are underpopulated for memory (or, with `charging : core`, on the extra cores a process gets for its memory). |
| `max_walltime` | the template's `--time` | The longest a job may run (`H:MM:SS`, or `D-HH:MM:SS`), instead of the template's `--time` (or a partition line's `--time=`). The first jobs of a step ask for this; with `walltime : auto` later ones ask for less once finished jobs show what they need. A relaxation stops a little before it and carries on in a new job; a single point can't, so give single points enough. |
| `walltime` | `auto` | `auto`: once jobs have finished, later ones ask for what they need (scaled from the measured time per SCF cycle) instead of the longest walltime; `fixed`: always the longest. |
| `walltime_safety` | `1.5` | Factor on the walltime estimate. |
| `sp_walltime` | — | The walltime (`H:MM:SS`) of the first single points, before there is anything to estimate from, if shorter than the longest. One that runs out is resubmitted with the longest. |
| `max_restarts` | `10` (`0` for C2, C3, C5, C6, C7) | How many times a job that runs out of time, or stops in a way the runner can recover from, is resubmitted; a relaxation carries on from its `.check`. |
| `delete_check` | `false` | Delete each calculation's `.check` file once it has finished; they can be large. |
| `dryrun_on` | `compute` | Where the dry runs go (one per distinct input, for the k-points and memory): `compute`, as small batch jobs from your template; `login`, where `dwf.py` runs, with `dryrun_cmd`. On ARCHER2, don't send them to the shared `serial` partition with a whole-node template, which it rejects: use `login`, or `dryrun_partition : standard`. |
| `dryrun_cmd` | `castep.mpi --dryrun {seed}` | The command for `dryrun_on : login`, e.g. `castep.serial --dryrun {seed}`. |
| `dryrun_timeout` | `900 s` | How long a login dry run may take. |
| `dryrun_jobscript` | the jobscript | A separate template for dry-run jobs. |
| `dryrun_partition`, `dryrun_time`, `dryrun_options` | — | The partition, time and other `#SBATCH` options of dry-run jobs. |
| `dryrun_cores`, `dryrun_mem_gb` | `1`, `4` | The processes and memory (GB) of a dry-run job (with `charging : core`, it asks for the cores that hold that memory). |
| `dryrun_pack` | `1` | With `dryrun_on : compute`: run up to N of a step's dry runs in one job, one process each (with `charging : core`, each with the cores for `dryrun_mem_gb`), side by side as many as a node holds at once (its cores, and its memory at `dryrun_mem_gb` each), the rest in turn on each core; `dryrun_time` must allow for them ([How a run works](#how-a-run-works)). Needs a jobscript (or `dryrun_jobscript`) whose CASTEP line uses `srun`. |
| `submit_cmd` | `sbatch` | The command that submits a job. |
| `status_cmd` | `squeue` or `qstat` | The command that lists queued and running jobs. |
| `poll` | `120 s` | Seconds between checks of the queue. |
| `max_jobs` | no limit | The most jobs, and dry runs, of one step in the queue at once (with `array`, each task of an array counts; with `pack` too, each task is a node of N jobs). |
| `tolerate_failures` | `false` (`true` for C2, C3, C5, C6, C7, C8, D1, D2, F1 and F2) | A job that fails is marked as failed and the others carry on, instead of the run stopping. `--resubmit-failed` starts failed jobs again. |
| `pack` | `1` | Run up to N of a step's jobs side by side on one node, in one job: with `charging : node`, each on `cores_per_node`/N cores and its share of the node's memory; with `charging : core`, each with its own cores, the pack asking for them all, as many as one node holds ([How a run works](#how-a-run-works)). With `array`, the packs go as job arrays, one pack a task. Needs `cores_per_node` and a jobscript whose CASTEP line uses `srun`. |
| `array` | `1` | With Slurm: send a step's jobs as job arrays of up to N tasks instead of one `sbatch` each, the jobs planned alike in one array (others in arrays of their own). `max_jobs` still counts every task, and the queue is topped up with a new array once `array_min` places are free; the queue is read with one `squeue -r` a poll. For steps of thousands of jobs, such as C2 of a big screen; keep N within the site's array limit. With `pack`, each task is a pack, a node running N jobs side by side ([How a run works](#how-a-run-works)). |
| `array_min` | a quarter of `array` (or of `max_jobs`, if smaller) | With `array`: how many places in the queue (of `max_jobs`) must be free before another array goes, so the fewest tasks such an array has (unless fewer jobs are left). Bigger: fewer, bigger arrays, but the queue runs lower meanwhile. Capped at `array` and `max_jobs` (so a step whose own block lowers `max_jobs` still tops up); ignored without `array`. |

`--prepare-only`, `--no-wait`, `--resubmit-failed` and `--force` are command-line switches
([Command line](#command-line)), not project-file keywords.

**Retired keywords.** A project file that still has one stops with what to do instead (`--split`
leaves D1–D5's out): `model`, `model_hybrid`, `model_hybrid_points`, `model_hybrid_fraction`,
`hybrid_param`, `hybrid_cell`, `model_rattles`, `model_rattle_stdev`, `model_force_tol`,
`model_rounds`, `search_distortions`, `search_rattles`, `search_fmax`, `search_steps`, `search_keep`
and `%block runner_hybrid`, `runner_dataset` and `runner_check` went with D1–D5, the ground-state
model of each defect (C6 ranks C5's relaxations, sharing distortions between charge states);
`series_defect`, `series_charge`, `series_sizes`, `series_choice` and `%block runner_series` went
with the CASTEP size series of one defect (B2 now scans every candidate); `series_tol` and `%block
runner_cells` with E0's size series of each defect (every defect now runs in the electronic cell,
B4, and C8 tests the finite-size error on a few); `finetune`, `finetune_args` and `%block
runner_finetune` with the old D2 (F2 trains its own model on the production runs); `scan_sizes` is
now `scan_lengths`, in Å; `finetuned_model` went (B2 and C1 use `mace_model`); `dispersion_check`,
`dispersion_tol`, `dispersion_defect` and `%block runner_dispersion` went with C2b (the screening
goes on with its Γ-point results), and `size_check`, `size_check_window`, `size_check_max`,
`size_check_defects`, `size_check_cell` and `%block runner_sizecheck` with C2c; `triage`,
`triage_window` (now `configuration_margin`) and `zpl_window` (C7 filters on the ΔSCF ZPL) went
earlier.

### 2. The production `.param` (`param`)

Your own CASTEP `.param`. A1 copies it into `A_host/production.param`, and every later job
starts from that file. Any keyword not in this table (`xc_functional`, `elec_energy_tol`,
`smearing_width`, mixing, geometry tolerances, ...) is passed on to every job unchanged.

| Keyword | What the workflow does with it |
|---|---|
| `task` | Set for every job: `geometryoptimization` for relaxations, `singlepoint` for single points, `dielectric_task` for the A2.3 run. Yours is ignored. |
| `cut_off_energy` | A1 sets it on each convergence run and writes the converged value into `production.param`, replacing yours. With `converge : false`, yours is kept. The screening runs (C2, C3, C5, C6, C7) and A3.1's phases use `screen_cutoff` if it is set. |
| `basis_precision` | Removed by A1 when it sets `cut_off_energy`. |
| `charge` | Set per job, to the charge state; removed from `production.param` and from the host, competing-phase and pristine-supercell jobs. Leave it out. |
| `spin` | The starting total spin (unpaired electrons), set per defect job: from `screen_spins` in C2; from the final spin of the screening run in C5, C6, C7 and C8; from C7's relaxed ground state in D1, and D1's in D2, F1 and F2 (for a ΔSCF run, of its ground-state run). Removed from `production.param` and from the host and phase jobs (unless a phase line sets it). A starting value only: with `fix_occupancy : false`, CASTEP holds it for `spin_fix` SCF cycles and then lets it relax. |
| `spin_polarized` (or `spin_polarised`) | Either spelling works: a job that sets it replaces yours, whichever spelling you used. `true` on every screening run (C2, C5, C6, C7), and on C8, D1, D2, F1 and F2 runs with a non-zero spin; `false` for their spin-0 states unless `%block production_overrides` sets it, and for the A2.3 run without `dielectric_param`. The host and phase jobs use yours. |
| `fix_occupancy` | `true` for the A2.3 run without `dielectric_param`, `false` for `metal` phases; otherwise yours. |
| `geom_max_iter` | `screen_max_steps` on the screening relaxations; yours, or `%block production_overrides`, elsewhere. A production relaxation that reaches it is flagged and left out of phase E. |
| `write_cell_structure` | Set to `true` on relaxations: their final structure is read from `<seed>-out.cell`. |
| `run_time` | Set on relaxations, a little below the job's walltime, so CASTEP stops cleanly and writes its `.check` before the time limit (not on screening runs, which are not restarted). |
| `calculate_stress` | `true` on C8's runs (the stress tensor for its stress estimate); yours elsewhere. |
| `continuation` | Yours is removed. Set to `default` when a relaxation is resubmitted to carry on from its `.check` (which removes any `reuse`: CASTEP takes one or the other). |
| `reuse` | Yours is removed. The ΔSCF runs (C7, D2, F1) set it to their ground state's `.check`. |
| `metals_method` | `dm` on the ΔSCF runs and their ground states (C7, D2, F1; CASTEP's ΔSCF constraints need density mixing), unless `%block excite_overrides` sets it; yours elsewhere. |
| `calculate_deltascf`, `deltascf_method`, `deltascf_checkpoint`, `%block deltascf_constraints` | Set on the excited-state runs only (C7, D2, F1): `true`, `simple`, the ground state's `.check`, and the occupations of the two bands (the occupied one emptied, the empty one filled, in one spin channel; the band numbers, as CASTEP counts them in that channel, are the same at every k-point). |
| `pdos_calculate_weights` | `true` on the single points whose bands' weights on the atoms C7 and D2 read (their ground states), and on their ΔSCF runs (is the promoted electron still on the defect?): the `.pdos_weights` file the participation numbers come from. |

`%block screen_overrides` and `%block production_overrides` (in `NAME.parameters`) set further
keywords for the screening and production runs, and options on a `%block phases` line set them for
that phase.

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
structure is used. The settings come from `production.param` (with `screen_cutoff` and `%block
screen_overrides` in A3.1), the host `.cell`'s `species_pot` and other keywords, and the
line's options:

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
| `{NODES}`, `{TASKS_PER_NODE}`, `{CPUS_PER_TASK}` | the layout, with `cores_per_node`: nodes, processes per node, cores per process |
| `{JOBNAME}` | the job's name |
| `{NKPTS}` | the number of k-points |
| `{PARTITION}` | the partition chosen |
| `{MEM_GB}`, `{MEM_MB}` | the estimated total memory |
| `{MEM_PER_NODE_GB}` | the estimated memory per node |
| `{MEM_PER_CORE_GB}`, `{MEM_PER_CORE_MB}` | the estimated memory per process |

The runner also sets `#SBATCH --partition` (with the partition line's extra options) and
`--time`, replacing the template's in either spelling; the template's `--time` is the longest
any job may run, unless `max_walltime` is set. It sets `OMP_NUM_THREADS`, and with threads adds
`OMP_PLACES`, `SRUN_CPUS_PER_TASK` and `--cpus-per-task` if they are missing. With
`charging : core` it adds `#SBATCH --exclusive` to a job on whole nodes, and takes the template's
out of a job on part of a node. A dry-run job (`dryrun_on : compute`) is made from the same
template (or `dryrun_jobscript`), with the CASTEP line run as one process with `--dryrun`.

The example's template for Cirrus, `cirrus.sh` (with `charging : core`; Cirrus takes no `--mem`,
as the memory comes with the cores):

```bash
#!/bin/bash
#SBATCH --job-name={JOBNAME}
#SBATCH --time=24:00:00
#SBATCH --nodes={NODES}
#SBATCH --ntasks-per-node={TASKS_PER_NODE}
#SBATCH --cpus-per-task={CPUS_PER_TASK}
#SBATCH --account=<budget>
#SBATCH --partition=standard
#SBATCH --qos=standard

module load castep

export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

srun --distribution=block:block --hint=nomultithread castep.mpi {SEED}
```

`archer2.sh` is the same for ARCHER2 (`charging : node`): whole nodes, and
`module load castep/26.11`.

### 7. The MACE jobscript template (`mace_jobscript`)

A Slurm script for one MACE job: your usual header (account, partition, QoS, one node, a
`--time` long enough for the job's share of the relaxations), the modules and environment
mace-torch needs, and one line with `{COMMAND}`. On Cirrus (`cirrus_mace.sh`) the job asks for
the cores its workers use:

```bash
#!/bin/bash
#SBATCH --job-name={JOBNAME}
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={NCORES}
#SBATCH --account=<budget>
#SBATCH --partition=standard
#SBATCH --qos=standard
module load cray-python
{COMMAND}
```

On ARCHER2 (`archer2_mace.sh`), which hands out whole nodes, it asks for `--nodes=1` and
`--exclusive` instead.

**On GPUs**, where there are any, the MACE relaxations run far faster than on CPU cores. Give the
step's block `mace_device : cuda` and a GPU template, a small `mace_pack` (about 4, each worker
holding its own cell on the GPU) and `mace_threads` (2): the workers share the job's one GPU
(`dwf.py` gives them no GPUs of their own, so ask for one GPU a job) and its queue. The example's
for young-ng's A100s, `young_ng_gpu.sh`, which F2's model jobs (`model_jobscript`) use too: the
header asks for `--cpus-per-task={NCORES}` and `--gres=gpu:1`, and `CaO_screen.run` has B2's
GPU block, commented out, beside its CPU one.

| Placeholder | Replaced by |
|---|---|
| `{COMMAND}` | its line becomes one line per worker, `mace_pack` of them, each in the background, then `wait`: `env OMP_NUM_THREADS=<mace_threads> <mace_python> <dwf.py> --mace-worker tasks.json > worker_NN.log 2>&1 &` (every worker the same queue, `tasks.json`). Anything in front of `{COMMAND}` on its line goes in front of each (e.g. `srun --exact --ntasks=1 --cpus-per-task=8 {COMMAND}` starts each worker with its own `srun`; without it the workers run on the job's node directly, which is enough for one node) |
| `{JOBNAME}` | `dwf_mace_<step>_job_NNN` |
| `{NCORES}` | the cores the job's workers use: workers × `mace_threads` (at most `mace_pack` × `mace_threads`) |

The job's folder (`<phase>/jobs/mace_<step>/job_NNN/`) holds `job.sh`, the job's queue
`tasks.json` (the structures and settings, biggest first), `claims/` (one file for each
relaxation a worker has taken: which node, process and time), each worker's log (a line for each
relaxation it finished, and a last line once nothing was left for it: how many it did, and how
long it worked), `progress.txt` (every worker's finished relaxations in one list, in the order
they finished, each with the count so far, under the count: `200/398 relaxations done (1 failed),
the last at 14:02:11`; `tail -f` follows it, and `grep -m1 done job_*/progress.txt` gives every
job's count), `slurm-<id>.out` and, once every worker has finished, `finished`. Each worker writes every relaxation's result to its file in `B_candidates/mace/` (C4's
in `C_screen/mace/`) as soon as it has it. A step's own block in `NAME.run` (`%block runner_scan`
for B2, `runner_prerelax` for C1, `runner_distort` for C4) can give its jobs a template of their own
(`mace_jobscript`) and a `max_walltime`, which becomes their `--time` in place of the template's
([MACE](#mace)).

### 8. The model jobscript template (`model_jobscript`)

A Slurm script for one model job: one of F2's trainings or phonon calculations, best on one GPU
(`model_device : cuda`): your usual header, the environment mace-torch needs (with CUDA), and one
line with `{COMMAND}`. The example's for Cirrus, `cirrus_gpu.sh` (check the
partition, QoS and GPU request against the Cirrus documentation):

```bash
#!/bin/bash
#SBATCH --job-name={JOBNAME}
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --account=<budget>
#SBATCH --partition=gpu
#SBATCH --qos=gpu
# load the Python environment with mace-torch here
{COMMAND}
```

| Placeholder | Replaced by |
|---|---|
| `{COMMAND}` | `bash -c 'cd <folder> && <command> > <log> 2>&1'`, with anything in front of `{COMMAND}` on its line in front of it: the training (the command in the folder's `train.sh`, log `train.log`) or `env OMP_NUM_THREADS=<mace_threads> <mace_python> <dwf.py> --phonon-worker spec.json` (log `phonons.log`) |
| `{JOBNAME}` | `dwf_model_<defect>` or `dwf_phonons_<excitation>` |
| `{NCORES}` | `mace_threads` |

Each job's folder holds `job.sh`, `job.json` (its Slurm id) and its log.

### 9. The job runner on its own: `.conv` and batch files

Outside the workflow, the job runner reads its own input files: `python3 dwf.py --conv NAME` (a
size series, `NAME.conv`; `python3 dwf.py --conv-example` writes an annotated one) and
`python3 dwf.py --batch FILE`; the full description is in `README_runner.md`. Its runner keywords are those under
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
| `energy` | `auto` | `auto`, `final`, `free` or `0K`, as in `dwf.py`'s `NAME.parameters`. |
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

The entries below are a log, oldest first, and each uses the step labels of its time; the last
says what they are now.

| Test project | What it covered |
|---|---|
| `M1`, `M2` | A to F with the MACE cell strategy: MACE jobs through the mock Slurm (`mace_jobscript`, `mace_pack`, `mace_max_jobs`), doped's compact cells (16 to 144 atoms), the scan of every candidate, the Γ-only floor from the band masses, C1 reusing B2's relaxations, E0 converging at the size below the screening cell (`M1`) and going a size up to the cap (`M2`, a tighter `series_tol`), `keep_structures : 2` carried into another cell, E2 |
| `M3`, `M4` | MACE jobs that end without their relaxations (resubmitted, then a stop after three tries); `--no-wait` with the jobs running in the background; a floor beyond the ladder |
| `Final` | A to F with MACE: convergence, DFPT, competing phases and limits, generated candidates, `scan : false` with `supercell : 2`, C1's MACE run here (no `mace_jobscript`), `configuration_margin`, `require_gap_states`, `hull_margin`, distortions and sharing, E2's fine-tuning data, Fermi level with a quench |
| `Opts2` | A to F with `mace : false`, a `transform` (cubic cells), antisites, native and dopant pairs, `screen_cutoff`, `%block screen_overrides`, `require_spin`, `require_bound`, `distort_charges : screened`, `distort_around : vacancy`, `%block oxidation_states`, `keep_structures : 2`, `%block production_overrides`, `freeze_radius`, `charge_correction : mp`, `gap`, `fermi_temperature` |
| `User` | A to F with Voronoi interstitials (`interstitials : dopants`), `supercell : 2` below the floor (warned), a quench |
| `Opts` | phase A with `converge : false`, `kspacing`, `dielectric`, `%block chemical_potentials` |
| `Async` | `--no-wait`, run again until the steps finish |
| `ExciteA` | `Charges` with D7: relaxed excited states, `excite_window`, `delete_check : true`, a restart of an excited state, `%block runner_excite` |
| `ExciteB` | `Charges` with D7: vertical energies (`excite_relax : false`), no window, a failed excited-state run |
| `ExciteC` | `Opts2` with D7: native defects past `require_spin` and `require_bound`, `keep_structures : 2` |

`dwf.py` replaces five scripts (one per phase) that passed results on through more files. The
`Final`, `Opts2` and `User` projects were run from scratch with both, and give the same results at
every step: the energies, spins and gap levels of every run, the survivors, distortions and
rankings, the production energies, the formation energies, levels and Fermi levels. Also tested
with `dwf.py`: carrying on after it was killed part-way through a step; a changed setting
(`require_spin`), which re-ran C3 onwards and reused the finished jobs; `--redo`; running single
steps and phases; `--until`; `--status`; and the example folder's phase A, with dry runs on the
login node.

Later the screening was simplified: C2 starts each charge state from one spin (`screen_spins :
triplet`, a doublet for an odd electron count) instead of two, C2b (the Γ-only check) was removed,
and rank and excite were renumbered to leave room for the planned model (C11 and C12 then, D6 and
D7 since phase D was split out of C). In `Charges`, `Opts2`, `User`, `ExciteA`, `ExciteB`,
`ExciteC` and `Final`, C2 ran half as many jobs, and every best screened state (energy and final
spin), C3's survivors and their charge states, the structures D6 kept and every production and
thermodynamics result (E and F) were the same as before. The mock ends a run in the same spin state
from either start, so the second start added nothing there; with real CASTEP, a run started as a
triplet ends as a singlet only if the spin is free to relax (see 3 below).

Since then, A1 converges k-point grids (from `converge_kgrid_min` to `_max`) instead of spacings,
and `%block defects` takes `q=`. Run again, the test projects choose the same 8 × 8 × 8 grid and
cut-off as before, and phases B to E give the same results. The production spacing worked out
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
steps and `pack : 2` for A3.1 and E1, and gave the same results as without it at every step,
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

Then D7 (`excite`). The mock CASTEP now does a simple ΔSCF: it needs `metals_method : dm` and
the `.check` it reuses, stops if `reuse` and `continuation` are both set, raises the energy by
the gap between the two constrained bands (less 0.1 eV, or 0.25 eV and one atom moved for a
relaxation), and can stop a chosen relaxation at its `run_time`. `ExciteA` (window 1.0–1.5 eV):
the Sb_Ca–V_O complex (ZPL 1.95 eV) was dropped and did not reach E, Sb_Ca passed through its
q = +1 state (1.25 eV), and V_O went on as a native defect; the ground states' `.check` files
survived `delete_check : true`, and the excited state stopped at `run_time` carried on from its
own `.check`, with `reuse` removed. `ExciteB` gave the vertical energies, and its failed
excited-state run kept its defect as `unknown`, flagged again in F3. `ExciteC` took all 14
native defects past `require_spin : 2` and `require_bound` to E (22 defects instead of 8).
Without `excite`, `Charges`, `Opts2`, `User` and `Final` submit the same jobs as before and give
the same results at every step (`User` differs only in the mock's job ids in its failure
messages).

Then the first ZPL estimate went: C2's Kohn–Sham estimate, its column in C3's table, step C3b
and `zpl_window` (a project file that still has it stops with a note to use D7).
`require_gap_states : both` still checks for an occupied and an empty gap level in one spin
channel. The four test projects (`User` and `Final` without their `zpl_window`, which had kept
every survivor anyway) submit the same jobs and give the same results as before, apart from the
estimate itself and F3: without `excite`, F3 now counts deep levels only, so `Charges`'s
Sb_Ca–V_O complex, picked before only for its estimate, no longer goes to G (the other projects'
picks are the same, for their deep levels). `ExciteA` and `ExciteB` give the same results as
before.

Then the MACE cell strategy replaced the CASTEP size series of one defect (B3) and the single
supercell: B2 scans every candidate with MACE (as jobs on compute nodes with `mace_jobscript`), B3
takes the worst case with the Γ-only floor, and E0 converges each defect's production cell. The mock
CASTEP now treats any supercell of the rock-salt lattice as a defect cell (not only cubic ones), has
parabolic band edges (2.0 and 1.2 m_e) for B3's masses, and spreads its defect levels across the
k-points by 10 exp(−L/2.5 Å) eV. B3 found the masses (2.02 and 1.21 m_e, also from the irreducible
k-points alone, and for a made-up band edge at X, whose three points were found and folded onto Γ
only by supercells that should fold them). In `Charges`, `Opts2`, `User`, `Final`, `ExciteA`,
`ExciteB` and `ExciteC`, run with the transform's cubic cells and the floor (or `supercell : 2`),
the screening cell was the one the size series had chosen before, and phases A to D give the same
results as before at every step (in `ExciteB` apart from its excited-state run, which failed on
purpose before); E and F differ, as each defect now has its own cell. In E0, relaxations of 512-atom
cells that ran out of time were flagged and their defect kept the biggest size whose runs had
worked.

The real MACE (medium-mpa-0) on the mock's CaO showed something to watch for: its relaxation
energy of the neutral O vacancy grew with the cell, 0.48, 0.79 and 0.92 eV in 54-, 102- and
144-atom cells, with the six Ca around it pushed out by 0.12 to 0.17 Å, as for a charged
vacancy. A foundation model has no electrons to keep in the vacancy, so it relaxes the ions as if
there were none, and that relaxation is long-ranged. With `scan_tol : 0.02 eV` no candidate
converged within 14 Å, and the screening cell went to the largest size; the edge displacements
were small (0.01 to 0.03 Å). On a real host, look at B2's table before phase C: if vacancies
never converge for this reason, a larger `scan_tol` (or a cap with smaller `scan_lengths`) is
reasonable: the production runs are in the electronic cell anyway, and D8 can measure the
screening cell's error with CASTEP.

Last, phase D was split out of phase C, and the runner learned to plan jobs on a machine that
charges by the core (Cirrus). The split renames steps and folders only (C11 and C12 became D6 and
D7, D0 to D2 became E0 to E2, E1 to E3 became F1 to F3; rank's and excite's files moved to
`D_states/`, and the production and thermodynamics folders became `E_production/` and `F_thermo/`).
Checked: `--status`, choosing steps and phases by label and letter, the example project, and
`--migrate` on a made-up work directory in the old layout (folders moved, the paths in every
`results.json` and in `state.json` changed, labels and summaries updated, and a step still running
refused). With `charging : core`, the plans were checked on made-up dry runs: one core per process
where the memory allows; where it doesn't, more cores per process or `highmem`, whichever takes
fewer cores; whole nodes with `--exclusive` beyond one node; the dry runs' cores. A batch went
through a mock Slurm with `cirrus.sh` (dry runs, plans, job scripts, results), and the MACE jobs'
scripts were written with `--prepare-only` (`{NCORES}` = workers × 8). `pack` and `dryrun_pack` with
`charging : core` went through the mock Slurm too: seven dry runs (2 cores each) in one job; four
jobs on `standard` in one job of 252 cores and two on `highmem` in one of 192 (one of them with 2
cores per process), while a fifth on `standard` ran on its own; each job with its own `srun --exact`
and `--mem-per-cpu` (the pack's `SLURM_MEM_PER_CPU` when Slurm sets it, the partition's GB per core
when not). With `charging : node` (ARCHER2) the runner plans exactly the same jobs as before (576
made-up dry runs over five sets of settings, `pack` included), and its packs' job scripts are
unchanged. The whole workflow has not been run against the mock since this split. The MACE jobs were
then checked through the mock Slurm on their own: with `dwf.py` run by a venv's Python reached
through a linked folder, the jobs got its `/work` path; with a Python the job couldn't find, the run
stopped after three jobs with the end of the worker's log (as did a job stopped at its time limit,
with Slurm's message), and `--resubmit-failed` gave the relaxations three more. B2's line test was
checked on the CaO:Sb scan (d001 converges at n144, 6 meV off the line at worst, and d002 at n102, 5
meV) and on EMT copper standing in for MACE, through B2 itself, with `scan_test : flat` giving the
same results as before. The band-state check was tested on `.bands`-shaped data built from
d001's −1 screening run (one electron left in the conduction band, found whether the per-spin
electron counts are the final ones or the starting ones), a bound −1 and a hole; and C3 on
made-up screening results like CaO:Sb's: the −1 in a band is carried on but no longer counts as
stable, with `gap` or without, and with `require_spin : 2` d001 is kept, pending. C4's `auto`
was checked with ShakeNBreak on Sb<sub>Ca</sub> + V<sub>O</sub> in the n144 cell: the
nearest-neighbour pair is distorted around its centre (−0.3 moves Sb 0.36 Å toward the vacancy),
the ⟨111⟩ pair around Sb and then the vacant site. D8 was run on made-up results for three
defects in the n144 cell and the sizes above it (n216 to n1000): energies on a known line
E_inf + b/N, and stress tensors written as CASTEP prints them, chosen so that each charge state's
stress energy is its real tail (or, for one state, isn't). It found E_inf and b, the stress
estimate passed where the stress held the tail and failed where it didn't, MACE's one c failed
where the charge states' tails differ, a failed run in the biggest size was left out, and a
band state was skipped; the stress energy matched ½ V p²/B for a hydrostatic stress and
½ V τ²/C44 for a shear. The whole workflow has not been run against the mock since.

Then the second version of phases D to G (this one): the electronic cell (B4), the model
(D1–D5) and D6's ranking of its minima, the new D7, D8's Γ check, the new E (E1, E2), F's Fermi
level per dopant set and F3's choice, and phase G. Each step was tested in four test scripts on
made-up CASTEP output written in CASTEP's formats (`.castep` geometry steps with forces, `.bands`,
`.pdos_weights` in either byte order), with a stand-in trainer (a command that writes a model
file) and a stand-in MACE (every atom pulled to its rock-salt site, an exactly harmonic model):
B4's folding of CaO's X points (the 144-atom cell folds one of three, the 320-atom one all three)
and its floor; D7's participation numbers, triplet filter, levels inside a band, ΔSCF jobs and
verdicts, and a ground state that slid away from its start; D1's filters and split, D2's picks,
D3's command (its heads and charge embedding read back), a D4 round that trains again, D5's search
(its head, charge and cell); E2's band-number check (a level that disperses falls back to Γ,
with a Γ ground state of its own); F2's dopant sets (native, Sb, Bi, Bi+Sb) and their Fermi
levels; F3's choice; G1's parabolas, which recover the phonon energies and Huang–Rhys factors of
known curves; the lineshape against Poisson's weights at 0 K and the Debye–Waller factor at
300 K; and G2 from end to end, with its phonon calculation run as a job of `dwf.py`, whose
Huang–Rhys factor matches the one worked out by hand for the stand-in model to 0.1 %, with ΔF and
with ΔR. The example project was read with every optional keyword on. The whole workflow has not
been run against the mock CASTEP since.

Then the cell split (`distort_cell`): C1–C3 in the screening cell, C4 on in the electronic cell.
Tested the same way on CaO:Sb (Sb<sub>Ca</sub> + V<sub>O</sub>, screening cell n144, electronic cell
n216 for speed): C4 carried both charge states into n216 (215 atoms, the Sb where B4's cell puts it)
and distorted them there; C5 ran those with one single point of the pristine n216 cell, and a level
8.45 eV above the VBM counted as in the band against n216's conduction-band edge and as a gap level
against n144's, as it should; D6 ranked the n216 runs against each charge state's carried-over run,
shared the −1 winner with the 0 state in n216 (reusing C5's band edges) and kept n216 structures;
with `excite : false` the lowest of each went on; D1 took C5's n216 steps (not C2's) and rattled
n216 cells, D2's hybrid single points and D4's relaxations and single points ran in n216, and D5
started from the lowest C5 structure there. With `distort_cell : screening`, or when the two cells
are the same, C4–C5, D1–D4 and D6 work in the screening cell as before.

Then `decay_length_multiple`, which was fixed at 4 before. Tested on CaO-like masses and gap: the
keyword is read (4 when it isn't given, 0 and −2 refused); the floors of B3 and B4 are that many of
the same decay lengths at 3, 3.5, 4 and 5, and `screen_min_image` still sets B3's; at 3, B3 and B4
pick smaller cells of the test ladder (n144 and n512, against n512 and n1000 at 4) and their tables
name the multiple; changing it runs B3 and B4 again. At the default, the floor's record and B4's
reason are what they were, so a work directory made before runs B3 and B4 again (no jobs) and
nothing after them.

Then D2's fixed budget, `model_hybrid_points`, which replaced `model_hybrid_fraction` (about 90 to
150 hybrid single points per defect with 12 distortions in three charge states). Tested on made-up
relaxations in three charge states: six points give two per charge state, its lowest structure and
its most distorted start; four give one each and the spare to the charge state with the most steps;
two still give one each; a bigger budget takes the first and last steps of the relaxations before
any other step; the rattled pristine cells are never picked, and two defects share one
pristine-cell job. In D3 the `hse` structures are repeated to a tenth of the `pbe` ones and
validated on themselves, without the repeats adding up in D4's next round, and G2's training
command is what it was. The stage-3 test (D1 to D5 on made-up CASTEP output) passes with four
points over two charge states.

Then C5's ranking. Tested on made-up relaxations in two charge states: a distortion 0.4 eV below
the start is a new minimum, one that relaxed back into the start's structure shares its letter, a
lower structure 0.03 eV down is not a new minimum, a failed run comes last with its lowest
converged step (its later, unconverged step left out), a run with an unconverged step says so, and
with `distort_cell : screening` the start is C2's screened run. The cell-split test runs C5 end to
end in both cells with it.

Then `results.dwf`. Tested on a made-up state: every step's status, the project file's keywords,
the phases that have run in order (none for one that hasn't), each step's table once, the file
written again rather than added to when a step finishes, and `--status` bringing it up to date; and
with `--status` on the example project, from the project file as read (its runner blocks on one
line each).

Then C3's charge states (`distort_charges : relevant`) and the four distortions. Tested on made-up
candidates, each in its own composition group, with stand-in single points (a `.bands` and a
`.pdos_weights` that put the carriers' bands where each case needs them): C3 makes one batch, in
`C_screen/jobs/filter`, of single points with the band weights, only of the band states one charge
beyond the state stable at a band edge (every band state of a candidate with none bound). A −1
whose electron is shallow and on six atoms is carried on, and so is a hole one beyond the state
stable at the VBM; a −1 spread over half the 63-atom cell, or 1.5 eV into the conduction band, is
left out, with its participation number, limit and depth recorded; one whose electron is in a gap
level in the single point, or whose single point failed, is carried on unjudged. A bound −1 0.30
eV above the lowest at the top of the gap is carried on with `charge_margin : 0.5` and not with
0.2, one 1.00 eV above is left out, and with `gap : 7.09 eV` both become stable above the PBE
conduction-band edge and go on. A band state next to no stable one is left out without a single
point; a candidate with no bound state is kept by a band state that passes, and dropped with none.
With `require_spin : 2`, a candidate whose −1 passes is kept, pending, and one whose −1 is spread
out is dropped. `stable` and `screened` make no single points and carry on what they did before.

Then `dopant_pairs : same`. Tested on CaO with Sb and N, vacancies, antisites, interstitials on the
Voronoi vertices and pairs up to 8.5 Å: `false` gives 336 candidates and no pair of two dopant
defects, `same` 400 (the 32 pairs of each dopant with itself, such as Sb_Ca + Sb_O, and no Sb +
N), `true` 452 (the 52 Sb + N pairs as well); never two interstitials; `true` and `false` still
read as before, so a project that gave them keeps its B1. On ADAQ's CaO set (33 dopants, the rest
as above), B1 makes 4,709 candidates in 511 composition groups in under a minute.

Then job arrays (`array`), for steps of thousands of jobs. Tested on a mock Slurm whose `sbatch`
runs an array's tasks one after another: seven single points planned alike with `array : 3` and
`max_jobs : 5` went as arrays of 3, 2 and 2 (`max_jobs` counting each task), each task running in
its own job's directory with its output there and its job id `<array job id>_<task>`; a group of one
went as a plain job; 8- and 64-atom jobs, planned on different numbers of processes, went in arrays
of their own; two relaxations of one array that stopped at `run_time` each carried on from its
`.check` in a job of its own; `array` without `sbatch` or at 0 is refused; `squeue -r`
is read once a poll; with `array_min : 3` the top-up of 2 waits, giving arrays of 3 and 3 and the
last job on its own. With `array : 1` nothing changes. The runner now keeps each job's
`job_info.json` in memory and doesn't read a finished job's output again: 1,000 mock jobs took about
110 s of the runner's own work before and 9 s after, and 10,000 about a minute.

Then arrays of packs (`array` with `pack`), on the same mock Slurm (40-core nodes, `pack : 4`):
nine single points went as one array of two tasks, each a node running four jobs side by side on
10 cores each (the template's `#SBATCH` lines for one whole node, a loop over the task's four lines
of `tasks.txt`, `--exact` and a quarter of three quarters of the node's memory each), with the ninth
on its own on the whole node; each job ran in its own folder, with its id `<array job id>_<task>`
and its output there. Ten jobs gave the array and a pack of the last two (half a node each); six,
too few for two full packs, two packs of three (a third each); twenty with `max_jobs : 3` arrays of
three and then two tasks. Relaxations of an array of packs that stopped at `run_time` carried on
from their `.check` in jobs of their own. The array's script ran under `dash` as well as `bash`.
With `charging : core`, four 8-core jobs went to a task that asked for their 32 cores of a shared
node, each job with `--mem-per-cpu`; without `mem_per_core`, 8- and 16-atom jobs on 10 cores each
shared one array, each asking for the most memory any needed. `pack` without `array` gave the same
job scripts as before, byte for byte, in eight cases (whole nodes and `charging : core`, OpenMP
threads, partitions with options, relaxations, `dryrun_pack`). 10,000 mock single points with
`pack : 40` went as one array of 250 tasks, with about a minute of the runner's own work, as before.

Then `dryrun_pack` above what a node holds at once, on the same mock Slurm (`dryrun_on : compute`,
40-core nodes of 4.5 GB a core). With `dryrun_mem_gb : 30` a node holds six dry runs at a time, so
fourteen different dry runs in one pack ran as six lanes of two or three in turn (the job asking
for six processes, each dry run for 30 GB), then the jobs ran; the lanes ran under `dash` too.
With `charging : core` (seven cores for 30 GB) they ran as five lanes on a shared node, each dry
run with `--mem-per-cpu`. A pack the node holds at once gives the same script as before (the eight
cases above, byte for byte). `dryrun_pack` above a node's cores is accepted; a dry run too big for
a node is refused.

Then phases C and D were merged, and the project file split in two. The ground-state model of
each defect (D1–D5: its training data, hybrid single points, training, checks and search) was
removed, and C6 now ranks C5's relaxations, sharing winning distortions between charge states as it
did with `model : false`. Rank, excite and validate (D6–D8) became C6–C8, in `C_screen/`, and
production, thermodynamics and lineshapes became phases D, E and F (`D_production/`, `E_thermo/`,
`F_lineshape/`). A project is now `NAME.parameters` and `NAME.run`, and B2 and C1 can give their
MACE jobs settings of their own (`%block runner_scan`, `runner_prerelax`). Tested: `--example NAME`
writes the pair (the example folder's), which reads back with each keyword in its own file; a runner
keyword, a MACE job keyword or a runner block in `.parameters`, or a science keyword in `.run`,
stops with where it goes, and D1–D5's keywords and blocks with a note; `NAME`, `NAME.parameters` and
`NAME.run` all name the project, a `.parameters` without its `.run` stops, and an old `NAME.dwf` is
told to split. `--split` on the old example kept every setting but D1–D5's (73 of 78), each in its
file, with the comments' labels and section headings as they are now (a phase letter alone only
after "to", "for", "in" and the like, so an energy E stays E); the pair runs `--status`, and
`--split` doesn't overwrite it. With `%block runner_scan`, `--prepare-only` wrote B2's 20 MACE
relaxations as 3 jobs (`mace_max_jobs : 3`) of 2 workers with 4 threads and `--time=6:00:00`, while
C1's kept the `.run` file's settings and the template's `--time`; a block that asks for more cores
than a node has, or holds a keyword it can't, stops. `--migrate` was tested on a work directory the
old `dwf.py` filled in step by step (A1 to G2, D1–D5 among them, made-up results naming files in
every phase folder): it is refused while C6 runs or with `D_production/` there already; then C6–C8's
jobs, structures and results move to `C_screen/` (D1–D5's stay in `D_states/`), the phase folders
are renamed, the paths in every `results.json` and in `state.json` change (and `for_G` to `for_F`),
D1–D5 leave `state.json`, and the labels and summaries are updated. Afterwards every step but C6 is
up to date; C6 run again with the same results leaves C7 up to date (with other results, not);
`--status` works; and `--migrate` again has nothing to do. The step tests pass as before, but those
of D1–D5, which went with them.

Then `phases_from : doped` with an older doped (a user's A3.1 stopped with `unexpected keyword
argument 'energy_above_hull'`). doped 3.1 renamed `e_above_hull` to `energy_above_hull`, took the
dopants into `CompetingPhases` (`extrinsic`) in place of `ExtrinsicCompetingPhases`, and moved the
band gap into the entries' summary; A3.1 now works with doped 2.x and 3.0 as well. Tested with
stand-ins for doped 2.4.7, 3.0.0 (current and legacy keys) and 3.1 on (3.2.1's naming), each with
that version's signatures and entries (from its source), for made-up phases of CaO with and
without Sb: each was called its own way, and all gave the same phases, molecules, metals and
names (two polymorphs named alike get `_2`), and the same CASTEP inputs from A3.1. When doped
fails, the error names its version and repeats doped's advice on what to install; with 2.x it
says to update doped (2.x needs the legacy API, which the Materials Project was closing in
September 2025), and a key in doped's message shows only its first four characters (2.x prints the
whole key). Then a user's doped 3.2.1 called the key invalid: doped 3 checks a key with a query
for `mp-1` and calls any failure of it an invalid key, whether the Materials Project refused the
key or the query never reached it. The error now adds the query's own error, where the key came
from (`mp_api_key` or `~/.pmgrc.yaml`), its first four characters and its length, and what that
points to: refused (HTTP 401 or 403), never checked (no connection), or not 32 characters
(pymatgen's "Invalid or old API key"). Tested with doped 3.2.1's own check, against a refusal
like the Materials Project's, a closed port and a quoted key, on Python 3.13 and on Python 3.10
with pymatgen 2025.10.7 (the newest for 3.10), where the whole suite passes too.

That user's next error showed the key worked: the Materials Project answered, but pymatgen
2025.10.7 couldn't read the answer. Its entries now come as `pymatgen.core.entries` (pymatgen
2026), with `entry_id` as a dict, which pymatgen reads from 2026.9.23 on, and that needs Python
3.11. Replayed with a stand-in Materials Project serving made-up replies in that format (as in the
user's error, emmet's `PotcarSpec` and r2SCAN entries included): Python 3.10 with pymatgen
2025.10.7 fails as the user's did, and the error now says that the key works and what pymatgen
it needs; a fresh Python 3.11 install (pymatgen 2026.10.24, monty 2026.10.21, doped 3.2.1)
passes doped's key check, has a wrong key refused (401), and A3.1 gets both Ca phases, the O2
molecule, CaO and CaO2. The whole suite passes on that install too. Still not run against the
Materials Project itself.

Then A3.1 sat for 5-10 minutes in doped's queries, with nothing in the log. pymatgen's
`get_entries`, which doped uses, asks for every material of the chemical system on every hull
the Materials Project has (GGA/GGA+U, r2SCAN and their mix), structures and all, then decodes it
all, corrects it all and throws most of it away, and it does so for the host's elements and again
for them with each dopant; nothing was kept, so every run asked again. `dwf.py` now makes those
queries itself (one request each: the GGA/GGA+U hull only, up to `phases_e_above_hull` + 0.1
eV/atom, the fields doped reads), keeps the replies and the phases doped chose, and logs each
query (documents, MB, seconds). Tested against a stand-in Materials Project that answers as the
real one does (thermo documents per hull type, with their entries in its current form; filters,
fields, both kinds of pages, a summary route, the key), with 25 made-up Ca-O-Sb materials on and
off the hull: doped through pymatgen's own queries chose 9 phases from 325 kB of thermo
documents, and through `dwf.py` the same 9 from 63 kB; a second run asked nothing; without
`phases_doped.json` only doped's key check went out; without dopants, the host's phases came
from the kept replies; a Materials Project that refuses the short query (400) gets pymatgen's own
queries, with the same phases; a wrong key still stops, saying so. On Python 3.11 and 3.13
(pymatgen 2026.10.24); not with the real Materials Project.

Then the same user's 33-dopant screen (CaO with H to Bi): the queries took 41 s (one request for
the host's systems and one for each dopant's, and a summary request with each), then the run never
moved again. Given all its dopants at once, doped ends by building a pymatgen phase diagram of
every element together, here 35, kept for reference only; its convex hull grows about twofold
with each element (made-up Ca-O-X compounds: 15 elements 1.6 s, 19 elements 8 s, 23 elements 76 s),
so 35 would take many hours. doped chooses each dopant's phases on their own anyway, from the hull
of the host's elements and that dopant. `dwf.py` now gives doped one dopant a call, asks for every
chemical system of every dopant in one request (doped's key check once a run), and keeps the
documents by chemical system and the summaries by material, so a dopant added later costs one
request for its systems. Tested on the stand-in Materials Project with Sr as a second dopant (and
an Sr-Sb oxide neither needs): doped given Sb and Sr at once, through pymatgen's own queries,
chose 12 phases from 7 requests and 433 kB of thermo documents, building phase diagrams of up to 4
elements; `dwf.py` chose the same 12 from 3 requests (the key check, every system, the summaries)
and 61 kB, with none above 3 elements, and never asked for Sr-Sb; Sb alone and no dopants then came
from the kept replies; a run with Sb, then one with Sb and Sr, asked only for Sr's four systems;
the fallback and a wrong key as before. The whole suite passes on Python 3.10, 3.11 and 3.13.

Then the competing phases were split three ways, after the same user's screen got 287 of them from
doped, to be relaxed at the production settings before any screening. A3.1 (`ref_phases`) and A3.2
(`ref_limits`) now run, at the screening settings, a single point of the host, the host's own phases
and each dopant's element (the lowest of each composition). C3 uses them to drop a defect more than
`site_margin` (0.5 eV) above the lowest defect with its dopant atoms everywhere in the host's
stability region and gap (the dopant's μ cancels; a linear program finds the closest point exactly),
and any defect whose least formation energy, the dopants at their elements, is above
`max_formation_energy` (2 eV). A new C4 (`dopant_phases`) relaxes the compounds of the dopants C3
kept, and cuts again with them. The competing phases at the production settings became E1.1 and
E1.2, for the dopants of D1's defects only, and without the host's own composition (relaxed in
another cell, it could come out a hair below the host and stop E1.2). C4 to C8 became C5 to C9, and
E1 to E3 became E2 to E4. Tested on made-up energies for CaO with Sb, Bi and Sr, checked by hand:
A3.1 chose the host, CaO₂, Ca, O₂, Sb and Bi (the lowest of each; not a second CaO, nor compounds)
at `screen_cutoff`; A3.2 gave the Ca-rich and O-rich corners with Sb and Bi at their elements (Sr
has none). C3 dropped Sb<sub>i</sub> + O<sub>i</sub> (always 0.80 eV above Sb<sub>Ca</sub>),
Sb<sub>i</sub> (1.25 eV above the Sb set's lowest at best, and 5 eV), V<sub>Ca</sub> (3 eV) and
Bi<sub>Ca</sub> (2.6 eV), and kept Sb<sub>Ca</sub> + O<sub>i</sub> (0.30 eV at best) and
Sr<sub>Ca</sub> (not judged); looser settings, and no limits, kept more. C4 relaxed Sb₂O₃, Sb₂O₅,
CaSb₂O₆ and SrO (not the higher Sb₂O₃, BiSbO₄, or Bi₂O₃: no Bi defect was left); with μ_Sb held 4.75
eV down by CaSb₂O₆ at the O-rich corner, it dropped Sb<sub>Ca</sub> (now 5.25 eV) and
Sb<sub>Ca</sub> + O<sub>i</sub> (5.55 eV), and Sr<sub>Ca</sub> through SrO (2.4 eV), and kept
Sb<sub>O</sub> and V<sub>O</sub>. E1.1 ran every phase of Ca, O and Sb but CaO's own, and E1.2's
limits had Sb alone. `--migrate` on the old work directory now also moves the old A3.1's jobs and
A3.1's and A3.2's results to `E_thermo/`: afterwards A3.1 can run, A3.2 and C3 to C7 wait for it,
E1.1 and E1.2 run again and every other step is up to date. A work directory from between the two
(merged, with those phases still in `A_host/` and their step running) has them moved by `--status`,
its record dropped with a warning to cancel the old jobs. The whole suite passes on Python 3.10,
3.11 and 3.13.

Then the same user's A3.1 at `screen_cutoff : 500 eV`: Be's and CaO₂'s cell relaxations never
converged. In Be's, the energy jumped by 0.18 eV and the pressure by 1.4 GPa each time c crossed
3.9177 Å, where CASTEP's FFT grid along c needs another point, and the zero-stress point lay in
the jump, so the optimiser reverted back and forth until its 30 steps ran out; its
`dEtot/dlog(Ecut)` (−5.4 eV for two atoms) and CaO₂'s, O₂'s and the host's (about −18 eV per O)
showed how far from converged Be and O are at 500 eV, while Ca and the heavier elements were
converged. Li ran out of steps without reverting. Every phase's relaxation now gets
`phases_max_steps` (100) geometry steps, and `fine_gmax` in `%block screen_overrides` gives the
screening runs and A3.1's and C4's phases a finer density grid alike (checklist 16).

Then `phases_molecule_box`, after the same user's dry run of O₂ in doped's 30 Å box was stopped
about 30 s in, twice, its `.castep` cut off in the pseudopotential report: with
`fine_gmax : 35 1/ang` that box has about 335 grid points a side instead of 192, and in a pack each
dry run has `dryrun_mem_gb` (4 GB). doped's molecules now go in 15 Å boxes. Tested with doped
3.2.1's own molecules: O₂ and Cl₂ come out in 15.01 × 15 × 14.99 Å boxes, centred, with doped's
bonds, options and names, from `phases_doped.json` as from doped; the solids are left alone;
`30 ang` gives doped's structures back exactly, 12 Å a 12.01 × 12 × 11.99 Å box, and 6.5 Å is
refused for Cl₂ (4.5 Å from its images), as is 0; a molecule split across the old box's edge comes
out whole; a `%block phases` molecule keeps its file's box. On a mock Slurm and CASTEP, with the dry
runs in a pack under `--mem=4096M`: O₂'s dry run in doped's box is killed and the run stops, as the
user's did; with the 15 Å box the next run sets O₂'s job up again (it had never run) and finishes,
and the solid keeps its dry run (checklist 17).

Then the host's own compounds moved from A3.1 to C4, as the same user asked: A3.1 now runs only the
defects' constituents (the host and the pure elements), so A3.2's limits are against the elements
alone, the widest region there can be; C3's cuts with them can only keep too much, and C4, with the
host's compounds (CaO₂) beside the dopants', makes both again, natives included. Tested on the
made-up CaO energies of the references' test: A3.1 runs Ca, O₂, Sb and Bi but not CaO₂; C4 runs CaO₂
with the compounds of Sb and Sr, and where CaO₂ lies above CaO and O₂ every result is as before.
With a CaO₂ that holds μ_O 0.3 eV below O₂'s at the O-rich corner, and a dopant (Ba) with its
element only, C3 keeps O_i (1.9 eV with O₂'s μ_O) and Ba_Ca + O_i (0.45 eV above Ba_Ca), and C4
drops them (2.2 eV; 0.75 eV above Ba_Ca), leaving the defects C3 keeps with CaO₂ among its limits
from the start.

Then, as the same user asked, C4 `dopant_phases` and `max_formation_energy` went, and the
distortions are searched with MACE. A3.1 still runs the host and the pure elements (the dopants'
too: C3's table gives each defect's least formation energy with them), C3 cuts only across a
dopant's sites, so that every dopant keeps its lowest defect, and the compounds run only in E1.1. C4
carries each survivor's screened structures into the electronic cell, makes ShakeNBreak's
distortions of every charge state there (its full set by default, each charge state with its own
number of neighbours) and relaxes them all with MACE, the starts too; MACE has no charge, so they
are pooled by defect, and C5 relaxes in CASTEP, in every charge state, the start and the
`distort_starts` (2) lowest distinct minima. C5 to C9 are C4 to C8 again. Tested with MACE stood in
for, on CaO:Sb's Sb<sub>Ca</sub> + V<sub>O</sub> in −1 and 0 in the 216-atom cell: one batch of MACE
relaxations (each charge state's start and its distortions, around 4 and 3 neighbours), the −0.3
distortions of both charge states relaxing into one structure and merged, the two lowest minima
relaxed in both charge states beside the starts, and C5's ranking naming them `MACE 1` and `MACE 2`
(a MACE minimum that beats the start is a new minimum); `distort_starts` 1 and 5; MACE jobs still
queued; `mace : false`, where C5 relaxes the distortions themselves, four by default; and the
screening cell, where the screened structures join the pool. On the references' made-up CaO energies
C3 drops only on the site cut (Sb<sub>i</sub> + O<sub>i</sub> at 0.8 eV, Sb<sub>i</sub> at 1.25 eV)
and keeps V<sub>Ca</sub> and Bi<sub>Ca</sub>, whose least formation energies (3 and 2.6 eV) are now
only shown; a project that still gives `max_formation_energy` is told it has gone. A work directory
with C4 `dopant_phases` drops its record and results (its jobs stay), and one of the older layout
migrates as before, C4 to C6 then running again.

B2's re-seeding and B3's `scan_unconverged` were checked on CaO with Sb<sub>Ca</sub> and
Sb<sub>Ca</sub> + V<sub>O</sub> 2.4 to 8.0 Å apart, n × n × n cells of 54 to 686 atoms and a
stand-in for MACE (relaxation energies on a line R∞ − c/N, 0.8 eV less in a shallow minimum that
one pair's relaxations from the ideal structure fell into in two of its sizes, and another pair
whose energy wanders by size whatever its start). The pairs 7.2 Å and more apart were reported as
fitting too few sizes, with the warning (the ladder tests pairs up to 6.78 Å); the jumping pair was
re-seeded from its deepest size and converged, and the wandering one was re-seeded and didn't. B3
cut one arrangement, dropped the unconverged survivors, sized the cell for the rest (the floor,
128 atoms) and with `largest` went to 686 atoms; C1 left out the dropped ones, took B2's re-seeded
relaxation in its cell without a new one, and ran its own from B2's minimum in a cell outside the
ladder. A rerun of B2 reused every relaxation, the re-seeded ones too. `scan_test : flat` and
`supercell` (nothing dropped) were run as well. Not run with the real MACE.

`mace_rattle` was checked with Lennard-Jones standing in for MACE, on an fcc crystal with an atom on
its octahedral hole: the energy before stays the ideal structure's, only the atoms within 4 Å of
the defect move (7 there), the same rattle gives the same relaxation, and a MACE job's worker
applies it from `tasks.json` as a relaxation here does; and with the B2 to C1 stand-in above: every
first relaxation rattled and no re-seeded one, C1 still reading B2's relaxations in a size of the
ladder, and a rerun of B2 reusing every one.

The MACE jobs' limit was checked with Slurm and the workers stood in for: 120 relaxations,
`mace_max_jobs` 4 and `mace_pack` 5, every job reaching its time limit after 12 relaxations with 3
more started. Never more than 4 jobs were queued at once, the leftovers went into rounds of 4 jobs
(30, then about 19, then about 6 relaxations each), and an attempt was counted for exactly the
relaxations started and not finished; jobs that started nothing stopped the run after three, with
the attempts kept for `--resubmit-failed`, which then finished them; and a job of the older
`dwf.py` (its attempts counted when it was submitted) was not counted again. (The older `dwf.py`
gave each finished job's leftovers up to `mace_max_jobs` new jobs of `mace_pack` relaxations, on
top of those still queued: 50 jobs that reached a 4-hour limit together became 995 jobs of about
20, until Slurm's job limit stopped the run.)

The CASTEP runner's restarts were checked the same way, through `run_batch` with Slurm and CASTEP
stood in for: 30 single points in packs of 5, `max_jobs` 3, every pack running out of time once.
The older `dwf.py` restarted each pack's calculations straight away, one job each, and had 15 jobs
queued at once; now the restarts waited for places and there were never more than 3, with all 30
done, as with `--resubmit-failed` (`max_restarts : 0`) resubmitting the packs' calculations. A run
stopped with restarts held sent them on its next run as they were, without planning them again
from their dry runs; and with job arrays (one calculation a task) the queue stayed at `max_jobs`
too.

Not tested: real Cirrus jobs (see 10 below), `phases_from : doped` (no Materials Project access
here), C3's cut on real energies (see 15 below), C4's search with the real MACE (see 8 below), real
`mace_run_train` trainings with the charge embedding (F2: the command was checked, not run), real
CASTEP ΔSCF (C7, D2, F1) and `.pdos_weights` files (C3, C7; their layout is taken from CASTEP's
source), real CASTEP stress tensors (C8), real Slurm for the MACE and model jobs and for job arrays
(`array`, with `pack` too), a phonon job on a real GPU, and real CASTEP output for the DFPT
permittivity and the `.bands` of spin-polarised runs.

On the first real runs, check:

0. **The MACE jobs** (B2): before the first run, `python3 dwf.py NAME B2 --prepare-only`
   writes the jobs' scripts to `B_candidates/jobs/mace_scan/prepared/`: check that `job.sh`
   loads what mace-torch needs and that `mace_python` and `dwf.py` are on `/work`. Then look
   at B2's table in `B_candidates/summary.txt` (see the O-vacancy note above), B3's choice,
   floor and band-edge folding, and B4's electronic cell, before phase C. CaO's conduction-band
   minimum is usually reported at X in PBE (an indirect Γ–X gap), which an odd multiple of the
   FCC primitive cell doesn't fold onto Γ: B3's table shows where A2.2 found it, and B4's which
   cells fold all three X points; a cubic `transform`'s cells always do. For a big screen, give
   B2's jobs settings of their own in `%block runner_scan` (in `NAME.run`): more of them
   (`mace_max_jobs`), and the time they need (`max_walltime`).

1. **A2.3 dielectric:** CASTEP's DFPT needs norm-conserving pseudopotentials, fixed occupancies
   and no spin polarisation. Give `dielectric_cell` (with an NCP `species_pot`) and
   `dielectric_param`, or set `dielectric` yourself. `dwf.py` warns if the potentials don't
   look norm-conserving. Check that the `.castep` has both "Optical Permittivity
   (f->infinity)" and "DC Permittivity (f=0)", and that `A_host/summary.txt` shows the values
   printed there.
2. **The `.bands` reader** (A2.2, C2): the electron count per spin channel of a spin-polarised
   run is taken from the header when CASTEP gives one number per channel, otherwise from the
   Fermi energy. Compare the gap levels of one run (`screen` in `C_screen/results.json`) with
   its `.bands`. The band carriers use the total count, so they don't depend on this: check one
   run C3 lists as in a band against its `.castep` (its Fermi energy at the host's band edge, its
   spin spread thinly over many atoms).
3. **The screening spin** (C2): each run starts from a triplet (`spin : 2`, or a doublet for an
   odd electron count) with `SPIN=` on the dopant. With `fix_occupancy : false` (smearing)
   CASTEP holds it only for its first `spin_fix` SCF cycles (10 by default) and then lets it
   relax, so a defect whose ground state is a singlet can still end there; the results record
   the final spin. With `fix_occupancy : true` the spin is not freed, so every even-electron
   state would stay a triplet.
4. **Gap levels** are compared with the host's band edges at the same settings without potential
   alignment: fine for sorting candidates, not for production levels (those come from E3).
5. **The band edges** come from A2.2's uniform Γ-centred grid, not a band-structure path: lower
   `bands_kspacing` if the edges may lie between grid points, or give `gap`.
6. **Packs** (`pack`, `dryrun_pack`): the mock Slurm doesn't check what a pack's job script asks
   for. On the first packed step, look at a `jobs/<step>/packs/pack_NNN/job.sh` (or
   `dryrun_NNN`), and check with
   `sacct -j <job id> --format=JobID,JobName,NTasks,AllocCPUS,Start,End,MaxRSS` that its job
   steps ran side by side, each with its own processes. On Cirrus, also check that the steps were
   allowed their `--mem-per-cpu` (each core's share of the pack's memory): if Slurm refused it,
   each job's `slurm-<id>.out` says so, and if the steps ran one after another (their Start
   times in `sacct`), their memory didn't fit in the pack's at once.
7. **C7 (`excite`)**, before relying on it: run one defect (e.g. `%block defects` with
   `Sb_Ca + Vac_O  all  q=-1`, and `--until C7`). In `C_screen/jobs/excite_gs/`, check that
   CASTEP wrote a `.pdos_weights` file and that C7's table of levels gives the defect's levels a
   small participation number (a few atoms) and the host's bands a large one: if every band
   looks localised, or none does, the file was read wrongly (`read_pdos_weights` tries both byte
   orders). In `C_screen/jobs/excite/`, check that the `.castep` shows CASTEP applying the
   constraints to the two bands named in the `.param`, that the excited state relaxed (or set
   `excite_relax : false` if CASTEP won't relax it), and that the ZPL in `C_screen/summary.txt`
   is near ADAQ's 0.53 eV for Sb_Ca–V_O⁻ (ADAQ used VASP with PAW potentials, so expect some
   difference). If the SCF won't converge with the constraints, try `deltascf_smearing : 0.01`
   (or more `nextra_bands`) in `%block excite_overrides`.
8. **C4–C5 in the electronic cell**: for CaO:Sb with the default cells, C4's MACE relaxations and
   C5's CASTEP relaxations are in n432 (`C_screen/mace/`; `C_screen/jobs/drelax/*_n432` and
   `host_n432`). C5's are the biggest jobs of phase C, so check their plan (the runner's
   `%block runner_drelax`) before the batch goes, and stop after C3 (`--until C3`) to count the
   survivors first. Then stop after C4 (`--until C4`): its table in `C_screen/summary.txt` lists
   each defect's lowest MACE minima, with how many relaxations ended in each and from which charge
   states and distortions, so you can look at those C5 will start from
   (`C_screen/structures/distortions/d001_m1_n432.cell`, ...) before it relaxes each in every charge
   state; the times in C4's MACE jobs (`C_screen/jobs/mace_distort/`) tell you what
   `%block runner_distort` needs. After C5 (`--until C5`), its ranking in `C_screen/summary.txt`
   shows whether a MACE minimum beat the carried-over structure in each charge state, which runs
   ended in the same structure and which failed (and how low they got first), and
   `C_screen/results.json` which charge states still leave an electron or hole in a band there (each
   run's `band`).
9. **C3's band-state test** (`distort_charges : relevant`): it reads `.pdos_weights` files as C7
   does. Run `--until C3` first. In `C_screen/jobs/filter/` (for CaO:Sb, `d001_qm1_bw`, the
   single point of d001's −1), check that each single point wrote one, and that C3's table of
   band states in `C_screen/summary.txt` gives the carriers a spread that makes sense: a few
   atoms to a few tens for one on the defect, a good part of the cell for a free one. If every
   carrier looks spread out, or none does, the file was read wrongly (see 7).
10. **Cirrus** (`charging : core`): look at the first `dryrun.sh` and `job.sh` of a step:
   `--nodes`, `--ntasks-per-node` and `--cpus-per-task` should match the plan in the log, with
   `--exclusive` only on jobs bigger than a node. Then check with
   `sacct -j <job id> --format=JobID,Partition,AllocCPUS,ReqMem,MaxRSS,Elapsed` that each job got
   the cores (and so the memory) it asked for, and how much memory it used: `mem_replicated : auto`
   measures it from CASTEP's peak, so the plans tighten as jobs finish.
11. **C8 (`validate`)**: in the first relaxation's `.castep` (`C_screen/jobs/validate/`), check
   that CASTEP printed a "Stress Tensor" block for the final structure (C8 reads the last one), and
   that the pristine single points did too ("Symmetrised Stress Tensor"). Then, in
   `C_screen/summary.txt`, that `fit worst` is a few meV: if not, the sizes above the screening
   cell aren't on 1/N yet, and a bigger `validate_sizes` (or longer `scan_lengths`) is needed
   before the verdicts mean anything.
12. **D2**: the table says which excitations ran at the production k-points and which fell back
   to Γ, and why. For one that ran with k-points, check in its `.castep` that the constrained
   occupations were applied at every k-point.
13. **F1 and F2**: F1's misfit column should be a few meV; a large one, or ħω_g far from ħω_e,
   means the equal-mode approximation is poor for that excitation (look at the points in
   `<excitation>_ccd.dat`, and check in the ΔSCF runs at the ends of the path that the same bands
   were constrained). For F2, try a smaller `lineshape_length` first to time the phonon job, and
   check the model's test error and the imaginary modes in F2's table.
14. **Job arrays** (`array`): on the first array, look at `packs/array_NNN/job.sh` and
   `tasks.txt` in the step's jobs folder, and check with `squeue -r -u $USER` that its tasks are
   queued as `<array job id>_<task>`, and afterwards that each task's
   `slurm-<array job id>_<task>.out` and `.castep` are in its own job's folder. If `sbatch` refuses
   the array (a limit on the jobs you may queue, or an array bigger than the site allows), lower
   `array` and `max_jobs`. With `pack` too, the script is `packs/pack_array_NNN/job.sh`: check
   that each task ran its jobs side by side on one node, each within its memory
   (`sacct -j <array job id>_<task> -o JobID,NodeList,NCPUS,MaxRSS,Elapsed` lists a step for each
   job), and that each job's `slurm-<array job id>_<task>.out` in its folder shows no `srun`
   errors.
15. **C3's cut across a dopant's sites** (with competing phases): in C3's table, look at the
   `above site` and `least E_f` columns for defects you know (in CaO:Sb, Sb<sub>Ca</sub> +
   V<sub>O</sub>). The screening settings (`screen_cutoff`, Γ only, few geometry steps) can put
   formation energies some tenths of an eV off: once phase E has run, compare A3.2's Δμ with E1.2's,
   and a survivor's least formation energy with E3's. If a defect you expect is dropped, raise
   `site_margin` (C3 runs again; its jobs are reused).
16. **A3.1's relaxations at a low `screen_cutoff`**: list the ones that failed, and how
   often each reverted, from `A_host/jobs/ref_phases`:
   `grep -l "failed to converge" */*.castep` and `grep -c "reverting to earlier" */*.castep`.
   Many reverts mean an energy that jumps as the grid changes (check that the steps' energies
   fall into two groups): add `fine_gmax` to `%block screen_overrides` (glossary) and
   `--redo ref_phases`. A failure without reverts usually just needs more `phases_max_steps`.
   Each cell relaxation's `.castep` also gives `dEtot/dlog(Ecut)` near the top: the larger it is
   per atom, the further that element is from converged at `screen_cutoff`.
17. **doped's molecules** (`phases_from : doped`): A3.1's log names them and their box. 15 Å leaves
   at least 13 Å of vacuum around each, plenty for a small neutral molecule at Γ. To check, relax
   one by hand in a 20 Å box with the same settings (its folder in `A_host/jobs/ref_phases/` has
   them): its energy should change by no more than a few meV.
18. **B2's re-seeding and B3's drops**: B2's log says how many candidates were re-seeded and how
   many of them then converged. In B2's table, `^` marks a size whose relaxation from the deepest
   minimum went deeper, and the last column the size that minimum came from: for a few, check that
   the jumps are gone (R on one smooth curve with size). B3's table lists what `scan_unconverged :
   drop` dropped; for a candidate you care about there, its B2 row says why (a pair with too few
   sizes needs more `scan_lengths` or a smaller `pair_max_distance`).
19. **`mace_rattle`**: the first run with it relaxes B2's and C1's candidates again (their starts
   are new). The steps per relaxation (the workers' logs, `progress.txt`) should go up only a
   little; for a few interstitials, compare R with an earlier run's: a larger one means the
   rattle let it leave a symmetric site.
