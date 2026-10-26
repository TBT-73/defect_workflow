# dwf_core.py

The code the five phase scripts share. You don't run it yourself: `dwf_host.py`,
`dwf_candidates.py`, `dwf_screen.py`, `dwf_production.py` and `dwf_thermo.py` import it, and it
imports `supercell_convergence.py` (the job runner), so keep all seven files in one directory.

This README is for reading or changing the code. To run the workflow, start with
`README_workflow.md`.

---

## Contents

1. [What it provides](#what-it-provides)
2. [The project file](#the-project-file)
3. [Steps and checkpoints](#steps-and-checkpoints)
4. [Running CASTEP jobs](#running-castep-jobs)
5. [Structures and defects](#structures-and-defects)
6. [Reading CASTEP output](#reading-castep-output)
7. [MACE](#mace)
8. [Adding a step](#adding-a-step)

---

## What it provides

| Part | Functions |
|---|---|
| project file | `read_project`, `PROJECT_KEYWORDS`, `RUNNER_KEYS`, `GROUP_DEFAULTS`, `EXAMPLE_PROJECT` |
| the project | `Project`: settings (`p.s`), files, directories, checkpoints, runner settings, CASTEP batches, production settings, relaxed host, supercell matrix, MACE models |
| steps and checkpoints | `Step`, `run_phase`, `inputs_hash`, `outputs_hash`, `reset_step`, `print_status` |
| structures | `atoms_to_dict`, `atoms_from_dict`, `save_structure`, `load_structure`, `read_structure_file`, `host_supercell`, `counts` |
| symmetry | `symmetry_ops`, `equivalent_sites` (spglib) |
| defects | `build_defect`, `comp_name`, `defect_name`, `composition_change`, `electron_parity`, `spin_states`, `qtag` |
| CASTEP output | `read_bands`, `occupied_bands`, `band_edges`, `gap_levels`, `zpl_estimate`, `read_permittivity`, `read_trajectory` |
| corrections | `correction` (Lany–Zunger or Makov–Payne, from `supercell_convergence.py`) |
| MACE | `Project.mace`, `Project.model_id`, `mace_relax` |
| files | `write_json`, `read_json`, `write_table` |

Every function has a comment block above it saying what it does, which functions call it and
which it calls, as in `supercell_convergence.py`. `Project.out()`, `Project.phase_dir()`,
`read_json()` and `write_json()` are used by almost every step and are left out of those lists.

---

## The project file

`read_project(path)` reads `<name>.dwf` with `supercell_convergence.py`'s CASTEP-style parser.
Each keyword is either:

- a **project keyword** in `PROJECT_KEYWORDS` (`keyword: (converter, default)`): converted with
  the same converters as `supercell_convergence.py` (`_num`, `_ints`, `_bool`, `_choice`, ...)
  or `_floats`. Two special kinds: `"lines"` blocks keep their rows (`%block phases`,
  `%block defects`, `%block chemical_potentials`, `%block oxidation_states`) and `"kv"` blocks
  hold `keyword : value` lines (`%block screen_overrides`, `%block production_overrides`);
- a **runner keyword** in `RUNNER_KEYS`, kept as parsed items for `supercell_convergence.batch_args`;
- `%block runner_<group>` for `host`, `series`, `screen` or `production`, holding runner keywords.

A typo stops it with the closest keyword suggested. `host_cell`, `param` and `jobscript` are
required.

`Project.runner(group)` builds the runner settings for a group of steps: the project's runner
keywords, then the group's defaults (`GROUP_DEFAULTS`: screening `max_restarts : 0` and
`tolerate_failures : true`; production `tolerate_failures : true`), then the group's block,
each replacing the one before, then the command-line switches (`--no-wait`, `--prepare-only`,
`--force`, `--resubmit-failed`).

---

## Steps and checkpoints

A phase script lists its steps as `Step(name, label, title, function, keys=..., files=...,
needs=..., owns=...)`:

- `function(p)` does the work and returns the step's outputs (a JSON-able dict, with an
  optional `_note` for the log and `--status`), or `None` when it isn't finished (jobs queued
  with `--no-wait`, `--prepare-only`);
- `keys`: the project keywords its results depend on; `files`: keywords naming files it reads;
  `needs`: earlier steps it uses; `owns`: paths under the work directory that belong to it
  (moved aside by `--redo`).

`run_phase(phase, steps, doc)` is the whole command line of a phase script. For each step:

1. `inputs_hash`: an MD5 of the step's settings (as written in the project file), the contents of
   its files, and the `out_hash` of each step it needs. A needed step that hasn't finished stops
   the script, naming the script that runs it.
2. If `state.json` has the step done with the same hash, it is skipped.
3. Otherwise the function runs. If it returns outputs, `state.json` records them with
   `outputs_hash`: an MD5 of the outputs and of the contents of every file they name, so a
   step whose results change makes the later steps run again, and one whose results come out
   the same doesn't.

`state.json` is written through a temporary file, so an interrupted write never corrupts it.
`--redo STEP` (`reset_step`) moves the step's `owns` paths to `<name>.old-<time>` and forgets
the step; an owned path with a `*` (e.g. `C_screen/share_*`) moves every match.

---

## Running CASTEP jobs

`Project.batch(subdir, jobs, group, param_text, cell_text=None, spacing=None)` calls
`supercell_convergence.run_batch` in `<workdir>/<subdir>` with the group's runner settings, the
host `.cell` (or `cell_text`) as the template for `species_pot` and the rest, the given `.param`
text, and the project's STOP file. Each job is a dict (`name`, `atoms`, `kind` geom/sp, `task`,
`overrides`, `kgrid`, `kpoint_offset`, `fix_cell`, `symmetry`, `cell_extra`); see `run_batch`'s
comment in `supercell_convergence.py`. It returns `{'complete', 'jobs', 'calcs'}`; a step returns
`None` while `complete` is false.

`Project.production_param()` is A1's `production.param` once A1 has finished (the project's
`.param` before). `Project.kspacing()` is A1's spacing, else `kspacing`, else the spacing of the
host `.cell`'s grid.

---

## Structures and defects

Structures move between steps as JSON (`save_structure`/`load_structure`): species, cell,
fractional positions and starting spins, plus any extra information (e.g. where the defect is).

A defect is a list of components, each a dict:

| key | meaning |
|---|---|
| `new` | the element put there, or `None` for a vacancy |
| `host` | the host element replaced, or `None` for an interstitial |
| `prim_site` | index of its site in the primitive cell (substitutions, vacancies) |
| `frac` | fractional position in the primitive cell (interstitials) |
| `vec` | vector (Å) from the first component (the anchor), for the others |
| `spin` | starting spin of an added atom |

`build_defect(prim, P, comps)` puts it into the supercell P × (primitive cell):

- every component after the anchor goes at its vector from the anchor, onto the host atom of
  its element there (within 0.1 Å), or for an interstitial at that point;
- the anchor goes where the whole defect is as close to the cell centre as it can be: of the
  copies of its primitive-cell site, and of the sites a pure translation of the crystal makes
  identical to it (`sc.translation_copies`, e.g. all four Ca of a conventional rocksalt cell),
  the one that puts the mean position of the components nearest the centre
  (`sc.anchor_site`; ties go to the one whose farthest component is nearest). An interstitial
  anchor takes the best image of its position the same way. The complex is only moved, never
  turned, so it is the same in every cell.

It stops if a site can't be found or the complex would touch its own periodic images. It
returns the structure and `info`: the defect's centre (the mean position of its components),
each component's position, the anchor's atom index (`None` for a vacancy anchor) and the
vacancies' positions, which later steps use (spin on the anchor atom, the distortion centre,
the far-field region of B2, the freeze radius of D2).

`composition_change(comps)` gives the change in atoms (e.g. `{'Sb': 1, 'Ca': -1, 'O': -1}`) and
its text (`+Sb-Ca-O`): arrangements with the same text are compared directly on the defect hull.

`electron_parity(atoms, q)` uses the sum of the atomic numbers minus the charge: core shells are
full, so this has the parity of the valence electron count whatever the pseudopotentials.
`spin_states` gives the starting spins the screen tries (0 or 1, and +2).

---

## Reading CASTEP output

- `read_bands(path)`: a `.bands` file: k-points, weights, eigenvalues (eV, `[spin][k][band]`),
  electrons (one number per spin channel, or the total) and Fermi energies.
- `occupied_bands(b)`: occupied bands per spin channel, from the electron counts (or from the
  Fermi energy at the first k-point when CASTEP gives only the total for a spin-polarised run).
- `band_edges(b)`: VBM and CBM over all k-points and spins.
- `gap_levels(b, vbm, cbm, margin)`: the eigenvalues at the first k-point that lie more than
  `margin` inside the band edges of the host, with spin channel, energy above the VBM and
  occupation. No potential alignment: good enough to sort candidates.
- `zpl_estimate(levels)`: the smallest occupied → empty step within one spin channel.
- `read_permittivity(text)`: the optical and static (DC) permittivity tensors from an efield or
  phonon+efield run.
- `read_trajectory(text)`: every structure of a `.castep` with its energy and forces (the last
  lattice and fractional-coordinates table before each `Final energy`, and the Forces table
  after it), for MACE fine-tuning data.

---

## MACE

`Project.mace(model)` loads a model once (`mace_mp`, float64, `mace_threads` torch threads):
`mace_model` by default (a foundation model's name or a model file); C1 passes
`finetuned_model` when it is set. `mace_relax(p, atoms, cache, model)` relaxes positions with
ASE's LBFGS to `mace_fmax` (at most `mace_steps`) and keeps the result in a JSON cache keyed by
the structure, the model (`Project.model_id`: its name, or the MD5 of a model file, so a model
retrained under the same name isn't mistaken for the old one) and the settings, so a rerun
doesn't repeat it. MACE ignores charge and spin; it only prepares structures.

---

## Adding a step

1. Write `def step_x(p):` in the phase script: read what it needs with `p.out("<step>")` and
   `p.s.<keyword>`, run jobs with `p.batch(...)` (return `None` if not complete), write its
   files under `p.phase_dir("<letter>")`, and return its outputs.
2. Add `core.Step("x", "C7", "title", step_x, keys=(...), files=(...), needs=(...),
   owns=(...))` to the script's `STEPS`.
3. Add new keywords to `PROJECT_KEYWORDS` (and to `EXAMPLE_PROJECT`), and the step's name to
   `STEP_OWNER`.
