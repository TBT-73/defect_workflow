# dwf_host.py — phase A: the host

The reference calculations every later phase depends on: the production cut-off and k-point
spacing, the relaxed host, its band edges and dielectric constant, and the chemical-potential
limits from its competing phases. Flowchart boxes A1 to A3.

```bash
python3 dwf_host.py CaO_Sb.dwf --prepare-only     # write the A1 inputs and dry runs, submit nothing
nohup python3 dwf_host.py CaO_Sb.dwf > A.log 2>&1 &
python3 dwf_host.py CaO_Sb.dwf --status
```

The options (`--redo`, `--no-wait`, `--resubmit-failed`, ...) are the same for every phase:
see `README_workflow.md`. Keep `dwf_core.py` and `supercell_convergence.py` in the same
directory.

---

## Contents

1. [Steps](#steps)
2. [A1 converge](#a1-converge)
3. [A2.1 relax, A2.2 bands](#a21-relax-a22-bands)
4. [A2.3 dielectric](#a23-dielectric)
5. [A3.1 phases, A3.2 limits](#a31-phases-a32-limits)
6. [Keywords](#keywords)
7. [Output](#output)
8. [Notes and limitations](#notes-and-limitations)

---

## Steps

| Label | Step | What it does | Stops if |
|---|---|---|---|
| A1 | `converge` | single points of the primitive cell over cut-offs and k-point spacings; writes `production.param` | a convergence run has no converged energy; only the last value is within tolerance |
| A2.1 | `relax` | variable-cell relaxation of the host | the relaxation doesn't finish |
| A2.2 | `bands` | single point on a denser Γ-centred grid: VBM, CBM, gap | no `.bands` file |
| A2.3 | `dielectric` | DFPT for the static and optical dielectric constants, or the value you give | no permittivity in the output, or only the optical one |
| A3.1 | `phases` | relaxes the competing phases | a phase doesn't finish |
| A3.2 | `limits` | chemical-potential limits | a pure element of the host is missing; the host is above the hull |

All the CASTEP jobs run through `supercell_convergence.py`'s batch mode as runner group `host`
(restarts on, a failure stops the step), in `<workdir>/A_host/<step>/`.

Every job copies the host `.cell`'s `species_pot` (and anything else in it apart from the
structure and k-points), so list the dopants' potentials there too: the competing phases with
Sb in them need one.

---

## A1 converge

One batch of single points of the primitive cell as given (with symmetry):

- every cut-off in `converge_cutoffs`, on the finest k-grid (`kspacing` if set, else the
  smallest of `converge_kspacings`);
- every spacing in `converge_kspacings` at the highest cut-off (skipped if `kspacing` is set).
  Spacings that give the same grid run once.

**Decision:** the converged value is the smallest cut-off (and the largest spacing) whose energy
per atom, and that of every finer setting, is within `converge_tol` of the finest one. If only
the finest value qualifies, the list hasn't shown convergence and the step stops: add higher
cut-offs (or smaller spacings) and rerun; the finished single points are reused.

`A_host/production.param` is your `.param` with `cut_off_energy` set to the converged cut-off
(any `cut_off_energy` or `basis_precision` line removed). `charge` and `spin` lines are left out
too, with a note: every job sets its own. With `converge : false` nothing runs: your `.param`
(without `charge`/`spin`) becomes `production.param`, and the k-point spacing is `kspacing`, or
the spacing of the host `.cell`'s grid.

---

## A2.1 relax, A2.2 bands

**A2.1** relaxes the lattice and positions of the primitive cell (with symmetry) at the
production settings. The relaxed cell goes to `host_relaxed.json` (read by every later phase) and
`host_relaxed.cell` (the size series' cell); `e_bulk` is the final energy per atom.

**A2.2** runs a single point of the relaxed cell on a Γ-centred grid at `bands_kspacing` (default
half the production spacing). VBM, CBM and gap are the highest occupied and lowest empty
eigenvalues over that grid and both spin channels. The `.bands` file is used again in E2 for the
free carriers.

---

## A2.3 dielectric

The charge corrections (B3 with a charged series, C3, E1) need the static dielectric constant
(electronic + ionic).

- With `dielectric` in the project file, nothing runs (`dielectric_inf` is optional and only
  recorded).
- Otherwise one DFPT job on the relaxed host: `task : dielectric_task` (default
  `phonon+efield`) at q = Γ (`%block phonon_kpoint_list`), at the production k-point spacing.
  ε₀ is the average of the diagonal of CASTEP's DC permittivity, ε∞ that of the optical one; a
  warning is printed if the diagonal varies by more than 10 %.

CASTEP's DFPT needs norm-conserving pseudopotentials, fixed occupancies and no spin
polarisation. Without `dielectric_param` the job uses `production.param` with
`fix_occupancy : true` and `spin_polarized : false`. If the `species_pot` it would use doesn't
look norm-conserving (not a `.recpot`, no "NCP" in the name or on-the-fly string, or no
`species_pot` at all), the script warns before submitting. The fix is a `.cell` with NCP
potentials (`dielectric_cell`) and a `.param` with a cut-off suited to them
(`dielectric_param`), or `dielectric` from elsewhere.

`task : efield` gives only the optical permittivity: the step stops and asks for
`phonon+efield` or `dielectric`.

---

## A3.1 phases, A3.2 limits

**Where the phases come from** (first that applies):

1. `%block chemical_potentials`: the limits themselves, nothing runs.
2. `%block phases`, one line per phase: `name  structure-file  [options]`. Options:
   `molecule` (positions only, in its fixed box, Γ point, no symmetry), `metal`
   (`fix_occupancy : false` unless the line sets it; give `smearing_width=...` too),
   `kspacing=0.03` (its own spacing), and any other `keyword=value` as a `.param` keyword
   for that phase only (`spin=2 spin_polarized=true` for O₂).
3. `phases_from : doped`: doped's `CompetingPhases` from the Materials Project (entries within
   `phases_e_above_hull` of the hull, elements, molecules in a box; needs `doped` and
   `mp_api_key`). O₂ gets `spin=2`, metals `fix_occupancy=false`.
4. None of these: no limits, and E2 gives energies without the μ terms (comparable only within
   one composition).

Solids relax lattice and positions (with symmetry) at the production settings.

**A3.2** builds a pymatgen phase diagram from the host and the phases made of its own elements
only. It needs a phase of each pure element of the host, and stops if the host lies above the
hull (it would decompose, so it has no stable chemical potentials). Each corner of the host's
stability region is a limit, named after the phases there (e.g. `CaO-Ca`, `CaO-O2`). Each dopant
X then gets, at every corner, the highest μ_X its phases allow:

  μ_X = min over phases P containing X of (E_P − Σ_{i≠X} n_i μ_i) / n_X

and the phase that sets it is recorded (`dopants limited by` in `limits.txt`). Only phases made
of the host elements and that one dopant count.

`limits.txt` gives each limit as Δμ against the pure-element phases (where those were run); the
absolute μ (CASTEP's energy scale) are in `state.json` and used by E2. Values in
`%block chemical_potentials` are absolute too: `O-rich  Ca=-1008.6  O=-432.6  Sb=-505.1`
(eV per atom).

---

## Keywords

Defaults in brackets.

| Keyword | Step | Meaning |
|---|---|---|
| `host_cell`, `param` | all | host primitive cell; production settings (required) |
| `converge` [true] | A1 | converge the cut-off and k-points; `false` uses the `.param` and k-points as given |
| `converge_cutoffs` [300 … 800 eV] | A1 | cut-offs to try |
| `converge_kspacings` [0.1 … 0.03 1/Å] | A1 | spacings to try (CASTEP's 2π units) |
| `converge_tol` [0.001 eV/atom] | A1 | convergence tolerance |
| `kspacing` | A1 | use this spacing instead of converging it |
| `bands_kspacing` [half the production spacing] | A2.2 | the band-edge grid |
| `dielectric`, `dielectric_inf` | A2.3 | give the dielectric constants instead of running DFPT |
| `dielectric_task` [phonon+efield] | A2.3 | the DFPT task |
| `dielectric_param`, `dielectric_cell` | A2.3 | separate `.param` / `.cell` for the DFPT job |
| `%block phases` | A3.1 | competing phases: `name file [options]` |
| `phases_from` [none] | A3.1 | `doped`: get them from the Materials Project |
| `mp_api_key`, `phases_e_above_hull` [0.05 eV/atom] | A3.1 | for `phases_from : doped` |
| `%block chemical_potentials` | A3.2 | the limits themselves (absolute eV/atom) |
| `dopants` | A3 | the dopants whose μ are limited |

Runner settings for these jobs: `%block runner_host` (see `README_workflow.md`).

---

## Output

In `<workdir>/A_host/`:

| File | From | Contents |
|---|---|---|
| `converge/`, `relax/`, `bands/`, `dielectric/`, `phases/` | all | the CASTEP job directories |
| `converge.txt` | A1 | energy per atom against cut-off and k-grid |
| `production.param` | A1 | the `.param` every later job starts from |
| `host_relaxed.json`, `host_relaxed.cell` | A2.1 | the relaxed primitive cell |
| `limits.txt` | A3.2 | Δμ at each limit, and the phase limiting each dopant |

`--status` (or `state.json`) has the cut-off and spacing, `e_bulk` and the lattice, VBM, CBM and
gap, ε₀ and ε∞ (and the tensors), the competing-phase energies, and the absolute μ of every
limit.

---

## Notes and limitations

- **The band edges come from a uniform Γ-centred grid**, not a band-structure path: a VBM or CBM
  between grid points is missed by up to the band dispersion between them. Lower
  `bands_kspacing` if the edges matter, or give `gap` (E2) from a separate calculation.
- **A2.3 has only been tested against a mock CASTEP.** Check the first real run: the `.castep`
  should contain both "Optical Permittivity (f->infinity)" and "DC Permittivity (f=0)", and
  `--status` should show the values printed there.
- **Competing phases use the production settings**, including its cut-off, so that their
  energies are on the same scale as the host's. Metals need smearing (`metal`, `smearing_width`)
  and usually a finer k-point spacing (`kspacing=` on their line).
- **`phases_from : doped` has not been tested** (no Materials Project access here). Once A2.1 has
  finished, run `python3 dwf_host.py CaO_Sb phases --prepare-only` and look at the job
  directories it writes in `A_host/phases/` before anything is submitted.
- Changing `dopants` reruns A3.1 (new phases may be needed) and A3.2; jobs already done are
  reused.
