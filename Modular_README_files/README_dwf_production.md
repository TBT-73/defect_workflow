# dwf_production.py — phase D: production relaxations

Relaxes the structures phase C kept, in each of their charge states, at the production settings,
and runs the pristine supercell at the same settings. These are the energies phase E uses.
Flowchart boxes D1 and D2. Phase C must have finished.

```bash
nohup python3 dwf_production.py CaO_Sb.dwf > D.log 2>&1 &
python3 dwf_production.py CaO_Sb.dwf --status
```

The options (`--redo production`, `--no-wait`, `--resubmit-failed`, ...) are the same for every
phase: see `README_workflow.md`.

---

## What it runs

One step, D1 `production`:

- **one relaxation per structure in `C_screen/rank.json`** (the lowest `keep_structures` distinct
  structures of each survivor and charge state):
  - the production `.param` (A1's cut-off) with `%block production_overrides`, the production
    k-point spacing, `charge : q`;
  - the starting spin from phase C (the final spin of its screening run, with the parity the
    charge state needs): `spin_polarized : true`, `spin : S` and `SPIN=S` on the anchor atom.
    A spin-0 state runs without spin polarisation, unless `production_overrides` sets
    `spin_polarized` (`spin_polarized : true` runs those spin-polarised too);
  - positions only (the supercell is fixed at the host's relaxed lattice), no symmetry;
- **one single point of the pristine supercell** with the same settings (with symmetry): E_host
  for the formation energies.

The jobs run as runner group `production`: restarts are on, so a relaxation that runs out of
walltime is carried on from its `.check` (as in the size series), and with
`tolerate_failures : true` (the default for this group) a job that fails is flagged while the
others carry on. A failed relaxation is left out of phase E with a warning; fix the cause and
rerun with `--resubmit-failed`. The step stops if the pristine single point fails.

## D2: freezing the far atoms (an untested idea)

With `freeze_radius : r` (Å), every atom further than r from all of the defect's components is
held fixed (`%block ionic_constraints`, three constraints per atom), so the relaxation moves only
the atoms near the defect. This cuts the geometry steps of large cells but changes the relaxation
energy; it has not been compared with full relaxations. Check it on one defect (with and without)
before relying on it. `D.log` says how many atoms each relaxation holds.

---

## Keywords

| Keyword | Meaning |
|---|---|
| `%block production_overrides` | `.param` keywords for the production runs, e.g. `geom_energy_tol`, `geom_max_iter`, `elec_energy_tol`, `spin_polarized` |
| `freeze_radius` [0] | D2: hold atoms further than this (Å) from every component fixed |
| `keep_structures` [1] | (phase C) how many structures per charge state reach D |

Runner settings: `%block runner_production` (e.g. a longer `walltime`, more `max_nodes`).

---

## Output

In `<workdir>/D_production/`:

| File | Contents |
|---|---|
| `relax/<id>_<q>_k<n>/` | the CASTEP jobs (`q`: `qm1`, `q0`, `qp1`, ...; `k`: rank of the structure in C6) |
| `production.json` | E_host, and per run: candidate, charge, energy, final spin, status, steps, restarts, frozen atoms |
| `structures/` | the relaxed structures |

---

## Notes and limitations

- **Energies count only if the last SCF cycle converged** and the job finished; otherwise the
  run is listed with no energy and E2 uses the other structures of that charge state, if any.
- **The final spin** of each run is in `production.json` (`spin_final`); E2 uses it for the spin
  degeneracy. If it differs from the starting spin, CASTEP found a different spin state.
- **Raising `keep_structures`** after D has run adds only the new structures' jobs; the others
  are reused.
- With many structures, `max_jobs` (how many jobs are queued at once) keeps the queue manageable.
