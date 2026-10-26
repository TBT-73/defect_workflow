# dwf_screen.py — phase C: screening funnel and ground-state search

Takes every candidate from phase B through cheap CASTEP relaxations, keeps those that are
thermodynamically competitive and have the properties you ask for, then searches for their
ground-state structures with ShakeNBreak distortions. Flowchart boxes C1 to C6. Phase B must
have finished.

```bash
nohup python3 dwf_screen.py CaO_Sb.dwf > C.log 2>&1 &
python3 dwf_screen.py CaO_Sb.dwf --status
```

The options (`--redo`, `--no-wait`, `--resubmit-failed`, ...) are the same for every phase:
see `README_workflow.md`.

---

## Contents

1. [Steps](#steps)
2. [C1 prerelax](#c1-prerelax)
3. [C2 screen](#c2-screen)
4. [C3 filter](#c3-filter)
5. [C3b zpl](#c3b-zpl)
6. [C4 distort, C5 drelax](#c4-distort-c5-drelax)
7. [C6 rank](#c6-rank)
8. [Keywords](#keywords)
9. [Output](#output)
10. [Notes and limitations](#notes-and-limitations)

---

## Steps

| Label | Step | What it does | Decision |
|---|---|---|---|
| C1 | `prerelax` | builds every candidate in the B5 supercell, relaxes it with MACE | pairs too far apart for the cell left out; `triage` drops arrangements far above the lowest |
| C2 | `screen` | one cheap CASTEP relaxation per candidate, charge state and starting spin | — |
| C3 | `filter` | defect hull, duplicates, property filters | within `hull_margin` of the hull; the `require_*` filters |
| C3b | `zpl` | optional ZPL window | first ZPL estimate inside `zpl_window` |
| C4 | `distort` | ShakeNBreak distortions of the survivors | which charge states (`distort_charges`) |
| C5 | `drelax` | cheap relaxations of the distortions | — |
| C6 | `rank` | ranks the structures, shares winning distortions | a distortion counts if it beats the screened structure by `distortion_min_gain` |

The CASTEP jobs of C2, C5 and C6 run as runner group `screen`: no restarts and a job that fails
is flagged and the others carry on (`max_restarts : 0`, `tolerate_failures : true`, unless
`%block runner_screen` says otherwise).

---

## C1 prerelax

Each candidate is built in the B5 supercell, as close to the cell centre as it can be (see
"Where a defect goes in the supercell" in `README_dwf_candidates.md`). A pair whose components
are more than half the shortest image distance apart is left out.

With `mace : true`, each structure is relaxed with MACE (positions only, the cell fixed), with
`finetuned_model` if you set one (B4), else `mace_model`. The results are cached
(`C_screen/prerelax/`), so a rerun doesn't repeat them.

**Triage** (`triage : true`, needs MACE): within each composition group, arrangements whose MACE
energy is more than `triage_window` above the group's lowest are dropped before any CASTEP run.
MACE knows nothing of charge or spin, so keep the window generous.

The structures go to `C_screen/structures/<id>.json` with where the defect is (its centre, the
position of each component, the anchor atom, the vacancies).

---

## C2 screen

For each candidate, each charge state in `screen_charges` and each starting spin, one CASTEP
relaxation at the screening settings:

- the Γ point only, no symmetry, positions only;
- `screen_cutoff` (default: the production cut-off), at most `screen_max_steps` geometry steps,
  and any `.param` keywords in `%block screen_overrides`;
- `charge : q`, `spin_polarized : true`, `spin : S` and `SPIN=S` on the anchor atom.

The starting spins (`screen_spins`): `low` is the lowest the electron count allows (0 or 1, from
the parity of ΣZ − q); `both` (the default) adds that + 2. They are starting values: CASTEP may
end elsewhere, and the results record the final spin, which is what the later filters use.

The pristine supercell gets a single point at the same settings: its Γ-point band edges are the
reference for the gap levels, and its energy is the host energy for the binding energies.

For every run the step keeps: the energy (only if the last SCF cycle converged), whether the
geometry converged, the final spin, the defect levels in the gap (eigenvalues at Γ more than
`level_margin` inside the host's band edges, with spin channel and occupation), the first ZPL
estimate (below) and the final structure. A run that failed has no usable energy and is left
out of what follows; `--resubmit-failed` tries it again.

---

## C3 filter

1. **Best state.** For each candidate and charge state, the lowest energy over its starting
   spins.
2. **Defect hull.** Within a composition group, the formation energy of every arrangement and
   charge state across the Fermi level is compared without the host energy and chemical
   potentials, which are the same for all of them: E + E_corr + q (E_VBM + E_F), with
   `charge_correction` and the A2.3 dielectric constant, for E_F from 0 to the screening gap.
   The hull is the lowest of them at each E_F. An arrangement is kept if its own lowest comes
   within `hull_margin` of the hull somewhere in the gap. Its *stable* charge states are those
   that are its lowest somewhere in the gap.
3. **Duplicates.** An arrangement that relaxed into the same structure as a lower one of its
   group, in every charge state both have, is dropped (e.g. Sb_O + Vac_Ca that relaxed into
   Sb_Ca + Vac_O).
4. **Property filters**, each met by at least one stable charge state:
   - `require_gap_states`: `occupied` (an occupied level in the gap), `empty`, or `both` (an
     occupied and an empty level in one spin channel, i.e. an internal transition);
   - `require_spin`: the final spin (unpaired electrons) within 0.25 of this value, e.g. `2`
     for a triplet;
   - `require_bound` (pairs): the binding energy against the isolated components,
     E_b = Σ E_f(parts) − E_f(pair), positive somewhere in the gap. The components must be
     candidates themselves (screened as single defects).

The charge states carried on are the stable ones (`distort_charges : stable`) or every one with
a usable energy (`screened`). `filter.txt` gives every candidate's verdict and why.

---

## C3b zpl

Optional (`zpl_window : lo hi`). The first ZPL estimate is the smallest step from an occupied to
an empty gap level within one spin channel, from the screening run of a stable charge state
that met the property filters. A survivor is kept if one of its estimates is inside the window;
one with no occupied → empty step has no internal transition and is dropped. Kohn–Sham
eigenvalue differences at the screening settings are a rough guide: keep the window wide.

---

## C4 distort, C5 drelax

For each survivor and each charge state carried on, starting from its screened structure (or
that of the nearest screened charge state), ShakeNBreak's `distort_and_rattle`:

- **how many neighbours**: ShakeNBreak's rule, n = q − Σ (oxidation state put there − that of
  the site) over the components, and |n| neighbours if |n| ≤ 4, else 8 − |n|. The oxidation
  states come from `%block oxidation_states`, or from pymatgen's guess for the host and, for
  each dopant, its common oxidation state closest to that of the element it replaces;
- **around what**: the anchor atom, or with `distort_around : vacancy` (or a vacancy anchor)
  the vacant site; a defect with no vacancy always uses its anchor;
- **how far**: each factor in `distortions` (−0.4 = bonds 40 % shorter); the other atoms are
  rattled (`rattle_stdev`, default 10 % of the host bond length; `rattle_dmin`, 80 %). With no
  neighbours to distort, the structure is only rattled.

The starting spin is the final spin of the screened run, with the parity the charge state
needs. C5 relaxes every distorted structure at the C2 settings.

---

## C6 rank

For each survivor and charge state, the screened structure and every distortion relaxation with
a usable energy are pooled, identical structures merged (pymatgen's StructureMatcher with tight
tolerances: `ltol 0.05`, `stol 0.05`, `angle_tol 2`, so structures a few tenths of an Ångström
apart around the defect stay distinct) and ranked by energy.

**Sharing.** A distortion that wins in one charge state (beats the screened structure by
`distortion_min_gain`) and that another charge state of the same defect doesn't have yet is
relaxed in that charge state too. This repeats for up to `share_rounds` rounds, or until nothing
new turns up (jobs in `C_screen/share_<n>/`).

The lowest `keep_structures` distinct structures of each charge state go to phase D
(`rank.json`); `rank.txt` shows, for each charge state, the screened energy, the best energy,
the gain and where the best structure came from.

---

## Keywords

Defaults in brackets.

| Keyword | Step | Meaning |
|---|---|---|
| `mace`, `finetuned_model` | C1 | MACE pre-relaxation, and the model for it |
| `triage` [false], `triage_window` [1.0 eV] | C1 | drop arrangements far above the lowest by MACE energy |
| `screen_charges` [-1 0 1] | C2 | charge states screened |
| `screen_spins` [both] | C2 | `low`, or `both` (also +2 unpaired electrons) |
| `screen_cutoff` [production] | C2, C5, C6 | screening cut-off |
| `screen_max_steps` [50] | C2, C5, C6 | geometry steps per screening relaxation |
| `%block screen_overrides` | C2, C5, C6 | other `.param` keywords for screening |
| `level_margin` [0.1 eV] | C2 | how far inside the band edges a gap level must be |
| `hull_margin` [0.2 eV] | C3 | distance from the defect hull that is kept |
| `charge_correction` [lz] | C3 | `lz`, `mp` or `none` |
| `require_gap_states` [none], `require_spin`, `require_bound` [false] | C3 | property filters |
| `distort_charges` [stable] | C3 | `stable` or `screened` |
| `zpl_window` | C3b | e.g. `0.8 3.5 eV` |
| `distortions` [−0.6 … 0.6] | C4 | bond-distortion factors |
| `distort_around` [anchor] | C4 | `anchor` or `vacancy` |
| `rattle_stdev`, `rattle_dmin` | C4 | rattling (default 10 % and 80 % of the host bond) |
| `%block oxidation_states` | C4 | e.g. `Ca +2`, `O -2`, `Sb +3` |
| `distortion_min_gain` [0.05 eV] | C6 | when a distortion counts as found |
| `share_rounds` [2] | C6 | rounds of sharing between charge states |
| `keep_structures` [1] | C6 | structures per charge state for phase D |

Runner settings: `%block runner_screen`.

---

## Output

In `<workdir>/C_screen/`:

| File | From | Contents |
|---|---|---|
| `structures/`, `prerelax/` | C1 | the starting structures; MACE caches |
| `screen/`, `screen_structures/`, `screen.json` | C2 | the jobs; final structures; every run's results and the host's band edges |
| `filter.txt`, `filter.json` | C3 | every candidate's verdict; the survivors |
| `zpl.txt`, `zpl.json` | C3b | the ZPL estimates and verdicts (with `zpl_window`) |
| `distortions/`, `distort.json` | C4 | the distorted structures |
| `drelax/`, `drelax_structures/`, `drelax.json` | C5 | their relaxations |
| `share_<n>/`, `rank.txt`, `rank.json` | C6 | sharing jobs; the ranking; the structures for phase D |

---

## Notes and limitations

- **Screening energies are only for ranking**: Γ point only, possibly a lower cut-off, and
  relaxations that may stop at `screen_max_steps`. Choose `hull_margin` to allow for their
  error; phase D recomputes everything at the production settings.
- **Gap levels are not aligned**: they are compared with the pristine supercell's Γ-point band
  edges at the same settings. Good enough to sort candidates, not for transition levels (E2).
- **Spins:** CASTEP starts each run from the spin given and may change it (with
  `fix_occupancy : false` it is held for the first `spin_fix` SCF cycles, 10 by default; with
  `fix_occupancy : true` the occupancies, and so the spin, stay as set). Both starting spins can
  end in the same state; the results record the final one.
- **A charge state with no usable run** (all failed) is missing from that candidate's hull;
  a candidate with none at all is dropped ("no usable screening run"). The C2 note says how many
  runs failed: look at their `.castep`/`.err` files and rerun with `--resubmit-failed`.
- **Cost:** C2 runs (number of candidates) × (charge states) × (1 or 2 spins) relaxations,
  and C5 (survivors) × (charge states) × (distortions) more. Triage, a tighter
  `hull_margin` and fewer `distortions` are the main levers.
