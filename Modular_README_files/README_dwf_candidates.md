# dwf_candidates.py — phase B: candidate defects and supercell size

Lists the defects to screen (yours, or generated from the dopants and the host's symmetry) and
chooses the supercell every later phase uses, from a MACE size scan and a CASTEP size series.
Flowchart boxes B1 to B5. Phase A must have finished.

```bash
nohup python3 dwf_candidates.py CaO_Sb.dwf > B.log 2>&1 &
python3 dwf_candidates.py CaO_Sb.dwf generate      # just the candidate list (quick), to check it
```

The options (`--redo`, `--no-wait`, `--prepare-only`, ...) are the same for every phase: see
`README_workflow.md`.

---

## Contents

1. [Steps](#steps)
2. [B1 generate](#b1-generate)
3. [Where a defect goes in the supercell](#where-a-defect-goes-in-the-supercell)
4. [B2 scan](#b2-scan)
5. [B3 series](#b3-series)
6. [B4 finetune](#b4-finetune)
7. [B5 choose](#b5-choose)
8. [Keywords](#keywords)
9. [Output](#output)
10. [Notes and limitations](#notes-and-limitations)

---

## Steps

| Label | Step | What it does | Decision |
|---|---|---|---|
| B1 | `generate` | the candidate defects | `%block defects` given (flowchart "yes") or generate them ("no") |
| B2 | `scan` | MACE relaxations of one defect in growing cells | the size where its relaxation energy levels off caps B3 |
| B3 | `series` | CASTEP size series of that defect (`supercell_convergence.py`) | converged when E_f* of two sizes agree within `series_tol` |
| B4 | `finetune` | optional: MACE training data from B3's CASTEP steps | — |
| B5 | `choose` | the supercell for phases C and D | the smaller (or larger) of the two converged sizes, or `supercell` |

With `supercell : n`, B2 and B3 are skipped and B5 takes that size (`finetune : true` then
stops at B4, which needs the series).

---

## B1 generate

**Your defects (`%block defects`).** One defect per line, components joined by `+`:

```
%block defects
Sb_Ca                  # Sb on a Ca site
Sb_Ca + Vac_O          # the partner on its nearest O site to the Sb
Sb_Ca + Vac_O   all    # every arrangement up to pair_max_distance
Vac_O:2                # the 2nd O site of the primitive cell (for inequivalent sites)
Sb_Ca@2                # with a starting spin of 2 on the Sb
%endblock defects
```

The tokens are those of `supercell_convergence.py` (substitutions `X_Y` and vacancies `Vac_Y`;
interstitials can only be generated). `all` works for pairs: each symmetry-distinct arrangement
becomes a candidate.

**Generated defects** (no `%block defects`). On each symmetry-inequivalent site of the primitive
cell (spglib):

| Option | Adds |
|---|---|
| `vacancies : true` | a vacancy |
| `dopants : Sb ...` | a substitution by each dopant |
| `antisites : true` | every other host element on it |
| `interstitials : dopants` / `all` | the dopants (or every element) on each Voronoi interstitial site (pymatgen-analysis-defects), labelled `Sb_ia`, `Sb_ib`, ... |
| `pairs : true` | dopant defect + native defect pairs |
| `native_pairs : true` | native + native pairs as well |
| `dopant_pairs : true` | dopant + dopant pairs as well |

Pairs never have two interstitials. Each pair type gives every symmetry-distinct arrangement up
to `pair_max_distance` apart: two arrangements are the same if a symmetry operation that keeps
the first component in place turns one into the other (for two identical components, also turned
round). An interstitial must be at least 0.5 Å from the other component (1 Å from a vacancy). For
each arrangement the table records how many equivalent orientations it has and how many sites of
its first component the primitive cell has; E2 uses both for concentrations.

**Composition groups.** Each candidate gets its change in composition, e.g. `+Sb-Ca-O` for
Sb_Ca + Vac_O. Candidates with the same change (e.g. Sb_Ca + Vac_O and Sb_O + Vac_Ca) are
directly comparable, which phase C's defect hull uses.

**Ids and names.** `d001`, `d002`, ... sorted by number of components, composition, separation
and name. A site number (`Vac_O:2`) is added only when the element has inequivalent sites.

How many candidates you get grows quickly: each extra option multiplies the pair types, and
each pair type has more arrangements the larger `pair_max_distance` is. Run
`python3 dwf_candidates.py <project> generate` and read `candidates.txt` before going on: every
candidate costs about 2 × (number of `screen_charges`) screening runs in phase C.

---

## Where a defect goes in the supercell

Every defect cell of the workflow (B2, B3, C1) is built the same way:

1. the first component (the anchor) and the vectors from it to the other components fix the
   complex;
2. the complex is put as close to the cell centre as it can be: of all the places the anchor can
   go (copies of its primitive-cell site, and of the sites a pure translation of the crystal
   makes identical to it, such as all four Ca of a conventional rocksalt cell), the one that
   puts the mean position of the components nearest the centre; ties go to the one whose
   farthest component is nearest the centre;
3. the complex is only ever moved, never turned, so it is identical in every cell size (B3's
   size series picks its own orientation, the same at every size: see [B3 series](#b3-series)).

`supercell_convergence.py` prints the distance between the defect's centre and the cell centre
for each size in its size table (`off-centre/A`). It is not always zero: a defect can only sit
on lattice sites, e.g. a Ca–O pair's mean position is half a bond from the nearest site.

---

## B2 scan

With `mace : true` and `scan : true`: the representative defect (below) is built in each size of
`scan_sizes` (n × `transform`) and relaxed with MACE (positions only). For each size the step
records the relaxation energy (unrelaxed − relaxed MACE energy) and the largest displacement of
the atoms more than 35 % of the shortest image distance from the defect (`far-field move`).

**Decision:** the scan stops at the first size whose relaxation energy is within `scan_tol` of
the size before; that size caps the CASTEP series. If it never levels off there is no cap (a
note says so).

**The representative defect** is `series_defect` if given (tokens, e.g. `Sb_Ca@1 + Vac_O`),
otherwise the nearest arrangement of the first pair that contains a dopant (otherwise the first
candidate). An interstitial can't be the representative: set `series_defect` or `supercell`.

---

## B3 series

The size series is a normal `supercell_convergence.py` run in `B_candidates/series/`. The step
writes its input, `series.conv`, from the project:

- `cell`: the relaxed host (`A_host/host_relaxed.cell`); `param`: `series.param` (the production
  `.param` with the series' charge and spin); `e_bulk` from A2.1; `kspacing` from A1;
  `transform`; `energy`;
- `sizes`: `series_sizes` up to the B2 cap (at least two); `tol`: `series_tol`;
  `singlepoint : false`;
- `series_charge`, with `charge_correction` and the A2.3 dielectric constant if it is not 0;
- with `mace : true`, `prerelax : mace` and the project's MACE settings;
- the runner settings: the project's runner keywords and `%block runner_series`.

This is the only step in which `supercell_convergence.py` builds its own defect cells; every
other CASTEP step runs structures the scripts build. They are the same cells as B2 and C1 use:
the same relaxed host, `transform` and centring, and the tokens name the primitive-cell site each
component sits on. The complex can be a rotated, symmetry-equivalent copy of the candidate's,
because the series picks the direction to the nearest site of each kind by its own rule. That is
the same calculation unless the supercell shape breaks the host's symmetry (see "How the scripts
use supercell_convergence.py" in `README_workflow.md`). Only the chosen size and B4's training
data are used later; the series' structures and energies are not.

A charged or odd-electron defect runs with `spin_polarized : true` and a starting spin (the
lowest its electron count allows, also put on the first component as `@s`); a neutral
even-electron one keeps your `.param`'s `spin_polarized`. The spin is a starting value: CASTEP is
free to change it.

`supercell_convergence.py` runs as a subprocess (its output appears in `B.log`) and the step
reads its `run/convergence.json`. `--no-wait`, `--prepare-only`, `--force` and
`--resubmit-failed` are passed on to it.

**Decision:** converged when the last two sizes' E_f* differ by less than `series_tol`. If not,
B5 stops and asks for bigger sizes (or a looser `series_tol`, or `supercell`). Add the sizes and
rerun: finished calculations are reused.

---

## B4 finetune

Optional (`finetune : true`). Every geometry step of the series' smallest-cell relaxation (all
its runs) with an energy and forces goes into `B_candidates/finetune/train.xyz` (`REF_energy`,
`REF_forces`); steps with a force above 2 eV/Å are left out. `finetune.sh` runs `mace_run_train`
on it, starting from `mace_model` (`finetune_args` adds options). Run it yourself, on a GPU
node if you can:

```bash
cd dwf_CaO_Sb/B_candidates/finetune && ./finetune.sh
```

then put the model it writes in the project file:

```
finetuned_model : dwf_CaO_Sb/B_candidates/finetune/dwf_finetuned.model
```

C1 then uses it (and runs again if it already ran). B2 and B3 keep `mace_model`, so they don't.

One relaxation is a small training set: compare a few MACE and CASTEP energies before trusting
the fine-tuned model more than the foundation one.

---

## B5 choose

`supercell : n` if given; otherwise the size series' choice: the smaller of the two sizes that
agreed (`series_choice : smaller`, the default) or the larger (`larger`). The step records the
size, its matrix, its number of atoms and its shortest image distance. Every arrangement of a
composition is built in this one cell, so their energies can be compared directly.

---

## Keywords

Defaults in brackets.

| Keyword | Step | Meaning |
|---|---|---|
| `dopants` | B1 | the extrinsic elements |
| `%block defects` | B1 | your own defects (skips generation) |
| `vacancies` [true], `antisites` [false] | B1 | native single defects |
| `interstitials` [none] | B1 | `dopants` or `all` (Voronoi sites) |
| `pairs` [true], `native_pairs` [false], `dopant_pairs` [false] | B1 | which pairs |
| `pair_max_distance` [5.0 Å] | B1 | longest pair separation |
| `scan` [true], `scan_sizes` [2 3 4 5], `scan_tol` [0.02 eV] | B2 | the MACE size scan |
| `series_defect` | B2, B3 | the representative defect, as tokens |
| `series_charge` [0] | B3 | its charge |
| `series_sizes` [2 3 4], `series_tol` [0.05 eV] | B3 | the CASTEP sizes and tolerance |
| `series_choice` [smaller] | B5 | `smaller` or `larger` of the converged pair |
| `supercell` | B3–B5 | use this size and skip the series |
| `finetune` [false], `finetune_args` | B4 | fine-tuning data and script |
| `finetuned_model` | C1 | the fine-tuned model file |
| `transform` | B2–B5 | supercell shape (n × transform) |
| `mace` [true], `mace_model`, `mace_fmax`, `mace_steps`, `mace_threads` | B2, B3 | MACE settings |

Runner settings for the series: `%block runner_series`.

---

## Output

In `<workdir>/B_candidates/`:

| File | From | Contents |
|---|---|---|
| `candidates.txt`, `candidates.json` | B1 | id, name, composition, separation, orientations, sites per cell (and the components) |
| `scan.txt`, `scan.json`, `scan/` | B2 | relaxation energy and far-field move per size; MACE caches |
| `series/series.conv`, `series.param`, `series/run/` | B3 | the size series (its `convergence.txt` is the result) |
| `finetune/train.xyz`, `finetune/finetune.sh` | B4 | fine-tuning data and script |

B5's choice is in `state.json` (`--status`).

---

## Notes and limitations

- **The size series uses one defect**, at one charge state. A charged series needs the A2.3
  dielectric constant for its correction; E_f* of a charged defect converges more slowly
  (about as 1/L) even after correction.
- **A pair farther apart than half the shortest image distance of the chosen cell** is left out
  in C1 (it would interact with its own image more than with its partner).
- **Interstitials** come only from pymatgen's Voronoi generator; check their positions in
  `C_screen/structures/` (C1) before screening many of them.
- **`%block defects` and generation are exclusive**: with the block, nothing is generated.
- Generation needs spglib (and pymatgen-analysis-defects for interstitials); a host cell that is
  not exactly symmetric (e.g. slightly distorted by a relaxation without symmetry) gives more
  inequivalent sites than expected.
