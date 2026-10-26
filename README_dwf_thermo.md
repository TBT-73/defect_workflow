# dwf_thermo.py — phase E: thermodynamics

Turns the production energies of phase D into formation energies, transition levels and (if you
ask) the self-consistent Fermi level and defect concentrations, and lists the defects worth the
optical lineshape calculations of phase F. Flowchart boxes E1 to E3. No CASTEP jobs: it runs in
seconds once phase D has finished.

```bash
python3 dwf_thermo.py CaO_Sb.dwf
python3 dwf_thermo.py CaO_Sb.dwf formation       # just E2 again, e.g. after changing gap
```

---

## Contents

1. [Steps](#steps)
2. [E1 corrections](#e1-corrections)
3. [E2 formation](#e2-formation)
4. [E3 interest](#e3-interest)
5. [Keywords](#keywords)
6. [Output](#output)
7. [Notes and limitations](#notes-and-limitations)

---

## Steps

| Label | Step | What it does | Decision |
|---|---|---|---|
| E1 | `corrections` | finite-size correction for each charge state that reached D | — |
| E2 | `formation` | formation energies at every chemical-potential limit, transition levels, plot, Fermi level | — |
| E3 | `interest` | which defects go on to phase F | a deep transition level, or an internal transition found in C |

---

## E1 corrections

For each charge state q that reached phase D, the image-charge correction of the B5 supercell
with the static dielectric constant ε₀ from A2.3 (`supercell_convergence.py`'s
`charge_correction`):

- `mp` (Makov–Payne, first order): E₁ = q²α_M / (2ε₀L), a point charge with its periodic images
  in a neutralising background;
- `lz` (Lany–Zunger, the default): [1 + c_sh (1 − 1/ε₀)] E₁, with the shape factor c_sh of the
  supercell (simple cubic, FCC, BCC; about −1/3 for other shapes);
- `none`: 0.

No potential alignment and no dipole or quadrupole terms. The values are in `corrections.txt`.

---

## E2 formation

For each defect that reached phase D, its lowest production energy in each charge state gives

  E_f(q, E_F) = E_D − E_H − Σᵢ nᵢ μᵢ + q (E_VBM + E_F) + E_corr

with E_D and E_H (the pristine supercell) from D1, nᵢ the atoms added (removed: negative), μᵢ
from each A3.2 limit (absolute, on CASTEP's energy scale), E_VBM from A2.2 and E_corr from E1.
Without chemical-potential limits (no competing phases), the μ terms are left out and only
defects of one composition can be compared.

- **Transition levels**: walking up the Fermi level from the VBM to the gap, every point where
  the defect's lowest-energy charge state changes, written as (q/q′) and its energy above the
  VBM. They don't depend on the limit. The gap is `gap` if given (e.g. experimental or hybrid),
  else A2.2's.
- **`formation.txt`**: E_f at the VBM and at the CBM for every limit, defect and charge state,
  with the run it came from; **`levels.txt`**: the transition levels;
  **`formation.png`**: one panel per limit, each defect's lowest charge state against E_F, with
  its transition levels marked (at most eight defects; the tables have them all).
- **Fermi level** (with `fermi_temperature`): at each limit, the E_F where the charges of the
  defects, free electrons and holes balance. Each defect's concentration in charge state q is
  (sites × orientations × g_q / V) exp(−E_f / kT), with g_q = |final spin| + 1 from phase D (a
  pair of two identical components is counted once). The free carriers come from A2.2's band
  energies on its k-point grid (the conduction bands shifted by the difference if `gap` is set).
  With `quench_temperature`, each defect's total concentration is frozen at
  `fermi_temperature` and shared out again between its charge states at the lower temperature
  (`fermi.txt`).

---

## E3 interest

A defect goes on to phase F (optical lineshapes, not scripted yet) if:

- one of its transition levels lies at least `deep_level_margin` from both band edges (a deep
  level), or
- phase C found an internal transition for it (a first ZPL estimate between occupied and empty
  gap levels, in a stable charge state that met the filters).

`interest.txt` gives the reason for each defect; `for_F.json` lists the ones picked.

---

## Keywords

Defaults in brackets.

| Keyword | Step | Meaning |
|---|---|---|
| `charge_correction` [lz] | E1 | `lz`, `mp` or `none` |
| `gap` [A2.2's] | E2 | band gap for the E_F range, the levels and the carriers |
| `fermi_temperature` | E2 | temperature (K) for the self-consistent Fermi level |
| `quench_temperature` | E2 | freeze the concentrations at `fermi_temperature`, then this |
| `deep_level_margin` [0.5 eV] | E3 | how far from both edges a "deep" level must be |

---

## Output

In `<workdir>/E_thermo/`:

| File | From | Contents |
|---|---|---|
| `corrections.txt` | E1 | E_corr for each charge |
| `formation.txt`, `formation.json` | E2 | formation energies (and everything else of E2, as JSON) |
| `levels.txt` | E2 | transition levels |
| `formation.png` | E2 | the plot |
| `fermi.txt` | E2 | Fermi level, electrons and holes at each limit (with `fermi_temperature`) |
| `interest.txt`, `for_F.json` | E3 | which defects go on to phase F, and why |

---

## Notes and limitations

- **Only the defects that reached phase D** enter the Fermi level. Native defects dropped in
  phase C (or never screened) can pin it elsewhere; include the important ones as candidates if
  the Fermi level matters.
- **No potential alignment** in the corrections, and the correction uses the average of ε₀'s
  diagonal. For strongly anisotropic hosts or large corrections, check against a full scheme
  (e.g. Freysoldt–Neugebauer–Van de Walle).
- **The band edges** come from A2.2's uniform Γ-centred grid; give `gap` if they are not the
  true ones (or to use an experimental or hybrid gap).
- **Semi-local DFT gaps are too small**: transition levels near the CBM and the carrier densities
  inherit that, even with `gap` set (the defect energies themselves are not corrected).
- Changing `gap`, `fermi_temperature` or `quench_temperature` reruns E2 and E3 only; nothing is
  submitted.
