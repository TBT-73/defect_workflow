#!/usr/bin/env python3
"""
dwf_thermo.py - phase E of the defect workflow: thermodynamics

Steps:
  E1  corrections  finite-size correction of every production run: Lany-Zunger (default)
                   or Makov-Payne with the A2.3 dielectric constant (charge_correction)
  E2  formation    formation energies of each defect and charge state at every
                   chemical-potential limit (A3.2), transition levels, plots, and with
                   fermi_temperature the self-consistent Fermi level and concentrations
                   (optionally quenched to quench_temperature)
  E3  interest     which defects go on to the optical lineshape (phase F, not scripted
                   yet): a transition level deeper than deep_level_margin from both band
                   edges, or an internal transition found in phase C

  E_f(q, E_F) = E_defect - E_host - sum_i n_i mu_i + q (E_VBM + E_F) + E_corr
with E_defect and E_host from phase D, E_VBM from A2.2 and mu_i from A3.2.

Usage:  python3 dwf_thermo.py <project>.dwf [steps] [--redo STEP] [--status] [--example]
"""

import math

import numpy as np

import dwf_core as core
from dwf_core import die, log

# ==========================================================================================
# What each step reads and writes (under <workdir>/E_thermo/):
#   E1 corrections  D1, A2.3, B5         -> corrections.txt
#   E2 formation    E1, D1, A2.2, A3.2   -> formation.txt, levels.txt, formation.png,
#                                           fermi.txt (fermi_temperature), formation.json
#   E3 interest     E2, C3b              -> interest.txt, for_F.json
# ==========================================================================================

KB = 8.617333262e-5                # eV/K
# chart colours (reference palette: light surface, categorical slots in their fixed order)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


# The production results (D1) of the runs that finished, by candidate and charge state, keeping
# the lowest energy of each (with the run it came from).
#   Called by: step_corrections(), step_formation()
def best_production(p):
    prod = core.read_json(p, p.out("production")["results"])
    best = {}
    for r in prod["runs"]:
        if r["energy"] is None:
            continue
        cur = best.setdefault(r["cand"], {}).get(r["q"])
        if cur is None or r["energy"] < cur["energy"]:
            best[r["cand"]][r["q"]] = r
    return prod["host_energy"], best


# E1: the finite-size correction (charge_correction, with the static dielectric constant from
# A2.3) for every charge state that reached phase D, in E_thermo/corrections.txt.
#   Called by: dwf_core.run_phase() (step E1)
#   Calls:     best_production(), dwf_core.Project.host(), dwf_core.correction(),
#              dwf_core.host_supercell(), dwf_core.write_table()
def step_corrections(p):
    _, best = best_production(p)
    prim, P = p.host(), np.array(p.out("choose")["matrix"])
    cell = core.host_supercell(prim, P).cell
    eps = p.out("dielectric").get("eps0")
    qs = sorted({q for st in best.values() for q in st})
    corr = {str(q): core.correction(q, cell, eps, p.s.charge_correction) for q in qs}
    rows = [[q, corr[str(q)]] for q in qs]
    print(core.write_table(p.phase_dir("E") / "corrections.txt", ["q", "E_corr/eV"], rows,
                           f"E1 {p.s.charge_correction} corrections (epsilon {eps})"))
    return {"corrections": corr, "method": p.s.charge_correction, "eps0": eps,
            "_note": ", ".join(f"q {q:+d}: {corr[str(q)]:.3f} eV" for q in qs if q)}


# The band gap used for the Fermi-level range: gap from the project file (e.g. experimental or
# hybrid) or the A2.2 one.
#   Called by: step_formation()
def band_gap(p):
    return p.s.gap or p.out("bands")["gap"]


# The chemical-potential limits (A3.2) as {name: {element: mu}}; without any, one entry 'mu=0' with
# no chemical-potential terms (energies are then only comparable within one composition).
#   Called by: step_formation()
#   Calls:     dwf_core.Project.finished()
def limits(p):
    lim = p.out("limits")["limits"] if p.finished("limits") else {}
    return {k: v["mu"] for k, v in lim.items()} or {"mu=0": {}}


# The thermodynamic transition levels of one defect: walking up the Fermi level from 0 to the gap,
# each point where its lowest-energy charge state changes, as (q_before, q_after, level).
#   Called by: step_formation()
def transition_levels(ef0, gap):
    qs = sorted(ef0)
    out, cur, x = [], None, 0.0
    while True:
        cur = min(qs, key=lambda q: ef0[q] + q * x)
        nxt = None
        for q in qs:
            if q < cur:                              # lower charges take over as E_F rises
                level = (ef0[q] - ef0[cur]) / (cur - q)
                if level > x + 1e-9 and (nxt is None or level < nxt[1]):
                    nxt = (q, level)
        if nxt is None or nxt[1] >= gap:
            return out
        out.append((cur, nxt[0], nxt[1]))
        x = nxt[1]


# The free carriers at Fermi level mu (absolute, eV) and temperature T, per cm^3, from the host's
# A2.2 band energies (k-point weights; a scissor shift of the conduction bands if gap is set):
# (electrons, holes).
#   Called by: fermi_level()
#   Calls:     dwf_core.occupied_bands()
def carriers(bands, mu, T, shift, volume_cm3):
    occ, eig, w = core.occupied_bands(bands), bands["eig"], bands["weights"]
    g = 2.0 if bands["nspin"] == 1 else 1.0
    n = h = 0.0
    for s in range(bands["nspin"]):
        cb, vb = eig[s, :, occ[s]:] + shift, eig[s, :, :occ[s]]
        n += g * np.nansum(w[:, None] / (1.0 + np.exp(np.clip((cb - mu) / (KB * T), -500, 500))))
        h += g * np.nansum(w[:, None] / (1.0 + np.exp(np.clip((mu - vb) / (KB * T), -500, 500))))
    return n / volume_cm3, h / volume_cm3


# The self-consistent Fermi level: the E_F (above the VBM) where the defect charges, free electrons
# and holes balance, found by bisection. Each defect's concentration per cm^3 in charge state q is
# sites x orientations x (spin degeneracy) / V x exp(-E_f / kT), or, with fixed totals (quenched),
# its total shared out between charge states by the same Boltzmann weights. Returns (E_F, n, p,
# {cand: {q: concentration}}).
#   Called by: step_formation()
#   Calls:     carriers()
def fermi_level(defects, bands, vbm, gap, T, shift, vol_cm3, totals=None):
    # The net charge per cm^3 at Fermi level x (eV above the VBM), and the concentrations.
    def charge(x):
        conc = {}
        for cid, d in defects.items():
            wts = {q: d["g"][q] * math.exp(-min(700.0, (d["ef0"][q] + q * x) / (KB * T)))
                   for q in d["ef0"]}
            if totals is None:
                conc[cid] = {q: d["sites"] / vol_cm3 * wt for q, wt in wts.items()}
            else:
                z = sum(wts.values()) or 1e-300
                conc[cid] = {q: totals[cid] * wt / z for q, wt in wts.items()}
        n, h = carriers(bands, vbm + x, T, shift, vol_cm3)
        return sum(q * c for cc in conc.values() for q, c in cc.items()) + h - n, n, h, conc

    lo, hi = -1.0, gap + 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if charge(mid)[0] > 0:
            lo = mid
        else:
            hi = mid
    x = 0.5 * (lo + hi)
    _, n, h, conc = charge(x)
    return x, n, h, conc


# Draws the formation energies against the Fermi level, one panel per chemical-potential limit
# (shared y-axis), each defect's lowest charge state as a line with its transition levels marked.
# At most eight defects (the palette's slots in order); the rest are in the tables.
#   Called by: step_formation()
def plot(path, curves, gap, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("note: matplotlib is not installed, so no plot")
        return None
    names = list(next(iter(curves.values())))[:8]
    fig, axes = plt.subplots(1, len(curves), figsize=(4.2 * len(curves) + 1.6, 4.4), sharey=True,
                             squeeze=False, facecolor=SURFACE)
    x = np.linspace(0, gap, 600)
    for ax, (lim, defs) in zip(axes[0], curves.items()):
        ax.set_facecolor(SURFACE)
        for i, name in enumerate(names):
            ef0, levels = defs[name]
            y = np.min([ef0[q] + q * x for q in ef0], axis=0)
            ax.plot(x, y, color=SERIES[i], lw=2, label=name, solid_capstyle="round")
            for _, _, lv in levels:
                ax.plot([lv], [min(ef0[q] + q * lv for q in ef0)], "o", ms=6, color=SERIES[i],
                        mec=SURFACE, mew=1.5, zorder=3)
        ax.set_xlim(0, gap)
        ax.set_title(lim, color=INK, fontsize=10)
        ax.set_xlabel("Fermi level above the VBM (eV)", color=INK2, fontsize=9)
        ax.grid(color=GRID, lw=0.8)
        ax.tick_params(colors=INK2, labelsize=8)
        for s in ax.spines.values():
            s.set_color(GRID)
    axes[0][0].set_ylabel("formation energy (eV)", color=INK2, fontsize=9)
    leg = axes[0][-1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
                             fontsize=8)
    for t in leg.get_texts():
        t.set_color(INK)
    fig.suptitle(title, color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


# E2: formation energies, transition levels and (with fermi_temperature) the Fermi level.
# For each defect that reached phase D, its lowest energy in each charge state gives
# E_f(q, E_F=0) = E_D - E_host - sum n_i mu_i + q E_VBM + E_corr at every limit; its thermodynamic
# transition levels in the gap (transition_levels; the same at every limit) go in levels.txt, the
# energies in formation.txt and the plot formation.png. With fermi_temperature, fermi_level finds
# the self-consistent Fermi level at each limit from these defects and the host's free carriers
# (quench_temperature: concentrations frozen at fermi_temperature, then charges redistributed).
#   Called by: dwf_core.run_phase() (step E2)
#   Calls:     band_gap(), best_production(), fermi_level(), limits(), plot(), transition_levels(),
#              dwf_core.Project.host(), dwf_core.read_bands(), dwf_core.write_table()
def step_formation(p):
    d = p.phase_dir("E")
    host_e, best = best_production(p)
    if not best:
        die("no production run finished, so there are no formation energies")
    cands = {c["id"]: c for c in core.read_json(p, p.out("generate")["candidates"])}
    corr = {int(q): v for q, v in p.out("corrections")["corrections"].items()}
    vbm, gap = p.out("bands")["vbm"], band_gap(p)
    lims = limits(p)
    table, curves, levels_rows, results = [], {}, [], {}
    for lim, mu in lims.items():
        curves[lim] = {}
        for cid in sorted(best):
            c = cands[cid]
            missing = [el for el in c["delta"] if mu and el not in mu]
            if missing:
                die(f"limit {lim} has no chemical potential for " + ", ".join(missing))
            dmu = sum(n * mu.get(el, 0.0) for el, n in c["delta"].items())
            ef0 = {q: r["energy"] - host_e - dmu + q * vbm + corr.get(q, 0.0)
                   for q, r in best[cid].items()}
            lv = transition_levels(ef0, gap)
            curves[lim][f"{cid} {c['name']}"] = (ef0, lv)
            results.setdefault(cid, {"name": c["name"], "levels": [list(x) for x in lv],
                                     "formation": {}})["formation"][lim] = {str(q): v
                                                                           for q, v in ef0.items()}
            for q in sorted(ef0):
                table.append([lim, cid, c["name"], f"{q:+d}", ef0[q], ef0[q] + q * gap,
                              best[cid][q]["job"]])
    for cid in sorted(best):
        lv = results[cid]["levels"]
        levels_rows.append([cid, cands[cid]["name"],
                            ", ".join(f"({a:+d}/{b:+d}) {e:.3f}" for a, b, e in lv) or
                            "one charge state across the gap"])
    print(core.write_table(d / "formation.txt", ["limit", "id", "defect", "q", "E_f(VBM)/eV",
                                                 "E_f(CBM)/eV", "from run"], table,
                           f"E2 formation energies (E_VBM {vbm:.4f} eV, gap {gap:.3f} eV)"))
    print(core.write_table(d / "levels.txt", ["id", "defect", "transition levels (eV above VBM)"],
                           levels_rows, "E2 thermodynamic transition levels in the gap"))
    png = plot(d / "formation.png", curves, gap, f"{p.name}: formation energies")
    fermi = {}
    if p.s.fermi_temperature:
        bands = core.read_bands(p.dir / p.out("bands")["bands_file"])
        prim = p.host()
        vol = prim.get_volume() * 1e-24
        shift = gap - p.out("bands")["gap"]
        for lim in lims:
            defects = {}
            for cid in best:
                c = cands[cid]
                # a pair of two identical components is counted from both ends: halve it
                same = len(c["comps"]) == 2 and all(c["comps"][0].get(k) == c["comps"][1].get(k)
                                                    for k in ("new", "host", "prim_site", "frac"))
                sites = c["sites"] * c["multiplicity"] / (2 if same else 1)
                g = {q: (abs(round(r["spin_final"] or 0)) + 1) for q, r in best[cid].items()}
                defects[cid] = {"ef0": {int(q): float(v) for q, v in
                                        results[cid]["formation"][lim].items()},
                                "g": g, "sites": sites}
            T = p.s.fermi_temperature
            x, n, h, conc = fermi_level(defects, bands, vbm, gap, T, shift, vol)
            out = {"T": T, "E_F": x, "electrons": n, "holes": h,
                   "concentrations": {cid: {str(q): v for q, v in cc.items()}
                                      for cid, cc in conc.items()}}
            if p.s.quench_temperature:
                totals = {cid: sum(cc.values()) for cid, cc in conc.items()}
                xq, nq, hq, cq = fermi_level(defects, bands, vbm, gap, p.s.quench_temperature,
                                             shift, vol, totals)
                out["quenched"] = {"T": p.s.quench_temperature, "E_F": xq, "electrons": nq,
                                   "holes": hq,
                                   "concentrations": {cid: {str(q): v for q, v in cc.items()}
                                                      for cid, cc in cq.items()}}
            fermi[lim] = out
        rows = []
        for lim, f in fermi.items():
            rows.append([lim, f"{f['T']:g} K", f["E_F"], f"{f['electrons']:.2e}",
                         f"{f['holes']:.2e}"])
            if "quenched" in f:
                qf = f["quenched"]
                rows.append([lim, f"quenched {qf['T']:g} K", qf["E_F"], f"{qf['electrons']:.2e}",
                             f"{qf['holes']:.2e}"])
        print(core.write_table(d / "fermi.txt", ["limit", "T", "E_F above VBM/eV", "n/cm^-3",
                                                 "p/cm^-3"], rows,
                               "E2 self-consistent Fermi level (only the defects of phase D)"))
        log("note: the Fermi level includes only the defects that reached phase D; defects "
            "left out (e.g. native defects dropped in C) may change it")
    rel = core.write_json(p, d / "formation.json", {"vbm": vbm, "gap": gap, "limits": lims,
                                                     "defects": results, "fermi": fermi})
    return {"results": rel, "plot": str(png.relative_to(p.dir)) if png else None,
            "_note": f"{len(best)} defects at {len(lims)} limit(s)"}


# E3: the decision before phase F. A defect is interesting if one of its transition levels lies
# more than deep_level_margin from both band edges (a deep level), or if phase C found an internal
# transition for it (a first ZPL estimate between occupied and empty gap levels). The list goes to
# E_thermo/for_F.json; phase F (optical lineshapes) is not scripted yet.
#   Called by: dwf_core.run_phase() (step E3)
#   Calls:     dwf_core.write_table()
def step_interest(p):
    form = core.read_json(p, p.out("formation")["results"])
    gap, m = form["gap"], p.s.deep_level_margin
    surv = {s["cand"]: s for s in core.read_json(p, p.out("zpl")["survivors"])}
    picks, rows = [], []
    for cid, dd in sorted(form["defects"].items()):
        why = [f"deep level ({a:+d}/{b:+d}) at {e:.2f} eV" for a, b, e in dd["levels"]
               if m <= e <= gap - m]
        s = surv.get(cid)
        if s:
            zpl = [s["best"][str(q)]["zpl"] for q in s.get("good", [])
                   if s["best"].get(str(q), {}).get("zpl") is not None]
            if zpl:
                why.append(f"internal transition, first estimate {min(zpl):.2f} eV")
        rows.append([cid, dd["name"], "yes: " + "; ".join(why) if why else "no"])
        if why:
            picks.append({"cand": cid, "name": dd["name"], "why": why})
    print(core.write_table(p.phase_dir("E") / "interest.txt", ["id", "defect", "on to F?"], rows,
                           f"E3 deep levels (>= {m:g} eV from both edges) or internal transitions"))
    rel = core.write_json(p, p.phase_dir("E") / "for_F.json", picks)
    return {"for_F": rel, "count": len(picks),
            "_note": f"{len(picks)} defect(s) for the lineshape phase (F, not scripted yet)"}


STEPS = [
    core.Step("corrections", "E1", "finite-size corrections", step_corrections,
              keys=("charge_correction",), needs=("production", "dielectric", "choose"),
              owns=("E_thermo/corrections.txt",)),
    core.Step("formation", "E2", "formation energies, levels, Fermi level", step_formation,
              keys=("gap", "fermi_temperature", "quench_temperature"),
              needs=("corrections", "production", "bands", "limits", "generate"),
              owns=("E_thermo/formation.json", "E_thermo/formation.png",
                    "E_thermo/formation.txt", "E_thermo/levels.txt", "E_thermo/fermi.txt")),
    core.Step("interest", "E3", "which defects go on to the lineshape (F)", step_interest,
              keys=("deep_level_margin",), needs=("formation", "zpl"),
              owns=("E_thermo/interest.txt", "E_thermo/for_F.json")),
]

if __name__ == "__main__":
    core.run_phase("E", STEPS, __doc__)
