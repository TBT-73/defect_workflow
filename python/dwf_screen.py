#!/usr/bin/env python3
"""
dwf_screen.py - phase C of the defect workflow: screening funnel and ground-state search

Steps (run in this order; each is skipped once it has finished with the same inputs):
  C1   prerelax  build every candidate in the chosen supercell and relax it with MACE
                 (positions only); with triage : true, drop arrangements more than
                 triage_window above the lowest of the same composition
  C2   screen    one cheap CASTEP relaxation per candidate, charge state and starting spin
                 (Gamma point, screen_cutoff, screen_max_steps, no restarts), plus the
                 pristine supercell at the same settings
  C3   filter    defect hull per composition (keep arrangements within hull_margin of the
                 lowest formation energy somewhere in the gap), then the property filters
                 (require_gap_states, require_spin, require_bound)
  C3b  zpl       optional (zpl_window): drop survivors whose first estimate of the
                 zero-phonon line (smallest occupied -> empty step between gap levels in one
                 spin channel) is outside the window
  C4   distort   ShakeNBreak bond distortions and rattling of each survivor, for each
                 charge state of interest, from its screened structure
  C5   drelax    cheap CASTEP relaxations of the distortions (C2 settings)
  C6   rank      rank by energy, merge identical structures, try a distortion that won in
                 one charge state in the others (up to share_rounds), keep the lowest
                 keep_structures per charge state for phase D

Usage:  python3 dwf_screen.py <project>.dwf [steps] [--redo STEP] [--status] [--no-wait]
        [--prepare-only] [--resubmit-failed] [--force] [--example]
"""

import numpy as np

import dwf_core as core
from dwf_core import die, log, sc

# ==========================================================================================
# What each step reads and writes (under <workdir>/C_screen/):
#   C1 prerelax   B1, B5                 -> structures/<id>.json (MACE caches in prerelax/)
#   C2 screen     C1                     -> screen/ (jobs), screen.json, screen_structures/
#   C3 filter     C2, A2.3               -> filter.json / .txt (best_states, relative_formation,
#                                           duplicates)
#   C3b zpl       C3                     -> zpl.json / .txt
#   C4 distort    C3b, C1                -> distortions/<job>.json, distort.json
#   C5 drelax     C4                     -> drelax/ (jobs), drelax.json, drelax_structures/
#   C6 rank       C5 (+ share_<n>/ jobs) -> rank.json / .txt: the structures for phase D
# ==========================================================================================


# The candidates (B1) and the chosen supercell (B5): (candidates by id, primitive host, matrix).
#   Called by: step_distort(), step_filter(), step_prerelax(), step_screen()
#   Calls:     dwf_core.Project.host()
def setup(p):
    cands = {c["id"]: c for c in core.read_json(p, p.out("generate")["candidates"])}
    return cands, p.host(), np.array(p.out("choose")["matrix"])


# The .param keywords of the screening runs: screen_cutoff, screen_max_steps and
# %block screen_overrides.
#   Called by: screen_job(), step_screen()
def screen_overrides(p):
    o = {"geom_max_iter": p.s.screen_max_steps}
    if p.s.screen_cutoff:
        o["cut_off_energy"] = f"{p.s.screen_cutoff:g} eV"
    o.update(p.s.screen_overrides)
    return o


# A screening or distortion job for one structure in charge state q, starting with spin S
# (spin_polarized, spin : S, and SPIN=S on the anchor atom when there is one): a Gamma-point
# relaxation without symmetry, with the screening keywords.
#   Called by: run_screen_jobs(), step_screen()
#   Calls:     screen_overrides()
def screen_job(p, name, atoms, info, q, S):
    a = atoms.copy()
    mags = np.zeros(len(a))
    if S and info.get("anchor_index") is not None:
        mags[info["anchor_index"]] = S
    a.set_initial_magnetic_moments(mags)
    o = dict(screen_overrides(p), charge=q, spin=S, spin_polarized="true")
    return {"name": name, "atoms": a, "kind": "geom", "kgrid": [1, 1, 1], "symmetry": False,
            "overrides": o}


# What a finished screening-type job gave: energy (None unless the last SCF converged), whether
# its geometry converged, final spin, defect levels in the gap and the first ZPL estimate (from its
# .bands against the host's band edges at the same settings), and its final structure, saved in
# structdir.
#   Called by: run_screen_jobs(), step_screen()
#   Calls:     dwf_core.gap_levels(), dwf_core.read_bands(), dwf_core.save_structure(),
#              dwf_core.zpl_estimate(), sc.final_atoms()
def job_summary(p, res, name, edges, structdir):
    r, calc = res["jobs"][name], res["calcs"][name]
    out = {"status": r["status"], "reason": r["reason"], "geom_converged": r["geom_converged"],
           "energy": r["energy"] if r["scf_converged"] else None, "spin": r["spin"],
           "steps": r["geom_steps"], "levels": None, "zpl": None, "structure": None}
    bands = calc.dir / f"{calc.seed}.bands"
    if bands.exists() and edges:
        lv = core.gap_levels(core.read_bands(bands), edges[0], edges[1], p.s.level_margin)
        out["levels"], out["zpl"] = lv, core.zpl_estimate(lv)
    if r["energy"] is not None:
        atoms = sc.final_atoms(calc)
        if atoms is not None:
            path = structdir / f"{name}.json"
            core.save_structure(path, atoms, energy=out["energy"])
            out["structure"] = str(path.relative_to(p.dir))
    return out


# --------------------------------------------------------------------------
# C1, C2
# --------------------------------------------------------------------------
# C1: builds every candidate in the B5 supercell (core.build_defect) and, with mace : true, relaxes
# it with MACE (positions only, the cell fixed; finetuned_model if given, else mace_model). A pair
# too far apart for this cell (more than half the shortest image distance) is left out. With
# triage : true (and MACE), arrangements more than triage_window above the lowest MACE energy of
# the same composition are dropped. Each structure is saved in C_screen/structures/<id>.json with
# where its defect is.
#   Called by: dwf_core.run_phase() (step C1)
#   Calls:     setup(), dwf_core.Project.file(), dwf_core.build_defect(), dwf_core.mace_relax(),
#              dwf_core.save_structure()
def step_prerelax(p):
    cands, prim, P = setup(p)
    d = p.phase_dir("C")
    L = p.out("choose")["L_min"]
    structures, skipped, energies = {}, {}, {}
    model = p.file("finetuned_model") if p.s.finetuned_model else None
    for cid, c in cands.items():
        if c["separation"] and c["separation"] > L / 2 - 0.05:
            skipped[cid] = f"components {c['separation']:.2f} A apart, too far for this cell"
            continue
        atoms, info = core.build_defect(prim, P, c["comps"], f"{cid} ({c['name']})")
        e = None
        if p.s.mace:
            log(f"C1: {cid} {c['name']}: MACE pre-relaxation ({len(atoms)} atoms)")
            atoms, e0, e, steps = core.mace_relax(p, atoms, d / "prerelax" / f"{cid}.json",
                                                  model)
        path = d / "structures" / f"{cid}.json"
        core.save_structure(path, atoms, mace_energy=e, **info)
        structures[cid] = str(path.relative_to(p.dir))
        energies[cid] = e
    triaged = []
    if p.s.triage and p.s.mace:
        groups = {}
        for cid in structures:
            groups.setdefault(cands[cid]["group"], []).append(cid)
        for g, ids in groups.items():
            low = min(energies[i] for i in ids)
            for i in ids:
                if energies[i] - low > p.s.triage_window:
                    triaged.append(i)
                    skipped[i] = (f"MACE energy {energies[i] - low:.2f} eV above the lowest "
                                  f"{g} arrangement (triage)")
                    structures.pop(i)
    for cid, why in skipped.items():
        log(f"C1: {cid} {cands[cid]['name']} left out: {why}")
    return {"structures": structures, "skipped": skipped, "mace_energies": energies,
            "_note": f"{len(structures)} candidates ready" + (f", {len(skipped)} left out"
                                                             if skipped else "")}


# C2: the screening pass. One CASTEP relaxation per candidate, charge state (screen_charges) and
# starting spin (the lowest the electron count allows and, with screen_spins : both, that + 2;
# the spin is not held fixed), plus the pristine supercell at the same settings for the band edges
# and host energy. All at the Gamma point with the screening keywords; failures are flagged, not
# rescued (runner group 'screen': max_restarts 0, tolerate_failures). Results, with the gap levels
# of each run, go to C_screen/screen.json.
#   Called by: dwf_core.run_phase() (step C2)
#   Calls:     job_summary(), screen_job(), screen_overrides(), setup(), dwf_core.Project.batch(),
#              dwf_core.Project.production_param(), dwf_core.band_edges(),
#              dwf_core.host_supercell(), dwf_core.load_structure(), dwf_core.qtag(),
#              dwf_core.read_bands(), dwf_core.spin_states()
def step_screen(p):
    cands, prim, P = setup(p)
    d = p.phase_dir("C")
    structs = p.out("prerelax")["structures"]
    host = core.host_supercell(prim, P)
    jobs = [{"name": "host", "atoms": host, "kind": "sp", "kgrid": [1, 1, 1], "symmetry": True,
             "overrides": {k: v for k, v in screen_overrides(p).items() if k != "geom_max_iter"}}]
    plan = {}
    for cid in structs:
        atoms, info = core.load_structure(p.dir / structs[cid])
        for q in p.s.screen_charges:
            for S in core.spin_states(atoms, q, p.s.screen_spins):
                name = f"{cid}_{core.qtag(q)}_s{S}"
                jobs.append(screen_job(p, name, atoms, info, q, S))
                plan[name] = {"cand": cid, "q": q, "spin_start": S}
    res = p.batch("C_screen/screen", jobs, "screen", p.production_param())
    if not res["complete"]:
        return None
    hc = res["calcs"]["host"]
    if res["jobs"]["host"]["status"] != "done" or not (hc.dir / f"{hc.seed}.bands").exists():
        die(f"the pristine screening cell did not finish ({hc.dir}); the band edges and host "
            "energy come from it")
    vbm, cbm = core.band_edges(core.read_bands(hc.dir / f"{hc.seed}.bands"))
    host_out = {"energy": res["jobs"]["host"]["energy"], "vbm": vbm, "cbm": cbm, "gap": cbm - vbm}
    runs = {}
    for name, meta in plan.items():
        runs[name] = dict(meta, **job_summary(p, res, name, (vbm, cbm), d / "screen_structures"))
    bad = sum(r["energy"] is None for r in runs.values())
    rel = core.write_json(p, d / "screen.json", {"host": host_out, "runs": runs})
    return {"results": rel, "host": host_out,
            "_note": f"{len(runs)} runs, gap {cbm - vbm:.2f} eV at Gamma"
                     + (f", {bad} without a usable energy" if bad else "")}


# --------------------------------------------------------------------------
# C3, C3b
# --------------------------------------------------------------------------
# The best screening result of each candidate in each charge state: the lowest energy over its
# starting spins, among runs with a converged last SCF (an unconverged geometry is kept, but
# flagged). Returns {cand: {q: run}}.
#   Called by: step_filter()
def best_states(runs):
    best = {}
    for name, r in runs.items():
        if r["energy"] is None:
            continue
        cur = best.setdefault(r["cand"], {}).get(r["q"])
        if cur is None or r["energy"] < cur["energy"]:
            best[r["cand"]][r["q"]] = dict(r, name=name)
    return best


# The formation energy of a candidate's charge states against the Fermi level, without the host
# energy and chemical potentials (they cancel within one composition): E + E_corr + q (VBM + E_F)
# on the grid ef. Returns {q: array}.
#   Called by: step_filter()
#   Calls:     dwf_core.correction()
def relative_formation(states, ef, vbm, cell, eps, method):
    return {q: s["energy"] + core.correction(q, cell, eps, method) + q * (vbm + ef)
            for q, s in states.items()}


# The arrangements that relaxed into the same structure as another of their composition group: in
# every charge state both have, their best screened structures match (same_structures). Each is
# mapped to the one with the lowest energy, which stays. Returns {duplicate: the one it matches}.
#   Called by: step_filter()
#   Calls:     same_structures(), dwf_core.load_structure()
def duplicates(p, best, groups):
    dup = {}
    for ids in groups.values():
        ids = sorted(ids, key=lambda c: min(s["energy"] for s in best[c].values()))
        for i, a in enumerate(ids):
            if a in dup:
                continue
            for b in ids[i + 1:]:
                common = sorted(set(best[a]) & set(best[b]))
                if b in dup or not common or not all(best[a][q]["structure"] and
                                                     best[b][q]["structure"] for q in common):
                    continue
                if all(len(same_structures([core.load_structure(p.dir / best[x][q]["structure"])[0]
                                            for x in (a, b)])) == 1 for q in common):
                    dup[b] = a
    return dup


# C3: the defect hull and the property filters.
# For each composition group and Fermi level across the screening gap, the lowest formation energy
# of any arrangement and charge state (relative_formation); an arrangement is kept if its own
# lowest is within hull_margin of that somewhere in the gap. Its stable charge states are those
# lowest for it somewhere in the gap. Then, if asked: gap levels (require_gap_states: occupied,
# empty, or both = an occupied and an empty level in one spin channel), the final spin
# (require_spin) and binding against its isolated components (require_bound: the components must
# have been screened too) - each met by at least one stable charge state. Arrangements that relaxed
# into the same structure as a lower one of their group in every charge state both have
# (duplicates) are dropped too. The survivors, with the charge states to carry on
# (distort_charges), go to C_screen/filter.json; every candidate's verdict to filter.txt.
#   Called by: dwf_core.run_phase() (step C3)
#   Calls:     best_states(), duplicates(), relative_formation(), setup(),
#              dwf_core.host_supercell(), dwf_core.write_table(), dwf_core.zpl_estimate()
def step_filter(p):
    cands, prim, P = setup(p)
    d = p.phase_dir("C")
    scr = core.read_json(p, p.out("screen")["results"])
    host, runs = scr["host"], scr["runs"]
    best = best_states(runs)
    eps = p.out("dielectric").get("eps0")
    cell = core.host_supercell(prim, P).cell
    ef = np.linspace(0.0, host["gap"], 401)
    form = {cid: relative_formation(st, ef, host["vbm"], cell, eps, p.s.charge_correction)
            for cid, st in best.items()}
    env = {cid: np.min(np.array(list(f.values())), axis=0) for cid, f in form.items()}
    groups = {}
    for cid in best:
        groups.setdefault(cands[cid]["group"], []).append(cid)
    hull = {g: np.min(np.array([env[c] for c in ids]), axis=0) for g, ids in groups.items()}
    singles = {(c["comps"][0]["new"], c["comps"][0]["host"], c["comps"][0].get("prim_site"),
                tuple(c["comps"][0].get("frac") or ())): cid
               for cid, c in cands.items() if len(c["comps"]) == 1}
    dup = duplicates(p, best, groups)
    left_out = p.out("prerelax")["skipped"]
    survivors, rows = [], []
    for cid in sorted(cands):
        c = cands[cid]
        if cid not in best:
            why = (f"left out in C1: {left_out[cid]}" if cid in left_out
                   else "dropped: no usable screening run")
            rows.append([cid, c["name"], c["group"], "", "", "", "", "", why])
            continue
        dist = float(np.min(env[cid] - hull[c["group"]]))
        lowest = np.argmin(np.array([form[cid][q] for q in sorted(form[cid])]), axis=0)
        stable = sorted({sorted(form[cid])[i] for i in np.unique(lowest)})
        why = [f"relaxed into the structure of {dup[cid]}"] if cid in dup else []
        if dist > p.s.hull_margin:
            why.append(f"{dist:.2f} eV above the hull")
        props = {q: best[cid][q] for q in stable}

        # Whether a charge state has the gap levels asked for.
        def levels_ok(s):
            lv = s.get("levels") or []
            want = p.s.require_gap_states
            return (want == "none" or (want == "occupied" and any(x["occupied"] for x in lv))
                    or (want == "empty" and any(not x["occupied"] for x in lv))
                    or (want == "both" and core.zpl_estimate(lv) is not None))

        # Whether a charge state has the spin asked for.
        def spin_ok(s):
            return (p.s.require_spin is None or
                    (s["spin"] is not None and abs(abs(s["spin"]) - p.s.require_spin) < 0.25))

        good = [q for q, s in props.items() if levels_ok(s) and spin_ok(s)]
        if not good:
            asked = [x for x, on in (("gap levels", p.s.require_gap_states != "none"),
                                     ("spin", p.s.require_spin is not None)) if on]
            why.append("no stable charge state with the " + " and ".join(asked)
                       + " asked for")
        bind = None
        if len(c["comps"]) > 1:
            parts = [singles.get((x["new"], x["host"], x.get("prim_site"),
                                  tuple(x.get("frac") or ()))) for x in c["comps"]]
            if all(pt in env for pt in parts):
                eb = sum(env[pt] for pt in parts) - env[cid] - (len(parts) - 1) * host["energy"]
                bind = (float(eb.min()), float(eb.max()))
        if p.s.require_bound and len(c["comps"]) > 1:
            if bind is None:
                why.append("binding energy unknown (its components were not screened)")
            elif bind[1] <= 0:
                why.append(f"not bound (E_b {bind[0]:.2f} to {bind[1]:.2f} eV)")
        charges = stable if p.s.distort_charges == "stable" else sorted(best[cid])
        spins = ", ".join(f"{q:+d}:{best[cid][q]['spin']:.1f}" if best[cid][q]["spin"] is not None
                          else f"{q:+d}:-" for q in stable)
        zpls = [best[cid][q]["zpl"] for q in stable if best[cid][q]["zpl"] is not None]
        rows.append([cid, c["name"], c["group"], dist, " ".join(f"{q:+d}" for q in stable), spins,
                     f"{min(zpls):.2f}" if zpls else "",
                     f"{bind[0]:.2f}..{bind[1]:.2f}" if bind else "",
                     "kept" if not why else "dropped: " + "; ".join(why)])
        if not why:
            survivors.append({"cand": cid, "name": c["name"], "group": c["group"],
                              "hull_distance": dist, "stable": stable, "charges": charges,
                              "good": good, "binding": bind,
                              "best": {str(q): best[cid][q] for q in best[cid]}})
    print(core.write_table(d / "filter.txt", ["id", "defect", "composition", "above hull/eV",
                                              "stable q", "spin (q:S)", "ZPL est./eV",
                                              "E_b/eV", "verdict"], rows,
                           f"C3 defect hull (margin {p.s.hull_margin:g} eV) and property filters"))
    rel = core.write_json(p, d / "filter.json", survivors)
    return {"survivors": rel, "count": len(survivors),
            "_note": f"{len(survivors)} of {len(cands)} candidates kept"}


# C3b (optional, zpl_window): keeps the survivors with a first ZPL estimate (zpl_estimate, from
# the screening run of a stable charge state that met the property filters) inside zpl_window; one
# with no occupied -> empty step in any spin channel has no internal transition and is dropped.
# Without zpl_window every survivor carries on.
#   Called by: dwf_core.run_phase() (step C3b)
#   Calls:     dwf_core.write_table()
def step_zpl(p):
    survivors = core.read_json(p, p.out("filter")["survivors"])
    if not p.s.zpl_window:
        return {"survivors": p.out("filter")["survivors"], "count": len(survivors),
                "_note": "no zpl_window, so no ZPL filter"}
    lo, hi = p.s.zpl_window
    kept, rows = [], []
    for s in survivors:
        est = [s["best"][str(q)]["zpl"] for q in s["good"] if s["best"][str(q)]["zpl"] is not None]
        inside = [e for e in est if lo <= e <= hi]
        shown = ", ".join(f"{e:.2f}" for e in est)
        verdict = ("kept" if inside else "dropped: " + (f"estimates {shown} eV outside" if est
                                                        else "no internal transition"))
        rows.append([s["cand"], s["name"], ", ".join(f"{e:.2f}" for e in est), verdict])
        if inside:
            kept.append(dict(s, zpl=inside))
    d = p.phase_dir("C")
    print(core.write_table(d / "zpl.txt", ["id", "defect", "ZPL estimates/eV", "verdict"], rows,
                           f"C3b first ZPL estimates, window {lo:g}-{hi:g} eV"))
    rel = core.write_json(p, d / "zpl.json", kept)
    return {"survivors": rel, "count": len(kept),
            "_note": f"{len(kept)} of {len(survivors)} inside the window"}


# --------------------------------------------------------------------------
# C4-C6: ground-state search
# --------------------------------------------------------------------------
# The oxidation states for counting the neighbours to distort: %block oxidation_states, else
# pymatgen's guess for the host and, for each dopant, its common oxidation state closest to that
# of the element it replaces (or its most positive one).
#   Called by: step_distort()
#   Calls:     sc.element()
def oxidation_states(p, prim, cands):
    if p.s.oxidation_states:
        return {sc.element(r[0]): float(r[1]) for r in p.s.oxidation_states}
    try:
        from pymatgen.core import Composition, Element
    except ImportError:
        die("give %block oxidation_states (e.g. Ca +2 / O -2 / Sb +3): guessing them needs "
            "pymatgen")
    guess = Composition(prim.get_chemical_formula()).oxi_state_guesses()
    if not guess:
        die("pymatgen could not guess the host's oxidation states: give %block oxidation_states")
    ox = dict(guess[0])
    for dop in p.s.dopants:
        common = Element(dop).common_oxidation_states or (0,)
        sites = [x["host"] for c in cands.values() for x in c["comps"] if x["new"] == dop]
        target = ox.get(sites[0], 0) if sites and sites[0] else max(common)
        ox[dop] = min(common, key=lambda v: (abs(v - target), -v))
    return ox


# How many neighbours ShakeNBreak's rule distorts for a defect in charge state q: n = sum over the
# components of -(oxidation state put there - that of the site) + q, and |n| neighbours if |n| <= 4,
# else 8 - |n|.
#   Called by: step_distort()
def neighbours_to_distort(comps, q, ox):
    n = q
    for c in comps:
        new = ox.get(c["new"], 0) if c["new"] else 0
        site = ox.get(c["host"], 0) if c["host"] else 0
        n += -(new - site)
    n = int(round(n))
    return abs(n) if abs(n) <= 4 else abs(8 - abs(n))


# The final spin of a charge state's best screening run, as a starting spin with the right parity
# for its electron count.
#   Called by: step_distort(), step_rank()
#   Calls:     dwf_core.electron_parity()
def start_spin(atoms, q, spin):
    par = core.electron_parity(atoms, q)
    if spin is None:
        return par
    s = int(round(abs(spin)))
    return s if s % 2 == par else s + 1


# C4: the distortions. For each survivor and each of its charge states (charges from C3), starting
# from its screened structure in that charge state (or the nearest screened one), ShakeNBreak's
# distort_and_rattle moves the neighbours_to_distort nearest neighbours of the defect (the anchor
# atom, or with distort_around : vacancy / a vacancy anchor, the first vacant site; a defect with
# no vacancy always uses its anchor atom) in or out by each of distortions, and rattles the other
# atoms (rattle_stdev, default 10% of the host bond length; rattle_dmin, 80%); with no neighbours
# to distort it is only rattled. The structures go to C_screen/distortions/.
#   Called by: dwf_core.run_phase() (step C4)
#   Calls:     neighbours_to_distort(), oxidation_states(), setup(), start_spin(),
#              dwf_core.host_supercell(), dwf_core.load_structure(), dwf_core.qtag(),
#              dwf_core.save_structure()
def step_distort(p):
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        from shakenbreak.distortions import distort_and_rattle
    except ImportError as err:
        die(f"the distortion step needs ShakeNBreak (pip install shakenbreak): {err}")
    import warnings
    warnings.filterwarnings("ignore")
    cands, prim, P = setup(p)
    d = p.phase_dir("C") / "distortions"
    survivors = core.read_json(p, p.out("zpl")["survivors"])
    structs = p.out("prerelax")["structures"]
    ox = oxidation_states(p, prim, cands)
    host = core.host_supercell(prim, P)
    bond = float(np.sort(host.get_all_distances(mic=True)[0])[1])
    stdev = p.s.rattle_stdev or 0.1 * bond
    dmin = p.s.rattle_dmin or 0.8 * bond
    jobs = []
    for s in survivors:
        cid = s["cand"]
        _, info = core.load_structure(p.dir / structs[cid])
        screened = {int(q): b for q, b in s["best"].items()}
        for q in s["charges"]:
            src = screened.get(q) or screened[min(screened, key=lambda x: (abs(x - q), x))]
            atoms, _ = core.load_structure(p.dir / src["structure"])
            S = start_spin(atoms, q, src["spin"] if src["q"] == q else None)
            nb = neighbours_to_distort(cands[cid]["comps"], q, ox)
            struct = AseAtomsAdaptor.get_structure(atoms)
            around = info.get("anchor_index")
            centre = {}
            vacancy = (p.s.distort_around == "vacancy" or around is None) and info.get("vacancies")
            if vacancy:                     # (a defect with no vacancy uses its anchor atom)
                centre["frac_coords"] = np.linalg.solve(np.asarray(atoms.cell).T,
                                                        np.asarray(info["vacancies"][0])) % 1.0
            elif around is not None:
                centre["site_index"] = around
            else:
                die(f"{cid}: no anchor atom and no vacancy to distort around")
            factors = [f for f in p.s.distortions if abs(f) > 1e-9] if nb else ["rattled"]
            for f in factors:
                tag = "rattled" if f == "rattled" else f"b{'m' if f < 0 else 'p'}{abs(f):.2f}"
                out = distort_and_rattle(struct, 1.0 if f == "rattled" else 1.0 + f, nb,
                                         stdev=stdev, d_min=dmin, **centre)
                new = AseAtomsAdaptor.get_atoms(out["distorted_structure"])
                new.set_initial_magnetic_moments(atoms.get_initial_magnetic_moments())
                name = f"{cid}_{core.qtag(q)}_{tag}"
                path = d / f"{name}.json"
                core.save_structure(path, new, cand=cid, q=q, spin=S, factor=f, neighbours=nb,
                                    **info)
                jobs.append({"name": name, "cand": cid, "q": q, "spin": S, "factor": f,
                             "neighbours": nb, "structure": str(path.relative_to(p.dir))})
    rel = core.write_json(p, p.phase_dir("C") / "distort.json", jobs)
    return {"jobs": rel, "count": len(jobs), "oxidation_states": ox,
            "_note": f"{len(jobs)} distorted structures for {len(survivors)} survivors"}


# Runs a list of distortion-type jobs (name, structure, q, spin) with the screening settings in
# <subdir>, and returns {name: job_summary} (final structures saved in <subdir>_structures).
#   Called by: step_drelax(), step_rank()
#   Calls:     job_summary(), screen_job(), dwf_core.Project.batch(),
#              dwf_core.Project.production_param(), dwf_core.load_structure()
def run_screen_jobs(p, subdir, todo):
    jobs = []
    for t in todo:
        atoms, info = core.load_structure(p.dir / t["structure"])
        jobs.append(screen_job(p, t["name"], atoms, info, t["q"], t["spin"]))
    res = p.batch(f"C_screen/{subdir}", jobs, "screen", p.production_param())
    if not res["complete"]:
        return None
    edges = (core.read_json(p, p.out("screen")["results"])["host"])
    out = {}
    for t in todo:
        out[t["name"]] = dict(t, **job_summary(p, res, t["name"], (edges["vbm"], edges["cbm"]),
                                               p.dir / "C_screen" / f"{subdir}_structures"))
    return out


# C5: relaxes every distorted structure with the screening settings (C2's, Gamma point, no
# restarts). Results go to C_screen/drelax.json.
#   Called by: dwf_core.run_phase() (step C5)
#   Calls:     run_screen_jobs()
def step_drelax(p):
    todo = core.read_json(p, p.out("distort")["jobs"])
    out = run_screen_jobs(p, "drelax", todo)
    if out is None:
        return None
    rel = core.write_json(p, p.phase_dir("C") / "drelax.json", out)
    bad = sum(r["energy"] is None for r in out.values())
    return {"results": rel, "_note": f"{len(out)} relaxations" + (f", {bad} without a usable "
                                                                     "energy" if bad else "")}


# Groups structures that are the same (pymatgen's StructureMatcher, which also finds a distortion
# repeated along another direction), as lists of indices; without pymatgen, by comparing atom by
# atom (all come from the same cell with the same atom order). The tolerances are tight (stol 0.05,
# i.e. about 0.1 A for a typical oxide): defect ground states can differ by a few tenths of an
# Angstrom around the defect, which pymatgen's defaults would merge.
#   Called by: distinct_states(), duplicates(), step_rank()
#   Calls:     sc.displacements()
def same_structures(atoms_list):
    try:
        from pymatgen.analysis.structure_matcher import StructureMatcher
        from pymatgen.io.ase import AseAtomsAdaptor
        sm = StructureMatcher(ltol=0.05, stol=0.05, angle_tol=2, primitive_cell=False,
                              scale=False, attempt_supercell=False)
        structs = [AseAtomsAdaptor.get_structure(a) for a in atoms_list]

        # Whether two structures match.
        def match(i, j):
            return sm.fit(structs[i], structs[j])
    except ImportError:
        # Whether every atom of two structures is within 0.05 A of its counterpart.
        def match(i, j):
            return len(atoms_list[i]) == len(atoms_list[j]) and \
                sc.displacements(atoms_list[i], atoms_list[j]).max() < 0.05
    groups = []
    for i in range(len(atoms_list)):
        for g in groups:
            if match(g[0], i):
                g.append(i)
                break
        else:
            groups.append([i])
    return groups


# The distinct relaxed structures of each candidate and charge state, lowest energy first: the
# screened one (C2) and every distortion relaxation with a usable energy, identical ones merged
# (same_structures, keeping the lowest). Returns {(cand, q): [entries]}.
#   Called by: step_rank()
#   Calls:     same_structures(), dwf_core.load_structure()
def distinct_states(p, survivors, results):
    pool = {}
    for s in survivors:
        for q in s["charges"]:
            b = s["best"].get(str(q))
            if b and b["structure"]:
                pool.setdefault((s["cand"], q), []).append(
                    {"name": b["name"], "energy": b["energy"], "spin": b["spin"],
                     "structure": b["structure"], "source": "screened", "factor": None})
    for name, r in results.items():
        if r["energy"] is not None and r["structure"] and (r["cand"], r["q"]) in pool:
            pool[(r["cand"], r["q"])].append(
                {"name": name, "energy": r["energy"], "spin": r["spin"],
                 "structure": r["structure"], "source": r.get("source", "distortion"),
                 "factor": r.get("factor")})
    out = {}
    for key, entries in pool.items():
        entries.sort(key=lambda e: e["energy"])
        atoms = [core.load_structure(p.dir / e["structure"])[0] for e in entries]
        groups = same_structures(atoms)
        firsts = sorted(min(g) for g in groups)
        out[key] = [dict(entries[i], copies=len(next(g for g in groups if min(g) == i)))
                    for i in firsts]
    return out


# C6: ranks the relaxed structures of each survivor and charge state (distinct_states). A
# distortion counts as found when it beats the screened structure by distortion_min_gain. A
# structure found in one charge state that no other charge state has is then relaxed in those
# charge states too (a batch per round, C_screen/share_<n>), for up to share_rounds rounds or
# until nothing new turns up. The lowest keep_structures distinct structures of each charge state
# go to phase D (C_screen/rank.json; table in rank.txt).
#   Called by: dwf_core.run_phase() (step C6)
#   Calls:     distinct_states(), run_screen_jobs(), same_structures(), start_spin(),
#              dwf_core.load_structure(), dwf_core.qtag(), dwf_core.save_structure(),
#              dwf_core.write_table()
def step_rank(p):
    survivors = core.read_json(p, p.out("zpl")["survivors"])
    results = dict(core.read_json(p, p.out("drelax")["results"]))
    tried = set()
    for rnd in range(1, p.s.share_rounds + 1):
        states = distinct_states(p, survivors, results)
        todo = []
        for (cid, q), entries in states.items():
            best = entries[0]
            if best["source"] == "screened" or best["name"] in tried:
                continue
            e0 = next((b["energy"] for s in survivors if s["cand"] == cid
                       for qq, b in s["best"].items() if int(qq) == q), None)
            if e0 is not None and e0 - best["energy"] < p.s.distortion_min_gain:
                continue
            tried.add(best["name"])
            atoms, _ = core.load_structure(p.dir / best["structure"])
            for (c2, q2), ent2 in states.items():
                if c2 != cid or q2 == q:
                    continue
                others = [core.load_structure(p.dir / e["structure"])[0] for e in ent2]
                if any(len(g) > 1 and 0 in g for g in same_structures([atoms] + others)):
                    continue                         # already has this structure
                spin = start_spin(atoms, q2, ent2[0]["spin"])
                name = f"{best['name']}_to_{core.qtag(q2)}"
                path = p.dir / "C_screen" / f"share_{rnd}" / f"{name}.json"
                a2 = atoms.copy()
                core.save_structure(path, a2, **core.load_structure(
                    p.dir / p.out("prerelax")["structures"][cid])[1])
                todo.append({"name": name, "cand": cid, "q": q2, "spin": spin,
                             "factor": best["factor"], "source": f"from {core.qtag(q)}",
                             "structure": str(path.relative_to(p.dir))})
        if not todo:
            break
        log(f"C6 round {rnd}: trying {len(todo)} structures found in one charge state in the "
            "others")
        out = run_screen_jobs(p, f"share_{rnd}", todo)
        if out is None:
            return None
        results.update(out)
    states = distinct_states(p, survivors, results)
    screened = {(s["cand"], int(q)): b["energy"] for s in survivors for q, b in s["best"].items()}
    keep, rows = [], []
    for (cid, q), entries in sorted(states.items()):
        best = entries[0]
        e0 = screened.get((cid, q), best["energy"])
        gain = e0 - best["energy"]
        rows.append([cid, f"{q:+d}", e0, best["energy"], gain,
                     best["source"] + (f" {best['factor']}" if best["factor"] is not None else ""),
                     len(entries)])
        for k, e in enumerate(entries[:p.s.keep_structures]):
            atoms, _ = core.load_structure(p.dir / e["structure"])
            keep.append({"cand": cid, "q": q, "k": k, "energy": e["energy"],
                         "spin": start_spin(atoms, q, e["spin"]), "structure": e["structure"],
                         "source": e["source"], "factor": e["factor"]})
    print(core.write_table(p.phase_dir("C") / "rank.txt", ["id", "q", "screened E/eV",
                                                           "best E/eV", "gain/eV", "from",
                                                           "distinct"], rows,
                           "C6 ground-state search (energies at the screening settings)"))
    rel = core.write_json(p, p.phase_dir("C") / "rank.json", keep)
    found = sum(1 for r in rows if r[4] > p.s.distortion_min_gain)
    return {"structures": rel, "count": len(keep),
            "_note": f"{len(keep)} structures for phase D; a distortion won in {found} of "
                     f"{len(rows)} charge states"}


SCREEN_KEYS = ("screen_cutoff", "screen_max_steps", "screen_overrides", "level_margin")
STEPS = [
    core.Step("prerelax", "C1", "build the candidates and pre-relax them with MACE",
              step_prerelax, keys=("mace", "mace_model", "mace_fmax", "mace_steps", "triage",
                                   "triage_window"), files=("finetuned_model",),
              needs=("generate", "choose", "relax"), owns=("C_screen/structures",
                                                          "C_screen/prerelax")),
    core.Step("screen", "C2", "cheap CASTEP screening of every candidate", step_screen,
              keys=SCREEN_KEYS + ("screen_charges", "screen_spins"),
              needs=("prerelax", "converge"), owns=("C_screen/screen", "C_screen/screen.json",
                                                    "C_screen/screen_structures")),
    core.Step("filter", "C3", "defect hull and property filters", step_filter,
              keys=("hull_margin", "require_gap_states", "require_spin", "require_bound",
                    "distort_charges", "charge_correction"),
              needs=("screen", "dielectric", "prerelax"),
              owns=("C_screen/filter.json", "C_screen/filter.txt")),
    core.Step("zpl", "C3b", "quick ZPL filter", step_zpl, keys=("zpl_window",), needs=("filter",),
              owns=("C_screen/zpl.json", "C_screen/zpl.txt")),
    core.Step("distort", "C4", "ShakeNBreak distortions of the survivors", step_distort,
              keys=("distortions", "distort_around", "rattle_stdev", "rattle_dmin",
                    "oxidation_states"), needs=("zpl", "prerelax"),
              owns=("C_screen/distortions", "C_screen/distort.json")),
    core.Step("drelax", "C5", "cheap CASTEP relaxations of the distortions", step_drelax,
              keys=SCREEN_KEYS, needs=("distort",),
              owns=("C_screen/drelax", "C_screen/drelax.json", "C_screen/drelax_structures")),
    core.Step("rank", "C6", "rank, merge, share distortions between charge states", step_rank,
              keys=SCREEN_KEYS + ("distortion_min_gain", "share_rounds", "keep_structures"),
              needs=("drelax",), owns=("C_screen/rank.json", "C_screen/rank.txt",
                                       "C_screen/share_*")),
]

if __name__ == "__main__":
    core.run_phase("C", STEPS, __doc__)
