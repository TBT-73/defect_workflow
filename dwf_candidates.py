#!/usr/bin/env python3
"""
dwf_candidates.py - phase B of the defect workflow: candidate defects and supercell size

Steps (run in this order; each is skipped once it has finished with the same inputs):
  B1  generate  the candidate defects: the ones in %block defects (B1 'yes'), or generated
                (B1 'no'): vacancies, substitutions by the named dopants, optionally
                antisites and interstitials, and dopant + native pairs up to
                pair_max_distance apart, with duplicates removed by symmetry
  B2  scan      MACE size scan of the representative defect: where does its relaxation
                energy level off? (caps the CASTEP series)
  B3  series    CASTEP size series of the representative defect with
                supercell_convergence.py (MACE pre-relaxation with mace : true)
  B4  finetune  optional (finetune : true): MACE fine-tuning data from the series'
                smallest-cell relaxation, and a mace_run_train script
  B5  choose    the supercell used from here on (series result, or supercell : n)

Usage:  python3 dwf_candidates.py <project>.dwf [steps] [--redo STEP] [--status] [--no-wait]
        [--prepare-only] [--resubmit-failed] [--force] [--example]
"""

import itertools
import json
import math
import subprocess
import sys

import numpy as np

import dwf_core as core
from dwf_core import die, log, sc

# ==========================================================================================
# What each step reads and writes (under <workdir>/B_candidates/):
#   B1 generate   A2.1 host, dopants ... -> candidates.json / .txt (single_defects, generate,
#                                           pair_arrangements, or user_defects)
#   B2 scan       B1 (representative)    -> scan/ (MACE caches), scan.txt, the size cap
#   B3 series     B1, B2, A1, A2         -> series/ (series.conv run by supercell_convergence.py)
#   B4 finetune   B3                     -> finetune/train.xyz, finetune.sh
#   B5 choose     B3 or supercell        -> the supercell size used in C and D
# ==========================================================================================


# --------------------------------------------------------------------------
# B1: candidates
# --------------------------------------------------------------------------
# The single defects of the host primitive cell, one per symmetry-inequivalent site: vacancies
# (vacancies : true), substitutions by each dopant, antisites (antisites : true), and Voronoi
# interstitials of the dopants or of every element (interstitials : dopants / all; needs
# pymatgen-analysis-defects). Each is a component dict (new, host, prim_site or frac) with
# 'sites': how many such sites there are per primitive cell.
#   Called by: generate()
#   Calls:     interstitial_sites(), dwf_core.equivalent_sites()
def single_defects(p, prim):
    syms, cls = prim.get_chemical_symbols(), core.equivalent_sites(prim)
    out = []
    for i in sorted(set(cls)):
        host, n = syms[i], cls.count(i)
        news = ([None] if p.s.vacancies else []) + list(p.s.dopants)
        if p.s.antisites:
            news += [x for x in sorted(set(syms)) if x != host]
        for new in news:
            out.append({"new": new, "host": host, "prim_site": i, "sites": n})
    if p.s.interstitials != "none":
        out += interstitial_sites(p, prim)
    return out


# The Voronoi interstitial sites of the host primitive cell (pymatgen's
# VoronoiInterstitialGenerator), for the dopants or every element. Labelled a, b, c, ... in the
# order pymatgen gives them.
#   Called by: single_defects()
def interstitial_sites(p, prim):
    try:
        from pymatgen.analysis.defects.generators import VoronoiInterstitialGenerator
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError as err:
        die(f"interstitials : {p.s.interstitials} needs pymatgen-analysis-defects: {err}")
    import warnings
    warnings.filterwarnings("ignore")
    species = list(p.s.dopants) + (sorted(set(prim.get_chemical_symbols()))
                                   if p.s.interstitials == "all" else [])
    out, labels = [], {}
    for d in VoronoiInterstitialGenerator().generate(AseAtomsAdaptor.get_structure(prim),
                                                     insert_species=species):
        el = d.site.species_string
        labels[el] = labels.get(el, 0) + 1
        out.append({"new": el, "host": None, "frac": d.site.frac_coords.tolist(),
                    "site_label": "abcdefghijklmnopqrstuvwxyz"[(labels[el] - 1) % 26],
                    "sites": int(getattr(d, "multiplicity", 1))})
    return out


# The position (cartesian, in the primitive cell) of a single defect.
#   Called by: pair_arrangements()
def comp_position(prim, c):
    if c["host"] is None:
        return np.asarray(c["frac"]) @ np.asarray(prim.cell)
    return prim.positions[c["prim_site"]].copy()


# Every place a component of type b can sit in the crystal: the images, within reach, of the
# primitive-cell sites equivalent to b's (or of b's interstitial site and its symmetry copies).
#   Called by: pair_arrangements()
#   Calls:     dwf_core.equivalent_sites(), dwf_core.symmetry_ops()
def candidate_points(prim, b, reach):
    cell = np.asarray(prim.cell)
    if b["host"] is None:
        rots, trans = core.symmetry_ops(prim)
        f0 = np.asarray(b["frac"])
        fracs = []
        for R, t in zip(rots, trans):
            f = (R @ f0 + t) % 1.0
            if not any(np.linalg.norm(((f - g + 0.5) % 1.0 - 0.5) @ cell) < 1e-3 for g in fracs):
                fracs.append(f)
        base = [f @ cell for f in fracs]
    else:
        cls = core.equivalent_sites(prim)
        base = [prim.positions[j] for j in range(len(prim)) if cls[j] == cls[b["prim_site"]]]
    n = [int(math.ceil(reach / g)) + 1 for g in 1 / np.linalg.norm(np.linalg.inv(cell).T, axis=1)]
    shifts = np.array(list(itertools.product(*[range(-k, k + 1) for k in n]))) @ cell
    return np.array([r + s for r in base for s in shifts])


# The cartesian rotations of the symmetry operations that leave a point of the primitive cell where
# it is (up to a lattice vector): the site symmetry used to tell arrangements apart.
#   Called by: pair_arrangements()
#   Calls:     dwf_core.symmetry_ops()
def site_rotations(prim, point):
    cell = np.asarray(prim.cell)
    f0 = np.linalg.solve(cell.T, point)
    rots, trans = core.symmetry_ops(prim)
    out = []
    for R, t in zip(rots, trans):
        d = (R @ f0 + t - f0 + 0.5) % 1.0 - 0.5
        if np.linalg.norm(d @ cell) < 1e-3:
            out.append(cell.T @ R @ np.linalg.inv(cell.T))
    return out


# The symmetry-distinct arrangements of a pair: b at every allowed point within pair_max_distance
# of a, grouped by the site symmetry of a (two vectors are the same arrangement if a symmetry
# operation that keeps a in place turns one into the other; for two identical components, also if
# it turns one into the other reversed). Each arrangement's vector is one that starts from a's own
# site (a reversed vector may only fit the other end, e.g. in wurtzite). Filters: interstitials at
# least 0.5 A from anything and 1 A from a vacancy. Returns [(vector, separation, how many
# equivalent orientations)], nearest first.
#   Called by: generate(), user_defects()
#   Calls:     candidate_points(), comp_position(), site_rotations()
def pair_arrangements(prim, a, b, dmax):
    ra = comp_position(prim, a)
    pts = candidate_points(prim, b, dmax)
    vecs = pts - ra
    dist = np.linalg.norm(vecs, axis=1)
    lo = 1.0 if (a["host"] is None) != (b["host"] is None) and not (a["new"] and b["new"]) else 0.5
    keep = (dist > max(0.1, lo if (a["host"] is None or b["host"] is None) else 0.1)) & \
           (dist <= dmax + 1e-6)
    rots = site_rotations(prim, ra)
    same = (a["new"], a["host"], a.get("prim_site"), tuple(a.get("frac") or ())) == \
           (b["new"], b["host"], b.get("prim_site"), tuple(b.get("frac") or ()))
    # A vector as a sortable tuple.
    def tup(x):
        return tuple(np.round(x, 3) + 0.0)

    groups = {}
    for v in vecs[keep]:
        imgs = [R @ v for R in rots]
        key = min(tup(x) for x in imgs + ([-x for x in imgs] if same else []))
        groups.setdefault(key, []).append(min(imgs, key=tup))     # (not reversed: fits a's site)
    out = []
    for key, reps in groups.items():
        rep = np.array(min(tup(x) for x in reps))
        out.append((rep, float(np.linalg.norm(rep)), len(reps)))
    return sorted(out, key=lambda x: (round(x[1], 3), x[0].tolist()))


# Turns a component token (Sb_Ca, Vac_O, Vac_O:2, Sb_Ca@2) into a component dict; 'sites' is how
# many symmetry-equivalent sites of that kind the primitive cell has.
#   Called by: representative(), user_defects()
#   Calls:     dwf_core.equivalent_sites(), sc.parse_defect()
def parse_token(tok, prim):
    comps = sc.parse_defect([tok], prim)
    c = comps[0]
    cls = core.equivalent_sites(prim)
    i = c["prim_site"]
    return {"new": c["new"], "host": c["host"], "prim_site": i, "spin": c["spin"],
            "sites": cls.count(cls[i])}


# The candidates from %block defects (B1 'yes'): one line per defect, components joined by '+'
# (e.g. Sb_Ca + Vac_O). A pair with 'all' on its line gives every arrangement up to
# pair_max_distance; otherwise each further component goes on the nearest site of its kind (as in
# the size series).
#   Called by: step_generate()
#   Calls:     candidate(), pair_arrangements(), parse_token(), sc.defect_vectors(),
#              sc.parse_defect()
def user_defects(p, prim):
    out = []
    for row in p.s.defects:
        words = [w for w in row if w.lower() != "all"]
        every = len(words) < len(row)
        toks = [t for t in " ".join(words).replace("+", " ").split() if t]
        comps = [parse_token(t, prim) for t in toks]
        if len(comps) == 1:
            out.append(candidate(prim, comps, None, 1))
            continue
        if every and len(comps) == 2:
            for v, d, m in pair_arrangements(prim, comps[0], comps[1], p.s.pair_max_distance):
                out.append(candidate(prim, [comps[0], dict(comps[1], vec=v.tolist())], d, m))
            continue
        if every:
            die(f"%block defects: 'all' works for pairs only ({' + '.join(toks)})")
        vecs = sc.defect_vectors(prim, sc.parse_defect(toks, prim))
        full = [comps[0]] + [dict(c, vec=np.asarray(v).tolist()) for c, v in zip(comps[1:], vecs)]
        sep = float(np.linalg.norm(vecs[0])) if vecs else None
        out.append(candidate(prim, full, sep, 1))
    return out


# A candidate record: its components, name, composition change and group (composition_change),
# the separation of a pair, its orientational multiplicity and the number of anchor sites per
# primitive cell (both for concentrations in E).
#   Called by: generate(), user_defects()
#   Calls:     dwf_core.composition_change(), dwf_core.defect_name()
def candidate(prim, comps, sep, mult):
    delta, group = core.composition_change(comps)
    comps = [{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in c.items()}
             for c in comps]
    return {"name": core.defect_name(comps, prim), "comps": comps, "group": group,
            "delta": delta, "separation": sep, "multiplicity": mult,
            "sites": comps[0].get("sites", 1)}


# The generated candidates (B1 'no'): the single defects (single_defects), then the pairs
# (pairs : true) of a dopant defect with a native one, and with native_pairs / dopant_pairs also
# native + native and dopant + dopant pairs; never two interstitials. Each pair type gives its
# symmetry-distinct arrangements up to pair_max_distance (pair_arrangements).
#   Called by: step_generate()
#   Calls:     candidate(), pair_arrangements(), single_defects()
def generate(p, prim):
    singles = single_defects(p, prim)
    out = [candidate(prim, [s], None, 1) for s in singles]
    if not p.s.pairs:
        return out
    dop = set(p.s.dopants)

    # Whether a single defect involves a dopant.
    def is_dopant(s):
        return s["new"] in dop

    for i, a in enumerate(singles):
        for j, b in enumerate(singles):
            if j < i or (a["host"] is None and b["host"] is None):
                continue
            kinds = (is_dopant(a), is_dopant(b))
            ok = ((kinds in ((True, False), (False, True))) or
                  (kinds == (False, False) and p.s.native_pairs) or
                  (kinds == (True, True) and p.s.dopant_pairs))
            if not ok:
                continue
            first, second = (b, a) if (not kinds[0] and kinds[1]) else (a, b)
            for v, d, m in pair_arrangements(prim, first, second, p.s.pair_max_distance):
                out.append(candidate(prim, [first, dict(second, vec=v.tolist())], d, m))
    return out


# B1: lists the candidate defects (user_defects or generate) in B_candidates/candidates.json, with
# ids d001, d002, ... sorted by composition group and separation, and a readable table.
#   Called by: dwf_core.run_phase() (step B1)
#   Calls:     generate(), user_defects(), dwf_core.Project.host(), dwf_core.write_table()
def step_generate(p):
    d = p.phase_dir("B")
    prim = p.host()
    cands = user_defects(p, prim) if p.s.defects else generate(p, prim)
    if not cands:
        die("no candidate defects: name some dopants or list defects in %block defects")
    cands.sort(key=lambda c: (len(c["comps"]), c["group"], c["separation"] or 0, c["name"]))
    for k, c in enumerate(cands):
        c["id"] = f"d{k + 1:03d}"
    groups = sorted({c["group"] for c in cands})
    rows = [[c["id"], c["name"], c["group"],
             f"{c['separation']:.3f}" if c["separation"] is not None else "",
             c["multiplicity"], c["sites"]] for c in cands]
    print(core.write_table(d / "candidates.txt", ["id", "defect", "composition", "separation/A",
                                                  "orientations", "sites/cell"], rows,
                           f"B1 candidates: {len(cands)} in {len(groups)} composition groups"))
    core.write_json(p, d / "candidates.json", cands)
    return {"candidates": "B_candidates/candidates.json", "count": len(cands),
            "groups": len(groups), "_note": f"{len(cands)} candidates in {len(groups)} groups"}


# --------------------------------------------------------------------------
# B2-B5: cell size
# --------------------------------------------------------------------------
# The defect the size scan and series use: series_defect (tokens, e.g. Sb_Ca@1 + Vac_O) or the
# nearest arrangement of the first pair with a dopant (else the first candidate). Returns
# (component dicts for build_defect, tokens for supercell_convergence.py, its name). A token names
# the primitive-cell site (:k) its component sits on when its element has more than one, so that
# supercell_convergence.py builds the same arrangement.
#   Called by: step_scan(), step_series()
#   Calls:     parse_token(), sc.defect_vectors(), sc.parse_defect()
def representative(p, prim):
    if p.s.series_defect:
        toks = p.s.series_defect.replace("+", " ").split()
        comps = sc.parse_defect(toks, prim)
        vecs = sc.defect_vectors(prim, comps)
        full = [parse_token(toks[0], prim)] + [
            dict(parse_token(t, prim), vec=np.asarray(v).tolist()) for t, v in zip(toks[1:], vecs)]
        return full, toks, " + ".join(toks)
    cands = core.read_json(p, p.out("generate")["candidates"])
    pairs = [c for c in cands if len(c["comps"]) == 2 and any(x["new"] in p.s.dopants
                                                              for x in c["comps"])]
    c = min(pairs, key=lambda c: c["separation"]) if pairs else cands[0]
    if any(x["host"] is None for x in c["comps"]):
        die(f"the representative defect {c['name']} has an interstitial, which the size series "
            "can't build: set series_defect (e.g. Sb_Ca + Vac_O) or supercell")
    prim_syms, frac, cell = prim.get_chemical_symbols(), prim.get_scaled_positions(), prim.cell
    ra = prim.positions[c["comps"][0]["prim_site"]]
    toks = []
    for x in c["comps"]:
        site = x["prim_site"]
        if "vec" in x:                              # the site its vector lands on
            f = np.linalg.solve(np.asarray(cell).T, ra + np.asarray(x["vec"]))
            d = (frac - f + 0.5) % 1.0 - 0.5
            site = int(np.argmin(np.linalg.norm(d @ np.asarray(cell), axis=1)))
        same = [i for i, s in enumerate(prim_syms) if s == x["host"]]
        tok = f"{x['new'] or 'Vac'}_{x['host']}"
        if len(same) > 1:
            tok += f":{same.index(site) + 1}"
        toks.append(tok)
    return c["comps"], toks, c["name"]


# B2: MACE size scan. Builds the representative defect in each of scan_sizes, relaxes it with
# MACE (positions only) and records its relaxation energy (unrelaxed - relaxed) and the largest
# displacement of the atoms in the outer part of the cell (beyond 35% of the shortest image
# distance from the defect). The CASTEP series goes no further than the first size whose
# relaxation energy is within scan_tol of the size before (the cap). Skipped with mace : false,
# scan : false or supercell (the cap is only for the series).
#   Called by: dwf_core.run_phase() (step B2)
#   Calls:     representative(), dwf_core.Project.host(), dwf_core.Project.supercell_matrix(),
#              dwf_core.build_defect(), dwf_core.mace_relax(), dwf_core.write_table(),
#              sc.displacements(), sc.get_distances(), sc.min_image_distance()
def step_scan(p):
    if not (p.s.mace and p.s.scan) or p.s.supercell:
        return {"cap": None, "_note": "skipped (" + ("supercell given" if p.s.supercell else
                                                    "mace or scan is false") + ")"}
    d = p.phase_dir("B")
    prim = p.host()
    comps, _, name = representative(p, prim)
    rows, out, prev, cap = [], [], None, None
    for n in sorted(p.s.scan_sizes):
        P = p.supercell_matrix(n)
        atoms, info = core.build_defect(prim, P, comps, name)
        L = sc.min_image_distance(atoms.cell)
        log(f"B2: {name} in size {n} ({len(atoms)} atoms): MACE relaxation")
        relaxed, e0, e1, steps = core.mace_relax(p, atoms, d / "scan" / f"n{n}.json")
        moved = sc.displacements(atoms, relaxed)
        _, r = sc.get_distances(np.array(info["centre"])[None], atoms.positions, cell=atoms.cell,
                                pbc=True)
        far = moved[r[0] > 0.35 * L]
        erel = e0 - e1
        row = {"n": n, "atoms": len(atoms), "L_min": L, "E_relax": erel, "steps": steps,
               "far_move": float(far.max()) if far.size else 0.0}
        row["change"] = None if prev is None else erel - prev
        out.append(row)
        rows.append([n, len(atoms), L, erel, row["change"], row["far_move"], steps])
        if prev is not None and abs(erel - prev) < p.s.scan_tol and cap is None:
            cap = n
            break
        prev = erel
    print(core.write_table(d / "scan.txt", ["n", "atoms", "L_min/A", "E_relax/eV", "change/eV",
                                            "far-field move/A", "MACE steps"], rows,
                           f"B2 MACE size scan of {name} (scan_tol {p.s.scan_tol:g} eV)"))
    if cap is None:
        log(f"note: the MACE relaxation energy has not levelled off within scan_sizes "
            f"{p.s.scan_sizes}; the CASTEP series is not capped")
    core.write_json(p, d / "scan.json", out)
    return {"cap": cap, "table": out,
            "_note": f"levels off at size {cap}" if cap else "did not level off"}


# The runner settings of a group, as lines of a .conv file: the project's (and the group's) runner
# keywords, with file names made absolute (the .conv is elsewhere). Batch-only keywords are left
# out.
#   Called by: step_series()
#   Calls:     dwf_core.Project.runner()
def runner_conv_lines(p, group):
    p.runner(group)                                   # checks them
    own = p.runner_groups.get(group, [])

    # The keyword or block name of an item.
    def key(x):
        return x["name"] if x["type"] == "block" else x["key"]

    later = {key(y) for y in own}
    lines = []
    for it in [x for x in p.runner_items if key(x) not in later] + own:
        k = key(it)
        if k in ("max_jobs", "tolerate_failures"):
            continue
        if it["type"] == "block":
            lines += [f"%block {k}"] + it["body"] + [f"%endblock {k}"]
        elif k in ("jobscript", "dryrun_jobscript"):
            lines.append(f"{k} : {(p.base / it['value']).resolve()}")
        else:
            lines.append(f"{k} : {it['value']}")
    return lines


# B3: the CASTEP size series of the representative defect, run by supercell_convergence.py in
# B_candidates/series: its .conv is written from the project (relaxed host cell, production
# .param and k-points, e_bulk, transform, series_sizes up to the B2 cap, series_tol, MACE
# pre-relaxation with mace : true, series_charge with its correction), then the script runs in
# a subprocess (output to this terminal) and its convergence.json is read. A charged or
# odd-electron defect runs spin-polarised with a starting spin (the lowest its electron count
# allows, also put on the first component); otherwise the .param's own spin_polarized is kept.
# Not finished while jobs are queued (--no-wait).
#   Called by: dwf_core.run_phase() (step B3)
#   Calls:     representative(), runner_conv_lines(), dwf_core.Project.host(),
#              dwf_core.Project.kspacing(), dwf_core.Project.production_param(),
#              dwf_core.Project.supercell_matrix(), dwf_core.build_defect(),
#              dwf_core.electron_parity(), sc.strip_comment()
def step_series(p):
    if p.s.supercell:
        return {"skipped": True, "_note": f"supercell : {p.s.supercell} given"}
    d = p.phase_dir("B") / "series"
    d.mkdir(parents=True, exist_ok=True)
    prim = p.host()
    _, toks, name = representative(p, prim)
    cap = p.out("scan").get("cap")
    sizes = [n for n in sorted(p.s.series_sizes) if cap is None or n <= cap]
    if len(sizes) < 2:
        sizes = sorted(p.s.series_sizes)[:2]
    q = p.s.series_charge
    spin = core.electron_parity(core.build_defect(prim, p.supercell_matrix(sizes[0]),
                                                  representative(p, prim)[0], name)[0], q)
    drop = r"(charge|spin|spin_polarized)" if spin or q else r"(charge|spin)"   # set below
    param = "\n".join(line for line in p.production_param().splitlines()
                      if not sc.re.match(rf"\s*{drop}\b", sc.strip_comment(line),
                                         sc.re.I)).rstrip() + "\n"
    if q:
        param += f"charge : {q}\n"
    if spin or q:
        param += f"spin_polarized : true\nspin : {spin}\n"
    (d / "series.param").write_text(param)
    host = p.out("relax")
    toks = [toks[0] + (f"@{spin}" if spin and "@" not in toks[0] and not
                       toks[0].lower().startswith("vac") else "")] + toks[1:]
    conv = [f"# size series of {name} (written by dwf_candidates.py, step B3)",
            f"cell : {(p.dir / host['cell_file']).resolve()}", "param : series.param",
            "%block defect"] + toks + ["%endblock defect",
                                       f"e_bulk : {host['e_bulk']:.10f} eV",
                                       f"sizes : {' '.join(map(str, sizes))}",
                                       f"kspacing : {p.kspacing():.8f}",
                                       f"tol : {p.s.series_tol:g} eV", "singlepoint : false",
                                       f"energy : {p.s.energy}", "workdir : run"]
    if p.s.transform:
        conv += ["%block transform"] + [" ".join(map(str, p.s.transform[i:i + 3]))
                                         for i in (0, 3, 6)] + ["%endblock transform"]
    if q:
        eps = p.out("dielectric")["eps0"]
        conv += [f"charge_correction : {p.s.charge_correction}", f"dielectric : {eps:.6f}"]
    if p.s.mace:
        conv += ["prerelax : mace", f"prerelax_model : {p.s.mace_model}",
                 f"prerelax_device : {p.s.mace_device}", f"prerelax_threads : {p.s.mace_threads}",
                 f"prerelax_fmax : {p.s.mace_fmax:g}", f"prerelax_steps : {p.s.mace_steps}"]
    conv += runner_conv_lines(p, "series")
    (d / "series.conv").write_text("\n".join(conv) + "\n")
    cmd = [sys.executable, str(core.HERE / "supercell_convergence.py"), "series.conv"]
    for flag in ("no_wait", "prepare_only", "force", "resubmit_failed"):
        if getattr(p.cl, flag, False):
            cmd.append("--" + flag.replace("_", "-"))
    log(f"B3: size series of {name}, sizes {sizes}, in {d}")
    sys.stdout.flush()
    ret = subprocess.run(cmd, cwd=d)
    if ret.returncode != 0:
        die(f"the size series stopped with an error (above); fix it and rerun ({d})")
    res = d / "run" / "convergence.json"
    if not res.exists():
        return None
    rows = json.loads(res.read_text())
    ns = [int(sc.re.findall(r"\d+", r["size"])[0]) for r in rows]
    converged = len(rows) > 1 and abs(rows[-1]["dE_f"]) < p.s.series_tol
    if not converged and ns[-1] < sizes[-1]:
        return None                                   # more sizes still to come
    table = [{"n": n, "atoms": r["atoms"], "E_f": r["E_f"], "dE_f": r.get("dE_f")}
             for n, r in zip(ns, rows)]
    return {"converged": converged, "sizes": ns, "table": table,
            "choice": (ns[-2] if p.s.series_choice == "smaller" else ns[-1]) if converged else None,
            "_note": (f"converged: E_f* changed by {rows[-1]['dE_f']:+.4f} eV from size {ns[-2]} "
                      f"to {ns[-1]}" if converged else f"not converged within sizes {ns}")}


# B4 (optional): MACE fine-tuning data from the series' smallest-cell relaxation. Every geometry
# step with energy and forces (core.read_trajectory, all runs of the job; steps with a force above
# 2 eV/A left out) goes into B_candidates/finetune/train.xyz (REF_energy, REF_forces), and
# finetune.sh runs mace_run_train from mace_model on it (finetune_args adds options). Run it
# yourself (a GPU helps), then set finetuned_model to the .model it writes: C1 uses it (B2 and
# B3 keep mace_model, so they don't run again).
#   Called by: dwf_core.run_phase() (step B4)
#   Calls:     dwf_core.read_trajectory()
def step_finetune(p):
    if not p.s.finetune:
        return {"skipped": True, "_note": "finetune : false"}
    import ase.io
    d = p.phase_dir("B") / "finetune"
    d.mkdir(parents=True, exist_ok=True)
    series = p.out("series")
    if series.get("skipped"):
        die("finetune : true needs the size series (B3), which supercell skipped")
    run = p.dir / "B_candidates" / "series" / "run"
    castep = sorted(run.glob("*/sc_*_defect_geom.castep"),
                    key=lambda f: len(f.read_text(errors="replace")))
    if not castep:
        die(f"no relaxation output found under {run}")
    first = min(castep, key=lambda f: int(sc.re.findall(r"\d+", f.parent.name)[0]))
    images = []
    for a, e, f in core.read_trajectory(first.read_text(errors="replace")):
        if np.abs(f).max() > 2.0:                     # far from the minimum: leave out
            continue
        a.info["REF_energy"] = e
        a.arrays["REF_forces"] = f
        images.append(a)
    if not images:
        log(f"WARNING: no geometry steps with forces in {first}; nothing to fine-tune on")
        return {"images": 0, "_note": "no training data"}
    ase.io.write(str(d / "train.xyz"), images, format="extxyz")
    script = ("#!/bin/bash\n# Fine-tune the MACE foundation model on the size series' CASTEP data "
              "(written by dwf_candidates.py, step B4)\n"
              f"mace_run_train --name=dwf_finetuned --foundation_model={p.s.mace_model} "
              "--multiheads_finetuning=False --train_file=train.xyz --valid_fraction=0.1 "
              "--energy_key=REF_energy --forces_key=REF_forces --E0s=average --loss=weighted "
              "--energy_weight=1 --forces_weight=10 --max_num_epochs=50 --batch_size=2 "
              f"--default_dtype=float64 --device={p.s.mace_device} {p.s.finetune_args}\n"
              f"# then set  finetuned_model : {(d / 'dwf_finetuned.model').resolve()}  in the "
              "project file (C1 uses it)\n")
    (d / "finetune.sh").write_text(script)
    (d / "finetune.sh").chmod(0o755)
    return {"images": len(images), "train": str((d / "train.xyz").relative_to(p.dir)),
            "_note": f"{len(images)} structures in train.xyz; run finetune.sh"}


# B5: the supercell for C and D: supercell : n if given, else the size series' choice (the
# smaller of the two sizes that agreed within series_tol, or the larger with series_choice :
# larger). Stops if the series did not converge. Every arrangement of one composition uses this
# cell, so their energies can be compared directly.
#   Called by: dwf_core.run_phase() (step B5)
#   Calls:     dwf_core.Project.host(), dwf_core.Project.supercell_matrix(),
#              dwf_core.host_supercell(), sc.min_image_distance()
def step_choose(p):
    if p.s.supercell:
        n, why = p.s.supercell, "supercell in the project file"
    else:
        s = p.out("series")
        if not s.get("converged"):
            die(f"the size series did not converge within sizes {s.get('sizes')}: add bigger "
                "sizes to series_sizes (or raise series_tol), or choose the cell yourself with "
                "supercell : n")
        n, why = s["choice"], "size series"
    prim = p.host()
    P = p.supercell_matrix(n)
    host = core.host_supercell(prim, P)
    L = sc.min_image_distance(host.cell)
    return {"n": n, "matrix": P.tolist(), "natoms": len(host), "L_min": L,
            "_note": f"size {n}: {len(host)} atoms, images {L:.2f} A apart ({why})"}


STEPS = [
    core.Step("generate", "B1", "candidate defects", step_generate,
              keys=("dopants", "defects", "vacancies", "antisites", "interstitials", "pairs",
                    "pair_max_distance", "native_pairs", "dopant_pairs"), needs=("relax",),
              owns=("B_candidates/candidates.json", "B_candidates/candidates.txt")),
    core.Step("scan", "B2", "MACE size scan", step_scan,
              keys=("mace", "scan", "scan_sizes", "scan_tol", "mace_model", "mace_fmax",
                    "mace_steps", "transform", "series_defect", "supercell"), needs=("generate",),
              owns=("B_candidates/scan", "B_candidates/scan.txt", "B_candidates/scan.json")),
    core.Step("series", "B3", "CASTEP size series", step_series,
              keys=("series_defect", "series_charge", "series_sizes", "series_tol",
                    "series_choice", "supercell", "transform", "mace", "mace_model",
                    "charge_correction", "energy"),
              needs=("scan", "relax", "converge", "dielectric"), owns=("B_candidates/series",)),
    core.Step("finetune", "B4", "MACE fine-tuning data", step_finetune,
              keys=("finetune", "finetune_args", "mace_model"), needs=("series",),
              owns=("B_candidates/finetune",)),
    core.Step("choose", "B5", "choose the supercell", step_choose,
              keys=("supercell", "transform"), needs=("series", "relax")),
]

if __name__ == "__main__":
    core.run_phase("B", STEPS, __doc__)
