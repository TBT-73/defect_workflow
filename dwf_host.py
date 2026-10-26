#!/usr/bin/env python3
"""
dwf_host.py - phase A of the defect workflow: host reference calculations

Steps (run in this order; each is skipped once it has finished with the same inputs):
  A1    converge    cut-off and k-point spacing: single points of the host primitive cell
                    over converge_cutoffs and converge_kspacings; the smallest values within
                    converge_tol (eV/atom) of the best become the production settings
                    (A_host/production.param and the k-point spacing)
  A2.1  relax       variable-cell relaxation of the host -> relaxed cell, e_bulk
  A2.2  bands       single point on a denser Gamma-centred grid -> VBM, CBM, band gap
  A2.3  dielectric  DFPT (phonon+efield) -> static and optical dielectric constants
                    (or the dielectric value from the project file)
  A3.1  phases      relax the competing phases (%block phases, or doped + Materials
                    Project with phases_from : doped)
  A3.2  limits      chemical-potential limits of the host (pymatgen phase diagram), each
                    dopant limited by its most stable phase (or %block chemical_potentials)

Usage:  python3 dwf_host.py <project>.dwf [steps] [--redo STEP] [--status] [--no-wait]
        [--prepare-only] [--resubmit-failed] [--force] [--example]
"""

import numpy as np

import dwf_core as core
from dwf_core import die, log, sc

# ==========================================================================================
# What each step reads and writes (under <workdir>/A_host/):
#   A1 converge   host_cell, param       -> converge/ (jobs), converge.txt, production.param
#   A2.1 relax    A1                     -> relax/, host_relaxed.json and .cell, e_bulk
#   A2.2 bands    A2.1                   -> bands/ (the .bands for E2), VBM, CBM, gap
#   A2.3 dielectric  A2.1                -> dielectric/, epsilon_0 and epsilon_inf
#   A3.1 phases   A2.1, %block phases    -> phases/ (jobs), their energies
#   A3.2 limits   A3.1 (+ A2.1)          -> limits.txt, the chemical-potential limits
# ==========================================================================================


# The .param keywords every host and competing-phase job sets: no charge, and no starting spin
# left over from a defect calculation (a phase that needs one gives it on its line in %block
# phases).
HOST_OVERRIDES = {"charge": None, "spin": None}


# The pseudopotential entries of a .cell that don't look norm-conserving (DFPT needs NCP): a file
# that isn't a .recpot or named NCP, or an on-the-fly string without NCP in it; no species_pot
# block at all means CASTEP's default on-the-fly ultrasoft potentials.
#   Called by: step_dielectric()
#   Calls:     sc.block_rows(), sc.parse_castep_input()
def ultrasoft_entries(cell_text):
    pots = [it for it in sc.parse_castep_input(cell_text)
            if it["type"] == "block" and it["name"] == "species_pot"]
    if not pots:
        return ["(no species_pot: CASTEP's default ultrasoft potentials)"]
    bad = []
    for row in sc.block_rows(pots[0]):
        entry = " ".join(row[1:]) if len(row) > 1 else row[0]
        if not (entry.lower().endswith(".recpot") or "ncp" in entry.lower()):
            bad.append(" ".join(row))
    return bad


# --------------------------------------------------------------------------
# A1: cut-off and k-points
# --------------------------------------------------------------------------
# The production .param: the project's .param without charge or spin (every job sets its own)
# and, given a cut-off, with cut_off_energy set to it (any cut_off_energy or basis_precision in it
# removed). Returns (text, the keywords removed).
#   Called by: step_converge()
#   Calls:     sc.strip_comment()
def production_param_text(text, cutoff=None):
    drop = ("charge", "spin") + (("cut_off_energy", "basis_precision") if cutoff else ())
    keep, gone = [], []
    for line in text.splitlines():
        m = sc.re.match(r"\s*(\w+)\b", sc.strip_comment(line))
        if m and m.group(1).lower() in drop:
            gone.append(m.group(1).lower())
        else:
            keep.append(line)
    out = "\n".join(keep).rstrip() + "\n" + (f"cut_off_energy : {cutoff:g} eV\n" if cutoff else "")
    return out, [k for k in gone if k in ("charge", "spin")]


# The smallest setting whose energy per atom, and that of every better setting, is within tol of
# the best one (the last of values, which must be ordered from worst to best). None if only the
# best one is (the list does not show convergence).
#   Called by: step_converge()
def converged_value(values, energies, tol):
    best = energies[-1]
    for i in range(len(values) - 1):
        if all(abs(e - best) < tol for e in energies[i:]):
            return values[i]
    return None


# A1: converges the plane-wave cut-off and the k-point spacing.
# One batch of single points of the host primitive cell as given: every cut-off in
# converge_cutoffs on the finest k-grid, and every spacing in converge_kspacings (unless kspacing
# is set) at the highest cut-off; spacings that give the same grid are run once. Picks the
# smallest cut-off and the largest spacing within converge_tol (eV/atom) of the best, and writes
# A_host/production.param (any charge or spin in the .param left out: every job sets its own).
# With converge : false, the .param and k-points are used as given.
#   Called by: dwf_core.run_phase() (step A1)
#   Calls:     converged_value(), production_param_text(), dwf_core.Project.batch(),
#              dwf_core.Project.file(), dwf_core.Project.kspacing(), dwf_core.read_structure_file(),
#              dwf_core.write_table(), sc.kgrid()
def step_converge(p):
    d = p.phase_dir("A")
    text = p.file("param").read_text()
    if production_param_text(text)[1]:
        log(f"note: {p.file('param').name} sets " + " and ".join(production_param_text(text)[1])
            + "; production.param leaves them out (every job sets its own)")
    if not p.s.converge:
        m = sc.re.search(r"^\s*cut_off_energy\s*[:=\s]\s*([\d.]+)", text, sc.re.I | sc.re.M)
        (d / "production.param").write_text(production_param_text(text)[0])
        return {"cutoff": float(m.group(1)) if m else None, "kspacing": p.kspacing(),
                "param_file": "A_host/production.param", "_note": "converge : false - the "
                "project's .param and k-points are used as they are"}
    prim = core.read_structure_file(p.file("host_cell"))
    cuts = sorted(p.s.converge_cutoffs)
    spacings = sorted(p.s.converge_kspacings, reverse=True)       # coarse to fine
    fine = p.s.kspacing or spacings[-1]
    grids = {}
    for s in ([] if p.s.kspacing else spacings):
        grids.setdefault(tuple(sc.kgrid(prim.cell, s)), s)        # one job per distinct grid
    jobs = [{"name": f"cut_{c:g}", "atoms": prim, "kind": "sp", "symmetry": True,
             "kgrid": sc.kgrid(prim.cell, fine),
             "overrides": dict(HOST_OVERRIDES, cut_off_energy=f"{c:g} eV")} for c in cuts]
    jobs += [{"name": "k_" + "x".join(map(str, g)), "atoms": prim, "kind": "sp",
              "symmetry": True, "kgrid": list(g),
              "overrides": dict(HOST_OVERRIDES, cut_off_energy=f"{cuts[-1]:g} eV")}
             for g in grids]
    res = p.batch("A_host/converge", jobs, "host", text)
    if not res["complete"]:
        return None
    r, n = res["jobs"], len(prim)
    bad = [k for k, v in r.items() if v["energy"] is None or not v["scf_converged"]]
    if bad:
        die("the convergence single points " + ", ".join(bad) + " have no converged energy "
            "- check their .castep files in " + str(p.dir / "A_host/converge"))
    e_cut = [r[f"cut_{c:g}"]["energy"] / n for c in cuts]
    cutoff = converged_value(cuts, e_cut, p.s.converge_tol)
    rows = [[f"{c:g} eV", "x".join(map(str, sc.kgrid(prim.cell, fine))), e, e - e_cut[-1]]
            for c, e in zip(cuts, e_cut)]
    kspacing = p.s.kspacing
    if not kspacing:
        gl = list(grids)
        e_k = [r["k_" + "x".join(map(str, g))]["energy"] / n for g in gl]
        kspacing = converged_value([grids[g] for g in gl], e_k, p.s.converge_tol)
        rows += [[f"{cuts[-1]:g} eV", "x".join(map(str, g)) + f" ({grids[g]:g})", e, e - e_k[-1]]
                 for g, e in zip(gl, e_k)]
    table = core.write_table(d / "converge.txt", ["cut-off", "k-grid (spacing)", "E/atom (eV)",
                                                  "vs best"], rows,
                             f"A1 convergence, tolerance {p.s.converge_tol:g} eV/atom")
    print(table)
    if cutoff is None or kspacing is None:
        die("not converged within the values given: only the "
            + ("highest cut-off" if cutoff is None else "finest k-point spacing")
            + f" is within {p.s.converge_tol:g} eV/atom of itself. Add "
            + ("higher cut-offs to converge_cutoffs" if cutoff is None
               else "smaller spacings to converge_kspacings") + " and rerun (see "
            + str(d / "converge.txt") + ")")
    (d / "production.param").write_text(production_param_text(text, cutoff)[0])
    return {"cutoff": cutoff, "kspacing": kspacing, "param_file": "A_host/production.param",
            "_note": f"cut-off {cutoff:g} eV, k-point spacing {kspacing:g} 1/A"}


# --------------------------------------------------------------------------
# A2: bulk properties
# --------------------------------------------------------------------------
# A2.1: relaxes the host primitive cell (lattice and positions, with symmetry) at the production
# settings. Saves the relaxed cell (A_host/host_relaxed.json, and as a .cell for the size series)
# and the energy per atom, e_bulk.
#   Called by: dwf_core.run_phase() (step A2.1)
#   Calls:     dwf_core.Project.batch(), dwf_core.Project.cell_text(), dwf_core.Project.file(),
#              dwf_core.Project.kspacing(), dwf_core.Project.production_param(),
#              dwf_core.read_structure_file(), dwf_core.save_structure(), sc.final_atoms(),
#              sc.kgrid(), sc.parse_castep_input(), sc.write_cell()
def step_relax(p):
    d = p.phase_dir("A")
    prim = core.read_structure_file(p.file("host_cell"))
    res = p.batch("A_host/relax", [{"name": "host", "atoms": prim, "kind": "geom",
                                    "fix_cell": False, "symmetry": True,
                                    "overrides": HOST_OVERRIDES}],
                  "host", p.production_param(), spacing=p.kspacing())
    if not res["complete"]:
        return None
    r = res["jobs"]["host"]
    if r["status"] != "done" or r["energy"] is None:
        die(f"the host relaxation did not finish: {r['reason'] or 'see ' + r['dir']}")
    atoms = sc.final_atoms(res["calcs"]["host"])
    core.save_structure(d / "host_relaxed.json", atoms, energy=r["energy"])
    keep = [it for it in sc.parse_castep_input(p.cell_text())
            if not (it["type"] == "block" and it["name"] in sc.STRUCT_BLOCKS | sc.DROP_BLOCKS)
            and not (it["type"] == "kw" and it["key"] in sc.SET_KEYS)]
    sc.write_cell(d / "host_relaxed.cell", atoms, sc.kgrid(atoms.cell, p.kspacing()), keep,
                  f"host relaxed in A2.1 ({atoms.get_chemical_formula()})")
    a, b, c, al, be, ga = atoms.cell.cellpar()
    e_bulk = r["energy"] / len(atoms)
    return {"structure": "A_host/host_relaxed.json", "cell_file": "A_host/host_relaxed.cell",
            "energy": r["energy"], "e_bulk": e_bulk, "natoms": len(atoms),
            "cellpar": [a, b, c, al, be, ga],
            "_note": f"a b c = {a:.4f} {b:.4f} {c:.4f} A, e_bulk = {e_bulk:.6f} eV/atom"}


# A2.2: band edges of the relaxed host from a single point on a Gamma-centred grid at
# bands_kspacing (default half the production spacing): VBM, CBM and gap from the .bands file,
# which the Fermi-level analysis (E2) also uses.
#   Called by: dwf_core.run_phase() (step A2.2)
#   Calls:     dwf_core.Project.batch(), dwf_core.Project.host(), dwf_core.Project.kspacing(),
#              dwf_core.Project.production_param(), dwf_core.band_edges(), dwf_core.read_bands(),
#              sc.kgrid()
def step_bands(p):
    host = p.host()
    spacing = p.s.bands_kspacing or p.kspacing() / 2
    res = p.batch("A_host/bands", [{"name": "bands", "atoms": host, "kind": "sp",
                                    "symmetry": True, "kgrid": sc.kgrid(host.cell, spacing),
                                    "kpoint_offset": "gamma", "overrides": HOST_OVERRIDES}],
                  "host", p.production_param())
    if not res["complete"]:
        return None
    calc = res["calcs"]["bands"]
    path = calc.dir / f"{calc.seed}.bands"
    if res["jobs"]["bands"]["status"] != "done" or not path.exists():
        die(f"the band-edge single point did not finish or wrote no .bands ({calc.dir})")
    b = core.read_bands(path)
    vbm, cbm = core.band_edges(b)
    return {"vbm": vbm, "cbm": cbm, "gap": cbm - vbm, "bands_file": str(path.relative_to(p.dir)),
            "kgrid": res["jobs"]["bands"]["kgrid"],
            "_note": f"VBM {vbm:.4f} eV, CBM {cbm:.4f} eV, gap {cbm - vbm:.3f} eV"}


# A2.3: the dielectric constant for the charge corrections. A DFPT run (dielectric_task, default
# phonon+efield, at Gamma) on the relaxed host, with dielectric_param and dielectric_cell if given
# (CASTEP's DFPT needs norm-conserving pseudopotentials, fixed occupancies and no spin
# polarisation: without dielectric_param the production .param is used with fix_occupancy : true
# and spin_polarized : false, and a species_pot that doesn't look norm-conserving is warned about);
# the static value (electronic + ionic) is the average of the diagonal of CASTEP's DC permittivity.
# With dielectric set in the project file, that is used instead and nothing runs.
#   Called by: dwf_core.run_phase() (step A2.3)
#   Calls:     ultrasoft_entries(), dwf_core.Project.batch(), dwf_core.Project.cell_text(),
#              dwf_core.Project.file(), dwf_core.Project.host(), dwf_core.Project.kspacing(),
#              dwf_core.Project.production_param(), dwf_core.read_permittivity()
def step_dielectric(p):
    if p.s.dielectric:
        return {"eps0": p.s.dielectric, "eps_inf": p.s.dielectric_inf, "source": "project file",
                "_note": f"dielectric {p.s.dielectric:g} from the project file"}
    host = p.host()
    task = p.s.dielectric_task
    extra = ("%block phonon_kpoint_list\n  0.0 0.0 0.0 1.0\n%endblock phonon_kpoint_list"
             if "phonon" in task.lower() else None)
    param = p.file("dielectric_param").read_text() if p.s.dielectric_param else p.production_param()
    over = ({} if p.s.dielectric_param else
            dict(HOST_OVERRIDES, fix_occupancy="true", spin_polarized="false"))
    cell = p.file("dielectric_cell").read_text() if p.s.dielectric_cell else None
    usp = ultrasoft_entries(cell or p.cell_text())
    if usp:
        log("WARNING: CASTEP's DFPT needs norm-conserving pseudopotentials, but these don't look "
            "norm-conserving: " + "; ".join(usp) + ". Give dielectric_cell (a .cell with an NCP "
            "species_pot) and dielectric_param, or set dielectric, if this job fails.")
    res = p.batch("A_host/dielectric", [{"name": "dielectric", "atoms": host, "kind": "sp",
                                         "task": task, "symmetry": True, "cell_extra": extra,
                                         "overrides": over}],
                  "host", param, cell_text=cell, spacing=p.kspacing())
    if not res["complete"]:
        return None
    eps_inf, eps0 = core.read_permittivity(res["calcs"]["dielectric"].output())
    if eps_inf is None:
        die("no permittivity table in the dielectric run's .castep "
            f"({res['calcs']['dielectric'].castep}); set dielectric in the project file instead")
    if eps0 is None:
        die(f"task {task} gave only the optical permittivity; the charge correction needs the "
            "static one (electronic + ionic): use dielectric_task : phonon+efield, or set "
            "dielectric")
    e0, ei = float(np.trace(eps0) / 3), float(np.trace(eps_inf) / 3)
    aniso = float(np.ptp(np.diag(eps0)))
    if aniso > 0.1 * e0:
        log(f"WARNING: the dielectric tensor is anisotropic (diagonal {np.diag(eps0).round(3)}); "
            "the charge corrections use its average")
    return {"eps0": e0, "eps_inf": ei, "eps0_tensor": eps0, "eps_inf_tensor": eps_inf,
            "source": "DFPT", "_note": f"epsilon_0 {e0:.3f}, epsilon_inf {ei:.3f}"}


# --------------------------------------------------------------------------
# A3: competing phases and chemical potentials
# --------------------------------------------------------------------------
# The competing phases to relax, from %block phases (name, structure file, options: molecule,
# metal (fix_occupancy : false unless the line says otherwise), kspacing=..., anything else a
# .param keyword) or from doped (phases_from : doped, with the Materials Project). Returns a list
# of (name, Atoms, options dict).
#   Called by: step_phases()
#   Calls:     doped_phases(), dwf_core.read_structure_file()
def phase_list(p):
    if p.s.phases:
        out = []
        for row in p.s.phases:
            if len(row) < 2:
                die("each line of %block phases is  name  structure-file  [options]")
            opts = {"overrides": {}}
            for o in row[2:]:
                k, eq, v = o.partition("=")
                if not eq:
                    opts[k.lower()] = True
                elif k.lower() == "kspacing":
                    opts["kspacing"] = float(v)
                else:
                    opts["overrides"][k.lower()] = v
            if opts.get("metal"):
                opts["overrides"].setdefault("fix_occupancy", "false")
            out.append((row[0], core.read_structure_file(p.base / row[1]), opts))
        return out
    if p.s.phases_from == "doped":
        return doped_phases(p)
    return []


# Competing phases from doped's CompetingPhases (Materials Project entries within
# phases_e_above_hull of the hull, elemental references and diatomic molecules in a box). Needs
# doped and an MP API key (mp_api_key, or doped's own configuration).
#   Called by: phase_list()
#   Calls:     dwf_core.Project.host()
def doped_phases(p):
    import warnings
    warnings.filterwarnings("ignore")
    try:
        from doped.chemical_potentials import CompetingPhases
        from pymatgen.io.ase import AseAtomsAdaptor
    except ImportError as err:
        die(f"phases_from : doped needs doped (pip install doped): {err}")
    host = p.host()
    formula = host.get_chemical_formula(empirical=True)
    try:
        cp = CompetingPhases(formula, energy_above_hull=p.s.phases_e_above_hull,
                             extrinsic=p.s.dopants or None, api_key=p.s.mp_api_key)
    except Exception as err:                       # network, API key, ...
        die(f"doped could not get the competing phases from the Materials Project: {err}")
    out = []
    for kind, entries in (("molecule", cp.molecular_entries), ("metal", cp.metallic_entries),
                          ("insulator", cp.nonmetallic_entries)):
        for e in entries:
            name = e.data.get("doped_name") or e.name
            atoms = AseAtomsAdaptor.get_atoms(e.structure)
            opts = {"overrides": {}}
            if kind == "molecule":
                opts["molecule"] = True
                if e.composition.reduced_formula == "O2":
                    opts["overrides"].update(spin="2", spin_polarized="true")
            if kind == "metal":
                opts["metal"] = True
                opts["overrides"].update(fix_occupancy="false")
            out.append((sc.re.sub(r"[^\w.\-]", "_", name), atoms, opts))
    if not out:
        die("doped found no competing phases")
    return out


# A3.1: relaxes the competing phases at the production settings. Solids: lattice and positions,
# with symmetry, at the production k-point spacing (or the phase's own kspacing). Molecules
# ('molecule'): positions only in a fixed box, Gamma point, no symmetry. Metals ('metal') get
# fix_occupancy : false (give e.g. smearing_width=0.1 on their line too). With %block
# chemical_potentials nothing runs.
#   Called by: dwf_core.run_phase() (step A3.1)
#   Calls:     phase_list(), dwf_core.Project.batch(), dwf_core.Project.kspacing(),
#              dwf_core.Project.production_param(), dwf_core.counts(), sc.kgrid()
def step_phases(p):
    if p.s.chemical_potentials:
        return {"phases": {}, "_note": "chemical potentials from the project file"}
    phases = phase_list(p)
    if not phases:
        return {"phases": {}, "_note": "no competing phases given, so no chemical-potential "
                "limits (E2 then gives energies without the mu terms)"}
    jobs = []
    for name, atoms, o in phases:
        mol = bool(o.get("molecule"))
        job = {"name": name, "atoms": atoms, "kind": "geom", "fix_cell": mol,
               "symmetry": not mol, "overrides": dict(HOST_OVERRIDES, **o["overrides"])}
        if mol:
            job["kgrid"] = [1, 1, 1]
        elif o.get("kspacing"):
            job["kgrid"] = sc.kgrid(atoms.cell, o["kspacing"])
        jobs.append(job)
    res = p.batch("A_host/phases", jobs, "host", p.production_param(), spacing=p.kspacing())
    if not res["complete"]:
        return None
    out = {}
    for name, atoms, o in phases:
        r = res["jobs"][name]
        if r["status"] != "done" or r["energy"] is None:
            die(f"competing phase {name} did not finish: {r['reason'] or 'see ' + r['dir']}")
        out[name] = {"energy": r["energy"], "counts": core.counts(atoms), "natoms": len(atoms),
                     "molecule": bool(o.get("molecule"))}
    return {"phases": out, "_note": f"{len(out)} phases"}


# The chemical-potential limits from the energies of the host and its competing phases.
# The host's own elements: the corners of its stability region on a pymatgen phase diagram of the
# phases made of only those elements (named after the phases there, e.g. CaO-O2). Each dopant: at
# each corner, the highest value its phases allow, mu_X = min over phases P containing X of
# (E_P - sum of the other elements' n mu) / n_X, with the limiting phase. Stops if the host is not
# stable against its competing phases.
#   Called by: step_limits()
def chemical_limits(host_counts, host_energy, phases, dopants):
    try:
        from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
        from pymatgen.core import Composition
    except ImportError as err:
        die(f"the chemical-potential limits need pymatgen (pip install pymatgen): {err}")
    own = set(host_counts)
    entries = [PDEntry(Composition(host_counts), host_energy,
                       name=Composition(host_counts).reduced_formula)]
    for name, ph in phases.items():
        if set(ph["counts"]) <= own:
            entries.append(PDEntry(Composition(ph["counts"]), ph["energy"], name=name))
    missing = [el for el in own if not any(set(e.composition.as_dict()) == {el}
                                           for e in entries)]
    if missing:
        die("the chemical-potential limits need a phase of each pure element of the host; "
            "missing: " + ", ".join(missing) + " (add it to %block phases)")
    pd = PhaseDiagram(entries)
    ehull = pd.get_e_above_hull(entries[0])
    if ehull > 1e-6:
        decomp = pd.get_decomp_and_e_above_hull(entries[0])[0]
        die(f"the host is {ehull * 1000:.1f} meV/atom above the hull of its competing phases "
            "(it would decompose into " + ", ".join(e.name for e in decomp) + "), so it has no "
            "stable chemical potentials. Check those phases and their settings.")
    limits = {}
    for facet, mus in pd.get_all_chempots(Composition(host_counts)).items():
        mu = {str(el): float(v) for el, v in mus.items()}
        name = facet
        limiting = {}
        for x in dopants:
            best = None
            for pname, ph in phases.items():
                c = ph["counts"]
                if x not in c or not set(c) <= own | {x}:
                    continue
                val = (ph["energy"] - sum(n * mu[el] for el, n in c.items() if el != x)) / c[x]
                if best is None or val < best[0]:
                    best = (val, pname)
            if best is None:
                die(f"no competing phase contains the dopant {x} (add its element at least)")
            mu[x], limiting[x] = best
        limits[name] = {"mu": mu, "limiting": limiting}
    return limits


# A3.2: the chemical-potential limits (chemical_limits on the A3.1 phases and the A2.1 host), or
# the ones in %block chemical_potentials (name  El=value ..., eV per atom, absolute). Also gives
# each limit as Delta mu against the pure elements, where those phases were run.
#   Called by: dwf_core.run_phase() (step A3.2)
#   Calls:     chemical_limits(), dwf_core.Project.host(), dwf_core.counts(), dwf_core.write_table()
def step_limits(p):
    d = p.phase_dir("A")
    if p.s.chemical_potentials:
        limits = {}
        for row in p.s.chemical_potentials:
            mu = {}
            for tok in row[1:]:
                k, _, v = tok.partition("=")
                try:
                    mu[core.sc.element(k)] = float(v)
                except ValueError:
                    die(f"%block chemical_potentials: bad entry '{tok}' (write El=value)")
            limits[row[0]] = {"mu": mu, "limiting": {}}
        return {"limits": limits, "_note": f"{len(limits)} limits from the project file"}
    phases = p.out("phases")["phases"]
    if not phases:
        return {"limits": {}, "_note": "no competing phases, so no limits"}
    host = p.out("relax")
    hc = core.counts(p.host())
    limits = chemical_limits(hc, host["energy"], phases, p.s.dopants)
    ref = {}
    for name, ph in phases.items():
        if len(ph["counts"]) == 1:
            el, n = next(iter(ph["counts"].items()))
            e = ph["energy"] / n
            ref[el] = min(ref.get(el, e), e)
    rows = []
    for name, lim in limits.items():
        lim["delta_mu"] = {el: (v - ref[el]) if el in ref else None for el, v in lim["mu"].items()}
        rows.append([name] + [lim["delta_mu"].get(el) for el in sorted(lim["mu"])]
                    + [", ".join(f"{k}: {v}" for k, v in lim["limiting"].items())])
    els = sorted(next(iter(limits.values()))["mu"])
    print(core.write_table(d / "limits.txt", ["limit"] + [f"dmu_{el}" for el in els]
                           + ["dopants limited by"], rows,
                           "A3 chemical-potential limits (eV per atom, against the pure elements)"))
    return {"limits": limits, "references": ref, "_note": f"{len(limits)} limits"}


STEPS = [
    core.Step("converge", "A1", "converge the cut-off and k-point spacing", step_converge,
              keys=("converge", "converge_cutoffs", "converge_kspacings", "converge_tol",
                    "kspacing"), files=("host_cell", "param"),
              owns=("A_host/converge", "A_host/converge.txt", "A_host/production.param")),
    core.Step("relax", "A2.1", "relax the host (cell and positions)", step_relax,
              files=("host_cell",), needs=("converge",),
              owns=("A_host/relax", "A_host/host_relaxed.json", "A_host/host_relaxed.cell")),
    core.Step("bands", "A2.2", "band edges on a Gamma-centred grid", step_bands,
              keys=("bands_kspacing",), needs=("relax",), owns=("A_host/bands",)),
    core.Step("dielectric", "A2.3", "dielectric constant (DFPT)", step_dielectric,
              keys=("dielectric", "dielectric_inf", "dielectric_task"),
              files=("dielectric_param", "dielectric_cell"), needs=("relax",),
              owns=("A_host/dielectric",)),
    core.Step("phases", "A3.1", "relax the competing phases", step_phases,
              keys=("phases", "phases_from", "phases_e_above_hull", "dopants",
                    "chemical_potentials"), needs=("relax",), owns=("A_host/phases",)),
    core.Step("limits", "A3.2", "chemical-potential limits", step_limits,
              keys=("chemical_potentials", "dopants"), needs=("phases", "relax"),
              owns=("A_host/limits.txt",)),
]

if __name__ == "__main__":
    core.run_phase("A", STEPS, __doc__)
