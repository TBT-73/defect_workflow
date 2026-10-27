#!/usr/bin/env python3
"""
dwf_production.py - phase D of the defect workflow: production relaxations

Steps:
  D1  production  converged CASTEP relaxations, at the production settings (A1 cut-off and
                  k-point spacing, %block production_overrides, fixed cell), of the
                  structures phase C kept for each survivor and charge state, starting from
                  their screened spin; plus a single point on the pristine supercell at the
                  same settings. Jobs that run out of time are carried on automatically.
                  D2 (an untested idea): with freeze_radius > 0, atoms further than that from
                  every component of the defect are held fixed (%block ionic_constraints).

Usage:  python3 dwf_production.py <project>.dwf [--redo production] [--status] [--no-wait]
        [--prepare-only] [--resubmit-failed] [--force] [--example]
"""

import numpy as np

import dwf_core as core
from dwf_core import die, log, sc

# ==========================================================================================
# What the step reads and writes (under <workdir>/D_production/):
#   D1 production  C6, B5, A1, A2.1      -> relax/ (jobs), production.json, structures/
# ==========================================================================================


# The %block ionic_constraints that holds still every atom further than radius (Ang) from all of
# the defect's components (points: their positions; three constraints per atom: x, y and z).
# CASTEP numbers the atoms of each species in the order of the .cell, which keeps the order
# within each species (written_order).
#   Called by: step_production()
#   Calls:     sc.get_distances()
def freeze_block(atoms, points, radius):
    _, dist = sc.get_distances(np.atleast_2d(points), atoms.positions, cell=atoms.cell, pbc=True)
    dist = dist.min(axis=0)
    syms, lines, k = atoms.get_chemical_symbols(), [], 0
    for i, s in enumerate(syms):
        if dist[i] <= radius:
            continue
        n = syms[:i + 1].count(s)
        for v in ("1 0 0", "0 1 0", "0 0 1"):
            k += 1
            lines.append(f"  {k:6d}  {s:<3s} {n:6d}   {v}")
    if not lines:
        return None, 0
    return ("%block ionic_constraints\n" + "\n".join(lines) + "\n%endblock ionic_constraints",
            k // 3)


# D1: the production relaxations. One job per structure phase C kept (C6): its charge state,
# starting spin (spin_polarized : true with spin S when S is not 0; a spin-0 state runs without
# spin polarisation unless %block production_overrides sets spin_polarized), the production .param
# with %block production_overrides, the production k-points, fixed cell, no symmetry; with
# freeze_radius, the far atoms fixed (freeze_block). Plus the pristine supercell single point (with
# symmetry). Runner group 'production' (restarts on; a job that fails is flagged and left out of
# E). Results in D_production/production.json.
#   Called by: dwf_core.run_phase() (step D1)
#   Calls:     freeze_block(), dwf_core.Project.batch(), dwf_core.Project.host(),
#              dwf_core.Project.kspacing(), dwf_core.Project.production_param(),
#              dwf_core.host_supercell(), dwf_core.load_structure(), dwf_core.qtag(),
#              dwf_core.save_structure(), sc.final_atoms()
def step_production(p):
    d = p.phase_dir("D")
    todo = core.read_json(p, p.out("rank")["structures"])
    if not todo:
        die("phase C kept no structures, so there is nothing to relax")
    prim, P = p.host(), np.array(p.out("choose")["matrix"])
    host = core.host_supercell(prim, P)
    structs = p.out("prerelax")["structures"]
    jobs = [{"name": "host", "atoms": host, "kind": "sp", "symmetry": True,
             "overrides": dict(p.s.production_overrides)}]
    frozen = {}
    for t in todo:
        atoms, _ = core.load_structure(p.dir / t["structure"])
        _, info = core.load_structure(p.dir / structs[t["cand"]])
        S = t["spin"]
        mags = np.zeros(len(atoms))
        if S and info.get("anchor_index") is not None:
            mags[info["anchor_index"]] = S
        atoms.set_initial_magnetic_moments(mags)
        o = dict(p.s.production_overrides, charge=t["q"])
        if S:
            o.update(spin_polarized="true", spin=S)
        else:                               # (production_overrides can ask for spin_polarized)
            o.setdefault("spin_polarized", "false")
            o["spin"] = None
        name = f"{t['cand']}_{core.qtag(t['q'])}_k{t['k']}"
        job = {"name": name, "atoms": atoms, "kind": "geom", "symmetry": False, "overrides": o}
        if p.s.freeze_radius:
            block, n = freeze_block(atoms, info.get("points") or [info["centre"]],
                                    p.s.freeze_radius)
            if block:
                job["cell_extra"] = block
                frozen[name] = n
        jobs.append(job)
        t["job"] = name
    if frozen:
        log(f"D2: freeze_radius {p.s.freeze_radius:g} A holds "
            f"{min(frozen.values())}-{max(frozen.values())} atoms fixed in each relaxation "
            "(untested idea: check it against a full relaxation)")
    res = p.batch("D_production/relax", jobs, "production", p.production_param(),
                  spacing=p.kspacing())
    if not res["complete"]:
        return None
    hr = res["jobs"]["host"]
    if hr["status"] != "done" or hr["energy"] is None:
        die(f"the pristine supercell single point did not finish ({hr['dir']}): the formation "
            "energies need it")
    out, failed = [], []
    for t in todo:
        r = res["jobs"][t["job"]]
        e = r["energy"] if r["scf_converged"] else None
        if r["status"] != "done" or e is None:
            failed.append(f"{t['job']} ({(r['reason'] or 'no converged energy').splitlines()[0]})")
        path = None
        if e is not None:
            atoms = sc.final_atoms(res["calcs"][t["job"]])
            if atoms is not None:
                path = d / "structures" / f"{t['job']}.json"
                core.save_structure(path, atoms, energy=e)
                path = str(path.relative_to(p.dir))
        out.append(dict(t, energy=e if r["status"] == "done" else None, energy_any=e,
                        spin_final=r["spin"], status=r["status"], reason=r["reason"],
                        steps=r["geom_steps"], restarts=r["restarts"], final=path,
                        frozen=frozen.get(t["job"], 0)))
    for f in failed:
        log(f"WARNING: production run {f} - left out of the formation energies; fix it and "
            "rerun with --resubmit-failed")
    rel = core.write_json(p, d / "production.json", {"host_energy": hr["energy"], "runs": out})
    return {"results": rel, "host_energy": hr["energy"],
            "_note": f"{len(out) - len(failed)} of {len(out)} relaxations done"
                     + (f", {len(failed)} failed" if failed else "")}


STEPS = [
    core.Step("production", "D1", "production relaxations", step_production,
              keys=("production_overrides", "freeze_radius"),
              needs=("rank", "choose", "converge", "relax"),
              owns=("D_production",)),
]

if __name__ == "__main__":
    core.run_phase("D", STEPS, __doc__)
