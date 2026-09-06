"""End-to-end experiment driver: generates every number reported in the paper."""
from __future__ import annotations

import json

import numpy as np

from . import adversarial as ADV
from . import assurance as A
from . import generator as G
from . import metrics as MT
from . import policies as P

# operating points used for the risk-coverage frontier
FRONTIER_QUANTILES = (0.80, 0.85, 0.90, 0.93, 0.95, 0.965, 0.98, 0.99, 0.995, 0.998)
CONF_THRESHOLDS = tuple(np.round(np.arange(0.20, 0.96, 0.02), 3))
# predeclared matched-workload target, fitted on nominal calibration records only
TARGET_NOMINAL_INTERVENTION = 0.10
# leave-one-seed-out coordinate-selection threshold on calibration harmful acceptance
LOSO_THRESHOLD = 0.002
MATCH_GRID = tuple(np.round(np.arange(0.80, 0.9991, 0.001), 4))


def _prepare(seed):
    d = G.generate_seed(seed)
    train = d["split"] == 0
    nom_train = train & (d["family"] == 0)
    nom_cal = (d["split"] == 1) & (d["family"] == 0)

    T_cal = G.fit_temperature(d["xbar"][train], d["truth"][train])
    p_cal = G.calibrated_confidence(d["xbar"], T_cal)
    p_raw = G.raw_confidence(d["xbar"])

    rho_i, rho = A.encode(d["evidence"])
    envelope = A.nominal_envelope(rho, nom_train)
    delta_bar = d["degradation"].mean(axis=1)
    raw = A.raw_coordinates(d["evidence"], rho_i, rho, envelope, p_cal, delta_bar)
    d.update(T_cal=T_cal, p_cal=p_cal, p_raw=p_raw, rho_i=rho_i, rho=rho,
             envelope=envelope, delta_bar=delta_bar, raw=raw,
             nom_train=nom_train, nom_cal=nom_cal)
    return d


def _metrics(action, d):
    return MT.policy_metrics(action, d["family"], d["split"], d["truth"], d["prediction"])


def run_seed(seed):
    d = _prepare(seed)
    gates = A.fit_gates(d["raw"], d["nom_train"])
    dispersion = d["raw"]["V"]

    actions = P.fixed_policies(d["p_raw"], d["p_cal"], d["degradation"],
                               d["mission"], d["resensing"], dispersion)
    gated, codes = P.gated_policies(gates, d["delta_bar"], d["mission"], d["resensing"])
    actions.update(gated)

    result = {name: _metrics(a, d) for name, a in actions.items()}

    # ---- single-gate ablation of B6 -------------------------------------
    # semantics: the coordinate is removed from the row-2 union, from the row-1
    # re-sense trigger, and from the row-3 core set.  Stated explicitly because
    # a row-2-only removal gives materially different numbers.
    ablation = {}
    for drop in A.STRUCTURAL:
        keys = [k for k in A.STRUCTURAL if k != drop]
        core = [k for k in ("S", "F", "Dv") if k != drop] or ["J"]
        trig = "S" if drop != "S" else keys[0]
        act, _ = P.cascade(gates, keys, core, trig,
                           d["delta_bar"], d["mission"], d["resensing"])
        if drop == "S":                       # row 1 disappears with its trigger
            act2, _ = P.cascade(gates, keys, core, keys[0],
                                d["delta_bar"], d["mission"], d["resensing"])
            act = np.where(act2 == P.RESENSE, P.REVIEW, act2)
        ablation[drop] = _metrics(act, d)["HAR"]
    result["_ablation"] = ablation

    # calibration-split ablation increments, the only input to the
    # leave-one-seed-out coordinate selector.  Evaluation records are never read.
    cal = (d["split"] == 1) & (d["family"] != 0)
    wrong = d["prediction"] != d["truth"]

    def cal_har(keys, core, trig):
        act, _ = P.cascade(gates, list(keys), list(core), trig,
                           d["delta_bar"], d["mission"], d["resensing"])
        return float(np.mean((act[cal] == P.ACCEPT) & wrong[cal]))

    base = cal_har(A.STRUCTURAL, ("S", "F", "Dv"), "S")
    cal_inc = {}
    for drop in A.STRUCTURAL:
        keys = [k for k in A.STRUCTURAL if k != drop]
        core = [k for k in ("S", "F", "Dv") if k != drop] or ["J"]
        trig = "S" if drop != "S" else keys[0]
        cal_inc[drop] = cal_har(keys, core, trig) - base
    result["_cal_ablation"] = cal_inc

    # ---- risk-coverage frontier -----------------------------------------
    frontier = {k: [] for k in list(P.POLICY_SETS) + ["B2"]}
    for q in FRONTIER_QUANTILES:
        g = A.fit_gates(d["raw"], d["nom_train"], quantile=q)
        for name, (keys, core, trig) in P.POLICY_SETS.items():
            act, _ = P.cascade(g, list(keys), list(core), trig,
                               d["delta_bar"], d["mission"], d["resensing"])
            m = _metrics(act, d)
            frontier[name].append(dict(q=float(q), HAR=m["HAR"], Cov=m["Cov"],
                                       NomInt=m["NomInt"], HAR_f6=m["HAR_f6"],
                                       HAR_f7=m["HAR_f7"], Cov_f6=m["Cov_f6"],
                                       Cov_f7=m["Cov_f7"]))
    for th in CONF_THRESHOLDS:
        act = np.where(d["p_cal"] >= th, P.ACCEPT, P.REVIEW)
        m = _metrics(act, d)
        frontier["B2"].append(dict(q=float(th), HAR=m["HAR"], Cov=m["Cov"],
                                   NomInt=m["NomInt"], HAR_f6=m["HAR_f6"],
                                   HAR_f7=m["HAR_f7"], Cov_f6=m["Cov_f6"],
                                   Cov_f7=m["Cov_f7"]))
    result["_frontier"] = frontier

    # ---- matched nominal-workload operating points -----------------------
    # the gate quantile / confidence threshold is chosen per seed so that the
    # nominal intervention measured on NOMINAL CALIBRATION RECORDS equals the
    # predeclared target.  No evaluation record is consulted.
    matched = {}
    for name, (keys, core, trig) in P.POLICY_SETS.items():
        best = None
        for q in MATCH_GRID:
            g = A.fit_gates(d["raw"], d["nom_train"], quantile=q)
            act, _ = P.cascade(g, list(keys), list(core), trig,
                               d["delta_bar"], d["mission"], d["resensing"])
            ni = float(np.mean(act[d["nom_cal"]] != P.ACCEPT))
            gap = abs(ni - TARGET_NOMINAL_INTERVENTION)
            if best is None or gap < best[0]:
                best = (gap, q, act)
        matched[name] = _metrics(best[2], d)
        matched[name]["param"] = float(best[1])
    best = None
    for th in np.round(np.arange(0.20, 0.96, 0.002), 4):
        act = np.where(d["p_cal"] >= th, P.ACCEPT, P.REVIEW)
        ni = float(np.mean(act[d["nom_cal"]] != P.ACCEPT))
        gap = abs(ni - TARGET_NOMINAL_INTERVENTION)
        if best is None or gap < best[0]:
            best = (gap, th, act)
    matched["B2"] = _metrics(best[2], d)
    matched["B2"]["param"] = float(best[1])
    result["_matched"] = matched

    # ---- observation-indistinguishability diagnostic ----------------------
    # fraction of family-c evaluation records whose compressed observation lies
    # inside the nominal central 96% interval on every scalar coordinate.
    obs, obs_coord = {}, {}
    lo, hi = {}, {}
    for k in ("Pc", "De", "V"):
        lo[k], hi[k] = np.quantile(d["raw"][k][d["nom_train"]], [0.02, 0.98])
    for k in ("mission", "resensing"):
        lo[k], hi[k] = np.quantile(d[k][d["nom_train"]], [0.02, 0.98])
    ev = d["split"] == 2
    for c in (0, 5, 6, 7):
        sel = ev & (d["family"] == c)
        inside = np.ones(int(sel.sum()), bool)
        for k in ("Pc", "De", "V"):
            v = d["raw"][k][sel]
            inside &= (v >= lo[k]) & (v <= hi[k])
        for k in ("mission", "resensing"):
            v = d[k][sel]
            inside &= (v >= lo[k]) & (v <= hi[k])
        obs[f"f{c}"] = float(inside.mean())
        for k in ("Pc", "De", "V"):
            v = d["raw"][k][sel]
            obs_coord[f"f{c}:{k}"] = float(np.mean((v >= lo[k]) & (v <= hi[k])))
        for k in ("mission", "resensing"):
            v = d[k][sel]
            obs_coord[f"f{c}:{k}"] = float(np.mean((v >= lo[k]) & (v <= hi[k])))
    result["_observation"] = obs
    result["_observation_coord"] = obs_coord

    result["_meta"] = dict(
        seed=int(seed), T_cal=float(d["T_cal"]),
        acc_nominal=float(np.mean(d["prediction"][d["family"] == 0] == d["truth"][d["family"] == 0])),
        acc_f6=float(np.mean(d["prediction"][d["family"] == 6] == d["truth"][d["family"] == 6])),
        acc_f7=float(np.mean(d["prediction"][d["family"] == 7] == d["truth"][d["family"] == 7])),
        praw_nominal=float(d["p_raw"][d["family"] == 0].mean()),
        praw_f6=float(d["p_raw"][d["family"] == 6].mean()),
        praw_f7=float(d["p_raw"][d["family"] == 7].mean()),
        pcal_nominal=float(d["p_cal"][d["family"] == 0].mean()),
        pcal_f6=float(d["p_cal"][d["family"] == 6].mean()),
        pcal_f7=float(d["p_cal"][d["family"] == 7].mean()),
        V_nominal=float(dispersion[d["family"] == 0].mean()),
        V_f6=float(dispersion[d["family"] == 6].mean()),
        V_f7=float(dispersion[d["family"] == 7].mean()),
        V_max=float(dispersion.max()),
    )
    return result


N_ADVERSARIAL_SEEDS = 3          # A2 is the expensive experiment
N_PROBE_SEEDS = 20


def run_adversarial(seeds=G.SEEDS, verbose=True):
    """Supervised representation probes and gate-aware attacks.  Reported separately
    because the evasion search dominates the runtime."""
    probe, mech, evade = [], [], []
    for i, s in enumerate(seeds):
        d = _prepare(s)
        gates = A.fit_gates(d["raw"], d["nom_train"])
        if i < N_PROBE_SEEDS:
            if verbose:
                print(f"  representation-probe seed {s} ...", flush=True)
            probe.append({f"f{fam}": ADV.representation_probe_frontier(d, family=fam, seed=s)
                          for fam in (6, 7)})
        if i < N_ADVERSARIAL_SEEDS:
            if verbose:
                print(f"  attacks seed {s} ...", flush=True)
            mech.append(ADV.attack_mechanism(d, gates, s))
            evade.append({name: [ADV.attack_evasion(d, gates, s, name, eps=e)
                                 for e in ADV.EPS_GRID]
                          for name in ADV.ATTACK_SETS})
    return dict(probe=probe, mechanism=mech, evasion=evade)


def aggregate_adversarial(raw):
    out = {}
    for fam in ("f6", "f7"):
        entry = {}
        for rep in ("obs", "struct"):
            curves = np.array([[p[1] for p in r[fam][rep]["curve"]] for r in raw["probe"]])
            entry[rep] = dict(
                ni_grid=list(ADV.NI_GRID),
                rejection=[float(x) for x in curves.mean(axis=0)],
                balanced_accuracy=float(np.mean([r[fam][rep]["balanced_accuracy"]
                                                 for r in raw["probe"]])),
                tv_lower_bound=float(np.mean([r[fam][rep]["tv_lower_bound"]
                                              for r in raw["probe"]])),
            )
        out[fam] = entry
    probe = {"n_seeds": len(raw["probe"]), "by_family": out}

    keyed = {}
    for rows in raw["mechanism"]:
        for r in rows:
            keyed.setdefault((r["rho"], r["innovation"], r["sigma"]), []).append(r)
    mech = []
    for k, v in keyed.items():
        rec = dict(rho=k[0], innovation=k[1], sigma=k[2])
        for f in v[0]:
            if f not in ("rho", "innovation", "sigma"):
                rec[f] = float(np.mean([x[f] for x in v]))
        mech.append(rec)
    mech.sort(key=lambda r: -r["harm_B6c"])

    evade = {}
    for name in ADV.ATTACK_SETS:
        evade[name] = []
        for i, eps in enumerate(ADV.EPS_GRID):
            vals = [r[name][i] for r in raw["evasion"]]
            evade[name].append({k: float(np.mean([v[k] for v in vals]))
                                for k in vals[0]})
    return dict(probe=probe, mechanism=mech, evasion=evade,
                n_attack_seeds=len(raw["mechanism"]))


def run_all(seeds=G.SEEDS, verbose=True):
    per_seed = []
    for s in seeds:
        if verbose:
            print(f"  seed {s} ...", flush=True)
        per_seed.append(run_seed(s))
    return per_seed


def aggregate(per_seed):
    out = {"version": G.VERSION, "n_seeds": len(per_seed),
           "seeds": [r["_meta"]["seed"] for r in per_seed]}

    primary = {}
    for pol in P.POLICY_ORDER:
        vals = {k: [r[pol][k] for r in per_seed]
                for k in per_seed[0][pol] if isinstance(per_seed[0][pol][k], float)}
        entry = {k: float(np.mean(v)) for k, v in vals.items()}
        for k in ("HAR", "HAR_f6", "HAR_f7"):
            entry[k + "_ci"] = MT.bootstrap_ci(vals[k])
        entry["HAR_seedwise"] = vals["HAR"]
        entry["HAR_f7_seedwise"] = vals["HAR_f7"]
        primary[pol] = entry
    out["primary"] = primary

    full = np.mean([r["B6"]["HAR"] for r in per_seed])
    out["ablation"] = {k: dict(HAR=float(np.mean([r["_ablation"][k] for r in per_seed])),
                               increase=float(np.mean([r["_ablation"][k] for r in per_seed]) - full))
                       for k in A.STRUCTURAL}
    out["ablation_full_HAR"] = float(full)

    front = {}
    for pol, pts in per_seed[0]["_frontier"].items():
        agg = []
        for i in range(len(pts)):
            agg.append({k: float(np.mean([r["_frontier"][pol][i][k] for r in per_seed]))
                        for k in pts[i]})
        front[pol] = agg
    out["frontier"] = front

    matched = {}
    for pol in ("B2", "B7", "B6", "B6c", "B8", "B8c"):
        vals = {k: [r["_matched"][pol][k] for r in per_seed] for k in per_seed[0]["_matched"][pol]}
        matched[pol] = {k: float(np.mean(v)) for k, v in vals.items()}
        matched[pol]["HAR_f7_seedwise"] = vals["HAR_f7"]
        matched[pol]["HAR_seedwise"] = vals["HAR"]
    out["matched"] = matched
    out["matched_target_nominal_intervention"] = TARGET_NOMINAL_INTERVENTION

    out["observation"] = {k: float(np.mean([r["_observation"][k] for r in per_seed]))
                          for k in per_seed[0]["_observation"]}
    out["observation_by_coordinate"] = {
        k: float(np.mean([r["_observation_coord"][k] for r in per_seed]))
        for k in per_seed[0]["_observation_coord"]}

    # leave-one-seed-out coordinate selection: the set applied to seed s is
    # determined only by the calibration ablations of the other 19 seeds.
    loso = {}
    for i, r in enumerate(per_seed):
        others = [x["_cal_ablation"] for j, x in enumerate(per_seed) if j != i]
        loso[r["_meta"]["seed"]] = sorted(
            k for k in A.STRUCTURAL
            if float(np.mean([o[k] for o in others])) > LOSO_THRESHOLD)
    sets = {tuple(v) for v in loso.values()}
    out["loso_selection"] = {
        "threshold": LOSO_THRESHOLD,
        "fitted_on": "calibration risk records of the other 19 seeds only",
        "per_seed": {str(k): v for k, v in loso.items()},
        "unanimous": len(sets) == 1,
        "selected_set": sorted(sets.pop()) if len(sets) == 1 else None,
        "mean_calibration_increase": {
            k: float(np.mean([r["_cal_ablation"][k] for r in per_seed]))
            for k in A.STRUCTURAL},
    }

    # aggregate risk-coverage frontier compared at matched coverage
    grid = [0.08, 0.12, 0.16, 0.20, 0.24, 0.28]
    matched_cov = {}
    for pol in ("B2", "B6", "B6c", "B7", "B8c"):
        pts = sorted(out["frontier"][pol], key=lambda p: p["Cov"])
        xs = [p["Cov"] for p in pts]
        ys = [p["HAR"] for p in pts]
        matched_cov[pol] = [
            (float(c), float(np.interp(c, xs, ys))) if xs[0] <= c <= xs[-1] else (float(c), None)
            for c in grid]
    out["matched_coverage"] = matched_cov
    out["meta"] = {k: float(np.mean([r["_meta"][k] for r in per_seed]))
                   for k in per_seed[0]["_meta"] if k != "seed"}

    # prespecified paired comparisons on the matched-workload operating point
    tests = {}
    # B6 carries no coordinate selection whatsoever, so the two primary tests
    # are independent of the selection procedure; the third tests the refinement.
    for comp in ("B2", "B7"):
        tests[f"B6_vs_{comp}_HAR_f7"] = MT.paired_wilcoxon(
            out["matched"][comp]["HAR_f7_seedwise"], out["matched"]["B6"]["HAR_f7_seedwise"])
    tests["B6c_vs_B7_HAR_f7"] = MT.paired_wilcoxon(
        out["matched"]["B7"]["HAR_f7_seedwise"], out["matched"]["B6c"]["HAR_f7_seedwise"])
    adj = MT.holm([tests[k]["pvalue"] for k in tests])
    for k, a in zip(tests, adj):
        tests[k]["pvalue_holm"] = float(a)
    out["tests"] = tests
    return out


def save(agg, path):
    with open(path, "w") as fh:
        json.dump(agg, fh, indent=2)
