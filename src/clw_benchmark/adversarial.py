"""Representation limits and gate-aware adversaries.

Two experiments live here, both added in package version 2 to close the two
open items of the pre-submission audit.

1.  `representation_probe_frontier` compares nominal/mechanism separation
    exposed by each *representation* with supervision and model class held constant.  A
    gradient-boosted discriminator is fitted on the training split with true
    mechanism labels, which no deployable router has, once on the compressed
    observation and once on the structural coordinates.  Sweeping its threshold
    traces an empirical supervised (nominal intervention, rejection) curve.
    Proposition 2 bounds every router by total variation, but this fitted model
    is not a certified likelihood-ratio router and therefore does not upper-bound
    all routers on the representation. Its balanced accuracy yields TV >= 2*BA-1.

2.  `attack_mechanism` and `attack_evasion` implement a gate-aware adversary
    that knows the encoder constants, the coordinate set, the fitted per-seed
    thresholds and the cascade rows.  It cannot alter the nominal training data
    and therefore cannot move the thresholds.
"""
from __future__ import annotations

import numpy as np

from . import assurance as A
from . import generator as G
from . import policies as P

OBS_KEYS = list(A.SCALAR)
STR_KEYS = list(A.STRUCTURAL)
CONTEXT = ("mission", "resensing")
NI_GRID = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
EPS_GRID = (0.05, 0.10, 0.20)

ATTACK_SETS = {
    # A2 focuses on the three configurations compared in the conference figure.
    "B7": list(A.SCALAR),
    "B6c": list(P.COMPACT),
    "B8c": list(P.COMPACT) + list(A.SCALAR),
}


# ---------------------------------------------------------------------------
# 1. supervised representation probe
# ---------------------------------------------------------------------------
def _design(prep, keys):
    cols = [prep["raw"][k] for k in keys] + [prep[c] for c in CONTEXT]
    return np.column_stack(cols)


def representation_probe_frontier(prep, family=7, seed=0):
    """Empirical (NI, rejection) curve of the same supervised probe per representation.

    The fitted curve is not a certified upper bound on all routers measurable
    with respect to the representation.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    fam = prep["family"]
    tr = (prep["split"] == 0) & ((fam == 0) | (fam == family))
    ev = (prep["split"] == 2) & ((fam == 0) | (fam == family))
    y_tr = (fam[tr] == family).astype(int)
    y_ev = (fam[ev] == family).astype(int)

    out = {}
    for name, keys in (("obs", OBS_KEYS), ("struct", STR_KEYS)):
        X = _design(prep, keys)
        clf = HistGradientBoostingClassifier(max_iter=300, random_state=seed)
        clf.fit(X[tr], y_tr)
        s = clf.predict_proba(X[ev])[:, 1]
        s0, s1 = s[y_ev == 0], s[y_ev == 1]
        thr = np.quantile(s0, 1.0 - np.array(NI_GRID))
        out[name] = dict(
            curve=[(float(ni), float(np.mean(s1 >= t)))
                   for ni, t in zip(NI_GRID, thr)],
            balanced_accuracy=float(0.5 * (np.mean(s1 >= 0.5) + np.mean(s0 < 0.5))),
        )
        out[name]["tv_lower_bound"] = 2.0 * out[name]["balanced_accuracy"] - 1.0
    return out


# Backward-compatible alias for v2.0 package users.
def oracle_frontier(prep, family=7, seed=0):
    return representation_probe_frontier(prep, family=family, seed=seed)


# ---------------------------------------------------------------------------
# 2. gate-aware adversary
# ---------------------------------------------------------------------------
def _standardise(raw, gates):
    z = {}
    for k, val in raw.items():
        o = A.ORIENTATION[k]
        mu, sd = gates["mu"][k], gates["sd"][k]
        if o == "+":
            z[k] = (val - mu) / sd
        elif o == "-":
            z[k] = (mu - val) / sd
        else:
            z[k] = np.abs(val - mu) / sd
    return z, {k: z[k] / gates["tau"][k] for k in raw}


def _coords(X, prep, gates, p_cal, delta_bar, keys=None):
    keys = list(gates["tau"]) if keys is None else list(dict.fromkeys(keys))
    structural = [k for k in keys if k in A.STRUCTURAL]
    if structural:
        rho_i, rho = A.encode(X)
    else:
        # Scalar-only policies do not require density-state construction.
        n = len(X)
        rho_i = np.empty((n, G.M, G.D, G.D))
        rho = np.empty((n, G.D, G.D))
    raw = A.raw_coordinates_subset(X, rho_i, rho, prep["envelope"], p_cal,
                                   delta_bar, keys)
    return _standardise(raw, gates)


def _synth_residual(rng, n, rho_adv, innovation, sigma):
    """Adversary-parameterised correlated-residual evidence."""
    X = np.empty((n, G.M, G.D))
    y = rng.choice(G.D, size=n, p=G.PRIORS[7])
    base = np.eye(G.D)[0][None, :]
    for t in range(n):
        nu = np.empty((G.M, G.D))
        nu[0] = rng.normal(0.0, 0.20, size=G.D)
        for i in range(1, G.M):
            nu[i] = rho_adv * nu[i - 1] + rng.normal(0.0, innovation, size=G.D)
        X[t] = base + nu + rng.normal(0.0, sigma, size=(G.M, G.D))
    return X, y


def attack_mechanism(prep, gates, seed, n=1500,
                     rho_grid=(0.0, 0.3, 0.5, 0.7, 0.90, 0.95, 0.99),
                     innovation_grid=(0.05, 0.10, 0.20, 0.35),
                     sigma_grid=(0.05, 0.15)):
    """A1: the adversary redesigns the mechanism knowing the fitted thresholds."""
    rng = np.random.default_rng(seed + 991)
    rows = []
    for rho_adv in rho_grid:
        for innovation in innovation_grid:
            for sigma in sigma_grid:
                X, y = _synth_residual(rng, n, rho_adv, innovation, sigma)
                xbar = X.mean(axis=1)
                yhat = G.softmax(xbar / G.T_AI).argmax(axis=1)
                p_cal = G.softmax(xbar / G.T_AI / prep["T_cal"]).max(axis=1)
                dbar = rng.uniform(*G.DELTA_RANGE[0], size=(n, G.M)).mean(axis=1)
                miss = np.clip(G.MBAR[0] + rng.normal(0, G.MISSION_SD, n), 0, 1)
                res = rng.uniform(G.RESENSE_LO, G.RESENSE_HI, n)
                z, u = _coords(X, prep, gates, p_cal, dbar)
                gg = dict(z=z, tau=gates["tau"], u=u)
                wrong = yhat != y
                row = dict(rho=float(rho_adv), innovation=float(innovation),
                           sigma=float(sigma), error_rate=float(wrong.mean()),
                           mean_confidence=float(p_cal.mean()))
                for pol, (keys, core, trig) in P.POLICY_SETS.items():
                    act, _ = P.cascade(gg, list(keys), list(core), trig, dbar, miss, res)
                    row[f"harm_{pol}"] = float(np.mean(wrong & (act == P.ACCEPT)))
                    row[f"reject_{pol}"] = float(np.mean(act != P.ACCEPT))
                rows.append(row)
    return rows


def attack_evasion(prep, gates, seed, policy, eps=0.10, n_records=150, iters=250):
    """A2: budgeted white-box search for a harmful record accepted by the full cascade.

    A successful evasion must satisfy ``action == ACCEPT`` after all cascade rows,
    not merely clear row-2 gate thresholds.  In particular, when mission severity
    is high, the row-3 core margin must also remain below ``CORE_MARGIN=0.60``.
    The perturbation is constrained to ||Delta||_F <= eps ||x||_F and the upstream
    prediction must remain wrong.
    """
    keys, core, trigger = P.POLICY_SETS[policy]
    keys, core = list(keys), list(core)
    ev = prep["split"] == 2
    pool = np.flatnonzero(ev & (prep["family"] == 7)
                          & (prep["prediction"] != prep["truth"]))
    rng = np.random.default_rng(seed + 7)
    idx = rng.choice(pool, size=min(n_records, len(pool)), replace=False)

    X0 = prep["evidence"][idx]
    y0 = prep["truth"][idx]
    dbar = prep["delta_bar"][idx]
    mission = prep["mission"][idx]
    resensing = prep["resensing"][idx]
    budget = np.linalg.norm(X0.reshape(len(idx), -1), axis=1) * eps

    def acceptance_margin(X):
        xb = X.mean(axis=1)
        pc = G.softmax(xb / G.T_AI / prep["T_cal"]).max(axis=1)
        z, u = _coords(X, prep, gates, pc, dbar, set(keys) | set(core))
        gate_margin = np.maximum.reduce([u[k] for k in keys])
        core_margin = np.maximum.reduce([u[k] for k in core])
        severity_margin = np.where(mission >= P.SEVERITY,
                                   core_margin / P.CORE_MARGIN, 0.0)
        # margin < 1 is equivalent to clearing row 2 and, where active, row 3.
        return np.maximum(gate_margin, severity_margin), xb.argmax(axis=1), z, u

    m0, _, _, _ = acceptance_margin(X0)
    best = m0.copy()
    Xb = X0.copy()
    step = budget / 4.0
    for _ in range(iters):
        cand = Xb + rng.normal(0.0, 1.0, X0.shape) * step[:, None, None]
        delta = (cand - X0).reshape(len(idx), -1)
        nrm = np.linalg.norm(delta, axis=1)
        over = nrm > budget
        if over.any():
            f = (budget[over] / nrm[over])[:, None]
            cand[over] = X0[over] + (delta[over] * f).reshape(-1, G.M, G.D)
        m, pred, _, _ = acceptance_margin(cand)
        ok = (m < best) & (pred != y0)
        best = np.where(ok, m, best)
        Xb[ok] = cand[ok]
        step = step * 0.995

    def accepted(X):
        xb = X.mean(axis=1)
        pc = G.softmax(xb / G.T_AI / prep["T_cal"]).max(axis=1)
        z, u = _coords(X, prep, gates, pc, dbar, set(keys) | set(core))
        gg = dict(z=z, tau=gates["tau"], u=u)
        act, _ = P.cascade(gg, keys, core, trigger, dbar, mission, resensing)
        return act == P.ACCEPT

    clean_accept = accepted(X0)
    attacked_accept = accepted(Xb)
    return dict(eps=float(eps), n=int(len(idx)),
                evaded_clean=float(np.mean(clean_accept)),
                evaded_attacked=float(np.mean(attacked_accept)),
                median_margin_clean=float(np.median(m0)),
                median_margin_attacked=float(np.median(best)))

