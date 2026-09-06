"""Evaluation population, metrics, and seed-level statistical inference."""
from __future__ import annotations

import numpy as np
from scipy import stats

from .policies import ACCEPT

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260516


def policy_metrics(action, family, split, truth, prediction):
    """HAR, coverage, selective risk, nominal intervention, review/re-sense split."""
    ev = split == 2
    risk = ev & (family != 0)
    nominal = ev & (family == 0)
    wrong = prediction != truth

    accepted = action[risk] == ACCEPT
    har = float(np.mean(accepted & wrong[risk]))
    cov = float(accepted.mean())
    out = dict(
        HAR=har,
        Cov=cov,
        SelRisk=float(har / cov) if cov > 0 else float("nan"),
        NomInt=float(np.mean(action[nominal] != ACCEPT)),
        Review=float(np.mean(action[risk] == 1)),
        Resense=float(np.mean(action[risk] == 2)),
    )
    for c in range(1, 8):
        sel = ev & (family == c)
        acc_c = action[sel] == ACCEPT
        out[f"HAR_f{c}"] = float(np.mean(acc_c & wrong[sel]))
        out[f"Cov_f{c}"] = float(acc_c.mean())
    return out


def seed_mean(per_seed, policy, key):
    return float(np.mean([s[policy][key] for s in per_seed]))


def bootstrap_ci(values, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED, alpha=0.05):
    """Percentile interval over seed-level values.  Records are never resampled."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(v), size=(draws, len(v)))
    means = v[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def paired_wilcoxon(comparator, proposed):
    """Exact one-sided signed-rank test that comparator - proposed > 0."""
    d = np.asarray(comparator, float) - np.asarray(proposed, float)
    if np.all(d > 0):
        # exact floor for n all-positive differences
        p = 2.0 ** (-len(d))
    else:
        p = float(stats.wilcoxon(d, alternative="greater",
                                 mode="exact" if len(d) <= 25 else "auto").pvalue)
    return dict(n=int(len(d)), n_positive=int(np.sum(d > 0)),
                mean_difference=float(d.mean()), pvalue=float(p))


def holm(pvalues):
    """Holm-Bonferroni step-down adjustment."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj.tolist()
