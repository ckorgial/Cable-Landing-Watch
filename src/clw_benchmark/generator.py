"""CLW v1.2 -- Cable-Landing Watch synthetic decision-level benchmark generator.

Version 1.2 differs from v1.1 in exactly one respect, documented in
NOEMAC-CLW-TR-2026 v1.2 Sec. V-C: the auxiliary channels (degradation range and
mission-severity base) of the two common-mode families -- coordinated spoof
(c=6) and correlated residual (c=7) -- are matched to the nominal family.

Rationale.  In v1.1 those families carried mission-severity base 0.85 and
degradation ranges [0.05,0.15] and [0.10,0.25] against nominal 0.10 and
[0.00,0.10].  Mission severity therefore separated nominal from the common-mode
families with zero overlap, and mission severity is inside the observation
tuple available to a scalar authority router.  Any claim that a scalar router
is representationally unable to separate those populations was consequently
untestable on v1.1: the router was handed a deterministic family label.
Version 1.2 removes that leak.  No other constant, family equation, prior,
split rule, seed, or metric changes.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Frozen constants
# ---------------------------------------------------------------------------
VERSION = "1.2"
M = 3                 # modalities
D = 5                 # classes == evidence dimension (K = d)
N_FAMILIES = 8
N_PER_FAMILY = 1000
N_TRAIN, N_CAL, N_EVAL = 300, 150, 550
T_AI = 0.45           # frozen upstream emulator softmax temperature
CAL_GRID = np.round(np.arange(0.5, 6.0 + 1e-9, 0.1), 4)

SEEDS = (20260516, 7, 11, 23, 41, 97, 137, 211, 313, 503,
         809, 1201, 1993, 2718, 3141, 4242, 5501, 6007, 7919, 8887)

FAMILY_NAMES = (
    "nominal benign",
    "UUV low SNR",
    "ambiguous diver/clutter",
    "novel motion",
    "high consequence",
    "cross-sensor contradiction",
    "coordinated spoof",
    "correlated residual",
)

# Class priors pi_c (rows sum to one)
PRIORS = np.array([
    [0.85, 0.10, 0.02, 0.02, 0.01],
    [0.10, 0.10, 0.10, 0.60, 0.10],
    [0.10, 0.40, 0.40, 0.05, 0.05],
    [0.10, 0.10, 0.10, 0.30, 0.40],
    [0.20, 0.10, 0.20, 0.40, 0.10],
    [0.05, 0.10, 0.30, 0.45, 0.10],
    [0.10, 0.10, 0.20, 0.50, 0.10],
    [0.10, 0.10, 0.20, 0.50, 0.10],
])

# Independent evidence-noise standard deviation varsigma_c
SIGMA = np.array([0.10, 0.45, 0.20, 0.25, 0.18, 0.10, 0.04, 0.05])

# Degradation interval [delta_c^-, delta_c^+].  v1.2: rows 6 and 7 == row 0.
DELTA_RANGE = np.array([
    [0.00, 0.10],
    [0.40, 0.80],
    [0.10, 0.30],
    [0.20, 0.50],
    [0.05, 0.20],
    [0.05, 0.20],
    [0.00, 0.10],   # v1.2 (v1.1 was [0.05, 0.15])
    [0.00, 0.10],   # v1.2 (v1.1 was [0.10, 0.25])
])

# Mission-severity base mbar_c.  v1.2: rows 6 and 7 == row 0.
MBAR = np.array([0.10, 0.40, 0.30, 0.70, 1.00, 0.85,
                 0.10,   # v1.2 (v1.1 was 0.85)
                 0.10])  # v1.2 (v1.1 was 0.85)

MISSION_SD = 0.04
RESENSE_LO, RESENSE_HI = 0.40, 0.95


def softmax(z, axis=-1):
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def twin(y: int) -> int:
    """Confusable-label map of family 2."""
    return 2 if y == 1 else 1


def _family_evidence(c, y, rng):
    """Return the (3, 5) noiseless family-specific evidence base."""
    ey = np.eye(D)[y]
    if c in (0, 1, 4):
        return np.tile(ey, (M, 1))
    if c == 2:
        return np.tile(0.55 * ey + 0.45 * np.eye(D)[twin(y)], (M, 1))
    if c == 3:
        n = rng.normal(size=D)
        return np.tile(0.60 * ey + 0.60 * n / np.linalg.norm(n), (M, 1))
    if c == 5:
        pool = [k for k in range(D) if k != y]
        lab = list(rng.choice(pool, size=M, replace=False))
        lab[0] = 0                       # force modality 1 to benign
        return np.stack([np.eye(D)[l] for l in lab])
    if c == 6:
        b = rng.normal(0.0, 0.30, size=D)
        return np.tile(np.eye(D)[0] + 0.90 * b, (M, 1))
    if c == 7:
        nu = np.empty((M, D))
        nu[0] = rng.normal(0.0, 0.20, size=D)
        for i in (1, 2):
            nu[i] = 0.90 * nu[i - 1] + rng.normal(0.0, 0.10, size=D)
        return np.eye(D)[0][None, :] + nu
    raise ValueError(f"unknown family {c}")


def generate_seed(seed: int) -> dict:
    """Generate one complete CLW seed.

    Draw order per record is normative: truth, degradation triple, mission
    severity, re-sensing feasibility, family-specific shared draw, evidence
    noise.  A single numpy Generator is initialised once per seed.
    """
    rng = np.random.default_rng(seed)
    n = N_FAMILIES * N_PER_FAMILY
    evidence = np.empty((n, M, D))
    truth = np.empty(n, dtype=np.int64)
    family = np.empty(n, dtype=np.int64)
    degradation = np.empty((n, M))
    mission = np.empty(n)
    resensing = np.empty(n)

    idx = 0
    for c in range(N_FAMILIES):
        for _ in range(N_PER_FAMILY):
            y = int(rng.choice(D, p=PRIORS[c]))
            delta = rng.uniform(DELTA_RANGE[c, 0], DELTA_RANGE[c, 1], size=M)
            m = float(np.clip(MBAR[c] + rng.normal(0.0, MISSION_SD), 0.0, 1.0))
            r = float(rng.uniform(RESENSE_LO, RESENSE_HI))
            base = _family_evidence(c, y, rng)
            noise = rng.normal(0.0, SIGMA[c], size=(M, D))
            evidence[idx] = base + noise
            truth[idx] = y
            family[idx] = c
            degradation[idx] = delta
            mission[idx] = m
            resensing[idx] = r
            idx += 1

    # frozen upstream emulator
    xbar = evidence.mean(axis=1)
    probs = softmax(xbar / T_AI)
    prediction = probs.argmax(axis=1)

    # stratified split, independent deterministic permutation inside each family
    split = np.empty(n, dtype=np.int64)   # 0 train, 1 cal, 2 eval
    for c in range(N_FAMILIES):
        pos = np.flatnonzero(family == c)
        perm = rng.permutation(pos)
        split[perm[:N_TRAIN]] = 0
        split[perm[N_TRAIN:N_TRAIN + N_CAL]] = 1
        split[perm[N_TRAIN + N_CAL:]] = 2

    return dict(seed=seed, evidence=evidence, truth=truth, prediction=prediction,
                probabilities=probs, degradation=degradation, mission=mission,
                resensing=resensing, family=family, split=split, xbar=xbar)


def fit_temperature(xbar_train, y_train):
    """Scalar temperature scaling on the full training split (Guo et al., 2017).

    Grid ties break toward the smallest candidate.
    """
    best_nll, best_T = np.inf, None
    for T in CAL_GRID:
        p = softmax(xbar_train / T_AI / T)
        nll = -np.mean(np.log(np.clip(p[np.arange(len(y_train)), y_train], 1e-300, None)))
        if nll < best_nll - 1e-15:
            best_nll, best_T = nll, float(T)
    return best_T


def calibrated_confidence(xbar, T_cal):
    return softmax(xbar / T_AI / T_cal).max(axis=1)


def raw_confidence(xbar):
    return softmax(xbar / T_AI).max(axis=1)
