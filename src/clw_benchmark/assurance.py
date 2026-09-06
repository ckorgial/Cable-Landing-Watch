"""POSEIDON-QIT density-state assurance layer.

Fully classical.  Every density matrix is a small real symmetric array that is
stored and diagonalised with LAPACK on conventional hardware.  "Quantum
information theory" names the matrix calculus, not the substrate.
"""
from __future__ import annotations

import numpy as np
from numpy.linalg import eigh

from .generator import D, M, softmax

T0 = 0.25          # encoder temperature
EPS = 1e-4         # state regulariser
BETA = 0.70        # engineered bipartite mixing
ALPHA_CLIP = (0.02, 0.95)
ALPHA_OFFSET = 0.5
EIG_FLOOR = 1e-15
SCALE_FLOOR = 1e-12
GATE_QUANTILE = 0.98

# structural (density-state + centred alignment) coordinates
STRUCTURAL = ("S", "F", "Dv", "J", "gm", "gf", "C")
# scalar coordinates available to a router restricted to the compressed observation
SCALAR = ("Pc", "De", "V")

# orientation: '+' upper tailed, '-' lower tailed, '2' two sided
ORIENTATION = {
    "S": "+",    # fused spectral dispersion
    "F": "-",    # loss of overlap with the nominal envelope
    "Dv": "+",   # directional departure from the envelope
    "J": "2",    # engineered bipartite deviation
    "gm": "-",   # least concentrated modality
    "gf": "2",   # fused concentration shift
    "C": "+",    # excessive centred alignment
    "Pc": "-",   # low calibrated confidence
    "De": "+",   # high aggregate degradation
    "V": "-",    # LOW evidence dispersion == excessive cross-sensor agreement
}

CODE = {"S": "QS2", "F": "QF2", "Dv": "QD1", "J": "QE1", "gm": "QP1",
        "gf": "QP2", "C": "QC1", "Pc": "QL1", "De": "QG1", "V": "QV1"}

# structural-code priority, highest first
CODE_PRIORITY = ("gf", "J", "S", "gm", "Dv", "F", "C", "V", "Pc", "De")


# ---------------------------------------------------------------------------
# encoder
# ---------------------------------------------------------------------------
def encode(evidence):
    """Evidence -> modality states rho_t^i and convexly fused state rho_t."""
    q = softmax(evidence / T0, axis=-1)
    v = np.sqrt(q)
    theta = evidence.std(axis=-1)                      # population sd, 1/d
    alpha = np.clip(theta / (ALPHA_OFFSET + theta), *ALPHA_CLIP)
    outer = v[..., :, None] * v[..., None, :]
    mixture = ((1.0 - alpha)[..., None, None] * outer
               + alpha[..., None, None] * np.eye(D) / D)
    rho_i = (mixture + EPS * np.eye(D)) / (1.0 + EPS * D)
    return rho_i, rho_i.mean(axis=1)


def nominal_envelope(rho, mask):
    env = rho[mask].mean(axis=0)
    return env / np.trace(env)


# ---------------------------------------------------------------------------
# density-state functionals
# ---------------------------------------------------------------------------
def von_neumann_entropy(R):
    w = np.clip(eigh(R)[0], EIG_FLOOR, None)
    return -(w * np.log(w)).sum(axis=-1)


def purity(R):
    return np.einsum("...ij,...ji->...", R, R)


def _spectral_map(R, f):
    w, U = eigh(R)
    w = np.clip(w, EIG_FLOOR, None)
    return (U * f(w)[..., None, :]) @ np.swapaxes(U, -1, -2)


def root_fidelity(R, xi):
    xs = _spectral_map(xi, np.sqrt)
    w = eigh(xs @ R @ xs)[0]
    return np.sqrt(np.clip(w, 0.0, None)).sum(axis=-1)


def relative_entropy(R, xi):
    return np.einsum("...ij,...ji->...", R,
                     _spectral_map(R, np.log) - _spectral_map(xi, np.log))


def _sigma_diag():
    d2 = D * D
    s = np.zeros((d2, d2))
    for k in range(D):
        e = np.zeros(d2)
        e[k * D + k] = 1.0
        s += np.outer(e, e)
    return s / D


def bipartite_score(rho_i):
    """J_t = S(Gamma_A) + S(Gamma_B) - S(Gamma), Gamma = beta rho1 (x) rho2 + (1-beta) sigma_diag.

    Marginals reduce in closed form to Gamma_A = beta rho1 + (1-beta) I/d and
    likewise for B, so the construction does not preserve the sensor states as
    marginals unless beta = 1.  Nonnegativity is subadditivity of S.
    """
    d2 = D * D
    r1, r2 = rho_i[:, 0], rho_i[:, 1]
    kron = np.einsum("nij,nkl->nikjl", r1, r2).reshape(-1, d2, d2)
    gamma = BETA * kron + (1.0 - BETA) * _sigma_diag()
    ga = BETA * r1 + (1.0 - BETA) * np.eye(D) / D
    gb = BETA * r2 + (1.0 - BETA) * np.eye(D) / D
    return von_neumann_entropy(ga) + von_neumann_entropy(gb) - von_neumann_entropy(gamma)


def centred_alignment(evidence):
    xt = evidence - evidence.mean(axis=-1, keepdims=True)
    nrm = np.linalg.norm(xt, axis=-1)
    acc = 0.0
    for i in range(M):
        for j in range(i + 1, M):
            acc = acc + (np.einsum("nk,nk->n", xt[:, i], xt[:, j])
                         / np.maximum(nrm[:, i] * nrm[:, j], EIG_FLOOR))
    return acc * 2.0 / (M * (M - 1))


def evidence_dispersion(evidence):
    xbar = evidence.mean(axis=1)
    return ((evidence - xbar[:, None, :]) ** 2).sum(axis=(1, 2)) / (M * D)


# ---------------------------------------------------------------------------
# coordinates, standardisation, gates
# ---------------------------------------------------------------------------
def raw_coordinates_subset(evidence, rho_i, rho, envelope, p_cal, delta_bar, keys):
    """Evaluate only requested coordinates; useful for compact-policy attacks/replay."""
    keys = set(keys)
    out = {}
    if "S" in keys: out["S"] = von_neumann_entropy(rho)
    if "F" in keys: out["F"] = root_fidelity(rho, envelope)
    if "Dv" in keys: out["Dv"] = relative_entropy(rho, envelope)
    if "J" in keys: out["J"] = bipartite_score(rho_i)
    if "gm" in keys: out["gm"] = purity(rho_i).min(axis=1)
    if "gf" in keys: out["gf"] = purity(rho)
    if "C" in keys: out["C"] = centred_alignment(evidence)
    if "Pc" in keys: out["Pc"] = p_cal
    if "De" in keys: out["De"] = delta_bar
    if "V" in keys: out["V"] = evidence_dispersion(evidence)
    return out


def raw_coordinates(evidence, rho_i, rho, envelope, p_cal, delta_bar):
    return raw_coordinates_subset(evidence, rho_i, rho, envelope, p_cal, delta_bar,
                                  STRUCTURAL + SCALAR)


def fit_gates(raw, nominal_train_mask, quantile=GATE_QUANTILE):
    """Per-seed standardisation and empirical gate thresholds on nominal training only."""
    mu, sd, z, tau = {}, {}, {}, {}
    for k, val in raw.items():
        mu[k] = float(val[nominal_train_mask].mean())
        sd[k] = float(max(val[nominal_train_mask].std(), SCALE_FLOOR))
        o = ORIENTATION[k]
        if o == "+":
            z[k] = (val - mu[k]) / sd[k]
        elif o == "-":
            z[k] = (mu[k] - val) / sd[k]
        else:
            z[k] = np.abs(val - mu[k]) / sd[k]
        tau[k] = float(max(np.quantile(z[k][nominal_train_mask], quantile), SCALE_FLOOR))
    margin = {k: z[k] / tau[k] for k in raw}
    return dict(mu=mu, sd=sd, z=z, tau=tau, u=margin)
