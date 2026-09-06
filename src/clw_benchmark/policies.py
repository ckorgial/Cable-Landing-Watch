"""Authority policies B0-B8.

Actions: 0 = accept, 1 = review, 2 = re-sense.

B4 note.  The v1.1 hand-coded router is retained verbatim for continuity, but
it is now reported with a warning that it is provably not a conservative
router: because its row 2 requires p_cal < 0.45, every record with
p_cal >= 0.60 is accepted at row 1 or row 3, hence

    A_B2 = {p_cal >= 0.60} subseteq A_B4,

so HAR_B4 >= HAR_B2 and Cov_B4 >= Cov_B2 identically.  Any release that
reports B4 below B2 on either quantity contains an implementation or reporting
error.  `assert_b4_contains_b2` in tests/ enforces this.
"""
from __future__ import annotations

import numpy as np

from .assurance import CODE, CODE_PRIORITY, SCALAR, STRUCTURAL

ACCEPT, REVIEW, RESENSE = 0, 1, 2

# policy coordinate sets: (gate set, core set for the severity row, re-sense trigger)
COMPACT = ("S", "gf", "C", "J")   # leave-one-seed-out selected, unanimous over 20 seeds

POLICY_SETS = {
    "B6": (STRUCTURAL, ("S", "F", "Dv"), "S"),
    "B6c": (COMPACT, ("S", "gf", "C"), "S"),
    "B7": (SCALAR, SCALAR, "Pc"),
    "B8": (STRUCTURAL + SCALAR, ("S", "F", "Dv", "Pc", "V"), "S"),
    "B8c": (COMPACT + SCALAR, ("S", "gf", "C", "Pc", "V"), "S"),
}

RESENSE_DELTA = 0.45
RESENSE_FEAS = 0.60
SEVERITY = 0.75
CORE_MARGIN = 0.60


def cascade(gates, keys, core, trigger, delta_bar, mission, resensing):
    """Priority-ordered authority cascade.  First matching row wins.

    1  z_trigger >= tau_trigger and delta_bar >= 0.45 and r >= 0.60   -> re-sense
    2  any gate in `keys` fires                                        -> review
    3  m >= 0.75 and max core margin >= 0.60                           -> review
    4  otherwise                                                       -> accept
    """
    z, tau, u = gates["z"], gates["tau"], gates["u"]
    n = len(delta_bar)
    action = np.full(n, ACCEPT, dtype=np.int64)
    fired = np.stack([z[k] >= tau[k] for k in keys], axis=1)
    any_fired = fired.any(axis=1)

    row1 = (z[trigger] >= tau[trigger]) & (delta_bar >= RESENSE_DELTA) & (resensing >= RESENSE_FEAS)
    row2 = (~row1) & any_fired
    core_margin = np.maximum.reduce([u[k] for k in core])
    row3 = (~row1) & (~row2) & (mission >= SEVERITY) & (core_margin >= CORE_MARGIN)

    action[row1] = RESENSE
    action[row2] = REVIEW
    action[row3] = REVIEW
    return action, fired


def primary_codes(action, fired, keys, mission):
    """Reason code of the selected branch; contextual QR1 is logged separately."""
    n = len(action)
    out = np.empty(n, dtype=object)
    order = [k for k in CODE_PRIORITY if k in keys]
    gate_code = np.empty(n, dtype=object)
    gate_code[:] = ""
    for k in reversed(order):
        col = keys.index(k)
        gate_code[fired[:, col]] = CODE[k]
    out[action == RESENSE] = "QS3"
    out[action == REVIEW] = gate_code[action == REVIEW]
    sev = (action == REVIEW) & (out == "")
    out[sev] = "QR1"
    out[action == ACCEPT] = "QF1"
    return out


def fixed_policies(p_raw, p_cal, delta, mission, resensing, dispersion):
    """The five threshold policies B0-B4 that carry no fitted quantity."""
    n = len(p_cal)
    delta_bar = delta.mean(axis=1)
    out = {}
    out["B0"] = np.full(n, ACCEPT, dtype=np.int64)
    out["B1"] = np.where(p_raw >= 0.60, ACCEPT, REVIEW)
    out["B2"] = np.where(p_cal >= 0.60, ACCEPT, REVIEW)
    out["B3"] = np.where(delta.max(axis=1) > 0.60, REVIEW, ACCEPT)

    a4 = np.full(n, REVIEW, dtype=np.int64)
    r1 = (p_cal >= 0.65) & (delta_bar < 0.30) & (dispersion < 0.45) & (mission < 0.70)
    r2 = (~r1) & (delta_bar > 0.55) & (resensing > 0.65) & (p_cal < 0.45)
    r3 = (~r1) & (~r2) & (p_cal >= 0.50)
    a4[r1] = ACCEPT
    a4[r2] = RESENSE
    a4[r3] = ACCEPT
    out["B4"] = a4
    return out


def gated_policies(gates, delta_bar, mission, resensing):
    """B5 (structural union only) and the three cascade policies B6, B7, B8."""
    z, tau = gates["z"], gates["tau"]
    n = len(delta_bar)
    out, codes = {}, {}
    union = np.stack([z[k] >= tau[k] for k in STRUCTURAL], axis=1).any(axis=1)
    out["B5"] = np.where(union, REVIEW, ACCEPT)
    for name, (keys, core, trig) in POLICY_SETS.items():
        act, fired = cascade(gates, list(keys), list(core), trig,
                             delta_bar, mission, resensing)
        out[name] = act
        codes[name] = primary_codes(act, fired, list(keys), mission)
    return out, codes


POLICY_ORDER = ("B0", "B1", "B2", "B3", "B4", "B5", "B7", "B6", "B6c", "B8", "B8c")

POLICY_LABEL = {
    "B0": "B0 always accept",
    "B1": "B1 raw confidence",
    "B2": "B2 temp.-scaled confidence",
    "B3": "B3 degradation warning",
    "B4": "B4 hand-coded classical router",
    "B5": "B5 structural gates only",
    "B7": "B7 scalar-observation cascade",
    "B6": "B6 POSEIDON-QIT",
    "B6c": "B6c POSEIDON-QIT (compact)",
    "B8": "B8 union cascade (full)",
    "B8c": "B8c union cascade (compact)",
}
