#!/usr/bin/env python3
"""Release verifier for CLW v1.2 / POSEIDON-QIT.

Every check below corresponds to a normative statement in
NOEMAC-CLW-TR-2026 v1.2 Sec. XI-D.  Check 9 is the assertion whose violation
invalidated the v1.1 result manifest and is therefore mandatory.

Exit status is non-zero if any check fails.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from clw_benchmark import assurance as A       # noqa: E402
from clw_benchmark import generator as G       # noqa: E402
from clw_benchmark import policies as P        # noqa: E402
from clw_benchmark import publication as PB    # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main(n_seeds=3):
    seeds = list(G.SEEDS)[:n_seeds]
    print(f"validating CLW v{G.VERSION} on {len(seeds)} seed(s)\n")

    # 1 seed list
    check("1  exactly 20 declared seeds", len(G.SEEDS) == 20, f"{len(G.SEEDS)}")

    # 2 auxiliary-channel matching (v1.2 requirement)
    ok = (np.allclose(G.DELTA_RANGE[6], G.DELTA_RANGE[0])
          and np.allclose(G.DELTA_RANGE[7], G.DELTA_RANGE[0])
          and G.MBAR[6] == G.MBAR[0] and G.MBAR[7] == G.MBAR[0])
    check("2  families 6,7 auxiliary channels matched to nominal", ok)

    for s in seeds:
        d = PB._prepare(s)
        n = len(d["truth"])
        fam = d["family"]

        check(f"3  seed {s}: 8000 records, 1000 per family",
              n == 8000 and all((fam == c).sum() == 1000 for c in range(8)))
        check(f"4  seed {s}: split 2400/1200/4400",
              [(d['split'] == k).sum() for k in (0, 1, 2)] == [2400, 1200, 4400])
        check(f"5  seed {s}: evidence shape and finiteness",
              d["evidence"].shape == (8000, 3, 5) and np.isfinite(d["evidence"]).all())
        check(f"6  seed {s}: channel ranges",
              bool(((d["degradation"] >= 0) & (d["degradation"] <= 1)).all()
                   and ((d["mission"] >= 0) & (d["mission"] <= 1)).all()
                   and ((d["resensing"] >= 0.40) & (d["resensing"] <= 0.95)).all()))
        check(f"7  seed {s}: probability rows sum to one",
              bool(np.allclose(d["probabilities"].sum(axis=1), 1.0)))

        rho_i, rho = d["rho_i"], d["rho"]
        herm = np.allclose(rho, np.swapaxes(rho, -1, -2), atol=1e-12)
        trace = np.allclose(np.trace(rho, axis1=-2, axis2=-1), 1.0, atol=1e-10)
        posdef = bool((np.linalg.eigvalsh(rho) > 0).all())
        check(f"8  seed {s}: fused states Hermitian, unit trace, positive definite",
              herm and trace and posdef)

        # --- the check that v1.1 failed ---
        disp = d["raw"]["V"]
        acts = P.fixed_policies(d["p_raw"], d["p_cal"], d["degradation"],
                                d["mission"], d["resensing"], disp)
        a2, a4 = acts["B2"], acts["B4"]
        inclusion = bool(np.all(a4[a2 == P.ACCEPT] == P.ACCEPT))
        check(f"9  seed {s}: A_B2 subseteq A_B4 (v1.1 manifest violated this)",
              inclusion)

        gates = A.fit_gates(d["raw"], d["nom_train"])
        gated, _ = P.gated_policies(gates, d["delta_bar"], d["mission"], d["resensing"])
        for name, act in gated.items():
            fr = [float(np.mean(act[(d["split"] == 2) & (fam != 0)] == k)) for k in (0, 1, 2)]
            check(f"10 seed {s}: {name} workload fractions sum to one",
                  abs(sum(fr) - 1.0) < 1e-12)

        m = PB._metrics(gated["B6c"], d)
        check(f"11 seed {s}: HAR = Cov * SelRisk for B6c",
              abs(m["HAR"] - m["Cov"] * m["SelRisk"]) < 1e-12)

        # monotone consistency of the confidence policies
        if d["T_cal"] >= 1.0:
            m1 = PB._metrics(acts["B1"], d)
            m2 = PB._metrics(acts["B2"], d)
            check(f"12 seed {s}: T_cal>=1 implies Cov_B2<=Cov_B1 and HAR_B2<=HAR_B1",
                  m2["Cov"] <= m1["Cov"] + 1e-12 and m2["HAR"] <= m1["HAR"] + 1e-12,
                  f"T_cal={d['T_cal']}")

    # 13 manifest agreement
    path = os.path.join(ROOT, "results", "paper_results.json")
    if os.path.exists(path):
        agg = json.load(open(path))
        check("13 manifest version is 1.2", agg.get("version") == "1.2")
        check("13 manifest covers 20 seeds", agg.get("n_seeds") == 20)
        b4, b2 = agg["primary"]["B4"], agg["primary"]["B2"]
        check("13 manifest respects the B4 inclusion",
              b4["HAR"] >= b2["HAR"] and b4["Cov"] >= b2["Cov"],
              f"B4 HAR {b4['HAR']:.3f} >= B2 HAR {b2['HAR']:.3f}")
        o = agg["observation"]
        check("13 observation-envelope rates are non-degenerate for family 7",
              o["f7"] > 0.05, f"Omega_7 = {o['f7']:.3f}")

        # 14 compact-selection and union naming/reporting consistency
        loso = agg.get("loso_selection", {})
        selected = tuple(sorted(loso.get("selected_set") or ()))
        check("14 LOSO selection is unanimous and equals the shipped B6c set",
              bool(loso.get("unanimous")) and selected == tuple(sorted(P.COMPACT)),
              f"selected={selected}, shipped={tuple(sorted(P.COMPACT))}")
        check("14 B8 and B8c are both present and distinct matched policies",
              "B8" in agg.get("matched", {}) and "B8c" in agg.get("matched", {})
              and P.POLICY_SETS["B8"] != P.POLICY_SETS["B8c"])

        import csv
        csv_path = os.path.join(ROOT, "results", "matched_workload.csv")
        if os.path.exists(csv_path):
            with open(csv_path, newline="") as fh:
                rows = list(csv.DictReader(fh))
            names = [r.get("policy") for r in rows]
            check("14 matched_workload.csv includes both B8 and B8c",
                  "B8" in names and "B8c" in names, f"policies={names}")
        else:
            check("14 matched_workload.csv present", False)
    else:
        check("13 result manifest present", False, "run scripts/build_publication.py")

    print()
    if FAILURES:
        print(f"{len(FAILURES)} CHECK(S) FAILED:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3,
                    help="number of predeclared seeds to verify in depth")
    sys.exit(main(ap.parse_args().seeds))
