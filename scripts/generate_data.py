#!/usr/bin/env python3
"""Deterministically regenerate every CLW v1.2 seed file into data/seeds/.

Usage
-----
    python scripts/generate_data.py              # all 20 predeclared seeds
    python scripts/generate_data.py --seeds 20260516 7
    python scripts/generate_data.py --dtype float32   # smaller files
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from clw_benchmark import generator as G   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="*", default=list(G.SEEDS))
    ap.add_argument("--dtype", default="float64", choices=["float64", "float32"])
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "seeds"))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dt = np.dtype(args.dtype)
    for s in args.seeds:
        d = G.generate_seed(s)
        path = os.path.join(args.out, f"clw_seed_{s}.npz")
        np.savez_compressed(
            path,
            evidence=d["evidence"].astype(dt),
            truth=d["truth"].astype(np.int8),
            prediction=d["prediction"].astype(np.int8),
            probabilities=d["probabilities"].astype(dt),
            degradation=d["degradation"].astype(dt),
            mission=d["mission"].astype(dt),
            resensing=d["resensing"].astype(dt),
            family=d["family"].astype(np.int8),
            split=d["split"].astype(np.int8),
            seed=np.int64(s),
            version=np.str_(G.VERSION),
        )
        print(f"wrote {path}  ({os.path.getsize(path)/1e6:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
