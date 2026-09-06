# Cable-Landing Watch (CLW) / POSEIDON-QIT reproducibility repository

This repository contains the **Cable-Landing Watch (CLW) v1.2** synthetic decision-level benchmark and the code used to reproduce the numerical experiments reported with **POSEIDON-QIT: Quantum-Information-Theoretic Density-State Assurance for Multi-Sensor Authority Routing**.

POSEIDON-QIT uses quantum-information-theoretic (QIT) density-state functionals only as **classical matrix and information-theoretic features**. No quantum hardware, quantum computation, or quantum speedup is assumed or claimed.

Repository: https://github.com/ckorgial/Cable-Landing-Watch

## What CLW is

CLW is a synthetic **decision-level multi-sensor benchmark**, not a raw sonar, radar, video, electro-optical, or AIS corpus. Each seed contains 8,000 records from eight authored mechanism families, with three modalities and five class scores per modality. The benchmark is designed to study post-prediction authority routing under correlated multi-sensor evidence.

Version 1.2 corrects the auxiliary-channel mismatch present in v1.1: the coordinated-spoof and correlated-residual families now use the nominal degradation interval and nominal mission-severity base. This prevents those common-mode families from being separated trivially through an auxiliary coordinate already available to the router.

## Main matched-workload results

At the predeclared nominal-intervention target of 0.10, thresholds are fitted **per seed using nominal calibration records only** and then evaluated on held-out evaluation records. The 20-seed means are:

| Policy | HAR ↓ | Coverage | Selective risk ↓ | Nominal intervention | Coordinated-spoof rejection ↑ | Correlated-residual rejection ↑ |
|---|---:|---:|---:|---:|---:|---:|
| B2 calibrated confidence | 0.150 | 0.375 | 0.399 | 0.092 | 0.451 | 0.386 |
| B7 scalar cascade | 0.077 | 0.087 | 0.886 | 0.107 | 0.992 | 0.403 |
| B6 full structural | 0.079 | 0.144 | 0.550 | 0.105 | 0.915 | 0.593 |
| **B6c compact structural** | **0.066** | **0.168** | **0.392** | **0.108** | **0.954** | **0.643** |
| B8 full union | 0.041 | 0.047 | 0.885 | 0.101 | 0.986 | 0.690 |
| B8c compact union | 0.036 | 0.042 | 0.849 | 0.105 | 0.994 | 0.726 |

The paper emphasizes **B8c**, not B8, when discussing the compact structural–scalar union. The two configurations are distinct:

- **B8** = full structural set + scalar set.
- **B8c** = compact B6c structural set + scalar set.

The compact structural set used by B6c is `{S, gf, C, J}`. It is supported by leave-one-seed-out selection on calibration-risk records: each held-out seed is excluded from the selection statistic, no evaluation record is used for selection, and the same set is returned on all 20 folds. The principal selector-independent comparisons use B6, which contains the full predeclared structural set.

## Supervised probe and adversarial evaluation

The gradient-boosted representation experiment is a **supervised probe**, not a certified upper bound on all possible routers. At nominal intervention 0.10, the correlated-residual rejection of the probe is 0.573 for the compressed observation and 0.863 for the structural representation under the same model class.

The gate-aware adversarial experiments are reported separately. Under budgeted white-box evasion at `epsilon = 0.10`, the acceptance/evasion rates are 0.802 for B7, 0.724 for B6c, and 0.596 for B8c. The structural-versus-scalar ordering is not monotone in the perturbation budget; at `epsilon = 0.20`, B6c is more easily evaded than B7. These results are conditional on the declared synthetic mechanisms and do not provide a family-wise, finite-sample, or deployment guarantee.

## Repository layout

```text
.
├── config/
│   └── clw_v1_2.json              frozen benchmark and policy configuration
├── data/seeds/
│   └── clw_seed_20260516.npz      deterministic float32 reference seed
├── results/
│   ├── paper_results.json         authoritative 20-seed result manifest
│   ├── matched_workload.csv       matched-workload B2/B7/B6/B6c/B8/B8c results
│   ├── primary_results.csv        frozen operating-point results
│   ├── mechanism_roc.csv          mechanism rejection vs nominal intervention
│   ├── gate_ablation.csv          B6 single-coordinate ablations
│   ├── paired_comparisons.csv     prespecified paired tests
│   ├── representation_probe.csv   supervised representation-probe results
│   ├── adaptive_mechanism.csv     mechanism-redesign sweep
│   ├── adaptive_evasion.csv       budgeted white-box evasion results
│   └── adversarial_results.json   full adversarial/probe manifest
├── scripts/
│   ├── generate_data.py           regenerate CLW seed files
│   ├── build_publication.py       rerun numerical experiments and manifests
│   ├── make_figures.py            regenerate figures from result manifests
│   └── validate_release.py        release-integrity and invariant checks
├── src/clw_benchmark/
│   ├── generator.py               CLW data generator
│   ├── assurance.py               density-state encoder and coordinates
│   ├── policies.py                B0–B8c authority policies
│   ├── metrics.py                 metrics, bootstrap, Wilcoxon/Holm utilities
│   ├── adversarial.py             supervised probes and adversarial experiments
│   └── publication.py             end-to-end experiment driver
├── tests/test_clw.py              unit/property/reproducibility tests
├── requirements-lock.txt          tested Python dependency versions
├── CITATION.cff                   citation metadata
├── CHECKSUMS.sha256               release integrity manifest
└── .gitignore
```

## Environment

The release was tested with **Python 3.13.5** and the exact package versions in `requirements-lock.txt`.

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
export PYTHONPATH="$PWD/src"
```

### Windows PowerShell

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-lock.txt
$env:PYTHONPATH = "$PWD\src"
```

## Quick verification

Run the test suite and the release validator:

```bash
pytest -q
python scripts/validate_release.py --seeds 3
```

The shipped release currently passes all 26 tests. The validator checks benchmark dimensions, split sizes, channel ranges, probability normalization, density-state validity, the repaired B4/B2 inclusion invariant, cascade workload identities, B6c metric identities, manifest versioning, B8/B8c distinction, and consistency of the compact set with the leave-one-seed-out result.

To validate more declared seeds, increase `--seeds` up to 20. This performs the same deterministic checks on additional seeds.

## Reproduce the shipped reference seed exactly

The committed reference seed is stored as **float32**. For byte-for-byte reproduction, explicitly request the same dtype:

```bash
python scripts/generate_data.py --seeds 20260516 --dtype float32 --out /tmp/clw_seedcheck
sha256sum /tmp/clw_seedcheck/clw_seed_20260516.npz
```

Expected SHA-256:

```text
2327d9908ddd2fb4c8d054f42b057c3eb723f1936f141089d73b1ee91eeae4bc
```

Using the generator's default `float64` output produces numerically equivalent values at float32 precision, but not the same file checksum.

## Regenerate all declared seed files

```bash
python scripts/generate_data.py --dtype float32
```

Generated seed files are ignored by Git except for the supplied reference seed, so the repository does not need to store all 20 `.npz` files.

## Reproduce the publication result manifests

```bash
export PYTHONPATH="$PWD/src"   # PowerShell: $env:PYTHONPATH = "$PWD\src"
python scripts/build_publication.py
python scripts/make_figures.py
```

`build_publication.py` regenerates the 20-seed result manifest, matched-workload tables, mechanism frontiers, ablations, paired tests, supervised probes, and gate-aware adversarial results. The adversarial search is the most computationally demanding part of the reproduction workflow.

After regeneration, rerun:

```bash
pytest -q
python scripts/validate_release.py --seeds 3
```

## Result provenance and interpretation

`results/paper_results.json` is the authoritative non-adversarial numerical manifest. `results/adversarial_results.json` is the authoritative supervised-probe/adversarial manifest. CSV files are human-readable exports of those manifests and should not be edited manually.

Harmful acceptance (HAR), risk coverage, selective risk, and nominal intervention must be interpreted jointly. Lower HAR can be obtained simply by withholding authority from more records; therefore mechanism rejection or HAR alone is not evidence of universally superior routing. In particular, the union configurations B8 and B8c achieve strong mechanism rejection at low coverage and high selective risk. B6c is retained as the primary reduced structural configuration for the clean matched-workload analysis.

## Statistical protocol

The **seed** is the inferential unit. The release uses 20 predeclared seeds, 10,000 seed-level bootstrap resamples for percentile intervals, and three prespecified exact one-sided Wilcoxon signed-rank comparisons on correlated-residual harmful acceptance with Holm correction. The two principal comparisons involving B6 do not depend on coordinate selection.

## Integrity check

After cloning or downloading a tagged release, verify the tracked release files with:

```bash
sha256sum -c CHECKSUMS.sha256
```

If you intentionally edit any tracked file, regenerate `CHECKSUMS.sha256` before creating a new release tag.

## Scope and limitations

CLW is an authored synthetic decision-level benchmark. The reported seed-level inference measures consistency across the declared synthetic seeds. The repository does **not** establish transfer to real sensors, field safety, deployment readiness, a quantum computational advantage, or a finite-sample risk-control guarantee. Independent validation on real multi-sensor evidence is required before treating the compact coordinate set or any operating point as a general recommendation.
