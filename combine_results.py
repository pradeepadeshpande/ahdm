"""
combine_results.py
==================
Merge the outputs of several run_experiments.py runs (e.g. one run for
synthetic/openstack/bgl and one for thunderbird) into a single results folder,
so that tables and figures cover all datasets together.

Nothing is re-run and no value is typed in: the per-run rows in metrics_long.csv
are concatenated, and summary.csv / significance.csv are recomputed from them
with exactly the same formulas as run_experiments.py.

    python combine_results.py --inputs results results_thunderbird --out results_combined
    python make_figures.py --results results_combined --eps 1.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

METHODS = ["IF", "LOF", "OCSVM", "DAE-only", "AHDM-fixed", "AHDM"]
METRICS = ["accuracy", "precision", "recall", "f1", "auc_roc", "avg_prec"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True, help="results folders to merge")
    ap.add_argument("--out", default="results_combined")
    a = ap.parse_args()

    frames, metas, scores = [], {}, {}
    seen = {}
    for folder in a.inputs:
        p = Path(folder)
        df = pd.read_csv(p / "metrics_long.csv")
        for d in df["dataset"].unique():
            if d in seen:
                raise SystemExit(f"Dataset '{d}' appears in both '{seen[d]}' and '{folder}'. "
                                 "Each dataset must come from exactly one run.")
            seen[d] = folder
        frames.append(df)
        metas[folder] = json.loads((p / "run_metadata.json").read_text(encoding="utf-8"))
        with np.load(p / "test_scores_seed0.npz") as z:
            scores.update({k: z[k] for k in z.files})

    df = pd.concat(frames, ignore_index=True)

    # ── consistency checks (warn, do not hide) ────────────────────────────────
    for folder, m in metas.items():
        if m.get("dae_backend") != "torch":
            print(f"WARNING: '{folder}' was run with DAE backend '{m.get('dae_backend')}', not torch.")
    for key in ("seeds", "epsilons"):
        vals = {f: json.dumps(m[key]) for f, m in metas.items()}
        if len(set(vals.values())) > 1:
            print(f"WARNING: runs differ in '{key}': {vals}")
    for key in ("python", "sklearn", "numpy", "pandas"):
        vals = {f: m.get(key) for f, m in metas.items()}
        if len(set(vals.values())) > 1:
            print(f"NOTE: runs used different '{key}' versions: {vals}")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "metrics_long.csv", index=False)

    # ── summary: mean +/- std over seeds (same as run_experiments.py) ─────────
    g = df.groupby(["dataset", "epsilon", "method"])[METRICS]
    summ = g.mean().join(g.std(), rsuffix="_std").reset_index()
    summ.to_csv(out / "summary.csv", index=False)

    # ── paired Wilcoxon: AHDM vs every other method (same as run_experiments.py)
    stats = []
    for (d, e), sub in df.groupby(["dataset", "epsilon"]):
        piv = {m: sub[sub.method == m].sort_values("seed") for m in METHODS}
        for other in METHODS:
            if other == "AHDM":
                continue
            for met in ("auc_roc", "f1"):
                x, y = piv["AHDM"][met].to_numpy(), piv[other][met].to_numpy()
                diff = x - y
                try:
                    p = 1.0 if np.allclose(diff, 0) else float(wilcoxon(x, y).pvalue)
                except ValueError:
                    p = float("nan")
                stats.append({"dataset": d, "epsilon": e, "vs": other, "metric": met,
                              "mean_diff": float(diff.mean()), "p_wilcoxon": p, "n_seeds": len(diff)})
    pd.DataFrame(stats).to_csv(out / "significance.csv", index=False)

    np.savez_compressed(out / "test_scores_seed0.npz", **scores)

    first = next(iter(metas.values()))
    meta = {k: first.get(k) for k in ("python", "sklearn", "numpy", "pandas", "torch_available",
                                      "dae_backend", "seeds", "epsilons")}
    meta["datasets"] = {d: m["datasets"][d] for m in metas.values() for d in m["datasets"]}
    meta["combined_from"] = {f: {"datasets": list(m["datasets"]), "runtime_s": m.get("runtime_s"),
                                 **{k: m.get(k) for k in ("python", "sklearn", "numpy", "pandas",
                                                          "torch_available", "dae_backend")}}
                             for f, m in metas.items()}
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Combined {len(seen)} datasets {sorted(seen)} -> {out}/")


if __name__ == "__main__":
    main()
