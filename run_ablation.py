"""
run_ablation.py
===============
Ablation study of AHDM (computed from data; nothing is hardcoded).

Configurations (all use the same splits, seeds and validation-based threshold
procedure as run_experiments.py unless stated):

  A1  IF only
  A2  DAE only
  A3  Fixed-weight fusion (alpha = 0.5)
  A4  AHDM with a FIXED threshold T = 0.5 on the fused score (no validation tuning)
  A5a AHDM with UNIFORM epsilon: every attribute gets epsilon_base
      (per-record loss d * epsilon_base: NOT budget-matched)
  A5b AHDM with UNIFORM epsilon, budget-matched: every attribute gets
      eps_rec / d, where eps_rec is the per-record total of the adaptive scheme
  A6  Full AHDM (sensitivity-class budgets, learned alpha, validation threshold)

Usage
-----
    python run_ablation.py --data-dir data --datasets synthetic openstack bgl \
        --epsilons 0.5 1 2 5 10 inf --seeds 8 --out results_ablation
    python run_ablation.py --data-dir data --datasets thunderbird --max-rows 20000 \
        --epsilons 0.5 1 2 5 10 inf --seeds 8 --out results_ablation_thunderbird
"""
from __future__ import annotations

import argparse, json, platform, time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import wilcoxon
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from ahdm import AHDM
from autoencoder import TORCH_AVAILABLE, TORCH_IMPORT_ERROR
from dp_engine import DPEngine
from run_experiments import load_dataset, prepare, metrics, parse_eps, log

CONFIGS = ["A1", "A2", "A3", "A4", "A5a", "A5b", "A6"]
NAMES = {"A1": "IF only", "A2": "DAE only", "A3": "Fixed fusion (alpha=0.5)",
         "A4": "AHDM, fixed threshold T=0.5", "A5a": "AHDM, uniform eps (eps_base each)",
         "A5b": "AHDM, uniform eps (budget-matched)", "A6": "Full AHDM"}


def perturb(Xs, schema, cols, eps, seed, allocation):
    eng = DPEngine(base_epsilon=eps, random_state=seed * 1000 + int(eps * 100))
    eng.register_schema(schema)
    out = tuple(eng.perturb_matrix(a, cols, allocation=allocation) for a in Xs)
    return tuple(a.astype(np.float32) for a in out), eng.record_epsilon


def fit_ahdm(Xtr, Xva, Xte, yva, seed):
    m = AHDM(input_dim=Xtr.shape[1], random_state=seed)
    m.fit(Xtr, X_val=Xva, y_val=yva, verbose=0)
    nv = m._score_scaler.transform(np.column_stack([m.if_scores(Xva), m.ae_scores(Xva)]))
    nt = m._score_scaler.transform(np.column_stack([m.if_scores(Xte), m.ae_scores(Xte)]))
    return m, nv, nt


def run_cell(Xs, ys, schema, cols, eps, seed):
    ytr, yva, yte = ys
    rows = []
    infin = not np.isfinite(eps)

    # ---- adaptive allocation (A1-A4, A6) -------------------------------------
    if infin:
        (Xtr, Xva, Xte), rec = tuple(a.astype(np.float32) for a in Xs), float("inf")
    else:
        (Xtr, Xva, Xte), rec = perturb(Xs, schema, cols, eps, seed, "adaptive")
    m, nv, nt = fit_ahdm(Xtr, Xva, Xte, yva, seed)
    iso = IsolationForest(n_estimators=100, random_state=seed).fit(Xtr)
    sc = {"A1": (-iso.score_samples(Xva), -iso.score_samples(Xte)),
          "A2": (nv[:, 1], nt[:, 1]),
          "A3": (0.5 * nv.sum(1), 0.5 * nt.sum(1)),
          "A6": (m.decision_scores(Xva), m.decision_scores(Xte))}
    for k, (sva, ste) in sc.items():
        rows.append({"config": k, "alpha": m.alpha_ if k == "A6" else np.nan,
                     "record_epsilon": rec, **metrics(yva, yte, sva, ste)})
    # A4: same scores as A6, fixed threshold
    ste = sc["A6"][1]
    pred = (ste >= 0.5).astype(int)
    rows.append({"config": "A4", "alpha": m.alpha_, "record_epsilon": rec,
                 "accuracy": accuracy_score(yte, pred),
                 "precision": precision_score(yte, pred, zero_division=0),
                 "recall": recall_score(yte, pred, zero_division=0),
                 "f1": f1_score(yte, pred, zero_division=0),
                 "auc_roc": roc_auc_score(yte, ste), "avg_prec": average_precision_score(yte, ste)})

    # ---- uniform allocation (A5a, A5b): only meaningful with DP ---------------
    if not infin:
        for k, alloc in (("A5a", "uniform"), ("A5b", "matched")):
            (Xtr, Xva, Xte), rec = perturb(Xs, schema, cols, eps, seed, alloc)
            m, nv, nt = fit_ahdm(Xtr, Xva, Xte, yva, seed)
            rows.append({"config": k, "alpha": m.alpha_, "record_epsilon": rec,
                         **metrics(yva, yte, m.decision_scores(Xva), m.decision_scores(Xte))})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--datasets", nargs="+", default=["synthetic", "openstack", "bgl"])
    ap.add_argument("--epsilons", nargs="+", type=parse_eps, default=[0.5, 1, 2, 5, 10, float("inf")])
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--out", default="results_ablation")
    a = ap.parse_args()
    if not TORCH_AVAILABLE:
        raise SystemExit(f"PyTorch could not be imported: {TORCH_IMPORT_ERROR}")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows, info, t0 = [], {}, time.time()
    for name in a.datasets:
        ds = load_dataset(name, Path(a.data_dir))
        n_full = len(ds.y)
        if a.max_rows and n_full > a.max_rows:
            keep, _ = train_test_split(np.arange(n_full), train_size=a.max_rows,
                                       stratify=ds.y, random_state=12345)
            keep = np.sort(keep); ds.X = ds.X.iloc[keep].reset_index(drop=True); ds.y = ds.y[keep]
        info[name] = {"summary": ds.summary(), "full_rows": n_full}
        log.info(ds.summary())
        for seed in range(a.seeds):
            Xs, ys, schema, cols = prepare(ds, seed)
            for eps in a.epsilons:
                for r in run_cell(Xs, ys, schema, cols, eps, seed):
                    rows.append({"dataset": name, "epsilon": eps, "seed": seed, **r})
            log.info("%s seed %d done (%.0fs)", name, seed, time.time() - t0)
    df = pd.DataFrame(rows); df.to_csv(out / "ablation_long.csv", index=False)
    M = ["accuracy", "precision", "recall", "f1", "auc_roc", "avg_prec"]
    g = df.groupby(["dataset", "epsilon", "config"])[M]
    g.mean().join(g.std(), rsuffix="_std").reset_index().to_csv(out / "ablation_summary.csv", index=False)
    stats = []
    for (d, e), sub in df.groupby(["dataset", "epsilon"]):
        piv = {c: sub[sub.config == c].sort_values("seed") for c in CONFIGS}
        for other in ("A1", "A2", "A3", "A4", "A5a", "A5b"):
            if piv[other].empty: continue
            for met in ("auc_roc", "f1"):
                x, y = piv["A6"][met].to_numpy(), piv[other][met].to_numpy(); diff = x - y
                try: p = 1.0 if np.allclose(diff, 0) else float(wilcoxon(x, y).pvalue)
                except ValueError: p = float("nan")
                stats.append({"dataset": d, "epsilon": e, "A6_vs": other, "metric": met,
                              "mean_diff": float(diff.mean()), "p_wilcoxon": p, "n_seeds": len(diff)})
    pd.DataFrame(stats).to_csv(out / "ablation_significance.csv", index=False)
    import torch
    (out / "ablation_metadata.json").write_text(json.dumps({
        "python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__,
        "pandas": pd.__version__, "torch": torch.__version__, "seeds": a.seeds,
        "epsilons": [str(e) for e in a.epsilons], "datasets": info,
        "runtime_s": round(time.time() - t0, 1), "configs": NAMES}, indent=2))
    log.info("Finished in %.0fs -> %s/", time.time() - t0, out)


if __name__ == "__main__":
    main()
