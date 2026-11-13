"""
run_experiments.py
==================
Computes every number reported for AHDM from the data.  Nothing is hardcoded.

Protocol (identical for all detectors)
--------------------------------------
  * Repeated stratified split per seed:  60 % train / 20 % validation / 20 % test.
  * MinMax scaling fitted on the training split only; val/test clipped to [0, 1].
  * Local Laplace DP applied to every split (see DPEngine.perturb_matrix).
  * Detectors are fitted on the (DP-perturbed) training split WITHOUT labels:
        Isolation Forest, LOF, One-Class SVM, DAE-only, AHDM-fixed (alpha=0.5), AHDM.
    AHDM additionally fits its meta-learner on validation labels.
  * Operating threshold: for EVERY detector, chosen on the validation split to
    maximise F1 (same label access for all).  Test split is never used for tuning.
  * Threshold-free metrics (AUC-ROC, Average Precision) are reported as primary.

Usage
-----
    python run_experiments.py --data-dir data --datasets synthetic openstack bgl \
        --epsilons 0.5 1 2 5 10 inf --seeds 5 --out results
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import platform
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import sklearn
from scipy.stats import wilcoxon
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM

import loaders
from ahdm import AHDM
from autoencoder import TORCH_AVAILABLE, TORCH_IMPORT_ERROR
from dp_engine import DPEngine

logging.basicConfig(level=logging.WARNING)
logging.getLogger("dp_engine").setLevel(logging.WARNING)
log = logging.getLogger("run")
log.setLevel(logging.INFO)
if not log.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
    log.addHandler(h)
    log.propagate = False

METHODS = ["IF", "LOF", "OCSVM", "DAE-only", "AHDM-fixed", "AHDM"]


# ─────────────────────────────────────────────────────────────────────────────
#  Data
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(name: str, data_dir: Path) -> loaders.LogDataset:
    if name == "synthetic":
        return loaders.generate_synthetic()
    if name == "openstack":
        normal = sorted(glob.glob(str(data_dir / "openstack_normal*.log")))
        abnormal = sorted(glob.glob(str(data_dir / "openstack_abnormal*.log")))
        if not normal or not abnormal:
            raise FileNotFoundError(f"OpenStack logs not found in {data_dir}")
        return loaders.load_openstack(normal, abnormal)
    if name == "bgl":
        cands = sorted(glob.glob(str(data_dir / "BGL*.csv")) + glob.glob(str(data_dir / "BGL*.log")))
        if not cands:
            raise FileNotFoundError(f"BGL file not found in {data_dir}")
        return loaders.load_bgl(cands[0])
    if name == "thunderbird":
        cands = sorted(glob.glob(str(data_dir / "Thunderbird*.log")) + glob.glob(str(data_dir / "Thunderbird*.csv")))
        if not cands:
            raise FileNotFoundError(f"Thunderbird slice not found in {data_dir}")
        return loaders.load_thunderbird(cands[0])
    raise ValueError(f"Unsupported dataset '{name}'.")


def prepare(ds: loaders.LogDataset, seed: int):
    """Split, derive train-only features, scale.  Returns arrays in [0,1] + schema."""
    idx = np.arange(len(ds.y))
    tr, tmp = train_test_split(idx, test_size=0.4, stratify=ds.y, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.5, stratify=ds.y[tmp], random_state=seed)

    X = ds.X.copy()
    schema = dict(ds.schema)

    if ds.meta.get("needs_rarity"):                        # BGL template rarity (train counts only)
        counts = X.iloc[tr]["template_id"].value_counts()
        n = len(tr)
        X["template_rarity"] = -np.log((X["template_id"].map(counts).fillna(0) + 1) / (n + 1))
        X = X.drop(columns=["template_id"])

    keep = [c for c in X.columns if X.iloc[tr][c].nunique() > 1]       # drop train-constant features
    X = X[keep]
    schema = {c: schema.get(c, "moderate") for c in keep}

    sc = MinMaxScaler().fit(X.iloc[tr].to_numpy(float))
    def tf(i): return np.clip(sc.transform(X.iloc[i].to_numpy(float)), 0.0, 1.0)
    return (tf(tr), tf(va), tf(te)), (ds.y[tr], ds.y[va], ds.y[te]), schema, keep


# ─────────────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────────────

def best_threshold(s_val: np.ndarray, y_val: np.ndarray) -> float:
    qs = np.unique(np.quantile(s_val, np.linspace(0.50, 0.995, 200)))
    best, best_f1 = qs[-1], -1.0
    for t in qs:
        f = f1_score(y_val, (s_val >= t).astype(int), zero_division=0)
        if f > best_f1:
            best, best_f1 = t, f
    return float(best)


def metrics(y_val, y_test, s_val, s_test) -> Dict[str, float]:
    t = best_threshold(s_val, y_val)
    pred = (s_test >= t).astype(int)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "auc_roc": roc_auc_score(y_test, s_test),
        "avg_prec": average_precision_score(y_test, s_test),
    }


# ─────────────────────────────────────────────────────────────────────────────
#  One run
# ─────────────────────────────────────────────────────────────────────────────

def run_once(Xs, ys, schema, cols, eps: float, seed: int, allow_fallback: bool):
    Xtr, Xva, Xte = Xs
    ytr, yva, yte = ys
    record_eps = float("inf")

    if np.isfinite(eps):
        eng = DPEngine(base_epsilon=eps, random_state=seed * 1000 + int(eps * 100))
        eng.register_schema(schema)
        Xtr, Xva, Xte = (eng.perturb_matrix(a, cols) for a in (Xtr, Xva, Xte))
        record_eps = eng.record_epsilon
    Xtr, Xva, Xte = (a.astype(np.float32) for a in (Xtr, Xva, Xte))

    scores: Dict[str, tuple] = {}

    m = IsolationForest(n_estimators=100, random_state=seed).fit(Xtr)
    scores["IF"] = (-m.score_samples(Xva), -m.score_samples(Xte))

    m = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(Xtr)
    scores["LOF"] = (-m.score_samples(Xva), -m.score_samples(Xte))

    m = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1).fit(Xtr)
    scores["OCSVM"] = (-m.decision_function(Xva), -m.decision_function(Xte))

    ahdm = AHDM(input_dim=Xtr.shape[1], random_state=seed, allow_fallback=allow_fallback)
    ahdm.fit(Xtr, X_val=Xva, y_val=yva, verbose=0)
    nv = ahdm._score_scaler.transform(np.column_stack([ahdm.if_scores(Xva), ahdm.ae_scores(Xva)]))
    nt = ahdm._score_scaler.transform(np.column_stack([ahdm.if_scores(Xte), ahdm.ae_scores(Xte)]))
    scores["DAE-only"] = (nv[:, 1], nt[:, 1])
    scores["AHDM-fixed"] = (0.5 * nv.sum(1), 0.5 * nt.sum(1))
    scores["AHDM"] = (ahdm.decision_scores(Xva), ahdm.decision_scores(Xte))

    rows = []
    for name in METHODS:
        s_va, s_te = scores[name]
        rows.append({"method": name, **metrics(yva, yte, s_va, s_te)})
    extra = {"alpha": ahdm.alpha_, "record_epsilon": record_eps, "dae_backend": ahdm._dae.backend}
    test_scores = {name: scores[name][1] for name in METHODS}
    return rows, extra, yte, test_scores


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_eps(v: str) -> float:
    return float("inf") if v.lower() in ("inf", "none", "nodp") else float(v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--datasets", nargs="+", default=["synthetic", "openstack", "bgl"])
    ap.add_argument("--epsilons", nargs="+", type=parse_eps, default=[0.5, 1, 2, 5, 10, float("inf")])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="results")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Stratified, seeded subsample of large datasets (keeps the anomaly rate). "
                         "Needed for Thunderbird: OCSVM/LOF do not scale to >100k rows.")
    ap.add_argument("--allow-fallback", action="store_true",
                    help="Allow the PCA stand-in if PyTorch is missing (NOT the paper's DAE).")
    args = ap.parse_args()

    if not TORCH_AVAILABLE and not args.allow_fallback:
        raise SystemExit(f"PyTorch could not be imported, so the DAE cannot run.\n"
                         f"Actual error: {TORCH_IMPORT_ERROR}\n"
                         "Fix the torch installation (see README); do not use --allow-fallback "
                         "for paper results.")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    saved: Dict[str, np.ndarray] = {}
    info: Dict[str, Dict] = {}
    t0 = time.time()

    for name in args.datasets:
        ds = load_dataset(name, Path(args.data_dir))
        n_full, a_full = len(ds.y), int(ds.y.sum())
        if args.max_rows and n_full > args.max_rows:
            keep, _ = train_test_split(np.arange(n_full), train_size=args.max_rows,
                                       stratify=ds.y, random_state=12345)
            keep = np.sort(keep)
            ds.X = ds.X.iloc[keep].reset_index(drop=True)
            ds.y = ds.y[keep]
            log.info("%s: stratified subsample %d -> %d rows", name, n_full, len(keep))
        info[name] = {"summary": ds.summary(), "label_note": ds.label_note, "unit": ds.unit,
                      "full_rows": n_full, "full_anomalies": a_full,
                      "max_rows": args.max_rows}
        log.info(ds.summary())
        for seed in range(args.seeds):
            Xs, ys, schema, cols = prepare(ds, seed)
            if seed == 0:
                info[name]["features_used"] = cols
                info[name]["split_sizes"] = [len(y) for y in ys]
                info[name]["test_anomalies_seed0"] = int(ys[2].sum())
            for eps in args.epsilons:
                r, extra, yte, sc = run_once(Xs, ys, schema, cols, eps, seed, args.allow_fallback)
                for row in r:
                    rows.append({"dataset": name, "epsilon": eps, "seed": seed, **extra, **row})
                if seed == 0:
                    saved[f"{name}|{eps}|y"] = yte
                    for mth, s in sc.items():
                        saved[f"{name}|{eps}|{mth}"] = s
            log.info("%s seed %d done (%.0fs elapsed)", name, seed, time.time() - t0)

    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics_long.csv", index=False)
    np.savez_compressed(out / "test_scores_seed0.npz", **saved)

    # ── summary (mean ± std over seeds) ───────────────────────────────────────
    g = df.groupby(["dataset", "epsilon", "method"])[["accuracy", "precision", "recall",
                                                      "f1", "auc_roc", "avg_prec"]]
    summ = g.mean().join(g.std(), rsuffix="_std").reset_index()
    summ.to_csv(out / "summary.csv", index=False)

    # ── paired significance: AHDM vs each other method, across seeds ──────────
    stats = []
    for (d, e), sub in df.groupby(["dataset", "epsilon"]):
        piv = {m: sub[sub.method == m].sort_values("seed") for m in METHODS}
        for other in METHODS:
            if other == "AHDM":
                continue
            for met in ("auc_roc", "f1"):
                a, b = piv["AHDM"][met].to_numpy(), piv[other][met].to_numpy()
                diff = a - b
                try:
                    p = 1.0 if np.allclose(diff, 0) else float(wilcoxon(a, b).pvalue)
                except ValueError:
                    p = float("nan")
                stats.append({"dataset": d, "epsilon": e, "vs": other, "metric": met,
                              "mean_diff": float(diff.mean()), "p_wilcoxon": p, "n_seeds": len(diff)})
    pd.DataFrame(stats).to_csv(out / "significance.csv", index=False)

    meta = {
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "numpy": np.__version__, "pandas": pd.__version__,
        "torch_available": TORCH_AVAILABLE, "dae_backend": str(df["dae_backend"].iloc[0]),
        "seeds": args.seeds, "epsilons": [str(e) for e in args.epsilons],
        "datasets": info, "runtime_s": round(time.time() - t0, 1),
    }
    (out / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    log.info("Finished in %.0fs. Results in %s/", time.time() - t0, out)


if __name__ == "__main__":
    main()
