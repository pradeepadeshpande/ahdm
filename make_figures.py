"""
make_figures.py
===============
Builds all figures and tables from results/ (written by run_experiments.py).
Reads only computed outputs; no values are typed in here.

    python make_figures.py --results results --eps 1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

METHODS = ["IF", "LOF", "OCSVM", "DAE-only", "AHDM-fixed", "AHDM"]
STYLE = {"IF": ("tab:blue", "--"), "LOF": ("tab:green", "--"), "OCSVM": ("tab:purple", "--"),
         "DAE-only": ("tab:orange", ":"), "AHDM-fixed": ("tab:gray", "-."), "AHDM": ("tab:red", "-")}


def eps_label(e: float) -> str:
    return "no DP" if np.isinf(e) else f"{e:g}"


def fig_utility(summ: pd.DataFrame, out: Path) -> None:
    datasets = list(summ["dataset"].unique())
    fig, axes = plt.subplots(2, len(datasets), figsize=(5 * len(datasets), 7), squeeze=False)
    for j, d in enumerate(datasets):
        sub = summ[summ.dataset == d]
        eps = sorted(sub["epsilon"].unique())
        x = np.arange(len(eps))
        for i, (met, lab) in enumerate([("auc_roc", "AUC-ROC"), ("f1", "F1 (val-tuned threshold)")]):
            ax = axes[i, j]
            for m in METHODS:
                s = sub[sub.method == m].set_index("epsilon").loc[eps]
                c, ls = STYLE[m]
                ax.errorbar(x, s[met], yerr=s[met + "_std"], label=m, color=c, ls=ls,
                            marker="o", ms=4, capsize=2, lw=1.6 if m == "AHDM" else 1.1)
            if met == "auc_roc":
                ax.axhline(0.5, color="k", lw=0.6, alpha=0.5)
            ax.set_xticks(x, [eps_label(e) for e in eps])
            ax.set_xlabel("per-attribute ε (base)")
            ax.set_ylabel(lab)
            ax.set_title(d if i == 0 else "")
            ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Privacy–utility trade-off (mean ± std over seeds)")
    fig.tight_layout()
    fig.savefig(out / "fig_privacy_utility.png", dpi=200)
    fig.savefig(out / "fig_privacy_utility.pdf")
    plt.close(fig)


def fig_curves(npz, datasets, eps: float, out: Path) -> None:
    show = ["IF", "OCSVM", "DAE-only", "AHDM"]
    fig, axes = plt.subplots(2, len(datasets), figsize=(5 * len(datasets), 8), squeeze=False)
    for j, d in enumerate(datasets):
        y = npz[f"{d}|{eps}|y"]
        for m in show:
            s = npz[f"{d}|{eps}|{m}"]
            c, ls = STYLE[m]
            fpr, tpr, _ = roc_curve(y, s)
            pr, rc, _ = precision_recall_curve(y, s)
            axes[0, j].plot(fpr, tpr, color=c, ls=ls, label=m)
            axes[1, j].plot(rc, pr, color=c, ls=ls, label=m)
        axes[0, j].plot([0, 1], [0, 1], "k:", lw=0.6)
        axes[1, j].axhline(y.mean(), color="k", ls=":", lw=0.6)
        axes[0, j].set(title=f"{d} (seed 0, ε={eps_label(eps)})", xlabel="FPR", ylabel="TPR")
        axes[1, j].set(xlabel="Recall", ylabel="Precision")
        for r in (0, 1):
            axes[r, j].grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"fig_roc_pr_eps{eps_label(eps).replace(' ', '')}.png", dpi=200)
    plt.close(fig)


def table_md(summ: pd.DataFrame, eps_list, out: Path) -> None:
    lines = []
    for e in eps_list:
        lines += [f"\n### ε = {eps_label(e)}\n",
                  "| Dataset | Method | Acc | Prec | Rec | F1 | AUC-ROC | AP |",
                  "|---|---|---|---|---|---|---|---|"]
        for d in summ["dataset"].unique():
            for m in METHODS:
                r = summ[(summ.dataset == d) & (summ.method == m) & (summ.epsilon == e)]
                if r.empty:
                    continue
                r = r.iloc[0]
                f = lambda k: f"{r[k]:.3f} ± {r[k + '_std']:.3f}"
                lines.append(f"| {d} | {m} | {f('accuracy')} | {f('precision')} | {f('recall')} | "
                             f"{f('f1')} | {f('auc_roc')} | {f('avg_prec')} |")
    (out / "results_tables.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--eps", type=float, default=1.0, help="epsilon for ROC/PR curves")
    a = ap.parse_args()
    out = Path(a.results)
    summ = pd.read_csv(out / "summary.csv")
    summ["epsilon"] = summ["epsilon"].astype(float)
    npz = np.load(out / "test_scores_seed0.npz")
    datasets = list(summ["dataset"].unique())

    fig_utility(summ, out)
    for e in (a.eps, float("inf")):
        fig_curves(npz, datasets, e, out)
    table_md(summ, [a.eps, float("inf")], out)
    print("Wrote figures and results_tables.md to", out)


if __name__ == "__main__":
    main()
