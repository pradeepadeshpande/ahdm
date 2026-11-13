"""
loaders.py
==========
Dataset loaders that turn raw log files into a numeric feature table with
labels and a DP sensitivity schema.  No label is invented: every label comes
from the provenance documented below.

OpenStack  (Loghub; DeepLog failure-injection logs)
    Files : openstack_normal1.log, openstack_normal2*.log, openstack_abnormal.log
    Unit  : one VM instance (all lines carrying "[instance: <uuid>]")
    Label : 1 if the instance appears in the failure-injection (abnormal) run,
            else 0.  This is a RUN-LEVEL label, not a per-VM ground truth.
    Note  : timestamps are NOT used as features, because the abnormal run was
            collected on a different date than the normal runs (confound).

BGL  (Loghub; Blue Gene/L RAS log)
    Files : BGL.log (raw) or BGL_*_structured.csv (Loghub structured)
    Unit  : one log line
    Label : 1 if the alert tag is not "-" (Loghub / original BGL convention)

Thunderbird  (Loghub; Sandia supercomputer syslog)
    Files : Thunderbird.log slice (raw) or a Loghub structured CSV
    Unit  : one log line
    Label : 1 if the alert tag is not "-".  The public 2k sample has NO alerts, so
            a labelled slice of the full log is required (see make_thunderbird_slice.py);
            the loader refuses to run with too few alerts.

Synthetic
    Generated with a fixed seed (see generate_synthetic).

Every loader returns a `LogDataset`.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from dp_engine import SensitivityLevel

H, M, L = SensitivityLevel.HIGH, SensitivityLevel.MODERATE, SensitivityLevel.LOW


@dataclass
class LogDataset:
    name: str
    X: pd.DataFrame                 # numeric features (raw scale)
    y: np.ndarray                   # 0 = normal, 1 = anomaly
    schema: Dict[str, str]          # feature -> sensitivity level
    unit: str                       # what one row represents
    label_note: str                 # how labels were obtained (for the paper)
    groups: Optional[np.ndarray] = None   # optional grouping key (no leakage across split)
    meta: Dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"{self.name}: {len(self.X)} {self.unit}s, {self.X.shape[1]} features, "
                f"{int(self.y.sum())} anomalies ({100 * self.y.mean():.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
_HEX = re.compile(r"\b0x[0-9a-fA-F]+\b|\b[0-9a-f]{16,}\b")
_IP = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?\b")
_NUM = re.compile(r"\d+(?:\.\d+)?")


def mask_template(msg: str) -> str:
    """Replace variable parts of a message so identical events share a key."""
    s = _UUID.sub("<ID>", msg)
    s = _IP.sub("<IP>", s)
    s = _HEX.sub("<HEX>", s)
    s = _NUM.sub("<N>", s)
    s = re.sub(r"/[\w./-]+", "<PATH>", s)
    return s.strip()[:120]


def template_id(msg: str) -> int:
    return int(hashlib.md5(mask_template(msg).encode()).hexdigest()[:8], 16)


def _entropy(counts: Sequence[int]) -> float:
    c = np.asarray(counts, dtype=float)
    p = c / c.sum()
    return float(-(p * np.log2(p + 1e-12)).sum())


# ─────────────────────────────────────────────────────────────────────────────
#  OpenStack
# ─────────────────────────────────────────────────────────────────────────────

_OS_LINE = re.compile(
    r"^(?P<logfile>\S+) (?P<date>\d{4}-\d\d-\d\d) (?P<time>\d\d:\d\d:\d\d\.\d+) "
    r"(?P<pid>\d+) (?P<level>[A-Z]+) (?P<component>\S+) (?P<rest>.*)$"
)
_OS_INSTANCE = re.compile(r"\[instance: ([0-9a-f-]{36})\] (.*)")
_LEVELS = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}


def _read_openstack_file(path: Path, label: int, run: str) -> pd.DataFrame:
    rows = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            m = _OS_LINE.match(line)
            if not m:
                continue
            inst = _OS_INSTANCE.search(m["rest"])
            if not inst:
                continue                       # lines without a VM id belong to no instance
            ts = pd.Timestamp(f"{m['date']} {m['time']}")
            rows.append((inst.group(1), ts, _LEVELS.get(m["level"], 1),
                         m["component"], inst.group(2), label, run))
    return pd.DataFrame(rows, columns=["instance", "ts", "level", "component",
                                       "msg", "label", "run"])


def load_openstack(
    normal_files: Sequence[str | Path],
    abnormal_files: Sequence[str | Path],
) -> LogDataset:
    """Aggregate per-VM-instance features from the Loghub OpenStack logs."""
    frames = [_read_openstack_file(Path(p), 0, Path(p).stem) for p in normal_files]
    frames += [_read_openstack_file(Path(p), 1, Path(p).stem) for p in abnormal_files]
    df = pd.concat(frames, ignore_index=True)

    # An instance id that occurs in both normal and abnormal files is ambiguous;
    # drop it rather than guess.
    both = df.groupby("instance")["label"].nunique()
    ambiguous = set(both[both > 1].index)
    df = df[~df["instance"].isin(ambiguous)]
    # an id repeated across the two normal files: merge (same label)

    df["tid"] = df["msg"].map(template_id)
    df["is_warn"] = (df["level"] >= 2).astype(int)
    df["is_err"] = (df["level"] >= 3).astype(int)

    feats: List[Dict] = []
    for inst, g in df.sort_values("ts").groupby("instance"):
        dur = (g["ts"].iloc[-1] - g["ts"].iloc[0]).total_seconds()
        gaps = g["ts"].diff().dt.total_seconds().dropna().to_numpy()
        feats.append({
            "instance": inst,
            "label": int(g["label"].iloc[0]),
            "n_events": len(g),
            "n_unique_events": g["tid"].nunique(),
            "event_entropy": _entropy(g["tid"].value_counts().to_numpy()),
            "n_warning": int(g["is_warn"].sum()),
            "n_error": int(g["is_err"].sum()),
            "duration_s": dur,
            "mean_gap_s": float(gaps.mean()) if len(gaps) else 0.0,
            "max_gap_s": float(gaps.max()) if len(gaps) else 0.0,
            "event_rate": len(g) / max(dur, 1.0),
            "n_components": g["component"].nunique(),
            "msg_len_mean": float(g["msg"].str.len().mean()),
            "n_compute_events": int(g["component"].str.contains("compute").sum()),
            "n_api_events": int(g["component"].str.contains("api|wsgi").sum()),
        })
    out = pd.DataFrame(feats).set_index("instance")
    y = out.pop("label").to_numpy().astype(int)

    schema = {
        "n_events": M, "n_unique_events": M, "event_entropy": M,
        "n_warning": M, "n_error": M,
        "duration_s": M, "mean_gap_s": M, "max_gap_s": M, "event_rate": M,
        "n_components": L, "msg_len_mean": L,
        "n_compute_events": L, "n_api_events": L,
    }
    return LogDataset(
        name="openstack", X=out, y=y, schema=schema, unit="VM instance",
        label_note=("Run-level labels: VM instances occurring in the failure-injection "
                    "run are anomalous, instances from the normal runs are normal. "
                    "No per-VM ground truth exists in the public data. Timestamps "
                    "excluded from features (runs were collected on different dates)."),
        meta={"ambiguous_dropped": len(ambiguous)},
    )


# ─────────────────────────────────────────────────────────────────────────────
#  BGL
# ─────────────────────────────────────────────────────────────────────────────

_SEVERITY = {"INFO": 0, "WARNING": 1, "ERROR": 2, "SEVERE": 3, "FATAL": 4, "FAILURE": 5}


def _read_bgl_raw(path: Path, max_lines: Optional[int]) -> pd.DataFrame:
    rows = []
    with open(path, errors="ignore") as fh:
        for i, line in enumerate(fh):
            if max_lines and i >= max_lines:
                break
            p = line.rstrip("\n").split(" ", 9)
            if len(p) < 10:
                continue
            rows.append({"Label": p[0], "Timestamp": int(p[1]), "Node": p[3],
                         "Type": p[6], "Component": p[7], "Level": p[8], "Content": p[9]})
    return pd.DataFrame(rows)


def load_bgl(path: str | Path, max_lines: Optional[int] = None) -> LogDataset:
    """
    Line-level BGL features.  Accepts the raw BGL.log or a Loghub structured CSV.
    Label = (alert tag != '-').
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, usecols=["Label", "Timestamp", "Node", "Component",
                                        "Level", "Content"], nrows=max_lines)
    else:
        df = _read_bgl_raw(path, max_lines)
    df = df.dropna(subset=["Content"]).reset_index(drop=True)
    y = (df["Label"].astype(str).str.strip() != "-").to_numpy().astype(int)

    content = df["Content"].astype(str)
    tid = content.map(template_id)
    comp = pd.Categorical(df["Component"]).codes

    X = pd.DataFrame({
        "severity": df["Level"].map(_SEVERITY).fillna(0).astype(int),
        "component_code": comp,
        "msg_len": content.str.len(),
        "n_tokens": content.str.split().str.len(),
        "n_digits": content.str.count(r"\d"),
        "n_hex": content.str.count(r"0x[0-9a-fA-F]+"),
        "upper_ratio": content.map(lambda s: sum(c.isupper() for c in s) / max(len(s), 1)),
        "template_id": tid,                       # converted to a rarity feature in the runner
        "node_rack": df["Node"].astype(str).str.extract(r"R(\d+)")[0].astype(float).fillna(-1),
        "node_is_io": df["Node"].astype(str).str.contains(r"-I:").astype(int),
    })
    schema = {
        "severity": M, "component_code": L, "msg_len": M, "n_tokens": M,
        "n_digits": M, "n_hex": M, "upper_ratio": L,
        "template_rarity": M,                     # derived from template_id on train only
        "node_rack": H, "node_is_io": H,
    }
    return LogDataset(
        name="bgl", X=X, y=y, schema=schema, unit="log line",
        label_note="Alert tag != '-' (original BGL / Loghub convention).",
        meta={"needs_rarity": True},
    )



# ─────────────────────────────────────────────────────────────────────────────
#  Thunderbird
# ─────────────────────────────────────────────────────────────────────────────

_TB_RAW = re.compile(
    r"^(?P<label>\S+) (?P<ts>\d+) (?P<date>\S+) (?P<node>\S+) (?P<mon>\w+) +(?P<day>\d+) "
    r"(?P<time>\S+) (?P<loc>\S+) (?P<rest>.*)$"
)
_TB_COMP = re.compile(r"^([^\[:\s]+)(?:\[(\d+)\])?:? ?(.*)$")
_ERR_KW = re.compile(r"fail|error|denied|refused|invalid|timeout|critical|fatal|panic|"
                     r"unable|cannot|can't|bad |corrupt|lost|abort", re.I)


def _read_tb_raw(path: Path, max_lines: Optional[int]) -> pd.DataFrame:
    rows = []
    with open(path, errors="ignore") as fh:
        for i, line in enumerate(fh):
            if max_lines and i >= max_lines:
                break
            m = _TB_RAW.match(line.rstrip("\r\n"))
            if not m:
                continue
            c = _TB_COMP.match(m["rest"])
            comp, pid, content = (c.group(1), c.group(2), c.group(3)) if c else ("", None, m["rest"])
            rows.append({"Label": m["label"], "Timestamp": int(m["ts"]), "Node": m["node"],
                         "Component": comp, "PID": pid, "Content": content})
    return pd.DataFrame(rows)


def load_thunderbird(path: str | Path, max_lines: Optional[int] = None,
                     min_anomalies: int = 30) -> LogDataset:
    """
    Line-level Thunderbird features from a raw slice or Loghub structured CSV.
    Label = (alert tag != '-').  Raises if the slice has fewer than `min_anomalies`
    alerts, because precision/recall/F1 would be undefined or meaningless.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, usecols=["Label", "Timestamp", "User", "Component",
                                        "PID", "Content"], nrows=max_lines)
        df = df.rename(columns={"User": "Node"})
    else:
        df = _read_tb_raw(path, max_lines)
    df = df.dropna(subset=["Content"]).reset_index(drop=True)
    y = (df["Label"].astype(str).str.strip() != "-").to_numpy().astype(int)
    if y.sum() < min_anomalies:
        raise ValueError(
            f"Thunderbird slice has only {int(y.sum())} alert lines out of {len(df)} "
            f"(need >= {min_anomalies}). The public 2k sample contains none. Extract a "
            "labelled slice from the full log with make_thunderbird_slice.py."
        )

    content = df["Content"].astype(str)
    X = pd.DataFrame({
        "component_code": pd.Categorical(df["Component"].fillna("")).codes,
        "has_pid": df["PID"].notna().astype(int),
        "msg_len": content.str.len(),
        "n_tokens": content.str.split().str.len(),
        "n_digits": content.str.count(r"\d"),
        "n_hex": content.str.count(r"0x[0-9a-fA-F]+"),
        "upper_ratio": content.map(lambda s: sum(c.isupper() for c in s) / max(len(s), 1)),
        "n_error_kw": content.map(lambda s: len(_ERR_KW.findall(s))),
        "mentions_root": content.str.contains(r"\broot\b").astype(int),
        "node_code": pd.Categorical(df["Node"].astype(str)).codes,
        "template_id": content.map(template_id),
    })
    schema = {
        "component_code": L, "has_pid": L, "msg_len": M, "n_tokens": M, "n_digits": M,
        "n_hex": M, "upper_ratio": L, "n_error_kw": M, "mentions_root": H,
        "node_code": H, "template_rarity": M,
    }
    return LogDataset(
        name="thunderbird", X=X, y=y, schema=schema, unit="log line",
        label_note=("Alert tag != '-' (Loghub convention) on a contiguous slice of the "
                    "full Thunderbird log; slice boundaries must be reported in the paper."),
        meta={"needs_rarity": True},
    )

# ─────────────────────────────────────────────────────────────────────────────
#  Synthetic
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic(n: int = 5000, anomaly_rate: float = 0.113, seed: int = 42) -> LogDataset:
    """Synthetic cloud-log features (fixed seed).  Anomalies = bursty/large sessions."""
    rng = np.random.default_rng(seed)
    n_a = int(n * anomaly_rate)
    n_n = n - n_a

    def block(k, anomalous):
        return pd.DataFrame({
            "trans_amt": rng.exponential(500 if anomalous else 50, k),
            "session_dur": rng.uniform(3600, 86400, k) if anomalous else rng.exponential(300, k),
            "access_count": rng.integers(50, 200, k) if anomalous else rng.poisson(10, k),
            "hour_of_day": rng.integers(0, 24, k),
            "is_night": rng.integers(0, 2, k),
            "resource_code": rng.integers(0, 3, k),
            "event_code": rng.integers(0, 3, k),
            "email_code": rng.integers(0, 999, k),
        })

    X = pd.concat([block(n_n, False), block(n_a, True)], ignore_index=True)
    y = np.r_[np.zeros(n_n, int), np.ones(n_a, int)]
    perm = rng.permutation(n)
    schema = {"trans_amt": M, "session_dur": M, "access_count": M, "hour_of_day": M,
              "is_night": M, "resource_code": L, "event_code": L, "email_code": H}
    return LogDataset(
        name="synthetic", X=X.iloc[perm].reset_index(drop=True), y=y[perm], schema=schema,
        unit="session", label_note=f"Synthetic, seed={seed}, anomaly rate={anomaly_rate}.",
    )
