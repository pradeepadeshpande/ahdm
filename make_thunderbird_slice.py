"""
make_thunderbird_slice.py
=========================
Transparent way to get a labelled slice of the full Thunderbird log (211 M lines).

Step 1 - survey (streams once, prints alert counts per block):
    python make_thunderbird_slice.py report  Thunderbird.log --block 1000000
Step 2 - extract a CONTIGUOUS window you choose from the report:
    python make_thunderbird_slice.py extract Thunderbird.log --start 12000000 \
        --lines 200000 --out data/Thunderbird_slice.log

Report the start offset, length and alert count in the paper.  Do not cherry-pick
individual lines: contiguity keeps the slice a faithful excerpt of the real log.
"""
import argparse
import itertools


def report(path: str, block: int) -> None:
    n = alerts = 0
    with open(path, errors="ignore") as fh:
        for i, line in enumerate(fh):
            n += 1
            alerts += not line.startswith("- ")
            if (i + 1) % block == 0:
                print(f"lines {i + 1 - block:>11,}-{i:>11,}: {alerts:>9,} alerts ({100 * alerts / block:.2f}%)")
                alerts = 0
    if n % block:
        print(f"final partial block: {alerts:,} alerts in {n % block:,} lines")
    print(f"total lines: {n:,}")


def extract(path: str, start: int, lines: int, out: str) -> None:
    alerts = 0
    with open(path, errors="ignore") as fh, open(out, "w") as o:
        for line in itertools.islice(fh, start, start + lines):
            o.write(line)
            alerts += not line.startswith("- ")
    print(f"wrote {lines:,} lines from offset {start:,}: {alerts:,} alerts ({100 * alerts / lines:.2f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["report", "extract"])
    ap.add_argument("log")
    ap.add_argument("--block", type=int, default=1_000_000)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--lines", type=int, default=200_000)
    ap.add_argument("--out", default="data/Thunderbird_slice.log")
    a = ap.parse_args()
    report(a.log, a.block) if a.mode == "report" else extract(a.log, a.start, a.lines, a.out)
