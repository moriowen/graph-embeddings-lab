"""Write slurm/jobs/*.txt, one `python -m src.run ...` per line. python -m src.jobs [stage]"""
import json
import sys

from config import BASELINES, MAIN_GRAPHS, MATRIX, PQ_GRID, RESULTS, ROOT, SEEDS, SPLITS, TUNE_SEEDS

J = ROOT / "slurm" / "jobs"
PY = "python -m src.run"


def line(g, c, s, split=0, extra=""):
    return f"{PY} --graph {g} --config {c} --seed {s} --split {split}{extra}"


def num(x):
    return f"{x:g}"


def write(name, lines):
    J.mkdir(parents=True, exist_ok=True)
    (J / f"{name}.txt").write_text("\n".join(lines) + "\n")
    print(name, len(lines))


def main():
    g2 = MAIN_GRAPHS
    # Small first in each list: a failure shows up quickly.
    write("tier0", [line(g, c, s) for g in g2 for c in ("dw_hsns", "n2v_ns") + BASELINES for s in SEEDS])
    write("controls", [line(g, c, s) for g in g2 for c in MATRIX + ("dw_repo", "n2v_repo") for s in SEEDS
                       if c not in ("dw_hsns", "n2v_ns")])
    write("tune", [line(g, f"n2v_p{num(p)}_q{num(q)}", s) for g in g2 for p in PQ_GRID for q in PQ_GRID for s in TUNE_SEEDS])
    # Split variation: the two main rows and the best baseline on the two other splits.
    write("splits", [line(g, c, s, k) for g in g2 for c in ("dw_hsns", "n2v_ns", "base_aa", "base_cn") for s in SEEDS
                     for k in SPLITS if k])
    p = RESULTS / "tuning.json"
    if p.exists():
        t = json.loads(p.read_text())
        rows = []
        for g in g2:
            c = t[g]["config"]
            rows += [line(g, c, s) for s in SEEDS]
            rows += [line(g, c, s, k) for s in SEEDS for k in SPLITS if k]
        write("tuned", rows)


if __name__ == "__main__":
    main()
