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
    # Part 2, Tier 2. Three seeds. Sweeps change one thing from n2v_ns (uniform walk, the shared config).
    s3 = SEEDS[:3]
    sweep = ["d=32", "d=64", "d=256", "l=20", "l=40", "w=5", "w=20", "epochs=1", "epochs=10"]
    write("sweep", [line(g, f"n2v_ns+{o}", s) for g in g2 for o in sweep for s in s3])
    write("hubs", [line(g, f"{c}+{o}", s) for g in g2 for c in ("dw_ns", "n2v_ns")
                   for o in ("hubdrop=0.01", "sample=0") for s in s3])
    fb = [line(g, c, s) for g in ("fb_small", "fb_large") for c in ("dw_hsns", "n2v_ns", "n2v_hs", "n2v_p0.25_q4", "n2v_p4_q0.25") + BASELINES
          for s in s3]
    write("replicate", fb)
    p = RESULTS / "tuning.json"
    if p.exists():
        t = json.loads(p.read_text())
        rows = []
        for g in g2:
            if g not in t:
                continue
            c = t[g]["config"]
            rows += [line(g, c, s) for s in SEEDS]
            rows += [line(g, c, s, k) for s in SEEDS for k in SPLITS if k]
        write("tuned", rows)


if __name__ == "__main__":
    main()
