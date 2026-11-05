"""Report figures as inline SVG, drawn from results/ only."""
import json

import networkx as nx
import numpy as np

from config import QUERIES, RESULTS, PQ_GRID
from src import report_lib as R
from src.graphs import load

S = json.loads((RESULTS / "summary.json").read_text())
U = json.loads((RESULTS / "understand.json").read_text())
TUN = json.loads((RESULTS / "tuning.json").read_text())
plt = R.mpl_setup()
GRAY = "#9a9994"
GR = {"small": "Small graph (222 nodes)", "large": "Large graph (4,589 nodes)"}


def q(g, c, k="20", m="ndcg"):
    return S["quality"][g][c]["0"]["test"][k][m]["mean"]


def lift(g, c, k="20"):
    return S["quality"][g][c]["0"]["label"][k]["lift"]["mean"]


def tradeoff():
    """Fig 1: held-out links against label lift. The two quality measures disagree."""
    rows = [("base_aa", "Adamic-Adar", GRAY), ("base_cn", "Common neighbours", GRAY), ("base_popularity", "Popularity", GRAY), ("base_random", "Random", GRAY),
            ("dw_hsns", "DeepWalk", R.BLUE), ("n2v_tuned", "node2vec tuned", R.ORANGE), ("n2v_hs", "node2vec p=q=1, HS", R.AQUA)]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
    off = {"small": {"n2v_tuned": (5, 7), "dw_hsns": (-34, -11), "n2v_hs": (5, -3), "base_cn": (-48, -11), "base_aa": (-14, 6)},
           "large": {"n2v_tuned": (-52, -3), "dw_hsns": (-16, -11), "n2v_hs": (5, 3), "base_cn": (-44, -11), "base_aa": (-16, 6)}}
    for ax, g in zip(axs, ("small", "large")):
        R.style_axes(ax)
        for c, name, col in rows:
            x, y = q(g, c), lift(g, c)
            ax.scatter([x], [y], s=34, color=col, edgecolor="white", linewidth=0.8, zorder=3)
            ax.annotate(name, (x, y), xytext=off[g].get(c, (5, 3)), textcoords="offset points", fontsize=6.6, color=R.INK2)
        ax.set_xlabel("Held-out links: test NDCG@20")
        ax.set_ylabel("Label lift@20")
        ax.set_title(GR[g], fontsize=8, loc="left")
        ax.xaxis.grid(True, color=R.GRID, linewidth=0.5)
        ax.margins(x=0.14, y=0.1)
    fig.tight_layout()
    return R.svg(fig)


def k_curves():
    """Fig: precision with its ceiling, and NDCG, against K."""
    Ks = ["1", "5", "10", "20", "50"]
    xs = [int(k) for k in Ks]
    rows = [("base_aa", "Adamic-Adar", GRAY), ("dw_hsns", "DeepWalk", R.BLUE), ("n2v_tuned", "node2vec tuned", R.ORANGE)]
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.1))
    for j, g in enumerate(("small", "large")):
        a, b = axs[2 * j], axs[2 * j + 1]
        for ax in (a, b):
            R.style_axes(ax)
            ax.set_xscale("log")
            ax.set_xticks(xs, Ks)
            ax.minorticks_off()
            ax.set_xlabel("K")
        for c, name, col in rows:
            a.plot(xs, [q(g, c, k, "p") for k in Ks], color=col, linewidth=1.6, marker="o", markersize=3, label=name)
            b.plot(xs, [q(g, c, k, "ndcg") for k in Ks], color=col, linewidth=1.6, marker="o", markersize=3)
        a.plot(xs, [q(g, "dw_hsns", k, "cap") for k in Ks], color=R.INK2, linewidth=1, linestyle="--", label="Precision ceiling")
        a.set_title(f"{g.capitalize()}: precision@K", fontsize=7.5, loc="left")
        b.set_title(f"{g.capitalize()}: NDCG@K", fontsize=7.5, loc="left")
        b.set_ylim(bottom=0)
        a.set_ylim(bottom=0)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout()
    return R.svg(fig)


def query_panels():
    """Fig: top-20 for each query on the small graph, under each algorithm, on one fixed layout of the graph."""
    g = load("small")
    G = nx.Graph()
    G.add_nodes_from(g.titles)
    G.add_edges_from((g.titles[a], g.titles[b]) for a, b in g.edges)
    pos = nx.spring_layout(G, seed=6220, iterations=120, k=0.3)
    xy = np.array([pos[t] for t in g.titles])
    fig, axs = plt.subplots(2, 5, figsize=(7.2, 3.0))
    L = U["lists"]["small"]
    for i, a in enumerate(("DeepWalk", "node2vec")):
        for j, t in enumerate(QUERIES):
            ax = axs[i, j]
            ax.scatter(xy[:, 0], xy[:, 1], s=2, color="#d9d8d4", linewidths=0, zorder=1)
            items = L[t][a]["20"]["items"]
            for x in items:
                p = pos[x["title"]]
                ax.plot([pos[t][0], p[0]], [pos[t][1], p[1]], color="#bdbcb7", linewidth=0.35, zorder=2)
            for x in items:
                p = pos[x["title"]]
                ax.scatter([p[0]], [p[1]], s=13, zorder=3, facecolor=R.BLUE if x["label"] else "white", edgecolor=R.BLUE, linewidth=0.7)
            ax.scatter([pos[t][0]], [pos[t][1]], s=34, marker="D", color=R.ORANGE, edgecolor="white", linewidth=0.6, zorder=4)
            ax.set_xticks([])
            ax.set_yticks([])
            for s_ in ax.spines.values():
                s_.set_visible(False)
            if i == 0:
                ax.set_title(t.replace("_", " "), fontsize=6.6)
            if j == 0:
                ax.set_ylabel(a, fontsize=7.5)
    h = [plt.Line2D([], [], marker="D", color=R.ORANGE, linestyle="", markersize=5, label="Query"),
         plt.Line2D([], [], marker="o", color=R.BLUE, linestyle="", markersize=4, label="Top-20, shares a subcategory"),
         plt.Line2D([], [], marker="o", markerfacecolor="white", markeredgecolor=R.BLUE, linestyle="", markersize=4, label="Top-20, no shared subcategory"),
         plt.Line2D([], [], marker="o", color="#d9d8d4", linestyle="", markersize=2.5, label="Other articles")]
    fig.legend(handles=h, loc="lower center", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(h_pad=0.2, w_pad=0.2)
    return R.svg(fig)


def runtime():
    """Fig: where the time goes on the large graph."""
    T = S["timing"]["large"]
    rows = [("dw_hs", "DeepWalk walks, HS"), ("dw_ns", "DeepWalk walks, NS"), ("n2v_ns", "node2vec p=q=1, NS"), ("n2v_p2_q0.5", "node2vec tuned, NS"),
            ("dw_hsns", "DeepWalk as written, HS+NS"), ("n2v_hsns", "node2vec p=q=1, HS+NS")]
    fig, ax = plt.subplots(figsize=(7.2, 1.9))
    R.style_axes(ax)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=R.GRID, linewidth=0.5)
    parts = [("precompute_s", "Probability precompute", R.ORANGE), ("walk_s", "Walk generation", R.AQUA), ("train_s", "Word2vec training", R.BLUE)]
    for i, (c, name) in enumerate(rows[::-1]):
        left = 0
        for key, lab, col in parts:
            v = T[c]["1"][key]["mean"]
            ax.barh(i, v, left=left, color=col, height=0.62, edgecolor="white", linewidth=1, label=lab if i == 0 else None)
            left += v
        ax.text(left + 6, i, f"{left:.0f} s", va="center", fontsize=6.8, color=R.INK2)
    ax.set_yticks(range(len(rows)), [n for _, n in rows[::-1]])
    ax.set_xlabel("Seconds, one worker, mean of 3 repeats")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=7)
    ax.set_xlim(right=max(T[c]["1"]["total_s"]["mean"] for c, _ in rows) * 1.1)
    fig.tight_layout()
    return R.svg(fig)


def workers():
    T = S["timing"]["large"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 1.9), sharex=True)
    W = ["1", "2", "4", "8"]
    xs = [int(w) for w in W]
    for ax, key, title in zip(axs, ("train_s", "total_s"), ("Word2vec training (s)", "Total including walks and precompute (s)")):
        R.style_axes(ax)
        for c, n, col in (("dw_ns", "DeepWalk walks, NS", R.BLUE), ("n2v_ns", "node2vec p=q=1, NS", R.ORANGE)):
            ax.plot(xs, [T[c][w][key]["mean"] for w in W], color=col, linewidth=1.6, marker="o", markersize=3, label=n)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs, W)
        ax.minorticks_off()
        ax.set_ylim(bottom=0)
        ax.set_title(title, fontsize=7.5, loc="left")
        ax.set_xlabel("Workers")
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return R.svg(fig)


def heatmaps():
    fig, axs = plt.subplots(1, 2, figsize=(5.6, 2.25))
    for ax, g in zip(axs, ("small", "large")):
        C = TUN[g]["cells"]
        M = np.array([[C[f"n2v_p{p:g}_q{q_:g}"]["val"]["mean"] for q_ in PQ_GRID] for p in PQ_GRID])
        ax.imshow(M, cmap="Blues", origin="lower", aspect="auto")
        for i in range(5):
            for j in range(5):
                on = M[i, j] > M.min() + 0.62 * (M.max() - M.min())
                sel = PQ_GRID[i] == TUN[g]["p"] and PQ_GRID[j] == TUN[g]["q"]
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=6, color="white" if on else R.INK, fontweight="bold" if sel else "normal")
        ax.set_xticks(range(5), [f"{x:g}" for x in PQ_GRID])
        ax.set_yticks(range(5), [f"{x:g}" for x in PQ_GRID])
        ax.tick_params(length=0, labelsize=7, colors=R.INK2)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_xlabel("q")
        ax.set_ylabel("p")
        ax.set_title(f"{g.capitalize()}: validation NDCG@20", fontsize=7.5, loc="left")
    fig.tight_layout()
    return R.svg(fig)


def sweeps():
    """Fig: each hyperparameter moved alone from the shared config."""
    fams = [("Dimension d", "d", [32, 64, 128, 256], R.BLUE), ("Walk length l", "l", [20, 40, 80], R.ORANGE), ("Window w", "w", [5, 10, 20], R.AQUA), ("Epochs", "epochs", [1, 5, 10], "#7a5bb5")]
    base = dict(d=128, l=80, w=10, epochs=5)
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.6))
    for ax, g in zip(axs, ("small", "large")):
        R.style_axes(ax)
        ax.xaxis.grid(True, color=R.GRID, linewidth=0.5)
        for name, key, vals, col in fams:
            xs, ys = [], []
            for v in vals:
                c = "n2v_ns" if base[key] == v else f"n2v_ns+{key}={v}"
                xs.append(q(g, c))
                ys.append(lift(g, c))
                if base[key] != v:   # every line passes through the shared config, marked once below
                    ax.annotate(str(v), (xs[-1], ys[-1]), xytext=(3, 3), textcoords="offset points", fontsize=6, color=R.INK2)
            ax.plot(xs, ys, color=col, linewidth=1.4, marker="o", markersize=3.5, label=name)
        ax.scatter([q(g, "n2v_ns")], [lift(g, "n2v_ns")], s=38, color=R.INK, zorder=5, label="Shared config")
        ax.set_xlabel("Test NDCG@20")
        ax.set_ylabel("Label lift@20")
        ax.set_title(GR[g], fontsize=8, loc="left")
    axs[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    return R.svg(fig)
