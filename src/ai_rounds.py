"""Write report/ai_analysis_r1.md and r2.md, the recorded AI analysis rounds. Numbers come from results/. python -m src.ai_rounds

R1 reads only the required rows, on purpose. The assistant had already seen the control rows when it wrote R1, so R1 is a
restricted reading, not a blind one. The report says so.
"""
import json

from config import REPORT, RESULTS
from src.prompts import ROUND_INSTRUCTIONS

S = json.loads((RESULTS / "summary.json").read_text())
U = json.loads((RESULTS / "understand.json").read_text())
P2 = {g: json.loads((RESULTS / f"part2_{g}.json").read_text()) for g in ("small", "large")}


def q(g, c, part="test", k="20", m="ndcg"):
    e = S["quality"][g][c]["0"][part][k]
    return e[m]["mean"]


def lift(g, c, k="20"):
    return S["quality"][g][c]["0"]["label"][k]["lift"]["mean"]


def pr(g, key, a, b):
    return S["pairs"][f"{g}:{key}:{a}-{b}"]


T = "n2v_p2_q0.5"
t = S["timing"]


def main():
    ls, ll = P2["small"], P2["large"]
    r1 = f"""# AI analysis, round 1 (restricted reading)

Instruction: {ROUND_INSTRUCTIONS[0][2]}

- **C1.** DeepWalk beats tuned node2vec on the large graph: NDCG@20 {q('large','dw_hsns'):.3f} vs {q('large','n2v_tuned'):.3f}, diff {-pr('large','ndcg20',T,'dw_hsns')['diff']:.3f}, Wilcoxon p = {pr('large','ndcg20',T,'dw_hsns')['wilcoxon_p']:.1e}. So uniform walks are better than biased walks on this graph.
- **C2.** On the small graph the two tie ({q('small','dw_hsns'):.3f} vs {q('small','n2v_tuned'):.3f}), so the DeepWalk advantage appears as the graph grows.
- **C3.** Adamic-Adar beats both embeddings on held-out links on both graphs ({q('small','base_aa'):.3f} and {q('large','base_aa'):.3f}), so embeddings are not worth their cost for link recommendation on this data.
- **C4.** Embeddings beat every baseline on label lift ({lift('large','dw_hsns'):.2f} vs {lift('large','base_aa'):.2f} on the large graph), so they understand topic better than the heuristics do.
- **C5.** Embeddings find relevant nodes the heuristics cannot reach, because a walk goes further than two hops.
- **C6.** Tuned node2vec costs more than DeepWalk because of its second-order walk and probability table.
- **C7.** node2vec's tuning found p=2, q=0.5 on both graphs. q below 1 favours moving outward, so these graphs reward DFS-like exploration.
- **C8.** Hubs drive the embedding lists, since United_States has degree 1,621 and walks pass through it constantly.
- **C9.** Cosine is the neutral choice of embedding distance, and the other metrics would give similar lists.
"""
    (REPORT / "ai_analysis_r1.md").write_text(r1)
    aa_hub = ll["hubs"]["table"]["base_aa"]
    dwh = ll["hubs"]["table"]["dw_hs"]
    dh = ll["hubs"]["table"]["dw_hsns"]
    td, tn = t["large"]["dw_hs"]["1"], t["large"]["n2v_ns"]["1"]
    dhs = t["large"]["dw_hsns"]["1"]
    r2 = f"""# AI analysis, round 2 (tests of round 1)

Instruction: {ROUND_INSTRUCTIONS[1][2]}

- **C1 fails as worded.** The gap is the loss, not the walk. DeepWalk's code trains HS+NS, and node2vec's default trains NS. With the same loss the two tie on the large graph: NS {pr('large','ndcg20','dw_ns','n2v_ns')['diff']:+.4f}, HS {pr('large','ndcg20','dw_hs','n2v_hs')['diff']:+.4f}, HS+NS {pr('large','ndcg20','dw_hsns','n2v_hsns')['diff']:+.4f}. Moving the loss from NS to HS+NS adds {pr('large','ndcg20','n2v_hsns','n2v_ns')['diff']:.3f}.
- **C2 fails.** It reads a loss effect as a size effect, and the small graph's noise hides the same loss effect ({pr('small','ndcg20','n2v_hsns','n2v_ns')['diff']:+.3f}, seed sd about {pr('small','ndcg20','n2v_hsns','n2v_ns')['seed_sd_b']:.3f}).
- **C3 is too strong.** The win is real on cosine lists. With Euclidean distance on the same vectors, DeepWalk reaches {ll['distance']['dw_hsns']['euclidean']['ndcg20']:.3f} on the large graph, above Adamic-Adar's {q('large','base_aa'):.3f}. That comparison was found after looking at test, so it is a lead, not a result. With hubs out of the candidates and the held-out set, HS gets {dwh['ndcg_nohub']:.3f}, ahead of common neighbours ({ll['hubs']['table']['base_cn']['ndcg_nohub']:.3f}), though Adamic-Adar stays ahead ({aa_hub['ndcg_nohub']:.3f}).
- **C4 holds, but its explanation does not.** The lift gap is real and its cause is unproven. Adamic-Adar gives {aa_hub['hub_slots']:.0%} of its slots to the top 1% of nodes and embeddings about {dh['hub_slots']:.0%}. The two measures reward different lists, so "understand topic" overstates it.
- **C5 fails.** {ll['strata']['two_hop_share']:.0%} of large-graph test edges are within two hops, and {ll['strata']['table']['dw_hsns']['top_in_2hop']:.0%} of DeepWalk's slots land there. Only {ll['strata']['n_far']} test edges lie further out, and DeepWalk finds about {ll['strata']['table']['dw_hsns']['far_hits']:.0f} of them.
- **C6 needs its comparison fixed.** Tuned node2vec ({t['large'][T]['1']['total_s']['mean']:.0f} s) is cheaper than DeepWalk as written ({dhs['total_s']['mean']:.0f} s), because DeepWalk pays for two losses. With the same loss the walk does cost more ({tn['total_s']['mean']:.0f} s vs {t['large']['dw_ns']['1']['total_s']['mean']:.0f} s).
- **C7 overreaches.** On the seeds that selected it, p=2, q=0.5 scored {U['selection']['small']['selected_test_tuneseeds']:.3f} on small test; on fresh seeds it scores {q('small','n2v_tuned'):.3f}, against uniform's {q('small','n2v_ns'):.3f}. On the large graph the gain is {pr('large','ndcg20',T,'n2v_ns')['diff']:+.4f}, about a tenth of the loss effect. The pattern is real but small. q below 1 is the paper's outward setting, and p=2 also discourages returning.
- **C8 fails.** Embeddings give hubs about {dh['hub_slots']:.0%} of their slots. Removing hubs from the walks lowers NDCG only from {q('large','n2v_ns'):.3f} to {q('large','n2v_ns+hubdrop=0.01'):.3f}. The heuristics are the hub-driven method.
- **C9 fails.** Euclidean lists share about {ll['distance']['dw_hsns']['euclidean']['jaccard_cos']:.2f} Jaccard with cosine lists and recommend nodes of mean degree {ll['distance']['dw_hsns']['euclidean']['rec_degree']:.0f} rather than {ll['distance']['dw_hsns']['cosine']['rec_degree']:.0f}. Embedding norm falls with degree (Spearman {ll['distance']['dw_hsns']['cosine']['norm_deg_spearman']:.2f}), so dot product picks low-degree nodes. That is the opposite of what the brief expected.
"""
    (REPORT / "ai_analysis_r2.md").write_text(r2)
    print(r1[:600])
    print(r2)


if __name__ == "__main__":
    main()
