# AI analysis, round 2 (tests of round 1)

Instruction: Now test every round-1 claim against everything else in results/: the control matrix, the sweeps, the seed-level noise, the embedding analyses, the replication. For each claim say whether it survives, what it confused, and the number that shows it. Do not soften real findings.

- **C1 fails as worded.** The gap is the loss, not the walk. DeepWalk's code trains HS+NS, and node2vec's default trains NS. With the same loss the two tie on the large graph: NS -0.0011, HS -0.0003, HS+NS -0.0011. Moving the loss from NS to HS+NS adds 0.024.
- **C2 fails.** It reads a loss effect as a size effect, and the small graph's noise hides the same loss effect (+0.009, seed sd about 0.010).
- **C3 is too strong.** The win is real on cosine lists. With Euclidean distance on the same vectors, DeepWalk reaches 0.229 on the large graph, above Adamic-Adar's 0.223. That comparison was found after looking at test, so it is a lead, not a result. With hubs out of the candidates and the held-out set, HS gets 0.185, ahead of common neighbours (0.176), though Adamic-Adar stays ahead (0.206).
- **C4 holds, but its explanation does not.** The lift gap is real and its cause is unproven. Adamic-Adar gives 35% of its slots to the top 1% of nodes and embeddings about 2%. The two measures reward different lists, so "understand topic" overstates it.
- **C5 fails.** 94% of large-graph test edges are within two hops, and 97% of DeepWalk's slots land there. Only 1333 test edges lie further out, and DeepWalk finds about 25 of them.
- **C6 needs its comparison fixed.** Tuned node2vec (305 s) is cheaper than DeepWalk as written (394 s), because DeepWalk pays for two losses. With the same loss the walk does cost more (303 s vs 202 s).
- **C7 overreaches.** On the seeds that selected it, p=2, q=0.5 scored 0.349 on small test; on fresh seeds it scores 0.340, against uniform's 0.339. On the large graph the gain is +0.0026, about a tenth of the loss effect. The pattern is real but small. q below 1 is the paper's outward setting, and p=2 also discourages returning.
- **C8 fails.** Embeddings give hubs about 2% of their slots. Removing hubs from the walks lowers NDCG only from 0.147 to 0.136. The heuristics are the hub-driven method.
- **C9 fails.** Euclidean lists share about 0.48 Jaccard with cosine lists and recommend nodes of mean degree 134 rather than 44. Embedding norm falls with degree (Spearman -0.84), so dot product picks low-degree nodes. That is the opposite of what the brief expected.
