# AI analysis, round 1 (restricted reading)

Instruction: Read only the required rows of results/summary.json: the DeepWalk row, the tuned node2vec row and the four baselines, on both graphs. Write the conclusions a report would draw from those rows. One numbered claim per point. Do not use the control rows, the sweeps or the Part 2 analyses.

- **C1.** DeepWalk beats tuned node2vec on the large graph: NDCG@20 0.170 vs 0.149, diff 0.020, Wilcoxon p = 3.0e-133. So uniform walks are better than biased walks on this graph.
- **C2.** On the small graph the two tie (0.338 vs 0.340), so the DeepWalk advantage appears as the graph grows.
- **C3.** Adamic-Adar beats both embeddings on held-out links on both graphs (0.505 and 0.223), so embeddings are not worth their cost for link recommendation on this data.
- **C4.** Embeddings beat every baseline on label lift (3.39 vs 2.85 on the large graph), so they understand topic better than the heuristics do.
- **C5.** Embeddings find relevant nodes the heuristics cannot reach, because a walk goes further than two hops.
- **C6.** Tuned node2vec costs more than DeepWalk because of its second-order walk and probability table.
- **C7.** node2vec's tuning found p=2, q=0.5 on both graphs. q below 1 favours moving outward, so these graphs reward DFS-like exploration.
- **C8.** Hubs drive the embedding lists, since United_States has degree 1,621 and walks pass through it constantly.
- **C9.** Cosine is the neutral choice of embedding distance, and the other metrics would give similar lists.
