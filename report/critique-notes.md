# Running notes for the 4b critiques (scaffolding, not submitted)

Every item happened in this homework. File and number evidence lives in the report's 4b text.

1. Brief said `hs=1` means hierarchical softmax. Code passes `hs=1` and leaves `negative=5`, so gensim trains both.
2. Brief said `word2vec.Vocab` is gone. It still exists in 4.3.3. The import failures are `Mapping` and `Iterable`.
3. Brief said the precompute is memory-hungry and banned the Mac. Measured peak about 630 MB. It costs time (46 s), not memory.
4. Brief expected dot product to favour hubs. Norm falls with degree, so it does the reverse.
5. Plan estimated 10-15 min per large run and about 25 core-hours. Measured 100-500 s; the whole grid took about 16 core-hours.
6. Plan called the shared config "the paper's". The paper trains one epoch, ours 5. One epoch drops large NDCG@20 sharply.
7. Slurm array exceeded the 50-job QOS. Then 8-core jobs on one CPU model sat pending.
8. stats.py mixed tuning seeds into the tuned row. Would have reported a gain that fresh seeds don't show.
9. stats.py regex read fb_small as tag fb + graph small, so replication silently vanished.
10. Part 2 job failed: tuning.json was on the Mac only.
11. Extra list had only 2 cores until the ETA question exposed it.
12. Status slips: "39 of 39", "3 hours". Sync printed "synced" after rsync failed.
13. A/A check on the small graph shows a CI excluding 0. Node-level CI ignores seed noise. Large graph ties.
14. Blog returned 403 and was not read.
15. "Okay" (P14) was ambiguous. Read as the recommended option.
