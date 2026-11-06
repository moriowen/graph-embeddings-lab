"""Appendix A: every prompt typed to the AI assistant for HW5, verbatim, typos kept.

Pulled from the Claude Code session logs in ~/.claude/projects/-Users-atharvamohite-dev-acads-bds/
(fe9e03e6-....jsonl for P1, c034b172-....jsonl from P2). Times are US Eastern, Nov 5. "choice" entries are menu answers in the
assistant's question dialog, not typed text. Automated background-task notices are not prompts and are left out. Nothing here
carries a GTID.

ROUND_INSTRUCTIONS are not the student's prompts: the assistant wrote them for its own analysis rounds
(report/ai_analysis_r1.md, r2.md). They are listed separately so the report never presents them as typed by the student.
"""

PROMPTS = [
    ("P1", "11:29", "typed",
     "create a prompot for /Users/atharvamohite/dev/acads/bds/HW5/2026-ProgrammingAssignment-5.pdf the same way we did for "
     "/Users/atharvamohite/dev/acads/bds/HW4/PROMPT.md and follow the same setup structure as HW4"),
    ("P2", "11:40", "typed", "using @HW5/PROMPT.md implement all the tiers incrementally"),
    ("P3", "11:50", "choice",
     "Answers to three questions from the assistant. Plan (nested Wikispeedia graphs, K=5 and 20, d=128, l=80, r=10, w=10, three "
     "loss columns, about 234 runs) = \"Approved, build it (Recommended)\". Run the vendored repos after a permission block = "
     "\"Yes, run the vendored repos (Recommended)\". PACE timed out, is the VPN on = \"VPN is on now\"."),
    ("P4", "12:11", "typed", "is the run still active?"),
    ("P5", "12:12", "typed", "which GPU is this being run on? An A100 should suffice for this right?"),
    ("P6", "12:36", "typed", "what is the queue status?"),
    ("P7", "12:38", "typed", "PACE is connected now"),
    ("P8", "13:03", "typed", "VPN is on how"),
    ("P9", "13:15", "typed", "what is the status of the job?"),
    ("P10", "13:30", "typed", "what is the status"),
    ("P11", "13:38", "typed", "what will be generally the ETA for all the runs total?"),
    ("P12", "13:41", "typed", "is it possible to get access to a GPU to run this since CPUs are in use"),
    ("P13", "13:42", "typed", "can we run this on this PC? are do we need better specs?"),
    ("P14", "13:42", "typed", "Okay"),
    ("P15", "15:29", "typed", "what is the status?"),
    ("P16", "16:44", "pasted", "Codex's first review of the report, pasted with no added words. Verbatim in the zip: report/outside_review_1.md."),
    ("P17", "17:00", "pasted", "Codex's second review of the revised report, pasted with no added words. Verbatim in the zip: report/outside_review_2.md."),
    ("P18", "17:07", "typed + pasted", "\"blog\" followed by the text of the handout's blog, copied from a browser (reference: Esteve 2021)."),
    ("P19", "19:46", "typed", "just like the older assignments, can you please create a vercel site for this? to go over the experiments and results. name it graph embeddings lab"),
    ("P20", "19:56", "typed", "can you link the prompt md file from github into this pdf report in the prompt log section. push files to remote and link it"),
]

ROUND_INSTRUCTIONS = [
    ("R1", "assistant-written",
     "Read only the required rows of results/summary.json: the DeepWalk row, the tuned node2vec row and the four baselines, on both "
     "graphs. Write the conclusions a report would draw from those rows. One numbered claim per point. Do not use the control rows, the "
     "sweeps or the Part 2 analyses."),
    ("R2", "assistant-written",
     "Now test every round-1 claim against everything else in results/: the control matrix, the sweeps, the seed-level noise, the "
     "embedding analyses, the replication. For each claim say whether it survives, what it confused, and the number that shows it. Do not "
     "soften real findings."),
]
