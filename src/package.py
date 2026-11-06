"""Build the submission ZIP: report PDF, source, vendored repos and patch, graphs, every run record, Slurm scripts and logs.

    python -m src.package                      # reuses the existing PDF if its ID line is filled in
    STUDENT_ID=... python -m src.package       # rebuilds the PDF first

Refuses to package a PDF with a blank ID line. Embeddings and per-node arrays stay out: they are large and reproducible.
"""
import os
import re
import sys
import zipfile

from pypdf import PdfReader

from config import ROOT
from src import build_pdf

NAME = "HW5_P_Mohite_Atharva"
ZIP = ROOT / f"{NAME}.zip"


def members():
    yield build_pdf.PDF
    yield from (ROOT / f for f in ("README.md", "requirements.txt", "config.py", "PROMPT.md", "PLAN.md", "IMPLEMENTATION.md", "AGENTS.md", "sync.sh"))
    for pattern in ("src/*.py", "vendor/patch.diff", "vendor/deepwalk/*", "vendor/node2vec/*", "data/provenance.json", "data/*/nodes.tsv",
                    "data/*/edges.tsv", "data/*/labels.json", "data/*/split_*.npz", "results/*.json", "slurm/*.sbatch", "slurm/*.sh",
                    "slurm/jobs/*", "slurm/logs/*.out", "report/report.html", "report/ai_analysis_r*.md", "report/outside_review_*.md", "src/site_template.html", "public/index.html", "vercel.json"):
        yield from sorted(p for p in ROOT.glob(pattern) if p.is_file() and "run_dev_" not in p.name and "__pycache__" not in str(p))


def pdf_has_id():
    if not build_pdf.PDF.exists():
        return False
    m = re.search(r"Student ID:\s*(\S+)", PdfReader(build_pdf.PDF).pages[0].extract_text())
    return bool(m and m.group(1)[0].isdigit())


def main():
    if os.environ.get("STUDENT_ID", "").strip():
        build_pdf.main()
    elif not pdf_has_id():
        sys.exit("the PDF has no student ID; run with STUDENT_ID=... to rebuild it")
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in members():
            z.write(f, f"{NAME}/{f.relative_to(ROOT)}")
    with zipfile.ZipFile(ZIP) as z:
        n = len(z.namelist())
    print(f"wrote {ZIP.name}  {n} files  {ZIP.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
