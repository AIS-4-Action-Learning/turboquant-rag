"""
extract.py — Extract text from PDFs into a corpus JSON for the RAG library.

Two-stage workflow (Approach B from our planning):
  1. Extract text from EVERY page → corpus_raw.json (no filtering)
  2. Apply page-skip rules + length threshold → corpus.json (filtered)
  3. Generate an extraction report so you can decide what to skip

Run from the project root:
    python3 extract.py

Edit the CONFIG section below to point at your PDFs and to specify which
pages to drop (front matter, indexes, etc.) for each document.

Output files:
  - data/corpus_raw.json       Every page extracted, no filtering
  - data/corpus.json           Filtered version, ready for RAG.build_index()
  - data/extraction_report.md  Per-page summary to inspect what got kept/dropped
"""

import json
import re
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# CONFIG — edit these for your project
# ---------------------------------------------------------------------------

INPUT_DIR = Path("data/pdfs")
OUTPUT_RAW = Path("data/corpus_raw.json")
OUTPUT_FILTERED = Path("data/corpus.json")
OUTPUT_REPORT = Path("data/extraction_report.md")

# Pages to skip per file (front matter, indexes, references).
# Use 1-based page numbers. After running once with empty rules and looking
# at the report, come back and fill these in for your textbook.
#
# Example for a deep learning textbook:
#   "deep_learning_book.pdf": {
#       "skip_pages": list(range(1, 13)) + list(range(580, 620)),
#       # ↑ skip first 12 pages (front matter) and pages 580-619 (index)
#   }
SKIP_RULES = {
    "d2l-en.pdf": {
        "skip_pages": (
            list(range(1, 34))            # title, TOC, preface, added installation guide
            + list(range(1129, 1152))     # references / bibliography (extended to 1151)
        ),
    },
}

# Pages with fewer characters than this are dropped from corpus.json.
# Increase if blank-ish pages still slip through; decrease if legitimate
# short pages (e.g., chapter title pages with one sentence) get filtered.
MIN_CHAR_THRESHOLD = 100

# How aggressively to detect repeated header/footer lines.
# A line appearing on more than this fraction of pages is treated as a
# header/footer and stripped. 0.3 = 30% of pages.
HEADER_FOOTER_RATIO = 0.3


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize whitespace, fix broken sentences, normalize bullets.

    Same logic as the original extract.py — kept because it works well.
    """
    # Normalize line endings
    text = text.replace("\r", "\n")

    # Fix broken lines inside sentences (lowercase letter starting next line
    # means the sentence was wrapped mid-sentence by the PDF layout)
    text = re.sub(r"\n(?=[a-z])", " ", text)

    # Normalize bullets
    text = text.replace("•", "- ")
    text = re.sub(r"-\s*\n", "- ", text)

    # Reduce excessive newlines
    text = re.sub(r"\n{2,}", "\n", text)

    # Normalize spaces
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def detect_repeated_lines(all_pages_text, threshold_ratio=HEADER_FOOTER_RATIO):
    """Find lines that appear on many pages — likely headers/footers."""
    all_lines = []
    for text in all_pages_text:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        all_lines.extend(lines)

    line_counts = Counter(all_lines)
    total_pages = len(all_pages_text)

    return {
        line for line, count in line_counts.items()
        if count > total_pages * threshold_ratio and len(line) < 200
    }


def remove_headers_footers(text: str, repeated_lines: set) -> str:
    """Strip lines flagged as headers/footers."""
    lines = text.split("\n")
    cleaned = [line for line in lines if line.strip() not in repeated_lines]
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# Per-PDF extraction
# ---------------------------------------------------------------------------

def extract_pdf(pdf_path: Path) -> list:
    """Extract all pages from a single PDF.

    Returns a list of dicts with keys: source, page, text, char_count.
    No filtering — every page is included, including blanks and front matter.
    """
    doc = fitz.open(pdf_path)

    # Pass 1: raw text per page (needed for header/footer detection)
    raw_pages = [page.get_text("text") for page in doc]

    # Detect repeated headers/footers across the whole document
    repeated_lines = detect_repeated_lines(raw_pages)

    # Pass 2: clean each page
    results = []
    for page_num, raw_text in enumerate(raw_pages, start=1):
        text = clean_text(raw_text)
        text = remove_headers_footers(text, repeated_lines)

        results.append({
            "source": pdf_path.name,
            "page": page_num,
            "text": text,
            "char_count": len(text),
        })

    doc.close()
    return results


# ---------------------------------------------------------------------------
# Filtering (Approach B: filter after extraction)
# ---------------------------------------------------------------------------

def apply_filters(raw_corpus: list) -> tuple:
    """Filter the raw corpus into a usable corpus.

    Returns (filtered_corpus, filter_log) where filter_log is a list of
    (source, page, reason) tuples for every dropped page — useful for
    the report.
    """
    filtered = []
    filter_log = []

    for entry in raw_corpus:
        source = entry["source"]
        page = entry["page"]
        char_count = entry["char_count"]

        # Rule 1: explicit skip list per source
        skip_rule = SKIP_RULES.get(source, {})
        if page in skip_rule.get("skip_pages", []):
            filter_log.append((source, page, "SKIP_RULES"))
            continue

        # Rule 2: too short to be useful
        if char_count < MIN_CHAR_THRESHOLD:
            filter_log.append((source, page, f"too short ({char_count} chars)"))
            continue

        filtered.append(entry)

    return filtered, filter_log


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(raw_corpus, filtered_corpus, filter_log) -> None:
    """Write a markdown report covering the extraction.

    The report is the main tool for deciding what to put in SKIP_RULES.
    Run extraction once with empty SKIP_RULES, read the report, identify
    page ranges that should be dropped, then re-run.
    """
    lines = []
    lines.append("# Extraction Report\n")

    # Summary
    by_source = {}
    for entry in raw_corpus:
        by_source.setdefault(entry["source"], []).append(entry)

    total_raw = len(raw_corpus)
    total_kept = len(filtered_corpus)
    total_dropped = len(filter_log)

    lines.append("## Summary\n")
    lines.append(f"- Source PDFs: {len(by_source)}")
    lines.append(f"- Total pages extracted: {total_raw}")
    lines.append(f"- Pages kept after filtering: {total_kept}")
    lines.append(f"- Pages dropped: {total_dropped}\n")

    # Per-source breakdown
    lines.append("## Per-source breakdown\n")
    for source, entries in by_source.items():
        kept = sum(1 for e in entries if e in filtered_corpus)
        lines.append(f"### {source}")
        lines.append(f"- Pages: {len(entries)}, kept: {kept}, dropped: {len(entries) - kept}")
        lines.append("")

    # Dropped pages
    lines.append("## Dropped pages\n")
    if not filter_log:
        lines.append("(none)\n")
    else:
        lines.append("| Source | Page | Reason |")
        lines.append("|--------|------|--------|")
        for source, page, reason in filter_log:
            lines.append(f"| {source} | {page} | {reason} |")
        lines.append("")

    # Per-page preview (first 200 chars of each KEPT page) — useful for
    # spotting front matter, indexes, etc. that should be added to SKIP_RULES.
    lines.append("## Page previews (first 200 chars of each kept page)\n")
    lines.append("Use this to spot pages that look like front matter, indexes,")
    lines.append("references, etc., and add them to SKIP_RULES at the top of")
    lines.append("extract.py.\n")
    for entry in filtered_corpus:
        preview = entry["text"][:200].replace("\n", " ")
        lines.append(f"- **{entry['source']}** p.{entry['page']} ({entry['char_count']} chars): {preview}")

    OUTPUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory not found: {INPUT_DIR}\n"
            f"Create it and put your PDFs inside, e.g.:\n"
            f"    mkdir -p {INPUT_DIR}\n"
            f"    cp /path/to/your/book.pdf {INPUT_DIR}/"
        )

    pdf_files = sorted(INPUT_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {INPUT_DIR}")

    print(f"Found {len(pdf_files)} PDF(s) in {INPUT_DIR}")

    # Stage 1: extract everything
    raw_corpus = []
    for pdf in pdf_files:
        print(f"  Extracting {pdf.name}...")
        raw_corpus.extend(extract_pdf(pdf))
    print(f"Extracted {len(raw_corpus)} pages total.")

    # Save raw extraction (the checkpoint we can rebuild filtered corpus from)
    OUTPUT_RAW.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_RAW, "w", encoding="utf-8") as f:
        json.dump(raw_corpus, f, ensure_ascii=False, indent=2)
    print(f"  Raw corpus → {OUTPUT_RAW}")

    # Stage 2: apply filters
    filtered_corpus, filter_log = apply_filters(raw_corpus)
    print(f"After filtering: {len(filtered_corpus)} pages kept, {len(filter_log)} dropped.")

    OUTPUT_FILTERED.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILTERED, "w", encoding="utf-8") as f:
        json.dump(filtered_corpus, f, ensure_ascii=False, indent=2)
    print(f"  Filtered corpus → {OUTPUT_FILTERED}")

    # Stage 3: report
    generate_report(raw_corpus, filtered_corpus, filter_log)
    print(f"  Report → {OUTPUT_REPORT}")

    print()
    print("Next steps:")
    print(f"  1. Open {OUTPUT_REPORT} and look at the page previews.")
    print("  2. Add front-matter / index / references page numbers to SKIP_RULES.")
    print("  3. Re-run this script. The filtered corpus is what RAG.build_index() uses.")


if __name__ == "__main__":
    main()