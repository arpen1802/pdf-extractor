# pdf-extractor

Detect-then-crop pipeline for extracting text from very large PDFs — built
for documents like building floor plans, where labels are small relative to
the page and naive image patching cuts text boxes in half.

## Approach

The full design is a hybrid: classical OCR guarantees coverage of every text
region; a vision LLM agent only spends compute on the hard cases (re-zooming
into low-confidence crops, grouping labels with their dimensions, etc.).

Pipeline stages:

| Stage | Purpose | Implemented |
|-------|---------|-------------|
| 0 | Render PDF pages at high DPI | yes |
| 1 | Page-level overview pass (vision LLM) | not yet |
| 2 | Exhaustive text detection (PaddleOCR) | yes |
| 3 | Cluster nearby regions (label + dim + arrow) | yes |
| 4 | Cheap OCR pass with confidence gating | yes |
| 5 | Agentic zoom/refinement (Claude vision) | not yet |
| 6 | Semantic grouping of text into logical units | not yet |
| 7 | Structured JSON output | yes (skeleton) |

This repo currently contains the deterministic skeleton (stages 0, 2, 3, 4,
7). Stages 1 and 5–6, which use the Claude API, are next.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

First run downloads PaddleOCR models (~200 MB) into `~/.paddlex/`.

## Usage

```bash
.venv/bin/python -m pdf_extractor.cli path/to/document.pdf \
    --dpi 300 \
    --confidence 0.85 \
    --merge-distance 60
```

Options:

- `--dpi` — render resolution. 300–600 typical for floor plans.
- `--confidence` — regions below this OCR score are flagged
  (`needs_agent=true`) for the future Stage 5 refinement loop.
- `--merge-distance` — pixel gap under which adjacent regions cluster
  together. Tune lower for dense pages, higher for sparse plans.

## Output

Written to `output/` by default:

- `extraction.json` — every region with bbox, polygon, text, confidence,
  cluster id, and the `needs_agent` flag
- `pages/page_NNN.png` — the rendered source page
- `page_NNN_detect.png` — every detected box drawn with text and confidence
- `page_NNN_clusters.png` — same regions colored by cluster, with cluster
  envelopes drawn

The two overlay PNGs are the fastest way to sanity-check detection and
clustering quality before tuning parameters.

## Why detect-then-crop instead of pure agentic zoom

Vision LLMs are surprisingly bad at "spot the text-like blob in a 20MP
image" and worse at outputting precise pixel coordinates to zoom into. A
~5 MB classical text detector does that job better, deterministically, and
guarantees coverage. The LLM earns its compute on what it's actually good
at: judgment, semantics, and re-reading hard crops at higher zoom.
