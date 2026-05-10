"""CLI entry point: `python -m pdf_extractor extract <pdf>`."""
from pathlib import Path

import typer

from .pipeline import run

app = typer.Typer(add_completion=False, help="PDF text-extraction skeleton (stages 0-4).")


@app.command()
def extract(
    pdf: Path = typer.Argument(..., exists=True, readable=True, help="Path to a PDF file."),
    out: Path = typer.Option(Path("output"), "--out", "-o", help="Output directory."),
    dpi: int = typer.Option(400, "--dpi", help="Render DPI (300-600 typical for floor plans)."),
    confidence: float = typer.Option(
        0.85, "--confidence", "-c",
        help="Below this OCR confidence, regions are flagged for the agent stage.",
    ),
    merge_distance: int = typer.Option(
        60, "--merge-distance", help="Pixel gap under which regions cluster together."
    ),
):
    """Run the deterministic skeleton (stages 0, 2, 3, 4) on a PDF."""
    run(
        pdf_path=pdf,
        out_dir=out,
        dpi=dpi,
        conf_threshold=confidence,
        merge_distance=merge_distance,
    )


if __name__ == "__main__":
    app()
