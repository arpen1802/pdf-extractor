"""Stage 0: PDF -> high-DPI page images."""
from pathlib import Path

import fitz  # PyMuPDF


def render_pdf(pdf_path: Path, dpi: int = 400, out_dir: Path | None = None) -> list[Path]:
    """Render each PDF page to a PNG at the requested DPI.

    Returns the list of image paths in page order.
    """
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir) if out_dir else pdf_path.parent / "pages"
    out_dir.mkdir(parents=True, exist_ok=True)

    zoom = dpi / 72  # PDF user units are 1/72 inch
    matrix = fitz.Matrix(zoom, zoom)

    paths: list[Path] = []
    doc = fitz.open(pdf_path)
    try:
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            out = out_dir / f"page_{i:03d}.png"
            pix.save(out)
            paths.append(out)
    finally:
        doc.close()
    return paths
