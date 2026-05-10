"""Orchestrator for the deterministic skeleton (stages 0, 2, 3, 4)."""
import json
from pathlib import Path

from rich.console import Console

from .cluster import cluster_regions
from .detect import detect_and_recognize
from .render import render_pdf
from .visualize import draw_regions

console = Console()


def run(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = 400,
    conf_threshold: float = 0.85,
    merge_distance: int = 60,
) -> dict:
    pdf_path = Path(pdf_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pages_dir = out_dir / "pages"

    console.log(f"[bold cyan]Stage 0[/]: rendering [italic]{pdf_path.name}[/] at {dpi} DPI")
    page_paths = render_pdf(pdf_path, dpi=dpi, out_dir=pages_dir)
    console.log(f"  -> {len(page_paths)} page image(s) in {pages_dir}")

    all_regions: list = []
    page_summaries = []

    for i, page_path in enumerate(page_paths, start=1):
        console.log(f"[bold cyan]Stages 2+4[/]: detect + recognize page {i}")
        regions = detect_and_recognize(page_path, page_num=i)
        console.log(f"  -> {len(regions)} text regions")

        console.log(f"[bold cyan]Stage 3[/]: cluster page {i} (merge_distance={merge_distance}px)")
        regions = cluster_regions(regions, merge_distance=merge_distance)
        n_clusters = len({r.cluster_id for r in regions})
        console.log(f"  -> {n_clusters} clusters")

        # Confidence gating (stand-in for the agent decision later)
        for r in regions:
            r.needs_agent = (r.confidence or 0.0) < conf_threshold
        n_low = sum(1 for r in regions if r.needs_agent)
        console.log(
            f"  -> {n_low}/{len(regions)} regions below conf {conf_threshold} "
            f"(would route to agent)"
        )

        # Debug visualizations
        det_overlay = out_dir / f"page_{i:03d}_detect.png"
        clu_overlay = out_dir / f"page_{i:03d}_clusters.png"
        draw_regions(page_path, regions, det_overlay, by_cluster=False)
        draw_regions(page_path, regions, clu_overlay, by_cluster=True)
        console.log(f"  -> overlays: {det_overlay.name}, {clu_overlay.name}")

        all_regions.extend(regions)
        page_summaries.append(
            {
                "page": i,
                "image": str(page_path),
                "n_regions": len(regions),
                "n_clusters": n_clusters,
                "n_low_confidence": n_low,
            }
        )

    out_json = out_dir / "extraction.json"
    payload = {
        "pdf": str(pdf_path),
        "dpi": dpi,
        "conf_threshold": conf_threshold,
        "merge_distance": merge_distance,
        "pages": page_summaries,
        "regions": [r.model_dump() for r in all_regions],
    }
    out_json.write_text(json.dumps(payload, indent=2))
    console.log(f"[bold green]Done.[/] {len(all_regions)} regions -> {out_json}")
    return payload
