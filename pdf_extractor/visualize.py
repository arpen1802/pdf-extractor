"""Debug overlays: draw detected/clustered regions on the page."""
from pathlib import Path

from PIL import Image, ImageDraw

from .models import Region

_PALETTE = [
    "#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#9A6324", "#800000", "#808000",
    "#000075", "#469990", "#bfef45", "#fabed4", "#ffe119",
]


def draw_regions(
    image_path: Path,
    regions: list[Region],
    out_path: Path,
    by_cluster: bool = False,
) -> None:
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    color_map: dict[str, str] = {}

    def color_for(key: str) -> str:
        if key not in color_map:
            color_map[key] = _PALETTE[len(color_map) % len(_PALETTE)]
        return color_map[key]

    for r in regions:
        key = (r.cluster_id or r.id) if by_cluster else r.id
        color = color_for(key)
        b = r.bbox
        draw.rectangle([b.x0, b.y0, b.x1, b.y1], outline=color, width=2)
        if r.text and not by_cluster:
            label = f"{r.text[:24]}"
            if r.confidence is not None:
                label += f" {r.confidence:.2f}"
            draw.text((b.x0, max(0, b.y0 - 11)), label, fill=color)

    if by_cluster:
        # Draw a bounding box around each cluster
        clusters: dict[str, list[Region]] = {}
        for r in regions:
            clusters.setdefault(r.cluster_id or r.id, []).append(r)
        for cid, group in clusters.items():
            if len(group) < 2:
                continue
            x0 = min(r.bbox.x0 for r in group)
            y0 = min(r.bbox.y0 for r in group)
            x1 = max(r.bbox.x1 for r in group)
            y1 = max(r.bbox.y1 for r in group)
            draw.rectangle([x0 - 4, y0 - 4, x1 + 4, y1 + 4], outline=color_for(cid), width=4)

    img.save(out_path)
