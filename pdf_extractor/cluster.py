"""Stage 3: cluster nearby text regions into composite groups.

Floor-plan labels often appear with neighbours that belong together
(label + dimensions + arrows). We use simple union-find on bbox proximity
so the agent stage can later operate on composite regions.
"""
from .models import Region


def _gap(a, b) -> tuple[int, int]:
    gap_x = max(a.x0 - b.x1, b.x0 - a.x1, 0)
    gap_y = max(a.y0 - b.y1, b.y0 - a.y1, 0)
    return gap_x, gap_y


def cluster_regions(regions: list[Region], merge_distance: int = 60) -> list[Region]:
    """Assign cluster_id to each region; merges any pair whose bboxes are
    within merge_distance px in both axes.
    """
    if not regions:
        return regions

    parent = list(range(len(regions)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            gx, gy = _gap(regions[i].bbox, regions[j].bbox)
            if gx <= merge_distance and gy <= merge_distance:
                union(i, j)

    for i, r in enumerate(regions):
        r.cluster_id = f"p{r.page}_c{find(i):04d}"
    return regions
