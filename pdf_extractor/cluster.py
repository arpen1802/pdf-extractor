"""Stage 3: cluster text regions into multi-line blocks using OBB proximity.

Axis-aligned bbox distance is wrong for rotated text (floor-plan labels can
be horizontal, vertical, or any angle). Instead we:

  1. Group regions that share the same angle bucket (within ANGLE_TOLERANCE).
  2. Within each angle group, project region centres onto the text's local
     coordinate frame (parallel u, perpendicular v).
  3. Two regions belong to the same block if their spans overlap along u
     (they're part of the same column/line sequence) AND the perpendicular
     gap along v is small (≤ line_height × PERP_GAP_FACTOR).
  4. After clusters are formed, sort members in reading order and join text
     into merged_text.
"""
import math
from .models import Region

ANGLE_TOLERANCE = 15.0   # degrees — angle difference within which two regions are "same orientation"
PERP_GAP_FACTOR = 0.8    # max perpendicular gap as a multiple of the shorter region's height
PARA_OVERLAP_MIN = 0.0   # parallel overlap required (0 = touching is enough)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _obb(region: Region) -> tuple[float, float, float, float, float]:
    """Return (cx, cy, half_para, half_perp, angle_rad) for a region.

    half_para  = half-length along the text direction (u axis)
    half_perp  = half-height perpendicular to text direction (v axis)
    """
    poly = region.polygon
    angle_rad = math.radians(region.angle)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    # Project all 4 corners into the local (u, v) frame
    us = [ cos_a * x + sin_a * y for x, y in poly]
    vs = [-sin_a * x + cos_a * y for x, y in poly]

    cx = (cos_a * region.bbox.x0 + cos_a * region.bbox.x1) / 2 + \
         (sin_a * region.bbox.y0 + sin_a * region.bbox.y1) / 2
    cy = (-sin_a * region.bbox.x0 - sin_a * region.bbox.x1) / 2 + \
         (cos_a * region.bbox.y0 + cos_a * region.bbox.y1) / 2

    cx = sum(cos_a * x + sin_a * y for x, y in poly) / 4
    cy = sum(-sin_a * x + cos_a * y for x, y in poly) / 4

    half_para = (max(us) - min(us)) / 2
    half_perp = (max(vs) - min(vs)) / 2
    return cx, cy, half_para, half_perp, angle_rad


def _angle_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two angles in degrees."""
    diff = abs(a - b) % 180
    return min(diff, 180 - diff)


def _same_block(a: Region, b: Region) -> bool:
    """True if a and b look like consecutive lines of the same text block."""
    if _angle_diff(a.angle, b.angle) > ANGLE_TOLERANCE:
        return False

    # Work in the local frame of the region with the larger area (more reliable OBB)
    ref = a if a.bbox.width * a.bbox.height >= b.bbox.width * b.bbox.height else b
    angle_rad = math.radians(ref.angle)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    def project(r: Region):
        cx = sum(x for x, _ in r.polygon) / 4
        cy = sum(y for _, y in r.polygon) / 4
        us = [ cos_a * x + sin_a * y for x, y in r.polygon]
        vs = [-sin_a * x + cos_a * y for x, y in r.polygon]
        return (
            min(us), max(us),   # parallel extent
            min(vs), max(vs),   # perp extent
            cx * cos_a + cy * sin_a,  # parallel centre
        )

    a_u0, a_u1, a_v0, a_v1, _ = project(a)
    b_u0, b_u1, b_v0, b_v1, _ = project(b)

    # Parallel spans must overlap (or touch)
    overlap = min(a_u1, b_u1) - max(a_u0, b_u0)
    if overlap < PARA_OVERLAP_MIN:
        return False

    # Perpendicular gap must be small relative to the shorter region's height
    a_h = a_v1 - a_v0
    b_h = b_v1 - b_v0
    shorter_h = min(a_h, b_h)
    perp_gap = max(a_v0 - b_v1, b_v0 - a_v1, 0)
    return perp_gap <= shorter_h * PERP_GAP_FACTOR


# ---------------------------------------------------------------------------
# Union-find
# ---------------------------------------------------------------------------

def _make_uf(n: int):
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    return find, union


# ---------------------------------------------------------------------------
# Reading-order sort within a cluster
# ---------------------------------------------------------------------------

def _reading_order_key(r: Region) -> tuple[float, float]:
    """Sort key: primary = position along text direction, secondary = perp."""
    angle_rad = math.radians(r.angle)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
    cx = sum(x for x, _ in r.polygon) / 4
    cy = sum(y for _, y in r.polygon) / 4
    para  =  cos_a * cx + sin_a * cy
    perp  = -sin_a * cx + cos_a * cy
    return perp, para   # perp first: new lines, then left-to-right within line


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def cluster_regions(regions: list[Region]) -> list[Region]:
    """Assign cluster_id and merged_text to each region."""
    if not regions:
        return regions

    find, union = _make_uf(len(regions))

    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            if _same_block(regions[i], regions[j]):
                union(i, j)

    for i, r in enumerate(regions):
        r.cluster_id = f"p{r.page}_c{find(i):04d}"

    # Build merged_text per cluster
    clusters: dict[str, list[Region]] = {}
    for r in regions:
        clusters.setdefault(r.cluster_id, []).append(r)

    for members in clusters.values():
        sorted_members = sorted(members, key=_reading_order_key)
        joined = "\n".join(m.text for m in sorted_members if m.text)
        for m in members:
            m.merged_text = joined

    return regions
