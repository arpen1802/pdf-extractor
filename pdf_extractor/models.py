from typing import Optional
from pydantic import BaseModel


class BBox(BaseModel):
    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    def pad(self, px: int, max_w: int, max_h: int) -> "BBox":
        return BBox(
            x0=max(0, self.x0 - px),
            y0=max(0, self.y0 - px),
            x1=min(max_w, self.x1 + px),
            y1=min(max_h, self.y1 + px),
        )


class Region(BaseModel):
    id: str
    page: int
    bbox: BBox
    polygon: list[tuple[int, int]] = []
    text: Optional[str] = None
    confidence: Optional[float] = None
    cluster_id: Optional[str] = None
    needs_agent: bool = False
