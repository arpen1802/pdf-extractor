"""Stages 2 + 4: PaddleOCR (3.x) detection + recognition.

We run detect+recognize in one shot for the deterministic skeleton; the
agent stage (later) can re-recognize specific crops at higher zoom for
low-confidence regions.
"""
import math
from pathlib import Path

from .models import BBox, Region


def _polygon_angle(polygon: list[tuple[int, int]]) -> float:
    """Return the text-line angle in degrees from a 4-point polygon.

    PaddleOCR orders points clockwise from the top-left corner, so the
    bottom edge (p0 → p1) runs along the text baseline. We use that edge
    to determine orientation, then normalise to (-90, 90] so that both
    left-to-right and right-to-left reads of the same physical line get
    the same angle bucket.
    """
    (x0, y0), (x1, y1) = polygon[0], polygon[1]
    angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
    # Fold into (-90, 90] so 270° vertical == -90° vertical
    if angle > 90:
        angle -= 180
    elif angle <= -90:
        angle += 180
    return angle

_paddle = None


def _get_paddle(use_server_models: bool = False):
    """Lazy-init PaddleOCR. Mobile models are far faster on CPU; server
    models give better accuracy but are slow without a GPU.
    """
    global _paddle
    if _paddle is None:
        from paddleocr import PaddleOCR

        kwargs = dict(
            lang="en",
            use_textline_orientation=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
        )
        if not use_server_models:
            kwargs["text_detection_model_name"] = "PP-OCRv5_mobile_det"
            kwargs["text_recognition_model_name"] = "en_PP-OCRv5_mobile_rec"
        _paddle = PaddleOCR(**kwargs)
    return _paddle


def detect_and_recognize(image_path: Path, page_num: int) -> list[Region]:
    paddle = _get_paddle()
    result = paddle.predict(str(image_path))

    regions: list[Region] = []
    if not result:
        return regions

    page_result = result[0]
    texts: list[str] = list(page_result.get("rec_texts", []))
    scores: list[float] = list(page_result.get("rec_scores", []))
    polys = list(page_result.get("rec_polys", []))

    for i, (text, score, poly) in enumerate(zip(texts, scores, polys)):
        # poly is a numpy array of shape (4, 2)
        polygon = [(int(p[0]), int(p[1])) for p in poly]
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        bbox = BBox(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))
        regions.append(
            Region(
                id=f"p{page_num}_r{i:04d}",
                page=page_num,
                bbox=bbox,
                polygon=polygon,
                angle=_polygon_angle(polygon),
                text=text,
                confidence=float(score),
            )
        )
    return regions
