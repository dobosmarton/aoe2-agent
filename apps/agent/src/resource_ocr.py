"""Local, LLM-free resource-bar reader.

Reads ``food / wood / gold / stone / population`` (+ ``age``, best-effort) from an
AoE2:DE screenshot **without** calling an LLM — the strategist's source of truth
for the HUD, which replaced Claude vision. The output dict uses the same keys as
``StrategistProvider``'s ``ResourceReadings`` and scores directly against
``gameplay_agent.strategist_eval.evaluate_resource_readings``.

Backends (``read_resource_bar(..., backend=...)``)
--------------------------------------------------
- ``"rapidocr"`` (production): PaddleOCR models on onnxruntime — pip-only, no
  system binary. Used by the live strategist.
- ``"template"``: OpenCV NCC against per-digit glyph crops; needs only ``opencv``
  plus per-resolution templates. Useful with no OCR engine, and as a
  lone-single-digit fallback for the engine backends.
- ``"tesseract"`` (optional): ``pytesseract`` digit-whitelist OCR; needs the
  Tesseract binary.

Field geometry is resolution-specific and supplied two ways, in precedence order:
``autodetect_calibration`` localizes the bar from the live frame at runtime
(resolution-independent, no assets needed), and a hand-tuned
``resource_ocr_assets/calibration.<W>x<H>.yaml`` overrides it when present (see
``resource_ocr_assets/README.md``). Geometry is data, never hardcoded.

On-screen field order is Wood → Food → Gold → Stone → Population
(``prompts/strategist.md``), which differs from the ResourceReadings field order.

Run ``python -m gameplay_agent.resource_ocr --selftest`` to verify the OCR
machinery with synthesized digits (no real screenshots needed).
"""

from __future__ import annotations

import io
import statistics
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NamedTuple, NotRequired, TypedDict, cast

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Sequence

    from PIL import ImageFont
    from rapidocr_onnxruntime import RapidOCR

    # RapidOCR.__call__ is untyped; this is the structural shape we consume.
    _OcrLine = tuple[Sequence[Sequence[float]], str, float]  # (quad, text, score)
    _OcrResult = list[_OcrLine]

# On-screen left-to-right order of the four resource counters + population.
RESOURCE_FIELDS: tuple[str, ...] = ("wood", "food", "gold", "stone")
POP_FIELD = "population"
# Canonical glyph size every segmented digit / template is resized to before NCC.
_GLYPH_HW: tuple[int, int] = (28, 20)

Backend = Literal["template", "tesseract", "rapidocr"]

# Per-field character sets — used as the Tesseract whitelist and to filter the
# RapidOCR output (which has no whitelist option).
_DIGITS = "0123456789"
_DIGITS_SLASH = "0123456789/"
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz "


def _filter_charset(text: str, charset: str) -> str:
    return "".join(c for c in text if c in charset)


# ---------------------------------------------------------------------------
# Calibration (resolution-specific; loaded from YAML, never hardcoded)
# ---------------------------------------------------------------------------


@dataclass
class FieldBox:
    """Pixel box (inclusive-exclusive) of one reading in the full screenshot."""

    x0: int
    y0: int
    x1: int
    y1: int

    def crop(self, img: np.ndarray) -> np.ndarray:
        return cast("np.ndarray", img[self.y0 : self.y1, self.x0 : self.x1])


class _CalibrationData(TypedDict):
    """Shape of a calibration YAML — the boundary type for ``yaml.safe_load``."""

    width: int
    height: int
    fields: dict[str, list[int]]
    template_dir: NotRequired[str]


@dataclass
class Calibration:
    """Where each reading sits, for a single capture resolution.

    `fields` maps a field name (wood/food/gold/stone/population, optionally age)
    to its FieldBox. `template_dir` holds `0.png`..`9.png` and `slash.png`
    glyph crops taken from a real screenshot at this resolution.
    """

    width: int
    height: int
    fields: dict[str, FieldBox]
    template_dir: Path

    def field_rects(self) -> dict[str, tuple[int, int, int, int]]:
        """Per-field ``(x0, y0, x1, y1)`` rectangles in screenshot pixels.

        A plain-tuple view of ``fields`` so callers (e.g. the debug overlay) can
        draw the reading regions without importing ``FieldBox``.
        """
        return {name: (box.x0, box.y0, box.x1, box.y1) for name, box in self.fields.items()}

    @classmethod
    def from_yaml(cls, path: str | Path) -> Calibration:
        import yaml

        path = Path(path)
        data = cast("_CalibrationData", yaml.safe_load(path.read_text()))
        fields = {
            name: FieldBox(int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            for name, box in data["fields"].items()
        }
        tdir = Path(data.get("template_dir", "templates"))
        if not tdir.is_absolute():
            tdir = path.parent / tdir
        return cls(
            width=int(data["width"]),
            height=int(data["height"]),
            fields=fields,
            template_dir=tdir,
        )


ASSETS_DIR = Path(__file__).parent / "resource_ocr_assets"


def calibration_for(width: int, height: int) -> Calibration | None:
    """Load the calibration matching a capture resolution, or None if absent.

    Lets callers (e.g. the strategist) auto-select the right per-resolution
    calibration from ``resource_ocr_assets/calibration.<W>x<H>.yaml``.
    """
    path = ASSETS_DIR / f"calibration.{width}x{height}.yaml"
    return Calibration.from_yaml(path) if path.exists() else None


# ---------------------------------------------------------------------------
# Image helpers (kept dependency-light; cv2 imported lazily so import works
# even where OpenCV is absent — only the template backend needs it)
# ---------------------------------------------------------------------------


def _decode_gray(screenshot_bytes: bytes) -> np.ndarray:
    """Decode JPEG/PNG bytes to a grayscale uint8 array."""
    img = Image.open(io.BytesIO(screenshot_bytes)).convert("L")
    return np.asarray(img, dtype=np.uint8)


def _decode_rgb(screenshot_bytes: bytes) -> np.ndarray:
    """Decode JPEG/PNG bytes to an RGB uint8 array (RapidOCR detection input)."""
    img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _binarize_digits(gray: np.ndarray) -> np.ndarray:
    """Threshold so digits are white (255) on black (0).

    AoE2 resource numbers are light glyphs on a dark bar. We Otsu-threshold and,
    if the result came out mostly white (dark text on light), invert so the
    foreground is always the digits.
    """
    import cv2

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if binary.mean() > 127:  # foreground should be the minority (the digits)
        binary = cv2.bitwise_not(binary)
    return binary


def _normalize_glyph(glyph: np.ndarray) -> np.ndarray:
    """Tight-crop the foreground, then aspect-preserving-resize into _GLYPH_HW.

    Aspect preservation keeps '1' narrow and '0' round; plain stretching to a
    fixed box destroys that, which is exactly the cue NCC needs to separate
    digits on a fixed font. Applied identically to templates and candidates.
    """
    import cv2

    g = glyph if glyph.dtype == np.uint8 else cast("np.ndarray", glyph.astype(np.uint8))
    ys, xs = np.where(g > 0)
    if ys.size == 0:
        return np.zeros(_GLYPH_HW, dtype=np.uint8)
    ymin, ymax = cast("int", ys.min()), cast("int", ys.max())
    xmin, xmax = cast("int", xs.min()), cast("int", xs.max())
    g = cast("np.ndarray", g[ymin : ymax + 1, xmin : xmax + 1])
    out_h, out_w = _GLYPH_HW
    h, w = cast("tuple[int, int]", g.shape)
    scale = min(out_h / h, out_w / w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    resized = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros(_GLYPH_HW, dtype=np.uint8)
    y0, x0 = (out_h - nh) // 2, (out_w - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-mean normalized cross-correlation of two equal-size arrays."""
    af = cast("np.ndarray", a.astype(np.float32).ravel())
    bf = cast("np.ndarray", b.astype(np.float32).ravel())
    af -= cast("float", af.mean())
    bf -= cast("float", bf.mean())
    denom = cast("float", np.linalg.norm(af)) * cast("float", np.linalg.norm(bf))
    if denom == 0.0:
        return 0.0
    return cast("float", np.dot(af, bf)) / denom


# ---------------------------------------------------------------------------
# Template backend
# ---------------------------------------------------------------------------


def load_templates(template_dir: Path, *, include_slash: bool) -> dict[str, np.ndarray]:
    """Load digit (+ optional slash) glyph templates, binarized and resized."""
    chars = [str(d) for d in range(10)]
    paths = {c: template_dir / f"{c}.png" for c in chars}
    if include_slash:
        paths["/"] = template_dir / "slash.png"
    templates: dict[str, np.ndarray] = {}
    for char, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"missing glyph template: {p}")
        gray = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
        templates[char] = _normalize_glyph(_binarize_digits(gray))
    return templates


def _segment_glyphs(field_binary: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Connected-component digit segmentation, returned left-to-right.

    Returns (x, glyph_binary) pairs. Filters specks by height relative to the
    field so noise/anti-aliasing doesn't become phantom digits.
    """
    import cv2

    n, _labels, stats, _centroids = cast(
        "tuple[int, np.ndarray, np.ndarray, np.ndarray]",
        cv2.connectedComponentsWithStats(field_binary),
    )
    field_h = cast("int", field_binary.shape[0])
    glyphs: list[tuple[int, np.ndarray]] = []
    for i in range(1, n):  # skip background label 0
        stat_row = cast("np.ndarray", stats[i])  # [x, y, w, h, area]
        x, y, w, h, area = (int(v) for v in cast("list[int]", stat_row.tolist()))
        if h < 0.3 * field_h or area < 6:
            continue
        glyph = cast("np.ndarray", field_binary[y : y + h, x : x + w])
        glyphs.append((x, _normalize_glyph(glyph)))
    glyphs.sort(key=lambda g: g[0])
    return glyphs


def _classify(glyph: np.ndarray, templates: dict[str, np.ndarray]) -> str:
    best_char, best_score = "", -2.0
    for char, tmpl in templates.items():
        score = _ncc(glyph, tmpl)
        if score > best_score:
            best_char, best_score = char, score
    return best_char


def _read_field(field_img: np.ndarray, templates: dict[str, np.ndarray]) -> str:
    binary = _binarize_digits(field_img)
    glyphs = _segment_glyphs(binary)
    return "".join(_classify(g, templates) for _x, g in glyphs)


# ---------------------------------------------------------------------------
# OCR-engine backends (game glyphs are white-on-dark; both engines want
# dark-on-light with margin, so binarize → invert → pad → upscale first)
# ---------------------------------------------------------------------------


def _preprocess_for_ocr(field_img: np.ndarray, *, pad: int, binarize: bool = True) -> Image.Image:
    """Prep a field crop for an OCR engine.

    Digits → binarize/invert/pad (clean, tuned for tiny white-on-dark glyphs).
    Larger text (age) → raw grayscale upscale: the engines read it more reliably
    than an over-thresholded binarization, which corrupts the bigger letters.
    """
    import cv2

    if binarize:
        proc = cast("np.ndarray", cv2.bitwise_not(_binarize_digits(field_img)))  # black on white
        proc = cast(
            "np.ndarray",
            cv2.copyMakeBorder(proc, pad, pad, pad + 10, pad + 10, cv2.BORDER_CONSTANT, value=255),
        )
        scale = 4
    else:
        proc = field_img  # raw grayscale; engines handle light-on-dark text fine
        scale = 3
    pil = Image.fromarray(proc)
    return pil.resize((pil.width * scale, pil.height * scale), Image.Resampling.LANCZOS)


def _read_field_tesseract(field_img: np.ndarray, *, whitelist: str, binarize: bool = True) -> str:
    import pytesseract

    pil = _preprocess_for_ocr(field_img, pad=16, binarize=binarize)
    # psm 7 (one text line) handles multi-glyph fields; it can return empty on a
    # lone single glyph, so fall back to psm 10 (single character).
    for psm in (7, 10):
        text = pytesseract.image_to_string(
            pil, config=f"--psm {psm} -c tessedit_char_whitelist={whitelist}"
        )
        cleaned = _filter_charset(text, whitelist)
        if cleaned:
            return cleaned
    return ""


_RAPIDOCR_ENGINE = None


def _rapidocr_engine() -> RapidOCR:
    """Lazily build the RapidOCR engine (expensive init — reuse across calls)."""
    global _RAPIDOCR_ENGINE
    if _RAPIDOCR_ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR

        _RAPIDOCR_ENGINE = RapidOCR()
    return _RAPIDOCR_ENGINE


def _read_field_rapidocr(field_img: np.ndarray, *, whitelist: str, binarize: bool = True) -> str:
    """RapidOCR (PaddleOCR models on onnxruntime) — pip-only, no system binary.
    RapidOCR has no whitelist, so we filter its recognized text to `whitelist`.
    """
    pil = _preprocess_for_ocr(field_img, pad=20, binarize=binarize).convert("RGB")
    # RapidOCR.__call__ is untyped — cast to the structural result shape we rely on.
    raw, _elapse = cast("tuple[_OcrResult | None, object]", _rapidocr_engine()(np.asarray(pil)))
    if not raw:
        return ""
    return _filter_charset("".join(line[1] for line in raw), whitelist)


_AGE_KEYWORDS = (
    ("imperial", "Imperial Age"),
    ("castle", "Castle Age"),
    ("feudal", "Feudal Age"),
    ("dark", "Dark Age"),
)


def _map_age(text: str) -> str:
    """Map (possibly noisy) OCR'd age text to a canonical age by keyword, else ""."""
    low = text.lower()
    for keyword, canonical in _AGE_KEYWORDS:
        if keyword in low:
            return canonical
    return ""


# ---------------------------------------------------------------------------
# Runtime auto-calibration: localize the resource bar in ONE live frame.
#
# These detection helpers (``_bbox/_detect/_extract/_column_centers/_assign``)
# are the same logic the offline ``scripts/calibrate_resource_bar.py`` uses to
# build a per-resolution YAML — they live here so the live agent and the CLI run
# one implementation. ``autodetect_calibration`` runs them on a single frame to
# synthesize a ``Calibration`` on the fly, so a brand-new capture resolution
# needs no hand-made YAML or templates (the production ``rapidocr`` backend reads
# fine with zero templates).
# ---------------------------------------------------------------------------


class Box(NamedTuple):
    """Axis-aligned pixel box (inclusive-exclusive) from text detection."""

    x0: int
    y0: int
    x1: int
    y1: int


def _bbox(quad: Sequence[Sequence[float]]) -> Box:
    """Axis-aligned bbox of a RapidOCR quad (four [x, y] points)."""
    xs = [p[0] for p in quad]
    ys = [p[1] for p in quad]
    return Box(int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def _detect(engine: RapidOCR, image: np.ndarray, top_frac: float) -> list[tuple[Box, str]]:
    """RapidOCR detections in the top band, returned as (bbox, text) full-frame."""
    band_h = int(cast("int", image.shape[0]) * top_frac)
    band = cast("np.ndarray", image[:band_h, :])
    # RapidOCR.__call__ is untyped — cast to the structural result shape we rely on.
    raw, _elapse = cast("tuple[_OcrResult | None, object]", engine(band))
    dets: list[tuple[Box, str]] = []
    for quad, text, _score in raw or []:
        dets.append((_bbox(quad), "".join(text.split())))
    return dets


def _extract(dets: list[tuple[Box, str]]) -> tuple[list[Box], Box | None, Box | None]:
    """Per-frame: (main-row numeric boxes, population box, age box).

    Population (the only "/"-bearing field) anchors the main row's y; resources
    share it and villager sub-counts sit lower → excluded. Numerics are returned
    UNASSIGNED — assignment to wood/food/gold/stone happens by x-column, so a
    field missing in a frame doesn't shift the others.
    """
    numeric = [b for b, t in dets if t.isdigit()]
    pop_dets = [b for b, t in dets if "/" in t and t.replace("/", "").isdigit()]
    age = [b for b, t in dets if _map_age(t)]
    pop_box = max(pop_dets, key=lambda b: b.y1 - b.y0) if pop_dets else None
    if pop_box is not None:
        yc = (pop_box.y0 + pop_box.y1) / 2
        band = 0.5 * (pop_box.y1 - pop_box.y0)
        main = [b for b in numeric if abs((b.y0 + b.y1) / 2 - yc) <= band]
    else:
        hmax = max((b.y1 - b.y0 for b in numeric), default=0)
        main = [b for b in numeric if (b.y1 - b.y0) >= 0.8 * hmax]
    return main, pop_box, (age[0] if age else None)


def _column_centers(x0s: list[int], k: int = 4, gap: int = 70) -> list[int]:
    """Cluster numeric left-edges into the k resource columns (1D, split on gaps)."""
    if not x0s:
        return []
    xs = sorted(x0s)
    groups: list[list[int]] = [[xs[0]]]
    for x in xs[1:]:
        (groups.append([x]) if x - groups[-1][-1] > gap else groups[-1].append(x))
    groups.sort(key=len, reverse=True)  # keep the k most-populated columns
    return sorted(int(statistics.median(g)) for g in groups[:k])


def _assign(main: list[Box], centers: list[int], tol: float) -> dict[str, Box]:
    """Assign each main-row numeric to its nearest resource column (within tol)."""
    out: dict[str, Box] = {}
    if not centers:
        return out
    for b in main:
        idx = min(range(len(centers)), key=lambda j: abs(b.x0 - centers[j]))
        if idx < len(RESOURCE_FIELDS) and abs(b.x0 - centers[idx]) <= tol:
            name = RESOURCE_FIELDS[idx]
            if name not in out or abs(b.x0 - centers[idx]) < abs(out[name].x0 - centers[idx]):
                out[name] = b
    return out


_Y_TOP_PAD = 2  # tiny top margin; bottom is left tight to exclude the sub-count row


def _build_fields_single_frame(
    assigned: dict[str, Box],
    pop: Box | None,
    age: Box | None,
    *,
    frame_w: int,
    pad: int,
) -> dict[str, FieldBox]:
    """Build field boxes from ONE frame, hugging each detected number tightly.

    Boxes are deliberately TIGHT — RapidOCR grows unreliable with empty margin
    around small text (it invents trailing digits), so we don't reserve "growth
    room": each strategist tick re-detects, so the box always fits the current
    value. Three edges are anchored to avoid the surrounding UI:
      * left  = detected x0 (the resource/population icon sits just left of it),
      * bottom = detected y1, un-padded (the villager gather sub-count sits just
        below and to the right; a looser box reads its digits),
      * right = detected x1 + a few px, capped before the next field.
    Resource fields share one y-band so a 1-digit field reads like a 4-digit one.
    """
    present: dict[str, Box] = {n: assigned[n] for n in RESOURCE_FIELDS if n in assigned}
    if pop is not None:
        present[POP_FIELD] = pop
    order = [n for n in (*RESOURCE_FIELDS, POP_FIELD) if n in present]
    if not order:
        return {}
    row_boxes = [present[n] for n in order]
    y0 = max(0, min(b.y0 for b in row_boxes) - _Y_TOP_PAD)
    y1 = max(b.y1 for b in row_boxes)

    x0s = [present[n].x0 for n in order]
    gaps = [b - a for a, b in pairwise(x0s)]
    pitch = statistics.median(gaps) if gaps else 150
    safety = max(2, round(0.06 * pitch))

    fields: dict[str, FieldBox] = {}
    for i, name in enumerate(order):
        b = present[name]
        x0 = max(0, b.x0)
        x1 = b.x1 + pad
        if i + 1 < len(order):
            x1 = min(x1, present[order[i + 1]].x0 - safety)  # never reach the next field
        x1 = min(max(x1, b.x1), frame_w)
        fields[name] = FieldBox(x0, y0, x1, y1)

    if age is not None:  # age is large standalone text (no icon-digit / sub-count risk)
        fields["age"] = FieldBox(
            max(0, age.x0), max(0, age.y0 - pad), min(frame_w, age.x1 + pad), age.y1 + pad
        )
    return fields


def autodetect_calibration(
    screenshot_bytes: bytes,
    *,
    top_frac: float = 0.15,
    pad: int = 4,
) -> Calibration | None:
    """Derive a ``Calibration`` from ONE live frame via RapidOCR localization.

    Returns ``None`` when the bar can't be localized (no population box AND fewer
    than two resource columns) so the caller falls back to last-known state — the
    single most important guard against a hidden HUD / wrong layout / blank frame.
    ``template_dir`` points at the per-resolution templates dir when it exists,
    else a guaranteed-absent sentinel so ``read_resource_bar``'s template fallback
    is simply skipped (the ``rapidocr`` backend needs no templates).
    """
    try:
        rgb = _decode_rgb(screenshot_bytes)
    except Exception:  # any decode failure (truncated/empty bytes) → caller falls back
        return None
    h, w = cast("tuple[int, int]", rgb.shape[:2])
    dets = _detect(_rapidocr_engine(), rgb, top_frac)
    main, pop, age = _extract(dets)
    centers = _column_centers([b.x0 for b in main])
    pitch = (centers[-1] - centers[0]) / (len(centers) - 1) if len(centers) >= 2 else 150
    assigned = _assign(main, centers, 0.4 * pitch)
    if pop is None and len(assigned) < 2:  # acceptance gate
        return None
    fields = _build_fields_single_frame(assigned, pop, age, frame_w=w, pad=pad)
    if not fields:
        return None
    tdir = ASSETS_DIR / "templates" / f"{w}x{h}"
    if not tdir.exists():
        tdir = ASSETS_DIR / "templates" / "__autodetect_none__"  # absent → no template fallback
    return Calibration(width=w, height=h, fields=fields, template_dir=tdir)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def read_resource_bar(
    screenshot_bytes: bytes,
    calibration: Calibration,
    *,
    backend: Backend = "template",
) -> dict[str, object]:
    """Read the resource bar into a ResourceReadings-shaped dict.

    Returns keys ``food, wood, gold, stone`` (int), ``population`` (str "cur/max"),
    and ``age`` (str; "" if not calibrated / not read). Fields that fail to parse
    are omitted or left at a sentinel so the scorer reports a clear failure rather
    than a crash.
    """
    gray = _decode_gray(screenshot_bytes)
    # Load templates when available. Required for the template backend; also used
    # as a per-field fallback for the tesseract backend, which structurally can't
    # read a lone isolated digit (e.g. a single "1") that template matching gets.
    templates_num = templates_pop = None
    try:
        templates_num = load_templates(calibration.template_dir, include_slash=False)
        templates_pop = load_templates(calibration.template_dir, include_slash=True)
    except FileNotFoundError:
        if backend == "template":
            raise  # templates are mandatory for this backend

    def ocr(crop: np.ndarray, whitelist: str, *, binarize: bool = True) -> str:
        if backend == "rapidocr":
            return _read_field_rapidocr(crop, whitelist=whitelist, binarize=binarize)
        return _read_field_tesseract(crop, whitelist=whitelist, binarize=binarize)

    def read_num(crop: np.ndarray) -> str:
        if backend == "template":
            assert templates_num is not None
            return _read_field(crop, templates_num)
        digits = ocr(crop, _DIGITS)
        if not digits and templates_num is not None:  # engine empty → template fallback
            digits = _read_field(crop, templates_num)
        return digits

    def read_pop(crop: np.ndarray) -> str:
        if backend == "template":
            assert templates_pop is not None
            return _read_field(crop, templates_pop)
        raw = ocr(crop, _DIGITS_SLASH)
        if "/" not in raw and templates_pop is not None:
            raw = _read_field(crop, templates_pop)
        return raw

    out: dict[str, object] = {}

    for name in RESOURCE_FIELDS:
        box = calibration.fields.get(name)
        if box is None:
            continue
        digits = read_num(box.crop(gray))
        if digits.isdigit():
            out[name] = int(digits)

    pop_box = calibration.fields.get(POP_FIELD)
    if pop_box is not None:
        raw = read_pop(pop_box.crop(gray))
        if "/" in raw:
            out[POP_FIELD] = raw

    # Age is text ("Dark/Feudal/Castle/Imperial Age") — OCR it and keyword-map.
    # The pure-template backend has no OCR engine, so age is left "" there.
    age_box = calibration.fields.get("age")
    if age_box is not None and backend != "template":
        # Age is large text — OCR the raw crop (binarize corrupts the bigger letters).
        out["age"] = _map_age(ocr(age_box.crop(gray), _LETTERS, binarize=False))
    else:
        out["age"] = ""

    return out


# ---------------------------------------------------------------------------
# Self-test: synthesize digits with PIL and verify the template backend.
# Proves the machinery runs on the installed stack (cv2 + PIL) with NO real
# screenshots. Run: python -m gameplay_agent.resource_ocr --selftest
# ---------------------------------------------------------------------------


_SELFTEST_FONTS = (
    "/System/Library/Fonts/Supplemental/Arial.ttf",  # macOS
    "/Library/Fonts/Arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
    "DejaVuSans.ttf",
)


def _selftest_font(size: int = 32) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    from PIL import ImageFont

    for path in _SELFTEST_FONTS:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    try:  # Pillow >= 10 accepts a size for the built-in bitmap font
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _render_digit_image(text: str) -> np.ndarray:
    """Render `text` as white glyphs on black, tight-cropped to content.

    Tight-cropping makes the glyphs fill the field height regardless of which
    font is available, so segmentation's height filter behaves like it does on a
    real (digit-row-tight) resource-bar crop.
    """
    from PIL import ImageDraw

    font = _selftest_font()
    canvas = Image.new("L", (400, 80), color=0)
    ImageDraw.Draw(canvas).text((10, 10), text, fill=255, font=font)
    bbox = canvas.getbbox()  # non-zero (white text) region
    if bbox is None:
        return np.zeros((40, 40), dtype=np.uint8)
    x0, y0, x1, y1 = bbox
    pad = 3
    crop = canvas.crop((max(0, x0 - pad), max(0, y0 - pad), x1 + pad, y1 + pad))
    return np.asarray(crop, dtype=np.uint8)


def _selftest() -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp)
        # Render single-glyph templates 0-9 and slash from the same font.
        for d in range(10):
            Image.fromarray(_render_digit_image(str(d))).save(tdir / f"{d}.png")
        Image.fromarray(_render_digit_image("/")).save(tdir / "slash.png")
        templates_num = load_templates(tdir, include_slash=False)
        templates_pop = load_templates(tdir, include_slash=True)

        cases_num = ["0", "7", "42", "200", "1530"]
        cases_pop = ["8/15", "12/200"]
        ok = True
        for want in cases_num:
            got = _read_field(_render_digit_image(want), templates_num)
            mark = "ok " if got == want else "ERR"
            if got != want:
                ok = False
            print(f"  [{mark}] number  want={want!r:>8}  got={got!r}")
        for want in cases_pop:
            got = _read_field(_render_digit_image(want), templates_pop)
            mark = "ok " if got == want else "ERR"
            if got != want:
                ok = False
            print(f"  [{mark}] pop     want={want!r:>8}  got={got!r}")

    print("\nSELFTEST:", "PASS" if ok else "FAIL")
    print(
        "(This validates the segmentation+classification machinery on a synthetic\n"
        " font. Real-game accuracy still requires calibration + real screenshots —\n"
        " see resource_ocr_assets/README.md.)"
    )
    return 0 if ok else 1


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Local resource-bar OCR.")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Synthesize digits and verify the template backend (no screenshots needed).",
    )
    args = parser.parse_args()
    if cast("bool", args.selftest):
        return _selftest()
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
