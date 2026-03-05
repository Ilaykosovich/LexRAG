from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import fitz  # PyMuPDF
import pdfplumber
import pandas as pd


CASE_RE = re.compile(r"\b(?:CFI|ARB|CA)\s*\d+/\d{4}\b")
LAW_RE = re.compile(r"\bLaw\s+No\.?\s*\d+\s*of\s*\d{4}\b", re.IGNORECASE)


def _norm_spaces(s: str) -> str:
    s = s.replace("\u00a0", " ")  # nbsp
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\s+\n", "\n", s)
    s = re.sub(r"\n\s+", "\n", s)
    return s.strip()


def _clean_line(s: str) -> str:
    # Склеиваем переносы внутри слова типа "infor-\nmation" -> "information"
    s = re.sub(r"(\w)-\s+(\w)", r"\1\2", s)
    s = s.replace("\n", " ")
    s = _norm_spaces(s)
    return s


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    v = sorted(values)
    n = len(v)
    return v[n // 2] if n % 2 == 1 else (v[n // 2 - 1] + v[n // 2]) / 2


@dataclass
class ExtractItem:
    type: str  # "line" | "paragraph" | "table"
    text: str
    source: str
    page: int  # 0-indexed
    bbox: Optional[Tuple[float, float, float, float]] = None
    meta: Optional[Dict[str, Any]] = None


class PDFProcessor:
    """
    Извлечение текста + layout (PyMuPDF) и таблиц (pdfplumber).
    Возвращает единый список объектов ExtractItem.
    """

    def __init__(
        self,
        merge_lines_to_paragraphs: bool = True,
        paragraph_y_gap_factor: float = 1.35,
        drop_short_lines: bool = True,
        min_line_chars: int = 2,
        extract_tables: bool = True,
        table_row_docs: bool = True,
        table_max_rows_for_markdown: int = 200,
    ):
        self.merge_lines_to_paragraphs = merge_lines_to_paragraphs
        self.paragraph_y_gap_factor = paragraph_y_gap_factor
        self.drop_short_lines = drop_short_lines
        self.min_line_chars = min_line_chars
        self.extract_tables = extract_tables
        self.table_row_docs = table_row_docs
        self.table_max_rows_for_markdown = table_max_rows_for_markdown

    # ---------- Public API ----------

    def process_pdf(self, pdf_path: Union[str, Path]) -> List[Dict[str, Any]]:
        pdf_path = Path(pdf_path)
        items: List[ExtractItem] = []

        # 1) Layout-aware lines via PyMuPDF
        lines = self._extract_lines_pymupdf(pdf_path)
        items.extend(lines)

        # 2) Paragraphs (optional)
        if self.merge_lines_to_paragraphs:
            paragraphs = self._lines_to_paragraphs(lines)
            items.extend(paragraphs)

        # 3) Tables via pdfplumber (optional)
        if self.extract_tables:
            tables = self._extract_tables_pdfplumber(pdf_path)
            items.extend(tables)

        # Convert dataclasses -> dicts
        return [self._item_to_dict(it) for it in items]

    def process_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = False,
        pattern: str = "*.pdf",
    ) -> List[Dict[str, Any]]:
        directory_path = Path(directory_path)
        files = directory_path.rglob(pattern) if recursive else directory_path.glob(pattern)

        all_items: List[Dict[str, Any]] = []
        for pdf_file in files:
            if not pdf_file.is_file():
                continue
            print(f"[PDFProcessor] Processing: {pdf_file.name}")
            all_items.extend(self.process_pdf(pdf_file))

        return all_items

    # ---------- PyMuPDF extraction ----------

    def _extract_lines_pymupdf(self, pdf_path: Path) -> List[ExtractItem]:
        doc = fitz.open(str(pdf_path))
        out: List[ExtractItem] = []

        for page_idx, page in enumerate(doc):
            d = page.get_text("dict")  # contains blocks -> lines -> spans
            for b in d.get("blocks", []):
                if b.get("type") != 0:  # 0=text
                    continue

                for line in b.get("lines", []):
                    spans = line.get("spans", [])
                    if not spans:
                        continue

                    raw_text = "".join(s.get("text", "") for s in spans)
                    text = _clean_line(raw_text)
                    if not text:
                        continue
                    if self.drop_short_lines and len(text) < self.min_line_chars:
                        continue

                    # bbox can be in line or computed from spans
                    bbox = tuple(line.get("bbox", spans[0].get("bbox")))  # type: ignore

                    sizes = [float(s.get("size", 0.0)) for s in spans if s.get("size") is not None]
                    fonts = [s.get("font") for s in spans if s.get("font")]

                    meta = {
                        "spans": [
                            {
                                "text": _clean_line(s.get("text", "")),
                                "bbox": tuple(s.get("bbox")) if s.get("bbox") else None,
                                "size": s.get("size"),
                                "font": s.get("font"),
                            }
                            for s in spans
                        ],
                        "font_size_median": _median(sizes),
                        "fonts": sorted(set(fonts)),
                        "anchors": self._anchors_from_text(text),
                    }

                    out.append(
                        ExtractItem(
                            type="line",
                            text=text,
                            source=pdf_path.name,
                            page=page_idx,
                            bbox=bbox,  # type: ignore
                            meta=meta,
                        )
                    )
        doc.close()
        return out

    def _lines_to_paragraphs(self, lines: List[ExtractItem]) -> List[ExtractItem]:
        """
        Склейка линий в параграфы на основе:
        - одинаковой страницы
        - близости по Y
        - похожего X0 (левый отступ)
        """
        out: List[ExtractItem] = []

        # Group by page
        by_page: Dict[int, List[ExtractItem]] = {}
        for ln in lines:
            if ln.type != "line":
                continue
            by_page.setdefault(ln.page, []).append(ln)

        for page_idx, page_lines in by_page.items():
            # sort by y0 then x0
            def key_fn(it: ExtractItem):
                x0, y0, x1, y1 = it.bbox or (0, 0, 0, 0)
                return (y0, x0)

            page_lines = sorted(page_lines, key=key_fn)

            # estimate baseline line height from font sizes
            font_sizes = []
            for it in page_lines:
                if it.meta and it.meta.get("font_size_median"):
                    font_sizes.append(float(it.meta["font_size_median"]))
            base = _median(font_sizes) or 10.0
            y_gap_thresh = base * self.paragraph_y_gap_factor

            cur: List[ExtractItem] = []

            def flush():
                nonlocal cur
                if not cur:
                    return
                text = " ".join(it.text for it in cur)
                text = _norm_spaces(text)
                # bbox union
                xs0, ys0, xs1, ys1 = [], [], [], []
                for it in cur:
                    if it.bbox:
                        x0, y0, x1, y1 = it.bbox
                        xs0.append(x0); ys0.append(y0); xs1.append(x1); ys1.append(y1)
                bbox = (min(xs0), min(ys0), max(xs1), max(ys1)) if xs0 else None

                anchors = self._anchors_from_text(text)
                out.append(
                    ExtractItem(
                        type="paragraph",
                        text=text,
                        source=cur[0].source,
                        page=page_idx,
                        bbox=bbox,
                        meta={
                            "line_count": len(cur),
                            "anchors": anchors,
                            "from": "pymupdf_lines",
                        },
                    )
                )
                cur = []

            prev_bbox = None
            prev_x0 = None

            for it in page_lines:
                if not it.bbox:
                    continue
                x0, y0, x1, y1 = it.bbox

                if not cur:
                    cur = [it]
                    prev_bbox = it.bbox
                    prev_x0 = x0
                    continue

                px0, py0, px1, py1 = prev_bbox or (x0, y0, x1, y1)

                y_gap = y0 - py1
                x_shift = abs(x0 - (prev_x0 or x0))

                # Heuristics: break paragraph if big vertical gap or big indent change
                if y_gap > y_gap_thresh or x_shift > 60:
                    flush()
                    cur = [it]
                else:
                    cur.append(it)

                prev_bbox = it.bbox
                prev_x0 = x0

            flush()

        return out

    # ---------- Table extraction ----------

    def _extract_tables_pdfplumber(self, pdf_path: Path) -> List[ExtractItem]:
        out: List[ExtractItem] = []

        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                # extract_tables returns list[list[list[str]]]
                tables = page.extract_tables()
                if not tables:
                    continue

                for t_idx, t in enumerate(tables):
                    # Build DataFrame
                    df = pd.DataFrame(t)
                    # If the first row looks like header, you can optionally set it as header.
                    # For contests it's safer to keep raw.

                    rows = df.fillna("").astype(str).values.tolist()

                    # markdown (limit rows to avoid huge chunks)
                    md = ""
                    if len(rows) <= self.table_max_rows_for_markdown:
                        try:
                            md = df.to_markdown(index=False)
                        except Exception:
                            md = "\n".join([" | ".join(r) for r in rows])

                    meta = {
                        "table_index": t_idx,
                        "n_rows": len(rows),
                        "n_cols": len(rows[0]) if rows else 0,
                        "anchors": self._anchors_from_rows(rows),
                        "from": "pdfplumber",
                    }

                    # Table as one doc
                    if md.strip():
                        out.append(
                            ExtractItem(
                                type="table",
                                text=md.strip(),
                                source=pdf_path.name,
                                page=page_idx,
                                bbox=None,  # pdfplumber table bbox не всегда даёт стабильно без extra settings
                                meta={**meta, "format": "markdown"},
                            )
                        )
                    else:
                        # fallback: stringify
                        txt = "\n".join([" | ".join(r) for r in rows])
                        out.append(
                            ExtractItem(
                                type="table",
                                text=txt.strip(),
                                source=pdf_path.name,
                                page=page_idx,
                                bbox=None,
                                meta={**meta, "format": "pipe_text"},
                            )
                        )

                    # Optional: index each row as separate doc (очень полезно для names/numbers)
                    if self.table_row_docs and rows:
                        for r_idx, r in enumerate(rows):
                            row_txt = " | ".join([c.strip() for c in r if c is not None])
                            row_txt = _norm_spaces(row_txt)
                            if not row_txt:
                                continue
                            out.append(
                                ExtractItem(
                                    type="table_row",
                                    text=row_txt,
                                    source=pdf_path.name,
                                    page=page_idx,
                                    bbox=None,
                                    meta={
                                        **meta,
                                        "row_index": r_idx,
                                        "format": "row_text",
                                        "anchors": self._anchors_from_text(row_txt),
                                    },
                                )
                            )

        return out

    # ---------- Anchors ----------

    def _anchors_from_text(self, text: str) -> Dict[str, List[str]]:
        cases = sorted(set(m.group(0).replace("  ", " ").strip() for m in CASE_RE.finditer(text)))
        laws = sorted(set(m.group(0).strip() for m in LAW_RE.finditer(text)))
        return {"cases": cases, "laws": laws}

    def _anchors_from_rows(self, rows: List[List[str]]) -> Dict[str, List[str]]:
        blob = " ".join(" ".join(r) for r in rows[:50])  # limit
        return self._anchors_from_text(blob)

    # ---------- Serialize ----------

    def _item_to_dict(self, it: ExtractItem) -> Dict[str, Any]:
        d = {
            "type": it.type,
            "text": it.text,
            "source": it.source,
            "page": it.page,
            "bbox": it.bbox,
            "meta": it.meta or {},
        }
        return d