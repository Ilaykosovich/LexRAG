from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PdfProcessor import ExtractItem

article_PART_RE = re.compile(r"\d+|\([^)]+\)")


def split_article_parts(article: str) -> List[str]:
    if not article:
        return []
    return article_PART_RE.findall(article)


def article_prefixes(article: str) -> List[str]:
    parts = split_article_parts(article)
    out = []
    cur = ""
    for p in parts:
        cur += p
        out.append(cur)
    return out


def stable_id(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.md5(raw).hexdigest()


@dataclass
class IndexDocument:
    doc_id: str
    text: str
    metadata: Dict[str, Any]


class LegalChunkBuilder:
    def __init__(
            self,
            include_lines: bool = False,
            include_tables: bool = True,
            max_group_chars: int = 420,
            group_overlap_docs: int = 1,
            merge_lines_to_paragraphs = True
    ):
        self.include_lines = include_lines
        self.include_tables = include_tables
        self.max_group_chars = max_group_chars
        self.group_overlap_docs = max(0, group_overlap_docs)
        self.merge_lines_to_paragraphs = merge_lines_to_paragraphs

    def build(self, items: List[Dict[str, Any]]) -> Tuple[List[IndexDocument], Dict[str, List[str]]]:
        atomic_docs = self._build_atomic_docs(items)
        grouped_docs = self._build_grouped_docs(atomic_docs)

        all_docs = atomic_docs + grouped_docs
        article_index = self._build_article_index(all_docs)

        return all_docs, article_index



    # -------------------------
    # Atomic docs
    # -------------------------

    def _build_atomic_docs(self, items: List[Dict[str, Any]]) -> List[IndexDocument]:
        docs: List[IndexDocument] = []

        sequence_no = 0
        sorted_items = sorted(
            items,
            key=lambda it: (
                it.get("source", ""),
                it.get("page", 0),
                (it.get("bbox")[1] if it.get("bbox") else 0.0),
                (it.get("bbox")[0] if it.get("bbox") else 0.0),
            ),
        )

        for it in sorted_items:
            typ = it.get("type")
            article = it.get("article")

            text_raw = (it.get("text") or "").strip()
            text_clean = (it.get("text_clean") or "").strip()

            if not text_raw:
                continue

            # для индексации можно использовать clean,
            # но если он пуст — брать raw
            text_for_index = text_clean if text_clean else text_raw

            if typ == "paragraph":
                if not article:
                    continue
                docs.append(
                    self._make_doc(
                        kind="atomic_paragraph",
                        text=text_for_index,
                        item=it,
                        sequence_no=sequence_no,
                        text_raw=text_raw,
                        text_clean=text_clean,
                    )
                )
                sequence_no += 1

            elif typ == "line" and self.include_lines:
                if not article:
                    continue
                docs.append(
                    self._make_doc(
                        kind="atomic_line",
                        text=text_for_index,
                        item=it,
                        sequence_no=sequence_no,
                        text_raw=text_raw,
                        text_clean=text_clean,
                    )
                )
                sequence_no += 1

            elif typ in {"table", "table_row"} and self.include_tables:
                docs.append(
                    self._make_doc(
                        kind=f"atomic_{typ}",
                        text=text_for_index,
                        item=it,
                        sequence_no=sequence_no,
                        text_raw=text_raw,
                        text_clean=text_clean,
                    )
                )
                sequence_no += 1

        return docs

    def _make_doc(
            self,
            kind: str,
            text: str,
            item: Dict[str, Any],
            sequence_no: int,
            text_raw: str,
            text_clean: str,
    ) -> IndexDocument:
        bbox = item.get("bbox")
        bbox_y0 = bbox[1] if bbox else None

        article = item.get("article")
        md = {
            "kind": kind,
            "source": item.get("source"),
            "page": item.get("page"),
            "bbox": bbox,
            "bbox_y0": bbox_y0,
            "type": item.get("type"),
            "article": article,
            "article_prefixes": article_prefixes(article or ""),
            "sequence_no": sequence_no,
            "text_raw": text_raw,
            "text_clean": text_clean,
            "meta": item.get("meta", {}),
        }

        doc_id = stable_id(
            {
                "kind": kind,
                "source": md["source"],
                "page": md["page"],
                "article": md["article"],
                "sequence_no": sequence_no,
                "text_raw": text_raw[:500],
            }
        )
        return IndexDocument(doc_id=doc_id, text=text, metadata=md)

    # -------------------------
    # Grouped docs
    # -------------------------

    def _build_grouped_docs(self, atomic_docs: List[IndexDocument]) -> List[IndexDocument]:
        """
        Объединяем только документы с одинаковым exact article.
        Разрешаем переход:
        - внутри той же страницы
        - на следующую страницу (page + 1)
        Запрещаем склейку через разрыв в 2+ страниц.

        Дополнительно поддерживаем overlap по atomic docs:
        следующий grouped chunk начинаетcя с последних N docs предыдущего chunk.
        """
        buckets: Dict[Tuple[str, str], List[IndexDocument]] = {}

        for doc in atomic_docs:
            article = doc.metadata.get("article")
            source = doc.metadata.get("source")
            kind = doc.metadata.get("kind")

            if not article:
                continue

            if kind not in {"atomic_paragraph", "atomic_line"}:
                continue

            key = (source, article)
            buckets.setdefault(key, []).append(doc)

        grouped: List[IndexDocument] = []

        for (source, article), docs in buckets.items():
            docs = sorted(
                docs,
                key=lambda d: (
                    d.metadata.get("page", 0),
                    d.metadata.get("sequence_no", 0),
                    d.metadata.get("bbox_y0", 0.0) or 0.0,
                ),
            )

            cur_docs: List[IndexDocument] = []
            cur_len = 0
            part_no = 1
            prev_page = None

            def docs_text_len(ds: List[IndexDocument]) -> int:
                total = 0
                for i, dd in enumerate(ds):
                    piece = dd.text.strip()
                    if not piece:
                        continue
                    total += len(piece)
                    if i > 0:
                        total += 1  # "\n" between docs in grouped text
                return total

            def flush_current() -> None:
                nonlocal cur_docs, cur_len, part_no
                if not cur_docs:
                    return

                grouped.append(
                    self._make_group_doc(
                        source=source,
                        article=article,
                        docs=cur_docs,
                        part_no=part_no,
                    )
                )
                part_no += 1

                if self.group_overlap_docs > 0:
                    cur_docs = cur_docs[-self.group_overlap_docs:]
                    cur_len = docs_text_len(cur_docs)
                else:
                    cur_docs = []
                    cur_len = 0

            for d in docs:
                piece = d.text.strip()
                if not piece:
                    continue

                page = d.metadata.get("page")

                # Если дыра между страницами > 1, разрываем цепочку и НЕ переносим overlap
                if cur_docs and prev_page is not None and page is not None:
                    if page - prev_page > 1:
                        grouped.append(
                            self._make_group_doc(
                                source=source,
                                article=article,
                                docs=cur_docs,
                                part_no=part_no,
                            )
                        )
                        part_no += 1
                        cur_docs = []
                        cur_len = 0

                add_len = len(piece) + (1 if cur_docs else 0)

                if cur_docs and (cur_len + add_len > self.max_group_chars):
                    flush_current()
                    add_len = len(piece) + (1 if cur_docs else 0)

                    # если overlap сам по себе уже почти забил лимит,
                    # выкидываем overlap и начинаем новый chunk чисто с текущего doc
                    if cur_docs and (cur_len + add_len > self.max_group_chars):
                        cur_docs = []
                        cur_len = 0
                        add_len = len(piece)

                cur_docs.append(d)
                cur_len += add_len
                prev_page = page

            if cur_docs:
                grouped.append(
                    self._make_group_doc(
                        source=source,
                        article=article,
                        docs=cur_docs,
                        part_no=part_no,
                    )
                )

        return grouped

    def _lines_to_paragraphs(self, lines: List[ExtractItem]) -> List[ExtractItem]:
        out: List[ExtractItem] = []

        by_page: Dict[int, List[ExtractItem]] = {}
        for ln in lines:
            if ln.type != "line":
                continue
            by_page.setdefault(ln.page, []).append(ln)

        for page_idx, page_lines in by_page.items():
            def key_fn(it: ExtractItem):
                x0, y0, x1, y1 = it.bbox or (0, 0, 0, 0)
                return (y0, x0)

            page_lines = sorted(page_lines, key=key_fn)

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

                xs0, ys0, xs1, ys1 = [], [], [], []
                for it in cur:
                    if it.bbox:
                        x0, y0, x1, y1 = it.bbox
                        xs0.append(x0)
                        ys0.append(y0)
                        xs1.append(x1)
                        ys1.append(y1)

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
                            "article": (cur[0].meta or {}).get("article"),
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
                cur_article = (it.meta or {}).get("article")

                if not cur:
                    cur = [it]
                    prev_bbox = it.bbox
                    prev_x0 = x0
                    continue

                px0, py0, px1, py1 = prev_bbox or (x0, y0, x1, y1)
                y_gap = y0 - py1
                x_shift = abs(x0 - (prev_x0 or x0))

                current_paragraph_article = (cur[0].meta or {}).get("article")
                article_changed = cur_article != current_paragraph_article

                if y_gap > y_gap_thresh or x_shift > 60 or article_changed:
                    flush()
                    cur = [it]
                else:
                    cur.append(it)

                prev_bbox = it.bbox
                prev_x0 = x0

            flush()

        return out

    def _make_group_doc(
        self,
        source: str,
        article: str,
        docs: List[IndexDocument],
        part_no: int,
    ) -> IndexDocument:
        text = "\n".join(d.text for d in docs if d.text.strip()).strip()

        pages = [d.metadata.get("page") for d in docs if d.metadata.get("page") is not None]
        seqs = [d.metadata.get("sequence_no") for d in docs if d.metadata.get("sequence_no") is not None]

        md = {
            "kind": "grouped_article_chunk",
            "source": source,
            "article": article,
            "article_prefixes": article_prefixes(article),
            "page_start": min(pages) if pages else None,
            "page_end": max(pages) if pages else None,
            "sequence_start": min(seqs) if seqs else None,
            "sequence_end": max(seqs) if seqs else None,
            "part_no": part_no,
            "child_doc_ids": [d.doc_id for d in docs],
        }

        doc_id = stable_id(
            {
                "kind": "grouped_article_chunk",
                "source": source,
                "article": article,
                "part_no": part_no,
                "child_doc_ids": md["child_doc_ids"],
            }
        )
        return IndexDocument(doc_id=doc_id, text=text, metadata=md)

    # -------------------------
    # article index
    # -------------------------

    def _build_article_index(self, docs: List[IndexDocument]) -> Dict[str, List[str]]:
        idx: Dict[str, List[str]] = {}

        for doc in docs:
            article = doc.metadata.get("article")
            if not article:
                continue

            for pref in article_prefixes(article):
                idx.setdefault(pref, []).append(doc.doc_id)

        return idx