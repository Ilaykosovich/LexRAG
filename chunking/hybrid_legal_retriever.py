from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi


@dataclass
class IndexDocument:
    doc_id: str
    text: str
    metadata: Dict[str, Any]


@dataclass
class SearchHit:
    doc_id: str
    score: float
    text: str
    metadata: Dict[str, Any]
    match_reasons: List[str]


article_EXACT_RE = re.compile(r"^\s*(\d+(?:\([A-Za-z0-9ivxlcdmIVXLCDM]+\))*)\s*$")
article_IN_TEXT_RE = re.compile(r"\b(\d+(?:\([A-Za-z0-9ivxlcdmIVXLCDM]+\))*)\b")


def normalize_article(article: str) -> str:
    return re.sub(r"\s+", "", article or "").strip()


def split_article_parts(article: str) -> List[str]:
    return re.findall(r"\d+|\([A-Za-z0-9ivxlcdmIVXLCDM]+\)", normalize_article(article))


def article_prefixes(article: str) -> List[str]:
    parts = split_article_parts(article)
    out: List[str] = []
    cur = ""
    for p in parts:
        cur += p
        out.append(cur)
    return out


def looks_like_article_query(query: str) -> bool:
    return bool(article_EXACT_RE.match(normalize_article(query)))


def extract_article_candidates(query: str) -> List[str]:
    found = [normalize_article(m.group(1)) for m in article_IN_TEXT_RE.finditer(query or "")]
    out: List[str] = []
    seen = set()

    for addr in found:
        for pref in reversed(article_prefixes(addr)):
            if pref not in seen:
                seen.add(pref)
                out.append(pref)

    return out


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zа-яё0-9]+|\([a-zа-яё0-9]+\)", (text or "").lower())


@dataclass
class BM25Index:
    bm25: Optional[BM25Okapi]
    doc_ids: List[str]

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        if self.bm25 is None:
            return []

        tokens = simple_tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        pairs = list(zip(self.doc_ids, scores))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:k]


def build_bm25_index(docs: List[IndexDocument]) -> BM25Index:
    tokenized = [(d.doc_id, simple_tokenize(d.text)) for d in docs]
    tokenized = [(doc_id, toks) for doc_id, toks in tokenized if toks]

    if not tokenized:
        return BM25Index(bm25=None, doc_ids=[])

    corpus = [toks for _, toks in tokenized]
    doc_ids = [doc_id for doc_id, _ in tokenized]

    bm25 = BM25Okapi(corpus)
    return BM25Index(bm25=bm25, doc_ids=doc_ids)



@dataclass
class VectorIndex:
    doc_ids: List[str]
    matrix: np.ndarray


def _l2_normalize_matrix(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def _l2_normalize_vector(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x)
    if norm == 0:
        return x
    return x / norm


def build_vector_index(
    docs: List[IndexDocument],
    embedding_function: Callable[[List[str]], List[List[float]]],
    vector_doc_filter: Optional[Callable[[IndexDocument], bool]] = None,
) -> Tuple[VectorIndex, Callable[[str, int], List[Tuple[str, float]]]]:
    if vector_doc_filter is None:
        def vector_doc_filter(doc: IndexDocument) -> bool:
            return doc.metadata.get("kind") in {"grouped_article_chunk", "atomic_paragraph", "atomic_line"}

    selected_docs = [d for d in docs if vector_doc_filter(d)]
    doc_ids = [d.doc_id for d in selected_docs]
    texts = [d.text for d in selected_docs]

    if not texts:
        empty = VectorIndex(doc_ids=[], matrix=np.zeros((0, 1), dtype=np.float32))

        def empty_search(query: str, k: int = 10) -> List[Tuple[str, float]]:
            return []

        return empty, empty_search

    matrix = np.array(embedding_function(texts), dtype=np.float32)
    matrix = _l2_normalize_matrix(matrix)
    vector_index = VectorIndex(doc_ids=doc_ids, matrix=matrix)

    def vector_search(query: str, k: int = 10) -> List[Tuple[str, float]]:
        q = np.array(embedding_function([query])[0], dtype=np.float32)
        q = _l2_normalize_vector(q)

        sims = vector_index.matrix @ q
        top_idx = np.argsort(-sims)[:k]
        return [(vector_index.doc_ids[i], float(sims[i])) for i in top_idx]

    return vector_index, vector_search


class HybridLegalRetriever:
    def __init__(
        self,
        doc_key: str,
        docs: List[IndexDocument],
        article_index: Dict[str, List[str]],
        bm25_index: BM25Index,
        vector_index: VectorIndex,
        vector_search_fn: Callable[[str, int], List[Tuple[str, float]]],
        *,
        bm25_weight: float = 0.35,
        vector_weight: float = 0.65,
        article_weight: float = 0.30,
        paragraph_weight: float  = 0.20



    ):
        self.doc_key = doc_key
        self.docs = docs
        self.doc_store = {d.doc_id: d for d in docs}
        self.article_index = article_index
        self.bm25_index = bm25_index
        self.vector_index = vector_index
        self.vector_search_fn = vector_search_fn
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight
        self.article_weight = article_weight
        self.paragraph_weight = paragraph_weight

    @classmethod
    def from_documents(
        cls,
        doc_key: str,
        docs: List[IndexDocument],
        article_index: Dict[str, List[str]],
        embedding_function: Callable[[List[str]], List[List[float]]],
        *,
        vector_doc_filter: Optional[Callable[[IndexDocument], bool]] = None,
        bm25_weight: float = 0.35,
        vector_weight: float = 0.35,
        article_weight: float = 0.30,
    ) -> "HybridLegalRetriever":
        bm25_index = build_bm25_index(docs)
        vector_index, vector_search_fn = build_vector_index(
            docs=docs,
            embedding_function=embedding_function,
            vector_doc_filter=vector_doc_filter,
        )
        return cls(
            doc_key = doc_key,
            docs=docs,
            article_index=article_index,
            bm25_index=bm25_index,
            vector_index=vector_index,
            vector_search_fn=vector_search_fn,
            bm25_weight=bm25_weight,
            vector_weight=vector_weight,
            article_weight=article_weight,
        )

    def search_exact_terms_in_docs(self, terms: List[str], k: int = 5) -> List[SearchHit]:
        terms_norm = [t.strip().lower() for t in terms if t and t.strip()]
        hits = []

        for doc in self.docs:
            text_norm = (doc.text or "").lower()
            matched = [t for t in terms_norm if t in text_norm]
            if not matched:
                continue

            score = float(len(matched))

            hits.append(
                SearchHit(
                    doc_id=self.doc_key,
                    score=score,
                    text=doc.text,
                    metadata=doc.metadata,
                    match_reasons=[f"exact_term:{t}" for t in matched],
                )
            )

        hits.sort(key=lambda x: x.score, reverse=True)
        hits = self._dedupe_by_text(hits)
        return hits[:k]

    def search_metadata_article(self, query: str) -> List[Tuple[IndexDocument, str]]:
        """
        Возвращает кандидатов только по metadata['article'].
        tuple = (doc, reason)
        reason: article_exact | article_prefix | article_in_text | article_prefix_in_text
        """
        results: List[Tuple[IndexDocument, str]] = []

        if looks_like_article_query(query):
            q = normalize_article(query)
            exact_ids = self.article_index.get(q, [])
            if exact_ids:
                return [(self.doc_store[i], "article_exact") for i in exact_ids if i in self.doc_store]

            for pref in reversed(article_prefixes(q)[:-1]):
                ids = self.article_index.get(pref, [])
                if ids:
                    return [(self.doc_store[i], "article_prefix") for i in ids if i in self.doc_store]

            return []

        for addr in extract_article_candidates(query):
            ids = self.article_index.get(addr, [])
            if ids:
                results.extend((self.doc_store[i], "article_in_text") for i in ids if i in self.doc_store)
            else:
                for pref in reversed(article_prefixes(addr)[:-1]):
                    pref_ids = self.article_index.get(pref, [])
                    if pref_ids:
                        results.extend((self.doc_store[i], "article_prefix_in_text") for i in pref_ids if i in self.doc_store)
                        break

        return self._dedupe_doc_reason(results)

    def _dedupe_by_text(self, hits: List[SearchHit]) -> List[SearchHit]:
        seen = set()
        out: List[SearchHit] = []

        for hit in hits:
            norm_text = re.sub(r"\s+", " ", (hit.text or "").strip()).lower()
            if not norm_text:
                continue
            if norm_text in seen:
                continue
            seen.add(norm_text)
            out.append(hit)

        return out

    def search_metadata_paragraph(self, query: str) -> List[Tuple[Any, str]]:
        q = (query or "").strip()
        if not q:
            return []

        q_norm = q.casefold()
        results: List[Tuple[Any, str]] = []

        for doc in self.doc_store.values():
            meta = doc.metadata or {}

            candidates = [
                meta.get("paragraph_title"),
                meta.get("parent_paragraph_title"),
                meta.get("title"),
            ]
            candidates = [str(x).strip() for x in candidates if x]

            matched_reason = None

            for val in candidates:
                v = val.casefold()

                if q_norm == v:
                    matched_reason = "paragraph_exact"
                    break
                elif v.startswith(q_norm):
                    matched_reason = "paragraph_prefix"
                elif q_norm in v and matched_reason is None:
                    matched_reason = "paragraph_in_text"

            if matched_reason:
                results.append((doc, matched_reason))

        return results

    def search(self, query: str, k: int = 10, candidate_multiplier: int = 5) -> List[SearchHit]:
        query = (query or "").strip()
        if not query:
            return []

        candidate_scores: Dict[str, Dict[str, Any]] = {}

        def ensure(doc_id: str) -> None:
            if doc_id not in candidate_scores:
                candidate_scores[doc_id] = {
                    "vector_raw": 0.0,
                    "article_raw": 0.0,
                    "paragraph_raw": 0.0,
                    "match_reasons": set(),
                }

        k1 = 3

        # 1) metadata article search FIRST
        meta_hits = self.search_metadata_article(query)
        for doc, reason in meta_hits:
            ensure(doc.doc_id)
            if reason == "article_exact":
                candidate_scores[doc.doc_id]["article_raw"] = k1*1.0
            elif reason == "article_prefix":
                candidate_scores[doc.doc_id]["article_raw"] = k1*0.8
            elif reason == "article_in_text":
                candidate_scores[doc.doc_id]["article_raw"] = k1*0.9
            elif reason == "article_prefix_in_text":
                candidate_scores[doc.doc_id]["article_raw"] = k1*0.65
            candidate_scores[doc.doc_id]["match_reasons"].add(reason)

        # 1.5) metadata paragraph title search
        paragraph_hits = self.search_metadata_paragraph(query)
        for doc, reason in paragraph_hits:
            ensure(doc.doc_id)
            if reason == "paragraph_exact":
                candidate_scores[doc.doc_id]["paragraph_raw"] = k1*1.0
            elif reason == "paragraph_prefix":
                candidate_scores[doc.doc_id]["paragraph_raw"] = k1*0.8
            elif reason == "paragraph_in_text":
                candidate_scores[doc.doc_id]["paragraph_raw"] = k1*0.7
            elif reason == "paragraph_fuzzy":
                candidate_scores[doc.doc_id]["paragraph_raw"] = k1*0.6
            candidate_scores[doc.doc_id]["match_reasons"].add(reason)

        # 2) Vector
        vector_hits = self.vector_search_fn(query, k=max(k * candidate_multiplier, 20))
        for doc_id, score in vector_hits:
            if doc_id not in self.doc_store:
                continue
            ensure(doc_id)
            candidate_scores[doc_id]["vector_raw"] = max(
                candidate_scores[doc_id]["vector_raw"],
                float(score),
            )
            candidate_scores[doc_id]["match_reasons"].add("vector")

        if not candidate_scores:
            return []



        vector_norm = self._minmax_norm(candidate_scores, "vector_raw")
        article_norm = self._minmax_norm(candidate_scores, "article_raw")
        paragraph_norm = self._minmax_norm(candidate_scores, "paragraph_raw")

        hits: List[SearchHit] = []
        query_is_article = looks_like_article_query(query)


        for doc_id, parts in candidate_scores.items():
            score = (
                    self.vector_weight * vector_norm.get(doc_id, 0.0) +
                    self.article_weight * article_norm.get(doc_id, 0.0) +
                    self.paragraph_weight * paragraph_norm.get(doc_id, 0.0)
            )

            doc = self.doc_store[doc_id]

            if doc.metadata.get("kind") == "grouped_article_chunk":
                score += 0.13

            if query_is_article and "article_exact" in parts["match_reasons"]:
                score += 0.10

            if "paragraph_exact" in parts["match_reasons"]:
                score += 0.08
            elif "paragraph_prefix" in parts["match_reasons"]:
                score += 0.04

            hits.append(
                SearchHit(
                    doc_id=self.doc_key,
                    score=float(score),
                    text=doc.text,
                    metadata=doc.metadata,
                    match_reasons=sorted(parts["match_reasons"]),
                )
            )

        hits.sort(key=lambda x: x.score, reverse=True)
        hits = self._dedupe_by_text(hits)
        return hits[:k]

    def search0(self, query: str, k: int = 10, candidate_multiplier: int = 5) -> List[SearchHit]:
        query = (query or "").strip()
        if not query:
            return []

        candidate_scores: Dict[str, Dict[str, Any]] = {}

        def ensure(doc_id: str) -> None:
            if doc_id not in candidate_scores:
                candidate_scores[doc_id] = {
                    "vector_raw": 0.0,
                    "article_raw": 0.0,
                    "match_reasons": set(),
                }

        # 1) metadata article search FIRST
        meta_hits = self.search_metadata_article(query)
        for doc, reason in meta_hits:
            ensure(doc.doc_id)
            if reason == "article_exact":
                candidate_scores[doc.doc_id]["article_raw"] = 1.0
            elif reason == "article_prefix":
                candidate_scores[doc.doc_id]["article_raw"] = 0.8
            elif reason == "article_in_text":
                candidate_scores[doc.doc_id]["article_raw"] = 0.9
            elif reason == "article_prefix_in_text":
                candidate_scores[doc.doc_id]["article_raw"] = 0.65
            candidate_scores[doc.doc_id]["match_reasons"].add(reason)

        # 2) Vector
        vector_hits = self.vector_search_fn(query, k=max(k * candidate_multiplier, 20))
        for doc_id, score in vector_hits:
            if doc_id not in self.doc_store:
                continue
            ensure(doc_id)
            candidate_scores[doc_id]["vector_raw"] = max(candidate_scores[doc_id]["vector_raw"], float(score))
            candidate_scores[doc_id]["match_reasons"].add("vector")

        if not candidate_scores:
            return []

        vector_norm = self._minmax_norm(candidate_scores, "vector_raw")
        article_norm = self._minmax_norm(candidate_scores, "article_raw")

        hits: List[SearchHit] = []
        query_is_article = looks_like_article_query(query)

        for doc_id, parts in candidate_scores.items():
            score = (
                    self.vector_weight * vector_norm.get(doc_id, 0.0) +
                    self.article_weight * article_norm.get(doc_id, 0.0)
            )

            doc = self.doc_store[doc_id]

            if doc.metadata.get("kind") == "grouped_article_chunk":
                score += 0.13

            if query_is_article and "article_exact" in parts["match_reasons"]:
                score += 0.10

            hits.append(
                SearchHit(
                    doc_id=self.doc_key,
                    score=float(score),
                    text=doc.text,
                    metadata=doc.metadata,
                    match_reasons=sorted(parts["match_reasons"]),
                )
            )

        hits.sort(key=lambda x: x.score, reverse=True)
        hits = self._dedupe_by_text(hits)
        return hits[:k]

    def _minmax_norm(self, candidate_scores: Dict[str, Dict[str, Any]], field: str) -> Dict[str, float]:
        values = [float(v.get(field, 0.0)) for v in candidate_scores.values()]
        if not values:
            return {}

        vmin, vmax = min(values), max(values)
        out: Dict[str, float] = {}

        for doc_id, parts in candidate_scores.items():
            raw = float(parts.get(field, 0.0))
            if math.isclose(vmin, vmax):
                out[doc_id] = 1.0 if raw > 0 else 0.0
            else:
                out[doc_id] = (raw - vmin) / (vmax - vmin)

        return out

    def _dedupe_doc_reason(self, items: List[Tuple[IndexDocument, str]]) -> List[Tuple[IndexDocument, str]]:
        priority = {
            "article_exact": 4,
            "article_in_text": 3,
            "article_prefix": 2,
            "article_prefix_in_text": 1,
        }
        best: Dict[str, Tuple[IndexDocument, str]] = {}

        for doc, reason in items:
            prev = best.get(doc.doc_id)
            if prev is None or priority[reason] > priority[prev[1]]:
                best[doc.doc_id] = (doc, reason)

        return list(best.values())