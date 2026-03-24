import json
import math
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List
from RouteQuestion import RouteQuestion

class DocumentRouter:
    """
    Лексический router/ranker документов по первым N токенам.

    Что делает:
    1. Загружает block-level records из JSON
    2. Группирует их по file_name
    3. Для каждого файла собирает текст, берет первые token_limit токенов
    4. Строит:
       - unigram document frequency
       - bigram document frequency
    5. Для вопроса считает score документа:
       score =
           unigram_weight * sum(idf(token) for token in matched_unigrams)
           + bigram_weight * sum(idf(bigram) for bigram in matched_bigrams)

    Ожидаемый формат records:
    [
      {
        "file_name": "...pdf",
        "page_number": 1,
        "content_type": "text_block",
        "text": "...",
        "metadata": {
            "item_index": 1
        }
      },
      ...
    ]
    """

    def __init__(
        self,
        token_limit: int = 50,
        unigram_weight: float = 1.0,
        bigram_weight: float = 3.0,
    ):
        self.token_limit = token_limit
        self.unigram_weight = unigram_weight
        self.bigram_weight = bigram_weight

        self.documents: List[Dict[str, Any]] = []
        self.unigram_df: Dict[str, int] = {}
        self.bigram_df: Dict[str, int] = {}
        self.num_documents: int = 0
        self.stopwords = {
    "the", "of", "in", "to", "a", "an", "and", "or", "by", "for",
    "on", "at", "with", "from"
}

    @staticmethod
    def load_json(input_path: str | Path) -> List[Dict[str, Any]]:
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Файл не найден: {input_path}")

        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Ожидался список records в JSON: {input_path}")

        return data

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.lower()
        text = text.replace("\u00a0", " ")
        # сохраняем / чтобы не ломать токены вроде 005/2025
        text = re.sub(r"[^\w\s/]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [tok for tok in text.split() if tok]

    @staticmethod
    def _build_bigrams(tokens: List[str]) -> List[str]:
        return [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]

    @staticmethod
    def _idf(df: int, n_docs: int) -> float:
        return math.log((n_docs + 1) / (df + 1)) + 1.0

    def _group_records_by_file(self, records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for record in records:
            file_name = record.get("source")
            if not file_name:
                continue
            grouped[file_name].append(record)

        return grouped

    def _build_document_representation(
            self,
            file_name: str,
            file_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Строит представление одного документа по блокам первой страницы.
        """

        if not isinstance(file_records, list) or not all(isinstance(record, dict) for record in file_records):
            raise ValueError("Ожидается список словарей (file_records) для одного документа.")

        first_page_records = [
            record for record in file_records
            if record.get("page") == 1 or record.get("page") == 0
        ]

        # sorted_records = sorted(
        #     first_page_records,
        #     key=lambda r: r.get("metadata", {}).get("item_index", 10 ** 9),
        # )

        merged_text_parts = []
        for rec in first_page_records:
            text = rec.get("text", "")
            if text:
                merged_text_parts.append(text)

        merged_text = " ".join(merged_text_parts).strip()
        normalized = self._normalize_text(merged_text)

        tokens = self._tokenize(normalized)[: self.token_limit]
        filtered_tokens = self._filter_stopwords(tokens)

        unigrams = filtered_tokens[:]
        bigrams = self._build_bigrams(filtered_tokens)

        return {
            "file_name": file_name,
            f"tokens_{self.token_limit}": tokens,
            f"unigrams_{self.token_limit}": sorted(set(unigrams)),
            f"bigrams_{self.token_limit}": sorted(set(bigrams)),
        }

    def _filter_stopwords(self, tokens: List[str]) -> List[str]:
        return [tok for tok in tokens if tok not in self.stopwords]




    def build_stats(self, records: List[Dict[str, Any]]) -> None:
        grouped = self._group_records_by_file(records)

        documents: List[Dict[str, Any]] = []
        unigram_df = Counter()
        bigram_df = Counter()

        for file_name, file_records in grouped.items():
            doc = self._build_document_representation(file_name, file_records)
            documents.append(doc)

            unigrams = set(doc[f"unigrams_{self.token_limit}"])
            bigrams = set(doc[f"bigrams_{self.token_limit}"])

            unigram_df.update(unigrams)
            bigram_df.update(bigrams)

        self.documents = documents
        self.unigram_df = dict(unigram_df)
        self.bigram_df = dict(bigram_df)
        self.num_documents = len(documents)

    def build_stats_from_json(self, input_path: str | Path) -> None:
        records = self.load_json(input_path)
        self.build_stats(records)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "documents": self.documents,
            "unigram_df": self.unigram_df,
            "bigram_df": self.bigram_df,
            "num_documents": self.num_documents,
        }

    def rank_question(self, question: str, top_k: int = 6) -> List[Dict[str, Any]]:
        if not self.documents:
            raise ValueError("Статистика не построена. Сначала вызови build_stats(...) или build_stats_from_json(...).")

        normalized_question = self._normalize_text(question)
        question_tokens = self._tokenize(normalized_question)[: self.token_limit]
        question_unigrams = set(question_tokens)
        question_bigrams = set(self._build_bigrams(question_tokens))

        results: List[Dict[str, Any]] = []

        unigrams_key = f"unigrams_{self.token_limit}"
        bigrams_key = f"bigrams_{self.token_limit}"
        tokens_key = f"tokens_{self.token_limit}"

        for doc in self.documents:
            doc_unigrams = set(doc.get(unigrams_key, []))
            doc_bigrams = set(doc.get(bigrams_key, []))

            matched_unigrams = sorted(question_unigrams & doc_unigrams)
            matched_bigrams = sorted(question_bigrams & doc_bigrams)
            if len(matched_bigrams) < 1:
                continue

            unigram_score = sum(
                1 / self.unigram_df.get(tok, self.num_documents)  # Изменение здесь
                for tok in matched_unigrams
            )
            bigram_score = sum(
                1 / self.bigram_df.get(bg, self.num_documents)  # Аналогичное изменение для биграм
                for bg in matched_bigrams
            )

            score = self.unigram_weight * unigram_score + self.bigram_weight * bigram_score

            results.append({
                "file_name": doc["file_name"],
                "score": score,
                "unigram_score": unigram_score,
                "bigram_score": bigram_score,
                "unigram_overlap": len(matched_unigrams),
                "bigram_overlap": len(matched_bigrams),
                "matched_unigrams": matched_unigrams,
                "matched_bigrams": matched_bigrams,
                "question_tokens": question_tokens,
                "document_tokens": doc.get(tokens_key, []),
            })

        results.sort(
            key=lambda x: (x["score"], x["bigram_overlap"], x["unigram_overlap"]),
            reverse=True,
        )

        return results[:top_k]

    def rank_question_from_text(
            self,
            question_text: str,
            top_k: int | None = 5,
            percentage_cutoff: float = 0.15,
            print_results: bool = True,
            measure_time: bool = True,
    ) -> Dict[str, Any]:
        question_started_at = time.perf_counter() if measure_time else None
        ranked = self.rank_question(question_text, top_k=None)

        question_elapsed = (
            time.perf_counter() - question_started_at
            if measure_time and question_started_at is not None
            else None
        )

        if ranked:
            max_score = ranked[0]["score"]
            cutoff_score = max_score * percentage_cutoff
            ranked = [doc for doc in ranked if doc["score"] >= cutoff_score and doc["score"] > 1]

        if top_k is not None:
            ranked = ranked[:top_k]

        result_item = {
            "question": question_text,
            "elapsed_seconds": question_elapsed,
            "top_documents": ranked,
        }

        if print_results:
            print(f"\nQuestion: {question_text}")
            if question_elapsed is not None:
                print(f"Elapsed: {question_elapsed:.6f} sec")
            print("Top documents:")

            for rank_idx, doc in enumerate(ranked, start=1):
                print(
                    f"  {rank_idx}. {doc['file_name']} | "
                    f"score={doc['score']:.4f} | "
                    f"unigrams={doc['unigram_overlap']} | "
                    f"bigrams={doc['bigram_overlap']}"
                )
                print(f"     matched_unigrams: {doc['matched_unigrams']}")
                print(f"     matched_bigrams: {doc['matched_bigrams']}")

        return result_item



    def rank_questions(self, questions: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        all_results: List[Dict[str, Any]] = []

        for item in questions:
            question_text = item.get("question", "")
            ranked = self.rank_question(question_text, top_k=top_k)

            all_results.append({
                "id": item.get("id"),
                "question": question_text,
                "answer_type": item.get("answer_type"),
                "top_documents": ranked,
            })

        return all_results

    def rank_questions_from_file(
            self,
            questions_path: str | Path,
            top_k: int = 5,
            percentage_cutoff: float = 0.2,  # Добавлен порог (20% по умолчанию)
            print_results: bool = True,
            measure_time: bool = True,
    ) -> List[Dict[str, Any]]:
        questions = self.load_json(questions_path)

        all_results: List[Dict[str, Any]] = []
        total_started_at = time.perf_counter() if measure_time else None

        for idx, item in enumerate(questions, start=1):
            question_text = item.get("question", "")
            question_id = item.get("id")
            answer_type = item.get("answer_type")

            question_started_at = time.perf_counter() if measure_time else None
            ranked = self.rank_question(question_text, top_k=None)  # Изменено: убираем ограничение top_k при сортировке
            question_elapsed = (
                time.perf_counter() - question_started_at
                if measure_time and question_started_at is not None
                else None
            )

            # Фильтр "не более X% от первого места"
            if ranked:
                max_score = ranked[0]["score"]  # Берём максимальный (первый) результат
                cutoff_score = max_score * percentage_cutoff  # Рассчитываем порог
                ranked = [doc for doc in ranked if doc["score"] >= cutoff_score]

            # Ограничиваем top_k, если оно всё ещё задано
            if top_k is not None:
                ranked = ranked[:top_k]

            result_item = {
                "id": question_id,
                "question": question_text,
                "answer_type": answer_type,
                "elapsed_seconds": question_elapsed,
                "top_documents": ranked,
            }
            all_results.append(result_item)

            if print_results:
                print(f"\n[{idx}] Question ID: {question_id}")
                print(f"Question: {question_text}")
                print(f"Answer type: {answer_type}")
                if question_elapsed is not None:
                    print(f"Elapsed: {question_elapsed:.6f} sec")
                print("Top documents:")

                for rank_idx, doc in enumerate(ranked, start=1):
                    print(
                        f"  {rank_idx}. {doc['file_name']} | "
                        f"score={doc['score']:.4f} | "
                        f"unigrams={doc['unigram_overlap']} | "
                        f"bigrams={doc['bigram_overlap']}"
                    )
                    print(f"     matched_unigrams: {doc['matched_unigrams']}")
                    print(f"     matched_bigrams: {doc['matched_bigrams']}")

        if print_results and measure_time and total_started_at is not None:
            total_time = time.perf_counter() - total_started_at
            avg_time = total_time / len(questions) if questions else 0.0

            print("\n" + "=" * 80)
            print(f"Всего вопросов: {len(questions)}")
            print(f"Общее время: {total_time:.4f} сек")
            print(f"Среднее время на вопрос: {avg_time:.4f} сек")
            print("=" * 80)

        return all_results

    def print_stats_summary(self, top_n: int = 20) -> None:
        if not self.documents:
            print("Статистика пока не построена.")
            return

        print(f"Documents: {self.num_documents}")
        print(f"Unique unigrams: {len(self.unigram_df)}")
        print(f"Unique bigrams: {len(self.bigram_df)}")

        top_unigrams = sorted(self.unigram_df.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_bigrams = sorted(self.bigram_df.items(), key=lambda x: x[1], reverse=True)[:top_n]

        print("\nTop unigrams by document frequency:")
        for token, freq in top_unigrams:
            print(f"  {token}: {freq}")

        print("\nTop bigrams by document frequency:")
        for token, freq in top_bigrams:
            print(f"  {token}: {freq}")

