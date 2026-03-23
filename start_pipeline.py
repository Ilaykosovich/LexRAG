from DocumentRouter import DocumentRouter
from PdfProcessor1 import PDFProcessor1
from Chunking.HybridLegalRetriever import HybridLegalRetriever
from Chunking.LegalChankBuilder import LegalChunkBuilder
from sentence_transformers import SentenceTransformer
import time
from typing import Any, Dict, List, Optional, Counter
import tiktoken
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile
import json
from dotenv import load_dotenv
ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from arlc import (  # noqa: E402
    EvaluationClient,
    RetrievalRef,
    SubmissionAnswer,
    SubmissionBuilder,
    Telemetry,
    TelemetryTimer,
    TimingMetrics,
    UsageMetrics,
    get_config,
    normalize_retrieved_pages,
)

CONFIG = get_config()
TOKENIZER = tiktoken.encoding_for_model("gpt-4o-mini")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.1
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
)


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Directories generated at runtime inside starter_kit — skip when archiving
_EXCLUDE_DIRS = {"__pycache__", "docs_corpus", "storage", ".venv", "venv", "env"}
# Specific files generated at runtime — skip when archiving
_EXCLUDE_FILES = {".env", "submission.json", "questions.json", "code_archive.zip"}

_TYPE_INSTRUCTIONS: dict[str, str] = {
    "number":    "Return only the numeric value (integer or decimal). No units, no explanation.",
    "boolean":   "Return only 'true' or 'false'. No explanation.",
    "name":      "Return only the exact name or entity as it appears in the documents. No explanation.",
    "names":     "Return only a semicolon-separated list of names. No explanation.",
    "date":      "Return only the date in YYYY-MM-DD format. No explanation.",
    "free_text": "Answer in full sentences using only the provided context.",
}


def embedding_function(texts: List[str]) -> List[List[float]]:
    return model.encode(texts, normalize_embeddings=False).tolist()


def build_context_with_metadata(docs: list) -> str:
    blocks = []

    for i, doc in enumerate(docs, 1):
        doc_id = getattr(doc, "doc_id", "unknown")
        text = getattr(doc, "text", "")

        metadata = getattr(doc, "metadata", {}) or {}
        page_start = metadata.get("page_start")
        page_end = metadata.get("page_end")

        page_repr = "unknown"
        try:
            if page_start is not None and page_end is not None:
                start = int(page_start) + 1
                end = int(page_end) + 1
                page_repr = str(start) if start == end else f"{start}-{end}"
            elif page_start is not None:
                page_repr = str(int(page_start) + 1)
        except (TypeError, ValueError):
            page_repr = "unknown"

        label = "page" if "-" not in page_repr else "pages"
        header = f"[CHUNK {i}] doc_id={doc_id}; {label}={page_repr}"
        blocks.append(f"{header}\n{text}")

    return "\n\n".join(blocks)


def ensure_code_archive(archive_path: Path) -> Path:
    """Archive the entire starter_kit directory, excluding generated/runtime artifacts."""
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    archive_resolved = archive_path.resolve()
    with ZipFile(archive_path, "w", compression=ZIP_DEFLATED) as zip_file:
        for file_path in ROOT_DIR.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.resolve() == archive_resolved:
                continue
            parts = set(file_path.relative_to(ROOT_DIR).parts)
            if parts & _EXCLUDE_DIRS:
                continue
            if file_path.name in _EXCLUDE_FILES:
                continue
            zip_file.write(file_path, file_path.relative_to(ROOT_DIR))
    return archive_path

def build_chunk_lookup(docs: list) -> dict[int, RetrievalRef]:
    lookup = {}

    for i, doc in enumerate(docs, 1):
        doc_id = getattr(doc, "doc_id", None)
        metadata = getattr(doc, "metadata", {}) or {}
        page_start = metadata.get("page_start")
        page_end = metadata.get("page_end")

        if not doc_id:
            continue

        pages = []
        try:
            if page_start is not None and page_end is not None:
                start = int(page_start) + 1
                end = int(page_end) + 1
                if start <= end:
                    pages = list(range(start, end + 1))
            elif page_start is not None:
                pages = [int(page_start) + 1]
        except (TypeError, ValueError):
            continue

        if pages:
            lookup[i] = RetrievalRef(doc_id=doc_id, page_numbers=pages)

    return lookup

def _parse_answer_by_type(raw: str, answer_type: str):
    text = (raw or "").strip()
    at = str(answer_type or "free_text").lower()

    if at == "number":
        try:
            return float(text.replace(",", "."))
        except (TypeError, ValueError):
            return text

    if at == "boolean":
        lower = text.lower()
        if lower in ("true", "yes", "1"):
            return True
        if lower in ("false", "no", "0"):
            return False
        return text

    if at == "date":
        return text

    if at == "names":
        parts = [p for p in (x.strip() for x in text.replace(",", ";").split(";")) if p]
        return parts if parts else [text]

    if at == "null":
        return None

    return text



def build_prompt(context: str, question_text: str, answer_type: str = "free_text") -> str:
    instruction = _TYPE_INSTRUCTIONS.get(answer_type, _TYPE_INSTRUCTIONS["free_text"])

    return f"""
You are given retrieved chunks from legal documents.

Each chunk is formatted like:
[CHUNK N] doc_id=<doc_id>; page=<page_number>
or
[CHUNK N] doc_id=<doc_id>; pages=<start-end>

Answer the question using only the provided context.

Return valid JSON only in this exact format:
{{
  "answer": "your answer here",
  "used_chunks": [1, 2]
}}

Rules:
- The answer must follow this instruction: {instruction}
- Use only the provided context.
- Include in used_chunks only chunk numbers that directly support the final answer.
- Do not include any explanation outside JSON.

Context:
{context}

Question:
{question_text}
""".strip()


def parse_llm_response(raw: str, answer_type: str, chunk_lookup: dict[int, RetrievalRef]):
    text = (raw or "").strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {
            "answer": _parse_answer_by_type(text, answer_type),
            "retrieval_refs": [],
        }

    raw_answer = data.get("answer", "")
    used_chunks = data.get("used_chunks", [])

    if isinstance(raw_answer, list):
        raw_answer_for_parse = "; ".join(map(str, raw_answer))
    elif raw_answer is None:
        raw_answer_for_parse = ""
    else:
        raw_answer_for_parse = str(raw_answer)

    answer = _parse_answer_by_type(raw_answer_for_parse, answer_type)

    refs = []
    for idx in used_chunks:
        try:
            idx = int(idx)
        except (TypeError, ValueError):
            continue

        if idx in chunk_lookup:
            refs.append(chunk_lookup[idx])

    refs = normalize_retrieved_pages(refs)

    return {
        "answer": answer,
        "retrieval_refs": refs,
    }



path = "C:\\Users\\ilayk\\PycharmProjects\\LexRAG\\docling_layout_output\\records_processed3.json"
path_to_pdf ="C:\\Users\\ilayk\\PycharmProjects\\LexRAG\\dataset_documents\\Test1"
questionPath = "C:\\Users\\ilayk\\OneDrive\\Desktop\\Конкурс\\starter-kit\\questions1.json"

processor = PDFProcessor1()
client = EvaluationClient.from_env()

start_time = time.time()
items = processor.process_directory(path_to_pdf)
elapsed_time = time.time() - start_time

print(f"Время парсинга pdf: {elapsed_time:.4f} секунд")

documents_dic = {}
for item in items:
    if item['source'] not in documents_dic:
        documents_dic[item['source']] = []
    documents_dic[item['source']].append(item)

router = DocumentRouter(token_limit=50, unigram_weight=1.0, bigram_weight=3.0)
router.build_stats(items)

router.print_stats_summary(top_n=10)

questions = DocumentRouter.load_json(questionPath)
retriever_dic = {}
for key_doc,val_doc in documents_dic.items():
    builder = LegalChunkBuilder()
    docs, address_index = builder.build(val_doc)
    retriever = HybridLegalRetriever.from_documents(
        docs=docs,
        address_index=address_index,
        embedding_function=embedding_function,
    )
    retriever_dic[key_doc] = retriever

builder = SubmissionBuilder(
    architecture_summary="Custom RAG with document routing, chunk retrieval, and LLM answer generation with cited chunk pages",
)

for index_number, question_item in enumerate(questions, 1):
    question_text = question_item["question"]
    question_id = question_item["id"]
    answer_type = question_item.get("answer_type", "free_text")

    print(f"[{index_number}/{len(questions)}] {question_id}")

    ranked_question = router.rank_question_from_text(question_text)

    all_hits = []
    for doc in ranked_question["top_documents"]:
        name_doc = doc["file_name"]
        if name_doc in retriever_dic:
            hits = retriever_dic[name_doc].search(question_text, k=5)
            all_hits.extend(hits)

    if not all_hits:
        answer = ""
        retrieval_refs = []
        response_text = ""
        prompt = question_text
        timing = TimingMetrics(ttft_ms=0, tpot_ms=0, total_time_ms=0)
    else:
        all_hits = sorted(all_hits, key=lambda x: x.score, reverse=True)[:10]

        context = build_context_with_metadata(all_hits)
        chunk_lookup = build_chunk_lookup(all_hits)
        prompt = build_prompt(context, question_text, answer_type)

        telemetry_timer = TelemetryTimer()
        response_chunks = []

        for chunk in llm.stream(prompt):
            telemetry_timer.mark_token()
            response_chunks.append(chunk.content)

        response_text = "".join(response_chunks)
        finished_timing = telemetry_timer.finish()

        parsed = parse_llm_response(response_text, answer_type, chunk_lookup)
        answer = parsed["answer"]
        retrieval_refs = parsed["retrieval_refs"]

        timing = TimingMetrics(
            ttft_ms=finished_timing.ttft_ms,
            tpot_ms=finished_timing.tpot_ms,
            total_time_ms=finished_timing.total_time_ms,
        )

    usage = UsageMetrics(
        input_tokens=len(TOKENIZER.encode(prompt)),
        output_tokens=len(TOKENIZER.encode(response_text)),
    )

    telemetry = Telemetry(
        timing=timing,
        retrieval=retrieval_refs,
        usage=usage,
        model_name="gpt-4o-mini",
    )

    builder.add_answer(
        SubmissionAnswer(
            question_id=question_id,
            answer=answer,
            telemetry=telemetry,
        )
    )
submission_path = builder.save(str(CONFIG.submission_path))
code_archive_path = ensure_code_archive(CONFIG.code_archive_path)
print("\nSaved submission.json")
print(f"Using code archive: {code_archive_path}")
print("Submitting...")
response = client.submit_submission(submission_path, code_archive_path)
print(response)
