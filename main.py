from PdfProcessor import PDFProcessor
from DoclingPdfRagIngestor import DoclingPdfRagIngestor



path = "C:\\Users\\ilayk\\PycharmProjects\\LexRAG\\dataset_documents\\Test"

ingestor = DoclingPdfRagIngestor(recursive=True, include_table_summaries=True)
items = ingestor.process_pdf_folder(path)

processor = PDFProcessor(
    merge_lines_to_paragraphs=True,
    extract_tables=True,
    table_row_docs=True,
)

items1 = processor.process_directory(path, recursive=False)



print("items:", len(items1))

# посмотреть первые 10
for it in items[:10]:
    print(it["type"], it["source"], it["page"], it["text"][:120])