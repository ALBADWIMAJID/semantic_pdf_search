# ✅ مشروع: البحث الدلالي في ملفات PDF باستخدام نموذج محلي (nomic-embed-text)

import os
import glob
import time
import random
import numpy as np
import pyarrow as pa
from typing import List, Dict
from docling.chunking import HybridChunker
 # type: ignore
from docling.document_converter import DocumentConverter # type: ignore
import lancedb # type: ignore
from lancedb.pydantic import LanceModel # type: ignore
from transformers.tokenization_utils_base import PreTrainedTokenizerBase # type: ignore
from tiktoken import get_encoding
from dotenv import load_dotenv
from IPython.display import display
import ipywidgets as widgets # type: ignore
from langchain.embeddings import OllamaEmbeddings

load_dotenv()

# ✅ إعداد النموذج المحلي
embedding_model = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434"
)



# ✅ إعداد التوكنيزر
class OpenAITokenizerWrapper(PreTrainedTokenizerBase):
    def __init__(self, model_name: str = "cl100k_base", max_length: int = 8191, **kwargs):
        super().__init__(model_max_length=max_length, **kwargs)
        self.tokenizer = get_encoding(model_name)
        self._vocab_size = self.tokenizer.max_token_value

    def tokenize(self, text: str, **kwargs) -> List[str]:
        return [str(t) for t in self.tokenizer.encode(text)]

    def _tokenize(self, text: str) -> List[str]:
        return self.tokenize(text)

    def _convert_token_to_id(self, token: str) -> int:
        return int(token)

    def _convert_id_to_token(self, index: int) -> str:
        return str(index)

    def get_vocab(self) -> Dict[str, int]:
        return dict(enumerate(range(self.vocab_size)))

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def save_vocabulary(self, *args):
        return ()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls()

# ✅ إعداد القيم العامة
tokenizer = OpenAITokenizerWrapper()
MAX_TOKENS = 8191

# ✅ تحويل PDF إلى كائن Docling
def convert_pdf_to_document(pdf_path):
    converter = DocumentConverter()
    if not pdf_path.startswith('http'):
        pdf_path = os.path.abspath(pdf_path)
    return converter.convert(pdf_path)

# ✅ معالجة كل ملفات PDF في مجلد
def process_pdf_documents(pdf_dir):
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    if not pdf_files:
        print(f"\n❌ لا يوجد ملفات PDF في: {pdf_dir}")
        return []

    all_chunks = []
    for pdf_file in pdf_files:
        print(f"🔍 Обработка файла: {pdf_file}")
        result = convert_pdf_to_document(pdf_file)
        chunker = HybridChunker(tokenizer=tokenizer, max_tokens=MAX_TOKENS, merge_peers=True)
        chunks = list(chunker.chunk(dl_doc=result.document))
        print(f"✅ Извлечено {len(chunks)} чанков из документа {pdf_file}")
        all_chunks.extend(chunks)
    return all_chunks

# ✅ إنشاء أو الاتصال بقاعدة LanceDB
def create_or_connect_db(db_path="data/lancedb"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    return lancedb.connect(db_path)

# ✅ إنشاء الجدول وتخزين التضمينات
def create_and_fill_table(db, chunks, table_name="pdf_chunks"):
    print(f"🧠 Создаем эмбеддинги для {len(chunks)} чанков...")
    schema = pa.schema([
        pa.field("text", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), 768)),
        pa.field("doc_name", pa.string()),
        pa.field("chunk_id", pa.int32())
    ])
    table = db.create_table(table_name, schema=schema, mode="overwrite")

    processed_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)
        doc_name = chunk.metadata.get("file_name", f"doc_{i//2}") if hasattr(chunk, "metadata") else f"doc_{i//2}"
        processed_chunks.append({"text": chunk_text, "doc_name": doc_name, "chunk_id": i})

    progress = widgets.IntProgress(value=0, min=0, max=len(processed_chunks), description='Прогресс:', bar_style='info')
    display(progress)

    for i, chunk in enumerate(processed_chunks):
        try:
            print(f"   🧮 Эмбеддинг чанка {i+1}/{len(processed_chunks)}")
            vector = embedding_model.embed_query(chunk["text"])
            chunk["vector"] = vector
            table.add([chunk])
            progress.value = i + 1
        except Exception as e:
            print(f"❌ Ошибка при добавлении чанка {i+1}: {e}")

    return table

# ✅ البحث داخل الجدول
def search_in_table(query_text, table, limit=3):
    print(f"🔍 Запрос: '{query_text}'")
    try:
        embedding = embedding_model.embed_query(query_text)
        results = table.search(embedding).limit(limit).to_pandas()
        return results
    except Exception as e:
        print(f"❌ Ошибка поиска: {e}")
        return None

# ✅ عرض النتائج
def display_search_results(results):
    if results is None or len(results) == 0:
        print("❌ لا يوجد نتائج.")
        return

    for i, row in results.iterrows():
        print(f"\n🔹 Результат {i+1} — Документ: {row['doc_name']} | Чанк ID: {row['chunk_id']}")
        print(f"📝 {row['text'][:300]}...")

# ✅ تشغيل الدورة الكاملة
def run_pipeline(pdf_dir="pdfs", db_path="./lancedb", table_name="pdf_docs"):
    print("\n==================================================")
    print(f"🚀 Обработка PDF: {pdf_dir}")
    chunks = process_pdf_documents(pdf_dir)
    if not chunks:
        return False
    db = create_or_connect_db(db_path)
    table = create_and_fill_table(db, chunks, table_name)
    return True

if __name__ == '__main__':
    success = run_pipeline("pdfs", "./lancedb", "pdf_docs")
    if success:
        db = lancedb.connect("./lancedb")
        table = db.open_table("pdf_docs")
        while True:
            question = input("Введите вопрос или 'exit': ")
            if question.lower() == 'exit':
                break
            results = search_in_table(question, table)
            display_search_results(results)
