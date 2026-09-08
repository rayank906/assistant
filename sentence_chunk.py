import re
import fitz
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

PDF_PATH = "eecs280notes.pdf"
model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_into_sentences(text):
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s for s in sentences if s]

def chunk_text(text, chunk_size=1500, overlap_sentences=2):
    sentences = split_into_sentences(text)
    chunks = []
    current = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > chunk_size and current:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_len = sum(len(s) for s in current)
        current.append(sentence)
        current_len += len(sentence)

    if current:
        chunks.append(" ".join(current))
    return chunks

def create_embeddings(chunks):
    embeddings = model.encode(chunks, normalize_embeddings=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))

    return index, embeddings

def retrieve_context(query, index, chunks, k=10):
    query_embedding = model.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_embedding.astype('float32'), k)
    return [chunks[i] for i in indices[0]]

def answer_question(question):
    # Build vector DB
    text = extract_text_from_pdf(PDF_PATH)
    chunks = chunk_text(text)
    index, embeddings = create_embeddings(chunks)

    # Retrieve relevant context
    context_chunks = retrieve_context(question, index, chunks)

    print(f"Question: {question}")
    print(f"\nRetrieved context ({len(context_chunks)} chunks):")
    for i, chunk in enumerate(context_chunks, 1):
        print(f"\n--- Chunk {i} ---")
        print(chunk[:200] + "..." if len(chunk) > 200 else chunk)

    return context_chunks

if __name__ == "__main__":
    question = "What is the difference between a stack and a queue?"
    answer_question(question)
