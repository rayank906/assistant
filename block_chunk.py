import fitz
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

PDF_PATH = "eecs280notes.pdf"
model = SentenceTransformer('all-MiniLM-L6-v2')

def is_probable_prose(text, min_chars=120, min_alpha_ratio=0.6):
    text = text.strip()
    if len(text) < min_chars:
        return False
    non_space = [c for c in text if not c.isspace()]
    if not non_space:
        return False
    alpha_ratio = sum(c.isalpha() for c in non_space) / len(non_space)
    return alpha_ratio > min_alpha_ratio

def extract_paragraphs_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    paragraphs = []
    for page in doc:
        for block in page.get_text("blocks"):
            block_text = block[4].replace("\n", " ").strip()
            if is_probable_prose(block_text):
                paragraphs.append(block_text)
    return paragraphs

def chunk_paragraphs(paragraphs, chunk_size=1500, overlap_paragraphs=1):
    chunks = []
    current = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) > chunk_size and current:
            chunks.append("\n\n".join(current))
            current = current[-overlap_paragraphs:] if overlap_paragraphs else []
            current_len = sum(len(p) for p in current)
        current.append(para)
        current_len += len(para)

    if current:
        chunks.append("\n\n".join(current))
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
    paragraphs = extract_paragraphs_from_pdf(PDF_PATH)
    chunks = chunk_paragraphs(paragraphs)
    index, embeddings = create_embeddings(chunks)

    # Retrieve relevant context
    context_chunks = retrieve_context(question, index, chunks)

    print(f"Question: {question}")
    print(f"\nRetrieved context ({len(context_chunks)} chunks):")
    for i, chunk in enumerate(context_chunks, 1):
        print(f"\n--- Chunk {i} ---")
        print(chunk)

    return context_chunks

if __name__ == "__main__":
    question = "What is the difference between a stack and a queue?"
    answer_question(question)
