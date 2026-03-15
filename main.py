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

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def create_embeddings(chunks):
    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings).astype('float32'))

    return index, embeddings

def retrieve_context(query, index, chunks, k=3):
    query_embedding = model.encode([query])
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