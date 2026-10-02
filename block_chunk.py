"""Chapter/section chunks for the numbered EECS 280 notes PDF."""

from bisect import bisect_right
from dataclasses import dataclass
import re

import fitz
from sentence_transformers import SentenceTransformer

PDF_PATH = "eecs280notes.pdf"
model = SentenceTransformer("all-MiniLM-L6-v2")


@dataclass
class SectionChunk:
    chapter: str
    section: str | None
    text: str
    page_start: int
    page_end: int


def extract_section_chunks(pdf_path):
    """Concatenate body pages, then split chapters and their top-level sections.

    The PDF outline validates heading titles and pages. Subsections remain inside
    their parent section; introductions and chapters without sections are retained.
    Page metadata uses one-based PDF page numbers, not printed page labels.
    """
    with fitz.open(pdf_path) as doc:
        outline = doc.get_toc()
        chapters = [entry for entry in outline if entry[0] == 2]
        if not chapters:
            raise ValueError("Expected chapter entries at level 2 in the PDF outline")

        pages = []
        offsets = [0]
        for page in doc:
            # This PDF places running headers above y=50 and footers below y=730.
            lines = []
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    if 50 <= line["bbox"][1] < page.rect.height - 62:
                        lines.append("".join(span["text"] for span in line["spans"]))
            text = "\n".join(lines) + "\n"
            pages.append(text)
            offsets.append(offsets[-1] + len(text))
        full_text = "".join(pages)

        boundaries = []
        chapter_number = 0
        for level, title, page_number in outline:
            if level not in (1, 2, 3):
                continue
            page_start = offsets[page_number - 1]
            page_text = pages[page_number - 1]
            if level == 1:
                match = re.search(r"(?m)^Part [IVXLCDM]+\s*$", page_text)
            elif level == 2:
                chapter_number += 1
                match = re.search(r"(?m)^CHAPTER\n[^\n]+\n" + re.escape(title.upper()) + r"\s*$", page_text)
            else:
                # Match a complete numbered heading, never a reference in prose.
                match = re.search(
                    r"(?m)^(" + str(chapter_number) + r"\.\d+)\s+"
                    + re.escape(title) + r"[ \t]*$", page_text
                )
            if match is None:
                raise ValueError(f"Could not locate outline heading {title!r} on PDF page {page_number}")
            label = f"{chapter_number} {title}" if level == 2 else title
            if level == 3:
                label = f"{match.group(1)} {title}"
            boundaries.append((page_start + match.start(), level, label))

        chunks = []
        chapter = None
        for i, (start, level, label) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(full_text)
            if level == 1:
                chapter = None
                continue
            if level == 2:
                chapter = label
                # Strip the display heading, keeping only chapter introduction.
                body_start = full_text.index("\n", start)
                body_start = full_text.index("\n", body_start + 1)
                body_start = full_text.index("\n", body_start + 1) + 1
            else:
                body_start = start
            body = full_text[body_start:end].strip()
            if not body:
                continue
            last_content = body_start + len(full_text[body_start:end].rstrip()) - 1
            chunks.append(SectionChunk(
                chapter=chapter,
                section=label if level == 3 else None,
                text=body,
                page_start=bisect_right(offsets, body_start),
                page_end=bisect_right(offsets, last_content),
            ))
        return chunks



def create_search_passages(chunks):
    """Index bounded passages while retaining complete sections for retrieval."""
    encoder = model
    tokenizer = encoder.tokenizer
    passages, parents = [], []
    for parent, chunk in enumerate(chunks):
        heading = chunk.chapter + (" > " + chunk.section if chunk.section else " > Introduction")
        prefix = tokenizer.encode(heading + "\n", add_special_tokens=False)
        budget = encoder.max_seq_length - tokenizer.num_special_tokens_to_add(pair=False) - len(prefix)
        if budget <= 0:
            raise ValueError("Heading exceeds the embedding model's token budget")
        tokens = tokenizer.encode(chunk.text, add_special_tokens=False)
        for start in range(0, len(tokens), budget):
            passages.append(tokenizer.decode(prefix + tokens[start:start + budget]))
            parents.append(parent)
    return passages, parents


def create_embeddings(passages):
    import faiss
    import numpy as np
    if not passages:
        raise ValueError("No passages to index")
    embeddings = model.encode(passages, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype="float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def retrieve_context(query, index, chunks, passage_parents, k=10):
    """Rank sections by their best passage, returning each section once."""
    if k <= 0:
        return []
    query_embedding = model.encode([query], normalize_embeddings=True)
    # Expand the search until enough distinct sections have been found.
    count = min(max(k, 1), index.ntotal)
    while count:
        _, indices = index.search(query_embedding.astype("float32"), count)
        selected = list(dict.fromkeys(passage_parents[i] for i in indices[0] if i >= 0))
        if len(selected) >= k or count == index.ntotal:
            return [chunks[i] for i in selected[:k]]
        count = min(count * 2, index.ntotal)
    return []


def answer_question(question):
    chunks = extract_section_chunks(PDF_PATH)
    passages, parents = create_search_passages(chunks)
    index, _ = create_embeddings(passages)
    context_chunks = retrieve_context(question, index, chunks, parents)
    print(f"Question: {question}")
    print(f"\nRetrieved context ({len(context_chunks)} sections):")
    for i, chunk in enumerate(context_chunks, 1):
        print(f"\n--- Chunk {i}: {chunk.chapter} / {chunk.section or 'Introduction'} "
              f"(PDF pages {chunk.page_start}–{chunk.page_end}) ---")
        print(chunk.text)
    return context_chunks


if __name__ == "__main__":
    answer_question("What is the difference between a stack and a queue?")
