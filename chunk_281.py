from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata
import fitz


@dataclass
class SectionChunk281:
    course: str
    source_file: str
    chapter: str
    section: str | None
    subsection: str | None
    text: str
    page_start: int
    page_end: int


def _normalize(text):
    text = unicodedata.normalize('NFKC', text)
    text = text.replace('※', '').replace('(✽)', '').replace('✽', '')
    return ' '.join(text.split()).casefold()


def extract_section_chunks(pdf_path, long_section_chars=10000):
    """Keep sections intact; split long sections at existing subsection headings.

    The threshold triggers structural splitting, not a hard size limit. A long
    subsection or a section without subsections remains whole. Original text and
    code are retained; diagrams are not interpreted.
    """
    if long_section_chars <= 0:
        raise ValueError('long_section_chars must be positive')
    path = Path(pdf_path)
    with fitz.open(path) as doc:
        outline = doc.get_toc()
        chapters = [entry for entry in outline if entry[0] == 1]
        if len(chapters) != 1:
            raise ValueError(f'{path.name}: expected one chapter at outline level 1')
        chapter = chapters[0][1]
        chapter_number = chapter.split()[0]
        pages, offsets, page_lines = [], [0], []
        for page in doc:
            lines = []
            # Text flags exclude image payloads and retain the same text as get_text().
            for block in page.get_text('dict', flags=fitz.TEXTFLAGS_TEXT)['blocks']:
                for line in block.get('lines', []):
                    if line['bbox'][1] < 55:
                        continue  # printed page number and running chapter/section header
                    lines.append(''.join(span['text'] for span in line['spans']))
            positions = []
            text = ''
            for line in lines:
                positions.append((len(text), line))
                text += line + '\n'
            page_lines.append(positions)
            pages.append(text)
            offsets.append(offsets[-1] + len(text))
        full_text = ''.join(pages)
        boundaries = []
        for level, title, page_number in outline:
            if level not in (2, 3):
                continue
            number, heading = title.split(' ', 1)
            if not re.fullmatch(re.escape(chapter_number) + r'\.\d+' + (r'\.\d+' if level == 3 else ''), number):
                raise ValueError(f'{path.name}: unexpected heading number {number}')
            lines = page_lines[page_number - 1]
            matches = []
            for i, (position, line) in enumerate(lines):
                normalized = _normalize(line)
                if normalized == _normalize(title):
                    matches.append(offsets[page_number - 1] + position)
                elif normalized == number:
                    # Outline wording can differ from the printed title (e.g.
                    # Big-O versus mathematical symbols); the unique number
                    # on the outline's destination page anchors the boundary.
                    matches.append(offsets[page_number - 1] + position)
            if not matches:
                # Some supplied bookmarks point to page 1 instead of the heading.
                # Require a unique numbered heading across the entire chapter.
                for page_index, candidate_lines in enumerate(page_lines):
                    for position, line in candidate_lines:
                        normalized = _normalize(line)
                        if normalized in (number, _normalize(title)):
                            matches.append(offsets[page_index] + position)
            if len(matches) != 1:
                raise ValueError(f'{path.name}: expected one match for {title!r} on page {page_number}; found {len(matches)}')
            boundaries.append((matches[0], level, title))
        if boundaries != sorted(boundaries):
            raise ValueError(f'{path.name}: heading positions are out of order')

        chunks = []

        def emit(start, end, section=None, subsection=None):
            raw = full_text[start:end]
            text = raw.strip()
            if not text:
                return
            first = start + len(raw) - len(raw.lstrip())
            last = start + len(raw.rstrip()) - 1
            chunks.append(SectionChunk281(
                course='281', source_file=path.name, chapter=chapter,
                section=section, subsection=subsection, text=text,
                page_start=bisect_right(offsets, first),
                page_end=bisect_right(offsets, last),
            ))

        # Preserve chapter introductions, excluding the chapter display heading.
        chapter_lines = page_lines[chapters[0][2] - 1]
        intro_start = None
        for i, (position, line) in enumerate(chapter_lines):
            if _normalize(line) == _normalize(f'Chapter {chapter_number}'):
                heading = ''
                for j in range(i + 1, min(i + 5, len(chapter_lines))):
                    heading += ' ' + chapter_lines[j][1]
                    if _normalize(heading) == _normalize(chapter.split(' ', 1)[1]):
                        intro_start = offsets[chapters[0][2] - 1] + chapter_lines[j][0] + len(chapter_lines[j][1]) + 1
                        break
                break
        if intro_start is None:
            raise ValueError(f'{path.name}: could not locate chapter display heading')
        sections = [b for b in boundaries if b[1] == 2]
        emit(intro_start, sections[0][0] if sections else len(full_text))
        for i, (start, _, title) in enumerate(sections):
            end = sections[i + 1][0] if i + 1 < len(sections) else len(full_text)
            subsections = [b for b in boundaries if b[1] == 3 and start <= b[0] < end]
            if end - start <= long_section_chars or not subsections:
                emit(start, end, title)
                continue
            # Keep any section introduction; do not create heading-only chunks.
            prefix = full_text[start:subsections[0][0]]
            if _normalize(prefix) != _normalize(title):
                emit(start, subsections[0][0], title)
            for j, (sub_start, _, sub_title) in enumerate(subsections):
                sub_end = subsections[j + 1][0] if j + 1 < len(subsections) else end
                emit(sub_start, sub_end, title, sub_title)
        return chunks


def extract_course_chunks(folder='eecs281', long_section_chars=10000):
    paths = sorted(Path(folder).glob('chapter-*.pdf'), key=lambda p: int(p.stem.split('-')[-1]))
    if not paths:
        raise ValueError(f'No chapter PDFs found in {folder}')
    return [chunk for path in paths for chunk in extract_section_chunks(path, long_section_chars)]


if __name__ == '__main__':
    chunks = extract_course_chunks()
    print(chunks)
    print(f'Parsed {len(chunks)} chunks from {len({c.source_file for c in chunks})} chapter PDFs')
