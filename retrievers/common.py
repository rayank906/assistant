from dataclasses import dataclass, field


@dataclass
class Hit:
    section_id: str
    parent_index: int
    passage_index: int
    score: float
    score_type: str
    evidence: list[int] = field(default_factory=list)
    components: dict = field(default_factory=dict)


def section_id(chunk):
    chapter = chunk.chapter.split()[0]
    section = chunk.section.split()[0] if chunk.section else 'intro'
    return f'{chunk.course}:{chunk.source_file}:chapter-{chapter}:{section}'


def collapse_passages(corpus, passage_indices, scores, top_k, score_type):
    hits, seen = [], set()
    for passage, score in zip(passage_indices, scores):
        passage = int(passage)
        if passage < 0:
            continue
        parent = corpus.parents[passage]
        if parent in seen:
            continue
        seen.add(parent)
        hits.append(Hit(corpus.section_ids[parent], parent, passage, float(score), score_type, [passage]))
        if len(hits) >= top_k:
            break
    return hits
