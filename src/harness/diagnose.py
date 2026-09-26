"""Classify eval-run failures: lost-in-the-middle, retrieval mismatch, low-recall hallucination."""

import json
import sys
from pathlib import Path

from harness.eval import heading_match, load_questions
from schemas.chunk import Chunk
from schemas.response import REFUSE


def _chunk(ref: dict) -> Chunk:
    return Chunk(
        chunk_id="",
        document=ref["document"],
        version=ref["version"],
        section=ref["section"],
        section_title=ref["section_title"],
        subsection=ref.get("subsection", ""),
        subsection_title=ref.get("subsection_title", ""),
        text=ref.get("text", ""),
        part=ref.get("part", 0),
        page=ref.get("page"),
        embedding=[],
    )


def classify(result: dict, question) -> list[str]:
    if question.type == "unanswerable" or not question.gold_section:
        return []
    hits = result["retrieved_chunks"][: question.retrieve_k]
    gold_at = [i for i, ref in enumerate(hits) if heading_match(_chunk(ref), question)]
    answer_ok = result["include_ok"] and result["exclude_ok"]
    refused = result["answer"].strip().lower() == REFUSE.lower()
    labels: list[str] = []
    if not gold_at:
        labels.append("retrieval_mismatch")
        if not refused:
            labels.append("hallucination_under_low_recall")
        return labels
    rank = gold_at[0]
    if not answer_ok and rank not in (0, len(hits) - 1):
        labels.append("lost_in_the_middle")
    return labels


def main(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = {q.id: q for q in load_questions()}
    counts = {
        "lost_in_the_middle": 0,
        "retrieval_mismatch": 0,
        "hallucination_under_low_recall": 0,
    }
    for result in payload["results"]:
        labels = classify(result, questions[result["id"]])
        for label in labels:
            counts[label] += 1
        if labels:
            print(f"{result['id']:24} {', '.join(labels)}")
    print(
        f"\nlost_in_the_middle={counts['lost_in_the_middle']}  "
        f"retrieval_mismatch={counts['retrieval_mismatch']}  "
        f"hallucination_under_low_recall={counts['hallucination_under_low_recall']}"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))