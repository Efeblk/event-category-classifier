"""Convert the LLM-written synthetic utterances in data/synthetic/part*.jsonl to BIO-tagged rows.

Each input line is {"text", "spans": [[slot, substring], ...]}. Every substring is located left to right, after the end
of the previous span, and must sit on token boundaries. Rows whose text contains an event_name span of a test set are
dropped (decontamination), as are rows whose text equals a test text and duplicate texts.

Two outputs:
- Training set E: parts 1-4, decontaminated against the messy test set. data/clean/synthetic_train.jsonl and
  synthetic_stats.json.
- The pool for the data curve (train_bert.py --synthetic N): parts 1-8, decontaminated against the messy test set and
  every Codex test file. data/clean/synthetic_pool_train.jsonl and synthetic_pool_stats.json.
"""

import json
import re
from pathlib import Path

from slots import TOKEN, bio_tags, dedupe, drop_leaks, read_jsonl, slot_counts, spans, write_lines

ROOT = Path(__file__).resolve().parent
PARTS = ("part1", "part2", "part3", "part4")
POOL_PARTS = PARTS + ("part5", "part6", "part7", "part8")


def locate(text, substring, after):
    """Return (start, end) of the first occurrence of substring at or after position after that starts and ends on
    token boundaries, so "la" is not found inside "playing"."""
    starts = {m.start() for m in re.finditer(TOKEN, text)}
    ends = {m.end() for m in re.finditer(TOKEN, text)}
    start = text.find(substring, after) if substring else -1
    while start >= 0 and not (start in starts and start + len(substring) in ends):
        start = text.find(substring, start + 1)
    if start < 0:
        raise ValueError(f"span {substring!r} not found as whole tokens in {text!r} after position {after}")
    return start, start + len(substring)


def to_row(row_id, obj):
    """Tokens and BIO tags for one {"text", "spans"} object. Raise ValueError on bad or overlapping spans."""
    text, found, pos = obj["text"], [], 0
    for slot, substring in obj["spans"]:
        start, end = locate(text, substring, pos)
        found.append((start, end, slot))
        pos = end
    for start, end, slot in found:
        for match in re.finditer(TOKEN, text):
            touches = match.start() < end and match.end() > start
            if touches and not (match.start() >= start and match.end() <= end):
                raise ValueError(f"{row_id}: {slot} span does not match token boundaries in {text!r}")
    tagged = bio_tags(text, found)
    if tagged is None:
        raise ValueError(f"{row_id}: overlapping spans in {text!r}")
    tokens, tags = tagged
    return {"id": row_id, "text": text, "tokens": tokens, "tags": tags}


def read_part(path):
    """Rows of one part file, split on newlines; ids use the file stem and the 1-based line number."""
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), 1):
        if line.strip():
            rows.append(to_row(f"synthetic:{path.stem}:{number}", json.loads(line)))
    return rows


def test_event_names(test_rows):
    """Lowercased token tuples of every event_name span in the test rows."""
    return {tuple(w.lower() for w in r["tokens"][s:e]) for r in test_rows for kind, s, e in spans(r["tags"])
            if kind == "event_name"}


def has_phrase(tokens, phrase):
    """True when phrase (lowercased tokens) occurs as a contiguous run in tokens (lowercased)."""
    words = [t.lower() for t in tokens]
    n = len(phrase)
    return any(tuple(words[i:i + n]) == phrase for i in range(len(words) - n + 1))


def decontaminate(rows, test_rows):
    """Drop rows containing a test event name, then rows whose lowercased text equals a test text.

    Return (kept, dropped for event names, dropped for test text).
    """
    names = test_event_names(test_rows)
    kept = [r for r in rows if not any(has_phrase(r["tokens"], name) for name in names)]
    by_name = len(rows) - len(kept)
    kept, by_text = drop_leaks(kept, {r["text"].lower() for r in test_rows})
    return kept, by_name, by_text


def build(parts, test_rows, name):
    """Read the parts, decontaminate against test_rows, dedupe, and write data/clean/<name>_train.jsonl and
    <name>_stats.json. Return the stats."""
    rows = [row for part in parts for row in read_part(ROOT / "data" / "synthetic" / f"{part}.jsonl")]
    kept, by_name, by_text = decontaminate(rows, test_rows)
    kept, duplicates = dedupe(kept)
    stats = {
        "rows_in": len(rows),
        "dropped_test_event_names": by_name,
        "dropped_test_text": by_text,
        "duplicates_dropped": duplicates,
        "rows_out": len(kept),
        "spans": slot_counts(kept),
    }
    write_lines(ROOT / "data" / "clean" / f"{name}_train.jsonl", [json.dumps(r, ensure_ascii=False) for r in kept])
    write_lines(ROOT / "data" / "clean" / f"{name}_stats.json", [json.dumps(stats, indent=2, ensure_ascii=False)])
    return stats


def main():
    messy = read_jsonl(ROOT / "data" / "messy" / "messy_test.jsonl")
    codex = [row for path in sorted((ROOT / "data" / "codex").glob("*.jsonl")) for row in read_jsonl(path)]
    print("set E:", json.dumps(build(PARTS, messy, "synthetic"), ensure_ascii=False))
    print("curve pool:", json.dumps(build(POOL_PARTS, messy + codex, "synthetic_pool"), ensure_ascii=False))


if __name__ == "__main__":
    main()
