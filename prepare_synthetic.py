"""Convert the LLM-written synthetic utterances in data/synthetic/part*.jsonl to BIO-tagged rows for training set E.

Each input line is {"text", "spans": [[slot, substring], ...]}. Every substring is located left to right, after the end
of the previous span, and must sit on token boundaries. Rows whose text contains an event_name span of the messy test
set are dropped (decontamination), as are rows whose text equals a messy test text and duplicate texts.

Output: data/clean/synthetic_train.jsonl and data/clean/synthetic_stats.json.
"""

import json
import re
from pathlib import Path

from slots import TOKEN, bio_tags, dedupe, drop_leaks, read_jsonl, slot_counts, spans, write_lines

ROOT = Path(__file__).resolve().parent
PARTS = ("part1", "part2", "part3", "part4")


def locate(text, substring, after):
    """Return (start, end) of the first occurrence of substring at or after position after."""
    start = text.find(substring, after) if substring else -1
    if start < 0:
        raise ValueError(f"span {substring!r} not found in {text!r} after position {after}")
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


def messy_event_names(messy_rows):
    """Lowercased token tuples of every event_name span in the messy test set."""
    return {tuple(w.lower() for w in r["tokens"][s:e]) for r in messy_rows for kind, s, e in spans(r["tags"])
            if kind == "event_name"}


def has_phrase(tokens, phrase):
    """True when phrase (lowercased tokens) occurs as a contiguous run in tokens (lowercased)."""
    words = [t.lower() for t in tokens]
    n = len(phrase)
    return any(tuple(words[i:i + n]) == phrase for i in range(len(words) - n + 1))


def decontaminate(rows, messy_rows):
    """Drop rows containing a messy event name, then rows whose lowercased text equals a messy text.

    Return (kept, dropped for event names, dropped for messy text).
    """
    names = messy_event_names(messy_rows)
    kept = [r for r in rows if not any(has_phrase(r["tokens"], name) for name in names)]
    by_name = len(rows) - len(kept)
    kept, by_text = drop_leaks(kept, {r["text"].lower() for r in messy_rows})
    return kept, by_name, by_text


def main():
    rows = [row for part in PARTS for row in read_part(ROOT / "data" / "synthetic" / f"{part}.jsonl")]
    messy = read_jsonl(ROOT / "data" / "messy" / "messy_test.jsonl")

    kept, by_name, by_text = decontaminate(rows, messy)
    kept, duplicates = dedupe(kept)
    stats = {
        "rows_in": len(rows),
        "dropped_messy_event_names": by_name,
        "dropped_messy_text": by_text,
        "duplicates_dropped": duplicates,
        "rows_out": len(kept),
        "spans": slot_counts(kept),
    }
    write_lines(ROOT / "data" / "clean" / "synthetic_train.jsonl",
                [json.dumps(r, ensure_ascii=False) for r in kept])
    write_lines(ROOT / "data" / "clean" / "synthetic_stats.json", [json.dumps(stats, indent=2, ensure_ascii=False)])
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
