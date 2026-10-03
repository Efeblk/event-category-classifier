"""Build a frozen text dataset from an existing event snapshot. No network calls."""

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

LABELS = {"Konser": "concert", "Tiyatro": "theatre", "Stand-up": "stand_up"}
COLUMNS = ["id", "text", "label", "group_id", "title", "source", "source_url"]


def normalize(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(str.maketrans({"I": "ı", "İ": "i"})).lower()
    return re.sub(r"\W+", " ", text).strip()


def fingerprint(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def page_url(value):
    parsed = urlsplit(value)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", ""))


def prepare(records, min_words=3):
    """Deduplicate text. Keep connected event families in one data partition."""
    parent = list(range(len(records)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    seen = {}
    texts = {}
    rejected = Counter()
    for index, event in enumerate(records):
        if not isinstance(event, dict) or event.get("category") not in LABELS:
            rejected["unsupported_label"] += 1
            continue
        title = str(event.get("title") or "").strip()
        description = str(event.get("description") or "").strip()
        # Do not count the title twice when the source repeats it as the description.
        text = title if normalize(title) == normalize(description) else f"{title}\n{description}".strip()
        texts[index] = text
        keys = [("text", normalize(text)), ("title", normalize(title))]
        if len(normalize(description).split()) >= 8:
            keys.append(("description", normalize(description)))
        for field in ("productionKey", "url"):
            value = str(event.get(field) or "")
            if value:
                keys.append((field, page_url(value) if field == "url" else value))
        for kind, value in keys:
            if not value:
                continue
            key = (kind, value)
            if key in seen:
                parent[find(index)] = find(seen[key])
            else:
                seen[key] = index

    families = defaultdict(list)
    for index in texts:
        families[find(index)].append(index)
    output = []
    for members in families.values():
        # A provider disagreement is unknown ground truth. Exclude the whole family.
        if len({records[index]["category"] for index in members}) != 1:
            rejected["conflicting_family_records"] += len(members)
            continue
        group_id = fingerprint("\n".join(sorted({normalize(texts[index]) for index in members})))
        unique = set()
        # Keep the same representative when the snapshot order changes.
        for index in sorted(members, key=lambda item: (
            texts[item], str(records[item].get("title") or ""),
            str(records[item].get("source") or ""), str(records[item].get("url") or ""),
        )):
            event, text = records[index], texts[index]
            normalized = normalize(text)
            if len(normalized.split()) < min_words:
                rejected["short_text_records"] += 1
                continue
            if normalized in unique:
                rejected["duplicate_session_records"] += 1
                continue
            unique.add(normalized)
            output.append({
                "id": fingerprint(normalized), "text": text,
                "label": LABELS[event["category"]], "group_id": group_id,
                "title": str(event.get("title") or ""),
                "source": str(event.get("source") or ""),
                "source_url": str(event.get("url") or ""),
            })
    output.sort(key=lambda row: row["id"])
    audit = {
        "input_session_records": len(records), "unique_examples": len(output),
        "event_families": len({row["group_id"] for row in output}),
        "class_counts": dict(sorted(Counter(row["label"] for row in output).items())),
        "excluded_records": dict(sorted(rejected.items())), "min_words": min_words,
        "labels": LABELS,
        "label_origin": "Existing collector categories. No independent human label review.",
        "license_status": "No explicit license for provider text was supplied. Corpus stays outside Git.",
        "group_rule": "Connected normalized title, exact text, long exact description, productionKey or page URL.",
    }
    return output, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/events.csv"))
    parser.add_argument("--min-words", type=int, default=3)
    args = parser.parse_args()
    if args.min_words < 1:
        parser.error("--min-words must be positive")
    raw = args.snapshot.read_bytes()
    records = json.loads(raw)
    if not isinstance(records, list):
        parser.error("Snapshot must be a JSON array of event records")
    rows, audit = prepare(records, args.min_words)
    if len(rows) < 1000 or set(audit["class_counts"]) != set(LABELS.values()):
        parser.error(f"Need at least 1,000 distinct examples and all three labels. Found {audit['unique_examples']}.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    audit["snapshot_sha256"] = hashlib.sha256(raw).hexdigest()
    audit["dataset_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
