"""Download public Gametime event listings and write the labeled title corpus.

Source: rebrowser/gametime-dataset on Hugging Face (CC BY-NC 4.0), pinned to one
commit so the same files are read on every run. Only the event title is used as
text. The ticket site's own category is the label.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen


# Keep this copy in sync with classifier.LABELS. This script does not import sklearn.
LABELS = ("concert", "theatre", "stand_up")
CATEGORY_TO_LABEL = {"music": "concert", "theater": "theatre", "comedy": "stand_up"}
COLUMNS = ("id", "text", "label", "group_id", "title", "source", "source_url")
SOURCE = "gametime"
DATASET = "rebrowser/gametime-dataset"
REVISION = "f24a84e1bedee99bef273332fe705cc7ed1958be"
LICENSE = "CC BY-NC 4.0"

# Ticketing notes such as "(21+ Event)" or "(Rescheduled from 3/28)" are not about the event.
ADMIN_NOTE = re.compile(
    r"\((?:\d+\+(?: event)?|\d+ event|rescheduled[^)]*|moved from[^)]*|[\d/ -]+|"
    r"open caption|audio described[^)]*|preview|early show|late show|night \d+|"
    r"multiple dates[^)]*)\)",
    re.IGNORECASE,
)
# Gametime files film festival screenings under "theater". Films are not one of our categories.
FILM_SCREENING = re.compile(r"\bfilm festival\b", re.IGNORECASE)


def clean_title(title: str) -> str:
    text = ADMIN_NOTE.sub(" ", unicodedata.normalize("NFKC", title))
    return re.sub(r"\s+", " ", text).strip(" -")


def normalize(text: str) -> str:
    """Return stable lowercase text for duplicate checks and row IDs."""
    return re.sub(r"[^\w]+", " ", unicodedata.normalize("NFKC", text).lower()).strip()


def fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_url(path: str) -> str:
    return f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{path}"


def download_events() -> list[tuple[str, dict[str, str]]]:
    """Read every daily events file at the pinned revision."""
    listing = f"https://huggingface.co/api/datasets/{DATASET}/tree/{REVISION}/events/data"
    with urlopen(listing, timeout=30) as response:
        paths = sorted(item["path"] for item in json.load(response)
                       if item["path"].endswith(".csv"))
    events = []
    for path in paths:
        with urlopen(file_url(path), timeout=60) as response:
            text = response.read().decode("utf-8")
        events.extend((path, row) for row in csv.DictReader(io.StringIO(text)))
    return events


def build_rows(events: list[tuple[str, dict[str, str]]]) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Keep the three categories, then drop duplicates and conflicting labels."""
    by_event: dict[str, tuple[str, dict[str, str]]] = {}
    film_screenings = set()
    for path, event in events:
        if event.get("category") not in CATEGORY_TO_LABEL:
            continue
        if FILM_SCREENING.search(event["name"]):
            film_screenings.add(event["eventId"])
            continue
        by_event[event["eventId"]] = (path, event)

    # One row per cleaned title. Repeated dates of one show become one example.
    by_text: dict[str, list[tuple[str, dict[str, str]]]] = defaultdict(list)
    for path, event in by_event.values():
        key = normalize(clean_title(event["name"]))
        if key:
            by_text[key].append((path, event))

    rows: list[dict[str, str]] = []
    conflicting_titles = 0
    for key, matches in by_text.items():
        labels = {CATEGORY_TO_LABEL[event["category"]] for _, event in matches}
        if len(labels) > 1:
            conflicting_titles += 1
            continue
        path, event = min(matches, key=lambda match: match[1]["eventId"])
        performers = sorted(ast.literal_eval(event["performerIds"] or "[]"))
        # Titles by the same performer stay in one partition. Events without one use the title.
        group = performers[0] if performers else "title-" + fingerprint(key)[:16]
        rows.append({
            "id": fingerprint(key),
            "text": clean_title(event["name"]),
            "label": labels.pop(),
            "group_id": group,
            "title": event["name"],
            "source": SOURCE,
            "source_url": file_url(path),
        })

    labels_by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        labels_by_group[row["group_id"]].add(row["label"])
    mixed = {group for group, labels in labels_by_group.items() if len(labels) > 1}
    dropped_mixed = sum(row["group_id"] in mixed for row in rows)
    rows = [row for row in rows if row["group_id"] not in mixed]
    rows.sort(key=lambda row: row["id"])

    audit: dict[str, object] = {
        "unique_examples": len(rows),
        "groups": len({row["group_id"] for row in rows}),
        "class_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        "events_in_categories": len(by_event),
        "dropped_film_festival_events": len(film_screenings),
        "dropped_conflicting_titles": conflicting_titles,
        "dropped_rows_in_mixed_label_groups": dropped_mixed,
        "labels": list(LABELS),
        "category_mapping": CATEGORY_TO_LABEL,
        "languages": ["English"],
        "provenance": f"Public Gametime event listings from {DATASET} on Hugging Face.",
        "source_revision": REVISION,
        "license": LICENSE,
        "text_field": "Event title with ticketing notes removed.",
        "group_rule": "Rows share a group_id when the first listed performer is the same.",
    }
    return rows, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/events.csv"))
    args = parser.parse_args()

    rows, audit = build_rows(download_events())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    audit["output"] = str(args.output)
    audit["dataset_sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(".audit.json").write_text(
        json.dumps(audit, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
