"""Download Amazon MASSIVE en-US and build BIO-tagged utterances for the date, city and event_name slots.

Raw store: data/raw/massive/en-US.jsonl and the license, copied out of the MASSIVE tarball. Output:
data/clean/massive_{train,dev,test}.jsonl and massive_stats.json.
"""

import argparse
import json
import re
import shutil
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

from slots import bio_tags, clean_split, read_jsonl, slot_counts, write_lines

URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz"
USER_AGENT = "prepare_massive.py/1.0 (NLP course project)"
SPLITS = ("train", "dev", "test")
ROOT = Path(__file__).resolve().parent
ARCHIVE_NAME = "amazon-massive-dataset-1.0.tar.gz"
DATA_NAME = "1.0/data/en-US.jsonl"
SLOT_MAP = {"date": "date", "place_name": "city", "event_name": "event_name"}
EVENT_INTENT = "recommendation_events"
ANNOT_RE = re.compile(r"\[(\w+) : ([^\]]*)\]")


def fetch_archive(archive):
    """Download the MASSIVE tarball with a User-Agent header."""
    request = urllib.request.Request(URL, headers={"User-Agent": USER_AGENT})
    archive.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=300) as response, open(archive, "wb") as f:
        shutil.copyfileobj(response, f)


def extract_files(archive, raw_dir):
    """Copy en-US.jsonl and the license out of the tarball, writing only by basename."""
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if member.name == DATA_NAME or name.upper().startswith("LICENSE"):
                raw_dir.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, open(raw_dir / name, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                print(f"extracted {name}")


def ensure_raw(raw_dir):
    if (raw_dir / "en-US.jsonl").exists():
        return
    archive = raw_dir / ARCHIVE_NAME
    if not archive.exists():
        fetch_archive(archive)
    extract_files(archive, raw_dir)
    archive.unlink()


def parse_annot(annot):
    """Return (plain text, [(start, end, slot)]) from a MASSIVE annot_utt string."""
    text, spans, pos = "", [], 0
    for match in ANNOT_RE.finditer(annot):
        text += annot[pos:match.start()]
        value = match.group(2)
        spans.append((len(text), len(text) + len(value), match.group(1)))
        text += value
        pos = match.end()
    return text + annot[pos:], spans


def map_slot(intent, slot):
    """Return the output slot name, or None when the slot is dropped."""
    if slot == "event_name" and intent != EVENT_INTENT:
        return None
    return SLOT_MAP.get(slot)


def extract(rows):
    """Return (split -> kept rows, split -> counters) for en-US rows."""
    kept = {split: [] for split in SPLITS}
    counts = {split: Counter() for split in SPLITS}
    for row in rows:
        split = row["partition"]
        counts[split]["raw_rows"] += 1
        text, raw_spans = parse_annot(row["annot_utt"])
        if text != row["utt"]:
            counts[split]["mismatch"] += 1
            continue
        spans = [(start, end, map_slot(row["intent"], slot)) for start, end, slot in raw_spans]
        spans = [span for span in spans if span[2]]
        if not spans:
            counts[split]["empty_dropped"] += 1
            continue
        tagged = bio_tags(text, spans)
        if tagged is None:
            counts[split]["conflicts"] += 1
            continue
        tokens, tags = tagged
        kept[split].append({"id": f"massive:{row['id']}", "text": text, "tokens": tokens, "tags": tags})
    return kept, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "massive")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "clean")
    args = parser.parse_args()

    ensure_raw(args.raw_dir)
    kept, counts = extract(read_jsonl(args.raw_dir / "en-US.jsonl"))

    stats, final = {}, []
    for split in SPLITS:
        rows, duplicates, leakage = clean_split(kept[split], final)
        final.append(rows)
        stats[split] = {
            "raw_rows": counts[split]["raw_rows"],
            "rows": len(rows),
            "mismatch": counts[split]["mismatch"],
            "empty_dropped": counts[split]["empty_dropped"],
            "duplicates_dropped": duplicates,
            "leakage_dropped": leakage,
            "conflicts": counts[split]["conflicts"],
            "spans": slot_counts(rows),
        }
        write_lines(args.out_dir / f"massive_{split}.jsonl", [json.dumps(r, ensure_ascii=False) for r in rows])

    write_lines(args.out_dir / "massive_stats.json", [json.dumps(stats, indent=2, ensure_ascii=False)])
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
