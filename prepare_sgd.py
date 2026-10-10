"""Download Google's Schema-Guided Dialogue (SGD) and build BIO-tagged utterances for the Events services.

Raw store: data/raw/sgd/{train,dev,test}.json, one JSON list per split with only the dialogues that have an Events*
service, plus LICENSE.txt. Output: data/clean/{train,dev,test}.jsonl and stats.json.
"""

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

from slots import bio_tags, clean_split, slot_counts, write_lines

BASE = "https://raw.githubusercontent.com/google-research-datasets/dstc8-schema-guided-dialogue/master/"
USER_AGENT = "prepare_sgd.py/1.0 (NLP course project)"
SPLITS = ("train", "dev", "test")
ROOT = Path(__file__).resolve().parent
SLOT_MAP = {"city_of_event": "city"}
IGNORED_SLOTS = {"category", "subcategory", "event_type"}


def download(url):
    """Return the response body, or None when the server answers 404."""
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def has_events(dialogue):
    return any(name.startswith("Events") for name in dialogue["services"])


def fetch_license(raw_dir):
    target = raw_dir / "LICENSE.txt"
    if target.exists():
        return
    for name in ("LICENSE.txt", "LICENSE"):
        data = download(BASE + name)
        if data is not None:
            raw_dir.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            return
    print("warning: no LICENSE file found in the SGD repo")


def fetch_split(split, raw_dir):
    """Download dialogues_NNN.json until a 404 and save the Events dialogues of all files as one raw file."""
    target = raw_dir / f"{split}.json"
    if target.exists():
        return
    dialogues, number = [], 1
    while True:
        data = download(f"{BASE}{split}/dialogues_{number:03d}.json")
        if data is None:
            break
        kept = [d for d in json.loads(data) if has_events(d)]
        dialogues.extend(kept)
        print(f"{split} file {number:03d}: {len(kept)} Events dialogues")
        number += 1
    write_lines(target, [json.dumps(dialogues, ensure_ascii=False)])


def load_dialogues(raw_dir, split):
    return json.loads((raw_dir / f"{split}.json").read_text(encoding="utf-8"))


def normalize_slot(name):
    return SLOT_MAP.get(name, name)


def extract(dialogues):
    """Return (rows, empty_dropped, conflicts) for the USER turns with Events frames."""
    rows, empty, conflicts = [], 0, 0
    for dialogue in dialogues:
        for index, turn in enumerate(dialogue["turns"]):
            if turn["speaker"] != "USER":
                continue
            spans = {
                (s["start"], s["exclusive_end"], normalize_slot(s["slot"]))
                for frame in turn.get("frames", [])
                if frame["service"].startswith("Events")
                for s in frame.get("slots", [])
                if s["slot"] not in IGNORED_SLOTS
            }
            if not spans:
                empty += 1
                continue
            tagged = bio_tags(turn["utterance"], spans)
            if tagged is None:
                conflicts += 1
                continue
            tokens, tags = tagged
            rows.append({
                "id": f"{dialogue['dialogue_id']}:{index}",
                "text": turn["utterance"],
                "tokens": tokens,
                "tags": tags,
            })
    return rows, empty, conflicts


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw" / "sgd")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "clean")
    args = parser.parse_args()

    fetch_license(args.raw_dir)
    for split in SPLITS:
        fetch_split(split, args.raw_dir)

    stats, kept = {}, []
    for split in SPLITS:
        rows, empty, conflicts = extract(load_dialogues(args.raw_dir, split))
        rows, duplicates, leakage = clean_split(rows, kept)
        kept.append(rows)
        stats[split] = {
            "rows": len(rows),
            "empty_dropped": empty,
            "duplicates_dropped": duplicates,
            "leakage_dropped": leakage,
            "conflicts": conflicts,
            "spans": slot_counts(rows),
        }
        write_lines(args.out_dir / f"{split}.jsonl", [json.dumps(r, ensure_ascii=False) for r in rows])

    write_lines(args.out_dir / "stats.json", [json.dumps(stats, indent=2, ensure_ascii=False)])
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
