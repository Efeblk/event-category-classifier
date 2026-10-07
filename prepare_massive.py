"""Prepare the pinned MASSIVE v1.0 Turkish official partitions (no sklearn)."""

import hashlib
import json
import re
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen

from classifier import normalize_text  # classifier imports sklearn only inside functions


URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz"
SHA256 = "7df623fd2d300a4d235d6ee5bd396c9a28258d3a0ccb29abdb054506eba153f8"
MEMBERS = ("1.0/data/tr-TR.jsonl", "1.0/LICENSE")


def parse_annotation(text, annotated):
    """Turn "[type : words]" annotations into whitespace tokens and BIO tags."""
    plain = ""
    spans = []
    end = 0
    for match in re.finditer(r"\[([^\[\]:]+)\s*:\s*(.*?)\]", annotated):
        plain += annotated[end:match.start()]
        start = len(plain)
        plain += match[2]
        spans.append((start, len(plain), match[1].strip()))
        end = match.end()
    plain += annotated[end:]
    if plain != text:
        raise ValueError("De-annotated text does not match utt.")

    tokens = list(re.finditer(r"\S+", text))
    tags = ["O"] * len(tokens)
    for start, end, kind in spans:
        starts_inside = start > 0 and not text[start - 1].isspace()
        ends_inside = end < len(text) and not text[end].isspace()
        if starts_inside or ends_inside:
            raise ValueError("Slot boundary falls inside a whitespace word.")
        indices = [i for i, token in enumerate(tokens) if start <= token.start() and token.end() <= end]
        if not indices:
            raise ValueError("Empty slot span.")
        for offset, index in enumerate(indices):
            tags[index] = ("B-" if offset == 0 else "I-") + kind
    return [token[0] for token in tokens], tags


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def extract_archive(archive, destination):
    """Check the pinned hash, then extract only the two files we need."""
    if digest(archive) != SHA256:
        raise ValueError("MASSIVE archive sha256 mismatch.")
    with tarfile.open(archive, "r:gz") as handle:
        members = [handle.getmember(name) for name in MEMBERS]
        if any(not member.isfile() for member in members):
            raise ValueError("Expected regular dataset and license files.")
        handle.extractall(destination, members=members, filter="data")


def write_audit(directory, audit):
    text = json.dumps(audit, ensure_ascii=False, indent=2) + "\n"
    (directory / "massive_tr.audit.json").write_text(text, encoding="utf-8")


def prepare(data_dir="data"):
    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "massive-1.0.tar.gz"
    if not archive.exists() or digest(archive) != SHA256:
        with urlopen(URL, timeout=60) as response, archive.open("wb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)

    audit = {
        "source_url": URL,
        "sha256": SHA256,
        "license": "CC BY 4.0; Amazon.com Inc.",
        "partition_counts": {},
        "intent_counts": {},
        "slot_counts": {},
        "inside_word_boundaries": 0,
        "annotation_mismatches": 0,
        "boundary_rule": "Reject boundaries inside whitespace words; no such boundaries in pinned tr-TR.",
    }
    rows = []
    partitions, intents, slots = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    duplicates = defaultdict(set)
    with tempfile.TemporaryDirectory() as temporary:
        extract_archive(archive, temporary)
        source = Path(temporary) / MEMBERS[0]
        for line in source.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["locale"] != "tr-TR" or row["partition"] not in {"train", "dev", "test"}:
                raise ValueError("Unexpected locale or partition.")
            try:
                tokens, tags = parse_annotation(row["utt"], row["annot_utt"])
            except ValueError as error:
                # Save the audit so the failing row is counted, then stop.
                key = "inside_word_boundaries" if "boundary" in str(error) else "annotation_mismatches"
                audit[key] += 1
                write_audit(directory, audit)
                raise ValueError(f"Row {row['id']}: {error}") from error

            part = row["partition"]
            partitions[part]["rows"] += 1
            intents[part][row["intent"]] += 1
            slots[part].update(tag[2:] for tag in tags if tag.startswith("B-"))
            duplicates[normalize_text(row["utt"])].add(part)
            rows.append({
                "id": row["id"],
                "partition": part,
                "intent": row["intent"],
                "scenario": row["scenario"],
                "text": row["utt"],
                "tokens": tokens,
                "tags": tags,
            })
        (directory / "LICENSE").write_bytes((Path(temporary) / MEMBERS[1]).read_bytes())

    all_intents = set().union(*(set(counts) for counts in intents.values()))
    audit["missing_intents"] = {part: sorted(all_intents - set(counts)) for part, counts in intents.items()}
    audit["official_split_note"] = "Dev lacks audio_volume_other; test lacks cooking_query. Keep official rows and 60 output labels."
    audit.update(
        partition_counts={part: counts["rows"] for part, counts in partitions.items()},
        intent_counts=dict(intents),
        slot_counts=dict(slots),
        cross_partition_normalized_duplicates=sum(len(found) > 1 for found in duplicates.values()),
    )
    lines = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    (directory / "massive_tr.jsonl").write_text(lines, encoding="utf-8")
    write_audit(directory, audit)
    return audit


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=True, indent=2))
