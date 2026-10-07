"""Prepare the pinned MASSIVE v1.0 Turkish official partitions (no sklearn)."""
import hashlib
import json
import re
import tarfile
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlopen

URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz"
SHA256 = "7df623fd2d300a4d235d6ee5bd396c9a28258d3a0ccb29abdb054506eba153f8"
MEMBERS = ("1.0/data/tr-TR.jsonl", "1.0/LICENSE")


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).translate(str.maketrans({"I": "?", "?": "i"})).lower().split())


def parse_annotation(text, annotated):
    """Parse exact whitespace-aligned slots; reject partial-word boundaries."""
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
        if (start and not text[start-1].isspace()) or (end < len(text) and not text[end].isspace()):
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
    if digest(archive) != SHA256:
        raise ValueError("MASSIVE archive sha256 mismatch.")
    with tarfile.open(archive, "r:gz") as handle:
        members = [handle.getmember(name) for name in MEMBERS]
        if any(not member.isfile() for member in members):
            raise ValueError("Expected regular dataset and license files.")
        handle.extractall(destination, members=members, filter="data")


def prepare(data_dir="data"):
    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / "massive-1.0.tar.gz"
    if not archive.exists() or digest(archive) != SHA256:
        with urlopen(URL, timeout=60) as response, archive.open("wb") as handle:
            while chunk := response.read(1024*1024):
                handle.write(chunk)
    audit = {"source_url": URL, "sha256": SHA256, "license": "CC BY 4.0; Amazon.com Inc.",
             "partition_counts": {}, "intent_counts": {}, "slot_counts": {},
             "inside_word_boundaries": 0, "annotation_mismatches": 0,
             "boundary_rule": "Reject boundaries inside whitespace words; no such boundaries in pinned tr-TR."}
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
                key = "inside_word_boundaries" if "boundary" in str(error) else "annotation_mismatches"
                audit[key] += 1
                (directory / "massive_tr.audit.json").write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
                raise ValueError(f"Row {row['id']}: {error}") from error
            part = row["partition"]
            partitions[part]["rows"] += 1
            intents[part][row["intent"]] += 1
            slots[part].update(tag[2:] for tag in tags if tag.startswith("B-"))
            duplicates[normalized(row["utt"])].add(part)
            rows.append({key: row[key] for key in ("id", "partition", "intent", "scenario")} | {
                "text": row["utt"], "tokens": tokens, "tags": tags})
        (directory / "LICENSE").write_bytes((Path(temporary) / MEMBERS[1]).read_bytes())
    audit.update(partition_counts={p: c["rows"] for p,c in partitions.items()},
                 intent_counts=dict(intents), slot_counts=dict(slots),
                 cross_partition_normalized_duplicates=sum(len(parts)>1 for parts in duplicates.values()))
    (directory / "massive_tr.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")
    (directory / "massive_tr.audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return audit


if __name__ == "__main__":
    print(json.dumps(prepare(), ensure_ascii=True, indent=2))
