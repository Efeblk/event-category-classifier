"""Prepare reproducible article-grouped PTC technique classification data.

Only the standard library is required. Original article bytes and offsets are
preserved; cleaned model inputs normalize Unicode whitespace, not meaning.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import tarfile
import tempfile
from urllib.request import urlopen


URL = "https://zenodo.org/records/3952415/files/datasets-v2.tgz?download=1"
SHA256 = "fd2d2c358d42b2e326e7f6f148007b384f0a0ffac5ab6db21e3889313a8516bb"
SOURCE_PARTITION = "official_train"
PARTITIONS = ("train", "dev", "test")
MEMBER_PATTERN = re.compile(
    r"datasets/(?:train-articles/article[0-9]+\.txt|"
    r"train-labels-task2-technique-classification/article[0-9]+\.task2-TC\.labels|"
    r"README\.md|LICENSE(?:\.txt|\.md)?)"
)
CONTEXT_MARGIN = 350
MAX_CONTEXT_CHARS = 1500
DEVELOPMENT_CONTEXT_MARGIN = 1000


def digest(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def normalize_whitespace(text: str) -> str:
    """Collapse Unicode whitespace while retaining words, case and punctuation."""
    return " ".join(text.split())


def extract_archive(archive: str | Path, destination: str | Path) -> None:
    """Verify before opening the archive; extract whitelisted regular files."""
    if digest(archive) != SHA256:
        raise ValueError("PTC archive SHA256 mismatch; no archive files were read.")
    with tarfile.open(archive, "r:gz") as handle:
        members = [item for item in handle.getmembers() if MEMBER_PATTERN.fullmatch(item.name)]
        names = [item.name for item in members]
        if len(set(names)) != len(names):
            raise ValueError("PTC archive contains duplicate whitelisted paths.")
        if any(not item.isfile() for item in members):
            raise ValueError("PTC articles and annotations must be regular files.")
        articles = [name for name in names if "/train-articles/" in name]
        annotations = [name for name in names if "/train-labels-task2-" in name]
        if not articles or not annotations or "datasets/README.md" not in names:
            raise ValueError("PTC archive is missing required source files.")
        handle.extractall(destination, members=members, filter="data")


def download_archive(path: Path) -> None:
    """Download to a temporary sibling and replace the cache after verification."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and digest(path) == SHA256:
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            with urlopen(URL, timeout=60) as response:
                while chunk := response.read(1024 * 1024):
                    handle.write(chunk)
            handle.close()
            if digest(temporary) != SHA256:
                raise ValueError("Downloaded PTC archive SHA256 mismatch.")
            temporary.replace(path)
        finally:
            handle.close()
            if temporary.exists():
                temporary.unlink()


def read_annotation_file(path: str | Path, expected_article_id: str,
                         text: str) -> list[dict]:
    """Validate source annotations using original Unicode code point offsets."""
    from classifier import LABELS

    records = []
    for line_number, line in enumerate(Path(path).read_bytes().decode("utf-8").splitlines(), 1):
        fields = line.split("\t")
        location = f"{Path(path).name}:{line_number}"
        if len(fields) != 4:
            raise ValueError(f"{location}: expected four tab-separated fields.")
        article_id, label, start_text, end_text = fields
        if article_id != expected_article_id:
            raise ValueError(f"{location}: article ID does not match its source file.")
        if label not in LABELS:
            raise ValueError(f"{location}: unknown technique {label!r}.")
        try:
            start, end = int(start_text), int(end_text)
        except ValueError as error:
            raise ValueError(f"{location}: offsets must be integers.") from error
        if not 0 <= start < end <= len(text):
            raise ValueError(f"{location}: offsets must satisfy 0 <= start < end <= article length.")
        if not normalize_whitespace(text[start:end]):
            raise ValueError(f"{location}: annotation contains only whitespace.")
        records.append({"article_id": article_id, "label": label, "start": start,
                        "end": end, "source_line": line_number})
    return records


def make_context(text: str, start: int, end: int) -> str:
    """Return the excerpt and up to 350 original characters on either side."""
    if not 0 <= start < end <= len(text):
        raise ValueError("Context offsets must lie inside the original article.")
    if end - start > MAX_CONTEXT_CHARS:
        raise ValueError("Source excerpt exceeds the context character limit.")
    allowance = (MAX_CONTEXT_CHARS - (end - start)) // 2
    margin = min(CONTEXT_MARGIN, allowance)
    return normalize_whitespace(text[max(0, start - margin):min(len(text), end + margin)])


def build_rows(raw_dir: str | Path) -> tuple[list[dict], dict, dict[str, str]]:
    """Extract annotations, remove only identical records and retain conflicts."""
    directory = Path(raw_dir)
    article_paths = sorted((directory / "articles").glob("article*.txt"))
    annotation_paths = sorted((directory / "annotations").glob("article*.task2-TC.labels"))
    articles = {re.fullmatch(r"article([0-9]+)\.txt", path.name)[1]:
                path.read_bytes().decode("utf-8") for path in article_paths
                if re.fullmatch(r"article([0-9]+)\.txt", path.name)}
    annotation_ids = {re.fullmatch(r"article([0-9]+)\.task2-TC\.labels", path.name)[1]
                      for path in annotation_paths
                      if re.fullmatch(r"article([0-9]+)\.task2-TC\.labels", path.name)}
    if not articles or set(articles) != annotation_ids:
        raise ValueError("Every source article needs exactly one corresponding annotation file.")
    seen, rows, removed = set(), [], []
    changed_text = changed_context = source_count = 0
    for path in annotation_paths:
        match = re.fullmatch(r"article([0-9]+)\.task2-TC\.labels", path.name)
        if match is None:
            raise ValueError(f"Unexpected annotation filename: {path.name}.")
        article_id = match[1]
        original = articles[article_id]
        for annotation in read_annotation_file(path, article_id, original):
            source_count += 1
            start, end, label = annotation["start"], annotation["end"], annotation["label"]
            identity = (article_id, start, end, label)
            if identity in seen:
                removed.append(annotation)
                continue
            seen.add(identity)
            text = original[start:end]
            context = make_context(original, start, end)
            clean_text = normalize_whitespace(text)
            changed_text += clean_text != text
            raw_context = original[max(0, start - CONTEXT_MARGIN):min(len(original), end + CONTEXT_MARGIN)]
            changed_context += context != raw_context
            rows.append({"id": f"{article_id}:{start}:{end}:{label}",
                         "article_id": article_id, "text": text, "clean_text": clean_text,
                         "context": context, "label": label, "start": start, "end": end,
                         "source_partition": SOURCE_PARTITION})
    rows.sort(key=lambda row: (row["article_id"], row["start"], row["end"], row["label"]))
    spans, inputs, snippets = defaultdict(set), defaultdict(set), defaultdict(set)
    input_counts, snippet_counts = Counter(), Counter()
    for row in rows:
        spans[(row["article_id"], row["start"], row["end"])].add(row["label"])
        model_input = (row["clean_text"], row["context"])
        inputs[model_input].add(row["label"])
        snippets[row["clean_text"]].add(row["label"])
        input_counts[model_input] += 1
        snippet_counts[row["clean_text"]] += 1
    audit = {
        "source_annotation_rows": source_count,
        "cleaned_rows": len(rows), "source_articles": len(articles),
        "supervised_articles": len({row["article_id"] for row in rows}),
        "articles_without_technique_annotations": sorted(set(articles) - {row["article_id"] for row in rows}),
        "exact_duplicate_annotations_removed": len(removed), "removed_annotations": removed,
        "distinct_source_spans": len(spans),
        "multiple_label_source_spans": sum(len(labels) > 1 for labels in spans.values()),
        "whitespace_normalized_snippets": changed_text,
        "whitespace_normalized_contexts": changed_context,
        "normalized_snippet_duplicate_groups": sum(count > 1 for count in snippet_counts.values()),
        "normalized_snippet_conflicting_label_groups": sum(len(labels) > 1 for labels in snippets.values()),
        "identical_model_input_groups": sum(count > 1 for count in input_counts.values()),
        "identical_model_input_conflicting_label_groups": sum(len(labels) > 1 for labels in inputs.values()),
        "invalid_offsets": 0, "hand_changed_labels": 0,
        "normalization": "Unicode whitespace collapse only; exact original excerpts retained in text.",
        "context": f"Up to {CONTEXT_MARGIN} original characters on either side; maximum {MAX_CONTEXT_CHARS} characters.",
    }
    hashes = {article_id: hashlib.sha256(text.encode("utf-8")).hexdigest()
              for article_id, text in articles.items()}
    audit["identical_source_article_groups"] = sum(count > 1 for count in Counter(hashes.values()).values())
    return rows, audit, hashes


def assign_partitions(rows: list[dict], seed: int = 42,
                      article_hashes: dict[str, str] | None = None) -> dict:
    """Assign shuffled article groups without consulting labels or scores."""
    from classifier import LABELS, normalize_text

    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Split seed must be an integer.")
    articles = sorted({row["article_id"] for row in rows})
    parents = {article_id: article_id for article_id in articles}

    def root(article_id):
        while parents[article_id] != article_id:
            parents[article_id] = parents[parents[article_id]]
            article_id = parents[article_id]
        return article_id

    def join(first, second):
        first, second = sorted((root(first), root(second)))
        parents[second] = first

    hashes, inputs = {}, {}
    for article_id in articles:
        article_hash = (article_hashes or {}).get(article_id)
        if article_hash is not None:
            if article_hash in hashes:
                join(article_id, hashes[article_hash])
            hashes[article_hash] = article_id
    for row in rows:
        model_input = (normalize_text(row["text"]), normalize_text(row["context"]))
        if model_input in inputs:
            join(row["article_id"], inputs[model_input])
        inputs[model_input] = row["article_id"]
    groups = defaultdict(list)
    for article_id in articles:
        groups[root(article_id)].append(article_id)
    group_ids = sorted(groups)
    random.Random(seed).shuffle(group_ids)
    train_end, dev_end = int(len(group_ids) * 0.65), int(len(group_ids) * 0.80)
    if not 0 < train_end < dev_end < len(group_ids):
        raise ValueError("At least seven independent article groups are required for all three partitions.")
    assignment = {}
    for index, group_id in enumerate(group_ids):
        partition = "train" if index < train_end else "dev" if index < dev_end else "test"
        for article_id in groups[group_id]:
            assignment[article_id] = partition
    for row in rows:
        row["partition"] = assignment[row["article_id"]]
    train_labels = {row["label"] for row in rows if row["partition"] == "train"}
    missing = sorted(set(LABELS) - train_labels)
    if missing:
        raise ValueError(f"Fixed article-grouped training split lacks techniques: {', '.join(missing)}.")
    label_counts = {partition: dict(sorted(Counter(row["label"] for row in rows
                    if row["partition"] == partition).items())) for partition in PARTITIONS}
    return {
        "seed": seed, "strategy": "Label-independent shuffled article groups; 65/15/20 percent of groups.",
        "source_partition": SOURCE_PARTITION, "official_test_comparable": False,
        "grouping": "Whole articles, identical original article hashes and identical model-normalized (NFKC/lower/whitespace) excerpt/context pairs.",
        "group_count": len(groups),
        "group_counts": {partition: sum(assignment[groups[group_id][0]] == partition
                                       for group_id in groups) for partition in PARTITIONS},
        "article_counts": dict(Counter(assignment.values())),
        "row_counts": dict(Counter(row["partition"] for row in rows)),
        "label_counts": label_counts,
        "missing_labels": {partition: sorted(set(LABELS) - set(label_counts[partition]))
                           for partition in PARTITIONS},
        "article_partitions": dict(sorted(assignment.items())),
        "article_groups": {group_id: groups[group_id] for group_id in sorted(groups)},
    }


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def build_development_rows(data_dir: str | Path = "data") -> tuple[list[dict], dict]:
    """Reconstruct wider-context train/dev rows without opening test sources.

    The original split is metadata, not a new selection decision. The caller
    writes the returned rows/audit separately from the original benchmark data.
    """
    from classifier import LABELS, normalize_text

    directory = Path(data_dir)
    manifest_path = directory / "cleaned" / "splits.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, ValueError) as error:
        raise ValueError("Development preparation needs the original valid split manifest.") from error
    if not isinstance(manifest, dict) or manifest.get("source_partition") != SOURCE_PARTITION:
        raise ValueError("Development preparation requires the official-training source partition.")
    assignments = manifest.get("article_partitions")
    if not isinstance(assignments, dict) or not assignments:
        raise ValueError("Development preparation needs article partition assignments.")
    allowed = {}
    for article_id, partition in assignments.items():
        if not isinstance(article_id, str) or not article_id.isascii() or not article_id.isdigit():
            raise ValueError("Split article IDs must contain ASCII digits only.")
        if partition not in PARTITIONS:
            raise ValueError("Split manifest contains an unsupported partition.")
        if partition in ("train", "dev"):
            allowed[article_id] = partition
    rows, seen, source_hashes, article_hashes = [], set(), {}, defaultdict(set)
    duplicate_count = source_rows = 0
    for article_id, partition in sorted(allowed.items()):
        article_path = directory / "raw" / "articles" / f"article{article_id}.txt"
        annotation_path = directory / "raw" / "annotations" / f"article{article_id}.task2-TC.labels"
        try:
            article_bytes = article_path.read_bytes()
            annotation_bytes = annotation_path.read_bytes()
            article = article_bytes.decode("utf-8")
        except (OSError, ValueError) as error:
            raise ValueError(f"Unable to read UTF-8 {partition} source article {article_id}.") from error
        source_hashes[article_path.relative_to(directory).as_posix()] = hashlib.sha256(article_bytes).hexdigest()
        source_hashes[annotation_path.relative_to(directory).as_posix()] = hashlib.sha256(annotation_bytes).hexdigest()
        article_hashes[hashlib.sha256(article_bytes).hexdigest()].add(partition)
        for annotation in read_annotation_file(annotation_path, article_id, article):
            source_rows += 1
            start, end, label = annotation["start"], annotation["end"], annotation["label"]
            identity = (article_id, start, end, label)
            if identity in seen:
                duplicate_count += 1
                continue
            seen.add(identity)
            text = article[start:end]
            context = normalize_whitespace(article[max(0, start - DEVELOPMENT_CONTEXT_MARGIN):
                                                   min(len(article), end + DEVELOPMENT_CONTEXT_MARGIN)])
            rows.append({"id": f"{article_id}:{start}:{end}:{label}",
                         "article_id": article_id, "text": text,
                         "clean_text": normalize_whitespace(text), "context": context,
                         "label": label, "start": start, "end": end,
                         "partition": partition, "source_partition": SOURCE_PARTITION})
    rows.sort(key=lambda row: (row["article_id"], row["start"], row["end"], row["label"]))
    row_counts = {partition: sum(row["partition"] == partition for row in rows)
                  for partition in ("train", "dev")}
    expected_counts = manifest.get("row_counts", {})
    if not isinstance(expected_counts, dict) or any(
        type(expected_counts.get(partition)) is not int or expected_counts[partition] < 1
        for partition in ("train", "dev")
    ):
        raise ValueError("Split manifest needs positive integer train/dev row counts.")
    if any(not row_counts[partition] or row_counts[partition] != expected_counts.get(partition)
           for partition in row_counts):
        raise ValueError("Development train/dev row counts differ from the original split manifest.")
    if {row["label"] for row in rows if row["partition"] == "train"} != set(LABELS):
        raise ValueError("Development training rows must contain all 14 techniques.")
    article_parts, input_parts = defaultdict(set), defaultdict(set)
    for row in rows:
        article_parts[row["article_id"]].add(row["partition"])
        input_parts[(normalize_text(row["text"]), normalize_text(row["context"]))].add(row["partition"])
    article_overlap = sum(len(parts) > 1 for parts in article_parts.values())
    original_article_overlap = sum(len(parts) > 1 for parts in article_hashes.values())
    input_overlap = sum(len(parts) > 1 for parts in input_parts.values())
    if article_overlap or original_article_overlap or input_overlap:
        raise ValueError("Development articles or identical excerpt/context inputs cross train/dev.")
    audit = {
        "schema_version": 1, "source_partition": SOURCE_PARTITION,
        "purpose": "Wider-context development-only modeling; original benchmark and partitions unchanged.",
        "partitions": ["train", "dev"], "context_margin": DEVELOPMENT_CONTEXT_MARGIN,
        "context": "Exact excerpt plus up to 1,000 original characters on either side; Unicode whitespace collapse only, no artificial context cap.",
        "rows": len(rows), "row_counts": row_counts,
        "article_counts": {partition: sum(partition in parts for parts in article_parts.values())
                           for partition in ("train", "dev")},
        "label_counts": {partition: dict(sorted(Counter(row["label"] for row in rows
                         if row["partition"] == partition).items())) for partition in ("train", "dev")},
        "source_annotation_rows": source_rows,
        "exact_duplicate_annotations_removed": duplicate_count,
        "max_context_characters": max(len(row["context"]) for row in rows),
        "cross_partition_article_overlap": article_overlap,
        "cross_partition_original_article_overlap": original_article_overlap,
        "cross_partition_model_input_overlap": input_overlap,
        "test_rows_loaded": 0, "test_source_files_opened": 0,
        "split_seed": manifest.get("seed"),
        "split_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_file_sha256": dict(sorted(source_hashes.items())),
        "label_order": list(LABELS),
        "inference_requirement": "Use the same selected excerpt and equally broad surrounding context; this is not a new held-out test result.",
    }
    return rows, audit


def prepare(data_dir: str | Path = "data", seed: int = 42) -> dict:
    from classifier import normalize_text

    directory = Path(data_dir)
    archive = directory / "source" / "datasets-v2.tgz"
    download_archive(archive)
    raw = directory / "raw"
    (raw / "articles").mkdir(parents=True, exist_ok=True)
    (raw / "annotations").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temporary:
        extract_archive(archive, temporary)
        source = Path(temporary) / "datasets"
        for path in sorted((source / "train-articles").glob("article*.txt")):
            shutil.copyfile(path, raw / "articles" / path.name)
        for path in sorted((source / "train-labels-task2-technique-classification").glob("*.labels")):
            shutil.copyfile(path, raw / "annotations" / path.name)
        shutil.copyfile(source / "README.md", raw / "SOURCE_README.md")
    rows, audit, hashes = build_rows(raw)
    if audit["source_articles"] != 371 or audit["source_annotation_rows"] != 6129:
        raise ValueError("Pinned PTC source must contain 371 articles and 6,129 annotations.")
    if len(rows) != 6128 or audit["exact_duplicate_annotations_removed"] != 1:
        raise ValueError("Pinned PTC cleanup must remove precisely one identical annotation.")
    splits = assign_partitions(rows, seed, hashes)
    model_partitions = defaultdict(set)
    span_partitions = defaultdict(set)
    for row in rows:
        model_partitions[(normalize_text(row["text"]), normalize_text(row["context"]))].add(row["partition"])
        span_partitions[row["clean_text"]].add(row["partition"])
    audit.update({"source_url": URL, "source_sha256": SHA256,
                  "license": "CC BY 4.0", "split": splits,
                  "cross_partition_article_overlap": 0,
                  "cross_partition_model_input_overlap": sum(len(parts) > 1 for parts in model_partitions.values()),
                  "cross_partition_normalized_snippet_overlap": sum(len(parts) > 1 for parts in span_partitions.values()),
                  "interpretation": "Curated annotated news corpus, not an original web scrape. Custom splits of official training data; no official test reproduction."})
    if audit["cross_partition_model_input_overlap"]:
        raise ValueError("Identical model inputs leaked across partitions.")
    cleaned = directory / "cleaned"
    cleaned.mkdir(parents=True, exist_ok=True)
    (cleaned / "techniques.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8", newline="\n")
    write_json(cleaned / "audit.json", audit)
    write_json(cleaned / "splits.json", splits)
    development_rows, development_audit = build_development_rows(directory)
    (cleaned / "development.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in development_rows),
        encoding="utf-8", newline="\n")
    write_json(cleaned / "development_audit.json", development_audit)
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    arguments = parser.parse_args()
    try:
        result = prepare(arguments.data_dir, arguments.seed)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Preparation failed: {error}\n")
    print(json.dumps({"source_annotations": result["source_annotation_rows"],
                      "cleaned_rows": result["cleaned_rows"],
                      "split_rows": result["split"]["row_counts"],
                      "split_articles": result["split"]["article_counts"]}, indent=2))
