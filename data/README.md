# Dataset

Run `prepare_requests.py` to create `requests.csv`. The script makes no network or paid API calls.

The public prototype contains 364 original synthetic requests. It has 208 Turkish examples and 156 English examples. Each label has 91 examples. The labels are `concert`, `theatre`, `stand_up`, and `unclear`.

The corpus contains 52 intent families. Each family has a stable `class-NN` group ID. Turkish and English phrasings of the same intent use the same group ID. Training keeps each family in one partition.

The source value is `generated-bootstrap`. The `title` and `source_url` fields are empty. The CSV columns are:

```text
id,text,label,group_id,title,source,source_url
```

The row ID is the SHA-256 hash of normalized text. The generator rejects normalized duplicates and label conflicts.

## Provenance and license

An AI assistant authored the prototype examples for this repository. They are not observed real-user requests. They contain no copied ticket-provider text.

The original examples use the [CC0-1.0 license](LICENSE). This license statement covers the authored dataset examples. It does not change the license of application code or archived third-party material.

The current corpus does not satisfy the course requirement for at least 1,000 examples. Do not describe it as user-collected data. Do not claim measured real-user accuracy.

The original provider-description experiment is in `archive/event-descriptions/`. The first request-model run is in `archive/request-prototype-v1/`. The active training process does not use the old provider corpus.

## Challenge cases

`request_challenge.json` contains 40 additional AI-authored cases. It includes Turkish and English requests for all four labels. It covers exclusions, mixed intent, price-only text, music playback, movies, spelling errors, and polite language.

An AI assistant inspected these cases while it fixed guard defects. Therefore, the file is a regression set. It is not a blind benchmark or a real-user evaluation set.

## Move toward 1,000 examples

Confirm the accepted data method with the teacher. If human-authored synthetic requests are acceptable, five team members can each write 200 requests under one label guide.

Record the author code, consent, source, license, language, label, and paraphrase family. Review mixed, vague, unsupported, and negated cases with a second person. Keep translations and paraphrases in one family. Reserve a human-reviewed evaluation set before model and guard changes.
