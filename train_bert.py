"""Fine-tune a pre-trained transformer for slot tagging on training set D (SGD + MASSIVE + typo copies, from
train_classic).

Usage: python train_bert.py [--model bert-base-uncased|roberta-base|vinai/bertweet-base]. The output tag is the model
name's first part (bert, roberta, bertweet).

Labels are O plus B-/I- for city, date and event_name. Only the first sub-token of each word is labelled and its
prediction is used for the word; predictions go through repair_bio. Runs up to 4 epochs of AdamW with linear warmup,
bf16 autocast on CUDA and seed 42. The epoch with the best span micro F1 on SGD dev + MASSIVE dev is kept, then
scored once on SGD test, MASSIVE test and the messy test set. Writes the summary to results/<tag>_results.json, the
per-row messy predictions to reports/<tag>_messy_predictions.csv and the model to artifacts/<tag>/. Requires torch
with a CUDA GPU (see requirements-gpu.txt).
"""

import argparse
import csv
import json
import math
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from torch.optim import AdamW
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_linear_schedule_with_warmup

from slots import describe, read_jsonl, repair_bio, slot_f1, span_scores
from train_classic import load, set_d_rows

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = "bert-base-uncased"
SLOTS = ["city", "date", "event_name"]
LABELS = ["O"] + [f"{prefix}-{slot}" for slot in SLOTS for prefix in ("B", "I")]
LABEL_ID = {label: i for i, label in enumerate(LABELS)}
SEED = 42
LR = 3e-5
WEIGHT_DECAY = 0.01
BATCH = 32
EPOCHS = 4
WARMUP = 0.1
MAX_LEN = 128


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def encode(tokenizer, rows):
    """Per sentence: sub-token ids, labels (first sub-token of each word only, -100 elsewhere) and each word's
    first position.

    Each word is tokenized alone with a leading space (RoBERTa's byte-level BPE needs it; BERT and BERTweet ignore it),
    so one manual alignment works for all three models without word_ids(). Specials are added around the content,
    which is cut to MAX_LEN - 2 sub-tokens like a truncating tokenizer would. A word with no sub-token gets <unk>.
    Words cut off by truncation get None as position and are predicted as O.
    """
    limit = MAX_LEN - 2
    examples, missing = [], 0
    for r in rows:
        content, labels, starts = [], [], []
        for word, tag in zip(r["tokens"], r["tags"]):
            sub = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(" " + word)) or [tokenizer.unk_token_id]
            if len(content) >= limit:
                starts.append(None)
                continue
            starts.append(1 + len(content))
            labels += [LABEL_ID[tag]] + [-100] * (len(sub) - 1)
            content += sub
        content, labels = content[:limit], labels[:limit]
        ids = [tokenizer.cls_token_id] + content + [tokenizer.sep_token_id]
        labels = [-100] + labels + [-100]
        missing += starts.count(None)
        examples.append({"ids": ids, "labels": labels, "word_pos": starts})
    if missing:
        print(f"warning: {missing} words have no sub-token position and will be predicted as O", flush=True)
    return examples


def collate(items, pad_id):
    """Pad a batch to its longest sentence and move it to the GPU."""
    n = max(len(e["ids"]) for e in items)
    ids = torch.full((len(items), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(items), n), dtype=torch.long)
    labels = torch.full((len(items), n), -100, dtype=torch.long)
    for i, e in enumerate(items):
        size = len(e["ids"])
        ids[i, :size] = torch.tensor(e["ids"])
        mask[i, :size] = 1
        labels[i, :size] = torch.tensor(e["labels"])
    return ids.cuda(), mask.cuda(), labels.cuda()


@torch.no_grad()
def predict(model, examples, pad_id):
    """BIO tags per sentence, read from each word's first sub-token and repaired with repair_bio."""
    model.eval()
    tags = []
    for start in range(0, len(examples), BATCH):
        chunk = examples[start:start + BATCH]
        ids, mask, _ = collate(chunk, pad_id)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=mask).logits
        best = logits.argmax(-1).cpu().tolist()
        for e, row in zip(chunk, best):
            tags.append(repair_bio([LABELS[row[pos]] if pos is not None else "O" for pos in e["word_pos"]]))
    return tags


def evaluate(model, examples, rows, pad_id):
    """Span scores against the gold tags of rows, plus the predicted tags."""
    pred = predict(model, examples, pad_id)
    return span_scores([r["tags"] for r in rows], pred), pred


def train(model, train_ex, dev_ex, dev_rows, pad_id):
    """Train for EPOCHS epochs; return the per-epoch history and the epoch whose weights are loaded at the end."""
    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(groups, lr=LR)
    total = math.ceil(len(train_ex) / BATCH) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP * total), total)
    order_rng = torch.Generator().manual_seed(SEED)
    history, best = [], None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        order = torch.randperm(len(train_ex), generator=order_rng).tolist()
        for start in range(0, len(order), BATCH):
            ids, mask, labels = collate([train_ex[i] for i in order[start:start + BATCH]], pad_id)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            losses.append(loss.item())
        dev, _ = evaluate(model, dev_ex, dev_rows, pad_id)
        history.append({"epoch": epoch, "train_loss": sum(losses) / len(losses), "dev_f1": dev["f1"]})
        print(f"epoch {epoch}: train loss {history[-1]['train_loss']:.4f}, dev F1 {dev['f1']:.4f}", flush=True)
        if best is None or dev["f1"] > best[0]:
            best = (dev["f1"], epoch, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[2])
    return history, best[1]


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face model id")
    model_id = parser.parse_args().model
    tag = model_id.split("/")[-1].split("-")[0].lower()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    set_seed()
    results_dir, reports, artifacts = ROOT / "results", ROOT / "reports", ROOT / "artifacts" / tag
    results_dir.mkdir(exist_ok=True)
    reports.mkdir(exist_ok=True)

    train_rows = set_d_rows()
    dev_rows = load("dev") + load("massive_dev")
    tests = {"SGD test": load("test"), "MASSIVE test": load("massive_test"),
             "messy": read_jsonl(ROOT / "data" / "messy" / "messy_test.jsonl")}

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pad_id = tokenizer.pad_token_id
    model = AutoModelForTokenClassification.from_pretrained(
        model_id, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL_ID).cuda()
    train_ex, dev_ex = encode(tokenizer, train_rows), encode(tokenizer, dev_rows)
    test_ex = {name: encode(tokenizer, rows) for name, rows in tests.items()}

    train_started = time.perf_counter()
    history, chosen = train(model, train_ex, dev_ex, dev_rows, pad_id)
    train_seconds = time.perf_counter() - train_started

    scores, messy_pred = {}, None
    for name, rows in tests.items():
        scores[name], pred = evaluate(model, test_ex[name], rows, pad_id)
        if name == "messy":
            messy_pred = pred

    # Saving the BERTweet tokenizer loses its BPE merges, so store the base id and reload the original.
    model.config.base_model_id = model_id
    model.save_pretrained(artifacts)

    gpu = torch.cuda.get_device_name(0)
    results = {
        "model": model_id, "labels": LABELS, "seed": SEED,
        "hyperparameters": {"lr": LR, "weight_decay": WEIGHT_DECAY, "batch": BATCH, "max_epochs": EPOCHS,
                            "warmup": WARMUP, "max_len": MAX_LEN, "grad_clip": 1.0, "precision": "bf16 autocast"},
        "train_sentences": len(train_rows), "dev_sentences": len(dev_rows),
        "epochs": history, "chosen_epoch": chosen, "dev_f1_chosen": history[chosen - 1]["dev_f1"],
        "test": scores, "train_seconds": train_seconds,
        "gpu": gpu,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "cuda": torch.version.cuda, "transformers": transformers.__version__},
    }
    with open(results_dir / f"{tag}_results.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        f.write("\n")

    with open(reports / f"{tag}_messy_predictions.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["id", "text", "gold_spans", "predicted_spans"])
        for r, p in zip(tests["messy"], messy_pred):
            writer.writerow([r["id"], r["text"], describe(r["tokens"], r["tags"]), describe(r["tokens"], p)])

    print()
    head = f"{'SGD test F1':>12}{'MASSIVE F1':>12}{'messy F1':>10}  " + "  ".join(f"{s:<10}" for s in SLOTS)
    print(head)
    print("-" * len(head))
    per_slot = "  ".join(f"{slot_f1(scores['messy'], s):<10}" for s in SLOTS)
    print(f"{scores['SGD test']['f1']:>12.3f}{scores['MASSIVE test']['f1']:>12.3f}"
          f"{scores['messy']['f1']:>10.3f}  {per_slot}")
    print(f"\nchosen epoch {chosen}, train time {train_seconds:.1f}s, total {time.perf_counter() - started:.1f}s, "
          f"GPU {gpu}")


if __name__ == "__main__":
    main()
