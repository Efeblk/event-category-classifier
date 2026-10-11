"""Store trained transformer taggers at lower precision and check that their scores hold.

For each model directory (from train_bert.py) it scores three versions on the same dev and test sets as train_bert:
32-bit floats (as trained), 16-bit floats, and 8-bit integers (PyTorch dynamic quantization of the linear layers and
the word embeddings). Size is the bytes of the saved weights. 32- and 16-bit run on the GPU, 8-bit on the CPU, which is
where PyTorch's int8 kernels run. Writes results/quantized_results.json.

Usage: python quantize.py artifacts/bertweet_E artifacts/bertweet_6L_E_kd0.7 artifacts/bertweet_4L_bottom_E_kd0.3
(BERTweet-based models; int8 BERT-mini fails in PyTorch's quantized embedding).
"""

import argparse
import io
import json
from pathlib import Path

import torch
from torch.ao.quantization import float_qparams_weight_only_qconfig, per_channel_dynamic_qconfig, quantize_dynamic
from transformers import AutoModelForTokenClassification, AutoTokenizer

from slots import read_jsonl, repair_bio, span_scores
from train_bert import BATCH, LABELS, encode
from train_classic import load

ROOT = Path(__file__).resolve().parent


def weight_bytes(model):
    """Size of the saved state dict in bytes."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getbuffer().nbytes


@torch.no_grad()
def predict(model, examples, pad_id, device):
    """BIO tags per sentence from each word's first sub-token, like train_bert.predict but on any device."""
    model.eval()
    tags = []
    for start in range(0, len(examples), BATCH):
        chunk = examples[start:start + BATCH]
        n = max(len(e["ids"]) for e in chunk)
        ids = torch.full((len(chunk), n), pad_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), n), dtype=torch.long)
        for i, e in enumerate(chunk):
            ids[i, :len(e["ids"])] = torch.tensor(e["ids"])
            mask[i, :len(e["ids"])] = 1
        best = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.argmax(-1).cpu().tolist()
        for e, row in zip(chunk, best):
            tags.append(repair_bio([LABELS[row[pos]] if pos is not None else "O" for pos in e["word_pos"]]))
    return tags


def int8(model):
    """Dynamic int8 quantization: linear layers get int8 weights with one scale per output row (one scale per layer
    collapsed the scores on the onednn engine), embeddings int8 weights with float scales."""
    torch.backends.quantized.engine = "onednn" if "onednn" in torch.backends.quantized.supported_engines else "x86"
    spec = {torch.nn.Linear: per_channel_dynamic_qconfig, torch.nn.Embedding: float_qparams_weight_only_qconfig}
    return quantize_dynamic(model, spec, dtype=torch.qint8)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("models", nargs="+", type=Path, help="model directories written by train_bert.py")
    args = parser.parse_args()

    sets = {"dev": load("dev") + load("massive_dev"), "SGD test": load("test"), "MASSIVE test": load("massive_test"),
            "messy": read_jsonl(ROOT / "data" / "messy" / "messy_test.jsonl"),
            "codex": read_jsonl(ROOT / "data" / "codex" / "codex_test.jsonl")}
    out = []
    for path in args.models:
        model = AutoModelForTokenClassification.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(model.config.base_model_id)
        examples = {name: encode(tokenizer, rows) for name, rows in sets.items()}
        parameters = sum(p.numel() for p in model.parameters())
        versions = {"float32": (lambda m: m.cuda(), "cuda"), "float16": (lambda m: m.half(), "cuda"),
                    "int8": (lambda m: int8(m.float().cpu()), "cpu")}
        for precision, (convert, device) in versions.items():
            model = convert(model)
            scores = {name: span_scores([r["tags"] for r in rows],
                                        predict(model, examples[name], tokenizer.pad_token_id, device))
                      for name, rows in sets.items()}
            entry = {"model": path.name, "precision": precision, "parameters": parameters,
                     "megabytes": weight_bytes(model) / 1e6, "f1": {name: s["f1"] for name, s in scores.items()}}
            out.append(entry)
            print(f"{path.name:28} {precision:8} {entry['megabytes']:7.1f} MB  " +
                  "  ".join(f"{name} {s['f1']:.3f}" for name, s in scores.items()), flush=True)
        del model
        torch.cuda.empty_cache()

    with open(ROOT / "results" / "quantized_results.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
