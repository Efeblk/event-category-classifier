"""Tests for the distillation loss in train_bert.py. Skipped when torch is not installed (CI uses the CPU lock file)."""

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HAS_TORCH = (importlib.util.find_spec("torch") is not None
             and importlib.util.find_spec("transformers") is not None)


@unittest.skipUnless(HAS_TORCH, "needs torch and transformers")
class DistillLossTest(unittest.TestCase):
    def setUp(self):
        import torch
        import torch.nn.functional as F

        from train_bert import LABELS, distill_loss
        self.torch, self.F, self.loss = torch, F, distill_loss
        torch.manual_seed(0)
        self.logits = torch.randn(2, 4, len(LABELS))
        self.labels = torch.tensor([[-100, 0, 3, -100], [-100, 1, -100, -100]])
        self.has_soft = self.labels != -100

    def test_alpha_one_is_plain_cross_entropy(self):
        soft = self.torch.full_like(self.logits, 1 / len(self.logits[0, 0]))
        expected = self.F.cross_entropy(self.logits.reshape(-1, self.logits.size(-1)), self.labels.reshape(-1),
                                        ignore_index=-100)
        got = self.loss(self.logits, self.labels, soft, self.has_soft, alpha=1.0)
        self.assertAlmostEqual(got.item(), expected.item(), places=5)

    def test_soft_part_is_zero_when_student_matches_teacher(self):
        soft = self.F.softmax(self.logits / 2.0, dim=-1)
        got = self.loss(self.logits, self.labels, soft, self.has_soft, alpha=0.0, temperature=2.0)
        self.assertAlmostEqual(got.item(), 0.0, places=5)

    def test_soft_part_is_positive_when_they_differ(self):
        soft = self.torch.zeros_like(self.logits)
        soft[..., 0] = 1.0
        got = self.loss(self.logits, self.labels, soft, self.has_soft, alpha=0.0)
        self.assertGreater(got.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
