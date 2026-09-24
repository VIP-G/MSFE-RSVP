import unittest

import torch

from openrec.losses import build_loss


class CTCLossTest(unittest.TestCase):
    def test_ctc_forward_and_backward_are_finite(self):
        loss_fn = build_loss({"name": "CTCLoss", "zero_infinity": True})
        logits = torch.randn(2, 32, 5, requires_grad=True)
        labels = torch.tensor(
            [[1, 2, 3, 4, 0], [2, 3, 1, 0, 0]], dtype=torch.long
        )
        lengths = torch.tensor([4, 3], dtype=torch.long)
        result = loss_fn(logits, [None, labels, lengths])
        loss = result["loss"]
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == "__main__":
    unittest.main()
