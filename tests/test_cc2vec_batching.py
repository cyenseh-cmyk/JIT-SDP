"""Check that batching lines preserves sequence order, state routing and gradients."""
import copy
from pathlib import Path
import unittest

import torch

from jit_sdp.config import load
from jit_sdp.neural import representation


class CC2VecBatchingTests(unittest.TestCase):
    def test_batched_encoder_matches_sequential_encoder(self):
        cfg = load(Path(__file__).resolve().parents[1] / "configs/smoke.json")
        state = {"code": list(range(24)), "message": list(range(12))}
        generator = torch.Generator().manual_seed(37)
        sequential = representation(cfg, state).double()
        fast_cfg = copy.deepcopy(cfg)
        fast_cfg["cc2vec"]["batch_lines"] = True
        batched = representation(fast_cfg, state).double()
        batched.load_state_dict(sequential.state_dict())
        for batch in (1, 3):
            with self.subTest(batch=batch):
                tokens = torch.randint(24, (batch, 2, 3, 7), generator=generator)
                tokens[:, 0, 1:] = 0
                hidden = cfg["cc2vec"]["hidden_size"]
                states = [torch.randn(2, batch, hidden, dtype=torch.float64, generator=generator)
                          for _ in range(3)]
                direction = torch.randn(batch, 2 * hidden, dtype=torch.float64, generator=generator)
                def differentiate(model):
                    model.zero_grad(set_to_none=True)
                    output = model.forward_code(tokens, states)
                    (output * direction).sum().backward()
                    return output, {k: p.grad for k, p in model.named_parameters() if p.grad is not None}

                expected, ref_grads = differentiate(sequential)
                actual, grads = differentiate(batched)
                torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-9)
                self.assertEqual(grads.keys(), ref_grads.keys())
                for name in grads:
                    torch.testing.assert_close(grads[name], ref_grads[name], atol=1e-10, rtol=1e-9)
