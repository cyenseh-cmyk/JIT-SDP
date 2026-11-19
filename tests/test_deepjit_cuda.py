"""CUDA regression coverage for routing max-pool gradients to the winning windows."""
import importlib.util
import unittest

import torch
import torch.nn.functional as F


@unittest.skipUnless(torch.cuda.is_available() and torch.version.cuda is not None
                     and torch.cuda.get_device_capability()[0] >= 8
                     and importlib.util.find_spec("triton") is not None, "requires NVIDIA SM80+ and Triton")
class PooledConvTests(unittest.TestCase):
    def test_gradients_match_cpu_with_ties_padding_and_partial_tiles(self):
        from jit_sdp.deepjit_cuda import _PooledConv

        generator = torch.Generator().manual_seed(11)
        def integers(shape):
            return torch.randint(-2, 3, shape, generator=generator).float()

        for rows, length, width, filters in ((1, 5, 7, 3), (35, 19, 33, 27)):
            for taps in (1, 2, 3):
                with self.subTest(rows=rows, taps=taps):
                    x = integers((rows, length, width))
                    x[:, length // 2:] = 1
                    weight = integers((filters, 1, taps, width))
                    bias = integers((filters,))
                    bias[0] = -4096
                    weight[1] = 0
                    bias[1] = 0
                    direction = integers((rows, filters))
                    expected = None
                    for device in ("cpu", "cuda", "cuda"):
                        args = [v.to(device).detach().requires_grad_() for v in (x, weight, bias)]
                        if device == "cpu":
                            activation = F.relu(F.conv2d(args[0].unsqueeze(1), *args[1:]).squeeze(3))
                            output = F.max_pool1d(activation, activation.shape[2]).squeeze(2)
                        else:
                            output = _PooledConv.apply(*args)
                        grads = torch.autograd.grad(output, args, direction.to(device))
                        actual = [v.detach().cpu() for v in (output, *grads)]
                        if expected is None:
                            expected = actual
                        else:
                            for ref, value in zip(expected, actual):
                                torch.testing.assert_close(value, ref, atol=0, rtol=0)
