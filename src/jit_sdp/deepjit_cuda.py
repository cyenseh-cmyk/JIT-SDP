"""CUDA backward for DeepJIT's valid convolution, ReLU and global max pool."""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _input_gradient(winners, upstream, weights, result,
                    LENGTH: tl.constexpr, WIDTH: tl.constexpr,
                    FILTERS: tl.constexpr, K: tl.constexpr,
                    TILE_F: tl.constexpr, TILE_T: tl.constexpr, TILE_D: tl.constexpr):
    row = tl.program_id(0)
    first = tl.program_id(1) * TILE_T
    token = first + tl.arange(0, TILE_T)
    dim = tl.program_id(2) * TILE_D + tl.arange(0, TILE_D)
    filt = tl.arange(0, TILE_F)
    start = tl.load(winners + row * FILTERS + filt, filt < FILTERS, other=-K).to(tl.int32)
    grad = tl.load(upstream + row * FILTERS + filt, filt < FILTERS, other=0)
    dx = tl.full((TILE_T, TILE_D), 0, tl.float32)
    # Each filter routes its gradient through one window per sequence.
    overlaps = (start < first + TILE_T) & (start + K > first) & (grad != 0)
    if tl.sum(overlaps.to(tl.int32), axis=0) > 0:
        for tap in tl.static_range(K):
            contribution = tl.where(token[:, None] == start[None, :] + tap, grad[None, :], 0.)
            offset = (filt[:, None] * K + tap) * WIDTH + dim[None, :]
            w = tl.load(weights + offset, (filt[:, None] < FILTERS) & (dim[None, :] < WIDTH), other=0.)
            dx += tl.dot(contribution, w, input_precision="ieee")
    offset = (row * LENGTH + token[:, None]) * WIDTH + dim[None, :]
    tl.store(result + offset, dx, (token[:, None] < LENGTH) & (dim[None, :] < WIDTH))


@triton.jit
def _weight_partials(x, winners, upstream, partials,
                     ROWS: tl.constexpr, LENGTH: tl.constexpr, WIDTH: tl.constexpr,
                     FILTERS: tl.constexpr, K: tl.constexpr, TILE_N: tl.constexpr, TILE_D: tl.constexpr):
    group = tl.program_id(0)
    filt_tap = tl.program_id(1)
    filt, tap = filt_tap // K, filt_tap % K
    row = group * TILE_N + tl.arange(0, TILE_N)
    dim = tl.program_id(2) * TILE_D + tl.arange(0, TILE_D)
    start = tl.load(winners + row * FILTERS + filt, row < ROWS, other=0).to(tl.int32)
    grad = tl.load(upstream + row * FILTERS + filt, row < ROWS, other=0.)
    offset = (row * LENGTH + start + tap)[:, None] * WIDTH + dim[None, :]
    values = tl.load(x + offset, (row[:, None] < ROWS) & (dim[None, :] < WIDTH), other=0.)
    dw = tl.sum(values * grad[:, None], axis=0)
    tl.store(partials + (group * FILTERS * K + filt_tap) * WIDTH + dim, dw, dim < WIDTH)


class _PooledConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        activation = F.relu(F.conv2d(x.unsqueeze(1), weight, bias).squeeze(3))
        values, winners = F.max_pool1d(activation, activation.shape[2], return_indices=True)
        values = values.squeeze(2)
        ctx.save_for_backward(x, weight, winners.contiguous(), values > 0)
        return values

    @staticmethod
    def backward(ctx, grad_values):
        x, weight, winners, positive = ctx.saved_tensors
        rows, length, width = x.shape
        filters, _, taps, _ = weight.shape
        grad = (grad_values * positive).contiguous()
        dx = torch.empty_like(x, memory_format=torch.contiguous_format)
        groups = triton.cdiv(rows, 32)
        partials = x.new_empty((groups, filters, 1, taps, width))
        # Each tile has one writer. The final sum uses a fixed reduction order.
        with torch.cuda.device(x.device):
            _input_gradient[(rows, triton.cdiv(length, 16), triton.cdiv(width, 32))](
                winners, grad, weight, dx, length, width, filters, taps,
                max(16, triton.next_power_of_2(filters)), 16, 32)
            _weight_partials[(groups, filters * taps, triton.cdiv(width, 32))](
                x, winners, grad, partials, rows, length, width, filters, taps, 32, 32)
        return dx, partials.sum(dim=0), grad.sum(dim=0)


def pooled_convolutions(x, convs):
    x = x.contiguous()
    return torch.cat([_PooledConv.apply(x, conv.weight, conv.bias) for conv in convs], dim=1)


def accelerated_model(base, args):
    class CudaDeepJIT(base):
        def forward_msg(self, x, convs):
            # The short file-level convolution and inference already run efficiently.
            if (torch.is_grad_enabled() and x.is_cuda and x.dtype == torch.float32
                    and not torch.is_autocast_enabled() and x.shape[1] >= 32
                    and self.args.num_filters <= 128 and max(self.args.filter_sizes) <= 8):
                return pooled_convolutions(x, convs)
            return super().forward_msg(x, convs)

    return CudaDeepJIT(args)
