# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from re import I
from typing import Any, List, Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn, normal

from .args import ModelArgs

# **NOTE**: This code is not runnable without installing `torch` and `fairscale`
# dependencies. These dependencies are not part of the default dependencies
# (requirements.txt) of the `llama-models` package.


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def apply_scaling(freqs: torch.Tensor) -> torch.Tensor:
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * torch.pi / freqs
    new_freqs = torch.where(wavelen > low_freq_wavelen, freqs / scale_factor, freqs)
    smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    return torch.where(
        (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen),
        (1 - smooth) * new_freqs / scale_factor + smooth * new_freqs,
        new_freqs,
    )


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False):
    # --- THE FIX: Explicitly set device="cpu" so it escapes the meta trap ---
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device="cpu")[: (dim // 2)].float() / dim))
    # ------------------------------------------------------------------------

    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(
        self,
        args: ModelArgs,
        kv_cache_compressor: Optional[Any] = None,
        outlier_compressor: Optional[Any] = None,
        normal_compressor: Optional[Any] = None
    ):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        world_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // world_size
        self.n_local_kv_heads = self.n_kv_heads // world_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )
        self.use_compressed_kv_cache = args.use_compressed_kv_cache
        self.kv_cache_compressor: Optional[Any] = kv_cache_compressor
        self.outlier_compressor: Optional[Any] = outlier_compressor
        self.normal_compressor: Optional[Any] = normal_compressor

        self.is_mixed_precision = outlier_compressor is not None and normal_compressor is not None
        self.bypass_kv_cache = False

        self.kv_cache_block_size = 0
        self.kv_cache_bit_width = 0
        self.kv_cache_n_blocks = 0
        self.kv_cache_bstring_bytes = 0
        self.kv_cache_qjl_bytes = 0

        self.cache_k = None
        self.cache_v = None
        self.cache_k_bstring = None
        self.cache_k_qjl = None
        self.cache_k_original_l2 = None
        self.cache_k_residual_l2 = None
        self.cache_v_bstring = None
        self.cache_v_qjl = None
        self.cache_v_original_l2 = None
        self.cache_v_residual_l2 = None

        if self.use_compressed_kv_cache:
            if self.is_mixed_precision:
                self.outlier_dim = self.outlier_compressor.block_size
                self.normal_dim = self.normal_compressor.block_size

                # Outlier Channels Allocation
                self.cache_k_bstring_outlier, self.cache_k_qjl_outlier, self.cache_k_orig_outlier, self.cache_k_res_outlier = self._allocate_cache(self.outlier_compressor, args)
                self.cache_v_bstring_outlier, self.cache_v_qjl_outlier, self.cache_v_orig_outlier, self.cache_v_res_outlier = self._allocate_cache(self.outlier_compressor, args)

                # Normal Channels Allocation
                self.cache_k_bstring_normal, self.cache_k_qjl_normal, self.cache_k_orig_normal, self.cache_k_res_normal = self._allocate_cache(self.normal_compressor, args)
                self.cache_v_bstring_normal, self.cache_v_qjl_normal, self.cache_v_orig_normal, self.cache_v_res_normal = self._allocate_cache(self.normal_compressor, args)
            else:
                # Standard Unified Allocation
                self.cache_k_bstring, self.cache_k_qjl, self.cache_k_orig, self.cache_k_res = self._allocate_cache(self.kv_cache_compressor, args)
                self.cache_v_bstring, self.cache_v_qjl, self.cache_v_orig, self.cache_v_res = self._allocate_cache(self.kv_cache_compressor, args)
        else:
            self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim))
            self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim))

    def _allocate_cache(self, compressor, args):
        """Helper to cleanly build the massive tensors based on the dynamic budget"""
        b_bytes = (int(compressor.bit_width) * int(compressor.block_size) + 7) // 8
        q_bytes = (int(compressor.block_size) + 7) // 8
        shape = (args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, 1) # n_blocks is 1 since slice matches block_size

        return (
            torch.zeros((*shape, b_bytes), dtype=torch.uint8),
            torch.zeros((*shape, q_bytes), dtype=torch.uint8),
            torch.zeros(shape, dtype=torch.float32),
            torch.zeros(shape, dtype=torch.float32)
        )


    def _store_compressed_cache(
        self,
        tensor: torch.Tensor,
        bsz: int,
        seqlen: int,
        start_pos: int,
        compressor,
        c_bstring: torch.Tensor,
        c_qjl: torch.Tensor,
        c_orig: torch.Tensor,
        c_res: torch.Tensor,
    ) -> None:
        """Vectorized storage: Uses bulk conversion to avoid Python loops."""
        tensor = tensor.float().contiguous()

        # Flatten input tensor for compression
        blocks = list(torch.unbind(tensor.view(-1, compressor.block_size), dim=0))

        compressed_results = compressor.compress_chunk(blocks)

        # Prevent Python's Garbage Collector from destroying tensor_f32
        # before the asynchronous CUDA kernel finishes reading it!
        if tensor.device.type == "cuda":
            torch.cuda.synchronize()

        # Bulk convert bytes into flat buffers using list comprehensions (faster than loops)
        all_bstrings = b"".join([res[2] for res in compressed_results])
        all_qjls = b"".join([res[3] for res in compressed_results])

        b_bytes = (int(compressor.bit_width) * int(compressor.block_size) + 7) // 8
        q_bytes = (int(compressor.block_size) + 7) // 8

        # Convert to tensors and reshape to match the cache structure
        # Shape: (bsz, seqlen, heads, n_blocks, ...)
        b_tensor = torch.frombuffer(bytearray(all_bstrings), dtype=torch.uint8).view(
            bsz, seqlen, self.n_local_kv_heads, 1, b_bytes
        )
        q_tensor = torch.frombuffer(bytearray(all_qjls), dtype=torch.uint8).view(
            bsz, seqlen, self.n_local_kv_heads, 1, q_bytes
        )

        # Original and Residual L2s
        orig_l2 = torch.tensor([res[0] for res in compressed_results], dtype=torch.float32).view(
            bsz, seqlen, self.n_local_kv_heads, 1
        )
        res_l2 = torch.tensor([res[1] for res in compressed_results], dtype=torch.float32).view(
            bsz, seqlen, self.n_local_kv_heads, 1
        )

        # Batch assignment to the cache slices
        end_pos = start_pos + seqlen
        c_bstring[:bsz, start_pos:end_pos] = b_tensor.to(c_bstring.device)
        c_qjl[:bsz, start_pos:end_pos] = q_tensor.to(c_qjl.device)
        c_orig[:bsz, start_pos:end_pos] = orig_l2.to(c_orig.device)
        c_res[:bsz, start_pos:end_pos] = res_l2.to(c_res.device)

    def _fetch_decompressed_cache(
        self,
        bsz: int,
        cache_len: int,
        compressor: Any,
        c_bstring: torch.Tensor,
        c_qjl: torch.Tensor,
        c_orig: torch.Tensor,
        c_res: torch.Tensor,
        target_device: torch.device,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Vectorized fetching: Converts cache slices to C-input in bulk."""
        # 1. Slice and flatten everything to CPU
        o_flat = c_orig[:bsz, :cache_len].reshape(-1).cpu().tolist()
        r_flat = c_res[:bsz, :cache_len].reshape(-1).cpu().tolist()

        b_bytes = (int(compressor.bit_width) * int(compressor.block_size) + 7) // 8
        q_bytes = (int(compressor.block_size) + 7) // 8

        # Reshape bitstrings to (N, BytesPerBlock) before calling .numpy()
        b_flat = c_bstring[:bsz, :cache_len].contiguous().reshape(-1, b_bytes).cpu().numpy()
        q_flat = c_qjl[:bsz, :cache_len].contiguous().reshape(-1, q_bytes).cpu().numpy()

        # 2. Reconstruct the list of tuples for the decompressor
        compressed_blocks = [
            (o_flat[i], r_flat[i], b_flat[i].tobytes(), q_flat[i].tobytes())
            for i in range(len(o_flat))
        ]

        # 3. Batch Decompress (The C-API Multi-threaded call)
        decompressed_blocks = compressor.decompress_chunk(compressed_blocks)

        # 4. Stack and view back to (B, S, H, D)
        stacked = torch.stack(decompressed_blocks, dim=0).to(device=target_device, dtype=target_dtype)

        return stacked.view(bsz, cache_len, self.n_local_kv_heads, compressor.block_size)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if getattr(self, "bypass_kv_cache", False):
            keys = xk
            values = xv
        elif self.use_compressed_kv_cache:
            cache_len = start_pos + seqlen

            if self.is_mixed_precision:
                # 1. Split Keys and Values by dimensions
                xk_out = xk[..., :self.outlier_dim]
                xk_norm = xk[..., self.outlier_dim:]
                xv_out = xv[..., :self.outlier_dim]
                xv_norm = xv[..., self.outlier_dim:]

                # 2. Store independently
                self._store_compressed_cache(xk_out, bsz, seqlen, start_pos, self.outlier_compressor, self.cache_k_bstring_outlier, self.cache_k_qjl_outlier, self.cache_k_orig_outlier, self.cache_k_res_outlier)
                self._store_compressed_cache(xk_norm, bsz, seqlen, start_pos, self.normal_compressor, self.cache_k_bstring_normal, self.cache_k_qjl_normal, self.cache_k_orig_normal, self.cache_k_res_normal)
                self._store_compressed_cache(xv_out, bsz, seqlen, start_pos, self.outlier_compressor, self.cache_v_bstring_outlier, self.cache_v_qjl_outlier, self.cache_v_orig_outlier, self.cache_v_res_outlier)
                self._store_compressed_cache(xv_norm, bsz, seqlen, start_pos, self.normal_compressor, self.cache_v_bstring_normal, self.cache_v_qjl_normal, self.cache_v_orig_normal, self.cache_v_res_normal)

                # 3. Fetch independently
                k_out = self._fetch_decompressed_cache(bsz, cache_len, self.outlier_compressor, self.cache_k_bstring_outlier, self.cache_k_qjl_outlier, self.cache_k_orig_outlier, self.cache_k_res_outlier, xq.device, xq.dtype)
                k_norm = self._fetch_decompressed_cache(bsz, cache_len, self.normal_compressor, self.cache_k_bstring_normal, self.cache_k_qjl_normal, self.cache_k_orig_normal, self.cache_k_res_normal, xq.device, xq.dtype)
                v_out = self._fetch_decompressed_cache(bsz, cache_len, self.outlier_compressor, self.cache_v_bstring_outlier, self.cache_v_qjl_outlier, self.cache_v_orig_outlier, self.cache_v_res_outlier, xq.device, xq.dtype)
                v_norm = self._fetch_decompressed_cache(bsz, cache_len, self.normal_compressor, self.cache_v_bstring_normal, self.cache_v_qjl_normal, self.cache_v_orig_normal, self.cache_v_res_normal, xq.device, xq.dtype)

                # 4. Concatenate back to 128-dim head
                keys = torch.cat([k_out, k_norm], dim=-1)
                values = torch.cat([v_out, v_norm], dim=-1)

            else:
                self._store_compressed_cache(xk, bsz, seqlen, start_pos, self.kv_cache_compressor, self.cache_k_bstring, self.cache_k_qjl, self.cache_k_orig, self.cache_k_res)
                self._store_compressed_cache(xv, bsz, seqlen, start_pos, self.kv_cache_compressor, self.cache_v_bstring, self.cache_v_qjl, self.cache_v_orig, self.cache_v_res)

                keys = self._fetch_decompressed_cache(bsz, cache_len, self.kv_cache_compressor, self.cache_k_bstring, self.cache_k_qjl, self.cache_k_orig, self.cache_k_res, xq.device, xq.dtype)
                values = self._fetch_decompressed_cache(bsz, cache_len, self.kv_cache_compressor, self.cache_v_bstring, self.cache_v_qjl, self.cache_v_orig, self.cache_v_res, xq.device, xq.dtype)

        else:
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
            self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x)
        self.w2 = RowParallelLinear(hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x)
        self.w3 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        layer_id: int,
        args: ModelArgs,
        kv_cache_compressor: Optional[Any] = None,
        outlier_compressor: Optional[Any] = None,
        normal_compressor: Optional[Any] = None
    ):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args,
                                   kv_cache_compressor=kv_cache_compressor,
                                   outlier_compressor=outlier_compressor,
                                   normal_compressor=normal_compressor
                                   )

        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        params: ModelArgs,
        kv_cache_compressor: Optional[Any] = None,
        outlier_compressor: Optional[Any] = None,
        normal_compressor: Optional[Any] = None
    ):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = VocabParallelEmbedding(params.vocab_size, params.dim, init_method=lambda x: x)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(
                TransformerBlock(
                    layer_id,
                    params,
                    kv_cache_compressor=kv_cache_compressor,
                    outlier_compressor=outlier_compressor,
                    normal_compressor=normal_compressor
                )
            )

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(params.dim, params.vocab_size, bias=False, init_method=lambda x: x)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
            params.use_scaled_rope,
        )

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _, seqlen = tokens.shape

        # Capture the original token device so we can return the logits to the right place
        original_token_device = tokens.device

        # 1. EMBEDDING PHASE (Dynamic)
        # Move tokens to wherever the embedding layer lives (CPU for offload, GPU for pure CUDA)
        embed_device = self.tok_embeddings.weight.device

        # 1. CPU PHASE: Embeddings
        # Ensure tokens are on CPU for the lookup
        h = self.tok_embeddings(tokens.to(embed_device))

        # 2. COMPUTE PHASE (Dynamic)
        # Figure out where the heavy Transformer layers are
        compute_device = next(self.layers[0].parameters()).device
        h = h.to(compute_device)

        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)

            # https://github.com/pytorch/pytorch/issues/100005
            # torch.triu is buggy when the device is mps: filled values are
            # nan instead of 0.
            if mask.device.type == torch.device("mps").type:
                mask = torch.nan_to_num(mask, nan=0.0)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack([torch.zeros((seqlen, start_pos), device=tokens.device), mask]).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)

        # 3. OUTPUT PHASE (Dynamic)
        # Move hidden state to wherever the output projection layer lives
        out_device = self.output.weight.device
        h_out = h.to(out_device)
        output = self.output(h_out).float()

        return output.to(original_token_device)

    @torch.inference_mode()
    def get_embeddings(self, tokens: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = tokens.shape
        original_token_device = tokens.device

        original_state = self.params.use_compressed_kv_cache
        self.params.use_compressed_kv_cache = False

        # Use setattr to dynamically bypass Pyright's ModuleList type checking
        for layer in self.layers:
            setattr(layer.attention, "bypass_kv_cache", True)

        try:
            # 1. EMBEDDING PHASE (Dynamic)
            embed_device = self.tok_embeddings.weight.device
            start_pos = 0
            h = self.tok_embeddings(tokens.to(embed_device))

            # 2. COMPUTE PHASE (Dynamic)
            compute_device = next(self.layers[0].parameters()).device
            h = h.to(compute_device)

            self.freqs_cis = self.freqs_cis.to(h.device)
            freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

            mask = None
            if seqlen > 1:
                # Build mask on the compute device so it matches attention scores
                mask = torch.full((seqlen, seqlen), float("-inf"), device=h.device)
                mask = torch.triu(mask, diagonal=1)
                if mask.device.type == torch.device("mps").type:
                    mask = torch.nan_to_num(mask, nan=0.0)
                mask = torch.hstack([
                    torch.zeros((seqlen, start_pos), device=h.device),
                    mask
                ]).type_as(h)

            for layer in self.layers:
                h = layer(h, start_pos, freqs_cis, mask)

            h = self.norm(h)

            # h shape: (bsz, seqlen, dim) — works for any batch size
            last_token_embedding = h[:, -1, :]

            return F.normalize(last_token_embedding, p=2, dim=-1).to(original_token_device)
        except Exception as e:
            print(f"[get_embeddings] error: {e}")
            return torch.zeros((bsz, self.params.dim), dtype=torch.float32, device=original_token_device)
        finally:
            # Safely restore the state for the Generator
            for layer in self.layers:
                setattr(layer.attention, "bypass_kv_cache", False)
            self.params.use_compressed_kv_cache = original_state
