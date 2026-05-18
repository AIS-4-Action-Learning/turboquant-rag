# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from typing import Any, List, Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn

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
            if self.kv_cache_compressor is None:
                raise ValueError(
                    "use_compressed_kv_cache=True requires an externally created TurboQuant compressor"
                )

            required_attrs = ("compress_chunk", "decompress_chunk", "block_size", "bit_width")
            for attr in required_attrs:
                if not hasattr(self.kv_cache_compressor, attr):
                    raise TypeError(
                        f"kv_cache_compressor must provide '{attr}' (expected app.turboquant compressor)"
                    )

            self.kv_cache_block_size = int(self.kv_cache_compressor.block_size)
            self.kv_cache_bit_width = int(self.kv_cache_compressor.bit_width)
            if self.kv_cache_block_size <= 0:
                raise ValueError("kv_cache_compressor.block_size must be > 0")
            if self.kv_cache_bit_width <= 0:
                raise ValueError("kv_cache_compressor.bit_width must be > 0")

            self.kv_cache_n_blocks = (
                self.head_dim + self.kv_cache_block_size - 1
            ) // self.kv_cache_block_size
            self.kv_cache_bstring_bytes = (
                self.kv_cache_bit_width * self.kv_cache_block_size + 7
            ) // 8
            self.kv_cache_qjl_bytes = (self.kv_cache_block_size + 7) // 8

            compressed_shape = (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.kv_cache_n_blocks,
            )

            self.cache_k_bstring = torch.zeros(
                (*compressed_shape, self.kv_cache_bstring_bytes), dtype=torch.uint8
            )
            self.cache_k_qjl = torch.zeros(
                (*compressed_shape, self.kv_cache_qjl_bytes), dtype=torch.uint8
            )
            self.cache_k_original_l2 = torch.zeros(compressed_shape, dtype=torch.float32)
            self.cache_k_residual_l2 = torch.zeros(compressed_shape, dtype=torch.float32)

            self.cache_v_bstring = torch.zeros(
                (*compressed_shape, self.kv_cache_bstring_bytes), dtype=torch.uint8
            )
            self.cache_v_qjl = torch.zeros(
                (*compressed_shape, self.kv_cache_qjl_bytes), dtype=torch.uint8
            )
            self.cache_v_original_l2 = torch.zeros(compressed_shape, dtype=torch.float32)
            self.cache_v_residual_l2 = torch.zeros(compressed_shape, dtype=torch.float32)
        else:
            self.cache_k = torch.zeros(
                (
                    args.max_batch_size,
                    args.max_seq_len,
                    self.n_local_kv_heads,
                    self.head_dim,
                )
            )
            self.cache_v = torch.zeros(
                (
                    args.max_batch_size,
                    args.max_seq_len,
                    self.n_local_kv_heads,
                    self.head_dim,
                )
            )

    def _flatten_for_compression(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        flat_blocks = tensor.view(-1, self.kv_cache_block_size)
        return list(torch.unbind(flat_blocks, dim=0))

    def _store_compressed_cache(
        self,
        tensor: torch.Tensor,
        bsz: int,
        seqlen: int,
        start_pos: int,
        cache_bstring: torch.Tensor,
        cache_qjl: torch.Tensor,
        cache_original_l2: torch.Tensor,
        cache_residual_l2: torch.Tensor,
        label: str = "",
    ) -> None:
        """Vectorized storage: Uses bulk conversion to avoid Python loops."""
        tensor = tensor.float().contiguous()
        
        blocks = self._flatten_for_compression(tensor)
        compressed_results = self.kv_cache_compressor.compress_chunk(blocks)

        # --- ASYNC MEMORY BARRIER ---
        # Prevent Python's Garbage Collector from destroying tensor_f32
        # before the asynchronous CUDA kernel finishes reading it!
        if tensor.device.type == "cuda":
            torch.cuda.synchronize()
        # ------------------------------

        # Bulk convert bytes into flat buffers using list comprehensions (faster than loops)
        all_bstrings = b"".join([res[2] for res in compressed_results])
        all_qjls = b"".join([res[3] for res in compressed_results])

        # Convert to tensors and reshape to match the cache structure
        # Shape: (bsz, seqlen, heads, n_blocks, ...)
        b_tensor = torch.frombuffer(all_bstrings, dtype=torch.uint8).view(
            bsz, seqlen, self.n_local_kv_heads, self.kv_cache_n_blocks, -1
        )
        q_tensor = torch.frombuffer(all_qjls, dtype=torch.uint8).view(
            bsz, seqlen, self.n_local_kv_heads, self.kv_cache_n_blocks, -1
        )

        # Original and Residual L2s
        orig_l2 = torch.tensor([res[0] for res in compressed_results], dtype=torch.float32).view(
            bsz, seqlen, self.n_local_kv_heads, self.kv_cache_n_blocks
        )
        res_l2 = torch.tensor([res[1] for res in compressed_results], dtype=torch.float32).view(
            bsz, seqlen, self.n_local_kv_heads, self.kv_cache_n_blocks
        )

        # Batch assignment to the cache slices
        end_pos = start_pos + seqlen
        cache_bstring[:bsz, start_pos:end_pos] = b_tensor.to(cache_bstring.device)
        cache_qjl[:bsz, start_pos:end_pos] = q_tensor.to(cache_qjl.device)
        cache_original_l2[:bsz, start_pos:end_pos] = orig_l2.to(cache_original_l2.device)
        cache_residual_l2[:bsz, start_pos:end_pos] = res_l2.to(cache_residual_l2.device)

    def _fetch_decompressed_cache(
        self,
        bsz: int,
        cache_len: int,
        cache_bstring: torch.Tensor,
        cache_qjl: torch.Tensor,
        cache_original_l2: torch.Tensor,
        cache_residual_l2: torch.Tensor,
        target_device: torch.device,
        target_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Vectorized fetching: Converts cache slices to C-input in bulk."""
        # 1. Slice and flatten everything to CPU
        o_flat = cache_original_l2[:bsz, :cache_len].reshape(-1).cpu().tolist()
        r_flat = cache_residual_l2[:bsz, :cache_len].reshape(-1).cpu().tolist()

        # Reshape bitstrings to (N, BytesPerBlock) before calling .numpy()
        b_flat = cache_bstring[:bsz, :cache_len].contiguous().reshape(-1, self.kv_cache_bstring_bytes).cpu().numpy()
        q_flat = cache_qjl[:bsz, :cache_len].contiguous().reshape(-1, self.kv_cache_qjl_bytes).cpu().numpy()

        # 2. Reconstruct the list of tuples for the decompressor
        compressed_blocks = [
            (o_flat[i], r_flat[i], b_flat[i].tobytes(), q_flat[i].tobytes())
            for i in range(len(o_flat))
        ]

        # 3. Batch Decompress (The C-API Multi-threaded call)
        decompressed_blocks = self.kv_cache_compressor.decompress_chunk(compressed_blocks)

        # 4. Stack and view back to (B, S, H, D)
        stacked = torch.stack(decompressed_blocks, dim=0).to(device=target_device, dtype=target_dtype)

        return stacked.view(bsz, cache_len, self.n_local_kv_heads, self.head_dim)

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

        if self.use_compressed_kv_cache:
            self._store_compressed_cache(
                xk,
                bsz=bsz,
                seqlen=seqlen,
                start_pos=start_pos,
                cache_bstring=self.cache_k_bstring,
                cache_qjl=self.cache_k_qjl,
                cache_original_l2=self.cache_k_original_l2,
                cache_residual_l2=self.cache_k_residual_l2,
                label="K",
            )
            self._store_compressed_cache(
                xv,
                bsz=bsz,
                seqlen=seqlen,
                start_pos=start_pos,
                cache_bstring=self.cache_v_bstring,
                cache_qjl=self.cache_v_qjl,
                cache_original_l2=self.cache_v_original_l2,
                cache_residual_l2=self.cache_v_residual_l2,
                label="V",
            )

            cache_len = start_pos + seqlen

            keys = self._fetch_decompressed_cache(
                bsz=bsz,
                cache_len=cache_len,
                cache_bstring=self.cache_k_bstring,
                cache_qjl=self.cache_k_qjl,
                cache_original_l2=self.cache_k_original_l2,
                cache_residual_l2=self.cache_k_residual_l2,
                target_device=xq.device,
                target_dtype=xq.dtype,
            )
            values = self._fetch_decompressed_cache(
                bsz=bsz,
                cache_len=cache_len,
                cache_bstring=self.cache_v_bstring,
                cache_qjl=self.cache_v_qjl,
                cache_original_l2=self.cache_v_original_l2,
                cache_residual_l2=self.cache_v_residual_l2,
                target_device=xq.device,
                target_dtype=xq.dtype,
            )
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
    ):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args, kv_cache_compressor=kv_cache_compressor)
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
        _bsz, seqlen = tokens.shape
        
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
