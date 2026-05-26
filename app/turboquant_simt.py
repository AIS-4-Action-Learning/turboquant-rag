"""
TurboQuant SIMT (CUDA) implementation.
Includes both single-stream and multi-stream variants.
"""

import ctypes
import os
import sys
from typing import Tuple, List, Optional

import torch

# Import common structures and base class from SIMD module
from app.turboquant_simd import Vector, QuantizationResult, TurboQuantCompressorBase


# ============================================================================
# SIMT Structures
# ============================================================================

class TurboQuantContext(ctypes.Structure):
    """Mirrors turboquant_context_t from simt headers."""
    _fields_ = [
        ("mse_quantizer", ctypes.c_void_p),
        ("mse_buffer", ctypes.c_void_p),
        ("y", ctypes.c_void_p),
        ("h_bstring", ctypes.POINTER(ctypes.c_uint8)),
        ("d_bstring", ctypes.POINTER(ctypes.c_uint8)),
        ("bstring_size", ctypes.c_size_t),
        ("h_qjl", ctypes.POINTER(ctypes.c_uint8)),
        ("d_qjl", ctypes.POINTER(ctypes.c_uint8)),
        ("qjl_size", ctypes.c_size_t),
        ("compute_stream", ctypes.c_void_p),
        ("is_init", ctypes.c_uint8),
    ]


class TurboQuantBatchContext(ctypes.Structure):
    """Mirrors turboquant_batch_context_t from simt-multi headers."""
    _fields_ = [
        ("contexts", ctypes.POINTER(ctypes.POINTER(TurboQuantContext))),
        ("n_streams", ctypes.c_uint8),
        ("dims", ctypes.c_size_t),
        ("bit_width", ctypes.c_uint8),
        ("batch_result_storage", ctypes.c_void_p),
        ("batch_bstring_storage", ctypes.c_void_p),
        ("batch_qjl_storage", ctypes.c_void_p),
        ("batch_result_capacity", ctypes.c_uint32),
        ("batch_output_ptrs", ctypes.c_void_p),
        ("batch_output_storage", ctypes.c_void_p),
        ("batch_output_device_storage", ctypes.c_void_p),
        ("batch_output_capacity", ctypes.c_uint32),
        ("is_init", ctypes.c_uint8),
    ]


class QuantizationBatchResult(ctypes.Structure):
    """Mirrors quantization_batch_result from simt headers."""
    _fields_ = [
        ("results", ctypes.POINTER(QuantizationResult)),
        ("n_results", ctypes.c_uint32),
    ]


# ============================================================================
# SIMT Single (CUDA Single-Stream)
# ============================================================================

class SIMTSingleCompressor(TurboQuantCompressorBase):
    """
    TurboQuant SIMT Single-Stream (CUDA) implementation.
    Uses context pointer with stream.
    """
    
    def _init_library(self):
        """Setup context-pointer function signatures."""
        lib = self._lib
        
        # Context lifecycle with pointer
        lib.turboquant_init.argtypes = [
            ctypes.POINTER(ctypes.POINTER(TurboQuantContext)),
            ctypes.c_size_t,
            ctypes.c_uint8,
        ]
        lib.turboquant_init.restype = ctypes.c_uint8
        
        lib.turboquant_context_destroy.argtypes = [
            ctypes.POINTER(ctypes.POINTER(TurboQuantContext))
        ]
        lib.turboquant_context_destroy.restype = None
        
        lib.turboquant_clean.argtypes = [ctypes.POINTER(TurboQuantContext)]
        lib.turboquant_clean.restype = None
        
        lib.turboquant_init_load.argtypes = [
            ctypes.POINTER(TurboQuantContext),
            ctypes.c_char_p,
        ]
        lib.turboquant_init_load.restype = ctypes.c_uint8
        
        lib.turboquant_save.argtypes = [
            ctypes.POINTER(TurboQuantContext),
            ctypes.c_char_p,
        ]
        lib.turboquant_save.restype = ctypes.c_uint8
        
        # Result management
        lib.turboquant_quantization_result_init.argtypes = []
        lib.turboquant_quantization_result_init.restype = ctypes.POINTER(QuantizationResult)
        lib.turboquant_quantization_result_destroy.argtypes = [
            ctypes.POINTER(ctypes.POINTER(QuantizationResult))
        ]
        lib.turboquant_quantization_result_destroy.restype = None
        
        # Product quantization with context
        lib.turboquant_prod_quantization.argtypes = [
            ctypes.POINTER(TurboQuantContext),
            ctypes.POINTER(Vector),
            ctypes.POINTER(QuantizationResult),
        ]
        lib.turboquant_prod_quantization.restype = ctypes.c_uint8
        
        lib.turboquant_prod_dequantization.argtypes = [
            ctypes.POINTER(TurboQuantContext),
            ctypes.POINTER(QuantizationResult),
        ]
        lib.turboquant_prod_dequantization.restype = ctypes.POINTER(Vector)
    
    def _init_context(self):
        """Initialize CUDA context."""
        # Allocate context pointer
        self._context = ctypes.POINTER(TurboQuantContext)()
        
        status = self._lib.turboquant_init(
            ctypes.byref(self._context),
            ctypes.c_size_t(self.block_size),
            ctypes.c_uint8(self.bit_width),
        )
        if status != 0:
            raise RuntimeError(f"turboquant_init failed with code {status}")
        
        if not self._context or not self._context.contents.is_init:
            raise RuntimeError("Failed to initialize TurboQuant context")
        
        # Try to load existing context file
        if os.path.exists(self.context_path):
            status = self._lib.turboquant_init_load(
                self._context, self.context_path.encode("utf-8")
            )
    
    def compress_block(self, block: torch.Tensor) -> Tuple[float, float, bytes, bytes]:
        """Compress a single block using CUDA."""
        # Ensure CUDA tensor
        if not block.is_cuda:
            block = block.to("cuda")
        
        block = block.detach().to(dtype=torch.bfloat16).contiguous()
        
        if block.numel() > self.block_size:
            raise ValueError(f"Block has {block.numel()} elements, expected <= {self.block_size}")
        
        # Handle padding
        if block.numel() < self.block_size:
            padding = torch.zeros(
                self.block_size - block.numel(),
                dtype=torch.bfloat16,
                device="cuda"
            )
            block = torch.cat([block, padding])
        
        original_l2 = torch.linalg.norm(block.float()).item()
        
        if original_l2 < 1e-12:
            n_bstring = (self.bit_width * self.block_size + 7) // 8
            n_qjl = (self.block_size + 7) // 8
            return (original_l2, 0.0, bytes(n_bstring), bytes(n_qjl))
        
        # Convert to FP32 for C API
        block_f32 = block.float().contiguous()
        
        # Create C vector (CUDA pointer)
        c_ptr = ctypes.cast(block_f32.data_ptr(), ctypes.POINTER(ctypes.c_float))
        vec = Vector(n=self.block_size, vector=c_ptr)
        
        # Allocate result
        result_ptr = self._lib.turboquant_quantization_result_init()
        if not result_ptr:
            raise MemoryError("Failed to allocate quantization result")
        
        try:
            status = self._lib.turboquant_prod_quantization(
                self._context,
                ctypes.byref(vec),
                result_ptr
            )
            if status != 0:
                raise RuntimeError(f"turboquant_prod_quantization failed with code {status}")
            
            res = result_ptr.contents
            n_bstring = (self.bit_width * self.block_size + 7) // 8
            n_qjl = (self.block_size + 7) // 8
            
            # Copy from device to host
            bstring_bytes = ctypes.string_at(res.bstring, n_bstring)
            qjl_bytes = ctypes.string_at(res.qjl, n_qjl)
            
            return (original_l2, res.residual_l2, bstring_bytes, qjl_bytes)
        finally:
            self._lib.turboquant_quantization_result_destroy(ctypes.byref(result_ptr))
    
    def compress_chunk(self, blocks: List[torch.Tensor]) -> List[Tuple[float, float, bytes, bytes]]:
        """Sequential compression for SIMT single."""
        return [self.compress_block(block) for block in blocks]
    
    def decompress_chunk(self, results: List[Tuple[float, float, bytes, bytes]]) -> List[torch.Tensor]:
        """Sequential decompression for SIMT single."""
        outputs = []
        n_bstring = (self.bit_width * self.block_size + 7) // 8
        n_qjl = (self.block_size + 7) // 8
        
        # Load CUDA runtime for device memory management
        cudart = None
        cuda_memcpy = None
        for cudart_name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
            try:
                cudart = ctypes.CDLL(cudart_name)
                break
            except OSError:
                continue
        
        if cudart is None:
            raise RuntimeError("CUDA runtime not found for decompression")
        
        # Setup cudaMemcpy function
        cudart.cudaMemcpy.argtypes = [
            ctypes.c_void_p,  # dst
            ctypes.c_void_p,  # src
            ctypes.c_size_t,  # count
            ctypes.c_int,     # kind
        ]
        cudart.cudaMemcpy.restype = ctypes.c_int
        cuda_memcpy = cudart.cudaMemcpy
        cuda_memcpy_device_to_device = 3  # cudaMemcpyDeviceToDevice
        cuda_memcpy_host_to_device = 1    # cudaMemcpyHostToDevice
        
        for original_l2, residual_l2, bstring_bytes, qjl_bytes in results:
            if original_l2 < 1e-12:
                outputs.append(torch.zeros(self.block_size, dtype=torch.bfloat16, device="cuda"))
                continue
            
            # Allocate GPU memory for compressed data and copy from host
            bstring_gpu = torch.empty(n_bstring, dtype=torch.uint8, device="cuda")
            qjl_gpu = torch.empty(n_qjl, dtype=torch.uint8, device="cuda")
            
            # Copy bytes to GPU tensors using cudaMemcpy
            bstring_arr = (ctypes.c_uint8 * n_bstring).from_buffer_copy(bstring_bytes)
            qjl_arr = (ctypes.c_uint8 * n_qjl).from_buffer_copy(qjl_bytes)
            
            cuda_memcpy(
                ctypes.c_void_p(bstring_gpu.data_ptr()),
                ctypes.cast(bstring_arr, ctypes.c_void_p),
                ctypes.c_size_t(n_bstring),
                ctypes.c_int(cuda_memcpy_host_to_device),
            )
            cuda_memcpy(
                ctypes.c_void_p(qjl_gpu.data_ptr()),
                ctypes.cast(qjl_arr, ctypes.c_void_p),
                ctypes.c_size_t(n_qjl),
                ctypes.c_int(cuda_memcpy_host_to_device),
            )
            
            # Create quantization result with DEVICE pointers
            result = QuantizationResult(
                bstring=ctypes.cast(bstring_gpu.data_ptr(), ctypes.POINTER(ctypes.c_uint8)),
                qjl=ctypes.cast(qjl_gpu.data_ptr(), ctypes.POINTER(ctypes.c_uint8)),
                residual_l2=ctypes.c_float(residual_l2),
            )
            
            vec_ptr = self._lib.turboquant_prod_dequantization(
                self._context,
                ctypes.byref(result)
            )
            if not vec_ptr:
                raise RuntimeError("turboquant_prod_dequantization returned null")
            
            # Copy result from device to device (C API returns device pointer)
            output = torch.empty(self.block_size, dtype=torch.float32, device="cuda")
            vec = vec_ptr.contents
            
            cuda_memcpy(
                ctypes.c_void_p(output.data_ptr()),
                ctypes.cast(vec.vector, ctypes.c_void_p),
                ctypes.c_size_t(self.block_size * 4),
                ctypes.c_int(cuda_memcpy_device_to_device),
            )

            # --- THE FINAL BRIDGE FIX ---
            # 1. Force the C++ output (norm ~3.16) back to a unit-sphere (norm 1.0)
            current_norm = output.norm(p=2).clamp_min(1e-6)
            output.div_(current_norm)
            
            # 2. Now multiply by the target Llama magnitude
            output.mul_(original_l2)
            # ----------------------------

            outputs.append(output.to(dtype=torch.bfloat16))
        
        return outputs
    
    def close(self):
        """Cleanup CUDA context."""
        if hasattr(self, '_context') and self._context and self._lib:
            self._lib.turboquant_context_destroy(ctypes.byref(self._context))
            self._context = None


# ============================================================================
# SIMT Multi (CUDA Multi-Stream)
# ============================================================================

class SIMTBatchCompressor(TurboQuantCompressorBase):
    """
    TurboQuant SIMT Multi-Stream (CUDA) implementation.
    Uses batch context with multiple CUDA streams.
    """
    
    def __init__(self, lib_path: str, context_path: str, block_size: int, bit_width: int, n_streams: int = 16):
        self.n_streams = n_streams
        self._batch_ctx = None
        self._libc = None
        self._cudart = None
        self._cuda_memcpy = None
        self._cuda_memcpy_device_to_device = 3  # cudaMemcpyDeviceToDevice
        super().__init__(lib_path, context_path, block_size, bit_width)
    
    def _init_library(self):
        """Setup batch stream function signatures."""
        lib = self._lib
        
        # Load CUDA runtime for memcpy
        self._libc = ctypes.CDLL(None)
        self._libc.free.argtypes = [ctypes.c_void_p]
        self._libc.free.restype = None
        
        for cudart_name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11.0"):
            try:
                self._cudart = ctypes.CDLL(cudart_name)
                break
            except OSError:
                continue
        
        if self._cudart is not None:
            self._cudart.cudaMemcpy.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            self._cudart.cudaMemcpy.restype = ctypes.c_int
            self._cuda_memcpy = self._cudart.cudaMemcpy
            self._cudart.cudaFree.argtypes = [ctypes.c_void_p]
            self._cudart.cudaFree.restype = ctypes.c_int
            self._cuda_free = self._cudart.cudaFree
        
        # Batch context lifecycle
        lib.turboquant_batch_init.argtypes = [
            ctypes.POINTER(ctypes.POINTER(TurboQuantBatchContext)),
            ctypes.c_size_t,
            ctypes.c_uint8,
            ctypes.c_uint8,
        ]
        lib.turboquant_batch_init.restype = ctypes.c_uint8
        
        lib.turboquant_batch_destroy.argtypes = [
            ctypes.POINTER(ctypes.POINTER(TurboQuantBatchContext))
        ]
        lib.turboquant_batch_destroy.restype = None
        
        lib.turboquant_batch_init_load.argtypes = [
            ctypes.POINTER(TurboQuantBatchContext),
            ctypes.c_char_p,
        ]
        lib.turboquant_batch_init_load.restype = ctypes.c_uint8
        
        lib.turboquant_batch_save.argtypes = [
            ctypes.POINTER(TurboQuantBatchContext),
            ctypes.c_char_p,
        ]
        lib.turboquant_batch_save.restype = ctypes.c_uint8
        
        # Batch operations
        lib.turboquant_prod_quantization_batch.argtypes = [
            ctypes.POINTER(TurboQuantBatchContext),
            ctypes.POINTER(ctypes.POINTER(Vector)),
            ctypes.POINTER(QuantizationBatchResult),
            ctypes.c_uint32,
        ]
        lib.turboquant_prod_quantization_batch.restype = ctypes.c_uint8
        
        lib.turboquant_prod_dequantization_batch.argtypes = [
            ctypes.POINTER(TurboQuantBatchContext),
            ctypes.POINTER(QuantizationBatchResult),
        ]
        lib.turboquant_prod_dequantization_batch.restype = ctypes.POINTER(ctypes.POINTER(Vector))

        lib.turboquant_prod_quantization_batch_direct.argtypes = [
            ctypes.POINTER(TurboQuantBatchContext),
            ctypes.c_void_p,  # d_inputs
            ctypes.c_void_p,  # d_bstrings
            ctypes.c_void_p,  # d_qjls
            ctypes.c_void_p,  # d_residual_l2s
            ctypes.c_uint32,  # batch_size
        ]
        lib.turboquant_prod_quantization_batch_direct.restype = ctypes.c_uint8
        
        lib.turboquant_prod_dequantization_batch.argtypes = [
            ctypes.POINTER(TurboQuantBatchContext),
            ctypes.POINTER(QuantizationBatchResult),
        ]
        lib.turboquant_prod_dequantization_batch.restype = ctypes.POINTER(ctypes.POINTER(Vector))
        
        # Zero-copy direct GPU batch dequantization (no host round-trip)
        lib.turboquant_prod_dequantization_batch_direct.argtypes = [
            ctypes.POINTER(TurboQuantBatchContext),
            ctypes.c_void_p,  # d_bstrings
            ctypes.c_void_p,  # d_qjls
            ctypes.c_void_p,  # d_residual_l2s
            ctypes.c_void_p,  # d_outputs
            ctypes.c_uint32,  # batch_size
        ]
        lib.turboquant_prod_dequantization_batch_direct.restype = ctypes.c_uint8

        # Define the new Fused Attention binding
        lib.turboquant_fused_attention_direct.argtypes = [
            ctypes.POINTER(TurboQuantBatchContext),
            ctypes.c_void_p, # xq
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, # Keys
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, # Values
            ctypes.c_void_p, # mask
            ctypes.c_uint32, # seqlen
            ctypes.c_uint32, # head_dim
            ctypes.c_uint32, # n_local_heads
            ctypes.c_uint32, # n_local_kv_heads
            ctypes.c_void_p  # d_output
        ]
        lib.turboquant_fused_attention_direct.restype = ctypes.c_uint8

        # 2. Mixed Precision Fused Attention
        lib.turboquant_fused_attention_mixed.argtypes = [
            ctypes.POINTER(TurboQuantBatchContext), # Outlier Context
            ctypes.POINTER(TurboQuantBatchContext), # Normal Context
            ctypes.c_void_p,  # xq

            # Outlier History
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, # K out
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, # V out

            # Normal History
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, # K norm
            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, # V norm

            ctypes.c_void_p,  # mask
            ctypes.c_uint32,  # seqlen (cache_len)
            ctypes.c_uint32,  # head_dim
            ctypes.c_uint32,  # outlier_dim
            ctypes.c_uint32,  # n_local_heads
            ctypes.c_uint32,  # n_local_kv_heads
            ctypes.c_void_p   # d_output
        ]
        lib.turboquant_fused_attention_mixed.restype = ctypes.c_uint8

    def _init_context(self):
        """Initialize batch context."""
        # Allocate batch context pointer
        self._batch_ctx = ctypes.POINTER(TurboQuantBatchContext)()
        
        status = self._lib.turboquant_batch_init(
            ctypes.byref(self._batch_ctx),
            ctypes.c_size_t(self.block_size),
            ctypes.c_uint8(self.bit_width),
            ctypes.c_uint8(self.n_streams),
        )
        if status != 0:
            raise RuntimeError(f"turboquant_batch_init failed with code {status}")
        
        if not self._batch_ctx or not self._batch_ctx.contents.is_init:
            raise RuntimeError("Failed to initialize batch context")
        
        # Try to load existing context file
        if os.path.exists(self.context_path):
            status = self._lib.turboquant_batch_init_load(
                self._batch_ctx, self.context_path.encode("utf-8")
            )

    def fused_attention(self, xq: torch.Tensor,
                    k_b: torch.Tensor, k_q: torch.Tensor, k_r: torch.Tensor, k_o: torch.Tensor,
                    v_b: torch.Tensor, v_q: torch.Tensor, v_r: torch.Tensor, v_o: torch.Tensor,
                    mask: torch.Tensor, seqlen: int, head_dim: int,
                    n_local_heads: int, n_local_kv_heads: int) -> torch.Tensor:

        if xq.shape[0] != 1 or xq.shape[1] != 1:
            raise ValueError(
                "SIMT fused attention currently supports decode-only tensors "
                f"with shape (1, 1, heads, dim), got {tuple(xq.shape)}"
            )

        original_dtype = xq.dtype
        xq_f32 = xq.detach().to(dtype=torch.float32).contiguous()
        output_f32 = torch.empty_like(xq_f32)

        mask_ptr = ctypes.c_void_p(0)
        if mask is not None:
            mask_f32 = mask.to(dtype=torch.float32).contiguous()
            mask_ptr = ctypes.c_void_p(mask_f32.data_ptr())

        k_b = k_b.contiguous()
        k_q = k_q.contiguous()
        k_r = k_r.contiguous()
        k_o = k_o.contiguous()
        v_b = v_b.contiguous()
        v_q = v_q.contiguous()
        v_r = v_r.contiguous()
        v_o = v_o.contiguous()

        status = self._lib.turboquant_fused_attention_direct(
            self._batch_ctx,
            ctypes.c_void_p(xq_f32.data_ptr()),
            ctypes.c_void_p(k_b.data_ptr()),
            ctypes.c_void_p(k_q.data_ptr()),
            ctypes.c_void_p(k_r.data_ptr()),
            ctypes.c_void_p(k_o.data_ptr()),
            ctypes.c_void_p(v_b.data_ptr()),
            ctypes.c_void_p(v_q.data_ptr()),
            ctypes.c_void_p(v_r.data_ptr()),
            ctypes.c_void_p(v_o.data_ptr()),
            mask_ptr,
            ctypes.c_uint32(seqlen),
            ctypes.c_uint32(head_dim),
            ctypes.c_uint32(n_local_heads),
            ctypes.c_uint32(n_local_kv_heads),
            ctypes.c_void_p(output_f32.data_ptr())
        )
        if status != 0:
            raise RuntimeError(f"Fused attention failed with code {status}")

        return output_f32.to(dtype=original_dtype)

    def mixed_fused_attention(
        self,
        normal_compressor: 'SIMTBatchCompressor',
        xq: torch.Tensor,

        # Outlier History
        k_b_out: torch.Tensor, k_q_out: torch.Tensor, k_r_out: torch.Tensor, k_o_out: torch.Tensor,
        v_b_out: torch.Tensor, v_q_out: torch.Tensor, v_r_out: torch.Tensor, v_o_out: torch.Tensor,

        # Normal History
        k_b_norm: torch.Tensor, k_q_norm: torch.Tensor, k_r_norm: torch.Tensor, k_o_norm: torch.Tensor,
        v_b_norm: torch.Tensor, v_q_norm: torch.Tensor, v_r_norm: torch.Tensor, v_o_norm: torch.Tensor,

        mask: Optional[torch.Tensor],
        cache_len: int,
        head_dim: int,
        n_local_heads: int,
        n_local_kv_heads: int
    ) -> torch.Tensor:
        """
        Zero-copy fused attention + dequantization for mixed-precision compression.
        Call this on the OUTLIER compressor and pass the NORMAL compressor as the first arg.
        """
        if xq.shape[0] != 1 or xq.shape[1] != 1:
            raise ValueError(
                "SIMT mixed fused attention currently supports decode-only tensors "
                f"with shape (1, 1, heads, dim), got {tuple(xq.shape)}"
            )

        original_dtype = xq.dtype
        xq_f32 = xq.detach().to(dtype=torch.float32).contiguous()
        output_f32 = torch.empty_like(xq_f32)

        mask_ptr = ctypes.c_void_p(0)
        if mask is not None:
            mask_f32 = mask.to(dtype=torch.float32).contiguous()
            mask_ptr = ctypes.c_void_p(mask_f32.data_ptr())

        # Get the dimension split directly from the compressor properties
        outlier_dim = self.block_size

        k_b_out = k_b_out.contiguous()
        k_q_out = k_q_out.contiguous()
        k_r_out = k_r_out.contiguous()
        k_o_out = k_o_out.contiguous()
        v_b_out = v_b_out.contiguous()
        v_q_out = v_q_out.contiguous()
        v_r_out = v_r_out.contiguous()
        v_o_out = v_o_out.contiguous()
        k_b_norm = k_b_norm.contiguous()
        k_q_norm = k_q_norm.contiguous()
        k_r_norm = k_r_norm.contiguous()
        k_o_norm = k_o_norm.contiguous()
        v_b_norm = v_b_norm.contiguous()
        v_q_norm = v_q_norm.contiguous()
        v_r_norm = v_r_norm.contiguous()
        v_o_norm = v_o_norm.contiguous()

        status = self._lib.turboquant_fused_attention_mixed(
            self._batch_ctx,                 # Outlier context (self)
            normal_compressor._batch_ctx,    # Normal context
            ctypes.c_void_p(xq_f32.data_ptr()),

            # Outlier pointers
            ctypes.c_void_p(k_b_out.data_ptr()), ctypes.c_void_p(k_q_out.data_ptr()),
            ctypes.c_void_p(k_r_out.data_ptr()), ctypes.c_void_p(k_o_out.data_ptr()),
            ctypes.c_void_p(v_b_out.data_ptr()), ctypes.c_void_p(v_q_out.data_ptr()),
            ctypes.c_void_p(v_r_out.data_ptr()), ctypes.c_void_p(v_o_out.data_ptr()),

            # Normal pointers
            ctypes.c_void_p(k_b_norm.data_ptr()), ctypes.c_void_p(k_q_norm.data_ptr()),
            ctypes.c_void_p(k_r_norm.data_ptr()), ctypes.c_void_p(k_o_norm.data_ptr()),
            ctypes.c_void_p(v_b_norm.data_ptr()), ctypes.c_void_p(v_q_norm.data_ptr()),
            ctypes.c_void_p(v_r_norm.data_ptr()), ctypes.c_void_p(v_o_norm.data_ptr()),

            mask_ptr,
            ctypes.c_uint32(cache_len),
            ctypes.c_uint32(head_dim),
            ctypes.c_uint32(outlier_dim),
            ctypes.c_uint32(n_local_heads),
            ctypes.c_uint32(n_local_kv_heads),
            ctypes.c_void_p(output_f32.data_ptr())
        )

        if status != 0:
            raise RuntimeError(f"turboquant_fused_attention_mixed failed with code {status}")

        return output_f32.to(dtype=original_dtype)

    def compress_block(self, block: torch.Tensor) -> Tuple[float, float, bytes, bytes]:
        """Single block - delegates to batch with size 1."""
        results = self.compress_chunk([block])
        return results[0]

    def compress_chunk_tensor_direct(
        self,
        blocks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Direct GPU batch quantization for a contiguous `(N, block_size)` tensor.

        Returns:
            bstrings: `(N, b_bytes)` uint8 tensor on CUDA
            qjls: `(N, q_bytes)` uint8 tensor on CUDA
            original_l2s: `(N,)` float32 tensor on CUDA
            residual_l2s: `(N,)` float32 tensor on CUDA
        """
        if blocks.ndim != 2 or blocks.shape[1] != self.block_size:
            raise ValueError(f"Expected blocks shaped (N, {self.block_size}), got {tuple(blocks.shape)}")

        if not blocks.is_cuda:
            blocks = blocks.to("cuda")

        blocks_f32 = blocks.detach().to(dtype=torch.bfloat16).float().contiguous()
        batch_size = blocks_f32.shape[0]
        b_bytes = (self.bit_width * self.block_size + 7) // 8
        q_bytes = (self.block_size + 7) // 8

        bstrings = torch.zeros((batch_size, b_bytes), dtype=torch.uint8, device=blocks_f32.device)
        qjls = torch.zeros((batch_size, q_bytes), dtype=torch.uint8, device=blocks_f32.device)
        original_l2s = torch.linalg.norm(blocks_f32, dim=1)
        residual_l2s = torch.zeros(batch_size, dtype=torch.float32, device=blocks_f32.device)

        active_indices = (original_l2s >= 1e-12).nonzero(as_tuple=False).flatten()
        if active_indices.numel() == 0:
            return bstrings, qjls, original_l2s, residual_l2s

        active_blocks = blocks_f32.index_select(0, active_indices).contiguous()
        active_count = active_blocks.shape[0]
        active_bstrings = torch.empty((active_count, b_bytes), dtype=torch.uint8, device=blocks_f32.device)
        active_qjls = torch.empty((active_count, q_bytes), dtype=torch.uint8, device=blocks_f32.device)
        active_residual_l2s = torch.empty(active_count, dtype=torch.float32, device=blocks_f32.device)

        status = self._lib.turboquant_prod_quantization_batch_direct(
            self._batch_ctx,
            ctypes.c_void_p(active_blocks.data_ptr()),
            ctypes.c_void_p(active_bstrings.data_ptr()),
            ctypes.c_void_p(active_qjls.data_ptr()),
            ctypes.c_void_p(active_residual_l2s.data_ptr()),
            ctypes.c_uint32(active_count),
        )
        if status != 0:
            raise RuntimeError(f"turboquant_prod_quantization_batch_direct failed with code {status}")

        bstrings.index_copy_(0, active_indices, active_bstrings)
        qjls.index_copy_(0, active_indices, active_qjls)
        residual_l2s.index_copy_(0, active_indices, active_residual_l2s)

        return bstrings, qjls, original_l2s, residual_l2s

    def compress_chunk(self, blocks: List[torch.Tensor]) -> List[Tuple[float, float, bytes, bytes]]:
        batch_size = len(blocks)
        if batch_size == 0:
            return []
        
        original_l2s = []
        active_indices = []
        active_blocks_f32 = []
        
        for i, block in enumerate(blocks):
            if not block.is_cuda:
                block = block.to("cuda")
            
            gpu_block = block.detach().to(dtype=torch.bfloat16).contiguous()
            
            if gpu_block.numel() < self.block_size:
                padding = torch.zeros(self.block_size - gpu_block.numel(), dtype=torch.bfloat16, device="cuda")
                gpu_block = torch.cat([gpu_block, padding])
            
            original_l2 = torch.linalg.norm(gpu_block.float()).item()
            original_l2s.append(original_l2)
            
            if original_l2 >= 1e-12:
                active_indices.append(i)
                active_blocks_f32.append(gpu_block.float().contiguous())
        
        n_bstring = (self.bit_width * self.block_size + 7) // 8
        n_qjl = (self.block_size + 7) // 8
        
        results = [(original_l2, 0.0, bytes(n_bstring), bytes(n_qjl)) for original_l2 in original_l2s]
        
        if not active_blocks_f32:
            return results
        
        chunk_size = len(active_blocks_f32)
        
        # --- THE MASSIVE ONE-SHOT ARRAY ---
        vectors = []
        vec_ptrs = []
        for block in active_blocks_f32:
            c_ptr = ctypes.cast(block.data_ptr(), ctypes.POINTER(ctypes.c_float))
            vec = Vector(n=self.block_size, vector=c_ptr)
            vectors.append(vec)
            vec_ptrs.append(ctypes.pointer(vec))
        
        vec_array = (ctypes.POINTER(Vector) * chunk_size)(*vec_ptrs)
        
        batch_result = QuantizationBatchResult()
        
        # FIRE ONCE
        status = self._lib.turboquant_prod_quantization_batch(
            self._batch_ctx,
            vec_array,
            ctypes.byref(batch_result),
            ctypes.c_uint32(chunk_size), # <--- 32-bit cast
        )
        
        if status != 0:
            raise RuntimeError(f"turboquant_prod_quantization_batch failed with code {status}")
        
        for chunk_i, original_i in enumerate(active_indices):
            res = batch_result.results[chunk_i]
            bstring_bytes = ctypes.string_at(res.bstring, n_bstring)
            qjl_bytes = ctypes.string_at(res.qjl, n_qjl)
            results[original_i] = (original_l2s[original_i], float(res.residual_l2), bstring_bytes, qjl_bytes)
            
        return results


    def decompress_chunk(self, results: List[Tuple[float, float, bytes, bytes]]) -> List[torch.Tensor]:
        batch_size = len(results)
        if batch_size == 0:
            return []
        
        if not self._batch_ctx or not self._batch_ctx.contents.is_init:
            raise RuntimeError("TurboQuant batch context is not initialized")
        
        n_bstring = (self.bit_width * self.block_size + 7) // 8
        n_qjl = (self.block_size + 7) // 8
        
        outputs = [torch.zeros(self.block_size, dtype=torch.bfloat16, device="cuda") for _ in range(batch_size)]
        
        active_indices = [i for i, (orig_l2, _, _, _) in enumerate(results) if orig_l2 >= 1e-12]
        if not active_indices:
            return outputs
        
        chunk_size = len(active_indices)

        all_bstrings = b"".join([results[i][2] for i in active_indices])
        all_qjls = b"".join([results[i][3] for i in active_indices])

        massive_bstring_arr = (ctypes.c_uint8 * len(all_bstrings)).from_buffer_copy(all_bstrings)
        massive_qjl_arr = (ctypes.c_uint8 * len(all_qjls)).from_buffer_copy(all_qjls)

        base_bstring_ptr = ctypes.addressof(massive_bstring_arr)
        base_qjl_ptr = ctypes.addressof(massive_qjl_arr)
        
        # --- THE MASSIVE ONE-SHOT ARRAY ---
        c_results_array = (QuantizationResult * chunk_size)()
        
        for chunk_i, original_i in enumerate(active_indices):
            c_results_array[chunk_i] = QuantizationResult(
                bstring=ctypes.cast(base_bstring_ptr + chunk_i * n_bstring, ctypes.POINTER(ctypes.c_uint8)),
                qjl=ctypes.cast(base_qjl_ptr + chunk_i * n_qjl, ctypes.POINTER(ctypes.c_uint8)),
                residual_l2=ctypes.c_float(float(results[original_i][1])),
            )
        
        batch_input = QuantizationBatchResult(
            results=ctypes.cast(c_results_array, ctypes.POINTER(QuantizationResult)),
            n_results=ctypes.c_uint32(chunk_size), # <--- 32-bit cast
        )
        
        # FIRE ONCE
        c_vectors = self._lib.turboquant_prod_dequantization_batch(
            self._batch_ctx,
            ctypes.byref(batch_input)
        )
        
        if not c_vectors:
            raise RuntimeError("turboquant_prod_dequantization_batch returned null")
       
        # ONE MASSIVE PYTORCH ZERO-LOOP MEMCPY
        first_vector_ptr = c_vectors[0].contents.vector
        massive_out_f32 = torch.empty((chunk_size, self.block_size), dtype=torch.float32, device="cuda")

        self._cuda_memcpy(
            ctypes.c_void_p(massive_out_f32.data_ptr()),
            ctypes.cast(first_vector_ptr, ctypes.c_void_p),
            ctypes.c_size_t(chunk_size * self.block_size * 4),
            ctypes.c_int(self._cuda_memcpy_device_to_device),
        )

        original_l2s_tensor = torch.tensor([results[i][0] for i in active_indices], dtype=torch.float32, device="cuda").unsqueeze(1)
        current_norms = massive_out_f32.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)

        massive_out_f32.div_(current_norms).mul_(original_l2s_tensor)
        massive_out_bf16 = massive_out_f32.to(dtype=torch.bfloat16)

        list_of_tensors = list(torch.unbind(massive_out_bf16, dim=0))

        for i, original_i in enumerate(active_indices):
            outputs[original_i] = list_of_tensors[i]

        return outputs


    def decompress_chunk_tensor_direct(
        self,
        b_tensor: torch.Tensor,
        q_tensor: torch.Tensor,
        r_tensor: torch.Tensor,
        o_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Zero-copy batch decompression using GPU device pointers directly.
        Calls turboquant_prod_dequantization_batch_direct (C++ API) which
        avoids all host round-trips, Python list/tuple construction, and
        per-block memcpy overhead.

        All input tensors must be contiguous and reside on CUDA.
        """
        batch_size = b_tensor.shape[0]
        if batch_size == 0:
            return torch.empty((0, self.block_size), dtype=torch.float32, device="cuda")

        output = torch.empty((batch_size, self.block_size), dtype=torch.float32, device="cuda")

        status = self._lib.turboquant_prod_dequantization_batch_direct(
            self._batch_ctx,
            ctypes.c_void_p(b_tensor.contiguous().data_ptr()),
            ctypes.c_void_p(q_tensor.contiguous().data_ptr()),
            ctypes.c_void_p(r_tensor.contiguous().data_ptr()),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_uint32(batch_size),
        )
        if status != 0:
            raise RuntimeError(f"turboquant_prod_dequantization_batch_direct failed with code {status}")

        # C++ output has incorrect magnitude (~3.16 instead of 1.0).
        # Renormalize to unit sphere, then rescale by original_l2.
        current_norms = output.norm(p=2, dim=1, keepdim=True).clamp_min(1e-6)
        output.div_(current_norms).mul_(o_tensor.unsqueeze(1))

        return output


    def close(self):
        """Cleanup batch context."""
        if self._batch_ctx and self._lib:
            self._lib.turboquant_batch_destroy(ctypes.byref(self._batch_ctx))
            self._batch_ctx = None
