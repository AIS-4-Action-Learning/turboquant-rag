"""
TurboQuant SIMT (CUDA) implementation.
Includes both single-stream and multi-stream variants.
"""

import ctypes
import os
import sys
from typing import Tuple, List

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
        ("is_init", ctypes.c_uint8),
    ]


class QuantizationBatchResult(ctypes.Structure):
    """Mirrors quantization_batch_result from simt headers."""
    _fields_ = [
        ("results", ctypes.POINTER(QuantizationResult)),
        ("n_results", ctypes.c_uint8),
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
            if status != 0:
                print(f"Warning: Failed to load context file {self.context_path}")
    
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
        
        for original_l2, residual_l2, bstring_bytes, qjl_bytes in results:
            if original_l2 < 1e-12:
                outputs.append(torch.zeros(self.block_size, dtype=torch.bfloat16, device="cuda"))
                continue
            
            # Create quantization result with device pointers
            # Note: The C API expects device pointers for bstring/qjl
            # We need to allocate GPU memory and copy data there
            bstring_arr = (ctypes.c_uint8 * n_bstring).from_buffer_copy(bstring_bytes)
            qjl_arr = (ctypes.c_uint8 * n_qjl).from_buffer_copy(qjl_bytes)
            
            result = QuantizationResult(
                bstring=ctypes.cast(bstring_arr, ctypes.POINTER(ctypes.c_uint8)),
                qjl=ctypes.cast(qjl_arr, ctypes.POINTER(ctypes.c_uint8)),
                residual_l2=ctypes.c_float(residual_l2),
            )
            
            vec_ptr = self._lib.turboquant_prod_dequantization(
                self._context,
                ctypes.byref(result)
            )
            if not vec_ptr:
                raise RuntimeError("turboquant_prod_dequantization returned null")
            
            # Copy result from device to host
            vec = vec_ptr.contents
            output = torch.zeros(self.block_size, dtype=torch.float32, device="cuda")
            # Need cudaMemcpy or similar here - using torch from_dlpack or similar
            # For now, simplified approach
            output_host = torch.zeros(self.block_size, dtype=torch.float32)
            ctypes.memmove(output_host.data_ptr(), vec.vector, self.block_size * 4)
            output = output_host.to("cuda")
            output.mul_(original_l2)
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
    
    def __init__(self, lib_path: str, context_path: str, block_size: int, bit_width: int, n_streams: int = 8):
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
            ctypes.c_uint8,
        ]
        lib.turboquant_prod_quantization_batch.restype = ctypes.c_uint8
        
        lib.turboquant_prod_dequantization_batch.argtypes = [
            ctypes.POINTER(TurboQuantBatchContext),
            ctypes.POINTER(QuantizationBatchResult),
        ]
        lib.turboquant_prod_dequantization_batch.restype = ctypes.POINTER(ctypes.POINTER(Vector))
    
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
            if status != 0:
                print(f"Warning: Failed to load context file {self.context_path}")
    
    def compress_block(self, block: torch.Tensor) -> Tuple[float, float, bytes, bytes]:
        """Single block - delegates to batch with size 1."""
        results = self.compress_chunk([block])
        return results[0]
    
    def compress_chunk(self, blocks: List[torch.Tensor]) -> List[Tuple[float, float, bytes, bytes]]:
        """Parallel batch compression using multiple CUDA streams."""
        batch_size = len(blocks)
        if batch_size == 0:
            return []
        if batch_size > 255:
            raise ValueError(f"Batch size {batch_size} exceeds C API limit (max 255)")
        
        # Move blocks to CUDA and process
        original_l2s = []
        active_indices = []
        active_blocks_f32 = []
        
        for i, block in enumerate(blocks):
            if not block.is_cuda:
                block = block.to("cuda")
            
            gpu_block = block.detach().to(dtype=torch.bfloat16).contiguous()
            
            if gpu_block.numel() > self.block_size:
                raise ValueError(
                    f"Block at index {i} has {gpu_block.numel()} elements, expected <= {self.block_size}"
                )
            
            # Handle padding
            if gpu_block.numel() < self.block_size:
                padding = torch.zeros(
                    self.block_size - gpu_block.numel(),
                    dtype=torch.bfloat16,
                    device="cuda"
                )
                gpu_block = torch.cat([gpu_block, padding])
            
            original_l2 = torch.linalg.norm(gpu_block.float()).item()
            original_l2s.append(original_l2)
            
            if original_l2 >= 1e-12:
                active_indices.append(i)
                active_blocks_f32.append(gpu_block.float().contiguous())
        
        n_bstring = (self.bit_width * self.block_size + 7) // 8
        n_qjl = (self.block_size + 7) // 8
        
        # Initialize outputs
        results = [
            (original_l2, 0.0, bytes(n_bstring), bytes(n_qjl))
            for original_l2 in original_l2s
        ]
        
        if not active_blocks_f32:
            return results
        
        # Process in chunks <= n_streams
        for start in range(0, len(active_blocks_f32), self.n_streams):
            end = min(start + self.n_streams, len(active_blocks_f32))
            chunk_size = end - start
            chunk_blocks = active_blocks_f32[start:end]
            chunk_indices = active_indices[start:end]
            
            # Create vector array
            vectors = []
            vec_ptrs = []
            for block in chunk_blocks:
                c_ptr = ctypes.cast(block.data_ptr(), ctypes.POINTER(ctypes.c_float))
                vec = Vector(n=self.block_size, vector=c_ptr)
                vectors.append(vec)
                vec_ptrs.append(ctypes.pointer(vec))
            
            vec_array = (ctypes.POINTER(Vector) * chunk_size)(*vec_ptrs)
            
            # Allocate batch result
            batch_result = QuantizationBatchResult()
            batch_result.n_results = 0
            batch_result.results = None
            
            status = self._lib.turboquant_prod_quantization_batch(
                self._batch_ctx,
                vec_array,
                ctypes.byref(batch_result),
                ctypes.c_uint8(chunk_size),
            )
            
            if status != 0:
                raise RuntimeError(f"turboquant_prod_quantization_batch failed with code {status}")
            
            # Extract results
            for chunk_i, original_i in enumerate(chunk_indices):
                res = batch_result.results[chunk_i]
                bstring_bytes = ctypes.string_at(res.bstring, n_bstring)
                qjl_bytes = ctypes.string_at(res.qjl, n_qjl)
                results[original_i] = (
                    original_l2s[original_i],
                    float(res.residual_l2),
                    bstring_bytes,
                    qjl_bytes,
                )
        
        return results
    
    def decompress_chunk(self, results: List[Tuple[float, float, bytes, bytes]]) -> List[torch.Tensor]:
        """Parallel batch decompression using multiple CUDA streams."""
        batch_size = len(results)
        if batch_size == 0:
            return []
        if batch_size > 255:
            raise ValueError(f"Batch size {batch_size} exceeds C API limit (max 255)")
        
        if not self._batch_ctx or not self._batch_ctx.contents.is_init:
            raise RuntimeError("TurboQuant batch context is not initialized")
        if self._cuda_memcpy is None:
            raise RuntimeError("CUDA runtime is required for decompression")
        
        n_bstring = (self.bit_width * self.block_size + 7) // 8
        n_qjl = (self.block_size + 7) // 8
        
        outputs = [
            torch.zeros(self.block_size, dtype=torch.bfloat16, device="cuda")
            for _ in range(batch_size)
        ]
        active_indices = []
        
        for i, (original_l2, _, _, _) in enumerate(results):
            if original_l2 >= 1e-12:
                active_indices.append(i)
        
        if not active_indices:
            return outputs
        
        # Process in chunks
        for start in range(0, len(active_indices), self.n_streams):
            end = min(start + self.n_streams, len(active_indices))
            chunk_indices = active_indices[start:end]
            chunk_size = len(chunk_indices)
            
            # Create results array
            c_results_array = (QuantizationResult * chunk_size)()
            keepalive_bstring = []
            keepalive_qjl = []
            
            for chunk_i, original_i in enumerate(chunk_indices):
                _, residual_l2, bstring_bytes, qjl_bytes = results[original_i]
                
                bstring_arr = (ctypes.c_uint8 * n_bstring).from_buffer_copy(bstring_bytes)
                qjl_arr = (ctypes.c_uint8 * n_qjl).from_buffer_copy(qjl_bytes)
                keepalive_bstring.append(bstring_arr)
                keepalive_qjl.append(qjl_arr)
                
                c_results_array[chunk_i] = QuantizationResult(
                    bstring=ctypes.cast(bstring_arr, ctypes.POINTER(ctypes.c_uint8)),
                    qjl=ctypes.cast(qjl_arr, ctypes.POINTER(ctypes.c_uint8)),
                    residual_l2=ctypes.c_float(float(residual_l2)),
                )
            
            batch_input = QuantizationBatchResult(
                results=ctypes.cast(c_results_array, ctypes.POINTER(QuantizationResult)),
                n_results=ctypes.c_uint8(chunk_size),
            )
            
            c_vectors = self._lib.turboquant_prod_dequantization_batch(
                self._batch_ctx,
                ctypes.byref(batch_input)
            )
            
            if not c_vectors:
                raise RuntimeError("turboquant_prod_dequantization_batch returned null")
            
            try:
                for chunk_i, original_i in enumerate(chunk_indices):
                    vec_ptr = c_vectors[chunk_i]
                    if not vec_ptr:
                        continue
                    
                    vec = vec_ptr.contents
                    out_f32 = torch.empty(self.block_size, dtype=torch.float32, device="cuda")
                    
                    copy_status = self._cuda_memcpy(
                        ctypes.c_void_p(out_f32.data_ptr()),
                        ctypes.cast(vec.vector, ctypes.c_void_p),
                        ctypes.c_size_t(self.block_size * 4),
                        ctypes.c_int(self._cuda_memcpy_device_to_device),
                    )
                    
                    if copy_status == 0:
                        out_f32.mul_(float(results[original_i][0]))
                        outputs[original_i] = out_f32.to(dtype=torch.bfloat16)
            finally:
                self._libc.free(ctypes.cast(c_vectors, ctypes.c_void_p))
        
        return outputs
    
    def close(self):
        """Cleanup batch context."""
        if self._batch_ctx and self._lib:
            self._lib.turboquant_batch_destroy(ctypes.byref(self._batch_ctx))
            self._batch_ctx = None