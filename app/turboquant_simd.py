"""
TurboQuant SIMD (CPU) implementation.
Includes both single-threaded and multi-threaded variants.
"""

import ctypes
import os
import torch
from typing import Tuple, List
from abc import ABC, abstractmethod

# Force load PyTorch's OpenMP library globally so the C extension uses it
try:
    # This prevents the "__kmpc_for_static_fini" crash and thread collisions
    ctypes.CDLL("libiomp5.so", mode=ctypes.RTLD_GLOBAL)
except OSError:
    pass # If not found, it might already be loaded by torch

# ============================================================================
# ctypes Structures
# ============================================================================

class Vector(ctypes.Structure):
    """Mirrors vector_t from lin_alg.h"""
    _fields_ = [
        ("n", ctypes.c_size_t),
        ("vector", ctypes.POINTER(ctypes.c_float)),
    ]


class QuantizationResult(ctypes.Structure):
    """Mirrors quantization_result from turboquant.h"""
    _fields_ = [
        ("bstring", ctypes.POINTER(ctypes.c_uint8)),
        ("qjl", ctypes.POINTER(ctypes.c_uint8)),
        ("residual_l2", ctypes.c_float),
    ]


class TurboQuantBatchContextSIMD(ctypes.Structure):
    """Matches turboquant_batch_ctx_t from SIMD multi header."""
    _fields_ = [
        ("quantizer", ctypes.c_void_p),
        ("n_threads", ctypes.c_size_t),
        ("dims", ctypes.c_size_t),
        ("bit_width", ctypes.c_uint8),
        ("is_init", ctypes.c_uint8),
    ]


# ============================================================================
# Abstract Base
# ============================================================================

class TurboQuantCompressorBase(ABC):
    """Abstract base for all TurboQuant compressors."""

    def __init__(self, lib_path: str, context_path: str, block_size: int, bit_width: int):
        self.lib_path = lib_path
        self.context_path = context_path
        self.block_size = block_size
        self.bit_width = bit_width
        self._lib = None

        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"TurboQuant library not found: {lib_path}")

        self._lib = ctypes.CDLL(lib_path)
        self._init_library()
        self._init_context()

    @abstractmethod
    def _init_library(self):
        pass
    
    @abstractmethod
    def _init_context(self):
        pass
    
    @abstractmethod
    def compress_block(self, block: torch.Tensor) -> Tuple[float, float, bytes, bytes]:
        pass
    
    @abstractmethod
    def compress_chunk(self, blocks: List[torch.Tensor]) -> List[Tuple[float, float, bytes, bytes]]:
        pass
    
    @abstractmethod
    def decompress_chunk(self, results: List[Tuple[float, float, bytes, bytes]]) -> List[torch.Tensor]:
        pass
    
    @abstractmethod
    def close(self):
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ============================================================================
# SIMD Single (CPU Single-Threaded)
# ============================================================================

class SIMDSingleCompressor(TurboQuantCompressorBase):
    """
    TurboQuant SIMD Single-Threaded (CPU) implementation.
    Uses global state - no context pointer needed.
    """
    
    def _init_library(self):
        """Setup global-state function signatures."""
        lib = self._lib
        
        # Global functions
        lib.turboquant_init.argtypes = [ctypes.c_size_t, ctypes.c_uint8]
        lib.turboquant_init.restype = ctypes.c_uint8
        lib.turboquant_clean.argtypes = []
        lib.turboquant_clean.restype = None
        lib.turboquant_init_load.argtypes = [ctypes.c_char_p]
        lib.turboquant_init_load.restype = ctypes.c_uint8
        lib.turboquant_save.argtypes = [ctypes.c_char_p]
        lib.turboquant_save.restype = ctypes.c_uint8
        
        # Result management
        lib.turboquant_quantization_result_init.argtypes = []
        lib.turboquant_quantization_result_init.restype = ctypes.POINTER(QuantizationResult)
        lib.turboquant_quantization_result_destroy.argtypes = [
            ctypes.POINTER(ctypes.POINTER(QuantizationResult))
        ]
        lib.turboquant_quantization_result_destroy.restype = None
        
        # Product quantization
        lib.turboquant_prod_quantization.argtypes = [
            ctypes.POINTER(Vector),
            ctypes.POINTER(QuantizationResult),
        ]
        lib.turboquant_prod_quantization.restype = ctypes.c_uint8
        lib.turboquant_prod_dequantization.argtypes = [
            ctypes.POINTER(QuantizationResult)
        ]
        lib.turboquant_prod_dequantization.restype = ctypes.POINTER(Vector)
        
        # MSE quantization (alternative)
        lib.turboquant_mse_quantization.argtypes = [
            ctypes.POINTER(Vector),
            ctypes.POINTER(ctypes.c_uint8),
        ]
        lib.turboquant_mse_quantization.restype = ctypes.c_uint8
        lib.turboquant_mse_dequantization.argtypes = [ctypes.POINTER(ctypes.c_uint8)]
        lib.turboquant_mse_dequantization.restype = ctypes.POINTER(Vector)
    
    def _init_context(self):
        """Initialize global state."""
        status = self._lib.turboquant_init(
            ctypes.c_size_t(self.block_size),
            ctypes.c_uint8(self.bit_width)
        )
        if status != 0:
            raise RuntimeError(f"turboquant_init failed with code {status}")
        
        # Try to load existing context file
        if os.path.exists(self.context_path):
            status = self._lib.turboquant_init_load(self.context_path.encode("utf-8"))
            if status != 0:
                print(f"Warning: Failed to load context file {self.context_path}")
    
    def compress_block(self, block: torch.Tensor) -> Tuple[float, float, bytes, bytes]:
        """Compress a single block using CPU SIMD."""
        block = block.detach().to(dtype=torch.float32).contiguous()
        
        if block.numel() > self.block_size:
            raise ValueError(f"Block has {block.numel()} elements, expected <= {self.block_size}")
        
        # Handle padding
        if block.numel() < self.block_size:
            padding = torch.zeros(self.block_size - block.numel(), dtype=torch.float32)
            block = torch.cat([block, padding])
        
        original_l2 = torch.linalg.norm(block).item()
        
        if original_l2 < 1e-12:
            n_bstring = ((self.bit_width) * self.block_size + 7) // 8
            n_qjl = (self.block_size + 7) // 8
            return (original_l2, 0.0, bytes(n_bstring), bytes(n_qjl))
        
        # Create C vector
        c_ptr = ctypes.cast(block.data_ptr(), ctypes.POINTER(ctypes.c_float))
        vec = Vector(n=self.block_size, vector=c_ptr)
        
        # Allocate result
        result_ptr = self._lib.turboquant_quantization_result_init()
        if not result_ptr:
            raise MemoryError("Failed to allocate quantization result")
        
        try:
            status = self._lib.turboquant_prod_quantization(
                ctypes.byref(vec), result_ptr
            )
            if status != 0:
                raise RuntimeError(f"turboquant_prod_quantization failed with code {status}")
            
            res = result_ptr.contents
            n_bstring = ((self.bit_width) * self.block_size + 7) // 8
            n_qjl = (self.block_size + 7) // 8
            
            bstring_bytes = ctypes.string_at(res.bstring, n_bstring)
            qjl_bytes = ctypes.string_at(res.qjl, n_qjl)
            
            return (original_l2, res.residual_l2, bstring_bytes, qjl_bytes)
        finally:
            self._lib.turboquant_quantization_result_destroy(ctypes.byref(result_ptr))
    
    def compress_chunk(self, blocks: List[torch.Tensor]) -> List[Tuple[float, float, bytes, bytes]]:
        """Sequential compression for SIMD single."""
        return [self.compress_block(block) for block in blocks]
    
    def decompress_chunk(self, results: List[Tuple[float, float, bytes, bytes]]) -> List[torch.Tensor]:
        """Sequential decompression for SIMD single."""
        outputs = []
        n_bstring = ((self.bit_width) * self.block_size + 7) // 8
        
        for original_l2, residual_l2, bstring_bytes, qjl_bytes in results:
            if original_l2 < 1e-12:
                outputs.append(torch.zeros(self.block_size, dtype=torch.float32))
                continue
            
            # Create quantization result struct
            bstring_arr = (ctypes.c_uint8 * n_bstring).from_buffer_copy(bstring_bytes)
            qjl_arr = (ctypes.c_uint8 * ((self.block_size + 7) // 8)).from_buffer_copy(qjl_bytes)
            
            result = QuantizationResult(
                bstring=ctypes.cast(bstring_arr, ctypes.POINTER(ctypes.c_uint8)),
                qjl=ctypes.cast(qjl_arr, ctypes.POINTER(ctypes.c_uint8)),
                residual_l2=ctypes.c_float(residual_l2),
            )
            
            vec_ptr = self._lib.turboquant_prod_dequantization(ctypes.byref(result))
            if not vec_ptr:
                raise RuntimeError("turboquant_prod_dequantization returned null")
            
            vec = vec_ptr.contents
            output = torch.zeros(self.block_size, dtype=torch.float32)
            ctypes.memmove(output.data_ptr(), vec.vector, self.block_size * 4)
            output.mul_(original_l2)
            outputs.append(output)
            
            # Cleanup vector (using free from stdlib if available)
            libc = ctypes.CDLL(None)
            libc.free(vec.vector)
        
        return outputs
    
    def close(self):
        """Cleanup global state."""
        if self._lib:
            self._lib.turboquant_clean()


# ============================================================================
# SIMD Multi (CPU Multi-Threaded)
# ============================================================================

class SIMDBatchCompressor(TurboQuantCompressorBase):
    """
    TurboQuant SIMD Multi-Threaded (CPU) implementation.
    Uses batch context for parallel processing.
    """
    
    def __init__(self, lib_path: str, context_path: str, block_size: int, bit_width: int, n_threads: int = 8):
        self.n_threads = n_threads
        self._batch_ctx = None
        super().__init__(lib_path, context_path, block_size, bit_width)
    
    def _init_library(self):
        """Setup batch function signatures."""
        lib = self._lib
        
        # Global init/clean
        lib.turboquant_init.argtypes = [ctypes.c_size_t, ctypes.c_uint8]
        lib.turboquant_init.restype = ctypes.c_uint8
        lib.turboquant_clean.argtypes = []
        lib.turboquant_clean.restype = None
        
        # Batch context functions
        lib.turboquant_batch_init.argtypes = [
            ctypes.POINTER(ctypes.POINTER(TurboQuantBatchContextSIMD)),
            ctypes.c_size_t,
            ctypes.c_uint8,
            ctypes.c_size_t,
        ]
        lib.turboquant_batch_init.restype = ctypes.c_uint8
        
        lib.turboquant_batch_destroy.argtypes = [
            ctypes.POINTER(ctypes.POINTER(TurboQuantBatchContextSIMD))
        ]
        lib.turboquant_batch_destroy.restype = None
        
        lib.turboquant_batch_init_load.argtypes = [
            ctypes.POINTER(TurboQuantBatchContextSIMD),
            ctypes.c_char_p,
        ]
        lib.turboquant_batch_init_load.restype = ctypes.c_uint8
        
        lib.turboquant_batch_save.argtypes = [
            ctypes.POINTER(TurboQuantBatchContextSIMD),
            ctypes.c_char_p,
        ]
        lib.turboquant_batch_save.restype = ctypes.c_uint8
        
        # Batch operations
        lib.turboquant_batch_quantize.argtypes = [
            ctypes.POINTER(TurboQuantBatchContextSIMD),
            ctypes.POINTER(ctypes.POINTER(Vector)),
            ctypes.POINTER(QuantizationResult),
            ctypes.c_size_t,
        ]
        lib.turboquant_batch_quantize.restype = ctypes.c_uint8
        
        lib.turboquant_batch_dequantize.argtypes = [
            ctypes.POINTER(TurboQuantBatchContextSIMD),
            ctypes.POINTER(QuantizationResult),
            ctypes.c_size_t,
        ]
        lib.turboquant_batch_dequantize.restype = ctypes.POINTER(ctypes.POINTER(Vector))
        
        lib.turboquant_batch_results_destroy.argtypes = [
            ctypes.POINTER(ctypes.POINTER(ctypes.POINTER(Vector)))
        ]
        lib.turboquant_batch_results_destroy.restype = None
    
    def _init_context(self):
        """Initialize global state and batch context."""
        # First init global state
        status = self._lib.turboquant_init(
            ctypes.c_size_t(self.block_size),
            ctypes.c_uint8(self.bit_width)
        )
        if status != 0:
            raise RuntimeError(f"turboquant_init failed with code {status}")
        
        # Allocate batch context
        self._batch_ctx = ctypes.POINTER(TurboQuantBatchContextSIMD)()
        
        status = self._lib.turboquant_batch_init(
            ctypes.byref(self._batch_ctx),
            ctypes.c_size_t(self.block_size),
            ctypes.c_uint8(self.bit_width),
            ctypes.c_size_t(self.n_threads),
        )
        if status != 0:
            raise RuntimeError(f"turboquant_batch_init failed with code {status}")
        
        if not self._batch_ctx or not self._batch_ctx.contents.is_init:
            raise RuntimeError("Failed to initialize batch context")
        
        # Try to load existing context
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
        """Parallel batch compression using SIMD."""
        batch_size = len(blocks)
        if batch_size == 0:
            return []
        
        # Prepare input tensors on CPU
        cpu_blocks = []
        original_l2s = []
        
        for block in blocks:
            block = block.detach().to(dtype=torch.float32).contiguous()
            
            if block.numel() > self.block_size:
                raise ValueError(f"Block has {block.numel()} elements, expected <= {self.block_size}")
            
            if block.numel() < self.block_size:
                padding = torch.zeros(self.block_size - block.numel(), dtype=torch.float32)
                block = torch.cat([block, padding])
            
            original_l2 = torch.linalg.norm(block).item()
            original_l2s.append(original_l2)
            cpu_blocks.append(block)
        
        # Skip if all degenerate
        if all(l2 < 1e-12 for l2 in original_l2s):
            n_bstring = (self.bit_width * self.block_size + 7) // 8
            n_qjl = (self.block_size + 7) // 8
            return [(l2, 0.0, bytes(n_bstring), bytes(n_qjl)) for l2 in original_l2s]
        
        # Create vector array
        vectors = []
        vec_ptrs = []
        for block in cpu_blocks:
            c_ptr = ctypes.cast(block.data_ptr(), ctypes.POINTER(ctypes.c_float))
            vec = Vector(n=self.block_size, vector=c_ptr)
            vectors.append(vec)
            vec_ptrs.append(ctypes.pointer(vec))
        
        vec_array = (ctypes.POINTER(Vector) * batch_size)(*vec_ptrs)
        
        # Allocate results array
        results_array = (QuantizationResult * batch_size)()
        
        # Call batch quantize
        status = self._lib.turboquant_batch_quantize(
            self._batch_ctx,
            vec_array,
            results_array,
            ctypes.c_size_t(batch_size),
        )
        
        if status != 0:
            raise RuntimeError(f"turboquant_batch_quantize failed with code {status}")
        
        # Extract results
        n_bstring = (self.bit_width * self.block_size + 7) // 8
        n_qjl = (self.block_size + 7) // 8
        
        output = []
        for i, original_l2 in enumerate(original_l2s):
            if original_l2 < 1e-12:
                output.append((original_l2, 0.0, bytes(n_bstring), bytes(n_qjl)))
            else:
                res = results_array[i]
                bstring_bytes = ctypes.string_at(res.bstring, n_bstring)
                qjl_bytes = ctypes.string_at(res.qjl, n_qjl)
                output.append((original_l2, res.residual_l2, bstring_bytes, qjl_bytes))
        
        return output
    
    def decompress_chunk(self, results: List[Tuple[float, float, bytes, bytes]]) -> List[torch.Tensor]:
        """Parallel batch decompression using SIMD."""
        batch_size = len(results)
        if batch_size == 0:
            return []
        
        n_bstring = (self.bit_width * self.block_size + 7) // 8
        n_qjl = (self.block_size + 7) // 8
        
        # Create results array
        results_array = (QuantizationResult * batch_size)()
        keepalive_bstring = []
        keepalive_qjl = []
        
        for i, (original_l2, residual_l2, bstring_bytes, qjl_bytes) in enumerate(results):
            bstring_arr = (ctypes.c_uint8 * n_bstring).from_buffer_copy(bstring_bytes)
            qjl_arr = (ctypes.c_uint8 * n_qjl).from_buffer_copy(qjl_bytes)
            keepalive_bstring.append(bstring_arr)
            keepalive_qjl.append(qjl_arr)
            
            results_array[i] = QuantizationResult(
                bstring=ctypes.cast(bstring_arr, ctypes.POINTER(ctypes.c_uint8)),
                qjl=ctypes.cast(qjl_arr, ctypes.POINTER(ctypes.c_uint8)),
                residual_l2=ctypes.c_float(residual_l2),
            )
        
        # Call batch dequantize
        c_vectors_ptr = self._lib.turboquant_batch_dequantize(
            self._batch_ctx,
            results_array,
            ctypes.c_size_t(batch_size),
        )
        
        if not c_vectors_ptr:
            raise RuntimeError("turboquant_batch_dequantize returned null")
        
        outputs = []
        try:
            for i, (original_l2, _, _, _) in enumerate(results):
                if original_l2 < 1e-12:
                    outputs.append(torch.zeros(self.block_size, dtype=torch.float32))
                else:
                    vec_ptr = c_vectors_ptr[i]
                    vec = vec_ptr.contents
                    output = torch.zeros(self.block_size, dtype=torch.float32)
                    ctypes.memmove(output.data_ptr(), vec.vector, self.block_size * 4)
                    output.mul_(original_l2)
                    outputs.append(output)
        finally:
            self._lib.turboquant_batch_results_destroy(ctypes.byref(c_vectors_ptr))
        
        return outputs
    
    def close(self):
        """Cleanup batch context and global state."""
        if self._batch_ctx and self._lib:
            self._lib.turboquant_batch_destroy(ctypes.byref(self._batch_ctx))
            self._batch_ctx = None
        if self._lib:
            self._lib.turboquant_clean()


# ============================================================================
# Factory Function
# ============================================================================

def get_compressor_for_variant(lib_path: str, context_path: str, block_size: int, bit_width: int, variant: str = "auto", n_streams: int = 32):
    """
    Factory function to create the appropriate compressor instance based on variant.
    
    Args:
        lib_path: Path to the TurboQuant shared library
        context_path: Path to the context file
        block_size: Block/dimension size for quantization
        bit_width: Bit width for quantization
        variant: Variant to use (simd, simd-multi, simt, simt-multi, auto, etc.)
        n_streams: Number of streams/threads for batch variants
    
    Returns:
        Instantiated compressor object
    """
    variant = variant.lower().strip()
    
    # Normalize legacy names
    variant_map = {
        "simt-batch": "simt-multi",
        "cuda-batch": "simt-multi",
        "cuda": "simt",
        "cpu": "simd",
        "cpu-batch": "simd-multi",
        "cpu-multi": "simd-multi",
        "auto": "simd-multi",  # Default to CPU batch
    }
    variant = variant_map.get(variant, variant)
    
    if variant == "simd":
        return SIMDSingleCompressor(lib_path, context_path, block_size, bit_width)
    elif variant == "simd-multi":
        return SIMDBatchCompressor(lib_path, context_path, block_size, bit_width, n_streams)
    elif variant == "simt":
        from app.turboquant_simt import SIMTSingleCompressor
        return SIMTSingleCompressor(lib_path, context_path, block_size, bit_width)
    elif variant == "simt-multi":
        from app.turboquant_simt import SIMTBatchCompressor
        return SIMTBatchCompressor(lib_path, context_path, block_size, bit_width, n_streams)
    else:
        raise ValueError(f"Unknown variant: {variant}. Use: simd, simd-multi, simt, simt-multi")


# Backward compatibility aliases
TurboQuantCompressor = SIMDSingleCompressor
TurboQuantBatchCompressor = SIMDBatchCompressor

