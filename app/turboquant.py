import ctypes
import os
import torch
from typing import Tuple, List


class Vector(ctypes.Structure):
    """Mirrors vector_t from turboquant.h"""
    _fields_ = [
        ("n", ctypes.c_size_t),
        ("vector", ctypes.POINTER(ctypes.c_float)),
    ]


class QuantizationResult(ctypes.Structure):
    """
    Mirrors quantization_result from turboquant.h.
    Allocated via turboquant_quantization_result_init() on the C side.
    """
    _fields_ = [
        ("bstring", ctypes.POINTER(ctypes.c_uint8)),  # packed centroid indices
        ("qjl", ctypes.POINTER(ctypes.c_uint8)),      # packed residual sign bits
        ("residual_l2", ctypes.c_float),              # L2 norm of residual
    ]


class TurboQuantContext(ctypes.Structure):
    """Mirrors turboquant_context_t from turboquant.h"""
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


class TurboQuantCompressor:
    """
    Python wrapper for TurboQuant CUDA quantization library (single stream).
    Handles context initialization, single-block processing, and cleanup.
    """

    def __init__(
        self,
        lib_path: str,
        context_path: str,
        block_size: int,
        bit_width: int,
    ):
        self.block_size = block_size
        self.bit_width = bit_width
        self.context_path = context_path
        self._lib = None
        self._context = None
        self._init_library(lib_path)
        self._init_context()

    def _init_library(self, lib_path: str):
        """Load the TurboQuant shared library and setup function signatures."""
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"TurboQuant library not found: {lib_path}")

        self._lib = ctypes.CDLL(lib_path)
        lib = self._lib

        # turboquant_init - allocate and initialize context
        lib.turboquant_init.argtypes = [
            ctypes.POINTER(ctypes.POINTER(TurboQuantContext)),
            ctypes.c_size_t,
            ctypes.c_uint8,
        ]
        lib.turboquant_init.restype = ctypes.c_uint8

        # turboquant_init_load - load context from file
        lib.turboquant_init_load.argtypes = [
            ctypes.POINTER(TurboQuantContext),
            ctypes.c_char_p,
        ]
        lib.turboquant_init_load.restype = ctypes.c_uint8

        # turboquant_clean - cleanup context resources
        lib.turboquant_clean.argtypes = [ctypes.POINTER(TurboQuantContext)]
        lib.turboquant_clean.restype = None

        # turboquant_context_destroy - destroy context completely
        lib.turboquant_context_destroy.argtypes = [
            ctypes.POINTER(ctypes.POINTER(TurboQuantContext))
        ]
        lib.turboquant_context_destroy.restype = None

        # turboquant_quantization_result_init
        lib.turboquant_quantization_result_init.argtypes = []
        lib.turboquant_quantization_result_init.restype = ctypes.POINTER(
            QuantizationResult
        )

        # turboquant_quantization_result_destroy
        lib.turboquant_quantization_result_destroy.argtypes = [
            ctypes.POINTER(ctypes.POINTER(QuantizationResult))
        ]
        lib.turboquant_quantization_result_destroy.restype = None

        # turboquant_prod_quantization - main quantization function
        lib.turboquant_prod_quantization.argtypes = [
            ctypes.POINTER(TurboQuantContext),
            ctypes.POINTER(Vector),
            ctypes.POINTER(QuantizationResult),
        ]
        lib.turboquant_prod_quantization.restype = ctypes.c_uint8

    def _init_context(self):
        """Initialize TurboQuant context, loading from file if available."""
        # Allocate context pointer
        self._context = ctypes.POINTER(TurboQuantContext)()

        # First, initialize with dimensions and bit_width
        status = self._lib.turboquant_init(
            ctypes.byref(self._context),
            ctypes.c_size_t(self.block_size),
            ctypes.c_uint8(self.bit_width),
        )
        if status != 0:
            raise RuntimeError(f"turboquant_init failed with code {status}")

        # Try to load existing context file
        if os.path.exists(self.context_path):
            status = self._lib.turboquant_init_load(
                self._context, self.context_path.encode("utf-8")
            )
            if status != 0:
                print(f"Warning: Failed to load context file {self.context_path}")
                print("Using freshly initialized context (codebook may differ)")

        if not self._context or not self._context.contents.is_init:
            raise RuntimeError("Failed to initialize TurboQuant context")

    def compress_block(self, block: torch.Tensor) -> Tuple[float, float, bytes, bytes]:
        """
        Compress a single block of weights.

        Args:
            block: Tensor of shape [block_size] with float values

        Returns:
            Tuple of (original_l2, residual_l2, bstring_bytes, qjl_bytes)
        """
        # Keep the Python-side path in BF16 on GPU.
        block = block.detach().to(dtype=torch.bfloat16, device="cuda").contiguous()

        if block.numel() > self.block_size:
            raise ValueError(
                f"Block has {block.numel()} elements, expected <= {self.block_size}"
            )

        # Handle padding if block is smaller than block_size
        if block.numel() < self.block_size:
            padding = torch.zeros(
                self.block_size - block.numel(),
                dtype=torch.bfloat16,
                device=block.device,
            )
            block = torch.cat([block, padding])

        # Compute original L2 norm in FP32 for numerical stability.
        original_l2 = torch.linalg.norm(block.float()).item()

        # Handle degenerate blocks
        if original_l2 < 1e-12:
            n_bstring = (self.bit_width * self.block_size + 7) // 8
            n_qjl = (self.block_size + 7) // 8
            return (original_l2, 0.0, bytes(n_bstring), bytes(n_qjl))

        # C API currently expects float* (FP32), so cast right before handoff.
        block_f32 = block.float().contiguous()

        # Create C vector structure
        c_ptr = ctypes.cast(block_f32.data_ptr(), ctypes.POINTER(ctypes.c_float))
        vec = Vector(n=self.block_size, vector=c_ptr)

        # Allocate result structure
        result_ptr = self._lib.turboquant_quantization_result_init()
        if not result_ptr:
            raise MemoryError("Failed to allocate quantization result")

        try:
            # Run product quantization
            status = self._lib.turboquant_prod_quantization(
                self._context, ctypes.byref(vec), result_ptr
            )
            if status != 0:
                raise RuntimeError(f"turboquant_prod_quantization failed with code {status}")

            res = result_ptr.contents

            # Extract compressed data sizes
            n_bstring = (self.bit_width * self.block_size + 7) // 8
            n_qjl = (self.block_size + 7) // 8

            # Copy data from C buffers to Python bytes
            bstring_bytes = ctypes.string_at(res.bstring, n_bstring)
            qjl_bytes = ctypes.string_at(res.qjl, n_qjl)

            return (original_l2, res.residual_l2, bstring_bytes, qjl_bytes)

        finally:
            self._lib.turboquant_quantization_result_destroy(
                ctypes.byref(result_ptr)
            )

    def compress_chunk(
        self, blocks: List[torch.Tensor]
    ) -> List[Tuple[float, float, bytes, bytes]]:
        """
        Compress a chunk of weight blocks (128 blocks at a time).

        Args:
            blocks: List of tensors, each of shape [block_size]

        Returns:
            List of (original_l2, residual_l2, bstring_bytes, qjl_bytes) tuples
        """
        return [self.compress_block(block) for block in blocks]

    def close(self):
        """Clean up TurboQuant context."""
        if self._context and self._lib:
            self._lib.turboquant_context_destroy(ctypes.byref(self._context))
            self._context = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class TurboQuantBatchContext(ctypes.Structure):
    """Mirrors turboquant_batch_context_t from turboquant.h"""
    _fields_ = [
        ("contexts", ctypes.POINTER(ctypes.POINTER(TurboQuantContext))),
        ("n_streams", ctypes.c_uint8),
        ("dims", ctypes.c_size_t),
        ("bit_width", ctypes.c_uint8),
        ("is_init", ctypes.c_uint8),
    ]


class QuantizationBatchResult(ctypes.Structure):
    """Mirrors quantization_batch_result from turboquant.h"""
    _fields_ = [
        ("results", ctypes.POINTER(QuantizationResult)),
        ("n_results", ctypes.c_uint8),
    ]


class TurboQuantBatchCompressor:
    """
    Python wrapper for TurboQuant CUDA quantization library with multi-stream support.
    Processes multiple blocks in parallel using multiple CUDA streams.
    """

    def __init__(
        self,
        lib_path: str,
        context_path: str,
        block_size: int,
        bit_width: int,
        n_streams: int = 8,
    ):
        self.block_size = block_size
        self.bit_width = bit_width
        self.context_path = context_path
        self.n_streams = n_streams
        self._lib = None
        self._batch_ctx = None
        self._libc = None
        self._cudart = None
        self._cuda_memcpy = None
        self._cuda_memcpy_device_to_device = 3
        self._init_library(lib_path)
        self._init_batch_context()

    def _init_library(self, lib_path: str):
        """Load the TurboQuant shared library and setup function signatures."""
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"TurboQuant library not found: {lib_path}")

        self._lib = ctypes.CDLL(lib_path)
        lib = self._lib
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

        # Batch context functions
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

        # Batch quantization function
        lib.turboquant_prod_quantization_batch.argtypes = [
            ctypes.POINTER(TurboQuantBatchContext),
            ctypes.POINTER(ctypes.POINTER(Vector)),
            ctypes.POINTER(QuantizationBatchResult),
            ctypes.c_uint8,
        ]
        lib.turboquant_prod_quantization_batch.restype = ctypes.c_uint8

        # Batch dequantization function
        lib.turboquant_prod_dequantization_batch.argtypes = [
            ctypes.POINTER(TurboQuantBatchContext),
            ctypes.POINTER(QuantizationBatchResult),
        ]
        lib.turboquant_prod_dequantization_batch.restype = ctypes.POINTER(
            ctypes.POINTER(Vector)
        )

    def _init_batch_context(self):
        """Initialize TurboQuant batch context with multiple streams."""
        # Allocate batch context pointer
        self._batch_ctx = ctypes.POINTER(TurboQuantBatchContext)()

        # Initialize with multiple streams
        status = self._lib.turboquant_batch_init(
            ctypes.byref(self._batch_ctx),
            ctypes.c_size_t(self.block_size),
            ctypes.c_uint8(self.bit_width),
            ctypes.c_uint8(self.n_streams),
        )
        if status != 0:
            raise RuntimeError(f"turboquant_batch_init failed with code {status}")

        # Try to load existing context file
        if os.path.exists(self.context_path):
            status = self._lib.turboquant_batch_init_load(
                self._batch_ctx, self.context_path.encode("utf-8")
            )
            if status != 0:
                print(f"Warning: Failed to load context file {self.context_path}")
                print("Using freshly initialized context (codebook may differ)")

        if not self._batch_ctx or not self._batch_ctx.contents.is_init:
            raise RuntimeError("Failed to initialize TurboQuant batch context")

    def compress_batch(self, blocks: List[torch.Tensor]) -> List[Tuple[float, float, bytes, bytes]]:
        """
        Compress a batch of weight blocks using multiple CUDA streams.

        Args:
            blocks: List of tensors, each of shape [block_size] on GPU

        Returns:
            List of (original_l2, residual_l2, bstring_bytes, qjl_bytes) tuples
        """
        batch_size = len(blocks)
        if batch_size == 0:
            return []
        if batch_size > 255:
            raise ValueError(
                f"Batch size {batch_size} exceeds C API limit (max 255)"
            )

        original_l2s = []
        active_indices = []
        active_blocks_f32 = []

        for i, block in enumerate(blocks):
            gpu_block = block.detach().to(dtype=torch.bfloat16, device="cuda").contiguous()

            if gpu_block.numel() > self.block_size:
                raise ValueError(
                    f"Block at index {i} has {gpu_block.numel()} elements, expected <= {self.block_size}"
                )

            # Handle padding if needed
            if gpu_block.numel() < self.block_size:
                padding = torch.zeros(
                    self.block_size - gpu_block.numel(),
                    dtype=torch.bfloat16,
                    device=gpu_block.device,
                )
                gpu_block = torch.cat([gpu_block, padding])

            # Compute norms in FP32 for numerical stability.
            original_l2 = torch.linalg.norm(gpu_block.float()).item()
            original_l2s.append(original_l2)

            # Keep only non-degenerate blocks for C API call.
            if original_l2 >= 1e-12:
                active_indices.append(i)
                active_blocks_f32.append(gpu_block.float().contiguous())

        n_bstring = (self.bit_width * self.block_size + 7) // 8
        n_qjl = (self.block_size + 7) // 8

        # Initialize outputs; degenerate entries remain zero-payload.
        results = [
            (original_l2, 0.0, bytes(n_bstring), bytes(n_qjl))
            for original_l2 in original_l2s
        ]

        if not active_blocks_f32:
            return results

        active_batch_size = len(active_blocks_f32)
        # C batch API uses per-stream shared buffers; process in micro-batches
        # <= n_streams to avoid pointer aliasing across outputs.
        for start in range(0, active_batch_size, self.n_streams):
            end = min(start + self.n_streams, active_batch_size)
            chunk_size = end - start
            chunk_blocks = active_blocks_f32[start:end]
            chunk_indices = active_indices[start:end]

            vectors = []
            vec_ptrs = []
            for block in chunk_blocks:
                c_ptr = ctypes.cast(block.data_ptr(), ctypes.POINTER(ctypes.c_float))
                vec = Vector(n=self.block_size, vector=c_ptr)
                vectors.append(vec)
                vec_ptrs.append(ctypes.pointer(vec))

            vec_array = (ctypes.POINTER(Vector) * chunk_size)(*vec_ptrs)

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
            if not batch_result.results:
                raise RuntimeError("turboquant_prod_quantization_batch returned null results")
            if batch_result.n_results < chunk_size:
                raise RuntimeError(
                    f"turboquant_prod_quantization_batch returned {batch_result.n_results} results for {chunk_size} inputs"
                )
            try:
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
            finally:
                self._libc.free(ctypes.cast(batch_result.results, ctypes.c_void_p))

        return results

    def compress_chunk(self, blocks: List[torch.Tensor]) -> List[Tuple[float, float, bytes, bytes]]:
        """
        Compress a chunk of weight blocks using batch processing.

        Args:
            blocks: List of tensors, each of shape [block_size]

        Returns:
            List of (original_l2, residual_l2, bstring_bytes, qjl_bytes) tuples
        """
        return self.compress_batch(blocks)

    def decompress_chunk(self, results: List[Tuple[float, float, bytes, bytes]]) -> List[torch.Tensor]:
        """
        Decompress a chunk of weight blocks using batch processing.

        Args:
            results: List of (original_l2, residual_l2, bstring_bytes, qjl_bytes)

        Returns:
            List of decompressed tensors in BF16, each of shape [block_size].
        """
        batch_size = len(results)
        if batch_size == 0:
            return []
        if batch_size > 255:
            raise ValueError(
                f"Batch size {batch_size} exceeds C API limit (max 255)"
            )

        if not self._batch_ctx or not self._batch_ctx.contents.is_init:
            raise RuntimeError("TurboQuant batch context is not initialized")
        if self._cuda_memcpy is None:
            raise RuntimeError("CUDA runtime (libcudart) is required for decompression")

        n_bstring = (self.bit_width * self.block_size + 7) // 8
        n_qjl = (self.block_size + 7) // 8

        outputs = [
            torch.zeros(self.block_size, dtype=torch.bfloat16, device="cuda")
            for _ in range(batch_size)
        ]
        active_indices = []

        for i, (original_l2, _residual_l2, bstring_bytes, qjl_bytes) in enumerate(results):
            if len(bstring_bytes) != n_bstring:
                raise ValueError(
                    f"Invalid bstring size at index {i}: got {len(bstring_bytes)}, expected {n_bstring}"
                )
            if len(qjl_bytes) != n_qjl:
                raise ValueError(
                    f"Invalid qjl size at index {i}: got {len(qjl_bytes)}, expected {n_qjl}"
                )
            if original_l2 >= 1e-12:
                active_indices.append(i)

        if not active_indices:
            return outputs

        active_batch_size = len(active_indices)

        # C batch API uses per-stream context buffers; process in chunks
        # <= n_streams so each output maps to a unique context buffer.
        for start in range(0, active_batch_size, self.n_streams):
            end = min(start + self.n_streams, active_batch_size)
            chunk_indices = active_indices[start:end]
            chunk_size = len(chunk_indices)

            c_results_array = (QuantizationResult * chunk_size)()
            keepalive_bstring_arrays = []
            keepalive_qjl_arrays = []
            for chunk_i, original_i in enumerate(chunk_indices):
                _original_l2, residual_l2, bstring_bytes, qjl_bytes = results[original_i]

                bstring_arr = (ctypes.c_uint8 * n_bstring).from_buffer_copy(bstring_bytes)
                qjl_arr = (ctypes.c_uint8 * n_qjl).from_buffer_copy(qjl_bytes)
                keepalive_bstring_arrays.append(bstring_arr)
                keepalive_qjl_arrays.append(qjl_arr)

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
                self._batch_ctx, ctypes.byref(batch_input)
            )
            if not c_vectors:
                raise RuntimeError("turboquant_prod_dequantization_batch returned null")

            try:
                for chunk_i, original_i in enumerate(chunk_indices):
                    vec_ptr = c_vectors[chunk_i]
                    if not vec_ptr:
                        raise RuntimeError(
                            f"turboquant_prod_dequantization_batch returned null vector at index {chunk_i}"
                        )

                    vec = vec_ptr.contents
                    if vec.n != self.block_size:
                        raise RuntimeError(
                            f"Decompressed vector has {vec.n} elements, expected {self.block_size}"
                        )

                    out_f32 = torch.empty(self.block_size, dtype=torch.float32, device="cuda")
                    copy_status = self._cuda_memcpy(
                        ctypes.c_void_p(out_f32.data_ptr()),
                        ctypes.cast(vec.vector, ctypes.c_void_p),
                        ctypes.c_size_t(self.block_size * ctypes.sizeof(ctypes.c_float)),
                        ctypes.c_int(self._cuda_memcpy_device_to_device),
                    )
                    if copy_status != 0:
                        raise RuntimeError(
                            f"cudaMemcpy failed with code {copy_status} while copying decompressed vector"
                        )

                    # TurboQuant core quantizes normalized vectors; restore original scale.
                    out_f32.mul_(float(results[original_i][0]))
                    outputs[original_i] = out_f32.to(dtype=torch.bfloat16)
            finally:
                self._libc.free(ctypes.cast(c_vectors, ctypes.c_void_p))

        return outputs

    def close(self):
        """Clean up TurboQuant batch context."""
        if self._batch_ctx and self._lib:
            self._lib.turboquant_batch_destroy(ctypes.byref(self._batch_ctx))
            self._batch_ctx = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
