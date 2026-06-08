import traceback as tr
import ctypes
import os
from pathlib import Path
from typing import List, Any, Optional, Tuple

from numpy import isin
import torch
import json

from app import (
    PROJECT_ROOT, CHECKPOINT_PATH, CONTEXT_PATH, DEFAULT_BIT_WIDTH,
    LIB_PATH, DEFAULT_DIMENSIONS, PARAMS_PATH, 
    TOKENIZER_PATH, get_turboquant_lib_path, TURBOQUANT_VARIANT,
    get_compressor_for_variant
)

from llama_models.llama3.args import ModelArgs
from llama_models.llama3.model import Transformer
from llama_models.llama3.tokenizer import Tokenizer


def _setup_single_process_distributed(device="cuda"):
    """Initialize distributed environment for single-process inference."""
    import torch.distributed as dist
    import fairscale.nn.model_parallel.initialize as fs_init

    backend = "nccl" if device == "cuda" else "gloo"

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="tcp://127.0.0.1:23456",
            world_size=1,
            rank=0
        )

    if not fs_init.model_parallel_is_initialized():
        fs_init.initialize_model_parallel(1)
        print(f"✅ Model parallel group initialized (backend={backend}).")

    # --- Vacuum the NCCL Ghost Error ---
    if backend == "nccl" and device == "cuda":
        try:
            cudart = ctypes.CDLL("libcudart.so")
            cudart.cudaDeviceSynchronize()
            _ = cudart.cudaGetLastError() # This consumes Error 11 from the driver queue
        except Exception:
            pass

class Llama:
    def __init__(self, device : str = "cuda", is_batch : bool = True):
        try:
            # Determine which variant to use
            # Priority: explicit LIB_PATH > TURBOQUANT_VARIANT > auto (device + is_batch)
            self.lib_path = get_turboquant_lib_path(device=device, is_batch=is_batch)

            # Determine variant name from the library path for factory function
            # The variant is the parent directory name of libturboquant.so
            lib_dir = Path(self.lib_path).parent.name

            # If LIB_PATH is set, use the directory name as variant
            # Otherwise use TURBOQUANT_VARIANT or auto-derived
            if LIB_PATH:
                self.variant = lib_dir
            else:
                self.variant = TURBOQUANT_VARIANT.lower() if TURBOQUANT_VARIANT else (
                    "simt-multi" if device == "cuda" and is_batch else
                    "simt" if device == "cuda" else
                    "simd-multi" if is_batch else "simd"
                )

            # Read the hyperparameters from params.json
            with open(PARAMS_PATH, "r") as f:
                self.params = json.load(f)

            # Initialize the tokenizer from tokenizer.model
            self.tokenizer = Tokenizer(TOKENIZER_PATH)

            # Set the device attribute to the input device argument
            self.device = device

            self.checkpoints = torch.load(
                str(CHECKPOINT_PATH),
                mmap=True,
                weights_only=True,
                map_location="cpu" # map to CPU address space first
            )


        except Exception as e:
            tr.print_exc()

    def input_encoding(self, input_seq: str) -> tuple[List[int], torch.Tensor]:
      try:
          tokens = self.tokenizer.encode(input_seq, bos=True, eos=False)
          # 1. Create on CPU first (Standard LongTensor)
          tokens_tensor = torch.tensor(tokens, dtype=torch.long)
          # 2. Explicitly move to the T4 GPU device
          tokens_tensor = tokens_tensor.to(self.device).unsqueeze(0)

          return tokens, tokens_tensor
      except Exception:
        import traceback as tr
        tr.print_exc()
        return [], torch.empty(0)


class LlamaBF16(Llama):
    def __init__(self, max_seq_length : int, batch_size : int, device : str = "cuda", 
                 is_batch : bool = True):
        # Setup fairscale for single-process usage
        _setup_single_process_distributed(device)

        super().__init__(device, is_batch)

        # Initialize model arguments (hyperparams, context window, and batch size)
        self.model_args = ModelArgs(
            **self.params,
            max_seq_len=max_seq_length,
            max_batch_size=batch_size,
            use_compressed_kv_cache=False
        )

        # --- THE FIX: Meta-Device Initialization ---
        # 1. Build model architecture on the META device (Takes 0 GB VRAM)
        with torch.device("meta"):
            self.model = Transformer(self.model_args)

        self.model.load_state_dict(self.checkpoints, strict=False, assign=True)

        # --- THE FLAWLESS GHOST CACHE MATERIALIZER ---
        # 1. Native PyTorch sweep for the model tree
        for m in self.model.modules():
            # Catch registered PyTorch buffers (like cache_k, cache_v)
            for name, buf in m.named_buffers(recurse=False):
                if buf is not None and buf.device.type == "meta":
                    m.register_buffer(name, torch.zeros(buf.shape, dtype=buf.dtype, device=device))

            # Catch loose attributes attached to the module (like cache_original_l2)
            for name, attr in vars(m).items():
                if isinstance(attr, torch.Tensor) and attr.device.type == "meta":
                    # FIX 1: Added tuple() cast
                    setattr(m, name, torch.zeros(tuple(attr.shape), dtype=attr.dtype, device=device))
        # ---------------------------------------------


        # --- Aggressive Memory Management ---
        # 3. DESTROY the checkpoints dictionary to free up system overhead
        del self.checkpoints
        import gc
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        # 4. Stream the model to GPU safely (Layer by Layer with CPU Offloading)
        if device == "cuda":
            # Keep Embeddings on CPU (bfloat16)
            self.model.tok_embeddings = self.model.tok_embeddings.to(device="cpu", dtype=torch.bfloat16)

            # Move the 32 Transformer blocks to CUDA
            for i, layer in enumerate(self.model.layers):
                self.model.layers[i] = layer.to(device="cuda", dtype=torch.bfloat16)
                if device == "cuda":
                    torch.cuda.empty_cache()

            self.model.norm = self.model.norm.to(device=device, dtype=torch.bfloat16)

            # Keep Output on CPU to prevent CPU bfloat16 GEMM upcast crashes
            self.model.output = self.model.output.to(device="cpu", dtype=torch.bfloat16)

            # Catch the precomputed rotary embeddings
            if device == "cuda":
                self.model.freqs_cis = self.model.freqs_cis.to(device="cuda")

        self.model.eval()


class LlamaCompressed(Llama):
    def __init__(self, max_seq_length: int,
                 batch_size: int,
                 device: str = "cuda",
                 is_batch: bool = True,
                 bit_width: float = DEFAULT_BIT_WIDTH,
                 dims: int = DEFAULT_DIMENSIONS
                 ):
        # Setup fairscale for single-process usage
        _setup_single_process_distributed(device)

        super().__init__(device, is_batch)

        self.bit_width = bit_width
        self.dims = dims

        # Initialize model arguments (hyperparams, context window, and batch size)
        self.model_args = ModelArgs(
            **self.params,
            max_seq_len=max_seq_length,
            max_batch_size=batch_size,
            use_compressed_kv_cache=True,
        )

        # Use default context path if not set, looking in artifacts folder
        # context_path = CONTEXT_PATH
        # if not context_path:
        #    from app import PROJECT_ROOT
        #    context_path = str(PROJECT_ROOT / "artifacts" / f"turboquant_ctx_{dims}d_{bit_width + 1}b.bin")

        self.is_mixed_precision = isinstance(bit_width, float) and not bit_width.is_integer()

        if self.is_mixed_precision:
            # 1. Calculate Split Logic based on the TurboQuant Paper
            if bit_width == 3.5:
                # (64 channels * 4 bits) + (64 channels * 3 bits) / 128 = 3.5
                self.outlier_dims = 64
                outlier_total_bits = 4

                self.normal_dims = dims - self.outlier_dims
                normal_total_bits = 3

            elif bit_width == 2.5:
                # (32 channels * 3 bits) + (96 channels * 2 bits) / 128 = 2.5
                self.outlier_dims = 32
                outlier_total_bits = 3

                self.normal_dims = dims - self.outlier_dims
                normal_total_bits = 2

            else:
                raise ValueError(f"Mixed precision for {bit_width} bits is not defined. Use 2.5 or 3.5.")

            # 2. Subtract 1 for the MSE bit budgets
            outlier_mse_bits = outlier_total_bits - 1
            normal_mse_bits = normal_total_bits - 1

            # 3. Dynamically generate context paths so they don't overwrite each other
            outlier_ctx_path = str(PROJECT_ROOT / "artifacts" / f"turboquant_outlier_{self.outlier_dims}d_{outlier_total_bits}b.bin")
            normal_ctx_path = str(PROJECT_ROOT / "artifacts" / f"turboquant_normal_{self.normal_dims}d_{normal_total_bits}b.bin")

            # 4. Initialize both compressors
            self.outlier_compressor = get_compressor_for_variant(
                lib_path=str(self.lib_path),
                context_path=outlier_ctx_path,
                block_size=self.outlier_dims,
                bit_width=outlier_mse_bits,
                variant=self.variant,
            )

            self.normal_compressor = get_compressor_for_variant(
                lib_path=str(self.lib_path),
                context_path=normal_ctx_path,
                block_size=self.normal_dims,
                bit_width=normal_mse_bits,
                variant=self.variant,
            )
            self.kv_compressor = None # Nullify the standard compressor

        else:
            int_bit_width = int(bit_width)
            mse_bits = int_bit_width - 1

            context_path = str(PROJECT_ROOT / "artifacts" / f"turboquant_ctx_{dims}d_{int_bit_width}b.bin")

            self.kv_compressor = get_compressor_for_variant(
                lib_path=str(self.lib_path),
                context_path=context_path,
                block_size=dims,
                bit_width=mse_bits,
                variant=self.variant,
            )
            self.outlier_compressor = None
            self.normal_compressor = None


        # --- THE FIX: Meta-Device Initialization ---
        # 1. Build model architecture on the META device (Takes 0 GB VRAM)
        with torch.device("meta"):
            self.model = Transformer(
                self.model_args,
                self.kv_compressor,
                self.outlier_compressor,
                self.normal_compressor
            )

        self.model.load_state_dict(self.checkpoints, strict=False, assign=True)

        # --- THE FLAWLESS GHOST CACHE MATERIALIZER ---
        # 1. Native PyTorch sweep for the model tree
        for m in self.model.modules():
            # Catch registered PyTorch buffers (like cache_k, cache_v)
            for name, buf in m.named_buffers(recurse=False):
                if buf is not None and buf.device.type == "meta":
                    m.register_buffer(name, torch.zeros(buf.shape, dtype=buf.dtype, device=device))

            # Catch loose attributes attached to the module
            for name, attr in vars(m).items():
                if isinstance(attr, torch.Tensor) and attr.device.type == "meta":
                    setattr(m, name, torch.zeros(tuple(attr.shape), dtype=attr.dtype, device=device))

        # 2. Safe sweep for the custom KV Compressors (Catching lists/dicts)
        compressors_to_sweep = []
        if self.is_mixed_precision:
            compressors_to_sweep = [self.outlier_compressor, self.normal_compressor]
        elif self.kv_compressor is not None:
            compressors_to_sweep = [self.kv_compressor]

        for compressor in compressors_to_sweep:
            for name, attr in vars(compressor).items():
                if isinstance(attr, torch.Tensor) and attr.device.type == "meta":
                    setattr(compressor, name, torch.zeros(attr.shape, dtype=attr.dtype, device=device))
                elif isinstance(attr, list):
                    for i, v in enumerate(attr):
                        if isinstance(v, torch.Tensor) and v.device.type == "meta":
                            attr[i] = torch.zeros(v.shape, dtype=v.dtype, device=device)
                elif isinstance(attr, dict):
                    for k, v in attr.items():
                        if isinstance(v, torch.Tensor) and v.device.type == "meta":
                            attr[k] = torch.zeros(v.shape, dtype=v.dtype, device=device)

        # --- Aggressive Memory Management ---
        # 3. DESTROY the checkpoints dictionary to free up system overhead
        del self.checkpoints
        import gc
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        # 4. Stream the model to device safely (Layer by Layer with Offloading)
        if device == "cuda":
            # Keep Embeddings on CPU (bfloat16)
            self.model.tok_embeddings = self.model.tok_embeddings.to(device="cpu", dtype=torch.bfloat16)

            # Move the 32 Transformer blocks to CUDA
            for i, layer in enumerate(self.model.layers):
                self.model.layers[i] = layer.to(device="cuda", dtype=torch.bfloat16)

            if device == "cuda":
                torch.cuda.empty_cache()

            # Move final Norm to CUDA
            self.model.norm = self.model.norm.to(device=device, dtype=torch.bfloat16)

            # Keep Output on CPU to prevent bfloat16 GEMM upcast crashes
            self.model.output = self.model.output.to(device="cpu", dtype=torch.bfloat16)

            # Catch the precomputed rotary embeddings
            if device == "cuda":
                self.model.freqs_cis = self.model.freqs_cis.to(device="cuda")

        self.model.eval()


class LlamaGenerator:
    def generate(self, tensor_tokens: torch.Tensor,
                 token_ids: Optional[List[int]],
                 llama: LlamaBF16 | LlamaCompressed,
                 max_gen_len: int = 1024) -> str:

        try:
            generated_token = []

            with torch.no_grad():
                current_pos = 0
                seq_len = tensor_tokens.shape[1]

                # Warmup loop for compressed KV cache
                if seq_len > 1:

                    # Pass all prompt tokens up to the second-to-last one simultaneously.
                    prefill_prompt = tensor_tokens[:, :-1].contiguous()
                    logits = llama.model.forward(prefill_prompt, start_pos=0)

                    del logits, prefill_prompt
                    current_pos = seq_len - 1

                    if llama.device == "cuda":
                        torch.cuda.synchronize()

                current_token = tensor_tokens[:, -1:].contiguous()

                for _ in range(max_gen_len):
                    logits = llama.model.forward(current_token, current_pos)

                    next_token = torch.argmax(logits[:, -1], dim=-1)
                    next_token_id = next_token.item()

                    if next_token_id == llama.tokenizer.eos_id:
                        break

                    generated_token.append(next_token_id)

                    current_token = next_token.unsqueeze(0)
                    current_pos += logits.shape[1]

            return llama.tokenizer.decode(generated_token).strip()
        except Exception as e:
            raise RuntimeError(f"Failed to generate response. Reason: {e}")



def format_prompt(prompt: str, context: str, sysprompt: str) -> str:
    """Build a plain-text prompt for the Llama 3.1 base model.

    The checkpoint configured in `.env` is `Llama3.1-8B`, which is the base
    model. It should receive a single instruction-style prompt, not the
    chat-template role tokens used by instruct-tuned checkpoints.
    """

    sections = []

    if sysprompt.strip():
        sections.append(
            "System:\n"
            f"{sysprompt.strip()}"
        )

    sections.append(
        "Context:\n"
        f"{context.strip()}"
    )

    sections.append(
        "Question:\n"
        f"{prompt.strip()}"
    )

    sections.append("Answer:")

    return "\n\n".join(sections)
