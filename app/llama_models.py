import torch
import json
import os
from pathlib import Path
import traceback as tr
from app import (
    CHECKPOINT_PATH, CONTEXT_PATH, DEFAULT_BIT_WIDTH,
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
            print(e)
            tr.print_exc()

    def input_encoding(self, input_seq : str):
      try:
          tokens = self.tokenizer.encode(input_seq, bos=True, eos=False)
          # 1. Create on CPU first (Standard LongTensor)
          tokens_tensor = torch.tensor(tokens, dtype=torch.long)
          # 2. Explicitly move to the T4 GPU device
          tokens_tensor = tokens_tensor.to(self.device).unsqueeze(0)

          return tokens, tokens_tensor
      except Exception as e:
        print(f"CUDA Error in encoding: {e}")
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

        # BF16 model standard
        with torch.device(device):
            self.model = Transformer(self.model_args).bfloat16()

        self.model.load_state_dict(self.checkpoints, strict=False, assign=True)

        self.model.eval()


class LlamaCompressed(Llama):
    def __init__(self, max_seq_length : int,
                 batch_size : int,
                 device : str = "cuda",
                 is_batch : bool = True,
                 bit_width : int = DEFAULT_BIT_WIDTH,
                 dims : int = DEFAULT_DIMENSIONS):
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
            use_compressed_kv_cache=True
        )

        # Use default context path if not set, looking in artifacts folder
        context_path = CONTEXT_PATH
        if not context_path:
            from app import PROJECT_ROOT
            context_path = str(PROJECT_ROOT / "artifacts" / f"turboquant_ctx_{dims}d_{bit_width}b.bin")

        # Use factory function to get appropriate compressor for the variant
        self.kv_compressor = get_compressor_for_variant(
            lib_path=str(self.lib_path),
            context_path=context_path,
            block_size=dims,
            bit_width=bit_width,
            variant=self.variant,
        )

        with torch.device(device):
            self.model = Transformer(self.model_args, self.kv_compressor)

        if device == "cuda":
            self.model = self.model.bfloat16()

        self.model.load_state_dict(self.checkpoints, strict=False, assign=True)

        self.model.eval()


class LlamaGenerator:
    def generate(self, tensor_tokens : torch.Tensor,
                 llama : LlamaBF16 | LlamaCompressed,
                 max_gen_len : int = 1024):
        try:
            generated_token = []
            with torch.no_grad():
                current_pos = 0
                for _ in range(max_gen_len):
                    logits = llama.model.forward(tensor_tokens, current_pos)
                    next_token = torch.argmax(logits[:, -1], dim=-1)

                    if next_token.item() == llama.tokenizer.eos_id:
                        break

                    generated_token.append(next_token.item())

                    tensor_tokens = next_token.unsqueeze(0)
                    current_pos += logits.shape[1]

            return generated_token
        except Exception as e:
            print(e)
            tr.print_exc()
