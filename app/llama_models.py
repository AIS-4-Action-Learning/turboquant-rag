import torch
import json
import os
from app.turboquant import TurboQuantBatchCompressor
from app import (CHECKPOINT_PATH, CONTEXT_PATH, DEFAULT_BIT_WIDTH,
                 LIB_PATH, DEFAULT_DIMENSIONS)
from llama_models.llama3.args import ModelArgs
from llama_models.llama3.model import Transformer
from llama_models.llama3.tokenizer import Tokenizer


def _setup_single_process_distributed():
    """Initialize distributed environment for single-process inference."""
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="gloo",
            init_method="file:///tmp/turboquant_dist_init",
            world_size=1,
            rank=0
        )
    import fairscale.nn.model_parallel.initialize as fs_init
    if not fs_init.model_parallel_is_initialized():
        fs_init.initialize_model_parallel(1)


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


class Llama:
    def __init__(self, device : str = "cpu"):
        try:
            # Read the hyperparameters from params.json
            with open(PARAMS_PATH, "r") as f:
                self.params = json.load(f)

            # Initialize the tokenizer from tokenizer.model
            self.tokenizer = Tokenizer(TOKENIZER_PATH)

            # Set the device attribute to the input device argument
            self.device = device

            # Load the weights to device memory (CPU RAM by default)
            # Consider mmap on disk 
            self.checkpoints = torch.load(
                str(CHECKPOINT_PATH),
                map_location=device,
                weights_only=True
            )

        except Exception as e:
            print(e)

    def input_encoding(self, input_seq : str):
        try:
            tokens = self.tokenizer.encode(input_seq, bos=True, eos=False)
            tokens_tensor = torch.tensor(tokens, dtype=torch.long, device=self.device)

            return tokens, tokens_tensor
        except Exception as e:
            print(e)


class LlamaBF16(Llama):
    def __init__(self, max_seq_length : int, batch_size : int, device : str = "cpu"):
        try:
            # Setup fairscale for single-process usage
            _setup_single_process_distributed()
            
            super().__init__(device)

            # Initialize model arguments (hyperparams, context window, and batch size)
            self.model_args = ModelArgs(
                **self.params,
                max_seq_len=max_seq_length,
                max_batch_size=batch_size,
                use_compressed_kv_cache=False
            )

            # BF16 model standard
            self.model = Transformer(self.model_args).to(device).bfloat16()

            self.model.load_state_dict(self.checkpoints)

            self.model.eval()
        except Exception as e:
            print(e)


class LlamaCompressed(Llama):
    def __init__(self, max_seq_length : int,
                 batch_size : int,
                 device : str = "cpu",
                 bit_width : int = DEFAULT_BIT_WIDTH,
                 dims : int = DEFAULT_DIMENSIONS):
        # Setup fairscale for single-process usage
        _setup_single_process_distributed()
        
        super().__init__(device)

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

        self.kv_compressor = TurboQuantBatchCompressor(LIB_PATH, context_path, dims, bit_width)
        self.model = Transformer(self.model_args, self.kv_compressor)
        self.model.load_state_dict(self.checkpoints)

        self.model.eval()
