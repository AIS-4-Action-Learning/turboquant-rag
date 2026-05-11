import torch
import json
from app import PARAMS_PATH, CHECKPOINT_PATH, TOKENIZER_PATH
from llama_models.llama3.args import ModelArgs
from llama_models.llama3.model import Transformer
from llama_models.llama3.tokenizer import Tokenizer


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
    def __init__(self, max_seq_length : int, batch_size : int, device : str = "cpu"):
        try:
            # Read the hyperparameters from params.json
            with open(PARAMS_PATH, "r") as f:
                params = json.load(f)

            # Initialize model arguments (hyperparams, context window, and batch size)
            self.model_args = ModelArgs(
                **params,
                max_seq_len=max_seq_length,
                max_batch_size=batch_size
            )

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
            tokens_tensor = torch.Tensor(tokens).to(self.device)

            return tokens, tokens_tensor
        except Exception as e:
            print(e)


class LlamaBF16(Llama):
    def __init__(self, max_seq_length : int, batch_size : int, device : str = "cpu"):
        try:
            super().__init__(max_seq_length, batch_size, device)

            # BF16 model standard
            self.model = Transformer(self.model_args).to(device).bfloat16()

            self.model.load_state_dict(self.checkpoints)

            self.model.eval()
        except Exception as e:
            print (e)


class LlamaCompressed(Llama):
    def __init__(self, max_seq_length : int, 
                 batch_size : int,
                 device : str = "cpu",
                 bit_width : int = 3):
        super().__init__(max_seq_length, batch_size, device)

        self.model = Transformer(self.model_args)

        self.model.load_state_dict(self.checkpoints)

        self.model.eval()
