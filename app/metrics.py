import torch
import torch.nn.functional as F
from typing import List


def perplexity(
    logits: torch.Tensor,   # (1, seq_len, vocab_size) or (seq_len, vocab_size)
    token_ids: List[int]
) -> float:
    if logits.dim() == 3:
        logits = logits.squeeze(0)              # → (seq_len, vocab_size)

    seqlen = logits.shape[0]
    shift_logits  = logits[:-1]                 # (seq_len - 1, vocab_size)
    shift_targets = torch.tensor(
        token_ids[1:seqlen], dtype=torch.long, device=logits.device
    )                                           # (seq_len - 1,)

    log_probs_full = F.log_softmax(shift_logits, dim=-1)
    log_probs = torch.gather(
        log_probs_full,
        dim=1,
        index=shift_targets.unsqueeze(-1)       # (seq_len - 1, 1)
    ).squeeze(-1)                               # (seq_len - 1,)

    avg_neg_log_prob = -torch.mean(log_probs)
    return torch.exp(avg_neg_log_prob).detach().cpu().item()


def mse(raw_token: torch.Tensor, dequant_token: torch.Tensor) -> float:
    if raw_token.shape != dequant_token.shape:
        raise RuntimeError("The shapes of the raw token and dequantized tokens are different.")

    diff = raw_token.float() - dequant_token.float()
    squared_diff = torch.square(diff)

    return torch.mean(squared_diff).item()


def rmse(mses: torch.Tensor) -> float:
    return torch.sqrt(torch.mean(mses)).item()


def zero_shot_accuracy(
    logits: torch.Tensor,
    label: List[str]) -> float:
    return 0.0
