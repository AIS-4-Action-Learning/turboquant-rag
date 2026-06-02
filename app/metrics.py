from typing import List
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from rag_library.embedder import Embedder

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
    fqa_accuracy: float,
    oosqa_accuracy: float,
    crqa_accuracy: float
) -> float:
    return (fqa_accuracy + oosqa_accuracy + crqa_accuracy) / 3


def eval_correctness(
    predicted_answer: str,
    ground_truth_answer: str,
    embedding_model: "Embedder",
    threshold: float = 0.82
) -> float:
    if not predicted_answer.strip():
        return 0.0

    try:
        embd = embedding_model.embed([predicted_answer, ground_truth_answer])

        pred_tensor = torch.tensor(embd[0], dtype=torch.float32).flatten()
        gt_tensor = torch.tensor(embd[1], dtype=torch.float32).flatten()

        similarity = F.cosine_similarity(pred_tensor.unsqueeze(0), gt_tensor.unsqueeze(0)).item()

        return 1.0 if similarity >= threshold else 0.0
    except Exception as e:
        raise RuntimeError(f"Failed to obtain evaluation result. Reason: {e}")


def question_answering_accuracy(
    results: torch.Tensor
) -> float:
    if results.numel() == 0:
        return -1.0

    try:
        flat_results = results.flatten()
        n_total = flat_results.shape[0]
        n_accurate = results.count_nonzero().item()
        return float(n_accurate / n_total)
    except Exception as e:
        raise RuntimeError(f"Failed to obtain question answering accuracy. Reason: {e}")
