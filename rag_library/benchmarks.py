"""Question-level benchmark runner for profiling RAG inference."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union

import torch

from .telemetry import (
    MeasurementResult,
    NvidiaSmiPoller,
    exact_match_score,
    f1_score,
    get_vram_snapshot,
    load_results,
    make_profiler,
    reset_vram_tracker,
    save_result,
    synchronize_cuda,
)

if TYPE_CHECKING:
    from .rag import RAG

PathLike = Union[str, Path]

NO_RELEVANT_DOCS_MESSAGE = (
    "[Database: No relevant documents found. You MUST reply with exactly: "
    "I can't answer this question.]"
)


@dataclass
class QuestionItem:
    """Normalized question entry loaded from questions.json."""

    question_id: str
    question: str
    reference_answer: str = ""
    metadata: Dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: Dict, fallback_index: int) -> "QuestionItem":
        question = payload.get("question") or payload.get("query") or payload.get("prompt")
        if not question:
            raise ValueError("Each question entry must include a 'question' field.")

        question_id = str(
            payload.get("id")
            or payload.get("question_id")
            or payload.get("qid")
            or fallback_index
        )

        reference_answer = (
            payload.get("reference_answer")
            or payload.get("answer")
            or payload.get("gold_answer")
            or ""
        )

        metadata = {
            key: value
            for key, value in payload.items()
            if key not in {"id", "question_id", "qid", "question", "query", "prompt", "reference_answer", "answer", "gold_answer"}
        }

        return cls(
            question_id=question_id,
            question=str(question),
            reference_answer=str(reference_answer) if reference_answer is not None else "",
            metadata=metadata,
        )


def _maybe_shuffle(items: List[QuestionItem], shuffle: bool, seed: Optional[int]) -> List[QuestionItem]:
    if not shuffle:
        return items

    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    return shuffled


def load_questions(
    questions_path: PathLike,
    limit: Optional[int] = None,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> List[QuestionItem]:
    """Load questions from a JSON file.

    Accepted formats:
    - a JSON list of question objects
    - a JSON object with a top-level `questions` key
    """
    path = Path(questions_path)
    if not path.exists():
        raise FileNotFoundError(f"Questions file not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        if "questions" in payload:
            payload = payload["questions"]
        elif "data" in payload:
            payload = payload["data"]
        else:
            raise ValueError(
                "questions.json must be either a list or an object with a "
                "'questions' or 'data' key."
            )

    if not isinstance(payload, list):
        raise ValueError("questions.json must contain a list of question objects.")

    items = [QuestionItem.from_dict(item, i + 1) for i, item in enumerate(payload)]
    items = _maybe_shuffle(items, shuffle=shuffle, seed=seed)

    if limit is not None:
        items = items[: max(limit, 0)]

    return items


def _is_llama_generator(generator) -> bool:
    return hasattr(generator, "llama") and hasattr(generator, "_format_prompt")


def _build_context(rag: RAG, question: str, k: Optional[int] = None) -> List[Dict]:
    retrieved = rag.retrieve(question, k=k)
    if hasattr(rag, "rerank_filter"):
        retrieved = rag.rerank_filter(question, retrieved)
    return retrieved


def _format_context(rag: RAG, retrieved: List[Dict]) -> str:
    if not retrieved:
        return NO_RELEVANT_DOCS_MESSAGE
    return rag._format_context(retrieved)


def _safe_cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_llama_question(
    rag: RAG,
    question: QuestionItem,
    config_name: str,
    max_new_tokens: Optional[int] = None,
    omit_sysprompt: bool = False,
    use_profiler: bool = False,
) -> MeasurementResult:
    """Run a single question through a Llama-backed RAG pipeline."""
    generator = rag.generator
    llama = generator.llama
    max_tokens = max_new_tokens or getattr(generator, "max_tokens", 128)

    retrieved = _build_context(rag, question.question, k=rag.top_k)
    context = _format_context(rag, retrieved)

    formatted_prompt = generator._format_prompt(question.question, context, omit_sysprompt)
    token_ids, prompt_tensors = llama.input_encoding(formatted_prompt)
    prompt_tensors = prompt_tensors.to(llama.device)

    reset_vram_tracker()
    poller = NvidiaSmiPoller()
    profiler = make_profiler() if use_profiler else None
    if profiler is not None:
        profiler.__enter__()

    poller.start()
    _safe_cuda_sync()
    start = time.perf_counter()

    try:
        generated_tokens: List[int] = []
        with torch.no_grad():
            seq_len = int(prompt_tensors.shape[1]) if prompt_tensors.ndim >= 2 else 0
            if seq_len > 1:
                prefill_prompt = prompt_tensors[:, :-1].contiguous()
                _ = llama.model.forward(prefill_prompt, start_pos=0)
                current_pos = seq_len - 1
            else:
                current_pos = 0

            _safe_cuda_sync()
            after_prefill = time.perf_counter()

            current_token = prompt_tensors[:, -1:].contiguous()
            for _ in range(max_tokens):
                logits = llama.model.forward(current_token, current_pos)
                next_token = torch.argmax(logits[:, -1], dim=-1)
                next_token_id = int(next_token.item())
                if next_token_id == getattr(llama.tokenizer, "eos_id", None):
                    break
                generated_tokens.append(next_token_id)
                current_token = next_token.unsqueeze(0)
                current_pos += 1

            _safe_cuda_sync()
            end = time.perf_counter()

        answer = llama.tokenizer.decode(generated_tokens).strip()
        if not answer:
            answer = "I don't have enough information to answer this."

        vram = get_vram_snapshot()
        smi_peak = poller.peak_memory_mb()
        mean_util = poller.mean_util_pct()

        ttft_ms = (after_prefill - start) * 1000
        decode_ms = (end - after_prefill) * 1000
        total_ms = (end - start) * 1000
        tpot_ms = decode_ms / max(len(generated_tokens), 1)

        exact = None
        f1 = None
        if question.reference_answer:
            exact = exact_match_score(answer, question.reference_answer)
            f1 = f1_score(answer, question.reference_answer)

        model_type = type(llama).__name__
        result = MeasurementResult(
            config_name=config_name,
            question_id=question.question_id,
            question=question.question,
            answer=answer,
            reference_answer=question.reference_answer,
            status="ok",
            ttft_ms=ttft_ms,
            decode_ms=decode_ms,
            tpot_ms=tpot_ms,
            total_ms=total_ms,
            peak_vram_torch_mb=vram["peak_mb"],
            peak_vram_smi_mb=smi_peak,
            mean_gpu_util_pct=mean_util,
            context_length=int(prompt_tensors.shape[1]),
            batch_size=getattr(llama, "model_args", {}).max_batch_size if hasattr(llama, "model_args") else 1,
            model_type=model_type,
            generated_tokens=len(generated_tokens),
            retrieved_count=len(retrieved),
            retrieved=[
                {
                    "source": chunk.get("source"),
                    "page": chunk.get("page"),
                    "score": chunk.get("score"),
                    "chunk_index": chunk.get("chunk_index"),
                }
                for chunk in retrieved
            ],
            exact_match=exact,
            f1=f1,
        )

        if use_profiler and profiler is not None:
            profiler.step()

        return result
    except Exception as exc:
        return MeasurementResult(
            config_name=config_name,
            question_id=question.question_id,
            question=question.question,
            reference_answer=question.reference_answer,
            status="error",
            error=str(exc),
            retrieved_count=len(retrieved),
            retrieved=[
                {
                    "source": chunk.get("source"),
                    "page": chunk.get("page"),
                    "score": chunk.get("score"),
                    "chunk_index": chunk.get("chunk_index"),
                }
                for chunk in retrieved
            ],
        )
    finally:
        poller.stop()
        if profiler is not None:
            profiler.__exit__(None, None, None)


def run_generic_question(
    rag: RAG,
    question: QuestionItem,
    config_name: str,
) -> MeasurementResult:
    """Fallback runner for non-Llama generators.

    TTFT/TPOT are not measured here because the generator does not expose the
    model internals needed to isolate prefill vs decode timing.
    """
    start = time.perf_counter()
    query_result = rag.query(question.question)
    total_ms = (time.perf_counter() - start) * 1000
    retrieved = query_result["retrieved"]
    answer = query_result["answer"]

    exact = None
    f1 = None
    if question.reference_answer:
        exact = exact_match_score(answer, question.reference_answer)
        f1 = f1_score(answer, question.reference_answer)

    return MeasurementResult(
        config_name=config_name,
        question_id=question.question_id,
        question=question.question,
        answer=answer,
        reference_answer=question.reference_answer,
        status="ok",
        ttft_ms=math.nan,
        decode_ms=math.nan,
        tpot_ms=math.nan,
        total_ms=total_ms,
        retrieved_count=len(retrieved),
        retrieved=[
            {
                "source": chunk.get("source"),
                "page": chunk.get("page"),
                "score": chunk.get("score"),
                "chunk_index": chunk.get("chunk_index"),
            }
            for chunk in retrieved
        ],
        exact_match=exact,
        f1=f1,
        notes="Non-Llama generator fallback; TTFT/TPOT unavailable.",
    )


def summarize_measurements(rows: Sequence[Dict]) -> Dict[str, float]:
    """Compute aggregate statistics from a benchmark result list."""
    if not rows:
        return {}

    def _values(key: str) -> List[float]:
        out = []
        for row in rows:
            value = row.get(key)
            if isinstance(value, (int, float)) and not math.isnan(value):
                out.append(float(value))
        return out

    summary = {
        "count": float(len(rows)),
        "ok_count": float(sum(1 for row in rows if row.get("status") == "ok")),
        "error_count": float(sum(1 for row in rows if row.get("status") != "ok")),
    }

    for key in [
        "ttft_ms",
        "tpot_ms",
        "total_ms",
        "peak_vram_torch_mb",
        "peak_vram_smi_mb",
        "mean_gpu_util_pct",
        "exact_match",
        "f1",
    ]:
        values = _values(key)
        if values:
            summary[f"mean_{key}"] = mean(values)

    return summary


class QuestionBenchmarkRunner:
    """Run a RAG pipeline over a user-defined question set."""

    def __init__(
        self,
        rag: RAG,
        questions_path: PathLike = "questions.json",
        limit: Optional[int] = None,
        shuffle: bool = False,
        seed: Optional[int] = None,
        output_dir: PathLike = "results/question_benchmark",
        run_name: Optional[str] = None,
        config_name: str = "question_benchmark",
        max_new_tokens: Optional[int] = None,
        omit_sysprompt: bool = False,
        use_profiler: bool = False,
    ):
        required_attrs = ("query", "retrieve", "generator", "top_k")
        missing = [name for name in required_attrs if not hasattr(rag, name)]
        if missing:
            raise TypeError(
                "rag must expose query(), retrieve(), generator, and top_k "
                f"attributes; missing: {', '.join(missing)}"
            )

        self.rag = rag
        self.questions_path = Path(questions_path)
        self.limit = limit
        self.shuffle = shuffle
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config_name = config_name
        self.max_new_tokens = max_new_tokens
        self.omit_sysprompt = omit_sysprompt
        self.use_profiler = use_profiler

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_file = self.output_dir / f"{self.run_name}.jsonl"

    def load_questions(self) -> List[QuestionItem]:
        return load_questions(
            self.questions_path,
            limit=self.limit,
            shuffle=self.shuffle,
            seed=self.seed,
        )

    def run_one(self, question: QuestionItem) -> MeasurementResult:
        if _is_llama_generator(self.rag.generator):
            return run_llama_question(
                rag=self.rag,
                question=question,
                config_name=self.config_name,
                max_new_tokens=self.max_new_tokens,
                omit_sysprompt=self.omit_sysprompt,
                use_profiler=self.use_profiler,
            )
        return run_generic_question(
            rag=self.rag,
            question=question,
            config_name=self.config_name,
        )

    def run(self) -> List[dict]:
        questions = self.load_questions()
        rows = []

        for index, question in enumerate(questions, start=1):
            print(f"[{index}/{len(questions)}] {question.question_id}: {question.question}")
            result = self.run_one(question)
            save_result(result, self.output_file)
            rows.append(result.to_dict())
            print(result.summary())
            print()

        return rows

    def load_existing_results(self) -> List[dict]:
        return load_results(self.output_file)
