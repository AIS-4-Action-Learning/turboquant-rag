"""Shared telemetry and result helpers for profiling RAG experiments."""

from __future__ import annotations

import json
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch

PathLike = Union[str, Path]

MB = 1024 ** 2


def cuda_available() -> bool:
    """Return True when CUDA telemetry is available."""
    return torch.cuda.is_available()


def synchronize_cuda() -> None:
    """Synchronize CUDA when available."""
    if cuda_available():
        torch.cuda.synchronize()


def reset_vram_tracker() -> None:
    """Reset PyTorch peak memory counters for a clean measurement window."""
    if not cuda_available():
        return

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()


def get_vram_snapshot() -> Dict[str, float]:
    """Capture current and peak VRAM usage from the PyTorch allocator."""
    if not cuda_available():
        return {"allocated_mb": 0.0, "peak_mb": 0.0, "reserved_mb": 0.0}

    return {
        "allocated_mb": torch.cuda.memory_allocated() / MB,
        "peak_mb": torch.cuda.max_memory_allocated() / MB,
        "reserved_mb": torch.cuda.memory_reserved() / MB,
    }


class NvidiaSmiPoller:
    """Poll nvidia-smi on a background thread at a fixed interval.

    This captures system-level GPU memory usage, which can differ from the
    PyTorch allocator's view when CUDA context overhead is present.
    """

    def __init__(self, interval_seconds: float = 1.0):
        self.interval = interval_seconds
        self.readings: List[dict] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._nvidia_smi = shutil.which("nvidia-smi")

    def _poll(self) -> None:
        if self._nvidia_smi is None:
            return

        while not self._stop.is_set():
            try:
                out = subprocess.run(
                    [
                        self._nvidia_smi,
                        "--query-gpu=timestamp,memory.used,memory.free,utilization.gpu,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                    check=False,
                )
                if out.returncode == 0:
                    parts = [part.strip() for part in out.stdout.strip().split(",")]
                    if len(parts) == 5:
                        self.readings.append(
                            {
                                "timestamp": parts[0],
                                "memory_used_mb": float(parts[1]),
                                "memory_free_mb": float(parts[2]),
                                "gpu_util_pct": float(parts[3]),
                                "temp_c": float(parts[4]),
                            }
                        )
            except Exception:
                pass

            time.sleep(self.interval)

    def start(self) -> None:
        self.readings.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self) -> List[dict]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3)
        return self.readings

    def peak_memory_mb(self) -> float:
        if not self.readings:
            return 0.0
        return max(reading["memory_used_mb"] for reading in self.readings)

    def mean_util_pct(self) -> float:
        if not self.readings:
            return 0.0
        return sum(reading["gpu_util_pct"] for reading in self.readings) / len(self.readings)


def make_profiler():
    """Create a PyTorch profiler configured for CUDA bottleneck analysis."""
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        with_flops=True,
    )


def _normalize_text(text: str) -> str:
    import re
    import string

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text


def exact_match_score(prediction: str, reference: str) -> float:
    """Return 1.0 when normalized strings match exactly, else 0.0."""
    return float(_normalize_text(prediction) == _normalize_text(reference))


def f1_score(prediction: str, reference: str) -> float:
    """Compute token-level F1 on normalized text."""
    pred_tokens = _normalize_text(prediction).split()
    ref_tokens = _normalize_text(reference).split()

    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0

    common = {}
    for token in pred_tokens:
        common[token] = common.get(token, 0) + 1

    overlap = 0
    for token in ref_tokens:
        if common.get(token, 0) > 0:
            overlap += 1
            common[token] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


@dataclass
class MeasurementResult:
    """Measurement row for a single benchmark question or inference run."""

    config_name: str
    question_id: str = ""
    question: str = ""
    answer: str = ""
    reference_answer: str = ""
    status: str = "ok"
    error: str = ""
    ttft_ms: float = 0.0
    decode_ms: float = 0.0
    tpot_ms: float = 0.0
    total_ms: float = 0.0
    peak_vram_torch_mb: float = 0.0
    peak_vram_smi_mb: float = 0.0
    mean_gpu_util_pct: float = 0.0
    context_length: int = 0
    batch_size: int = 1
    model_type: str = "unknown"
    generated_tokens: int = 0
    retrieved_count: int = 0
    retrieved: List[Dict] = field(default_factory=list)
    exact_match: Optional[float] = None
    f1: Optional[float] = None
    notes: str = ""

    MLPERF_TTFT_MS = 2000.0
    MLPERF_TPOT_MS = 100.0

    def mlperf_ttft_pass(self) -> bool:
        return self.ttft_ms <= self.MLPERF_TTFT_MS

    def mlperf_tpot_pass(self) -> bool:
        return self.tpot_ms <= self.MLPERF_TPOT_MS

    def to_dict(self) -> dict:
        return {
            "timestamp": datetime.now().isoformat(),
            "config_name": self.config_name,
            "question_id": self.question_id,
            "question": self.question,
            "answer": self.answer,
            "reference_answer": self.reference_answer,
            "status": self.status,
            "error": self.error,
            "ttft_ms": self.ttft_ms,
            "decode_ms": self.decode_ms,
            "tpot_ms": self.tpot_ms,
            "total_ms": self.total_ms,
            "peak_vram_torch_mb": self.peak_vram_torch_mb,
            "peak_vram_smi_mb": self.peak_vram_smi_mb,
            "mean_gpu_util_pct": self.mean_gpu_util_pct,
            "context_length": self.context_length,
            "batch_size": self.batch_size,
            "model_type": self.model_type,
            "generated_tokens": self.generated_tokens,
            "retrieved_count": self.retrieved_count,
            "retrieved": self.retrieved,
            "exact_match": self.exact_match,
            "f1": self.f1,
            "mlperf_ttft_pass": self.mlperf_ttft_pass(),
            "mlperf_tpot_pass": self.mlperf_tpot_pass(),
            "notes": self.notes,
        }

    def summary(self) -> str:
        ttft_icon = "✅" if self.mlperf_ttft_pass() else "❌"
        tpot_icon = "✅" if self.mlperf_tpot_pass() else "❌"
        lines = [
            f"[{self.config_name}]",
            f"  Question ID:      {self.question_id or '-'}",
            f"  Status:           {self.status}",
            f"  TTFT:             {self.ttft_ms:.1f} ms  {ttft_icon}",
            f"  TPOT:             {self.tpot_ms:.1f} ms  {tpot_icon}",
            f"  Total time:       {self.total_ms:.1f} ms",
            f"  VRAM (torch):     {self.peak_vram_torch_mb:.1f} MB",
            f"  VRAM (smi):       {self.peak_vram_smi_mb:.1f} MB",
            f"  GPU utilization:  {self.mean_gpu_util_pct:.1f}%",
            f"  Retrieved chunks: {self.retrieved_count}",
        ]
        if self.exact_match is not None:
            lines.append(f"  Exact match:      {self.exact_match:.3f}")
        if self.f1 is not None:
            lines.append(f"  F1:               {self.f1:.3f}")
        if self.notes:
            lines.append(f"  Notes:            {self.notes}")
        return "\n".join(lines)


def save_result(result: MeasurementResult, output_file: PathLike) -> Path:
    """Append a measurement result to a JSONL file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(result.to_dict(), ensure_ascii=True) + "\n")
    return output_path


def load_results(output_file: PathLike) -> List[dict]:
    """Load all measurement rows from a JSONL file."""
    output_path = Path(output_file)
    if not output_path.exists():
        return []

    rows = []
    with open(output_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
