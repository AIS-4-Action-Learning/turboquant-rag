"""Experiment runner for TurboQuant RAG evaluations."""

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from app import N_TRIALS
from app.llama_models import LlamaBF16, LlamaCompressed
from app.metrics import (
    eval_correctness,
    question_answering_accuracy,
    zero_shot_accuracy,
)
from rag_library import (
    BF16LlamaGenerator,
    Chunker,
    Embedder,
    RAG,
    TurboQuantLlamaGenerator,
    VectorStore,
)


class Experiment:
    """Run and persist a benchmark experiment for a Llama-backed RAG pipeline.

    The experiment loads a question set, runs each question through the RAG
    pipeline, evaluates answer correctness with embedding similarity, and
    writes per-trial and aggregate experiment metrics to CSV files.
    """

    INDEX_PATH = Path("data/faiss_index.index")
    CHUNKS_PATH = Path("data/chunks.json")

    TRIALS_SCHEMA = [
        "trial_number",
        "bit_width",
        "group_size",
        "model_type",
        "context_question",
        "question_type",
        "evaluation",
        "perplexity",
        "rmse_key",
        "rmse_value",
    ]
    EXPERIMENT_SCHEMA = [
        "experiment_id",
        "bit_width",
        "group_size",
        "mean_perplexity",
        "mean_rmse_key",
        "mean_rmse_value",
        "fqa_accuracy",
        "oosqa_accuracy",
        "crqa_accuracy",
        "zero_shot_accuracy",
    ]

    def __init__(
        self,
        experiment_id: int,
        llama_model: LlamaBF16 | LlamaCompressed,
        embedder: Embedder,
        max_gen_len: int = 20,
        chunk_size: int = 600,
        overlap: int = 150,
        questions_path: str = "data/questions.json",
        corpus_path: str = "data/corpus.json",
        trial_results_path: str = "results/trial_results.csv",
        experiment_results_path: str = "results/experiment_results.csv",
        chunk: bool = False,
    ) -> None:
        """Initialize the experiment and prepare the RAG pipeline.

        Args:
            experiment_id: Identifier written to aggregate experiment results.
            llama_model: BF16 or TurboQuant-compressed Llama model instance.
            embedder: Embedder used by retrieval and answer evaluation.
            max_gen_len: Maximum number of tokens generated per answer.
            chunk_size: Target chunk size, in characters, for indexing.
            overlap: Character overlap between adjacent chunks.
            questions_path: JSON file containing benchmark questions.
            corpus_path: JSON corpus file used when rebuilding the RAG index.
            trial_results_path: CSV file for per-question results.
            experiment_results_path: CSV file for aggregate experiment results.
            chunk: When True, rebuild and save the RAG index from ``corpus_path``.

        Raises:
            RuntimeError: If initialization fails.
        """
        try:
            self.experiment_id = experiment_id
            self.llama_model = llama_model
            self.embedder = embedder
            self.questions_path = Path(questions_path)
            self.corpus_path = Path(corpus_path)
            self.trial_results_path = Path(trial_results_path)
            self.experiment_results_path = Path(experiment_results_path)

            self.generator = self._build_generator(max_gen_len)
            self.rag = RAG(
                embedder=self.embedder,
                generator=self.generator,
                chunker=Chunker(
                    chunk_size=chunk_size,
                    overlap=overlap,
                    skip_noisy_pages=False,
                ),
                vector_store=VectorStore(),
                top_k=5,
            )

            Path("data").mkdir(parents=True, exist_ok=True)
            self.trial_results_path.parent.mkdir(parents=True, exist_ok=True)
            self.experiment_results_path.parent.mkdir(parents=True, exist_ok=True)

            self.questions = self._load_questions(self.questions_path)
            self._prepare_index(chunk)

            self.trials_results = self._load_results(
                self.trial_results_path,
                self.TRIALS_SCHEMA,
            )
            self.experiment_results = self._load_results(
                self.experiment_results_path,
                self.EXPERIMENT_SCHEMA,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize experiment. Reason: {exc}"
            ) from exc

    def _build_generator(
        self,
        max_gen_len: int,
    ) -> BF16LlamaGenerator | TurboQuantLlamaGenerator:
        """Create the generator that matches the configured Llama model.

        Args:
            max_gen_len: Maximum number of generated tokens.

        Returns:
            A Llama generator compatible with the model type.

        Raises:
            TypeError: If ``self.llama_model`` is not supported.
        """
        if isinstance(self.llama_model, LlamaCompressed):
            self.model_type = "Compressed"
            self.bit_width = self.llama_model.bit_width
            self.dims = self.llama_model.dims
            return TurboQuantLlamaGenerator(self.llama_model, max_gen_len)

        if isinstance(self.llama_model, LlamaBF16):
            self.model_type = "BF16"
            self.bit_width = None
            self.dims = None
            return BF16LlamaGenerator(self.llama_model, max_gen_len)

        raise TypeError(
            "llama_model must be an instance of LlamaCompressed or LlamaBF16; "
            f"got {type(self.llama_model).__name__}."
        )

    @staticmethod
    def _load_questions(path: Path) -> list[dict[str, Any]]:
        """Load benchmark questions from a JSON file.

        Args:
            path: Path to the question JSON file.

        Returns:
            A list of question dictionaries.

        Raises:
            ValueError: If the JSON root is not a list.
        """
        with path.open("r", encoding="utf-8") as questions_file:
            questions = json.load(questions_file)

        if not isinstance(questions, list):
            raise ValueError(f"Questions file must contain a list: {path}")

        return questions

    @staticmethod
    def _load_results(path: Path, schema: list[str]) -> pd.DataFrame:
        """Load a results CSV and normalize it to the expected schema.

        Args:
            path: CSV file path.
            schema: Ordered list of expected columns.

        Returns:
            A DataFrame containing exactly the expected columns.
        """
        if not path.exists():
            return pd.DataFrame(columns=schema)

        results = pd.read_csv(path)
        results.columns = results.columns.str.strip()
        results = results.loc[:, ~results.columns.str.startswith("Unnamed:")]

        for column in schema:
            if column not in results.columns:
                results[column] = np.nan

        return results[schema]

    def _prepare_index(self, rebuild: bool) -> None:
        """Build or load the persisted RAG index.

        Args:
            rebuild: When True, rebuild the index from ``self.corpus_path``.
        """
        if rebuild:
            self.rag.build_index(self.corpus_path)
            self.rag.save(self.INDEX_PATH, self.CHUNKS_PATH)
            return

        self.rag.load(self.INDEX_PATH, self.CHUNKS_PATH)

    def _get_expected_answer(self, category: str, index: int) -> str:
        """Return the expected answer for a benchmark question.

        Args:
            category: Question category from the benchmark data.
            index: Index of the question in ``self.questions``.

        Returns:
            The expected answer string used for correctness evaluation.
        """
        if category == "out-of-scope":
            return "I can't answer this question"

        if category == "ambiguous":
            return "Could you clarify?"

        return self.questions[index]["expected_answer"]

    @staticmethod
    def _set_evaluation(
        category: str,
        evaluation: float,
        index: int,
        evals: dict[str, dict[int, float]],
    ) -> None:
        """Store an evaluation score in the category-specific accumulator.

        Args:
            category: Benchmark question category.
            evaluation: Numeric correctness score for the answer.
            index: Trial index.
            evals: Mutable mapping of category accumulators.
        """
        category_key_map = {
            "factual": "fqa_evals",
            "cross-reference": "crqa_evals",
            "out-of-scope": "oosqa_evals",
        }
        eval_key = category_key_map.get(category)

        if eval_key is not None:
            evals[eval_key][index] = evaluation

    def _log_trial(
        self,
        index: int,
        question: str,
        category: str,
        evaluation: float,
        perplexity: float,
        rmse_k: float,
        rmse_v: float,
    ) -> None:
        """Record metrics for a single benchmark trial.

        Args:
            index: Trial number.
            question: Question text sent to the RAG pipeline.
            category: Benchmark question category.
            evaluation: Correctness score for the generated answer.
            perplexity: Perplexity returned by the generator.
            rmse_k: Key-cache RMSE returned by the generator.
            rmse_v: Value-cache RMSE returned by the generator.
        """
        row = {
            "trial_number": index,
            "bit_width": self.bit_width,
            "group_size": self.dims,
            "model_type": self.model_type,
            "context_question": question,
            "question_type": category,
            "evaluation": evaluation,
            "perplexity": perplexity,
            "rmse_key": rmse_k,
            "rmse_value": rmse_v,
        }
        self.trials_results.loc[index, self.TRIALS_SCHEMA] = row

    @staticmethod
    def _mean_numeric(series: pd.Series) -> float:
        """Return the mean of a numeric result series, ignoring invalid values.

        Args:
            series: Result values loaded from or prepared for CSV.

        Returns:
            The numeric mean, or ``nan`` when no numeric values exist.
        """
        return float(pd.to_numeric(series, errors="coerce").mean())

    def _log_experiment(self, evals: dict[str, dict[int, float]]) -> None:
        """Record aggregate metrics for the completed experiment.

        Args:
            evals: Per-category correctness scores collected during ``run``.
        """
        fqa_evals_tensor = torch.tensor(list(evals["fqa_evals"].values()))
        crqa_evals_tensor = torch.tensor(list(evals["crqa_evals"].values()))
        oosqa_evals_tensor = torch.tensor(list(evals["oosqa_evals"].values()))

        fqa_accuracy = question_answering_accuracy(fqa_evals_tensor)
        crqa_accuracy = question_answering_accuracy(crqa_evals_tensor)
        oosqa_accuracy = question_answering_accuracy(oosqa_evals_tensor)
        zs_accuracy = zero_shot_accuracy(
            fqa_accuracy,
            oosqa_accuracy,
            crqa_accuracy,
        )

        row = {
            "experiment_id": self.experiment_id,
            "bit_width": self.bit_width,
            "group_size": self.dims,
            "mean_perplexity": self._mean_numeric(
                self.trials_results["perplexity"]
            ),
            "mean_rmse_key": self._mean_numeric(self.trials_results["rmse_key"]),
            "mean_rmse_value": self._mean_numeric(
                self.trials_results["rmse_value"]
            ),
            "fqa_accuracy": fqa_accuracy,
            "oosqa_accuracy": oosqa_accuracy,
            "crqa_accuracy": crqa_accuracy,
            "zero_shot_accuracy": zs_accuracy,
        }
        self.experiment_results.loc[
            len(self.experiment_results),
            self.EXPERIMENT_SCHEMA,
        ] = row

    def run(self, top_k: int = 5) -> None:
        """Run all configured benchmark trials and persist the results.

        Args:
            top_k: Number of retrieved chunks to pass to the generator.

        Raises:
            RuntimeError: If a trial or result persistence fails.
        """
        try:
            print("=" * 15)
            print("RUNNING TURBOQUANT BENCHMARKING EXPERIMENT")
            print(f"CONFIGURATION: {self.bit_width}")
            print("=" * 15)

            self.trials_results = pd.DataFrame(columns=self.TRIALS_SCHEMA)
            evals: dict[str, dict[int, float]] = {
                "fqa_evals": {},
                "crqa_evals": {},
                "oosqa_evals": {},
            }
            trial_count = min(N_TRIALS, len(self.questions))
            trial_durations: list[float] = []

            for index in range(trial_count):
                trial_start_time = time.perf_counter()
                question_data = self.questions[index]
                question = question_data["question"]
                category = question_data["category"]

                print(f"----- TRIAL {index + 1} -----")
                print(f"Question Category: {category}")
                print("Running...")
                expected_answer = self._get_expected_answer(category, index)

                response = self.rag.query(question, top_k)

                print(
                    f"Query: {response['query']}\n"
                    f"Response: {response['answer']}"
                )

                evaluation = eval_correctness(
                    response["answer"],
                    expected_answer,
                    self.embedder,
                )
                self._set_evaluation(category, evaluation, index, evals)
                self._log_trial(
                    index=index,
                    question=question,
                    category=category,
                    evaluation=evaluation,
                    perplexity=response["perplexity"],
                    rmse_k=response["rmse_k"],
                    rmse_v=response["rmse_v"],
                )
                trial_duration = time.perf_counter() - trial_start_time
                trial_durations.append(trial_duration)
                average_trial_duration = sum(trial_durations) / len(
                    trial_durations
                )
                remaining_trials = trial_count - len(trial_durations)
                estimated_time_left = average_trial_duration * remaining_trials

                print(f"Evaluation: {evaluation}")
                print(
                    "Timing: "
                    f"trial={trial_duration / 60:.2f} min, "
                    f"avg={average_trial_duration / 60:.2f} min, "
                    f"time_left={estimated_time_left / 60:.2f} min"
                )
                print("-" * 15)

            self._log_experiment(evals)
            self.trials_results.to_csv(self.trial_results_path, index=False)
            self.experiment_results.to_csv(
                self.experiment_results_path,
                index=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to run experiment. Reason: {exc}"
            ) from exc
