"""
Generator interface and implementations.

A Generator takes a user query and retrieved context, and produces an answer.
This is the component that gets swapped between OpenAI, BF16 Llama, and
TurboQuant-compressed Llama for our research benchmarks.

Class hierarchy:
    Generator (ABC)
    ├── OpenAIGenerator
    ├── BF16LlamaGenerator      ──┐
    └── TurboQuantLlamaGenerator ──┴── both inherit shared logic from _LlamaGeneratorBase
"""

from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Public abstract base class
# ---------------------------------------------------------------------------

class Generator(ABC):
    """Abstract base class for answer generators.

    Any concrete generator must inherit from this class and implement
    the `generate` method.
    """

    @abstractmethod
    def generate(self, query: str, context: str) -> str:
        """Generate an answer given a query and retrieved context.

        Args:
            query: The user's question.
            context: Concatenated retrieved chunks, formatted for the model.

        Returns:
            The generated answer as a string.
        """
        pass


# ---------------------------------------------------------------------------
# OpenAI implementation
# ---------------------------------------------------------------------------

class OpenAIGenerator(Generator):
    """Generator that uses OpenAI's chat completion API.

    Used for prototyping and as a sanity-check baseline.
    Llama-based generators below will be used for the research benchmarks.
    """

    def __init__(
        self,
        client,
        model: str = "gpt-4o-mini",
        temperature: float = 0,
        max_tokens: int = 500,
        cost_tracker=None,
    ):
        """
        Args:
            client: An initialized OpenAI client (passed in, not created here).
            model: The OpenAI model name to use.
            temperature: Sampling temperature (0 for deterministic).
            max_tokens: Maximum tokens in the response.
            cost_tracker: Optional callable that takes the response object to
                          track API costs. Pass None to disable.
        """
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cost_tracker = cost_tracker

        self.system_prompt = """You are a helpful assistant.

        Answer the question based STRICTLY on the provided context.

        Rules:
        - If the context doesn't contain the answer, say 'I don't have enough information to answer this.'
        - Always cite which source your answer comes from using the format: (Source: <filename>, Page <number>).
        - Use the context to answer, applying reasonable inference.
        - If the answer comes from multiple chunks/sources, combine and cite all of them.
        - If the question is yes/no, answer with only 'Yes' or 'No'
        - If the question asks what NOT to do, infer the answer from what the context has
        """

    def generate(self, query: str, context: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        if self.cost_tracker is not None:
            self.cost_tracker(response)

        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Shared base for Llama-based generators
# ---------------------------------------------------------------------------

class _LlamaGeneratorBase:
    """Shared logic for Llama-based generators (BF16 and TurboQuant).

    Holds prompt formatting and any other code that doesn't depend on the
    inference path. Concrete subclasses below differ in how they actually
    run the model.

    NOTE: This is a private helper (leading underscore). Users should not
    instantiate it directly — they use BF16LlamaGenerator or
    TurboQuantLlamaGenerator.
    """

    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant.

    Answer the question based STRICTLY on the provided context.

    Rules:
    - If the context doesn't contain the answer, say 'I don't have enough information to answer this.'
    - Always cite which source your answer comes from using the format: (Source: <filename>, Page <number>).
    - Use the context to answer, applying reasonable inference.
    - If the answer comes from multiple chunks/sources, combine and cite all of them.
    - If the question is yes/no, answer with only 'Yes' or 'No'
    - If the question asks what NOT to do, infer the answer from what the context has
    """

    def _format_prompt(self, query: str, context: str) -> str:
        """Format query + context into a single prompt string for Llama.

        TODO (when implementing): wrap with the actual Llama 3.1 chat template
        once we settle on the inference framework (HuggingFace transformers,
        llama.cpp, vLLM, etc.).
        """
        return (
            f"{self.system_prompt}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )


# ---------------------------------------------------------------------------
# Llama BF16 (baseline) — STUB, to be implemented
# ---------------------------------------------------------------------------

class BF16LlamaGenerator(_LlamaGeneratorBase, Generator):
    """Generator using BF16 (uncompressed) Llama 3.1 8B.

    This is the BASELINE for our TurboQuant research. It runs Llama 3.1 8B
    at full BF16 precision, no compression. Used to establish the reference
    VRAM and latency numbers that TurboQuantLlamaGenerator will be compared
    against.

    STATUS: stub. The inference framework (HuggingFace transformers, vLLM,
    llama.cpp, etc.) will be decided when Hamza is ready to implement.
    Both Llama generators should use the SAME framework for fair comparison.
    """

    def __init__(self, model_path: str, max_tokens: int = 500):
        """
        Args:
            model_path: Path or HF identifier for the Llama 3.1 8B BF16 model.
            max_tokens: Maximum tokens in the response.
        """
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.system_prompt = self.DEFAULT_SYSTEM_PROMPT

        # TODO: load the model here once framework is decided.

    def generate(self, query: str, context: str) -> str:
        raise NotImplementedError(
            "BF16LlamaGenerator.generate not yet implemented. "
            "Awaiting framework decision and integration."
        )


# ---------------------------------------------------------------------------
# Llama TurboQuant (compressed) — STUB, to be implemented
# ---------------------------------------------------------------------------

class TurboQuantLlamaGenerator(_LlamaGeneratorBase, Generator):
    """Generator using TurboQuant-compressed Llama 3.1 8B.

    This is the EXPERIMENTAL configuration for our research. It runs Llama
    3.1 8B with TurboQuant compression applied to the KV cache (3-bit
    target). Inference uses Hamza's custom C kernels for the compressed
    attention path.

    STATUS: stub. To be implemented once Hamza's TurboQuant kernels are
    integrated as a Python library and the BF16 framework is chosen.
    """

    def __init__(self, model_path: str, bit_width: int = 3, max_tokens: int = 500):
        """
        Args:
            model_path: Path or HF identifier for the Llama 3.1 8B model.
            bit_width: TurboQuant compression bit-width (2, 3, or 4).
                       Default 3 is the target configuration for the research.
            max_tokens: Maximum tokens in the response.
        """
        self.model_path = model_path
        self.bit_width = bit_width
        self.max_tokens = max_tokens
        self.system_prompt = self.DEFAULT_SYSTEM_PROMPT

        # TODO: load the model + apply TurboQuant compression here.

    def generate(self, query: str, context: str) -> str:
        raise NotImplementedError(
            "TurboQuantLlamaGenerator.generate not yet implemented. "
            "Awaiting Hamza's TurboQuant kernel integration."
        )


# --- rag-specific ---

# Data files (large PDFs and processed corpora)
data/

# FAISS indexes (regeneratable from corpus.json)
*.faiss

# macOS
.DS_Store

# Editor configs (uncomment if your team agrees to ignore these)
.vscode/
.idea/
