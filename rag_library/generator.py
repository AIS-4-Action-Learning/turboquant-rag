"""
Generator interface and implementations.

A Generator takes a user query and retrieved context, and produces an answer.
This is the component that gets swapped between OpenAI, Gemini, BF16 Llama,
and TurboQuant-compressed Llama for our research benchmarks.

Class hierarchy:
    Generator (ABC)
    ├── OpenAIGenerator
    ├── GeminiGenerator
    ├── BF16LlamaGenerator      ──┐
    └── TurboQuantLlamaGenerator ──┴── both inherit shared logic from _LlamaGeneratorBase
"""

import time
from abc import ABC, abstractmethod
from typing import no_type_check
from app.llama_models import Llama, LlamaBF16, LlamaCompressed, LlamaGenerator, format_prompt

# ---------------------------------------------------------------------------
# Public abstract base class
# ---------------------------------------------------------------------------

class Generator(ABC):
    """Abstract base class for answer generators.

    Any concrete generator must inherit from this class and implement
    the `generate` method.
    """

    @abstractmethod
    def generate(self, query: str, context: str, omit_sysprompt: bool) -> str:
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

    def generate(self, query: str, context: str, omit_sysprompt: bool) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt if not omit_sysprompt else ""},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        if self.cost_tracker is not None:
            self.cost_tracker(response)

        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Gemini implementation (Google's free-tier-friendly API)
# ---------------------------------------------------------------------------

class GeminiGenerator(Generator):
    """Generator that uses Google's Gemini API.

    Gemini has a genuinely free tier (no credit card, no expiration), making
    it the best option for team prototyping without any out-of-pocket cost.

    Defaults to `gemini-2.5-flash-lite` because it has the most generous free
    rate limits (15 RPM, 1000 requests/day). Switch to `gemini-2.5-flash` or
    `gemini-2.5-pro` for higher-quality responses (with lower daily caps).

    NOTE: requests are rate-limited automatically by sleeping between calls
    to stay under the per-minute RPM cap.
    """

    # Conservative defaults (one request every ~4.5s) keep us safely under the
    # 15 RPM free-tier ceiling. Override in the constructor if needed.
    DEFAULT_MIN_INTERVAL_SECONDS = 4.5

    def __init__(
        self,
        client,
        model: str = "gemini-2.5-flash-lite",
        temperature: float = 0,
        max_tokens: int = 500,
        min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS,
    ):
        """
        Args:
            client: An initialized google.genai.Client (passed in, not created
                    here — same pattern as OpenAIGenerator).
            model: Gemini model name. Free-tier options:
                   - "gemini-2.5-flash-lite" (15 RPM, 1000/day — fastest free)
                   - "gemini-2.5-flash"      (10 RPM, 250/day  — better quality)
                   - "gemini-2.5-pro"        (5 RPM, 100/day   — best quality)
            temperature: Sampling temperature (0 for deterministic).
            max_tokens: Maximum tokens in the response.
            min_interval_seconds: Minimum seconds between requests. Default
                                  4.5s keeps us under 15 RPM for Flash-Lite.
                                  Increase if using lower-RPM models.
        """
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.min_interval_seconds = min_interval_seconds
        self._last_request_time = 0.0

        # System prompt is identical to OpenAIGenerator's — kept aligned so
        # behavior is comparable across baselines.
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

    def _throttle(self) -> None:
        """Sleep if needed to respect the per-minute rate limit."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)
        self._last_request_time = time.time()

    def generate(self, query: str, context: str, omit_sysprompt: bool) -> str:
        # Gemini doesn't have a separate "system message" role like OpenAI;
        # we prepend the system prompt to the user content.
        sysprompt = self.system_prompt if not omit_sysprompt else ""
        prompt = (
            f"{sysprompt}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}"
        )

        self._throttle()

        # Lazy import so the SDK is only required when actually used.
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )

        return response.text


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
    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that answers questions about Deep Learning.
You must:
Answer based ONLY on the provided context, if the question is not very specific to the context or out of scope, reply with: "I can't answer this question". Other wise:
1. Cite your sources (document name and page number).
2. Be concise and accurate.
3. If the question lacks clarification, request for clarification.
"""

    def _format_prompt(self, query: str, context: str, omit_sysprompt: bool) -> str:
        """Format query + context into a Llama 3.1 chat-template string.

        Llama 3.1 is instruction-tuned on a strict chat template. Without it,
        the model produces incoherent noise and repetitive loops.
        """
        sysprompt = self.DEFAULT_SYSTEM_PROMPT if not omit_sysprompt else ""

        # Llama 3.1 chat template (IDs: <|begin_of_text|>=128000,
        # <|start_header_id|>=128006, <|end_header_id|>=128007, <|eot_id|>=128009)
        return format_prompt(query, context, sysprompt)

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

    def __init__(self, model: LlamaBF16, max_tokens: int = 500):
        """
        Args:
            model_path: Path or HF identifier for the Llama 3.1 8B BF16 model.
            max_tokens: Maximum tokens in the response.
        """
        self.llama = model
        self.max_tokens = max_tokens
        self.system_prompt = self.DEFAULT_SYSTEM_PROMPT

        # TODO: load the model here once framework is decided.

        self.llama_generator = LlamaGenerator()

    def generate(self, query: str, context: str, omit_sysprompt: bool) -> str:
        formatted_prompt = self._format_prompt(query, context, omit_sysprompt)
        token_ids, prompt_tensors = self.llama.input_encoding(formatted_prompt)
        gen_limit = self.max_tokens

        response = self.llama_generator.generate(
            tensor_tokens=prompt_tensors,
            token_ids=token_ids,
            llama=self.llama,
            max_gen_len=gen_limit
        )

        return response


# ---------------------------------------------------------------------------
# Llama TurboQuant (compressed) — STUB, to be implemented
# ---------------------------------------------------------------------------

class TurboQuantLlamaGenerator(_LlamaGeneratorBase, Generator):
    """Generator using TurboQuant-compressed Llama 3.1 8B.

    This is the EXPERIMENTAL configuration for our research. It runs Llama
    3.1 8B with TurboQuant compression applied to the KV cache (3-bit
    target). Inference uses the custom C kernels for the compressed
    attention path.

    STATUS: stub. To be implemented once the TurboQuant kernels are
    integrated as a Python library and the BF16 framework is chosen.
    """

    def __init__(self, model: LlamaCompressed, max_tokens: int = 500):
        """
        Args:
            model_path: Path or HF identifier for the Llama 3.1 8B model.
            bit_width: TurboQuant compression bit-width (2, 3, or 4).
                       Default 3 is the target configuration for the research.
            max_tokens: Maximum tokens in the response.
        """
        self.max_tokens = max_tokens
        self.system_prompt = self.DEFAULT_SYSTEM_PROMPT

        # TODO: load the model + apply TurboQuant compression here.
        self.llama = model
        self.llama_generator = LlamaGenerator()

    def generate(self, query: str, context: str, omit_sysprompt: bool) -> str:
        formatted_prompt = self._format_prompt(query, context, omit_sysprompt)
        token_ids, prompt_tensors = self.llama.input_encoding(formatted_prompt)
        gen_limit = self.max_tokens

        response = self.llama_generator.generate(
            tensor_tokens=prompt_tensors,
            token_ids=token_ids,
            llama=self.llama,
            max_gen_len=gen_limit
        )

        return response
