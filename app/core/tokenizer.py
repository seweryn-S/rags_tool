"""Utility helpers for configurable tokenization.

Provides a lightweight abstraction over different tokenizer backends so that
the rest of the codebase can rely on a common encode/decode/count interface
without hard-coding the implementation (e.g., tiktoken vs Hugging Face).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence
import warnings


class TokenizerError(RuntimeError):
    """Raised when a tokenizer cannot be initialized or used."""


@dataclass
class TokenizerAdapter:
    """Minimal encode/decode interface shared by chunking and embedding logic."""

    name: str
    encode: Callable[[str], List[int]]
    decode: Callable[[Sequence[int]], str]
    extra_token_count: int = 0  # tokens automatically added by the backend


# Build a TokenizerAdapter backed by tiktoken for the given spec.
def _build_tiktoken_adapter(spec: str) -> TokenizerAdapter:
    try:  # pragma: no cover - optional dependency
        import tiktoken  # type: ignore
    except Exception as exc:  # pragma: no cover - hard failure
        raise TokenizerError(
            "Tokenization requires 'tiktoken'. Install it with `pip install tiktoken`."
        ) from exc

    encoding_name = spec.split(":", 1)[1] or "cl100k_base"
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception as exc:  # pragma: no cover - defensive
        raise TokenizerError(
            f"Failed to initialize tiktoken encoding '{encoding_name}'."
        ) from exc

    return TokenizerAdapter(
        name=f"tiktoken:{encoding_name}",
        encode=encoding.encode,
        decode=lambda tokens: encoding.decode(list(tokens)),
        extra_token_count=0,
    )


# Build a TokenizerAdapter backed by Hugging Face transformers for the spec.
def _build_hf_adapter(spec: str) -> TokenizerAdapter:
    try:  # pragma: no cover - optional dependency
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:  # pragma: no cover - hard failure
        raise TokenizerError(
            "Tokenization requires 'transformers'. Install it with `pip install transformers`."
        ) from exc

    model_name = spec.split(":", 1)[1]
    if not model_name:
        raise TokenizerError("Tokenizer spec 'hf:' must include a model name, e.g. 'hf:bert-base-uncased'.")

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )  # pragma: no cover - runtime path
    except Exception as exc:  # pragma: no cover - defensive
        raise TokenizerError(f"Failed to load Hugging Face tokenizer '{model_name}'.") from exc

    extra = 0
    try:
        extra = int(tokenizer.num_special_tokens_to_add(pair=False))
    except Exception:
        extra = 0

    def encode(text: str) -> List[int]:
        # Ignore special tokens; backend usually injects them automatically.
        return tokenizer.encode(text, add_special_tokens=False)  # type: ignore[arg-type]

    def decode(tokens: Sequence[int]) -> str:
        return tokenizer.decode(tokens, clean_up_tokenization_spaces=False, skip_special_tokens=True)

    return TokenizerAdapter(
        name=f"hf:{model_name}",
        encode=encode,
        decode=decode,
        extra_token_count=extra,
    )


# Load a tokenizer adapter according to configuration string.
def load_tokenizer(spec: str | None) -> TokenizerAdapter:
    """Instantiate a tokenizer adapter based on configuration string."""

    normalized = (spec or "tiktoken:cl100k_base").strip()
    if not normalized:
        normalized = "tiktoken:cl100k_base"

    lowered = normalized.lower()
    if lowered.startswith("tiktoken:"):
        return _build_tiktoken_adapter(normalized)
    if lowered.startswith("hf:") or lowered.startswith("huggingface:"):
        try:
            return _build_hf_adapter(normalized)
        except TokenizerError as exc:
            warnings.warn(
                (
                    f"{exc} Falling back to tiktoken:cl100k_base for local chunking. "
                    "Remote embedding calls still use the configured embedding model."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            return _build_tiktoken_adapter("tiktoken:cl100k_base")

    raise TokenizerError(
        "Unsupported tokenizer spec. Use 'tiktoken:<encoding>' or 'hf:<model_name>'."
    )


# Count tokens with the adapter, including backend-added specials.
def count_tokens(adapter: TokenizerAdapter, text: str) -> int:
    """Count tokens using adapter, including backend-added specials."""

    if not text:
        return 0
    return len(adapter.encode(text)) + adapter.extra_token_count


# Truncate text so total tokens (incl. specials) do not exceed max_tokens.
def truncate_to_tokens(adapter: TokenizerAdapter, text: str, max_tokens: int) -> str:
    """Trim text so that total tokens (including specials) do not exceed max_tokens."""

    if max_tokens <= 0 or not text:
        return ""

    budget = max_tokens - adapter.extra_token_count
    if budget <= 0:
        return ""

    tokens = adapter.encode(text)
    if len(tokens) <= budget:
        return text
    return adapter.decode(tokens[:budget])


# Yield token-aware sliding windows with a configurable overlap.
def sliding_windows(
    adapter: TokenizerAdapter,
    text: str,
    target_tokens: int,
    overlap_tokens: int,
) -> Iterable[str]:
    """Yield text windows that respect token limits for the given adapter."""

    tokens = adapter.encode(text)
    if not tokens:
        return []

    body_target = max(1, target_tokens - adapter.extra_token_count)
    body_overlap = max(0, overlap_tokens - adapter.extra_token_count)

    # Prevent non-progress loops
    if body_overlap >= body_target:
        body_overlap = max(body_target - 1, 0)

    chunks: List[str] = []
    start = 0
    n = len(tokens)
    while start < n:
        end = min(n, start + body_target)
        piece_tokens = tokens[start:end]
        piece = adapter.decode(piece_tokens)
        if piece.strip():
            chunks.append(piece)
        if end >= n:
            break
        start = end - body_overlap if body_overlap else end
    return chunks
