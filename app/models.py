"""Pydantic request/response models used by the API."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from app.core.search import (
    DEFAULT_MMR_LAMBDA,
    DEFAULT_PER_DOC_LIMIT,
    DEFAULT_SCORE_NORM,
)
from app.settings import get_settings

settings = get_settings()


class About(BaseModel):
    name: str = settings.app_name
    version: str = settings.app_version
    author: str = "Seweryn Sitarski (seweryn.sitarski@gmail.com) with support from Kat"
    description: str = "Dwustopniowy RAG ze streszczeniami + hybryda dense/TF‑IDF (Qdrant)"


class InitCollectionsRequest(BaseModel):
    collection_name: str = Field(default_factory=lambda: settings.collection_name)
    force_dim_probe: bool = False


class CollectionsExportRequest(BaseModel):
    collection_names: Optional[List[str]] = Field(
        default=None,
        description="Deprecated filter; eksporter zawsze pobiera wszystkie kolekcje.",
    )


class CollectionsImportRequest(BaseModel):
    archive_base64: str = Field(
        ...,
        description=(
            "Base64-encoded tar.gz snapshot bundle (Qdrant + TF-IDF) produced by /collections/export. "
            "Prefer multipart upload with 'archive_file' when using HTTP clients."
        ),
    )
    replace_existing: bool = Field(
        True,
        description="Drop (and recreate) existing collections and TF-IDF artifacts before import. Should stay true in most cases.",
    )


class ScanRequest(BaseModel):
    base_dir: str
    glob: str = "**/*"
    recursive: bool = True


class ScanResponse(BaseModel):
    files: List[str]


class IngestBuildRequest(BaseModel):
    base_dir: str
    glob: str = "**/*"
    recursive: bool = True
    reindex: bool = False
    chunk_tokens: int = Field(default_factory=lambda: settings.chunk_tokens)
    chunk_overlap: int = Field(default_factory=lambda: settings.chunk_overlap)
    language_hint: Optional[str] = None
    collection_name: str = Field(default_factory=lambda: settings.collection_name)
    enable_sparse: bool = True
    rebuild_tfidf: bool = True
    # Force regeneration of summaries and overwrite sidecar cache
    force_regen_summary: bool = Field(
        False,
        description="When true, bypass sidecar cache and regenerate LLM summary + dense vector, overwriting .summary/*.json.gz",
    )


class SummariesGenerateRequest(BaseModel):
    files: List[str]


class SearchQuery(BaseModel):
    query: List[str] = Field(
        ...,
        description=(
            "List of focused queries (each 3–12 words). Provide synonyms or variants to improve recall. "
            "Accepted shapes: string | List[str] | List[List[str]] (nested lists are flattened)."
        ),
    )
    top_m: int = Field(
        60,
        description=(
            "Stage‑1 candidate document count (summaries). Default 60; typical 50–200. "
            "Increase for broad topics, decrease for precise queries."
        ),
    )
    top_k: int = Field(
        10,
        description=(
            "Global final result count after Stage‑2 selection. Typical 5–10. "
            "Use per_doc_limit to prevent dominance of a single document."
        ),
    )
    mode: str = Field(
        "auto",
        description=(
            "Retrieval mode: auto|current|archival|all. 'current' filters is_active=true; 'archival' false. "
            "Heuristic in 'auto': queries with 'obowiązując*' → current; with 'archiwaln*' or explicit years → archival."
        ),
    )
    use_hybrid: bool = Field(True, description="Enable hybrid scoring (dense + TF‑IDF) for query.")
    dense_weight: float = Field(0.6, description="Weight of dense similarity in hybrid relevance [0..1].")
    sparse_weight: float = Field(0.4, description="Weight of sparse (TF‑IDF) similarity in hybrid relevance [0..1].")
    mmr_lambda: float = Field(DEFAULT_MMR_LAMBDA, description="MMR relevance-vs-diversity balance [0..1]. Higher = more relevance.")
    per_doc_limit: int = Field(DEFAULT_PER_DOC_LIMIT, description="Max results per single document in Stage-2.")
    score_norm: str = Field(DEFAULT_SCORE_NORM, description="Score normalization: minmax|zscore|none.")
    rep_alpha: Optional[float] = Field(None, description="Redundancy alpha in hybrid MMR (dense contribution). Defaults to dense_weight.")
    mmr_stage1: bool = Field(True, description="Apply hybrid MMR already at Stage-1 (summaries).")
    summary_mode: str = Field("first", description="Document summary duplication: none|first|all. 'first' shows once per doc.")
    # Entities-aware controls for LLM callers
    entities: Optional[List[str]] = Field(
        default=None,
        description=(
            "List of entities extracted from the user intent (names/IDs/years/quoted phrases). "
            "When omitted and AUTO_EXTRACT_QUERY_ENTITIES=true, backend applies heuristics."
        ),
    )
    entity_strategy: str = Field(
        default="optional",
        description=(
            "How to use entities: 'optional' (soft bonus only; no hard filter and default), "
            "'auto' (backend decides; typically similar to optional), 'boost' (soft bonus only), "
            "'must_any' (filter: any entity present), 'must_all' (filter: all entities present), "
            "'exclude' (exclude docs/chunks with these entities)."
        ),
    )
    # Runtime chunk-merging removed in 2.0.0. Blocks are built directly
    # from section-aware chunks generated at ingest time.
    result_format: str = Field(
        "blocks",
        description=(
            "Response shape: flat|grouped|blocks. Default 'blocks' (recommended for tools). When 'blocks', merged evidence blocks are returned (text + path + score)."
        ),
    )

    # Optional restriction to a known subset of documents by doc_id.
    # LLM guidance: Use this ONLY when you have previously retrieved
    # the candidate list via POST /browse/doc-ids. Otherwise leave it empty.
    restrict_doc_ids: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional allowlist of doc_id to restrict search scope. "
            "Use only when you already have a prior list of interesting documents (e.g., from POST /browse/doc-ids). "
            "When omitted or empty, the whole corpus is considered."
        ),
    )

    # Walidator wejścia 'query': akceptuje string, listę stringów lub listę list (zagnieżdżenia),
    # a następnie spłaszcza i czyści wartości do List[str]. Dzięki temu żądania typu
    # "query": [["a", "b" ,"c"]] nie kończą się 422 i są interpretowane jako "query": ["a","b","c"].
    @field_validator("query", mode="before")
    @classmethod
    def _normalize_query(cls, v):  # type: ignore[override]
        # Funkcja pomocnicza: przekształca dowolną strukturę do listy niepustych stringów.
        def to_list_of_str(x) -> List[str]:
            if x is None:
                return []
            if isinstance(x, str):
                s = x.strip()
                return [s] if s else []
            if isinstance(x, (list, tuple, set)):
                acc: List[str] = []
                for item in x:
                    acc.extend(to_list_of_str(item))
                return acc
            # Dla innych typów (np. liczby) użyj reprezentacji tekstowej
            s = str(x).strip()
            return [s] if s else []

        out = to_list_of_str(v)
        # Zabezpieczenie: jeżeli po konwersji nic nie zostało, przekaż oryginał (pozwoli Pydanticowi zgłosić 422)
        return out if out else v


class SearchHit(BaseModel):
    doc_id: str = Field(..., description="Stable document identifier (sha1 over absolute path).")
    path: str = Field(..., description="Absolute document path (for citation).")
    title: Optional[str] = Field(default=None, description="Document title extracted during ingest.")
    doc_date: Optional[str] = Field(default=None, description="Document date (YYYY, YYYY-MM, or YYYY-MM-DD) or 'brak'.")
    is_active: Optional[bool] = Field(default=None, description="Whether the document is marked as current (true) or archival (false).")
    doc_url: Optional[str] = Field(default=None, description="External source URL for the document (e.g., WIKAMP post).")
    section: Optional[str] = Field(default=None, description="Optional document section identifier, if present.")
    chunk_id: int = Field(..., description="Chunk index within the document (0-based).")
    score: float = Field(..., description="Hybrid relevance score (normalized according to score_norm).")
    snippet: str = Field(..., description="Short text snippet of the chunk or summary fallback.")
    summary: Optional[str] = Field(default=None, description="Document-level summary (presence controlled by summary_mode).")


class SearchResponse(BaseModel):
    took_ms: int = Field(..., description="Total search latency in milliseconds.")
    hits: List[SearchHit] = Field(..., description="Flat hit list (chunk-level).")
    groups: Optional[List["SearchGroup"]] = Field(default=None, description="Grouped results per document (summary + chunks).")
    blocks: Optional[List["MergedBlock"]] = Field(default=None, description="Merged blocks per (doc_id, section). Prefer for tool use.")


class SearchChunk(BaseModel):
    chunk_id: int = Field(..., description="Chunk index within the document (0-based).")
    score: float = Field(..., description="Hybrid relevance score (normalized).")
    snippet: str = Field(..., description="Short text snippet of the chunk.")


class SearchGroup(BaseModel):
    doc_id: str = Field(..., description="Stable document identifier.")
    path: str = Field(..., description="Absolute document path.")
    title: Optional[str] = Field(default=None, description="Document title extracted during ingest.")
    doc_date: Optional[str] = Field(default=None, description="Document date (YYYY, YYYY-MM, or YYYY-MM-DD) or 'brak'.")
    is_active: Optional[bool] = Field(default=None, description="Whether the document is marked as current (true) or archival (false).")
    doc_url: Optional[str] = Field(default=None, description="External source URL for the document (e.g., WIKAMP post).")
    summary: Optional[str] = Field(default=None, description="Document-level summary (single copy per document).")
    score: float = Field(..., description="Max score among group's chunks.")
    chunks: List[SearchChunk] = Field(..., description="Chunk-level results belonging to this document.")


class MergedBlock(BaseModel):
    doc_id: str = Field(..., description="Stable document identifier.")
    path: str = Field(..., description="Absolute document path.")
    title: Optional[str] = Field(default=None, description="Document title extracted during ingest.")
    doc_date: Optional[str] = Field(default=None, description="Document date (YYYY, YYYY-MM, or YYYY-MM-DD) or 'brak'.")
    is_active: Optional[bool] = Field(default=None, description="Whether the document is marked as current (true) or archival (false).")
    section: Optional[str] = Field(default=None, description="Optional section identifier.")
    doc_url: Optional[str] = Field(default=None, description="External source URL for the document (e.g., WIKAMP post).")
    first_chunk_id: int = Field(..., description="First chunk id (inclusive) in this merged block.")
    last_chunk_id: int = Field(..., description="Last chunk id (inclusive) in this merged block.")
    score: float = Field(..., description="Block score = max score among its member chunks.")
    summary: Optional[str] = Field(default=None, description="Document/section summary if requested by summary_mode.")
    text: str = Field(..., description="Merged textual content of the block (joined contiguous chunks).")
    # Pola opcjonalne dla rerankera (jeśli włączony):
    ranker_score: Optional[float] = Field(default=None, description="Ocena jakości nadana przez ranker (0..1).")


# Rebuild forward refs
SearchResponse.model_rebuild()


# --- Quotes (find occurrences in restricted documents) ---

class QuotesFindRequest(BaseModel):
    """Find exact occurrences of a query within a restricted set of documents.

    Requires a prior list of doc_ids (e.g., from POST /browse/doc-ids). Enumerates
    occurrences at the chosen granularity without using MMR/top_k. Always paginated.
    """

    restrict_doc_ids: List[str] = Field(..., description="List of doc_id to scan. Required.")
    query: object = Field(..., description="Phrase or tokens to search for. Accepts string or list of strings.")
    match: str = Field(
        "phrase",
        description="Match mode: phrase|any|all|regex. Default 'phrase' (exact substring, case-insensitive by default).",
    )
    case_sensitive: bool = Field(False, description="Case-sensitive matching. Default false (case-insensitive).")
    granularity: str = Field(
        "occurrence",
        description="Result granularity: occurrence|chunk. 'occurrence' returns each match; 'chunk' deduplicates per chunk.",
    )
    context_chars: int = Field(80, ge=0, le=400, description="Left/right context size around a match (characters).")
    limit: int = Field(200, ge=1, le=1000, description="Max items returned in one page.")
    cursor: Optional[str] = Field(default=None, description="Opaque cursor for pagination; pass from previous response.")

    @field_validator("query", mode="before")
    @classmethod
    def _coerce_query(cls, v):
        def to_list_of_str(x):
            if x is None:
                return []
            if isinstance(x, str):
                s = x.strip()
                return [s] if s else []
            if isinstance(x, (list, tuple, set)):
                acc = []
                for it in x:
                    acc.extend(to_list_of_str(it))
                return acc
            s = str(x).strip()
            return [s] if s else []
        out = to_list_of_str(v)
        return out if out else v


class QuoteItem(BaseModel):
    doc_id: str
    path: str
    title: Optional[str] = None
    doc_date: Optional[str] = None
    is_active: Optional[bool] = None
    section: Optional[str] = None
    chunk_id: int
    start: int
    end: int
    left_context: Optional[str] = None
    text: str
    right_context: Optional[str] = None


class QuotesFindResponse(BaseModel):
    took_ms: int
    total_quotes: int
    returned: int
    complete: bool
    next_cursor: Optional[str] = None
    quotes: List[QuoteItem]


class SparseQuery(BaseModel):
    indices: List[int]
    values: List[float]


# --- Browse/analytics models ---

class BrowseDocItem(BaseModel):
    """Lightweight document metadata returned by browse endpoints."""

    doc_id: str = Field(..., description="Stable document identifier (sha1 over absolute path).")
    title: Optional[str] = Field(default=None, description="Document title extracted during ingest.")
    doc_date: Optional[str] = Field(default=None, description="Document date (YYYY, YYYY-MM, or YYYY-MM-DD) or 'brak'.")
    is_active: Optional[bool] = Field(default=None, description="Whether the document is marked as current (true) or archival (false).")
    doc_kind: Optional[str] = Field(default=None, description="Coarse document kind inferred from title/signature (e.g., order, resolution, regulation).")
    doc_url: Optional[str] = Field(default=None, description="External source URL for the document (e.g., WIKAMP post).")


class BrowseQuery(BaseModel):
    """Lightweight browse request over Stage-1 candidates."""

    query: object
    top_m: int = 100
    mode: str = "auto"  # auto|current|archival|all
    use_hybrid: bool = True
    status: str = Field(
        "active",
        description=(
            "Document activity filter: 'active' (default), 'inactive', or 'all'. "
            "Applies a chunk-level is_active filter accordingly."
        ),
    )
    kinds: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional post-selection filter by inferred document kind. "
            "Accepts ASCII identifiers such as: resolution, order, announcement, notice, decision, regulation, policy, procedure, instruction, statute, other."
        ),
    )
    # Entities-aware (chunk-level) filters for browse
    entities: Optional[List[str]] = Field(
        default=None,
        description=(
            "Optional list of entities for chunk-level filtering. Entities are propagated to chunk payloads during ingest."
        ),
    )
    entity_strategy: str = Field(
        "auto",
        description=(
            "How to use entities: 'auto' (same as must_any), 'must_any' (at least one), 'must_all' (all required), 'exclude' (exclude docs/chunks with these entities), 'optional' (OR logic: content match OR entity match)."
        ),
    )
    text_match: str = Field(
        "none",
        description=(
            "Literal text requirement in chunk content: 'none' (default), 'phrase' (query substring), 'any' (any token from query), 'all' (all tokens from query)."
        ),
    )

    @field_validator("query", mode="before")
    @classmethod
    def _normalize_query(cls, v):  # type: ignore[override]
        def to_list_of_str(x) -> List[str]:
            if x is None:
                return []
            if isinstance(x, str):
                s = x.strip()
                return [s] if s else []
            if isinstance(x, (list, tuple, set)):
                acc: List[str] = []
                for item in x:
                    acc.extend(to_list_of_str(item))
                return acc
            s = str(x).strip()
            return [s] if s else []

        out = to_list_of_str(v)
        # Always return a list (possibly empty) so callers don't accidentally
        # treat None as a literal string.
        return out


class BrowseIdsResponse(BaseModel):
    took_ms: int
    total: int
    approx: bool = False
    candidates_total: Optional[int] = None
    docs: List[BrowseDocItem]


"""Facets models removed in 2.43.0. Use /browse/doc-ids and aggregate client-side."""


# --- Simplified doc-ids params (FTS-based) ---

class DocIdsQuery(BaseModel):
    """Simplified request for /browse/doc-ids."""

    query: object
    match: str = Field("phrase", description="Match mode: phrase|any|all (default: 'phrase'; prefer 'phrase' for single keywords).")
    status: str = Field("active", description="active|inactive|all (filters is_active)")
    kinds: Optional[List[str]] = None

    @field_validator("query", mode="before")
    @classmethod
    def _norm_query(cls, v):
        def to_list_of_str(x):
            if x is None:
                return []
            if isinstance(x, str):
                s = x.strip()
                return [s] if s else []
            if isinstance(x, (list, tuple, set)):
                acc: List[str] = []
                for it in x:
                    acc.extend(to_list_of_str(it))
                return acc
            s = str(x).strip()
            return [s] if s else []
        out = to_list_of_str(v)
        return out if out else v


class DocStatsResponse(BaseModel):
    total_docs: int
    active_docs: int
    inactive_docs: int


# --- Multi-query debug models (mirror /search/query with step-by-step outputs) ---

class DebugMultiEmbedRequest(BaseModel):
    # Accept single query or list of queries (flattened)
    query: object
    mode: str = "auto"
    use_hybrid: bool = True
    top_m: int = 100
    top_k: int = 10
    per_doc_limit: int = DEFAULT_PER_DOC_LIMIT
    score_norm: str = DEFAULT_SCORE_NORM
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    mmr_lambda: float = DEFAULT_MMR_LAMBDA
    rep_alpha: Optional[float] = None
    mmr_stage1: bool = True
    result_format: str = "blocks"
    summary_mode: str = "first"

    @field_validator("query", mode="before")
    @classmethod
    def _coerce_query_multi(cls, v):
        def to_list_of_str(x):
            if x is None:
                return []
            if isinstance(x, str):
                s = x.strip()
                return [s] if s else []
            if isinstance(x, (list, tuple, set)):
                acc = []
                for it in x:
                    acc.extend(to_list_of_str(it))
                return acc
            s = str(x).strip()
            return [s] if s else []
        out = to_list_of_str(v)
        return out if out else v


class DebugMultiEmbedResponse(BaseModel):
    step: str = "embed_multi"
    queries: List[str]
    mode: str
    q_vecs: List[List[float]]
    q_vec_lens: List[int]
    content_sparse_queries: Optional[List[Optional[SparseQuery]]] = None
    summary_sparse_queries: Optional[List[Optional[SparseQuery]]] = None
    _next: Optional[dict] = None


class DebugMultiStage1Request(BaseModel):
    queries: List[str]
    q_vecs: List[List[float]]
    mode: str = "auto"
    use_hybrid: bool = True
    top_m: int = 100
    top_k: int = 10
    per_doc_limit: int = DEFAULT_PER_DOC_LIMIT
    score_norm: str = DEFAULT_SCORE_NORM
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    mmr_stage1: bool = True
    mmr_lambda: float = DEFAULT_MMR_LAMBDA
    rep_alpha: Optional[float] = None
    summary_sparse_queries: Optional[List[Optional[SparseQuery]]] = None
    content_sparse_queries: Optional[List[Optional[SparseQuery]]] = None
    # shaping params for parity
    result_format: str = "blocks"
    summary_mode: str = "first"


class DebugMultiStage1Response(BaseModel):
    step: str = "stage1_multi"
    cand_doc_ids_list: List[List[str]]
    doc_maps: List[dict]
    _next: Optional[dict] = None


class DebugMultiStage2Request(BaseModel):
    queries: List[str]
    q_vecs: List[List[float]]
    mode: str = "auto"
    # Stage-1 outputs
    cand_doc_ids_list: Optional[List[List[str]]] = None
    doc_maps: Optional[List[dict]] = None
    # optional sparse from embed
    content_sparse_queries: Optional[List[Optional[SparseQuery]]] = None
    # scoring params
    top_m: int = 100
    top_k: int = 10
    per_doc_limit: int = DEFAULT_PER_DOC_LIMIT
    score_norm: str = DEFAULT_SCORE_NORM
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    mmr_lambda: float = DEFAULT_MMR_LAMBDA
    rep_alpha: Optional[float] = None
    # shaping params
    result_format: str = "blocks"
    summary_mode: str = "first"


class DebugMultiHit(BaseModel):
    doc_id: str
    path: Optional[str] = None
    section: Optional[str] = None
    chunk_id: int
    score: float
    snippet: Optional[str] = None


class DebugMultiStage2Response(BaseModel):
    step: str = "stage2_multi"
    per_query_hits: List[List[DebugMultiHit]]
    fused_hits: List[DebugMultiHit]
    _next: Optional[dict] = None


class DebugMultiShapeRequest(BaseModel):
    fused_hits: List[DebugMultiHit]
    result_format: str = "blocks"
    summary_mode: str = "first"


class DebugMultiShapeResponse(BaseModel):
    step: str = "shape_multi"
    results: List[SearchHit]
    groups: Optional[List[SearchGroup]] = None
    blocks: Optional[List[MergedBlock]] = None


# --- Golden QA (generator/edytor) ---

class GoldenGenerateRequest(BaseModel):
    base_dir: str
    glob: str = "**/*"
    recursive: bool = True
    out_dir: str = Field(default="data/golden", description="Katalog wyjściowy (pliki golden_qa.jsonl, golden_documents.jsonl)")
    limit_docs: Optional[int] = Field(default=None, description="Maksymalna liczba dokumentów do przetworzenia")
    per_doc_qa: int = Field(default=2, ge=1, description="Maksymalna liczba QA na dokument")
    target_qa: Optional[int] = Field(default=None, description="Zatrzymaj się po przybliżonej liczbie QA")
    seed: int = Field(default=123, description="Ziarno deterministycznego próbkowania")


class GoldenGenerateResponse(BaseModel):
    documents: int
    qa_items: int
    took_ms: int
    seed: int
    use_llm: bool
    llm_model: Optional[str] = None
    qa_path: str


class GoldenItem(BaseModel):
    id: str
    query: str
    expected_answer: str
    answer_type: Optional[str] = None
    score_rule: Optional[str] = None
    unanswerable: Optional[bool] = None
    difficulty: Optional[str] = None
    key_values: Optional[List[dict]] = None
    meta: Optional[dict] = None


class GoldenListResponse(BaseModel):
    items: List[GoldenItem]
    qa_path: str


class GoldenUpdateRequest(BaseModel):
    out_dir: str = Field(default="data/golden")
    id: str
    query: str
    expected_answer: str


class GoldenRegenerateRequest(BaseModel):
    out_dir: str = Field(default="data/golden")
    id: str
    # Random regeneration over whole corpus
    use_random_doc: bool = Field(default=True, description="Gdy true, wybiera losowy dokument z korpusu zamiast trzymać się pierwotnego")
    base_dir: Optional[str] = Field(default=None, description="Opcjonalny katalog bazowy do skanowania korpusu (fallback, gdy brak golden_documents.jsonl)")
    glob: Optional[str] = Field(default=None, description="Wzorzec glob (domyślnie **/*)")
    recursive: Optional[bool] = Field(default=True, description="Skanuj rekurencyjnie")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperatura dla LLM podczas regeneracji")
    ensure_different: bool = Field(default=True, description="Wymagaj, aby nowa para Q/A różniła się od poprzedniej i nie duplikowała istniejących")
    seed: Optional[int] = Field(default=None, description="Ziarno deterministycznego wyboru dokumentu (opcjonalnie)")


class GoldenAnswerRequest(BaseModel):
    out_dir: str = Field(default="data/golden")
    id: str
    temperature: float = Field(default=0.4, ge=0.0, le=2.0, description="Temperatura generowania odpowiedzi")
