"""Application configuration loading using pydantic-settings."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class SummRAGSettings(BaseSettings):
    """Centralised configuration for the rags_tool service."""

    app_name: str = "rags_tool"
    app_version: str = "2.45.0"

    qdrant_url: str = Field(default="http://127.0.0.1:6333", alias="QDRANT_URL")
    qdrant_api_key: Optional[str] = Field(default=None, alias="QDRANT_API_KEY")
    qdrant_request_timeout: float = Field(default=60.0, alias="QDRANT_TIMEOUT")
    embedding_api_url: str = Field(default="http://127.0.0.1:8000/v1", alias="EMBEDDING_API_URL")
    embedding_api_key: str = Field(default="sk-no-key", alias="EMBEDDING_API_KEY")
    embedding_model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")
    # Tokenizer spec controlling chunking + local token limits
    embedding_tokenizer: str = Field(
        default="tiktoken:cl100k_base", alias="EMBEDDING_TOKENIZER"
    )
    # Max tokens per single input to the embedding endpoint (safety cap)
    embedding_max_tokens: int = Field(default=512, alias="EMBEDDING_MAX_TOKENS")
    # (Debugging note) embedding_max_tokens is a local safety cap only; the backend
    # may use different tokenization.
    # Prefixes used by some retrieval models that expect instruction-style inputs
    # Defaults align with sdadas/mmlw-retrieval-roberta-large-v2
    embedding_query_prefix: str = Field(default="query: ", alias="EMBEDDING_QUERY_PREFIX")
    embedding_passage_prefix: str = Field(default="passage: ", alias="EMBEDDING_PASSAGE_PREFIX")
    summary_api_url: str = Field(default="http://127.0.0.1:8001/v1", alias="SUMMARY_API_URL")
    summary_api_key: str = Field(default="sk-no-key", alias="SUMMARY_API_KEY")
    summary_model: str = Field(default="gpt-4o-mini", alias="SUMMARY_MODEL")
    summary_system_prompt: str = Field(
        default="Jesteś zwięzłym ekstrakcyjnym streszczaczem.",
        alias="SUMMARY_SYSTEM_PROMPT",
    )
    summary_prompt: str = Field(
        default=(
            "Streść poniższy tekst w maks. 5 zdaniach. Wypisz też sekcje: 'TITLE' (krótki, "
            " jednoznaczny tytuł dokumentu — preferuj pierwszą linię lub numer aktu; "
            " pojedyncza fraza, bez dodatkowego komentarza), 'SIGNATURE' (10–20 lematów "
            " kluczowych), 'ENTITIES' (nazwy własne/ID/zakresy dat), 'DATE' (data "
            " wprowadzenia/ogłoszenia dokumentu; preferuj format YYYY-MM-DD lub YYYY; wpisz "
            " dokładnie 'brak', jeśli brak informacji) oraz 'REPLACEMENT' (krótka lista tytułów, "
            " zawsze w mianowniku; może składać się jedynie z krótkich tytułów aktów zastępowanych, "
            " jeśli tekst jednolity wypisz wszystkie tytuły aktów które ujednolica, rozdziel „;”; "
            " wpisz dokładnie 'brak', jeżeli brak danych). Dodatkowo podaj 'SUBTITLE' "
            " (string; krótki jednoznaczny podtytuł, najczęściej określa funkcje dokumentu "
            " i nie jest numerowany, max 100 znaków, zawsze w mianowniku; jeśli nie umiesz "
            " zidentyfikować wpisz dokładnie 'brak'). Bez komentarzy.\\n\\nFORMAT:\\nTITLE: ...\\nSUBTITLE: ...\\nSUMMARY: ...\\nSIGNATURE: "
            " lemma1, lemma2, ...\\nENTITIES: ...\\nDATE: ...\\nREPLACEMENT: ...\\n\\nTEKST:\\n"
        ),
        alias="SUMMARY_PROMPT",
    )
    summary_prompt_json: str = Field(
        default=(
            "Zwróć wyłącznie poprawny JSON bez komentarzy i bez kodu. Klucze: 'title' (string; "
            " krótki jednoznaczny tytuł, preferuj pierwszą linię lub numer dokumentu; max 200 znaków, "
            " zawsze w mianowniku), 'subtitle' (string; krótki jednoznaczny podtytuł, "
            " najczęściej określa funkcje dokumentu i nie jest numerowany, max 100 znaków, "
            " zawsze w mianowniku; jeśli nie umiesz zidentyfikować wpisz dokładnie 'brak'), "
            " 'summary' (string; max 5 zdań po polsku), 'signature' "
            " (lista 10–20 lematów kluczowych jako strings), 'entities' (lista stringów z nazwami "
            " własnymi/ID/zakresami dat), 'doc_date' (string; data wprowadzenia/ogłoszenia dokumentu "
            " w formacie 'YYYY-MM-DD' lub 'YYYY-MM' lub 'YYYY'; jeśli brak informacji wpisz dokładnie 'brak'), "
            " 'replacement' (string; krótkie tytuły aktów zastępowanych, zawsze w mianowniku; może składać się jedynie z krótkich tytułów aktów "
            " zastępowanych, jeśli tekst jednolity wypisz wszystkie tytuły aktów które ujednolica, separator ';'; wpisz dokładnie 'brak', jeśli brak informacji), "
            " 'is_active' (bool; oceń WYŁĄCZNIE na podstawie kontekstu PATH, czy dokument jest obowiązujący. USTAW false TYLKO, gdy PATH zawiera jednoznaczne słowa-klucze wskazujące archiwum (np. 'archiwum', 'archiwal', 'archive', 'archives', 'archival', 'old', 'stare', 'stary', 'history', 'deprecated', 'zarchiwizowane'). NIE traktuj samych lat/dat w PATH jako oznaki archiwalności; w przeciwnym razie ustaw true)."
        ),
        alias="SUMMARY_PROMPT_JSON",
    )
    collection_name: str = Field(default="rags_tool", alias="COLLECTION_NAME")
    summary_collection_name: Optional[str] = Field(
        default=None, alias="SUMMARY_COLLECTION_NAME"
    )
    content_collection_name: Optional[str] = Field(
        default=None, alias="CONTENT_COLLECTION_NAME"
    )
    debug: bool = Field(default=False, alias="DEBUG")
    vector_store_dir: Path = Field(default=Path(".rags_tool_store"), alias="VECTOR_STORE_DIR")
    # Embedding vector dimension for the chosen embedding model
    embedding_dim: int = Field(default=1024, alias="EMBEDDING_DIM")
    # Prefer JSON responses for summaries (OpenAI JSON mode). Fallback to text parser if unsupported.
    summary_json_mode: bool = Field(default=True, alias="SUMMARY_JSON_MODE")

    # Browse search limits (controls for /browse/doc-ids)
    # Oversample factor for Qdrant search limit relative to top_m. Higher when
    # text_match is enabled to reduce false negatives of literal filters.
    browse_search_oversample: int = Field(default=10, alias="BROWSE_SEARCH_OVERSAMPLE")
    # Absolute cap for Qdrant search limit to avoid pathological queries.
    browse_search_limit_max: int = Field(default=4000, alias="BROWSE_SEARCH_LIMIT_MAX")

    # Chunking defaults (token-based). Tune per embedding model.
    chunk_tokens: int = Field(default=400, alias="CHUNK_TOKENS")
    chunk_overlap: int = Field(default=64, alias="CHUNK_OVERLAP")
    section_merge_level: str = Field(default="ust", alias="SECTION_MERGE_LEVEL")

    # OpenAPI / tool description used by the /search/query endpoint. Can be overridden
    # via .env to tailor the wording for a specific corpus (e.g., PŁ documents).
    search_tool_description: str = Field(
        default=(
            "Dwustopniowe wyszukiwanie RAG (streszczenia → pełne treści) zwracające krótkie, "
            "cytowalne bloki ('blocks') jako materiał dowodowy do odpowiedzi. Endpoint nie służy do liczenia ani listowania dokumentów.\n\n"
            "Zakres domyślny:\n"
            "- Gdy 'mode' = 'auto', traktuj zapytanie jako 'current' (obowiązujące akty), chyba że kontekst wyraźnie wskazuje inaczej: \n"
            "  • 'archiwal*', 'stara', 'wersja z ...' lub konkretne lata → użyj 'archival',\n"
            "  • 'wszystkie', 'cała historia', 'pełen zakres' → użyj 'all'.\n\n"
            "Zawężanie po doc_id (ważne):\n"
            "- Jeśli WCZEŚNIEJ pobrałeś listę kandydatów przez POST /browse/doc-ids, przekaż ją w polu 'restrict_doc_ids' (lista doc_id), aby ograniczyć zakres wyszukiwania.\n"
            "- W przeciwnym razie nie ustawiaj 'restrict_doc_ids' — wyszukiwanie obejmuje cały korpus zgodnie z trybem.\n"
            "- Heurystyka po stronie serwera (dla 'restrict_doc_ids'): automatycznie podnosimy N do minimalnych wartości, aby nie urywać cytatów:\n"
            "  • top_m ≥ 500,\n"
            "  • top_k ≥ 50,\n"
            "  • per_doc_limit ≥ 15.\n\n"
            "Zachowanie:\n"
            "- Etap 1 selekcjonuje dokumenty po streszczeniach (hybryda dense + TF‑IDF, opcjonalny MMR).\n"
            "- Etap 2 wyszukuje w chunkach wybranych dokumentów i buduje zmergowane sekcje ('blocks').\n"
            "- Opcjonalny reranker porządkuje gotowe bloki.\n\n"
            "Jak wołać (dla modeli LLM):\n"
            "- Podawaj 2–8 zwięzłych wariantów 'query' (tytuły/sygnatury/datacje/słowa kluczowe).\n"
            "- Preferuj wynik 'result_format' = 'blocks' (domyślnie).\n"
            "- Utrzymuj 'top_k' w zakresie 5–10; kontroluj dominację jednego dokumentu 'per_doc_limit'.\n"
            "- Jeśli masz encje (nazwy/ID/lata/cytaty), przekaż w 'entities' i wybierz 'entity_strategy' (optional/auto/boost/must_any/must_all/exclude). Domyślnie 'optional'.\n\n"
            "Czego NIE robić tym endpointem:\n"
            "- Nie proś o liczbę dokumentów ani same listy doc_id/tytułów. Do tego używaj: \n"
            "  • POST /browse/doc-ids — lista doc_id + meta (tytuł, data, is_active, doc_kind) oraz 'candidates_total'; dla samej liczby ustaw 'limit=0' i zaufaj 'candidates_total' (nie stosuj sond 'limit:1').\n\n"
            "Używaj wyłącznie języka polskiego. Cały korpus oraz metadane są po polsku."
        ),
        alias="SEARCH_TOOL_DESCRIPTION",
    )

    # Globalny przełącznik: pomiń Etap 1 (streszczenia) i szukaj od razu w całym korpusie (chunkach).
    # Sterowany wyłącznie przez admina z .env (brak parametru w API).
    search_skip_stage1_default: bool = Field(
        default=False, alias="SEARCH_SKIP_STAGE1_DEFAULT"
    )

    # --- Hybryda 2‑query (dense + sparse w dwóch zapytaniach) ---
    # Gdy true, Stage 1 i Stage 2 wykonują dwa zapytania: osobno po dense i sparse,
    # a następnie łączą wyniki po stronie aplikacji. Pozwala docelowo usunąć TF‑IDF
    # z payloadów Qdranta (mniejsze rekordy i niższe CPU po stronie serwera).
    search_dual_query_sparse: bool = Field(default=False, alias="SEARCH_DUAL_QUERY_SPARSE")
    dual_query_rrf_k: int = Field(default=60, alias="DUAL_QUERY_RRF_K")
    dual_query_oversample: int = Field(default=2, alias="DUAL_QUERY_OVERSAMPLE")
    dual_query_dense_for_mmr: bool = Field(default=True, alias="DUAL_QUERY_DENSE_FOR_MMR")

    # Redukcja payloadów (wybór pól with_payload). Zalecane pozostawić włączone.
    search_minimal_payload: bool = Field(default=True, alias="SEARCH_MINIMAL_PAYLOAD")

    # Batchowanie sekcji per dokument (jedna kwerenda scroll łącząca sekcje po prefiksach)
    batch_section_fetch: bool = Field(default=True, alias="BATCH_SECTION_FETCH")

    # Deduplication: skip identical files during ingest based on content hash
    dedupe_on_ingest: bool = Field(default=True, alias="DEDUPE_ON_INGEST")

    # --- Entities-aware search (filters and boosting) ---
    # Soft boost for entity matches at Stage 1 (summaries)
    entity_boost_stage1: float = Field(default=0.15, alias="ENTITY_BOOST_STAGE1")
    # Soft boost for entity matches at Stage 2 (chunks/sections)
    entity_boost_stage2: float = Field(default=0.10, alias="ENTITY_BOOST_STAGE2")
    # Auto-extract entities from user queries when not provided explicitly
    auto_extract_query_entities: bool = Field(default=True, alias="AUTO_EXTRACT_QUERY_ENTITIES")

    # --- Validators for forgiving .env parsing (blank strings) ---
    @field_validator(
        "search_skip_stage1_default",
        "search_dual_query_sparse",
        "dual_query_dense_for_mmr",
        "search_minimal_payload",
        "batch_section_fetch",
        "dedupe_on_ingest",
        "auto_extract_query_entities",
        mode="before",
    )
    @classmethod
    def _coerce_bool_env(cls, v):  # type: ignore[override]
        if isinstance(v, str) and v.strip() == "":
            return False
        return v

    @field_validator("dual_query_rrf_k", mode="before")
    @classmethod
    def _coerce_rrf_k(cls, v):  # type: ignore[override]
        if isinstance(v, str) and v.strip() == "":
            return 60
        return v

    @field_validator("dual_query_oversample", mode="before")
    @classmethod
    def _coerce_oversample(cls, v):  # type: ignore[override]
        if isinstance(v, str) and v.strip() == "":
            return 2
        return v

    # --- Reranker (OpenAI-compatible) minimal configuration ---
    # Pusty BASE_URL lub MODEL oznacza wyłączony ranker i brak rerankingu.
    # K i N są kontrolowane z .env, nie przez publiczne API.
    ranker_base_url: Optional[str] = Field(default=None, alias="RANKER_BASE_URL")
    ranker_api_key: Optional[str] = Field(default=None, alias="RANKER_API_KEY")
    ranker_model: Optional[str] = Field(default=None, alias="RANKER_MODEL")
    # MAX caps (synchronised with request params at runtime). Legacy env names
    # RERANK_TOP_N and RETURN_TOP_K are still accepted as fallbacks.
    rerank_top_n_max: int = Field(default=50, alias="RERANK_TOP_N_MAX")
    return_top_k_max: int = Field(default=50, alias="RETURN_TOP_K_MAX")
    ranker_score_threshold: float = Field(default=0.2, alias="RANKER_SCORE_THRESHOLD")
    # Długość kontekstu dla pojedynczego passage wysyłanego do rankera (znaki, przybliżenie).
    # Jeśli model ma twardy limit tokenów, rekomendujemy ustawić konserwatywnie (np. 2048 znaków).
    ranker_max_length: int = Field(default=2048, alias="RANKER_MAX_LENGTH")
    # Absolute minimum score; items below are never returned (no backfill)
    ranker_hard_threshold: float = Field(default=0.65, alias="RANKER_HARD_THRESHOLD")

    # Backwards-compatibility: accept legacy env vars when MAX variants are not provided
    @field_validator("rerank_top_n_max", mode="before")
    @classmethod
    def _fallback_rerank_top_n_max(cls, v):  # type: ignore[override]
        if isinstance(v, str) and v.strip() != "":
            return v
        try:
            import os
            legacy = os.getenv("RERANK_TOP_N")
            if legacy is not None and legacy.strip() != "":
                return int(legacy)
        except Exception:
            pass
        return v

    @field_validator("return_top_k_max", mode="before")
    @classmethod
    def _fallback_return_top_k_max(cls, v):  # type: ignore[override]
        if isinstance(v, str) and v.strip() != "":
            return v
        try:
            import os
            legacy = os.getenv("RETURN_TOP_K")
            if legacy is not None and legacy.strip() != "":
                return int(legacy)
        except Exception:
            pass
        return v

    @property
    def qdrant_summary_collection(self) -> str:
        base = self.summary_collection_name or f"{self.collection_name}_summaries"
        return f"{base}_active"

    @property
    def qdrant_content_collection(self) -> str:
        base = self.content_collection_name or f"{self.collection_name}_content"
        return f"{base}_active"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


@lru_cache(maxsize=1)
def get_settings() -> SummRAGSettings:
    """Return cached application settings instance."""

    return SummRAGSettings()
