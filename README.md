# rags_tool (2.45.0)

## Nowości w 2.45.0
- Qdrant: wprowadzono parę aliasów `*_summaries_active` / `*_content_active` dla głównego korpusu (i dodatkowych baz), dzięki czemu pełny reindex buduje nową wersję kolekcji w tle, a po zakończeniu aliasy są atomowo przełączane na nową parę (blue‑green). Stara para kolekcji pozostaje w Qdrant i może zostać skasowana ręcznie po weryfikacji.
- Ingest: endpoint `POST /ingest/build` z `reindex=true` nie usuwa już aktywnej kolekcji na początku przebudowy. Zamiast tego tworzy nową bazę `COLLECTION_NAME_vYYYYMMDDhhmmss`, indeksuje do niej dane i dopiero na końcu przełącza aliasy, co eliminuje wielogodzinne okno „pustego” korpusu.

## Nowości w 2.44.0
- Ingest: podczas wczytywania korpusu ładowana jest mapa linków WIKAMP z pliku `wikamp_normative_acts_map_doc.csv` znajdującego się w katalogu głównym korpusu. Linki są parowane po nazwie pliku (bez rozszerzenia) z priorytetem dla `.doc`/`.docx`, następnie `.pdf`, a na końcu pierwszego dostępnego wpisu.
- Search/Browse: wyniki `/search/query` (hits/groups/blocks) oraz `/browse/doc-ids` zwracają dodatkowe pole `doc_url` z linkiem źródłowym dokumentu, które można cytować w odpowiedziach LLM.

## Nowości w 2.43.0
- Usunięto endpoint `POST /browse/facets`. Dla list i liczby dokumentów korzystaj z `POST /browse/doc-ids` (pole `candidates_total`). Rozkłady/filtry twórz po stronie klienta na podstawie zwróconych metadanych (`is_active`, `doc_date`, `doc_kind`).

Dwustopniowy serwis RAG zbudowany na FastAPI. System wspiera streszczanie dokumentów, indeksowanie w Qdrant oraz wyszukiwanie hybrydowe (dense + TF-IDF). Administrator może globalnie pominąć Etap 1 (streszczenia) i wyszukiwać bezpośrednio w całym korpusie chunków — patrz `SEARCH_SKIP_STAGE1_DEFAULT`.

## Nowości w 2.41.1
- Browse: poprawka — przy `limit=0` próbka dokumentów nie jest dołączana, gdy jedynym parametrem jest `status` (np. `status=all` lub `status=inactive`). Próbka pojawia się tylko, gdy podano `query` lub `kinds`.

## Nowości w 2.41.0
- Browse: `POST /browse/doc-ids` przy `limit=0` zwraca teraz próbkę do 15 dokumentów (ORDER BY doc_date DESC) RAZEM z dokładnym `candidates_total`, jeśli podano zawężenie treścią (`query`) lub filtrem `kinds`. Gdy brak takich zawężeń (pełny korpus), `limit=0` zwraca wyłącznie `candidates_total` (bez listy). Pole `approx=true` sygnalizuje próbkę niepełną.

## Nowości w 2.40.1
- FTS5: uproszczono kod — usunięto automatyczną detekcję/migrację schematu. Jeśli zmienisz schemat (np. po aktualizacji), usuń plik indeksu FTS lub wywołaj `POST /fts/rebuild` aby odbudować indeks.

## Nowości w 2.40.0
- FTS5: rozszerzony schemat o `doc_kind` (UNINDEXED). Indeks FTS przechowuje teraz także rodzaj dokumentu (ASCII: np. `resolution`, `order`, `regulation`), wyznaczany heurystycznie z tytułu.
- Browse: `/browse/doc-ids` z filtrem `kinds` liczy kandydatów SQL‑owo (COUNT DISTINCT doc_id) także dla `limit=0` — szybciej i bez skanowania metadanych.
- Migracja: po zmianie schematu FTS usuń plik indeksu lub wywołaj `POST /fts/rebuild`; automatyczna migracja między schematami nie jest wykonywana.

## Nowości w 2.38.0
- Search: zawsze włączone sortowanie wtórne po dacie dokumentu (`doc_date` DESC) jako tie‑break po score Etapu 1 (i ewentualnym rerankerze). Stosowane przed przycięciem listy kandydatów. Braki/„brak” traktowane jako najstarsze.
- FTS5: rozszerzony schemat o `doc_date` i `doc_date_ord` (UNINDEXED) oraz zapytania z grupowaniem po `doc_id` i sortowaniem po `MAX(doc_date_ord)` dla wydajnego `ORDER BY` po dacie.
- Migracja: przy starcie usługi, jeśli brakuje lokalnego indeksu FTS5 (lub jest pusty), zostanie on zbudowany od nowa (drop+create + zasilenie z Qdrant). Jeśli indeks istnieje, nie jest dotykany — zawsze możesz go przebudować ręcznie w Admin UI.
- Browse: `POST /browse/doc-ids` zwraca listę dokumentów posortowaną malejąco po `doc_date` (gdy `limit>0`). Od 2.41.0 dla `limit=0` zwracane jest zawsze dokładne `candidates_total`, a przy zawężeniach (query lub kinds) także próbka do 15 dokumentów.

## Nowości w 2.38.1
- Startup: wyeliminowano ostrzeżenie przy pre‑warm TF‑IDF (dopasowano wywołanie `prepare_tfidf`).
- FTS5: dodano okresowe logi postępu podczas przebudowy indeksu na starcie (np. „FTS rebuild: progress | rows=… pages=… elapsed=…s”), aby było jasne, dlaczego serwis chwilowo „wisi”.

## Nowości w 2.37.0
- Reranker: dodano twardy próg `RANKER_HARD_THRESHOLD` (domyślnie 0.65). Elementy poniżej tego progu nie są nigdy zwracane (brak dopełniania do K). Wciąż działa miękki próg `RANKER_SCORE_THRESHOLD` (domyślnie 0.9): najpierw wybierane są elementy ≥ soft‑progu, a jeśli brakuje — dopełniane są elementy z przedziału [HARD, SOFT).

## Nowości w 2.36.2
- Dokumentacja: doprecyzowano heurystyki `restrict_doc_ids` (minima: `top_m≥500`, `top_k≥50`, `per_doc_limit≥15`) – zgodnie z implementacją.

## Nowości w 2.36.0
- Reranker: synchronizacja limitów z parametrami żądania. `top_k` z requestu jest teraz respektowane (z limitem serwerowym). Zmieniono ustawienia na limity maksymalne:
  - `RERANK_TOP_N_MAX` (zamiast `RERANK_TOP_N`) — górny limit kandydatów dla rerankera,
  - `RETURN_TOP_K_MAX` (zamiast `RETURN_TOP_K`) — górny limit zwracanych bloków. 
  Backend tnie do `min(req.top_k, RETURN_TOP_K_MAX)`. Wciąż honorowane są stare zmienne środowiskowe jako fallback.

## Nowości w 2.35.0
- Domyślna strategia encji dla `/search/query`: `entity_strategy=optional` (miękki boost bez twardego filtra). Ułatwia to uzyskanie wyników, gdy encje bywają różnie zapisywane lub nie są kompletne w payloadzie.

## Nowości w 2.34.3
- Search: poprawka filtra encji w `/search/query` — dopasowanie `entities` działa teraz bezpiecznie względem wielkości liter (uwzględnia formy raw + casefold). Ułatwia to wyszukiwanie typu `entity_strategy=must_any` dla skrótów wielkimi literami.

## Nowości w 2.34.2
- Admin UI: dodano gotowy przykład „Search (restricted by doc_ids)” wywołujący `/search/query` z polem `restrict_doc_ids` i podniesionymi limitami.

## Nowości w 2.34.1
- Search: możliwość zawężenia wyszukiwania do zadanego podzbioru dokumentów poprzez `restrict_doc_ids` (lista `doc_id`).
  - Używaj TYLKO, gdy wcześniej pozyskałeś listę doc_id z `POST /browse/doc-ids` (np. po filtrowaniu/kindach/encjach).
  - W przeciwnym razie nie ustawiaj tego pola — przeszukiwanie obejmie cały korpus zgodnie z trybem (`mode`).
  - Wydajność: filtr wykonuje się pre‑search w Qdrant (indeks `keyword` na `doc_id`), działa szybko dla setek–tysięcy wartości.
  - Heurystyka serwera dla `restrict_doc_ids`: aby nie ucinać cytatów, backend automatycznie podnosi minimalne limity:
    - `top_m ≥ 500`,
    - `top_k ≥ 50`,
    - `per_doc_limit ≥ 15`.

Przykład (POST /search/query):

```
{
  "query": ["regulamin bezpieczeństwa danych"],
  "top_k": 5,
  "result_format": "blocks",
  "restrict_doc_ids": ["<doc_id_1>", "<doc_id_2>", "<doc_id_3>"]
}
```

## Nowości w 2.31.0
- Browse: dodano `text_match` — wymaganie literalnego dopasowania zapytania w treści chunków. Wartości: `none` (domyślnie), `phrase` (cała fraza jako substring), `any` (dowolny token), `all` (wszystkie tokeny). Działa w `POST /browse/doc-ids`.
- Opis: `entities` + `entity_strategy='optional'` łączy wyniki po treści ORAZ encjach; w strategiach ścisłych encje działają jak filtr (AND) już podczas wektorowego wyszukiwania.

## Nowości w 2.30.0
- Browse: jawny filtr aktywności dokumentów (`status`): `active` (domyślnie), `inactive`, `all`. Ma pierwszeństwo wobec heurystyki `mode`.
- Browse: encje — dodano strategię `optional` (logika OR): dokument wejdzie jako kandydat, jeśli pasuje po treści LUB ma wskazane encje. Dostępne strategie: `auto` (=must_any), `must_any`, `must_all`, `exclude`, `optional`.
- Browse: przypomnienie — zapytania domyślnie wyszukują po treści (wektory + TF‑IDF). Encje są używane tylko, gdy przekażesz `entities`/`entity_strategy`.

## Nowości w 2.29.1
- Browse: poprawka filtrów encji — dopasowanie obejmuje zarówno oryginalny zapis, jak i wersję casefold (w celu łagodzenia różnic wielkości liter w payloadach). Dodano też fallback „entities‑only” (scroll), gdy połączenie filtra encji i wyszukiwania wektorowego/TF‑IDF nie zwróci wystarczającej liczby kandydatów.

## Nowości w 2.29.0
- Browse: dodano filtry po encjach (entities) na poziomie treści (chunków).
  - Nowe pola w `BrowseQuery`: `entities` (lista stringów) oraz `entity_strategy` (`auto`=must_any, `must_any`, `must_all`, `exclude`).
  - Działa w: `POST /browse/doc-ids`.
- doc_kind: klasyfikacja rodzaju dokumentu oparta wyłącznie o tytuł (title). Sygnatury/keywords nie wpływają na rozpoznanie rodzaju.
- Admin UI: przykłady browse z `entities`.

## Nowości w 2.28.2
- Poprawka: inferencja `doc_kind` preferuje teraz dopasowanie w tytule, a dopiero potem sprawdza sygnaturę/keywords. Zapobiega to błędnym klasyfikacjom (np. „Regulamin …”/„Uchwała …” rozpoznane jako `order` z powodu słowa „zarządzenie” w sygnaturze).

## Nowości w 2.28.1
- Poprawka: w browse uzupełnianie metadanych (title/date/signature) dla kandydatów odbywa się także, gdy `doc_map` zawierało wcześniej tylko `is_active` (z chunków). Dzięki temu inferencja `doc_kind` korzysta z tytułu i nie zwraca błędnie `other`.

## Nowości w 2.28.0
- Browse: dodano inferencję rodzaju dokumentu (doc_kind) „w locie” z tytułów/sygnatur bez zmiany schematu Qdrant.
  - Filtrowanie po `kinds` (np. `order`, `resolution`, `regulation`) wspierane w:
    - `POST /browse/doc-ids` (odpowiedź zawiera pole `doc_kind` i `candidates_total`).
  - Obsługiwane identyfikatory: `resolution`, `order`, `announcement`, `notice`, `decision`, `regulation`, `policy`, `procedure`, `instruction`, `statute`, `other`.
- Admin UI: dodano operacje dla browse (`browse-doc-ids`).

Uwaga: filtrowanie `kinds` wykonywane jest post‑selekcyjnie (bez zmian w schemacie), więc nie wpływa na etap wyszukiwania po chunkach — zawęża wynik kandydatów i listę kandydatów.

## Nowości w 2.27.1
- Prompt (summary): doprecyzowano regułę ustalania `is_active` na podstawie `PATH` — sama obecność roku/daty w ścieżce nie oznacza archiwalności. `is_active=false` ustawiaj wyłącznie wtedy, gdy `PATH` zawiera jednoznaczne słowa‑klucze (np. `archiwum`, `archiwal`, `archive`, `archives`, `archival`, `old`, `stare`, `stary`, `history`, `deprecated`, `zarchiwizowane`).

## Nowości w 2.27.0
- Ingest: LLM ocenia pole `is_active` podczas streszczenia na podstawie ścieżki pliku (`PATH`). Jeśli ścieżka sugeruje archiwum (np. `archiwum/`, `archive/`, `stare/`, `old/`), model powinien zwrócić `is_active=false`; w przeciwnym razie `true`. Gdy model nie zwróci pola, przyjmujemy `true`.
- Sidecar: `is_active` jest zapisywane dodatkowo w pliku cache streszczenia. Starsze cache bez tego pola pozostają ważne — ich `is_active` domyślnie traktujemy jako `true`.
- Zasady istniejące (REPLACEMENT): dotychczasowy mechanizm ustawiania `is_active=false` dla dokumentów zastępowanych nadal działa i ma zastosowanie po ingestcie, niezależnie od powyższej oceny LLM.

- Browse po treści: `POST /browse/doc-ids` selekcjonuje kandydatów wyłącznie na podstawie treści (chunków), bez przeszukiwania streszczeń. Streszczenia mogą być użyte jedynie do wzbogacenia metadanych (tytuł/data) po selekcji.
- OpenAPI: ustalone `operation_id` dla endpointu browse (tag `tools`):
  - `POST /browse/doc-ids` → `rags_tool_browse_doc_ids`.
- Wyszukiwanie: przywrócono dotychczasowe zachowanie — `search` zwraca streszczenie raz na dokument (zgodnie z `summary_mode`), a snippety mają fallback do streszczenia tylko gdy brak tekstu chunku.
- Zasada podsumowań: funkcje podsumowujące nie tworzą „streszczeń streszczeń”. Dozwolone jest korzystanie z `entities` i `signatures`.

## Nowości w 2.25.0
- Domyślny zakres wyszukiwania dla `mode=auto` to teraz dokumenty obowiązujące (`current`). Tylko gdy kontekst wyraźnie wskazuje inaczej używamy `archival` (np. słowa „archiwalne”, „stara”, konkretne lata, „wersja z …”) lub `all` (np. „wszystkie”, „cała historia”).
- Opis narzędzia `/search/query` w OpenAPI doprecyzowany (RAG‑only, domyślny zakres, odsyłacz do `/browse/*` dla liczenia/listowania).
 
## Nowości w 2.24.0
- Wydzielenie funkcji przeglądowych (LLM‑friendly) od wyszukiwania odpowiedzi:
  - Endpointy browse:
    - `POST /browse/doc-ids` — lista `doc_id` z metadanymi (tytuł, `doc_date`, `is_active`, `doc_kind`) oraz pole `candidates_total` (liczba kandydatów po filtrach, niezależna od `limit`). Dla samej liczby ustaw `limit=0` i odczytaj `candidates_total` (nie stosuj sond `limit:1`). Gdy podano zawężenia treścią (query) lub `kinds`, odpowiedź może zawierać próbkę do 15 dokumentów (`approx=true` sygnalizuje próbkę niepełną).
    - `POST /browse/facets` — (usunięty w 2.43.0) proste rozkłady po kandydatach.
  - Wspólne helpery dostępu do magazynu przeniesione do `app/core/store_access.py`.
  - Bez zmian w istniejącym endpointzie `POST /search/query` — podział dotyczy struktury i nowych tras.

## Nowości w 2.13.0
- Ingest: deduplikacja identycznych plików po sumie kontrolnej SHA‑256 (bajtów pliku). Podczas budowania korpusu, jeżeli napotkany plik ma identyczną treść jak wcześniej przetworzony (w tym w poprzednich biegach), zostanie pominięty przed kosztowną obróbką (streszczenie/embedding). W logu INFO pojawi się komunikat:
  - `Duplicate skipped | sha256=<hash> existing=<ścieżka_już_zaindeksowana> duplicate=<bieżąca_ścieżka>`
  - W odpowiedzi endpointu `/ingest/build` zwracane jest pole `duplicates_skipped` z liczbą pominięć.
- Konfiguracja: `DEDUPE_ON_INGEST=true` (domyślnie włączone). Wyłączenie przywraca pełne przetwarzanie wszystkich plików.
- Qdrant: do payloadu streszczeń dodano `content_sha256` i indeks payload dla szybkiego sprawdzania duplikatów.

Uwaga: aby deduplikacja między biegami działała na istniejących kolekcjach, potrzebne jest ponowne zbudowanie indeksu (lub re‑ingest) tak, by dotychczasowe dokumenty otrzymały `content_sha256` w payloadzie.

## Nowości w 2.9.1
- Admin UI: widoczne, automatycznie generowane opisy dla operacji (funkcji) dostępnych w panelu. Sekcja dokumentacji per‑endpoint zawiera teraz listę parametrów, ich typy, wartości domyślne oraz — gdy dotyczy — dozwolone wartości (np. `auto|current|archival|all`, `flat|grouped|blocks`). Opisy są generowane dynamicznie na podstawie modeli Pydantic i metadanych endpointów FastAPI, dzięki czemu pozostają spójne z dokumentacją i nie wymagają duplikacji treści.
- Brak zmian w API — zmiana dotyczy wyłącznie warstwy UI/dokumentacji w panelu administracyjnym.

## Nowości w 2.9.0
- API: do wyników dodano pola `title`, `doc_date`, `is_active` we wszystkich formatach odpowiedzi (`flat`, `grouped`, `blocks`).
- API: usunięto pole `token_estimate` z bloków (`result_format=blocks`).
- Dokumentacja endpointu zaktualizowana (lista pól `blocks`).

## Nowości w 2.8.1
- Poprawka: `POST /search/query` z `result_format="flat"` lub `"grouped"` kończył się błędem 500 (`UnboundLocalError: blocks_payload`). Zmienna pomocnicza była niezainicjalizowana w gałęzi nie‑"blocks" funkcji kształtującej wynik. Naprawiono przez jednoznaczne ustawienie `blocks_payload = None` dla tych formatów. Bez zmian funkcjonalnych.

## Nowości w 2.7.2
- Dokumentacja: dodano lub uzupełniono docstringi/komentarze dla wszystkich funkcji i metod w modułach API, wyszukiwania, Qdrant oraz przetwarzania. Wszystkie docstringi są po angielsku (zgodnie z wytycznymi), bez zmian funkcjonalnych.

## Nowości w 2.7.1
- Poprawka: proces budowania z regeneracją streszczeń nie kończy się błędem 500 w przypadku błędnej konfiguracji endpointu LLM (np. odpowiedź 405 Not Allowed z reverse proxy/nginx). Dodano bezpieczny fallback lokalny w `llm_summary`, który tworzy minimalne streszczenie na podstawie treści, a w logach pojawia się czytelny komunikat diagnostyczny z `SUMMARY_API_URL`.

## Nowości w 2.7.0
- Chunki zapisują teraz kanoniczną ścieżkę sekcji `section_path` (z separatorem ` > `) oraz listę prefiksów `section_path_prefixes`. Pozwala to pobierać całe sekcje i ich podsekcje jednym filtrem w Qdrant bez scrollowania całego dokumentu.
- Podczas inicjalizacji kolekcji tworzone są indeksy payload (`doc_id`, `point_type`, `is_active`, `section_path`, `section_path_prefixes`) dla streszczeń i chunków, co znacząco skraca scalanie wyników.

### Dodanie indeksów do istniejących kolekcji
Jeżeli kolekcje zostały utworzone przed wersją 2.7.0, uruchom jednorazowo poniższy skrypt (w katalogu projektu, z poprawnie ustawionym `.env`), aby dołożyć brakujące indeksy:

```bash
python - <<'PY'
from app.qdrant_utils import qdrant
from app.settings import get_settings
from qdrant_client.http import models as qm

settings = get_settings()
collections = [
    settings.qdrant_summary_collection,
    settings.qdrant_content_collection,
]
index_specs = (
    ("doc_id", {"type": "keyword"}),
    ("point_type", {"type": "keyword"}),
    ("is_active", {"type": "bool"}),
    ("section_path", {"type": "keyword"}),
    ("section_path_prefixes", {"type": "keyword"}),
)

for coll in collections:
    for field, params in index_specs:
        try:
            qdrant.create_payload_index(
                collection_name=coll,
                field_name=field,
                field_schema=qm.PayloadIndexParams(**params),
            )
            print(f"[OK] {coll}: {field}")
        except Exception as exc:
            msg = str(exc).lower()
            if "already exists" in msg or "index with params conflicts" in msg or getattr(exc, "status_code", None) == 409:
                print(f"[SKIP] {coll}: {field} (już istnieje)")
            else:
                raise
PY
```

## Nowości w 2.6.0
- Scalanie bloków `result_format="blocks"` można teraz sprowadzić do wskazanego poziomu hierarchii (`SECTION_MERGE_LEVEL`, domyślnie `ust`). Wszystkie chunki z poziomu docelowego oraz jego potomków są łączone w jeden blok, co pozwala uzyskać pełne `ust.` wraz z `pkt`/`lit.` bez utraty kontekstu.
- Ingest zapisuje przy każdym chunku metadane `section_path` (kanoniczna ścieżka), dzięki czemu wyszukiwanie rekonstruuje sekcje szybciej i bez zgadywania struktury etykiet.


## Nowości w 2.2.1
- Poprawka: błąd inicjalizacji Admin UI (literówka `true` → `True` w specyfikacji operacji importu) uniemożliwiał start serwisu.

## Nowości w 2.3.0
- Globalny rerank po fuzji (RRF) i twarde cięcie do K, gdy reranker jest włączony:
  - Po zebraniu kandydatów z wielu zapytań i deduplikacji (po `(doc_id, section, chunk_id)`), wykonywany jest jeden globalny rerank wszystkich bloków.
  - Zwracane są najwyżej `min(top_k, RETURN_TOP_K_MAX)` bloków (cap konfigurowalny w `.env`). Parametry kapujące: `RERANK_TOP_N_MAX`, `RETURN_TOP_K_MAX`.
  - Jeżeli reranker jest wyłączony, zachowanie bez zmian: wynik jest ucinany do `top_k` z żądania.
  - Dzięki temu, niezależnie od liczby wariantów zapytania, klient otrzymuje „te kilka najlepszych” bloków.

## Nowości w 2.0.0

- Breaking: usunięto runtime scalanie chunków w odpowiedzi wyszukiwania. Usunięte parametry: `merge_chunks`, `merge_group_budget_tokens`, `max_merged_per_group`, `expand_neighbors`, `block_join_delimiter`.
- Format `blocks` zwracał pojedyncze chunki sekcyjne z ingestu (pojedynczy chunk = pojedynczy blok). Długość kontrolujesz przez `CHUNK_TOKENS`/`CHUNK_OVERLAP` i strategię `merge_up_to` w `chunk_text_by_sections`.

## Nowości w 2.4.0

- Przywrócono scalanie po sekcji dla `result_format="blocks"`:
  - Blok odpowiada pełnej sekcji dokumentu. Po fuzji RRF (na chunkach) dociągane są wszystkie chunki danej sekcji z Qdrant i łączone w jeden tekst.
  - Pola `first_chunk_id` i `last_chunk_id` reprezentują zakres id najmniejszego i największego chunku włączonego do bloku sekcji.
  - Reranker działa na zmergowanych sekcjach (po scaleniu), a nie na pojedynczych chunkach. Sortowanie końcowe odbywa się według `ranker_score` (gdy skonfigurowany) z respektowaniem `per_doc_limit`.
  - Dla wyników bez etykiety `section` stosowany jest fallback: blok budowany jest z trafionych chunków (bez doczytywania całej sekcji).
- Uporządkowano payload bloków: usunięto pola `ranker_applied` i `ranker_model`. Pole `ranker_score` pozostaje i jest ustawiane, gdy reranker jest aktywny.

## Nowości w 2.5.0

- Sidecar cache streszczeń i wektorów w katalogu `.summary/` obok plików źródłowych:
  - Dla każdego dokumentu zapisywany jest plik `.summary/<basename>_summary.json.gz` zawierający: `title`, `summary`, `signature`, `entities`, `replacement`, `doc_date` oraz wektor gęsty `summary_dense`.
  - Plik jest kompresowany gzip (duża oszczędność przy listach floatów).
  - Spójność weryfikowana jest przez `document.content_sha256` oraz `schema_version` (1.0.0). Przy rozbieżności cache jest ignorowany.
  - Podczas ingestu, gdy cache jest zgodny, pomijane są wywołania LLM oraz embedding streszczenia (znaczące skrócenie czasu i kosztu).
  - Wymuszenie przebudowy:
    - Użyj flagi `force_regen_summary: true` w `POST /ingest/build` (checkbox w Admin UI), aby pominąć cache i nadpisać pliki `.summary/*.json.gz`.
    - Alternatywnie usuń plik/katalog `.summary/` dla wybranych dokumentów lub zmień plik źródłowy (hash zmieni się automatycznie).
    - Uwaga: parametr `reindex` dotyczy Qdrant/TF‑IDF i nie czyści plików sidecar.

## Nowości w 2.5.2

- Debug logi dla cache sidecar:
  - Podczas ingestu raportowane jest: obecność pliku sidecar, decyzja o użyciu (hit) lub odrzuceniu (miss/stale), pominięcie z powodu `force_regen_summary` oraz zapis nowego pliku sidecar.

## Nowości w 2.5.3

- Uporządkowanie logów: nazwa pliku sidecar bez pełnej ścieżki (mniej hałasu w logach, łatwiejsza lektura).

## Nowości w 1.9.0

- Globalny przełącznik w `.env`: `SEARCH_SKIP_STAGE1_DEFAULT` (domyślnie `false`).
  - Gdy `true`, endpoint `/search/query` pomija selekcję dokumentów po streszczeniach (Etap 1) i od razu wyszukuje w całej kolekcji chunków.
  - Zachowane są wszystkie pozostałe mechanizmy: hybryda dense/TF‑IDF, MMR, `per_doc_limit`, `summary_mode`, `result_format` oraz (jeśli skonfigurowany) reranker.
  - Uwaga dla UI/testów: w tym trybie `top_m` ogranicza początkową pulę chunków (zamiast liczby dokumentów po Etapie 1). Testy, które oczekują wywołania Etapu 1, powinny uwzględnić nowy tryb.

## Nowości w 1.7.1

- Poprawka zgodności wejścia: pole `query` w `/search/query` akceptuje teraz również zagnieżdżone listy
  (np. `[["a","b","c"]]`). Wejście jest spłaszczane do `["a","b","c"]`, co eliminuje 422 przy takim formacie.

## Nowości w 1.7.0

- Reranker (OpenAI‑compatible): dodano opcjonalny krok rerankingu po wyszukiwaniu wektorowym.
- Minimalne zmienne w `.env`: `RANKER_BASE_URL`, `RANKER_API_KEY`, `RANKER_MODEL`,
    `RERANK_TOP_N_MAX`, `RETURN_TOP_K_MAX`, `RANKER_SCORE_THRESHOLD`, `RANKER_MAX_LENGTH`.
  - Integracja w endpointzie `/search/query`: jeżeli ranker jest włączony, wyniki w formacie `blocks`
    są sortowane i filtrowane wg `ranker_score` po wywołaniu `POST {RANKER_BASE_URL}/v1/rerank`.
  - Wielozapytaniowość: lista zapytań jest łączona w jeden ciąg (separator ` || `) na potrzeby rankera.
  - Fallback: na błąd/timeout lub brak konfiguracji rankera zwracane są wyniki wektorowe (bez `ranker_score`).

## Nowości w 1.6.0

- Embedding: dodano konfigurowalne prefiksy dla modeli retrievalujących (instruction-style). Nowe zmienne: `EMBEDDING_QUERY_PREFIX` i `EMBEDDING_PASSAGE_PREFIX`. Domyślne wartości odpowiadają modelowi sdadas/mmlw-retrieval-roberta-large-v2 (`"query: "` i `"passage: "`).
- API: zapytania (dense) embedowane są z prefiksem `query`, a streszczenia i treść dokumentów z prefiksem `passage`.
- Chunking: rozmiar chunku i overlap są konfigurowalne w `.env` (zmienne `CHUNK_TOKENS`, `CHUNK_OVERLAP`) i domyślnie dostosowane do modeli z limitem ~512 tokenów. Endpoint `/ingest/build` domyślnie korzysta z tych wartości, ale można je nadpisać w żądaniu.

## Nowości w 1.5.0

- Wyszukiwanie: pole `query` przyjmuje teraz listę zapytań (`List[str]`). Każde zapytanie jest wykonywane kolejno, a wyniki są łączone metodą RRF (Reciprocal Rank Fusion) i ograniczane globalnym `top_k`.
- Parametry łączenia są wewnętrzne (brak nowych pól w modelu) i mają sensowne domyślne wartości; domyślna strategia to `rrf`.


## Nowości w 1.4.2

- Import: czyszczenie `VECTOR_STORE_DIR` nie usuwa już samego katalogu (co kończyło się błędem „Device or resource busy” przy montażu jako wolumen Dockera). Czyścimy zawartość katalogu, a brakujące pliki są nadpisywane podczas importu.

## Nowości w 1.4.1

- Dodano zależność `python-multipart` wymaganą do obsługi importu pliku w endpointzie `/collections/import` (multipart/form-data). Obraz Dockera i instrukcja lokalnej instalacji zostały zaktualizowane.

## Nowości w 1.4.0

- Endpoint `/collections/import` przyjmuje teraz archiwum `.tar.gz` bezpośrednio jako przesyłany plik (multipart/form-data) albo surowe body HTTP, co upraszcza wykorzystanie plików wygenerowanych przez eksport.
- Panel Admin UI pozwala wskazać plik eksportu z dysku i wysłać go do API bez ręcznego kodowania base64; wciąż można użyć JSON-a z `archive_base64` dla automatyzacji.

## Nowości w 1.3.1

- Systemowy prompt używany przy generowaniu streszczeń (`SUMMARY_SYSTEM_PROMPT`) przeniesiono do konfiguracji `.env`, co pozwala sterować tonem i rolą modelu bez edycji kodu.

## Nowości w 1.3.0

- Prompty do streszczeń (`SUMMARY_PROMPT`, `SUMMARY_PROMPT_JSON`) są teraz konfigurowalne przez zmienne środowiskowe, dzięki czemu można łatwo dostosować instrukcje dla modeli LLM bez zmian w kodzie.

## Nowości w 1.2.1

- Poprawiono eksport/import snapshotów: jeśli klient Python nie udostępnia metod snapshot, używamy bezpośrednich endpointów REST (tworzenie, pobieranie, upload). Gdy snapshoty nie są dostępne, eksport automatycznie przełącza się na tryb JSONL.

## Nowości w 1.2.0

- Eksport kolekcji wykorzystuje natywne snapshoty Qdrant; archiwum zawiera pliki snapshotów (`snapshots/<kolekcja>/<plik>.snapshot`) gotowe do ponownego wgrania.
- Import odtwarza kolekcje przez upload i recovery snapshotów (z zachowaniem opcji `replace_existing`).
- Zachowana kompatybilność wstecz: archiwa 1.1.x oparte na JSON-ach są nadal obsługiwane.
- W odpowiedzi eksportu pojawia się nagłówek `X-Rags-Snapshots` z listą wygenerowanych plików snapshotów.

## Nowości w 1.1.2

- Naprawiono eksport punktów Qdrant: stronicowanie wykorzystuje teraz iterator odporny na różnice w typie zwracanym przez `qdrant.scroll`, dzięki czemu pliki `points.jsonl` zawierają pełną zawartość nawet dla dużych kolekcji.

## Nowości w 1.1.1

- Eksport kolekcji działa strumieniowo (plik `points.jsonl` per kolekcja) i nie buforuje już całej zawartości w pamięci, dzięki czemu obsługuje duże zbiory.
- Import wspiera zarówno nowe archiwum `.tar.gz`, jak i format z wersji 1.1.0; pliki TF-IDF są odtwarzane po stronie serwera, a istniejące zasoby mogą zostać zachowane lub nadpisane.
- W odpowiedzi eksportu dodano nagłówek `X-Rags-Vector-Store` z listą plików TF-IDF do szybkiej inspekcji archiwum.

## Nowości w 1.1.0

- Eksport zawsze obejmuje wszystkie kolekcje Qdrant oraz artefakty TF-IDF z katalogu `VECTOR_STORE_DIR`; dane trafiają do archiwum `.tar.gz` kompatybilnego z nowym importem.
- Import odtwarza kolekcje i pliki indeksów, opcjonalnie zastępując istniejące zasoby (w tym katalog TF-IDF) po ustawieniu `replace_existing=true`.
- Panel Admin UI aktualizuje helper eksportu/importu do nowego formatu (archiwum `.tar.gz`).

## Nowości w 1.0.0

- Panel Admin UI otrzymał dwa helpery: eksport wszystkich kolekcji Qdrant do pojedynczego archiwum JSON.gz (`/collections/export`) oraz import plików wygenerowanych w ten sposób (`/collections/import`) z opcją zastąpienia istniejących kolekcji.
- UI automatycznie pobiera plik eksportu (bezpośredni download z przeglądarki) i przyjmuje dane importu w formacie base64.

## Nowości w 0.9.5

- Naprawiono błąd `Unknown arguments: ['timeout']` podczas `upsert` w klientach Qdrant bez wsparcia parametru per-zapytanie; limit czasu ustawiany jest teraz wyłącznie globalnie.

## Nowości w 0.9.4

- Rozbito upserty do Qdrant na mniejsze batch-e (256 punktów) i dodano ustawialny timeout, co zapobiega błędom `ResponseHandlingException: timed out` przy dużych dokumentach.
- Nowa zmienna środowiskowa `QDRANT_TIMEOUT` (domyślnie 60 s) pozwala dostosować limit czasu na operacje HTTP do konfiguracji klastra.

## Nowości w 0.9.3

- Ujednolicono etykiety sekcji generowane przez spaCy (np. § 5 ust. 3 pkt 2), dzięki czemu payload Qdrant zachowuje pełną hierarchię.
- Uproszczony fallback chunk_text_by_sections gwarantuje pary {text, section_path} także bez spaCy, co zabezpiecza ingest.

## Nowości w 0.9.2

- Nowe narzędzie CLI: wydobywanie wzorców sekcjonowania z korpusu (`tools/mine_section_patterns.py`).
  - Skanowanie plików zgodne z ingest/scan (`SUPPORTED_EXT`, ten sam parser `extract_text`).
  - Heurystyka + fallback do LLM z użyciem tego samego modelu co streszczenia (`SUMMARY_*`).
  - Wynikiem jest snippet Pythona z `SECTION_LEVELS` i `LEVEL_PATTERNS` (unia poziomów z całego korpusu).

## Nowości w 0.9.1

- Refaktor: podział kodu na logiczne moduły (`app/api.py`, `app/core/*`, `app/models.py`, `app/qdrant_utils.py`).
- Zmieniony sposób uruchomienia: `uvicorn main:app` (wcześniej `uvicorn app:app`).
- README i Dockerfile dostosowane do nowej struktury.

## Nowości w 0.9.0

- Sekcjony podział punktów i podpunktów w obrębie paragrafów (§): wykrywamy listy numerowane (`1)`, `2.`), literowe (`a)`, `lit. b)`), rzymskie (`i)`, `IV)`), a także tirety (`-`, `–`, `•`). Każdy wykryty element staje się osobną podsekcją z etykietą np. „§ 7 pkt 3 lit. b)”, która trafia do payloadu jako `section_path`.
- Heurystyki unikające szumu: segmentujemy tylko gdy poziom ma co najmniej 2 elementy i elementy mają sensowną długość; krótkie i pojedyncze pozycje pozostają w rodzicu.
- Integracja z dotychczasowym chunkingiem: podsekcje są dalej dzielone tokenowo, a grupowanie `blocks` i `grouped` zyskuje bardziej precyzyjne granice.

## Nowości w 0.8.0

- Sekcyjny chunking dla dokumentów regulaminowych i prawnych: parser rozpoznaje nagłówki „Rozdział …”, paragrafy „§ …”, a także bloki „Załącznik …”. Tekst jest dzielony w granicach sekcji i paragrafów, bez ich przecinania.
- Payloady Qdrant zawierają teraz pole `section_path` dla każdego chunku (np. „Rozdział 1 > § 1”), co poprawia prezentację wyników (`blocks`, `grouped`).
- Lepsze cytowanie: odpowiedzi zawierają `section` (oparte o `section_path`), co ułatwia odwołania do konkretnych fragmentów dokumentu.

## Nowości w 0.7.2

- Streszczenia w JSON: funkcja streszczeń preferuje teraz tryb JSON (`response_format={"type":"json_object"}`) i oczekuje kluczy `summary`, `signature`, `entities`. Jeśli serwer nie wspiera JSON‑mode, automatycznie używany jest dotychczasowy parser tekstowy. Przełącznik: `SUMMARY_JSON_MODE` (domyślnie `true`).

## Nowości w 0.7.1

- Refaktor: dalszy podział długich funkcji — osobne helpery dla klasyfikacji trybu, kształtowania odpowiedzi i skanowania plików; `ingest_scan` oraz `ingest_build` respektują teraz flagę `recursive` przy wyszukiwaniu plików.

## Nowości w 0.7.0

- Wymiar embeddingu z konfiguracji: kolekcja Qdrant jest tworzona na podstawie `EMBEDDING_DIM` (zamiast wykonywać zapytanie do API, by odczytać wymiar). Zmniejsza to koszty i przyspiesza start. Dla zgodności nadal można wymusić sondowanie ustawiając `force_dim_probe=true` w `POST /collections/init`.
- Poprawka startu: uniknięto błędu `NameError: SearchQuery` poprzez użycie forward refs w typach helperów.

## Nowości w 0.6.2

- Refaktor: podział długich funkcji (`ingest_build`, `search_query`) na mniejsze helpery ułatwiające utrzymanie i testowanie.
- Panel Admin UI: szablon przeniesiony do pliku `templates/admin.html` (łatwiejsza edycja). Serwer wczytuje szablon z pliku i wstrzykuje operacje przez prostą podmianę tokenu `__OPERATIONS__`.

## Nowości w 0.6.1

- Dokładne chunkowanie po tokenach: `chunk_text` wykorzystuje teraz `tiktoken` do liczenia tokenów, dzięki czemu fragmenty lepiej mieszczą się w budżecie modeli i są bardziej spójne (mniej zbyt małych/zbyt dużych chunków). Jeśli `tiktoken` nie jest dostępny, stosowany jest bezpieczny fallback heurystyczny (~4 znaki/token). Zmiana wpływa tylko na sposób wyznaczania granic chunków — API pozostaje bez zmian.

## Nowości w 0.6.0

- (Zastąpione w 2.0.0).

## Nowości w 0.5.3

- Stabilne OpenAPI dla narzędzi: dodano `operation_id` (`rags_tool_search`) i `tags: ["tools"]` dla `/search/query`, aby uniknąć problemów z importerami narzędzi (np. OpenWebUI).

## Nowości w 0.5.2

- Rozszerzone opisy w OpenAPI dla narzędzia LLM (parametry `SearchQuery`, pola odpowiedzi i opis endpointu `/search/query`).

## Nowości w 0.5.1

- Poprawka uruchomienia Pydantic: przesunięto `model_rebuild()` po definicji `MergedBlock` (naprawia błąd `PydanticUndefinedAnnotation`).

## Nowości w 0.5.0

- Dodano format wyników `blocks` (zwraca bloki treści przygotowane do cytowania przez narzędzia). W 2.0.0 bloki odpowiadają pojedynczym chunkom sekcyjnym z ingestu (bez runtime scalania).

## Nowości w 0.4.0

- Kontrola duplikacji streszczeń w wynikach: nowy parametr `summary_mode` (`none` | `first` | `all`). Domyślnie `first` — streszczenie dokumentu pojawia się tylko przy pierwszym trafieniu z danego dokumentu (eliminuje powtarzanie).
- Nowy format wyników: `result_format` (`flat` | `grouped`). Domyślnie `flat`. W trybie `grouped` wyniki są grupowane per dokument (jedno streszczenie na dokument + lista trafionych fragmentów).
- Zaktualizowany panel Admin UI ma domyślnie `summary_mode: "first"` i `result_format: "blocks"` w predefiniowanym żądaniu `search-query`.

## Nowości w 0.3.0

- MMR i ranking liczone w przestrzeni hybrydowej (dense + TF‑IDF) z normalizacją wyników; końcowe sortowanie po score hybrydowym.
- Limit per‑doc w Etapie 2 (domyślnie 2) przeciwdziała dominacji jednego dokumentu.
- Opcje normalizacji: `minmax` (domyślna), `zscore`, `none` — stabilizują wagi dense/sparse.
- Opcjonalny MMR na Etapie 1 (streszczenia) z re‑pulsywnością hybrydową.
- Nowe parametry zapytania: `per_doc_limit`, `score_norm`, `rep_alpha`, `mmr_stage1` (szczegóły niżej).

## Wymagania

- Python 3.11+
- Zewnętrzna instancja Qdrant (SaaS lub self-hosted)
- Endpoint zgodny z protokołem OpenAI zapewniający embeddingi i model konwersacyjny (np. vLLM, Ollama, OpenAI)

## Konfiguracja środowiska

Ustaw wymagane zmienne środowiskowe przed uruchomieniem aplikacji:

```bash
export QDRANT_URL="http://127.0.0.1:6333"
export QDRANT_API_KEY=""
export QDRANT_TIMEOUT="60"
export EMBEDDING_API_URL="http://127.0.0.1:8000/v1"
export EMBEDDING_API_KEY="sk-embed-xxx"
export EMBEDDING_MODEL="BAAI/bge-m3"
export EMBEDDING_TOKENIZER="tiktoken:cl100k_base"   # lub hf:<model> dla tokenizerów Hugging Face
export EMBEDDING_DIM="1024"          # stały wymiar wektora dla używanego modelu
 
export SUMMARY_API_URL="http://127.0.0.1:8001/v1"
export SUMMARY_API_KEY="sk-summary-xxx"
export SUMMARY_MODEL="gpt-4o-mini"
export SUMMARY_SYSTEM_PROMPT="Jesteś zwięzłym ekstrakcyjnym streszczaczem."  # opcjonalna rola systemowa
export SUMMARY_PROMPT="..."         # opcjonalnie nadpisz instrukcję tekstowego promptu
export SUMMARY_PROMPT_JSON="..."     # opcjonalnie nadpisz instrukcję promptu JSON
export SEARCH_TOOL_DESCRIPTION="..." # opcjonalny opis endpointu /search/query widoczny w OpenAPI i narzędziu LLM
export COLLECTION_NAME="rags_tool"
export VECTOR_STORE_DIR=".rags_tool_store"
export DEBUG="false"  # ustaw na "true", aby włączyć logi debugujące

# Reranker (OpenAI‑compatible). Pozostaw puste, aby wyłączyć.
# Ustaw bazę BEZ '/v1' — klient sam dołączy '/v1/rerank' (lub '/rerank', jeśli baza kończy się na '/v1').
export RANKER_BASE_URL="http://127.0.0.1:8002"
export RANKER_API_KEY="sk-ranker-xxx"
export RANKER_MODEL="sdadas/polish-reranker-roberta-v3"
export RERANK_TOP_N_MAX="50"       # maks. ilu kandydatów przekazać do rankera
export RETURN_TOP_K_MAX="50"       # maks. ilu wyników zwrócić po rankingu
export RANKER_SCORE_THRESHOLD="0.2"# próg minimalnego score
export RANKER_MAX_LENGTH="2048"    # przybliżony limit znaków na jeden passage
```

Możesz także umieścić te wartości w pliku `.env`; aplikacja wczyta je automatycznie dzięki `pydantic-settings`. W repo znajdziesz przykładowy plik `.env`, który możesz skopiować i dostosować. Flaga `DEBUG=true` włącza szczegółowe logi z przebiegu ingestu (po jednym wpisie na dokument).

## Uruchomienie lokalne

1. Utwórz i aktywuj wirtualne środowisko.
2. Zainstaluj zależności:
   ```bash
   pip install fastapi uvicorn qdrant-client openai pydantic pydantic-settings python-multipart tiktoken transformers scikit-learn markdown2 beautifulsoup4 html2text PyPDF2
   ```
3. Start serwera:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8080
   ```
4. Dokumentacja API jest dostępna pod `/docs` (Swagger UI) oraz `/openapi.json`.

## Struktura projektu

```
.
├── app/
│   ├── __init__.py
│   ├── api.py              # endpointy FastAPI
│   ├── core/
│   │   ├── __init__.py
│   │   ├── chunking.py     # chunk_text, count_tokens, segmentacja sekcji
│   │   ├── embedding.py    # embed_text, TF-IDF (load/save/fit/vector)
│   │   ├── parsing.py      # extract_text, html_to_text, split_into_paragraphs
│   │   ├── search.py       # _stage1, _stage2, MMR, shaping
│   │   └── summary.py      # llm_summary i prompty
│   ├── models.py           # modele Pydantic (Request/Response)
│   ├── qdrant_utils.py     # ensure_collections, upsert punktów, klient Qdrant
│   └── settings.py         # konfiguracja aplikacji
├── templates/
│   └── admin.html
├── .env
├── main.py                 # wejście startowe (uvicorn main:app)
└── Dockerfile
```

## Uruchomienie w Dockerze

1. Zbuduj obraz:
   ```bash
   docker build -t rags_tool:latest .
   ```
2. Uruchom kontener, przekazując wymagane zmienne środowiskowe:
   ```bash
   docker run --rm -p 8080:8080 \
     --env-file .env \
    -v $(pwd)/.rags_tool_store:/app/.rags_tool_store \
    rags_tool:latest
   ```

> **Uwaga**: Qdrant i modele LLM muszą działać poza kontenerem i być dostępne pod adresami przekazanymi w zmiennych środowiskowych.

## Parametry LLM (OpenAI‑compatible)

Serwis korzysta z dwóch endpointów zgodnych z protokołem OpenAI:

- Embedding API — do wektoryzacji treści i streszczeń
  - `EMBEDDING_API_URL` (np. `http://127.0.0.1:8000/v1`)
  - `EMBEDDING_API_KEY` (token; może być pusty jeśli serwer nie wymaga)
  - `EMBEDDING_MODEL` (np. `BAAI/bge-m3` lub inny zgodny z `/v1/embeddings`)
  - `EMBEDDING_QUERY_PREFIX` (prefiks dla zapytań; domyślnie `"query: "`)
  - `EMBEDDING_PASSAGE_PREFIX` (prefiks dla dokumentów/fragmentów; domyślnie `"passage: "`)
  - `CHUNK_TOKENS` (domyślny docelowy rozmiar chunku w tokenach)
  - `CHUNK_OVERLAP` (domyślny overlap chunków w tokenach)
  - `SECTION_MERGE_LEVEL` (poziom sekcji używany przy scalaniu bloków; np. `ust`, `pkt`, `lit`)
  - Wymagania: endpoint `/v1/embeddings` przyjmuje `{"model": str, "input": List[str]}` i zwraca `{"data": [{"embedding": List[float]}, ...]}`.

- Summary (Chat) API — do generowania streszczeń dokumentów
  - `SUMMARY_API_URL` (np. `http://127.0.0.1:8001/v1`)
  - `SUMMARY_API_KEY`
  - `SUMMARY_MODEL` (np. `gpt-4o-mini` lub kompatybilny model czatowy)
  - Wymagania: endpoint `/v1/chat/completions` przyjmuje `{"model": str, "messages": [{role, content}, ...], "temperature": float, "max_tokens": int}` i zwraca `{"choices": [{"message": {"content": str}}]}`.

## Konfiguracja wyszukiwania

- `SEARCH_SKIP_STAGE1_DEFAULT` (bool; domyślnie `false`)
  - Jeśli `true`, Etap 1 (wyszukiwanie po streszczeniach) jest globalnie wyłączony. `/search/query` wyszukuje bezpośrednio w całej kolekcji chunków, respektując filtr trybu (`current`/`archival`), hybrydę dense/TF‑IDF, MMR, limity per‑doc, `summary_mode`, formatowanie wyników oraz ewentualny reranker.
  - W tym trybie `top_m` działa jako limit początkowej puli chunków do rozważenia w Etapie 2.

### Jak działają wywołania

- Embedding:
  - Aplikacja odpytuje `/v1/embeddings` batchowo (`input: List[str]`).
  - Dla modeli oczekujących instrukcji (np. sdadas/mmlw-retrieval-roberta-large-v2) dołącza prefiks `EMBEDDING_QUERY_PREFIX` dla zapytań oraz `EMBEDDING_PASSAGE_PREFIX` dla streszczeń i treści.
  - Rozmiar chunków kontrolują `CHUNK_TOKENS` i `CHUNK_OVERLAP`. Dla modeli z limitem ~512 zalecane wartości startowe to `400` i `64`; dla modeli 1k–2k można rozważyć większe okna (np. `900`/`150`).
  - Wymiar wektora (dim) wykrywany jest sondą z tekstem `"test"` – musi być stały między wywołaniami; zmiana modelu = zmiana wymiaru.
  - W przypadku zmiany modelu embedującego warto użyć `POST /collections/init` z `force_dim_probe=true` i/lub przebudować indeks.

- Streszczenia:
  - Aplikacja woła `/v1/chat/completions` z `temperature=0.0` i `max_tokens=300`.
  - Wysyłany jest polski prompt proszący o format:
    - `SUMMARY: ...`
    - `SIGNATURE: lemma1, lemma2, ...`
    - `ENTITIES: ...`
  - Parser oczekuje powyższych prefiksów linii; jeśli model ich nie zwróci, używa fallbacku (pierwsze ~600 znaków odpowiedzi jako `summary`).

### Przykłady cURL

- Embeddings:

```bash
curl -sS "$EMBEDDING_API_URL/embeddings" \
  -H "Authorization: Bearer $EMBEDDING_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "'"${EMBEDDING_MODEL}"'", "input": ["hello", "world"]}'
```

- Chat (streszczenia):

```bash
curl -sS "$SUMMARY_API_URL/chat/completions" \
  -H "Authorization: Bearer $SUMMARY_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"${SUMMARY_MODEL}"'",
    "temperature": 0.0,
    "max_tokens": 300,
    "messages": [
      {"role": "system", "content": "Jesteś zwięzłym ekstrakcyjnym streszczaczem."},
      {"role": "user", "content": "SUMMARY: ...\nSIGNATURE: lemma1, lemma2, ...\nENTITIES: ...\n\nTEKST:\nTo jest przykładowy tekst do streszczenia."}
    ]
  }'
```

### Dobór modeli i wskazówki

- Embedding:
  - `BAAI/bge-m3` — uniwersalny, wielojęzyczny, dobry baseline (COSINE).
  - Inne modele są ok, jeśli zwracają spójną długość wektora i wspierają `/v1/embeddings`.
- Streszczenia:
  - Model czatowy musi wspierać `/v1/chat/completions` i generację w języku polskim.
  - Ustaw `temperature=0.0` dla deterministycznych, ekstrakcyjnych streszczeń.

### Rozwiązywanie problemów

- „Api key is used with an insecure connection.” — komunikat ostrzegawczy z klienta Qdrant przy użyciu HTTP z kluczem API. Najlepiej przejść na HTTPS (`QDRANT_URL`).
- „Dimension mismatch” po zmianie modelu embedującego — uruchom `POST /collections/init` z `force_dim_probe=true` i/lub przebuduj indeks (`/ingest/build`).
- Model czatowy nie zwraca prefiksów `SUMMARY:`/`SIGNATURE:`/`ENTITIES:` — sprawdź, czy endpoint respektuje wiadomość systemową i format prośby; w razie czego parser użyje fallbacku.
- Qdrant 500 „Service internal error: No such file or directory (os error 2)” podczas `upsert` — aplikacja próbuje automatycznie odzyskać działanie: ponawia po `ensure_collections`, dzieli batch na mniejsze (64), a jeśli to nie pomaga, wysyła punkty bez składników TF‑IDF (tylko dense). Jeśli błąd się utrzymuje, sprawdź uprawnienia/zapisywalność storage Qdrant oraz wersję serwera (zalecana 1.7+ z obsługą named sparse vectors).

## Narzędzie: wzorce sekcji

Narzędzie CLI do wydobywania poziomów sekcji i bezpiecznych regexów na podstawie całego korpusu.

- Plik: `tools/mine_section_patterns.py`
- Skanuje te same formaty co ingest (`SUPPORTED_EXT`), używa identycznego parsera treści (`extract_text`).
- Heurystyka (szybka) + LLM fallback (ten sam model co streszczenia: `SUMMARY_API_URL`, `SUMMARY_API_KEY`, `SUMMARY_MODEL`).

Przykład użycia:

```bash
python tools/mine_section_patterns.py /app/data \
  --glob "**/*" \
  --recursive \
  --out section_patterns.py
```

Przykładowy wynik (snippet Pythona):

```python
SECTION_LEVELS = ["doc", "chapter", "par", "ust", "pkt", "lit", "dash"]
LEVEL_PATTERNS = {
  "chapter": r"^\s*(rozdział|dział)\s+([IVXLCDM]+|\d+)",
  "par":     r"^\s*§\s*(\d+[a-z]?)",
  "ust":     r"\bust\.?\s*(\d+[a-z]?)\b",
  "pkt":     r"(\bpkt\.?\s*(\d+[a-z]?)|^\s*\(?\d+[a-z]?\))",
  "lit":     r"\blit\.?\s*([a-z])\)",
  "dash":    r"^\s*[-–—]\s+",
}
```

Wartości `SECTION_LEVELS` to unia poziomów znalezionych we wszystkich dokumentach; `doc` jest zawsze dołączany na górze hierarchii. `LEVEL_PATTERNS` zawiera tylko te poziomy, dla których wykryto wzorce.

## Najważniejsze endpointy

- `GET /about` – metadane usługi
- `GET /health` – sprawdzenie dostępności Qdrant
- `POST /collections/init` – utworzenie/aktualizacja kolekcji w Qdrant
- `POST /ingest/scan` – skanowanie katalogu z dokumentami
- `POST /ingest/build` – pełny pipeline ingestu (streszczenia + embedding + indeks)
- `POST /summaries/generate` – generowanie streszczeń pojedynczych plików
- `POST /search/query` – zapytania dwustopniowe z hybrydowym rankingiem

### Parametry wyszukiwania (`POST /search/query`)

To narzędzie służy wyłącznie do wyszukiwania RAG i zwracania bloków dowodowych (`blocks`) do cytowania. Nie używaj go do liczenia ani listowania dokumentów — do tego służą:

<!-- count endpoint usunięty -->
- `POST /browse/doc-ids` — lista `doc_id` + `title` + `doc_date` + `is_active`,
<!-- Endpoint `POST /browse/facets` został usunięty w 2.43.0. Używaj `/browse/doc-ids` i agreguj po stronie klienta. -->

Najważniejsze pola `POST /search/query`:

- `query`: lista krótkich wariantów (3–12 słów) — tytuły/sygnatury/datacje/hasła; warianty zwiększają recall; wyniki łączone są globalnie.
- `top_m`: liczba kandydatów po Etapie 1 (streszczenia).
- `top_k`: liczba końcowych bloków (Etap 2). Zalecane 5–10.
- `use_hybrid`: hybryda dense+TF‑IDF (domyślnie true); `dense_weight`/`sparse_weight` ustawiają proporcje.
- `mmr_lambda`, `per_doc_limit`, `score_norm`, `rep_alpha`, `mmr_stage1`: kontrola dywersyfikacji i fuzji.
- `summary_mode`: `none` | `first` | `all` — steruje duplikowaniem streszczeń w wynikach.
- `result_format`: `blocks` (zalecane) | `grouped` | `flat`.

Przykładowe zapytanie (flat, bez duplikacji streszczeń):

```json
{
  "query": ["Jak działa rags_tool?", "architektura rags_tool"],
  "top_m": 10,
  "top_k": 5,
  "mode": "auto",
  "use_hybrid": true,
  "dense_weight": 0.6,
  "sparse_weight": 0.4,
  "mmr_lambda": 0.3,
  "per_doc_limit": 2,
  "score_norm": "minmax",
  "rep_alpha": 0.6,
  "mmr_stage1": true,
  "summary_mode": "first",
  "result_format": "flat"
}
```

Przykładowe zapytanie (grouped):

```json
{
  "query": ["Jak działa rags_tool?"],
  "top_m": 10,
  "top_k": 5,
  "result_format": "grouped",
  "summary_mode": "first"
}
```

Przykładowe zapytanie (blocks):

```json
{
  "query": ["Jak działa rags_tool?", "architektura rags_tool"],
  "top_m": 10,
  "top_k": 5,
  "result_format": "blocks",
  "summary_mode": "first"
}
```

## Licencja

MIT
## Nowości w 2.23.0
- Entity‑aware search: nowe pola w zapytaniu `POST /search/query` sterowane przez LLM:
  - `entities` (opcjonalnie): lista encji (nazwy/ID/lata/cytowane frazy).
  - `entity_strategy`: `auto|boost|must_any|must_all|exclude`.
    - `auto/boost` – miękki bonus do rankingów (Stage‑1 i Stage‑2), bez ryzyka 0 wyników.
    - `must_any/must_all` – twardy filtr po encjach na Etapie 1 (i Etapie 2, gdy Stage‑1 pominięty).
    - `exclude` – wykluczenie dokumentów/chunków z tymi encjami.
- Ustawienia w `.env` (szyny, poza kontrolą LLM):
  - `ENTITY_BOOST_STAGE1=0.15`, `ENTITY_BOOST_STAGE2=0.10` – siła bonusu encji.
  - `AUTO_EXTRACT_QUERY_ENTITIES=true` – heurystyczne wydobycie encji z zapytania, gdy `entities` pominięte.
- Ingest: encje dokumentu są replikowane do payloadu chunków, aby filtry encji działały także przy `SEARCH_SKIP_STAGE1_DEFAULT=true`.
- Dokumentacja OpenAPI uzupełniona o opis pól `entities` i `entity_strategy` oraz sekcję „DocList” bez zmian w dotychczasowych polach.
## Nowości w 2.20.2
- OpenAPI dla narzędzi: ukryto pozostałe funkcje narzędziowe w specyfikacji (`include_in_schema=false`) tak, aby importer LLM widział wyłącznie `POST /search/query` (operation_id `rags_tool_search`). Brak zmian w działaniu endpointów — nadal dostępne HTTP, ale niewidoczne w OpenAPI.

## Nowości w 2.20.1
- Dokumentacja narzędzia: dodano opis dwóch intencji dla `/search/query` (evidence i doc_list), heurystyki wykrywania (słowa kluczowe) oraz preset „DocList” z zalecanymi parametrami. Zaktualizowano opis w OpenAPI (widoczny dla narzędzia LLM). Bez zmian w API.
## Nowości w 1.9.1

- Refaktor: wydzielono kod panelu `/admin` do oddzielnego modułu `app/admin_routes.py` i podpinany jest przez `attach_admin_routes(app)`. Kod funkcjonalny pozostaje w `app/api.py`.
## Nowości w 2.8.0
- Streszczenia zawierają teraz pole `doc_date` (data wprowadzenia/ogłoszenia dokumentu). Pole jest:
  - wydobywane przez model (prompt uzupełniony o instrukcję),
  - zapisywane w cache `.summary/*.json.gz`,
  - dodawane do payloadu Qdrant dla punktów typu `summary`,
  - włączane do wektoryzacji TF‑IDF streszczeń (lepsze dopasowania zapytań z datami).
- Nowy indeks payload w Qdrant: `doc_date` (typ `keyword`). Tworzony automatycznie przy inicjalizacji kolekcji.

### Dodanie indeksu `doc_date` do istniejących kolekcji
Jeśli kolekcje powstały przed 2.8.0, możesz dodać brakujący indeks jednym skryptem:

```bash
python - <<'PY'
from app.qdrant_utils import qdrant
from app.settings import get_settings
from qdrant_client.http import models as qm

settings = get_settings()
for coll in (settings.qdrant_summary_collection, settings.qdrant_content_collection):
    try:
        qdrant.create_payload_index(
            collection_name=coll,
            field_name="doc_date",
            field_schema=qm.PayloadIndexParams(type="keyword"),
        )
        print(f"[OK] {coll}: doc_date")
    except Exception as exc:
        msg = str(exc).lower()
        if "already exists" in msg or getattr(exc, "status_code", None) == 409:
            print(f"[SKIP] {coll}: doc_date (już istnieje)")
        else:
            raise
PY
```
## Nowości w 2.22.0
- ENTITIES: utrwalanie i indeksowanie
  - Streszczenia zapisują teraz pole `entities` tak samo jak `signature`:
    - w cache sidecar `.summary/<basename>_summary.json.gz` (sekcja `summary.entities` jako lista stringów),
    - w payloadzie punktów typu `summary` w Qdrant (`entities: [string, ...]`).
  - Tworzony jest indeks payload `entities` (typ `keyword`) w obu kolekcjach, aby przyspieszyć filtrowanie/przeglądanie po nazwach i identyfikatorach (field condition `MatchAny`).
  - `/search/query` dołącza `entities` do minimalnego payloadu dla Etapu 1 i propaguje je do wyników (flat/grouped/blocks) — tak jak `signature`.
  - W JSON‑mode prompt streszczeń oczekuje teraz `entities` jako listy stringów.
  - Zgodność wsteczna: istniejące pliki sidecar bez `entities` pozostają ważne; pole pojawi się po kolejnej regeneracji streszczeń.
## Nowości w 2.33.0
- /browse/doc-ids: uproszczone parametry (query, match=phrase|any|all, status=active|inactive|all, kinds) i pełny korpus dzięki lokalnemu indeksowi FTS (SQLite FTS5). Zwraca `candidates_total` przed przycięciem do `limit`.
- Indeks FTS buduje się automatycznie przy pierwszym użyciu (źródło: Qdrant, kolekcja chunków). Plik: `<VECTOR_STORE_DIR>/chunks_fts.sqlite`.
 - Nowy endpoint: `GET /docs/stats` — szybka statystyka liczby dokumentów w korpusie (aktywnych/nieaktywnych/łącznie) na podstawie FTS (distinct doc_id).

## Nowości w 2.32.1
- doc_kind: wzorce wyrażeń regularnych są kotwiczone na początku tytułu (^) — tytuły zawsze zaczynają się słowem określającym rodzaj. Zmniejsza to fałszywe dopasowania.
## Nowości w 2.39.0
- Streszczenia: dodano pole `subtitle` (podtytuł) generowane przez LLM.
  - JSON: nowy klucz `'subtitle'` (string; krótki jednoznaczny podtytuł, max 100 znaków, zawsze w mianowniku; jeśli nie można ustalić, wpisz dokładnie `'brak'`).
  - Tryb tekstowy: nowa sekcja `SUBTITLE: ...` zaraz pod `TITLE:`.
  - Sidecar (`.summary/*.json.gz`): zapis/odczyt `subtitle`.
- Qdrant (kolekcja streszczeń): zapis pól `subtitle`, `subtitle_norm` (znormalizowany, do porównań), `doc_date_ord` (YYYYMMDD), `ingested_ts` (unix epoch). Dodano indeksy payload dla tych pól.
- Ingest: wykrywanie konfliktów podtytułów — jeżeli istnieje kilka dokumentów z tym samym `subtitle_norm`, starsze zostają oznaczone `is_active=false`. Jako „najnowszy” wybierany jest dokument z największym `doc_date_ord`, a w razie remisu z większym `ingested_ts`.
- Addytywność decyzji i logi:
  - decyzje z `replacement` i `subtitle` są agregowane w jednym przebiegu; system stosuje wyłącznie przejścia 1→0 (nigdy nie re‑aktywuje),
  - log zbiorczy: `Aggregated is_active updates | deactivate=… changed=… (rep=…, sub=…, sub_groups=…)`.
 - Sidecar: podniesiono wersję schematu do `2.0.0` (tryb ścisły). Starsze pliki cache (1.x) są ignorowane.
   - Jeśli chcesz odświeżyć cache, uruchom ingest z `force_regen_summary=true` lub usuń katalogi `.summary/` obok plików.
## Nowości w 2.41.2
- Browse: twarda normalizacja pola `query` w `/browse/doc-ids` (po stronie endpointu) — unika sytuacji, w której `null` byłby traktowany jako string "None" i aktywował próbkowanie przy `limit=0`.
## Nowości w 2.42.0
- Nowy endpoint: `POST /quotes/find` — deterministyczna enumeracja cytatów (wystąpień frazy/wyrażeń) w wybranych dokumentach (`restrict_doc_ids`).
  - Tryby dopasowania: `match=phrase|any|all|regex` (domyślnie `phrase`, bez rozróżniania wielkości liter).
  - Granularność: `granularity=occurrence|chunk` (domyślnie `occurrence`).
  - Stronicowanie: `limit` + `cursor`; odpowiedź zawiera `total_quotes`, `returned`, `complete`, `next_cursor`.
  - Przeznaczenie: pełna lista cytatów dla już zawężonego zbioru dokumentów (np. po `POST /browse/doc-ids`). Endpoint nie używa `top_k` ani MMR i nie „tnie” wyników.
- Admin UI: uporządkowano kolejność operacji i dodano gotowy przykład wywołania `Quotes: znajdź cytaty (restrict_doc_ids)`.
