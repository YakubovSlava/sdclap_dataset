#!/usr/bin/env python3
"""
Извлечение фрагментов прямой речи с описательной характеристикой звучания из текстов книг
с помощью локальной LLM (Gemma 4 E4B-it QAT, GGUF, через llama-server).

Запускать из корня репозитория (пути в CSV относительные), venv/ активирован.

Актуальная команда запуска сервера и разбор параметров — docs/llm_setup.md. Вкратце (с MTP,
даёт +27% скорости без потери качества):
    ./.local/llama.cpp/build/bin/llama-server \
        -m unsloth/gemma-4-E4B-it-qat-GGUF/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf \
        -md unsloth/gemma-4-E4B-it-qat-GGUF/mtp-gemma-4-E4B-it.gguf \
        --spec-type draft-mtp --spec-draft-n-max 4 \
        -a gemma-4-e4b -ngl 999 -fa off -c $((16 * 8192)) -np 16 \
        -ub 512 -b 1024 --host 127.0.0.1 --port 1234

Запуск скрипта (сначала на небольшой выборке для проверки качества):
    python scripts/01_extract_speech_fragments.py --limit 10

Параллельный запуск на нескольких машинах — раздели общий CSV непересекающимися диапазонами
--offset/--limit (одна книга не попадёт на два хоста), результаты (по одному .jsonl на книгу)
объединяются простым копированием в общую data/processed/speech_fragments/:
    # машина A:
    python scripts/01_extract_speech_fragments.py --offset 0 --limit 700
    # машина B:
    python scripts/01_extract_speech_fragments.py --offset 700
"""
import argparse
import json
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

from text_utils import (
    chunk_paragraphs,
    epub_paragraphs,
    filter_dialogue_relevant_paragraphs,
    is_junk_description,
    normalize_for_match,
)
from tfidf_pos_hybrid_filter import HybridModel, hybrid_pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("extract_speech_fragments")

# счётчики guard'а (speech in chunk_text) — раньше отброшенные тут фрагменты нигде не считались
# и терялись молча (см. history.md); нужны для видимости масштаба, не только для самого фильтра
guard_stats_lock = threading.Lock()
guard_stats = {"raw": 0, "kept": 0, "dropped_by_old_exact_guard": 0, "dropped_by_new_norm_guard": 0}

FRAGMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "fragments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "speech": {
                        "type": "string",
                        "description": "Дословная прямая речь персонажа, точная подстрока текста фрагмента.",
                    },
                    "description": {
                        "type": "string",
                        "description": (
                            "Авторское описание того, КАК произнесена реплика (тон, темп, "
                            "громкость, эмоция, манера) — не сама реплика и не нейтральный "
                            "глагол вроде 'сказал'."
                        ),
                    },
                    "character": {
                        "type": ["string", "null"],
                        "description": "Имя персонажа, которому принадлежит реплика, если понятно из текста, иначе null.",
                    },
                },
                "required": ["speech", "description", "character"],
            },
        }
    },
    "required": ["fragments"],
}

SYSTEM_PROMPT = """Ты помогаешь собирать датасет для обучения модели, связывающей звучание речи \
с её текстовым описанием. Тебе дают фрагмент текста художественной книги на русском языке.

Найди в тексте все места, где есть ПРЯМАЯ РЕЧЬ персонажа (реплика в кавычках/после тире) И РЯДОМ \
С НЕЙ авторское описание того, КАК именно она ЗВУЧАЛА — тон, громкость, темп, интонация, \
эмоциональная окраска голоса, тембр и т.п. (например: "прошептал он, едва сдерживая ярость", \
"выкрикнула она срывающимся голосом", "медленно и тихо произнёс старик").

description должен относиться СТРОГО к звучанию — к тому, что можно услышать в аудиозаписи этой \
реплики. Критерий: могла бы модель, слушая только аудио (без картинки), определить это свойство?

НЕ считай описанием:
- нейтральные слова без окраски: "сказал", "ответил", "спросил", "произнёс" сами по себе — без \
наречия/деепричастного оборота/сравнения, передающего манеру звучания;
- визуальные детали и жесты, которые нельзя услышать: поклонился, улыбнулся, нахмурился, \
пожал плечами, посмотрел, кивнул и т.п. — даже если это стоит рядом с репликой, это НЕ описание \
звучания и такой фрагмент включать нельзя (если только рядом нет ОТДЕЛЬНОГО указания на сам звук \
голоса — тогда бери в description только звуковую часть, а не жест).

Для каждого найденного места верни:
- speech — дословная цитата реплики персонажа, точная подстрока из данного текста (без изменений).
- description — дословная или близкая к тексту формулировка того, как реплика ЗВУЧАЛА (можно \
взять прямо из авторского текста, без визуальных/жестовых деталей).
- character — имя персонажа, если оно понятно из фрагмента, иначе null.

ВАЖНОЕ ПРАВИЛО про реплики, прерванные авторской вставкой. Если реплика ОДНОГО персонажа \
прерывается посередине словами автора о том, кто и как это сказал («— Часть первая, — сказал X, \
— часть вторая.») — НЕЛЬЗЯ класть весь этот кусок текста в один speech целиком, включая слова \
автора. Обе половины реплики физически разделены в тексте чужими словами — их нужно вернуть как \
ДВА ОТДЕЛЬНЫХ элемента списка fragments (один на каждую половину, у обоих один и тот же \
character). speech каждого элемента — точная непрерывная подстрока ТОЛЬКО своей половины реплики, \
БЕЗ единого слова из авторской вставки (без тире, без "сказал X" и т.п.) — как будто автора там \
вообще не было.

Пример. В тексте: \
«— Спасибо, — с благодарностью откликнулся Гарри. — Рад, что этим летом меня ждёт что-то приятное.»

НЕПРАВИЛЬНО (весь кусок одним speech, авторская вставка не вырезана):
{"speech": "Спасибо, — с благодарностью откликнулся Гарри. — Рад, что этим летом меня ждёт что-то приятное.", "description": "с благодарностью откликнулся", "character": "Гарри"}

ПРАВИЛЬНО (два отдельных элемента, авторская вставка полностью убрана из обоих speech):
{"speech": "Спасибо,", "description": "с благодарностью откликнулся", "character": "Гарри"}
{"speech": "Рад, что этим летом меня ждёт что-то приятное.", "description": "с благодарностью откликнулся", "character": "Гарри"}

Если в тексте таких мест нет — верни пустой список fragments. Не выдумывай реплики и описания, \
которых нет в тексте."""


def call_llm(session: requests.Session, server_url: str, chunk_text: str, timeout: float, max_retries: int) -> list[dict]:
    payload = {
        "model": "gemma-4-e4b-it-qat",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": chunk_text},
        ],
        "temperature": 0.0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "speech_fragments", "schema": FRAGMENT_SCHEMA, "strict": True},
        },
        # thinking намеренно НЕ отключаем здесь: без него модель массово нарушала промпт
        # (описание типа "сказал"/"без описания манеры"/пустая строка вместо того, чтобы вообще
        # не включать такой фрагмент). thinking безлимитный (сервер без --reasoning-budget) —
        # см. docs/llm_setup.md
    }
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.post(f"{server_url}/v1/chat/completions", json=payload, timeout=timeout)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            data = json.loads(content)
            fragments = data.get("fragments", [])

            normalized_chunk = normalize_for_match(chunk_text)
            kept = []
            n_dropped_exact = 0  # отсеялся бы и старым (байтовым) guard'ом — "до"
            n_dropped_norm = 0   # отсеялся бы новым (нормализованным) guard'ом — "после"
            for f in fragments:
                speech = f.get("speech")
                if not speech:
                    continue
                exact_match = speech in chunk_text
                # отсекаем галлюцинации: речь обязана реально быть в чанке. Сравниваем
                # нормализованный вид с обеих сторон (регистр/кавычки-ёлочки-и-прямые/все виды
                # тире/пробелы унифицированы, но проверка остаётся строгим НЕПРЕРЫВНЫМ вхождением
                # подстроки — не fuzzy-поиском), а не сырые байты. LLM почти всегда чуть меняет
                # типографику при копировании реплики, и байтовое сравнение из-за этого молча
                # теряло валидные фрагменты (замер: старый guard пропускал только 17% реально
                # найденных LLM реплик, см. history.md).
                norm_match = normalize_for_match(speech) in normalized_chunk
                if not exact_match:
                    n_dropped_exact += 1
                if not norm_match:
                    n_dropped_norm += 1
                    continue
                # отсекаем заглушки вместо валидного описания манеры речи
                if is_junk_description(f.get("description"), speech):
                    continue
                kept.append(f)

            with guard_stats_lock:
                guard_stats["raw"] += len(fragments)
                guard_stats["kept"] += len(kept)
                guard_stats["dropped_by_old_exact_guard"] += n_dropped_exact
                guard_stats["dropped_by_new_norm_guard"] += n_dropped_norm

            return kept
        except Exception as e:
            last_err = e
            log.warning("llm call failed (attempt %d/%d): %s", attempt, max_retries, e)
            time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_err}")


def load_books(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["path_book"])
    # одна и та же книга (текст) может встречаться несколько раз — по одной строке на каждого
    # чтеца; экстракцию по тексту делаем один раз, актёра присоединяем позже (шаг 1.3)
    return df.drop_duplicates(subset=["path_book"])


def book_id_for(path_book: str) -> str:
    return re.sub(r"[^0-9a-zA-Zа-яА-Я]+", "_", path_book).strip("_")[-150:]


def write_book_output(state: dict) -> None:
    out_path = state["out_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in sorted(state["results"], key=lambda r: r["chunk_id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    state["done_marker"].touch()
    log.info("%s: %d fragments extracted", state["path_book"].name, len(state["results"]))


def prepare_book_states(df: pd.DataFrame, args: argparse.Namespace, hybrid_model: HybridModel | None) -> dict[str, dict]:
    """Парсит все epub заранее и строит per-книжное состояние + список чанков-задач."""
    book_states: dict[str, dict] = {}
    chunks_before_filter = 0
    chunks_after_dialogue = 0
    chunks_after_hybrid = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="parsing epub", unit="book"):
        path_book = Path(row["path_book"])
        book_id = book_id_for(str(row["path_book"]))
        out_path = Path(args.output_dir) / f"{book_id}.jsonl"
        done_marker = out_path.with_suffix(".done")
        if done_marker.exists() and not args.overwrite:
            continue
        if not path_book.exists():
            log.warning("missing epub, skip: %s", path_book)
            continue
        try:
            paragraphs = epub_paragraphs(path_book)
        except Exception as e:
            log.error("failed to parse epub %s: %s", path_book, e)
            continue

        chunks_before_filter += len(chunk_paragraphs(paragraphs, args.chunk_chars))
        if not args.skip_filter:
            paragraphs = filter_dialogue_relevant_paragraphs(paragraphs, context=args.dialogue_context)
        chunks_after_dialogue += len(chunk_paragraphs(paragraphs, args.chunk_chars))
        if hybrid_model is not None:
            paragraphs = hybrid_pass(hybrid_model, paragraphs, args.hybrid_threshold)
        chunks = chunk_paragraphs(paragraphs, args.chunk_chars)
        chunks_after_hybrid += len(chunks)
        if not chunks:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.touch()
            done_marker.touch()
            continue

        book_states[book_id] = {
            "path_book": path_book,
            "row": row,
            "chunks": chunks,
            "out_path": out_path,
            "done_marker": done_marker,
            "results": [],
            "remaining": len(chunks),
            "lock": threading.Lock(),
        }

    if not args.skip_filter and chunks_before_filter:
        dropped = chunks_before_filter - chunks_after_dialogue
        log.info(
            "dialogue filter: %d -> %d chunks (отсеяно %d, %.0f%%)",
            chunks_before_filter,
            chunks_after_dialogue,
            dropped,
            100 * dropped / chunks_before_filter,
        )
    if hybrid_model is not None and chunks_after_dialogue:
        dropped = chunks_after_dialogue - chunks_after_hybrid
        log.info(
            "hybrid filter (порог %.2f): %d -> %d chunks (отсеяно %d, %.0f%%)",
            args.hybrid_threshold,
            chunks_after_dialogue,
            chunks_after_hybrid,
            dropped,
            100 * dropped / chunks_after_dialogue,
        )
    return book_states


def process_chunk_task(
    book_id: str,
    chunk_id: int,
    book_states: dict[str, dict],
    session: requests.Session,
    args: argparse.Namespace,
) -> None:
    state = book_states[book_id]
    chunk_text = state["chunks"][chunk_id]
    try:
        fragments = call_llm(session, args.server_url, chunk_text, args.timeout, args.max_retries)
    except Exception as e:
        log.error("%s chunk %d failed permanently: %s", state["path_book"].name, chunk_id, e)
        fragments = []

    row = state["row"]
    with state["lock"]:
        for frag in fragments:
            state["results"].append(
                {
                    "book_id": book_id,
                    "path_book": str(state["path_book"]),
                    "book_name": row.get("book_names"),
                    "book_author": row.get("book_authors"),
                    "chunk_id": chunk_id,
                    "chunk_text": chunk_text,
                    "speech": frag["speech"],
                    "description": frag["description"],
                    "character": frag.get("character"),
                }
            )
        state["remaining"] -= 1
        book_finished = state["remaining"] == 0

    if book_finished:
        write_book_output(state)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", default="data/books_postdownload_filtered.csv")
    parser.add_argument("--output-dir", default="data/processed/speech_fragments")
    parser.add_argument("--server-url", default="http://127.0.0.1:1234")
    parser.add_argument("--chunk-chars", type=int, default=6000)
    parser.add_argument("--workers", type=int, default=16, help="должно совпадать с -np сервера")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--ramp-delay",
        type=float,
        default=0.5,
        help="пауза (сек) между первыми --workers запросами, чтобы не грузить сервер холодным залпом",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help=(
            "пропустить первые N книг перед --limit — для параллельного запуска на нескольких "
            "машинах: раздели общий CSV на непересекающиеся диапазоны --offset/--limit, "
            "результаты (по одному .jsonl на книгу в --output-dir) объединяются просто "
            "копированием файлов в общую директорию, дублей не будет"
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="обработать только первые N книг после --offset (для теста)")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-filter",
        action="store_true",
        help="не отсеивать абзацы без прямой речи (тире-диалог/кавычки) перед нарезкой на чанки",
    )
    parser.add_argument(
        "--dialogue-context",
        type=int,
        default=1,
        help="сколько соседних абзацев вокруг реплики оставлять как контекст для фильтра диалогов",
    )
    parser.add_argument(
        "--dry-run-filter",
        action="store_true",
        help="только посчитать, сколько чанков отсеет фильтр диалогов, без обращений к LLM",
    )
    parser.add_argument(
        "--hybrid-filter",
        action="store_true",
        help=(
            "включить гибридный TF-IDF+POS+контекст фильтр поверх диалогового (см. "
            "для статьи/tfidf_pos_hybrid_experiment.md) — обучение и веса: "
            "scripts/train_tfidf_pos_hybrid.py"
        ),
    )
    parser.add_argument(
        "--hybrid-threshold",
        type=float,
        default=0.25,
        help=(
            "порог вероятности для --hybrid-filter (выше — агрессивнее экономия, ниже recall). "
            "0.14 — консервативный (recall ~97%% на hold-out), 0.25 — сбалансированный "
            "(~2x меньше запросов при recall ~90-92%%), см. для статьи/"
            "tfidf_pos_hybrid_experiment.md §5.2 за полной таблицей порог/экономия/recall"
        ),
    )
    parser.add_argument(
        "--hybrid-model",
        default="scripts/models/tfidf_pos_hybrid_cv.pkl",
        help="путь к весам гибридной модели (см. --hybrid-filter)",
    )
    args = parser.parse_args()

    hybrid_model = None
    if args.hybrid_filter:
        hybrid_model = HybridModel.load(Path(args.hybrid_model))
        log.info("hybrid filter enabled: model=%s, threshold=%.2f", args.hybrid_model, args.hybrid_threshold)

    df = load_books(Path(args.csv))
    if args.offset:
        df = df.iloc[args.offset :]
    if args.limit:
        df = df.head(args.limit)
    log.info("books to process: %d (offset=%d, limit=%s)", len(df), args.offset, args.limit)

    book_states = prepare_book_states(df, args, hybrid_model)
    tasks = [(book_id, i) for book_id, state in book_states.items() for i in range(len(state["chunks"]))]
    log.info("books queued: %d, chunks queued: %d", len(book_states), len(tasks))

    if args.dry_run_filter:
        return

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=args.workers, pool_maxsize=args.workers)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    start = time.perf_counter()

    # общая очередь чанков по всему корпусу — не по одной книге за раз, чтобы все слоты
    # сервера (-np) были постоянно заняты вне зависимости от размера текущей книги
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = []
        for i, (book_id, chunk_id) in enumerate(tasks):
            # первые --workers запросов растягиваем по времени: если все N слотов разом
            # начинают prefill без flash attention, суммарное время первой волны легко
            # превышает --timeout, и все они синхронно отваливаются, потом повторяется
            # на ретраях — по факту зависание навсегда
            if i < args.workers and args.ramp_delay:
                time.sleep(args.ramp_delay)
            futures.append(pool.submit(process_chunk_task, book_id, chunk_id, book_states, session, args))
        for future in tqdm(as_completed(futures), total=len(futures), desc="chunks", unit="chunk"):
            future.result()

    elapsed = time.perf_counter() - start
    log.info("elapsed: %.1fs for %d chunks (%.2fs/chunk avg)", elapsed, len(tasks), elapsed / max(len(tasks), 1))
    raw = max(guard_stats["raw"], 1)
    log.info(
        "guard speech-in-chunk: raw=%d | ДО (байтовый guard) отброшено=%d (осталось бы %.1f%%) | "
        "ПОСЛЕ (нормализованный guard) отброшено=%d (осталось %.1f%%) | итог kept=%d",
        guard_stats["raw"],
        guard_stats["dropped_by_old_exact_guard"],
        100 * (guard_stats["raw"] - guard_stats["dropped_by_old_exact_guard"]) / raw,
        guard_stats["dropped_by_new_norm_guard"],
        100 * (guard_stats["raw"] - guard_stats["dropped_by_new_norm_guard"]) / raw,
        guard_stats["kept"],
    )


if __name__ == "__main__":
    main()
