#!/usr/bin/env python3
"""
Матчинг фрагментов прямой речи (результат шага 1.2, `data/processed/speech_fragments/*.jsonl`)
с участками аудиозаписи, где эта речь звучит (шаг 1.4).

Подробная документация (формат выхода, производительность, известные ограничения, масштаб
проблемы реестра книга/аудио) — `docs/audio_matching.md`. Ниже — краткая версия.

Один текст книги (`path_book`) может быть озвучен несколькими чтецами — реестр
`data/books_postdownload_filtered.csv` даёт все пары (path_book, reader, path_audio),
матчинг делается отдельно для каждой пары.

Скрипт работает в три этапа для каждой (книга, чтец):
  1. **Проверка на неправильный метчинг при скраппинге.** До 1400 книг в реестре — треть
     оказалась с полностью не тем текстом на не той аудиозаписи (title-коллизии/дубли при
     сопоставлении источников текста и аудио на шаге 1.1, см. `history.md`). Берутся до 3 целых абзацев
     (начало/середина/конец книги, каждый — цельный кусок chunk_text между `\n\n`, не произвольный
     срез в N символов — обрезка на полуслове/полупредложении систематически завышала расстояние
     даже у верных пар, см. `history.md`) и ищутся по всей аудио-транскрипции целиком (`edlib`,
     `HW`, без окна). Если ни один образец не находится — пара считается плохой.
  2. **Плохие пары выписываются** в `data/processed/bad_text_audio_pairs.csv` и полностью
     исключаются из дальнейшего матчинга (их фрагменты не попадают в выход шага 3).
  3. **Матчинг с аудио** — только для пар, прошедших проверку. Двухуровневый поиск через edlib
     (широкий контекст → сама реплика), с двумя ключевыми отличиями от наивного варианта:
     - Монотонное окно поиска: фрагменты обрабатываются по порядку (chunk_id, позиция внутри
       чанка), а поиск очередного фрагмента ведётся не по всей книге, а по суффиксу слов начиная
       с конца предыдущего найденного фрагмента. Это одновременно чинит проблему тай-брейка edlib
       (при точном совпадении текста возвращается список локаций, отсортированный по возрастанию
       позиции, и брать location[0] "в лоб" по всей книге — значит почти всегда попадать в самое
       раннее вхождение повторяющейся фразы, а не в нужное) и ускоряет поиск (окно сужается по
       мере продвижения по книге). Если в окне не находится разумного совпадения (текст пошёл не
       по порядку — вставка/перестановка), делается один fallback-поиск по всей книге.
     - Кэш аудио-транскрипции — один файл на (reader, title) в
       `data/processed/audio_words_cache/`, а не общий пикл на весь корпус (грузится лениво, по
       требованию, не съедает память сразу на весь корпус).

Порогов/отсева по качеству ПОФРАГМЕНТНОГО матчинга (этап 3) нет — все попытки, включая слабые,
пишутся в выход с сырыми метриками (edit distance, ASR confidence). Отсев — следующий шаг
(1.5, WER-фильтрация). Отсев на уровне этапа 1 (плохая пара книга/аудио целиком) — другое дело,
это не вопрос качества конкретного фрагмента, а испорченные исходные данные.

Запуск (сначала на маленькой выборке):
    python scripts/02_match_fragments_to_audio.py --limit 15
"""
import argparse
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import edlib
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("match_fragments_to_audio")

MISMATCH_RATIO_THRESHOLD = 0.35
CHECK_SAMPLE_CHARS = 300
# Порог провала окна адаптивный по книге (см. match_one) — эти константы задают, как именно:
# для первых MIN_BASELINE_SAMPLES фрагментов книги используется дефолтный порог (нет ещё своей
# статистики), дальше — медиана уже накопленных "успешных" block_distance_norm + запас, зажатая
# в [FLOOR, CEIL], чтобы не улетать в крайности на короткой книге/выбросах.
FALLBACK_TRIGGER_DEFAULT = 0.55
FALLBACK_TRIGGER_MARGIN = 0.2
FALLBACK_TRIGGER_FLOOR = 0.5
FALLBACK_TRIGGER_CEIL = 0.85
MIN_BASELINE_SAMPLES = 5
# Независимый от адаптивного порога триггер fallback — по итоговому качеству локального матча
# (match_distance_norm), а не только по блочному (block_distance_norm может выглядеть нормально,
# пока сам speech найден плохо внутри чуть промахнувшегося блока — см. match_one).
LOCAL_FALLBACK_TRIGGER_RATIO = 0.4

OUTPUT_COLUMNS = [
    "book_id", "path_book", "book_name", "book_author", "reader", "path_audio",
    "chunk_id", "chunk_text", "speech", "description", "character",
    "match_start_ts", "match_end_ts", "match_distance", "match_distance_norm",
    "block_distance_norm", "n_words_matched",
    "asr_confidence_mean", "asr_confidence_min",
    "match_status", "search_window_expanded",
]


def safe_name(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Zа-яА-Я]+", "_", s).strip("_")[-150:]


def load_fragments(fragments_dir: Path, limit: int | None = None) -> pd.DataFrame:
    rows = []
    for fp in sorted(fragments_dir.glob("*.jsonl")):
        with fp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if limit is not None and len(rows) >= limit:
            break
    df = pd.DataFrame(rows)
    if limit is not None:
        df = df.iloc[:limit]
    return df


def load_registry(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["path_book", "path_audio", "book_reader_names"])
    df = df.rename(columns={"book_reader_names": "reader"})
    return df[["path_book", "reader", "path_audio"]].drop_duplicates()


class AudioWords:
    """Слова аудио-транскрипции книги + предпосчитанный индекс для быстрого оконного поиска."""

    def __init__(self, words: list[dict]):
        self.words = words
        char_ids = []
        starts = np.zeros(len(words), dtype=np.int64)
        pos = 0
        for idx, w in enumerate(words):
            starts[idx] = pos
            n = len(w["word"]) + 1  # +1 за разделяющий пробел
            char_ids.extend([idx] * n)
            pos += n
        # регистр приводим к нижнему для устойчивости матчинга (капслок в тексте vs обычный
        # регистр в ASR-транскрипции, например выкрики: "ОТДАЙТЕ МНЕ МОЁ ПИСЬМО!") — .lower() не
        # меняет длину строки для кириллицы/латиницы, индексы остаются валидными
        self.joined_text = " ".join(w["word"] for w in words).lower()
        self.char_to_word = np.array(char_ids[:-1] if char_ids else [], dtype=np.int64)
        self.word_char_start = starts

    def __len__(self) -> int:
        return len(self.words)


def load_audio_words(path_audio: Path, cache_dir: Path) -> AudioWords | None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{safe_name(str(path_audio))}.npz"
    title = path_audio.name
    json_path = path_audio / "raw" / f"{title}.json"

    if cache_path.exists():
        try:
            return _load_cached_audio_words(cache_path)
        except Exception:
            log.warning("corrupt cache %s, rebuilding", cache_path)

    if not json_path.exists():
        return None

    with json_path.open(encoding="utf-8") as f:
        transcribed = json.load(f)

    words = [w for seg in transcribed for w in seg["words"]]
    for i in range(len(words) - 1):
        if words[i]["end"] > words[i + 1]["start"]:
            overlap = words[i]["end"] - words[i + 1]["start"]
            words[i]["end"] -= overlap / 2
            words[i + 1]["start"] += overlap / 2

    aw = AudioWords(words)
    _save_cached_audio_words(cache_path, aw)
    return aw


def _save_cached_audio_words(cache_path: Path, aw: AudioWords) -> None:
    starts = np.array([w["start"] for w in aw.words], dtype=np.float64)
    ends = np.array([w["end"] for w in aw.words], dtype=np.float64)
    scores = np.array([w.get("score", 1.0) for w in aw.words], dtype=np.float64)
    words_arr = np.array([w["word"] for w in aw.words], dtype=object)
    np.savez_compressed(cache_path, words=words_arr, starts=starts, ends=ends, scores=scores)


def _load_cached_audio_words(cache_path: Path) -> AudioWords:
    data = np.load(cache_path, allow_pickle=True)
    words = [
        {"word": w, "start": s, "end": e, "score": sc}
        for w, s, e, sc in zip(data["words"], data["starts"], data["ends"], data["scores"])
    ]
    return AudioWords(words)


def _edlib_locate(query: str, target: str) -> tuple[int, int, int] | None:
    """Возвращает (char_start, char_end_inclusive, distance) в target, либо None, если target пуст."""
    if not target or not query:
        return None
    res = edlib.align(query, target, mode="HW", task="locations")
    if res["editDistance"] < 0:
        return None
    start, end = res["locations"][0]
    if start is None or end is None or end < 0:
        return None
    return start, end, res["editDistance"]


def _pick_paragraph(chunk_text: str) -> str | None:
    """Берёт один ЦЕЛЫЙ абзац из chunk_text, а не произвольный срез в CHECK_SAMPLE_CHARS символов
    — обрезка посередине предложения/слова добавляет "хвост" без пары в аудио и systематически
    завышает edit distance даже для полностью верных пар (проверено эмпирически: один и тот же
    отрывок давал ratio 0.29 целым и 0.45 обрубленным на границе 300 символов)."""
    paragraphs = [p.strip() for p in chunk_text.split("\n\n") if p.strip()]
    if not paragraphs:
        return None
    candidates = [p for p in paragraphs if 100 <= len(p) <= 350]
    if candidates:
        return max(candidates, key=len)
    return max(paragraphs, key=len)[:CHECK_SAMPLE_CHARS]


def sample_chunks_for_check(fragments: list[dict], n: int = 3) -> list[str]:
    """Берёт до n образцов (по одному целому абзацу) из chunk_text, разнесённых по книге
    (начало/середина/конец) — для проверки, что книга и аудио вообще соответствуют друг другу
    (этап 1)."""
    chunk_by_id = {}
    for f in fragments:
        chunk_by_id.setdefault(f["chunk_id"], f["chunk_text"])
    chunk_ids = sorted(chunk_by_id)
    if not chunk_ids:
        return []
    if len(chunk_ids) <= n:
        picks = chunk_ids
    else:
        idxs = [0, len(chunk_ids) // 2, len(chunk_ids) - 1]
        picks = sorted(set(chunk_ids[i] for i in idxs))
    samples = [_pick_paragraph(chunk_by_id[cid]) for cid in picks]
    return [s for s in samples if s]


def check_pair_quality(sample_texts: list[str], full_audio_text: str) -> dict:
    """Этап 1: ищет каждый образец по всей аудио-транскрипции целиком (без окна). Если хотя бы
    один образец находится с разумной точностью — пара считается нормальной (одного плохого
    абзаца, например из-за сокращения при озвучке, недостаточно, чтобы забраковать книгу)."""
    ratios = []
    for s in sample_texts:
        if not s or not full_audio_text:
            ratios.append(None)
            continue
        s = s.lower()
        k = max(int(len(s) * (MISMATCH_RATIO_THRESHOLD + 0.15)), 10)
        res = edlib.align(s, full_audio_text, mode="HW", task="distance", k=k)
        dist = res["editDistance"]
        ratios.append(dist / len(s) if dist >= 0 else None)

    valid = [r for r in ratios if r is not None]
    if not sample_texts:
        status = "no_sample"
    elif not valid:
        status = "mismatch"
    else:
        status = "ok" if min(valid) < MISMATCH_RATIO_THRESHOLD else "mismatch"
    return {"ratios": ratios, "status": status}


def match_one(
    speech: str,
    chunk_text: str,
    aw: AudioWords,
    prev_end_word_idx: int,
    baseline: list[float],
) -> dict:
    """Матчит один фрагмент речи, начиная поиск с prev_end_word_idx (монотонность). Двухуровневый:
    сначала широкий контекст (chunk_text) внутри окна, затем сама реплика (speech) внутри найденного
    контекста. При провале в окне — один fallback-поиск по всей книге (search_window_expanded=True).

    Порог провала окна адаптивный (`baseline` — растущий список block_distance_norm успешных
    оконных матчей этой же книги): "шумовой пол" сильно различается по книгам (чище/грязнее ASR,
    рваность chunk_text от диалогового фильтра) — фиксированный порог либо триггерит fallback
    почти на каждом фрагменте шумной книги (дорого и бессмысленно, окно и так было верным), либо
    пропускает реальную порчу окна на чистой книге.
    """
    chunk_text = chunk_text.lower()
    speech = speech.lower()

    def try_window(word_offset: int) -> dict | None:
        char_offset = int(aw.word_char_start[word_offset]) if word_offset < len(aw) else len(aw.joined_text)
        window_text = aw.joined_text[char_offset:]
        if not window_text:
            return None

        block = _edlib_locate(chunk_text, window_text)
        if block is None:
            return None
        block_start, block_end, block_dist = block
        block_slice = window_text[block_start: block_end + 1]
        block_char_offset = char_offset + block_start

        local = _edlib_locate(speech, block_slice)
        if local is None:
            return None
        local_start, local_end, local_dist = local

        abs_start_char = block_char_offset + local_start
        abs_end_char = block_char_offset + local_end

        w0 = int(aw.char_to_word[min(abs_start_char, len(aw.char_to_word) - 1)])
        w1 = int(aw.char_to_word[min(abs_end_char, len(aw.char_to_word) - 1)])
        w0 = max(0, min(w0, len(aw) - 1))
        w1 = max(w0, min(w1, len(aw) - 1))

        scores = [aw.words[i]["score"] for i in range(w0, w1 + 1)]
        return {
            "match_start_ts": aw.words[w0]["start"],
            "match_end_ts": aw.words[w1]["end"],
            "match_distance": local_dist,
            "match_distance_norm": local_dist / max(len(speech), 1),
            "block_distance_norm": block_dist / max(len(chunk_text), 1),
            "n_words_matched": w1 - w0 + 1,
            "asr_confidence_mean": float(np.mean(scores)),
            "asr_confidence_min": float(np.min(scores)),
            "match_status": "ok",
            "end_word_idx": w1 + 1,
        }

    result = try_window(prev_end_word_idx)
    expanded = False

    if len(baseline) >= MIN_BASELINE_SAMPLES:
        threshold = min(max(float(np.median(baseline)) + FALLBACK_TRIGGER_MARGIN, FALLBACK_TRIGGER_FLOOR),
                         FALLBACK_TRIGGER_CEIL)
    else:
        threshold = FALLBACK_TRIGGER_DEFAULT

    # edlib почти никогда не возвращает None даже на мусорном совпадении (HW-режим всегда даёт
    # "лучший из худших" результат) — поэтому триггерим fallback по качеству матча, а не по факту
    # отсутствия результата. Без этого один плохой матч "заражает" prev_end_word_idx и каскадом
    # ломает все последующие фрагменты той же книги (окно схлопывается в мусор).
    #
    # Триггерим по ДВУМ метрикам, не только по block_distance_norm: качественный блочный матч не
    # гарантирует качественный локальный (speech внутри block_slice) — если block чуть промахнулся
    # мимо истинной позиции (типично при большом рваном chunk_text), speech может оказаться у самой
    # границы найденного среза или вовсе за ней, и локальный поиск довольствуется худшим совпадением
    # внутри, хотя реально хорошее совпадение есть рядом за границей блока. Обнаружено на реальных
    # данных полного прогона: block_distance_norm=0.54 (прошёл порог), но match_distance_norm=0.65,
    # хотя прямой поиск одной speech по всей книге дал 0.19 в той же точке — см. history.md.
    needs_fallback = (
        result is None
        or result["block_distance_norm"] > threshold
        or result["match_distance_norm"] > LOCAL_FALLBACK_TRIGGER_RATIO
    )
    if needs_fallback and prev_end_word_idx > 0:
        expanded = True
        fallback_result = try_window(0)
        if fallback_result is not None and (
            result is None or fallback_result["match_distance_norm"] < result["match_distance_norm"]
        ):
            result = fallback_result
    elif result is not None:
        baseline.append(result["block_distance_norm"])

    if result is None:
        return {
            "match_start_ts": None, "match_end_ts": None, "match_distance": None,
            "match_distance_norm": None, "block_distance_norm": None, "n_words_matched": 0,
            "asr_confidence_mean": None, "asr_confidence_min": None,
            "match_status": "no_match", "search_window_expanded": expanded,
            "end_word_idx": prev_end_word_idx,
        }

    result["search_window_expanded"] = expanded
    return result


def process_book_reader(
    path_book: str, reader: str, path_audio: Path, fragments: list[dict], cache_dir: Path,
) -> tuple[list[dict], dict | None]:
    """Возвращает (строки для matched_fragments, запись для bad_text_audio_pairs либо None)."""
    aw = load_audio_words(path_audio, cache_dir)
    if aw is None:
        out = [
            {**frag, "reader": reader, "path_audio": str(path_audio),
             "match_start_ts": None, "match_end_ts": None, "match_distance": None,
             "match_distance_norm": None, "block_distance_norm": None,
             "n_words_matched": 0, "asr_confidence_mean": None,
             "asr_confidence_min": None, "match_status": "no_audio_json",
             "search_window_expanded": False}
            for frag in fragments
        ]
        return out, None

    # Этап 1: проверка на неправильный метчинг книга↔аудио при скраппинге.
    samples = sample_chunks_for_check(fragments)
    check = check_pair_quality(samples, aw.joined_text)
    if check["status"] != "ok":
        bad_pair = {
            "path_book": path_book, "path_audio": str(path_audio), "reader": reader,
            "book_name": fragments[0].get("book_name"), "book_author": fragments[0].get("book_author"),
            "check_ratios": ";".join(f"{r:.3f}" if r is not None else "NA" for r in check["ratios"]),
            "n_fragments_skipped": len(fragments),
        }
        # Этап 2: плохая пара выписывается отдельно, в матчинг (этап 3) не идёт.
        return [], bad_pair

    # Этап 3: матчинг фрагментов с аудио — только для пар, прошедших проверку.
    out = []
    prev_end = 0
    baseline: list[float] = []
    # sorted(fragments, key=lambda f: (..., fragments.index(f))) был бы O(n²) (index() — линейный
    # поиск, и на дублирующихся dict возвращает один и тот же индекс) — сортируем через
    # enumerate(), исходный порядок внутри chunk_id сохраняется корректно и за O(n log n).
    for _, frag in sorted(enumerate(fragments), key=lambda x: (x[1]["chunk_id"], x[0])):
        m = match_one(frag["speech"], frag["chunk_text"], aw, prev_end, baseline)
        prev_end = m.pop("end_word_idx")
        out.append({**frag, "reader": reader, "path_audio": str(path_audio), **m})
    return out, None


def load_resume_state(jsonl_path: Path, bad_pairs_path: Path) -> dict:
    """Читает уже посчитанные выходные файлы прошлого (прерванного) прогона, чтобы можно было
    пропустить готовые пары книга/чтец вместо полного перезапуска."""
    resumed_pairs: set[tuple[str, str]] = set()
    n_rows = n_ok = n_expanded = 0
    if jsonl_path.exists():
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                resumed_pairs.add((r["path_book"], r["reader"]))
                n_rows += 1
                n_ok += r.get("match_status") == "ok"
                n_expanded += bool(r.get("search_window_expanded"))

    bad_pairs: list[dict] = []
    if bad_pairs_path.exists():
        bad_pairs = pd.read_csv(bad_pairs_path).to_dict("records")
        for bp in bad_pairs:
            resumed_pairs.add((bp["path_book"], bp["reader"]))

    return {"resumed_pairs": resumed_pairs, "n_rows": n_rows, "n_ok": n_ok,
            "n_expanded": n_expanded, "bad_pairs": bad_pairs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fragments-dir", default="data/processed/speech_fragments")
    parser.add_argument("--registry-csv", default="data/books_postdownload_filtered.csv")
    parser.add_argument("--output-dir", default="data/processed/matched_fragments")
    parser.add_argument("--cache-dir", default="data/processed/audio_words_cache")
    parser.add_argument("--bad-pairs-csv", default="data/processed/bad_text_audio_pairs.csv")
    parser.add_argument("--limit", type=int, default=None, help="ограничить число фрагментов (для теста)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true",
                         help="игнорировать уже посчитанные пары книга/чтец в выходных файлах и начать заново "
                              "(по умолчанию — резюмировать прерванный прогон, пропуская готовые пары)")
    args = parser.parse_args()

    fragments_dir = Path(args.fragments_dir)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "matched_fragments.jsonl"
    bad_pairs_path = Path(args.bad_pairs_csv)
    bad_pairs_path.parent.mkdir(parents=True, exist_ok=True)

    # Резюмирование: прогон занимает часы, при перезапуске после падения/прерывания по умолчанию
    # не начинаем заново, а пропускаем пары книга/чтец, уже посчитанные в прошлый раз (по
    # выходным файлам, не по .done-маркерам — их тут нет). --overwrite отключает это поведение.
    resumed_pairs: set[tuple[str, str]] = set()
    n_rows = n_ok = n_expanded = 0
    bad_pairs: list[dict] = []
    if not args.overwrite:
        state = load_resume_state(jsonl_path, bad_pairs_path)
        resumed_pairs, n_rows, n_ok, n_expanded, bad_pairs = (
            state["resumed_pairs"], state["n_rows"], state["n_ok"], state["n_expanded"], state["bad_pairs"],
        )
        if resumed_pairs:
            log.info("резюмирую прерванный прогон: %d пар книга/чтец уже готовы (%d строк), пропускаю их",
                      len(resumed_pairs), n_rows)

    log.info("loading fragments...")
    frag_df = load_fragments(fragments_dir, limit=args.limit)
    log.info("loaded %d fragments", len(frag_df))

    registry = load_registry(Path(args.registry_csv))
    joined = frag_df.merge(registry, on="path_book", how="left")

    unmatched_books = joined[joined["reader"].isna()]["path_book"].unique()
    if len(unmatched_books):
        log.warning("%d books have no reader/path_audio in registry (skipped)", len(unmatched_books))
    joined = joined.dropna(subset=["reader", "path_audio"])

    tasks = []
    for (path_book, reader, path_audio), group in joined.groupby(["path_book", "reader", "path_audio"]):
        if (path_book, reader) in resumed_pairs:
            continue
        tasks.append((path_book, reader, Path(path_audio), group.to_dict("records")))

    log.info("matching %d (book, reader) pairs across %d fragments...", len(tasks),
              sum(len(t[3]) for t in tasks))

    # Пишем инкрементально по мере готовности каждой (книга, чтец)-пары, а не всё разом в конце:
    # полный прогон занимает часы (см. docs/audio_matching.md), падение/прерывание на середине не
    # должно стирать уже сделанную работу.
    jsonl_mode = "a" if resumed_pairs else "w"
    with jsonl_path.open(jsonl_mode, encoding="utf-8") as jsonl_f, \
         ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_book_reader, pb, rd, pa, frags, cache_dir): (pb, rd)
            for pb, rd, pa, frags in tasks
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="book-reader pairs"):
            pb, rd = futures[fut]
            try:
                rows, bad_pair = fut.result()
            except Exception as e:
                log.error("failed for %s / %s: %s", pb, rd, e)
                continue

            for r in rows:
                row = {col: r.get(col) for col in OUTPUT_COLUMNS}
                jsonl_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
                n_ok += row["match_status"] == "ok"
                n_expanded += bool(row["search_window_expanded"])
            jsonl_f.flush()

            if bad_pair is not None:
                bad_pairs.append(bad_pair)
                pd.DataFrame(bad_pairs).to_csv(bad_pairs_path, index=False)

    if bad_pairs:
        n_skipped_fragments = sum(bp["n_fragments_skipped"] for bp in bad_pairs)
        log.info("этап 1/2: %d пар книга/аудио забракованы (%d фрагментов исключены из матчинга), записано в %s",
                  len(bad_pairs), n_skipped_fragments, bad_pairs_path)

    csv_path = output_dir / "matched_fragments.csv"
    if n_rows:
        pd.read_json(jsonl_path, lines=True).to_csv(csv_path, index=False)
    else:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(csv_path, index=False)

    log.info("этап 3: %d строк, %d ok (%.1f%%), %d через fallback-поиск по всей книге",
              n_rows, n_ok, 100 * n_ok / max(n_rows, 1), n_expanded)
    log.info("written: %s, %s", jsonl_path, csv_path)


if __name__ == "__main__":
    main()
