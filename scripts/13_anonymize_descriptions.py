#!/usr/bin/env python3
"""
Анонимизация имён/ролей персонажей в `description` (шаг 1.4c, до текстовых эмбеддингов) — убирает
буквальное упоминание `character` из `description`, чтобы текстовая башня CLAP не училась на
совпадении имени персонажа вместо манеры произнесения реплики (риск найден вручную на
nearest-neighbors bge-m3: "Слава стал орать" ранжировался по совпадению имени, см. history.md
2026-07-28).

Не трогает scores.csv/manifest.csv — строит lookup-таблицу description -> description_anon,
которую 11_/12_embed_descriptions*.py применяют через --anonymize (обновление отдельным шагом).

Метод:
  1. Джойн scores.csv (description) + dataset.csv (character) по fragment_id, тот же фильтр жёстких
     junk-флагов, что в 11_/12_/14_ (~junk_any_hard & ~junk_badmatch & ~junk_few_words &
     ~junk_few_asr_words) — та же база 89015 фрагментов.
  2. Для каждого уникального description собираем МНОЖЕСТВО связанных `character` — один и тот же
     текст description может встречаться у разных персонажей (3683 из 51059 случаев), это почти
     всегда бессодержательные фразы без имени.
  3. Местоимения-заглушки (он/она/оно/я/ты/вы/мы/они/self/unspecified/...) НЕ считаются кандидатом
     на удаление — LLM ставит их, когда персонаж не определён из контекста; вырезание было бы
     бессмысленным и ломало бы обычные предложения ("весело сказал он").
  4. Вырезаем: (a) точное вхождение character как целого слова/фразы (word-boundary regex);
     (b) слова description, чья pymorphy3-лемма совпадает с леммой любого слова character — ловит
     склонения ("Хоскинса" и т.п.), которые точное совпадение не поймает.
  5. Схлопываем лишние пробелы/висящие знаки препинания, оставшиеся после вырезания.

Выход: data/processed/audio_fragments/description_anonymization.csv
  (description, description_anon, changed)

Запуск (из корня репозитория, venv активирован):
    python scripts/10b_anonymize_descriptions.py
    python scripts/10b_anonymize_descriptions.py --limit 500   # тест на подвыборке фрагментов
"""
import argparse
import logging
import re
from pathlib import Path

import pandas as pd
import pymorphy3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("anonymize_descriptions")

BASE = "data/processed/audio_fragments"

PRONOUN_STOPLIST = {
    "он", "она", "оно", "я", "ты", "вы", "мы", "они",
    "self", "unspecified", "self/unspecified", "кто-то", "некто",
}

morph = pymorphy3.MorphAnalyzer()
_lemma_cache: dict[str, str] = {}


def lemma(word: str) -> str:
    wl = word.lower()
    cached = _lemma_cache.get(wl)
    if cached is None:
        cached = morph.parse(wl)[0].normal_form
        _lemma_cache[wl] = cached
    return cached


def load_fragments(limit: int | None) -> pd.DataFrame:
    scores = pd.read_csv(f"{BASE}/scores.csv",
                          usecols=["fragment_id", "description", "junk_any_hard", "junk_badmatch",
                                   "junk_few_words", "junk_few_asr_words"])
    hard_ok = (~scores["junk_any_hard"] & ~scores["junk_badmatch"]
               & ~scores["junk_few_words"] & ~scores["junk_few_asr_words"])
    scores = scores[hard_ok]
    dataset = pd.read_csv(f"{BASE}/dataset.csv", usecols=["fragment_id", "character"])
    df = scores.merge(dataset, on="fragment_id", how="left")
    if limit is not None:
        df = df.head(limit)
    return df


def is_real_character(character: object) -> bool:
    if not isinstance(character, str):
        return False
    c = character.strip()
    if not c:
        return False
    return c.lower() not in PRONOUN_STOPLIST


WORD_RE = re.compile(r"\w+", re.UNICODE)


def anonymize_one(description: str, characters: set[str]) -> tuple[str, bool]:
    text = description
    changed = False

    for character in characters:
        pattern = r"(?<!\w)" + re.escape(character) + r"(?!\w)"
        new_text = re.sub(pattern, " ", text, flags=re.UNICODE)
        if new_text != text:
            changed = True
            text = new_text

    # второй проход — по леммам, ловит склонения ("Хоскинса" и т.п.), которые точное совпадение
    # не поймало
    char_lemmas = {lemma(w) for character in characters for w in WORD_RE.findall(character)}

    def strip_by_lemma(m: re.Match) -> str:
        nonlocal changed
        word = m.group(0)
        if lemma(word) in char_lemmas:
            changed = True
            return " "
        return word

    text = WORD_RE.sub(strip_by_lemma, text)

    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[,;:.\-—]+\s*", "", text)
    text = re.sub(r"\s*[,;:\-—]+$", "", text)

    if not text:
        # редкий вырожденный случай (см. history.md/план): character сам по себе — описательная
        # фраза, целиком перекрывающая description (напр. character="ворчливый солдат",
        # description="ворчливый") — вырезание оставило бы пустую строку. Лучше сохранить
        # оригинал с утечкой, чем скормить модели пустой текст.
        return description, False

    return text, changed


def check_leak(text: str, characters: set[str]) -> bool:
    for character in characters:
        pattern = r"(?<!\w)" + re.escape(character) + r"(?!\w)"
        if re.search(pattern, text, flags=re.UNICODE):
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=f"{BASE}/description_anonymization.csv")
    ap.add_argument("--limit", type=int, default=None, help="ограничить число фрагментов (для теста)")
    args = ap.parse_args()

    df = load_fragments(args.limit)
    log.info("фрагментов после жёстких фильтров: %d", len(df))

    grouped = df.groupby("description")["character"].apply(
        lambda s: {c.strip() for c in s if is_real_character(c)}
    )
    n_with_name = int((grouped.apply(len) > 0).sum())
    log.info("уникальных description: %d, из них с хотя бы одним именем/ролью: %d", len(grouped), n_with_name)

    rows = []
    n_changed = 0
    n_residual = 0
    n_empty_reverted = 0
    for description, characters in grouped.items():
        if not characters:
            rows.append((description, description, False))
            continue
        anon, changed = anonymize_one(description, characters)
        if not changed and anon == description and check_leak(description, characters):
            n_empty_reverted += 1
        if changed:
            n_changed += 1
        if check_leak(anon, characters):
            n_residual += 1
        rows.append((description, anon, changed))

    out_df = pd.DataFrame(rows, columns=["description", "description_anon", "changed"])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    log.info("изменено %d из %d уникальных description -> %s", n_changed, len(out_df), out_path)
    log.info("остаточная утечка (точное совпадение) после анонимизации: %d уникальных description", n_residual)
    log.info("вырожденных случаев (анонимизация дала бы пустую строку, оставлен оригинал): %d", n_empty_reverted)


if __name__ == "__main__":
    main()
