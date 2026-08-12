"""
Экспериментальный POS-фильтр абзацев по наличию описания манеры звучания речи.

Статус: НЕ подключён к продовскому пайплайну (scripts/01_extract_speech_fragments.py).
Это отдельный модуль с логикой, которую отработали в сессии экспериментов
по ускорению (см. dataset_collection в history.md/plan.md, раздел 1.2) —
сохраняем, чтобы не потерять, окончательное решение о внедрении не принято.

Идея: применяется ПОСЛЕ уже существующего filter_dialogue_relevant_paragraphs
(тире/кавычки) и ПЕРЕД нарезкой на чанки — дополнительно отсеивает абзацы,
где диалог есть, но по морфологии/синтаксису не похоже, что рядом есть
описание манеры звучания (наречие, деепричастие, глагол речи с окраской,
существительное звучания, творительный падеж, предлог+сущ).

Два варианта:
- v1 (extended=False): наречие/деепричастие + узкий список глаголов/сущ.
- v2 (extended=True): v1 + творительный падеж существительного (граммема ablt,
  без привязки к конкретному слову) + предлог (с/со/без) + сущ./прил.

На реальных данных (50 книг, 15259 валидных фрагментов из
data/processed/speech_fragments/, измерено строгим методом — см.
для статьи/pos_filter_experiment.md, включая поправку об исправлении
более раннего, ошибочного замера):
  v1: ~11.4% экономии чанков, ~87.5% recall
  v2: ~7.6% экономии чанков, ~90.7% recall
v2 строго доминирует v1 на любом размере окна — если внедрять, то только v2.
ВАЖНО: TF-IDF+LogReg классификатор превосходит POS v2 на каждой сопоставимой
точке экономии (см. для статьи/tfidf_classifier_experiment.md) — POS остаётся
кандидатом только если приоритет простота/отсутствие обучающей выборки, а не
максимальный recall.

Сужение окна анализа (tag_window) даёт больше экономии ценой recall —
развёртка (v2): 15 симв. → 50%/57%, 150 симв. → 25%/83%,
весь абзац → 7.6%/91%. Компромисса с одновременно высокой экономией И
высоким recall не нашлось — либо-либо. Подробности — в истории сессии.

Требует pymorphy3 + pymorphy3-dicts-ru (см. requirements.txt).
"""
import re

import pymorphy3

morph = pymorphy3.MorphAnalyzer()

# базовый список — глаголы речи с явной окраской (не "сказал"/"ответил" и т.п.)
BASE_SPEECH_MANNER_LEMMAS = {
    "шептать", "прошептать", "зашептать",
    "крикнуть", "кричать", "закричать", "выкрикнуть",
    "заорать", "орать",
    "гаркнуть", "рявкнуть",
    "пробормотать", "забормотать", "бормотать",
    "простонать", "стонать",
    "взвыть", "выть",
    "взвизгнуть", "визжать",
    "прохрипеть", "хрипеть",
    "процедить",
    "пролепетать", "лепетать",
    "выдохнуть",
    "всхлипнуть", "всхлипывать",
    "промычать", "мычать",
    "буркнуть",
    "прорычать", "рычать",
    "провизжать",
    "пропищать", "пищать",
}

# найдены частотным анализом реальных description на 10 книгах (глаголы,
# которые модель реально использует, но не попали в базовый список) —
# см. историю сессии для примеров и частот
EXTRA_SPEECH_MANNER_LEMMAS = {
    "проворчать", "воскликнуть", "завопить", "прошипеть", "шепнуть",
    "вскричать", "пропеть", "выдавить", "отрезать", "фыркнуть", "хмыкнуть",
    "пробасить", "просипеть", "скомандовать", "напеть", "прокричать",
    "взреветь",
}

SPEECH_MANNER_LEMMAS = BASE_SPEECH_MANNER_LEMMAS | EXTRA_SPEECH_MANNER_LEMMAS

# узкий список существительных звучания (v1 и v2 базово)
SOUND_NOUN_LEMMAS = {"голос", "тон", "шёпот", "крик", "хрип", "интонация", "дрожь"}

# предлоги для паттерна "предлог + сущ./прил." (v2): "с гордостью", "без энтузиазма"
PREPS = {"с", "со", "без"}

WORD_RE = re.compile(r"[а-яёА-ЯЁ]+")

_morph_cache: dict[str, tuple[str, str, object]] = {}


def _tag_words(text: str) -> list[tuple[str, str, str, object]]:
    """Токенизация + морфоразбор с кэшем (одно слово — один разбор на весь процесс)."""
    out = []
    for w in WORD_RE.findall(text):
        wl = w.lower()
        if wl not in _morph_cache:
            p = morph.parse(w)[0]
            _morph_cache[wl] = (p.tag.POS, p.normal_form, p.tag)
        pos, lemma, tag = _morph_cache[wl]
        out.append((wl, pos, lemma, tag))
    return out


def pos_pass(text: str, extended: bool = True) -> bool:
    """True, если в тексте есть морфологический/синтаксический признак описания
    манеры звучания речи. extended=False — v1, extended=True — v2 (рекомендуется)."""
    words = _tag_words(text)
    for i, (wl, pos, lemma, tag) in enumerate(words):
        if pos in ("ADVB", "GRND"):
            return True
        if pos == "VERB" and lemma in SPEECH_MANNER_LEMMAS:
            return True
        if pos == "NOUN":
            if lemma in SOUND_NOUN_LEMMAS:
                return True
            if extended and "ablt" in tag:  # творительный падеж
                return True
        if extended and wl in PREPS and i + 1 < len(words) and words[i + 1][1] in ("NOUN", "ADJF"):
            return True
    return False


# маркеры реплики/тире-интро — используются для сужения окна анализа вокруг тега.
# ВНИМАНИЕ: сужение окна сильно роняет recall (проверено эмпирически, см. историю
# сессии) — 70 симв. даёт 81% recall вместо 91% на полном абзаце. Использовать
# window=None (весь абзац) по умолчанию, сужать только осознанно ради экономии.
_SPAN_RE = re.compile(r'«[^»]+»|"[^"\n]{2,}"|(?:^|\n)\s*[—–-]\s+[^—–\n]*')


def tag_window(paragraph: str, k: int | None) -> str:
    """Текст в пределах k символов сразу после каждого сегмента реплики.
    k=None — весь абзац без сужения (рекомендуется, см. предупреждение выше)."""
    if k is None:
        return paragraph
    spans = list(_SPAN_RE.finditer(paragraph))
    if not spans:
        return paragraph
    zones = []
    for i, sp in enumerate(spans):
        end = sp.end()
        next_start = spans[i + 1].start() if i + 1 < len(spans) else len(paragraph)
        zone_end = min(end + k, next_start)
        zones.append(paragraph[end:zone_end])
    return " ".join(zones)


def filter_paragraphs_pos(paragraphs: list[str], extended: bool = True, window: int | None = None) -> list[str]:
    """Применить POS-фильтр к списку уже отобранных (filter_dialogue_relevant_paragraphs)
    абзацев — оставляет только те, где похоже на описание манеры звучания."""
    return [p for p in paragraphs if pos_pass(tag_window(p, window), extended=extended)]
