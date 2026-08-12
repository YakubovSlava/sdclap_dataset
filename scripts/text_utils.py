"""
Общие текстовые утилиты для пайплайна экстракции (парсинг epub, нарезка на чанки, фильтр
диалоговых абзацев, отсев мусорных description) — используются и продовским
01_extract_speech_fragments.py, и обучающими/фильтрующими скриптами
(train_tfidf_pos_hybrid.py, tfidf_pos_hybrid_filter.py), чтобы не дублировать логику.
"""
import re
from pathlib import Path

from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub


def normalize_for_match(s: str) -> str:
    """Нормализация для проверки "речь реально есть в тексте" — унифицирует типографику
    (кавычки-ёлочки/прямые, тире всех видов, регистр, пробелы), не трогая сами слова. LLM почти
    всегда слегка меняет типографику при копировании реплики (другие кавычки, дефис вместо тире,
    другой регистр после тире-диалога) — байтовое сравнение из-за этого молча теряет валидные
    фрагменты (см. history.md, замер: guard пропускал только 17% реально найденных LLM реплик)."""
    s = s.lower()
    s = re.sub(r"[«»\"“”„]", '"', s)
    s = re.sub(r"[—–]", "-", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def epub_paragraphs(path: Path) -> list[str]:
    book = epub.read_epub(str(path))
    paragraphs = []
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "lxml")
        for tag in soup.find_all(["p", "div"]):
            text = tag.get_text(" ", strip=True)
            if text:
                paragraphs.append(text)
    return paragraphs


def chunk_paragraphs(paragraphs: list[str], max_chars: int) -> list[str]:
    chunks, current, current_len = [], [], 0
    for p in paragraphs:
        if current and current_len + len(p) > max_chars:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(p)
        current_len += len(p)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


# маркеры прямой речи в русском тексте: тире-диалог в начале абзаца/строки (— Реплика.)
# и кавычки-«ёлочки»/обычные кавычки, оборачивающие реплику. Если в чанке нет ни одного —
# в нём по определению нет прямой речи, и гонять его через LLM бессмысленно.
DIALOGUE_MARKER_RE = re.compile(
    r"(?:^|\n)\s*[—–-]\s+\S"  # тире в начале строки/абзаца, за которым текст
    r"|«[^»]+»"  # кавычки-ёлочки
    r'|"[^"\n]{2,}"',  # обычные кавычки вокруг текста без переноса строки
)


def chunk_has_dialogue(chunk_text: str) -> bool:
    return bool(DIALOGUE_MARKER_RE.search(chunk_text))


def filter_dialogue_relevant_paragraphs(paragraphs: list[str], context: int = 1) -> list[str]:
    """Оставляет только абзацы с прямой речью + соседние (контекст для описания манеры речи).

    На фиксированных окнах по max_chars почти каждое окно всё равно содержит хоть одну реплику
    (диалог встречается слишком часто), так что пост-фактум фильтр целых чанков почти не экономит.
    А вот отбросить заведомо нерелевантные абзацы (описания природы, экшн без диалогов и т.п.) до
    нарезки на чанки — даёт ощутимую экономию по объёму текста и числу запросов к LLM.
    """
    has_dialogue = [chunk_has_dialogue(p) for p in paragraphs]
    keep = [False] * len(paragraphs)
    for i, flag in enumerate(has_dialogue):
        if flag:
            for j in range(max(0, i - context), min(len(paragraphs), i + context + 1)):
                keep[j] = True
    return [p for p, k in zip(paragraphs, keep) if k]


# модель (даже с включённым thinking) иногда включает фрагмент без валидного описания манеры,
# подставляя вместо него заглушку — вместо того чтобы просто не включать такой фрагмент, как
# явно требует промпт. Ловим известные паттерны заглушек и такие фрагменты отбрасываем на клиенте.
JUNK_DESCRIPTION_RE = re.compile(
    r"^\s*$"  # пусто
    r"|^[-—–\s]*$"  # только тире/пробелы
    r"|^null$"
    r"|без\s+описани"  # "без описания манеры", "без описания манеры речи" и т.п.
    r"|нет\s+описани"  # "нет описания манеры произнесения"
    r"|не\s+указан"  # "не указано"
    r"|^нет\.?$",  # голое "нет"
    re.IGNORECASE,
)

# нейтральные глаголы речи без окраски — если description состоит только из такого глагола
# (+ опционально короткое местоимение), считаем его невалидным согласно тому же критерию, что
# и в самом промпте. Глаголы с окраской (рявкнул, прошептал, заорал и т.п.) сюда не входят —
# они сами по себе валидное описание манеры.
NEUTRAL_SPEECH_VERBS = {
    "сказал", "сказала", "сказали", "ответил", "ответила", "ответили",
    "спросил", "спросила", "спросили", "произнёс", "произнесла", "произнесли",
    "промолвил", "промолвила", "добавил", "добавила", "заметил", "заметила",
}
PRONOUNS = {"я", "он", "она", "они", "мы", "ты", "вы"}


def _normalize_for_dup_check(s: str) -> str:
    s = re.sub(r"^[—–-]\s*", "", s.strip())  # тире-диалог в начале
    return re.sub(r"[^\w\s]", "", s).strip().lower()


def is_junk_description(description: str, speech: str | None = None) -> bool:
    d = (description or "").strip()
    if JUNK_DESCRIPTION_RE.search(d):
        return True
    tokens = d.lower().rstrip(".").split()
    if tokens and tokens[0] in NEUTRAL_SPEECH_VERBS and (len(tokens) == 1 or (len(tokens) == 2 and tokens[1] in PRONOUNS)):
        return True
    # модель иногда просто дублирует speech в description вместо описания манеры —
    # если это одно и то же (без учёта тире/пунктуации), реальной информации там нет
    if speech and _normalize_for_dup_check(d) == _normalize_for_dup_check(speech):
        return True
    return False
