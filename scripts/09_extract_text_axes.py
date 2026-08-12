#!/usr/bin/env python3
"""
Разметка текстового `description` фрагмента по акустическим осям — текстовая половина сверки
«описание ↔ аудио» (шаг 1.4b, ось B). Для каждой реплики LLM определяет, какое звучание ОБЕЩАЕТ
авторское описание, по тем же осям, что меряются из сигнала в `scripts/04_extract_audio_features.py`:

- **loudness** — громкость (low = тихо/шёпот, high = громко/крик)
- **pitch**    — высота/подвижность тона (low = ровно/монотонно, high = звонко/взвинченно)
- **tempo**    — темп (low = медленно/растянуто, high = быстро/скороговоркой)
- **tension**  — напряжённость голоса (low = мягко/спокойно, high = сдавленно/зло/надрывно)

Каждая ось — одно из {low, neutral, high}. **neutral = в описании нет направленного сигнала по
этой оси** (а не «средне»): сверка проверяет только НЕнейтральные оси, поэтому важно, чтобы модель
не выдумывала направление там, где его нет. Отрицание и контекст («не повышая голоса», «без всякого
выражения») модель обязана учитывать — ради этого и берём LLM, а не словарь.

Оптимизация стоимости: `description` массово повторяются между фрагментами и чтецами, поэтому
размечаются УНИКАЛЬНЫЕ строки описания, а результат джойнится обратно по `description`. Это на
порядок меньше вызовов LLM, чем строк в манифесте.

Сервер — тот же локальный llama-server/OpenAI-совместимый эндпоинт, что и в шаге 1.2
(`docs/llm_setup.md`). Вход по умолчанию — манифест шага 03 (`dataset.csv`), берётся колонка
`description`. Выход — CSV `description + loudness + pitch + tempo + tension + axes_status`
(по умолчанию `data/processed/audio_fragments/text_axes.csv`); присоединяется к фрагментам по
`description` (см. `--join-output`).

Примеры:
    # разведочный прогон на 50 уникальных описаниях
    python scripts/05_extract_text_axes.py --limit 50
    # полный прогон
    python scripts/05_extract_text_axes.py --workers 16
    # докатить недостающие
    python scripts/05_extract_text_axes.py --resume
    # + сразу разложить метки по fragment_id
    python scripts/05_extract_text_axes.py --join-output data/processed/audio_fragments/dataset_axes.csv
"""
import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("extract_text_axes")

DEFAULT_MANIFEST = "data/processed/audio_fragments/dataset.csv"
DEFAULT_OUTPUT = "data/processed/audio_fragments/text_axes.csv"
AXES = ["loudness", "pitch", "tempo", "tension"]
LEVELS = ["low", "neutral", "high"]

_axis_prop = {
    "type": "string",
    "enum": LEVELS,
}
AXES_SCHEMA = {
    "type": "object",
    "properties": {
        "loudness": {**_axis_prop, "description": "Громкость: low=тихо/шёпот, high=громко/крик, neutral=нет сигнала."},
        "pitch": {**_axis_prop, "description": "Высота/подвижность тона: low=ровно/монотонно, high=звонко/взвинченно, neutral=нет сигнала."},
        "tempo": {**_axis_prop, "description": "Темп: low=медленно/растянуто, high=быстро/скороговоркой, neutral=нет сигнала."},
        "tension": {**_axis_prop, "description": "Напряжённость голоса: low=мягко/спокойно, high=сдавленно/зло/надрывно, neutral=нет сигнала."},
    },
    "required": AXES,
}

SYSTEM_PROMPT = """Тебе дают короткое авторское ОПИСАНИЕ того, как персонаж произнёс реплику \
(тон, громкость, темп, эмоция, манера). Твоя задача — определить, какое ЗВУЧАНИЕ это описание \
обещает, по четырём независимым осям. Для каждой оси выбери ровно одно значение: low, neutral \
или high.

Оси:
- loudness (громкость): low — тихо, шёпотом, вполголоса, еле слышно; high — громко, крик, вопль, \
рёв, во весь голос.
- pitch (высота и подвижность тона): low — ровно, монотонно, глухо, без интонаций; high — звонко, \
на высокой ноте, взвинченно, визгливо, с резкими перепадами.
- tempo (темп речи): low — медленно, растягивая, с паузами, вяло; high — быстро, торопливо, \
скороговоркой, взахлёб.
- tension (напряжённость голоса): low — мягко, спокойно, расслабленно, ласково; high — сдавленно, \
сквозь зубы, зло, надрывно, дрожащим от напряжения голосом.

ГЛАВНОЕ ПРАВИЛО: neutral означает, что в описании НЕТ направленного сигнала по этой оси — не \
«средне», а «про это здесь ничего не сказано». Не выдумывай направление там, где его нет. Большая \
часть осей для короткого описания будет neutral — это нормально.

Учитывай отрицание и контекст: «не повышая голоса» → loudness low; «без всякого выражения» → \
pitch low; «спокойно, но со сталью в голосе» → tension high, loudness neutral. Ориентируйся на \
реальное звучание, а не на эмоцию саму по себе (например «радостно» само по себе не задаёт \
громкость — если нет других слов, loudness=neutral).

Верни строго JSON с полями loudness, pitch, tempo, tension."""


def call_llm(session: requests.Session, server_url: str, model: str, description: str,
             timeout: float, max_retries: int) -> dict[str, str]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": description},
        ],
        "temperature": 0.0,
        # max_tokens НЕ ограничиваем: у сервера включён thinking (как в шаге 1.2) — жёсткий потолок
        # обрезает размышление до того, как модель выдаст JSON, и content приходит пустым.
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "speech_axes", "schema": AXES_SCHEMA, "strict": True},
        },
    }
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.post(f"{server_url}/v1/chat/completions", json=payload, timeout=timeout)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            if not content or not content.strip():
                raise ValueError(f"пустой content (finish_reason={resp.json()['choices'][0].get('finish_reason')})")
            data = json.loads(content)
            # грамматика гарантирует enum, но подстрахуемся: неизвестное → neutral
            return {ax: (data.get(ax) if data.get(ax) in LEVELS else "neutral") for ax in AXES}
        except Exception as e:  # noqa: BLE001
            last_err = e
            log.warning("llm call failed (attempt %d/%d): %s", attempt, max_retries, e)
            time.sleep(min(2 ** attempt, 20))
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_err}")


def process_one(session, args, description: str) -> dict:
    row: dict = {"description": description}
    try:
        axes = call_llm(session, args.server_url, args.model, description, args.timeout, args.max_retries)
        row.update(axes)
        row["axes_status"] = "ok"
    except Exception as e:  # noqa: BLE001
        for ax in AXES:
            row[ax] = "neutral"
        row["axes_status"] = f"error:{type(e).__name__}"
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help="Таблица уникальных описаний с осями.")
    ap.add_argument("--join-output", default=None,
                    help="Если задан — дополнительно записать метки, разложенные по fragment_id.")
    ap.add_argument("--readers", default=None,
                    help="Только описания этих чтецов (через запятую) — для разведочных прогонов.")
    ap.add_argument("--server-url", default="http://127.0.0.1:1234",
                    help="База эндпоинта (без /v1); скрипт добавит /v1/chat/completions.")
    ap.add_argument("--model", default="gemma-4-e4b")
    ap.add_argument("--workers", type=int, default=16, help="должно совпадать с -np сервера")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--ramp-delay", type=float, default=0.5,
                    help="пауза между первыми --workers запросами (холодный залп на сервер)")
    ap.add_argument("--offset", type=int, default=0,
                    help="пропустить первые N уникальных описаний — для параллельного прогона на "
                         "нескольких машинах непересекающимися диапазонами --offset/--limit "
                         "(порядок описаний детерминирован сортировкой)")
    ap.add_argument("--limit", type=int, default=None, help="разметить только N уникальных описаний после --offset")
    ap.add_argument("--resume", action="store_true", help="пропустить описания, уже размеченные в --output")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    if args.readers:
        wanted = {r.strip() for r in args.readers.split(",")}
        df = df[df["reader"].isin(wanted)].reset_index(drop=True)
    desc = df["description"].dropna().astype(str).str.strip()
    desc = desc[desc != ""]
    # сортировка -> порядок одинаков на всех машинах, диапазоны --offset/--limit не пересекаются
    uniq = pd.Index(sorted(desc.unique()))
    log.info("уникальных описаний: %d (из %d строк манифеста)", len(uniq), len(df))

    if args.offset:
        uniq = uniq[args.offset:]
    if args.limit:
        uniq = uniq[: args.limit]
    if args.resume and Path(args.output).exists():
        done = set(pd.read_csv(args.output, usecols=["description"])["description"].astype(str))
        before = len(uniq)
        uniq = uniq[~uniq.isin(done)]
        log.info("resume: пропущено %d уже размеченных", before - len(uniq))
    if len(uniq) == 0:
        log.info("нечего размечать")
    else:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=args.workers, pool_maxsize=args.workers)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        rows: list[dict] = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = []
            for i, d in enumerate(uniq):
                if i < args.workers and args.ramp_delay:
                    time.sleep(args.ramp_delay)
                futures.append(pool.submit(process_one, session, args, d))
            for fut in tqdm(as_completed(futures), total=len(futures), desc="axes", unit="descr"):
                rows.append(fut.result())

        out = pd.DataFrame(rows, columns=["description", *AXES, "axes_status"])
        n_err = int((out["axes_status"] != "ok").sum())
        if n_err:
            log.warning("ошибок разметки: %d", n_err)
        out_path = Path(args.output)
        if args.resume and out_path.exists():
            out.to_csv(out_path, mode="a", header=False, index=False)
        else:
            out.to_csv(out_path, index=False)
        log.info("готово: %d уникальных описаний -> %s", len(out), out_path)

    # разложить метки по fragment_id (по всему набору описаний из --output)
    if args.join_output and Path(args.output).exists():
        axes_tbl = pd.read_csv(args.output)
        merged = df.merge(axes_tbl, on="description", how="left")
        cols = ["fragment_id", "description", *AXES, "axes_status"]
        cols = [c for c in cols if c in merged.columns]
        merged[cols].to_csv(args.join_output, index=False)
        log.info("джойн по fragment_id -> %s (%d строк)", args.join_output, len(merged))


if __name__ == "__main__":
    main()
