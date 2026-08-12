#!/usr/bin/env python3
"""
Нарезка аудиофрагментов на диск (после шага 1.4, матчинга).

Вход — результат матчинга `data/processed/matched_fragments/matched_fragments.csv`
(колонки `path_audio`, `match_start_ts`, `match_end_ts`, `match_status`, метаданные фрагмента).
Для каждой строки со `match_status == "ok"` вырезается участок исходного mp3 книги по таймкодам
матчинга и сохраняется в FLAC (mono, 24000 Гц) через ffmpeg.

Полный набор без какого-либо отсева по качеству: пороги по `match_distance_norm`/WER здесь НЕ
применяются (WER-фильтр — отдельный шаг 1.5). Отбрасываются только строки без валидного матча
(`match_status != "ok"`, у них нет таймкодов).

Исходный mp3 книги достраивается как `<path_audio>/raw/*.mp3` (в каждой директории книги ровно
один mp3 — проверено на корпусе).

На выходе — CSV-датасет по аналогии с `matched_fragments.csv`: все исходные колонки строки плюс
`fragment_id`, `audio_fragment_path`, `fragment_duration`, `cut_status`. Одна строка = один
нарезанный фрагмент.

Резюмируемость: если целевой FLAC уже существует и непустой — фрагмент пропускается (перекодировать
заново — `--overwrite`). Для параллельного прогона на двух устройствах — `--offset`/`--limit`
(режут упорядоченный список ok-строк), манифест при этом пишется с суффиксом среза, чтобы не
затирать чужой.

Пример:
    python scripts/03_cut_audio_fragments.py --workers 16
    # два устройства:
    python scripts/03_cut_audio_fragments.py --offset 0      --limit 153000 --workers 16
    python scripts/03_cut_audio_fragments.py --offset 153000 --limit 153500 --workers 16
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm

DEFAULT_MATCHED_CSV = "data/processed/matched_fragments/matched_fragments.csv"
DEFAULT_OUT_DIR = "data/processed/audio_fragments"
SAMPLE_RATE = 24000


def safe_name(s: str) -> str:
    """Файловобезопасное имя (та же логика, что в 02_match_fragments_to_audio.py)."""
    return re.sub(r"[^0-9a-zA-Zа-яА-Я]+", "_", str(s)).strip("_")[-150:]


def source_mp3(path_audio: str, _cache: dict[str, str | None] = {}) -> str | None:
    """Путь к исходному mp3 книги внутри `<path_audio>/raw/`. Кэшируется по path_audio."""
    if path_audio not in _cache:
        hits = glob.glob(os.path.join(path_audio, "raw", "*.mp3"))
        _cache[path_audio] = hits[0] if len(hits) == 1 else (hits[0] if hits else None)
    return _cache[path_audio]


def verify_header(path: str, expected_dur: float, tol: float = 0.15) -> str | None:
    """Быстрая проверка целостности FLAC по заголовку (без декодирования, уровень 1).

    Возврат None если всё хорошо, иначе строка-причина. Для глубокой проверки (полный декод +
    сверка MD5) — отдельный `scripts/03b_verify_fragments.py --deep`.
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return "missing_or_empty"
    try:
        info = sf.info(path)
    except Exception as e:
        return f"header_unreadable:{type(e).__name__}"
    if info.frames == 0:
        return "zero_frames"
    if abs(info.frames / info.samplerate - expected_dur) > tol:
        return "dur_mismatch"
    return None


def cut_one(src: str, start: float, end: float, out_path: Path, sample_rate: int) -> tuple[bool, str]:
    """Вырезать [start, end] из src в FLAC (mono, sample_rate). Возврат (успех, сообщение)."""
    dur = end - start
    if not (dur > 0):
        return False, f"bad_duration({dur})"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".flac.part")
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-y",
        "-ss", f"{start:.3f}", "-i", src, "-t", f"{dur:.3f}",
        "-ac", "1", "-ar", str(sample_rate), "-c:a", "flac", "-f", "flac",
        str(tmp),
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        return False, "ffmpeg_timeout"
    if r.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        return False, f"ffmpeg_fail({r.returncode}):{r.stderr.strip()[:120]}"
    tmp.replace(out_path)
    return True, "ok"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matched-csv", default=DEFAULT_MATCHED_CSV)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Корень для нарезанных FLAC.")
    ap.add_argument("--manifest", default=None,
                    help="Путь итогового CSV-датасета (по умолчанию <out-dir>/dataset.csv, "
                         "с суффиксом среза при --offset/--limit).")
    ap.add_argument("--sample-rate", type=int, default=SAMPLE_RATE)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--offset", type=int, default=0, help="Пропустить первые N ok-строк.")
    ap.add_argument("--limit", type=int, default=None, help="Обработать не более N ok-строк после offset.")
    ap.add_argument("--overwrite", action="store_true", help="Перекодировать уже существующие FLAC.")
    ap.add_argument("--verify", action="store_true",
                    help="После нарезки — быстрая проверка целостности по заголовку FLAC (уровень 1). "
                         "Глубокая проверка (полный декод) — scripts/03b_verify_fragments.py --deep.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    df = pd.read_csv(args.matched_csv, dtype={"chunk_id": str})
    total_rows = len(df)
    df = df[df["match_status"] == "ok"].reset_index(drop=True)
    print(f"Прочитано {total_rows} строк, ok-фрагментов: {len(df)}", file=sys.stderr)

    # Глобальный fragment_id по всему ok-набору — стабилен независимо от среза offset/limit.
    df["fragment_id"] = [f"{i:07d}" for i in range(len(df))]

    sliced = df.iloc[args.offset: (args.offset + args.limit) if args.limit is not None else None].copy()
    print(f"Срез к обработке: {len(sliced)} (offset={args.offset}, limit={args.limit})", file=sys.stderr)

    # Целевые пути — чистая функция от строки, считаем заранее.
    def out_path_for(row) -> Path:
        return out_dir / safe_name(row["reader"]) / safe_name(row["book_name"]) / f"{row['fragment_id']}.flac"

    sliced["audio_fragment_path"] = sliced.apply(lambda r: str(out_path_for(r)), axis=1)
    sliced["fragment_duration"] = (sliced["match_end_ts"] - sliced["match_start_ts"]).round(3)

    jobs = []  # (idx, src, start, end, out_path)
    statuses: dict[int, str] = {}
    for idx, row in sliced.iterrows():
        out_path = Path(row["audio_fragment_path"])
        if not args.overwrite and out_path.exists() and out_path.stat().st_size > 0:
            statuses[idx] = "skipped_exists"
            continue
        src = source_mp3(row["path_audio"])
        if src is None:
            statuses[idx] = "missing_source"
            continue
        jobs.append((idx, src, float(row["match_start_ts"]), float(row["match_end_ts"]), out_path))

    print(f"К нарезке: {len(jobs)}, пропущено (уже есть): "
          f"{sum(v == 'skipped_exists' for v in statuses.values())}, "
          f"без исходника: {sum(v == 'missing_source' for v in statuses.values())}", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(cut_one, s, st, en, op, args.sample_rate): idx
                for (idx, s, st, en, op) in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="cutting", unit="frag"):
            idx = futs[fut]
            ok, msg = fut.result()
            statuses[idx] = "ok" if ok else msg

    sliced["cut_status"] = sliced.index.map(statuses).fillna("unknown")

    ok_n = (sliced["cut_status"].isin(["ok", "skipped_exists"])).sum()
    fail = sliced[~sliced["cut_status"].isin(["ok", "skipped_exists"])]
    print(f"Готово: {ok_n}/{len(sliced)} успешно, ошибок: {len(fail)}", file=sys.stderr)
    if len(fail):
        print("Примеры ошибок:", file=sys.stderr)
        for _, r in fail.head(5).iterrows():
            print(f"  {r['fragment_id']} {r['cut_status']}", file=sys.stderr)

    if args.manifest:
        manifest = Path(args.manifest)
    else:
        suffix = "" if (args.offset == 0 and args.limit is None) else f"_off{args.offset}_lim{args.limit}"
        manifest = out_dir / f"dataset{suffix}.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    sliced.to_csv(manifest, index=False)
    print(f"Манифест датасета: {manifest} ({len(sliced)} строк)", file=sys.stderr)

    if args.verify:
        to_check = sliced[sliced["cut_status"].isin(["ok", "skipped_exists"])]
        print(f"Проверка целостности (заголовок FLAC): {len(to_check)} фрагментов...", file=sys.stderr)
        bad = []
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(verify_header, r["audio_fragment_path"], float(r["fragment_duration"])): r["fragment_id"]
                    for _, r in to_check.iterrows()}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="verify", unit="frag"):
                reason = fut.result()
                if reason is not None:
                    bad.append((futs[fut], reason))
        if bad:
            print(f"ПРОБЛЕМНЫХ: {len(bad)} (примеры: {bad[:5]})", file=sys.stderr)
        else:
            print("Проверка пройдена: все фрагменты целы.", file=sys.stderr)


if __name__ == "__main__":
    main()
