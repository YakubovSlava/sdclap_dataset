#!/usr/bin/env python3
"""
Проверка целостности нарезанных аудиофрагментов (после `scripts/03_cut_audio_fragments.py`).

Два уровня проверки:

- **Уровень 1 (по умолчанию, быстро, без декодирования)** — чтение только заголовка FLAC
  (`STREAMINFO`) через `soundfile.info`. Проверяет: файл существует и читается, число сэмплов
  ненулевое, длительность из заголовка совпадает с `fragment_duration` из манифеста (в пределах
  `--tol` секунд). Ловит пустые/нефинализированные (обрубки), нечитаемые заголовки, грубое
  несоответствие длины. НЕ ловит порчу внутри аудиокадров при валидном заголовке.

- **Уровень 2 (`--deep`, полный декод)** — `flac -t`, декодирует каждый файл и сверяет
  встроенный MD5 несжатого аудио из заголовка. Золотой стандарт «аудио цело до последнего
  сэмпла», но требует полного чтения (параллелится).

Вход — манифест `dataset.csv` шага 03 (по умолчанию `data/processed/audio_fragments/dataset.csv`).
Проверяются строки со `cut_status in {ok, skipped_exists}`. Проблемные файлы выводятся в консоль и
пишутся в CSV (`--report`, по умолчанию рядом с манифестом: `<manifest>.verify_bad.csv`).

Пример:
    python scripts/03b_verify_fragments.py                       # уровень 1
    python scripts/03b_verify_fragments.py --deep --workers 16   # + полный декод
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm

DEFAULT_MANIFEST = "data/processed/audio_fragments/dataset.csv"
OK_STATUSES = {"ok", "skipped_exists"}


def check_header(path: str, expected_dur: float, tol: float) -> str | None:
    """Уровень 1: только заголовок. Возврат None если всё хорошо, иначе строка-причина."""
    p = Path(path)
    if not p.exists():
        return "missing"
    if p.stat().st_size == 0:
        return "empty"
    try:
        info = sf.info(path)
    except Exception as e:
        return f"header_unreadable:{type(e).__name__}"
    if info.frames == 0:
        return "zero_frames"
    got = info.frames / info.samplerate
    if abs(got - expected_dur) > tol:
        return f"dur_mismatch(got={got:.2f},exp={expected_dur:.2f})"
    return None


def check_deep(path: str) -> str | None:
    """Уровень 2: полный декод + сверка MD5 через `flac -t`."""
    if not Path(path).exists():
        return "missing"
    try:
        r = subprocess.run(["flac", "-t", "--silent", path],
                           capture_output=True, text=True, timeout=120)
    except FileNotFoundError:
        print("ОШИБКА: не найден бинарник `flac` (нужен для --deep)", file=sys.stderr)
        raise
    except subprocess.TimeoutExpired:
        return "flac_test_timeout"
    if r.returncode != 0:
        return f"flac_test_fail:{r.stderr.strip()[:120]}"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--deep", action="store_true", help="Уровень 2: полный декод + сверка MD5 (flac -t).")
    ap.add_argument("--tol", type=float, default=0.15, help="Допуск расхождения длительности, с (уровень 1).")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--report", default=None, help="CSV с проблемными файлами (по умолчанию <manifest>.verify_bad.csv).")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    df = df[df["cut_status"].isin(OK_STATUSES)].reset_index(drop=True)
    print(f"К проверке: {len(df)} фрагментов, уровень {'2 (deep)' if args.deep else '1 (header)'}",
          file=sys.stderr)

    rows = list(df[["fragment_id", "audio_fragment_path", "fragment_duration"]].itertuples(index=False))
    bad: list[tuple[str, str, str]] = []  # (fragment_id, path, reason)

    def work(t) -> tuple[str, str, str | None]:
        reason = check_deep(t.audio_fragment_path) if args.deep \
            else check_header(t.audio_fragment_path, float(t.fragment_duration), args.tol)
        return t.fragment_id, t.audio_fragment_path, reason

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, t) for t in rows]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="verify", unit="frag"):
            fid, path, reason = fut.result()
            if reason is not None:
                bad.append((fid, path, reason))

    print(f"\nПроверено: {len(rows)}, проблемных: {len(bad)}", file=sys.stderr)
    if bad:
        from collections import Counter
        cats = Counter(r.split("(")[0].split(":")[0] for _, _, r in bad)
        print("По категориям:", dict(cats), file=sys.stderr)
        for fid, path, reason in bad[:10]:
            print(f"  {fid} {reason}  {path}", file=sys.stderr)
        report = Path(args.report) if args.report else Path(str(args.manifest) + ".verify_bad.csv")
        pd.DataFrame(bad, columns=["fragment_id", "audio_fragment_path", "reason"]).to_csv(report, index=False)
        print(f"Отчёт: {report}", file=sys.stderr)
    else:
        print("Все фрагменты в порядке.", file=sys.stderr)


if __name__ == "__main__":
    main()
