#!/usr/bin/env python3
"""
Извлечение базовых акустических признаков из нарезанных аудиофрагментов
(после `scripts/03_cut_audio_fragments.py` / проверки `03b_verify_fragments.py`).

Только «сырые» сигнальные фичи — без моделей (SER, VAD, диаризация и т.п.) и без нормализации.
Нормализация на диктора и любая логика фильтрации/сверки с текстовым описанием — отдельные шаги,
которые работают уже поверх этой таблицы. Здесь задача одна: для каждого фрагмента посчитать
интерпретируемые числа и сложить их в CSV.

Набор фич (см. `FEATURE_COLUMNS`), сгруппированный по назначению:

- **Громкость и динамика** (`rms_*`, `loudness_dbfs`, `peak_dbfs`, `crest_db`, `dyn_range_db`) —
  уровень и его разброс; динамический диапазон ~ выразительность подачи.
- **Тишина, паузы, границы** (`silence_ratio`, `lead/trail_silence_s`, `n_pauses`,
  `edge_start_ratio`, `edge_end_ratio`) — брак нарезки: обрубленная на полуслове речь (высокая
  энергия у самого края) или лишняя тишина/паддинг.
- **Клиппинг** (`clipping_ratio`) — перегруз/порча записи.
- **Питч F0** (`f0_*`, `voiced_ratio`) — высота и её подвижность (в полутонах, что сопоставимо
  между дикторами); `voiced_ratio` — грубый индикатор «речь vs не-речь».
- **Спектр/тембр** (`spec_*`, `zcr_mean`, `hf_energy_ratio`) — речь vs музыка/шум/эффекты
  (`spec_flatness_mean`), яркость/напряжённость голоса (`hf_energy_ratio`, центроид).
- **Гармоничность** (`harmonic_ratio`) — доля гармонической составляющей (HPSS); фон-музыка и
  шум смещают её.
- **Ритм** (`onset_rate`) — грубый сигнальный прокси темпа (число онсетов в секунду), без ASR.

Опционально `--egemaps` — дополнительно 88 фич eGeMAPSv02 через openSMILE (колонки `egm_*`):
стандартный валидированный SER-набор с нормальным F0 в полутонах, loudness, HNR, jitter/shimmer,
спектральным наклоном (alphaRatio/Hammarberg — вокальное усилие, устойчивое к нормализации
громкости). openSMILE на C++, не сегфолтит как numba-pyin. Плюс `words_per_sec` — прокси темпа из
манифеста (ASR-слова / длительность), надёжнее сигнального `onset_rate`.

Все фичи считаются на нативном sr фрагмента (24 кГц моно у текущего датасета). Значения, которые
невозможно посчитать (нет озвученных кадров и т.п.), пишутся как пустые (NaN).

Вход — манифест `dataset.csv` шага 03. Обрабатываются строки со `cut_status in {ok,
skipped_exists}`. Выход — CSV `fragment_id + фичи + feat_status` (по умолчанию рядом с манифестом:
`audio_features.csv`), джойнится обратно по `fragment_id`.

Примеры:
    # разведочный прогон на паре чтецов
    python scripts/04_extract_audio_features.py --readers Александр_Клюквин,Владимир_Самойлов --limit 500
    # полный прогон
    python scripts/04_extract_audio_features.py --workers 12
    # докатить недостающие (пропустить уже посчитанные)
    python scripts/04_extract_audio_features.py --resume
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

DEFAULT_MANIFEST = "data/processed/audio_fragments/dataset.csv"
DEFAULT_OUTPUT = "data/processed/audio_fragments/audio_features.csv"
OK_STATUSES = {"ok", "skipped_exists"}

# --- параметры извлечения (при 24 кГц: n_fft ~43 мс, hop ~10.7 мс) ---
N_FFT = 1024
HOP = 256
F0_MIN = 65.0       # нижняя граница поиска F0, Гц (взрослый голос)
F0_MAX = 500.0      # верхняя граница (запас на экспрессию/женский голос)
HF_CUTOFF = 4000.0  # порог «высоких частот» для hf_energy_ratio, Гц
SILENCE_REL_DB = 35.0  # кадр считается тишиной, если он на столько дБ тише пикового кадра
EDGE_WIN_S = 0.10   # окно у краёв для детекции обрубленной речи, с
CLIP_THRESH = 0.999  # |сэмпл| выше — считаем клиппингом

FEATURE_COLUMNS = [
    "duration_s",
    # громкость / динамика
    "rms_mean", "rms_std", "rms_max",
    "loudness_dbfs", "peak_dbfs", "crest_db", "dyn_range_db",
    "clipping_ratio",
    # тишина / паузы / границы
    "silence_ratio", "lead_silence_s", "trail_silence_s", "n_pauses",
    "edge_start_ratio", "edge_end_ratio",
    # питч
    "f0_mean_hz", "f0_median_hz", "voiced_ratio",
    "f0_std_semitones", "f0_range_semitones",
    # спектр / тембр
    "spec_centroid_mean", "spec_centroid_std", "spec_bandwidth_mean",
    "spec_rolloff_mean", "spec_flatness_mean", "zcr_mean", "hf_energy_ratio",
    # гармоничность / ритм
    "harmonic_ratio", "onset_rate",
]


def _hz_to_semitones(f0: np.ndarray) -> np.ndarray:
    """Перевод частот (Гц) в полутоны относительно 1 Гц — для разбросов межнотовые интервалы
    сопоставимы между дикторами независимо от абсолютной высоты голоса."""
    return 12.0 * np.log2(f0)


def _f0_autocorr(y: np.ndarray, sr: int, fmin: float = F0_MIN, fmax: float = F0_MAX,
                 frame_s: float = 0.04, hop_s: float = 0.01,
                 voiced_thr: float = 0.3, energy_floor: float = 1e-4
                 ) -> tuple[np.ndarray, float]:
    """Покадровый F0 через автокорреляцию (чистый numpy, без numba).

    Сознательно НЕ используем `librosa.pyin`: его numba-ядро сегфолтит в текущем окружении
    (llvmlite/threading) — нативный краш рушит весь ProcessPool-прогон, а не ловится как
    исключение. Для наших грубых фич (разброс F0 в полутонах, доля озвученных кадров) точности
    автокорреляции достаточно. Возврат: (массив F0 озвученных кадров в Гц, voiced_ratio)."""
    fl, hl = int(frame_s * sr), int(hop_s * sr)
    if len(y) < fl:
        return np.empty(0), 0.0
    lag_min, lag_max = int(sr / fmax), int(sr / fmin)
    win = np.hanning(fl)
    f0s: list[float] = []
    voiced = total = 0
    for st in range(0, len(y) - fl + 1, hl):
        total += 1
        fr = y[st:st + fl] * win
        if np.sqrt(np.mean(fr ** 2)) < energy_floor:
            continue
        ac = np.correlate(fr, fr, "full")[fl - 1:]
        if ac[0] <= 0:
            continue
        seg = ac[lag_min:lag_max + 1]
        if not seg.size:
            continue
        peak = seg.max()
        if peak / ac[0] < voiced_thr:  # слабая периодичность → кадр не озвучен
            continue
        # антиоктавная защита: берём не глобальный максимум (может быть гармоникой на коротком
        # лаге = завышенный F0), а самый ДЛИННЫЙ лаг среди пиков, близких к максимуму —
        # это фундаментал. Без этого мужские голоса ловятся на 2×-3× (см. history.md 2026-07-20).
        strong = np.where(seg >= 0.85 * peak)[0]
        k = int(strong.max())
        voiced += 1
        f0s.append(sr / (lag_min + k))
    return np.asarray(f0s), (voiced / total if total else 0.0)


def extract_features(y: np.ndarray, sr: int) -> dict[str, float]:
    """Посчитать все фичи по моно-сигналу `y` (float32/64, диапазон ~[-1, 1]) на частоте `sr`.

    Импорт librosa — внутри функции: воркеры ProcessPool импортируют его в своём процессе, а не
    платят за импорт при старте (и модуль остаётся импортируемым без librosa для интроспекции)."""
    import librosa

    feat: dict[str, float] = {k: np.nan for k in FEATURE_COLUMNS}
    n = y.shape[0]
    duration = n / sr
    feat["duration_s"] = duration
    if n == 0:
        return feat

    # --- покадровая RMS-энергия (основа для громкости/динамики/тишины) ---
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
    eps = 1e-10
    feat["rms_mean"] = float(rms.mean())
    feat["rms_std"] = float(rms.std())
    feat["rms_max"] = float(rms.max())
    feat["loudness_dbfs"] = float(20.0 * np.log10(rms.mean() + eps))
    peak = float(np.abs(y).max())
    feat["peak_dbfs"] = float(20.0 * np.log10(peak + eps))
    feat["crest_db"] = feat["peak_dbfs"] - feat["loudness_dbfs"]
    # динамический диапазон: разброс громких/тихих кадров (p95/p5 RMS), в дБ
    p95, p5 = np.percentile(rms, 95), np.percentile(rms, 5)
    feat["dyn_range_db"] = float(20.0 * np.log10((p95 + eps) / (p5 + eps)))
    feat["clipping_ratio"] = float(np.mean(np.abs(y) >= CLIP_THRESH))

    # --- тишина / паузы / края (по маске тихих кадров) ---
    rms_db = 20.0 * np.log10(rms + eps)
    silence_mask = rms_db < (rms_db.max() - SILENCE_REL_DB)
    feat["silence_ratio"] = float(silence_mask.mean())
    frame_dur = HOP / sr
    # ведущая/хвостовая тишина
    voiced_idx = np.where(~silence_mask)[0]
    if voiced_idx.size:
        feat["lead_silence_s"] = float(voiced_idx[0] * frame_dur)
        feat["trail_silence_s"] = float((len(silence_mask) - 1 - voiced_idx[-1]) * frame_dur)
    else:
        feat["lead_silence_s"] = float(duration)
        feat["trail_silence_s"] = float(duration)
    # число внутренних пауз: сегменты тишины между первым и последним звучащим кадром
    if voiced_idx.size:
        inner = silence_mask[voiced_idx[0]:voiced_idx[-1] + 1]
        feat["n_pauses"] = float(int(np.sum(np.diff(inner.astype(int)) == 1)))
    else:
        feat["n_pauses"] = 0.0
    # энергия у самых краёв относительно средней — детектор обрубленной речи
    edge = max(1, int(EDGE_WIN_S * sr))
    ref = feat["rms_mean"] + eps
    feat["edge_start_ratio"] = float(np.sqrt(np.mean(y[:edge] ** 2)) / ref)
    feat["edge_end_ratio"] = float(np.sqrt(np.mean(y[-edge:] ** 2)) / ref)

    # --- питч F0 (автокорреляция, без numba — см. _f0_autocorr) ---
    voiced, voiced_ratio = _f0_autocorr(y, sr)
    feat["voiced_ratio"] = float(voiced_ratio)
    if voiced.size:
        feat["f0_mean_hz"] = float(np.mean(voiced))
        feat["f0_median_hz"] = float(np.median(voiced))
        semi = _hz_to_semitones(voiced)
        feat["f0_std_semitones"] = float(np.std(semi))
        feat["f0_range_semitones"] = float(np.percentile(semi, 95) - np.percentile(semi, 5))

    # --- спектр / тембр ---
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    cent = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    feat["spec_centroid_mean"] = float(cent.mean())
    feat["spec_centroid_std"] = float(cent.std())
    feat["spec_bandwidth_mean"] = float(librosa.feature.spectral_bandwidth(S=S, sr=sr)[0].mean())
    feat["spec_rolloff_mean"] = float(
        librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0].mean())
    feat["spec_flatness_mean"] = float(librosa.feature.spectral_flatness(S=S)[0].mean())
    feat["zcr_mean"] = float(librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP)[0].mean())
    # доля энергии выше HF_CUTOFF (яркость/напряжённость)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    power = S ** 2
    total = power.sum() + eps
    feat["hf_energy_ratio"] = float(power[freqs >= HF_CUTOFF, :].sum() / total)

    # --- гармоничность (HPSS) ---
    y_harm, y_perc = librosa.effects.hpss(y)
    e_h, e_p = float(np.sum(y_harm ** 2)), float(np.sum(y_perc ** 2))
    feat["harmonic_ratio"] = float(e_h / (e_h + e_p + eps))

    # --- ритм: онсеты в секунду (грубый прокси темпа без ASR) ---
    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP, units="frames")
    feat["onset_rate"] = float(len(onsets) / duration) if duration > 0 else np.nan

    return feat


_SMILE = None  # ленивый per-process экземпляр openSMILE (не пиклится через ProcessPool)


def _get_smile():
    """Ленивая инициализация openSMILE (eGeMAPSv02, Functionals) в процессе-воркере."""
    global _SMILE
    if _SMILE is None:
        import opensmile
        _SMILE = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    return _SMILE


def egemaps_columns() -> list[str]:
    """Имена 88 eGeMAPS-фич с префиксом egm_ (для сборки итоговой таблицы в main)."""
    import opensmile
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    return [f"egm_{n}" for n in smile.feature_names]


def _process_one(args: tuple[str, str, bool]) -> dict:
    """Воркер: (fragment_id, path, egemaps) -> строка результата с feat_status."""
    import soundfile as sf

    fragment_id, path, egemaps = args
    row: dict = {"fragment_id": fragment_id}
    try:
        y, sr = sf.read(path, dtype="float32", always_2d=False)
        if y.ndim > 1:  # на всякий случай сведём в моно
            y = y.mean(axis=1)
        feat = extract_features(np.asarray(y, dtype=np.float32), int(sr))
        row.update(feat)
        if egemaps:
            # openSMILE читает файл сам (C++, без numba — не сегфолтит как pyin)
            egm = _get_smile().process_file(path).iloc[0]
            row.update({f"egm_{k}": float(v) for k, v in egm.items()})
        row["feat_status"] = "ok"
    except Exception as e:  # noqa: BLE001 — любую ошибку логируем в статус, не роняем прогон
        for k in FEATURE_COLUMNS:
            row[k] = np.nan
        row["feat_status"] = f"error:{type(e).__name__}"
    return row


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST)
    ap.add_argument("--output", default=DEFAULT_OUTPUT, help="Куда писать таблицу фич.")
    ap.add_argument("--readers", default=None,
                    help="Только эти чтецы (через запятую), для разведочных прогонов.")
    ap.add_argument("--limit", type=int, default=None, help="Ограничить число фрагментов.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true",
                    help="Пропустить fragment_id, уже присутствующие в --output.")
    ap.add_argument("--egemaps", action="store_true",
                    help="Дополнительно посчитать 88 eGeMAPS-фич через openSMILE (колонки egm_*).")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    df = df[df["cut_status"].isin(OK_STATUSES)].reset_index(drop=True)
    if args.readers:
        wanted = {r.strip() for r in args.readers.split(",")}
        df = df[df["reader"].isin(wanted)].reset_index(drop=True)
    if args.resume and Path(args.output).exists():
        done = set(pd.read_csv(args.output, usecols=["fragment_id"])["fragment_id"])
        before = len(df)
        df = df[~df["fragment_id"].isin(done)].reset_index(drop=True)
        print(f"resume: пропущено {before - len(df)} уже посчитанных", file=sys.stderr)
    if args.limit:
        df = df.head(args.limit).reset_index(drop=True)

    tasks = [(fid, path, args.egemaps)
             for fid, path in df[["fragment_id", "audio_fragment_path"]].itertuples(index=False, name=None)]
    print(f"К обработке: {len(tasks)} фрагментов, воркеров: {args.workers}"
          f"{', + eGeMAPS' if args.egemaps else ''}", file=sys.stderr)
    if not tasks:
        return

    rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_process_one, t) for t in tasks]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="features", unit="frag"):
            rows.append(fut.result())

    feat_cols = list(FEATURE_COLUMNS)
    if args.egemaps:
        feat_cols += egemaps_columns()
    out = pd.DataFrame(rows, columns=["fragment_id", *feat_cols, "feat_status"])

    # words_per_sec — прокси темпа из манифеста (ASR-слова / длительность), лучше onset_rate
    if {"n_words_matched", "fragment_duration"}.issubset(df.columns):
        wps = df[["fragment_id", "n_words_matched", "fragment_duration"]].copy()
        wps["words_per_sec"] = wps["n_words_matched"] / wps["fragment_duration"].replace(0, np.nan)
        out = out.merge(wps[["fragment_id", "words_per_sec"]], on="fragment_id", how="left")

    n_err = int((out["feat_status"] != "ok").sum())
    if n_err:
        from collections import Counter
        cats = Counter(s for s in out["feat_status"] if s != "ok")
        print(f"Ошибок: {n_err}, по категориям: {dict(cats)}", file=sys.stderr)

    # дозапись при --resume, иначе перезапись
    out_path = Path(args.output)
    if args.resume and out_path.exists():
        out.to_csv(out_path, mode="a", header=False, index=False)
    else:
        out.to_csv(out_path, index=False)
    print(f"Готово: {len(out)} строк -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
