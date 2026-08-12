#!/usr/bin/env python3
"""
Reader-disjoint train/val/test сплит (шаг 1.6) — назначает каждому ЧТЕЦУ ровно один сплит
(train/val/test), чтобы ни один голос не встречался больше чем в одном сплите — иначе модель
рискует выучить голос конкретного диктора вместо манеры речи (37 чтецов в корпусе).

Метод: жадная LPT-балансировка (largest-processing-time-first) по часам аудио, ЗАПУЩЕННАЯ
ОТДЕЛЬНО для male/female чтецов (каждая гендерная группа балансируется на те же --train-ratio/
--val-ratio/--test-ratio внутри себя) — это одновременно даёт точное попадание в целевые
пропорции по часам И не даёт результату перекоситься по полу в сторону одного сплита.
На 37 чтецах эмпирически даёт точное совпадение с целью (не приближение) — см. history.md
2026-07-30.

Вход: fragment-level манифест (`14_build_manifest.py`) — берутся только `reader`/`reader_gender`/
`duration_s`, агрегируются по чтецу.
Выход:
    data/reader_split.csv               — reader, gender, hours, split (для справки/аудита)
    data/processed/audio_fragments/manifest.csv — дополнен колонкой `split` (перезаписан)

Запуск (из корня репозитория, venv активирован):
    python scripts/15_assign_reader_split.py
    python scripts/15_assign_reader_split.py --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
"""
import argparse
import logging

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("assign_reader_split")

MANIFEST_PATH = "data/processed/audio_fragments/manifest.csv"
READER_SPLIT_PATH = "data/reader_split.csv"


def greedy_split_group(sub: pd.DataFrame, targets: dict[str, float], seed: int) -> dict[str, list[str]]:
    total = sub["hours"].sum()
    order = sub.sample(frac=1.0, random_state=seed).sort_values("hours", ascending=False)
    assigned = {k: 0.0 for k in targets}
    buckets: dict[str, list[str]] = {k: [] for k in targets}
    target_hours = {k: v * total for k, v in targets.items()}
    for _, row in order.iterrows():
        deficit = {k: target_hours[k] - assigned[k] for k in targets}
        best = max(deficit, key=deficit.get)
        assigned[best] += row["hours"]
        buckets[best].append(row["reader"])
    return buckets


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--manifest", default=MANIFEST_PATH)
    args = ap.parse_args()

    targets = {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio}
    assert abs(sum(targets.values()) - 1.0) < 1e-6, "пропорции должны суммироваться в 1.0"

    log.info("чтение манифеста...")
    df = pd.read_csv(args.manifest, usecols=["reader", "reader_gender", "duration_s"])
    g = df.groupby(["reader", "reader_gender"])["duration_s"].sum().reset_index()
    g["hours"] = g["duration_s"] / 3600
    total = g["hours"].sum()
    log.info("всего %.1fч, %d чтецов", total, len(g))

    reader_to_split: dict[str, str] = {}
    for gender in g["reader_gender"].unique():
        sub = g[g["reader_gender"] == gender]
        buckets = greedy_split_group(sub, targets, args.seed)
        for split, readers in buckets.items():
            for r in readers:
                reader_to_split[r] = split
        h = {k: sub[sub["reader"].isin(v)]["hours"].sum() for k, v in buckets.items()}
        log.info("%s (%.1fч, %d чтецов): %s",
                  gender, sub["hours"].sum(), len(sub),
                  {k: f"{v:.1f}ч ({v/sub['hours'].sum():.1%})" for k, v in h.items()})

    g["split"] = g["reader"].map(reader_to_split)
    g[["reader", "reader_gender", "hours", "split"]].to_csv(READER_SPLIT_PATH, index=False)
    log.info("сохранено: %s", READER_SPLIT_PATH)

    for split in targets:
        h = g[g["split"] == split]["hours"].sum()
        log.info("ИТОГО %s: %.1fч (%.1f%%), %d чтецов",
                  split, h, 100 * h / total, (g["split"] == split).sum())

    log.info("применяю к манифесту...")
    manifest = pd.read_csv(args.manifest)
    manifest["split"] = manifest["reader"].map(reader_to_split)
    assert manifest["split"].notna().all(), "не все фрагменты получили split — новый чтец?"
    manifest.to_csv(args.manifest, index=False)
    log.info("манифест обновлён: %s (+колонка split)", args.manifest)


if __name__ == "__main__":
    main()
