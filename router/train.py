"""Обучение модели: python -m router.train [--with-corrections]."""
import argparse
import json
from pathlib import Path

import pandas as pd

from router import model
from router.config import DATASET_PATH, DB_PATH, MODEL_PATH, REPORT_PATH
from router.data_generator import generate
from router.storage import Storage


def run(dataset_path: Path = DATASET_PATH, model_path: Path = MODEL_PATH, report_path: Path = REPORT_PATH,
        db_path: Path = DB_PATH, with_corrections: bool = False, n: int = 2400, seed: int = 42) -> dict:
    if dataset_path.exists():
        df = pd.read_csv(dataset_path)
    else:
        df = generate(n, seed)
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dataset_path, index=False)

    n_corrections = 0
    if with_corrections:
        corrections = Storage(db_path).corrections()
        n_corrections = len(corrections)
        df = pd.concat([df, corrections], ignore_index=True)

    pipeline, report = model.train(df, seed=seed)
    report["n_corrections"] = n_corrections
    model.save(pipeline, model_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Обучение модели маршрутизации обращений")
    parser.add_argument("--with-corrections", action="store_true", help="добавить исправления операторов из журнала")
    parser.add_argument("--n", type=int, default=2400, help="размер синтетического датасета")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    report = run(with_corrections=args.with_corrections, n=args.n, seed=args.seed)
    print(f"Обучающая выборка: {report['n_train']}, тестовая: {report['n_test']}, "
          f"исправлений операторов: {report['n_corrections']}")
    print(f"Accuracy: {report['accuracy']:.3f}   Macro-F1: {report['macro_f1']:.3f}")
    for label in report["labels"]:
        row = report["per_class"][label]
        print(f"  {label:<28} precision={row['precision']:.2f} recall={row['recall']:.2f} f1={row['f1-score']:.2f}")
    print(f"Модель сохранена: {MODEL_PATH}")


if __name__ == "__main__":
    main()
