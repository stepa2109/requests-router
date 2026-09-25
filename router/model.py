"""Модель классификации обращений по отделам."""
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from router.preprocessing import normalize


def build_pipeline() -> Pipeline:
    # Словесные n-граммы ловят смысл, символьные — устойчивы к опечаткам и словоформам.
    features = FeatureUnion([
        ("words", TfidfVectorizer(preprocessor=normalize, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("chars", TfidfVectorizer(preprocessor=normalize, analyzer="char_wb", ngram_range=(3, 5),
                                  min_df=2, sublinear_tf=True)),
    ])
    return Pipeline([
        ("features", features),
        ("clf", LogisticRegression(C=10, max_iter=2000)),
    ])


def train(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> tuple[Pipeline, dict]:
    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df["department"], test_size=test_size, random_state=seed, stratify=df["department"],
    )
    pipeline = build_pipeline().fit(x_train, y_train)
    predicted = pipeline.predict(x_test)
    labels = list(pipeline.classes_)
    report = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "macro_f1": float(f1_score(y_test, predicted, average="macro")),
        "per_class": classification_report(y_test, predicted, labels=labels, output_dict=True, zero_division=0),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_test, predicted, labels=labels).tolist(),
        "n_train": len(x_train),
        "n_test": len(x_test),
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    return pipeline, report


def save(pipeline: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)


def load(path: Path) -> Pipeline:
    return joblib.load(path)
