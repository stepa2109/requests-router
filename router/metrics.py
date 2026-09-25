"""Метрики мониторинга по журналу обращений."""
import pandas as pd

from router.config import MANUAL_ROUTING_MINUTES


def summary(df: pd.DataFrame, minutes_per_request: float = MANUAL_ROUTING_MINUTES) -> dict:
    total = len(df)
    if total == 0:
        return {"total": 0, "auto_share": 0.0, "accuracy": None, "avg_confidence": None,
                "pending": 0, "corrected": 0, "saved_hours": 0.0}
    processed = df[df["final_department"].notna()]
    auto = int(df["auto_routed"].sum())
    return {
        "total": total,
        "auto_share": auto / total,
        # Точность модели на обработанных обращениях: доля, которую не пришлось исправлять.
        "accuracy": None if processed.empty else float(1 - processed["corrected"].mean()),
        "avg_confidence": float(df["confidence"].mean()),
        "pending": total - len(processed),
        "corrected": int(df["corrected"].sum()),
        "saved_hours": auto * minutes_per_request / 60,
    }


def daily(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["date", "total", "auto", "corrected"])
    grouped = df.assign(date=df["created_at"].dt.date).groupby("date")
    return grouped.agg(
        total=("auto_routed", "size"),
        auto=("auto_routed", "sum"),
        corrected=("corrected", "sum"),
    ).reset_index()


def by_department(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=int)
    return df["final_department"].fillna(df["predicted"]).value_counts().rename("count")
