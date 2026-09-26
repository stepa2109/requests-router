"""Журнал обращений и действий операторов в SQLite."""
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

import pandas as pd

from router.config import URGENCY_HIGH
from router.service import RoutingResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    text TEXT NOT NULL,
    predicted TEXT NOT NULL,
    confidence REAL NOT NULL,
    urgency TEXT NOT NULL,
    auto_routed INTEGER NOT NULL,
    final_department TEXT,
    corrected INTEGER NOT NULL DEFAULT 0,
    processed_at TEXT
)
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Storage:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def log(self, text: str, result: RoutingResult, created_at: datetime | None = None) -> int:
        created = (created_at or datetime.now()).isoformat(timespec="seconds")
        final = result.department if result.auto_routed else None
        processed = created if result.auto_routed else None
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO requests (created_at, text, predicted, confidence, urgency, auto_routed,"
                " final_department, processed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (created, text, result.department, result.confidence, result.urgency,
                 int(result.auto_routed), final, processed),
            )
            return cursor.lastrowid

    def confirm(self, request_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE requests SET final_department = predicted, corrected = 0, processed_at = ? WHERE id = ?",
                (_now(), request_id),
            )

    def correct(self, request_id: int, department: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE requests SET final_department = ?, corrected = (predicted != ?), processed_at = ?"
                " WHERE id = ?",
                (department, department, _now(), request_id),
            )

    def get(self, request_id: int) -> dict | None:
        df = self._query("SELECT * FROM requests WHERE id = ?", (request_id,))
        return None if df.empty else df.iloc[0].to_dict()

    def clear(self) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM requests")

    def _query(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        with self._connect() as conn:
            df = pd.read_sql_query(sql, conn, params=params)
        if "created_at" in df:
            df["created_at"] = pd.to_datetime(df["created_at"])
        for column in ("auto_routed", "corrected"):
            if column in df:
                df[column] = df[column].astype(bool)
        return df

    def all_requests(self) -> pd.DataFrame:
        return self._query("SELECT * FROM requests ORDER BY created_at")

    def pending(self) -> pd.DataFrame:
        return self._query(
            "SELECT * FROM requests WHERE final_department IS NULL"
            " ORDER BY urgency = ? DESC, created_at",
            (URGENCY_HIGH,),
        )

    def corrections(self) -> pd.DataFrame:
        return self._query("SELECT DISTINCT text, final_department AS department FROM requests WHERE corrected = 1")

    def training_examples(self) -> pd.DataFrame:
        """Примеры для дообучения: исправления операторов и проверенные человеком обращения из ручной очереди.

        Верно направленные автоматически обращения не берём: на них модель и так права.
        """
        return self._query(
            "SELECT DISTINCT text, final_department AS department FROM requests"
            " WHERE corrected = 1 OR (auto_routed = 0 AND final_department IS NOT NULL)"
        )
