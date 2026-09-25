"""Общие объекты приложения: модель, журнал, отчёт."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from router import config  # noqa: E402
from router.service import RequestRouter  # noqa: E402
from router.storage import Storage  # noqa: E402


@st.cache_resource
def _load_router(path: str, mtime: float) -> RequestRouter:
    # mtime входит в ключ кэша: после переобучения (из UI или CLI) модель перечитывается.
    return RequestRouter.from_disk(Path(path))


def get_router() -> RequestRouter | None:
    if not config.MODEL_PATH.exists():
        return None
    return _load_router(str(config.MODEL_PATH), config.MODEL_PATH.stat().st_mtime)


def require_router() -> RequestRouter:
    router = get_router()
    if router is None:
        st.error("Модель ещё не обучена. Выполните в терминале из корня проекта:")
        st.code("python -m router.train", language="bash")
        st.stop()
    return router


def get_storage() -> Storage:
    return Storage(config.DB_PATH)


def load_report() -> dict | None:
    if not config.REPORT_PATH.exists():
        return None
    return json.loads(config.REPORT_PATH.read_text(encoding="utf-8"))
