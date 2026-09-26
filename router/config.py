"""Общие настройки проекта."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DATASET_PATH = DATA_DIR / "dataset.csv"
DB_PATH = DATA_DIR / "router.db"
MODEL_PATH = MODELS_DIR / "model.joblib"
REPORT_PATH = MODELS_DIR / "report.json"

DEPARTMENTS = [
    "Деканат",
    "Учебный отдел",
    "Бухгалтерия",
    "Приёмная комиссия",
    "IT-поддержка",
    "Справки и документы",
    "Практика и трудоустройство",
    "Студенческий офис",
]

# Ниже этой уверенности обращение уходит на ручной разбор оператору.
CONFIDENCE_THRESHOLD = 0.6
# Если второй по вероятности отдел набрал не меньше этого значения, в обращении, скорее всего,
# несколько вопросов в разные отделы — такое обращение тоже уходит на ручной разбор.
SECOND_OPTION_THRESHOLD = 0.15
MIN_TEXT_LEN = 3
# Сколько минут в среднем тратит сотрудник на ручную маршрутизацию одного обращения.
MANUAL_ROUTING_MINUTES = 5

URGENCY_HIGH = "высокая"
URGENCY_NORMAL = "обычная"
