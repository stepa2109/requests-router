# Маршрутизация обращений студентов — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Локальное ML-приложение, которое определяет отдел университета для обращения студента, решает «авто / ручной разбор», журналирует работу операторов и показывает мониторинг.

**Architecture:** Пакет `router/` — ядро без UI (генератор данных, модель, сервис маршрутизации, SQLite-журнал, метрики, CLI). `app/` — Streamlit-приложение из трёх страниц поверх ядра. Документация по этапам внедрения — в `docs/`.

**Tech Stack:** Python 3.12, scikit-learn, pandas, joblib, Streamlit (≥1.40, `st.navigation`), SQLite (stdlib), pytest.

**Spec:** `docs/superpowers/specs/2026-09-25-student-requests-router-design.md`

## Global Constraints

- Работает без GPU, без интернета во время работы и без внешних API.
- Точность на отложенной выборке: macro-F1 ≥ 0.85.
- Порог уверенности по умолчанию: 0.6 (`CONFIDENCE_THRESHOLD`).
- Минимальная длина текста после нормализации: 3 символа, иначе `ValueError`.
- Срочность — только два значения: `"высокая"` / `"обычная"`.
- Данные синтетические; это явно указано в README и docs.
- Все тексты интерфейса и документации — на русском.
- Команды запуска: `python -m router.train`, `python -m router.simulate`, `streamlit run app/main.py`.

## Review Focus

1. Текст только из пунктуации/пробелов/эмодзи (`"???"`, `"  "`) → `ValueError`, в UI — предупреждение, без трассировки. (Task 4)
2. Текст на латинице / не по теме (`"hello"`, `"купите слона"`) → результат возвращается, но с низкой уверенностью уходит на ручной разбор, а не падает. (Task 4)
3. Приложение открыто до обучения модели → понятное сообщение с командой обучения, без исключения. (Task 8)
4. Оператор «исправил» на тот же отдел, что предсказан → считается подтверждением (`corrected=0`), не портит метрику точности. (Task 5)
5. Мониторинг на пустой БД → нули/«нет данных», без деления на ноль; переобучение без исправлений → работает. (Tasks 6, 7)

---

## Файловая структура

```
router/__init__.py         пустой
router/config.py           отделы, пути, пороги
router/preprocessing.py    normalize(text)
router/data_generator.py   generate(n, seed, label_noise) -> DataFrame
router/urgency.py          detect_urgency(text)
router/model.py            build_pipeline, train, save, load
router/service.py          RoutingResult, RequestRouter
router/storage.py          Storage (SQLite)
router/metrics.py          summary, daily, by_department
router/train.py            run(...), main(argv) — CLI обучения
router/simulate.py         run(...), main(argv) — CLI имитации потока
app/main.py                точка входа Streamlit, навигация
app/common.py              кэшированные router/storage, общие хелперы
app/views/operator.py      страница «Оператор»
app/views/monitoring.py    страница «Мониторинг»
app/views/training.py      страница «Обучение»
tests/...                  по модулю на файл
docs/business_process.md, docs/technology_choice.md, docs/user_guide.md, docs/results.md
README.md, requirements.txt, pytest.ini
```

---

### Task 1: Каркас проекта, конфиг, нормализация текста

**Files:**
- Create: `requirements.txt`, `pytest.ini`, `router/__init__.py`, `router/config.py`, `router/preprocessing.py`
- Test: `tests/test_preprocessing.py`

**Interfaces:**
- Produces: `config.DEPARTMENTS: list[str]`, `config.BASE_DIR, DATA_DIR, MODELS_DIR, DATASET_PATH, DB_PATH, MODEL_PATH, REPORT_PATH: Path`, `config.CONFIDENCE_THRESHOLD = 0.6`, `config.MIN_TEXT_LEN = 3`, `config.MANUAL_ROUTING_MINUTES = 5`, `config.URGENCY_HIGH = "высокая"`, `config.URGENCY_NORMAL = "обычная"`; `preprocessing.normalize(text: str) -> str`.

- [ ] **Step 1: Файлы окружения**

`requirements.txt`:
```
scikit-learn>=1.5
pandas>=2.2
joblib>=1.4
streamlit>=1.40
pytest>=8.0
```

`pytest.ini`:
```ini
[pytest]
testpaths = tests
pythonpath = .
```

`router/__init__.py` — пустой файл.

Run: `.venv/bin/pip install -r requirements.txt`

- [ ] **Step 2: Написать падающий тест**

`tests/test_preprocessing.py`:
```python
from router.preprocessing import normalize


def test_lowercase_and_yo():
    assert normalize("Приёмная КОМИССИЯ") == "приемная комиссия"


def test_punctuation_and_spaces():
    assert normalize("  Здравствуйте!!!   Где,  расписание?? ") == "здравствуйте где расписание"


def test_keeps_latin_and_digits():
    assert normalize("Ошибка 500 в LMS") == "ошибка 500 в lms"


def test_only_punctuation_becomes_empty():
    assert normalize("?!... 🙂") == ""
```

- [ ] **Step 3: Запустить — падает**

Run: `.venv/bin/pytest tests/test_preprocessing.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'router.preprocessing'`

- [ ] **Step 4: Реализация**

`router/config.py`:
```python
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
MIN_TEXT_LEN = 3
# Сколько минут в среднем тратит сотрудник на ручную маршрутизацию одного обращения.
MANUAL_ROUTING_MINUTES = 5

URGENCY_HIGH = "высокая"
URGENCY_NORMAL = "обычная"
```

`router/preprocessing.py`:
```python
"""Нормализация текста обращений перед векторизацией."""
import re

_NON_WORD = re.compile(r"[^0-9a-zа-я]+")


def normalize(text: str) -> str:
    text = text.lower().replace("ё", "е")
    return _NON_WORD.sub(" ", text).strip()
```

- [ ] **Step 5: Запустить — проходит**

Run: `.venv/bin/pytest tests/test_preprocessing.py -v`
Expected: 4 passed

- [ ] **Step 6: Commit**

```bash
git add requirements.txt pytest.ini router tests
git commit -m "feat: project skeleton, config and text normalization"
```

---

### Task 2: Генератор синтетического датасета

**Files:**
- Create: `router/data_generator.py`
- Test: `tests/test_data_generator.py`

**Interfaces:**
- Consumes: `config.DEPARTMENTS`
- Produces: `generate(n: int = 2400, seed: int = 42, label_noise: float = 0.03) -> pd.DataFrame` с колонками `text: str`, `department: str`. Строк ровно `n // len(DEPARTMENTS) * len(DEPARTMENTS)`.

- [ ] **Step 1: Падающий тест**

`tests/test_data_generator.py`:
```python
from router.config import DEPARTMENTS
from router.data_generator import generate


def test_deterministic():
    assert generate(160, seed=1).equals(generate(160, seed=1))


def test_different_seeds_differ():
    assert not generate(160, seed=1).equals(generate(160, seed=2))


def test_balanced_without_noise():
    df = generate(800, seed=3, label_noise=0.0)
    counts = df["department"].value_counts()
    assert set(counts.index) == set(DEPARTMENTS)
    assert (counts == 100).all()


def test_noise_keeps_all_labels_valid():
    df = generate(800, seed=3, label_noise=0.1)
    assert set(df["department"]) <= set(DEPARTMENTS)
    assert len(df) == 800


def test_texts_non_empty_and_varied():
    df = generate(800, seed=4)
    assert (df["text"].str.len() > 10).all()
    assert df["text"].nunique() > 600
```

- [ ] **Step 2: Запустить — падает**

Run: `.venv/bin/pytest tests/test_data_generator.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Реализация**

`router/data_generator.py`:
```python
"""Генератор синтетического датасета обращений студентов.

Реальных обращений университета у нас нет, поэтому тексты собираются
из шаблонов: типовая суть вопроса + приветствие, контекст, вежливые
обороты и опечатки. Небольшая доля меток намеренно «ошибочна» —
так выглядят исторические данные, размеченные людьми вручную.
"""
import random

import pandas as pd

from router.config import DEPARTMENTS

SUBJECTS = [
    "высшей математике", "философии", "программированию на Python", "базам данных",
    "экономике", "английскому языку", "машинному обучению", "статистике",
    "истории России", "менеджменту", "праву", "маркетингу",
]
PROGRAMS = [
    "Прикладная информатика", "Экономика", "Менеджмент", "Юриспруденция",
    "Психология", "Бизнес-информатика", "Реклама и связи с общественностью",
]

TEMPLATES = {
    "Деканат": [
        "хочу оформить академический отпуск, какие документы нужны",
        "как записаться на пересдачу по {subject}",
        "меня не допустили к пересдаче по {subject}, что делать",
        "хочу перевестись с направления {program} на другое направление",
        "как перевестись с очной формы обучения на заочную",
        "как восстановиться после отчисления",
        "нужно согласовать индивидуальный график обучения",
        "прошу продлить сессию по уважительной причине, есть больничный",
        "хочу сменить научного руководителя курсовой работы",
        "как получить разрешение на свободное посещение занятий",
        "у меня академическая задолженность по {subject}, какие сроки ликвидации",
        "хочу подать заявление на перевод в другой вуз",
    ],
    "Учебный отдел": [
        "где посмотреть расписание сессии",
        "когда будет экзамен по {subject}",
        "в расписании стоят две пары одновременно, какая правильная",
        "не отображается оценка по {subject} в зачетной книжке",
        "где найти учебный план по направлению {program}",
        "когда начинается следующий семестр",
        "перенесли занятие по {subject}, какое новое время",
        "не могу найти список литературы по {subject}",
        "какие дисциплины по выбору есть на {course} курсе",
        "преподаватель по {subject} не выставил оценку в ведомость",
        "когда будет защита курсовых работ",
        "сколько зачетных единиц в семестре на {course} курсе",
    ],
    "Бухгалтерия": [
        "как оплатить обучение за следующий семестр",
        "оплатил обучение, но в личном кабинете висит задолженность",
        "нужна справка для налогового вычета за обучение",
        "можно ли оформить рассрочку оплаты обучения",
        "как получить копию договора на обучение и акт об оплате",
        "хочу вернуть деньги за обучение после отчисления",
        "платеж прошел дважды, верните лишнюю сумму",
        "почему выросла стоимость обучения по договору",
        "можно ли оплатить обучение материнским капиталом",
        "нужны реквизиты для оплаты обучения от организации",
        "как оформить скидку на обучение для многодетной семьи",
        "не пришел чек об оплате семестра",
    ],
    "Приёмная комиссия": [
        "хочу поступить на направление {program}, какие вступительные испытания",
        "до какого числа принимают документы абитуриентов",
        "можно ли поступить без ЕГЭ после колледжа",
        "как подать документы на поступление онлайн",
        "какой проходной балл был на {program} в прошлом году",
        "есть ли бюджетные места на {program}",
        "когда будут результаты вступительного экзамена",
        "я абитуриент, как узнать свое место в конкурсном списке",
        "хочу поступить в магистратуру, какие нужны документы",
        "можно ли поступить на второе высшее образование",
        "когда будет день открытых дверей",
        "учитываются ли индивидуальные достижения при поступлении",
    ],
    "IT-поддержка": [
        "не могу войти в личный кабинет, пишет неверный пароль",
        "как восстановить пароль от личного кабинета",
        "не открывается вебинар, черный экран",
        "в LMS не загружается задание по {subject}",
        "не могу прикрепить файл с работой, выдает ошибку",
        "приложение университета вылетает на телефоне",
        "не приходит код подтверждения на почту",
        "нет доступа к электронной библиотеке",
        "тест по {subject} завис и не засчитал ответы",
        "как подключить корпоративную почту студента",
        "пропал звук на онлайн лекции по {subject}",
        "сайт выдает ошибку 500 при входе",
    ],
    "Справки и документы": [
        "нужна справка об обучении для работы",
        "как заказать справку для военкомата",
        "нужна академическая справка",
        "как получить дубликат диплома",
        "нужна справка для пенсионного фонда, что я учусь очно",
        "как заверить копию зачетной книжки",
        "сколько делается справка об обучении",
        "хочу забрать оригинал аттестата из архива",
        "нужна справка-вызов для работодателя на сессию",
        "потерял студенческий билет, как восстановить",
        "как получить приложение к диплому на английском",
        "нужна справка о периоде обучения",
    ],
    "Практика и трудоустройство": [
        "где найти место для прохождения производственной практики",
        "как оформить договор на практику с организацией",
        "можно ли пройти практику по месту работы",
        "кто подписывает дневник практики",
        "когда сдавать отчет по преддипломной практике",
        "есть ли вакансии для студентов направления {program}",
        "помогите составить резюме для стажировки",
        "когда будет ярмарка вакансий",
        "руководитель практики от организации не отвечает",
        "нужно индивидуальное задание на практику",
        "как попасть на стажировку в компанию-партнер",
        "можно ли перенести сроки учебной практики",
    ],
    "Студенческий офис": [
        "как заселиться в общежитие",
        "сколько стоит проживание в общежитии",
        "хочу вступить в студенческий совет",
        "как записаться в спортивную секцию",
        "когда будет посвящение в студенты",
        "как оформить материальную помощь",
        "хочу организовать студенческий клуб",
        "есть ли волонтерские программы",
        "как оформить студенческий проездной",
        "в общежитии не работает отопление",
        "как попасть в студенческий театр",
        "можно ли переселиться в другую комнату в общежитии",
    ],
}

OPENERS = ["", "", "Здравствуйте!", "Добрый день.", "Добрый вечер!", "Привет.", "Уважаемые коллеги,"]
CONTEXTS = [
    "", "", "", "Я студент {course} курса, {program}.", "Учусь на заочной форме.",
    "Пишу уже второй раз.", "Номер договора {num}.", "Учусь дистанционно.",
]
PREFIXES = ["", "", "подскажите, ", "скажите пожалуйста, ", "у меня вопрос: ", "помогите, ", "не понимаю, "]
CLOSERS = ["", "", "Спасибо.", "Заранее спасибо!", "Жду ответа.", "Буду благодарен за помощь.", "Срочно!"]


def _typo(text: str, rng: random.Random) -> str:
    """Переставляет две соседние буквы в случайном слове."""
    words = text.split(" ")
    candidates = [i for i, w in enumerate(words) if len(w) > 4]
    if not candidates:
        return text
    i = rng.choice(candidates)
    w = words[i]
    j = rng.randrange(len(w) - 1)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def _compose(template: str, rng: random.Random) -> str:
    slots = {
        "subject": rng.choice(SUBJECTS),
        "program": rng.choice(PROGRAMS),
        "course": rng.randint(1, 5),
        "num": rng.randint(10000, 99999),
    }
    core = rng.choice(PREFIXES) + template.format(**slots)
    core = core[0].upper() + core[1:] + rng.choice(["?", ".", ""])
    if rng.random() < 0.15:
        core = _typo(core, rng)
    parts = [
        rng.choice(OPENERS),
        rng.choice(CONTEXTS).format(**slots),
        core,
        rng.choice(CLOSERS),
    ]
    return " ".join(p for p in parts if p)


def generate(n: int = 2400, seed: int = 42, label_noise: float = 0.03) -> pd.DataFrame:
    rng = random.Random(seed)
    per_class = n // len(DEPARTMENTS)
    rows = []
    for department in DEPARTMENTS:
        for _ in range(per_class):
            text = _compose(rng.choice(TEMPLATES[department]), rng)
            label = department
            if rng.random() < label_noise:
                label = rng.choice([d for d in DEPARTMENTS if d != department])
            rows.append((text, label))
    rng.shuffle(rows)
    return pd.DataFrame(rows, columns=["text", "department"])
```

- [ ] **Step 4: Запустить — проходит**

Run: `.venv/bin/pytest tests/test_data_generator.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add router/data_generator.py tests/test_data_generator.py
git commit -m "feat: synthetic student requests generator"
```

---

### Task 3: Определение срочности

**Files:**
- Create: `router/urgency.py`
- Test: `tests/test_urgency.py`

**Interfaces:**
- Consumes: `preprocessing.normalize`, `config.URGENCY_HIGH`, `config.URGENCY_NORMAL`
- Produces: `detect_urgency(text: str) -> str` (одно из двух значений).

- [ ] **Step 1: Падающий тест**

`tests/test_urgency.py`:
```python
import pytest

from router.config import URGENCY_HIGH, URGENCY_NORMAL
from router.urgency import detect_urgency


@pytest.mark.parametrize("text", [
    "Срочно! Не могу войти в личный кабинет",
    "Завтра экзамен, а тест не открывается",
    "Сегодня последний день оплаты",
    "Меня хотят отчислить, помогите",
    "Горит дедлайн по курсовой",
    "Нужно как можно скорее получить справку",
])
def test_high(text):
    assert detect_urgency(text) == URGENCY_HIGH


@pytest.mark.parametrize("text", [
    "Где посмотреть расписание сессии?",
    "Как оформить академический отпуск",
    "Хочу вступить в студенческий совет",
])
def test_normal(text):
    assert detect_urgency(text) == URGENCY_NORMAL
```

- [ ] **Step 2: Запустить — падает**

Run: `.venv/bin/pytest tests/test_urgency.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Реализация**

`router/urgency.py`:
```python
"""Правила определения срочности обращения по ключевым словам."""
import re

from router.config import URGENCY_HIGH, URGENCY_NORMAL
from router.preprocessing import normalize

_URGENT = re.compile(
    r"\b(срочн\w*|немедленно|как можно скорее|сегодня|завтра|дедлайн\w*|"
    r"горит|последний день|отчисл\w*|не допуст\w*)\b"
)


def detect_urgency(text: str) -> str:
    return URGENCY_HIGH if _URGENT.search(normalize(text)) else URGENCY_NORMAL
```

- [ ] **Step 4: Запустить — проходит**

Run: `.venv/bin/pytest tests/test_urgency.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add router/urgency.py tests/test_urgency.py
git commit -m "feat: rule-based urgency detection"
```

---

### Task 4: Модель и сервис маршрутизации

**Files:**
- Create: `router/model.py`, `router/service.py`
- Test: `tests/test_model.py`, `tests/test_service.py`, `tests/conftest.py`

**Interfaces:**
- Consumes: `normalize`, `generate`, `detect_urgency`, `config.*`
- Produces:
  - `model.build_pipeline() -> sklearn.pipeline.Pipeline`
  - `model.train(df: DataFrame[text, department], test_size=0.2, seed=42) -> tuple[Pipeline, dict]`; отчёт: `accuracy: float, macro_f1: float, per_class: dict[str, dict], labels: list[str], confusion_matrix: list[list[int]], n_train: int, n_test: int, trained_at: str (ISO)`
  - `model.save(pipeline, path: Path) -> None`, `model.load(path: Path) -> Pipeline`
  - `service.RoutingResult` (dataclass): `department: str, confidence: float, urgency: str, auto_routed: bool, top3: list[tuple[str, float]]`
  - `service.RequestRouter(pipeline, threshold=CONFIDENCE_THRESHOLD)`, `.route(text: str) -> RoutingResult` (ValueError на слишком коротком тексте), `RequestRouter.from_disk(path=MODEL_PATH) -> RequestRouter`
  - фикстура `trained_pipeline` (scope=session) в `tests/conftest.py`

- [ ] **Step 1: Падающие тесты**

`tests/conftest.py`:
```python
import pytest

from router.data_generator import generate
from router.model import train


@pytest.fixture(scope="session")
def trained():
    return train(generate(1600, seed=11))


@pytest.fixture(scope="session")
def trained_pipeline(trained):
    return trained[0]
```

`tests/test_model.py`:
```python
from router.config import DEPARTMENTS
from router.model import load, save


def test_quality_threshold(trained):
    _, report = trained
    assert report["macro_f1"] >= 0.85
    assert set(report["labels"]) == set(DEPARTMENTS)
    assert len(report["confusion_matrix"]) == len(DEPARTMENTS)
    assert report["n_train"] + report["n_test"] == 1600


def test_save_load_roundtrip(trained_pipeline, tmp_path):
    path = tmp_path / "m.joblib"
    save(trained_pipeline, path)
    texts = ["Где расписание сессии?", "Не могу войти в личный кабинет"]
    assert list(load(path).predict(texts)) == list(trained_pipeline.predict(texts))
```

`tests/test_service.py`:
```python
import pytest

from router.config import DEPARTMENTS, URGENCY_HIGH
from router.service import RequestRouter


def test_route_confident(trained_pipeline):
    result = RequestRouter(trained_pipeline).route("Не могу войти в личный кабинет, пишет неверный пароль")
    assert result.department == "IT-поддержка"
    assert result.auto_routed
    assert 0.6 <= result.confidence <= 1.0
    assert len(result.top3) == 3
    assert result.top3[0][0] == result.department


def test_urgency_passed_through(trained_pipeline):
    result = RequestRouter(trained_pipeline).route("Срочно! Завтра экзамен, где расписание?")
    assert result.urgency == URGENCY_HIGH


def test_threshold_one_forces_manual(trained_pipeline):
    result = RequestRouter(trained_pipeline, threshold=1.0).route("Как оплатить обучение?")
    assert not result.auto_routed


@pytest.mark.parametrize("text", ["", "  ", "???", "!!", "🙂🙂"])
def test_too_short_rejected(trained_pipeline, text):
    with pytest.raises(ValueError):
        RequestRouter(trained_pipeline).route(text)


@pytest.mark.parametrize("text", ["hello", "купите слона", "qwerty asdf"])
def test_off_topic_goes_to_manual(trained_pipeline, text):
    result = RequestRouter(trained_pipeline).route(text)
    assert result.department in DEPARTMENTS
    assert not result.auto_routed
```

- [ ] **Step 2: Запустить — падает**

Run: `.venv/bin/pytest tests/test_model.py tests/test_service.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'router.model'`

- [ ] **Step 3: Реализация**

`router/model.py`:
```python
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
```

`router/service.py`:
```python
"""Сервис маршрутизации: модель + правило уверенности + срочность."""
from dataclasses import dataclass
from pathlib import Path

from sklearn.pipeline import Pipeline

from router import model
from router.config import CONFIDENCE_THRESHOLD, MIN_TEXT_LEN, MODEL_PATH
from router.preprocessing import normalize
from router.urgency import detect_urgency


@dataclass
class RoutingResult:
    department: str
    confidence: float
    urgency: str
    auto_routed: bool
    top3: list[tuple[str, float]]


class RequestRouter:
    def __init__(self, pipeline: Pipeline, threshold: float = CONFIDENCE_THRESHOLD):
        self.pipeline = pipeline
        self.threshold = threshold

    @classmethod
    def from_disk(cls, path: Path = MODEL_PATH) -> "RequestRouter":
        return cls(model.load(path))

    def route(self, text: str) -> RoutingResult:
        if len(normalize(text)) < MIN_TEXT_LEN:
            raise ValueError("Текст обращения слишком короткий")
        probabilities = self.pipeline.predict_proba([text])[0]
        ranked = sorted(zip(self.pipeline.classes_, probabilities), key=lambda p: p[1], reverse=True)
        department, confidence = ranked[0]
        return RoutingResult(
            department=str(department),
            confidence=float(confidence),
            urgency=detect_urgency(text),
            auto_routed=bool(confidence >= self.threshold),
            top3=[(str(d), float(p)) for d, p in ranked[:3]],
        )
```

- [ ] **Step 4: Запустить — проходит**

Run: `.venv/bin/pytest tests/test_model.py tests/test_service.py -v`
Expected: all passed. Если `test_off_topic_goes_to_manual` падает — значит модель чрезмерно уверена на мусоре: уменьшить `C` (10 → 5) и перепроверить, что `test_route_confident` и `test_quality_threshold` проходят.

- [ ] **Step 5: Commit**

```bash
git add router/model.py router/service.py tests/conftest.py tests/test_model.py tests/test_service.py
git commit -m "feat: TF-IDF + logistic regression model and routing service"
```

---

### Task 5: SQLite-журнал обращений

**Files:**
- Create: `router/storage.py`
- Test: `tests/test_storage.py`

**Interfaces:**
- Consumes: `service.RoutingResult`
- Produces: `Storage(db_path: Path)`; методы:
  - `log(text: str, result: RoutingResult, created_at: datetime | None = None) -> int` — авто-обращение сразу считается обработанным (`final_department = predicted`, `processed_at = created_at`), ручное — в очереди (`final_department = NULL`).
  - `confirm(request_id: int) -> None`
  - `correct(request_id: int, department: str) -> None` — `corrected = 1` только если отдел отличается от предсказанного.
  - `pending() -> pd.DataFrame` — необработанные, сначала срочные, затем старые.
  - `all_requests() -> pd.DataFrame` — колонки `id, created_at (datetime64), text, predicted, confidence, urgency, auto_routed (bool), final_department, corrected (bool), processed_at`.
  - `corrections() -> pd.DataFrame[text, department]`.

- [ ] **Step 1: Падающий тест**

`tests/test_storage.py`:
```python
from datetime import datetime

import pytest

from router.service import RoutingResult
from router.storage import Storage


def result(department="Бухгалтерия", auto=True, urgency="обычная"):
    return RoutingResult(department, 0.9 if auto else 0.4, urgency, auto, [(department, 0.9)])


@pytest.fixture
def storage(tmp_path):
    return Storage(tmp_path / "db.sqlite")


def test_empty(storage):
    assert storage.all_requests().empty
    assert storage.pending().empty
    assert storage.corrections().empty


def test_auto_request_is_processed(storage):
    storage.log("Как оплатить?", result(auto=True))
    row = storage.all_requests().iloc[0]
    assert row["final_department"] == "Бухгалтерия"
    assert not row["corrected"]
    assert storage.pending().empty


def test_manual_request_waits_then_confirmed(storage):
    request_id = storage.log("Непонятный вопрос", result(auto=False))
    assert list(storage.pending()["id"]) == [request_id]
    storage.confirm(request_id)
    assert storage.pending().empty
    assert storage.all_requests().iloc[0]["final_department"] == "Бухгалтерия"


def test_correction_recorded(storage):
    request_id = storage.log("Справка для вычета", result(auto=True))
    storage.correct(request_id, "Справки и документы")
    row = storage.all_requests().iloc[0]
    assert row["corrected"]
    assert row["final_department"] == "Справки и документы"
    assert storage.corrections().to_dict("records") == [
        {"text": "Справка для вычета", "department": "Справки и документы"}
    ]


def test_correct_to_same_department_is_confirmation(storage):
    request_id = storage.log("Как оплатить?", result(auto=False))
    storage.correct(request_id, "Бухгалтерия")
    assert not storage.all_requests().iloc[0]["corrected"]
    assert storage.corrections().empty


def test_pending_urgent_first(storage):
    storage.log("старое обычное", result(auto=False), created_at=datetime(2026, 1, 1))
    urgent_id = storage.log("новое срочное", result(auto=False, urgency="высокая"), created_at=datetime(2026, 1, 2))
    assert storage.pending().iloc[0]["id"] == urgent_id


def test_created_at_is_datetime(storage):
    storage.log("текст", result(), created_at=datetime(2026, 3, 5, 10, 0))
    assert storage.all_requests().iloc[0]["created_at"] == datetime(2026, 3, 5, 10, 0)
```

- [ ] **Step 2: Запустить — падает**

Run: `.venv/bin/pytest tests/test_storage.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Реализация**

`router/storage.py`:
```python
"""Журнал обращений и действий операторов в SQLite."""
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

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

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

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

    def _query(self, sql: str) -> pd.DataFrame:
        with self._connect() as conn:
            df = pd.read_sql_query(sql, conn)
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
            " ORDER BY urgency = 'высокая' DESC, created_at"
        )

    def corrections(self) -> pd.DataFrame:
        return self._query("SELECT text, final_department AS department FROM requests WHERE corrected = 1")
```

- [ ] **Step 4: Запустить — проходит**

Run: `.venv/bin/pytest tests/test_storage.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add router/storage.py tests/test_storage.py
git commit -m "feat: SQLite request log with operator confirmations and corrections"
```

---

### Task 6: Метрики мониторинга

**Files:**
- Create: `router/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: DataFrame из `Storage.all_requests()`, `config.MANUAL_ROUTING_MINUTES`
- Produces:
  - `summary(df, minutes_per_request=MANUAL_ROUTING_MINUTES) -> dict` с ключами `total: int, auto_share: float, accuracy: float | None, avg_confidence: float | None, pending: int, corrected: int, saved_hours: float`
  - `daily(df) -> pd.DataFrame` колонки `date, total, auto, corrected`
  - `by_department(df) -> pd.Series` (индекс — отдел, значение — количество; для необработанных берётся предсказанный отдел)

- [ ] **Step 1: Падающий тест**

`tests/test_metrics.py`:
```python
import pandas as pd
import pytest

from router.metrics import by_department, daily, summary

COLUMNS = ["created_at", "predicted", "confidence", "auto_routed", "final_department", "corrected"]


def frame(rows):
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def test_empty():
    s = summary(frame([]))
    assert s == {"total": 0, "auto_share": 0.0, "accuracy": None, "avg_confidence": None,
                 "pending": 0, "corrected": 0, "saved_hours": 0.0}
    assert daily(frame([])).empty
    assert by_department(frame([])).empty


def sample():
    return frame([
        ("2026-09-01 10:00", "Бухгалтерия", 0.9, True, "Бухгалтерия", False),
        ("2026-09-01 11:00", "Деканат", 0.8, True, "Учебный отдел", True),
        ("2026-09-02 09:00", "Деканат", 0.5, False, "Деканат", False),
        ("2026-09-02 12:00", "Бухгалтерия", 0.4, False, None, False),
    ])


def test_summary():
    s = summary(sample(), minutes_per_request=6)
    assert s["total"] == 4
    assert s["auto_share"] == 0.5
    assert s["accuracy"] == pytest.approx(2 / 3)
    assert s["avg_confidence"] == pytest.approx(0.65)
    assert s["pending"] == 1
    assert s["corrected"] == 1
    assert s["saved_hours"] == pytest.approx(0.2)  # 2 авто × 6 мин


def test_daily():
    d = daily(sample())
    assert list(d["total"]) == [2, 2]
    assert list(d["auto"]) == [2, 0]
    assert list(d["corrected"]) == [1, 0]


def test_by_department():
    counts = by_department(sample())
    assert counts.to_dict() == {"Бухгалтерия": 2, "Учебный отдел": 1, "Деканат": 1}
```

- [ ] **Step 2: Запустить — падает**

Run: `.venv/bin/pytest tests/test_metrics.py -v`
Expected: FAIL, `ModuleNotFoundError`

- [ ] **Step 3: Реализация**

`router/metrics.py`:
```python
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
```

- [ ] **Step 4: Запустить — проходит**

Run: `.venv/bin/pytest tests/test_metrics.py -v`
Expected: 4 passed. (`by_department` сравнивается через `to_dict()`, поэтому имя Series не влияет.)

- [ ] **Step 5: Commit**

```bash
git add router/metrics.py tests/test_metrics.py
git commit -m "feat: monitoring metrics over request log"
```

---

### Task 7: CLI обучения и имитации потока

**Files:**
- Create: `router/train.py`, `router/simulate.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `generate`, `model.train/save`, `Storage`, `RequestRouter`
- Produces:
  - `train.run(dataset_path=DATASET_PATH, model_path=MODEL_PATH, report_path=REPORT_PATH, db_path=DB_PATH, with_corrections=False, n=2400, seed=42) -> dict` — создаёт датасет, если его нет; при `with_corrections` добавляет `Storage(db_path).corrections()`; сохраняет модель и `report.json` (в отчёт добавляется `n_corrections: int`); возвращает отчёт.
  - `train.main(argv: list[str] | None = None) -> None` — флаги `--with-corrections`, `--n`, `--seed`.
  - `simulate.run(router: RequestRouter, storage: Storage, n=300, days=30, seed=2026) -> dict` — прогоняет `generate(n, seed, label_noise=0)` через роутер, раскидывает даты по последним `days` дням; ручные обращения оператор разбирает (`correct` на истинный отдел), авто-обращения с ошибкой «возвращаются отделом» (`correct`). Возвращает `metrics.summary`.
  - `simulate.main(argv=None)` — флаги `--n`, `--days`, `--seed`.

- [ ] **Step 1: Падающий тест**

`tests/test_cli.py`:
```python
import json

from router import simulate, train
from router.service import RequestRouter, RoutingResult
from router.storage import Storage


def paths(tmp_path):
    return dict(dataset_path=tmp_path / "d.csv", model_path=tmp_path / "m.joblib",
                report_path=tmp_path / "r.json", db_path=tmp_path / "db.sqlite")


def test_train_creates_artifacts(tmp_path):
    p = paths(tmp_path)
    report = train.run(**p, n=800)
    assert p["dataset_path"].exists() and p["model_path"].exists()
    assert json.loads(p["report_path"].read_text(encoding="utf-8"))["macro_f1"] == report["macro_f1"]
    assert report["n_corrections"] == 0


def test_train_with_corrections_empty_db(tmp_path):
    report = train.run(**paths(tmp_path), n=800, with_corrections=True)
    assert report["n_corrections"] == 0


def test_train_with_corrections_uses_them(tmp_path):
    p = paths(tmp_path)
    storage = Storage(p["db_path"])
    request_id = storage.log("Где взять дневник практики?", RoutingResult("Деканат", 0.5, "обычная", False, []))
    storage.correct(request_id, "Практика и трудоустройство")
    report = train.run(**p, n=800, with_corrections=True)
    assert report["n_corrections"] == 1
    assert report["n_train"] + report["n_test"] == 801


def test_simulate_fills_log(tmp_path, trained_pipeline):
    storage = Storage(tmp_path / "db.sqlite")
    stats = simulate.run(RequestRouter(trained_pipeline), storage, n=80, days=10)
    df = storage.all_requests()
    assert len(df) == 80 == stats["total"]
    assert stats["pending"] == 0
    assert df["created_at"].dt.date.nunique() > 1
    assert stats["accuracy"] >= 0.8
```

- [ ] **Step 2: Запустить — падает**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: FAIL, `ImportError: cannot import name 'simulate'`

- [ ] **Step 3: Реализация**

`router/train.py`:
```python
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
```

`router/simulate.py`:
```python
"""Имитация потока обращений для наполнения мониторинга: python -m router.simulate."""
import argparse
import random
from datetime import datetime, timedelta

from router import metrics
from router.config import DB_PATH, MODEL_PATH
from router.data_generator import generate
from router.service import RequestRouter
from router.storage import Storage


def run(router: RequestRouter, storage: Storage, n: int = 300, days: int = 30, seed: int = 2026) -> dict:
    rng = random.Random(seed)
    now = datetime.now()
    requests = generate(n, seed=seed, label_noise=0.0)
    for text, true_department in requests.itertuples(index=False):
        created_at = now - timedelta(days=rng.uniform(0, days))
        result = router.route(text)
        request_id = storage.log(text, result, created_at=created_at)
        # Оператор разбирает ручную очередь; ошибочно направленные авто-обращения возвращает отдел.
        if not result.auto_routed or result.department != true_department:
            storage.correct(request_id, true_department)
    return metrics.summary(storage.all_requests())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Имитация потока обращений студентов")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args(argv)

    if not MODEL_PATH.exists():
        raise SystemExit("Модель не обучена. Сначала выполните: python -m router.train")
    stats = run(RequestRouter.from_disk(), Storage(DB_PATH), n=args.n, days=args.days, seed=args.seed)
    print(f"Всего обращений в журнале: {stats['total']}")
    print(f"Автомаршрутизация: {stats['auto_share']:.1%}   Точность: {stats['accuracy']:.1%}")
    print(f"Сэкономлено времени сотрудников: {stats['saved_hours']:.1f} ч")


if __name__ == "__main__":
    main()
```

Примечание: `generate(n=80)` при 8 отделах даёт ровно 80 строк.

- [ ] **Step 4: Запустить — проходит**

Run: `.venv/bin/pytest tests/test_cli.py -v`
Expected: 4 passed

- [ ] **Step 5: Прогон на реальных путях**

Run: `.venv/bin/python -m router.train && .venv/bin/python -m router.simulate`
Expected: печать метрик, Macro-F1 ≥ 0.85, файлы `data/dataset.csv`, `data/router.db`, `models/model.joblib`, `models/report.json`. Если доля автомаршрутизации < 50 % — снизить порог невозможно без согласования со спецификацией; вместо этого проверить `C` в модели и зафиксировать наблюдение в `docs/results.md`.

- [ ] **Step 6: Commit**

`models/` и `data/dataset.csv` коммитятся (чтобы проверяющий мог сразу запустить приложение); `data/router.db` в `.gitignore`.

```bash
git add router/train.py router/simulate.py tests/test_cli.py data/dataset.csv models/
git commit -m "feat: training and simulation CLIs"
```

---

### Task 8: Веб-приложение Streamlit

**Files:**
- Create: `app/main.py`, `app/common.py`, `app/views/operator.py`, `app/views/monitoring.py`, `app/views/training.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `RequestRouter`, `Storage`, `metrics.*`, `train.run`, `generate`, `config.*`
- Produces: `common.get_router() -> RequestRouter | None` (кэш `st.cache_resource`), `common.require_router() -> RequestRouter` (иначе `st.error` + `st.stop()`), `common.get_storage() -> Storage`, `common.load_report() -> dict | None`. Пути берутся из `router.config` в момент вызова (чтобы тесты могли подменить через monkeypatch).

- [ ] **Step 1: Падающий тест**

`tests/test_app.py`:
```python
import pytest
from streamlit.testing.v1 import AppTest

from router import config, model

PAGES = ["app/views/operator.py", "app/views/monitoring.py", "app/views/training.py"]


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "m.joblib")
    monkeypatch.setattr(config, "REPORT_PATH", tmp_path / "r.json")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "db.sqlite")
    monkeypatch.setattr(config, "DATASET_PATH", tmp_path / "d.csv")
    import app.common as common
    common.get_router.clear()
    return tmp_path


@pytest.mark.parametrize("page", PAGES)
def test_pages_without_model_do_not_crash(isolated, page):
    at = AppTest.from_file(page).run(timeout=30)
    assert not at.exception


@pytest.mark.parametrize("page", PAGES)
def test_pages_with_model(isolated, trained_pipeline, page):
    model.save(trained_pipeline, config.MODEL_PATH)
    at = AppTest.from_file(page).run(timeout=30)
    assert not at.exception


def test_operator_routes_text(isolated, trained_pipeline):
    model.save(trained_pipeline, config.MODEL_PATH)
    at = AppTest.from_file("app/views/operator.py").run(timeout=30)
    at.text_area[0].input("Не могу войти в личный кабинет, пишет неверный пароль")
    at.button(key="route").click().run(timeout=30)
    assert not at.exception
    assert any("IT-поддержка" in m.value for m in at.markdown)


def test_operator_short_text_warns(isolated, trained_pipeline):
    model.save(trained_pipeline, config.MODEL_PATH)
    at = AppTest.from_file("app/views/operator.py").run(timeout=30)
    at.text_area[0].input("??")
    at.button(key="route").click().run(timeout=30)
    assert not at.exception
    assert at.warning


def test_main_entrypoint(isolated):
    at = AppTest.from_file("app/main.py").run(timeout=30)
    assert not at.exception
```

Также создать пустой `app/__init__.py` и `app/views/__init__.py` (нужны, чтобы тест мог сделать `import app.common`).

- [ ] **Step 2: Запустить — падает**

Run: `.venv/bin/pytest tests/test_app.py -v`
Expected: FAIL (файлов приложения нет)

- [ ] **Step 3: Реализация**

`app/common.py`:
```python
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
def get_router() -> RequestRouter | None:
    if not config.MODEL_PATH.exists():
        return None
    return RequestRouter.from_disk(config.MODEL_PATH)


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
```

Каждая страница начинается с одинакового импорта, чтобы работать и из `main.py`, и из `AppTest.from_file`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
```

`app/main.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Маршрутизация обращений студентов", page_icon="📨", layout="wide")

pages = st.navigation([
    st.Page("views/operator.py", title="Оператор", icon="📨", default=True),
    st.Page("views/monitoring.py", title="Мониторинг", icon="📊"),
    st.Page("views/training.py", title="Обучение", icon="🎓"),
])
pages.run()
```

`app/views/operator.py`:
```python
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app.common import get_storage, require_router  # noqa: E402
from router.config import DEPARTMENTS, URGENCY_HIGH  # noqa: E402
from router.data_generator import generate  # noqa: E402

st.title("📨 Рабочее место оператора")
st.caption("ИИ предлагает отдел для обращения студента. Оператор подтверждает или исправляет решение.")

router = require_router()
storage = get_storage()

if "draft" not in st.session_state:
    st.session_state.draft = ""

if st.button("Взять пример обращения"):
    st.session_state.draft = generate(8, seed=random.randint(0, 10**6)).iloc[0]["text"]

text = st.text_area("Текст обращения", key="draft", height=120)

if st.button("Определить отдел", type="primary", key="route"):
    try:
        result = router.route(text)
    except ValueError:
        st.warning("Введите текст обращения (не короче 3 символов).")
    else:
        request_id = storage.log(text, result)
        st.session_state.last = (request_id, result)

if "last" in st.session_state:
    request_id, result = st.session_state.last
    mode = "✅ направлено автоматически" if result.auto_routed else "🕵️ отправлено на ручной разбор"
    urgency = "🔥 высокая" if result.urgency == URGENCY_HIGH else "обычная"
    st.markdown(f"### Отдел: **{result.department}**")
    st.markdown(f"Уверенность: **{result.confidence:.0%}** · Срочность: **{urgency}** · {mode}")
    st.markdown("Другие варианты: " + ", ".join(f"{d} ({p:.0%})" for d, p in result.top3[1:]))
    with st.form(f"review_{request_id}"):
        chosen = st.selectbox("Правильный отдел", DEPARTMENTS, index=DEPARTMENTS.index(result.department))
        if st.form_submit_button("Сохранить решение"):
            storage.correct(request_id, chosen)
            st.success("Решение сохранено" + (" (исправление учтётся при дообучении)"
                                               if chosen != result.department else ""))
            del st.session_state.last

st.divider()
st.subheader("Очередь ручного разбора")
queue = storage.pending()
if queue.empty:
    st.info("Очередь пуста.")
for row in queue.head(20).itertuples():
    with st.expander(f"#{row.id} · {'🔥 ' if row.urgency == URGENCY_HIGH else ''}{row.text[:80]}"):
        st.write(row.text)
        st.caption(f"Предложено: {row.predicted} ({row.confidence:.0%})")
        department = st.selectbox("Отдел", DEPARTMENTS, index=DEPARTMENTS.index(row.predicted),
                                  key=f"dept_{row.id}")
        if st.button("Направить", key=f"send_{row.id}"):
            storage.correct(row.id, department)
            st.rerun()
```

`app/views/monitoring.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.common import get_storage, load_report  # noqa: E402
from router import metrics  # noqa: E402

st.title("📊 Мониторинг внедрения")

df = get_storage().all_requests()
s = metrics.summary(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Обращений", s["total"])
c2.metric("Автомаршрутизация", f"{s['auto_share']:.0%}")
c3.metric("Точность по журналу", "нет данных" if s["accuracy"] is None else f"{s['accuracy']:.1%}")
c4.metric("Сэкономлено, ч", f"{s['saved_hours']:.1f}")
st.caption(f"В очереди ручного разбора: {s['pending']} · исправлений операторов: {s['corrected']}. "
           "Экономия считается как число автоматически направленных обращений × 5 минут ручной маршрутизации.")

if df.empty:
    st.info("Журнал пуст. Обработайте обращения на странице «Оператор» "
            "или заполните журнал демо-данными: `python -m router.simulate`.")
else:
    left, right = st.columns(2)
    with left:
        st.subheader("Обращения по дням")
        st.line_chart(metrics.daily(df).set_index("date")[["total", "auto", "corrected"]])
    with right:
        st.subheader("Нагрузка по отделам")
        st.bar_chart(metrics.by_department(df))
    st.subheader("Последние исправления операторов")
    fixes = df[df["corrected"]].sort_values("created_at", ascending=False).head(10)
    st.dataframe(fixes[["created_at", "text", "predicted", "final_department", "confidence"]],
                 hide_index=True, use_container_width=True)

st.divider()
st.subheader("Качество модели на тестовой выборке")
report = load_report()
if report is None:
    st.info("Отчёт об обучении не найден. Выполните `python -m router.train`.")
else:
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{report['accuracy']:.3f}")
    c2.metric("Macro-F1", f"{report['macro_f1']:.3f}")
    c3.metric("Обучено", report["trained_at"].replace("T", " "))
    per_class = pd.DataFrame({label: report["per_class"][label] for label in report["labels"]}).T
    st.dataframe(per_class[["precision", "recall", "f1-score", "support"]].round(3), use_container_width=True)
    st.markdown("**Матрица ошибок** (строки — истинный отдел, столбцы — предсказанный)")
    st.dataframe(pd.DataFrame(report["confusion_matrix"], index=report["labels"], columns=report["labels"]),
                 use_container_width=True)
```

`app/views/training.py`:
```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app.common import get_router, get_storage, load_report  # noqa: E402
from router import config, train  # noqa: E402

st.title("🎓 Обучение сотрудников")

st.markdown(f"""
### Как работает система
1. Студент пишет обращение в свободной форме.
2. Модель ИИ определяет отдел и оценивает **уверенность** (от 0 до 100 %).
3. Если уверенность ≥ **{config.CONFIDENCE_THRESHOLD:.0%}**, обращение уходит в отдел автоматически.
   Иначе — попадает в **очередь ручного разбора** на странице «Оператор».
4. Срочность («высокая» / «обычная») определяется по ключевым словам: «срочно», «завтра», «дедлайн» и т. п.
   Срочные обращения стоят первыми в очереди.

### Что делает оператор
- Разбирает очередь: выбирает правильный отдел и нажимает «Направить».
- Если отдел вернул ошибочно направленное обращение — исправляет отдел в карточке.
- Каждое исправление сохраняется и используется для **дообучения** модели.

### Частые вопросы
**Модель ошиблась — это плохо?** Нет. Исправьте отдел: так модель учится на реальных примерах.

**Почему обращение ушло на ручной разбор?** Текст неоднозначный или не похож на типовые вопросы.

**Что делать, если отдел не подходит ни один?** Выберите ближайший и сообщите администратору системы.
""")

st.divider()
st.subheader("Дообучение модели на исправлениях операторов")
n_corrections = len(get_storage().corrections())
st.write(f"Накоплено исправлений: **{n_corrections}**")

before = load_report()
if st.button("Переобучить модель", type="primary", key="retrain"):
    with st.spinner("Обучение..."):
        after = train.run(config.DATASET_PATH, config.MODEL_PATH, config.REPORT_PATH, config.DB_PATH,
                          with_corrections=True)
    get_router.clear()
    c1, c2 = st.columns(2)
    c1.metric("Macro-F1 до", "—" if before is None else f"{before['macro_f1']:.3f}")
    delta = None if before is None else f"{after['macro_f1'] - before['macro_f1']:+.3f}"
    c2.metric("Macro-F1 после", f"{after['macro_f1']:.3f}", delta=delta)
    st.success(f"Модель переобучена (учтено исправлений: {after['n_corrections']}).")
```

Примечание к `train.run` в UI: аргументы передаются позиционно из `config` в момент клика, чтобы monkeypatch в тестах и реальные пути работали одинаково (значения по умолчанию в сигнатуре `run` зафиксированы при импорте).

- [ ] **Step 4: Запустить — проходит**

Run: `.venv/bin/pytest tests/test_app.py -v`
Expected: all passed. Если `AppTest` не находит `app.common` — проверить наличие `app/__init__.py` и `sys.path` вставку в странице.

- [ ] **Step 5: Ручная проверка**

Run: `.venv/bin/streamlit run app/main.py --server.headless true` — открыть http://localhost:8501, проверить три страницы, ввести обращение, подтвердить, исправить, переобучить.

- [ ] **Step 6: Commit**

```bash
git add app tests/test_app.py
git commit -m "feat: Streamlit app with operator, monitoring and training pages"
```

---

### Task 9: Документация

**Files:**
- Create: `README.md`, `docs/business_process.md`, `docs/technology_choice.md`, `docs/user_guide.md`, `docs/results.md`

**Interfaces:**
- Consumes: реальные цифры из `models/report.json` и вывода `python -m router.simulate` (Task 7, Step 5) — вписываются в `docs/results.md` и README, не выдумываются.

- [ ] **Step 1: `docs/business_process.md`** — этап 1 задания:
  - контекст: Университет «Синергия», большое число онлайн- и заочных студентов, обращения через ЛК, почту, чат;
  - схема процесса «как было» (mermaid flowchart): студент → общий ящик → сотрудник читает и пересылает → отдел (узкие места: ручная сортировка, пересылки по цепочке, потеря срочных, нет статистики);
  - перечень процессов-кандидатов на ИИ (маршрутизация обращений, прогноз отчислений, проверка работ на заимствования, чат-бот FAQ, прогноз нагрузки) с оценкой эффекта/сложности; обоснование, почему первой выбрана маршрутизация;
  - схема «как стало» (mermaid): студент → модель → авто в отдел / очередь оператора → исправления → дообучение;
  - KPI внедрения: доля автомаршрутизации, точность, время до попадания в отдел, сэкономленные часы.

- [ ] **Step 2: `docs/technology_choice.md`** — этап 2: таблица сравнения (правила по ключевым словам / TF-IDF + логистическая регрессия / fine-tuning BERT (ruBERT) / LLM через API) по критериям: точность, требования к железу, стоимость, объяснимость, данные для обучения, скорость внедрения, защита персональных данных (152-ФЗ). Выбор и обоснование: TF-IDF + LR для отдела, правила для срочности; путь развития — ruBERT при накоплении реальных данных.

- [ ] **Step 3: `docs/user_guide.md`** — этап 4: инструкция для сотрудника (как работать на каждой странице, что значит уверенность, как исправлять, когда переобучать), план обучения персонала (вводный семинар 1 ч, практикум, памятка, контакт поддержки), анкета удовлетворённости (5–7 вопросов по шкале 1–5) — пригодится для кейса 4.

- [ ] **Step 4: `docs/results.md`** — этап 5: фактические метрики из `models/report.json` (accuracy, macro-F1, per-class таблица, наиболее частые путаницы из матрицы ошибок) и из симуляции (доля авто, точность по журналу, сэкономленные часы); выявленные ограничения (синтетические данные, шаблонность, правила срочности, нет интеграции с ЛК) — задел для кейсов 4–5.

- [ ] **Step 5: `README.md`** — цель и связь с кейс-задачей №3 (таблица «этап → где в проекте»), архитектура (mermaid), быстрый старт:
  ```bash
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  python -m router.train
  python -m router.simulate
  streamlit run app/main.py
  pytest
  ```
  структура проекта, ключевые результаты (цифры из results.md), оговорка про синтетические данные, ссылки на docs.

- [ ] **Step 6: Проверка**

Run: `.venv/bin/pytest -q` — все тесты проходят; сверить цифры в README/results.md с `models/report.json`.

- [ ] **Step 7: Commit**

```bash
git add README.md docs
git commit -m "docs: business process analysis, technology choice, user guide, results"
```
