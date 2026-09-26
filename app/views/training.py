import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from app.common import get_storage, load_report  # noqa: E402
from router import config, train  # noqa: E402

st.title("🎓 Обучение сотрудников")

st.markdown(f"""
### Как работает система
1. Студент пишет обращение в свободной форме.
2. Модель ИИ определяет отдел и оценивает **уверенность** (от 0 до 100 %).
3. Если уверенность ≥ **{config.CONFIDENCE_THRESHOLD:.0%}**, обращение уходит в отдел автоматически.
   Иначе — попадает в **очередь ручного разбора** на странице «Оператор».
   Туда же уходят обращения, где второй вариант отдела набрал ≥ **{config.SECOND_OPTION_THRESHOLD:.0%}**:
   скорее всего, в них несколько вопросов (пометка ⚠️).
4. Срочность («высокая» / «обычная») определяется по ключевым словам: «срочно», «завтра», «дедлайн» и т. п.
   Срочные обращения стоят первыми в очереди.

### Что делает оператор
- Разбирает очередь: выбирает правильный отдел и нажимает «Направить».
- Если отдел вернул ошибочно направленное обращение — указывает его номер в разделе
  «Возврат обращения из отдела» на странице «Оператор» и выбирает правильный отдел.
- Каждое исправление и каждое проверенное обращение из очереди ручного разбора
  сохраняются и используются для **дообучения** модели.

### Частые вопросы
**Модель ошиблась — это плохо?** Нет. Исправьте отдел: так модель учится на реальных примерах.

**Почему обращение ушло на ручной разбор?** Текст неоднозначный или не похож на типовые вопросы.

**Что делать, если не подходит ни один отдел?** Выберите ближайший и сообщите администратору системы.
""")

st.divider()
st.subheader("Дообучение модели на исправлениях операторов")
storage = get_storage()
n_feedback, n_corrections = len(storage.training_examples()), len(storage.corrections())
st.write(f"Накоплено примеров от операторов: **{n_feedback}** "
         f"(исправлений: {n_corrections}, подтверждений из ручной очереди: {n_feedback - n_corrections})")

before = load_report()
if st.button("Переобучить модель", type="primary", key="retrain"):
    with st.spinner("Обучение..."):
        after = train.run(config.DATASET_PATH, config.MODEL_PATH, config.REPORT_PATH, config.DB_PATH,
                          with_corrections=True)
    c1, c2 = st.columns(2)
    c1.metric("Macro-F1 до", "—" if before is None else f"{before['macro_f1']:.3f}")
    delta = None if before is None else f"{after['macro_f1'] - before['macro_f1']:+.3f}"
    c2.metric("Macro-F1 после", f"{after['macro_f1']:.3f}", delta=delta)
    st.success(f"Модель переобучена (учтено примеров от операторов: {after['n_feedback']}).")
