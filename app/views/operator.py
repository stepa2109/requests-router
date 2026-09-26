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
    st.markdown(f"### Обращение #{request_id} → отдел: **{result.department}**")
    st.markdown(f"Уверенность: **{result.confidence:.0%}** · Срочность: **{urgency}** · {mode}")
    st.markdown("Другие варианты: " + ", ".join(f"{d} ({p:.0%})" for d, p in result.top3[1:]))
    if result.multi_topic:
        st.markdown(f"⚠️ Похоже, в обращении несколько вопросов: второй вариант — **{result.top3[1][0]}** "
                    f"({result.top3[1][1]:.0%}). Проверьте, не нужно ли переслать обращение в оба отдела.")
    with st.form(f"review_{request_id}"):
        chosen = st.selectbox("Правильный отдел", DEPARTMENTS, index=DEPARTMENTS.index(result.department))
        if st.form_submit_button("Сохранить решение"):
            storage.correct(request_id, chosen)
            st.success("Решение сохранено" + (" (исправление учтётся при дообучении)"
                                               if chosen != result.department else ""))
            del st.session_state.last

st.divider()
st.subheader("Возврат обращения из отдела")
st.caption("Если отдел сообщил, что обращение пришло не по адресу, укажите его номер и правильный отдел.")
c1, c2, c3 = st.columns([1, 2, 1], vertical_alignment="bottom")
return_id = c1.number_input("Номер обращения", min_value=1, step=1, key="return_id")
return_dept = c2.selectbox("Правильный отдел", DEPARTMENTS, key="return_dept")
if c3.button("Исправить", key="return_save"):
    returned = storage.get(int(return_id))
    if returned is None:
        st.warning(f"Обращение #{int(return_id)} не найдено.")
    else:
        storage.correct(int(return_id), return_dept)
        st.success(f"Обращение #{int(return_id)} перенаправлено: {returned['predicted']} → {return_dept}.")

st.divider()
st.subheader("Очередь ручного разбора")
queue = storage.pending()
if queue.empty:
    st.info("Очередь пуста.")
elif len(queue) > 20:
    st.caption(f"Показаны 20 из {len(queue)} обращений в очереди.")
for row in queue.head(20).itertuples():
    with st.expander(f"#{row.id} · {'🔥 ' if row.urgency == URGENCY_HIGH else ''}{row.text[:80]}"):
        st.write(row.text)
        st.caption(f"Предложено: {row.predicted} ({row.confidence:.0%})")
        department = st.selectbox("Отдел", DEPARTMENTS, index=DEPARTMENTS.index(row.predicted),
                                  key=f"dept_{row.id}")
        if st.button("Направить", key=f"send_{row.id}"):
            storage.correct(row.id, department)
            st.rerun()
