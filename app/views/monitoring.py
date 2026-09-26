import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.common import get_storage, load_report  # noqa: E402
from router import metrics  # noqa: E402
from router.config import MANUAL_ROUTING_MINUTES  # noqa: E402

st.title("📊 Мониторинг внедрения")

df = get_storage().all_requests()
s = metrics.summary(df)


def percent(value):
    return "нет данных" if value is None else f"{value:.1%}"


c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Обращений", s["total"])
c2.metric("Автомаршрутизация", f"{s['auto_share']:.0%}")
c3.metric("Точность автомаршрутизации", percent(s["auto_accuracy"]),
          help="Доля верных среди обращений, направленных в отдел без участия человека.")
c4.metric("Точность модели", percent(s["accuracy"]),
          help="Доля обработанных обращений, где отдел, предложенный моделью, не пришлось исправлять "
               "(включая очередь ручного разбора).")
c5.metric("Сэкономлено, ч", f"{s['saved_hours']:.1f}")
st.caption(f"В очереди ручного разбора: {s['pending']} · исправлений операторов: {s['corrected']} "
           f"(из них ошибок автомаршрутизации: {s['auto_errors']}). "
           f"Экономия = число автоматически направленных обращений × {MANUAL_ROUTING_MINUTES} мин "
           "ручной маршрутизации.")

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
                 hide_index=True, width="stretch")

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
    st.dataframe(per_class[["precision", "recall", "f1-score", "support"]].round(3), width="stretch")
    st.markdown("**Матрица ошибок** (строки — истинный отдел, столбцы — предсказанный)")
    st.dataframe(pd.DataFrame(report["confusion_matrix"], index=report["labels"], columns=report["labels"]),
                 width="stretch")
