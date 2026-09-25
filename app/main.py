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
