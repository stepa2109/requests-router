import pytest
from streamlit.testing.v1 import AppTest

from router import config, model

APP = config.BASE_DIR / "app"
OPERATOR = str(APP / "views" / "operator.py")
PAGES = [OPERATOR, str(APP / "views" / "monitoring.py"), str(APP / "views" / "training.py")]


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODEL_PATH", tmp_path / "m.joblib")
    monkeypatch.setattr(config, "REPORT_PATH", tmp_path / "r.json")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "db.sqlite")
    monkeypatch.setattr(config, "DATASET_PATH", tmp_path / "d.csv")
    import app.common as common
    common._load_router.clear()
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


def test_operator_without_model_explains_training(isolated):
    at = AppTest.from_file(OPERATOR).run(timeout=30)
    assert at.error
    assert any("router.train" in c.value for c in at.code)


def test_operator_routes_text(isolated, trained_pipeline):
    model.save(trained_pipeline, config.MODEL_PATH)
    at = AppTest.from_file(OPERATOR).run(timeout=30)
    at.text_area[0].input("Не могу войти в личный кабинет, пишет неверный пароль")
    at.button(key="route").click().run(timeout=30)
    assert not at.exception
    assert any("IT-поддержка" in m.value for m in at.markdown)


def test_operator_short_text_warns(isolated, trained_pipeline):
    model.save(trained_pipeline, config.MODEL_PATH)
    at = AppTest.from_file(OPERATOR).run(timeout=30)
    at.text_area[0].input("??")
    at.button(key="route").click().run(timeout=30)
    assert not at.exception
    assert at.warning


def test_monitoring_with_filled_log(isolated, trained_pipeline):
    from router import simulate, train
    from router.service import RequestRouter
    from router.storage import Storage

    train.run(config.DATASET_PATH, config.MODEL_PATH, config.REPORT_PATH, config.DB_PATH, n=800)
    simulate.run(RequestRouter(trained_pipeline), Storage(config.DB_PATH), n=80, days=5)
    at = AppTest.from_file(str(APP / "views" / "monitoring.py")).run(timeout=30)
    assert not at.exception
    assert at.metric[0].value == "80"


def test_main_entrypoint(isolated):
    at = AppTest.from_file(str(APP / "main.py")).run(timeout=30)
    assert not at.exception


def test_model_trained_after_app_start_is_picked_up(isolated, trained_pipeline):
    at = AppTest.from_file(OPERATOR).run(timeout=30)
    assert at.error
    model.save(trained_pipeline, config.MODEL_PATH)
    at = AppTest.from_file(OPERATOR).run(timeout=30)
    assert not at.error


def test_operator_can_correct_returned_auto_request(isolated, trained_pipeline):
    from router.service import RoutingResult
    from router.storage import Storage

    model.save(trained_pipeline, config.MODEL_PATH)
    storage = Storage(config.DB_PATH)
    request_id = storage.log("Как оплатить обучение?", RoutingResult("Бухгалтерия", 0.9, "обычная", True, []))
    at = AppTest.from_file(OPERATOR).run(timeout=30)
    at.number_input(key="return_id").set_value(request_id)
    at.selectbox(key="return_dept").select("Деканат")
    at.button(key="return_save").click().run(timeout=30)
    assert not at.exception
    row = storage.all_requests().iloc[0]
    assert row["corrected"] and row["final_department"] == "Деканат"


def test_return_unknown_id_warns(isolated, trained_pipeline):
    model.save(trained_pipeline, config.MODEL_PATH)
    at = AppTest.from_file(OPERATOR).run(timeout=30)
    at.number_input(key="return_id").set_value(999)
    at.button(key="return_save").click().run(timeout=30)
    assert not at.exception
    assert at.warning
