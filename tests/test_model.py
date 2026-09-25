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
