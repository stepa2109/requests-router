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


def test_simulate_hard_requests_produce_manual_work(tmp_path, trained_pipeline):
    storage = Storage(tmp_path / "db.sqlite")
    stats = simulate.run(RequestRouter(trained_pipeline), storage, n=80, days=10, hard_share=1.0)
    assert stats["auto_share"] < 0.9
    assert stats["corrected"] > 0
    assert (storage.all_requests()["text"].str.contains("И ещё:")).any()


def test_retrain_with_corrections_keeps_same_holdout(tmp_path):
    p = paths(tmp_path)
    baseline = train.run(**p, n=800)
    storage = Storage(p["db_path"])
    for _ in range(5):
        request_id = storage.log("Где взять дневник практики? И ещё: как оплатить?",
                                 RoutingResult("Деканат", 0.5, "обычная", False, []))
        storage.correct(request_id, "Практика и трудоустройство")
    report = train.run(**p, n=800, with_corrections=True)
    assert report["n_test"] == baseline["n_test"]


def test_simulate_reset_does_not_duplicate(tmp_path, trained_pipeline, monkeypatch):
    from router import model as model_module
    model_path = tmp_path / "m.joblib"
    model_module.save(trained_pipeline, model_path)
    monkeypatch.setattr(simulate, "MODEL_PATH", model_path)
    monkeypatch.setattr(simulate, "DB_PATH", tmp_path / "db.sqlite")
    simulate.main(["--n", "16", "--reset"])
    simulate.main(["--n", "16", "--reset"])
    assert len(Storage(tmp_path / "db.sqlite").all_requests()) == 16


def test_train_uses_confirmed_manual_requests(tmp_path):
    p = paths(tmp_path)
    storage = Storage(p["db_path"])
    fixed = storage.log("Где взять дневник практики?", RoutingResult("Деканат", 0.5, "обычная", False, []))
    storage.correct(fixed, "Практика и трудоустройство")
    confirmed = storage.log("Как оплатить семестр картой?", RoutingResult("Бухгалтерия", 0.5, "обычная", False, []))
    storage.confirm(confirmed)
    report = train.run(**p, n=800, with_corrections=True)
    assert report["n_corrections"] == 1
    assert report["n_feedback"] == 2
