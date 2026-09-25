"""Имитация потока обращений для наполнения мониторинга: python -m router.simulate."""
import argparse
import random
from datetime import datetime, timedelta

from router import metrics
from router.config import DB_PATH, MODEL_PATH
from router.data_generator import generate
from router.service import RequestRouter
from router.storage import Storage


def run(router: RequestRouter, storage: Storage, n: int = 300, days: int = 30, seed: int = 2026,
        hard_share: float = 0.25) -> dict:
    rng = random.Random(seed)
    now = datetime.now()
    requests = generate(n, seed=seed, label_noise=0.0)
    # Второй поток — для «сложных» обращений, где студент задаёт сразу два вопроса в разные отделы.
    extra = generate(n, seed=seed + 1, label_noise=0.0)
    for (text, true_department), (extra_text, extra_department) in zip(
            requests.itertuples(index=False), extra.itertuples(index=False)):
        if extra_department != true_department and rng.random() < hard_share:
            text = f"{text} И ещё: {extra_text}"
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
    parser.add_argument("--hard-share", type=float, default=0.25, help="доля обращений с двумя вопросами сразу")
    parser.add_argument("--reset", action="store_true", help="очистить журнал перед имитацией")
    args = parser.parse_args(argv)

    if not MODEL_PATH.exists():
        raise SystemExit("Модель не обучена. Сначала выполните: python -m router.train")
    storage = Storage(DB_PATH)
    if args.reset:
        storage.clear()
    stats = run(RequestRouter.from_disk(MODEL_PATH), storage, n=args.n, days=args.days, seed=args.seed,
                hard_share=args.hard_share)
    accuracy = "нет данных" if stats["accuracy"] is None else f"{stats['accuracy']:.1%}"
    print(f"Всего обращений в журнале: {stats['total']}")
    print(f"Автомаршрутизация: {stats['auto_share']:.1%}   Точность: {accuracy}")
    print(f"Сэкономлено времени сотрудников: {stats['saved_hours']:.1f} ч")


if __name__ == "__main__":
    main()
