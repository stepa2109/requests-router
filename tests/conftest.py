import pytest

from router.data_generator import generate
from router.model import train


@pytest.fixture(scope="session")
def trained():
    return train(generate(1600, seed=11))


@pytest.fixture(scope="session")
def trained_pipeline(trained):
    return trained[0]


class FakePipeline:
    """Подставная модель с заранее заданными вероятностями — для проверки правил маршрутизации."""

    def __init__(self, probabilities: dict[str, float]):
        self.classes_ = list(probabilities)
        self._probabilities = list(probabilities.values())

    def predict_proba(self, texts):
        return [self._probabilities for _ in texts]
