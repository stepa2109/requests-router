import pytest

from router.data_generator import generate
from router.model import train


@pytest.fixture(scope="session")
def trained():
    return train(generate(1600, seed=11))


@pytest.fixture(scope="session")
def trained_pipeline(trained):
    return trained[0]
