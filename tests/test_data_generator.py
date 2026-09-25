from router.config import DEPARTMENTS
from router.data_generator import generate


def test_deterministic():
    assert generate(160, seed=1).equals(generate(160, seed=1))


def test_different_seeds_differ():
    assert not generate(160, seed=1).equals(generate(160, seed=2))


def test_balanced_without_noise():
    df = generate(800, seed=3, label_noise=0.0)
    counts = df["department"].value_counts()
    assert set(counts.index) == set(DEPARTMENTS)
    assert (counts == 100).all()


def test_noise_keeps_all_labels_valid():
    df = generate(800, seed=3, label_noise=0.1)
    assert set(df["department"]) <= set(DEPARTMENTS)
    assert len(df) == 800


def test_texts_non_empty_and_varied():
    df = generate(800, seed=4)
    assert (df["text"].str.len() > 10).all()
    assert df["text"].nunique() > 600
