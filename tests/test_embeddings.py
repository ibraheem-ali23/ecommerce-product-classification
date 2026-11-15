"""Tests for src/embeddings.py (tiny fake GloVe file, no download needed)."""
import numpy as np
import pytest

pytest.importorskip("keras")

import config
from src import embeddings


@pytest.fixture
def glove_file(tmp_path):
    """A fake GloVe file in the same format as the real one (with the optional header line)."""
    path = tmp_path / "glove.txt"
    vec = " ".join(["0.5"] * config.EMBEDDING_DIM)
    path.write_text(f"3 {config.EMBEDDING_DIM}\nbluetooth {vec}\nnovel {vec}\nthe {vec}\n", encoding="utf-8")
    return path


def test_vocabulary_uses_most_frequent_words_and_reserves_0_and_1():
    index = embeddings.build_vocabulary(["a a a b b c", "a b"], max_words=4)
    assert index == {"a": 2, "b": 3}            # only 4 - 2 words fit
    assert 0 not in index.values() and 1 not in index.values()


def test_sequences_are_padded_truncated_and_mark_unknown_words():
    index = {"red": 2, "shirt": 3}
    seq = embeddings.texts_to_sequences(["red shirt purple", "shirt"], index, max_len=4)
    assert seq.shape == (2, 4)
    assert seq[0].tolist() == [2, 3, embeddings.UNKNOWN, 0]
    assert seq[1].tolist() == [3, 0, 0, 0]
    assert embeddings.texts_to_sequences(["red shirt red shirt red"], index, max_len=3).shape == (1, 3)


def test_load_glove_keeps_only_vocabulary_words_and_skips_header(glove_file):
    vectors = embeddings.load_glove_vectors(glove_file, {"novel": 2, "missing": 3})
    assert list(vectors) == ["novel"] and vectors["novel"].shape == (config.EMBEDDING_DIM,)


def test_embedding_matrix_rows(glove_file):
    index = {"novel": 2, "missing": 3}
    matrix = embeddings.build_embedding_matrix(index, embeddings.load_glove_vectors(glove_file, index))
    assert matrix.shape == (4, config.EMBEDDING_DIM)
    assert np.allclose(matrix[2], 0.5)                    # known word = GloVe vector
    assert not matrix[[0, 1, 3]].any()                    # padding, unknown, word missing in GloVe = zeros


def test_frozen_and_fine_tuned_models_differ_only_in_trainable_embedding():
    matrix = np.random.RandomState(0).rand(20, config.EMBEDDING_DIM).astype("float32")
    frozen = embeddings.build_model(matrix, 4, trainable=False)
    tuned = embeddings.build_model(matrix, 4, trainable=True)
    dense = config.EMBEDDING_DIM * 4 + 4
    assert sum(int(np.prod(w.shape)) for w in frozen.trainable_weights) == dense
    assert sum(int(np.prod(w.shape)) for w in tuned.trainable_weights) == dense + 20 * config.EMBEDDING_DIM
    assert frozen.count_params() == tuned.count_params()


def test_frozen_embedding_does_not_change_but_fine_tuned_one_does():
    rng = np.random.RandomState(0)
    matrix = rng.rand(10, config.EMBEDDING_DIM).astype("float32")
    X = rng.randint(2, 10, size=(40, config.MAX_LEN)).astype("int32")
    y = rng.randint(0, 4, size=40)
    for trainable in (False, True):
        model = embeddings.build_model(matrix, 4, trainable)
        embeddings.train_model(model, X, y, epochs=2)
        changed = not np.allclose(model.layers[0].get_weights()[0], matrix)
        assert changed == trainable


def test_predict_labels_returns_valid_class_names():
    matrix = np.random.RandomState(1).rand(10, config.EMBEDDING_DIM).astype("float32")
    X = np.random.RandomState(2).randint(2, 10, size=(5, config.MAX_LEN)).astype("int32")
    names = ["Electronics", "Household", "Books", "Clothing & Accessories"]
    labels = embeddings.predict_labels(embeddings.build_model(matrix, 4, False), X, names)
    assert len(labels) == 5 and set(labels) <= set(names)
