"""GloVe word embeddings with a very small Keras model (Submission 2).

Pipeline in five easy steps:
    1. build a vocabulary from the TRAINING texts
    2. turn every text into a fixed-length list of word numbers
    3. copy the pretrained GloVe vector of each vocabulary word into a matrix
    4. model:  Embedding  ->  average of the word vectors  ->  Dense (4 classes)
    5. train it twice: embedding frozen, and embedding fine-tuned
"""
import gzip
import shutil
import time
import urllib.request
import zipfile
from collections import Counter

import keras
import numpy as np
from keras import layers

import config

PAD, UNKNOWN = 0, 1   # word number 0 = padding, word number 1 = unknown word


# --------------------------------------------------------------------------- #
# GloVe file
# --------------------------------------------------------------------------- #
def download_glove(folder=config.GLOVE_DIR):
    """Download glove.6B.100d.txt into `folder` (skipped if it already exists)."""
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / config.GLOVE_FILENAME
    if target.exists():
        return target
    for url, kind in config.GLOVE_URLS:
        try:
            print("Downloading", url)
            download_file = folder / "download.tmp"
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=60) as response, open(download_file, "wb") as out:
                shutil.copyfileobj(response, out)
            if kind == "gz":
                with gzip.open(download_file, "rb") as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
            else:
                with zipfile.ZipFile(download_file) as z, z.open(config.GLOVE_FILENAME) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
            download_file.unlink()
            return target
        except Exception as error:
            print("  failed:", error)
    raise FileNotFoundError(
        "Could not download GloVe. Download glove.6B.zip from https://nlp.stanford.edu/projects/glove/ "
        f"and put {config.GLOVE_FILENAME} into the folder {folder}")


def load_glove_vectors(path, word_index):
    """Read the GloVe file and keep only the words that are in our vocabulary."""
    vectors = {}
    with open(path, encoding="utf-8") as file:
        for line in file:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in word_index and len(parts) == config.EMBEDDING_DIM + 1:   # skips the header line too
                vectors[word] = np.asarray(parts[1:], dtype="float32")
    return vectors


# --------------------------------------------------------------------------- #
# Text -> numbers
# --------------------------------------------------------------------------- #
def build_vocabulary(train_texts, max_words=config.MAX_WORDS):
    """Map the most frequent TRAINING words to numbers 2, 3, 4, ... (0 = padding, 1 = unknown)."""
    counts = Counter(word for text in train_texts for word in text.split())
    most_common = [word for word, _ in counts.most_common(max_words - 2)]
    return {word: number for number, word in enumerate(most_common, start=2)}


def texts_to_sequences(texts, word_index, max_len=config.MAX_LEN):
    """Each text becomes a list of exactly `max_len` numbers (cut or padded with 0)."""
    sequences = np.zeros((len(texts), max_len), dtype="int32")
    for row, text in enumerate(texts):
        numbers = [word_index.get(word, UNKNOWN) for word in text.split()][:max_len]
        sequences[row, :len(numbers)] = numbers
    return sequences


def build_embedding_matrix(word_index, glove_vectors):
    """Row i of the matrix = GloVe vector of the word with number i (zeros if GloVe lacks the word)."""
    matrix = np.zeros((len(word_index) + 2, config.EMBEDDING_DIM), dtype="float32")
    for word, number in word_index.items():
        if word in glove_vectors:
            matrix[number] = glove_vectors[word]
    return matrix


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def build_model(embedding_matrix, n_classes, trainable):
    """Embedding -> average pooling -> Dense. `trainable=False` freezes GloVe, `True` fine-tunes it."""
    keras.utils.set_random_seed(config.RANDOM_SEED)
    model = keras.Sequential([
        keras.Input(shape=(config.MAX_LEN,)),
        layers.Embedding(input_dim=embedding_matrix.shape[0], output_dim=config.EMBEDDING_DIM,
                         embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                         mask_zero=True,            # padding is ignored when averaging
                         trainable=trainable),
        layers.GlobalAveragePooling1D(),            # average of the word vectors = one vector per text
        layers.Dense(n_classes, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(config.LEARNING_RATE),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(model, X_train, y_train, epochs=config.EPOCHS, validation_split=0.0):
    """Train and return (history, seconds).

    By default the model sees ALL training data (like the TF-IDF models). Use
    validation_split=0.1 only for a diagnostic run that watches a held-out part of the TRAINING data.
    """
    start = time.perf_counter()
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=config.BATCH_SIZE,
                        validation_split=validation_split, verbose=0)
    return history, round(time.perf_counter() - start, 2)


def predict_labels(model, X, class_names):
    """Predicted class names (the class with the highest probability)."""
    probabilities = model.predict(X, verbose=0)
    return np.array(class_names)[probabilities.argmax(axis=1)]
