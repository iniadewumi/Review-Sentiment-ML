import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import re
data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

word_index = data.get_word_index()

word_index = {k: (v + 3) for k, v in word_index.items()}


word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reversed_word_index = dict([(value, key) for (key, value) in word_index.items()])
train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=256)

def decode_review(text):
    return " ".join([reversed_word_index.get(i, "?") for i in text])
def encode_review(text):
    return [word_index.get(i, 2) for i in text]


def train():
    global model
    model = keras.Sequential([
        keras.layers.Embedding(50000, 16),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(16, activation=tf.nn.relu),
        keras.layers.Dense(8, activation=tf.nn.relu),
        keras.layers.Dense(1, activation=tf.nn.sigmoid)
    ])


    model.summary()
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    mf = model.fit(train_data, train_labels, epochs=40, batch_size=512, validation_split=0.2, validation_data=(test_data, test_labels))

    results = model.evaluate(x=test_data, y=test_labels)

    model.save("MODELS/Text_Classification.h5")


model = keras.models.load_model("MODELS/Text_Classification.h5")

raw_review = "Ew"

alphab_string = re.sub(r'[^A-Za-z0-9 ]+', '', raw_review)
input_string = encode_review(alphab_string.lower())
input_array = keras.preprocessing.sequence.pad_sequences([input_string], value=word_index["<PAD>"], padding="post", maxlen=256)

out = ["Negative", "Positive"]

result = 1 - model.predict(input_array)[0][0]

print(result, out[round(result)])
