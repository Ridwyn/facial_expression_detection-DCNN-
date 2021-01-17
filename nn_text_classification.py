import tensorflow as td
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)



# Data filtering
word_index = data.get_word_index()

word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])



# # Model Layers
# model = keras.Sequential()
# model.add(keras.layers.Embedding(100000,16))
# model.add(keras.layers.GlobalAveragePooling1D())
# model.add(keras.layers.Dense(16, activation="relu"))
# # Ouput Layer or last layer
# model.add(keras.layers.Dense(1, activation="sigmoid"))

# model.summary()
# # Set model compile format
# model.compile(optimizer="adam", loss="binary_crossentropy", metrics="accuracy")

# x_val = train_data[:10000]
# x_train = train_data[10000:]

# y_val = train_labels[:10000]
# y_train = train_labels[10000:]

# # Training Model
# fitModel = model.fit(x_train, y_train, epochs=10, batch_size=512 , validation_data=(x_val,y_val), verbose=1)



# Save model after traingng
# model.save("model.h5")


# View model results
# results = model.evaluate(test_data, test_labels)

# print(results)

# text_review = test_data[0]
# prediction = model.predict(text_review)
# print("Review: ")
# print(decode_review(text_review))
# print("Prediction: " + str(prediction[0]))
# print("Actual: " + str(test_labels[0]))
# print(results)


# ##### LOAD IN SAVED MODEL AND TEST
model = keras.models.load_model("neural_network/model.h5")

def review_encode(s):
    encode = [1]
    for word in s:
        if word in word_index:
            encode.append(word_index[word.lower()])
        else:
            encode.append(2)
    return encode

with open("neural_network/review.txt", encoding="utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace(".", "").replace("(","").replace(")","").replace(":","").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250)
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])