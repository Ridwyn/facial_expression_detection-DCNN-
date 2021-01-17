import tensorflow 
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images,train_labels),(test_images,test_labels)= data.load_data()


class_names =["T-shirts/top", "trouser","Pullover","Dress","Coat",
                "Sandal","Shirt","Sneaker","Bag","ankle boot"]

train_images = train_images/255.0
test_images = test_images/255.0


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
    ])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=5)

# Tetsing the model
# test_loss,test_acc = model.evaluate(test_images, test_labels)

# print("Test Acc: ", test_acc)


# Making prediction with model
prediction = model.predict(test_images)
print(class_names[np.argmax(prediction[0])])

for i  in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("prediction " + class_names[np.argmax(prediction[i])])
    plt.show()