import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

import seaborn as sns
from matplotlib import pyplot

GDRIVEPATH='fer2013.csv'

data_frame = pd.read_csv(GDRIVEPATH)
print(data_frame.shape)

data_frame.emotion.unique()

emotion_label_to_text = {0:'anger', 1:'disgust', 2:'fear', 3:'happiness', 4: 'sadness', 5: 'surprise', 6: 'neutral'}

data_frame.emotion.value_counts()

sns.countplot(data_frame.emotion)
pyplot.show()


import tensorflow as tf
# def augment_pixels(px, IMG_SIZE = 48):
#     image = np.array(px.split(' ')).reshape(IMG_SIZE, IMG_SIZE).astype('float32')
#     image = tf.image.random_flip_left_right(image.reshape(IMG_SIZE,IMG_SIZE,1))
#     # Pad image size
#     image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 12, IMG_SIZE + 12) 
#     # Random crop  to the original image size
#     image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 1])
#     # Random brightness to the mage
#     image = tf.image.random_brightness(image, max_delta=0.5) 
#     image = tf.clip_by_value(image, 0, 255)
#     augmented = image.numpy().reshape(IMG_SIZE,IMG_SIZE)
#     str_augmented = ' '.join(augmented.reshape(IMG_SIZE*IMG_SIZE).astype('int').astype(str))
#     return str_augmented

# valcounts = data_frame.emotion.value_counts()
# valcounts_diff = valcounts[valcounts.idxmax()] - valcounts
# for emotion_idx, aug_count in valcounts_diff.iteritems():
#     sampled = data_frame.query("emotion==@emotion_idx").sample(aug_count, replace=True)
#     sampled['pixels'] = sampled.pixels.apply(augment_pixels)
#     data_frame = pd.concat([data_frame, sampled])
#     print(emotion_idx, aug_count)



data_frame.Usage.unique()
# Check again to see if dataset size is similar across emotions.
sns.countplot(data_frame.emotion)
pyplot.show()


ig = pyplot.figure(1, (14, 14))

k = 0
for label in sorted(data_frame.emotion.unique()):
    for j in range(7):
        px = data_frame[data_frame.emotion==label].pixels.iloc[k]
        px = np.array(px.split(' ')).reshape(48, 48).astype('float32')

        k += 1
        ax = pyplot.subplot(7, 7, k)
        ax.imshow(px, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(emotion_label_to_text[label])
        pyplot.tight_layout()



INTERESTED_LABELS = [0, 1, 2, 3, 4, 5, 6]
data_frame = data_frame[data_frame.emotion.isin(INTERESTED_LABELS)]
data_frame.shape

img_array = data_frame.pixels.apply(lambda x: np.array(x.split(' ')).reshape(48, 48, 1).astype('float32'))
img_array = np.stack(img_array, axis=0)


img_array.shape

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
le = LabelEncoder()
img_labels = le.fit_transform(data_frame.emotion)
img_labels = np_utils.to_categorical(img_labels)
img_labels.shape

le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)



from keras.utils import to_categorical

from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(img_array, img_labels,
                                                    shuffle=True, stratify=img_labels,
                                                    test_size=0.1)
X_train.shape, X_valid.shape, y_train.shape, y_valid.shape

img_width = X_train.shape[1]
img_height = X_train.shape[2]
img_depth = X_train.shape[3]
num_classes = y_train.shape[1]



# Normalizing results, as neural networks are very sensitive to unnormalized data.
X_train = X_train / 255.
X_valid = X_valid / 255.


from keras import layers
from keras import models
from keras import optimizers
from keras.layers import Dropout, BatchNormalization, Activation
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D

model = models.Sequential()
model.add(Conv2D(6, (5, 5), input_shape=(img_width, img_height, img_depth), padding='same', activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(16, (5, 5), padding='same', activation = 'relu'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')


model.summary()

"""Using callbacks one is `early stopping` for avoiding overfitting training data
and other `ReduceLROnPlateau` for learning rate.
"""
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    min_delta=0.00005,
    patience=10,
    verbose=1,
    restore_best_weights=True,
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.6,
    patience=7,
    min_lr=1e-7,
    verbose=1,
)

callbacks = [
    early_stopping,
    lr_scheduler,
]

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
    rotation_range=40,
    rescale=1./255,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.4,
    horizontal_flip=True,
)
train_datagen.fit(X_train)

batch_size = 32 #batch size of 32 performs the best.
epochs = 20

history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_valid, y_valid),
    steps_per_epoch=len(X_train) / batch_size,
    epochs=epochs,
    callbacks=callbacks,
)

# model_yaml = model.to_yaml()
# with open("model.yaml", "w") as yaml_file:
#     yaml_file.write(model_yaml)
    
# model.save("model.h5")