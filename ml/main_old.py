import os

import numpy as np
import pandas as pd
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from keras.preprocessing import image
from keras.utils import to_categorical
from keras_preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import tensorflow as tf

dataset = pd.read_csv('data/labels.csv')

TARGET_SIZE = (128, 128, 1)

checkpoint_path = "models/training.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

train_image = []
for i in tqdm(range(dataset.shape[0])):
    img = image.load_img('data/train/' + dataset['id'][i] + '.jpg', target_size=TARGET_SIZE,
                         grayscale=False)
    img = img_to_array(img)
    img = img / 255
    train_image.append(img)
x = np.array(train_image)

y = dataset['breed'].values
dic = dict(zip(np.unique(y), range(0, np.unique(y).__len__() + 1)))

y = dataset['breed'].map(dic).values

y = to_categorical(y, num_classes=dic.__len__())

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)


def build_model():
    _model = Sequential()
    _model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=tuple(map(sum, zip(TARGET_SIZE, (0, 0, 2))))))
    _model.add(Conv2D(64, (3, 3), activation='relu'))
    _model.add(MaxPooling2D(pool_size=(2, 2)))
    _model.add(Dropout(0.25))
    _model.add(Flatten())
    _model.add(Dense(128, activation='relu'))
    _model.add(Dropout(0.5))
    _model.add(Dense(dic.__len__(), activation='softmax'))
    _model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
    return _model


model = build_model()
model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), callbacks=[cp_callback])
model.save('models/dog-recognition.h5')