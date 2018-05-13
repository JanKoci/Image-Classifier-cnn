#!/usr/bin/env python3

########################################
# IKR project
# Authors: Petr Zubalik, Jan Koci
# Date: 26.4.2018
# Brief: image classifier using keras
#       Convolutional neural network
########################################
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical
import numpy as np
import random
import cv2
import os

# build the convolution neural network
def build_cnn(width, height, depth, classes):
    model = Sequential()
    input_shape = (height, width, depth)

    # if channels are stored first
    if K.image_data_format() == "channels_first":
        input_shape = (depth, height, width)

    # first layer
    model.add(Conv2D(20, (5, 5), input_shape=input_shape))
    model.add(Activation("relu"))

    # pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # second layer
    model.add(Conv2D(50, (5, 5)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))

    # softmax classifier
    model.add(Dense(classes))
    model.add(Activation("softmax"))

    # return the network
    return model


epochs = 50
batch_size = 32
learning_rate = 1e-3

all_files = []
data = []
labels = []

# directories with images
target_train_dir = 'target_train/'
non_target_train_dir = 'non_target_train/'
target_test_dir = 'target_dev/'
non_target_test_dir = 'non_target_dev/'
all_dirs = [target_train_dir, non_target_train_dir, target_test_dir, non_target_test_dir]

# get all files from all directories
for directory in all_dirs:
    for f in os.listdir(directory):
        if (f.endswith(".png")):
            all_files.append(directory + f)

# shuffle these files
random.seed(150)
random.shuffle(all_files)

# read files, convert them into numpy array
# and store their label (target or non_target)
for f in all_files:
    image = cv2.imread(f)
    image = img_to_array(image)
    data.append(image)

    directory, f = f.split('/')
    if directory in ("target_train", "target_dev"):
        labels.append(1) # target = 1
    elif directory in ("non_target_train", "non_target_dev"):
        labels.append(0) # non_target = 0


# convert data to float and normalize them to range [0,1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# split data randomly into train and test set
(train_data, test_data, train_labels, test_labels) = train_test_split(data,
	labels, test_size=0.25, random_state=150)


train_labels = to_categorical(train_labels, num_classes=2)
test_labels = to_categorical(test_labels, num_classes=2)

# create image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")


# buil the Convolutional neural network (cnn)
model = build_cnn(width=80, height=80, depth=3, classes=2)

# prepare optimizer function
opt = Adam(lr=learning_rate, decay=learning_rate / epochs)

# compile the model
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# train the cnn
history = model.fit_generator(aug.flow(train_data, train_labels, batch_size=batch_size),
	validation_data=(test_data, test_labels), steps_per_epoch=len(train_data) // batch_size,
	epochs=epochs, verbose=1)

# save the model
model.save("cnn.model")
