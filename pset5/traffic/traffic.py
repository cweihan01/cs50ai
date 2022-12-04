"""
Neural network to predict the label on a traffic sign image.
German Traffic Sign Recognition Benchmark (GTSRB) dataset.
"""

import cv2
import numpy as np
import os
import sys

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Activation
from tensorflow.keras.layers import Flatten, BatchNormalization, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt

# Declare constants and hyperparameters
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43  # 3 or 43 depending on data directory used
EPOCHS = 10
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    print("Loading data...")
    images, labels = load_data(sys.argv[1])
    print("Done loading data.")

    # Split data into training and testing sets
    labels = to_categorical(labels)  # categorical encoding for labels
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    print("Training model...")
    model = get_model()
    model.summary()

    # Callbacks (for larger epochs)
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=8, verbose=1)
    early_stopping = EarlyStopping(monitor="val_loss", patience=15, verbose=1)

    # Fit model on training data
    hist = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        callbacks=[reduce_lr, early_stopping],
        validation_data=(x_test, y_test),
    )
    print("Done training model.")

    # Evaluate neural network performance
    model.evaluate(x_test, y_test)

    # Model result analysis
    plot_model_loss(hist)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = list()
    labels = list()

    # Iterate through each folder (and its files) in data_dir
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
            images.append(image)
            labels.append(int(folder))
        print(f"Loaded folder {folder}.")

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # Image input shape - width * height * 3 RGB channels
    input_shape = (IMG_WIDTH, IMG_HEIGHT, 3)

    # Create a convolutional neural network
    model = Sequential(
        [
            # 1st convolutional layer
            Conv2D(
                16, (3, 3), padding="same", activation="relu", input_shape=input_shape
            ),
            BatchNormalization(),
            MaxPool2D((2, 2)),
            # 2nd convolutional layer
            Dropout(0.4),
            Conv2D(32, (3, 3), padding="same", activation="relu"),
            MaxPool2D((2, 2)),
            Flatten(),
            # Dense layers
            Dense(128),
            BatchNormalization(),
            Activation("relu"),
            Dropout(0.3),
            Dense(64),
            Activation("relu"),
            # Output layer, with `NUM_CATEGORIES` output units,
            # using `softmax` activation to predict probability for each class
            Dense(NUM_CATEGORIES),
            BatchNormalization(),
            Activation("softmax"),
        ]
    )

    # Compile model
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def plot_model_loss(hist):
    """Plot loss function of the trained model"""

    plt.plot(hist.history["loss"], color="#0000ff")
    plt.plot(hist.history["val_loss"], color="#ff0000")
    plt.title("Model Loss during Training")
    plt.xlabel("Epoch")
    plt.ylabel("Categorical Cross-Entropy Loss")
    plt.legend(["Training Loss", "Testing Loss"])
    plt.show()


if __name__ == "__main__":
    main()
