#
# Script used to create and train the model.
# Now that you have training data, it’s time to build and train your network!
#
# Use Keras to train a network to do the following:
#
# Take in an image from the center camera of the car. This is the input to your neural network.
# Output a new steering angle for the car.
#     You don’t have to worry about the throttle for this project, that will be set for you.
#
# Save your model architecture as model.json, and the weights as model.h5.

import cv2
import numpy as np
import random
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.core import Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import ELU

random.seed(4532)

CHANNELS = 3
HEIGHT = 160
WIDTH = 320
CROP_H = int(HEIGHT / 3)
CROP_YT = int(HEIGHT / 3)
CROP_YB = CROP_YT + CROP_H
CROP_W = WIDTH
CROP_XL = 0
CROP_XR = CROP_XL + CROP_W

DRIVING_LOG_FILE = 'driving_log.csv'


def preprocess_image(image, flip=False):
    # image = cv2.resize(image, (WIDTH, HEIGHT))
    # image = cv2.normalize(image, min, max, cv2.NORM_MINMAX)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image = image[CROP_YT:CROP_YB, CROP_XL:CROP_XR]
    # image = cv2.GaussianBlur(image, (5, 5), 0)
    # image = cv2.Canny(image, 100, 200)
    if flip:
        image = cv2.flip(image, 1)
    image = np.reshape(image, (image.shape[0], image.shape[1], CHANNELS))
    return image


def read_data_log(data_folder):
    driving_log_csv = '/'.join([data_folder, 'driving_log.csv'])

    # read driving log data
    line_count = sum(1 for _ in open(driving_log_csv))
    all_data = np.genfromtxt(driving_log_csv, usecols=(0, 1, 2, 3), delimiter=',', autostrip=True, dtype=None)

    assert (len(all_data) == line_count),  "unexpected size of data array"
    return np.array(all_data[:].tolist())


def fetch_random_row(all_data, weights, start=0, min_dist=0.1, min_prob=0.1):
    max_index = len(all_data)
    # pick random angle weighted by magnitude (distance from 0.0)
    prob = random.random()
    acc = 0.0
    pick = start
    while acc < prob:
        pick += 1
        if pick >= max_index:
            pick = 0
        if abs(all_data[pick, 3].astype(float)) > min_dist or random.random() < min_prob:
            acc += weights[pick]

    angles = np.empty(3, dtype=float)
    names = np.empty(3, dtype="<U40")

    angle = all_data[pick, 3].astype(float)
    angles[0] = angle
    angles[1] = angle
    angles[2] = angle
    names[0] = str(all_data[pick, 0], 'utf8')
    names[1] = str(all_data[pick, 1], 'utf8')
    names[2] = str(all_data[pick, 2], 'utf8')

    return pick, names, angles


def load_images(data_folder, image_names, steering_angles):
    count = len(steering_angles)
    images = np.ndarray(shape=(count * 2, CROP_H,  CROP_W, CHANNELS), dtype=float)
    angles = np.ndarray(count * 2)
    for (i, name) in enumerate(image_names):
        image = cv2.imread('/'.join([data_folder, name]))
        images[i][:, :] = preprocess_image(image, False)
        angles[i] = steering_angles[i]
        images[count + i][:, :] = preprocess_image(image, True)
        angles[count + i] = steering_angles[i] * -1

    return images, angles


def drive_log_generator(data_folder):
    all_data = read_data_log(data_folder)
    all_data = shuffle(all_data)

    sum_total = sum(all_data[:, 3].astype(float))
    weights = (abs(all_data[:, 3].astype(float)) + 0.0000001) / float(sum_total)

    next_start = -1
    while 1:
        next_start, image_names, steering_angles = \
            fetch_random_row(all_data, weights, start=next_start, min_dist=0.5, min_prob=0.05)

        # load images resized to expected dimensions
        images, steering_angles = load_images(data_folder, image_names, steering_angles)

        yield images, steering_angles


# model started with these two references then evolved through experimentation:
# https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
# https://github.com/commaai/research/blob/master/train_steering_model.py
def define_model(input_shape):
    model = Sequential([
        # Lambda(lambda x: x/127.5 - 1., input_shape=input_shape, output_shape=input_shape),
        Convolution2D(16, 8, 8, activation='elu', subsample=(4, 4), border_mode="same", input_shape=input_shape),
        Convolution2D(24, 5, 5, activation='elu', border_mode='valid', subsample=(2, 2)),
        Convolution2D(36, 5, 5, activation='elu', border_mode='valid', subsample=(2, 2)),
        Convolution2D(48, 3, 3, activation='elu', border_mode='valid', subsample=(2, 2)),
        Convolution2D(64, 3, 3, activation='elu', border_mode='valid', subsample=(1, 1)),
        # MaxPooling2D(),
        Flatten(),
        Dense(1164, activation='elu'),
        Dropout(.5),
        Dense(100, activation='elu'),
        Dropout(.5),
        Dense(50, activation='elu'),
        Dropout(.5),
        Dense(10, activation='elu'),
        Dropout(.5),
        Dense(1)
    ])

    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam,
                  loss='mse')

    return model


def commaai_model(input_shape):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
                     input_shape=input_shape,
                     output_shape=input_shape))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", input_shape=input_shape))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))

    adam = Adam(lr=0.0001)
    model.compile(optimizer=adam, loss="mse")

    return model


def save_model(model, name):
    import os
    from datetime import datetime

    stamp = datetime.now().strftime("%y-%m-%d-%H-%M")

    # architecture as model.json, and the weights as model.h5.
    save_dir = '/'.join(['./outputs', name, stamp])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    weights_file = '/'.join([save_dir, 'model.h5'])
    model_file = '/'.join([save_dir, 'model.json'])

    model.save_weights(weights_file, True)
    with open(model_file, 'w') as model_out:
        model_out.write(model.to_json())

    return save_dir


def main():
    from time import time
    train_data_name = 'track1-given'
    valid_data_name = 'bc-track-1'

    base_data_folder = './data'
    valid_data_folder = '/'.join([base_data_folder, valid_data_name])
    train_data_folder = '/'.join([base_data_folder, train_data_name])

    training_count = sum(1 for _ in open('/'.join([train_data_folder, DRIVING_LOG_FILE])))
    valid_count = sum(1 for _ in open('/'.join([valid_data_folder, DRIVING_LOG_FILE])))

    model = commaai_model((CROP_H, CROP_W, CHANNELS))
    # model = commaai_model((HEIGHT, WIDTH, CHANNELS))
    model.summary()

    t0 = time()
    # history = model.fit(X_train, y_train,  batch_size=64, nb_epoch=10, validation_split=0.2)
    model.fit_generator(
        drive_log_generator(train_data_folder),
        samples_per_epoch=training_count * 3,
        nb_epoch=20,
        validation_data=drive_log_generator(valid_data_folder),
        nb_val_samples=2000,
        max_q_size=5,
        verbose=1)
    print("Duration: {}s".format(round(time() - t0, 3)))

    save_dir = save_model(model, '{}'.format(train_data_name))
    print('Model data saved to: {}'.format(save_dir))


if __name__ == "__main__":
    main()
