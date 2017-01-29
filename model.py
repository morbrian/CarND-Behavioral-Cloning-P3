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
import matplotlib.pyplot as plt
import random

random.seed(123)

HEIGHT = 160
WIDTH = 320
DRIVING_LOG_FILE = 'driving_log.csv'


# image display helper
def show_image(image):
    plt.figure()
    plt.imshow(image)
    plt.show()


# Returns (X_train, y_train) from (camera_images, steering_angle)
# data_folder: full or relative path to folder containing csv log and IMG directory
# driving_log_csv: name of csv file in the data_folder
def load_driving_data(data_folder):
    driving_log_csv = '/'.join([data_folder, 'driving_log.csv'])
    # read driving log data
    data_log = np.genfromtxt(driving_log_csv, usecols=(0, 3), delimiter=',', dtype=None)

    # split data into image names and steering angles
    image_names = data_log[1:, [0]].flatten().astype(str)
    steering_angle = data_log[1:, [1]].flatten().astype(float)

    # load images resized to expected dimensions
    images = np.ndarray(shape=(len(image_names), HEIGHT, WIDTH, 3))
    for (i, name) in enumerate(image_names):
        images[i][:, :] = cv2.resize(cv2.imread('/'.join([data_folder, image_names[i]])), (WIDTH, HEIGHT))

    return images, steering_angle


def process_driving_record(csv, data_folder, skip_straight=0.9):
    fields = csv.split(',')
    image_name = fields[0]
    steering_angle = float(fields[3])
    if steering_angle == 0.0 and random.uniform(0.0, 1.0) >= skip_straight:
        return np.array([]), np.array([])
    else:
        image = cv2.resize(cv2.imread('/'.join([data_folder, image_name])), (WIDTH, HEIGHT))
        flipped = cv2.flip(image, 0)
        return np.array([image, flipped]), np.array([steering_angle, -steering_angle])


def p2_dr_gen(data_folder, batch_size=500):
    driving_log_csv = '/'.join([data_folder, 'driving_log.csv'])
    # read driving log data
    line_count = sum(1 for _ in open(driving_log_csv))

    loop_count = 0
    while 1:
        data_log = np.genfromtxt(driving_log_csv, usecols=(0, 3), delimiter=',',
                                 skip_header=int(loop_count/2), max_rows=batch_size, dtype=None)

        # convert from tuple items to array of items
        data_log = np.array(data_log[:].tolist())
        # split data into image names and steering angles
        image_names = data_log[1:, [0]].flatten().astype(str)
        steering_angle = data_log[1:, [1]].flatten().astype(float)

        # load images resized to expected dimensions
        images = np.ndarray(shape=(len(image_names), HEIGHT, WIDTH, 3))
        for (i, name) in enumerate(image_names):
            images[i][:, :] = cv2.resize(cv2.imread('/'.join([data_folder, image_names[i]])), (WIDTH, HEIGHT))
            if loop_count % 2 == 1:
                images[i][:, :] = cv2.flip(images[i], 0)

        loop_count += 1
        if loop_count / 2 * batch_size >= line_count + batch_size:
            loop_count = 0
        yield images, steering_angle


def driving_data_generator(data_folder, batch=1000, skip_straight=0.9):
    driving_log_csv = '/'.join([data_folder, DRIVING_LOG_FILE])
    while 1:
        f = open(driving_log_csv)
        for line in f:
            images, steering_angles = process_driving_record(line, data_folder, skip_straight)
            if len(images) == 0:
                continue
            yield (images, steering_angles)
        f.close()


# model started with these two references then evolved through experimentation:
# https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
# https://github.com/commaai/research/blob/master/train_steering_model.py
def define_model(input_shape):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Flatten, Lambda
    from keras.layers.convolutional import Convolution2D
    from keras.layers.advanced_activations import ELU

    print("shape: {}".format(input_shape))
    model = Sequential([
        Lambda(lambda x: x/127.5 - 1., input_shape=input_shape, output_shape=input_shape),
        Convolution2D(24, 5, 5, activation='relu', border_mode='valid', subsample=(2, 2), input_shape=input_shape),
        Convolution2D(36, 5, 5, activation='relu', border_mode='valid', subsample=(2, 2)),
        Convolution2D(48, 3, 3, activation='relu', border_mode='valid', subsample=(2, 2)),
        Convolution2D(64, 3, 3, activation='relu', border_mode='valid', subsample=(1, 1)),
        Flatten(),
        Dense(1164),
        ELU(),
        Dropout(.2),
        Dense(100),
        ELU(),
        Dropout(.3),
        Dense(50),
        ELU(),
        Dropout(.5),
        Dense(10),
        ELU(),
        Dropout(.5),
        Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])

    return model


def tinker_model(input_shape):
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Flatten, Lambda
    from keras.layers.convolutional import Convolution2D
    from keras.layers.advanced_activations import ELU

    print("shape: {}".format(input_shape))
    model = Sequential([
        Lambda(lambda x: x/127.5 - 1., input_shape=input_shape, output_shape=input_shape),
        Convolution2D(24, 5, 5, activation='relu', border_mode='valid', subsample=(2, 2), input_shape=input_shape),
        ELU(),
        Dropout(.1),
        Convolution2D(36, 5, 5, activation='relu', border_mode='valid', subsample=(2, 2)),
        ELU(),
        Dropout(.1),
        Convolution2D(48, 3, 3, activation='relu', border_mode='valid', subsample=(2, 2)),
        ELU(),
        Dropout(.2),
        Convolution2D(64, 3, 3, activation='relu', border_mode='valid', subsample=(1, 1)),
        Flatten(),
        Dense(1164),
        ELU(),
        Dropout(.2),
        Dense(100),
        ELU(),
        Dropout(.5),
        Dense(50),
        ELU(),
        Dropout(.5),
        Dense(10),
        ELU(),
        Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss='mse',
                  metrics=['accuracy'])

    return model


def main():
    valid_data_folder = './data/simtracks'
    train_data_folder = './data/track1-given'

    # train_data_folder = './data/turn-left'
    # valid_data_folder = './data/turn-left'

    training_count = sum(1 for _ in open('/'.join([train_data_folder, DRIVING_LOG_FILE])))
    valid_count = sum(1 for _ in open('/'.join([valid_data_folder, DRIVING_LOG_FILE])))

    model = define_model((HEIGHT, WIDTH, 3))
    model.summary()

    # history = model.fit(X_train, y_train,  batch_size=64, nb_epoch=10, validation_split=0.2)
    model.fit_generator(p2_dr_gen(train_data_folder, batch_size=1000), samples_per_epoch=training_count * 2,
                        nb_epoch=500, validation_data=driving_data_generator(valid_data_folder), nb_val_samples=2000,
                        max_q_size=5, verbose=1)


if __name__ == "__main__":
    main()
