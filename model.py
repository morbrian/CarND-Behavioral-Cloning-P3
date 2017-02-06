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

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.core import Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.advanced_activations import ELU
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import prep as p
import cv2


def drive_log_generator(data_folder, batch_size=500):
    # generator = ImageDataGenerator(
    #     samplewise_center=True,
    #     samplewise_std_normalization=True,
    #     width_shift_range=0.1,
    #     height_shift_range=0.1,
    #     channel_shift_range=[-50, 50],
    #     fill_mode="reflect",
    #     dim_ordering="tf"
    # )
    # images = data_folder.load_images()
    # while True:
    #     yield generator.flow(images, data_folder.angles,
    #                          batch_size=batch_size, shuffle=True,
    #                          save_to_dir='/'.join([data_folder.base_folder, 'keras_generated']))

    true_end = len(data_folder.angles)
    assert batch_size < true_end, "Error: batch_size must be less than total number of samples"

    next_start = 0
    next_end = next_start + batch_size
    while 1:
        angles = data_folder.angles[next_start:next_end]
        names = data_folder.names[next_start:next_end]
        images = np.ndarray((len(names), p.CROP_H, p.CROP_W, p.CHANNELS))
        for i, name in enumerate(names):
            # images[i][:, :] = p.prepare_image(cv2.imread('/'.join([data_folder.base_folder, name])), crop=False)
            images[i][:, :] = cv2.imread('/'.join([data_folder.base_folder, name]))

        next_start = next_end if next_end >= true_end else 0
        next_end = next_start + batch_size
        if next_end > true_end:
            next_end = true_end

        # should the denom instead be over the entire sample set?
        # sample_weights = (abs(angles) + 0.0001) / sum(angles)

        yield images, angles  # , sample_weights


# model started with these two references then evolved through experimentation:
# https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
# https://github.com/commaai/research/blob/master/train_steering_model.py
def define_model(input_shape):
    model = Sequential([
        Lambda(lambda x: x/127.5 - 1., input_shape=input_shape, output_shape=input_shape),
        Convolution2D(12, 8, 8, activation='relu', border_mode="valid", input_shape=input_shape),
        # Dropout(.1),
        Convolution2D(24, 5, 5, activation='relu', border_mode='valid', subsample=(2, 2)),
        # Dropout(.2),
        Convolution2D(36, 5, 5, activation='relu', border_mode='valid', subsample=(2, 2)),
        # Dropout(.3),
        Convolution2D(48, 3, 3, activation='relu', border_mode='valid', subsample=(2, 2)),
        # Dropout(.4),
        # MaxPooling2D(),
        Flatten(),
        Dense(1164, activation='relu'),
        # Dropout(.5),
        Dense(100, activation='relu'),
        # Dropout(.5),
        Dense(50, activation='relu'),
        # Dropout(.5),
        Dense(10, activation='relu'),
        # Dropout(.5),
        Dense(1)
    ])

    adam = Adam(lr=0.001)
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

    adam = Adam(lr=0.001)
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
    train_data_name = 'data/track1-given'
    valid_data_name = 'data/bc-track-1'

    training_folder = p.DataFolder(train_data_name, 'balanced_log.csv')
    training_folder.load_data_log()
    valid_folder = p.DataFolder(valid_data_name, 'balanced_log.csv')
    valid_folder.load_data_log()

    model = define_model((p.CROP_H, p.CROP_W, p.CHANNELS))
    model.summary()

    generator = ImageDataGenerator(
                samplewise_center=True,
                samplewise_std_normalization=True,
                width_shift_range=0.1,
                height_shift_range=0.1,
                # channel_shift_range=[-50, 50],
                fill_mode="reflect",
                dim_ordering="tf"
            )
    images = training_folder.load_images(log_every_n=1000)
    angles = training_folder.angles


    t0 = time()
    # history = model.fit(X_train, y_train,  batch_size=64, nb_epoch=10, validation_split=0.2)
    model.fit_generator(
        # drive_log_generator(training_folder, batch_size=2000),
        generator.flow(images, angles,
                       batch_size=256, shuffle=True,
                       save_to_dir=training_folder.provide_keras_folder()),
        samples_per_epoch=20480,
        nb_epoch=20,
        validation_data=drive_log_generator(valid_folder),
        nb_val_samples=2000,
        verbose=1)
    print("Duration: {}s".format(round(time() - t0, 3)))

    save_dir = save_model(model, '{}'.format(train_data_name))
    print('Model data saved to: {}'.format(save_dir))


if __name__ == "__main__":
    main()
