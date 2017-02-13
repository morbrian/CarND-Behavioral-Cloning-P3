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
from keras.layers import Cropping2D
from keras.layers.advanced_activations import ELU
from keras.layers.pooling import MaxPooling2D
from keras.layers.local import LocallyConnected2D
import prep as p


def moriarty_model(input_shape):
    """
    model started with the nvidia model and was modified to try other ideas.
    1. added dropout to discourage overfitting
    2. added maxpooling layer
    :param input_shape: initial shape of the input data
    :return: None
    """
    model = Sequential([
        Lambda(lambda x: x/127.5 - 1., input_shape=input_shape, output_shape=input_shape),
        Cropping2D(cropping=((65, 0), (0, 0))),
        Convolution2D(12, 8, 8, activation='elu', border_mode="valid"),
        Convolution2D(24, 5, 5, activation='elu', border_mode='valid', subsample=(2, 2)),
        Convolution2D(36, 5, 5, activation='elu', border_mode='valid', subsample=(2, 2)),
        Dropout(.3),
        Convolution2D(48, 3, 3, activation='elu', border_mode='valid', subsample=(2, 2)),
        Dropout(.4),
        Convolution2D(64, 3, 3, activation='elu', border_mode='valid', subsample=(2, 2)),
        MaxPooling2D((2, 2), strides=(2, 2)),
        Dropout(.4),
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

    adam = Adam(lr=0.001)
    model.compile(optimizer=adam,
                  loss='mse')

    return model


def nvidia_model(input_shape):
    """
    model started with this reference and includes minor adjustments:
    https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/
    adjustments include cropping
    :param input_shape: initial shape of the input data
    :return: None
    """
    model = Sequential([
        Lambda(lambda x: x/127.5 - 1., input_shape=input_shape, output_shape=input_shape),
        Cropping2D(cropping=((65, 0), (0, 0))),
        Convolution2D(24, 5, 5, activation='elu', border_mode='valid', subsample=(2, 2)),
        Convolution2D(36, 5, 5, activation='elu', border_mode='valid', subsample=(2, 2)),
        Convolution2D(48, 5, 5, activation='elu', border_mode='valid', subsample=(2, 2)),
        Convolution2D(64, 3, 3, activation='elu', border_mode='valid', subsample=(2, 2)),
        Convolution2D(64, 3, 3, activation='elu', border_mode='valid', subsample=(2, 2)),
        Flatten(),
        Dense(1164, activation='elu'),
        Dense(100, activation='elu'),
        Dense(50, activation='elu'),
        Dense(10, activation='elu'),
        Dense(1)
    ])

    adam = Adam(lr=0.001)
    model.compile(optimizer=adam,
                  loss='mse')

    return model


def commaai_model(input_shape):
    """
    model started with this reference and includes minor adjustments:
    https://github.com/commaai/research/blob/master/train_steering_model.py
    :param input_shape: initial shape of the input data
    :return: None
    """
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
                     input_shape=input_shape,
                     output_shape=input_shape))
    model.add(Cropping2D(cropping=((65, 0), (0, 0)))),
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


def save_model(model, data_path):
    import os
    from datetime import datetime

    stamp = datetime.now().strftime("%y-%m-%d-%H-%M")

    # architecture as model.json, and the weights as model.h5.
    save_dir = '/'.join(['./outputs', os.path.split(data_path)[1], stamp])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model_file = '/'.join([save_dir, 'model.h5'])
    model.save(model_file)

    return model_file


def build_and_train_model(train_data_path, model_name):
    from time import time
    training_folder = p.DataFolder(train_data_path, 'pair_log.csv', raw_file='driving_log.csv',
                                   name_cols=(0, 1, 2), angle_col=3)
    training_folder.load_data_log()
    training_folder.store_data_metrics()
    training_folder.persist_pairs()

    model = None
    input_shape = (p.HEIGHT, p.WIDTH, p.CHANNELS)
    if model_name == 'moriarty':
        model = moriarty_model(input_shape)
    elif model_name == 'nvidia':
        nvidia_model(input_shape)
    elif model_name == 'commaai':
        commaai_model(input_shape)
    else:
        print("ERROR: unknown model specified: {}", model_name)

    assert model is not None, "model required choose [moriarty, nvidia, commai]"

    model.summary()

    t0 = time()
    model.fit_generator(
        training_folder.data_generator(min_keep_prob=0.98,
                                       flip_prob=0.5,
                                       shift_prob=0.2,
                                       shift_channel_prob=0.8,
                                       blur_prob=0.0,
                                       shear_prob=0.5,
                                       noise_prob=0.2,
                                       shade_prob=0.8),
        samples_per_epoch=20000,
        nb_epoch=40,
        verbose=1)
    print("Duration: {}s".format(round(time() - t0, 3)))

    save_dir = save_model(model, '{}-{}'.format(model_name, train_data_path))
    print('Model data saved to: {}'.format(save_dir))


def main():
    import optparse

    parser = optparse.OptionParser()
    parser.add_option('-d', '--data_folder', dest='data_folder', default='./data/moriarty-borrowed-and-augmented',
                      help="path to folder containing driving_log.csv and image data.")
    parser.add_option('-m', '--model', dest='model', default='moriarty',
                      help="neural network model to use (moriarty, nvidia, commai)")
    options, args = parser.parse_args()

    build_and_train_model(options.data_folder, options.model)


if __name__ == "__main__":
    main()
