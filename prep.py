
import os
import os.path as path
from time import time
from sklearn.utils import shuffle
from random import randint

import numpy as np
import csv
import cv2

CHANNELS = 3
HEIGHT = 160
WIDTH = 320
CROP_H = 95
CROP_YT = 65
CROP_YB = CROP_YT + CROP_H
CROP_W = WIDTH
CROP_XL = 0
CROP_XR = CROP_XL + CROP_W

IMG_NAME_TYPE = "<U100"


class DataFolder:
    """
    helper class for loading csv and image data from folder
    """

    base_folder = "" #dtype(string)
    pair_file = None #dtype(string)
    raw_file = None #dtype(string)
    img_output = None #dtype(string)
    name_cols = () #dtype((int))
    angle_col = 1 #dtype(int)
    data_log = None #dtype (whatever np.genfromtxt returns)
    names = None #dtype np.array() of "U<100"
    angles = None #dtype np.array() of float

    def __init__(self, base_folder, pair_file, raw_file=None, img_output='img_output',
                 name_cols=(0,), angle_col=1):
        """
        DataFolder constructor
        :param base_folder: string path to data folder
        :param pair_file: string short name of file
        :param raw_file: origin file name to read from to create the pair_file this object represents
        :param img_output: output folder for image processing output
        :param name_cols: (int) list of columns to associate to angle
        :param angle_col: int column containing angle value
        """
        self.base_folder = base_folder
        self.raw_file = raw_file
        self.pair_file = pair_file
        self.img_output = img_output
        self.name_cols = name_cols
        self.angle_col = angle_col

    # path to raw csv file
    def get_raw_path(self):
        return '/'.join([self.base_folder, self.raw_file])

    # path to paired csv file
    def get_pair_path(self):
        return '/'.join([self.base_folder, self.pair_file])

    def get_img_output_path(self):
        return '/'.join([self.base_folder, self.img_output])

    def provide_keras_folder(self, clean=True):
        output_path = '/'.join([self.base_folder, 'keras_generated'])
        if not path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        elif clean:
            for file in os.listdir(output_path):
                os.remove('/'.join([output_path, file]))

        # clean the output dir
        for file in os.listdir(output_path):
            os.remove(file)

        return output_path

    def load_data_log(self):
        """
        load data from csv to initialize data_log, name_vals and angle_vals properties
        :return: None
        """
        if self.raw_file is not None:
            csv_path = self.get_raw_path()
        elif self.pair_file is not None:
            csv_path = self.get_pair_path()
        else:
            csv_path = None

        assert csv_path is not None, "Must specify either raw_file or pair_file to load data"
        assert path.exists(csv_path), "File not found: {}".format(csv_path)
        print("data loading from: {}".format(csv_path))

        usecols = self.name_cols + (self.angle_col,)
        self.data_log = \
            np.array(np.genfromtxt(csv_path, usecols=usecols, delimiter=',', autostrip=True, dtype=None)[:].tolist())

        # we convert the genfromtxt output to something more readily usable,
        # and if multiple image colums were specified we reduce to single name list
        # and broadcast the angle value associated with each name
        broadcast_size = len(self.data_log) * len(self.name_cols)
        col_count = len(self.name_cols)
        self.names = np.ndarray(broadcast_size, dtype=IMG_NAME_TYPE)
        self.angles = np.empty(broadcast_size, dtype=float)
        for i, row in enumerate(self.data_log):
            angle = row[col_count].astype(float)
            for col in range(len(self.name_cols)):
                self.names[i * col_count + col] = row[col].astype(IMG_NAME_TYPE)
                self.angles[i * col_count + col] = angle

    def persist_pairs(self):
        """
        Save the loaded name/angle pairs to the pair_file log
        :return: None
        """
        assert self.names is not None, "names values not initialized, try loading the data_log first."
        assert self.angles is not None, "angles values not initialized, try loading the data_log first."

        csv_path = self.get_pair_path()
        if not path.exists(path.dirname(csv_path)):
            os.makedirs(path.dirname(csv_path))

        print("storing data as: {}".format(csv_path))
        with open(csv_path, 'w') as file:
            writer = csv.writer(file, delimiter=',')
            for name, angle in zip(self.names, self.angles):
                writer.writerow([name, angle])

    def augment_data(self, min_filter=0.001, clean=False):
        """
        use existing data, plus generate additional data through transformations on the existing data
        :param min_filter: skip augmenting images with angle magnitude less than min_filter
        :param clean: use True to empty the output folder before createing new images.
        :return: None
        """
        output_path = self.get_img_output_path()
        if not path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        elif clean:
            for file in os.listdir(output_path):
                os.remove('/'.join([output_path, file]))

        # clean the output dir
        for file in os.listdir(output_path):
            os.remove(file)

        exclude_min = abs(self.angles) > min_filter
        filtered_names = self.names[exclude_min]
        filtered_angles = self.angles[exclude_min]

        new_size = len(self.names) + len(filtered_names)
        new_names = np.ndarray(new_size, dtype=self.names.dtype)
        new_angles = np.ndarray(new_size, dtype=self.angles.dtype)

        print("Duplicate {} images in {}".format(len(self.names), output_path))
        for i, (name, angle) in enumerate(zip(self.names, self.angles)):
            new_names[i], new_angles[i], new_image = \
                DataFolder._duplicate(name, angle, cv2.imread('/'.join([self.base_folder, name])), {'prefix': self.img_output})
            cv2.imwrite('/'.join([self.base_folder, new_names[i]]), new_image)

        print("Create {} flipped images in {}".format(len(filtered_names), output_path))
        for i, (name, angle) in enumerate(zip(filtered_names, filtered_angles), start=len(self.angles)):
            new_names[i], new_angles[i], new_image = \
                DataFolder._hflip(name, angle, cv2.imread('/'.join([self.base_folder, name])), {'prefix': self.img_output})
            cv2.imwrite('/'.join([self.base_folder, new_names[i]]), new_image)

        self.names = new_names
        self.angles = new_angles

    def process_data(self, clean=False):
        """
        Apply image processing preparation to the images, uses the same routines that will
        be used when getting data from the simulator during auto-driving.
        :param clean: use True to empty the output folder before creating new images.
        :return: None
        """
        output_path = self.get_img_output_path()
        if not path.exists(output_path):
            os.makedirs(output_path, exist_ok=True)
        elif clean:
            for file in os.listdir(output_path):
                os.remove('/'.join([output_path, file]))

        new_names = np.ndarray(len(self.names), dtype=self.names.dtype)

        print("Process {} images into {}".format(len(self.names), output_path))
        for i, name in enumerate(self.names):
            new_names[i], new_image = \
                DataFolder._preprocess(name, cv2.imread('/'.join([self.base_folder, name])), {'prefix': self.img_output})
            cv2.imwrite('/'.join([self.base_folder, new_names[i]]), new_image)

        self.names = new_names

    def balance_data(self, min_filter=0.001, min_keep_prob=0.1, max_filter=1.0, max_keep_prob=1.0):
        """
        Produce a balanced selection of data from the already processed set.
        :param max_keep_prob: probability of keeping large angles
        :param max_filter: angles greater than this are considered to large
        :param min_filter: angles less than this are treated as zero
        :param min_keep_prob: probability of keeping zero valued angles
        :return: None
        """
        exclude_min = np.any([abs(self.angles) > min_filter, np.random.rand(len(self.angles)) < min_keep_prob], axis=0)
        exclude_max = np.any([abs(self.angles) < max_filter, np.random.rand(len(self.angles)) < max_keep_prob], axis=0)
        exclude_filter = np.all([exclude_min, exclude_max], axis=0)

        interim_names, interim_angles = shuffle(self.names[exclude_filter], self.angles[exclude_filter])

        bin_count = 20
        hist, bin_edges = np.histogram(interim_angles, bins=bin_count, range=(-1.0, 1.0), density=False)
        min_count = 100
        balanced_names = np.empty(min_count * bin_count, dtype=IMG_NAME_TYPE)
        balanced_angles = np.empty(len(interim_angles))
        for edge in range(bin_count):
            for i in range(min_count):
                pick_i = DataFolder._pick_in_range(interim_angles, bin_edges[edge], bin_edges[edge + 1])
                balanced_names[min_count * edge + i] = interim_names[pick_i]
                balanced_angles[min_count * edge + i] = interim_angles[pick_i]

        self.names = balanced_names
        self.angles = balanced_angles

    def load_images(self, log_every_n=1000, max_load=-1):
        t0 = time()
        if max_load < 0:
            max_load = len(self.names)
        images = np.ndarray(shape=(max_load, CROP_H, CROP_W, CHANNELS), dtype=int)
        for i, name in enumerate(self.names[:max_load]):
            if i % log_every_n == 0:
                print("Loaded {} of {} total images.".format(i, max_load))
            images[i][:, :] = cv2.imread('/'.join([self.base_folder, name]))

        print("Loaded {} images".format(len(images)))
        print("Duration: {}s".format(round(time() - t0, 3)))

        return images

    @staticmethod
    def _pick_in_range(values, rmin=-1.0, rmax=1.0, pick_limit=20000, pick_creep=0.05):
        min_i = 0
        max_i = len(values) - 1
        pick_count = 0
        creep = 0
        while True:
            pick_count += 1
            if pick_count % pick_limit == 0:
                creep += pick_creep
            index = randint(min_i, max_i)
            if rmin - creep < values[index] < rmax + creep:
                break

        return index

    @staticmethod
    def _preprocess(name, image, props):
        return DataFolder._new_name(name, props['prefix'], ''), prepare_image(image)

    @staticmethod
    def _duplicate(name, angle, image, props):
        return DataFolder._new_name(name, props['prefix'], ''), angle, image

    @staticmethod
    def _hflip(name, angle, image, props):
        return DataFolder._new_name(name, props['prefix'], '-hflip'), -1 * angle, cv2.flip(image, 1)

    @staticmethod
    def _new_name(name, prefix, tag):
        """
        replace prefixed directory, and add tag befor extension.
        example: path1/image.jpg becomes path2/image-tag.jpg
        :param name: original image name relative to datafolder
        :param prefix: the new prefix
        :param tag: tag name to insert
        :return: reformed name string
        """
        name_only = name.partition('/')[2]
        name_short = name_only.partition('.')[0]
        name_ext = name_only.partition('.')[2]
        return "{}/{}{}.{}".format(prefix, name_short, tag, name_ext)


def prepare_image(image):
    # input_img_size=(HEIGHT, WIDTH, CHANNELS)
    # image = cv2.resize(image, (WIDTH, HEIGHT))
    image = image[CROP_YT:CROP_YB, CROP_XL:CROP_XR]
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.normalize(image, -1.0, 1.0, cv2.NORM_MINMAX)
    # image = cv2.GaussianBlur(image, (5, 5), 0)
    # image = cv2.Canny(image, 100, 200)
    # image = np.reshape(image, (image.shape[0], image.shape[1], CHANNELS))
    return image


def initialize_folder(input_folder_name, min_filter=0.001, min_keep_prob=0.1, max_filter=0.8, max_keep_prob=0.2):
    # the original input data (such as from udacity, shared by others, or generated by me in the sim)
    input_folder = DataFolder(input_folder_name, 'pair_log.csv', raw_file='driving_log.csv',
                              name_cols=(0, 1, 2), angle_col=3)
    input_folder.load_data_log()
    input_folder.persist_pairs()

    # load pair data and create augmented image folder
    # see the method docs, but this does things like flipping and blurring images
    augmented_folder = DataFolder(input_folder_name, 'augmented_log.csv',
                                  raw_file='pair_log.csv', img_output='img_augmented',)
    augmented_folder.load_data_log()
    t0 = time()
    augmented_folder.augment_data(clean=True)
    print("Duration: {}s".format(round(time() - t0, 3)))
    augmented_folder.persist_pairs()

    # preprocessed data folder contains modified/cropped images we can use for training
    processed_folder = DataFolder(input_folder_name, 'processed_log.csv',
                                  raw_file='augmented_log.csv', img_output='img_processed')
    processed_folder.load_data_log()
    t0 = time()
    processed_folder.process_data(clean=True)
    print("Duration: {}s".format(round(time() - t0, 3)))
    processed_folder.persist_pairs()

    # balanced data folder contains new CSV file listing balanced subset of total data
    balanced_folder = DataFolder(input_folder_name, 'balanced_log.csv',
                                 raw_file='processed_log.csv', img_output='img_balanced')
    balanced_folder.load_data_log()
    balanced_folder.balance_data(min_filter=min_filter, min_keep_prob=min_keep_prob, max_filter=0.8, max_keep_prob=0.2)
    balanced_folder.persist_pairs()


def main():
    # set up our various data storage folders

    initialize_folder('./data/track1-given', min_filter=0.001, min_keep_prob=0.1)
    initialize_folder('./data/bc-track-1', min_filter=0.001, min_keep_prob=0.01)
    initialize_folder('./data/recover', min_filter=0.001, min_keep_prob=0.00, max_filter=0.8, max_keep_prob=0.2)
    initialize_folder('./data/first-curve', min_filter=0.001, min_keep_prob=0.01)
    # initialize_folder('./data/combine', min_filter=0.001, min_keep_prob=0.01)

if __name__ == "__main__":
    main()
