
import os
import os.path as path
from time import time
from sklearn.utils import shuffle
from random import randint
import matplotlib.pyplot as plt

import numpy as np
import csv
import cv2

from scipy import ndimage
import random

CHANNELS = 3
HEIGHT = 160
WIDTH = 320

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

    def get_raw_path(self):
        """ path to raw csv file, the optional unprocessed source we will filter """
        return '/'.join([self.base_folder, self.raw_file])

    def get_pair_path(self):
        """ path to paired csv file, two column data for filename-angle pairs """
        return '/'.join([self.base_folder, self.pair_file])

    def get_img_output_path(self):
        """ optional output path where we will write images if image processing is requested """
        return '/'.join([self.base_folder, self.img_output])

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
        Save the loaded name/angle pairs to the csv_path log
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

    def load_images(self, log_every_n=1000, max_load=-1):
        """
        Load max_load images into memory and return the image array.
        Log a message every log_every_n images to show status.
        :param log_every_n: how often to log loading status
        :param max_load: max number of images to laod
        :return: array of images
        """
        t0 = time()
        if max_load < 0:
            max_load = len(self.names)
        images = np.ndarray(shape=(max_load, HEIGHT, WIDTH, CHANNELS), dtype=int)
        for i, name in enumerate(self.names[:max_load]):
            if i % log_every_n == 0:
                print("Loaded {} of {} total images.".format(i, max_load))
            images[i][:, :] = cv2.imread('/'.join([self.base_folder, name]))

        print("Loaded {} images".format(len(images)))
        print("Duration: {}s".format(round(time() - t0, 3)))

        return images

    def store_data_metrics(self, bins=20):
        """
        Save a histogram of angles with title and the number of bins
        :param bins: number of bins
        """
        title = "{}-{}".format(path.split(self.base_folder)[1], path.splitext(self.pair_file)[0])
        data_count = len(self.angles)
        data_range = (np.amin(self.angles), np.amax(self.angles))
        hist, bin_edges = np.histogram(self.angles, bins=bins, range=(-1.0, 1.0), density=False)
        plt.figure()
        plt.hist(self.angles, bins=bins, range=(-1.0, 1.0))
        plt.title(title)
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.savefig("/".join([self.base_folder, "{}.{}".format(title, 'png')]))
        plt.clf()

        with open('/'.join([self.base_folder, "metrics.txt"]), "a") as metrics_file:
            metrics_file.write("===== {} =====\n".format(title))
            metrics_file.write("data_count: {}\n data range: {}\n".format(data_count, data_range))
            metrics_file.write("hist: {}\n".format(hist))
            metrics_file.write("bin_edges: {}\n".format(bin_edges))

    def data_generator(self, batch_size=256, loop_forever=True, min_filter=0.001, min_keep_prob=0.1,
                       max_filter=1.0, max_keep_prob=1.0, bin_count=20):
        """
        Repeated select random batches of data from the dataset.
        The pool of data can be reduced by filtering large values or values near zero
        in order to improve how the data is balanced.
        The algorithm attempts to select samples evenly distributed across the
        spectrum of angles between -1.0 and 1.0.
        :param batch_size: number of samples yielded per batch
        :param max_keep_prob: probability of keeping large angles
        :param max_filter: angles greater than this are considered to large
        :param min_filter: angles less than this are treated as zero
        :param min_keep_prob: probability of keeping zero valued angles
        :param bin_count: number of histo bins to fill with balanced data
        """
        # first we filter out some of the angles new zero or very large angles
        interim_names, interim_angles = \
            self.filter_extremes(min_filter=min_filter, min_keep_prob=min_keep_prob, max_filter=max_filter,
                                 max_keep_prob=max_keep_prob)
        hist, bin_edges = np.histogram(interim_angles, bins=bin_count, range=(-1.0, 1.0), density=False)
        one_run = True
        while loop_forever or one_run:
            images = np.ndarray(shape=(batch_size, HEIGHT, WIDTH, CHANNELS))
            angles = np.ndarray(shape=(batch_size,))
            for i in range(batch_size):
                random_edge = random.randint(0, len(bin_edges)-2)
                pick_i = DataFolder.pick_in_range(interim_angles, bin_edges[random_edge], bin_edges[random_edge + 1])
                angles[i], images[i][:, :] = \
                    DataFolder.jitter_data(interim_angles[pick_i],
                                           ndimage.imread('/'.join([self.base_folder, interim_names[pick_i]])))
            if loop_forever:
                yield images, angles
            else:
                one_run = False
                return images, angles

    def filter_extremes(self, min_filter=0.001, min_keep_prob=0.1,
                        max_filter=1.0, max_keep_prob=1.0):
        """
        Filter the image/angle samples by excluding some larger angles or angles near zero.
        :param max_keep_prob: probability of keeping large angles
        :param max_filter: angles greater than this are considered to large
        :param min_filter: angles less than this are treated as zero
        :param min_keep_prob: probability of keeping zero valued angles
        :return:
        """
        exclude_min = np.any([abs(self.angles) > min_filter,
                              np.random.rand(len(self.angles)) < min_keep_prob], axis=0)
        exclude_max = np.any([abs(self.angles) < max_filter,
                              np.random.rand(len(self.angles)) < max_keep_prob], axis=0)
        exclude_filter = np.all([exclude_min, exclude_max], axis=0)
        interim_names, interim_angles = shuffle(self.names[exclude_filter], self.angles[exclude_filter])

        return interim_names, interim_angles

    @staticmethod
    def jitter_data(angle, image, shift_prob=0.8, shift_height_range=0.05, shift_width_range=0.05,
                    shift_channel_prob=0.3, flip_prob=0.5):

        """
        Perform random maniplations on the generated images, controlled by this method's parameters.
        Both the angle and image can be modified as part of the return value.
        :param angle: angle value for image
        :param image: original image
        :param shift_prob: probability of applying any shift
        :param shift_height_range: range to shift height as proportion of total height
        :param shift_width_range: range to shift width as proportion of total width
        :param shift_channel_prob: probability of shifting the rgb channels
        :param flip_prob: probability the image and angle will be flipped
        :return: jittered image and angle
        """
        if random.random() < flip_prob:
            image = np.fliplr(image)
            # image = cv2.flip(image, 1)
            angle *= -1

        if random.random() < shift_prob:
            h, w, c = image.shape
            h_dist = h * shift_height_range
            w_dist = w * shift_width_range
            c_dist = random.randint(1, 2) if random.random() < shift_channel_prob else 0
            mode = 'nearest' if random.random() < 0.5 else 'wrap'
            image = ndimage.shift(image,
                                  (random.uniform(-h_dist, h_dist),
                                   random.uniform(-w_dist, w_dist),
                                   random.uniform(-c_dist, c_dist)),
                                  mode=mode, order=0)

        return angle, image

    @staticmethod
    def pick_in_range(values, rmin=-1.0, rmax=1.0, pick_limit=20000, pick_creep=0.05):
        """
        find index of a random value from values that falls between rmin and rmax
        a value is guaranteed to be picked, but it is not guaranteed to be in original rmin-rmax range.
        if a value in range is not found within pick_limit iterations, the range is increased
        by pick_creep in both the positive and negative directions.
        :param values: array like, values to pick from
        :param rmin: minimum range
        :param rmax: maximum range
        :param pick_limit: max iterations before expand range constraint
        :param pick_creep: increment of range when pick not found after iteration limit
        :return: index of random value as close as possible to range constraint
        """
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


def initialize_folder(input_folder_name):
    """
    Reads raw csv file output by simulator and processes into list of file-angle pairs,
    and duplicates and augments image data into new subfolder
    :param input_folder_name: location of input csv file and images
    """
    # the original input data (such as from udacity, shared by others, or generated by me in the sim)
    input_folder = DataFolder(input_folder_name, 'pair_log.csv', raw_file='driving_log.csv',
                              name_cols=(0, 1, 2), angle_col=3)
    input_folder.load_data_log()
    input_folder.store_data_metrics()
    input_folder.persist_pairs()


def main():
    import optparse

    parser = optparse.OptionParser()
    parser.add_option('-d', '--data_folder', dest='data_folder', default='./data/moriarty-track1-combine',
                      help="path to folder containing driving_log.csv and image data.")
    options, args = parser.parse_args()

    initialize_folder(options.data_folder)

if __name__ == "__main__":
    main()
