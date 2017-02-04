
import model as m
import analyze as a
import cv2
import numpy as np
from os import path

data_folder = 'data/track1-given'
img_name_1 = 'center_2017_01_22_11_45_11_872.jpg'
img_name_2 = 'center_2017_01_22_13_31_14_078.jpg'
img_name_3 = 'center_2017_01_22_12_00_10_496.jpg'
img_name_4 = 'center_2017_01_22_12_00_14_573.jpg'


def test_preprocess_image():
    def rel_image_name(name):
        return '/'.join([data_folder, 'IMG', name])

    def test_on_image(name):
        print("show image: {}".format(name))
        assert path.isfile(name), "file does not exist"
        img = cv2.imread(name)
        a.show_image(img)
        img_processed = m.preprocess_image(img)
        a.show_image(img_processed, cmap='gray')
        img_flipped = m.preprocess_image(img, flip=True)
        a.show_image(img_flipped, cmap='gray')

    test_on_image(rel_image_name(img_name_1))
    test_on_image(rel_image_name(img_name_2))
    test_on_image(rel_image_name(img_name_3))
    test_on_image(rel_image_name(img_name_4))


def test_read_data_log():
    data = m.read_data_log(data_folder)

    assert 29396 == len(data), "unexpected data length"


def test_fetch_random_row():

    def show_result(all_data, sum_total, next_start):
        next_start, names, angles = m.fetch_random_row(all_data, sum_total, next_start)
        print("---")
        for name, angle in zip(names, angles):
            print("{}: {}, {}".format(next_start, name, angle))

        return next_start

    all_data = m.read_data_log(data_folder)

    sum_total = sum(all_data[:, 3].astype(float))
    weights = (abs(all_data[:, 3].astype(float)) + 0.0000001) / float(sum_total)
    print("sum_total: {}".format(sum_total))

    next_start = -1
    for i in range(3):
        next_start = show_result(all_data, weights, next_start)

# def test_fetch_rows():
#     BATCH=1
#
#     def test_batch(batch_size=BATCH, start=0):
#         names, angles = m.fetch_rows(all_data, batch_size, start=start)
#
#         assert len(names) == batch_size * 3, "unexpected row count for names"
#         assert len(angles) == batch_size * 3, "unexpected row count for angles"
#
#         for name, angle in zip(names, angles):
#             print("{}, {}".format(name, angle))
#
#     all_data = m.read_data_log(data_folder)
#     test_batch(batch_size=BATCH, start=5553)


def test_load_images():
    def test_load(image_names, mock_angles):
        images, angles = m.load_images(data_folder, image_names, mock_angles)
        for i in range(len(angles)):
            print("ang: {}".format(angles[i]))

        for i in range(len(images)):
            a.show_image(images[i], cmap='gray')

        assert len(images) == len(angles), "image and angle results should be same length"
        assert len(images) == len(image_names) * 2 and len(angles) == len(mock_angles) * 2, "inputs and outputs should be same length"

    image_names = np.array(['/'.join(['IMG', img_name_1]),
                            '/'.join(['IMG', img_name_2])])
    mock_angles = np.array([1.1, 2.2])

    test_load(image_names, mock_angles)


# test_preprocess_image()
# test_read_data_log()
test_fetch_random_row()

# test_load_images()
