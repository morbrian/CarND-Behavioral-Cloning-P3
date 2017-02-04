import model as m
import numpy as np
import matplotlib.pyplot as plt

data_folder = 'data/track1-given'


# image display helper
def show_image(image, cmap='jet'):
    if image.shape[2] == 1:
        image = image[:, :, 0] * 3
    plt.figure()
    plt.imshow(image, cmap=cmap)
    plt.show()


def histogram(title, numbers):
    data_range = (np.amin(numbers), np.amax(numbers))
    print("data range: {}".format(data_range))
    plt.hist(numbers, bins=21, range=data_range)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    #
    # fig = plt.gcf()

    plt.show()


def steering_histo(title):
    data = m.read_data_log(data_folder)
    sum_total = sum(data[:, 3].astype(float))
    weights = (abs(data[:, 3].astype(float)) + 0.0000001) / float(sum_total)
    fetch_count = len(data)
    angles = np.ndarray(fetch_count * 3)
    for i in range(fetch_count):
        next_start, names, angles[i * 3:i * 3 + 3] = m.fetch_random_row(data,
                                                                        weights=weights,
                                                                        start=-1,
                                                                        min_dist=0.5,
                                                                        min_prob=0.05)

    # for i in range(len(angles)):
    #     print("{}: {}".format(i, angles[i]))

    histogram(title, angles)


def main():
    # steering_histo("All Data")
    steering_histo("Filtered Data")


if __name__ == "__main__":
    main()
