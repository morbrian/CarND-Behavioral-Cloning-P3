import model as m
import numpy as np
import matplotlib.pyplot as plt
import prep as p


# image display helper
def show_image(image, cmap='jet'):
    if image.shape[2] == 1:
        image = image[:, :, 0] * 3
    plt.figure()
    plt.imshow(image, cmap=cmap)
    plt.show()


def histogram(title, numbers, bins=20):
    hist, bin_edges = np.histogram(numbers, bins=bins, range=(-1.0, 1.0), density=False)
    print("hist: {}".format(hist))
    print("bin_edges: {}".format(bin_edges))

    data_count = len(numbers)
    data_range = (np.amin(numbers), np.amax(numbers))
    print("data_count: {}, data range: {}".format(data_count, data_range))
    plt.hist(numbers, bins=bins, range=data_range)
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    #
    # fig = plt.gcf()

    plt.show()


def review_data_folder(input_folder_name):
    # review processed data
    processed_folder = p.DataFolder(input_folder_name, 'processed_log.csv')
    processed_folder.load_data_log()
    histogram("Processed Data", processed_folder.angles)

    # review balanced data
    balanced_folder = p.DataFolder(input_folder_name, 'balanced_log.csv')
    balanced_folder.load_data_log()
    histogram("Balanced Data", balanced_folder.angles)


def main():
    review_data_folder('./data/track1-given')
    review_data_folder('./data/bc-track-1')
    review_data_folder('./data/recover')
    review_data_folder('./data/first-curve')
    review_data_folder('./data/combine2')


if __name__ == "__main__":
    main()
