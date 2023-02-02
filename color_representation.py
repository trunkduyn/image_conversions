import os
import random

import cv2
import numpy
import numpy as np
from PIL import Image
from dotenv import find_dotenv, load_dotenv
from sklearn.cluster import KMeans
from trunky.data_vault.data_vault_config import DataVaultConfig
from trunky.data_vault.image_data_set.artistic_images import ArtisticImageDataSet
from trunky.data_vault.image_data_set.image_data_set import ImageDataSet
from trunky.data_vault.image_data_set.photos_2022_image_data_set import Photos2022ImageDataSet

from entropy_scenes import demo_entropy_mask, demo_entropy_enhanced, image_size_from_reduction, get_entropy_image, \
    display_n_images, non_zero_fraction


def centroid_histogram(clt):
    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    # return the histogram
    return hist


def plot_colors(hist, centroids):
    # initialize the bar chart representing the relative frequency
    # of each of the colors
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    # loop over the percentage of each cluster and the color of
    # each cluster
    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar


# USAGE
# python color_kmeans.py --image images/jp.png --clusters 3


def load_env():
    dotenv_path = find_dotenv(".env.prd")
    if dotenv_path:
        load_dotenv(dotenv_path)

    load_dotenv(find_dotenv(raise_error_if_not_found=True))


def create_color_representation(image_path, n_clusters):
    # load the image and convert it from BGR to RGB so that
    # we can dispaly it with matplotlib
    import matplotlib.pyplot as plt

    image = plt.imread(image_path)
    # image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # show our image
    plt.figure()
    plt.axis("off")
    plt.imshow(image)

    try:
        # reshape the image to be a list of pixels
        image_as_pixel_list = image.reshape((image.shape[0] * image.shape[1], 3))
    except ValueError as e:
        print(e)
        print("reshaping issue")
        return
    # cluster the pixel intensities
    print(image_path)
    print("make clustering")
    clt = KMeans(n_clusters=n_clusters)
    clt.fit(image_as_pixel_list)
    print("create label image")
    result = clt.labels_
    clusters = clt.cluster_centers_
    color_result = numpy.array([[int(clusters[r, i]) for i in range(3)] for r in result])
    labelled_image = color_result.reshape((image.shape[0], image.shape[1], 3))
    plt.figure()
    plt.axis("off")
    plt.imshow(labelled_image)
    temp_path = 'temp.png'
    plt.savefig(fname=temp_path, format='png')

    print("make color histogram")
    # build a histogram of clusters and then create a figure
    # representing the number of pixels labeled to each color
    hist = centroid_histogram(clt)
    bar = plot_colors(hist, clt.cluster_centers_)

    # show our color bart
    plt.figure()
    plt.axis("off")
    plt.imshow(bar)
    plt.show()

    print("Apply entropy")
    # since is converted to grey -> change labelled image into greyscale spread out image
    # sort centroid colors by brightness -> map evenly across greyscale -> convert and do entropy
    cluster_brightness = []
    for i in range(n_clusters):
        R, G, B = clt.cluster_centers_[i]
        Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
        cluster_brightness.append((Y, R, G, B, i))
    cluster_brightness.sort()
    cluster_mapping = {}
    cur_val = 0
    step_size = int(255 / n_clusters)
    for Y, R, G, B, i in cluster_brightness:
        cluster_mapping[i] = cur_val + step_size
        cur_val = cluster_mapping[i]
    color_result = numpy.array([cluster_mapping[r] for r in result])
    enhanced = color_result.reshape((image.shape[0], image.shape[1]))

    img = Image.fromarray(np.uint8(enhanced), 'L')
    img.show()
    temp_path = 'temp.png'
    img.save(temp_path)

    demo_entropy_sketch(temp_path, 2)


def value_to_bin(val, boundaries):
    for _min, _max in boundaries:
        if val >= _min and val <= _max:
            return _min
    return _min


def f_reverse2(orig, entr, entropy_max):
    # multiple f_reverse with the higher (brighter) half of the original image values (binned)
    # using entropy image, so if orig& entropy high -> return bright, else return reverse entropy
    if orig > (255 * 0.8) and entr > entropy_max * 0.8:
        return (orig / 255) * entr

    # # if there is a dark patch with low entropy -> make it dark
    # if orig < 255 / 10 and entr < entropy_max * 0.2:
    #     return entr

    # return reversed entropy
    return entropy_max - entr

def demo_entropy_sketch(image_path, image_reduction=10):
    with Image.open(image_path) as image:
        size = image.size
    size = image_size_from_reduction(image_reduction, size)

    entropy_image, input_image = get_entropy_image(image_path, size)

    mask_coverage = non_zero_fraction(entropy_image, 1.5)

    n_bins = 10
    entropy_max = max(max(e) for e in entropy_image)
    stepsize = entropy_max / n_bins
    boundaries = [[x * stepsize, x * stepsize + stepsize] for x in range(n_bins)]

    f_reverse = lambda x: value_to_bin(entropy_max - x, boundaries)
    reversed_image = []
    for x in entropy_image:
        reversed_image.append(list(map(f_reverse, x)))

    large_coverage = mask_coverage > 0.6
    if large_coverage:
        sketch = []
        for i, x in enumerate(reversed_image):
            sketch.append([])
            for j in range(len(x)):
                val = value_to_bin(f_reverse2(input_image[i][j], entropy_image[i][j], entropy_max), boundaries)
                sketch[-1].append(val)
    else:
        sketch = reversed_image

    display_n_images([input_image, entropy_image, reversed_image, sketch],
                     ["input_image", "entropy_image", f"reversed entropy {mask_coverage:.2f} {large_coverage}", "sketch"])



def get_image_sample(n_images):
    load_env()
    data_vault_config = DataVaultConfig.from_env()
    # image_data_set = ImageDataSet(os.path.join(data_vault_config.artistic_images_folder, 'colorz'), 'test', data_vault_config)

    # image_data_set = ImageDataSet(os.path.join(data_vault_config.artistic_images_folder, 'Drawings animals'), 'test', data_vault_config)
    # image_data_set = ImageDataSet(os.path.join(data_vault_config.artistic_images_folder, 'Drawings fantasy'), 'test', data_vault_config)
    # image_data_set = ArtisticImageDataSet(data_vault_config)
    image_data_set = Photos2022ImageDataSet(data_vault_config)

    images = image_data_set.get_image_description_data()

    for image in random.sample(images, n_images):
        yield image


if __name__ == '__main__':
    n_clusters = 10

    for image in get_image_sample(15):
        create_color_representation(image.image_path, n_clusters)
