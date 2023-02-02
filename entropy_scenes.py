import json
import random

import cv2
import imagehash as imagehash
import matplotlib.pyplot as plt
import numpy
import skimage
from PIL import Image, ImageOps
from numpy import count_nonzero
from skimage import img_as_ubyte
from skimage.filters.rank import entropy
from skimage.metrics import structural_similarity
from skimage.morphology import disk
from trunky.data_model.image_description import ImageDescription
from trunky.data_vault.data_vault_config import DataVaultConfig
from trunky.data_vault.image_data_set.artistic_images import ArtisticImageDataSet
from trunky.data_vault.image_data_set.photos_2022_image_data_set import Photos2022ImageDataSet

from compositions import Compositions
from original_rotation_converter import OriginalRotationConverter
from script_utils import load_env

rotator = OriginalRotationConverter()


def compare_images(image1, image2):
    return skimage.util.compare_images(image1, image2)


def get_entropy_image(image_path, size, entropy_disk_size=5):
    entropy_disk = disk(entropy_disk_size)
    with Image.open(image_path) as image:
        im = rotator.apply_orientation(image)
        grey_image = ImageOps.grayscale(im)
        # x = grey_image.thumbnail(size, Image.Resampling.LANCZOS)
        grey_image = grey_image.resize(size)
        input_image = img_as_ubyte(grey_image, force_copy=True)
    entropy_image = entropy(input_image, entropy_disk)
    return entropy_image, input_image


def get_masks(size):
    pass


def cut_below_mask(image_nd_array, cutoff_value):
    f = lambda x: x if x >= cutoff_value else 0
    mask = []
    for x in image_nd_array:
        mask.append(list(map(f, x)))
    return mask


def cut_above_mask(image_nd_array, cutoff_value):
    f = lambda x: x if x <= cutoff_value else 0
    mask = []
    for x in image_nd_array:
        mask.append(list(map(f, x)))
    return mask


def cut_off_mask(image_nd_array, cutoff_value):
    f = lambda x: 1 if x >= cutoff_value else 0
    mask = []
    for x in image_nd_array:
        mask.append(list(map(f, x)))
    return mask


def get_double_entropy_image(image_path, size, entropy_disk_size=5):
    entropy_disk = disk(entropy_disk_size)
    with Image.open(image_path) as image:
        grey_image = ImageOps.grayscale(image)
        grey_image = grey_image.resize(size)
        full_image = img_as_ubyte(grey_image, force_copy=True)
    entropy_image = entropy(full_image, entropy_disk)
    rescaled_entropy_image = entropy_image / numpy.max(numpy.abs(entropy_image), axis=0)
    double_entropy_image = entropy(rescaled_entropy_image, entropy_disk)
    return entropy_image, double_entropy_image, full_image


def display_n_images(images, titles, cmaps=None):
    n_images = len(images)
    assert n_images == len(titles)
    if cmaps:
        assert len(cmaps) == n_images
    else:
        cmaps = ["gray"] * len(images)

    fig, subplots = plt.subplots(ncols=n_images, figsize=(12, 4),
                                 sharex=True, sharey=True)

    for i, ax in enumerate(subplots):
        img0 = ax.imshow(images[i], cmap=cmaps[i])
        ax.set_title(titles[i])
        ax.axis("off")
        fig.colorbar(img0, ax=ax)

    fig.tight_layout()

    plt.show()


def demo_double_entropy(image_path, image_reduction=10):
    with Image.open(image_path) as image:
        size = image.size
    size = image_size_from_reduction(image_reduction, size)

    entropy_image, double_entropy_image, input_image = get_double_entropy_image(image_path, size)

    display_n_images([input_image, entropy_image, double_entropy_image],
                     ["input_image", "entropy_image", "double_entropy_image"])


def non_zero_fraction(mask, cutoff=None):
    if cutoff is not None:
        mask = [[v if v > cutoff else 0 for v in x] for x in mask]
    # count_nonzero: Default is None, meaning that non-zeros will be counted along a flattened version of a.
    return count_nonzero(mask) / sum(len(x) for x in mask)


def demo_entropy_mask(image_path, image_reduction=10):
    with Image.open(image_path) as image:
        size = image.size
    size = image_size_from_reduction(image_reduction, size)

    entropy_image, input_image = get_entropy_image(image_path, size)

    mask, _, _ = get_dynamic_max_20percentvalues(entropy_image)

    mask_coverage = non_zero_fraction(mask)

    display_n_images([input_image, entropy_image, mask],
                     ["input_image", "entropy_image", f"mask cov {mask_coverage:.2f}"])
def demo_entropy_enhanced(image_path, image_reduction=10):
    with Image.open(image_path) as image:
        size = image.size

    contrast, brightness = 1.2, -10
    size = image_size_from_reduction(image_reduction, size)

    entropy_image, input_image = get_entropy_image(image_path, size)

    enhanced = cv2.addWeighted(numpy.array(entropy_image), contrast, numpy.array(entropy_image), 0, brightness)

    display_n_images([input_image, entropy_image, enhanced],
                     ["input_image", "entropy_image", f"enhanced contrast"])


def get_dynamic_max_20percentvalues(entropy_image):
    max_ent = max(max(x) for x in entropy_image)
    min_ent = min(min(x) for x in entropy_image)
    # keep top 20 % of entropy range of image
    cutoff = max_ent - ((max_ent - min_ent) / 5)
    # mask = cut_off_mask(entropy_image, cutoff_value=cutoff)
    mask = cut_below_mask(entropy_image, cutoff_value=cutoff)
    return mask, max_ent, min_ent


def get_dynamic_min_20percentvalues(entropy_image):
    max_ent = max(max(x) for x in entropy_image)
    min_ent = min(min(x) for x in entropy_image)
    # keep top 20 % of entropy range of image
    cutoff = min_ent + ((max_ent - min_ent) / 5)
    mask = cut_above_mask(entropy_image, cutoff_value=cutoff)
    return mask, max_ent, min_ent


def demo_entropy(image_path, image_reduction=10):
    with Image.open(image_path) as image:
        size = image.size
    size = image_size_from_reduction(image_reduction, size)

    entropy_image, input_image = get_entropy_image(image_path, size)

    display_n_images([input_image, entropy_image], ["input_image", "entropy_image"])


def demo_mask_to_sketch(image_path, image_reduction):
    with Image.open(image_path) as image:
        size = image.size
    size = image_size_from_reduction(image_reduction, size)

    entropy_image, input_image = get_entropy_image(image_path, size)

    mask, max_entropy, min_entropy = get_dynamic_max_20percentvalues(entropy_image)
    mask_coverage = non_zero_fraction(mask)

    # worked really bad
    # mask_lower, max_entropy, min_entropy = get_dynamic_min_20percentvalues(entropy_image)

    f_reverse = lambda x: 1 if not x else 1 - (x / max_entropy)
    reversed_image = []
    for x in mask:
        reversed_image.append(list(map(f_reverse, x)))

    display_n_images([input_image, entropy_image, reversed_image, mask],
                     ["input_image", "entropy_image", f"mask reversed {mask_coverage:.2f}", "mask"])


def image_size_from_reduction(image_reduction, size):
    size = tuple(int(s / image_reduction) for s in size)
    return size


def sort_by_mask_coverage(dataset, image_reduction):
    images = dataset.get_image_description_data()
    by_coverage = []
    n_check = 70
    for i, image in enumerate(images):
        if i % 100 == 0:
            print(f"[{i}/{len(images)}]")
        with Image.open(image.image_path) as im:
            size = im.size
        size = image_size_from_reduction(image_reduction, size)

        entropy_image, input_image = get_entropy_image(image.image_path, size)
        mask, max_entropy, min_entropy = get_dynamic_max_20percentvalues(entropy_image)
        mask_coverage = non_zero_fraction(mask)
        by_coverage.append((mask_coverage, i, image))

        # if i == 800:
        #     break

    by_coverage.sort()
    top_n, bottom_n = by_coverage[:n_check], by_coverage[-n_check:]

    json_obj = {im.image_path: mc for mc, i, im in top_n}
    json_obj.update({im.image_path: mc for mc, i, im in bottom_n})
    with open("output.json", "w") as outfile:
        json.dump(json_obj, outfile, indent=4)

    for coverage_cat in [top_n, bottom_n]:
        for c, i, image in coverage_cat:
            with Image.open(image.image_path) as im2:
                size = im2.size
            size = image_size_from_reduction(image_reduction, size)
            entropy_image, input_image = get_entropy_image(image.image_path, size)
            mask, max_entropy, min_entropy = get_dynamic_max_20percentvalues(entropy_image)
            display_n_images([input_image, entropy_image, mask],
                             ["input_image", "entropy_image", "mask"])


def image_hash_difference(im1, im2):
    hash1 = imagehash.average_hash(im1)
    hash2 = imagehash.average_hash(im2)
    # should be percentage true or something?
    return hash1 - hash2

def compare_entropy_to_compositions(im: ImageDescription, image_reduction):
    with Image.open(im.image_path) as im2:
        original_size = im2.size
    size = image_size_from_reduction(image_reduction, original_size)
    entropy_image, input_image = get_entropy_image(im.image_path, size)


    composition_matches_ssi = []
    composition_matches_hash = []

    Xrescaled_entropy_image = entropy_image / numpy.max(numpy.abs(entropy_image), axis=0)

    compositions = Compositions()
    for composition_image in compositions.iter_all_images(size):

        ssi = structural_similarity(img_as_ubyte(composition_image), img_as_ubyte(Xrescaled_entropy_image), win_size=5)
        composition_matches_ssi.append((ssi, composition_image))

        hash_diff_entropy = Image.fromarray(entropy_image)
        hash_dif = image_hash_difference(composition_image, hash_diff_entropy)
        composition_matches_hash.append((hash_dif, composition_image))

    images = [input_image, entropy_image]
    images.extend(im for v, im in composition_matches_ssi[:2])
    images.extend(im for v, im in composition_matches_hash[:2])
    display_n_images(images, ["input", "entropy", "ssi 1", "ssi 2", "hash 1", "hash 2"])


if __name__ == '__main__':
    # output_path = os.path.join(config.get_string("CONVERSION_RESULT_FOLDER"), "similar_images_ssi")

    load_env()
    data_vault_config = DataVaultConfig.from_env()

    # image_data_set = ArtisticImageDataSet(data_vault_config)
    # reduction=5
    image_data_set = Photos2022ImageDataSet(data_vault_config)
    reduction = 10
    n_images = 5

    # sort_by_mask_coverage(image_data_set, reduction)

    images = image_data_set.get_image_description_data()

    # for image in images[:10]:
    for image in random.sample(images, n_images):
        # demo_double_entropy(image.image_path, image_reduction=reduction)
        # demo_entropy_mask(image.image_path, image_reduction=reduction)
        demo_entropy_enhanced(image.image_path, image_reduction=reduction)

        # compare_entropy_to_compositions(image, reduction)
        # demo_mask_to_sketch(image.image_path, image_reduction=reduction)
        # demo_entropy(image.image_path, image_reduction=reduction)
