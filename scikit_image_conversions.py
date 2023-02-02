import os
import random
import shutil
from collections import defaultdict

import matplotlib.pyplot as plt
from skimage.filters.rank import entropy
import numpy
from PIL import Image

from skimage.metrics import structural_similarity, hausdorff_pair, hausdorff_distance, variation_of_information
from skimage.morphology import disk
from skimage.util import img_as_ubyte
# ============================================
# ============================================
from trunky.data_vault.data_vault_config import DataVaultConfig
from trunky.data_vault.image_data_set.image_data_set import ImageDataSet
from trunky.data_vault.image_data_set.photos_2022_image_data_set import Photos2022ImageDataSet
from trunky.general_utils import config
from trunky.general_utils.path import create_dir_if_doesnt_exist

from script_utils import load_env
from temp_vars import PERSONAL_PHOTO_FOLDER, MAC_PHOTOS_LABELS_FOLDER


def get_sample_image_descriptions(n_comparison_sample: int, description_by_label):
    samples = {}
    for key in description_by_label.keys():
        if key == "unlabelled":
            continue
        image_descriptions = description_by_label[key]
        if len(image_descriptions) < n_comparison_sample:
            continue
        samples[key] = random.sample(image_descriptions, n_comparison_sample)
    return samples


def x(image_path):
    size = 128, 128
    with Image.open(image_path) as im:
        im.thumbnail(size)


def haussdorf():
    # ssi = structural_similarity(img_as_ubyte(im1), img_as_ubyte(resized), win_size=5, multichannel=True)
    # hausd = hausdorff_distance(numpy.array(im1), numpy.array(resized))
    # hausdorf_distance_per_label[label].append((hausd, unlabelled_image.image_path))
    # if len(hausdorf_distance_per_label[label]) > max_n:
    #     hausdorf_distance_per_label[label] = sorted(hausdorf_distance_per_label[label])[:max_n]
    #
    # hausdorff_pair
    # variation_of_information
    pass


def find_most_similar_images():
    load_env()
    data_vault_config = DataVaultConfig.from_env()
    photos_2022 = Photos2022ImageDataSet(data_vault_config)
    directory_2022 = photos_2022.top_directory.replace("2022", "2021")
    photos2021 = ImageDataSet(directory_2022, "photos 2021", data_vault_config)
    output_path = os.path.join(config.get_string("CONVERSION_RESULT_FOLDER"), "similar_images_ssi")

    closest_n = 20
    comparison_sample = 20

    description_by_label = get_description_by_label(photos_2022)
    samples = get_sample_image_descriptions(comparison_sample, description_by_label)

    print("Load photos 2021")
    unlabelled_photos = photos2021.get_image_description_data()
    print("finished loading, start calculating")

    sample_n = 0
    print("Sample for: {}".format("\n".join(f"{key[0]} {key[1]}" for key in samples.keys())))
    for key, image_sample in samples.items():
        print("Calculations for category", key)
        sample_n += 1
        ssi_distances = []
        output_folder = os.path.join(output_path, f"{key[0]}_{key[1]}")
        if os.path.exists(output_folder):
            print(output_folder, "already exists, continue with rest")
            continue

        for i_s, labelled_image in enumerate(image_sample):
            with Image.open(labelled_image.image_path) as im1:
                # for each unlabelled image
                for i_unlab, unlabelled_image in enumerate(unlabelled_photos):
                    with Image.open(unlabelled_image.image_path) as im2:
                        if (i_unlab + 1) % 5 == 0:
                            comparison_str = f"\t[cat {sample_n}/{i_s}/{len(samples)}][unl {i_unlab}/{len(unlabelled_photos)}] input2 image {unlabelled_image.image_path}"
                            print(comparison_str)

                        new_size = tuple(int(c / 5) for c in im2.size)
                        im2_resized = im2.resize(new_size)
                        im1_resized = im1.resize(new_size)
                        # if "IMG-20210604-WA0007" in unlabelled_image.image_path:
                        # # if "IMG-20210604-WA0008" in unlabelled_image.image_path:
                        #     im2_resized.show()
                        #     im1_resized.show()
                        #     x=2
                        ssi = structural_similarity(img_as_ubyte(im1_resized), img_as_ubyte(im2_resized), win_size=5,
                                                    multichannel=True)
                        ssi_distances.append((ssi, unlabelled_image.image_path))
                    # if i_unlab == 50:
                    #     break

        print("write to file")
        # create on the fly and only if doesnt exist already (early selection)
        create_dir_if_doesnt_exist(output_folder)
        ssi_distances.sort()
        i_written = set()
        for d, file_path in ssi_distances:
            name = photos_2022.file_name_from_path(file_path)
            if name in i_written:
                continue
            i_written.add(name)
            f_name = "{:.2}_{}".format(d, name)
            f_path = os.path.join(output_folder, f_name)
            shutil.copyfile(file_path, f_path)
            if len(i_written) == closest_n:
                break


def get_description_by_label(photos_2022):
    description_by_label = defaultdict(list)
    for i_desc, image_description in enumerate(photos_2022.get_image_description_data()):
        all_labels = list(image_description.iterate_all_labels())
        if not all_labels:
            description_by_label["unlabelled"].append(image_description)
        for cat, label in all_labels:
            description_by_label[(cat, label)].append(image_description)
    n_total = 0
    for label, desc_list in description_by_label.items():
        print(label[1], len(desc_list))
        n_total += len(desc_list)
    print(f"n total: {n_total}")
    return description_by_label


if __name__ == '__main__':
    find_most_similar_images()

    # ImageDataSet()
