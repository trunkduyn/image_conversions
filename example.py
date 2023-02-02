import os

from PIL import Image

from open_cv_converter import OpenCVConverter
from original_rotation_converter import OriginalRotationConverter

cur_dir = os.path.dirname(os.path.abspath(__file__))
example_data_folder = os.path.join(cur_dir, "example_data")
print(list(os.listdir(example_data_folder)))
converter = OpenCVConverter()

# prev_sketch = None
# for f_name in os.listdir(example_data_folder):
#     with Image.open(os.path.join(example_data_folder, f_name)) as im:
#
#         # create pixelated
#         pixelated_image = converter.pixelate(im, pixelation_scale=0.06)
#         converter.display_image(pixelated_image)
#
#         # create sketch
#         sketch_image = converter.sketch_image(im, k_size=7)
#         converter.display_image(sketch_image)
#
#         # create pixelated sketch
#         pixelated_sketch = converter.pixelate(sketch_image, pixelation_scale=0.01)
#         converter.display_image(pixelated_sketch)
#
#         # blend between this sketch and previous
#         if prev_sketch is not None:
#             blend_image = converter.blend_images(prev_sketch, sketch_image)
#             converter.display_image(blend_image)
#
#         prev_sketch = sketch_image

# full_folder = "/Users/trunkiekaekel/OneDrive/Pictures/2022"
# target_folder = "/Users/trunkiekaekel/artificial_intelligence/sketches_personal_high2"
# for i, f_name in enumerate(os.listdir(full_folder)):
#     if i%200==0 or i < 10:
#         print(i)
#     try:
#         with Image.open(os.path.join(full_folder, f_name)) as im:
#             target_path = os.path.join(target_folder, f_name)
#             # if os.path.exists(target_path):
#             #     continue
#             # create sketch
#             sketch_image = converter.sketch_image(image=im, k_size=25)
#             converter.display_image(sketch_image)
#             converter.save_opencv_image(sketch_image, target_path)
#     except Exception as e:
#         print(e)
#         continue

full_folder = "/Users/trunkiekaekel/OneDrive/Pictures/2022"
target_folder = "/Users/trunkiekaekel/artificial_intelligence/rotated_2022"
rotation_converter = OriginalRotationConverter()
for i, f_name in enumerate(os.listdir(full_folder)):
    if i%200==0 or i < 10:
        print(i)
    try:
        with Image.open(os.path.join(full_folder, f_name)) as im:
            target_path = os.path.join(target_folder, f_name)
            # im.show()
            rotated_im = rotation_converter.apply_orientation(im)
            # rotated_im.show()
            sketch_image = converter.sketch_image(image=rotated_im, k_size=25)
            converter.display_image(sketch_image)
    except Exception as e:
        print(e)
        continue