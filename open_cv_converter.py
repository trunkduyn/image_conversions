import cv2
import numpy
from PIL.Image import Image


class OpenCVConverter:

    def pillow_image_to_opencv(self, image: Image):
        open_cv_image = numpy.array(image)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        return open_cv_image

    def convert_if_is_pillow_image(self, image):
        type(Image)
        if isinstance(image, Image):
            return self.pillow_image_to_opencv(image)
        else:
            return image

    def sketch_image(self, image: Image, k_size: int):
        open_cv_image = self.convert_if_is_pillow_image(image)
        # Convert to Grey Image
        grey_img = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)
        # Invert Image
        invert_img = cv2.bitwise_not(grey_img)
        # Blur image
        blur_img = cv2.GaussianBlur(invert_img, (k_size, k_size), 0)
        # Invert Blurred Image
        invblur_img = cv2.bitwise_not(blur_img)
        # Create Sketch Image by combining
        sketch_img = cv2.divide(grey_img, invblur_img, scale=256.0)

        return sketch_img

    def display_image(self, image, title='image'):
        open_cv_image = self.convert_if_is_pillow_image(image)
        # Display sketch
        cv2.imshow(title, open_cv_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def blend_images(self, image1: Image, image2: Image, alpha=0.5, beta=0.5, gamma=0):
        image1 = self.convert_if_is_pillow_image(image1)
        image2 = self.convert_if_is_pillow_image(image2)
        image2_resized = self.resize_set_value(image2, image1.shape[1], image1.shape[0])
        blended_image = cv2.addWeighted(src1=image1, alpha=alpha, src2=image2_resized, beta=beta, gamma=gamma)

        cv2.imshow('Blended Image', blended_image)

        cv2.waitKey(0)

    def resize_by_factor(self, img, scale_factor: float):
        width = int(img.shape[1] * scale_factor)
        height = int(img.shape[0] * scale_factor)
        dim = (width, height)

        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        return resized

    def resize_set_value(self, img, width=None, height=None):
        width = img.shape[1] if width is None else width
        height = img.shape[0] if height is None else height
        # resize image
        resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
        return resized

    def pixelate(self, img, pixelation_scale=0.05):
        image = self.convert_if_is_pillow_image(img)

        # Get input size
        height, width = image.shape[:2]
        # Desired "pixelated" size
        w, h = (int(width*pixelation_scale), int(height*pixelation_scale))
        # Resize input to "pixelated" size
        temp = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        # Initialize output image
        output = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
        return output

    def save_opencv_image(self, image, path):
        # Save Sketch
        cv2.imwrite(path, image)
