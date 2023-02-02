from PIL import Image


class OriginalRotationConverter:

    def __init__(self):
        self.orientation_funcs = [None,
                             lambda x: x,
                             self.flip_horizontal,
                             self.rotate_180,
                             self.flip_vertical,
                             self.transpose,
                             self.rotate_270,
                             self.transverse,
                             self.rotate_90
                             ]

    def flip_horizontal(self, im):
        return im.transpose(Image.FLIP_LEFT_RIGHT)

    def flip_vertical(self, im):
        return im.transpose(Image.FLIP_TOP_BOTTOM)

    def rotate_180(self, im):
        return im.transpose(Image.ROTATE_180)

    def rotate_90(self, im):
        return im.transpose(Image.ROTATE_90)

    def rotate_270(self, im):
        return im.transpose(Image.ROTATE_270)

    def transpose(self, im):
        return self.rotate_90(self.flip_horizontal(im))

    def transverse(self, im):
        return self.rotate_90(self.flip_vertical(im))



    def apply_orientation(self, im):
        """
        Extract the oritentation EXIF tag from the image, which should be a PIL Image instance,
        and if there is an orientation tag that would rotate the image, apply that rotation to
        the Image instance given to do an in-place rotation.

        :param Image im: Image instance to inspect
        :return: A possibly transposed image instance
        """

        try:
            kOrientationEXIFTag = 0x0112
            if hasattr(im, '_getexif'):  # only present in JPEGs
                e = im._getexif()  # returns None if no EXIF data
                if e is not None:
                    # log.info('EXIF data found: %r', e)
                    orientation = e[kOrientationEXIFTag]
                    f = self.orientation_funcs[orientation]
                    return f(im)
        except:
            # We'd be here with an invalid orientation value or some random error?
            pass  # log.exception("Error applying EXIF Orientation tag")
        return im