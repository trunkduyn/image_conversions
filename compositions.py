from PIL import Image, ImageDraw, ImageOps, ImageColor


class Compositions:
    GOLDEN_RATIO = 1.618

    def __init__(self):
        self.ink_color = ImageColor.getrgb('white')

    @staticmethod
    def get_new_color_image(size, black_background=True):
        color = (0, 0, 0)
        if not black_background:
            color = (255, 255, 255)
        # create an image
        out = Image.new("RGB", size, color)
        return out

    @staticmethod
    def get_new_grey_image(size, black_background=True):
        color = (0, 0, 0)
        if not black_background:
            color = (255, 255, 255)
        # create an image
        out = Image.new("RGB", size, color)
        grey_image = ImageOps.grayscale(out)
        return grey_image

    def get_ellipse_images(self, size):
        center = self.get_center(size)

        ellipse_definitions = []
        ellipse_images = []

        center_ellipses = self.get_center_circel_definitions(center, size, n_circels=5)
        ellipse_definitions.extend(center_ellipses)

        for ellipse in ellipse_definitions:
            image = self.get_new_grey_image(size)
            # get a drawing context
            d = ImageDraw.Draw(image)
            d.ink = ImageColor.getcolor('white', d.mode)
            d.ellipse(ellipse)
            ellipse_images.append(image)
        return ellipse_images

    @staticmethod
    def get_center(size):
        width, height = size
        center = (int(width / 2), int(height / 2))
        return center

    @staticmethod
    def get_center_circel_definitions(center, size, n_circels=5):
        width, height = size
        max_d = min(width, height)
        step_size = max_d / n_circels
        center_ellipses = []
        for i in range(n_circels):
            s = max_d
            s -= step_size * i
            x = int(s / 2)
            center_ellipses.append([center[0] - x, center[1] - x, center[0] + x, center[1] + x])
        return center_ellipses

    def get_horizontal_circel_images(self, size, n_circels=2):
        width, height = size
        base_horizon = self.get_base_horizon(height)
        center = self.get_center(size)
        ellipse_definitions = []
        circle_size_start = int(int(max(height, width) / 10) / 2)

        step_size = center[1] / self.GOLDEN_RATIO / n_circels
        width_positions = []
        for i in range(n_circels):
            width_positions.append(center[1] - ((i + 1) * step_size))
            width_positions.append(center[1] + ((i + 1) * step_size))

        circle_size = circle_size_start
        for i in range(3):
            for width_pos in width_positions:
                for factor in [1, 0.6, 0.4]:
                    ellipse_definitions.append([(base_horizon * factor) - circle_size, width_pos - circle_size,
                                                base_horizon * factor + circle_size, width_pos + circle_size])
            circle_size *= 1.5

        ellipse_images = []
        for ellipse in ellipse_definitions:
            image = self.get_new_grey_image(size)
            # get a drawing context
            d = ImageDraw.Draw(image)
            d.ink = ImageColor.getcolor('white', d.mode)
            d.ellipse(ellipse)
            ellipse_images.append(image)
        return ellipse_images

    def single_line_image(self, size, xy):
        image = self.get_new_grey_image(size)
        d = ImageDraw.Draw(image)
        d.ink = ImageColor.getcolor('white', d.mode)
        d.line(xy, width=5)
        return image

    def get_horizon_images(self, size, n_heights=3):
        width, height = size
        # base_horizon * self.GOLDEN_RATIO = height
        base_horizon = self.get_base_horizon(height)
        step_size = (height - base_horizon) / (n_heights - 1) / 2
        low_horizon = base_horizon - (2 * step_size)

        horizon_images = []
        for i in range(n_heights):
            horizon_height = int(low_horizon + (i * step_size))
            xy = [0, horizon_height, width, horizon_height]
            # add straight horizon at height horizon_height
            img = self.single_line_image(size, xy)
            horizon_images.append(img)
            # add tilted horizons
            for f1, f2 in [(0.9, 1), (1, 0.9), (0.8, 1), (1, 0.8)]:
                h1 = int(horizon_height * f1)
                h2 = int(horizon_height * f2)
                xy = [0, h1, width, h2]
                horizon_images.append(self.single_line_image(size, xy))
        return horizon_images

    def get_base_horizon(self, height):
        base_horizon = height - (height / self.GOLDEN_RATIO)
        return base_horizon

    def iter_all_images(self, size):
        for i in self.get_horizontal_circel_images(size):
            yield i
        for i in self.get_ellipse_images(size):
            yield i
        for i in self.get_horizon_images(size):
            yield i
