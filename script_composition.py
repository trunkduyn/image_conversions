from compositions import Compositions

if __name__ == '__main__':
    ideal_thumbnail_large = (1280, 720)
    ideal_thumbnail = [int(x/10) for x in ideal_thumbnail_large]

    # size = ideal_thumbnail
    # size = (512,512)
    size = (212,212)
    compositions = Compositions()

    for img in compositions.get_horizontal_circel_images(size=size):
        img.show()
        x=2
    for img in compositions.get_ellipse_images(size=size):
        img.show()
    for img in compositions.get_horizon_images(size, n_heights=5):
        img.show()


