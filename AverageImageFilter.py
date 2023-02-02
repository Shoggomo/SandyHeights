from typing import List

import numpy as np

import config


class AverageImageFilter:
    images: List[np.ndarray]

    def __init__(self):
        self.images = []

    def add_image(self, image):
        if len(self.images) >= config.AVERAGE_WINDOW_SIZE:
            self.images.pop(0)

        self.images.append(image)

    def get_averaged_image(self):
        assert self._are_images_same_size_(), "Images are of different size, maybe scale them before"

        shape = self.images[0].shape
        arr = np.zeros(shape, float)

        # Build up average pixel intensities, casting each image as an array of floats
        for im in self.images:
            imarr = np.array(im, dtype=float)
            arr = arr + imarr / len(self.images)

        # Round values in array and cast as 8-bit integer
        average_image = np.array(np.round(arr), dtype=np.uint8)

        return average_image

    def _are_images_same_size_(self):
        assert len(self.images) > 0, "Method called to early, no image added yet"

        first_image = self.images[0]
        for im in self.images[1:]:
            if first_image.shape != im.shape:
                return False

        return True
