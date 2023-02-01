from __future__ import annotations

import numpy as np


class BlackPixelRemover:
    last_image: np.ndarray | None

    def __init__(self):
        self.last_image = None

    def remove_from_image(self, image: np.ndarray):
        if self.last_image is None:
            self.last_image = image
            return image

        # TODO this is too innevicient. We should use the new image as a mask for the last image and add the result with the image
        # for x, line in enumerate(image):
        #     for y, pixel in enumerate(line):
        #         if (pixel == 0).all():
        #             image[x][y] = self.last_image[x][y]

        self.last_image = image
        return image
