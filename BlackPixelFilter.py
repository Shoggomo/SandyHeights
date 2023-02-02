from __future__ import annotations

import numpy as np


class BlackPixelFilter:
    last_image: np.ndarray | None

    def __init__(self):
        self.last_image = None

    def remove_from_image(self, image: np.ndarray):
        if self.last_image is None:
            self.last_image = image
            return image

        # create mask for black pixels and replace with old pixels
        black_mask = (image == (0, 0, 0)).all(axis=2)
        image[black_mask] = self.last_image[black_mask]

        self.last_image = image
        return image
