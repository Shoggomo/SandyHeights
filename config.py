# DEBUGGING
PRINT_DEFAULT_CONFIG = False  # print the devices default config
ONLY_SHOW_MARKERS = False  # only show markers (e.g. for printing them out)
CROP_IMAGE = False  # crop the image or show markers

CROP_MARKER_OFFSET = 10  # offset from marker for cropping in pixels (TODO unused yet)
CROP_MARKER_IDS = [1, 2]  # markers ids to use for cropping
AVERAGE_WINDOW_SIZE = 3  # number of images to take the average from
SANDBOX_TOP_DISTANCE = 0.25  # distance from top of sandbox in m (-0.1)
SANDBOX_HEIGHT = 0.20  # height of sandbox in m (+0.1)
SCALED_IMAGE_RESOLUTION = (640, 480)  # resolution to scale the final image to
APPLY_DISPARITY_FILTER = True
