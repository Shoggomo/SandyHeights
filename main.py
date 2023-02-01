import json
import time
import pyrealsense2 as rs
import numpy as np
import cv2

from BlackPixelRemover import BlackPixelRemover
from ImageAverager import ImageAverager
from MarkerCropper import MarkerCropper

PRINT_DEFAULT_CONFIG = False
ONLY_SHOW_MARKERS = False
CROP_IMAGE = False

marker_cropper = MarkerCropper()
image_averager = ImageAverager()
black_pixel_remover = BlackPixelRemover()

if ONLY_SHOW_MARKERS:
    marker_cropper.show_markers_and_exit()

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
advnc_mode = rs.rs400_advanced_mode(device)

# ensure advanced mode is active
while not advnc_mode.is_enabled():
    print("Trying to enable advanced mode...")
    advnc_mode.toggle_advanced_mode(True)
    # At this point the device will disconnect and re-connect.
    print("Sleeping for 5 seconds...")
    time.sleep(5)
    # The 'dev' object will become invalid and we need to initialize it again
    dev = pipeline_profile.get_device()
    advnc_mode = rs.rs400_advanced_mode(dev)
    print("Advanced mode is", "enabled" if advnc_mode.is_enabled() else "disabled")

# print default config
if PRINT_DEFAULT_CONFIG:
    serialized_string = advnc_mode.serialize_json()
    print("Default config: \n", serialized_string)

# load controls config from json file
with open("controls_config.json", "r") as f:
    json_string = str(json.load(f)).replace("'", '\"')
    advnc_mode.load_json(json_string)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

# Start streaming
profile = pipeline.start(config)

# Create an align object
# rs.align allows us to perform alignment of depth frames to others frames
# The "align_to" is the stream type to which we plan to align depth frames.
align_to = rs.stream.color
align = rs.align(align_to)

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        # depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # colorize depth image
        colorizer = rs.colorizer(0)  # 0 is Jet color scheme
        depth_colormap = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

        # try to remove black pixels
        depth_colormap = black_pixel_remover.remove_from_image(depth_colormap)

        # average images (do this before crop. resolution of images must be equal)
        image_averager.add_image(depth_colormap)
        depth_colormap = image_averager.get_averaged_image()

        # detect and draw aruco markers or crop images
        if marker_cropper.should_detect():
            marker_cropper.detect_markers(color_image)
        if CROP_IMAGE:
            color_image, depth_colormap = marker_cropper.try_crop_images_to_markers(color_image, depth_colormap)
        else:
            color_image = marker_cropper.draw_detected_markers(color_image)

        images = np.hstack((color_image, depth_colormap))

        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
