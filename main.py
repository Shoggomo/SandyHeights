import json
import time
import pyrealsense2 as rs
import numpy as np
import cv2
import config

from BlackPixelFilter import BlackPixelFilter
from AverageImageFilter import AverageImageFilter
from MarkerCropFilter import MarkerCropFilter

marker_crop_filter = MarkerCropFilter()
average_image_filter = AverageImageFilter()
black_pixel_filter = BlackPixelFilter()

if config.ONLY_SHOW_MARKERS:
    marker_crop_filter.show_markers_and_exit()

# Configure depth and color streams
pipeline = rs.pipeline()
device_config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = device_config.resolve(pipeline_wrapper)
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
if config.PRINT_DEFAULT_CONFIG:
    serialized_string = advnc_mode.serialize_json()
    print("Default config: \n", serialized_string)

# load controls config from json file
with open("controls_config.json", "r") as f:
    json_string = str(json.load(f)).replace("'", '\"')
    advnc_mode.load_json(json_string)

device_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
device_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)

# Start streaming
profile = pipeline.start(device_config)

# setup filters
threshold_filter = rs.threshold_filter()
disparity_filter = rs.disparity_transform(config.APPLY_DISPARITY_FILTER)
color_filter = rs.colorizer()
align = rs.align(rs.stream.color)

# configure filters
# threshold_filter.set_option(rs.option.min_distance, config.SANDBOX_TOP_DISTANCE)
# threshold_filter.set_option(rs.option.max_distance, config.SANDBOX_TOP_DISTANCE + config.SANDBOX_HEIGHT)

# color_filter.set_option(rs.option.histogram_equalization_enabled, True)
# color_filter.set_option(rs.option.color_scheme, 9)
# color_filter.set_option(rs.option.visual_preset, 1)
# color_filter.set_option(rs.option.min_distance, config.SANDBOX_TOP_DISTANCE)
# color_filter.set_option(rs.option.max_distance, config.SANDBOX_TOP_DISTANCE + config.SANDBOX_HEIGHT)
# print(color_filter.get_option(rs.option.min_distance))

# sensor = device.query_sensors()[0]
# print(sensor.get_supported_options())
# sensor.set_option(rs.option.min_distance, 0)

# TODO add self calibration here
# TODO add other filters. See viewer
# TODO remove blackpixelfilter, use builtin hole filling filter

# Streaming loop
try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        depth_image = aligned_frames.get_depth_frame()
        color_image = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not depth_image or not color_image:
            continue

        # apply realsense filters
        depth_image = threshold_filter.process(depth_image)
        depth_image = disparity_filter.process(depth_image)
        depth_image = color_filter.colorize(depth_image).get_data()

        # turn images into numpy arrays
        color_image = np.asanyarray(color_image.get_data())
        depth_image = np.asanyarray(depth_image)

        # remove flickering black areas
        # depth_image = black_pixel_filter.remove_from_image(depth_image)

        # average images (do this before crop. resolution of images must be equal)
        average_image_filter.add_image(depth_image)
        depth_image = average_image_filter.get_averaged_image()

        # detect and draw aruco markers or crop images via markers
        if marker_crop_filter.should_detect():
            marker_crop_filter.detect_markers(color_image)

        if config.CROP_IMAGE:
            color_image, depth_image = marker_crop_filter.try_crop_images_to_markers(color_image, depth_image)
        else:
            color_image = marker_crop_filter.draw_detected_markers(color_image)

        # scale images up again
        color_image = cv2.resize(color_image, dsize=config.SCALED_IMAGE_RESOLUTION, interpolation=cv2.INTER_LANCZOS4)
        depth_image = cv2.resize(depth_image, dsize=config.SCALED_IMAGE_RESOLUTION, interpolation=cv2.INTER_LANCZOS4)

        images = np.hstack((color_image, depth_image))

        cv2.namedWindow('Align Example', cv2.WINDOW_NORMAL)
        cv2.imshow('Align Example', images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()
