## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

###############################################
##      Open CV and Numpy integration        ##
###############################################
import json
import sys
import time

import pyrealsense2 as rs
import numpy as np
import cv2

from MarkerCropper import MarkerCropper

PRINT_DEFAULT_CONFIG = False
ONLY_SHOW_MARKERS = True
CROP_IMAGE = True

marker_cropper = MarkerCropper()

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


# set depth control
print("Depth Control: \n", advnc_mode.get_depth_control())
#To get the minimum and maximum value of each control use the mode value:
# query_min_values_mode = 1
# query_max_values_mode = 2
# current_std_depth_control_group = advnc_mode.get_depth_control()
# min_std_depth_control_group = advnc_mode.get_depth_control(query_min_values_mode)
# print(min_std_depth_control_group)
# max_std_depth_control_group = advnc_mode.get_depth_control(query_max_values_mode)
# print(max_std_depth_control_group)

# current_std_depth_control_group.scoreThreshA = 150
# current_std_depth_control_group.scoreThreshB = 150
# advnc_mode.set_depth_control(current_std_depth_control_group)

# Serialize all controls to a Json string
# serialized_string = advnc_mode.serialize_json()
# print("Controls as JSON: \n", serialized_string)
# as_json_object = json.loads(serialized_string)

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

# Getting the depth sensor's depth scale (see rs-align example for explanation)
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale is: " , depth_scale)

# We will be removing the background of objects more than
#  clipping_distance_in_meters meters away
clipping_distance_in_meters = 1 #1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

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
        # frames.get_depth_frame() is a 640x360 depth image

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        # depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Render images:
        #   depth align to color on left
        #   depth on right
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        colorizer = rs.colorizer()
        depth_colormap = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

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