import sys
import time

import cv2

UPDATE_THRESHOLD = 2 * 1e+9  # = 2 seconds


class MarkerCropper:

    def __init__(self):
        self.ids = []
        self.corners = []
        self.last_time_detected = 0
        self.detector_params = cv2.aruco.DetectorParameters()
        self.marker_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.detector = cv2.aruco.ArucoDetector(self.marker_dict, self.detector_params)

    def detect_markers(self, color_image):
        corners, ids, rejected_imgP_points = self.detector.detectMarkers(color_image)
        self.corners = corners
        self.ids = ids
        self.last_time_detected = time.time_ns()

    def should_detect(self):
        return (time.time_ns() - self.last_time_detected) > UPDATE_THRESHOLD

    def draw_detected_markers(self, color_image):
        return cv2.aruco.drawDetectedMarkers(color_image, self.corners, self.ids)

    def try_crop_images_to_markers(self, color_image, depth_image):
        if len(self.corners) >= 2:
            x1, y1 = self.corners[0][0][0].astype(int)
            x2, y2 = self.corners[1][0][0].astype(int)

            from_x = min(x1, x2)
            to_x = max(x1, x2)
            from_y = min(y1, y2)
            to_y = max(y1, y2)

            color_image = color_image[from_y:to_y, from_x:to_x]
            depth_image = depth_image[from_y:to_y, from_x:to_x]
        return color_image, depth_image

    def show_markers_and_exit(self, amount_markers=4):
        for marker_id in range(1, amount_markers+1):
            marker_image = cv2.aruco.generateImageMarker(self.marker_dict, marker_id, 200)
            cv2.imshow(f'{marker_id} Marker', marker_image)
        cv2.waitKey(0)
        sys.exit(0)
