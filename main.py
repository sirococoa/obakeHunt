from collections import deque
import time

import pyxel

import js
import numpy
from scipy.spatial import distance

WINDOW_W = 400
WINDOW_H = 400


class Hand:
    POINT_SIZE = 7
    POINT_COLOR = 7

    TARGET_SIZE = 3
    TARGET_COLOR = 8

    def __init__(self, landmarks, aspect, senshi):
        self.points = []
        for landmark in landmarks:
            x, y, z = landmark["x"], landmark["y"], landmark["z"]
            if aspect < 1:
                self.points.append(numpy.asarray((x, 0.5 - (y - 0.5) / aspect, z)))
            else:
                self.points.append(numpy.asarray((0.5 - (x - 0.5) * aspect, y, z)))

        self.target = self.calc_target(senshi)

    def draw(self):
        for point in self.points:
            pyxel.circ(point[0] * WINDOW_W, point[1] * WINDOW_H, self.POINT_SIZE, self.POINT_COLOR)
        pyxel.circ(self.target[0] * WINDOW_W, self.target[1] * WINDOW_H, self.TARGET_SIZE, self.TARGET_COLOR)

    def thumb_length(self):
        return distance.euclidean(self.points[2], self.points[4])

    def index_finger_length(self):
        return distance.euclidean(self.points[5], self.points[8])

    def index_finger_vector(self):
        return self.points[8] - self.points[5]
    
    def index_finger_base(self):
        return self.points[5]
    
    def calc_target(self, senshi):
        target_vector = self.index_finger_base() + self.index_finger_vector() / self.thumb_length() * senshi
        return target_vector


class ShotDetector:
    TARGET_HISTORY_NUM = 60
    SHOT_DETECTION_LENGTH = 0.25
    SHOT_DETECTION_ACCURACY = 0.1
    MARK_DETECTION_FLAME = 5
    MARK_DETECTION_ACCURACY = 0.05

    def __init__(self):
        self.target_history = deque(maxlen=self.TARGET_HISTORY_NUM)
        self.position = None
        self.mark = None
        self.marked_index = 0
        self.shot_flag = False

    def update(self, target):
        self.target_history.append(target)
        if len(self.target_history) < self.TARGET_HISTORY_NUM:
            return

        self.update_mark()
        self.detect_shot()

    def update_mark(self):
        if self.mark is not None:
            self.marked_index -= 1
            if self.marked_index < 0:
                self.mark = None
        current = self.target_history[-1]
        distance_list = []
        for i in range(1, self.MARK_DETECTION_FLAME + 1):
            distance_list.append(distance.euclidean(current, self.target_history[-i]))
        if all(d < self.MARK_DETECTION_ACCURACY for d in distance_list):
            self.mark = current
            self.marked_index = len(self.target_history) - 1
    
    def detect_shot(self):
        self.shot_flag = False
        current = self.target_history[-1]
        if self.mark is not None and distance.euclidean(self.mark, current) < self.SHOT_DETECTION_ACCURACY:
            for i in range(self.marked_index, len(self.target_history)):
                if self.mark[1] - self.target_history[i][1] > self.SHOT_DETECTION_LENGTH:
                    self.position = self.mark
                    self.mark = None
                    self.shot_flag = True
                    break

    def is_shot(self):
        return self.shot_flag
    
    def shot_position(self):
        return self.position


class App:
    def __init__(self):
        pyxel.init(WINDOW_W, WINDOW_H)
        self.hands = []
        self.senshi = 0.5
        self.shot_detector = ShotDetector()
        while True:
            videoWidth = js.videoWidth
            videoHeight = js.videoHeight
            if videoHeight != 0 and videoWidth != 0:
                self.videoAspect = videoWidth / videoHeight
                break
            else:
                time.sleep(0.1)
        pyxel.run(self.update, self.draw)

    def update(self):
        landmarks = js.getLandmarks().to_py()
        self.hands = [Hand(landmark, self.videoAspect, self.senshi) for landmark in landmarks]
        if self.hands:
            self.shot_detector.update(self.hands[0].target)

    def draw(self):
        pyxel.cls(0)
        for hand in self.hands:
            hand.draw()
        fire_position = self.shot_detector.shot_position()
        if fire_position is not None:
            pyxel.circ(fire_position[0] * WINDOW_W, fire_position[1] * WINDOW_H, 3, 11)


App()