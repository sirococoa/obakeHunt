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


class App:
    def __init__(self):
        pyxel.init(WINDOW_W, WINDOW_H)
        self.hands = []
        self.senshi = 0.5
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

    def draw(self):
        pyxel.cls(0)
        for hand in self.hands:
            hand.draw()

App()