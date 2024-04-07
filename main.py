import time

import pyxel

import js

WINDOW_W = 400
WINDOW_H = 400

class Hand:
    POINT_SIZE = 7
    POINT_COLOR = 7

    def __init__(self, landmarks, aspect):
        self.points = []
        for landmark in landmarks:
            self.points.append((landmark["x"], landmark["y"], landmark["z"]))
        
        self.aspect = aspect

    def draw(self):
        for point in self.points:
            if self.aspect < 1:
                pyxel.circ(WINDOW_W - point[0] * WINDOW_W, (0.5 - (point[1] - 0.5) / self.aspect) * WINDOW_H, self.POINT_SIZE, self.POINT_COLOR)
            else:
                pyxel.circ((0.5 - (point[0] - 0.5) * self.aspect) * WINDOW_W, point[1] * WINDOW_H, self.POINT_SIZE, self.POINT_COLOR)

class App:
    def __init__(self):
        pyxel.init(WINDOW_W, WINDOW_H)
        self.hands = []
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
        self.hands = [Hand(landmark, self.videoAspect) for landmark in landmarks]

    def draw(self):
        pyxel.cls(0)
        for hand in self.hands:
            hand.draw()

App()