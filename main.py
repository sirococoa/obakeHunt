import pyxel

import js

WINDOW_W = 400
WINDOW_H = 400

class Hand:
    POINT_SIZE = 7
    POINT_COLOR = 7

    def __init__(self, landmarks):
        self.points = []
        for landmark in landmarks:
            self.points.append((landmark["x"], landmark["y"], landmark["z"]))

    def draw(self):
        for point in self.points:
            pyxel.circ(WINDOW_W - point[0] * WINDOW_W, point[1] * WINDOW_H, self.POINT_SIZE, self.POINT_COLOR)

class App:
    def __init__(self):
        pyxel.init(WINDOW_W, WINDOW_H)
        self.hands = []
        pyxel.run(self.update, self.draw)

    def update(self):
        landmarks = js.getLandmarks().to_py()
        self.hands = [Hand(landmark) for landmark in landmarks]

    def draw(self):
        pyxel.cls(0)
        for hand in self.hands:
            hand.draw()

App()