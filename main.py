import pyxel

import js

class Hand:
    POINT_SIZE = 7
    POINT_COLOR = 7

    def __init__(self, landmarks):
        self.points = []
        for landmark in landmarks:
            self.points.append((landmark["x"], landmark["y"], landmark["z"]))

    def draw(self):
        for point in self.points:
            pyxel.circ(400 - point[0] * 400, point[1] * 400, self.POINT_SIZE, self.POINT_COLOR)

class App:
    def __init__(self):
        pyxel.init(400, 400)
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