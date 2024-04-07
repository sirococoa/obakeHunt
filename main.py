import pyxel

import js

class App:
    def __init__(self):
        pyxel.init(400, 400)
        self.landmarks = []
        pyxel.run(self.update, self.draw)

    def update(self):
        self.landmarks = js.getLandmarks().to_py()

    def draw(self):
        pyxel.cls(0)
        for landmark in self.landmarks:
            pyxel.circ(400 - float(landmark["x"]) * 400, float(landmark["y"]) * 400, 10, 7)


App()