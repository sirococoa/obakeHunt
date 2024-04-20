from collections import deque
import random
import time
from typing import Any

import pyxel

import js
import numpy
from scipy.spatial import distance

WINDOW_W = 256
WINDOW_H = 256


class Hand:
    POINT_SIZE = 7
    POINT_COLOR = 7

    TARGET_SIZE = 3
    TARGET_COLOR = 8

    def __init__(self, landmarks: Any, aspect: float, senshi: float) -> None:
        self.points = []
        for landmark in landmarks:
            x, y, z = landmark["x"], landmark["y"], landmark["z"]
            if aspect < 1:
                self.points.append(numpy.asarray((x, 0.5 - (y - 0.5) / aspect, z)))
            else:
                self.points.append(numpy.asarray((0.5 - (x - 0.5) * aspect, y, z)))

        self.target = self.calc_target(senshi)

    def draw(self) -> None:
        for point in self.points:
            pyxel.circ(point[0] * WINDOW_W, point[1] * WINDOW_H, self.POINT_SIZE, self.POINT_COLOR)
        pyxel.circ(self.target[0] * WINDOW_W, self.target[1] * WINDOW_H, self.TARGET_SIZE, self.TARGET_COLOR)

    def thumb_length(self) -> float:
        return distance.euclidean(self.points[2], self.points[4])
    
    def thumb_tip_point(self) -> numpy.ndarray:
        return self.points[4]

    def index_finger_length(self) -> float:
        return distance.euclidean(self.points[5], self.points[8])

    def index_finger_vector(self) -> numpy.ndarray:
        return self.points[8] - self.points[5]
    
    def index_finger_base(self) -> numpy.ndarray:
        return self.points[5]
    
    def ring_finger_pip_point(self) -> numpy.ndarray:
        return self.points[14]
    
    def calc_target(self, senshi) -> numpy.ndarray:
        target_vector = self.index_finger_base() + self.index_finger_vector() / self.thumb_length() * senshi
        return target_vector[:2]


class ShotDetector:
    TARGET_HISTORY_NUM = 60
    SHOT_DETECTION_LENGTH = 0.25
    SHOT_DETECTION_ACCURACY = 0.1
    MARK_DETECTION_FLAME = 5
    MARK_DETECTION_ACCURACY = 0.05

    def __init__(self) -> None:
        self.target_history = deque(maxlen=self.TARGET_HISTORY_NUM)
        self.position: numpy.ndarray | None = None
        self.mark: numpy.ndarray | None = None
        self.marked_index = 0
        self.shot_flag = False

    def update(self, target: numpy.ndarray) -> None:
        self.target_history.append(target)
        if len(self.target_history) < self.TARGET_HISTORY_NUM:
            return

        self.update_mark()
        self.detect_shot()

    def update_mark(self) -> None:
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
    
    def detect_shot(self) -> None:
        self.shot_flag = False
        current = self.target_history[-1]
        if self.mark is not None and distance.euclidean(self.mark, current) < self.SHOT_DETECTION_ACCURACY:
            for i in range(self.marked_index, len(self.target_history)):
                if self.mark[1] - self.target_history[i][1] > self.SHOT_DETECTION_LENGTH:
                    self.position = self.mark
                    self.mark = None
                    self.shot_flag = True
                    break

    def is_shot(self) -> bool:
        return self.shot_flag
    
    def shot_position(self) -> numpy.ndarray:
        return self.position


class ReloadDetector:
    RELOAD_DETECTION_DISTANCE = 0.1

    def __init__(self) -> None:
        self.reload_flag = False

    def update(self, hand: Hand) -> None:
        if distance.euclidean(hand.thumb_tip_point(), hand.ring_finger_pip_point()) < hand.thumb_length():
            self.reload_flag = True
        else:
            self.reload_flag = False
    
    def is_reload(self) -> None:
        return self.reload_flag


class ObakeImage:
    ASSET_FILE = "./assets/obake.png"
    I = 1
    U = 0
    V = 0
    W = 32
    H = 32
    COLKEY = 15

    def __init__(self) -> None:
        self.load()

    def load(self) -> None:
        pyxel.images[self.I].load(self.U, self.V, self.ASSET_FILE)
    
    def draw(self, x: int, y: int) -> None:
        pyxel.blt(x, y, self.I, self.U, self.V, self.W, self.H, self.COLKEY)


class Obake:
    W = 32
    H = 32
    COLLISION_MARGIN = 0

    def __init__(self, x: int, y: int, shot_detector: ShotDetector, obake_image: ObakeImage) -> None:
        self.x = x
        self.y = y
        self.shot_detector = shot_detector
        self.image = obake_image
        self.active = True

    def update(self) -> None:
         if self.shot_detector.is_shot():
            shot_position = self.shot_detector.shot_position()
            if self.collision(shot_position[0] * WINDOW_W, shot_position[1] * WINDOW_H):
                self.active = False

    def collision(self, sx: int, sy: int) -> bool:
        return (-self.COLLISION_MARGIN <= sx - self.x < self.W + self.COLLISION_MARGIN)\
            and (-self.COLLISION_MARGIN <= sy - self.y < self.H + self.COLLISION_MARGIN)
    
    def is_active(self):
        return self.active

    def draw(self) -> None:
        self.image.draw(self.x, self.y)


class BackGroundImage:
    ASSET_FILE = "./assets/background.png"
    X = 0
    Y = 0
    I = 0
    U = 0
    V = 0
    W = 256
    H = 256

    def __init__(self) -> None:
        self.load()

    def load(self) -> None:
        pyxel.images[self.I].load(self.U, self.V, self.ASSET_FILE)
    
    def draw(self) -> None:
        pyxel.blt(self.X, self.Y, self.I, self.U, self.V, self.W, self.H)


class BulletImage:
    ASSET_FILE = "./assets/bullet.png"
    X = 0
    Y = 0
    I = 1
    U = 32
    V = 32
    W = 10
    H = 25
    COLKEY = 15

    def __init__(self) -> None:
        self.load()

    def load(self) -> None:
        pyxel.images[self.I].load(self.U, self.V, self.ASSET_FILE)
    
    def draw(self, x: int, y: int) -> None:
        pyxel.blt(x, y, self.I, self.U, self.V, self.W, self.H, self.COLKEY)

class BulletEmptyImage:
    ASSET_FILE = "./assets/bullet_empty.png"
    X = 0
    Y = 0
    I = 1
    U = 42
    V = 32
    W = 10
    H = 25
    COLKEY = 15

    def __init__(self) -> None:
        self.load()

    def load(self) -> None:
        pyxel.images[self.I].load(self.U, self.V, self.ASSET_FILE)
    
    def draw(self, x: int, y: int) -> None:
        pyxel.blt(x, y, self.I, self.U, self.V, self.W, self.H, self.COLKEY)

class BulletUI:
    X = 100
    Y = 190
    BULLET_W = 10 # BULLET_W * 3 + MARGIN_X * 2 = 56
    BULLET_H = 25 # BULLET_W * 2 + MARGIN_X * 1 = 60
    MARGIN_X = 13
    MARGIN_Y = 10

    def __init__(self) -> None:
        self.bullet_image = BulletImage()
        self.bullet_empty_image = BulletEmptyImage()

    def draw(self, num: int) -> None:
        for i in range(2):
            for j in range(3):
                x = self.X + j * (self.BULLET_W + self.MARGIN_X)
                y = self.Y + i * (self.BULLET_H + self.MARGIN_Y)
                if (1 - i) * 3 + j < num:
                    self.bullet_image.draw(x, y)
                else:
                    self.bullet_empty_image.draw(x, y)


class App:
    BULLET_NUM = 6
    def __init__(self) -> None:
        pyxel.init(WINDOW_W, WINDOW_H)
        self.hands = []
        self.senshi = 0.5
        self.bullet_num = self.BULLET_NUM
        self.shot_detector = ShotDetector()
        self.reload_detector = ReloadDetector()
        self.obake_list = []
        self.back_ground = BackGroundImage()
        self.obake_image = ObakeImage()
        self.bullet_ui = BulletUI()
        while True:
            videoWidth = js.videoWidth
            videoHeight = js.videoHeight
            if videoHeight != 0 and videoWidth != 0:
                self.videoAspect = videoWidth / videoHeight
                break
            else:
                time.sleep(0.1)
        pyxel.run(self.update, self.draw)

    def update(self) -> None:
        landmarks = js.getLandmarks().to_py()
        self.hands = [Hand(landmark, self.videoAspect, self.senshi) for landmark in landmarks]
        if self.hands:
            self.shot_detector.update(self.hands[0].target)
            self.reload_detector.update(self.hands[0])
        if self.shot_detector.is_shot():
            self.bullet_num -= 1
        if self.reload_detector.is_reload():
            self.bullet_num = self.BULLET_NUM

        for obake in self.obake_list:
            obake.update()
        self.obake_list = [obake for obake in self.obake_list if obake.is_active()]
        if len(self.obake_list) < 5:
            self.obake_list.append(Obake(random.randrange(0, WINDOW_W), random.randrange(0, WINDOW_H), self.shot_detector, self.obake_image))


    def draw(self) -> None:
        pyxel.cls(0)
        self.back_ground.draw()
        for hand in self.hands:
            hand.draw()
        fire_position = self.shot_detector.shot_position()
        if fire_position is not None:
            pyxel.circ(fire_position[0] * WINDOW_W, fire_position[1] * WINDOW_H, 3, 11)
        pyxel.text(10, 10, str(self.bullet_num), 7)
        for obake in self.obake_list:
            obake.draw()
        self.bullet_ui.draw(self.bullet_num)


App()
