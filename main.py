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

    def __init__(self, landmarks: Any, aspect: float, sens: float) -> None:
        self.points = []
        for landmark in landmarks:
            x, y, z = landmark['x'], landmark['y'], landmark['z']
            if aspect < 1:
                self.points.append(numpy.asarray((x, 0.5 - (y - 0.5) / aspect, z)))
            else:
                self.points.append(numpy.asarray((0.5 - (x - 0.5) * aspect, y, z)))

        self.target = self.calc_target(sens)

    def draw(self) -> None:
        for point in self.points:
            pyxel.circ(point[0] * WINDOW_W, point[1] * WINDOW_H, self.POINT_SIZE, self.POINT_COLOR)
        pyxel.circ(self.target[0] * WINDOW_W, self.target[1] * WINDOW_H, self.TARGET_SIZE, self.TARGET_COLOR)

    def thumb_length(self) -> float:
        return distance.euclidean(self.points[2], self.points[3]) + distance.euclidean(self.points[3], self.points[4])

    def thumb_tip_point(self) -> numpy.ndarray:
        return self.points[4]

    def index_finger_length(self) -> float:
        return distance.euclidean(self.points[5], self.points[8])

    def index_finger_vector(self) -> numpy.ndarray:
        return self.points[8] - self.points[5]

    def index_finger_base(self) -> numpy.ndarray:
        return self.points[5]
    
    def index_finger_tip_point(self) -> numpy.ndarray:
        return self.points[8]
    
    def middle_finger_tip_point(self) -> numpy.ndarray:
        return self.points[12]

    def ring_finger_pip_point(self) -> numpy.ndarray:
        return self.points[14]

    def calc_target(self, sens) -> numpy.ndarray:
        target_vector = self.index_finger_base() + self.index_finger_vector() / self.thumb_length() * sens
        return target_vector[:2]


class ShootDetector:
    TARGET_HISTORY_NUM = 60
    SHOOT_DETECTION_LENGTH = 0.25
    SHOOT_DETECTION_ACCURACY = 0.1
    MARK_DETECTION_FLAME = 5
    MARK_DETECTION_ACCURACY = 0.05

    def __init__(self) -> None:
        self.target_history = deque(maxlen=self.TARGET_HISTORY_NUM)
        self.position: numpy.ndarray | None = None
        self.mark: numpy.ndarray | None = None
        self.marked_index = 0
        self.shoot_flag = False

    def update(self, target: numpy.ndarray) -> None:
        self.target_history.append(target)
        if len(self.target_history) < self.TARGET_HISTORY_NUM:
            return

        self.update_mark()
        self.detect_shoot()

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

    def detect_shoot(self) -> None:
        self.shoot_flag = False
        current = self.target_history[-1]
        if self.mark is not None and distance.euclidean(self.mark, current) < self.SHOOT_DETECTION_ACCURACY:
            for i in range(self.marked_index, len(self.target_history)):
                if self.mark[1] - self.target_history[i][1] > self.SHOOT_DETECTION_LENGTH:
                    self.position = self.mark
                    self.mark = None
                    self.shoot_flag = True
                    break

    def is_shoot(self) -> bool:
        return self.shoot_flag

    def shoot_position(self) -> numpy.ndarray:
        return self.position


class ReloadDetector:
    def __init__(self) -> None:
        self.reload_flag = False

    def update(self, hand: Hand) -> None:
        if distance.euclidean(hand.thumb_tip_point(), hand.ring_finger_pip_point()) < hand.thumb_length() and \
            distance.euclidean(hand.index_finger_tip_point(), hand.middle_finger_tip_point()) < hand.thumb_length():
            self.reload_flag = True
        else:
            self.reload_flag = False

    def is_reload(self) -> None:
        return self.reload_flag


class PointDetector:
    DETECTION_TIME = 30
    DETECTION_DETECTION_ACCURACY = 0.05

    RADIUS = 20
    DETECTION_DRAW_START_TIME = 10

    def __init__(self) -> None:
        self.point_history = deque(maxlen=self.DETECTION_TIME)

    def update(self, hand: Hand) -> None:
        point = hand.index_finger_tip_point()
        self.point_history.append([point[0], point[1]])

    def selected_point(self) -> list[int]:
        if len(self.point_history) != self.DETECTION_TIME:
            return []
        current = self.point_history[-1]
        for point in self.point_history:
            if distance.euclidean(current, point) > self.DETECTION_DETECTION_ACCURACY:
                return []
        return [int(current[0] * WINDOW_W), int(current[1] * WINDOW_H)]

    def draw(self) -> None:
        if len(self.point_history) != self.DETECTION_TIME:
            return
        current = self.point_history[-1]
        for i in range(len(self.point_history)):
            point = self.point_history[-(i + 1)]
            if distance.euclidean(current, point) > self.DETECTION_DETECTION_ACCURACY:
                break
        if i < self.DETECTION_DRAW_START_TIME:
            return
        pyxel.circb(int(current[0] * WINDOW_W), int(current[1] * WINDOW_H), self.RADIUS, 8)
        r = int((i - self.DETECTION_DRAW_START_TIME) / (self.DETECTION_TIME - self.DETECTION_DRAW_START_TIME) * self.RADIUS)
        pyxel.circ(int(current[0] * WINDOW_W), int(current[1] * WINDOW_H), r, 8)


class ObakeDeadImage:
    ASSET_FILE = './assets/obake_dead.png'
    I = 1
    U = 32
    V = 0
    W = 32
    H = 32
    COLKEY = 15

    def __init__(self) -> None:
        self.load()

    def load(self) -> None:
        pyxel.images[self.I].load(self.U, self.V, self.ASSET_FILE)

    def draw(self, x: int, y: int, flip: bool) -> None:
        if flip:
            pyxel.blt(x, y, self.I, self.U, self.V, -self.W, self.H, self.COLKEY)
        else:
            pyxel.blt(x, y, self.I, self.U, self.V, self.W, self.H, self.COLKEY)


class ObakeDeadParticle:
    obake_dead_particle_list = []
    obake_dead_image = None
    ACTIVE_TIME = 30
    OFFSET = 0.1

    def __init__(self, x: int, y: int, flip: bool) -> None:
        self.x = x
        self.y = y
        self.flip = flip
        self.count = 0
        self.active = True

    def _update(self) -> None:
        self.count += 1
        if self.count >= self.ACTIVE_TIME:
            self.active = False

    def _draw(self) -> None:
        dither = max(0, min(1, (self.ACTIVE_TIME - self.count) / self.ACTIVE_TIME + self.OFFSET))
        pyxel.dither(dither)
        self.obake_dead_image.draw(self.x, self.y, self.flip)
        pyxel.dither(1)

    @classmethod
    def add_particle(cls, x: int, y: int, flip: bool):
        if cls.obake_dead_image is not None:
            cls.obake_dead_particle_list.append(ObakeDeadParticle(x, y, flip))

    @classmethod
    def load(cls):
        if cls.obake_dead_image is None:
            cls.obake_dead_image = ObakeDeadImage()

    @classmethod
    def reset(cls):
        cls.obake_dead_particle_list = []

    @classmethod
    def update(cls):
        for particle in cls.obake_dead_particle_list:
            particle._update()
        cls.obake_dead_particle_list = [particle for particle in cls.obake_dead_particle_list if particle.active]

    @classmethod
    def draw(cls):
        for particle in cls.obake_dead_particle_list:
            particle._draw()


class ObakeImage:
    ASSET_FILE = './assets/obake.png'
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

    def draw(self, x: int, y: int, flip: bool) -> None:
        if flip:
            pyxel.blt(x, y, self.I, self.U, self.V, -self.W, self.H, self.COLKEY)
        else:
            pyxel.blt(x, y, self.I, self.U, self.V, self.W, self.H, self.COLKEY)


class Obake:
    obake_image = None

    W = 32
    H = 32
    COLLISION_MARGIN = 0
    UP_SPEED = 0.2
    LATERAL_SPEED = 0.3
    ZIGZAG_DURATION = 60
    MIN_FLIP_COUNT = 120
    MAX_FLIP_COUNT = 300
    APPEAR_TIME = 20

    def __init__(self, x: int, y: int, delay: int) -> None:
        self.x = x
        self.y = y
        if self.obake_image is None:
            self.obake_image = ObakeImage()
        self.delay = delay
        if random.random() < 0.5:
            self.direction = [self.LATERAL_SPEED, -self.UP_SPEED]
        else:
            self.direction = [-self.LATERAL_SPEED, -self.UP_SPEED]
        self.active = True
        self.count = 0
        self.next_flip_count = 0

    def update(self) -> None:
        self.count += 1
        if not self.is_active() or self.is_waiting() or self.is_appearing():
            return
        if self.count % (self.ZIGZAG_DURATION*2) < self.ZIGZAG_DURATION:
            self.x += self.direction[0] + 0.5*(self.direction[0] + self.direction[1])
            self.y += self.direction[1] + 0.5*(-self.direction[0] + self.direction[1])
        else:
            self.x += self.direction[0] + 0.5*(self.direction[0] - self.direction[1])
            self.y += self.direction[1] + 0.5*(self.direction[0] + self.direction[1])
        if self.next_flip_count < self.count or self.is_edge():
            self.direction[0] = -self.direction[0]
            self.next_flip_count = self.count + random.randint(self.MIN_FLIP_COUNT, self.MAX_FLIP_COUNT)
        if self.is_outside():
            self.active = False

    def is_outside(self) -> bool:
        if -self.W < self.x < WINDOW_W:
            if -self.H < self.y < WINDOW_H:
                return False
        return True
    
    def is_edge(self) -> bool:
        if self.direction[0] < 0:
            # go left
            return self.x <= 0
        else:
            # go right
            return self.x + self.W >= WINDOW_W

    def shot(self, position: numpy.ndarray) -> None:
        if not self.is_active() or self.is_waiting() or self.is_appearing():
            return
        if self.collision(position[0] * WINDOW_W, position[1] * WINDOW_H):
            Score.add_score(self.x, self.y, 1000)
            if self.direction[0] < 0:
                ObakeDeadParticle.add_particle(self.x, self.y, False)
            else:
                ObakeDeadParticle.add_particle(self.x, self.y, True)
            self.active = False

    def collision(self, sx: int, sy: int) -> bool:
        return (-self.COLLISION_MARGIN <= sx - self.x < self.W + self.COLLISION_MARGIN)\
            and (-self.COLLISION_MARGIN <= sy - self.y < self.H + self.COLLISION_MARGIN)

    def is_active(self) -> bool:
        return self.active

    def is_waiting(self) -> bool:
        return self.count < self.delay

    def is_appearing(self) -> bool:
        return self.delay < self.count < self.delay + self.APPEAR_TIME

    def draw(self) -> None:
        if not self.is_active() or self.is_waiting():
            return
        if self.is_appearing():
            dither = max(0, min(1, (self.count - self.delay) / self.APPEAR_TIME))
            pyxel.dither(dither)
        if self.direction[0] < 0:
            self.obake_image.draw(self.x, self.y, False)
        else:
            self.obake_image.draw(self.x, self.y, True)
        pyxel.dither(1)


class BackGroundImage:
    ASSET_FILE = './assets/background.png'
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


class BackGround:
    back_ground_image = None

    @classmethod
    def draw(cls):
        if cls.back_ground_image is None:
            cls.back_ground_image = BackGroundImage()
        cls.back_ground_image.draw()


class BulletImage:
    ASSET_FILE = './assets/bullet.png'
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
    ASSET_FILE = './assets/bullet_empty.png'
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
    W = 56
    H = 60
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


class ReloadImage:
    ASSET_FILE = './assets/reload.png'
    I = 1
    U = 0
    V = 57
    W = 68
    H = 21
    COLKEY = 0

    def __init__(self) -> None:
        self.load()

    def load(self) -> None:
        pyxel.images[self.I].load(self.U, self.V, self.ASSET_FILE)

    def draw(self, x: int, y: int, progress: float) -> None:
        h = int(self.H * min(max(progress, 0), 1))
        y = y + self.H - h
        v = self.V + self.H - h
        pyxel.blt(x, y, self.I, self.U, v, self.W, h, self.COLKEY)


class ReloadUI:
    W = 68
    H = 21
    X = BulletUI.X + (BulletUI.W - W) // 2
    Y = BulletUI.Y + (BulletUI.H - H) // 2

    def __init__(self) -> None:
        self.reload_image = ReloadImage()

    def draw(self, progress: float) -> None:
        self.reload_image.draw(self.X, self.Y, progress)


class BulletManager:
    BULLET_MAX_NUM = 6
    RELOAD_TIME = 60
    RELOAD_DISPLAY_OFFSET = 0.1

    def __init__(self) -> None:
        self.bullet_ui = BulletUI()
        self.reload_ui = ReloadUI()
        self.bullet_num = self.BULLET_MAX_NUM
        self.reload_count = self.RELOAD_TIME

    def update(self) -> None:
        if self.reload_count < self.RELOAD_TIME:
            self.reload_count += 1
            if self.reload_count == self.RELOAD_TIME:
                self.bullet_num = self.BULLET_MAX_NUM

    def shoot(self) -> bool:
        if self.is_reloading():
            return False
        if self.is_out_of_ammo():
            return False
        self.bullet_num -= 1
        return True

    def reload(self) -> None:
        if self.reload_count == self.RELOAD_TIME and not self.is_max_of_ammo():
            self.reload_count = 0

    def reset(self) -> None:
        self.bullet_num = self.BULLET_MAX_NUM
        self.reload_count = self.RELOAD_TIME

    def is_reloading(self) -> bool:
        return self.reload_count < self.RELOAD_TIME

    def is_out_of_ammo(self) -> bool:
        return self.bullet_num <= 0
    
    def is_max_of_ammo(self) -> bool:
        return self.bullet_num >= self.BULLET_MAX_NUM

    def draw(self) -> None:
        self.bullet_ui.draw(self.bullet_num)
        if self.reload_count < self.RELOAD_TIME:
            self.reload_ui.draw(self.reload_count / self.RELOAD_TIME + self.RELOAD_DISPLAY_OFFSET)


class NumberImage:
    ASSET_FILE = './assets/number.png'
    I = 1
    U = 0
    V = 78
    W = 50 
    NUMBER_W = 5 # NUMBER_W * 10 = W
    H = 9
    COLKEY = 0

    def __init__(self) -> None:
        self.load()

    def load(self) -> None:
        pyxel.images[self.I].load(self.U, self.V, self.ASSET_FILE)

    def draw(self, x: int, y: int, number: int) -> None:
        for digit in str(number):
            u = self.U + self.NUMBER_W * int(digit)
            w = self.NUMBER_W
            pyxel.blt(x, y, self.I, u, self.V, w, self.H, self.COLKEY)
            x = x + self.NUMBER_W


class LargeNumberImage:
    ASSET_FILE = './assets/large_number.png'
    I = 1
    U = 0
    V = 87
    W = 176
    NUMBER_W = 16 # NUMBER_W * 11 = W
    H = 28
    COLKEY = 0

    def __init__(self) -> None:
        self.load()

    def load(self) -> None:
        pyxel.images[self.I].load(self.U, self.V, self.ASSET_FILE)

    def draw(self, x: int, y: int, number: str) -> None:
        for digit in number:
            if digit == '.':
                u = self.U + self.NUMBER_W * 10
            else:
                u = self.U + self.NUMBER_W * int(digit)
            w = self.NUMBER_W
            pyxel.blt(x, y, self.I, u, self.V, w, self.H, self.COLKEY)
            x = x + self.NUMBER_W


class Score:
    score_list = []
    total = 0
    number_image = None
    COUNT_TIME = 30
    TOTAL_MARGIN_X = 10
    TOTAL_MARGIN_Y = 10

    def __init__(self, x: int, y: int, score: int, count: int) -> None:
        self.x = x
        self.y = y
        self.score = score
        self.count = count
        self.active = True

    def _update(self) -> None:
        if self.count > 0:
            self.count -= 1
        if self.count == 0:
            self.active = False

    def _draw(self) -> None:
        y = self.y - (self.COUNT_TIME - self.count) // 2
        self.number_image.draw(self.x, y, self.score)

    @classmethod
    def add_score(cls, x: int, y: int, score: int):
        if cls.number_image is not None:
            cls.score_list.append(Score(x, y, score, cls.COUNT_TIME))
            cls.total += score

    @classmethod
    def load(cls):
        if cls.number_image is None:
            cls.number_image = NumberImage()

    @classmethod
    def reset(cls):
        cls.score_list = []
        cls.total = 0

    @classmethod
    def update(cls):
        for score in cls.score_list:
            score._update()
        cls.score_list = [score for score in cls.score_list if score.active]

    @classmethod
    def draw(cls):
        for score in cls.score_list:
            score._draw()
        tx = WINDOW_W - cls.TOTAL_MARGIN_X - NumberImage.NUMBER_W * len(str(cls.total))
        ty = WINDOW_H - cls.TOTAL_MARGIN_Y - NumberImage.H
        cls.number_image.draw(tx, ty, cls.total)


class Wave:
    SPAWN_NUM = (
        (2, ),
        (2, ),
        (3, ),
        (3, ),
        (2, 2),
        (2, 2),
        (3, 2),
        (3, 2),
        (3, 3),
        (3, 3),
        (6, ),
    )

    SPAWN_POINT = (
        (65, 110),
        (130, 110),
        (180, 110),
        (90, 160),
        (150, 155),
        (210, 140),
    )

    SPAWN_DELAY = 120

    def __init__(self) -> None:
        self.wave_count = 0

    def spawn(self) -> list[Obake]:
        if self.wave_count == len(self.SPAWN_NUM):
            return []

        obake_list = []
        for i, spawn_num in enumerate(self.SPAWN_NUM[self.wave_count]):
            for x, y in random.sample(self.SPAWN_POINT, spawn_num):
                delay = self.SPAWN_DELAY * i
                obake_list.append(Obake(x, y, delay))

        self.wave_count += 1
        return obake_list

    def reset(self):
        self.wave_count = 0


class TitleImage:
    ASSET_FILE = './assets/title.png'
    X = 20
    Y = 5
    I = 2
    U = 0
    V = 0
    W = 216
    H = 108

    def __init__(self) -> None:
        self.load()

    def load(self) -> None:
        pyxel.images[self.I].load(self.U, self.V, self.ASSET_FILE)

    def draw(self) -> None:
        pyxel.blt(self.X, self.Y, self.I, self.U, self.V, self.W, self.H)


class StartImage:
    ASSET_FILE = './assets/start.png'
    I = 2
    U = 0
    V = 108
    W = 127
    H = 28
    X = (WINDOW_W - W) // 2
    Y = TitleImage.Y + TitleImage.H + 10

    def __init__(self) -> None:
        self.load()

    def load(self) -> None:
        pyxel.images[self.I].load(self.U, self.V, self.ASSET_FILE)

    def draw(self) -> None:
        pyxel.blt(self.X, self.Y, self.I, self.U, self.V, self.W, self.H)


class SensImage:
    ASSET_FILE = './assets/sens.png'
    I = 2
    U = 127
    V = 108
    W = 98
    H = 28
    X = (WINDOW_W // 2 - W) // 2
    Y = StartImage.Y + StartImage.H + 20

    def __init__(self) -> None:
        self.load()

    def load(self) -> None:
        pyxel.images[self.I].load(self.U, self.V, self.ASSET_FILE)

    def draw(self) -> None:
        pyxel.blt(self.X, self.Y, self.I, self.U, self.V, self.W, self.H)


class UpDownButtonImage:
    ASSET_FILE = './assets/up_down_button.png'
    I = 2
    U = 0
    V = 136
    W = 32
    H = 28
    BUTTON_W = 16

    def __init__(self) -> None:
        self.load()

    def load(self) -> None:
        pyxel.images[self.I].load(self.U, self.V, self.ASSET_FILE)

    def draw(self, x, y, up: bool) -> None:
        if up:
            pyxel.blt(x, y, self.I, self.U + self.BUTTON_W, self.V, self.BUTTON_W, self.H)
        else:
            pyxel.blt(x, y, self.I, self.U, self.V, self.BUTTON_W, self.H)


class UpDownButton:
    W = UpDownButtonImage.BUTTON_W
    H = UpDownButtonImage.H

    def __init__(self, x, y, up: bool) -> None:
        self.x = x
        self.y = y
        self.up = up
        self.up_down_image = UpDownButtonImage()

    def collision(self, x, y) -> bool:
        if self.x <= x <= self.x + self.W:
            if self.y <= y <= self.y + self.H:
                return True
        return False
    
    def draw(self) -> None:
        self.up_down_image.draw(self.x, self.y, self.up)


class TitleMenu:
    NUMBER_Y = SensImage.Y + SensImage.H // 2 - LargeNumberImage.H // 2 
    BUTTON_Y = SensImage.Y + SensImage.H // 2 - LargeNumberImage.H // 2
    DOWN_BUTTON_X = WINDOW_W // 2 + 10
    UP_BUTTON_X = WINDOW_W - 10 - UpDownButton.W

    MAX_SENS = 1
    MIN_SENS = 0.1
    SENS_RESOLUTION = 0.1

    def __init__(self, init_sens: float) -> None:
        self.sens = init_sens
        self.up_button = UpDownButton(self.UP_BUTTON_X, self.BUTTON_Y, True)
        self.down_button = UpDownButton(self.DOWN_BUTTON_X, self.BUTTON_Y,False)
        self.title_image = TitleImage()
        self.start_image = StartImage()
        self.sens_image = SensImage()
        self.large_number_image = LargeNumberImage()

    def update(self) -> None:
        pass

    def select(self, x, y) -> bool:
        if StartImage.X <= x <= StartImage.X + StartImage.W:
            if StartImage.Y <= y <= StartImage.Y + StartImage.H:
                return True
        if self.up_button.collision(x, y):
            self.sens_increment()
        if self.down_button.collision(x, y):
            self.sens_decrement()
        return False

    def sens_increment(self):
        self.sens = min(self.sens + self.SENS_RESOLUTION, self.MAX_SENS)

    def sens_decrement(self):
        self.sens = max(self.sens - self.SENS_RESOLUTION, self.MIN_SENS)

    def draw(self) -> None:
        self.title_image.draw()
        self.start_image.draw()
        self.sens_image.draw()
        
        self.up_button.draw()
        self.down_button.draw()

        sens = "{:.1f}".format(self.sens)
        number_x = self.DOWN_BUTTON_X + UpDownButton.W + (
            (self.UP_BUTTON_X - self.DOWN_BUTTON_X - UpDownButton.W) - len(sens)*LargeNumberImage.NUMBER_W
            ) // 2
        self.large_number_image.draw(number_x, self.NUMBER_Y, sens)


class FinishImage:
    ASSET_FILE = './assets/finish.png'
    I = 2
    U = 0
    V = 164
    W = 135
    H = 28

    def __init__(self) -> None:
        self.load()

    def load(self) -> None:
        pyxel.images[self.I].load(self.U, self.V, self.ASSET_FILE)

    def draw(self, x, y) -> None:
        pyxel.blt(x, y, self.I, self.U, self.V, self.W, self.H)


class ScoreImage:
    ASSET_FILE = './assets/score.png'
    I = 2
    U = 0
    V = 195
    W = 126
    H = 28

    def __init__(self) -> None:
        self.load()

    def load(self) -> None:
        pyxel.images[self.I].load(self.U, self.V, self.ASSET_FILE)

    def draw(self, x, y) -> None:
        pyxel.blt(x, y, self.I, self.U, self.V, self.W, self.H)


class BackButtonImage:
    ASSET_FILE = './assets/back_button.png'
    I = 2
    U = 135
    V = 164
    W = 36
    H = 36

    def __init__(self) -> None:
        self.load()

    def load(self) -> None:
        pyxel.images[self.I].load(self.U, self.V, self.ASSET_FILE)

    def draw(self, x, y) -> None:
        pyxel.blt(x, y, self.I, self.U, self.V, self.W, self.H)


class BackButton:
    W = BackButtonImage.W
    H = BackButtonImage.H

    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.back_button_image = BackButtonImage()

    def collision(self, x, y) -> bool:
        if self.x <= x <= self.x + self.W:
            if self.y <= y <= self.y + self.H:
                return True
        return False
    
    def draw(self) -> None:
        self.back_button_image.draw(self.x, self.y)


class Result:
    FINISH_X = (WINDOW_W - FinishImage.W) // 2
    FINISH_Y = WINDOW_H // 4

    SCORE_Y = WINDOW_H // 2

    BACK_BUTTON_X = (WINDOW_W - BackButton.W) // 2
    BACK_BUTTON_Y = WINDOW_H // 4 * 3

    def __init__(self) -> None:
        self.finish_image = FinishImage()
        self.score_image = ScoreImage()
        self.large_number_image = LargeNumberImage()
        self.back_button = BackButton(self.BACK_BUTTON_X, self.BACK_BUTTON_Y)

    def update(self) -> None:
        pass

    def select(self, x, y) -> bool:
        return self.back_button.collision(x, y)

    def draw(self) -> None:
        self.finish_image.draw(self.FINISH_X, self.FINISH_Y)

        score = str(Score.total)
        score_image_and_score_width = ScoreImage.W + LargeNumberImage.NUMBER_W*(len(score) + 1)
        score_image_x = (WINDOW_H - score_image_and_score_width) // 2
        score_x = score_image_x + ScoreImage.W + LargeNumberImage.NUMBER_W
        self.score_image.draw(score_image_x, self.SCORE_Y)
        self.large_number_image.draw(score_x, self.SCORE_Y, score)

        self.back_button.draw()


class App:
    def __init__(self) -> None:
        pyxel.init(WINDOW_W, WINDOW_H)
        pyxel.mouse(True)
        self.hands = []
        self.sens = 0.5
        self.shoot_detector = ShootDetector()
        self.reload_detector = ReloadDetector()
        self.point_detector = PointDetector()
        self.obake_list = []
        self.bullet_manger = BulletManager()
        Score.load()
        ObakeDeadParticle.load()
        self.wave = Wave()
        self.title_menu = TitleMenu(self.sens)
        self.result = Result()
        self.status = "title"
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
        if self.status == "title":
            self.title_menu.update()
            landmarks = js.getLandmarks().to_py()
            self.hands = [Hand(landmark, self.videoAspect, self.sens) for landmark in landmarks]
            if self.hands:
                self.point_detector.update(self.hands[0])
            if pyxel.btnr(pyxel.MOUSE_BUTTON_LEFT):
                if self.title_menu.select(pyxel.mouse_x, pyxel.mouse_y):
                    self.status = "play"
            point = self.point_detector.selected_point()
            if point:
                if self.title_menu.select(point[0], point[1]):
                    self.status = "play"

        if self.status == "play":
            if pyxel.btn(pyxel.KEY_R):
                self.reset()
                self.status = "title"
                return

            landmarks = js.getLandmarks().to_py()
            self.hands = [Hand(landmark, self.videoAspect, self.sens) for landmark in landmarks]
            if self.hands:
                self.shoot_detector.update(self.hands[0].target)
                self.reload_detector.update(self.hands[0])

            self.bullet_manger.update()

            if self.shoot_detector.is_shoot():
                if self.bullet_manger.shoot():
                    for obake in self.obake_list:
                        obake.shot(self.shoot_detector.shoot_position())
            if self.reload_detector.is_reload():
                self.bullet_manger.reload()

            for obake in self.obake_list:
                obake.update()

            self.obake_list = [obake for obake in self.obake_list if obake.is_active()]
            if len(self.obake_list) == 0:
                obake_list = self.wave.spawn()
                if obake_list:
                    self.obake_list.extend(obake_list)
                else:
                    self.status = 'result'

            Score.update()
            ObakeDeadParticle.update()

        if self.status == 'result':
            self.result.update()
            landmarks = js.getLandmarks().to_py()
            self.hands = [Hand(landmark, self.videoAspect, self.sens) for landmark in landmarks]
            if self.hands:
                self.point_detector.update(self.hands[0])
            if pyxel.btnr(pyxel.MOUSE_BUTTON_LEFT):
                if self.result.select(pyxel.mouse_x, pyxel.mouse_y):
                    self.reset()
                    self.status = "title"
            point = self.point_detector.selected_point()
            if point:
                if self.result.select(point[0], point[1]):
                    self.reset()
                    self.status = "title"

    def reset(self) -> None:
        self.obake_list = []
        ObakeDeadParticle.reset()
        self.bullet_manger.reset()
        self.wave.reset()
        Score.reset()
        self.shoot_detector = ShootDetector()

    def draw(self) -> None:
        pyxel.cls(0)
        if self.status == "title":
            self.title_menu.draw()
            for hand in self.hands:
                hand.draw()
            self.point_detector.draw()
        if self.status == "play":
            BackGround.draw()
            for hand in self.hands:
                hand.draw()
            shoot_position = self.shoot_detector.shoot_position()
            if shoot_position is not None:
                pyxel.circ(shoot_position[0] * WINDOW_W, shoot_position[1] * WINDOW_H, 3, 11)
            for obake in self.obake_list:
                obake.draw()
            self.bullet_manger.draw()
            Score.draw()
            ObakeDeadParticle.draw()
        if self.status == 'result':
            self.result.draw()
            for hand in self.hands:
                hand.draw()
            self.point_detector.draw()


App()
