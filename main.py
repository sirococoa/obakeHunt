from collections import deque
import random
from typing import Any

import pyxel

import js

WINDOW_W = 256
WINDOW_H = 256


def distance(a: list[float], b: list[float]) -> float:
    return pyxel.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def subtraction(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


class Hand:
    POINT_SIZE = 7
    POINT_COLOR = 7

    TARGET_SIZE = 3
    TARGET_COLOR = 8

    def __init__(self, landmarks: Any, aspect: float, sens: float, time: float) -> None:
        self.points = []
        for landmark in landmarks:
            x, y, z = landmark['x'] - 0.5, landmark['y'] - 0.5, landmark['z']
            if aspect < 1:
                y = y / aspect
            else:
                x = x * aspect
            x, y = x + 0.5, y + 0.5
            self.points.append([1 - x, y, z])
        self.time = time

        self.target = self.calc_target(sens)

    def draw(self) -> None:
        for point in self.points:
            pyxel.circ(
                point[0] * WINDOW_W,
                point[1] * WINDOW_H,
                self.POINT_SIZE,
                self.POINT_COLOR,
            )
        pyxel.circ(
            self.target[0] * WINDOW_W,
            self.target[1] * WINDOW_H,
            self.TARGET_SIZE,
            self.TARGET_COLOR,
        )

    def thumb_length(self) -> float:
        return distance(self.points[2], self.points[3]) + distance(
            self.points[3], self.points[4]
        )

    def thumb_tip_point(self) -> list[float]:
        return self.points[4]

    def index_finger_length(self) -> float:
        return distance(self.points[5], self.points[8])

    def index_finger_vector(self) -> list[float]:
        return subtraction(self.points[8], self.points[5])

    def index_finger_base(self) -> list[float]:
        return self.points[5]

    def index_finger_tip_point(self) -> list[float]:
        return self.points[8]

    def middle_finger_tip_point(self) -> list[float]:
        return self.points[12]

    def ring_finger_pip_point(self) -> list[float]:
        return self.points[14]

    def calc_target(self, sens) -> list[float]:
        target_vector = []
        for base, vector in zip(self.index_finger_base(), self.index_finger_vector()):
            target_vector.append(base + vector / self.thumb_length() * sens)
        return target_vector[:2]


class CrossHairImage:
    ASSET_FILE = './assets/cross_hair.png'
    I = 1
    U = 0
    V = 115
    W = 20
    H = 20
    COLKEY = 0

    def __init__(self) -> None:
        self.load()

    def load(self) -> None:
        pyxel.images[self.I].load(self.U, self.V, self.ASSET_FILE)

    def draw(self, x: int, y: int) -> None:
        pyxel.blt(x, y, self.I, self.U, self.V, self.W, self.H, self.COLKEY)


class ShootDetector:
    SHOOT_DETECTION_LENGTH = 0.25
    MARK_DETECTION_ACCURACY = 0.05
    MARK_DETECTION_TIME = 0.5
    MARK_ACTIVE_TIME = 1

    def __init__(self) -> None:
        self.cross_hair_image = CrossHairImage()
        self.position: list[float] | None = None
        self.mark: list[float] | None = None
        self.shoot_flag = False
        self.mark_time  = -1

    def update(self) -> None:
        self.shoot_flag = False

    def detect(self, hand_history: list[Hand]) -> None:
        self.update_mark(hand_history)
        self.detect_shoot(hand_history[-1])

    def update_mark(self, hand_history: list[Hand]) -> None:
        current = hand_history[-1]
        if (
            self.mark is not None
            and current.time - self.mark_time > self.MARK_ACTIVE_TIME
        ):
            self.mark = None
        distance_list = []
        for hand in hand_history[::-1]:
            if current.time - hand.time > self.MARK_DETECTION_TIME:
                break
            distance_list.append(distance(current.target, hand.target))
        if all(d < self.MARK_DETECTION_ACCURACY for d in distance_list):
            self.mark = current.target
            self.mark_time = current.time

    def detect_shoot(self, current: Hand) -> None:
        if self.mark is None:
            return
        if self.mark[1] - current.target[1] > self.SHOOT_DETECTION_LENGTH:
            self.position = self.mark
            self.mark = None
            self.shoot_flag = True

    def is_shoot(self) -> bool:
        return self.shoot_flag

    def shoot_position(self) -> list[float]:
        return self.position

    def draw(self):
        if self.mark is not None:
            x = self.mark[0] * WINDOW_W - CrossHairImage.W // 2
            y = self.mark[1] * WINDOW_H - CrossHairImage.H // 2
            self.cross_hair_image.draw(x, y)


class ReloadDetector:
    def __init__(self) -> None:
        self.reload_flag = False

    def detect(self, hand: Hand) -> None:
        if (
            distance(hand.thumb_tip_point(), hand.ring_finger_pip_point())
            < hand.thumb_length()
            and distance(hand.index_finger_tip_point(), hand.middle_finger_tip_point())
            < hand.thumb_length()
        ):
            self.reload_flag = True
        else:
            self.reload_flag = False

    def is_reload(self) -> None:
        return self.reload_flag


class PointDetector:
    DETECTION_TIME = 1
    DETECTION_DETECTION_ACCURACY = 0.05

    POINT_INTERVAL = 20

    RADIUS = 20
    DETECTION_DRAW_START_TIME = 0.1

    def __init__(self) -> None:
        self.pointing_count = 0
        self.pointing_time = 0
        self.pointing_position = []

    def update(self) -> None:
        self.pointing_count += 1
        if self.pointing_count > self.POINT_INTERVAL:
            self.pointing_count = 0

    def detect(self, hand_history: list[Hand]) -> None:
        current = hand_history[-1]
        self.pointing_position = [
            int(current.index_finger_tip_point()[0] * WINDOW_W),
            int(current.index_finger_tip_point()[1] * WINDOW_H),
        ]
        self.pointing_time = 0
        for hand in hand_history[::-1]:
            self.pointing_time = current.time - hand.time
            if (
                distance(
                    current.index_finger_tip_point()[:1],
                    hand.index_finger_tip_point()[:1],
                )
                > self.DETECTION_DETECTION_ACCURACY
            ):
                break
        if self.pointing_time == 0:
            self.pointing_count = 0
            self.pointing_position = []

    def selected_point(self) -> list[int]:
        if self.pointing_count < self.POINT_INTERVAL:
            return []
        if self.pointing_time < self.DETECTION_TIME:
            return []
        return self.pointing_position

    def draw(self) -> None:
        if self.pointing_position and self.pointing_time > self.DETECTION_DRAW_START_TIME:
            pyxel.circb(self.pointing_position[0], self.pointing_position[1], self.RADIUS, 8)
            r = int(min(self.pointing_time / self.DETECTION_TIME, 1) * self.RADIUS)
            pyxel.circ(self.pointing_position[0], self.pointing_position[1], r, 8)


class VideoMarkImage:
    ASSET_FILE = './assets/video_mark.png'
    I = 1
    U = 0
    V = 135
    W = 60
    H = 30
    MARGIN = 10
    COLKEY = 15
    VIDEO_COLOR = 11

    def __init__(self) -> None:
        self.load()

    def load(self) -> None:
        pyxel.images[self.I].load(self.U, self.V, self.ASSET_FILE)

    def draw(self, color :int) -> None:
        x = self.MARGIN
        y = WINDOW_H - self.MARGIN - self.H
        pyxel.pal(self.VIDEO_COLOR, color)
        pyxel.blt(x, y, self.I, self.U, self.V, self.W, self.H, self.COLKEY)
        pyxel.pal(self.VIDEO_COLOR, self.VIDEO_COLOR)


class MediapipeManager:
    STORE_HAND_TIME = 2

    def __init__(self, sens: float) -> None:
        self.video_mark_image = VideoMarkImage()
        self.sens = sens
        self.connect_flag = False
        self.detect_flag = False
        self.update_flag = False
        self.videoAspect = 1
        self.hand_history: list[Hand] = []
        self.before_video_time = -1
        self.processing_time = 0
        self.shoot_detector = ShootDetector()
        self.reload_detector = ReloadDetector()
        self.point_detector = PointDetector()

    def connect(self) -> None:
        enable_webcam_flag = js.webcamRunning
        enable_detection_flag = js.detectionRunning
        if enable_webcam_flag and enable_detection_flag:
            videoWidth = js.videoWidth
            videoHeight = js.videoHeight
            self.videoAspect = videoWidth / videoHeight
            self.connect_flag = True

    def update(self) -> None:
        videoWidth = js.videoWidth
        videoHeight = js.videoHeight
        self.videoAspect = videoWidth / videoHeight

        self.get_landmarks()

        self.shoot_detector.update()
        self.point_detector.update()

        if self.update_flag and self.hand_history:
            self.shoot_detector.detect(self.hand_history)
            self.reload_detector.detect(self.hand_history[-1])
            self.point_detector.detect(self.hand_history)

    def latest_hand(self) -> Hand | None:
        if self.hand_history:
            return self.hand_history[-1]
        return None

    def get_landmarks(self) -> None:
        results = js.getResults().to_py()
        video_time = results['videoTime']
        landmarks = results['landmarks']
        if self.before_video_time == video_time:
            self.update_flag = False
            return
        if self.before_video_time > 0:
            self.processing_time = video_time - self.before_video_time
        self.update_flag = True
        self.before_video_time = video_time
        if len(landmarks) == 0:
            self.detect_flag = False
        else:
            self.detect_flag = True
            self.hand_history.append(Hand(landmarks[0], self.videoAspect, self.sens, video_time))
        self.hand_history = [
            hand
            for hand in self.hand_history
            if video_time - hand.time < self.STORE_HAND_TIME
        ]

    def is_detect(self) -> bool:
        return self.detect_flag

    def is_video_connect(self) -> bool:
        return self.connect_flag

    def draw(self) -> None:
        if self.hand_history:
            hand = self.hand_history[-1]
            if self.before_video_time == hand.time:
                hand.draw()
        if self.processing_time < 0.1:
            self.video_mark_image.draw(11)
        elif self.processing_time < 0.3:
            self.video_mark_image.draw(10)
        else:
            self.video_mark_image.draw(8)
        x = self.video_mark_image.MARGIN + 10
        y = WINDOW_H - self.video_mark_image.MARGIN - self.video_mark_image.H // 2
        pyxel.text(x, y, '{:.3f}'.format(self.processing_time), 7)


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
        dither = max(
            0, min(1, (self.ACTIVE_TIME - self.count) / self.ACTIVE_TIME + self.OFFSET)
        )
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
        cls.obake_dead_particle_list = [
            particle for particle in cls.obake_dead_particle_list if particle.active
        ]

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
        if self.count % (self.ZIGZAG_DURATION * 2) < self.ZIGZAG_DURATION:
            self.x += self.direction[0] + 0.5 * (self.direction[0] + self.direction[1])
            self.y += self.direction[1] + 0.5 * (-self.direction[0] + self.direction[1])
        else:
            self.x += self.direction[0] + 0.5 * (self.direction[0] - self.direction[1])
            self.y += self.direction[1] + 0.5 * (self.direction[0] + self.direction[1])
        if self.next_flip_count < self.count or self.is_edge():
            self.direction[0] = -self.direction[0]
            self.next_flip_count = self.count + random.randint(
                self.MIN_FLIP_COUNT, self.MAX_FLIP_COUNT
            )
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

    def shot(self, position: list[float]) -> None:
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
        return (
            -self.COLLISION_MARGIN <= sx - self.x < self.W + self.COLLISION_MARGIN
        ) and (-self.COLLISION_MARGIN <= sy - self.y < self.H + self.COLLISION_MARGIN)

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
    BULLET_W = 10  # BULLET_W * 3 + MARGIN_X * 2 = 56
    BULLET_H = 25  # BULLET_W * 2 + MARGIN_X * 1 = 60
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
            self.reload_ui.draw(
                self.reload_count / self.RELOAD_TIME + self.RELOAD_DISPLAY_OFFSET
            )


class NumberImage:
    ASSET_FILE = './assets/number.png'
    I = 1
    U = 0
    V = 78
    W = 50
    NUMBER_W = 5  # NUMBER_W * 10 = W
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
    NUMBER_W = 16  # NUMBER_W * 11 = W
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
        (2,),
        (2,),
        (3,),
        (3,),
        (2, 2),
        (2, 2),
        (3, 2),
        (3, 2),
        (3, 3),
        (3, 3),
        (6,),
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
            pyxel.blt(
                x, y, self.I, self.U + self.BUTTON_W, self.V, self.BUTTON_W, self.H
            )
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
        self.down_button = UpDownButton(self.DOWN_BUTTON_X, self.BUTTON_Y, False)
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

        sens = '{:.1f}'.format(self.sens)
        number_x = (
            self.DOWN_BUTTON_X
            + UpDownButton.W
            + (
                (self.UP_BUTTON_X - self.DOWN_BUTTON_X - UpDownButton.W)
                - len(sens) * LargeNumberImage.NUMBER_W
            )
            // 2
        )
        self.large_number_image.draw(number_x, self.NUMBER_Y, sens)


class FinishImage:
    ASSET_FILE = './assets/finish.png'
    I = 2
    U = 0
    V = 164
    W = 135
    H = 28
    COLKEY = 0

    def __init__(self) -> None:
        self.load()

    def load(self) -> None:
        pyxel.images[self.I].load(self.U, self.V, self.ASSET_FILE)

    def draw(self, x, y) -> None:
        pyxel.blt(x, y, self.I, self.U, self.V, self.W, self.H, self.COLKEY)


class ScoreImage:
    ASSET_FILE = './assets/score.png'
    I = 2
    U = 0
    V = 195
    W = 126
    H = 28
    COLKEY = 0

    def __init__(self) -> None:
        self.load()

    def load(self) -> None:
        pyxel.images[self.I].load(self.U, self.V, self.ASSET_FILE)

    def draw(self, x, y) -> None:
        pyxel.blt(x, y, self.I, self.U, self.V, self.W, self.H, self.COLKEY)


class BackButtonImage:
    ASSET_FILE = './assets/back_button.png'
    I = 2
    U = 135
    V = 164
    W = 36
    H = 36
    COLKEY = 15

    def __init__(self) -> None:
        self.load()

    def load(self) -> None:
        pyxel.images[self.I].load(self.U, self.V, self.ASSET_FILE)

    def draw(self, x, y) -> None:
        pyxel.blt(x, y, self.I, self.U, self.V, self.W, self.H, self.COLKEY)


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


class ObakeParticle:
    obake_particle_list = []
    obake_image = None
    SPEED = 1
    INTERVAL = 5
    MAX_RATE_SCORE = 40000

    def __init__(self) -> None:
        self.x = random.randint(0, WINDOW_W)
        self.y = WINDOW_H
        self.flip = random.random() < 0.5
        self.color = random.randint(1, 15)
        self.active = True

    def _update(self) -> None:
        self.y -= self.SPEED
        if self.y + ObakeDeadImage.H < 0:
            self.active = False

    def _draw(self) -> None:
        if self.obake_image is not None:
            pyxel.pal(7, self.color)
            self.obake_image.draw(self.x, self.y, self.flip)
            pyxel.pal()

    @classmethod
    def add_particle(cls):
        cls.obake_particle_list.append(ObakeParticle())

    @classmethod
    def load(cls):
        if cls.obake_image is None:
            cls.obake_image = ObakeImage()

    @classmethod
    def reset(cls):
        cls.obake_particle_list = []

    @classmethod
    def update(cls):
        if (
            pyxel.frame_count % cls.INTERVAL == 0
            and random.random() < Score.total / cls.MAX_RATE_SCORE
        ):
            cls.add_particle()
        for particle in cls.obake_particle_list:
            particle._update()
        cls.obake_particle_list = [
            particle for particle in cls.obake_particle_list if particle.active
        ]

    @classmethod
    def draw(cls):
        for particle in cls.obake_particle_list:
            particle._draw()


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
        score_image_and_score_width = ScoreImage.W + LargeNumberImage.NUMBER_W * (
            len(score) + 1
        )
        score_image_x = (WINDOW_H - score_image_and_score_width) // 2
        score_x = score_image_x + ScoreImage.W + LargeNumberImage.NUMBER_W
        self.score_image.draw(score_image_x, self.SCORE_Y)
        self.large_number_image.draw(score_x, self.SCORE_Y, score)

        self.back_button.draw()


class ShakeEffect:
    count = 0
    BREADTH = 3
    SHAKE_TIME = 10

    @classmethod
    def update(cls):
        if cls.count > 0:
            cls.count -= 1
        dx = random.randint(-cls.BREADTH, cls.BREADTH) * cls.count / cls.SHAKE_TIME
        dy = random.randint(-cls.BREADTH, cls.BREADTH) * cls.count / cls.SHAKE_TIME
        pyxel.camera(dx, dy)

    @classmethod
    def shake(cls):
        cls.count = cls.SHAKE_TIME

    @classmethod
    def reset(cls):
        cls.count = 0
        pyxel.camera()


class App:
    INIT_SENS = 0.5

    def __init__(self) -> None:
        pyxel.init(WINDOW_W, WINDOW_H, title='obakeHunt')
        pyxel.mouse(True)
        self.mediapipe_manager = MediapipeManager(self.INIT_SENS)
        self.obake_list = []
        self.bullet_manger = BulletManager()
        Score.load()
        ObakeDeadParticle.load()
        ObakeParticle.load()
        self.wave = Wave()
        self.title_menu = TitleMenu(self.INIT_SENS)
        self.result = Result()
        self.status = 'title'
        pyxel.run(self.update, self.draw)

    def update(self) -> None:
        if not self.mediapipe_manager.is_video_connect():
            self.mediapipe_manager.connect()
            return

        self.mediapipe_manager.update()

        if self.status == 'title':
            self.title_menu.update()
            if pyxel.btnr(pyxel.MOUSE_BUTTON_LEFT):
                if self.title_menu.select(pyxel.mouse_x, pyxel.mouse_y):
                    self.status = 'play'
            point = self.mediapipe_manager.point_detector.selected_point()
            if point:
                if self.title_menu.select(point[0], point[1]):
                    self.status = 'play'
            self.mediapipe_manager.sens = self.title_menu.sens

        if self.status == 'play':
            if pyxel.btn(pyxel.KEY_R):
                self.reset()
                self.status = 'title'
                return

            self.bullet_manger.update()

            if self.mediapipe_manager.shoot_detector.is_shoot():
                if self.bullet_manger.shoot():
                    for obake in self.obake_list:
                        obake.shot(
                            self.mediapipe_manager.shoot_detector.shoot_position()
                        )
                    ShakeEffect.shake()
            if self.mediapipe_manager.reload_detector.is_reload():
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
            ShakeEffect.update()

        if self.status == 'result':
            ObakeParticle.update()
            self.result.update()
            if pyxel.btnr(pyxel.MOUSE_BUTTON_LEFT):
                if self.result.select(pyxel.mouse_x, pyxel.mouse_y):
                    self.reset()
                    self.status = 'title'
            point = self.mediapipe_manager.point_detector.selected_point()
            if point:
                if self.result.select(point[0], point[1]):
                    self.reset()
                    self.status = 'title'

    def reset(self) -> None:
        self.obake_list = []
        ObakeDeadParticle.reset()
        ObakeParticle.reset()
        self.bullet_manger.reset()
        self.wave.reset()
        Score.reset()
        ShakeEffect.reset()

    def draw(self) -> None:
        pyxel.cls(0)
        if self.status == 'title':
            if self.mediapipe_manager.is_video_connect():
                videoWidth = js.videoWidth
                videoHeight = js.videoHeight
                pyxel.text(
                    WINDOW_W // 4,
                    WINDOW_H - 10,
                    'CAMERA {}x{}'.format(videoWidth, videoHeight),
                    7,
                )
                if self.mediapipe_manager.is_detect():
                    pyxel.text(WINDOW_W // 2 + 10, WINDOW_H - 10, 'HAND: found', 7)
                else:
                    pyxel.text(WINDOW_W // 2 + 10, WINDOW_H - 10, 'HAND: not found', 7)
            else:
                pyxel.text(
                    WINDOW_W // 4, WINDOW_H - 10, 'Waiting for camera to connect', 7
                )
            self.title_menu.draw()
            self.mediapipe_manager.draw()
            self.mediapipe_manager.point_detector.draw()
        if self.status == 'play':
            BackGround.draw()
            self.mediapipe_manager.draw()
            for obake in self.obake_list:
                obake.draw()
            self.mediapipe_manager.shoot_detector.draw()
            self.bullet_manger.draw()
            Score.draw()
            ObakeDeadParticle.draw()
        if self.status == 'result':
            ObakeParticle.draw()
            self.result.draw()
            self.mediapipe_manager.draw()
            self.mediapipe_manager.point_detector.draw()


App()
