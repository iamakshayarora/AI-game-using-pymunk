import random
import math
import numpy as np

import pygame
from pygame.color import THECOLORS

import pymunk
from pymunk.vec2d import Vec2d
from pymunk.pygame_util import DrawOptions 

# PyGame init
width = 1000
height = 500
pygame.init()
screen = pygame.display.set_mode((width, height))
clock = pygame.time.Clock()
screen.set_alpha(None)
show_sensors = True
draw_screen = True

class GameState:
    def __init__(self):
        self.crashed = False
        self.space = pymunk.Space()
        self.space.gravity = pymunk.Vec2d(0., 0.)
        self.draw_options = pymunk.pygame_util.DrawOptions(screen)
        self.create_car(100, 100, 0.5)
        self.num_steps = 0

        # static= [pymunk.Segment(self.space.static_body, (50, 50), (50, 550), 5)
        #     ,pymunk.Segment(self.space.static_body, (50, 550), (650, 550), 5)
        #     ,pymunk.Segment(self.space.static_body, (650, 550), (650, 50), 5)
        #     ,pymunk.Segment(self.space.static_body, (50, 50), (650, 50), 5)
        #     ]  

        static = [
            pymunk.Segment(
                self.space.static_body,
                (0, 1), (0, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, height), (width, height), 1),
            pymunk.Segment(
                self.space.static_body,
                (width-1, height), (width-1, 1), 1),
            pymunk.Segment(
                self.space.static_body,
                (1, 1), (width, 1), 1)
        ]
        for s in static:
            s.friction = 1.0
            s.group = 1
            s.collision_type = 1
            s.color = THECOLORS['red']
        self.space.add(static)

        self.obstacles = []
        self.obstacles.append(self.create_obstacle(300, 360, 100))
        self.obstacles.append(self.create_obstacle(790, 355, 125))
        self.obstacles.append(self.create_obstacle(575, 160, 35))
        self.obstacles.append(self.create_obstacle(200, 150, 35))

        self.create_cat()

    def create_obstacle(self, x, y, r):
        c_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        c_shape = pymunk.Poly.create_box(c_body,(200,100,r))
        c_shape.elasticity = 1.0
        c_body.position = x, y
        c_shape.color = THECOLORS["orange"]
        self.space.add(c_body, c_shape)
        return c_body

    def create_cat(self):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.cat_body = pymunk.Body(1, inertia)
        self.cat_body.position = 50, height - 100
        self.cat_shape = pymunk.Circle(self.cat_body, 30)
        self.cat_shape.color = THECOLORS["red"]
        self.cat_shape.elasticity = 1.0
        self.cat_shape.angle = 0.5
        direction = Vec2d(1, 0).rotated(self.cat_body.angle)
        self.space.add(self.cat_body, self.cat_shape)

    def create_car(self, x, y, r):
        inertia = pymunk.moment_for_circle(1, 0, 14, (0, 0))
        self.car_body = pymunk.Body(1, inertia)
        self.car_body.position = x, y
        self.car_shape = pymunk.Circle(self.car_body, 25)
        self.car_shape.color = THECOLORS["green"]
        self.car_shape.elasticity = 1.0
        self.car_body.angle = r
        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.apply_impulse_at_local_point(driving_direction)
        self.space.add(self.car_body, self.car_shape)

    def frame_step(self, action):
        if action == 0:  
            self.car_body.angle -= .2
        elif action == 1:  
            self.car_body.angle += .2

        if self.num_steps % 100 == 0:
            self.move_obstacles()

        if self.num_steps % 5 == 0:
            self.move_cat()

        driving_direction = Vec2d(1, 0).rotated(self.car_body.angle)
        self.car_body.velocity = 10 * driving_direction

        screen.fill(THECOLORS["white"])
        self.space.debug_draw(self.draw_options)
        self.space.step(1./10)
        if draw_screen:
            pygame.display.flip()
        clock.tick()

        x, y = self.car_body.position
        readings = self.get_sonar_readings(x, y, self.car_body.angle)
        normalized_readings = [(x-20.0)/20.0 for x in readings] 
        state = np.array([normalized_readings])

        if self.car_is_crashed(readings):
            self.crashed = True
            reward = -500
            self.recover_from_crash(driving_direction)
        else:
            reward = -5 + int(self.sum_readings(readings) / 10)
        self.num_steps += 1

        return reward, state

    def move_obstacles(self):
        # Randomly move obstacles around.
        for obstacle in self.obstacles:
            speed = random.randint(20, 50)
            direction = Vec2d(1, 0).rotated(self.car_body.angle + random.randint(-2, 2))
            obstacle.velocity = speed * direction

    def move_cat(self):
        speed = random.randint(10, 30)
        self.cat_body.angle -= random.randint(-1, 1)
        direction = Vec2d(1, 0).rotated(self.cat_body.angle)
        self.cat_body.velocity = speed * direction

    def car_is_crashed(self, readings):
        if readings[0] == 1 or readings[1] == 1 or readings[2] == 1:
            return True
        else:
            return False

    def recover_from_crash(self, driving_direction):
        while self.crashed:
            self.car_body.velocity = -10 * driving_direction
            self.crashed = False
            for i in range(10):
                self.car_body.angle += .2  
                screen.fill(THECOLORS["blue"])  
                self.space.debug_draw(self.draw_options)
                self.space.step(1./10)
                if draw_screen:
                    pygame.display.flip()
                clock.tick()

    def sum_readings(self, readings):
        tot = 0
        for i in readings:
            tot += i
        return tot

    def get_sonar_readings(self, x, y, angle):
        readings = []
        arm_left = self.make_sonar_arm(x, y)
        arm_middle = arm_left
        arm_right = arm_left
        readings.append(self.get_arm_distance(arm_left, x, y, angle, 0.75))
        readings.append(self.get_arm_distance(arm_middle, x, y, angle, 0))
        readings.append(self.get_arm_distance(arm_right, x, y, angle, -0.75))

        if show_sensors:
            pygame.display.update()
        return readings

    def get_arm_distance(self, arm, x, y, angle, offset):
        i = 0
        for point in arm:
            i += 1
            rotated_p = self.get_rotated_point(
                x, y, point[0], point[1], angle + offset
            )

            if rotated_p[0] <= 0 or rotated_p[1] <= 0 \
                    or rotated_p[0] >= width or rotated_p[1] >= height:
                return i 
            else:
                obs = screen.get_at(rotated_p)
                if self.get_track_or_not(obs) != 0:
                    return i

            if show_sensors:
                pygame.draw.circle(screen, (255, 255, 255), (rotated_p), 2)

        return i

    def make_sonar_arm(self, x, y):
        spread = 10
        distance = 20 
        arm_points = []
        for i in range(1, 40):
            arm_points.append((distance + x + (spread * i), y))
        return arm_points

    def get_rotated_point(self, x_1, y_1, x_2, y_2, radians):
        # Rotate x_2, y_2 around x_1, y_1 by angle.
        x_change = (x_2 - x_1) * math.cos(radians) + \
            (y_2 - y_1) * math.sin(radians)
        y_change = (y_1 - y_2) * math.cos(radians) - \
            (x_1 - x_2) * math.sin(radians)
        new_x = x_change + x_1
        new_y = height - (y_change + y_1)
        return int(new_x), int(new_y)

    def get_track_or_not(self, reading):
        if reading == THECOLORS['white']:
            return 0
        else:
            return 1


if __name__ == "__main__":
    game_state = GameState()
    running=True
    while running:
        game_state.frame_step((random.randint(0, 2)))
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                running = True
            if event.type == pygame.QUIT:
                running = False
