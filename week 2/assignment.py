from math import pi, sqrt, inf
from abc import ABC, abstractmethod


# Task 1
class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class Line:
    def __init__(self, point_1: Point, point_2: Point):
        self.point_1 = point_1
        self.point_2 = point_2

    @property
    def is_vertical(self):
        return self.point_1.x == self.point_2.x

    @property
    def is_horizontal(self):
        return self.point_1.y == self.point_2.y

    @property
    def slope(self) -> float:
        if self.is_vertical:
            return inf

        if self.is_horizontal:
            return 0.0

        return (self.point_2.y - self.point_1.y) / (self.point_2.x - self.point_1.x)


class Circle:
    def __init__(self, center: Point, radius: float):
        if radius <= 0:
            raise ValueError("Radius must be positive.")

        self.center = center
        self.radius = radius

    @property
    def area(self) -> float:
        return pi * (self.radius**2)

    @property
    def circumference(self) -> float:
        return 2 * pi * self.radius


class Polygon(ABC):
    def __init__(self, *vertices: Point):
        if len(vertices) < 3:
            raise ValueError("A polygon must have at least 3 vertices.")
        self.vertices = vertices

    @property
    @abstractmethod
    def perimeter(self) -> float:
        pass


class Quadrilateral(Polygon):
    def __init__(self, *vertices: Point):
        if len(vertices) != 4:
            raise ValueError("A quadrilateral must have exactly 4 vertices.")
        self.vertices = vertices

    @property
    def perimeter(self):
        return sum(
            GeometryUtils.get_distance(self.vertices[i], self.vertices[(i + 1) % 4])
            for i in range(4)
        )


class GeometryUtils:
    @staticmethod
    def get_distance(point_1: Point, point_2: Point) -> float:
        return sqrt((point_2.x - point_1.x) ** 2 + (point_2.y - point_1.y) ** 2)

    @staticmethod
    def are_parallel(line_1: Line, line_2: Line) -> bool:
        if line_1.is_vertical and line_2.is_vertical:
            return True
        if line_1.is_vertical or line_2.is_vertical:
            return False
        return line_1.slope == line_2.slope

    @staticmethod
    def are_perpendicular(line_1: Line, line_2: Line) -> bool:
        if line_1.is_vertical and line_2.is_horizontal:
            return True
        if line_1.is_horizontal and line_2.is_vertical:
            return True
        if line_1.is_vertical or line_2.is_vertical:
            return False
        return line_1.slope * line_2.slope == -1

    @staticmethod
    def are_circles_intersect(circle_1: Circle, circle_2: Circle) -> bool:
        center_distance = GeometryUtils.get_distance(circle_1.center, circle_2.center)
        return center_distance <= circle_1.radius + circle_2.radius

    @staticmethod
    def get_circle_area(circle: Circle) -> float:
        return pi * (circle.radius**2)


# Task 2
class Vector:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class Enemy:
    def __init__(self, label: str, point: Point, moving_vector: Vector):
        self.label = label
        self.point = point
        self.moving_vector = moving_vector
        self.life_point = 10

    def __str__(self) -> str:
        return f"{self.label}, ({self.point.x}, {self.point.y}), {self.life_point}"

    def is_alive(self) -> bool:
        return self.life_point > 0

    def move(self):
        self.point.x = self.point.x + self.moving_vector.x
        self.point.y = self.point.y + self.moving_vector.y

    def get_attacked(self, attack_point: int):
        if self.is_alive():
            self.life_point -= attack_point


class Tower:
    def __init__(self, label: str, center: Point, range_: int, attack_point: int):
        self.label = label
        self.center = center
        self.range_ = range_
        self.attack_point = attack_point

    def is_in_tower_range(self, enemy: Enemy) -> bool:
        distance = sqrt(
            (enemy.point.x - self.center.x) ** 2 + (enemy.point.y - self.center.y) ** 2
        )
        return distance <= self.range_

    def attack_enemy(self, enemy: Enemy):
        if self.is_in_tower_range(enemy):
            enemy.get_attacked(self.attack_point)


class BasicTower(Tower):
    def __init__(
        self, label: str, center: Point, range_: int = 2, attack_point: int = 1
    ):
        super().__init__(label, center, range_, attack_point)


class AdvancedTower(Tower):
    def __init__(
        self, label: str, center: Point, range_: int = 4, attack_point: int = 2
    ):
        super().__init__(label, center, range_, attack_point)


class GameProcedurer:
    def __init__(self, turn: int, enemies: list[Enemy], towers: list[Tower]):
        self.turn = turn
        self.enemies = enemies
        self.towers = towers

    def display_result(self):
        for enemy in self.enemies:
            print(enemy)

    def start_game(self):
        for _ in range(self.turn):
            self._process_turn()

        self.display_result()

    def _process_turn(self):
        self._move_enemies()
        self._towers_attack()

    def _move_enemies(self):
        for enemy in self.enemies:
            if enemy.is_alive():
                enemy.move()

    def _towers_attack(self):
        for tower in self.towers:
            self._attack_enemies_in_range(tower)

    def _attack_enemies_in_range(self, tower: Tower):
        for enemy in self.enemies:
            if enemy.is_alive() and tower.is_in_tower_range(enemy):
                tower.attack_enemy(enemy)


class TaskHandler:
    @staticmethod
    def run_task_1():
        line_a = Line(Point(2, 4), Point(-6, 1))
        line_b = Line(Point(2, 2), Point(-6, -1))
        print(GeometryUtils.are_parallel(line_a, line_b))

        line_c = Line(Point(-1, 6), Point(-4, -4))
        print(GeometryUtils.are_perpendicular(line_a, line_c))

        circle_a = Circle(Point(6, 3), 2)
        print(f"{GeometryUtils.get_circle_area(circle_a):.2f}")

        circle_b = Circle(Point(8, 1), 1)
        print(GeometryUtils.are_circles_intersect(circle_a, circle_b))

        quad = Quadrilateral(Point(2, 0), Point(-1, -2), Point(4, -4), Point(5, -1))
        print(f"{quad.perimeter:.2f}")

    @staticmethod
    def run_task_2():
        turn = 10
        enemies = [
            Enemy("E1", Point(-10, 2), Vector(2, -1)),
            Enemy("E2", Point(-8, 0), Vector(3, 1)),
            Enemy("E3", Point(-9, -1), Vector(3, 0)),
        ]

        towers = [
            BasicTower("T1", Point(-3, 2)),
            BasicTower("T2", Point(-1, -2)),
            BasicTower("T3", Point(4, 2)),
            BasicTower("T4", Point(7, 0)),
            AdvancedTower("A1", Point(1, 1)),
            AdvancedTower("A2", Point(4, -3)),
        ]

        GameProcedurer(turn, enemies, towers).start_game()


if __name__ == "__main__":
    task_handler = TaskHandler()
    task_handler.run_task_1()
    task_handler.run_task_2()
