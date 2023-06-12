import math
import sys

import numpy as np


class Point:
    def __init__(self, x: int | float, y: int | float, z: int | float):
        self.x = x
        self.y = y
        self.z = z

        self.coordinates = [x, y, z]

    def distance(self, other) -> int | float:
        dx = self.x - other.x
        dy = self.y - other.y
        dz = self.z - other.z
        return math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    def __getitem__(self, item: int):
        return self.coordinates[item]

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar):
        return Point(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar):
        return Point(self.x * scalar, self.y * scalar, self.z * scalar)

    def __truediv__(self, scalar):
        return Point(self.x / scalar, self.y / scalar, self.z / scalar)

    def __eq__(self, other):
        if isinstance(other, Point):
            return (self.x == other.x and
                    self.y == other.y and
                    self.z == other.z)

        return NotImplemented

    def __str__(self):
        return "Point({:.4f}, {:.4f}, {:.4f})".format(*self.coordinates)


class Angle(Point):
    def __init__(self, x: float, y: float, z: float):
        super().__init__(x, y, z)

    # Override coordinate transformation operators
    def __add__(self, other):
        x = (self.x + other.x) % (2 * math.pi)
        y = (self.y + other.y) % (2 * math.pi)
        z = (self.z + other.z) % (2 * math.pi)
        return Angle(x, y, z)

    def __sub__(self, other):
        x = (self.x - other.x) % (2 * math.pi)
        y = (self.y - other.y) % (2 * math.pi)
        z = (self.z - other.z) % (2 * math.pi)
        return Angle(x, y, z)

    def __mul__(self, scalar):
        x = (self.x * scalar) % (2 * math.pi)
        y = (self.y * scalar) % (2 * math.pi)
        z = (self.z * scalar) % (2 * math.pi)
        return Angle(x, y, z)

    def __truediv__(self, scalar):
        x = (self.x / scalar) % (2 * math.pi)
        y = (self.y / scalar) % (2 * math.pi)
        z = (self.z / scalar) % (2 * math.pi)
        return Angle(x, y, z)

    def toVector(self):
        return Vector(self.x, self.y, self.z)


class Vector:
    def __init__(self, *args):
        if len(args) == 1:
            assert isinstance(args[0], Point)
            self.point = args[0]
        elif len(args) == 3:
            assert all(map(isinstance, args, [(int, float)] * 3))
            self.point = Point(*args)

    def __str__(self):
        return "Vector({:.4f}, {:.4f}, {:.4f})".format(
            *self.point.coordinates)

    def length(self):
        return self.vs.init_pt.distance(self.point)

    def normalize(self):
        if self.length() == 0:
            return self

        return Vector(self.point / self.length())

    def __bool__(self):
        return bool(self.point)

    def __eq__(self, other: "Vector"):
        return self.point == other.point

    def __ne__(self, other: "Vector"):
        return self.point != other.point

    def __add__(self, other: "Vector"):
        return Vector(self.point + other.point)

    def __sub__(self, other):
        return Vector(self.point - other.point)

    def __mul__(self, other):
        if isinstance(other, Vector):
            return sum(self.point.coordinates[i] * other.point.coordinates[i]
                       for i in range(3))
        else:
            return Vector(self.point * other)

    def __rmul__(self, other):
        assert isinstance(other, (int, float))

        return Vector(self.point * other)

    def __truediv__(self, other):
        assert isinstance(other, (int, float))

        return Vector(self.point / other)




    def __pow__(self, other):
        x1 = self.point.coordinates[0]
        y1 = self.point.coordinates[1]
        z1 = self.point.coordinates[2]
        x2 = other.point.coordinates[0]
        y2 = other.point.coordinates[1]
        z2 = other.point.coordinates[2]

        x = self.vs.basis[0] * (y1 * z2 - y2 * z1)
        y = self.vs.basis[1] * -(x1 * z2 - x2 * z1)
        z = self.vs.basis[2] * (y2 * x1 - y1 * x2)

        return x + y + z

    def rotate(self, x_angle: float = 0, y_angle: float = 0,
               z_angle: float = 0):
        x_angle = math.pi * x_angle / 360
        y_angle = math.pi * y_angle / 360
        z_angle = math.pi * z_angle / 360

        # Поворот вокруг оси Ox
        y_old = self.point.coordinates[1]
        z_old = self.point.coordinates[2]
        self.point.coordinates[1] = y_old * math.cos(x_angle) \
                                    - z_old * math.sin(x_angle)
        self.point.coordinates[2] = y_old * math.sin(x_angle) \
                                    + z_old * math.cos(x_angle)

        # Поворот вокруг оси Oy
        x_old = self.point.coordinates[0]
        z_old = self.point.coordinates[2]
        self.point.coordinates[0] = x_old * math.cos(y_angle) \
                                    + z_old * math.sin(y_angle)
        self.point.coordinates[2] = x_old * -math.sin(y_angle) \
                                    + z_old * math.cos(y_angle)

        # Поворот вокруг оси Oz
        x_old = self.point.coordinates[0]
        y_old = self.point.coordinates[1]
        self.point.coordinates[0] = x_old * math.cos(z_angle) \
                                    - y_old * math.sin(z_angle)
        self.point.coordinates[1] = x_old * math.sin(z_angle) \
                                    + y_old * math.cos(z_angle)


class VectorSpace:
    init_pt = Point(0, 0, 0)
    basis = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]

    def __init__(self, init_pt: Point = init_pt, dir1: Vector = None,
                 dir2: Vector = None, dir3: Vector = None):
        self.init_pt = init_pt
        for i, d in enumerate((dir1, dir2, dir3)):
            if d is not None:
                VectorSpace.basis[i] = d.normalize()


Vector.vs = VectorSpace()


class Parameters:
    def __init__(self, position: Point, rotation: Vector):
        self.position = position
        self.rotation = rotation

    def move(self, move_to: Point):
        self.position = self.position + move_to

    def scaling(self, value):
        pass

    def rotate(self, x_angle: float = 0, y_angle: float = 0, z_angle: float = 0):
        self.rotation.rotate(x_angle, y_angle, z_angle)


class Object:
    def __init__(self, position: Point, rotation: Vector):
        self.position = position
        self.rotation = rotation

    def nearest_point(self, points: list[Point]) -> Point | None:
        nearest_point = None
        min_distance = float('inf')

        for point in points:
            distance = self.position.distance(point)

            if distance < min_distance:
                min_distance = distance
                nearest_point = point

        return nearest_point

    def contains(self, point: Point) -> bool:
        if self.__class__.function(self, point) == 0:
            return True

        return False

    def intersect(self, ray) -> Vector | None:
        t0 = (self.rotation * Vector(self.position[0], self.position[1], self.position[2]) -
              self.rotation * Vector(ray.position[0], ray.position[1], ray.position[2])) / (
                     self.rotation * ray.direction)

        if t0 < 0:
            return None

        return t0 * ray.direction.length()


class Plane(Object):
    def __init__(self, position: Point, rotation: Vector):
        super().__init__(position, rotation)
        self.parameters = Parameters(self.position, self.rotation)

    def _update(self):
        self.position = self.parameters.position
        self.rotation = self.parameters.rotation

    def contains(self, point: Point, eps=1e-6) -> bool:
        self._update()



        return abs(self.rotation * Vector(point - self.position)) < eps

    def intersect(self, ray) -> float:
        self._update()

        if self.rotation * ray.direction != 0 and not (
                self.contains(ray.position) and self.contains(ray.direction.point)):
            t0 = (self.rotation * Vector(self.position) -
                  self.rotation * Vector(ray.position)) / (self.rotation * ray.direction)
            if t0 >= 0:
                return t0 * ray.direction.length()
        elif self.contains(ray.position):
            return 0

    def nearest_point(self, *points: Point) -> Point:
        self._update()
        r_min = sys.maxsize
        min_point = Vector.vs.init_pt
        for point in points:
            r = abs(self.rotation * Vector(point - self.position)) / self.rotation.length()
            if r == 0:
                return point

            if r < r_min:
                r_min = r
                min_point = point

        return min_point

    def __str__(self):
        self._update()
        return f'Plane({self.position}, {str(self.rotation)})'


class BoundedPlaneParams(Parameters):
    def __init__(self, position: Point, rotation: Vector,
                 u, v, du, dv):
        super().__init__(position, rotation)
        self.u = u
        self.v = v
        self.du = du
        self.dv = dv

    def scaling(self, value):
        self.du = self.du * value
        self.dv = self.dv * value

    def rotate(self, x_angle=0, y_angle=0, z_angle=0):
        self.rotation.rotate(x_angle, y_angle, z_angle)
        self.u.rotate(x_angle, y_angle, z_angle)
        self.v.rotate(x_angle, y_angle, z_angle)


class BoundedPlane(Plane):
    def __init__(self, position: Point, rotation: Vector, du, dv):
        super().__init__(position, rotation)
        self.du = du
        self.dv = dv

        y_dir = Vector.vs.basis[1]
        if self.rotation.point == y_dir.point \
                or self.rotation.point == -1 * y_dir.point:
            y_dir = Vector.vs.basis[0]

        self.u: Vector = (self.rotation ** y_dir).normalize()
        self.v: Vector = (self.rotation ** self.u).normalize()

        self.parameters = BoundedPlaneParams(self.position, self.rotation,
                                             self.u, self.v, self.du, self.dv)

    def _update(self):
        self.position = self.parameters.position
        self.rotation = self.parameters.rotation
        self.u = self.parameters.u
        self.v = self.parameters.v
        self.du = self.parameters.du
        self.dv = self.parameters.dv

    def __str__(self):
        self._update()

        return f'Plane({self.position}, {self.rotation} du={self.du}, dv={self.dv})'

    def in_boundaries(self, point: Point) -> bool:
        self._update()

        corner = self.u * self.du + self.v * self.dv
        delta_x, delta_y, delta_z = corner.point.coordinates

        # print('This is here', corner.point)

        return abs(point.coordinates[0] - self.position.coordinates[0]) <= abs(delta_x) \
               and abs(point.coordinates[1] - self.position.coordinates[1]) <= abs(delta_y) \
               and abs(point.coordinates[2] - self.position.coordinates[2]) <= abs(delta_z)

    def contains(self, point: Point, eps=1e-6) -> bool:
        self._update()

        if self.in_boundaries(point):
            return abs(self.rotation * Vector(point - self.position)) < eps

        return False

    def intersect(self, ray) -> float or None:
        self._update()
        if self.rotation * ray.direction != 0:
            if self.contains(ray.position):
                return 0

            t0 = (self.rotation * Vector(self.position) -
                  self.rotation * Vector(ray.position)) / (self.rotation * ray.direction)
            int_point = ray.direction.point * t0 + ray.position
            if t0 >= 0 and self.in_boundaries(int_point):
                return int_point.distance(ray.position)




        elif self.rotation * Vector(ray.direction.point
                                    + ray.position - self.position) == 0:
            # Проекции вектора из точки центра плоскости
            # к точке начала вектора v на направляющие вектора плоскости
            r_begin = Vector(ray.position - self.position)
            # Если начало вектора совпадает с центром плоскости
            if r_begin.length() == 0:
                return 0

            begin_pr1 = r_begin * self.u * self.du / r_begin.length()
            begin_pr2 = r_begin * self.v * self.dv / r_begin.length()
            if abs(begin_pr1) <= 1 and abs(begin_pr2) <= 1:
                return 0

            # Проекции вектора из точки центра плоскости
            # к точке конца вектора v на направляющие вектора плоскости
            r_end = r_begin + ray.direction
            if r_end.length() == 0:
                if abs(begin_pr1) > 1 or abs(begin_pr2) > 1:
                    if begin_pr1 > 1:
                        begin_pr1 -= 1
                    elif begin_pr1 < -1:
                        begin_pr1 += 1

                    if begin_pr2 > 1:
                        begin_pr2 -= 1
                    elif begin_pr2 < -1:
                        begin_pr2 += 1

                    return begin_pr1 * self.du + begin_pr2 * self.dv

                return 0

            def find_point(ray1, ray2):
                if ray1.direction.point.coordinates[0] != 0:
                    x0 = ray1.position.coordinates[0]
                    y0 = ray1.position.coordinates[1]
                    xr = ray2.position.coordinates[0]
                    yr = ray2.position.coordinates[1]
                    vx = ray1.direction.point.coordinates[0]
                    vy = ray1.direction.point.coordinates[1]
                    ux = ray2.direction.point.coordinates[0]
                    uy = ray2.direction.point.coordinates[1]

                    t1 = ((x0 - xr) * vy / vx + yr - y0) \
                         / (uy - ux * vy / vx)
                    s1 = (t1 * ux + x0 - xr) / vx
                    return t1, s1

                elif ray1.direction.point.coordinates[1] != 0:
                    x0 = ray1.position.coordinates[0]
                    y0 = ray1.position.coordinates[1]
                    xr = ray2.position.coordinates[0]
                    yr = ray2.position.coordinates[1]
                    vx = ray1.direction.point.coordinates[0]
                    vy = ray1.direction.point.coordinates[1]
                    ux = ray2.direction.point.coordinates[0]
                    uy = ray2.direction.point.coordinates[1]
                    t1 = ((y0 - yr) * vx / vy + xr - x0) \
                         / (ux - uy * vx / vy)
                    s1 = (t0 * uy + y0 - yr) / vy
                    return t1, s1

                elif ray1.direction.point.coordinates[2] != 0:
                    z0 = ray1.position.coordinates[2]
                    y0 = ray1.position.coordinates[1]
                    zr = ray2.position.coordinates[2]
                    yr = ray2.position.coordinates[1]
                    vz = ray1.direction.point.coordinates[2]
                    vy = ray1.direction.point.coordinates[1]
                    uz = ray2.direction.point.coordinates[2]
                    uy = ray2.direction.point.coordinates[1]
                    t1 = ((z0 - zr) * vy / vz + yr - y0) / (
                            uy - uz * vy / vz)
                    s1 = (t0 * uz + z0 - zr) / vz
                    return t1, s1

            if abs(begin_pr1) > self.du:
                if self.u * ray.direction == 0:
                    return None

                sign = 1 if begin_pr1 > 0 else -1
                t0, s0 = find_point(
                    Ray(sign * self.du * self.u.point + self.position,
                        self.dv * self.v), ray)
                if s0 >= 0 and abs(t0) <= 1:
                    return s0 * ray.direction.length()

            elif abs(begin_pr2) > self.dv:
                if self.v * ray.direction == 0:
                    return None


                sign = 1 if begin_pr2 > 0 else -1
                t0, s0 = find_point(
                    Ray(sign * self.dv * self.v.point + self.position,
                        self.du * self.u), ray)
                if s0 >= 0 and abs(t0) <= 1:
                    return s0 * ray.direction.length()

    def nearest_point(self, *points: Point) -> Point:
        self._update()

        r_min = sys.maxsize
        min_point = Vector.vs.init_pt
        r = 0
        for point in points:
            r_begin = Vector(point - self.position)
            # Если начало вектора совпадает с центром плоскости
            if r_begin.length() == 0:
                return point

            projection1 = r_begin * self.rotation / r_begin.length()
            projection2 = r_begin * self.u * self.du / r_begin.length()
            projection3 = r_begin * self.v * self.dv / r_begin.length()
            sign = lambda x: 1 if x > 0 else -1
            if abs(projection2) <= 1 and abs(projection3) <= 1:
                r = projection1 * self.rotation.length()
            elif abs(projection2) > 1 and abs(projection3) > 1:
                proj2 = projection2 - sign(projection2)
                proj3 = projection3 - sign(projection3)
                r = self.rotation * -projection1 + self.u * proj2 \
                    + self.v * proj3 + Vector(point)
                r = r.length()
            elif abs(projection2) > 1:
                proj2 = projection2 - sign(projection2)
                r = self.rotation * -projection1 + self.u * proj2 \
                    + Vector(point)
                r = r.length()
            elif abs(projection3) > 1:
                proj3 = projection3 - sign(projection3)
                r = self.rotation * -projection1 + self.v * proj3 \
                    + Vector(point)
                r = r.length()

            if r < r_min:
                r_min = r
                min_point = point

        return min_point


class SphereParams(Parameters):
    def __init__(self, position: Point, rotation: Vector, radius):
        super().__init__(position, rotation)
        self.radius = radius

    def scaling(self, value):
        self.radius = self.radius * value


class Sphere(Object):
    def __init__(self, position: Point, rotation: Vector, radius):
        super().__init__(position, rotation)
        self.parameters = SphereParams(self.position, self.rotation.normalize() * radius,
                                       radius)

    def _update(self):
        self.position = self.parameters.position
        self.rotation = self.parameters.rotation
        self.radius = self.parameters.radius

    def __str__(self):
        self._update()
        return f'Sphere({self.position}, {str(self.rotation)}, radius={self.radius})'

    def contains(self, point: Point, eps=1e-6) -> bool:
        self._update()
        return self.position.distance(point) - self.radius <= eps

    def intersect(self, ray) -> float or None:
        self._update()

        a = ray.direction * ray.direction
        b = 2 * ray.direction * Vector(ray.position - self.position)
        c = Vector(self.position) * Vector(self.position) + \
            Vector(ray.position) * Vector(ray.position) \
            - 2 * Vector(self.position) * Vector(ray.position) - self.radius ** 2

        d = b ** 2 - 4 * a * c
        if d > 0:
            t1 = (-b + math.sqrt(d)) / (2 * a)
            t2 = (-b - math.sqrt(d)) / (2 * a)
            # Смотрим пересечения с поверхностью сферы
            if t2 < 0 <= t1 or 0 < t1 <= t2:
                return t1 * ray.direction.length()
            elif t1 < 0 <= t2 or 0 < t2 <= t1:
                return t2 * ray.direction.length()

        elif d == 0:
            t0 = -b / (2 * a)
            if t0 >= 0:
                return t0 * ray.direction.length()


    def nearest_point(self, *points: Point) -> Point:
        self._update()
        r_min = sys.maxsize
        min_point = Vector.vs.init_pt
        for point in points:
            r = self.position.distance(point)
            if r == 0:
                return point

            if r < r_min:
                r_min = r
                min_point = point

        return min_point


class CubeParams(Parameters):
    def __init__(self, position: Point, limit, rotations: [Vector],
                 edges: '[BoundedPlane]'):
        super().__init__(position, rotations[0])
        self.rotation2, self.rotation3 = rotations[1:]
        self.limit = limit
        self.edges = edges

    def move(self, move_to: Point):
        self.position = self.position + move_to

        for edge in self.edges:
            edge.position = edge.position + move_to

    def scaling(self, value):
        self.rotation = self.rotation * value
        self.rotation2 = self.rotation2 * value
        self.rotation3 = self.rotation3 * value
        rotations = [self.rotation, self.rotation2, self.rotation3]
        self.limit *= value

        for i, edge in enumerate(self.edges):
            edge.parameters.scaling(value)
            if i % 2 == 0:
                edge.parameters.position = self.position + rotations[i // 2].point
                edge._update()
            else:
                edge.parameters.position = self.position - rotations[i // 2].point
                edge._update()

    def rotate(self, x_angle=0, y_angle=0, z_angle=0):
        self.rotation.rotate(x_angle, y_angle, z_angle)
        self.rotation2.rotate(x_angle, y_angle, z_angle)
        self.rotation3.rotate(x_angle, y_angle, z_angle)

        rotations = [self.rotation, self.rotation2, self.rotation3]
        for i, edge in enumerate(self.edges):
            if i % 2 == 0:
                edge.parameters.position = self.position + rotations[i // 2].point
            else:
                edge.parameters.position = self.position - rotations[i // 2].point

            edge.parameters.rotate(x_angle, y_angle, z_angle)


class Cube(Object):
    def __init__(self, position: Point, rotation: Vector, size: float):
        super().__init__(position, rotation)
        # Ограничения размеров куба (половина длина ребра)
        self.limit = size / 2
        self.rotation = rotation.normalize() * self.limit

        # Ещё два ортогональных вектора из центра куба длины self.limit
        x_dir = Vector.vs.basis[0]
        if self.rotation.point == x_dir.point \
                or self.rotation.point == -1 * x_dir.point:
            x_dir = Vector.vs.basis[1]

        self.rotation2 = (x_dir ** self.rotation).normalize() * self.limit
        self.rotation3 = (self.rotation2 ** self.rotation).normalize() * self.limit

        # Создание граней куба
        self.edges = []
        for v in self.rotation, self.rotation2, self.rotation3:
            self.edges.append(BoundedPlane(v.point + self.position, v,
                                           du=self.limit,
                                           dv=self.limit))
            self.edges.append(BoundedPlane(-1 * v.point + self.position,
                                           -1 * v, du=self.limit,
                                           dv=self.limit))

        self.parameters = CubeParams(self.position, self.limit,
                                     [self.rotation, self.rotation2, self.rotation3],
                                     self.edges)

    def _update(self):
        self.position = self.parameters.position
        self.rotation = self.parameters.rotation
        self.rotation2 = self.parameters.rotation2
        self.rotation3 = self.parameters.rotation3
        self.limit = self.parameters.limit
        self.edges = self.parameters.edges

    def __str__(self):
        self._update()
        s = ", ".join(map(str, [self.rotation, self.rotation2, self.rotation3]))
        return f'Cube({self.position}, ({s}), limit={self.limit:.4f})'

    def contains(self, point: Point, eps=1e-6) -> bool:
        self._update()
        # Радиус-вектор из центра куба к точке
        v_tmp = Vector(point - self.position)
        # Если точка является центром куба
        if v_tmp.length() == 0:
            return True

        # Проекции вектора v_tmp на направляющие вектора куба
        rot1_pr = self.rotation * v_tmp / v_tmp.length()
        rot2_pr = self.rotation2 * v_tmp / v_tmp.length()
        rot3_pr = self.rotation3 * v_tmp / v_tmp.length()
        return all(abs(abs(parameters) - 1) <= eps
                   for parameters in (rot1_pr, rot2_pr, rot3_pr))

    def intersect(self, ray, eps=1e-6) -> float or None:
        self._update()
        # Пересечения куба с лучом, имеющей направляющий вектор ray.direction
        # и начальную точку ray.position
        # int_points = list(filter(lambda x: x is not None,
        #                       [edge.intersect(ray)
        #                        for edge in self.edges]))

        int_points = []
        for edge in self.edges:
            r = edge.intersect(ray)
            if r is not None:
                int_points.append(r)

        if len(int_points):
            return min(int_points)

    def nearest_point(self, *points: Point) -> Point:
        r_min = sys.maxsize
        min_point = Vector.vs.init_pt
        r = 0
        nearest = [edge.nearest_point(*points) for edge in self.edges]
        print(*nearest)
        for i, near_pt in enumerate(nearest):
            r_begin = Vector(near_pt - self.edges[i].position)
            # Если начало вектора совпадает с центром плоскости
            if r_begin.length() == 0:
                return near_pt

            projection1 = r_begin * self.edges[i].rotation / r_begin.length()
            projection2 = r_begin * self.edges[i].u * self.edges[i].du \
                          / r_begin.length()
            projection3 = r_begin * self.edges[i].v * self.edges[i].dv \
                          / r_begin.length()
            sign = lambda x: 1 if x > 0 else -1
            if abs(projection2) <= 1 and abs(projection3) <= 1:
                r = projection1 * self.edges[i].rotation.length()
            elif abs(projection2) > 1 and abs(projection3) > 1:
                proj2 = projection2 - sign(projection2)
                proj3 = projection3 - sign(projection3)
                r = self.edges[i].rotation * -projection1 \
                    + self.edges[i].u * proj2 \
                    + self.edges[i].v * proj3 + Vector(near_pt)
                r = r.length()
            elif abs(projection2) > 1:
                proj2 = projection2 - sign(projection2)
                r = self.edges[i].rotation * -projection1 \
                    + self.edges[i].u * proj2 + Vector(near_pt)
                r = r.length()
            elif abs(projection3) > 1:
                proj3 = projection3 - sign(projection3)
                r = self.edges[i].rotation * -projection1 \
                    + self.edges[i].v * proj3 + Vector(near_pt)
                r = r.length()

            if r < r_min:
                r_min = r
                min_point = near_pt

        return min_point


class Ray:
    def __init__(self, position: Point, direction: Vector):
        self.position = position
        self.direction = direction

    def __str__(self):
        return f"Ray({self.position}, {self.direction})"

    def intersect(self, map) -> list[float]:
        return [obj.intersect(self) for obj in map]


class Camera:
    def __init__(self, position, lookDir, FOV, drawDistance):
        self.WIDTH = 92  # Ширина экрана
        self.HEIGHT = 33  # Высота экрана

        self.position = position
        self.lookDir = lookDir.normalize()
        self.FOV = (FOV / 180 * math.pi) / 2
        self.VFOV = FOV / (self.WIDTH / self.HEIGHT)
        self.drawDistance = drawDistance

        self.screen = BoundedPlane(
            self.position + self.lookDir.point / math.tan(self.FOV),
            self.lookDir, math.tan(self.FOV), math.tan(self.VFOV))


    def send_rays(self) -> list[list[Ray]]:
        # считает расстояние от камеры до пересечения луча с объектами
        rays = []

        # Создаём лучи к каждому пикселю
        for i, s in enumerate(np.linspace(
                -self.screen.dv, self.screen.dv, self.HEIGHT)):
            rays.append([])
            for t in np.linspace(-self.screen.du, self.screen.du,
                                 self.WIDTH):
                direction: Vector = Vector(self.screen.position) \
                                    + self.screen.v * s + self.screen.u * t

                direction = direction - Vector(self.position)
                direction.point.coordinates[1] /= 15 / 48
                rays[i].append(Ray(self.position, direction.normalize()))

        return rays

    def rotate(self, x_angle=0, y_angle=0, z_angle=0):
        self.lookDir.rotate(x_angle, y_angle, z_angle)
        self.screen.parameters.rotate(x_angle, y_angle, z_angle)
        self.screen.parameters.position = self.position + self.lookDir.point

        self.screen._update()


class Map:
    def __init__(self):
        self.objects = []

    def append(self, *objs) -> None:  # Метод для добавления объектов в список
        self.objects.extend(objs)  # Расширение списка объектов

    def get_object(self, index):  # Метод для получения объекта по индексу
        return self.objects[index]  # Возвращает объект из списка по индексу

    def __getitem__(self, item):  # Переопределение оператора доступа к элементам класса
        return self.get_object(item)  # Возвращает объект из списка по индексу

    def __iter__(self):  # Переопределение метода итерации
        return iter(self.objects)  # Возвращает итератор для списка объектов


class Canvas:
    def __init__(self, map: Map, camera: Camera):
        self.map = map
        self.camera = camera

    def update(self):
        rays = self.camera.send_rays()
        dist_matrix = []
        for i in range(self.camera.HEIGHT):
            dist_matrix.append([])
            for j in range(self.camera.WIDTH):
                distances = rays[i][j].intersect(self.map)
                if all(d is None or d > self.camera.drawDistance
                       for d in distances):
                    dist_matrix[i].append(None)
                else:
                    dist_matrix[i].append(
                        min(filter(lambda x: x is not None, distances)))

        return dist_matrix


symbols = " .:!/r(l1Z4H3F8$@"


class Console(Canvas):
    def draw(self):
        dist_matrix = self.update()
        output_screen = ''

        for y in range(len(dist_matrix)):
            for x in range(len(dist_matrix[y])):
                if dist_matrix[y][x] is None:
                    output_screen += symbols[0]
                    continue

                try:
                    gradient = dist_matrix[y][x] / self.camera.drawDistance \
                               * (len(symbols) - 1)

                    output_screen += symbols[len(symbols)
                                             - round(gradient) - 1]
                except (IndexError, TypeError):
                    print(len(symbols) * dist_matrix[y][x]
                          / self.camera.drawDistance, dist_matrix[y][x])
                    raise

            output_screen += '\n'

        print(output_screen)



if __name__ == "__main__":
    sphere = Sphere(Point(0, 0, 124), Vector(0, 0, 0), 25)
    sphere2 = Sphere(Point(-75, 0, 100), Vector(2, 5, -25), 25)

    map = Map()

    map.append(sphere)
    map.append(sphere2)

    camera = Camera(Point(0, 0, 0), Vector(Point(0, 0, 1)), 90, 500)

    console = Console(map, camera)

    console.draw()
