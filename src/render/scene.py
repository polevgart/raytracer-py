import attr
import math
import numpy as np
import tqdm
from PIL import Image

from ..geometry import BaseObject, Color, Intersection, Ray, Vector, reflect, refract
from .matrix import look_at, point_matrix_multiply, vector_matrix_multiply


EPS = 1e-8


@attr.s(slots=True, kw_only=True)
class PointLight:
    origin: Vector = attr.ib()
    intensity: Vector = attr.ib()


@attr.s(slots=True, kw_only=True)
class CameraOptions:
    screen_width: float = attr.ib()
    screen_height: float = attr.ib()
    fov: float = attr.ib(default=math.pi / 2)
    look_from: Vector = attr.ib(factory=Vector)
    look_to: Vector = attr.ib()

    @look_to.default
    def _(self) -> Vector:
        return Vector(0, 0, -1)


@attr.s(slots=True, kw_only=True)
class Scene:
    objects: list[BaseObject] = attr.ib(factory=list, init=False)
    lights: list[PointLight] = attr.ib(factory=list, init=False)

    def add_object(self, obj: BaseObject) -> None:
        self.objects.append(obj)

    def add_light(self, light: PointLight) -> None:
        self.lights.append(light)

    def find_closest_intersection(self, ray: Ray) -> tuple[Intersection, BaseObject]:
        intersected_obj = None
        best_intersection = None
        for obj in self.objects:
            intersection = obj.intersect(ray)
            if intersection is None:
                continue
            if best_intersection is None or best_intersection.distance > intersection.distance:
                # recalculate normal for triangle
                best_intersection = intersection
                intersected_obj = obj

        return best_intersection, intersected_obj

    def is_point_illuminated(self, point: Vector, light_dir: Vector) -> bool:
        ray = Ray(origin=point, direction=light_dir)
        light_dist = light_dir.length
        for obj in self.objects:
            intersection = obj.intersect(ray)
            if intersection is not None and intersection.distance < light_dist:
                return False
        return True

    def get_intensity(self,
                      ray: Ray,
                      intersection: Intersection,
                      obj: BaseObject,
                      *,
                      inside: bool = False,
                      ) -> Vector:
        material = obj.material

        # ambient shading
        intensity = material.ambient_color

        if not inside and material.albedo[0] > EPS:
            pos = intersection.position
            norm = intersection.normal

            view_dir = -ray.direction
            new_pos = pos + EPS * norm

            diffuse_total = Color()
            specular_total = Color()

            for light in self.lights:
                light_dir = light.origin - new_pos
                if not self.is_point_illuminated(new_pos, light_dir):
                    continue
                light_dir.normalize()
                print('Illuminated!')

                # diffuse shading
                diffuse_total += light.intensity * max(0, norm.dot(light_dir))

                # specular shading
                specular_dot = view_dir.dot(reflect(-light_dir, norm))
                specular_total += light.intensity * max(0, specular_dot) ** material.specular_exponent
                print(light.intensity, diffuse_total, specular_total, material.albedo)

            intensity += (material.albedo[0] * material.diffuse_color) * diffuse_total
            intensity += (material.albedo[0] * material.specular_color) * specular_total

        return intensity

    def trace_ray(self, ray: Ray, *, depth: float, inside: bool = False) -> Vector:
        intersection, obj = self.find_closest_intersection(ray)
        if intersection is None:
            return None

        intensity = self.get_intensity(ray, intersection, obj, inside=inside)
        if depth <= 1:
            return intensity

        material = obj.material

        # reflection
        if not inside and material.albedo[1] > EPS:
            print('Reflect')
            new_dir = reflect(ray.direction, intersection.normal)
            new_ray = Ray(origin=intersecion.position + EPS * intersection.normal, direction=new_dir)
            reflected = self.trace_ray(new_ray, depth=depth - 1, inside=False)
            if reflected is not None:
                intensity += material.albedo[1] * reflected

        # refraction
        if inside or material.albedo[2] > EPS:
            print('Refract')
            eta = material.refraction_index
            if not inside:
                eta = 1 / eta
            new_dir = refract(ray.direction, intersection.normal, eta)
            if new_dir is not None:
                new_ray = Ray(origin=intersecion.position - EPS * intersection.normal, direction=new_dir)
                refracted = self.trace_ray(new_ray, depth=depth - 1, inside=inside ^ object.has_volume())
                if refracted is not None:
                    intensity += (1 if inside else material.albedo[2]) * refracted

        return intensity

    def tone_mapping(self, pixels):
        scale = pixels.max()
        for j, row in tqdm.tqdm(enumerate(pixels), desc='Tone mapping', total=len(pixels)):
            for i, pixel in enumerate(row):
                pixels[j, i] = pixel * (1 + pixel / (scale ** 2)) / (1 + pixel)

    def gamma_correction(self, pixels, *, gamma: float = 2.2):
        pixels **= 1 / gamma

    def postprocess(self, pixels, *, gamma: float = 2.2):
        self.tone_mapping(pixels)
        self.gamma_correction(pixels, gamma=gamma)

    def render(self,
               cam_options: CameraOptions,
               background_color: Color = Color(0, 0, 0),
               depth: float = 3,
               ) -> Image.Image:
        width = cam_options.screen_width
        height = cam_options.screen_height

        scale = math.tan(cam_options.fov / 2)
        aspect_ratio = width / height
        cam_to_world = look_at(cam_options.look_from, cam_options.look_to)
        origin = point_matrix_multiply(cam_to_world, Vector())

        pixels = np.empty((height, width, 3), dtype=float)
        for j in tqdm.tqdm(range(height), desc='Ray tracing'):
            for i in range(width):
                x = (2 * (i + 0.5) / width - 1) * aspect_ratio * scale
                y = (1 - 2 * (j + 0.5) / height) * scale
                direction = vector_matrix_multiply(cam_to_world, Vector(x, y, -1))
                ray = Ray(origin=origin, direction=direction)

                pixel = self.trace_ray(ray, depth=depth)
                if pixel is None:
                    pixel = background_color
                pixels[j, i] = pixel.to_array()

        self.postprocess(pixels)

        img = Image.fromarray(np.uint8(255 * pixels))
        return img
