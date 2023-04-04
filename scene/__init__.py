import attr
import geometry
import tqdm
from PIL import Image


@attr.s(slots=True, kw_only=True)
class AmbientLight:
    intensity: geometry.Vector = attr.ib()


@attr.s(slots=True, kw_only=True)
class PointLight:
    intensity: geometry.Vector = attr.ib()
    origin: geometry.Vector = attr.ib()


@attr.s(slots=True, kw_only=True)
class Scene:
    objects: list[geometry.BaseObject] = attr.ib(factory=list, init=False)
    lights: list[PointLight | AmbientLight] = attr.ib(init=False)
    # materials: list[Material]

    depth: int = attr.ib(init=True)

    @lights.default
    def _(self):
        return [
            AmbientLight(intensity=geometry.Vector(2, 1, 0) * 0.1),
            PointLight(origin=geometry.Vector(-6, 2, 0), intensity=geometry.Vector(0.8, 0.8, 0.8)),
        ]
    
    def add_object(self, obj):
        self.objects.append(obj)

    def ray_cast(self, ray: geometry.Ray) -> tuple[geometry.Intersection, geometry.BaseObject]:
        """обходит все объекты в сцене и находит пересечение с лучем"""
        intersected_obj = None
        best_intersection: (geometry.Intersection | None) = None
        for obj in self.objects:
            inersection = obj.intersect(ray)
            if not inersection:
                continue

            if not best_intersection or best_intersection.distance < inersection.distance:
                best_intersection = inersection
                intersected_obj = obj

        return best_intersection, intersected_obj
    
    def compute_lighting(self):
        intensity = geometry.Vector(0, 0, 0)
        for light in self.lights:
            if isinstance(light, AmbientLight):
                intensity += light.intensity
                continue

            # TODO: Point & Direction
    
    def ray_trace(self, ray: geometry.Ray):
        if (self.depth <= 0):
            return geometry.Color(0, 0, 0)
        
        intersection, obj = self.ray_cast(ray)
        if obj:
            return obj.material.color
        else:
            return geometry.Color(0, 0, 0)
        # material = obj.material
        # local_color = geometry.Vector(0, 0, 0)

    def render(self, pixels, size):
        w, h = pixels
        ws, hs = size
        dw = ws / w
        dh = hs / h
        img = Image.new("RGB", pixels)
        start_point = geometry.Vector(0, 0, 0)
        for j in tqdm.tqdm(range(h)):
            for i in range(w):
                ray = geometry.Ray(origin=start_point, direction=geometry.Vector(-i * dw + ws / 2.0, j * dh - hs / 2.0, 1))
                pixel = self.ray_trace(ray)
                img.putpixel((i, j), pixel.to_tuple())

        return img

