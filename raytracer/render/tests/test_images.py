import math
import numpy as np
import os
from pathlib import Path
from PIL import Image
import time

from ...geometry import Material, Sphere, Triangle, Vector
from .. import CameraOptions, PointLight, Scene


CURDIR = Path(os.path.relpath(__file__)).parent


def check_image(
    scene: Scene,
    filename: str | Path,
    cam_options: CameraOptions,
    depth: float,
    parallel: bool = True,
    timeout: float | None = None,
    artifact_path: str | Path | None = None,
) -> None:
    orig = Image.open(filename).convert('RGB')

    start_time = time.time()
    img = scene.render(cam_options, depth=depth, parallel=parallel, verbose=False)
    end_time = time.time()

    if artifact_path is not None:
        Path(artifact_path).parent.mkdir(exist_ok=True)
        img.save(artifact_path)

    assert orig.size == img.size

    diff = np.asarray(orig, dtype=float) - np.asarray(img, dtype=float)
    similarity = (np.abs(diff).sum(axis=-1) < 2).mean()
    assert similarity > 0.99

    assert timeout is None or end_time - start_time < timeout


class TestRender:
    def test_sphere(self):
        scene = Scene()

        # materials
        ambient = Material(
            ambient_color=Vector(0.5, 0, 0),
        )
        diffuse = Material(
            diffuse_color=Vector(0.5, 0, 0),
        )
        specular = Material(
            ambient_color=Vector(0.05, 0, 0),
            specular_color=Vector(0.5, 0, 0),
            specular_exponent=500,
        )

        # objects
        scene.add_object(Sphere(
            center=Vector(-0.35, 0, -0.5),
            radius=0.15,
            material=ambient,
        ))
        scene.add_object(Sphere(
            center=Vector(0, 0, -0.5),
            radius=0.15,
            material=diffuse,
        ))
        scene.add_object(Sphere(
            center=Vector(0.4, 0, -0.5),
            radius=0.15,
            material=specular,
        ))

        # light
        scene.add_light(PointLight(
            origin=Vector(-0.2, 0, 0),
            intensity=Vector(0.5),
        ))

        # render options
        cam_options = CameraOptions(
            screen_width=640,
            screen_height=480,
        )

        check_image(
            scene,
            CURDIR / 'images/spheres.png',
            cam_options,
            depth=1,
            parallel=True,
            timeout=15,
            artifact_path=CURDIR / 'artifacts/spheres.png'
        )

    def test_triangle(self):
        scene = Scene()

        # materials
        material = Material(
            diffuse_color=Vector(0, 0, 1),
        )

        # objects
        scene.add_object(Triangle([
                Vector(-1, 0, 0),
                Vector(0, 0, -1),
                Vector(1, 0, 0),
            ],
            material=material,
        ))

        # light
        scene.add_light(PointLight(
            origin=Vector(0, 2, 0),
            intensity=Vector(1),
        ))

        # render options
        cam_options = CameraOptions(
            screen_width=640,
            screen_height=480,
            look_from=Vector(0, 2, 0),
            look_to=Vector(0, 0, 0),
        )

        check_image(
            scene,
            CURDIR / 'images/triangle.png',
            cam_options,
            depth=1,
            parallel=True,
            timeout=15,
            artifact_path=CURDIR / 'artifacts/triangle.png'
        )

    def test_invisible_triangle(self):
        scene = Scene()

        # materials
        material = Material(
            diffuse_color=Vector(0, 0, 1),
        )

        # objects
        scene.add_object(Triangle([
                Vector(-1, 0, 0),
                Vector(0, 0, -1),
                Vector(1, 0, 0),
            ],
            material=material,
        ))

        # light
        scene.add_light(PointLight(
            origin=Vector(0, 2, 0),
            intensity=Vector(1),
        ))

        # render options
        cam_options = CameraOptions(
            screen_width=640,
            screen_height=480,
            look_from=Vector(0, -2, 0),
            look_to=Vector(0, 0, 0),
        )

        check_image(
            scene,
            CURDIR / 'images/invisible_triangle.png',
            cam_options,
            depth=1,
            parallel=True,
            timeout=15,
            artifact_path=CURDIR / 'artifacts/invisible_triangle.png'
        )

    def test_box(self):
        scene = Scene()

        # materials
        left_sphere_mtl = Material(
            albedo=Vector(0, 0.8, 0),
            specular_exponent=1024,
        )
        right_sphere_mtl = Material(
            albedo=Vector(0, 0.3, 0.7),
            specular_exponent=1024,
            refraction_index=1.8,
        )

        floor_mtl = Material(
            diffuse_color=Vector(0.725, 0.91, 0.88),
            specular_color=Vector(1.5, 1.5, 1.5),
            albedo=Vector(0.5, 0, 0),
            specular_exponent=10,
            refraction_index=1.5,
        )
        ceiling_mtl = Material(
            diffuse_color=Vector(0.2, 0.2, 0.5),
            specular_color=Vector(0.3, 0.3, 0.3),
            specular_exponent=1024,
            refraction_index=1.5,
        )

        back_wall_mtl = Material(
            diffuse_color=Vector(0.725, 0.91, 0.88),
            specular_color=Vector(1.5, 1.5, 1.5),
            albedo=Vector(0.5, 0, 0),
            specular_exponent=10,
            refraction_index=1.5,
        )
        right_wall_mtl = Material(
            diffuse_color=Vector(0.161, 0.133, 0.427),
            albedo=Vector(0.8, 0, 0),
            specular_exponent=10,
        )
        left_wall_mtl = Material(
            diffuse_color=Vector(0.33, 0.065, 0.05),
            specular_color=Vector(0, 0, 0),
            albedo=Vector(0.8, 0, 0),
            specular_exponent=10,
        )

        # objects
        scene.add_object(Sphere(
            center=Vector(-0.4, 0.3, -0.4),
            radius=0.3,
            material=left_sphere_mtl,
        ))
        scene.add_object(Sphere(
            center=Vector(0.3, 0.3, 0),
            radius=0.3,
            material=right_sphere_mtl,
        ))

        # walls
        scene.add_object(Triangle([
                Vector(1, 0, -1.04),
                Vector(-0.99, 0, -1.04),
                Vector(-1.01, 0, 0.99),
            ],
            material=floor_mtl,
        ))
        scene.add_object(Triangle([
                Vector(-1.01, 0, 0.99),
                Vector(1, 0, 0.99),
                Vector(1, 0, -1.04),
            ],
            material=floor_mtl,
        ))

        scene.add_object(Triangle([
                Vector(1, 1.59, -1.04),
                Vector(1, 1.59, 0.99),
                Vector(-1.02, 1.59, 0.99),
            ],
            material=ceiling_mtl,
        ))
        scene.add_object(Triangle([
                Vector(-1.02, 1.59, 0.99),
                Vector(-1.02, 1.59, -1.04),
                Vector(1, 1.59, -1.04),
            ],
            material=ceiling_mtl,
        ))

        scene.add_object(Triangle([
                Vector(1, 1.59, -1.04),
                Vector(-1.02, 1.59, -1.04),
                Vector(-0.99, 0, -1.04),
            ],
            material=back_wall_mtl,
        ))
        scene.add_object(Triangle([
                Vector(-0.99, 0, -1.04),
                Vector(1, 0, -1.04),
                Vector(1, 1.59, -1.04),
            ],
            material=back_wall_mtl,
        ))

        scene.add_object(Triangle([
                Vector(1, 1.59, 0.99),
                Vector(1, 1.59, -1.04),
                Vector(1, 0, -1.04),
            ],
            material=right_wall_mtl,
        ))
        scene.add_object(Triangle([
                Vector(1, 0, -1.04),
                Vector(1, 0, 0.99),
                Vector(1, 1.59, 0.99),
            ],
            material=right_wall_mtl,
        ))

        scene.add_object(Triangle([
                Vector(-1.02, 1.59, -1.04),
                Vector(-1.02, 1.59, 0.99),
                Vector(-1.01, 0, 0.99),
            ],
            material=left_wall_mtl,
        ))
        scene.add_object(Triangle([
                Vector(-1.01, 0, 0.99),
                Vector(-0.99, 0, -1.04),
                Vector(-1.02, 1.59, -1.04),
            ],
            material=left_wall_mtl,
        ))

        # light
        scene.add_light(PointLight(
            origin=Vector(0, 1.5899, 0),
            intensity=Vector(1, 1, 1),
        ))
        scene.add_light(PointLight(
            origin=Vector(0, 0.7, 1.98),
            intensity=Vector(0.5, 0.5, 0.5),
        ))

        # render options
        cam_options = CameraOptions(
            screen_width=640,
            screen_height=480,
            fov=math.pi / 3,
            look_from=Vector(0, 0.7, 1.75),
            look_to=Vector(0, 0.7, 0),
        )

        check_image(
            scene,
            CURDIR / 'images/box.png',
            cam_options,
            depth=4,
            parallel=True,
            timeout=30,
            artifact_path=CURDIR / 'artifacts/box.png'
        )

    def test_mirrors(self):
        scene = Scene()

        # materials
        back_mtl = Material(
            specular_color=Vector(0.95),
            albedo=Vector(10, 0.5, 0),
            specular_exponent=1024,
            refraction_index=1,
        )
        front_mtl = Material(
            specular_color=Vector(0.95),
            albedo=Vector(10, 0.5, 0),
            specular_exponent=1024,
            refraction_index=1,
        )
        left_mtl = Material(
            specular_color=Vector(0.95),
            albedo=Vector(10, 0.5, 0),
            specular_exponent=1024,
            refraction_index=1,
        )
        right_mtl = Material(
            specular_color=Vector(0.95),
            albedo=Vector(10, 0.5, 0),
            specular_exponent=1024,
            refraction_index=1,
        )
        floor_mtl = Material(
            diffuse_color=Vector(0.1),
        )
        ceil_mtl = Material(
            diffuse_color=Vector(0.4),
        )

        sphere_mtl = Material(
            ambient_color=Vector(0.01, 0.03, 0.03),
            diffuse_color=Vector(0.1, 0.3, 0.3),
        )

        # objects
        vertices = [
            Vector(),
            Vector(0, 0, 0),
            Vector(0, 0, -3),
            Vector(3, 0, -3),
            Vector(3, 0, 0),
            Vector(0, 3, 0),
            Vector(0, 3, -3),
            Vector(3, 3, -3),
            Vector(3, 3, 0),
        ]

        scene.add_object(Triangle([
                vertices[1],
                vertices[5],
                vertices[8],
            ],
            material=back_mtl,
        ))
        scene.add_object(Triangle([
                vertices[1],
                vertices[8],
                vertices[4],
            ],
            material=back_mtl,
        ))

        scene.add_object(Triangle([
                vertices[1],
                vertices[5],
                vertices[6],
            ],
            material=left_mtl,
        ))
        scene.add_object(Triangle([
                vertices[1],
                vertices[6],
                vertices[2],
            ],
            material=left_mtl,
        ))

        scene.add_object(Triangle([
                vertices[4],
                vertices[3],
                vertices[7],
            ],
            material=right_mtl,
        ))
        scene.add_object(Triangle([
                vertices[4],
                vertices[7],
                vertices[8],
            ],
            material=right_mtl,
        ))

        scene.add_object(Triangle([
                vertices[2],
                vertices[3],
                vertices[7],
            ],
            material=front_mtl,
        ))
        scene.add_object(Triangle([
                vertices[2],
                vertices[7],
                vertices[6],
            ],
            material=front_mtl,
        ))

        scene.add_object(Triangle([
                vertices[1],
                vertices[2],
                vertices[3],
            ],
            material=floor_mtl,
        ))
        scene.add_object(Triangle([
                vertices[1],
                vertices[3],
                vertices[4],
            ],
            material=floor_mtl,
        ))

        scene.add_object(Triangle([
                vertices[5],
                vertices[8],
                vertices[7],
            ],
            material=ceil_mtl,
        ))
        scene.add_object(Triangle([
                vertices[5],
                vertices[7],
                vertices[6],
            ],
            material=ceil_mtl,
        ))

        scene.add_object(Sphere(
            center=Vector(1, 0.5, -2),
            radius=0.5,
            material=sphere_mtl,
        ))

        # light
        scene.add_light(PointLight(
            origin=Vector(2.8, 2.8, -2.8),
            intensity=Vector(1),
        ))

        # render options
        cam_options = CameraOptions(
            screen_width=800,
            screen_height=600,
            look_from=Vector(2, 1.5, -0.1),
            look_to=Vector(1, 1.2, -2.8),
        )

        check_image(
            scene,
            CURDIR / 'images/mirrors.png',
            cam_options,
            depth=9,
            parallel=True,
            timeout=90,
            artifact_path=CURDIR / 'artifacts/mirrors.png'
        )
