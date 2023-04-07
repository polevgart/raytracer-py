import math
import numpy as np
import os
from pathlib import Path
from PIL import Image
import time

from ...geometry import *
from .. import *


CURDIR = Path(os.path.relpath(__file__)).parent


def check_image(
    scene: Scene,
    filename: str | Path,
    cam_options: CameraOptions,
    depth: float,
    timeout: float = None,
    artifact_path: str | Path | None = None,
) -> None:
    orig = Image.open(filename).convert('RGB')

    start_time = time.time()
    img = scene.render(cam_options, depth=depth, verbose=False)
    end_time = time.time()

    if artifact_path is not None:
        Path(artifact_path).parent.mkdir(exist_ok=True)
        img.save(artifact_path)

    assert orig.size == img.size

    diff = np.asarray(orig, dtype=float) - np.asarray(img, dtype=float)
    assert np.linalg.norm(diff) < 5.

    assert timeout is None or end_time - start_time < timeout

class TestRender:
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

        # spheres
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
            timeout=100,
            artifact_path=CURDIR / 'artifacts/box.png'
        )
