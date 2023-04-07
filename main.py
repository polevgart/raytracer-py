import math
import time

from pycallgraph import PyCallGraph
from pycallgraph.output import GraphvizOutput

from raytracer import CameraOptions, Material, PointLight, Scene, Sphere, Triangle, Vector


def test_triangle() -> None:
    material = Material(
        diffuse_color=Vector(0, 0, 1),
    )

    scene = Scene()
    scene.add_object(Triangle([
            Vector(-1, 0, 0),
            Vector(0, 0, -1),
            Vector(1, 0, 0),
        ],
        material=material,
    ))

    scene.add_light(PointLight(
        origin=Vector(0, 2, 0),
        intensity=Vector(1),
    ))

    cam_options = CameraOptions(
        screen_width=640,
        screen_height=480,
        look_from=Vector(0, 2, 0),
        look_to=Vector(0, 0, 0),
    )

    start_ts = time.time()
    img = scene.render(cam_options, depth=1)
    end_ts = time.time()

    print("Elapsed time:", end_ts - start_ts)
    img.save("image.png")


def test_invisible_triangle() -> None:
    material = Material(
        diffuse_color=Vector(0, 0, 1),
    )

    scene = Scene()
    scene.add_object(Triangle([
            Vector(-1, 0, 0),
            Vector(0, 0, -1),
            Vector(1, 0, 0),
        ],
        material=material,
    ))

    scene.add_light(PointLight(
        origin=Vector(0, 2, 0),
        intensity=Vector(1),
    ))

    cam_options = CameraOptions(
        screen_width=640,
        screen_height=480,
        look_from=Vector(0, -2, 0),
        look_to=Vector(0, 0, 0),
    )

    start_ts = time.time()
    img = scene.render(cam_options, depth=1)
    end_ts = time.time()

    print("Elapsed time:", end_ts - start_ts)
    img.save("image.png")


def test_box() -> None:
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
    wall_behind_mtl = Material(
        diffuse_color=Vector(0.2, 0.7, 0.8),
        specular_color=Vector(0, 0, 0),
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

    scene.add_object(Triangle([
            Vector(-1.02, 1.59, 0.99),
            Vector(-1.01, 0, 0.99),
            Vector(1, 0, 0.99),
        ],
        material=wall_behind_mtl,
    ))
    scene.add_object(Triangle([
            Vector(1, 0, 0.99),
            Vector(1, 1.59, 0.99),
            Vector(-1.02, 1.59, 0.99),
        ],
        material=wall_behind_mtl,
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

    start_ts = time.time()
    img = scene.render(cam_options, depth=4)
    end_ts = time.time()

    print("Elapsed time:", end_ts - start_ts)
    img.save("image.png")


def test_sphere() -> None:
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

    scene = Scene()
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

    scene.add_light(PointLight(
        origin=Vector(-0.2, 0, 0),
        intensity=Vector(0.5),
    ))

    cam_options = CameraOptions(
        screen_width=640,
        screen_height=480,
    )

    start_ts = time.time()
    img = scene.render(cam_options, depth=1)
    end_ts = time.time()

    print("Elapsed time:", end_ts - start_ts)
    img.save("image.png")


def main() -> None:
    with PyCallGraph(output=GraphvizOutput()):
        # test_sphere()
        # test_box()
        test_triangle()
        # test_invisible_triangle()


if __name__ == "__main__":
    main()
