import time

from src import Color, Material, Sphere, Vector
from src import CameraOptions, PointLight, Scene


def main() -> None:
    red = Material(
        ambient_color=Color(0.5, 0, 0),
        diffuse_color=Color(0.5, 0, 0),
        specular_color=Color(0.5, 0, 0),
        specular_exponent=500,
        # refraction_index=1,
    )

    scene = Scene()
    scene.add_object(Sphere(
        center=Vector(-0.35, 0, -0.5),
        radius=0.15,
        material=red,
    ))
    scene.add_light(PointLight(
        origin=Vector(-0.2, 0, 0),
        intensity=Vector(0.5, 0.5, 0.5),
    ))

    cam_options = CameraOptions(
        screen_width=700,
        screen_height=700,
    )

    start_ts = time.time()
    img = scene.render(cam_options, depth=1)
    end_ts = time.time()

    print("Elapsed time:", end_ts - start_ts)
    img.save("image.png")


if __name__ == "__main__":
    main()
