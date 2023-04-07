import time
import geometry
from scene import Scene


def main():    
    red = geometry.Material(color=geometry.Color(230, 0, 0))
    blue = geometry.Material(color=geometry.Color(0, 0, 230))
    green = geometry.Material(color=geometry.Color(0, 230, 0))
    yellow = geometry.Material(color=geometry.Color(230, 230, 0))
    orange = geometry.Material(color=geometry.Color(230, 140, 0))

    scene = Scene(depth=2)
    scene.add_object(geometry.Sphere(center=geometry.Vector(1.5, -1, 6), radius=1, material=red))
    pixels = 700, 700
    start_ts = time.time()
    from pycallgraph import PyCallGraph
    from pycallgraph.output import GraphvizOutput

    with PyCallGraph(output=GraphvizOutput()):
        scene.render(pixels, (1, 1))
        
    end_ts = time.time()

    print("Elapsed time:", end_ts - start_ts)
    # img.save("kekkok.bmp")
    # img.save("kekkok.jpg")
    # img.save("kekkok.png")
    # img.show()


if __name__ == "__main__":
    main()