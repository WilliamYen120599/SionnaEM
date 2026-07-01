#!/usr/bin/env python3
"""Create Sionna-rendered mobility previews and path geometry for Blender."""

from __future__ import annotations

import json
from pathlib import Path

import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")

import numpy as np
from PIL import Image
import sionna
from sionna.rt import Camera, PathSolver, PlanarArray, Receiver, Transmitter, load_scene


OUT = Path("Blender Preview Images/mobility/blender")
FS = 122.88e6


def array(pattern="iso"):
    return PlanarArray(num_rows=1, num_cols=1, pattern=pattern, polarization="V")


def setup_reflector():
    scene = load_scene(sionna.rt.scene.simple_reflector, merge_shapes=False)
    scene.frequency = 3.5e9
    scene.get("reflector").velocity = [0, 0, -20]
    scene.tx_array = array()
    scene.rx_array = array()
    scene.add(Transmitter("tx", [-25, 0.1, 50], power_dbm=0.0))
    scene.add(Receiver("rx", [25, 0.1, 50]))
    return scene


def setup_street():
    scene = load_scene(sionna.rt.scene.simple_street_canyon_with_cars, merge_shapes=False)
    scene.frequency = 3.5e9
    scene.tx_array = array("tr38901")
    scene.rx_array = array("tr38901")
    scene.add(Transmitter("tx", position=[22.7, 5.6, 0.75], orientation=[np.pi, 0, 0], power_dbm=0.0))
    scene.add(Receiver("rx", position=[-27.8, -4.9, 0.75]))
    v = np.array([10.0, 0.0, 0.0])
    for j in range(1, 6):
        scene.get(f"car_{j}").velocity = -v
    for j in range(6, 9):
        scene.get(f"car_{j}").velocity = v
    scene.get("tx").velocity = -v
    scene.get("rx").velocity = v
    return scene


def render(scene, camera, path: Path, paths=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    bitmap = scene.render(
        camera=camera,
        paths=paths,
        num_samples=256,
        resolution=(900, 650),
        return_bitmap=True,
    )
    arr = np.asarray(bitmap)
    arr = np.clip(arr[..., :3], 0.0, 1.0)
    Image.fromarray((arr * 255.0).astype(np.uint8)).save(path)


def path_lines(scene, paths) -> list[dict[str, object]]:
    sources = np.asarray(paths.sources.numpy(), dtype=float).T
    targets = np.asarray(paths.targets.numpy(), dtype=float).T
    vertices = np.asarray(paths.vertices.numpy(), dtype=float)
    valid = np.asarray(paths.valid.numpy(), dtype=bool)
    interactions = np.asarray(paths.interactions.numpy(), dtype=int)
    doppler = np.asarray(paths.doppler.numpy(), dtype=float)
    tau = np.asarray(paths.tau.numpy(), dtype=float)
    lines = []
    for rx in range(valid.shape[0]):
        for tx in range(valid.shape[1]):
            for p in range(valid.shape[2]):
                if not valid[rx, tx, p]:
                    continue
                pts = [sources[tx].tolist()]
                path_interactions = interactions[rx, tx, :, p]
                for depth_idx, inter in enumerate(path_interactions):
                    if inter != 0:
                        pts.append(vertices[rx, tx, depth_idx, p].tolist())
                pts.append(targets[rx].tolist())
                lines.append({
                    "points": pts,
                    "doppler_hz": float(doppler[rx, tx, p]),
                    "delay_s": float(tau[rx, tx, p]),
                    "interaction": [int(x) for x in path_interactions.tolist()],
                })
    return lines


def save_gif(frame_paths: list[Path], out_path: Path):
    images = [Image.open(p).convert("P", palette=Image.Palette.ADAPTIVE) for p in frame_paths]
    images[0].save(out_path, save_all=True, append_images=images[1:], duration=450, loop=0)


def reflector_outputs():
    out = OUT / "simple_reflector"
    scene = setup_reflector()
    paths = PathSolver()(scene=scene, max_depth=1)
    cam = Camera(position=[0, 100, 50], look_at=[0, 0, 30])
    render(scene, cam, out / "sionna_scene.png")
    render(scene, cam, out / "sionna_paths.png", paths=paths)

    frame_paths = []
    step = np.array([0.0, 0.0, -4.0 / 6.0])
    for i in range(7):
        p = out / f"frame_{i:02d}.png"
        render(scene, cam, p)
        frame_paths.append(p)
        scene.get("reflector").position += step
    save_gif(frame_paths, out / "reflector_motion.gif")

    data = {
        "scene_xml": str(sionna.rt.scene.simple_reflector),
        "tx": [-25, 0.1, 50],
        "rx": [25, 0.1, 50],
        "moving": {"reflector": {"velocity": [0, 0, -20], "frame_displacement": [0, 0, -4]}},
        "paths": path_lines(scene, paths),
    }
    (out / "blender_overlay.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def street_outputs():
    out = OUT / "street_canyon"
    scene = setup_street()
    paths = PathSolver()(scene=scene, max_depth=3, refraction=False, diffraction=False)
    cam = Camera(position=[50, 0, 130], look_at=[10, 0, 0])
    render(scene, cam, out / "sionna_scene.png")
    render(scene, cam, out / "sionna_paths.png", paths=paths)

    frame_paths = []
    displacement = np.array([2.0, 0.0, 0.0])
    for i in range(7):
        p = out / f"frame_{i:02d}.png"
        render(scene, cam, p)
        frame_paths.append(p)
        scene.get("tx").position -= displacement
        scene.get("rx").position += displacement
        for j in range(1, 6):
            scene.get(f"car_{j}").position -= displacement
        for j in range(6, 9):
            scene.get(f"car_{j}").position += displacement
    save_gif(frame_paths, out / "street_motion.gif")

    moving = {f"car_{j}": {"velocity": [-10, 0, 0], "frame_displacement": [-8, 0, 0]} for j in range(1, 6)}
    moving.update({f"car_{j}": {"velocity": [10, 0, 0], "frame_displacement": [8, 0, 0]} for j in range(6, 9)})
    moving["tx"] = {"velocity": [-10, 0, 0], "frame_displacement": [-8, 0, 0]}
    moving["rx"] = {"velocity": [10, 0, 0], "frame_displacement": [8, 0, 0]}
    data = {
        "scene_xml": str(sionna.rt.scene.simple_street_canyon_with_cars),
        "tx": [22.7, 5.6, 0.75],
        "rx": [-27.8, -4.9, 0.75],
        "moving": moving,
        "paths": path_lines(scene, paths),
    }
    (out / "blender_overlay.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    reflector_outputs()
    street_outputs()
    print(json.dumps({"out_dir": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
