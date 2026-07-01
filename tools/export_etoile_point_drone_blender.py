#!/usr/bin/env python3
"""Export the etoile point-drone trajectory to Blender and render 50 frames."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import sionna
from PIL import Image


ROOT = Path("etoile/point_drone_tx")
CONDITION = "point_drone_tx_to_rx_rooftop_e089__cam_x58p9_y32p0_z44p0"
SOURCE_RUN = ROOT / "5GHz" / CONDITION
OUT_DIR = ROOT / "blender_export"
FRAME_DIR = OUT_DIR / "frames"
BLENDER = Path("Blender/blender-4.2.8-linux-x64/blender")
SCENE_XML = Path(sionna.rt.scene.etoile)


def as_list(arr: np.ndarray) -> list[float]:
    return [float(x) for x in np.asarray(arr, dtype=float).reshape(-1)]


def fmt_coord(value: float) -> str:
    return f"{value:.1f}".replace("-", "m").replace(".", "p")


def coord_slug(prefix: str, pos: np.ndarray) -> str:
    return f"{prefix}_x{fmt_coord(pos[0])}_y{fmt_coord(pos[1])}_z{fmt_coord(pos[2])}"


def load_payload() -> dict[str, object]:
    with np.load(SOURCE_RUN / "rt_data.npz") as data:
        tx_positions = data["tx_positions_m"].astype(np.float64)
        tx_velocity = data["tx_velocity_mps"].astype(np.float64)
        times = data["frame_time_s"].astype(np.float64)
        rx_position = data["rx_position_m"].astype(np.float64)
        camera_position = data["camera_position_m"].astype(np.float64)
        camera_look_at = data["camera_look_at_m"].astype(np.float64)

    rx_slug = coord_slug("rx", rx_position)
    cam_slug = coord_slug("cam", camera_position)
    look_slug = coord_slug("look", camera_look_at)
    return {
        "scene": "etoile",
        "source_run": str(SOURCE_RUN.resolve()),
        "scene_xml": str(SCENE_XML.resolve()),
        "out_dir": str(OUT_DIR.resolve()),
        "frame_dir": str(FRAME_DIR.resolve()),
        "blend_path": str((OUT_DIR / f"etoile_point_drone__{rx_slug}__{cam_slug}.blend").resolve()),
        "gif_path": str((OUT_DIR / f"blender_motion__{rx_slug}__{cam_slug}__{look_slug}.gif").resolve()),
        "render_resolution": [960, 640],
        "fps": 10,
        "frame_dt_s": float(np.diff(times).mean()),
        "tx_positions_m": tx_positions.tolist(),
        "tx_velocity_mps": tx_velocity.tolist(),
        "frame_time_s": times.tolist(),
        "rx_position_m": as_list(rx_position),
        "camera_position_m": as_list(camera_position),
        "camera_look_at_m": as_list(camera_look_at),
        "point_drone_note": "The Blender drone is a red visual marker animated at the RF point-Transmitter positions; it is not a physical drone mesh.",
    }


def blender_script() -> str:
    return r'''
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import bpy
from mathutils import Vector


payload_path = Path(sys.argv[sys.argv.index("--") + 1])
DATA = json.loads(payload_path.read_text(encoding="utf-8"))

COLORS = {
    "itu_glass": (0.45, 0.70, 0.95, 0.45),
    "itu_marble": (0.72, 0.72, 0.68, 1.0),
    "itu_metal": (0.30, 0.31, 0.33, 1.0),
    "itu_concrete": (0.50, 0.51, 0.49, 1.0),
    "itu_brick": (0.62, 0.24, 0.16, 1.0),
    "drone_tx": (1.0, 0.05, 0.02, 1.0),
    "rx": (0.20, 0.15, 1.0, 1.0),
    "camera_marker": (0.0, 0.75, 0.35, 1.0),
    "trajectory": (1.0, 0.18, 0.02, 1.0),
    "label": (1.0, 1.0, 0.92, 1.0),
}


def reset():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = len(DATA["tx_positions_m"])
    scene.frame_set(1)
    scene.render.fps = DATA["fps"]


def material(name, color):
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.62
        if color[3] < 1.0:
            bsdf.inputs["Alpha"].default_value = color[3]
            mat.blend_method = "BLEND"
    return mat


def parse_shapes(xml_path):
    root = ET.parse(xml_path).getroot()
    for shape in root.findall("shape"):
        if shape.attrib.get("type") != "ply":
            continue
        filename = None
        mat_id = "itu_concrete"
        for child in shape:
            if child.tag == "string" and child.attrib.get("name") == "filename":
                filename = child.attrib["value"]
            elif child.tag == "ref" and child.attrib.get("name") == "bsdf":
                mat_id = child.attrib.get("id", mat_id)
        if filename:
            yield shape.attrib.get("id", Path(filename).stem).replace("mesh-", ""), xml_path.parent / filename, mat_id


def import_scene():
    xml_path = Path(DATA["scene_xml"])
    count = 0
    for name, ply_path, mat_id in parse_shapes(xml_path):
        bpy.ops.wm.ply_import(filepath=str(ply_path))
        obj = bpy.context.object
        obj.name = name
        obj.data.name = f"{name}_mesh"
        obj.data.materials.append(material(mat_id, COLORS.get(mat_id, (0.60, 0.60, 0.58, 1.0))))
        count += 1
    return count


def add_sphere(name, loc, radius, mat_name):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=radius, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material(mat_name, COLORS[mat_name]))
    return obj


def add_label(text, loc, size):
    bpy.ops.object.text_add(location=loc)
    obj = bpy.context.object
    obj.name = f"label_{text}"
    obj.data.body = text
    obj.data.align_x = "CENTER"
    obj.data.size = size
    obj.rotation_euler = (math.radians(65), 0, 0)
    obj.data.materials.append(material("label", COLORS["label"]))
    return obj


def add_curve(name, points, mat_name, bevel_depth):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 2
    curve.bevel_depth = bevel_depth
    curve.bevel_resolution = 3
    spl = curve.splines.new("POLY")
    spl.points.add(len(points) - 1)
    for point, co in zip(spl.points, points):
        point.co = (co[0], co[1], co[2], 1.0)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material(mat_name, COLORS[mat_name]))
    return obj


def add_arrow(name, start, end):
    start_v = Vector(start)
    end_v = Vector(end)
    add_curve(f"{name}_shaft", [start, end], "trajectory", 0.18)
    direction = end_v - start_v
    if direction.length > 0:
        bpy.ops.mesh.primitive_cone_add(vertices=32, radius1=1.4, depth=3.8, location=end_v)
        cone = bpy.context.object
        cone.name = f"{name}_head"
        cone.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
        cone.data.materials.append(material("trajectory", COLORS["trajectory"]))


def setup_devices_and_path():
    tx_positions = DATA["tx_positions_m"]
    rx = DATA["rx_position_m"]
    cam_pos = DATA["camera_position_m"]
    drone = add_sphere("point_drone_tx_animated", tx_positions[0], 2.8, "drone_tx")
    for i, pos in enumerate(tx_positions, start=1):
        bpy.context.scene.frame_set(i)
        drone.location = pos
        drone.keyframe_insert(data_path="location", frame=i)
    if drone.animation_data and drone.animation_data.action:
        for fcurve in drone.animation_data.action.fcurves:
            for key in fcurve.keyframe_points:
                key.interpolation = "LINEAR"

    add_curve("drone_tx_50_sample_trajectory", tx_positions, "trajectory", 0.16)
    add_arrow("trajectory_direction_start", tx_positions[3], tx_positions[7])
    add_arrow("trajectory_direction_mid", tx_positions[24], tx_positions[28])
    add_arrow("trajectory_direction_end", tx_positions[45], tx_positions[49])
    add_sphere("fixed_rooftop_rx", rx, 2.2, "rx")
    camera_marker = add_sphere("fixed_camera_position", cam_pos, 2.0, "camera_marker")
    camera_marker.hide_render = True
    add_label("Drone Tx start", [tx_positions[0][0], tx_positions[0][1], tx_positions[0][2] + 5.5], 3.2)
    add_label("Drone Tx end", [tx_positions[-1][0], tx_positions[-1][1], tx_positions[-1][2] + 5.5], 3.2)
    add_label("Fixed rooftop Rx", [rx[0], rx[1], rx[2] + 5.0], 3.0)
    camera_label = add_label("Fixed camera", [cam_pos[0], cam_pos[1], cam_pos[2] + 5.0], 3.0)
    camera_label.hide_render = True
    return drone


def setup_camera():
    cam_pos = Vector(DATA["camera_position_m"])
    look_at = Vector(DATA["camera_look_at_m"])
    bpy.ops.object.light_add(type="AREA", location=(cam_pos.x, cam_pos.y, cam_pos.z + 65.0))
    light = bpy.context.object
    light.name = "camera_area_light"
    light.data.energy = 700
    light.data.size = 80
    bpy.ops.object.camera_add(location=cam_pos)
    cam = bpy.context.object
    cam.name = "fixed_dataset_camera"
    bpy.context.scene.camera = cam
    direction = look_at - cam.location
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.data.lens = 20
    cam.data.angle = math.radians(70)
    return cam


def configure_render():
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.render.resolution_x = DATA["render_resolution"][0]
    scene.render.resolution_y = DATA["render_resolution"][1]
    scene.world = scene.world or bpy.data.worlds.new("World")
    scene.world.color = (0.04, 0.045, 0.05)


def render_frames():
    frame_dir = Path(DATA["frame_dir"])
    frame_dir.mkdir(parents=True, exist_ok=True)
    for frame in range(1, len(DATA["tx_positions_m"]) + 1):
        bpy.context.scene.frame_set(frame)
        out = frame_dir / f"blender_frame_{frame - 1:03d}.png"
        bpy.context.scene.render.filepath = str(out)
        bpy.ops.render.render(write_still=True)
        print(f"rendered blender frame {frame:02d}/{len(DATA['tx_positions_m'])}: {out}")


reset()
mesh_count = import_scene()
setup_devices_and_path()
setup_camera()
configure_render()
bpy.ops.wm.save_as_mainfile(filepath=DATA["blend_path"])
render_frames()
summary = {
    "mesh_count": mesh_count,
    "blend_path": DATA["blend_path"],
    "frame_dir": DATA["frame_dir"],
    "n_frames": len(DATA["tx_positions_m"]),
    "point_drone_note": DATA["point_drone_note"],
}
(Path(DATA["out_dir"]) / "blender_export_runtime_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
'''


def write_gif(frame_dir: Path, gif_path: Path) -> None:
    frame_paths = sorted(frame_dir.glob("blender_frame_*.png"))
    images = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in frame_paths]
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=120, loop=0)
    for image in images:
        image.close()


def main() -> None:
    if not BLENDER.exists():
        raise FileNotFoundError(f"Blender executable not found: {BLENDER}")
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    FRAME_DIR.mkdir(parents=True)

    payload = load_payload()
    payload_path = OUT_DIR / "blender_payload.json"
    script_path = OUT_DIR / "create_etoile_point_drone_blend.py"
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    script_path.write_text(blender_script(), encoding="utf-8")

    subprocess.run(
        [str(BLENDER), "--background", "--python", str(script_path), "--", str(payload_path.resolve())],
        check=True,
    )
    write_gif(FRAME_DIR, Path(payload["gif_path"]))

    summary = {
        "scene": "etoile",
        "source_run": str(SOURCE_RUN),
        "scene_xml": str(SCENE_XML),
        "n_frames": len(payload["tx_positions_m"]),
        "frame_dt_s": payload["frame_dt_s"],
        "fps": payload["fps"],
        "blend_path": payload["blend_path"],
        "frames_dir": str(FRAME_DIR),
        "gif": payload["gif_path"],
        "point_drone_note": payload["point_drone_note"],
        "rx_position_m": payload["rx_position_m"],
        "camera_position_m": payload["camera_position_m"],
        "camera_look_at_m": payload["camera_look_at_m"],
        "first_tx_position_m": payload["tx_positions_m"][0],
        "last_tx_position_m": payload["tx_positions_m"][-1],
    }
    (OUT_DIR / "blender_export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
