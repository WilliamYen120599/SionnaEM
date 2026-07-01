#!/usr/bin/env python3
"""Build Blender .blend files for rooftop-BS street-canyon cases."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path("/workspace/ycq/sionna")
OUT = ROOT / "Blender Preview Images/mobility/rooftop_bs"


COLORS = {
    "glass": (0.45, 0.70, 0.95, 0.55),
    "wood": (0.55, 0.32, 0.16, 1.0),
    "marble": (0.72, 0.72, 0.70, 1.0),
    "brick": (0.65, 0.18, 0.12, 1.0),
    "concrete": (0.50, 0.52, 0.52, 1.0),
    "metal": (0.25, 0.27, 0.30, 1.0),
    "bs": (0.05, 0.30, 1.0, 1.0),
    "rx_car": (1.0, 0.12, 0.08, 1.0),
    "rx_roof": (0.72, 0.20, 1.0, 1.0),
    "path_los": (0.05, 0.62, 1.0, 1.0),
    "path_ref": (1.0, 0.58, 0.08, 1.0),
    "motion": (0.1, 0.86, 0.30, 1.0),
    "label": (0.98, 0.98, 0.92, 1.0),
}


def reset():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = 80


def material(name: str, color):
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = 0.55
        if color[3] < 1:
            bsdf.inputs["Alpha"].default_value = color[3]
            mat.blend_method = "BLEND"
    return mat


def parse_xml(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    for shape in root.findall("shape"):
        if shape.attrib.get("type") != "ply":
            continue
        filename = None
        mat_id = "concrete"
        for child in shape:
            if child.tag == "string" and child.attrib.get("name") == "filename":
                filename = child.attrib["value"]
            if child.tag == "ref" and child.attrib.get("name") == "bsdf":
                mat_id = child.attrib["id"]
        if filename:
            yield shape.attrib.get("id", Path(filename).stem).replace("mesh-", ""), xml_path.parent / filename, mat_id


def import_meshes(xml_path: Path):
    objs = {}
    for name, ply_path, mat_id in parse_xml(xml_path):
        bpy.ops.wm.ply_import(filepath=str(ply_path))
        obj = bpy.context.object
        obj.name = name
        obj.data.name = f"{name}_mesh"
        obj.data.materials.append(material(mat_id, COLORS.get(mat_id, COLORS["concrete"])))
        objs[name] = obj
    return objs


def add_sphere(name: str, loc, radius: float, mat_name: str):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=radius, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material(mat_name, COLORS[mat_name]))
    return obj


def add_label(text: str, loc, size: float = 1.8):
    bpy.ops.object.text_add(location=loc, rotation=(math.radians(65), 0, 0))
    obj = bpy.context.object
    obj.name = f"label_{text}"
    obj.data.body = text
    obj.data.align_x = "CENTER"
    obj.data.size = size
    obj.data.materials.append(material("label", COLORS["label"]))
    return obj


def add_curve(name: str, points, mat_name: str, bevel: float):
    curve = bpy.data.curves.new(name, "CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = bevel
    curve.bevel_resolution = 3
    spl = curve.splines.new("POLY")
    spl.points.add(len(points) - 1)
    for p, co in zip(spl.points, points):
        p.co = (co[0], co[1], co[2], 1.0)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material(mat_name, COLORS[mat_name]))
    return obj


def add_arrow(name: str, start, delta):
    end = Vector(start) + Vector(delta)
    add_curve(f"motion_{name}", [start, end], "motion", 0.055)
    d = end - Vector(start)
    if d.length > 0:
        bpy.ops.mesh.primitive_cone_add(vertices=24, radius1=0.35, depth=0.9, location=end)
        cone = bpy.context.object
        cone.name = f"motion_head_{name}"
        cone.rotation_euler = d.to_track_quat("Z", "Y").to_euler()
        cone.data.materials.append(material("motion", COLORS["motion"]))


def keyframe(obj, delta):
    obj.keyframe_insert(data_path="location", frame=1)
    obj.location = obj.location + Vector(delta)
    obj.keyframe_insert(data_path="location", frame=80)


def add_paths(paths):
    for i, path in enumerate(paths):
        mat_name = "path_los" if len(path["points"]) == 2 else "path_ref"
        add_curve(f"path_{i:03d}_{path['tx']}_to_{path['rx']}", path["points"], mat_name, 0.045)


def setup_camera(name: str, loc, target, scale):
    bpy.ops.object.camera_add(location=loc)
    cam = bpy.context.object
    cam.name = name
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = scale
    direction = Vector(target) - Vector(loc)
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return cam


def render(path: Path):
    bpy.context.scene.render.engine = "BLENDER_WORKBENCH"
    bpy.context.scene.display.shading.light = "STUDIO"
    bpy.context.scene.display.shading.color_type = "MATERIAL"
    bpy.context.scene.display.shading.show_shadows = False
    bpy.context.scene.render.resolution_x = 1600
    bpy.context.scene.render.resolution_y = 1000
    bpy.ops.render.render(write_still=True)
    bpy.data.images["Render Result"].save_render(filepath=str(path))


def build_case(case_dir: Path, with_paths: bool):
    reset()
    overlay = json.loads((case_dir / "blender_overlay.json").read_text(encoding="utf-8"))
    objects = import_meshes(Path(overlay["scene_xml"]))

    for bs in overlay["base_stations"]:
        add_sphere(bs["name"], bs["position"], 1.0, "bs")
        add_label(bs["name"], [bs["position"][0], bs["position"][1], bs["position"][2] + 3.0])
    for rx in overlay["receivers"]:
        mat = "rx_car" if rx["host"].startswith("car_") else "rx_roof"
        marker = add_sphere(rx["name"], rx["position"], 0.75, mat)
        add_label(rx["name"], [rx["position"][0], rx["position"][1], rx["position"][2] + 2.4], 1.4)
        if rx["host"].startswith("car_"):
            delta = [12, 0, 0] if rx["velocity"][0] > 0 else [-12, 0, 0]
            keyframe(marker, delta)

    for name, info in overlay["moving"].items():
        obj = objects.get(name)
        if obj:
            add_arrow(name, obj.location, info["frame_displacement"])
            keyframe(obj, info["frame_displacement"])

    if with_paths:
        add_paths(overlay["paths"])

    bpy.ops.object.light_add(type="AREA", location=(45, 0, 170))
    light = bpy.context.object
    light.data.energy = 800
    light.data.size = 120

    cameras = [
        setup_camera("cam_top", (45, 0, 155), (0, 0, 0), 170),
        setup_camera("cam_oblique", (78, -88, 82), (0, 0, 15), 145),
        setup_camera("cam_street", (-65, -24, 18), (8, 0, 4), 95),
    ]
    suffix = "with_paths" if with_paths else "scene_only"
    bpy.context.scene.camera = cameras[0]
    bpy.ops.wm.save_as_mainfile(filepath=str(case_dir / f"{case_dir.name}_{suffix}.blend"))
    for cam in cameras:
        bpy.context.scene.camera = cam
        render(case_dir / f"{case_dir.name}_{suffix}_{cam.name.replace('cam_', '')}.png")


def main():
    for case in ["one_bs", "two_bs"]:
        case_dir = OUT / case
        build_case(case_dir, with_paths=False)
        build_case(case_dir, with_paths=True)
    print(json.dumps({"out_dir": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
