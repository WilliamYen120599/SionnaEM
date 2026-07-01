#!/usr/bin/env python3
"""Build Blender .blend previews for Sionna mobility toy scenes."""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path("/workspace/ycq/sionna")
OUT = ROOT / "Blender Preview Images/mobility/blender"


MATERIAL_COLORS = {
    "glass": (0.45, 0.70, 0.95, 0.55),
    "wood": (0.55, 0.32, 0.16, 1.0),
    "marble": (0.72, 0.72, 0.70, 1.0),
    "brick": (0.65, 0.18, 0.12, 1.0),
    "concrete": (0.48, 0.50, 0.50, 1.0),
    "metal": (0.28, 0.30, 0.32, 1.0),
    "reflector-mat": (0.85, 0.18, 0.12, 1.0),
    "device_tx": (0.05, 0.32, 1.0, 1.0),
    "device_rx": (1.0, 0.12, 0.08, 1.0),
    "path_los": (0.05, 0.60, 1.0, 1.0),
    "path_ref": (1.0, 0.60, 0.08, 1.0),
    "motion": (0.1, 0.9, 0.35, 1.0),
    "label": (0.98, 0.98, 0.92, 1.0),
}


def reset_scene():
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
        if color[3] < 1.0:
            bsdf.inputs["Alpha"].default_value = color[3]
            mat.blend_method = "BLEND"
    return mat


def parse_xml(xml_path: Path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    shapes = []
    for shape in root.findall("shape"):
        if shape.attrib.get("type") != "ply":
            continue
        filename = None
        mat_id = None
        for child in shape:
            if child.tag == "string" and child.attrib.get("name") == "filename":
                filename = child.attrib["value"]
            if child.tag == "ref" and child.attrib.get("name") == "bsdf":
                mat_id = child.attrib["id"]
        if filename:
            raw_id = shape.attrib.get("id", Path(filename).stem)
            name = raw_id.replace("mesh-", "")
            shapes.append((name, xml_path.parent / filename, mat_id or "concrete"))
    return shapes


def import_scene_meshes(xml_path: Path):
    objects = {}
    for name, ply_path, mat_id in parse_xml(xml_path):
        bpy.ops.wm.ply_import(filepath=str(ply_path))
        obj = bpy.context.object
        obj.name = name
        obj.data.name = f"{name}_mesh"
        obj.data.materials.append(material(mat_id, MATERIAL_COLORS.get(mat_id, (0.6, 0.6, 0.6, 1.0))))
        objects[name] = obj
    return objects


def add_sphere(name: str, loc, radius: float, mat_name: str):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=radius, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material(mat_name, MATERIAL_COLORS[mat_name]))
    return obj


def add_label(text: str, loc, size: float):
    bpy.ops.object.text_add(location=loc, rotation=(math.radians(65), 0, math.radians(0)))
    obj = bpy.context.object
    obj.name = f"label_{text}"
    obj.data.body = text
    obj.data.align_x = "CENTER"
    obj.data.size = size
    obj.data.materials.append(material("label", MATERIAL_COLORS["label"]))
    return obj


def add_curve(name: str, points, mat_name: str, bevel_depth: float):
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
    obj.data.materials.append(material(mat_name, MATERIAL_COLORS[mat_name]))
    return obj


def add_motion_arrow(name: str, start, delta, scale: float):
    end = Vector(start) + Vector(delta) * scale
    add_curve(f"motion_{name}", [start, end], "motion", 0.05)
    direction = end - Vector(start)
    if direction.length > 0:
        bpy.ops.mesh.primitive_cone_add(vertices=24, radius1=0.35, depth=0.9, location=end)
        cone = bpy.context.object
        cone.name = f"motion_head_{name}"
        cone.data.materials.append(material("motion", MATERIAL_COLORS["motion"]))
        cone.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()


def keyframe_motion(obj, delta):
    obj.keyframe_insert(data_path="location", frame=1)
    obj.location = obj.location + Vector(delta)
    obj.keyframe_insert(data_path="location", frame=80)


def add_paths(overlay):
    for i, path in enumerate(overlay["paths"]):
        mat_name = "path_los" if len(path["points"]) == 2 else "path_ref"
        obj = add_curve(f"path_{i:02d}", path["points"], mat_name, 0.08)
        mid = path["points"][len(path["points"]) // 2]
        add_label(f"{path['doppler_hz']:.0f} Hz", [mid[0], mid[1], mid[2] + 1.5], 1.2)
        obj["doppler_hz"] = path["doppler_hz"]
        obj["delay_s"] = path["delay_s"]


def setup_camera(scene_name: str):
    if scene_name == "simple_reflector":
        loc, target, scale = (0, 110, 70), (0, 0, 28), 105
    else:
        loc, target, scale = (50, 0, 155), (10, 0, 0), 170
    bpy.ops.object.light_add(type="AREA", location=(loc[0], loc[1], loc[2] + 35))
    light = bpy.context.object
    light.data.energy = 650
    light.data.size = 70
    bpy.ops.object.camera_add(location=loc)
    cam = bpy.context.object
    bpy.context.scene.camera = cam
    direction = Vector(target) - Vector(cam.location)
    cam.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = scale


def save_render(path: Path):
    bpy.context.scene.render.engine = "BLENDER_WORKBENCH"
    bpy.context.scene.display.shading.light = "STUDIO"
    bpy.context.scene.display.shading.color_type = "MATERIAL"
    bpy.context.scene.display.shading.show_shadows = False
    bpy.context.scene.render.resolution_x = 1600
    bpy.context.scene.render.resolution_y = 1000
    bpy.context.scene.world = bpy.context.scene.world or bpy.data.worlds.new("World")
    bpy.context.scene.world.color = (0.78, 0.80, 0.82)
    bpy.ops.render.render(write_still=True)
    bpy.data.images["Render Result"].save_render(filepath=str(path))


def build(scene_name: str, with_paths: bool):
    reset_scene()
    scene_dir = OUT / scene_name
    overlay = json.loads((scene_dir / "blender_overlay.json").read_text(encoding="utf-8"))
    objects = import_scene_meshes(Path(overlay["scene_xml"]))
    tx_marker = add_sphere("TX", overlay["tx"], 1.2 if scene_name == "simple_reflector" else 0.8, "device_tx")
    rx_marker = add_sphere("RX", overlay["rx"], 1.2 if scene_name == "simple_reflector" else 0.8, "device_rx")
    add_label("TX", [overlay["tx"][0], overlay["tx"][1], overlay["tx"][2] + 3], 2.0)
    add_label("RX", [overlay["rx"][0], overlay["rx"][1], overlay["rx"][2] + 3], 2.0)

    for name, info in overlay["moving"].items():
        target = objects.get(name)
        if target is not None:
            add_motion_arrow(name, target.location, info["frame_displacement"], 1.0)
            keyframe_motion(target, info["frame_displacement"])
        elif name == "tx":
            add_motion_arrow(name, overlay["tx"], info["frame_displacement"], 1.0)
            keyframe_motion(tx_marker, info["frame_displacement"])
        elif name == "rx":
            add_motion_arrow(name, overlay["rx"], info["frame_displacement"], 1.0)
            keyframe_motion(rx_marker, info["frame_displacement"])

    if with_paths:
        add_paths(overlay)

    setup_camera(scene_name)
    suffix = "with_paths" if with_paths else "scene_only"
    blend_path = scene_dir / f"{scene_name}_{suffix}.blend"
    render_path = scene_dir / f"{scene_name}_{suffix}.png"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
    save_render(render_path)


def main():
    for scene_name in ["simple_reflector", "street_canyon"]:
        build(scene_name, with_paths=False)
        build(scene_name, with_paths=True)
    print(json.dumps({"out_dir": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
