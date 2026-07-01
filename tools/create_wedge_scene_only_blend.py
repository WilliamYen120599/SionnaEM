#!/usr/bin/env python3
"""Create Blender files for the simple_wedge scene without propagation paths."""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
from pathlib import Path

import sionna


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create simple_wedge scene-only Blender files.")
    parser.add_argument("--out-dir", default="Blender Preview Images/simple_wedge/visualization")
    parser.add_argument("--blender", default="Blender/blender-4.2.8-linux-x64/blender")
    return parser.parse_args()


def read_binary_ply(path: Path) -> tuple[list[list[float]], list[list[int]]]:
    raw = path.read_bytes()
    header_end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:header_end].decode("ascii")
    vertex_count = 0
    face_count = 0
    for line in header.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
        elif len(parts) == 3 and parts[:2] == ["element", "face"]:
            face_count = int(parts[2])

    offset = header_end
    vertices = []
    for _ in range(vertex_count):
        x, y, z, _u, _v = struct.unpack_from("<5f", raw, offset)
        offset += 20
        vertices.append([x, y, z])

    faces = []
    for _ in range(face_count):
        n = struct.unpack_from("<B", raw, offset)[0]
        offset += 1
        face = struct.unpack_from("<" + "i" * n, raw, offset)
        offset += 4 * n
        faces.append(list(face))

    return vertices, faces


def blender_script(payload: dict[str, object]) -> str:
    return f"""
import json
import math
import bpy
from mathutils import Vector

DATA = json.loads({json.dumps(json.dumps(payload))})

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

def material(name, color):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = color
    return mat

wedge_mat = material('wedge_metal_gray', (0.42, 0.42, 0.45, 1.0))
bs0_mat = material('BS0_blue', (0.05, 0.32, 1.0, 1.0))
bs1_mat = material('BS1_green', (0.05, 0.72, 0.24, 1.0))
ue_mat = material('UE_red', (1.0, 0.10, 0.08, 1.0))

mesh = bpy.data.meshes.new('simple_wedge_mesh')
mesh.from_pydata(DATA['wedge_vertices'], [], DATA['wedge_faces'])
mesh.update()
wedge = bpy.data.objects.new('simple_wedge_only', mesh)
bpy.context.collection.objects.link(wedge)
wedge.data.materials.append(wedge_mat)

def add_sphere(name, loc, radius, mat):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=radius, location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(mat)
    return obj

def add_label(text, loc, size=2.2):
    bpy.ops.object.text_add(location=loc)
    obj = bpy.context.object
    obj.name = 'label_' + text
    obj.data.body = text
    obj.data.align_x = 'CENTER'
    obj.data.size = size
    return obj

if DATA['include_markers']:
    bs_positions = DATA['bs_positions']
    add_sphere('BS0', bs_positions['BS0'], 1.25, bs0_mat)
    add_label('BS0', [bs_positions['BS0'][0], bs_positions['BS0'][1], bs_positions['BS0'][2] + 3.0])
    add_sphere('BS1', bs_positions['BS1'], 1.25, bs1_mat)
    add_label('BS1', [bs_positions['BS1'][0], bs_positions['BS1'][1], bs_positions['BS1'][2] + 3.0])
    ue = DATA['ue_position']
    add_sphere('UE', ue, 1.35, ue_mat)
    add_label('UE', [ue[0], ue[1], ue[2] + 3.0])

def look_at(obj, target):
    direction = Vector(target) - Vector(obj.location)
    obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

bpy.ops.object.light_add(type='AREA', location=(25, -35, 55))
light = bpy.context.object
light.data.energy = 700
light.data.size = 55

bpy.ops.object.camera_add(location=(45, -70, 45))
camera = bpy.context.object
bpy.context.scene.camera = camera
camera.data.type = 'ORTHO'
camera.data.ortho_scale = 80
look_at(camera, (12, -2, 0))

bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
bpy.context.scene.render.resolution_x = 1600
bpy.context.scene.render.resolution_y = 1100
bpy.context.scene.world = bpy.context.scene.world or bpy.data.worlds.new('World')
bpy.context.scene.world.color = (0.78, 0.78, 0.78)

bpy.ops.wm.save_as_mainfile(filepath=DATA['blend_path'])
bpy.ops.render.render(write_still=True)
bpy.data.images['Render Result'].save_render(filepath=DATA['render_path'])
"""


def run_blender(blender: Path, script_path: Path) -> None:
    completed = subprocess.run(
        [str(blender), "--background", "--python", str(script_path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        print(completed.stdout)
        raise RuntimeError(f"Blender failed with return code {completed.returncode}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wedge_mesh = Path(sionna.rt.scene.simple_wedge).parent / "meshes" / "wedge.ply"
    vertices, faces = read_binary_ply(wedge_mesh)

    bs_positions = {
        "BS0": [43.30127, 25.0, 0.0],
        "BS1": [25.0, 43.30127, 0.0],
    }
    ue_position = [-1.545085, 4.7552824, 0.0]

    jobs = [
        {
            "name": "simple_wedge_scene_only",
            "include_markers": False,
            "description": "Environment only: simple_wedge mesh, no BS/UE, no propagation paths.",
        },
        {
            "name": "simple_wedge_setup_no_paths",
            "include_markers": True,
            "description": "Simulation setup: simple_wedge mesh plus BS0/BS1/UE markers, no propagation paths.",
        },
    ]

    outputs = {}
    blender_exe = Path(args.blender)
    for job in jobs:
        blend_path = out_dir / f"{job['name']}.blend"
        render_path = out_dir / f"{job['name']}.png"
        script_path = out_dir / f"{job['name']}_create.py"
        payload = {
            "wedge_vertices": vertices,
            "wedge_faces": faces,
            "bs_positions": bs_positions,
            "ue_position": ue_position,
            "include_markers": job["include_markers"],
            "blend_path": str(blend_path.resolve()),
            "render_path": str(render_path.resolve()),
        }
        script_path.write_text(blender_script(payload), encoding="utf-8")
        run_blender(blender_exe, script_path)
        outputs[job["name"]] = {
            "description": job["description"],
            "blend": str(blend_path),
            "png": str(render_path),
            "script": str(script_path),
        }

    summary = {
        "source_scene": str(sionna.rt.scene.simple_wedge),
        "note": "These Blender files intentionally exclude propagation/ray curves.",
        "outputs": outputs,
    }
    summary_path = out_dir / "simple_wedge_scene_only_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
