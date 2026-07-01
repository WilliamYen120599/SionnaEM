#!/usr/bin/env python3
"""Visualize the two-BS simple_wedge scene and ray paths."""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
from pathlib import Path

import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sionna
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from sionna.rt import ITURadioMaterial, PathSolver, PlanarArray, Receiver, Transmitter, load_scene


INTERACTION_NAMES = {
    0: "LoS",
    1: "Reflection",
    8: "Diffraction",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize the two-BS simple_wedge scene.")
    parser.add_argument("--out-dir", default="Blender Preview Images/simple_wedge/visualization")
    parser.add_argument("--blender", default="Blender/blender-4.2.8-linux-x64/blender")
    return parser.parse_args()


def read_binary_ply(path: Path) -> tuple[np.ndarray, list[tuple[int, ...]]]:
    raw = path.read_bytes()
    header_end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:header_end].decode("ascii")
    vertex_count = 0
    face_count = 0
    for line in header.splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[:2] == ["element", "vertex"]:
            vertex_count = int(parts[2])
        if len(parts) == 3 and parts[:2] == ["element", "face"]:
            face_count = int(parts[2])

    offset = header_end
    vertices = []
    for _ in range(vertex_count):
        x, y, z, _u, _v = struct.unpack_from("<5f", raw, offset)
        offset += 20
        vertices.append((x, y, z))

    faces = []
    for _ in range(face_count):
        n = struct.unpack_from("<B", raw, offset)[0]
        offset += 1
        face = struct.unpack_from("<" + "i" * n, raw, offset)
        offset += 4 * n
        faces.append(tuple(face))

    return np.asarray(vertices, dtype=np.float32), faces


def run_two_bs_scene():
    scene = load_scene(sionna.rt.scene.simple_wedge, merge_shapes=False)
    scene.frequency = 1e9
    scene.objects["wedge"].radio_material = ITURadioMaterial("metal", itu_type="metal", thickness=100)
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.rx_array = scene.tx_array

    bs_positions = {
        "BS0": np.array([43.30127, 25.0, 0.0], dtype=np.float32),
        "BS1": np.array([25.0, 43.30127, 0.0], dtype=np.float32),
    }
    ue_position = np.array([-1.545085, 4.7552824, 0.0], dtype=np.float32)

    for name, pos in bs_positions.items():
        scene.add(Transmitter(name=name, position=pos, orientation=[0, 0, 0], power_dbm=0.0))
    scene.add(Receiver(name="UE", position=ue_position, orientation=[0, 0, 0]))

    paths = PathSolver()(
        scene,
        max_depth=1,
        los=True,
        specular_reflection=True,
        diffraction=True,
        edge_diffraction=False,
        refraction=False,
        diffuse_reflection=False,
        synthetic_array=False,
    )
    a, tau = paths.cir(sampling_frequency=122.88e6, normalize_delays=False, out_type="numpy")
    return paths, a, tau, bs_positions, ue_position


def extract_path_lines(paths, a: np.ndarray, tau: np.ndarray, bs_positions, ue_position) -> list[dict[str, object]]:
    vertices = paths.vertices.numpy()
    interactions = paths.interactions.numpy()
    valid = paths.valid.numpy()

    path_lines = []
    bs_names = list(bs_positions.keys())
    for tx_idx, bs_name in enumerate(bs_names):
        for path_idx in range(valid.shape[-1]):
            if not bool(valid[0, 0, tx_idx, 0, path_idx]):
                continue
            interaction_code = int(interactions[0, 0, 0, tx_idx, 0, path_idx])
            interaction = INTERACTION_NAMES.get(interaction_code, f"Type {interaction_code}")
            power = float(np.abs(a[0, 0, tx_idx, 0, path_idx, 0]) ** 2)
            delay_s = float(tau[0, 0, tx_idx, 0, path_idx])

            points = [np.asarray(bs_positions[bs_name], dtype=float)]
            if interaction_code != 0:
                points.append(np.asarray(vertices[0, 0, 0, tx_idx, 0, path_idx, :], dtype=float))
            points.append(np.asarray(ue_position, dtype=float))

            path_lines.append(
                {
                    "bs": bs_name,
                    "path_index": path_idx,
                    "interaction": interaction,
                    "interaction_code": interaction_code,
                    "power_linear": power,
                    "power_db": float(10.0 * np.log10(max(power, 1e-30))),
                    "delay_ns": delay_s * 1e9,
                    "points": [p.tolist() for p in points],
                }
            )
    return path_lines


def save_matplotlib_png(
    out_path: Path,
    wedge_vertices: np.ndarray,
    wedge_faces: list[tuple[int, ...]],
    path_lines: list[dict[str, object]],
    bs_positions,
    ue_position,
) -> None:
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    polys = [[wedge_vertices[i] for i in face] for face in wedge_faces]
    wedge = Poly3DCollection(polys, alpha=0.26, facecolor="#9a9a9a", edgecolor="#444444", linewidth=0.8)
    ax.add_collection3d(wedge)

    colors = {"BS0": "#1f77b4", "BS1": "#2ca02c"}
    linestyles = {"LoS": "-", "Reflection": "--", "Diffraction": ":"}

    for name, pos in bs_positions.items():
        ax.scatter(pos[0], pos[1], pos[2], s=90, color=colors[name], depthshade=False)
        ax.text(pos[0], pos[1], pos[2] + 2.2, name, color=colors[name], weight="bold")

    ax.scatter(ue_position[0], ue_position[1], ue_position[2], s=110, color="#d62728", depthshade=False)
    ax.text(ue_position[0], ue_position[1], ue_position[2] + 2.2, "UE", color="#d62728", weight="bold")

    used_labels = set()
    for item in path_lines:
        pts = np.asarray(item["points"], dtype=float)
        label = f"{item['bs']} {item['interaction']}"
        plot_label = label if label not in used_labels else None
        used_labels.add(label)
        ax.plot(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            color=colors[item["bs"]],
            linestyle=linestyles.get(item["interaction"], "-"),
            linewidth=2.1,
            alpha=0.85,
            label=plot_label,
        )
        if len(pts) > 2:
            v = pts[1]
            ax.scatter(v[0], v[1], v[2], color=colors[item["bs"]], s=28, depthshade=False)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Two-BS simple_wedge scene: wedge, BS/UE positions, and ray paths")
    ax.view_init(elev=25, azim=-135)
    ax.set_xlim(-10, 50)
    ax.set_ylim(-35, 50)
    ax.set_zlim(-18, 18)
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def blender_script_text(
    blend_path: Path,
    render_path: Path,
    wedge_vertices: np.ndarray,
    wedge_faces: list[tuple[int, ...]],
    path_lines: list[dict[str, object]],
    bs_positions,
    ue_position,
) -> str:
    payload = {
        "blend_path": str(blend_path.resolve()),
        "render_path": str(render_path.resolve()),
        "wedge_vertices": wedge_vertices.tolist(),
        "wedge_faces": [list(f) for f in wedge_faces],
        "path_lines": path_lines,
        "bs_positions": {k: v.tolist() for k, v in bs_positions.items()},
        "ue_position": ue_position.tolist(),
    }
    return f"""
import json
import math
import bpy
from mathutils import Vector

DATA = json.loads({json.dumps(json.dumps(payload))})

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

def mat(name, color, alpha_blend=False, emission=False):
    m = bpy.data.materials.new(name)
    m.diffuse_color = color
    if alpha_blend:
        m.use_nodes = True
        bsdf = m.node_tree.nodes.get('Principled BSDF')
        if bsdf:
            bsdf.inputs['Alpha'].default_value = color[3]
            bsdf.inputs['Base Color'].default_value = color
        m.blend_method = 'BLEND'
        m.show_transparent_back = True
    if emission:
        m.use_nodes = True
        nodes = m.node_tree.nodes
        nodes.clear()
        emission_node = nodes.new('ShaderNodeEmission')
        emission_node.inputs['Color'].default_value = color
        emission_node.inputs['Strength'].default_value = 1.8
        out = nodes.new('ShaderNodeOutputMaterial')
        m.node_tree.links.new(emission_node.outputs['Emission'], out.inputs['Surface'])
    return m

wedge_mat = mat('semi_transparent_wedge_metal', (0.55, 0.55, 0.58, 0.22), alpha_blend=True)
bs0_mat = mat('BS0_blue', (0.05, 0.32, 1.0, 1.0), emission=True)
bs1_mat = mat('BS1_green', (0.05, 0.72, 0.24, 1.0), emission=True)
ue_mat = mat('UE_red', (1.0, 0.10, 0.08, 1.0), emission=True)
diff_mat = mat('diffraction_orange', (1.0, 0.55, 0.05, 1.0), emission=True)
refl_mat = mat('reflection_purple', (0.50, 0.25, 1.0, 1.0), emission=True)

mesh = bpy.data.meshes.new('simple_wedge_mesh')
mesh.from_pydata(DATA['wedge_vertices'], [], DATA['wedge_faces'])
mesh.update()
obj = bpy.data.objects.new('simple_wedge_metal_object', mesh)
bpy.context.collection.objects.link(obj)
obj.data.materials.append(wedge_mat)

def add_sphere(name, loc, radius, material):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=32, ring_count=16, radius=radius, location=loc)
    o = bpy.context.object
    o.name = name
    o.data.materials.append(material)
    return o

def add_label(text, loc, size=2.1):
    bpy.ops.object.text_add(location=loc, rotation=(0, 0, 0))
    o = bpy.context.object
    o.name = 'label_' + text
    o.data.body = text
    o.data.align_x = 'CENTER'
    o.data.size = size
    return o

materials = {{'BS0': bs0_mat, 'BS1': bs1_mat}}
for name, loc in DATA['bs_positions'].items():
    add_sphere(name, loc, 1.25, materials[name])
    add_label(name, [loc[0], loc[1], loc[2] + 3.0])
add_sphere('UE', DATA['ue_position'], 1.35, ue_mat)
add_label('UE', [DATA['ue_position'][0], DATA['ue_position'][1], DATA['ue_position'][2] + 3.0])

def add_curve(name, points, material, bevel_depth):
    curve = bpy.data.curves.new(name, 'CURVE')
    curve.dimensions = '3D'
    curve.resolution_u = 2
    curve.bevel_depth = bevel_depth
    curve.bevel_resolution = 4
    poly = curve.splines.new('POLY')
    poly.points.add(len(points) - 1)
    for p, co in zip(poly.points, points):
        p.co = (co[0], co[1], co[2], 1.0)
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)
    return obj

for item in DATA['path_lines']:
    base_mat = materials[item['bs']]
    if item['interaction'] == 'Reflection':
        material = refl_mat
    elif item['interaction'] == 'Diffraction':
        material = diff_mat
    else:
        material = base_mat
    add_curve(
        f"{{item['bs']}}_path{{item['path_index']}}_{{item['interaction']}}_{{item['power_db']:.1f}}dB",
        item['points'],
        material,
        0.16 if item['interaction'] == 'LoS' else 0.12,
    )

bpy.ops.object.light_add(type='AREA', location=(25, -35, 55))
light = bpy.context.object
light.name = 'large_softbox'
light.data.energy = 900
light.data.size = 60

def look_at(obj, target):
    loc = Vector(obj.location)
    direction = Vector(target) - loc
    obj.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

bpy.ops.object.camera_add(location=(20, 14, 112))
cam = bpy.context.object
bpy.context.scene.camera = cam
cam.data.type = 'ORTHO'
cam.data.ortho_scale = 82
look_at(cam, (17, 10, 0))

bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
bpy.context.scene.render.resolution_x = 1800
bpy.context.scene.render.resolution_y = 1200
bpy.context.scene.view_settings.view_transform = 'Filmic'
bpy.context.scene.view_settings.look = 'Medium High Contrast'
world = bpy.context.scene.world or bpy.data.worlds.new('World')
bpy.context.scene.world = world
world.color = (0.78, 0.78, 0.78)

bpy.ops.wm.save_as_mainfile(filepath=DATA['blend_path'])
bpy.ops.render.render(write_still=True)
bpy.data.images['Render Result'].save_render(filepath=DATA['render_path'])
"""


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    wedge_path = Path(sionna.rt.scene.simple_wedge).parent / "meshes" / "wedge.ply"
    wedge_vertices, wedge_faces = read_binary_ply(wedge_path)
    paths, a, tau, bs_positions, ue_position = run_two_bs_scene()
    path_lines = extract_path_lines(paths, a, tau, bs_positions, ue_position)

    mpl_png = out_dir / "simple_wedge_scene_3d_paths.png"
    blend_path = out_dir / "simple_wedge_two_bs_scene.blend"
    blender_png = out_dir / "simple_wedge_two_bs_scene_blender_render.png"
    blender_script = out_dir / "simple_wedge_blender_render.py"
    summary_path = out_dir / "simple_wedge_scene_visualization_summary.json"

    save_matplotlib_png(mpl_png, wedge_vertices, wedge_faces, path_lines, bs_positions, ue_position)
    blender_script.write_text(
        blender_script_text(blend_path, blender_png, wedge_vertices, wedge_faces, path_lines, bs_positions, ue_position),
        encoding="utf-8",
    )

    blender_exe = Path(args.blender)
    blender_result = None
    if blender_exe.exists():
        completed = subprocess.run(
            [str(blender_exe), "--background", "--python", str(blender_script)],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        blender_result = {
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-3000:],
        }
        if completed.returncode != 0:
            print(completed.stdout)
            raise RuntimeError(f"Blender failed with return code {completed.returncode}")

    summary = {
        "scene": "Sionna built-in simple_wedge, visualized for the generalized two-BS run",
        "bs_positions": {k: v.tolist() for k, v in bs_positions.items()},
        "ue_position": ue_position.tolist(),
        "path_lines": path_lines,
        "outputs": {
            "matplotlib_png": str(mpl_png),
            "blender_file": str(blend_path),
            "blender_render_png": str(blender_png),
            "blender_script": str(blender_script),
        },
        "blender_result": blender_result,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary["outputs"], indent=2))


if __name__ == "__main__":
    main()
