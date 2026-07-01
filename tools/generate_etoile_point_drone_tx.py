#!/usr/bin/env python3
"""Generate point-drone Tx mobility data in the built-in Paris etoile scene."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import struct
import time
from pathlib import Path

import mitsuba as mi

# Querying variants first avoids intermittent Mitsuba/Dr.Jit startup stalls.
mi.variants()
mi.set_variant("llvm_ad_mono_polarized")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sionna
from matplotlib.patches import Polygon, Rectangle
from PIL import Image, ImageDraw
from sionna.rt import Camera, PathSolver, PlanarArray, Receiver, Transmitter, load_scene
from sionna.rt.utils import subcarrier_frequencies


OUT_ROOT = Path("etoile/point_drone_tx")
SCENE_XML = Path(sionna.rt.scene.etoile)
MESH_DIR = SCENE_XML.parent / "meshes"

FREQS_HZ = [5.0e9, 28.0e9]
FS = 122.88e6
DF = 30e3
N_SC = 512
SUBCARRIER_FREQS = subcarrier_frequencies(N_SC, DF)

DT_S = 0.1
N_FRAMES = 50
DRONE_PATH_CONTROL_M = np.array(
    [
        [146.0, -52.0, 70.0],
        [154.0, -18.0, 70.0],
        [178.0, 14.0, 70.0],
        [204.0, -8.0, 70.0],
    ],
    dtype=np.float64,
)
DRONE_ALTITUDE_NOTE = "constant 70 m altitude, point-transmitter drone; no physical drone mesh"
DRONE_PATH_NOTE = (
    "arc-length-resampled cubic Bezier arc over the dense east-side etoile "
    "building cluster; start-to-end direction follows increasing frame index"
)

RX_OBJECT = "element_089"
RX_NAME = "rx_rooftop_e089"
TX_NAME = "drone_tx_point"
CLEAN_OUTPUT_ROOT = True

SOLVER_SETTINGS = {
    "merge_shapes": True,
    "max_depth": 3,
    "los": True,
    "specular_reflection": True,
    "diffuse_reflection": False,
    "diffraction": True,
    "edge_diffraction": True,
    "refraction": False,
    "synthetic_array": False,
}

RENDER_CAMERA_KEYFRAMES = [0, 25, 49]
RENDER_NUM_SAMPLES = 16
RENDER_RESOLUTION = (720, 480)
RENDER_FOV_DEG = 70.0


def read_ply_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    size_map = {
        "float": 4,
        "float32": 4,
        "double": 8,
        "float64": 8,
        "uchar": 1,
        "uint8": 1,
        "char": 1,
        "int8": 1,
        "int": 4,
        "int32": 4,
        "uint": 4,
        "uint32": 4,
    }
    with path.open("rb") as f:
        n_vertices = 0
        props = []
        in_vertex = False
        while True:
            line = f.readline().decode("ascii", "ignore").strip()
            if line.startswith("element "):
                parts = line.split()
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    n_vertices = int(parts[2])
            elif in_vertex and line.startswith("property "):
                props.append(line.split()[1:])
            if line == "end_header":
                break
        stride = sum(size_map.get(p[0], 4) for p in props if p[0] != "list")
        mn = np.array([math.inf, math.inf, math.inf], dtype=np.float64)
        mx = np.array([-math.inf, -math.inf, -math.inf], dtype=np.float64)
        for _ in range(n_vertices):
            data = f.read(stride)
            xyz = np.array(struct.unpack_from("<fff", data, 0), dtype=np.float64)
            mn = np.minimum(mn, xyz)
            mx = np.maximum(mx, xyz)
    return mn, mx


def mesh_bounds() -> dict[str, dict[str, list[float]]]:
    bounds: dict[str, dict[str, list[float]]] = {}
    for path in sorted(MESH_DIR.glob("*.ply")):
        if path.name == "Plane.ply":
            continue
        name = re.sub(r"-itu_.*$", "", path.stem)
        mn, mx = read_ply_bounds(path)
        if name in bounds:
            old_mn = np.array(bounds[name]["min"])
            old_mx = np.array(bounds[name]["max"])
            mn = np.minimum(old_mn, mn)
            mx = np.maximum(old_mx, mx)
        bounds[name] = {
            "min": mn.tolist(),
            "max": mx.tolist(),
            "center": ((mn + mx) / 2.0).tolist(),
        }
    return bounds


BOUNDS = mesh_bounds()


def roof_position(object_name: str, dz: float = 2.0) -> np.ndarray:
    b = BOUNDS[object_name]
    mn = np.array(b["min"], dtype=np.float64)
    mx = np.array(b["max"], dtype=np.float64)
    return np.array([(mn[0] + mx[0]) / 2.0, (mn[1] + mx[1]) / 2.0, mx[2] + dz], dtype=np.float64)


def cubic_bezier(control: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)[:, None]
    return (
        ((1.0 - t) ** 3) * control[0]
        + 3.0 * ((1.0 - t) ** 2) * t * control[1]
        + 3.0 * (1.0 - t) * (t ** 2) * control[2]
        + (t ** 3) * control[3]
    )


def sample_constant_speed_curve(control: np.ndarray, n_frames: int, dt_s: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    dense_t = np.linspace(0.0, 1.0, 4096)
    dense_pts = cubic_bezier(control, dense_t)
    seg_len = np.linalg.norm(np.diff(dense_pts, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_length = float(cumulative[-1])
    sample_dist = np.linspace(0.0, total_length, n_frames)
    sample_t = np.interp(sample_dist, cumulative, dense_t)
    positions = cubic_bezier(control, sample_t)
    nominal_speed = total_length / ((n_frames - 1) * dt_s)

    velocities = np.zeros_like(positions)
    for i in range(n_frames):
        if i == 0:
            direction = positions[1] - positions[0]
        elif i == n_frames - 1:
            direction = positions[-1] - positions[-2]
        else:
            direction = positions[i + 1] - positions[i - 1]
        norm = np.linalg.norm(direction)
        if norm > 0.0:
            velocities[i] = direction / norm * nominal_speed

    return positions, velocities, total_length, nominal_speed


RX_POSITION_M = roof_position(RX_OBJECT, dz=2.0)
CAMERA_POSITION_M = RX_POSITION_M + np.array([-20.0, 0.0, 15.0], dtype=np.float64)
FRAME_TIMES_S = np.arange(N_FRAMES, dtype=np.float64) * DT_S
DRONE_POSITIONS_M, DRONE_VELOCITIES_MPS, DRONE_PATH_LENGTH_M, DRONE_NOMINAL_SPEED_MPS = sample_constant_speed_curve(
    DRONE_PATH_CONTROL_M,
    N_FRAMES,
    DT_S,
)
DRONE_START_M = DRONE_POSITIONS_M[0]
CAMERA_LOOK_AT_M = np.mean(DRONE_POSITIONS_M, axis=0)


def array() -> PlanarArray:
    return PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )


def band_name(freq_hz: float) -> str:
    ghz = freq_hz / 1e9
    if abs(ghz - round(ghz)) < 1e-9:
        return f"{int(round(ghz))}GHz"
    return f"{str(ghz).replace('.', 'p')}GHz"


def fmt_coord(value: float) -> str:
    return f"{value:.1f}".replace("-", "m").replace(".", "p")


def coord_slug(prefix: str, pos: np.ndarray) -> str:
    return f"{prefix}_x{fmt_coord(pos[0])}_y{fmt_coord(pos[1])}_z{fmt_coord(pos[2])}"


def as_list(arr: np.ndarray) -> list[float]:
    return [float(x) for x in np.asarray(arr, dtype=float).reshape(-1)]


def setup_scene(freq_hz: float):
    scene = load_scene(sionna.rt.scene.etoile, merge_shapes=SOLVER_SETTINGS["merge_shapes"])
    scene.frequency = freq_hz
    scene.tx_array = array()
    scene.rx_array = array()

    tx = Transmitter(TX_NAME, position=as_list(DRONE_POSITIONS_M[0]), power_dbm=0.0)
    rx = Receiver(RX_NAME, position=as_list(RX_POSITION_M))
    scene.add(tx)
    scene.add(rx)
    scene.get(TX_NAME).velocity = as_list(DRONE_VELOCITIES_MPS[0])
    scene.get(RX_NAME).velocity = [0.0, 0.0, 0.0]
    scene.get(TX_NAME).look_at(scene.get(RX_NAME))
    return scene


def solve_frame(scene, solver: PathSolver, drone_position: np.ndarray, drone_velocity: np.ndarray):
    tx = scene.get(TX_NAME)
    tx.position = as_list(drone_position)
    tx.velocity = as_list(drone_velocity)
    tx.look_at(scene.get(RX_NAME))

    t0 = time.perf_counter()
    paths = solver(
        scene=scene,
        max_depth=SOLVER_SETTINGS["max_depth"],
        los=SOLVER_SETTINGS["los"],
        specular_reflection=SOLVER_SETTINGS["specular_reflection"],
        diffuse_reflection=SOLVER_SETTINGS["diffuse_reflection"],
        diffraction=SOLVER_SETTINGS["diffraction"],
        edge_diffraction=SOLVER_SETTINGS["edge_diffraction"],
        refraction=SOLVER_SETTINGS["refraction"],
        synthetic_array=SOLVER_SETTINGS["synthetic_array"],
    )
    solve_ms = (time.perf_counter() - t0) * 1e3
    return paths, solve_ms


def path_lines(paths, powers: np.ndarray) -> list[dict[str, object]]:
    sources = np.asarray(paths.sources.numpy(), dtype=float).T
    targets = np.asarray(paths.targets.numpy(), dtype=float).T
    valid = np.asarray(paths.valid.numpy(), dtype=bool)[:, 0, :, 0, :]
    interactions = np.asarray(paths.interactions.numpy(), dtype=int)[:, :, 0, :, 0, :]
    vertices = np.asarray(paths.vertices.numpy(), dtype=float)[:, :, 0, :, 0, :, :]
    doppler = np.asarray(paths.doppler.numpy(), dtype=float)[:, 0, :, 0, :]
    tau = np.asarray(paths.tau.numpy(), dtype=float)[:, 0, :, 0, :]

    lines = []
    for path_i in range(valid.shape[2]):
        if not valid[0, 0, path_i]:
            continue
        pts = [sources[0].tolist()]
        inter = interactions[:, 0, 0, path_i]
        for depth_i, interaction_type in enumerate(inter):
            if interaction_type != 0:
                pts.append(vertices[depth_i, 0, 0, path_i].tolist())
        pts.append(targets[0].tolist())
        power = float(powers[path_i]) if path_i < powers.size else 0.0
        lines.append(
            {
                "tx": TX_NAME,
                "rx": RX_NAME,
                "points": pts,
                "delay_ns": float(tau[0, 0, path_i] / 1e-9),
                "doppler_hz": float(doppler[0, 0, path_i]),
                "power_db": float(10.0 * np.log10(max(power, 1e-30))),
            }
        )
    return lines


def plot_scene_base(ax, title: str, zoom: bool = True) -> None:
    for name, b in BOUNDS.items():
        mn = np.array(b["min"])
        mx = np.array(b["max"])
        width = mx[0] - mn[0]
        height = mx[1] - mn[1]
        if width <= 0.0 or height <= 0.0:
            continue
        if name == "Arc_de_Triomphe":
            face, edge, alpha, lw = "#c8b081", "#6b5530", 0.9, 0.75
        elif name == RX_OBJECT:
            face, edge, alpha, lw = "#d9b8ff", "#6b21a8", 0.8, 0.7
        else:
            face, edge, alpha, lw = "#d8dde3", "#aeb7c2", 0.56, 0.28
        ax.add_patch(Rectangle((mn[0], mn[1]), width, height, facecolor=face, edgecolor=edge, linewidth=lw, alpha=alpha))

    if zoom:
        pts = np.vstack([DRONE_POSITIONS_M[:, :2], RX_POSITION_M[:2], CAMERA_POSITION_M[:2]])
        mn = pts.min(axis=0) - np.array([80.0, 85.0])
        mx = pts.max(axis=0) + np.array([80.0, 85.0])
        ax.set_xlim(float(mn[0]), float(mx[0]))
        ax.set_ylim(float(mn[1]), float(mx[1]))
    else:
        mins = np.array([b["min"] for b in BOUNDS.values()], dtype=float).min(axis=0)
        maxs = np.array([b["max"] for b in BOUNDS.values()], dtype=float).max(axis=0)
        ax.set_xlim(float(mins[0] - 20.0), float(maxs[0] + 20.0))
        ax.set_ylim(float(mins[1] - 20.0), float(maxs[1] + 20.0))

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.grid(alpha=0.16)


def add_camera_frustum(ax) -> None:
    cam_xy = CAMERA_POSITION_M[:2]
    target_xy = CAMERA_LOOK_AT_M[:2]
    direction = target_xy - cam_xy
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        return
    direction = direction / norm
    perp = np.array([-direction[1], direction[0]])
    far = norm + 70.0
    half_width = far * math.tan(math.radians(RENDER_FOV_DEG / 2.0))
    p1 = cam_xy
    p2 = cam_xy + direction * far + perp * half_width
    p3 = cam_xy + direction * far - perp * half_width
    ax.add_patch(Polygon([p1, p2, p3], closed=True, facecolor="#facc15", edgecolor="#b45309", alpha=0.12, linewidth=1.1, zorder=4))
    ax.arrow(cam_xy[0], cam_xy[1], direction[0] * 35.0, direction[1] * 35.0, color="#b45309", width=0.6, head_width=5.0, length_includes_head=True, zorder=8)


def add_trajectory_markers(ax, frame_idx: int | None = None, show_all_points: bool = False) -> None:
    ax.plot(DRONE_POSITIONS_M[:, 0], DRONE_POSITIONS_M[:, 1], color="#ef4444", linewidth=2.0, label="Drone Tx trajectory", zorder=7)
    for arrow_idx in [7, 17, 27, 37]:
        start = DRONE_POSITIONS_M[arrow_idx, :2]
        end = DRONE_POSITIONS_M[min(arrow_idx + 3, N_FRAMES - 1), :2]
        delta = end - start
        ax.arrow(
            start[0],
            start[1],
            delta[0],
            delta[1],
            color="#b91c1c",
            width=0.55,
            head_width=4.8,
            length_includes_head=True,
            alpha=0.82,
            zorder=8,
        )
    ax.scatter(DRONE_POSITIONS_M[0, 0], DRONE_POSITIONS_M[0, 1], s=80, marker="o", color="#16a34a", edgecolor="white", linewidth=0.7, zorder=9, label="Start")
    ax.scatter(DRONE_POSITIONS_M[-1, 0], DRONE_POSITIONS_M[-1, 1], s=90, marker="X", color="#dc2626", edgecolor="white", linewidth=0.7, zorder=9, label="End")
    if show_all_points:
        ax.scatter(DRONE_POSITIONS_M[:, 0], DRONE_POSITIONS_M[:, 1], s=10, color="#991b1b", alpha=0.35, zorder=8)
    if frame_idx is not None:
        pos = DRONE_POSITIONS_M[frame_idx]
        ax.scatter(pos[0], pos[1], s=150, marker="^", color="#f97316", edgecolor="black", linewidth=0.8, zorder=11, label="Current point drone Tx")
        ax.annotate(
            f"Drone Tx frame {frame_idx:03d}\nt={FRAME_TIMES_S[frame_idx]:.1f}s\nz={pos[2]:.1f}m",
            xy=(pos[0], pos[1]),
            xytext=(12, 16),
            textcoords="offset points",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#f97316", "alpha": 0.88},
            zorder=12,
        )
    ax.scatter(RX_POSITION_M[0], RX_POSITION_M[1], s=130, marker="s", color="#7c3aed", edgecolor="white", linewidth=0.8, zorder=10, label="Fixed rooftop Rx")
    ax.annotate(
        f"Fixed rooftop Rx\nx={RX_POSITION_M[0]:.1f}, y={RX_POSITION_M[1]:.1f}, z={RX_POSITION_M[2]:.1f}",
        xy=(RX_POSITION_M[0], RX_POSITION_M[1]),
        xytext=(10, -34),
        textcoords="offset points",
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#7c3aed", "alpha": 0.9},
        zorder=12,
    )
    ax.scatter(CAMERA_POSITION_M[0], CAMERA_POSITION_M[1], s=135, marker="D", color="#0f766e", edgecolor="white", linewidth=0.8, zorder=10, label="Fixed camera")
    ax.annotate(
        f"Fixed camera\nx={CAMERA_POSITION_M[0]:.1f}, y={CAMERA_POSITION_M[1]:.1f}, z={CAMERA_POSITION_M[2]:.1f}",
        xy=(CAMERA_POSITION_M[0], CAMERA_POSITION_M[1]),
        xytext=(-82, 14),
        textcoords="offset points",
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#0f766e", "alpha": 0.9},
        zorder=12,
    )
    add_camera_frustum(ax)


def save_motion_images(images_dir: Path, band: str) -> dict[str, object]:
    images_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    plot_scene_base(ax, f"{band} etoile point-drone Tx: scene and fixed camera", zoom=False)
    add_trajectory_markers(ax, show_all_points=True)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    scene_map = images_dir / "scene_map.png"
    fig.savefig(scene_map, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    plot_scene_base(ax, f"{band} etoile point-drone Tx: 50-frame trajectory", zoom=True)
    add_trajectory_markers(ax, show_all_points=True)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    motion_map = images_dir / "motion_map.png"
    fig.savefig(motion_map, dpi=180)
    plt.close(fig)

    frame_paths = []
    rx_slug = coord_slug("rx", RX_POSITION_M)
    cam_slug = coord_slug("cam", CAMERA_POSITION_M)
    for i, pos in enumerate(DRONE_POSITIONS_M):
        fig, ax = plt.subplots(figsize=(8.8, 7.2))
        plot_scene_base(ax, f"{band} point-drone Tx motion frame {i:03d}", zoom=True)
        add_trajectory_markers(ax, frame_idx=i, show_all_points=False)
        ax.legend(loc="upper right", fontsize=7)
        fig.tight_layout()
        tx_slug = coord_slug("tx", pos)
        frame_path = images_dir / f"motion_frame_{i:03d}__{tx_slug}__{rx_slug}__{cam_slug}.png"
        fig.savefig(frame_path, dpi=170)
        plt.close(fig)
        frame_paths.append(frame_path)

    images = [Image.open(p).convert("P", palette=Image.Palette.ADAPTIVE) for p in frame_paths]
    gif_path = images_dir / f"motion__{rx_slug}__{cam_slug}.gif"
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=120, loop=0)
    for image in images:
        image.close()

    return {
        "scene_map": str(scene_map),
        "motion_map": str(motion_map),
        "motion_gif": str(gif_path),
        "motion_frames": [str(p) for p in frame_paths],
    }


def save_path_plot(path: Path, band: str, frame_lines: dict[int, list[dict[str, object]]]) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    plot_scene_base(ax, f"{band} point-drone Tx: selected path geometry", zoom=True)
    colors = {0: "#2563eb", 25: "#f97316", 49: "#dc2626"}
    for frame_idx, lines in frame_lines.items():
        for line in lines:
            pts = np.array(line["points"], dtype=float)
            ax.plot(pts[:, 0], pts[:, 1], color=colors.get(frame_idx, "#475569"), alpha=0.58, linewidth=1.0)
    add_trajectory_markers(ax, show_all_points=True)
    for frame_idx, color in colors.items():
        pos = DRONE_POSITIONS_M[frame_idx]
        ax.scatter(pos[0], pos[1], s=95, color=color, edgecolor="white", linewidth=0.7, zorder=12, label=f"paths frame {frame_idx:03d}")
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_csi_plot(path: Path, csi_matrix: np.ndarray, band: str) -> None:
    mag_db = 20.0 * np.log10(np.maximum(np.abs(csi_matrix), 1e-30))
    finite = mag_db[np.isfinite(mag_db)]
    vmax = float(np.percentile(finite, 99.0)) if finite.size else 0.0
    vmin = vmax - 45.0
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    im = ax.imshow(
        mag_db,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[
            float(SUBCARRIER_FREQS[0] / 1e6),
            float(SUBCARRIER_FREQS[-1] / 1e6),
            float(FRAME_TIMES_S[0]),
            float(FRAME_TIMES_S[-1]),
        ],
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Subcarrier offset (MHz)")
    ax.set_ylabel("Trajectory time (s)")
    ax.set_title(f"{band} point-drone Tx CSI magnitude over 50 explicit frames")
    fig.colorbar(im, ax=ax, label="|H(f)| (dB)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_pathloss_plot(path: Path, pathloss_db: np.ndarray, band: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(FRAME_TIMES_S, pathloss_db[:, 0, 0], color="#2563eb", linewidth=1.7, marker="o", markersize=2.4)
    ax.set_xlabel("Trajectory time (s)")
    ax.set_ylabel("Pathloss (dB)")
    ax.set_title(f"{band} point-drone Tx pathloss, shape [frame, rx, tx] = {tuple(pathloss_db.shape)}")
    ax.grid(alpha=0.24)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_doppler_plot(path: Path, doppler_by_frame: list[np.ndarray], power_by_frame: list[np.ndarray], band: str) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    xs = []
    ys = []
    cs = []
    for i, doppler in enumerate(doppler_by_frame):
        power = power_by_frame[i]
        if doppler.size == 0:
            continue
        xs.extend([FRAME_TIMES_S[i]] * doppler.size)
        ys.extend(doppler.tolist())
        cs.extend((10.0 * np.log10(np.maximum(power, 1e-30))).tolist())
    if xs:
        sc = ax.scatter(xs, ys, c=cs, cmap="magma", s=18, alpha=0.75)
        fig.colorbar(sc, ax=ax, label="Per-path power (dB)")
    ax.axhline(0.0, color="#334155", linewidth=0.8, alpha=0.45)
    ax.set_xlabel("Trajectory time (s)")
    ax.set_ylabel("Per-path Doppler (Hz)")
    ax.set_title(f"{band} point-drone Tx Sionna per-path Doppler")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def delay_doppler_from_frames(csi_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h_delay = np.fft.ifft(np.fft.fftshift(csi_matrix, axes=1), axis=1, norm="ortho")
    dd = np.fft.fftshift(np.fft.fft(h_delay, axis=0, norm="ortho"), axes=0)
    doppler_bins = (np.arange(N_FRAMES) - N_FRAMES // 2) * ((1.0 / DT_S) / N_FRAMES)
    delay_bins_ns = np.arange(N_SC) * ((1.0 / DF) / N_SC) / 1e-9
    return dd.astype(np.complex64), delay_bins_ns, doppler_bins


def save_delay_doppler_plot(path: Path, dd: np.ndarray, delay_bins_ns: np.ndarray, doppler_bins_hz: np.ndarray, band: str) -> None:
    mag = np.abs(dd)
    mag_db = 20.0 * np.log10(np.maximum(mag / max(float(np.max(mag)), 1e-30), 1e-12))
    delay_mask = delay_bins_ns <= 900.0
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    im = ax.imshow(
        mag_db[:, delay_mask],
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=-55.0,
        vmax=0.0,
        extent=[
            float(delay_bins_ns[delay_mask][0]),
            float(delay_bins_ns[delay_mask][-1]),
            float(doppler_bins_hz[0]),
            float(doppler_bins_hz[-1]),
        ],
    )
    ax.set_xlabel("Delay (ns)")
    ax.set_ylabel("Doppler bin from 10 Hz frame sampling (Hz)")
    ax.set_title(f"{band} delay-Doppler diagnostic from 50 explicit frames")
    ax.text(
        0.01,
        0.98,
        "Low-rate diagnostic; physical per-path Doppler is saved separately.",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#64748b", "alpha": 0.85},
    )
    fig.colorbar(im, ax=ax, label="Normalized magnitude (dB)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def render_camera_drone_keyframes(scene, camera_dir: Path, band: str) -> dict[str, object]:
    camera_dir.mkdir(parents=True, exist_ok=True)
    camera = Camera(position=as_list(CAMERA_POSITION_M), look_at=as_list(CAMERA_LOOK_AT_M))
    rx_slug = coord_slug("rx", RX_POSITION_M)
    cam_slug = coord_slug("cam", CAMERA_POSITION_M)
    look_slug = coord_slug("look", CAMERA_LOOK_AT_M)
    outputs = []
    errors = []

    for frame_idx in RENDER_CAMERA_KEYFRAMES:
        pos = DRONE_POSITIONS_M[frame_idx]
        vel = DRONE_VELOCITIES_MPS[frame_idx]
        scene.get(TX_NAME).position = as_list(pos)
        scene.get(TX_NAME).velocity = as_list(vel)
        scene.get(TX_NAME).look_at(scene.get(RX_NAME))
        tx_slug = coord_slug("tx", pos)
        out_path = camera_dir / f"camera_drone_frame_{frame_idx:03d}__{tx_slug}__{rx_slug}__{cam_slug}.png"
        try:
            bitmap = scene.render(
                camera=camera,
                num_samples=RENDER_NUM_SAMPLES,
                resolution=RENDER_RESOLUTION,
                show_devices=True,
                show_orientations=False,
                fov=RENDER_FOV_DEG,
                return_bitmap=True,
            )
            arr = np.asarray(bitmap)
            arr = np.clip(arr[..., :3], 0.0, 1.0)
            img = Image.fromarray((arr * 255.0).astype(np.uint8)).convert("RGB")
            draw = ImageDraw.Draw(img)
            label = (
                f"{band} frame {frame_idx:03d}  t={FRAME_TIMES_S[frame_idx]:.1f}s\n"
                f"Tx(point drone)=({pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f})\n"
                f"Vtx=({vel[0]:.1f},{vel[1]:.1f},{vel[2]:.1f}) m/s\n"
                f"Rx=({RX_POSITION_M[0]:.1f},{RX_POSITION_M[1]:.1f},{RX_POSITION_M[2]:.1f})\n"
                f"Cam=({CAMERA_POSITION_M[0]:.1f},{CAMERA_POSITION_M[1]:.1f},{CAMERA_POSITION_M[2]:.1f})"
            )
            draw.rectangle((8, 8, 456, 98), fill=(255, 255, 255), outline=(15, 118, 110))
            draw.text((14, 14), label, fill=(0, 0, 0))
            img.save(out_path)
            outputs.append(str(out_path))
        except Exception as exc:  # pragma: no cover - render support varies by host
            errors.append({"frame": frame_idx, "error": repr(exc)})

    gif_path = None
    if outputs:
        gif_path = camera_dir / f"camera_drone_keyframes__{rx_slug}__{cam_slug}__{look_slug}.gif"
        images = [Image.open(p).convert("P", palette=Image.Palette.ADAPTIVE) for p in outputs]
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=550, loop=0)
        for image in images:
            image.close()

    return {
        "camera_position_m": as_list(CAMERA_POSITION_M),
        "camera_look_at_m": as_list(CAMERA_LOOK_AT_M),
        "camera_fov_deg": RENDER_FOV_DEG,
        "rendered_keyframes": outputs,
        "rendered_keyframes_gif": str(gif_path) if gif_path else None,
        "render_errors": errors,
    }


def write_trajectory_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "frame",
        "time_s",
        "tx_x_m",
        "tx_y_m",
        "tx_z_m",
        "tx_vx_mps",
        "tx_vy_mps",
        "tx_vz_mps",
        "rx_x_m",
        "rx_y_m",
        "rx_z_m",
        "camera_x_m",
        "camera_y_m",
        "camera_z_m",
        "pathloss_db",
        "n_paths",
        "solve_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def run_band(freq_hz: float) -> dict[str, object]:
    band = band_name(freq_hz)
    condition = f"point_drone_tx_to_{RX_NAME}__{coord_slug('cam', CAMERA_POSITION_M)}"
    out_dir = OUT_ROOT / band / condition
    images_dir = out_dir / "images"
    camera_dir = out_dir / "camera_drone_keyframes"
    out_dir.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "npz").mkdir(parents=True, exist_ok=True)

    print(f"Running {band} {condition}", flush=True)
    scene = setup_scene(freq_hz)
    solver = PathSolver()

    csi_rows = []
    a_frames = []
    tau_frames = []
    doppler_frames = []
    power_frames = []
    valid_frames = []
    frame_rows = []
    selected_lines: dict[int, list[dict[str, object]]] = {}

    for frame_idx, drone_pos in enumerate(DRONE_POSITIONS_M):
        drone_velocity = DRONE_VELOCITIES_MPS[frame_idx]
        paths, solve_ms = solve_frame(scene, solver, drone_pos, drone_velocity)
        a, tau = paths.cir(sampling_frequency=FS, normalize_delays=False, out_type="numpy")
        h = paths.cfr(
            frequencies=SUBCARRIER_FREQS,
            sampling_frequency=1.0 / DT_S,
            num_time_steps=1,
            normalize_delays=False,
            normalize=False,
            out_type="numpy",
        ).astype(np.complex64)
        doppler = np.asarray(paths.doppler.numpy(), dtype=np.float64)

        a_vec = np.asarray(a[0, 0, 0, 0, :, 0], dtype=np.complex64)
        tau_vec = np.asarray(tau[0, 0, 0, 0, :], dtype=np.float32)
        doppler_vec = np.asarray(doppler[0, 0, 0, 0, :], dtype=np.float64)
        valid = tau_vec > -1.0
        power_vec = (np.abs(a_vec.astype(np.complex128)) ** 2).astype(np.float64)

        total_power = float(np.sum(power_vec[valid])) if np.any(valid) else 0.0
        pathloss_db = float(-10.0 * np.log10(total_power)) if total_power > 0.0 else float("inf")
        n_paths = int(np.count_nonzero(valid))

        csi_rows.append(np.asarray(h[0, 0, 0, 0, 0, :], dtype=np.complex64))
        a_frames.append(a_vec)
        tau_frames.append(tau_vec)
        doppler_frames.append(doppler_vec[valid])
        power_frames.append(power_vec[valid])
        valid_frames.append(valid)

        if frame_idx in {0, 25, 49}:
            selected_lines[frame_idx] = path_lines(paths, power_vec)

        frame_rows.append(
            {
                "frame": frame_idx,
                "time_s": float(FRAME_TIMES_S[frame_idx]),
                "tx_x_m": float(drone_pos[0]),
                "tx_y_m": float(drone_pos[1]),
                "tx_z_m": float(drone_pos[2]),
                "tx_vx_mps": float(drone_velocity[0]),
                "tx_vy_mps": float(drone_velocity[1]),
                "tx_vz_mps": float(drone_velocity[2]),
                "rx_x_m": float(RX_POSITION_M[0]),
                "rx_y_m": float(RX_POSITION_M[1]),
                "rx_z_m": float(RX_POSITION_M[2]),
                "camera_x_m": float(CAMERA_POSITION_M[0]),
                "camera_y_m": float(CAMERA_POSITION_M[1]),
                "camera_z_m": float(CAMERA_POSITION_M[2]),
                "pathloss_db": pathloss_db,
                "n_paths": n_paths,
                "solve_ms": float(solve_ms),
            }
        )

        print(
            f"  {band} frame {frame_idx + 1:02d}/{N_FRAMES}: "
            f"paths={n_paths}, pathloss={pathloss_db:.2f} dB, solve={solve_ms:.1f} ms",
            flush=True,
        )

    csi_matrix = np.stack(csi_rows, axis=0)
    csi = csi_matrix.reshape(1, 1, 1, 1, N_FRAMES, N_SC).astype(np.complex64)
    pathloss_db = np.array([row["pathloss_db"] for row in frame_rows], dtype=np.float64).reshape(N_FRAMES, 1, 1)
    n_paths = np.array([row["n_paths"] for row in frame_rows], dtype=np.int32)
    solve_ms = np.array([row["solve_ms"] for row in frame_rows], dtype=np.float64)

    max_paths = max((len(x) for x in a_frames), default=0)
    a_by_frame = np.zeros((N_FRAMES, 1, 1, 1, 1, max_paths, 1), dtype=np.complex64)
    tau_by_frame = np.full((N_FRAMES, 1, 1, 1, 1, max_paths), -1.0, dtype=np.float32)
    doppler_by_frame = np.zeros((N_FRAMES, 1, 1, 1, 1, max_paths), dtype=np.float64)
    path_power_linear = np.zeros((N_FRAMES, 1, 1, max_paths), dtype=np.float64)
    valid_path_mask = np.zeros((N_FRAMES, 1, 1, 1, 1, max_paths), dtype=bool)
    for i in range(N_FRAMES):
        n = len(a_frames[i])
        a_by_frame[i, 0, 0, 0, 0, :n, 0] = a_frames[i]
        tau_by_frame[i, 0, 0, 0, 0, :n] = tau_frames[i]
        full_doppler = np.zeros(n, dtype=np.float64)
        valid = valid_frames[i]
        full_doppler[valid] = doppler_frames[i]
        doppler_by_frame[i, 0, 0, 0, 0, :n] = full_doppler
        power = np.abs(a_frames[i].astype(np.complex128)) ** 2
        path_power_linear[i, 0, 0, :n] = power
        valid_path_mask[i, 0, 0, 0, 0, :n] = valid

    dd, delay_bins_ns, doppler_bins_hz = delay_doppler_from_frames(csi_matrix)

    images = save_motion_images(images_dir, band)
    save_path_plot(out_dir / "paths.png", band, selected_lines)
    save_csi_plot(out_dir / "csi.png", csi_matrix, band)
    save_pathloss_plot(out_dir / "pathloss.png", pathloss_db, band)
    save_doppler_plot(out_dir / "doppler.png", doppler_frames, power_frames, band)
    save_delay_doppler_plot(out_dir / "delay_doppler_from_frames.png", dd, delay_bins_ns, doppler_bins_hz, band)
    camera_outputs = render_camera_drone_keyframes(scene, camera_dir, band)

    trajectory_csv = out_dir / "trajectory.csv"
    write_trajectory_csv(trajectory_csv, frame_rows)

    data_path = out_dir / "rt_data.npz"
    data_arrays = {
        "csi": csi,
        "csi_explicit_frames": csi_matrix.astype(np.complex64),
        "a_by_frame": a_by_frame,
        "tau_by_frame": tau_by_frame,
        "doppler_hz_by_frame": doppler_by_frame,
        "valid_path_mask": valid_path_mask,
        "path_power_linear": path_power_linear,
        "pathloss_db": pathloss_db,
        "n_paths": n_paths,
        "solve_ms": solve_ms,
        "frame_time_s": FRAME_TIMES_S,
        "trajectory_dt_s": np.array(DT_S, dtype=np.float64),
        "tx_positions_m": DRONE_POSITIONS_M.astype(np.float64),
        "tx_velocity_mps": DRONE_VELOCITIES_MPS.astype(np.float64),
        "tx_speed_mps": np.linalg.norm(DRONE_VELOCITIES_MPS, axis=1).astype(np.float64),
        "trajectory_control_points_m": DRONE_PATH_CONTROL_M.astype(np.float64),
        "trajectory_length_m": np.array(DRONE_PATH_LENGTH_M, dtype=np.float64),
        "trajectory_nominal_speed_mps": np.array(DRONE_NOMINAL_SPEED_MPS, dtype=np.float64),
        "rx_position_m": RX_POSITION_M.astype(np.float64),
        "camera_position_m": CAMERA_POSITION_M.astype(np.float64),
        "camera_look_at_m": CAMERA_LOOK_AT_M.astype(np.float64),
        "freqs": SUBCARRIER_FREQS,
        "delay_doppler_from_frames": dd,
        "delay_bins_ns": delay_bins_ns,
        "doppler_bins_hz_from_frame_sampling": doppler_bins_hz,
    }
    np.savez_compressed(data_path, **data_arrays)
    npz_copy = OUT_ROOT / "npz" / f"{band}__{condition}.npz"
    np.savez_compressed(npz_copy, **data_arrays)

    summary = {
        "scene": "etoile",
        "scene_xml": str(SCENE_XML),
        "frequency_hz": freq_hz,
        "band": band,
        "condition": condition,
        "modeling_assumption": DRONE_ALTITUDE_NOTE,
        "tx": {
            "name": TX_NAME,
            "role": "moving point drone transmitter",
            "start_position_m": as_list(DRONE_START_M),
            "end_position_m": as_list(DRONE_POSITIONS_M[-1]),
            "velocity_model": "per-frame tangent velocity along arc-length-resampled curved path",
            "start_velocity_mps": as_list(DRONE_VELOCITIES_MPS[0]),
            "end_velocity_mps": as_list(DRONE_VELOCITIES_MPS[-1]),
            "nominal_speed_mps": float(DRONE_NOMINAL_SPEED_MPS),
        },
        "rx": {
            "name": RX_NAME,
            "role": "fixed rooftop receiver",
            "host_object": RX_OBJECT,
            "position_m": as_list(RX_POSITION_M),
            "velocity_mps": [0.0, 0.0, 0.0],
        },
        "fixed_camera": {
            "role": "visualization camera near fixed rooftop Rx",
            "position_m": as_list(CAMERA_POSITION_M),
            "look_at_m": as_list(CAMERA_LOOK_AT_M),
            "fov_deg": RENDER_FOV_DEG,
            "note": "Camera is for images/GIFs only; it is not the radio receiver.",
        },
        "trajectory": {
            "path_type": "curved cubic Bezier arc resampled at approximately constant speed",
            "path_note": DRONE_PATH_NOTE,
            "control_points_m": DRONE_PATH_CONTROL_M.tolist(),
            "path_length_m": float(DRONE_PATH_LENGTH_M),
            "nominal_speed_mps": float(DRONE_NOMINAL_SPEED_MPS),
            "dt_s": DT_S,
            "num_frames": N_FRAMES,
            "times_s": FRAME_TIMES_S.tolist(),
            "position_sampling_note": "50 samples at 0.1 s cover t=0.0 through 4.9 s, i.e. a nominal 5 s capture window.",
        },
        "solver": SOLVER_SETTINGS,
        "arrays": {
            "pathloss_db_shape": list(pathloss_db.shape),
            "csi_shape": list(csi.shape),
            "csi_indexing": "csi[rx, rx_ant, tx, tx_ant, explicit_frame, subcarrier]",
            "a_by_frame_shape": list(a_by_frame.shape),
            "tau_by_frame_shape": list(tau_by_frame.shape),
            "doppler_hz_by_frame_shape": list(doppler_by_frame.shape),
            "tx_velocity_mps_shape": list(DRONE_VELOCITIES_MPS.shape),
        },
        "data_considerations": [
            "The drone is a point transmitter in this first design; no physical drone body scattering, blockage, or rotor micro-Doppler is modeled.",
            "PathSolver is rerun for every explicit trajectory frame.",
            "Per-path Doppler values are computed from Sionna geometry and the per-frame tangent velocity for the curved constant-speed path.",
            "The 0.1 s trajectory sampling is too slow to sample high carrier-frequency Doppler without aliasing; delay_doppler_from_frames is a low-rate diagnostic.",
        ],
        "frame_summary": {
            "pathloss_db_min": float(np.nanmin(pathloss_db)),
            "pathloss_db_max": float(np.nanmax(pathloss_db)),
            "pathloss_db_mean": float(np.nanmean(pathloss_db)),
            "n_paths_min": int(np.min(n_paths)),
            "n_paths_max": int(np.max(n_paths)),
            "solve_ms_mean": float(np.mean(solve_ms)),
        },
        "outputs": {
            "data": str(data_path),
            "npz_copy": str(npz_copy),
            "trajectory_csv": str(trajectory_csv),
            "paths_plot": str(out_dir / "paths.png"),
            "csi_plot": str(out_dir / "csi.png"),
            "pathloss_plot": str(out_dir / "pathloss.png"),
            "doppler_plot": str(out_dir / "doppler.png"),
            "delay_doppler_from_frames_plot": str(out_dir / "delay_doppler_from_frames.png"),
            "images": images,
            "camera_drone_keyframes": camera_outputs,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    if CLEAN_OUTPUT_ROOT and OUT_ROOT.exists():
        shutil.rmtree(OUT_ROOT)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = {band_name(freq_hz): run_band(freq_hz) for freq_hz in FREQS_HZ}
    (OUT_ROOT / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
