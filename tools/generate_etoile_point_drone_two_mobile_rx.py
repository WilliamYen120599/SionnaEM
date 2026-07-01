#!/usr/bin/env python3
"""Generate point-drone Tx data with two moving point receivers in etoile."""

from __future__ import annotations

import csv
import json
import math
import shutil
import time
from pathlib import Path

import generate_etoile_point_drone_tx as base
import numpy as np
from PIL import Image, ImageDraw
from sionna.rt import Camera, PathSolver, Receiver, Transmitter, load_scene


OUT_ROOT = base.OUT_ROOT
SCENE_XML = base.SCENE_XML
FREQS_HZ = base.FREQS_HZ
FS = base.FS
DF = base.DF
N_SC = base.N_SC
SUBCARRIER_FREQS = base.SUBCARRIER_FREQS
DT_S = base.DT_S
N_FRAMES = base.N_FRAMES
FRAME_TIMES_S = base.FRAME_TIMES_S
DRONE_POSITIONS_M = base.DRONE_POSITIONS_M
DRONE_VELOCITIES_MPS = base.DRONE_VELOCITIES_MPS
DRONE_PATH_CONTROL_M = base.DRONE_PATH_CONTROL_M
DRONE_PATH_LENGTH_M = base.DRONE_PATH_LENGTH_M
DRONE_NOMINAL_SPEED_MPS = base.DRONE_NOMINAL_SPEED_MPS
CAMERA_POSITION_M = base.CAMERA_POSITION_M
CAMERA_LOOK_AT_M = base.CAMERA_LOOK_AT_M
BOUNDS = base.BOUNDS

TX_NAME = "drone_tx_point"
RX_NAMES = ["rx_mobile_east", "rx_mobile_west"]
RX_LABELS = {"rx_mobile_east": "mobile_east", "rx_mobile_west": "mobile_west"}
CONDITION = f"point_drone_tx_to_two_mobile_rx__{base.coord_slug('cam', CAMERA_POSITION_M)}"
RX_ALTITUDE_M = 42.0
RX_LATERAL_OFFSET_M = 14.0
RENDER_CAMERA_KEYFRAMES = [0, 25, 49]
RENDER_NUM_SAMPLES = 12
RENDER_RESOLUTION = (720, 480)
RENDER_FOV_DEG = 70.0

SOLVER_SETTINGS = dict(base.SOLVER_SETTINGS)


def moving_rx_tracks() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
    forward_xy = DRONE_POSITIONS_M[-1, :2] - DRONE_POSITIONS_M[0, :2]
    direction = forward_xy / np.linalg.norm(forward_xy)
    perp = np.array([-direction[1], direction[0]], dtype=np.float64)
    east_offset = -perp * RX_LATERAL_OFFSET_M
    west_offset = perp * RX_LATERAL_OFFSET_M

    base_track = DRONE_POSITIONS_M[::-1].copy()
    base_track[:, 2] = RX_ALTITUDE_M
    reversed_velocity = -DRONE_VELOCITIES_MPS[::-1].copy()
    reversed_velocity[:, 2] = 0.0

    positions = {
        "rx_mobile_east": base_track + np.array([east_offset[0], east_offset[1], 0.0]),
        "rx_mobile_west": base_track + np.array([west_offset[0], west_offset[1], 0.0]),
    }
    velocities = {
        "rx_mobile_east": reversed_velocity.copy(),
        "rx_mobile_west": reversed_velocity.copy(),
    }
    design = {
        "rx_altitude_m": RX_ALTITUDE_M,
        "lateral_offset_m": RX_LATERAL_OFFSET_M,
        "average_drone_xy_direction_unit": direction.tolist(),
        "east_offset_xy_m": east_offset.tolist(),
        "west_offset_xy_m": west_offset.tolist(),
        "motion_note": (
            "Both moving point receivers follow the drone path in reverse with lateral offsets, "
            "so their nominal motion is opposite the moving drone Tx."
        ),
    }
    return positions, velocities, design


RX_POSITIONS_M, RX_VELOCITIES_MPS, RX_TRACK_DESIGN = moving_rx_tracks()


def setup_scene(freq_hz: float):
    scene = load_scene(base.sionna.rt.scene.etoile, merge_shapes=SOLVER_SETTINGS["merge_shapes"])
    scene.frequency = freq_hz
    scene.tx_array = base.array()
    scene.rx_array = base.array()
    scene.add(Transmitter(TX_NAME, position=base.as_list(DRONE_POSITIONS_M[0]), power_dbm=0.0))
    for rx_name in RX_NAMES:
        scene.add(Receiver(rx_name, position=base.as_list(RX_POSITIONS_M[rx_name][0])))
        scene.get(rx_name).velocity = base.as_list(RX_VELOCITIES_MPS[rx_name][0])
    scene.get(TX_NAME).velocity = base.as_list(DRONE_VELOCITIES_MPS[0])
    scene.get(TX_NAME).look_at(scene.get(RX_NAMES[0]))
    return scene


def update_frame(scene, frame_idx: int) -> None:
    tx = scene.get(TX_NAME)
    tx.position = base.as_list(DRONE_POSITIONS_M[frame_idx])
    tx.velocity = base.as_list(DRONE_VELOCITIES_MPS[frame_idx])
    for rx_name in RX_NAMES:
        rx = scene.get(rx_name)
        rx.position = base.as_list(RX_POSITIONS_M[rx_name][frame_idx])
        rx.velocity = base.as_list(RX_VELOCITIES_MPS[rx_name][frame_idx])
    tx.look_at(scene.get(RX_NAMES[0]))


def solve_frame(scene, solver: PathSolver, frame_idx: int):
    update_frame(scene, frame_idx)
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
    return paths, (time.perf_counter() - t0) * 1e3


def path_lines(paths, powers_by_rx: dict[int, np.ndarray]) -> list[dict[str, object]]:
    sources = np.asarray(paths.sources.numpy(), dtype=float).T
    targets = np.asarray(paths.targets.numpy(), dtype=float).T
    valid = np.asarray(paths.valid.numpy(), dtype=bool)[:, 0, :, 0, :]
    interactions = np.asarray(paths.interactions.numpy(), dtype=int)[:, :, 0, :, 0, :]
    vertices = np.asarray(paths.vertices.numpy(), dtype=float)[:, :, 0, :, 0, :, :]
    doppler = np.asarray(paths.doppler.numpy(), dtype=float)[:, 0, :, 0, :]
    tau = np.asarray(paths.tau.numpy(), dtype=float)[:, 0, :, 0, :]
    lines = []
    for rx_i, rx_name in enumerate(RX_NAMES):
        for path_i in range(valid.shape[2]):
            if not valid[rx_i, 0, path_i]:
                continue
            pts = [sources[0].tolist()]
            inter = interactions[:, rx_i, 0, path_i]
            for depth_i, interaction_type in enumerate(inter):
                if interaction_type != 0:
                    pts.append(vertices[depth_i, rx_i, 0, path_i].tolist())
            pts.append(targets[rx_i].tolist())
            power = float(powers_by_rx[rx_i][path_i])
            lines.append(
                {
                    "tx": TX_NAME,
                    "rx": rx_name,
                    "points": pts,
                    "delay_ns": float(tau[rx_i, 0, path_i] / 1e-9),
                    "doppler_hz": float(doppler[rx_i, 0, path_i]),
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
        else:
            face, edge, alpha, lw = "#d8dde3", "#aeb7c2", 0.56, 0.28
        ax.add_patch(base.Rectangle((mn[0], mn[1]), width, height, facecolor=face, edgecolor=edge, linewidth=lw, alpha=alpha))

    pts = [DRONE_POSITIONS_M[:, :2], CAMERA_POSITION_M[:2][None, :]]
    pts.extend(track[:, :2] for track in RX_POSITIONS_M.values())
    pts_arr = np.vstack(pts)
    if zoom:
        mn = pts_arr.min(axis=0) - np.array([55.0, 60.0])
        mx = pts_arr.max(axis=0) + np.array([55.0, 60.0])
    else:
        mn = np.array([b["min"] for b in BOUNDS.values()], dtype=float).min(axis=0)[:2] - 20.0
        mx = np.array([b["max"] for b in BOUNDS.values()], dtype=float).max(axis=0)[:2] + 20.0
    ax.set_xlim(float(mn[0]), float(mx[0]))
    ax.set_ylim(float(mn[1]), float(mx[1]))
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
    far = norm + 85.0
    half_width = far * math.tan(math.radians(RENDER_FOV_DEG / 2.0))
    p1 = cam_xy
    p2 = cam_xy + direction * far + perp * half_width
    p3 = cam_xy + direction * far - perp * half_width
    ax.add_patch(base.Polygon([p1, p2, p3], closed=True, facecolor="#facc15", edgecolor="#b45309", alpha=0.11, linewidth=1.1, zorder=4))
    ax.scatter(cam_xy[0], cam_xy[1], s=125, marker="D", color="#0f766e", edgecolor="white", linewidth=0.8, zorder=10, label="fixed camera")


def add_motion_markers(ax, frame_idx: int | None = None, show_points: bool = False) -> None:
    ax.plot(DRONE_POSITIONS_M[:, 0], DRONE_POSITIONS_M[:, 1], color="#ef4444", linewidth=2.0, label="drone Tx", zorder=7)
    rx_colors = {"rx_mobile_east": "#16a34a", "rx_mobile_west": "#2563eb"}
    for rx_name in RX_NAMES:
        track = RX_POSITIONS_M[rx_name]
        ax.plot(track[:, 0], track[:, 1], color=rx_colors[rx_name], linewidth=2.0, label=RX_LABELS[rx_name], zorder=7)
        if show_points:
            ax.scatter(track[:, 0], track[:, 1], s=10, color=rx_colors[rx_name], alpha=0.35, zorder=8)
        for arrow_idx in [7, 21, 35]:
            start = track[arrow_idx, :2]
            end = track[min(arrow_idx + 3, N_FRAMES - 1), :2]
            delta = end - start
            ax.arrow(start[0], start[1], delta[0], delta[1], color=rx_colors[rx_name], width=0.45, head_width=4.2, length_includes_head=True, alpha=0.82, zorder=8)
    for arrow_idx in [7, 21, 35]:
        start = DRONE_POSITIONS_M[arrow_idx, :2]
        end = DRONE_POSITIONS_M[min(arrow_idx + 3, N_FRAMES - 1), :2]
        delta = end - start
        ax.arrow(start[0], start[1], delta[0], delta[1], color="#b91c1c", width=0.5, head_width=4.6, length_includes_head=True, alpha=0.82, zorder=8)
    if frame_idx is not None:
        tx = DRONE_POSITIONS_M[frame_idx]
        ax.scatter(tx[0], tx[1], s=150, marker="^", color="#f97316", edgecolor="black", linewidth=0.8, zorder=12, label="current drone Tx")
        for rx_name in RX_NAMES:
            rx = RX_POSITIONS_M[rx_name][frame_idx]
            ax.scatter(rx[0], rx[1], s=130, marker="o", color=rx_colors[rx_name], edgecolor="black", linewidth=0.8, zorder=12, label=f"current {RX_LABELS[rx_name]}")
    ax.scatter(DRONE_POSITIONS_M[0, 0], DRONE_POSITIONS_M[0, 1], s=78, marker="o", color="#fee2e2", edgecolor="#b91c1c", linewidth=0.9, zorder=10)
    ax.scatter(DRONE_POSITIONS_M[-1, 0], DRONE_POSITIONS_M[-1, 1], s=88, marker="X", color="#dc2626", edgecolor="white", linewidth=0.8, zorder=10)
    add_camera_frustum(ax)


def save_motion_images(images_dir: Path, band: str) -> dict[str, object]:
    images_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = base.plt.subplots(figsize=(8.8, 7.2))
    plot_scene_base(ax, f"{band} etoile point-drone Tx + two moving Rx: scene map", zoom=False)
    add_motion_markers(ax, show_points=True)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout()
    scene_map = images_dir / "scene_map.png"
    fig.savefig(scene_map, dpi=180)
    base.plt.close(fig)

    fig, ax = base.plt.subplots(figsize=(8.8, 7.2))
    plot_scene_base(ax, f"{band} point-drone Tx + two moving Rx: motion map", zoom=True)
    add_motion_markers(ax, show_points=True)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout()
    motion_map = images_dir / "motion_map.png"
    fig.savefig(motion_map, dpi=180)
    base.plt.close(fig)

    frame_paths = []
    cam_slug = base.coord_slug("cam", CAMERA_POSITION_M)
    for i in range(N_FRAMES):
        fig, ax = base.plt.subplots(figsize=(8.8, 7.2))
        plot_scene_base(ax, f"{band} two-moving-Rx motion frame {i:03d}", zoom=True)
        add_motion_markers(ax, frame_idx=i, show_points=False)
        ax.legend(loc="upper right", fontsize=6.6, ncol=2)
        fig.tight_layout()
        tx_slug = base.coord_slug("tx", DRONE_POSITIONS_M[i])
        frame_path = images_dir / f"motion_frame_{i:03d}__{tx_slug}__{cam_slug}.png"
        fig.savefig(frame_path, dpi=170)
        base.plt.close(fig)
        frame_paths.append(frame_path)
    gif_path = images_dir / f"motion_two_mobile_rx__{cam_slug}.gif"
    images = [Image.open(p).convert("P", palette=Image.Palette.ADAPTIVE) for p in frame_paths]
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=100, loop=0)
    for image in images:
        image.close()
    return {
        "scene_map": str(scene_map),
        "motion_map": str(motion_map),
        "motion_gif": str(gif_path),
        "motion_frames": [str(p) for p in frame_paths],
    }


def save_pathloss_plot(path: Path, pathloss_db: np.ndarray, band: str) -> None:
    fig, ax = base.plt.subplots(figsize=(8.2, 4.4))
    colors = ["#16a34a", "#2563eb"]
    for rx_i, rx_name in enumerate(RX_NAMES):
        ax.plot(FRAME_TIMES_S, pathloss_db[:, rx_i, 0], color=colors[rx_i], linewidth=1.7, marker="o", markersize=2.4, label=RX_LABELS[rx_name])
    ax.set_xlabel("Trajectory time (s)")
    ax.set_ylabel("Pathloss (dB)")
    ax.set_title(f"{band} point-drone Tx to two moving Rx, shape {tuple(pathloss_db.shape)}")
    ax.grid(alpha=0.24)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    base.plt.close(fig)


def save_csi_plot(path: Path, csi_matrix: np.ndarray, band: str) -> None:
    fig, axes = base.plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    for rx_i, rx_name in enumerate(RX_NAMES):
        mag_db = 20.0 * np.log10(np.maximum(np.abs(csi_matrix[:, rx_i, :]), 1e-30))
        finite = mag_db[np.isfinite(mag_db)]
        vmax = float(np.percentile(finite, 99.0)) if finite.size else 0.0
        vmin = vmax - 45.0
        im = axes[rx_i].imshow(
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
        axes[rx_i].set_title(RX_LABELS[rx_name])
        axes[rx_i].set_xlabel("Subcarrier offset (MHz)")
        axes[rx_i].grid(alpha=0.12)
    axes[0].set_ylabel("Trajectory time (s)")
    fig.suptitle(f"{band} CSI magnitude, point-drone Tx to two moving Rx")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="|H(f)| (dB)")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    base.plt.close(fig)


def delay_doppler_from_frames(csi_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h_delay = np.fft.ifft(np.fft.fftshift(csi_matrix, axes=1), axis=1, norm="ortho")
    dd = np.fft.fftshift(np.fft.fft(h_delay, axis=0, norm="ortho"), axes=0)
    doppler_bins = (np.arange(N_FRAMES) - N_FRAMES // 2) * ((1.0 / DT_S) / N_FRAMES)
    delay_bins_ns = np.arange(N_SC) * ((1.0 / DF) / N_SC) / 1e-9
    return dd.astype(np.complex64), delay_bins_ns, doppler_bins


def save_delay_doppler_plot(path: Path, dd_by_rx: np.ndarray, delay_bins_ns: np.ndarray, doppler_bins_hz: np.ndarray, band: str) -> None:
    fig, axes = base.plt.subplots(1, 2, figsize=(12.0, 4.8), sharey=True)
    delay_mask = delay_bins_ns <= 900.0
    for rx_i, rx_name in enumerate(RX_NAMES):
        mag = np.abs(dd_by_rx[rx_i])
        mag_db = 20.0 * np.log10(np.maximum(mag / max(float(np.max(mag)), 1e-30), 1e-12))
        im = axes[rx_i].imshow(
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
        axes[rx_i].set_title(RX_LABELS[rx_name])
        axes[rx_i].set_xlabel("Delay (ns)")
    axes[0].set_ylabel("Doppler bin from 10 Hz pose sampling (Hz)")
    fig.suptitle(f"{band} low-rate delay-Doppler diagnostic from 50 explicit frames")
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Normalized magnitude (dB)")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    base.plt.close(fig)


def save_doppler_plot(path: Path, doppler_by_frame_rx: list[list[np.ndarray]], power_by_frame_rx: list[list[np.ndarray]], band: str) -> None:
    fig, ax = base.plt.subplots(figsize=(8.4, 4.8))
    colors = ["#16a34a", "#2563eb"]
    for rx_i, rx_name in enumerate(RX_NAMES):
        xs = []
        ys = []
        sizes = []
        for frame_i in range(N_FRAMES):
            doppler = doppler_by_frame_rx[frame_i][rx_i]
            power = power_by_frame_rx[frame_i][rx_i]
            if doppler.size == 0:
                continue
            xs.extend([FRAME_TIMES_S[frame_i]] * doppler.size)
            ys.extend(doppler.tolist())
            p_db = 10.0 * np.log10(np.maximum(power, 1e-30))
            sizes.extend(np.clip(p_db - np.min(p_db) + 8.0, 8.0, 34.0).tolist())
        if xs:
            ax.scatter(xs, ys, s=sizes, color=colors[rx_i], alpha=0.62, label=RX_LABELS[rx_name])
    ax.axhline(0.0, color="#334155", linewidth=0.8, alpha=0.45)
    ax.set_xlabel("Trajectory time (s)")
    ax.set_ylabel("Per-path Doppler (Hz)")
    ax.set_title(f"{band} per-path Doppler: drone Tx and two moving Rx")
    ax.grid(alpha=0.22)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    base.plt.close(fig)


def save_path_plot(path: Path, band: str, selected_lines: dict[int, list[dict[str, object]]]) -> None:
    fig, ax = base.plt.subplots(figsize=(8.8, 7.2))
    plot_scene_base(ax, f"{band} two-moving-Rx selected path geometry", zoom=True)
    colors = {"rx_mobile_east": "#16a34a", "rx_mobile_west": "#2563eb"}
    for frame_idx, lines in selected_lines.items():
        for line in lines:
            pts = np.array(line["points"], dtype=float)
            ax.plot(pts[:, 0], pts[:, 1], color=colors.get(line["rx"], "#475569"), alpha=0.35, linewidth=0.8)
    add_motion_markers(ax, show_points=True)
    ax.legend(loc="upper right", fontsize=6.5, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    base.plt.close(fig)


def render_camera_keyframes(scene, camera_dir: Path, band: str) -> dict[str, object]:
    camera_dir.mkdir(parents=True, exist_ok=True)
    camera = Camera(position=base.as_list(CAMERA_POSITION_M), look_at=base.as_list(CAMERA_LOOK_AT_M))
    cam_slug = base.coord_slug("cam", CAMERA_POSITION_M)
    outputs = []
    errors = []
    for frame_idx in RENDER_CAMERA_KEYFRAMES:
        update_frame(scene, frame_idx)
        tx_pos = DRONE_POSITIONS_M[frame_idx]
        tx_slug = base.coord_slug("tx", tx_pos)
        out_path = camera_dir / f"camera_drone_two_mobile_rx_frame_{frame_idx:03d}__{tx_slug}__{cam_slug}.png"
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
            arr = np.clip(np.asarray(bitmap)[..., :3], 0.0, 1.0)
            img = Image.fromarray((arr * 255.0).astype(np.uint8)).convert("RGB")
            draw = ImageDraw.Draw(img)
            label = (
                f"{band} frame {frame_idx:03d}  t={FRAME_TIMES_S[frame_idx]:.1f}s\n"
                f"Tx=({tx_pos[0]:.1f},{tx_pos[1]:.1f},{tx_pos[2]:.1f})\n"
                f"mobile_east=({RX_POSITIONS_M['rx_mobile_east'][frame_idx][0]:.1f},{RX_POSITIONS_M['rx_mobile_east'][frame_idx][1]:.1f},{RX_ALTITUDE_M:.1f})\n"
                f"mobile_west=({RX_POSITIONS_M['rx_mobile_west'][frame_idx][0]:.1f},{RX_POSITIONS_M['rx_mobile_west'][frame_idx][1]:.1f},{RX_ALTITUDE_M:.1f})"
            )
            draw.rectangle((8, 8, 470, 98), fill=(255, 255, 255), outline=(15, 118, 110))
            draw.text((14, 14), label, fill=(0, 0, 0))
            img.save(out_path)
            outputs.append(str(out_path))
        except Exception as exc:
            errors.append({"frame": frame_idx, "error": repr(exc)})
    gif_path = None
    if outputs:
        gif_path = camera_dir / f"camera_drone_two_mobile_rx_keyframes__{cam_slug}.gif"
        images = [Image.open(p).convert("P", palette=Image.Palette.ADAPTIVE) for p in outputs]
        images[0].save(gif_path, save_all=True, append_images=images[1:], duration=550, loop=0)
        for image in images:
            image.close()
    return {
        "camera_position_m": base.as_list(CAMERA_POSITION_M),
        "camera_look_at_m": base.as_list(CAMERA_LOOK_AT_M),
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
    ]
    for rx_name in RX_NAMES:
        label = RX_LABELS[rx_name]
        fieldnames.extend(
            [
                f"{label}_x_m",
                f"{label}_y_m",
                f"{label}_z_m",
                f"{label}_vx_mps",
                f"{label}_vy_mps",
                f"{label}_vz_mps",
                f"{label}_pathloss_db",
                f"{label}_n_paths",
            ]
        )
    fieldnames.append("solve_ms")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})


def run_band(freq_hz: float) -> dict[str, object]:
    band = base.band_name(freq_hz)
    out_dir = OUT_ROOT / band / CONDITION
    if out_dir.exists():
        shutil.rmtree(out_dir)
    images_dir = out_dir / "images"
    camera_dir = out_dir / "camera_drone_keyframes"
    out_dir.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "npz").mkdir(parents=True, exist_ok=True)

    print(f"Running {band} {CONDITION}", flush=True)
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

    for frame_idx in range(N_FRAMES):
        paths, solve_ms = solve_frame(scene, solver, frame_idx)
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

        csi_by_rx = []
        a_by_rx = []
        tau_by_rx = []
        doppler_valid_by_rx = []
        power_valid_by_rx = []
        valid_by_rx = []
        powers_by_rx = {}
        row = {
            "frame": frame_idx,
            "time_s": float(FRAME_TIMES_S[frame_idx]),
            "tx_x_m": float(DRONE_POSITIONS_M[frame_idx, 0]),
            "tx_y_m": float(DRONE_POSITIONS_M[frame_idx, 1]),
            "tx_z_m": float(DRONE_POSITIONS_M[frame_idx, 2]),
            "tx_vx_mps": float(DRONE_VELOCITIES_MPS[frame_idx, 0]),
            "tx_vy_mps": float(DRONE_VELOCITIES_MPS[frame_idx, 1]),
            "tx_vz_mps": float(DRONE_VELOCITIES_MPS[frame_idx, 2]),
            "solve_ms": float(solve_ms),
        }
        pathloss_for_print = []
        for rx_i, rx_name in enumerate(RX_NAMES):
            a_vec = np.asarray(a[rx_i, 0, 0, 0, :, 0], dtype=np.complex64)
            tau_vec = np.asarray(tau[rx_i, 0, 0, 0, :], dtype=np.float32)
            doppler_vec = np.asarray(doppler[rx_i, 0, 0, 0, :], dtype=np.float64)
            valid = tau_vec > -1.0
            power_vec = (np.abs(a_vec.astype(np.complex128)) ** 2).astype(np.float64)
            total_power = float(np.sum(power_vec[valid])) if np.any(valid) else 0.0
            pathloss_db = float(-10.0 * np.log10(total_power)) if total_power > 0.0 else float("inf")
            n_paths = int(np.count_nonzero(valid))
            label = RX_LABELS[rx_name]
            rx_pos = RX_POSITIONS_M[rx_name][frame_idx]
            rx_vel = RX_VELOCITIES_MPS[rx_name][frame_idx]
            row.update(
                {
                    f"{label}_x_m": float(rx_pos[0]),
                    f"{label}_y_m": float(rx_pos[1]),
                    f"{label}_z_m": float(rx_pos[2]),
                    f"{label}_vx_mps": float(rx_vel[0]),
                    f"{label}_vy_mps": float(rx_vel[1]),
                    f"{label}_vz_mps": float(rx_vel[2]),
                    f"{label}_pathloss_db": pathloss_db,
                    f"{label}_n_paths": n_paths,
                }
            )
            csi_by_rx.append(np.asarray(h[rx_i, 0, 0, 0, 0, :], dtype=np.complex64))
            a_by_rx.append(a_vec)
            tau_by_rx.append(tau_vec)
            doppler_valid_by_rx.append(doppler_vec[valid])
            power_valid_by_rx.append(power_vec[valid])
            valid_by_rx.append(valid)
            powers_by_rx[rx_i] = power_vec
            pathloss_for_print.append(f"{label}={pathloss_db:.2f}dB/{n_paths}p")

        if frame_idx in {0, 25, 49}:
            selected_lines[frame_idx] = path_lines(paths, powers_by_rx)
        csi_rows.append(np.stack(csi_by_rx, axis=0))
        a_frames.append(a_by_rx)
        tau_frames.append(tau_by_rx)
        doppler_frames.append(doppler_valid_by_rx)
        power_frames.append(power_valid_by_rx)
        valid_frames.append(valid_by_rx)
        frame_rows.append(row)
        print(f"  {band} frame {frame_idx + 1:02d}/{N_FRAMES}: " + ", ".join(pathloss_for_print) + f", solve={solve_ms:.1f} ms", flush=True)

    csi_matrix = np.stack(csi_rows, axis=0)
    csi = np.transpose(csi_matrix, (1, 0, 2)).reshape(len(RX_NAMES), 1, 1, 1, N_FRAMES, N_SC).astype(np.complex64)
    pathloss_db = np.zeros((N_FRAMES, len(RX_NAMES), 1), dtype=np.float64)
    n_paths = np.zeros((N_FRAMES, len(RX_NAMES)), dtype=np.int32)
    for frame_idx, row in enumerate(frame_rows):
        for rx_i, rx_name in enumerate(RX_NAMES):
            label = RX_LABELS[rx_name]
            pathloss_db[frame_idx, rx_i, 0] = row[f"{label}_pathloss_db"]
            n_paths[frame_idx, rx_i] = row[f"{label}_n_paths"]
    solve_ms = np.array([row["solve_ms"] for row in frame_rows], dtype=np.float64)

    max_paths = max((len(a_frames[i][rx_i]) for i in range(N_FRAMES) for rx_i in range(len(RX_NAMES))), default=0)
    a_by_frame = np.zeros((N_FRAMES, len(RX_NAMES), 1, 1, 1, max_paths, 1), dtype=np.complex64)
    tau_by_frame = np.full((N_FRAMES, len(RX_NAMES), 1, 1, 1, max_paths), -1.0, dtype=np.float32)
    doppler_by_frame = np.zeros((N_FRAMES, len(RX_NAMES), 1, 1, 1, max_paths), dtype=np.float64)
    path_power_linear = np.zeros((N_FRAMES, len(RX_NAMES), 1, max_paths), dtype=np.float64)
    valid_path_mask = np.zeros((N_FRAMES, len(RX_NAMES), 1, 1, 1, max_paths), dtype=bool)
    for frame_i in range(N_FRAMES):
        for rx_i in range(len(RX_NAMES)):
            n = len(a_frames[frame_i][rx_i])
            valid = valid_frames[frame_i][rx_i]
            a_by_frame[frame_i, rx_i, 0, 0, 0, :n, 0] = a_frames[frame_i][rx_i]
            tau_by_frame[frame_i, rx_i, 0, 0, 0, :n] = tau_frames[frame_i][rx_i]
            full_doppler = np.zeros(n, dtype=np.float64)
            full_doppler[valid] = doppler_frames[frame_i][rx_i]
            doppler_by_frame[frame_i, rx_i, 0, 0, 0, :n] = full_doppler
            path_power_linear[frame_i, rx_i, 0, :n] = np.abs(a_frames[frame_i][rx_i].astype(np.complex128)) ** 2
            valid_path_mask[frame_i, rx_i, 0, 0, 0, :n] = valid

    dd_by_rx = []
    for rx_i in range(len(RX_NAMES)):
        dd, delay_bins_ns, doppler_bins_hz = delay_doppler_from_frames(csi_matrix[:, rx_i, :])
        dd_by_rx.append(dd)
    dd_by_rx = np.stack(dd_by_rx, axis=0)

    images = save_motion_images(images_dir, band)
    save_path_plot(out_dir / "paths.png", band, selected_lines)
    save_csi_plot(out_dir / "csi.png", csi_matrix, band)
    save_pathloss_plot(out_dir / "pathloss.png", pathloss_db, band)
    save_doppler_plot(out_dir / "doppler.png", doppler_frames, power_frames, band)
    save_delay_doppler_plot(out_dir / "delay_doppler_from_frames.png", dd_by_rx, delay_bins_ns, doppler_bins_hz, band)
    camera_outputs = render_camera_keyframes(scene, camera_dir, band)

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
        "rx_names": np.array(RX_NAMES),
        "rx_positions_m": np.stack([RX_POSITIONS_M[name] for name in RX_NAMES], axis=0).astype(np.float64),
        "rx_velocity_mps": np.stack([RX_VELOCITIES_MPS[name] for name in RX_NAMES], axis=0).astype(np.float64),
        "camera_position_m": CAMERA_POSITION_M.astype(np.float64),
        "camera_look_at_m": CAMERA_LOOK_AT_M.astype(np.float64),
        "freqs": SUBCARRIER_FREQS,
        "delay_doppler_from_frames": dd_by_rx,
        "delay_bins_ns": delay_bins_ns,
        "doppler_bins_hz_from_frame_sampling": doppler_bins_hz,
    }
    np.savez_compressed(data_path, **data_arrays)
    npz_copy = OUT_ROOT / "npz" / f"{band}__{CONDITION}.npz"
    np.savez_compressed(npz_copy, **data_arrays)

    rx_summaries = {}
    for rx_i, rx_name in enumerate(RX_NAMES):
        rx_summaries[RX_LABELS[rx_name]] = {
            "name": rx_name,
            "role": "moving point receiver",
            "start_position_m": base.as_list(RX_POSITIONS_M[rx_name][0]),
            "end_position_m": base.as_list(RX_POSITIONS_M[rx_name][-1]),
            "start_velocity_mps": base.as_list(RX_VELOCITIES_MPS[rx_name][0]),
            "end_velocity_mps": base.as_list(RX_VELOCITIES_MPS[rx_name][-1]),
            "nominal_speed_mps": float(np.linalg.norm(RX_VELOCITIES_MPS[rx_name][0])),
            "pathloss_db_min": float(np.nanmin(pathloss_db[:, rx_i, 0])),
            "pathloss_db_max": float(np.nanmax(pathloss_db[:, rx_i, 0])),
            "pathloss_db_mean": float(np.nanmean(pathloss_db[:, rx_i, 0])),
            "n_paths_min": int(np.min(n_paths[:, rx_i])),
            "n_paths_max": int(np.max(n_paths[:, rx_i])),
        }

    summary = {
        "scene": "etoile",
        "scene_xml": str(SCENE_XML),
        "frequency_hz": freq_hz,
        "band": band,
        "condition": CONDITION,
        "modeling_assumption": "Moving drone and receivers are point devices; no physical drone/Rx mesh scattering or blockage is modeled.",
        "tx": {
            "name": TX_NAME,
            "role": "moving point drone transmitter",
            "start_position_m": base.as_list(DRONE_POSITIONS_M[0]),
            "end_position_m": base.as_list(DRONE_POSITIONS_M[-1]),
            "nominal_speed_mps": float(DRONE_NOMINAL_SPEED_MPS),
        },
        "rx": rx_summaries,
        "moving_rx_design": RX_TRACK_DESIGN,
        "fixed_camera": {
            "role": "visualization camera near the original rooftop Rx",
            "position_m": base.as_list(CAMERA_POSITION_M),
            "look_at_m": base.as_list(CAMERA_LOOK_AT_M),
            "note": "Camera is for visual artifacts only; it is not a radio node in this condition.",
        },
        "trajectory": {
            "dt_s": DT_S,
            "num_frames": N_FRAMES,
            "times_s": FRAME_TIMES_S.tolist(),
            "drone_control_points_m": DRONE_PATH_CONTROL_M.tolist(),
            "drone_path_length_m": float(DRONE_PATH_LENGTH_M),
            "drone_nominal_speed_mps": float(DRONE_NOMINAL_SPEED_MPS),
            "position_sampling_note": "50 samples at 0.1 s cover t=0.0 through 4.9 s, a nominal 5 s capture window.",
        },
        "solver": SOLVER_SETTINGS,
        "arrays": {
            "pathloss_db_shape": list(pathloss_db.shape),
            "pathloss_db_indexing": "pathloss_db[frame, rx, tx]",
            "csi_shape": list(csi.shape),
            "csi_indexing": "csi[rx, rx_ant, tx, tx_ant, explicit_frame, subcarrier]",
            "csi_explicit_frames_shape": list(csi_matrix.shape),
            "csi_explicit_frames_indexing": "csi_explicit_frames[frame, rx, subcarrier]",
            "a_by_frame_shape": list(a_by_frame.shape),
            "tau_by_frame_shape": list(tau_by_frame.shape),
            "doppler_hz_by_frame_shape": list(doppler_by_frame.shape),
            "rx_positions_m_shape": [len(RX_NAMES), N_FRAMES, 3],
        },
        "data_considerations": [
            "The two moving receivers are point receivers at 42 m altitude, placed to stay visible in the fixed camera and bird's-eye scene geometry.",
            "Both receivers move along reverse, laterally offset copies of the drone trajectory, so the receiver motion is nominally opposite the drone motion.",
            "PathSolver is rerun for every explicit trajectory frame with both Tx and Rx positions and velocities updated.",
            "The 0.1 s frame interval is for pose/image sampling; physical per-path Doppler is saved separately.",
        ],
        "frame_summary": {
            "solve_ms_mean": float(np.mean(solve_ms)),
            "rx": rx_summaries,
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
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = {base.band_name(freq_hz): run_band(freq_hz) for freq_hz in FREQS_HZ}
    summary_path = OUT_ROOT / "summary_two_mobile_rx.json"
    summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
