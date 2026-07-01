#!/usr/bin/env python3
"""Condensed pipeline for the uploaded custom scene, with street-canyon-like outputs."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import sys
import time
from pathlib import Path
from xml.etree import ElementTree as ET

import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sionna
from matplotlib.patches import Rectangle
from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene
from sionna.rt.utils import subcarrier_frequencies

TOOLS_DIR = Path(__file__).resolve().parent
if str(TOOLS_DIR) not in sys.path:
    sys.path.append(str(TOOLS_DIR))

from run_xml_rt_quick import _ensure_radio_materials, _flatten_mesh_groups, _prepare_scene_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run condensed custom-scene RT pipeline.")
    p.add_argument(
        "--scene",
        default="test_scenes/meshes/normalized_scene_sionna_materials_flat_shapes.xml",
        help="XML scene path.",
    )
    p.add_argument("--scene-name", default="uploaded_scene", help="Case folder name.")
    p.add_argument("--out-root", default="test_scenes/uploaded_scene_smoke/scene_map", help="Output root.")
    p.add_argument("--tx", nargs=3, type=float, default=(0.0, 0.0, 2.0), help="Tx x y z.")
    p.add_argument("--rx", nargs=3, type=float, default=(20.0, 0.0, 2.0), help="Rx x y z.")
    p.add_argument("--max-depth", type=int, default=1, help="PathSolver max depth.")
    p.add_argument("--sampling-frequency", type=float, default=122.88e6, help="CIR sampling frequency.")
    p.add_argument("--num-subcarriers", type=int, default=64, help="CFR subcarrier count.")
    p.add_argument("--subcarrier-spacing", type=float, default=30e3, help="Subcarrier spacing.")
    p.add_argument("--num-time-steps", type=int, default=64, help="CFR time steps.")
    p.add_argument("--motion-steps", type=int, default=7, help="Number of animation frames.")
    p.add_argument("--motion-distance", type=float, default=8.0, help="Total Tx/Rx displacement (m).")
    return p.parse_args()


def band_name(freq_hz: float) -> str:
    ghz = freq_hz / 1e9
    if abs(ghz - round(ghz)) < 1e-9:
        return f"{int(round(ghz))}GHz"
    return f"{str(ghz).replace('.', 'p')}GHz"


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
        props = []
        in_vertex = False
        n_vertices = 0
        while True:
            line = f.readline().decode("ascii", "ignore").strip()
            if line.startswith("element "):
                _, name, n = line.split()
                in_vertex = name == "vertex"
                if in_vertex:
                    n_vertices = int(n)
            elif in_vertex and line.startswith("property "):
                t, name, *_ = line.split()
                props.append((t, name))
            if line == "end_header":
                break

        stride = sum(size_map.get(t, 4) for t, _ in props if t != "list")
        mn = np.array([math.inf, math.inf, math.inf], dtype=np.float64)
        mx = np.array([-math.inf, -math.inf, -math.inf], dtype=np.float64)
        for _ in range(n_vertices):
            data = f.read(stride)
            if len(data) < 12:
                break
            xyz = np.array(struct.unpack_from("<fff", data, 0), dtype=np.float64)
            mn = np.minimum(mn, xyz)
            mx = np.maximum(mx, xyz)
        if not np.isfinite(mn).all():
            raise ValueError(f"No vertices in {path}")
        return mn, mx


def collect_shape_bounds(scene_xml: Path) -> list[dict[str, object]]:
    mesh_root = scene_xml.parent
    root = ET.parse(scene_xml).getroot()
    out = []
    for shape in root.findall("shape"):
        if shape.attrib.get("type") != "ply":
            continue
        sid = shape.attrib.get("id", "shape")
        filename_node = shape.find("string[@name='filename']")
        if filename_node is None:
            continue
        filename = filename_node.attrib.get("value")
        if not filename:
            continue
        p = (mesh_root / filename).resolve()
        if not p.exists():
            continue
        ref = shape.find("ref[@name='bsdf']")
        bsdf_id = ref.attrib.get("id") if ref is not None else ""
        bsdf_key = str(bsdf_id).lower()
        sid_l = sid.lower()
        fn_l = filename.lower()
        if "building" in sid_l or "roof" in sid_l or "building" in fn_l or "roof" in bsdf_key:
            kind = "building"
        elif "road" in sid_l or "road" in fn_l or "paths" in sid_l or "paths" in fn_l:
            kind = "road"
        else:
            kind = "other"
        try:
            mn, mx = read_ply_bounds(p)
            out.append({"id": sid, "kind": kind, "min": mn, "max": mx})
        except Exception:
            continue
    return out


def style_for_kind(kind: str) -> tuple[str, str]:
    if kind == "building":
        return "#bcbcbc", "#8d8d8d"
    if kind == "road":
        return "#efe2cb", "#a58a4f"
    return "#d0d0d0", "#9a9a9a"


def bbox_distance_sq_to_point(point: np.ndarray, mn: np.ndarray, mx: np.ndarray) -> float:
    point = np.asarray(point[:2], dtype=float)
    mn2 = np.asarray(mn[:2], dtype=float)
    mx2 = np.asarray(mx[:2], dtype=float)
    d = np.maximum(np.maximum(mn2 - point, 0.0), point - mx2)
    return float(np.dot(d, d))


def select_local_bounds(
    bounds: list[dict[str, object]],
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    padding: float,
    *,
    max_shapes: int = 24,
) -> list[dict[str, object]]:
    if not bounds:
        return []

    focus = 0.5 * (tx_pos[:2] + rx_pos[:2])
    ordered = sorted(
        bounds,
        key=lambda b: bbox_distance_sq_to_point(focus, b["min"], b["max"]),
    )
    candidates = [
        b for b in ordered
        if bbox_distance_sq_to_point(focus, b["min"], b["max"]) <= (padding + 10.0) ** 2
    ]

    if not candidates:
        return []

    return candidates[:max_shapes]


def compute_plot_box(
    bounds: list[dict[str, object]],
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    lines: list[dict[str, object]],
    padding: float,
) -> tuple[float, float, float, float]:
    x0 = float(min(tx_pos[0], rx_pos[0]) - padding)
    y0 = float(min(tx_pos[1], rx_pos[1]) - padding)
    x1 = float(max(tx_pos[0], rx_pos[0]) + padding)
    y1 = float(max(tx_pos[1], rx_pos[1]) + padding)

    for line in lines:
        pts = np.asarray(line.get("points", []), dtype=float)
        if pts.ndim >= 2 and pts.shape[0] >= 2:
            x0 = min(x0, float(pts[:, 0].min()) - padding)
            y0 = min(y0, float(pts[:, 1].min()) - padding)
            x1 = max(x1, float(pts[:, 0].max()) + padding)
            y1 = max(y1, float(pts[:, 1].max()) + padding)

    if not np.isfinite([x0, y0, x1, y1]).all():
        return -20.0, -20.0, 20.0, 20.0
    if x1 <= x0:
        x0 -= padding
        x1 += padding
    if y1 <= y0:
        y0 -= padding
        y1 += padding
    return x0, y0, x1, y1


def add_scene_rectangles(ax, bounds: list[dict[str, object]]) -> tuple[float, float, float, float]:
    x0 = math.inf
    y0 = math.inf
    x1 = -math.inf
    y1 = -math.inf
    for b in bounds:
        mn = b["min"]
        mx = b["max"]
        x, y = float(mn[0]), float(mn[1])
        w, h = float(mx[0] - mn[0]), float(mx[1] - mn[1])
        face, edge = style_for_kind(b["kind"])
        ax.add_patch(Rectangle((x, y), w, h, facecolor=face, edgecolor=edge, linewidth=0.4, alpha=0.8))
        x0 = min(x0, x)
        y0 = min(y0, y)
        x1 = max(x1, x + w)
        y1 = max(y1, y + h)
    return x0, y0, x1, y1


def path_points_from_solver(paths) -> list[dict[str, object]]:
    valid = np.asarray(paths.valid.numpy(), dtype=bool)
    if valid.ndim >= 5:
        valid = valid[:, 0, :, 0, :]
    valid = np.squeeze(valid)
    if valid.ndim == 0:
        valid = np.array([bool(valid)])

    sources = np.asarray(paths.sources.numpy(), dtype=float).T
    targets = np.asarray(paths.targets.numpy(), dtype=float).T
    max_rx, max_tx = sources.shape[0], 1 if targets.shape[0] == 1 else sources.shape[0]
    if valid.ndim == 1:
        valid = valid.reshape(max_rx, max_tx, -1)

    lines: list[dict[str, object]] = []
    for rx_i in range(min(valid.shape[0], sources.shape[0])):
        for tx_i in range(min(valid.shape[1], targets.shape[0])):
            for p_i in range(valid.shape[2]):
                if not valid[rx_i, tx_i, p_i]:
                    continue
                lines.append({"tx_i": int(tx_i), "rx_i": int(rx_i), "points": [sources[tx_i].tolist(), targets[rx_i].tolist()]})
    if not lines:
        lines = [{"tx_i": 0, "rx_i": 0, "points": [sources[0].tolist(), targets[0].tolist()]}]
    return lines


def save_paths_plot(
    path: Path,
    freq_hz: float,
    lines: list[dict[str, object]],
    bounds: list[dict[str, object]],
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    plot_box: tuple[float, float, float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    if bounds:
        add_scene_rectangles(ax, bounds)
    x0, y0, x1, y1 = plot_box
    for line in lines:
        pts = np.asarray(line["points"], dtype=float)
        ax.plot(pts[:, 0], pts[:, 1], color="#1f77b4", linewidth=1.3)
    ax.scatter([tx_pos[0]], [tx_pos[1]], marker="^", s=90, c="#1f4e79", edgecolor="white", linewidth=0.8, label="tx")
    ax.scatter([rx_pos[0]], [rx_pos[1]], marker="o", s=78, c="#222222", edgecolor="white", linewidth=0.8, label="rx")
    ax.set_title(f"{band_name(freq_hz)} link paths")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    margin = 2.0
    ax.set_xlim(x0 - margin, x1 + margin)
    ax.set_ylim(y0 - margin, y1 + margin)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _normalize_xy_points(values: np.ndarray | dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    if isinstance(values, np.ndarray):
        return {"tx": np.asarray(values, dtype=float)}
    normalized: dict[str, np.ndarray] = {}
    for name, val in values.items():
        normalized[str(name)] = np.asarray(val, dtype=float)
    return normalized


def save_scene_map(
    path: Path,
    freq_hz: float,
    bounds: list[dict[str, object]],
    tx_pos: np.ndarray | dict[str, np.ndarray],
    rx_pos: np.ndarray | dict[str, np.ndarray],
    lines: list[dict[str, object]],
    plot_box: tuple[float, float, float, float],
    show_motion: bool = False,
    tx_end: np.ndarray | dict[str, np.ndarray] | None = None,
    rx_end: np.ndarray | dict[str, np.ndarray] | None = None,
    tx_start: np.ndarray | dict[str, np.ndarray] | None = None,
    rx_start: np.ndarray | dict[str, np.ndarray] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    if bounds:
        add_scene_rectangles(ax, bounds)
    x0, y0, x1, y1 = plot_box
    for line in lines:
        pts = np.asarray(line["points"], dtype=float)
        ax.plot(pts[:, 0], pts[:, 1], color="#666", linewidth=1.0, alpha=0.55)

    tx_points = _normalize_xy_points(tx_pos)
    rx_points = _normalize_xy_points(rx_pos)
    for name, point in tx_points.items():
        ax.scatter([point[0]], [point[1]], marker="^", s=85, c="#1f4e79", edgecolor="white", linewidth=0.8, label=f"tx: {name}")
    for name, point in rx_points.items():
        ax.scatter([point[0]], [point[1]], marker="o", s=75, c="#222222", edgecolor="white", linewidth=0.8, label=f"rx: {name}")

    if show_motion and tx_end is not None and rx_end is not None:
        tx_points_end = _normalize_xy_points(tx_end)
        rx_points_end = _normalize_xy_points(rx_end)
        tx_points_start = tx_points if tx_start is None else _normalize_xy_points(tx_start)
        rx_points_start = rx_points if rx_start is None else _normalize_xy_points(rx_start)

        for name, end_pos in tx_points_end.items():
            start = tx_points_start.get(name, next(iter(tx_points.values())))
            ax.arrow(
                float(start[0]),
                float(start[1]),
                float(end_pos[0] - start[0]),
                float(end_pos[1] - start[1]),
                width=0.06,
                head_width=0.4,
                length_includes_head=True,
                color="#1f4e79",
                alpha=0.9,
            )
        for name, end_pos in rx_points_end.items():
            start = rx_points_start.get(name, next(iter(rx_points.values())))
            ax.arrow(
                float(start[0]),
                float(start[1]),
                float(end_pos[0] - start[0]),
                float(end_pos[1] - start[1]),
                width=0.06,
                head_width=0.4,
                length_includes_head=True,
                color="#222222",
                alpha=0.9,
            )

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles=handles, labels=labels, loc="upper right", fontsize=7)
    ax.set_title(f"{band_name(freq_hz)} scene map")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    margin = 2.0
    ax.set_xlim(x0 - margin, x1 + margin)
    ax.set_ylim(y0 - margin, y1 + margin)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_link_paths_map(
    path: Path,
    freq_hz: float,
    lines: list[dict[str, object]],
    bounds: list[dict[str, object]],
    plot_box: tuple[float, float, float, float],
) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    if bounds:
        add_scene_rectangles(ax, bounds)
    x0, y0, x1, y1 = plot_box
    for idx, line in enumerate(lines):
        pts = np.asarray(line["points"], dtype=float)
        ax.plot(pts[:, 0], pts[:, 1], marker="o", markersize=2, linewidth=1.5, label=f"path_{idx}")
    ax.set_title(f"{band_name(freq_hz)} link path geometry")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(x0 - 2, x1 + 2)
    ax.set_ylim(y0 - 2, y1 + 2)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_motion_series(
    out_dir: Path,
    bounds: list[dict[str, object]],
    lines: list[dict[str, object]],
    freq_hz: float,
    tx_pos: np.ndarray,
    rx_pos: np.ndarray,
    tx_delta: np.ndarray,
    rx_delta: np.ndarray,
    n_frames: int,
    plot_box: tuple[float, float, float, float],
) -> None:
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for i in range(max(1, n_frames)):
        alpha = i / max(1, n_frames - 1)
        cur_tx = tx_pos + alpha * tx_delta
        cur_rx = rx_pos + alpha * rx_delta
        frame = img_dir / f"motion_frame_{i:02d}.png"
        frame_lines = [{"tx_i": 0, "rx_i": 0, "points": [cur_tx.tolist(), cur_rx.tolist()]}]
        save_scene_map(
            frame,
            freq_hz,
            bounds,
            cur_tx,
            cur_rx,
            frame_lines,
            plot_box,
            show_motion=True,
            tx_end=cur_tx,
            rx_end=cur_rx,
            tx_start=tx_pos,
            rx_start=rx_pos,
        )
        frames.append(frame)
    if frames:
        images = [Image.open(p).convert("P", palette=Image.Palette.ADAPTIVE) for p in frames]
        images[0].save(img_dir / "motion.gif", save_all=True, append_images=images[1:], duration=450, loop=0)


def delay_doppler(h_time_freq: np.ndarray, sampling_frequency: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h_delay = np.fft.ifft(np.fft.fftshift(h_time_freq, axes=1), axis=1, norm="ortho")
    dd = np.fft.fftshift(np.fft.fft(h_delay, axis=0, norm="ortho"), axes=0)
    delay_ns = np.arange(h_time_freq.shape[1]) * ((1.0 / sampling_frequency) / h_time_freq.shape[1]) / 1e-9
    doppler_hz = (np.arange(h_time_freq.shape[0]) - h_time_freq.shape[0] // 2) * (sampling_frequency / h_time_freq.shape[0])
    return dd.astype(np.complex64), delay_ns.astype(float), doppler_hz.astype(float)


def save_doppler_plot(path: Path, h_time_freq: np.ndarray, tau_s: np.ndarray, doppler_hz: np.ndarray, title: str, sample_freq: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dd, delay_bins_ns, doppler_bins = delay_doppler(h_time_freq, sample_freq)
    mag_db = 20.0 * np.log10(np.maximum(np.abs(dd), np.finfo(np.float32).tiny))
    mag_db = mag_db - np.max(mag_db) if mag_db.size else mag_db
    tau_ns = tau_s / 1e-9 if tau_s.size else np.array([])
    delay_mask = delay_bins_ns <= max(700.0, float(np.max(tau_ns) + 150.0) if tau_ns.size else 700.0)
    doppler_max = float(np.max(np.abs(doppler_hz))) if doppler_hz.size else 0.0
    if doppler_max == 0.0:
        doppler_mask = np.ones_like(doppler_bins, dtype=bool)
    else:
        doppler_mask = np.abs(doppler_bins) <= max(350.0, 1.35 * max(1.0, doppler_max))
    fig, ax = plt.subplots(figsize=(7.6, 4.9))
    if mag_db.size and np.any(delay_mask) and np.any(doppler_mask):
        im = ax.imshow(
            mag_db[np.ix_(doppler_mask, delay_mask)],
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=-55.0,
            vmax=0.0,
            extent=[
                float(delay_bins_ns[delay_mask][0]),
                float(delay_bins_ns[delay_mask][-1]),
                float(doppler_bins[doppler_mask][0]),
                float(doppler_bins[doppler_mask][-1]),
            ],
        )
        if tau_ns.size:
            ax.scatter(tau_ns, doppler_hz, marker="x", color="#58f3ff", s=28, linewidths=1.0)
        fig.colorbar(im, ax=ax, label="Normalized magnitude (dB)")
        ax.set_xlabel("Delay (ns)")
        ax.set_ylabel("Doppler (Hz)")
        ax.set_title(title)
    else:
        ax.text(0.5, 0.5, "No Doppler content", ha="center", va="center")
    ax.grid(alpha=0.14)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return dd, delay_bins_ns, doppler_bins


def save_csi_plot(path: Path, h_freq: np.ndarray, pathloss_db: float) -> None:
    mag_db = 20.0 * np.log10(np.maximum(np.abs(h_freq), 1e-30))
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    ax.plot(mag_db, linewidth=1.2)
    ax.set_xlabel("Subcarrier index")
    ax.set_ylabel("|H(f)| (dB)")
    ax.set_title(f"CSI magnitude ({pathloss_db:.1f} dB pathloss)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_path_power_plot(path: Path, power: np.ndarray) -> None:
    power = np.asarray(power, dtype=np.float64).reshape(-1)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    if power.size:
        ax.stem(np.arange(power.size), power)
        ax.set_title("Path powers")
        ax.set_xlabel("Path index")
        ax.set_ylabel("Power")
        ax.set_yscale("log")
    else:
        ax.text(0.1, 0.5, "No paths found")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_for_frequency(
    scene_xml: Path,
    tx: np.ndarray,
    rx: np.ndarray,
    freq_hz: float,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, object]:
    scene = load_scene(str(scene_xml), remove_duplicate_vertices=False)
    scene.frequency = float(freq_hz)
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5, horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.add(Transmitter(name="tx", position=tx, orientation=[0, 0, 0], power_dbm=0.0))
    scene.add(Receiver(name="rx", position=rx, orientation=[0, 0, 0]))

    solver = PathSolver()
    t0 = time.perf_counter()
    paths = solver(
        scene=scene,
        max_depth=args.max_depth,
        los=True,
        specular_reflection=True,
        diffuse_reflection=True,
        diffraction=False,
        edge_diffraction=False,
        refraction=False,
        synthetic_array=False,
    )
    solve_ms = (time.perf_counter() - t0) * 1e3

    a, tau = paths.cir(
        sampling_frequency=args.sampling_frequency,
        normalize_delays=False,
        out_type="numpy",
    )
    a0 = np.asarray(a[0, 0, 0, 0, :, 0], dtype=np.complex128).reshape(-1)
    tau0 = np.asarray(tau[0, 0, 0, 0, :], dtype=np.float64).reshape(-1)
    power0 = np.abs(a0) ** 2
    path_gain = float(np.sum(power0))
    pathloss_db = float(-10.0 * np.log10(path_gain)) if path_gain > 0 else float("inf")
    rx_power_dbm = -pathloss_db if np.isfinite(pathloss_db) else float("-inf")
    if not np.isfinite(pathloss_db) or path_gain <= 0.0:
        print(
            f"[warn] No finite path gain for {band_name(freq_hz)}: tx={tx.tolist()}, rx={rx.tolist()}, path_gain={path_gain}"
        )

    subcarriers = subcarrier_frequencies(args.num_subcarriers, args.subcarrier_spacing)
    csi = paths.cfr(
        frequencies=subcarriers,
        sampling_frequency=args.subcarrier_spacing,
        num_time_steps=args.num_time_steps,
        normalize_delays=False,
        normalize=False,
        out_type="numpy",
    ).astype(np.complex64)

    h_link = np.asarray(csi[0, 0, 0, 0, :, :], dtype=np.complex64)
    doppler = np.asarray(paths.doppler.numpy(), dtype=float).reshape(-1)

    bounds = collect_shape_bounds(scene_xml)
    lines = path_points_from_solver(paths)

    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    tx_pos = tx.astype(float)
    rx_pos = rx.astype(float)
    tx_end = tx_pos + np.array([args.motion_distance, 0.0, 0.0], dtype=float)
    rx_end = rx_pos - np.array([args.motion_distance, 0.0, 0.0], dtype=float)
    tx_delta = (tx_end - tx_pos) / max(1, args.motion_steps - 1)
    rx_delta = (rx_end - rx_pos) / max(1, args.motion_steps - 1)

    local_bounds = select_local_bounds(bounds, tx_pos, rx_pos, padding=22.0)
    plot_box = compute_plot_box(local_bounds, tx_pos, rx_pos, lines, padding=8.0)
    save_paths_plot(out_dir / "paths.png", freq_hz, lines, local_bounds, tx_pos, rx_pos, plot_box)
    save_link_paths_map(images_dir / "link_paths_map.png", freq_hz, lines, local_bounds, plot_box)
    save_scene_map(
        images_dir / "scene_map.png",
        freq_hz,
        local_bounds,
        tx_pos,
        rx_pos,
        lines,
        plot_box,
    )
    save_scene_map(
        images_dir / "motion_map.png",
        freq_hz,
        local_bounds,
        tx_pos,
        rx_pos,
        lines,
        plot_box,
        show_motion=True,
        tx_end=tx_end,
        rx_end=rx_end,
        tx_start=tx_pos,
        rx_start=rx_pos,
    )
    save_motion_series(
        out_dir,
        local_bounds,
        lines,
        freq_hz,
        tx_pos,
        rx_pos,
        tx_delta,
        rx_delta,
        args.motion_steps,
        plot_box,
    )

    dd, delay_bins_ns, doppler_bins = save_doppler_plot(
        out_dir / "doppler.png",
        h_link,
        tau0,
        doppler,
        f"{band_name(freq_hz)} delay-Doppler",
        args.subcarrier_spacing,
    )
    save_csi_plot(out_dir / "csi.png", h_link[0], pathloss_db)
    save_path_power_plot(out_dir / "path_powers.png", power0)

    npz_path = out_dir / "rt_data.npz"
    np.savez_compressed(
        npz_path,
        a=a,
        tau=tau,
        csi=csi,
        freqs=subcarriers,
        doppler_hz=doppler,
        delay_doppler=dd,
        delay_bins_ns=delay_bins_ns,
        doppler_bins_hz=doppler_bins,
        frequency_hz=np.array([freq_hz], dtype=float),
        tx=tx.astype(np.float32),
        rx=rx.astype(np.float32),
        tx_power_dbm=np.array([0.0]),
        pathloss_db=np.array([pathloss_db]),
        rx_power_dbm=np.array([rx_power_dbm]),
        solve_ms=np.array([solve_ms]),
    )

    summary = {
        "frequency_hz": float(freq_hz),
        "solve_ms": float(solve_ms),
        "path_count": int(power0.size),
        "pathloss_db": pathloss_db,
        "rx_power_dbm": rx_power_dbm,
        "total_power": float(path_gain),
        "tx": tx.tolist(),
        "rx": rx.tolist(),
        "outputs": {
            "paths": str(out_dir / "paths.png"),
            "doppler": str(out_dir / "doppler.png"),
            "csi": str(out_dir / "csi.png"),
            "path_powers": str(out_dir / "path_powers.png"),
            "rt_data": str(npz_path),
            "images": str(images_dir),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary | {"npz": str(npz_path)}


def main() -> None:
    args = parse_args()
    os.environ.setdefault("DRJIT_LIBLLVM_PATH", "/usr/lib/x86_64-linux-gnu/libLLVM-14.so")

    scene_in = Path(args.scene)
    prepared = _prepare_scene_path(scene_in, "", Path("test_scenes"))
    prepared, n_conv = _ensure_radio_materials(prepared)
    prepared, n_flat = _flatten_mesh_groups(prepared)
    if n_conv:
        print(f"[info] converted {n_conv} materials")
    if n_flat:
        print(f"[info] flattened {n_flat} shapes")

    out_root = Path(args.out_root)
    out_base = out_root / args.scene_name
    out_base.mkdir(parents=True, exist_ok=True)
    npz_root = out_root / "npz"
    npz_root.mkdir(parents=True, exist_ok=True)

    tx = np.array(args.tx, dtype=np.float64)
    rx = np.array(args.rx, dtype=np.float64)
    bands = [5e9, 28e9]
    all_summary = {}
    for freq in bands:
        bname = band_name(freq)
        bdir = out_base / bname
        bdir.mkdir(parents=True, exist_ok=True)
        summary = run_for_frequency(prepared, tx, rx, freq, args, bdir)
        npz_copy = npz_root / f"{bname}__{args.scene_name}.npz"
        shutil.copyfile(summary["npz"], npz_copy)
        summary["npz_copy"] = str(npz_copy)
        all_summary[bname] = summary

    (out_root / "summary.json").write_text(json.dumps(all_summary, indent=2), encoding="utf-8")
    print(json.dumps(all_summary, indent=2))


if __name__ == "__main__":
    main()
