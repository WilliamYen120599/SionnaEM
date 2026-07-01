#!/usr/bin/env python3
"""Rooftop-BS street-canyon mobility experiment and visual export."""

from __future__ import annotations

import json
import math
import struct
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sionna
from sionna.rt import Camera, PathSolver, PlanarArray, Receiver, Transmitter, load_scene
from sionna.rt.utils import subcarrier_frequencies


OUT = Path("Blender Preview Images/mobility/rooftop_bs")
SCENE_XML = Path(sionna.rt.scene.simple_street_canyon_with_cars)
MESH_DIR = SCENE_XML.parent / "meshes"
FREQ = 3.5e9
FS = 122.88e6
N_SC = 512
DF = 30e3
FREQS = subcarrier_frequencies(N_SC, DF)
CAR_VELOCITY = 10.0


def read_ply_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    size_map = {
        "float": 4, "float32": 4, "double": 8, "float64": 8,
        "uchar": 1, "uint8": 1, "char": 1, "int8": 1,
        "int": 4, "int32": 4, "uint": 4, "uint32": 4,
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
    out = {}
    for p in sorted(MESH_DIR.glob("*.ply")):
        if p.stem.startswith(("building_", "car_", "floor")):
            mn, mx = read_ply_bounds(p)
            out[p.stem] = {
                "min": mn.tolist(),
                "max": mx.tolist(),
                "center": ((mn + mx) / 2.0).tolist(),
            }
    return out


BOUNDS = mesh_bounds()


def roof_position(building: str, dz: float = 1.5) -> list[float]:
    b = BOUNDS[building]
    mn = np.array(b["min"])
    mx = np.array(b["max"])
    return [float((mn[0] + mx[0]) / 2.0), float((mn[1] + mx[1]) / 2.0), float(mx[2] + dz)]


def car_position(car: str, dz: float = 0.55) -> list[float]:
    b = BOUNDS[car]
    mn = np.array(b["min"])
    mx = np.array(b["max"])
    return [float((mn[0] + mx[0]) / 2.0), float((mn[1] + mx[1]) / 2.0), float(mx[2] + dz)]


def configs() -> dict[str, dict[str, object]]:
    receivers = [
        {"name": "rx_car_3", "position": car_position("car_3"), "velocity": [-CAR_VELOCITY, 0, 0], "host": "car_3"},
        {"name": "rx_car_6", "position": car_position("car_6"), "velocity": [CAR_VELOCITY, 0, 0], "host": "car_6"},
        {"name": "rx_roof_b2", "position": roof_position("building_2", 1.2), "velocity": [0, 0, 0], "host": "building_2"},
    ]
    bs0 = {"name": "bs_b6", "position": roof_position("building_6"), "host": "building_6"}
    bs1 = {"name": "bs_b4", "position": roof_position("building_4"), "host": "building_4"}
    return {
        "one_bs": {"base_stations": [bs0], "receivers": receivers},
        "two_bs": {"base_stations": [bs0, bs1], "receivers": receivers},
    }


def array(pattern: str = "iso"):
    return PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern=pattern,
        polarization="V",
    )


def setup_scene(cfg: dict[str, object]):
    scene = load_scene(sionna.rt.scene.simple_street_canyon_with_cars, merge_shapes=False)
    scene.frequency = FREQ
    scene.tx_array = array()
    scene.rx_array = array()
    for bs in cfg["base_stations"]:
        scene.add(Transmitter(bs["name"], position=bs["position"], orientation=[0, 0, 0], power_dbm=0.0))
    for rx in cfg["receivers"]:
        scene.add(Receiver(rx["name"], position=rx["position"], orientation=[0, 0, 0]))
        scene.get(rx["name"]).velocity = rx["velocity"]
    for j in range(1, 6):
        scene.get(f"car_{j}").velocity = [-CAR_VELOCITY, 0, 0]
    for j in range(6, 9):
        scene.get(f"car_{j}").velocity = [CAR_VELOCITY, 0, 0]
    return scene


def solve(scene):
    t0 = time.perf_counter()
    paths = PathSolver()(
        scene=scene,
        max_depth=5,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        diffraction=True,
        edge_diffraction=True,
        refraction=False,
        synthetic_array=False,
    )
    return paths, (time.perf_counter() - t0) * 1e3


def path_lines(paths, tx_names: list[str], rx_names: list[str]) -> list[dict[str, object]]:
    sources = np.asarray(paths.sources.numpy(), dtype=float).T
    targets = np.asarray(paths.targets.numpy(), dtype=float).T
    vertices = np.asarray(paths.vertices.numpy(), dtype=float)
    valid = np.asarray(paths.valid.numpy(), dtype=bool)[:, 0, :, 0, :]
    interactions = np.asarray(paths.interactions.numpy(), dtype=int)[:, :, 0, :, 0, :]
    vertices = np.asarray(paths.vertices.numpy(), dtype=float)[:, :, 0, :, 0, :, :]
    doppler = np.asarray(paths.doppler.numpy(), dtype=float)[:, 0, :, 0, :]
    tau = np.asarray(paths.tau.numpy(), dtype=float)[:, 0, :, 0, :]
    lines = []
    for rx in range(valid.shape[0]):
        for tx in range(valid.shape[1]):
            for p in range(valid.shape[2]):
                if not valid[rx, tx, p]:
                    continue
                pts = [sources[tx].tolist()]
                path_interactions = interactions[:, rx, tx, p]
                for depth_idx, inter in enumerate(path_interactions):
                    if inter != 0:
                        pts.append(vertices[depth_idx, rx, tx, p].tolist())
                pts.append(targets[rx].tolist())
                lines.append({
                    "tx": tx_names[tx],
                    "rx": rx_names[rx],
                    "points": pts,
                    "doppler_hz": float(doppler[rx, tx, p]),
                    "delay_s": float(tau[rx, tx, p]),
                    "interaction": [int(x) for x in path_interactions.tolist()],
                })
    return lines


def render(scene, camera, path: Path, paths=None):
    bitmap = scene.render(
        camera=camera,
        paths=paths,
        num_samples=256,
        resolution=(950, 700),
        show_devices=True,
        return_bitmap=True,
    )
    arr = np.asarray(bitmap)
    arr = np.clip(arr[..., :3], 0.0, 1.0)
    Image.fromarray((arr * 255.0).astype(np.uint8)).save(path)


def save_pathloss_plot(case_dir: Path, labels: list[str], values: list[float]):
    fig, ax = plt.subplots(figsize=(9, 4.8))
    finite = [v for v in values if np.isfinite(v)]
    cap = (max(finite) + 10.0) if finite else 200.0
    plot_values = [v if np.isfinite(v) else cap for v in values]
    bars = ax.bar(labels, plot_values, color="#2f6fdd")
    ax.bar_label(bars, fmt="%.1f", fontsize=8)
    ax.set_ylabel("Pathloss (dB)")
    ax.set_title("Rooftop-BS street-canyon pathloss per link")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(case_dir / "pathloss_links.png", dpi=170)
    plt.close(fig)


def save_csi_plot(case_dir: Path, csi: np.ndarray, tx_names: list[str], rx_names: list[str]):
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for rx_i, rx_name in enumerate(rx_names):
        for tx_i, tx_name in enumerate(tx_names):
            h = csi[rx_i, 0, tx_i, 0, 0, :]
            mag = 20.0 * np.log10(np.maximum(np.abs(h), 1e-30))
            ax.plot(FREQS / 1e6, mag, linewidth=1.0, label=f"{tx_name}->{rx_name}")
    ax.set_xlabel("Subcarrier offset (MHz)")
    ax.set_ylabel("|H(f)| (dB)")
    ax.set_title("CSI magnitude for rooftop-BS links")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(case_dir / "csi_links.png", dpi=170)
    plt.close(fig)


def delay_doppler_spectrum(h_time_freq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return delay-Doppler magnitude plus delay/Doppler axes for [time, subcarrier] CFR."""
    h_delay = np.fft.ifft(np.fft.fftshift(h_time_freq, axes=1), axis=1, norm="ortho")
    dd = np.fft.fftshift(np.fft.fft(h_delay, axis=0, norm="ortho"), axes=0)
    num_time = h_time_freq.shape[0]
    doppler_res = DF / num_time
    delay_res = (1.0 / DF) / N_SC
    doppler_bins = (np.arange(num_time) - num_time // 2) * doppler_res
    delay_bins_ns = np.arange(N_SC) * delay_res / 1e-9
    return np.abs(dd), delay_bins_ns, doppler_bins


def save_delay_doppler_plots(case_dir: Path, csi: np.ndarray, tx_names: list[str], rx_names: list[str]):
    dd_dir = case_dir / "delay_doppler"
    dd_dir.mkdir(exist_ok=True)
    summaries = []
    for rx_i, rx_name in enumerate(rx_names):
        for tx_i, tx_name in enumerate(tx_names):
            h = csi[rx_i, 0, tx_i, 0, :, :]
            mag, delay_ns, doppler_hz = delay_doppler_spectrum(h)
            peak = np.unravel_index(np.argmax(mag), mag.shape)
            summaries.append({
                "tx": tx_name,
                "rx": rx_name,
                "peak_delay_ns": float(delay_ns[peak[1]]),
                "peak_doppler_hz": float(doppler_hz[peak[0]]),
                "peak_magnitude": float(mag[peak]),
            })

            delay_mask = delay_ns <= 700.0
            x, y = np.meshgrid(delay_ns[delay_mask], doppler_hz)
            z = mag[:, delay_mask]
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(x, y, z, cmap="viridis", edgecolor="none", linewidth=0, antialiased=True)
            ax.set_xlabel("Delay (ns)")
            ax.set_ylabel("Doppler (Hz)")
            ax.set_zlabel("Magnitude")
            ax.set_title(f"Delay-Doppler spectrum: {tx_name} -> {rx_name}")
            ax.view_init(elev=35, azim=-45)
            fig.tight_layout()
            fig.savefig(dd_dir / f"dd_3d_{tx_name}_to_{rx_name}.png", dpi=170)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(7.5, 4.8))
            im = ax.imshow(
                20.0 * np.log10(np.maximum(z, 1e-30)),
                aspect="auto",
                origin="lower",
                extent=[float(delay_ns[delay_mask][0]), float(delay_ns[delay_mask][-1]),
                        float(doppler_hz[0]), float(doppler_hz[-1])],
                cmap="magma",
            )
            ax.set_xlabel("Delay (ns)")
            ax.set_ylabel("Doppler (Hz)")
            ax.set_title(f"Delay-Doppler heatmap: {tx_name} -> {rx_name}")
            fig.colorbar(im, ax=ax, label="Magnitude (dB)")
            fig.tight_layout()
            fig.savefig(dd_dir / f"dd_heatmap_{tx_name}_to_{rx_name}.png", dpi=170)
            plt.close(fig)

    fig, axes = plt.subplots(len(rx_names), len(tx_names), figsize=(4.8 * len(tx_names), 3.4 * len(rx_names)), squeeze=False)
    for rx_i, rx_name in enumerate(rx_names):
        for tx_i, tx_name in enumerate(tx_names):
            h = csi[rx_i, 0, tx_i, 0, :, :]
            mag, delay_ns, doppler_hz = delay_doppler_spectrum(h)
            delay_mask = delay_ns <= 700.0
            ax = axes[rx_i, tx_i]
            ax.imshow(
                20.0 * np.log10(np.maximum(mag[:, delay_mask], 1e-30)),
                aspect="auto",
                origin="lower",
                extent=[float(delay_ns[delay_mask][0]), float(delay_ns[delay_mask][-1]),
                        float(doppler_hz[0]), float(doppler_hz[-1])],
                cmap="magma",
            )
            ax.set_title(f"{tx_name} -> {rx_name}")
            ax.set_xlabel("Delay (ns)")
            ax.set_ylabel("Doppler (Hz)")
    fig.suptitle("Delay-Doppler heatmap overview")
    fig.tight_layout()
    fig.savefig(case_dir / "delay_doppler_overview.png", dpi=170)
    plt.close(fig)
    (dd_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries


def run_case(name: str, cfg: dict[str, object]) -> dict[str, object]:
    case_dir = OUT / name
    case_dir.mkdir(parents=True, exist_ok=True)
    scene = setup_scene(cfg)
    paths, solve_ms = solve(scene)
    a, tau = paths.cir(sampling_frequency=FS, normalize_delays=False, out_type="numpy")
    csi = paths.cfr(
        frequencies=FREQS,
        sampling_frequency=FS,
        num_time_steps=64,
        normalize_delays=False,
        normalize=False,
        out_type="numpy",
    ).astype(np.complex64)

    tx_names = [bs["name"] for bs in cfg["base_stations"]]
    rx_names = [rx["name"] for rx in cfg["receivers"]]
    link_summary = []
    plot_labels = []
    plot_values = []
    for rx_i, rx_name in enumerate(rx_names):
        for tx_i, tx_name in enumerate(tx_names):
            power = np.abs(np.asarray(a[rx_i, 0, tx_i, 0, :, 0], dtype=np.complex128)) ** 2
            total = float(np.sum(power))
            pathloss = float(-10.0 * np.log10(total)) if total > 0 else float("inf")
            link_summary.append({
                "tx": tx_name,
                "rx": rx_name,
                "n_paths": int(np.count_nonzero(power > 0)),
                "total_power": total,
                "pathloss_db": pathloss,
            })
            plot_labels.append(f"{tx_name}\n{rx_name}")
            plot_values.append(pathloss)

    np.savez_compressed(case_dir / "rt_results.npz", a=a, tau=tau, csi=csi, freqs=FREQS)
    save_pathloss_plot(case_dir, plot_labels, plot_values)
    save_csi_plot(case_dir, csi, tx_names, rx_names)
    dd_summary = save_delay_doppler_plots(case_dir, csi, tx_names, rx_names)

    views = {
        "top": Camera(position=[45, 0, 155], look_at=[0, 0, 0]),
        "oblique": Camera(position=[78, -88, 82], look_at=[0, 0, 15]),
        "street": Camera(position=[-65, -24, 18], look_at=[8, 0, 4]),
    }
    for view_name, camera in views.items():
        render(scene, camera, case_dir / f"sionna_{view_name}_scene.png")
        render(scene, camera, case_dir / f"sionna_{view_name}_paths.png", paths=paths)

    frame_paths = []
    moving_scene = setup_scene(cfg)
    step = np.array([2.0, 0.0, 0.0])
    camera = views["top"]
    for i in range(7):
        frame = case_dir / f"motion_frame_{i:02d}.png"
        render(moving_scene, camera, frame)
        frame_paths.append(frame)
        for j in range(1, 6):
            moving_scene.get(f"car_{j}").position -= step
        for j in range(6, 9):
            moving_scene.get(f"car_{j}").position += step
        for rx in cfg["receivers"]:
            if rx["host"].startswith("car_"):
                delta = step if rx["velocity"][0] > 0 else -step
                moving_scene.get(rx["name"]).position += delta
    images = [Image.open(p).convert("P", palette=Image.Palette.ADAPTIVE) for p in frame_paths]
    images[0].save(case_dir / "motion.gif", save_all=True, append_images=images[1:], duration=450, loop=0)

    moving = {f"car_{j}": {"frame_displacement": [-12, 0, 0]} for j in range(1, 6)}
    moving.update({f"car_{j}": {"frame_displacement": [12, 0, 0]} for j in range(6, 9)})
    overlay = {
        "scene_xml": str(SCENE_XML),
        "case": name,
        "base_stations": cfg["base_stations"],
        "receivers": cfg["receivers"],
        "moving": moving,
        "paths": path_lines(paths, tx_names, rx_names),
        "mesh_bounds": BOUNDS,
    }
    (case_dir / "blender_overlay.json").write_text(json.dumps(overlay, indent=2), encoding="utf-8")
    summary = {
        "case": name,
        "frequency_hz": FREQ,
        "solve_ms": solve_ms,
        "base_stations": cfg["base_stations"],
        "receivers": cfg["receivers"],
        "links": link_summary,
        "outputs": {
            "npz": str(case_dir / "rt_results.npz"),
            "pathloss_plot": str(case_dir / "pathloss_links.png"),
            "csi_plot": str(case_dir / "csi_links.png"),
            "delay_doppler_overview": str(case_dir / "delay_doppler_overview.png"),
            "delay_doppler_dir": str(case_dir / "delay_doppler"),
            "motion_gif": str(case_dir / "motion.gif"),
        },
        "delay_doppler_peaks": dd_summary,
    }
    (case_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_summaries = {name: run_case(name, cfg) for name, cfg in configs().items()}
    (OUT / "summary.json").write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(OUT), "cases": list(all_summaries)}, indent=2))


if __name__ == "__main__":
    main()
