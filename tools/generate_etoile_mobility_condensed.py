#!/usr/bin/env python3
"""Generate compact Paris etoile mobility datasets."""

from __future__ import annotations

import json
import math
import re
import struct
import time
from pathlib import Path

import mitsuba as mi

# Querying available variants first avoids intermittent Dr.Jit/Mitsuba startup
# stalls seen in this environment when setting an LLVM variant directly.
mi.variants()
mi.set_variant("llvm_ad_mono_polarized")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sionna
from matplotlib.patches import Rectangle
from PIL import Image
from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene
from sionna.rt.utils import subcarrier_frequencies


OUT_ROOT = Path("etoile/mobility_single_vs_multi_bs")
SCENE_XML = Path(sionna.rt.scene.etoile)
MESH_DIR = SCENE_XML.parent / "meshes"
FREQS_HZ = [5.0e9, 28.0e9]
FS = 122.88e6
DF = 30e3
N_SC = 512
N_TIME = 512
SUBCARRIER_FREQS = subcarrier_frequencies(N_SC, DF)
MOBILE_VELOCITY = 10.0
FRAME_DT_S = 0.8
N_MOTION_FRAMES = 7


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


def roof_position(object_name: str, dz: float = 3.0) -> list[float]:
    b = BOUNDS[object_name]
    mn = np.array(b["min"])
    mx = np.array(b["max"])
    return [float((mn[0] + mx[0]) / 2.0), float((mn[1] + mx[1]) / 2.0), float(mx[2] + dz)]


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


TX_LOCATION_LABEL = {
    "bs_arc": "arc",
    "bs_maison": "maison",
}

RX_LOCATION_LABEL = {
    "rx_mobile_west": "mobile_west",
    "rx_mobile_east": "mobile_east",
    "rx_roof_e089": "e089",
}


def condition_name(case_name: str, tx_names: list[str], tx_name: str, rx_name: str) -> str:
    tx_labels = [TX_LOCATION_LABEL[n] for n in tx_names]
    tx_sel = TX_LOCATION_LABEL[tx_name]
    tx_part = "+".join(tx_labels)
    if len(tx_labels) > 1:
        return f"{case_name}__Tx={tx_part}__TxSel={tx_sel}__Rx={RX_LOCATION_LABEL[rx_name]}"
    return f"{case_name}__Tx={tx_sel}__Rx={RX_LOCATION_LABEL[rx_name]}"


def configs() -> dict[str, dict[str, object]]:
    receivers = [
        {
            "name": "rx_mobile_west",
            "position": [-40.0, 0.0, 10.0],
            "velocity": [-MOBILE_VELOCITY, 0.0, 0.0],
        },
        {
            "name": "rx_mobile_east",
            "position": [40.0, -25.0, 10.0],
            "velocity": [MOBILE_VELOCITY, 0.0, 0.0],
        },
        {
            "name": "rx_roof_e089",
            "position": roof_position("element_089", 2.0),
            "velocity": [0.0, 0.0, 0.0],
        },
    ]
    bs0 = {"name": "bs_arc", "position": roof_position("Arc_de_Triomphe", 3.0)}
    bs1 = {"name": "bs_maison", "position": roof_position("Maison_du_Danemark", 3.0)}
    return {
        "one_bs": {"base_stations": [bs0], "receivers": receivers},
        "two_bs": {"base_stations": [bs0, bs1], "receivers": receivers},
    }


def setup_scene(freq_hz: float, cfg: dict[str, object]):
    scene = load_scene(sionna.rt.scene.etoile, merge_shapes=False)
    scene.frequency = freq_hz
    scene.tx_array = array()
    scene.rx_array = array()

    for bs in cfg["base_stations"]:
        scene.add(Transmitter(bs["name"], position=bs["position"], power_dbm=0.0))
    for rx in cfg["receivers"]:
        scene.add(Receiver(rx["name"], position=rx["position"]))
        scene.get(rx["name"]).velocity = rx["velocity"]
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


def as_flat_list(x) -> list[float]:
    return np.asarray(x, dtype=float).reshape(-1).tolist()


def delay_doppler(h_time_freq: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h_delay = np.fft.ifft(np.fft.fftshift(h_time_freq, axes=1), axis=1, norm="ortho")
    dd = np.fft.fftshift(np.fft.fft(h_delay, axis=0, norm="ortho"), axes=0)
    doppler_bins = (np.arange(N_TIME) - N_TIME // 2) * (DF / N_TIME)
    delay_bins_ns = np.arange(N_SC) * ((1.0 / DF) / N_SC) / 1e-9
    return dd.astype(np.complex64), delay_bins_ns, doppler_bins


def path_lines(paths, tx_names: list[str], rx_names: list[str], powers: dict[tuple[int, int], np.ndarray]) -> list[dict[str, object]]:
    sources = np.asarray(paths.sources.numpy(), dtype=float).T
    targets = np.asarray(paths.targets.numpy(), dtype=float).T
    valid = np.asarray(paths.valid.numpy(), dtype=bool)[:, 0, :, 0, :]
    interactions = np.asarray(paths.interactions.numpy(), dtype=int)[:, :, 0, :, 0, :]
    vertices = np.asarray(paths.vertices.numpy(), dtype=float)[:, :, 0, :, 0, :, :]
    doppler = np.asarray(paths.doppler.numpy(), dtype=float)[:, 0, :, 0, :]
    tau = np.asarray(paths.tau.numpy(), dtype=float)[:, 0, :, 0, :]

    lines = []
    for rx_i in range(valid.shape[0]):
        for tx_i in range(valid.shape[1]):
            for path_i in range(valid.shape[2]):
                if not valid[rx_i, tx_i, path_i]:
                    continue
                pts = [sources[tx_i].tolist()]
                inter = interactions[:, rx_i, tx_i, path_i]
                for depth_i, interaction_type in enumerate(inter):
                    if interaction_type != 0:
                        pts.append(vertices[depth_i, rx_i, tx_i, path_i].tolist())
                pts.append(targets[rx_i].tolist())
                p = float(powers[(rx_i, tx_i)][path_i])
                lines.append(
                    {
                        "tx": tx_names[tx_i],
                        "rx": rx_names[rx_i],
                        "points": pts,
                        "delay_ns": float(tau[rx_i, tx_i, path_i] / 1e-9),
                        "doppler_hz": float(doppler[rx_i, tx_i, path_i]),
                        "power_db": float(10.0 * np.log10(max(p, 1e-30))),
                    }
                )
    return lines


def plot_scene_base(ax, title: str) -> None:
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
        ax.add_patch(Rectangle((mn[0], mn[1]), width, height, facecolor=face, edgecolor=edge, linewidth=lw, alpha=alpha))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    ax.grid(alpha=0.16)


def add_markers(ax, tx_positions: dict[str, list[float]], rx_positions: dict[str, list[float]], rx_velocities: dict[str, list[float]]) -> None:
    tx_colors = {"bs_arc": "#1769aa", "bs_maison": "#c2413b"}
    rx_colors = {"rx_mobile_west": "#111111", "rx_mobile_east": "#2f855a", "rx_roof_e089": "#7c3aed"}
    for tx_name, pos in tx_positions.items():
        ax.scatter(pos[0], pos[1], s=100, marker="^", color=tx_colors.get(tx_name, "#333333"), edgecolor="white", linewidth=0.8, zorder=6, label=tx_name)
    for rx_name, pos in rx_positions.items():
        ax.scatter(pos[0], pos[1], s=82, marker="o", color=rx_colors.get(rx_name, "#111111"), edgecolor="white", linewidth=0.8, zorder=7, label=rx_name)
        vel = np.array(rx_velocities[rx_name], dtype=float)
        if np.linalg.norm(vel[:2]) > 0:
            arrow = vel[:2] / np.linalg.norm(vel[:2]) * 22.0
            ax.arrow(pos[0], pos[1], arrow[0], arrow[1], color=rx_colors.get(rx_name, "#111111"), width=1.0, head_width=6.0, length_includes_head=True, alpha=0.82, zorder=5)
    ax.legend(loc="upper right", fontsize=7, ncol=2)


def save_scene_images(
    images_dir: Path,
    freq_hz: float,
    case_name: str,
    condition: str,
    tx_positions: dict[str, list[float]],
    rx_positions: dict[str, list[float]],
    rx_velocities: dict[str, list[float]],
    link_lines: list[dict[str, object]],
) -> None:
    images_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    plot_scene_base(ax, f"{band_name(freq_hz)} etoile {case_name}: scene map")
    add_markers(ax, tx_positions, rx_positions, rx_velocities)
    fig.tight_layout()
    fig.savefig(images_dir / "scene_map.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    plot_scene_base(ax, f"{band_name(freq_hz)} etoile {case_name}: motion map")
    add_markers(ax, tx_positions, rx_positions, rx_velocities)
    fig.tight_layout()
    fig.savefig(images_dir / "motion_map.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    plot_scene_base(ax, f"{band_name(freq_hz)} etoile: {condition} paths")
    for line in link_lines:
        pts = np.array(line["points"], dtype=float)
        ax.plot(pts[:, 0], pts[:, 1], color="#d94645", alpha=0.74, linewidth=1.1)
    add_markers(ax, tx_positions, rx_positions, rx_velocities)
    fig.tight_layout()
    fig.savefig(images_dir / "link_paths_map.png", dpi=180)
    plt.close(fig)

    frame_paths = []
    for i in range(N_MOTION_FRAMES):
        moved_rx = {}
        for rx_name, pos in rx_positions.items():
            displacement = np.array(rx_velocities[rx_name], dtype=float) * FRAME_DT_S * i
            moved_rx[rx_name] = (np.array(pos, dtype=float) + displacement).tolist()
        fig, ax = plt.subplots(figsize=(8.4, 7.2))
        plot_scene_base(ax, f"etoile motion frame {i:02d}")
        add_markers(ax, tx_positions, moved_rx, rx_velocities)
        fig.tight_layout()
        frame_path = images_dir / f"motion_frame_{i:02d}.png"
        fig.savefig(frame_path, dpi=180)
        plt.close(fig)
        frame_paths.append(frame_path)

    images = [Image.open(p).convert("P", palette=Image.Palette.ADAPTIVE) for p in frame_paths]
    images[0].save(images_dir / "motion.gif", save_all=True, append_images=images[1:], duration=450, loop=0)
    for image in images:
        image.close()


def save_paths_plot(path: Path, freq_hz: float, case_name: str, lines: list[dict[str, object]], tx_position: list[float], rx_position: list[float], tx_name: str, rx_name: str) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 7.2))
    plot_scene_base(ax, f"{band_name(freq_hz)} etoile {case_name}: {tx_name} to {rx_name}")
    for line in lines:
        pts = np.array(line["points"], dtype=float)
        ax.plot(pts[:, 0], pts[:, 1], color="#d94645", alpha=0.74, linewidth=1.1)
    ax.scatter(tx_position[0], tx_position[1], s=100, marker="^", color="#1769aa", edgecolor="white", linewidth=0.8, zorder=6, label=tx_name)
    ax.scatter(rx_position[0], rx_position[1], s=82, marker="o", color="#111111", edgecolor="white", linewidth=0.8, zorder=7, label=rx_name)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_doppler_plot(path: Path, h_time_freq: np.ndarray, tau_s: np.ndarray, doppler_hz: np.ndarray, title: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dd, delay_bins_ns, doppler_bins_hz = delay_doppler(h_time_freq)
    mag = np.abs(dd)
    mag_db = 20.0 * np.log10(np.maximum(mag / max(float(np.max(mag)), 1e-30), 1e-12))
    valid_tau = tau_s[tau_s > -1.0]
    tau_ns = valid_tau / 1e-9 if valid_tau.size else np.array([0.0])
    valid_doppler = doppler_hz[tau_s > -1.0] if valid_tau.size else np.array([0.0])
    delay_limit = min(float(delay_bins_ns[-1]), max(700.0, float(np.max(tau_ns) + 150.0)))
    doppler_limit = min(float(np.max(np.abs(doppler_bins_hz))), max(350.0, float(np.max(np.abs(valid_doppler)) * 1.4)))
    delay_mask = delay_bins_ns <= delay_limit
    doppler_mask = np.abs(doppler_bins_hz) <= doppler_limit

    fig, ax = plt.subplots(figsize=(7.6, 4.9))
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
            float(doppler_bins_hz[doppler_mask][0]),
            float(doppler_bins_hz[doppler_mask][-1]),
        ],
    )
    ax.scatter(tau_ns, valid_doppler, marker="x", color="#58f3ff", s=30, linewidths=1.0)
    ax.set_xlabel("Delay (ns)")
    ax.set_ylabel("Doppler (Hz)")
    ax.set_title(title)
    ax.grid(alpha=0.16)
    fig.colorbar(im, ax=ax, label="Normalized magnitude (dB)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return dd, delay_bins_ns, doppler_bins_hz


def save_csi_plot(path: Path, h_freq: np.ndarray, link_summary: dict[str, object], title: str) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    mag_db = 20.0 * np.log10(np.maximum(np.abs(h_freq), 1e-30))
    ax.plot(SUBCARRIER_FREQS / 1e6, mag_db, linewidth=1.25)
    ax.set_xlabel("Subcarrier offset (MHz)")
    ax.set_ylabel("|H(f)| (dB)")
    pl = link_summary["pathloss_db"]
    pl_text = f"{pl:.1f} dB" if np.isfinite(pl) else "inf"
    ax.set_title(f"{title}: PL {pl_text}")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_band(freq_hz: float) -> dict[str, dict[str, dict[str, str]]]:
    outputs = {}
    npz_dir = OUT_ROOT / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

    for case_name, cfg in configs().items():
        print(f"Solving {band_name(freq_hz)} {case_name}", flush=True)
        scene = setup_scene(freq_hz, cfg)
        paths, solve_ms = solve(scene)
        a, tau = paths.cir(sampling_frequency=FS, normalize_delays=False, out_type="numpy")
        csi = paths.cfr(
            frequencies=SUBCARRIER_FREQS,
            sampling_frequency=DF,
            num_time_steps=N_TIME,
            normalize_delays=False,
            normalize=False,
            out_type="numpy",
        ).astype(np.complex64)
        doppler = np.asarray(paths.doppler.numpy(), dtype=np.float64)

        tx_names = [bs["name"] for bs in cfg["base_stations"]]
        rx_names = [rx["name"] for rx in cfg["receivers"]]
        powers = {}
        for rx_i in range(len(rx_names)):
            for tx_i in range(len(tx_names)):
                powers[(rx_i, tx_i)] = np.abs(np.asarray(a[rx_i, 0, tx_i, 0, :, 0], dtype=np.complex128)) ** 2

        lines = path_lines(paths, tx_names, rx_names, powers)
        tx_positions = {name: as_flat_list(scene.get(name).position.numpy()) for name in tx_names}
        rx_positions = {name: as_flat_list(scene.get(name).position.numpy()) for name in rx_names}
        rx_velocities = {name: as_flat_list(scene.get(name).velocity.numpy()) for name in rx_names}
        case_outputs = {}

        for rx_i, rx_name in enumerate(rx_names):
            for tx_i, tx_name in enumerate(tx_names):
                condition = condition_name(case_name, tx_names, tx_name, rx_name)
                out_dir = OUT_ROOT / band_name(freq_hz) / condition
                images_dir = out_dir / "images"
                out_dir.mkdir(parents=True, exist_ok=True)

                power = powers[(rx_i, tx_i)]
                total_power = float(np.sum(power))
                pathloss_db = float(-10.0 * np.log10(total_power)) if total_power > 0 else float("inf")
                valid_mask = tau[rx_i, 0, tx_i, 0, :] > -1.0
                link_summary = {
                    "case": case_name,
                    "tx": tx_name,
                    "rx": rx_name,
                    "n_paths": int(np.count_nonzero(power > 0.0)),
                    "total_power_linear": total_power,
                    "pathloss_db": pathloss_db,
                    "rx_power_dbm_at_0dbm_tx": -pathloss_db if np.isfinite(pathloss_db) else float("-inf"),
                    "delays_ns": (tau[rx_i, 0, tx_i, 0, valid_mask] / 1e-9).astype(float).tolist(),
                    "doppler_hz": doppler[rx_i, 0, tx_i, 0, valid_mask].astype(float).tolist(),
                }
                link_lines = [line for line in lines if line["tx"] == tx_name and line["rx"] == rx_name]
                title = f"{band_name(freq_hz)} etoile {case_name}: {tx_name} to {rx_name}"
                h_time_freq = csi[rx_i, 0, tx_i, 0, :, :]
                dd, delay_bins_ns, doppler_bins_hz = save_doppler_plot(
                    out_dir / "doppler.png",
                    h_time_freq,
                    tau[rx_i, 0, tx_i, 0, :],
                    doppler[rx_i, 0, tx_i, 0, :],
                    title,
                )
                save_paths_plot(
                    out_dir / "paths.png",
                    freq_hz,
                    case_name,
                    link_lines,
                    tx_positions[tx_name],
                    rx_positions[rx_name],
                    tx_name,
                    rx_name,
                )
                save_csi_plot(out_dir / "csi.png", h_time_freq[0, :], link_summary, title)
                save_scene_images(
                    images_dir,
                    freq_hz,
                    case_name,
                    condition,
                    tx_positions,
                    rx_positions,
                    rx_velocities,
                    link_lines,
                )

                data_path = out_dir / "rt_data.npz"
                data_arrays = {
                    "a": a[rx_i:rx_i + 1, :, tx_i:tx_i + 1, :, :, :],
                    "tau": tau[rx_i:rx_i + 1, :, tx_i:tx_i + 1, :, :],
                    "csi": csi[rx_i:rx_i + 1, :, tx_i:tx_i + 1, :, :, :],
                    "freqs": SUBCARRIER_FREQS,
                    "doppler_hz": doppler[rx_i:rx_i + 1, :, tx_i:tx_i + 1, :, :],
                    "delay_doppler": dd,
                    "delay_bins_ns": delay_bins_ns,
                    "doppler_bins_hz": doppler_bins_hz,
                }
                np.savez_compressed(data_path, **data_arrays)
                npz_copy = npz_dir / f"{band_name(freq_hz)}__{condition}.npz"
                np.savez_compressed(npz_copy, **data_arrays)
                summary = {
                    "scene": "etoile",
                    "scene_xml": str(SCENE_XML),
                    "frequency_hz": freq_hz,
                    "band": band_name(freq_hz),
                    "subcarriers": N_SC,
                    "subcarrier_spacing_hz": DF,
                    "ofdm_time_steps": N_TIME,
                    "case_solve_ms": solve_ms,
                    "condition": condition,
                    "base_station": {tx_name: tx_positions[tx_name]},
                    "receiver": {rx_name: rx_positions[rx_name]},
                    "receiver_velocity_mps": rx_velocities[rx_name],
                    "link": link_summary,
                    "paths": link_lines,
                    "motion": {
                        "frame_dt_s": FRAME_DT_S,
                        "num_frames": N_MOTION_FRAMES,
                        "frame_displacement_m": (np.array(rx_velocities[rx_name], dtype=float) * FRAME_DT_S).tolist(),
                    },
                    "outputs": {
                        "data": str(data_path),
                        "npz_copy": str(npz_copy),
                        "paths_plot": str(out_dir / "paths.png"),
                        "doppler_plot": str(out_dir / "doppler.png"),
                        "csi_plot": str(out_dir / "csi.png"),
                        "images": {
                            "scene_map": str(images_dir / "scene_map.png"),
                            "motion_map": str(images_dir / "motion_map.png"),
                            "link_paths_map": str(images_dir / "link_paths_map.png"),
                            "motion_gif": str(images_dir / "motion.gif"),
                            "motion_frames": [
                                str(images_dir / "motion_frame_00.png"),
                                str(images_dir / "motion_frame_03.png"),
                                str(images_dir / "motion_frame_06.png"),
                            ],
                        },
                    },
                }
                (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                case_outputs[condition] = summary["outputs"]

        outputs[case_name] = case_outputs
    return outputs


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    outputs = {band_name(freq_hz): run_band(freq_hz) for freq_hz in FREQS_HZ}
    (OUT_ROOT / "summary.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
