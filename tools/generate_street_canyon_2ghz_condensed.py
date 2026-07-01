#!/usr/bin/env python3
"""Generate compact rooftop-BS street-canyon datasets."""

from __future__ import annotations

import json
import math
import struct
import time
from pathlib import Path

import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sionna
from matplotlib.patches import Rectangle
from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene
from sionna.rt.utils import subcarrier_frequencies


OUT_ROOT = Path("street_canyon/mobility")
SCENE_XML = Path(sionna.rt.scene.simple_street_canyon_with_cars)
MESH_DIR = SCENE_XML.parent / "meshes"
FREQS_HZ = [5.0e9, 28.0e9]
FS = 122.88e6
DF = 30e3
N_SC = 512
N_TIME = 512
SUBCARRIER_FREQS = subcarrier_frequencies(N_SC, DF)
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
    bounds = {}
    for path in sorted(MESH_DIR.glob("*.ply")):
        if path.stem.startswith(("building_", "car_", "floor")):
            mn, mx = read_ply_bounds(path)
            bounds[path.stem] = {
                "min": mn.tolist(),
                "max": mx.tolist(),
                "center": ((mn + mx) / 2.0).tolist(),
            }
    return bounds


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


def configs() -> dict[str, dict[str, object]]:
    receivers = [
        {"name": "rx_car_3", "position": car_position("car_3"), "velocity": [-CAR_VELOCITY, 0, 0]},
        {"name": "rx_car_6", "position": car_position("car_6"), "velocity": [CAR_VELOCITY, 0, 0]},
        {"name": "rx_roof_b2", "position": roof_position("building_2", 1.2), "velocity": [0, 0, 0]},
    ]
    bs0 = {"name": "bs_b6", "position": roof_position("building_6")}
    bs1 = {"name": "bs_b4", "position": roof_position("building_4")}
    return {
        "one_bs": {"base_stations": [bs0], "receivers": receivers},
        "two_bs": {"base_stations": [bs0, bs1], "receivers": receivers},
    }


def setup_scene(freq_hz: float, cfg: dict[str, object]):
    scene = load_scene(sionna.rt.scene.simple_street_canyon_with_cars, merge_shapes=False)
    scene.frequency = freq_hz
    scene.tx_array = array()
    scene.rx_array = array()

    for bs in cfg["base_stations"]:
        scene.add(Transmitter(bs["name"], position=bs["position"], power_dbm=0.0))
    for rx in cfg["receivers"]:
        scene.add(Receiver(rx["name"], position=rx["position"]))
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
                lines.append({
                    "tx": tx_names[tx_i],
                    "rx": rx_names[rx_i],
                    "points": pts,
                    "delay_ns": float(tau[rx_i, tx_i, path_i] / 1e-9),
                    "doppler_hz": float(doppler[rx_i, tx_i, path_i]),
                    "power_db": float(10.0 * np.log10(max(p, 1e-30))),
                })
    return lines


def save_paths_plot(path: Path, freq_hz: float, case_name: str, lines: list[dict[str, object]], tx_position: list[float], rx_position: list[float], tx_name: str, rx_name: str) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 6.2))
    for name, b in BOUNDS.items():
        mn = np.array(b["min"])
        mx = np.array(b["max"])
        if name.startswith("building_"):
            ax.add_patch(Rectangle((mn[0], mn[1]), mx[0] - mn[0], mx[1] - mn[1],
                                   facecolor="#d9d9d9", edgecolor="#a8a8a8", linewidth=0.5, alpha=0.75))
        elif name.startswith("car_"):
            ax.add_patch(Rectangle((mn[0], mn[1]), mx[0] - mn[0], mx[1] - mn[1],
                                   facecolor="#f1c36d", edgecolor="#9d782f", linewidth=0.4, alpha=0.85))

    colors = {"bs_b6": "#1769aa", "bs_b4": "#c2413b"}
    for line in lines:
        pts = np.array(line["points"], dtype=float)
        ax.plot(pts[:, 0], pts[:, 1], color=colors.get(line["tx"], "#333333"),
                alpha=0.72, linewidth=1.1)

    ax.scatter(tx_position[0], tx_position[1], s=95, marker="^", color=colors.get(tx_name, "#333333"),
               edgecolor="white", linewidth=0.8, zorder=5, label=tx_name)
    ax.scatter(rx_position[0], rx_position[1], s=85, marker="o", color="#111111",
               edgecolor="white", linewidth=0.8, zorder=5, label=rx_name)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{band_name(freq_hz)} {case_name}: {tx_name} to {rx_name}")
    ax.grid(alpha=0.18)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_doppler_plot(path: Path, h_time_freq: np.ndarray, tau_s: np.ndarray, doppler_hz: np.ndarray, title: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dd, delay_bins_ns, doppler_bins_hz = delay_doppler(h_time_freq)
    mag = np.abs(dd)
    mag_db = 20.0 * np.log10(np.maximum(mag / np.max(mag), 1e-12))
    tau_ns = tau_s / 1e-9
    delay_limit = min(float(delay_bins_ns[-1]), max(700.0, float(np.max(tau_ns) + 150.0)))
    doppler_limit = min(float(np.max(np.abs(doppler_bins_hz))), max(350.0, float(np.max(np.abs(doppler_hz)) * 1.4)))
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
    ax.scatter(tau_ns, doppler_hz, marker="x", color="#58f3ff", s=30, linewidths=1.0)
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
    ax.set_title(f"{title}: PL {link_summary['pathloss_db']:.1f} dB")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_band(freq_hz: float) -> dict[str, str]:
    outputs = {}
    npz_dir = OUT_ROOT / "npz"
    npz_dir.mkdir(parents=True, exist_ok=True)

    for case_name, cfg in configs().items():
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
        tx_positions = {name: scene.get(name).position.numpy().astype(float).tolist() for name in tx_names}
        rx_positions = {name: scene.get(name).position.numpy().astype(float).tolist() for name in rx_names}
        case_outputs = {}

        for rx_i, rx_name in enumerate(rx_names):
            for tx_i, tx_name in enumerate(tx_names):
                condition = f"{case_name}__{tx_name}_to_{rx_name}"
                out_dir = OUT_ROOT / band_name(freq_hz) / condition
                out_dir.mkdir(parents=True, exist_ok=True)

                power = powers[(rx_i, tx_i)]
                total_power = float(np.sum(power))
                pathloss_db = float(-10.0 * np.log10(total_power)) if total_power > 0 else float("inf")
                link_summary = {
                    "case": case_name,
                    "tx": tx_name,
                    "rx": rx_name,
                    "n_paths": int(np.count_nonzero(power > 0.0)),
                    "total_power_linear": total_power,
                    "pathloss_db": pathloss_db,
                    "rx_power_dbm_at_0dbm_tx": -pathloss_db if np.isfinite(pathloss_db) else float("-inf"),
                    "delays_ns": (tau[rx_i, 0, tx_i, 0, :] / 1e-9).astype(float).tolist(),
                    "doppler_hz": doppler[rx_i, 0, tx_i, 0, :].astype(float).tolist(),
                }
                link_lines = [line for line in lines if line["tx"] == tx_name and line["rx"] == rx_name]
                title = f"{band_name(freq_hz)} {case_name}: {tx_name} to {rx_name}"
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

                data_path = out_dir / "rt_data.npz"
                np.savez_compressed(
                    data_path,
                    a=a[rx_i:rx_i + 1, :, tx_i:tx_i + 1, :, :, :],
                    tau=tau[rx_i:rx_i + 1, :, tx_i:tx_i + 1, :, :],
                    csi=csi[rx_i:rx_i + 1, :, tx_i:tx_i + 1, :, :, :],
                    freqs=SUBCARRIER_FREQS,
                    doppler_hz=doppler[rx_i:rx_i + 1, :, tx_i:tx_i + 1, :, :],
                    delay_doppler=dd,
                    delay_bins_ns=delay_bins_ns,
                    doppler_bins_hz=doppler_bins_hz,
                )
                npz_copy = npz_dir / f"{band_name(freq_hz)}__{condition}.npz"
                np.savez_compressed(
                    npz_copy,
                    a=a[rx_i:rx_i + 1, :, tx_i:tx_i + 1, :, :, :],
                    tau=tau[rx_i:rx_i + 1, :, tx_i:tx_i + 1, :, :],
                    csi=csi[rx_i:rx_i + 1, :, tx_i:tx_i + 1, :, :, :],
                    freqs=SUBCARRIER_FREQS,
                    doppler_hz=doppler[rx_i:rx_i + 1, :, tx_i:tx_i + 1, :, :],
                    delay_doppler=dd,
                    delay_bins_ns=delay_bins_ns,
                    doppler_bins_hz=doppler_bins_hz,
                )
                summary = {
                    "scene": "simple_street_canyon_with_cars",
                    "frequency_hz": freq_hz,
                    "band": band_name(freq_hz),
                    "subcarriers": N_SC,
                    "subcarrier_spacing_hz": DF,
                    "ofdm_time_steps": N_TIME,
                    "case_solve_ms": solve_ms,
                    "condition": condition,
                    "base_station": {tx_name: tx_positions[tx_name]},
                    "receiver": {rx_name: rx_positions[rx_name]},
                    "receiver_velocity_mps": scene.get(rx_name).velocity.numpy().astype(float).tolist(),
                    "link": link_summary,
                    "paths": link_lines,
                    "outputs": {
                        "data": str(data_path),
                        "npz_copy": str(npz_copy),
                        "paths_plot": str(out_dir / "paths.png"),
                        "doppler_plot": str(out_dir / "doppler.png"),
                        "csi_plot": str(out_dir / "csi.png"),
                    },
                }
                (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
                case_outputs[condition] = summary["outputs"]

        outputs[case_name] = case_outputs
    return outputs


def main() -> None:
    outputs = {band_name(freq_hz): run_band(freq_hz) for freq_hz in FREQS_HZ}
    (OUT_ROOT / "summary.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
