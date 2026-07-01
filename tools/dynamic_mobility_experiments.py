#!/usr/bin/env python3
"""Run Sionna RT mobility toy-scene checks across frequency bands."""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.request
from pathlib import Path

import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sionna
from sionna.rt import Camera, PathSolver, PlanarArray, Receiver, Transmitter, load_scene
from sionna.rt.utils import subcarrier_frequencies


FS = 122.88e6
N_SC = 1024
DF = 30e3
FREQS = subcarrier_frequencies(N_SC, DF)
OFFICIAL_IMAGE_BASE = "https://nvlabs.github.io/sionna/_images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dynamic mobility checks.")
    parser.add_argument("--out-dir", default="Blender Preview Images/mobility")
    parser.add_argument("--bands", nargs="+", type=float, default=[3.5e9, 28e9])
    return parser.parse_args()


def band_name(freq_hz: float) -> str:
    ghz = freq_hz / 1e9
    if abs(ghz - round(ghz)) < 1e-9:
        return f"{int(round(ghz))}GHz"
    return f"{str(ghz).replace('.', 'p')}GHz"


def band_title(freq_hz: float) -> str:
    return band_name(freq_hz).replace("p", ".")


def mag_db(h: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(h), 1e-30))


def phase_deg(h: np.ndarray) -> np.ndarray:
    return np.unwrap(np.angle(h)) * 180.0 / np.pi


def power_to_pathloss(power: np.ndarray) -> tuple[float, float]:
    total = float(np.sum(power))
    if total <= 0.0:
        return total, float("inf")
    return total, float(-10.0 * np.log10(total))


def link_power(a: np.ndarray, tx_idx: int = 0, tx_ant: int = 0) -> np.ndarray:
    return np.abs(np.asarray(a[0, 0, tx_idx, tx_ant, :, 0], dtype=np.complex128)) ** 2


def make_array(tx_rows: int = 1, tx_cols: int = 1, pattern: str = "iso") -> PlanarArray:
    return PlanarArray(
        num_rows=tx_rows,
        num_cols=tx_cols,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern=pattern,
        polarization="V",
    )


def solve(scene, max_depth: int, diffraction: bool = False):
    t0 = time.perf_counter()
    paths = PathSolver()(
        scene=scene,
        max_depth=max_depth,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        diffraction=diffraction,
        edge_diffraction=False,
        refraction=False,
        synthetic_array=False,
    )
    return paths, (time.perf_counter() - t0) * 1e3


def cir_cfr(paths, num_time_steps: int = 1, normalize: bool = False):
    a, tau = paths.cir(sampling_frequency=FS, normalize_delays=False, out_type="numpy")
    h = paths.cfr(
        frequencies=FREQS,
        sampling_frequency=FS,
        num_time_steps=num_time_steps,
        normalize_delays=False,
        normalize=normalize,
        out_type="numpy",
    ).astype(np.complex64)
    return a, tau, h


def save_line(path: Path, title: str, ylabel: str, series: list[tuple[str, np.ndarray]], phase: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for label, h in series:
        y = phase_deg(h) if phase else mag_db(h)
        ax.plot(FREQS / 1e6, y, linewidth=1.1, label=label)
    ax.set_xlabel("Subcarrier offset (MHz)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_bar(path: Path, title: str, ylabel: str, labels: list[str], values: list[float], fmt: str = "%.2f") -> None:
    fig, ax = plt.subplots(figsize=(6.7, 4.3))
    bars = ax.bar(labels, values, color=["#2f6fdd", "#2c9a68", "#8a6edb", "#c9962d", "#c85d54"][:len(labels)])
    ax.bar_label(bars, fmt=fmt)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_path_stem(path: Path, title: str, tau: np.ndarray, power: np.ndarray, doppler: np.ndarray | None = None) -> None:
    tau_ns = np.asarray(tau).reshape(-1) / 1e-9
    power_db = 10.0 * np.log10(np.maximum(np.asarray(power).reshape(-1), 1e-30))
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    markerline, stemlines, baseline = ax.stem(tau_ns, power_db)
    markerline.set_markersize(5)
    plt.setp(stemlines, linewidth=1.2)
    ax.set_xlabel("Delay (ns)")
    ax.set_ylabel("Path power (dB)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    if doppler is not None:
        for x, y, d in zip(tau_ns, power_db, np.asarray(doppler).reshape(-1)):
            ax.annotate(f"{d:.0f} Hz", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def simple_reflector_scene(freq_hz: float, tx_rows: int = 1, tx_cols: int = 1, multi: bool = False):
    scene = load_scene(sionna.rt.scene.simple_reflector, merge_shapes=False)
    scene.frequency = float(freq_hz)
    scene.get("reflector").velocity = [0, 0, -20]
    scene.tx_array = make_array(tx_rows, tx_cols)
    scene.rx_array = make_array()
    scene.add(Transmitter("tx0", [-25, 0.1, 50], power_dbm=0.0))
    if multi:
        scene.add(Transmitter("tx1", [-25, 8.0, 50], power_dbm=0.0))
    scene.add(Receiver("rx", [25, 0.1, 50]))
    return scene


def street_scene(freq_hz: float, tx_rows: int = 1, tx_cols: int = 1, multi: bool = False):
    scene = load_scene(sionna.rt.scene.simple_street_canyon_with_cars, merge_shapes=False)
    scene.frequency = float(freq_hz)
    scene.tx_array = make_array(tx_rows, tx_cols, pattern="tr38901")
    scene.rx_array = make_array(pattern="tr38901")
    scene.add(Transmitter("tx0", position=[22.7, 5.6, 0.75], orientation=[np.pi, 0, 0], power_dbm=0.0))
    if multi:
        scene.add(Transmitter("tx1", position=[18.0, -5.6, 0.75], orientation=[0, 0, 0], power_dbm=0.0))
    scene.add(Receiver("rx", position=[-27.8, -4.9, 0.75]))
    velocity = np.array([10.0, 0.0, 0.0])
    for j in range(1, 6):
        scene.get(f"car_{j}").velocity = -velocity
    for j in range(6, 9):
        scene.get(f"car_{j}").velocity = velocity
    scene.get("tx0").velocity = -velocity
    if multi:
        scene.get("tx1").velocity = velocity
    scene.get("rx").velocity = velocity
    return scene


def run_single(scene_factory, freq_hz: float, out_dir: Path, scene_label: str, max_depth: int) -> dict[str, object]:
    band = band_title(freq_hz)
    scene = scene_factory(freq_hz)
    paths, solve_ms = solve(scene, max_depth=max_depth)
    a, tau, h = cir_cfr(paths, num_time_steps=25)
    h0 = h[0, 0, 0, 0, 0, :]
    h_last = h[0, 0, 0, 0, -1, :]
    power = link_power(a)
    total, pl = power_to_pathloss(power)
    doppler = np.asarray(paths.doppler.numpy()[0, 0, 0], dtype=np.float64)
    tau0 = np.asarray(paths.tau.numpy()[0, 0, 0], dtype=np.float64)
    np.savez_compressed(out_dir / "single_h.npz", csi=h, a=a, tau=tau, freqs=FREQS,
                        doppler_hz=paths.doppler.numpy(), solve_ms=solve_ms)
    save_line(out_dir / "single_mag.png", f"{band} {scene_label} single-link CSI magnitude", "|H(f,t)| (dB)",
              [("t0", h0), ("t24", h_last)])
    save_line(out_dir / "single_phase.png", f"{band} {scene_label} single-link CSI phase", "Unwrapped phase (deg)",
              [("t0", h0), ("t24", h_last)], phase=True)
    save_bar(out_dir / "single_pl.png", f"{band} {scene_label} single-link pathloss", "Pathloss (dB)", ["tx0"], [pl])
    save_path_stem(out_dir / "single_paths.png", f"{band} {scene_label} single-link paths", tau0, power, doppler)
    return {
        "csi_shape": list(h.shape),
        "n_paths": int(power.size),
        "total_power": total,
        "pathloss_db": pl,
        "doppler_hz": doppler.tolist(),
        "delay_ns": (tau0 / 1e-9).tolist(),
        "solve_ms": solve_ms,
    }


def run_multi(scene_factory, freq_hz: float, out_dir: Path, scene_label: str, max_depth: int) -> dict[str, object]:
    band = band_title(freq_hz)
    scene = scene_factory(freq_hz, multi=True)
    paths, solve_ms = solve(scene, max_depth=max_depth)
    a, tau, h = cir_cfr(paths, num_time_steps=25)
    h0 = h[0, 0, 0, 0, 0, :]
    h1 = h[0, 0, 1, 0, 0, :]
    h_sum = h0 + h1
    p0 = link_power(a, 0)
    p1 = link_power(a, 1)
    total0, pl0 = power_to_pathloss(p0)
    total1, pl1 = power_to_pathloss(p1)
    combined = total0 + total1
    combined_pl = float(-10.0 * np.log10(combined)) if combined > 0 else float("inf")
    np.savez_compressed(out_dir / "multi_h.npz", csi=h, a=a, tau=tau, freqs=FREQS, h_sum=h_sum,
                        doppler_hz=paths.doppler.numpy(), solve_ms=solve_ms)
    save_line(out_dir / "sep_mag.png", f"{band} {scene_label} separated CSI magnitude", "|H(f)| (dB)",
              [("tx0", h0), ("tx1", h1)])
    save_line(out_dir / "sep_phase.png", f"{band} {scene_label} separated CSI phase", "Unwrapped phase (deg)",
              [("tx0", h0), ("tx1", h1)], phase=True)
    save_line(out_dir / "sum_mag.png", f"{band} {scene_label} separated CSI vs sum", "|H(f)| (dB)",
              [("tx0", h0), ("tx1", h1), ("tx0+tx1", h_sum)])
    save_bar(out_dir / "pl_sep.png", f"{band} {scene_label} per-transmitter pathloss", "Pathloss (dB)",
             ["tx0", "tx1"], [pl0, pl1])
    save_bar(out_dir / "pl_sum.png", f"{band} {scene_label} combined received power equivalent", "Pathloss (dB)",
             ["tx0", "tx1", "sum"], [pl0, pl1, combined_pl])
    return {
        "csi_shape": list(h.shape),
        "tx0_total_power": total0,
        "tx0_pathloss_db": pl0,
        "tx1_total_power": total1,
        "tx1_pathloss_db": pl1,
        "combined_power_sum": combined,
        "combined_pathloss_db": combined_pl,
        "solve_ms": solve_ms,
    }


def run_antenna_b(scene_factory, freq_hz: float, out_dir: Path, scene_label: str, max_depth: int) -> dict[str, object]:
    band = band_title(freq_hz)
    scene = scene_factory(freq_hz, tx_rows=4, tx_cols=1)
    paths, solve_ms = solve(scene, max_depth=max_depth)
    a, tau, h = cir_cfr(paths, num_time_steps=1)
    h_ant = h[0, 0, 0, :, 0, :]
    pls = []
    totals = []
    for ant in range(h_ant.shape[0]):
        total, pl = power_to_pathloss(link_power(a, 0, ant))
        totals.append(total)
        pls.append(pl)
    np.savez_compressed(out_dir / "ant_b.npz", csi=h, a=a, tau=tau, freqs=FREQS,
                        doppler_hz=paths.doppler.numpy(), solve_ms=solve_ms)
    series = [(f"ant{ant}", h_ant[ant]) for ant in range(h_ant.shape[0])]
    save_line(out_dir / "ant_b_mag.png", f"{band} {scene_label} 4x1 TX-array CSI magnitude", "|H(f)| (dB)", series)
    save_bar(out_dir / "ant_b_pl.png", f"{band} {scene_label} per-antenna pathloss", "Pathloss (dB)",
             [f"a{i}" for i in range(len(pls))], pls)
    return {
        "csi_shape": list(h.shape),
        "pathloss_db_per_tx_ant": pls,
        "total_power_per_tx_ant": totals,
        "solve_ms": solve_ms,
    }


def run_pilots(out_dir: Path, freq_hz: float, scene_label: str) -> dict[str, object]:
    band = band_title(freq_hz)
    data = np.load(out_dir / "multi_h.npz")
    h = data["csi"][0, 0, :, 0, 0, :].astype(np.complex128)
    h0, h1 = h[0], h[1]
    x_orth = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)
    y_orth = x_orth @ h
    h_hat_orth = np.linalg.inv(x_orth) @ y_orth
    x_mix = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.complex128)
    y_mix = x_mix @ h
    h_hat_mix = np.linalg.pinv(x_mix) @ y_mix
    orth_err = float(np.max(np.abs(h_hat_orth - h)))
    mix_err = float(np.max(np.abs(h_hat_mix - h)))
    np.savez_compressed(out_dir / "pilot.npz", h=h, x_orth=x_orth, y_orth=y_orth,
                        h_hat_orth=h_hat_orth, x_mix=x_mix, y_mix=y_mix, h_hat_mix=h_hat_mix, freqs=FREQS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(FREQS / 1e6, mag_db(h0), label="true tx0")
    axes[0].plot(FREQS / 1e6, mag_db(h_hat_orth[0]), "--", label="estimated tx0")
    axes[0].plot(FREQS / 1e6, mag_db(h1), label="true tx1")
    axes[0].plot(FREQS / 1e6, mag_db(h_hat_orth[1]), "--", label="estimated tx1")
    axes[0].set_title(f"{band} {scene_label}: orthogonal pilots, max err={orth_err:.1e}")
    axes[1].plot(FREQS / 1e6, mag_db(h0 + h1), label="observable sum")
    axes[1].plot(FREQS / 1e6, mag_db(h_hat_mix[0]), "--", label="pinv tx0")
    axes[1].plot(FREQS / 1e6, mag_db(h_hat_mix[1]), "--", label="pinv tx1")
    axes[1].set_title(f"{band} {scene_label}: identical pilots are rank deficient")
    for ax in axes:
        ax.set_xlabel("Subcarrier offset (MHz)")
        ax.set_ylabel("|H(f)| (dB)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "pilot_cmp.png", dpi=170)
    plt.close(fig)
    save_bar(out_dir / "pilot_err.png", f"{band} {scene_label} pilot channel-estimation error", "Max |error|",
             ["orth", "same"], [orth_err, mix_err], fmt="%.2e")
    return {
        "physical_channel_has_pilots": False,
        "orthogonal_pilot_max_abs_error": orth_err,
        "same_pilot_rank": int(np.linalg.matrix_rank(x_mix)),
        "same_pilot_max_abs_error": mix_err,
    }


def reflector_delay_doppler(freq_hz: float, out_dir: Path) -> dict[str, object]:
    scene = simple_reflector_scene(freq_hz)
    scene.get("tx0").velocity = [30, 0, 0]
    paths, solve_ms = solve(scene, max_depth=1)
    num_symbols = 1024
    h = paths.cfr(
        frequencies=FREQS,
        sampling_frequency=DF,
        num_time_steps=num_symbols,
        normalize_delays=False,
        normalize=True,
        out_type="numpy",
    )
    h = np.squeeze(h)
    h_delay = np.fft.ifft(np.fft.fftshift(h, axes=1), axis=1, norm="ortho")
    dd = np.fft.fftshift(np.fft.fft(h_delay, axis=0, norm="ortho"), axes=0)
    doppler_res = DF / num_symbols
    delay_res = (1 / DF) / N_SC
    doppler_bins = np.arange(-num_symbols / 2 * doppler_res, num_symbols / 2 * doppler_res, doppler_res)
    delay_bins = np.arange(N_SC) * delay_res / 1e-9
    mag = np.abs(dd)
    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(111, projection="3d")
    offset = 20
    xs = slice(int(N_SC / 2) - offset, int(N_SC / 2) + offset)
    ys = slice(0, offset)
    x_grid, y_grid = np.meshgrid(delay_bins, doppler_bins)
    ax.plot_surface(x_grid[xs, ys], y_grid[xs, ys], mag[xs, ys], cmap="viridis", edgecolor="none")
    ax.set_xlabel("Delay (ns)")
    ax.set_ylabel("Doppler (Hz)")
    ax.set_zlabel("Magnitude")
    ax.view_init(elev=53, azim=-32)
    ax.set_title("Local delay-Doppler spectrum")
    fig.tight_layout()
    fig.savefig(out_dir / "local_delay_doppler.png", dpi=170)
    plt.close(fig)
    tau = np.asarray(paths.tau.numpy()[0, 0, 0], dtype=np.float64).reshape(-1)
    doppler = np.asarray(paths.doppler.numpy()[0, 0, 0], dtype=np.float64).reshape(-1)
    np.savez_compressed(out_dir / "delay_doppler.npz", h=h, h_delay_doppler=dd, freqs=FREQS,
                        delay_bins_ns=delay_bins, doppler_bins_hz=doppler_bins,
                        tau_s=tau, doppler_hz=doppler, solve_ms=solve_ms)
    return {
        "delay_ns": (tau / 1e-9).tolist(),
        "doppler_hz": doppler.tolist(),
        "solve_ms": solve_ms,
        "plot": str(out_dir / "local_delay_doppler.png"),
    }


def street_delay_doppler(freq_hz: float, out_dir: Path) -> dict[str, object]:
    scene = street_scene(freq_hz)
    paths, solve_ms = solve(scene, max_depth=3)
    num_symbols = 1024
    h = paths.cfr(
        frequencies=FREQS,
        sampling_frequency=DF,
        num_time_steps=num_symbols,
        normalize_delays=False,
        normalize=True,
        out_type="numpy",
    )
    h = np.squeeze(h)
    h_delay = np.fft.ifft(np.fft.fftshift(h, axes=1), axis=1, norm="ortho")
    dd = np.fft.fftshift(np.fft.fft(h_delay, axis=0, norm="ortho"), axes=0)

    doppler_res = DF / num_symbols
    delay_res = (1 / DF) / N_SC
    doppler_bins = np.arange(-num_symbols / 2 * doppler_res, num_symbols / 2 * doppler_res, doppler_res)
    delay_bins = np.arange(N_SC) * delay_res / 1e-9

    tau = np.asarray(paths.tau.numpy()[0, 0, 0], dtype=np.float64).reshape(-1)
    doppler = np.asarray(paths.doppler.numpy()[0, 0, 0], dtype=np.float64).reshape(-1)
    tau_ns = tau / 1e-9

    delay_upper = min(float(delay_bins[-1]), max(500.0, float(np.max(tau_ns) + 200.0)))
    doppler_abs = float(np.max(np.abs(doppler))) if doppler.size else 0.0
    doppler_upper = min(float(np.max(np.abs(doppler_bins))), max(600.0, doppler_abs * 1.35))
    delay_mask = delay_bins <= delay_upper
    doppler_mask = np.abs(doppler_bins) <= doppler_upper

    mag = np.abs(dd)
    mag_db_norm = 20.0 * np.log10(np.maximum(mag / np.max(mag), 1e-12))

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    image = ax.imshow(
        mag_db_norm[np.ix_(doppler_mask, delay_mask)],
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=-50.0,
        vmax=0.0,
        extent=[
            float(delay_bins[delay_mask][0]),
            float(delay_bins[delay_mask][-1]),
            float(doppler_bins[doppler_mask][0]),
            float(doppler_bins[doppler_mask][-1]),
        ],
    )
    ax.scatter(tau_ns, doppler, marker="x", color="cyan", s=36, linewidths=1.2, label="Sionna paths")
    ax.set_xlabel("Delay (ns)")
    ax.set_ylabel("Doppler (Hz)")
    ax.set_title(f"{band_title(freq_hz)} street_canyon delay-Doppler map")
    ax.grid(alpha=0.18)
    ax.legend(fontsize=8, loc="upper right")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Magnitude (dB, normalized)")
    fig.tight_layout()
    fig.savefig(out_dir / "local_delay_doppler.png", dpi=170)
    plt.close(fig)

    np.savez_compressed(
        out_dir / "delay_doppler.npz",
        h=h,
        h_delay_doppler=dd,
        freqs=FREQS,
        delay_bins_ns=delay_bins,
        doppler_bins_hz=doppler_bins,
        tau_s=tau,
        doppler_hz=doppler,
        solve_ms=solve_ms,
    )
    return {
        "delay_ns": tau_ns.tolist(),
        "doppler_hz": doppler.tolist(),
        "solve_ms": solve_ms,
        "plot": str(out_dir / "local_delay_doppler.png"),
        "npz": str(out_dir / "delay_doppler.npz"),
    }


def street_doppler_vs_movement(freq_hz: float, out_dir: Path) -> dict[str, object]:
    num_symbols = 25
    velocity = np.array([10.0, 0.0, 0.0])
    ofdm_symbol_duration = 1 / DF
    displacement = velocity * ofdm_symbol_duration

    scene = street_scene(freq_hz)
    paths, solve_ms = solve(scene, max_depth=3)
    h_dop = np.squeeze(paths.cfr(
        frequencies=FREQS,
        sampling_frequency=DF,
        num_time_steps=num_symbols,
        normalize_delays=False,
        out_type="numpy",
    ))

    scene_sim = street_scene(freq_hz)
    h_sim = []
    sim_solve_ms = 0.0
    for _ in range(num_symbols):
        paths_i, solve_i_ms = solve(scene_sim, max_depth=3)
        sim_solve_ms += solve_i_ms
        h_i = np.squeeze(paths_i.cfr(frequencies=FREQS, sampling_frequency=DF,
                                     normalize_delays=False, out_type="numpy"), axis=(0, 1, 2, 3))
        h_sim.append(h_i[0])
        scene_sim.get("tx0").position -= displacement
        scene_sim.get("rx").position += displacement
        for j in range(1, 6):
            scene_sim.get(f"car_{j}").position -= displacement
        for j in range(6, 9):
            scene_sim.get(f"car_{j}").position += displacement
    h_sim = np.asarray(h_sim)

    subcarriers = np.arange(0, N_SC, 256)
    timesteps = [0, 8, 16, 24]
    fig, axs = plt.subplots(4, 4, figsize=(17, 13))
    for i, j in enumerate(subcarriers):
        axs[0, i].plot(np.arange(num_symbols), np.real(h_sim[:, j]))
        axs[0, i].plot(np.arange(num_symbols), np.real(h_dop[:, j]), "--")
        axs[0, i].set_xlabel("Timestep")
        axs[0, i].set_ylabel(r"$\Re\{h(f,t)\}$")
        axs[0, i].set_title(f"Subcarrier {j}")
        axs[0, i].legend(["Movement", "Doppler"], fontsize=8)
    for i, j in enumerate(subcarriers):
        axs[1, i].plot(np.arange(num_symbols), np.imag(h_sim[:, j]))
        axs[1, i].plot(np.arange(num_symbols), np.imag(h_dop[:, j]), "--")
        axs[1, i].set_xlabel("Timestep")
        axs[1, i].set_ylabel(r"$\Im\{h(f,t)\}$")
        axs[1, i].set_title(f"Subcarrier {j}")
        axs[1, i].legend(["Movement", "Doppler"], fontsize=8)
    for i, j in enumerate(timesteps):
        axs[2, i].plot(np.arange(N_SC), np.real(h_sim[j, :]))
        axs[2, i].plot(np.arange(N_SC), np.real(h_dop[j, :]), "--")
        axs[2, i].set_xlabel("Subcarrier")
        axs[2, i].set_ylabel(r"$\Re\{h(f,t)\}$")
        axs[2, i].set_title(f"Timestep {j}")
        axs[2, i].legend(["Movement", "Doppler"], fontsize=8)
    for i, j in enumerate(timesteps):
        axs[3, i].plot(np.arange(N_SC), np.imag(h_sim[j, :]))
        axs[3, i].plot(np.arange(N_SC), np.imag(h_dop[j, :]), "--")
        axs[3, i].set_xlabel("Subcarrier")
        axs[3, i].set_ylabel(r"$\Im\{h(f,t)\}$")
        axs[3, i].set_title(f"Timestep {j}")
        axs[3, i].legend(["Movement", "Doppler"], fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "local_doppler_vs_movement.png", dpi=170)
    plt.close(fig)
    max_abs_error = float(np.max(np.abs(h_sim - h_dop)))
    rms_error = float(np.sqrt(np.mean(np.abs(h_sim - h_dop) ** 2)))
    np.savez_compressed(out_dir / "doppler_vs_movement.npz", h_sim=h_sim, h_dop=h_dop, freqs=FREQS)
    return {
        "h_shape": list(h_dop.shape),
        "max_abs_error": max_abs_error,
        "rms_error": rms_error,
        "doppler_solve_ms": solve_ms,
        "movement_total_solve_ms": sim_solve_ms,
        "plot": str(out_dir / "local_doppler_vs_movement.png"),
    }


def download_official_images(out_dir: Path) -> dict[str, str]:
    names = {
        "reflector_paths": "rt_tutorials_Mobility_21_0.png",
        "reflector_delay_doppler": "rt_tutorials_Mobility_27_0.png",
        "street_paths": "rt_tutorials_Mobility_32_0.png",
        "street_doppler_vs_movement": "rt_tutorials_Mobility_36_0.png",
    }
    saved = {}
    for key, filename in names.items():
        target = out_dir / f"official_{filename}"
        if not target.exists():
            urllib.request.urlretrieve(f"{OFFICIAL_IMAGE_BASE}/{filename}", target)
        saved[key] = str(target)
    return saved


def save_side_by_side(out_path: Path, local_path: Path, official_path: Path, title: str) -> None:
    local = plt.imread(local_path)
    official = plt.imread(official_path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(local)
    axes[0].set_title("Local")
    axes[1].imshow(official)
    axes[1].set_title("Sionna docs")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def write_csv(out_dir: Path, summary: dict[str, object]) -> None:
    rows = [
        ("freq_hz", summary["frequency_hz"]),
        ("reflector_single_pathloss_db", summary["simple_reflector"]["single"]["pathloss_db"]),
        ("reflector_multi_tx0_pathloss_db", summary["simple_reflector"]["multi"]["tx0_pathloss_db"]),
        ("reflector_multi_tx1_pathloss_db", summary["simple_reflector"]["multi"]["tx1_pathloss_db"]),
        ("street_single_pathloss_db", summary["street_canyon"]["single"]["pathloss_db"]),
        ("street_multi_tx0_pathloss_db", summary["street_canyon"]["multi"]["tx0_pathloss_db"]),
        ("street_multi_tx1_pathloss_db", summary["street_canyon"]["multi"]["tx1_pathloss_db"]),
        ("street_doppler_vs_movement_rms_error", summary["street_canyon"]["official_single_antenna_comparison"]["rms_error"]),
    ]
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def run_scene_group(scene_key: str, scene_factory, freq_hz: float, band_dir: Path, max_depth: int) -> dict[str, object]:
    out_dir = band_dir / scene_key
    out_dir.mkdir(parents=True, exist_ok=True)
    single = run_single(scene_factory, freq_hz, out_dir, scene_key, max_depth)
    multi = run_multi(scene_factory, freq_hz, out_dir, scene_key, max_depth)
    ant_b = run_antenna_b(scene_factory, freq_hz, out_dir, scene_key, max_depth)
    pilot = run_pilots(out_dir, freq_hz, scene_key)
    h_info = {
        "matrix_file": str(out_dir / "multi_h.npz"),
        "matrix_key": "csi",
        "plot_files": {"magnitude": str(out_dir / "sep_mag.png"), "phase": str(out_dir / "sep_phase.png")},
        "indexing": {"tx0": "csi[0,0,0,0,0,:]", "tx1": "csi[0,0,1,0,0,:]"},
        "matrix_shape": multi["csi_shape"],
    }
    (out_dir / "h_info.json").write_text(json.dumps(h_info, indent=2), encoding="utf-8")
    return {
        "single": single,
        "multi": multi,
        "antenna_experiment_b": ant_b,
        "channel_matrix": h_info,
        "pilot": pilot,
    }


def main() -> None:
    args = parse_args()
    base = Path(args.out_dir)
    base.mkdir(parents=True, exist_ok=True)
    official = download_official_images(base)
    all_summaries = {}

    for freq_hz in args.bands:
        band = band_name(freq_hz)
        band_dir = base / band
        band_dir.mkdir(parents=True, exist_ok=True)
        reflector = run_scene_group("simple_reflector", simple_reflector_scene, freq_hz, band_dir, max_depth=1)
        reflector["official_single_antenna_comparison"] = reflector_delay_doppler(freq_hz, band_dir / "simple_reflector")
        street = run_scene_group("street_canyon", street_scene, freq_hz, band_dir, max_depth=3)
        street["official_single_antenna_comparison"] = street_doppler_vs_movement(freq_hz, band_dir / "street_canyon")
        street["delay_doppler"] = street_delay_doppler(freq_hz, band_dir / "street_canyon")

        if abs(freq_hz - 3.5e9) < 1.0:
            save_side_by_side(
                band_dir / "simple_reflector" / "compare_delay_doppler_official.png",
                band_dir / "simple_reflector" / "local_delay_doppler.png",
                Path(official["reflector_delay_doppler"]),
                "Simple reflector delay-Doppler: local vs Sionna docs",
            )
            save_side_by_side(
                band_dir / "street_canyon" / "compare_doppler_vs_movement_official.png",
                band_dir / "street_canyon" / "local_doppler_vs_movement.png",
                Path(official["street_doppler_vs_movement"]),
                "Street canyon Doppler vs movement: local vs Sionna docs",
            )

        summary = {
            "frequency_hz": float(freq_hz),
            "band": band,
            "simple_reflector": reflector,
            "street_canyon": street,
            "official_docs": {
                "source": "https://nvlabs.github.io/sionna/rt/tutorials/Mobility.html",
                "downloaded_images": official,
                "reference_values_at_3p5GHz": {
                    "simple_reflector_delay_ns": [166.782, 372.93597],
                    "simple_reflector_doppler_hz_after_tx_velocity": [350.242, -261.056],
                },
            },
        }
        (band_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_csv(band_dir, summary)
        all_summaries[band] = summary

    (base / "summary.json").write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(base), "bands": list(all_summaries)}, indent=2))


if __name__ == "__main__":
    main()
