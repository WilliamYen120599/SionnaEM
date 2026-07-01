#!/usr/bin/env python3
"""Output helpers for Sionna RT quick scene checks."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EPS_POWER = 1e-30


def safe_scene_name(raw: str | Path) -> str:
    """Convert a scene name/path into a stable output-folder name."""
    name = Path(raw).stem if isinstance(raw, Path) else str(raw)
    name = Path(name).stem
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return name or "scene"


def scene_output_dir(base_out_dir: str | Path, scene_name: str | Path) -> Path:
    """Return `base_out_dir / safe_scene_name(scene_name)` and create it."""
    out_dir = Path(base_out_dir) / safe_scene_name(scene_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def subcarrier_frequencies(num_subcarriers: int, subcarrier_spacing: float) -> np.ndarray:
    """Return baseband OFDM subcarrier offsets centered around DC."""
    k = np.arange(num_subcarriers, dtype=np.float64) - (num_subcarriers - 1) / 2.0
    return k * float(subcarrier_spacing)


def first_link_path_data(a: np.ndarray, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract path coefficients, powers, and delays for the first Tx/Rx link."""
    a0 = np.asarray(a[0, 0, 0, 0, :, 0], dtype=np.complex128)
    if tau.ndim >= 5:
        tau0 = np.asarray(tau[0, 0, 0, 0, :], dtype=np.float64)
    elif tau.ndim == 2:
        tau0 = np.asarray(tau[0, :], dtype=np.float64)
    else:
        tau0 = np.asarray(tau).reshape(-1).astype(np.float64)
    n = min(a0.shape[0], tau0.shape[0])
    a0 = a0[:n]
    tau0 = tau0[:n]
    power = np.abs(a0) ** 2
    return a0, power, tau0


def link_budget_from_power(path_power: np.ndarray, tx_power_dbm: float = 0.0) -> tuple[float, float, float]:
    """Return path gain, pathloss dB, and received power dBm."""
    path_gain_linear = float(np.sum(path_power))
    if path_gain_linear <= 0.0:
        return 0.0, float("inf"), float("-inf")
    pathloss_db = float(-10.0 * np.log10(path_gain_linear))
    rx_power_dbm = float(tx_power_dbm - pathloss_db)
    return path_gain_linear, pathloss_db, rx_power_dbm


def save_rf_outputs(
    *,
    paths,
    a: np.ndarray,
    tau: np.ndarray,
    out_dir: Path,
    prefix: str,
    tx: np.ndarray,
    rx: np.ndarray,
    frequency: float,
    sampling_frequency: float,
    tx_power_dbm: float,
    num_subcarriers: int,
    subcarrier_spacing: float,
    num_time_steps: int = 1,
    solve_ms: float | None = None,
) -> dict[str, object]:
    """Save CSI/CFR arrays, pathloss data, and compact diagnostic plots."""
    out_dir.mkdir(parents=True, exist_ok=True)

    path_coeff, path_power, path_delay = first_link_path_data(a, tau)
    path_gain, pathloss_db, rx_power_dbm = link_budget_from_power(path_power, tx_power_dbm)

    csi_frequencies = subcarrier_frequencies(num_subcarriers, subcarrier_spacing)
    csi = paths.cfr(
        frequencies=csi_frequencies,
        sampling_frequency=sampling_frequency,
        num_time_steps=num_time_steps,
        normalize_delays=False,
        normalize=False,
        out_type="numpy",
    ).astype(np.complex64)

    npz_path = out_dir / f"{prefix}_csi_pathloss.npz"
    np.savez_compressed(
        npz_path,
        cir_a=a,
        cir_tau=tau,
        first_link_path_coefficients=path_coeff.astype(np.complex64),
        first_link_path_power_linear=path_power.astype(np.float64),
        first_link_path_delay_s=path_delay.astype(np.float64),
        csi=csi,
        csi_frequencies_hz=csi_frequencies.astype(np.float64),
        path_gain_linear=np.array(path_gain, dtype=np.float64),
        pathloss_db=np.array(pathloss_db, dtype=np.float64),
        rx_power_dbm=np.array(rx_power_dbm, dtype=np.float64),
        tx=np.asarray(tx, dtype=np.float32),
        rx=np.asarray(rx, dtype=np.float32),
        carrier_frequency_hz=np.array(frequency, dtype=np.float64),
        sampling_frequency_hz=np.array(sampling_frequency, dtype=np.float64),
        subcarrier_spacing_hz=np.array(subcarrier_spacing, dtype=np.float64),
        tx_power_dbm=np.array(tx_power_dbm, dtype=np.float64),
        solve_ms=np.array(np.nan if solve_ms is None else solve_ms, dtype=np.float64),
    )

    pathloss_plot = out_dir / f"{prefix}_pathloss.png"
    save_pathloss_plot(pathloss_plot, path_gain, pathloss_db, rx_power_dbm)

    csi_mag_plot = out_dir / f"{prefix}_csi_magnitude.png"
    save_csi_magnitude_plot(csi_mag_plot, csi, csi_frequencies)

    return {
        "npz": npz_path,
        "pathloss_plot": pathloss_plot,
        "csi_magnitude_plot": csi_mag_plot,
        "path_gain_linear": path_gain,
        "pathloss_db": pathloss_db,
        "rx_power_dbm": rx_power_dbm,
        "csi_shape": csi.shape,
    }


def save_path_power_plot(
    path: Path,
    power: np.ndarray,
    scene_label: str,
    delay: np.ndarray | None = None,
) -> None:
    """Save a deterministic path-power stem plot."""
    power = np.asarray(power, dtype=np.float64).reshape(-1)
    if delay is not None:
        delay = np.asarray(delay, dtype=np.float64).reshape(-1)
        n = min(power.size, delay.size)
        power = power[:n]
        delay = delay[:n]
        order = np.argsort(delay)
        power = power[order]

    fig, ax = plt.subplots()
    if power.size:
        ax.stem(np.arange(power.size), power)
        ax.set_title(f"Path powers ({scene_label})")
        ax.set_xlabel("Path index")
        ax.set_ylabel("Power")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.2)
    else:
        ax.text(0.1, 0.5, "No paths found", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_pathloss_plot(path: Path, path_gain: float, pathloss_db: float, rx_power_dbm: float) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    if np.isfinite(pathloss_db):
        bars = ax.bar(["Pathloss", "Rx power"], [pathloss_db, rx_power_dbm], color=["#2f6fdd", "#2c9a68"])
        ax.bar_label(bars, fmt="%.2f")
        ax.set_ylabel("dB / dBm")
        ax.set_title(f"Link pathloss, gain={path_gain:.3e}")
        ax.grid(axis="y", alpha=0.25)
    else:
        ax.text(0.5, 0.5, "No valid path: pathloss is infinite", ha="center", va="center")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_csi_magnitude_plot(path: Path, csi: np.ndarray, frequencies: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    h = np.asarray(csi[0, 0, 0, 0, 0, :], dtype=np.complex64)
    mag_db = 20.0 * np.log10(np.maximum(np.abs(h), EPS_POWER))
    # Remove floating-order noise that is far below plot precision.
    mag_db = np.round(mag_db, decimals=4)
    ax.plot(frequencies / 1e6, mag_db, marker=".", linewidth=1.2)
    ax.set_xlabel("Subcarrier offset (MHz)")
    ax.set_ylabel("|CSI| (dB)")
    ax.set_title("CSI magnitude, first Tx/Rx link")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
