#!/usr/bin/env python3
"""Create clearer delay-Doppler views from saved mobility outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot extra delay-Doppler views.")
    parser.add_argument("case_dir", type=Path, help="Folder containing delay_doppler.npz")
    parser.add_argument("--title", default="", help="Plot title prefix")
    return parser.parse_args()


def mag_db_norm(dd: np.ndarray) -> np.ndarray:
    mag = np.abs(dd)
    ref = float(np.max(mag))
    if ref <= 0.0:
        ref = 1.0
    return 20.0 * np.log10(np.maximum(mag / ref, 1e-12))


def path_powers_db(case_dir: Path, n_paths: int) -> np.ndarray:
    single = case_dir / "single_h.npz"
    if not single.exists():
        return np.zeros(n_paths, dtype=np.float64)
    data = np.load(single)
    try:
        a = np.asarray(data["a"][0, 0, 0, 0, :, 0], dtype=np.complex128).reshape(-1)
    finally:
        data.close()
    p = np.abs(a[:n_paths]) ** 2
    if p.size < n_paths:
        p = np.pad(p, (0, n_paths - p.size), constant_values=0.0)
    ref = float(np.max(p))
    if ref <= 0.0:
        return np.zeros(n_paths, dtype=np.float64)
    return 10.0 * np.log10(np.maximum(p / ref, 1e-30))


def roi_masks(delay_bins_ns: np.ndarray, doppler_bins_hz: np.ndarray, tau_ns: np.ndarray, doppler_hz: np.ndarray):
    delay_pad = 80.0
    doppler_pad = max(180.0, 0.25 * float(np.ptp(doppler_hz) if doppler_hz.size else 0.0))
    d_min = max(float(delay_bins_ns[0]), float(np.min(tau_ns)) - delay_pad)
    d_max = min(float(delay_bins_ns[-1]), float(np.max(tau_ns)) + delay_pad)
    f_min = max(float(doppler_bins_hz[0]), float(np.min(doppler_hz)) - doppler_pad)
    f_max = min(float(doppler_bins_hz[-1]), float(np.max(doppler_hz)) + doppler_pad)
    return (delay_bins_ns >= d_min) & (delay_bins_ns <= d_max), (doppler_bins_hz >= f_min) & (doppler_bins_hz <= f_max)


def plot_zoomed_heatmap(case_dir: Path, title: str, db: np.ndarray, delay_bins_ns: np.ndarray,
                        doppler_bins_hz: np.ndarray, tau_ns: np.ndarray, doppler_hz: np.ndarray,
                        power_db: np.ndarray) -> Path:
    delay_mask, doppler_mask = roi_masks(delay_bins_ns, doppler_bins_hz, tau_ns, doppler_hz)
    fig, ax = plt.subplots(figsize=(9.4, 5.9))
    image = ax.imshow(
        db[np.ix_(doppler_mask, delay_mask)],
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=-45.0,
        vmax=0.0,
        extent=[
            float(delay_bins_ns[delay_mask][0]),
            float(delay_bins_ns[delay_mask][-1]),
            float(doppler_bins_hz[doppler_mask][0]),
            float(doppler_bins_hz[doppler_mask][-1]),
        ],
    )
    sizes = 85.0 * np.clip(1.0 + power_db / 35.0, 0.22, 1.0)
    ax.scatter(tau_ns, doppler_hz, s=sizes, marker="o", facecolors="none",
               edgecolors="cyan", linewidths=1.7, label="Sionna paths")
    for i, (delay, doppler) in enumerate(zip(tau_ns, doppler_hz)):
        ax.annotate(f"P{i}", (delay, doppler), xytext=(5, 4), textcoords="offset points",
                    color="cyan", fontsize=8, weight="bold")
    ax.set_xlabel("Delay (ns)")
    ax.set_ylabel("Doppler (Hz)")
    ax.set_title(f"{title} delay-Doppler heatmap (zoomed)")
    ax.grid(alpha=0.22)
    ax.legend(fontsize=8, loc="upper right")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Magnitude (dB, normalized)")
    fig.tight_layout()
    out = case_dir / "local_delay_doppler_heatmap_zoom.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_3d_surface(case_dir: Path, title: str, db: np.ndarray, delay_bins_ns: np.ndarray,
                    doppler_bins_hz: np.ndarray, tau_ns: np.ndarray, doppler_hz: np.ndarray) -> Path:
    delay_mask, doppler_mask = roi_masks(delay_bins_ns, doppler_bins_hz, tau_ns, doppler_hz)
    d = delay_bins_ns[delay_mask]
    f = doppler_bins_hz[doppler_mask]
    z = np.clip(db[np.ix_(doppler_mask, delay_mask)], -50.0, 0.0)
    step_d = max(1, int(np.ceil(d.size / 140)))
    step_f = max(1, int(np.ceil(f.size / 140)))
    d = d[::step_d]
    f = f[::step_f]
    z = z[::step_f, ::step_d]
    x_grid, y_grid = np.meshgrid(d, f)

    fig = plt.figure(figsize=(9.2, 7.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x_grid, y_grid, z, cmap="magma", linewidth=0, antialiased=True, alpha=0.96)
    ax.scatter(tau_ns, doppler_hz, np.zeros_like(tau_ns), color="cyan", marker="x", s=45, label="Sionna paths")
    ax.set_xlabel("Delay (ns)")
    ax.set_ylabel("Doppler (Hz)")
    ax.set_zlabel("Magnitude (dB, normalized)")
    ax.set_zlim(-50.0, 0.0)
    ax.view_init(elev=32, azim=-48)
    ax.set_title(f"{title} delay-Doppler 3D surface")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out = case_dir / "local_delay_doppler_3d.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_path_summary(case_dir: Path, title: str, tau_ns: np.ndarray, doppler_hz: np.ndarray,
                      power_db: np.ndarray) -> Path:
    order = np.argsort(-power_db)
    fig, (ax, table_ax) = plt.subplots(1, 2, figsize=(12.8, 5.6), gridspec_kw={"width_ratios": [1.55, 1.0]})
    sc = ax.scatter(tau_ns, doppler_hz, c=power_db, s=95, cmap="viridis", edgecolor="black", linewidth=0.45)
    for i, (delay, doppler) in enumerate(zip(tau_ns, doppler_hz)):
        ax.annotate(f"P{i}", (delay, doppler), xytext=(6, 4), textcoords="offset points", fontsize=9, weight="bold")
    ax.set_xlabel("Delay (ns)")
    ax.set_ylabel("Doppler (Hz)")
    ax.set_title(f"{title} path delay/Doppler summary")
    ax.grid(alpha=0.25)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Relative path power (dB)")

    table_ax.axis("off")
    rows = [["Path", "Delay ns", "Doppler Hz", "Rel pwr dB"]]
    for idx in order:
        rows.append([f"P{idx}", f"{tau_ns[idx]:.2f}", f"{doppler_hz[idx]:.1f}", f"{power_db[idx]:.1f}"])
    table = table_ax.table(cellText=rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.35)
    for col in range(4):
        table[(0, col)].set_text_props(weight="bold")
    table_ax.set_title("Paths sorted by power", pad=14)

    fig.tight_layout()
    out = case_dir / "local_delay_doppler_path_summary.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def main() -> None:
    args = parse_args()
    data_path = args.case_dir / "delay_doppler.npz"
    data = np.load(data_path)
    try:
        dd = np.asarray(data["h_delay_doppler"])
        delay_bins_ns = np.asarray(data["delay_bins_ns"], dtype=np.float64)
        doppler_bins_hz = np.asarray(data["doppler_bins_hz"], dtype=np.float64)
        tau_ns = np.asarray(data["tau_s"], dtype=np.float64).reshape(-1) / 1e-9
        doppler_hz = np.asarray(data["doppler_hz"], dtype=np.float64).reshape(-1)
    finally:
        data.close()

    db = mag_db_norm(dd)
    power_db = path_powers_db(args.case_dir, tau_ns.size)
    title = args.title or args.case_dir.name
    outputs = [
        plot_zoomed_heatmap(args.case_dir, title, db, delay_bins_ns, doppler_bins_hz, tau_ns, doppler_hz, power_db),
        plot_3d_surface(args.case_dir, title, db, delay_bins_ns, doppler_bins_hz, tau_ns, doppler_hz),
        plot_path_summary(args.case_dir, title, tau_ns, doppler_hz, power_db),
    ]
    for out in outputs:
        print(out)


if __name__ == "__main__":
    main()
