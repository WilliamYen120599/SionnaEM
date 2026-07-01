#!/usr/bin/env python3
"""Compare 5 GHz and 28 GHz point-drone Tx outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("etoile/point_drone_tx")
CONDITION = "point_drone_tx_to_rx_rooftop_e089__cam_x58p9_y32p0_z44p0"
OUT = ROOT / "comparison_5GHz_vs_28GHz"
BANDS = {
    "5GHz": ROOT / "5GHz" / CONDITION / "rt_data.npz",
    "28GHz": ROOT / "28GHz" / CONDITION / "rt_data.npz",
}


def load_band(path: Path) -> dict[str, np.ndarray]:
    return {k: v for k, v in np.load(path).items()}


def db20(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.asarray(x), 1e-30))


def db10(x: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(x), 1e-30))


def valid_doppler(data: dict[str, np.ndarray]) -> np.ndarray:
    return data["doppler_hz_by_frame"][data["valid_path_mask"]]


def csi_stats(csi: np.ndarray) -> dict[str, np.ndarray | float]:
    mag = np.abs(csi)
    mag_db = db20(mag)
    frame_rms_db = db20(np.sqrt(np.mean(mag**2, axis=1)))
    frame_mean_db = db20(np.mean(mag, axis=1))
    return {
        "mag_db": mag_db,
        "frame_rms_db": frame_rms_db,
        "frame_mean_db": frame_mean_db,
        "median_db": float(np.median(mag_db)),
        "max_db": float(np.max(mag_db)),
        "min_db": float(np.min(mag_db)),
        "subcarrier_std_db_mean": float(np.mean(np.std(mag_db, axis=1))),
        "subcarrier_std_db_max": float(np.max(np.std(mag_db, axis=1))),
    }


def dd_peak(data: dict[str, np.ndarray]) -> dict[str, float]:
    dd = np.abs(data["delay_doppler_from_frames"])
    delay_bins = data["delay_bins_ns"]
    doppler_bins = data["doppler_bins_hz_from_frame_sampling"]
    idx = np.unravel_index(np.argmax(dd), dd.shape)
    return {
        "delay_ns": float(delay_bins[idx[1]]),
        "frame_doppler_bin_hz": float(doppler_bins[idx[0]]),
        "normalized_peak_db": 0.0,
    }


def save_pathloss_plot(data: dict[str, dict[str, np.ndarray]], out: Path) -> None:
    t = data["5GHz"]["frame_time_s"]
    pl5 = data["5GHz"]["pathloss_db"][:, 0, 0]
    pl28 = data["28GHz"]["pathloss_db"][:, 0, 0]
    diff = pl28 - pl5
    theory = 20.0 * np.log10(28.0 / 5.0)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(9.2, 7.0), sharex=True)
    ax0.plot(t, pl5, marker="o", markersize=2.8, label="5 GHz")
    ax0.plot(t, pl28, marker="o", markersize=2.8, label="28 GHz")
    ax0.set_ylabel("Pathloss (dB)")
    ax0.set_title("Point-drone pathloss: 5 GHz vs 28 GHz")
    ax0.grid(alpha=0.25)
    ax0.legend()

    ax1.plot(t, diff, color="#7c3aed", marker="o", markersize=2.8, label="28 GHz - 5 GHz")
    ax1.axhline(theory, color="#dc2626", linestyle="--", linewidth=1.2, label=f"20log10(28/5) = {theory:.2f} dB")
    ax1.set_xlabel("Trajectory time (s)")
    ax1.set_ylabel("Difference (dB)")
    ax1.grid(alpha=0.25)
    ax1.legend()
    fig.tight_layout()
    fig.savefig(out / "pathloss_5GHz_vs_28GHz.png", dpi=180)
    plt.close(fig)


def save_csi_plots(data: dict[str, dict[str, np.ndarray]], stats: dict[str, dict[str, object]], out: Path) -> None:
    t = data["5GHz"]["frame_time_s"]
    freqs = data["5GHz"]["freqs"] / 1e6

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    for band, color in [("5GHz", "#2563eb"), ("28GHz", "#dc2626")]:
        ax.plot(t, stats[band]["frame_rms_db"], color=color, linewidth=1.8, label=band)
    ax.set_xlabel("Trajectory time (s)")
    ax.set_ylabel("CSI frame RMS magnitude (dB)")
    ax.set_title("CSI magnitude over trajectory")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "csi_frame_rms_5GHz_vs_28GHz.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.9), sharey=True)
    for ax, band in zip(axes, ["5GHz", "28GHz"]):
        mag_db = stats[band]["mag_db"]
        vmax = float(np.percentile(mag_db, 99.5))
        vmin = vmax - 45.0
        im = ax.imshow(
            mag_db,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            extent=[float(freqs[0]), float(freqs[-1]), float(t[0]), float(t[-1])],
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(band)
        ax.set_xlabel("Subcarrier offset (MHz)")
        ax.grid(alpha=0.12)
        fig.colorbar(im, ax=ax, label="|CSI| (dB)")
    axes[0].set_ylabel("Trajectory time (s)")
    fig.suptitle("CSI heatmaps: 50 poses x 512 subcarriers")
    fig.tight_layout()
    fig.savefig(out / "csi_heatmaps_5GHz_vs_28GHz.png", dpi=180)
    plt.close(fig)


def save_path_count_plot(data: dict[str, dict[str, np.ndarray]], out: Path) -> None:
    t = data["5GHz"]["frame_time_s"]
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    ax.plot(t, data["5GHz"]["n_paths"], marker="o", markersize=2.8, label="5 GHz")
    ax.plot(t, data["28GHz"]["n_paths"], marker="o", markersize=2.8, label="28 GHz")
    ax.set_xlabel("Trajectory time (s)")
    ax.set_ylabel("Valid paths")
    ax.set_title("Valid path count over trajectory")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "path_count_5GHz_vs_28GHz.png", dpi=180)
    plt.close(fig)


def save_doppler_plots(data: dict[str, dict[str, np.ndarray]], out: Path) -> None:
    fs = 1.0 / float(data["5GHz"]["trajectory_dt_s"])
    nyq = fs / 2.0
    dop5 = valid_doppler(data["5GHz"])
    dop28 = valid_doppler(data["28GHz"])

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    bins = np.linspace(-1800, 1800, 80)
    ax.hist(dop5, bins=bins, alpha=0.58, label="5 GHz physical per-path Doppler")
    ax.hist(dop28, bins=bins, alpha=0.58, label="28 GHz physical per-path Doppler")
    ax.axvspan(-nyq, nyq, color="#22c55e", alpha=0.18, label=f"0.1 s frame Nyquist: +/-{nyq:.1f} Hz")
    ax.set_xlabel("Doppler (Hz)")
    ax.set_ylabel("Path samples")
    ax.set_title("Physical Doppler greatly exceeds 10 Hz frame sampling bandwidth")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "physical_doppler_vs_frame_sampling.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    bins5 = data["5GHz"]["doppler_bins_hz_from_frame_sampling"]
    ax.stem(bins5, np.ones_like(bins5), basefmt=" ", linefmt="#64748b", markerfmt=" ")
    ax.axvline(-nyq, color="#dc2626", linestyle="--")
    ax.axvline(nyq, color="#dc2626", linestyle="--")
    ax.set_xlabel("Delay-Doppler FFT bin from 50 pose samples (Hz)")
    ax.set_yticks([])
    ax.set_title("Low-rate frame-sampled Doppler bins only span about +/-5 Hz")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "frame_sampled_doppler_bins.png", dpi=180)
    plt.close(fig)


def save_delay_doppler_plot(data: dict[str, dict[str, np.ndarray]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), sharey=True)
    for ax, band in zip(axes, ["5GHz", "28GHz"]):
        dd = np.abs(data[band]["delay_doppler_from_frames"])
        mag_db = db20(dd / max(float(np.max(dd)), 1e-30))
        delay = data[band]["delay_bins_ns"]
        dop = data[band]["doppler_bins_hz_from_frame_sampling"]
        delay_mask = delay <= 900.0
        im = ax.imshow(
            mag_db[:, delay_mask],
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=-55,
            vmax=0,
            extent=[float(delay[delay_mask][0]), float(delay[delay_mask][-1]), float(dop[0]), float(dop[-1])],
        )
        ax.set_title(f"{band} low-rate diagnostic")
        ax.set_xlabel("Delay (ns)")
        fig.colorbar(im, ax=ax, label="Normalized magnitude (dB)")
    axes[0].set_ylabel("Frame-sampled Doppler bin (Hz)")
    fig.suptitle("Delay-Doppler from 0.1 s pose samples is not raw RF Doppler")
    fig.tight_layout()
    fig.savefig(out / "delay_doppler_low_rate_side_by_side.png", dpi=180)
    plt.close(fig)


def write_summary(data: dict[str, dict[str, np.ndarray]], stats: dict[str, dict[str, object]], out: Path) -> None:
    rows = []
    fs = 1.0 / float(data["5GHz"]["trajectory_dt_s"])
    nyq = fs / 2.0
    theory = 20.0 * np.log10(28.0 / 5.0)
    pl5 = data["5GHz"]["pathloss_db"][:, 0, 0]
    pl28 = data["28GHz"]["pathloss_db"][:, 0, 0]
    diff = pl28 - pl5

    for band in ["5GHz", "28GHz"]:
        d = data[band]
        pl = d["pathloss_db"][:, 0, 0]
        dop = valid_doppler(d)
        peak = dd_peak(d)
        rows.append(
            {
                "band": band,
                "pathloss_min_db": float(pl.min()),
                "pathloss_max_db": float(pl.max()),
                "pathloss_mean_db": float(pl.mean()),
                "pathloss_std_db": float(pl.std()),
                "n_paths_min": int(d["n_paths"].min()),
                "n_paths_max": int(d["n_paths"].max()),
                "n_paths_mean": float(d["n_paths"].mean()),
                "csi_shape": list(d["csi"].shape),
                "csi_median_mag_db": stats[band]["median_db"],
                "csi_max_mag_db": stats[band]["max_db"],
                "csi_frame_rms_mean_db": float(np.mean(stats[band]["frame_rms_db"])),
                "physical_doppler_min_hz": float(dop.min()),
                "physical_doppler_max_hz": float(dop.max()),
                "physical_doppler_abs_max_hz": float(np.max(np.abs(dop))),
                "low_rate_dd_peak_delay_ns": peak["delay_ns"],
                "low_rate_dd_peak_doppler_bin_hz": peak["frame_doppler_bin_hz"],
            }
        )

    comparison = {
        "condition": CONDITION,
        "pose_sampling_dt_s": float(data["5GHz"]["trajectory_dt_s"]),
        "pose_sampling_rate_hz": fs,
        "pose_sampling_nyquist_hz": nyq,
        "pathloss_28GHz_minus_5GHz_mean_db": float(diff.mean()),
        "pathloss_28GHz_minus_5GHz_min_db": float(diff.min()),
        "pathloss_28GHz_minus_5GHz_max_db": float(diff.max()),
        "theoretical_frequency_pathloss_delta_db_20log10_28_over_5": float(theory),
        "pathloss_delta_error_mean_minus_theory_db": float(diff.mean() - theory),
        "physical_doppler_abs_max_5GHz_hz": float(np.max(np.abs(valid_doppler(data["5GHz"])))),
        "physical_doppler_abs_max_28GHz_hz": float(np.max(np.abs(valid_doppler(data["28GHz"])))),
        "doppler_abs_max_ratio_28GHz_over_5GHz": float(
            np.max(np.abs(valid_doppler(data["28GHz"]))) / np.max(np.abs(valid_doppler(data["5GHz"])))
        ),
        "frequency_ratio_28GHz_over_5GHz": 28.0 / 5.0,
        "interpretation": (
            "The physical per-path Doppler values are hundreds of Hz at 5 GHz "
            "and over 1.6 kHz at 28 GHz, far above the +/-5 Hz Nyquist limit "
            "of 0.1 s pose sampling. Therefore the frame-sampled delay-Doppler "
            "plots are low-rate trajectory diagnostics, not raw RF Doppler spectra."
        ),
    }

    with (out / "numerical_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    (out / "comparison_summary.json").write_text(json.dumps({"bands": rows, "comparison": comparison}, indent=2), encoding="utf-8")

    md = [
        "# Point-Drone 5 GHz vs 28 GHz Comparison",
        "",
        f"Condition: `{CONDITION}`",
        "",
        "## Key Result",
        "",
        comparison["interpretation"],
        "",
        "## Sampling Check",
        "",
        f"- Pose/image sampling interval: `{comparison['pose_sampling_dt_s']:.3f} s`",
        f"- Pose/image sampling rate: `{comparison['pose_sampling_rate_hz']:.1f} Hz`",
        f"- Nyquist limit from pose samples: `+/-{comparison['pose_sampling_nyquist_hz']:.1f} Hz`",
        f"- 5 GHz physical Doppler max magnitude: `{comparison['physical_doppler_abs_max_5GHz_hz']:.2f} Hz`",
        f"- 28 GHz physical Doppler max magnitude: `{comparison['physical_doppler_abs_max_28GHz_hz']:.2f} Hz`",
        "",
        "This verifies that `0.1 s` sampling is appropriate for pose/image labels, but not for raw Doppler sampling.",
        "",
        "## Pathloss",
        "",
        f"- Mean 28 GHz - 5 GHz pathloss delta: `{comparison['pathloss_28GHz_minus_5GHz_mean_db']:.3f} dB`",
        f"- Theoretical frequency-only delta `20log10(28/5)`: `{comparison['theoretical_frequency_pathloss_delta_db_20log10_28_over_5']:.3f} dB`",
        f"- Mean delta error: `{comparison['pathloss_delta_error_mean_minus_theory_db']:.3f} dB`",
        "",
        "## Band Table",
        "",
        "| Band | Pathloss min/max/mean dB | Paths min/max/mean | CSI median/max dB | Physical Doppler min/max Hz |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        md.append(
            f"| {row['band']} | "
            f"{row['pathloss_min_db']:.2f}/{row['pathloss_max_db']:.2f}/{row['pathloss_mean_db']:.2f} | "
            f"{row['n_paths_min']}/{row['n_paths_max']}/{row['n_paths_mean']:.2f} | "
            f"{row['csi_median_mag_db']:.2f}/{row['csi_max_mag_db']:.2f} | "
            f"{row['physical_doppler_min_hz']:.2f}/{row['physical_doppler_max_hz']:.2f} |"
        )
    md.extend(
        [
            "",
            "## Generated Plots",
            "",
            "- `pathloss_5GHz_vs_28GHz.png`",
            "- `csi_frame_rms_5GHz_vs_28GHz.png`",
            "- `csi_heatmaps_5GHz_vs_28GHz.png`",
            "- `path_count_5GHz_vs_28GHz.png`",
            "- `physical_doppler_vs_frame_sampling.png`",
            "- `frame_sampled_doppler_bins.png`",
            "- `delay_doppler_low_rate_side_by_side.png`",
        ]
    )
    (out / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {band: load_band(path) for band, path in BANDS.items()}
    stats = {band: csi_stats(data[band]["csi_explicit_frames"]) for band in BANDS}

    save_pathloss_plot(data, OUT)
    save_csi_plots(data, stats, OUT)
    save_path_count_plot(data, OUT)
    save_doppler_plots(data, OUT)
    save_delay_doppler_plot(data, OUT)
    write_summary(data, stats, OUT)
    print(f"Wrote comparison to {OUT}")


if __name__ == "__main__":
    main()
