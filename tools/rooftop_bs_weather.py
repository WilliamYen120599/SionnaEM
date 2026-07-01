#!/usr/bin/env python3
"""Weather-condition post-processing for rooftop-BS street-canyon runs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE = Path("Blender Preview Images/mobility/rooftop_bs")
C = 299_792_458.0
DF = 30e3


# Excess specific attenuation used for this controlled simulation.
# Values are deliberately modest at 3.5 GHz; they are stress-test knobs, not a full
# ITU-R rain/fog/snow radiative transfer model.
WEATHER = {
    "sunshine": {
        "label": "Sunshine",
        "specific_attenuation_db_per_km": 0.0,
        "description": "Baseline clear-weather channel.",
    },
    "rain_light": {
        "label": "Light rain",
        "specific_attenuation_db_per_km": 0.08,
        "description": "Small excess attenuation applied per path length.",
    },
    "rain_heavy": {
        "label": "Heavy rain",
        "specific_attenuation_db_per_km": 0.70,
        "description": "Stronger excess attenuation, most visible on longer links.",
    },
    "snow": {
        "label": "Snow",
        "specific_attenuation_db_per_km": 0.20,
        "description": "Moderate excess attenuation for a snowy street canyon.",
    },
}


def mag_db(h: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(np.abs(h), 1e-30))


def delay_doppler_spectrum(h_time_freq: np.ndarray, freqs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h_delay = np.fft.ifft(np.fft.fftshift(h_time_freq, axes=1), axis=1, norm="ortho")
    dd = np.fft.fftshift(np.fft.fft(h_delay, axis=0, norm="ortho"), axes=0)
    num_time = h_time_freq.shape[0]
    num_sc = h_time_freq.shape[1]
    doppler_res = DF / num_time
    if num_sc > 1:
        subcarrier_spacing = float(np.median(np.diff(freqs)))
    else:
        subcarrier_spacing = DF
    delay_res = (1.0 / subcarrier_spacing) / num_sc
    doppler_bins = (np.arange(num_time) - num_time // 2) * doppler_res
    delay_bins_ns = np.arange(num_sc) * delay_res / 1e-9
    return np.abs(dd), delay_bins_ns, doppler_bins


def link_names(summary: dict[str, object]) -> tuple[list[str], list[str]]:
    tx_names = [x["name"] for x in summary["base_stations"]]
    rx_names = [x["name"] for x in summary["receivers"]]
    return tx_names, rx_names


def attenuate_case(case: str, weather_name: str, params: dict[str, object]) -> dict[str, object]:
    case_dir = BASE / case
    out_dir = case_dir / "weather" / weather_name
    out_dir.mkdir(parents=True, exist_ok=True)
    base_summary = json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))
    tx_names, rx_names = link_names(base_summary)
    data = np.load(case_dir / "rt_results.npz")
    a = data["a"].astype(np.complex128)
    tau = data["tau"].astype(np.float64)
    csi = data["csi"].astype(np.complex128)
    freqs = data["freqs"].astype(np.float64)
    gamma = float(params["specific_attenuation_db_per_km"])

    a_weather = a.copy()
    csi_weather = csi.copy()
    link_rows = []
    for rx_i, rx_name in enumerate(rx_names):
        for tx_i, tx_name in enumerate(tx_names):
            tau_link = tau[rx_i, 0, tx_i, 0, :]
            dist_km = np.maximum(tau_link * C / 1000.0, 0.0)
            amp_scale_per_path = 10.0 ** (-(gamma * dist_km) / 20.0)
            a_weather[rx_i, 0, tx_i, 0, :, 0] *= amp_scale_per_path

            base_power = np.abs(a[rx_i, 0, tx_i, 0, :, 0]) ** 2
            weather_power = np.abs(a_weather[rx_i, 0, tx_i, 0, :, 0]) ** 2
            base_total = float(np.sum(base_power))
            weather_total = float(np.sum(weather_power))
            if base_total > 0.0:
                link_amp_scale = np.sqrt(weather_total / base_total)
                weighted_dist_km = float(np.sum(base_power * dist_km) / base_total)
            else:
                link_amp_scale = 0.0
                weighted_dist_km = float("nan")
            csi_weather[rx_i, 0, tx_i, 0, :, :] *= link_amp_scale
            base_pl = float(-10.0 * np.log10(base_total)) if base_total > 0 else float("inf")
            weather_pl = float(-10.0 * np.log10(weather_total)) if weather_total > 0 else float("inf")
            link_rows.append({
                "tx": tx_name,
                "rx": rx_name,
                "weighted_path_distance_km": weighted_dist_km,
                "specific_attenuation_db_per_km": gamma,
                "excess_loss_db": weather_pl - base_pl,
                "pathloss_db": weather_pl,
                "baseline_pathloss_db": base_pl,
                "total_power": weather_total,
            })

    np.savez_compressed(out_dir / "rt_weather.npz", a=a_weather.astype(np.complex64),
                        tau=tau.astype(np.float32), csi=csi_weather.astype(np.complex64), freqs=freqs)
    save_pathloss_plot(out_dir, weather_name, link_rows)
    save_csi_plot(out_dir, weather_name, csi_weather, freqs, tx_names, rx_names)
    dd_summary = save_delay_doppler(out_dir, weather_name, csi_weather, freqs, tx_names, rx_names)
    summary = {
        "case": case,
        "weather": weather_name,
        "weather_parameters": params,
        "links": link_rows,
        "delay_doppler_peaks": dd_summary,
        "outputs": {
            "rt_weather": str(out_dir / "rt_weather.npz"),
            "pathloss_plot": str(out_dir / "pathloss_links.png"),
            "csi_plot": str(out_dir / "csi_links.png"),
            "delay_doppler_overview": str(out_dir / "delay_doppler_overview.png"),
            "delay_doppler_dir": str(out_dir / "delay_doppler"),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def save_pathloss_plot(out_dir: Path, weather_name: str, links: list[dict[str, object]]) -> None:
    labels = [f"{x['tx']}\n{x['rx']}" for x in links]
    values = [float(x["pathloss_db"]) for x in links]
    excess = [float(x["excess_loss_db"]) for x in links]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    bars = axes[0].bar(labels, values, color="#2f6fdd")
    axes[0].bar_label(bars, fmt="%.2f", fontsize=8)
    axes[0].set_ylabel("Pathloss (dB)")
    axes[0].set_title(f"{weather_name}: weather-adjusted pathloss")
    axes[0].grid(axis="y", alpha=0.25)
    bars = axes[1].bar(labels, excess, color="#c9962d")
    axes[1].bar_label(bars, fmt="%.3f", fontsize=8)
    axes[1].set_ylabel("Excess loss vs sunshine (dB)")
    axes[1].set_title("Weather-only excess loss")
    axes[1].grid(axis="y", alpha=0.25)
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(out_dir / "pathloss_links.png", dpi=170)
    plt.close(fig)


def save_csi_plot(out_dir: Path, weather_name: str, csi: np.ndarray, freqs: np.ndarray, tx_names: list[str], rx_names: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for rx_i, rx_name in enumerate(rx_names):
        for tx_i, tx_name in enumerate(tx_names):
            h = csi[rx_i, 0, tx_i, 0, 0, :]
            ax.plot(freqs / 1e6, mag_db(h), linewidth=1.0, label=f"{tx_name}->{rx_name}")
    ax.set_xlabel("Subcarrier offset (MHz)")
    ax.set_ylabel("|H(f)| (dB)")
    ax.set_title(f"{weather_name}: CSI magnitude")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "csi_links.png", dpi=170)
    plt.close(fig)


def save_delay_doppler(out_dir: Path, weather_name: str, csi: np.ndarray, freqs: np.ndarray, tx_names: list[str], rx_names: list[str]) -> list[dict[str, object]]:
    dd_dir = out_dir / "delay_doppler"
    dd_dir.mkdir(exist_ok=True)
    summaries = []
    for rx_i, rx_name in enumerate(rx_names):
        for tx_i, tx_name in enumerate(tx_names):
            mag, delay_ns, doppler_hz = delay_doppler_spectrum(csi[rx_i, 0, tx_i, 0, :, :], freqs)
            delay_mask = delay_ns <= 700.0
            peak = np.unravel_index(np.argmax(mag), mag.shape)
            summaries.append({
                "tx": tx_name,
                "rx": rx_name,
                "peak_delay_ns": float(delay_ns[peak[1]]),
                "peak_doppler_hz": float(doppler_hz[peak[0]]),
                "peak_magnitude": float(mag[peak]),
            })
            x, y = np.meshgrid(delay_ns[delay_mask], doppler_hz)
            z = mag[:, delay_mask]
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_surface(x, y, z, cmap="viridis", edgecolor="none", linewidth=0)
            ax.set_xlabel("Delay (ns)")
            ax.set_ylabel("Doppler (Hz)")
            ax.set_zlabel("Magnitude")
            ax.set_title(f"{weather_name}: {tx_name} -> {rx_name}")
            ax.view_init(elev=35, azim=-45)
            fig.tight_layout()
            fig.savefig(dd_dir / f"dd_3d_{tx_name}_to_{rx_name}.png", dpi=170)
            plt.close(fig)

    fig, axes = plt.subplots(len(rx_names), len(tx_names), figsize=(4.8 * len(tx_names), 3.4 * len(rx_names)), squeeze=False)
    for rx_i, rx_name in enumerate(rx_names):
        for tx_i, tx_name in enumerate(tx_names):
            mag, delay_ns, doppler_hz = delay_doppler_spectrum(csi[rx_i, 0, tx_i, 0, :, :], freqs)
            delay_mask = delay_ns <= 700.0
            ax = axes[rx_i, tx_i]
            ax.imshow(20.0 * np.log10(np.maximum(mag[:, delay_mask], 1e-30)),
                      aspect="auto", origin="lower",
                      extent=[float(delay_ns[delay_mask][0]), float(delay_ns[delay_mask][-1]),
                              float(doppler_hz[0]), float(doppler_hz[-1])],
                      cmap="magma")
            ax.set_title(f"{tx_name} -> {rx_name}")
            ax.set_xlabel("Delay (ns)")
            ax.set_ylabel("Doppler (Hz)")
    fig.suptitle(f"{weather_name}: delay-Doppler overview")
    fig.tight_layout()
    fig.savefig(out_dir / "delay_doppler_overview.png", dpi=170)
    plt.close(fig)
    (dd_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return summaries


def comparison_plots(all_summaries: dict[str, dict[str, dict[str, object]]]) -> None:
    comp = BASE / "weather" / "comparison"
    comp.mkdir(exist_ok=True)
    weather_order = list(WEATHER)
    for case, summaries in all_summaries.items():
        sunshine_links = summaries["sunshine"]["links"]
        link_keys = [(x["tx"], x["rx"]) for x in sunshine_links]
        labels = [f"{tx}\n{rx}" for tx, rx in link_keys]
        x = np.arange(len(link_keys))
        width = 0.18
        fig, ax = plt.subplots(figsize=(12, 5.2))
        for i, weather_name in enumerate(weather_order):
            links = {(r["tx"], r["rx"]): r for r in summaries[weather_name]["links"]}
            vals = [float(links[k]["pathloss_db"]) for k in link_keys]
            ax.bar(x + (i - 1.5) * width, vals, width, label=weather_name)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25)
        ax.set_ylabel("Pathloss (dB)")
        ax.set_title(f"{case}: pathloss by weather condition")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(comp / f"{case}_pathloss_weather.png", dpi=170)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(12, 5.2))
        for i, weather_name in enumerate(weather_order):
            links = {(r["tx"], r["rx"]): r for r in summaries[weather_name]["links"]}
            vals = [float(links[k]["excess_loss_db"]) for k in link_keys]
            ax.bar(x + (i - 1.5) * width, vals, width, label=weather_name)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25)
        ax.set_ylabel("Excess loss vs sunshine (dB)")
        ax.set_title(f"{case}: weather-only excess attenuation")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(comp / f"{case}_excess_loss_weather.png", dpi=170)
        plt.close(fig)

        make_csi_comparison(case, comp, link_keys[: min(3, len(link_keys))])
        make_dd_peak_comparison(case, summaries, comp, link_keys)
        make_delta_comparisons(case, summaries, comp, link_keys)
        make_side_by_side_comparisons(case, comp)
    (comp / "summary.json").write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")


def save_image_grid(out_path: Path, image_paths: list[Path], title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.reshape(-1)
    for ax, weather_name, image_path in zip(axes, WEATHER, image_paths):
        img = plt.imread(image_path)
        ax.imshow(img)
        ax.set_title(weather_name)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def make_side_by_side_comparisons(case: str, comp: Path) -> None:
    side_dir = comp / "side_by_side" / case
    side_dir.mkdir(parents=True, exist_ok=True)

    top_level_plots = {
        "pathloss_links": "pathloss_links.png",
        "csi_links": "csi_links.png",
        "delay_doppler_overview": "delay_doppler_overview.png",
    }
    for label, filename in top_level_plots.items():
        paths = [BASE / case / "weather" / weather_name / filename for weather_name in WEATHER]
        if all(p.exists() for p in paths):
            save_image_grid(side_dir / f"{case}_{label}_side_by_side.png", paths, f"{case}: {label} by weather")

    dd_ref_dir = BASE / case / "weather" / "sunshine" / "delay_doppler"
    for ref in sorted(dd_ref_dir.glob("dd_3d_*.png")):
        paths = [BASE / case / "weather" / weather_name / "delay_doppler" / ref.name for weather_name in WEATHER]
        if all(p.exists() for p in paths):
            save_image_grid(side_dir / f"{case}_{ref.stem}_side_by_side.png", paths, f"{case}: {ref.stem} by weather")


def make_csi_comparison(case: str, comp: Path, link_keys: list[tuple[str, str]]) -> None:
    fig, axes = plt.subplots(len(link_keys), 1, figsize=(9.5, 3.5 * len(link_keys)), squeeze=False)
    base_summary = json.loads((BASE / case / "summary.json").read_text(encoding="utf-8"))
    tx_names, rx_names = link_names(base_summary)
    for row, (tx, rx) in enumerate(link_keys):
        ax = axes[row, 0]
        for weather_name in WEATHER:
            data = np.load(BASE / case / "weather" / weather_name / "rt_weather.npz")
            csi = data["csi"]
            freqs = data["freqs"]
            tx_i = tx_names.index(tx)
            rx_i = rx_names.index(rx)
            ax.plot(freqs / 1e6, mag_db(csi[rx_i, 0, tx_i, 0, 0, :]), linewidth=1.0, label=weather_name)
        ax.set_title(f"{case}: CSI weather overlay, {tx}->{rx}")
        ax.set_xlabel("Subcarrier offset (MHz)")
        ax.set_ylabel("|H(f)| (dB)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(comp / f"{case}_csi_weather_overlay.png", dpi=170)
    plt.close(fig)


def make_dd_peak_comparison(case: str, summaries: dict[str, dict[str, object]], comp: Path, link_keys: list[tuple[str, str]]) -> None:
    labels = [f"{tx}\n{rx}" for tx, rx in link_keys]
    x = np.arange(len(link_keys))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 5.2))
    for i, weather_name in enumerate(WEATHER):
        peaks = {(r["tx"], r["rx"]): r for r in summaries[weather_name]["delay_doppler_peaks"]}
        vals = [20.0 * np.log10(max(float(peaks[k]["peak_magnitude"]), 1e-30)) for k in link_keys]
        ax.bar(x + (i - 1.5) * width, vals, width, label=weather_name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25)
    ax.set_ylabel("Peak DD magnitude (dB)")
    ax.set_title(f"{case}: delay-Doppler peak magnitude by weather")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(comp / f"{case}_delay_doppler_peak_weather.png", dpi=170)
    plt.close(fig)


def make_delta_comparisons(case: str, summaries: dict[str, dict[str, object]], comp: Path, link_keys: list[tuple[str, str]]) -> None:
    """Plots on delta scales so short-link weather effects are visible."""
    labels = [f"{tx}\n{rx}" for tx, rx in link_keys]
    x = np.arange(len(link_keys))
    width = 0.24

    fig, ax = plt.subplots(figsize=(12, 5.2))
    for i, weather_name in enumerate(["rain_light", "rain_heavy", "snow"]):
        links = {(r["tx"], r["rx"]): r for r in summaries[weather_name]["links"]}
        vals = [1000.0 * float(links[k]["excess_loss_db"]) for k in link_keys]
        ax.bar(x + (i - 1.0) * width, vals, width, label=weather_name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25)
    ax.set_ylabel("Excess loss vs sunshine (milli-dB)")
    ax.set_title(f"{case}: weather excess loss on a magnified scale")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(comp / f"{case}_excess_loss_millidb.png", dpi=170)
    plt.close(fig)

    base_summary = json.loads((BASE / case / "summary.json").read_text(encoding="utf-8"))
    tx_names, rx_names = link_names(base_summary)
    sunshine = np.load(BASE / case / "weather" / "sunshine" / "rt_weather.npz")
    h_sun = sunshine["csi"]
    freqs = sunshine["freqs"]
    selected = link_keys[: min(3, len(link_keys))]
    fig, axes = plt.subplots(len(selected), 1, figsize=(9.5, 3.5 * len(selected)), squeeze=False)
    for row, (tx, rx) in enumerate(selected):
        tx_i = tx_names.index(tx)
        rx_i = rx_names.index(rx)
        ref = mag_db(h_sun[rx_i, 0, tx_i, 0, 0, :])
        ax = axes[row, 0]
        for weather_name in ["rain_light", "rain_heavy", "snow"]:
            data = np.load(BASE / case / "weather" / weather_name / "rt_weather.npz")
            h = data["csi"]
            delta = mag_db(h[rx_i, 0, tx_i, 0, 0, :]) - ref
            ax.plot(freqs / 1e6, delta, linewidth=1.15, label=weather_name)
        ax.set_title(f"{case}: CSI delta vs sunshine, {tx}->{rx}")
        ax.set_xlabel("Subcarrier offset (MHz)")
        ax.set_ylabel("Delta |H(f)| (dB)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(comp / f"{case}_csi_delta_vs_sunshine.png", dpi=170)
    plt.close(fig)

    sunshine_peaks = {(r["tx"], r["rx"]): r for r in summaries["sunshine"]["delay_doppler_peaks"]}
    fig, ax = plt.subplots(figsize=(12, 5.2))
    for i, weather_name in enumerate(["rain_light", "rain_heavy", "snow"]):
        peaks = {(r["tx"], r["rx"]): r for r in summaries[weather_name]["delay_doppler_peaks"]}
        vals = []
        for key in link_keys:
            base = 20.0 * np.log10(max(float(sunshine_peaks[key]["peak_magnitude"]), 1e-30))
            val = 20.0 * np.log10(max(float(peaks[key]["peak_magnitude"]), 1e-30))
            vals.append(val - base)
        ax.bar(x + (i - 1.0) * width, vals, width, label=weather_name)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25)
    ax.set_ylabel("DD peak magnitude delta (dB)")
    ax.set_title(f"{case}: delay-Doppler peak delta vs sunshine")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(comp / f"{case}_delay_doppler_peak_delta_vs_sunshine.png", dpi=170)
    plt.close(fig)


def main() -> None:
    all_summaries = {}
    for case in ["one_bs", "two_bs"]:
        all_summaries[case] = {}
        for weather_name, params in WEATHER.items():
            all_summaries[case][weather_name] = attenuate_case(case, weather_name, params)
    comparison_plots(all_summaries)
    print(json.dumps({"out_dir": str(BASE), "comparison": str(BASE / "weather" / "comparison")}, indent=2))


if __name__ == "__main__":
    main()
