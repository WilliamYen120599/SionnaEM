#!/usr/bin/env python3
"""Run simple_wedge checks for CSI separation, antennas, channel plots, and pilots."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sionna
from sionna.rt import ITURadioMaterial, PathSolver, PlanarArray, Receiver, Transmitter, load_scene


BS0 = np.array([43.30127, 25.0, 0.0], dtype=np.float32)
BS1 = np.array([25.0, 43.30127, 0.0], dtype=np.float32)
UE = np.array([-1.545085, 4.7552824, 0.0], dtype=np.float32)
FS = 122.88e6
N_SC = 1024
DF = 120e3
FREQS = (np.arange(N_SC, dtype=np.float64) - (N_SC - 1) / 2.0) * DF


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple_wedge check experiments.")
    parser.add_argument("--out-dir", default="Blender Preview Images/wedge checks")
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


def pathloss_from_power(power: np.ndarray) -> tuple[float, float]:
    total = float(np.sum(power))
    if total <= 0.0:
        return total, float("inf")
    return total, float(-10.0 * np.log10(total))


def make_scene(freq_hz: float, tx_rows: int = 1, tx_cols: int = 1):
    scene = load_scene(sionna.rt.scene.simple_wedge, merge_shapes=False)
    scene.frequency = float(freq_hz)
    scene.objects["wedge"].radio_material = ITURadioMaterial("metal", itu_type="metal", thickness=100)
    scene.tx_array = PlanarArray(
        num_rows=tx_rows,
        num_cols=tx_cols,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    return scene


def solve(scene):
    return PathSolver()(
        scene,
        max_depth=1,
        los=True,
        specular_reflection=True,
        diffraction=True,
        edge_diffraction=False,
        refraction=False,
        diffuse_reflection=False,
        synthetic_array=False,
    )


def cir_cfr(paths):
    a, tau = paths.cir(sampling_frequency=FS, normalize_delays=False, out_type="numpy")
    h = paths.cfr(
        frequencies=FREQS,
        sampling_frequency=FS,
        normalize_delays=False,
        normalize=False,
        out_type="numpy",
    ).astype(np.complex64)
    return a, tau, h


def link_power(a: np.ndarray, tx_idx: int = 0, tx_ant: int = 0) -> np.ndarray:
    return np.abs(np.asarray(a[0, 0, tx_idx, tx_ant, :, 0], dtype=np.complex128)) ** 2


def link_delay(tau: np.ndarray, tx_idx: int = 0) -> np.ndarray:
    return np.asarray(tau[0, 0, tx_idx, 0, :], dtype=np.float64)


def save_line_plot(path: Path, title: str, ylabel: str, series: list[tuple[str, np.ndarray]], phase=False) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for label, h in series:
        y = phase_deg(h) if phase else mag_db(h)
        ax.plot(FREQS / 1e6, y, linewidth=1.15, label=label)
    ax.set_xlabel("Subcarrier offset (MHz)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_bar(path: Path, title: str, ylabel: str, labels: list[str], values: list[float], fmt="%.2f") -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = ax.bar(labels, values, color=["#2f6fdd", "#2c9a68", "#8a6edb", "#c9962d"][: len(labels)])
    ax.bar_label(bars, fmt=fmt)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def single_bs(freq_hz: float, out_dir: Path) -> dict[str, object]:
    band = band_title(freq_hz)
    scene = make_scene(freq_hz)
    scene.add(Transmitter(name="BS0", position=BS0, orientation=[0, 0, 0], power_dbm=0.0))
    scene.add(Receiver(name="UE", position=UE, orientation=[0, 0, 0]))
    paths = solve(scene)
    a, tau, h = cir_cfr(paths)
    h0 = h[0, 0, 0, 0, 0, :]
    power = link_power(a)
    total, pl = pathloss_from_power(power)

    np.savez_compressed(out_dir / "single_h.npz", csi=h, a=a, tau=tau, freqs=FREQS)
    save_line_plot(out_dir / "single_mag.png", f"{band} single-BS channel magnitude", "|H(f)| (dB)", [("BS0 -> UE", h0)])
    save_line_plot(out_dir / "single_phase.png", f"{band} single-BS channel phase", "Unwrapped phase (deg)", [("BS0 -> UE", h0)], phase=True)
    save_bar(out_dir / "single_pl.png", f"{band} single-BS pathloss", "Pathloss (dB)", ["BS0"], [pl])
    return {
        "csi_shape": list(h.shape),
        "total_power": total,
        "pathloss_db": pl,
    }


def multi_bs(freq_hz: float, out_dir: Path) -> dict[str, object]:
    band = band_title(freq_hz)
    scene = make_scene(freq_hz)
    scene.add(Transmitter(name="BS0", position=BS0, orientation=[0, 0, 0], power_dbm=0.0))
    scene.add(Transmitter(name="BS1", position=BS1, orientation=[0, 0, 0], power_dbm=0.0))
    scene.add(Receiver(name="UE", position=UE, orientation=[0, 0, 0]))
    paths = solve(scene)
    a, tau, h = cir_cfr(paths)
    h0 = h[0, 0, 0, 0, 0, :]
    h1 = h[0, 0, 1, 0, 0, :]
    h_sum = h0 + h1

    p0 = link_power(a, 0)
    p1 = link_power(a, 1)
    total0, pl0 = pathloss_from_power(p0)
    total1, pl1 = pathloss_from_power(p1)
    combined_power = total0 + total1
    combined_pl = float(-10.0 * np.log10(combined_power))

    np.savez_compressed(out_dir / "multi_h.npz", csi=h, a=a, tau=tau, freqs=FREQS, h_sum=h_sum)
    save_line_plot(out_dir / "sep_mag.png", f"{band} separated CSI magnitude by base station", "|H(f)| (dB)", [("BS0", h0), ("BS1", h1)])
    save_line_plot(out_dir / "sep_phase.png", f"{band} separated CSI phase by base station", "Unwrapped phase (deg)", [("BS0", h0), ("BS1", h1)], phase=True)
    save_line_plot(out_dir / "sum_mag.png", f"{band} separated CSI vs non-orthogonal sum", "|H(f)| (dB)", [("BS0", h0), ("BS1", h1), ("BS0+BS1", h_sum)])
    save_bar(out_dir / "pl_sep.png", f"{band} pathloss can be separated by base station", "Pathloss (dB)", ["BS0", "BS1"], [pl0, pl1])
    save_bar(out_dir / "pl_sum.png", f"{band} combined received power equivalent", "Pathloss (dB)", ["BS0", "BS1", "sum"], [pl0, pl1, combined_pl])
    return {
        "csi_shape": list(h.shape),
        "bs0_total_power": total0,
        "bs0_pathloss_db": pl0,
        "bs1_total_power": total1,
        "bs1_pathloss_db": pl1,
        "combined_power_sum": combined_power,
        "combined_pathloss_db": combined_pl,
    }


def antenna_experiment_b(freq_hz: float, out_dir: Path) -> dict[str, object]:
    band = band_title(freq_hz)
    """Different antenna counts by separate scene solves.

    Sionna RT uses scene.tx_array globally, so heterogeneous BS arrays are represented
    by separate solves and combined after loading/saving the channel tensors.
    """
    # BS0: one TX antenna
    s0 = make_scene(freq_hz, tx_rows=1, tx_cols=1)
    s0.add(Transmitter(name="BS0", position=BS0, orientation=[0, 0, 0], power_dbm=0.0))
    s0.add(Receiver(name="UE", position=UE, orientation=[0, 0, 0]))
    a0, tau0, h0 = cir_cfr(solve(s0))

    # BS1: four TX antennas
    s1 = make_scene(freq_hz, tx_rows=4, tx_cols=1)
    s1.add(Transmitter(name="BS1", position=BS1, orientation=[0, 0, 0], power_dbm=0.0))
    s1.add(Receiver(name="UE", position=UE, orientation=[0, 0, 0]))
    a1, tau1, h1 = cir_cfr(solve(s1))

    h0_vec = h0[0, 0, 0, 0, 0, :]
    h1_ant = h1[0, 0, 0, :, 0, :]

    p0 = link_power(a0, 0, 0)
    total0, pl0 = pathloss_from_power(p0)
    pl1_ant = []
    total1_ant = []
    for ant in range(h1_ant.shape[0]):
        p = link_power(a1, 0, ant)
        total, pl = pathloss_from_power(p)
        total1_ant.append(total)
        pl1_ant.append(pl)

    np.savez_compressed(out_dir / "ant_b.npz", bs0_csi=h0, bs1_csi=h1, bs0_a=a0, bs1_a=a1, bs0_tau=tau0, bs1_tau=tau1, freqs=FREQS)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(FREQS / 1e6, mag_db(h0_vec), label="BS0 1x1 ant0", linewidth=1.2)
    for ant in range(h1_ant.shape[0]):
        ax.plot(FREQS / 1e6, mag_db(h1_ant[ant]), label=f"BS1 4x1 ant{ant}", linewidth=0.95)
    ax.set_xlabel("Subcarrier offset (MHz)")
    ax.set_ylabel("|H(f)| (dB)")
    ax.set_title(f"{band} experiment B: different BS antenna counts via separate solves")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "ant_b_mag.png", dpi=170)
    plt.close(fig)

    labels = ["BS0 a0"] + [f"BS1 a{i}" for i in range(len(pl1_ant))]
    save_bar(out_dir / "ant_b_pl.png", f"{band} per-antenna pathloss for heterogeneous arrays", "Pathloss (dB)", labels, [pl0] + pl1_ant)
    return {
        "method": "Separate solves, because scene.tx_array is global in Sionna RT.",
        "bs0_csi_shape": list(h0.shape),
        "bs1_csi_shape": list(h1.shape),
        "bs0_pathloss_db": pl0,
        "bs1_pathloss_db_per_tx_ant": pl1_ant,
    }


def channel_matrix_check(out_dir: Path, multi: dict[str, object]) -> dict[str, object]:
    # The actual matrix is already saved in multi_h.npz; make an explicit pointer file.
    pointer = {
        "matrix_file": str(out_dir / "multi_h.npz"),
        "matrix_key": "csi",
        "plot_files": {
            "magnitude": str(out_dir / "sep_mag.png"),
            "phase": str(out_dir / "sep_phase.png"),
        },
        "indexing": {
            "BS0": "csi[0,0,0,0,0,:]",
            "BS1": "csi[0,0,1,0,0,:]",
        },
        "matrix_shape": multi["csi_shape"],
        "conversion": "|H(f)| plot uses 20*log10(abs(csi slice)); phase plot uses unwrap(angle(csi slice)).",
    }
    (out_dir / "h_info.json").write_text(json.dumps(pointer, indent=2), encoding="utf-8")
    return pointer


def pilot_experiment(freq_hz: float, out_dir: Path) -> dict[str, object]:
    band = band_title(freq_hz)
    data = np.load(out_dir / "multi_h.npz")
    h = data["csi"][0, 0, :, 0, 0, :].astype(np.complex128)  # [tx=2, freq]
    h0, h1 = h[0], h[1]

    # Orthogonal pilots over two pilot symbols.
    x_orth = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.complex128)  # [pilot_symbol, tx]
    y_orth = x_orth @ h
    h_hat_orth = np.linalg.inv(x_orth) @ y_orth

    # Non-orthogonal identical pilots: rank deficient. Least-squares/pinv cannot identify both channels.
    x_mix = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.complex128)
    y_mix = x_mix @ h
    h_hat_mix = np.linalg.pinv(x_mix) @ y_mix

    orth_err = float(np.max(np.abs(h_hat_orth - h)))
    mix_err = float(np.max(np.abs(h_hat_mix - h)))
    np.savez_compressed(out_dir / "pilot.npz", h=h, x_orth=x_orth, y_orth=y_orth, h_hat_orth=h_hat_orth, x_mix=x_mix, y_mix=y_mix, h_hat_mix=h_hat_mix, freqs=FREQS)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    axes[0].plot(FREQS / 1e6, mag_db(h0), label="true BS0")
    axes[0].plot(FREQS / 1e6, mag_db(h_hat_orth[0]), "--", label="estimated BS0")
    axes[0].plot(FREQS / 1e6, mag_db(h1), label="true BS1")
    axes[0].plot(FREQS / 1e6, mag_db(h_hat_orth[1]), "--", label="estimated BS1")
    axes[0].set_title(f"{band} orthogonal pilots recover both channels, max err={orth_err:.1e}")
    axes[1].plot(FREQS / 1e6, mag_db(h0 + h1), label="observable sum")
    axes[1].plot(FREQS / 1e6, mag_db(h_hat_mix[0]), "--", label="pinv estimate BS0")
    axes[1].plot(FREQS / 1e6, mag_db(h_hat_mix[1]), "--", label="pinv estimate BS1")
    axes[1].set_title(f"{band} identical pilots are rank deficient")
    for ax in axes:
        ax.set_xlabel("Subcarrier offset (MHz)")
        ax.set_ylabel("|H(f)| (dB)")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "pilot_cmp.png", dpi=170)
    plt.close(fig)

    save_bar(out_dir / "pilot_err.png", f"{band} pilot channel-estimation error", "Max |error|", ["orth", "same"], [orth_err, mix_err], fmt="%.2e")
    return {
        "physical_channel_has_pilots": False,
        "verification": "Sionna output csi has no pilot-symbol dimension; pilots are added after cfr as y = X @ H + noise.",
        "orthogonal_pilot_max_abs_error": orth_err,
        "same_pilot_rank": int(np.linalg.matrix_rank(x_mix)),
        "same_pilot_max_abs_error": mix_err,
        "recommended_method": "Generate H with Sionna RT, design pilot matrix X, simulate received pilots Y=XH+N, estimate H via least squares or LMMSE.",
    }


def write_csv(out_dir: Path, summary: dict[str, object]) -> None:
    rows = [
        ("freq_hz", summary["frequency_hz"]),
        ("single_pathloss_db", summary["single"]["pathloss_db"]),
        ("multi_bs0_pathloss_db", summary["multi"]["bs0_pathloss_db"]),
        ("multi_bs1_pathloss_db", summary["multi"]["bs1_pathloss_db"]),
        ("multi_sum_pathloss_db", summary["multi"]["combined_pathloss_db"]),
        ("pilot_orth_error", summary["pilot"]["orthogonal_pilot_max_abs_error"]),
        ("pilot_same_error", summary["pilot"]["same_pilot_max_abs_error"]),
    ]
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    base = Path(args.out_dir)
    base.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    for freq_hz in args.bands:
        out_dir = base / band_name(freq_hz)
        out_dir.mkdir(parents=True, exist_ok=True)
        single = single_bs(freq_hz, out_dir)
        multi = multi_bs(freq_hz, out_dir)
        ant_b = antenna_experiment_b(freq_hz, out_dir)
        h_info = channel_matrix_check(out_dir, multi)
        pilot = pilot_experiment(freq_hz, out_dir)

        summary = {
            "frequency_hz": float(freq_hz),
            "band": band_name(freq_hz),
            "single": single,
            "multi": multi,
            "antenna_experiment_b": ant_b,
            "channel_matrix": h_info,
            "pilot": pilot,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        write_csv(out_dir, summary)
        all_summaries[band_name(freq_hz)] = summary

    (base / "summary.json").write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(base), "bands": list(all_summaries)}, indent=2))


if __name__ == "__main__":
    main()
