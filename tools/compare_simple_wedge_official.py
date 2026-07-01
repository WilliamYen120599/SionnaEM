#!/usr/bin/env python3
"""Reproduce Sionna's simple_wedge tutorial plots and compare local outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")

import drjit as dr
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sionna  # noqa: E402
from sionna.rt import (
    ITURadioMaterial,
    InteractionType,
    PathSolver,
    PlanarArray,
    Receiver,
    Transmitter,
    load_scene,
)
from sionna.rt.utils import r_hat


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare local simple_wedge output with Sionna tutorial setup.")
    p.add_argument("--out-dir", default="Blender Preview Images/simple_wedge")
    p.add_argument("--local-npz", default="Blender Preview Images/simple_wedge/simple_wedge_pipeline_csi_pathloss.npz")
    p.add_argument("--num-rx", type=int, default=1000)
    p.add_argument("--rx-index", type=int, default=400)
    return p.parse_args()


def compute_gain_db(a: np.ndarray) -> np.ndarray:
    if len(a.shape) == 2:
        gain = np.abs(np.sum(a, axis=-1)) ** 2
    else:
        gain = np.abs(a) ** 2
    gain = np.where(gain == 0.0, 1e-24, gain)
    return np.squeeze(10.0 * np.log10(gain))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = load_scene(sionna.rt.scene.simple_wedge, merge_shapes=False)
    scene.frequency = 1e9
    scene.objects["wedge"].radio_material = ITURadioMaterial("metal", itu_type="metal", thickness=100)
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.rx_array = scene.tx_array

    tx_angle = 30.0 / 180.0 * dr.pi
    tx_dist = 50.0
    tx_pos = tx_dist * r_hat(dr.pi / 2.0, tx_angle)
    ref_boundary = float((dr.pi - tx_angle) / dr.pi * 180.0)
    los_boundary = float((dr.pi + tx_angle) / dr.pi * 180.0)
    scene.add(Transmitter(name="tx", position=tx_pos, orientation=[0, 0, 0]))

    rx_dist = 5.0
    phi = dr.linspace(mi.Float, 1e-2, 3.0 / 2.0 * dr.pi - 1e-2, num=args.num_rx)
    theta = dr.pi / 2.0 * dr.ones(mi.Float, args.num_rx)
    rx_pos = rx_dist * r_hat(theta, phi)
    for i in range(args.num_rx):
        scene.add(
            Receiver(
                name=f"rx-{i}",
                position=[rx_pos.x[i], rx_pos.y[i], rx_pos.z[i]],
                orientation=[0, 0, 0],
            )
        )

    paths = PathSolver()(
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
    a, tau = [np.squeeze(t) for t in paths.cir(out_type="numpy")]
    phi_deg = np.asarray(phi.numpy() / np.pi * 180.0, dtype=np.float64)

    n = args.rx_index
    a_n = np.asarray(a[n], dtype=np.complex128)
    tau_n = np.asarray(tau[n], dtype=np.float64)
    valid_n = np.abs(a_n) > 0.0
    path_power_n = np.abs(a_n[valid_n]) ** 2
    path_delay_n = tau_n[valid_n]
    total_power_n = float(np.sum(path_power_n))
    pathloss_n = float(-10.0 * np.log10(total_power_n))
    rx_power_n = -pathloss_n

    fig, ax = plt.subplots()
    ax.stem(path_delay_n / 1e-9, 10.0 * np.log10(path_power_n))
    ax.set_title(fr"Official-style CIR, receiver $\phi$: {int(phi_deg[n])}$^\circ$")
    ax.set_xlabel("Delay (ns)")
    ax.set_ylabel(r"$|a|^2$ (dB)")
    fig.tight_layout()
    cir_plot = out_dir / "official_style_simple_wedge_rx400_cir.png"
    fig.savefig(cir_plot, dpi=160)
    plt.close(fig)

    h_f_tot = np.sum(a, axis=-1)
    fig, ax = plt.subplots()
    ax.plot(phi_deg, 20.0 * np.log10(np.maximum(np.abs(h_f_tot), 1e-24)))
    ax.set_xlabel(r"Diffraction angle $\phi$ (deg)")
    ax.set_ylabel(r"Path gain $|H(f)|^2$ (dB)")
    ax.set_ylim([-100, -59])
    ax.set_xlim([0, phi_deg[-1]])
    fig.tight_layout()
    total_plot = out_dir / "official_style_simple_wedge_total_gain_vs_phi.png"
    fig.savefig(total_plot, dpi=160)
    plt.close(fig)

    interactions = np.squeeze(paths.interactions.numpy())
    valid = np.squeeze(paths.valid.numpy())
    a_los = []
    a_reflected = []
    a_diffracted = []
    for i in range(args.num_rx):
        los_index = np.where(np.logical_and(valid[i], interactions[i] == InteractionType.NONE))[0]
        ref_index = np.where(np.logical_and(valid[i], interactions[i] == InteractionType.SPECULAR))[0]
        dif_index = np.where(np.logical_and(valid[i], interactions[i] == InteractionType.DIFFRACTION))[0]
        a_los.append(0.0 if los_index.shape[0] == 0 else a[i][los_index][0])
        a_reflected.append(0.0 if ref_index.shape[0] == 0 else a[i][ref_index][0])
        a_diffracted.append(0.0 if dif_index.shape[0] == 0 else a[i][dif_index][0])
    a_los = np.asarray(a_los)
    a_reflected = np.asarray(a_reflected)
    a_diffracted = np.asarray(a_diffracted)

    g_tot_db = compute_gain_db(a)
    g_los_db = compute_gain_db(a_los)
    g_ref_db = compute_gain_db(a_reflected)
    g_dif_db = compute_gain_db(a_diffracted)
    ymax = float(np.max(g_tot_db) + 5.0)
    ymin = ymax - 45.0

    fig, ax = plt.subplots()
    ax.plot(phi_deg, g_tot_db)
    ax.plot(phi_deg, g_los_db)
    ax.plot(phi_deg, g_ref_db)
    ax.plot(phi_deg, g_dif_db)
    ax.set_ylim([ymin, ymax])
    ax.set_xlim([phi_deg[0], phi_deg[-1]])
    ax.legend(["Total", "LoS", "Reflected", "Diffracted"], loc="lower left")
    ax.set_xlabel(r"Diffraction angle $\phi$ (deg)")
    ax.set_ylabel(r"Path gain $|H(f)|^2$ (dB)")
    ax.axvline(x=ref_boundary, ymin=0, ymax=1, color="black", linestyle="--")
    ax.axvline(x=los_boundary, ymin=0, ymax=1, color="black", linestyle="--")
    ax.set_title('$f=1.0$ GHz ("metal")')
    fig.tight_layout()
    components_plot = out_dir / "official_style_simple_wedge_components_vs_phi.png"
    fig.savefig(components_plot, dpi=160)
    plt.close(fig)

    local = np.load(args.local_npz)
    local_power = np.asarray(local["first_link_path_power_linear"], dtype=np.float64)
    local_delay = np.asarray(local["first_link_path_delay_s"], dtype=np.float64)
    local_total_power = float(local["path_gain_linear"])
    local_pathloss = float(local["pathloss_db"])
    local_rx_power = float(local["rx_power_dbm"])
    local_csi_shape = tuple(np.asarray(local["csi"]).shape)

    summary = {
        "official_tutorial_reproduction": {
            "source": "https://nvlabs.github.io/sionna/rt/tutorials/Diffraction.html",
            "rx_index": n,
            "rx_angle_deg": float(phi_deg[n]),
            "n_paths": int(path_power_n.shape[0]),
            "path_power_linear": path_power_n.tolist(),
            "path_delay_s": path_delay_n.tolist(),
            "total_power_linear": total_power_n,
            "pathloss_db_at_0dbm_tx": pathloss_n,
            "rx_power_dbm_at_0dbm_tx": rx_power_n,
        },
        "local_single_receiver_pipeline": {
            "n_paths": int(local_power.shape[0]),
            "path_power_linear": local_power.tolist(),
            "path_delay_s": local_delay.tolist(),
            "total_power_linear": local_total_power,
            "pathloss_db": local_pathloss,
            "rx_power_dbm": local_rx_power,
            "csi_shape": local_csi_shape,
        },
        "differences": {
            "total_power_linear": local_total_power - total_power_n,
            "pathloss_db": local_pathloss - pathloss_n,
            "rx_power_dbm": local_rx_power - rx_power_n,
        },
        "plots": {
            "official_style_cir": str(cir_plot),
            "official_style_total_gain_vs_phi": str(total_plot),
            "official_style_components_vs_phi": str(components_plot),
        },
    }

    summary_path = out_dir / "official_style_simple_wedge_comparison.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"[pass] Saved comparison: {summary_path}")


if __name__ == "__main__":
    main()
