#!/usr/bin/env python3
"""Smoke-test pipeline on Sionna-RT's built-in simple_wedge scene."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sionna  # noqa: E402
from sionna.rt import (
    ITURadioMaterial,
    PathSolver,
    PlanarArray,
    Receiver,
    Transmitter,
    load_scene,
)
from rt_output_utils import save_path_power_plot, save_rf_outputs, scene_output_dir


def wedge_position(distance: float, phi_deg: float) -> np.ndarray:
    """Return the xy-plane position convention used by Sionna's diffraction tutorial."""
    phi = np.deg2rad(phi_deg)
    return np.array([distance * np.cos(phi), distance * np.sin(phi), 0.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Verify the built-in simple_wedge RT pipeline.")
    p.add_argument("--tx", nargs=3, type=float, default=tuple(wedge_position(50.0, 30.0)),
                   help="TX position x y z")
    p.add_argument("--rx", nargs=3, type=float, default=tuple(wedge_position(5.0, 108.0)),
                   help="RX position x y z")
    p.add_argument("--frequency", type=float, default=1e9, help="Carrier frequency (Hz)")
    p.add_argument("--sampling-frequency", type=float, default=122.88e6, help="CIR sampling frequency (Hz)")
    p.add_argument("--tx-power-dbm", type=float, default=0.0, help="Transmit power in dBm")
    p.add_argument("--num-subcarriers", type=int, default=64, help="Number of CSI/CFR subcarriers to save")
    p.add_argument("--subcarrier-spacing", type=float, default=30e3, help="CSI/CFR subcarrier spacing in Hz")
    p.add_argument("--max-depth", type=int, default=1, help="Path depth")
    p.add_argument("--diffraction", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable diffraction")
    p.add_argument("--edge-diffraction", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable edge diffraction")
    p.add_argument("--diffuse-reflection", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable diffuse reflection")
    p.add_argument("--specular-reflection", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable specular reflection")
    p.add_argument("--out-dir", default="Blender Preview Images", help="Directory for output images")
    p.add_argument("--out-prefix", default="simple_wedge_pipeline", help="Output file prefix")
    p.add_argument("--scene-name", default="simple_wedge", help="Output folder name under --out-dir")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = scene_output_dir(args.out_dir, args.scene_name)

    scene = load_scene(sionna.rt.scene.simple_wedge, merge_shapes=False)
    scene.frequency = float(args.frequency)
    scene.objects["wedge"].radio_material = ITURadioMaterial("metal", itu_type="metal", thickness=100)
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.add(
        Transmitter(name="tx", position=np.array(args.tx, dtype=np.float32),
                    power_dbm=float(args.tx_power_dbm), orientation=[0, 0, 0])
    )
    scene.add(
        Receiver(name="rx", position=np.array(args.rx, dtype=np.float32),
                 orientation=[0, 0, 0])
    )

    solver = PathSolver()
    t0 = time.perf_counter()
    paths = solver(
        scene,
        max_depth=args.max_depth,
        los=True,
        specular_reflection=args.specular_reflection,
        diffuse_reflection=args.diffuse_reflection,
        diffraction=args.diffraction,
        edge_diffraction=args.edge_diffraction,
        refraction=False,
        synthetic_array=False,
    )
    t_rt = time.perf_counter() - t0

    a, tau = paths.cir(
        sampling_frequency=args.sampling_frequency,
        normalize_delays=False,
        out_type="numpy",
    )

    a0 = np.asarray(a[0, 0, 0, 0, :, 0], dtype=np.complex128)
    tau0 = np.asarray(tau[0, 0, 0, 0, :], dtype=np.float64)
    n_paths = a0.shape[0]
    power = np.abs(a0) ** 2
    total_power = float(np.sum(power))

    print(f"[pass] Loaded scene: {sionna.rt.scene.simple_wedge}")
    print(f"[pass] Path count = {n_paths}, total power = {total_power:.3e}, solve = {t_rt*1e3:.2f} ms")

    npz_out = out_dir / f"{args.out_prefix}_result.npz"
    np.savez_compressed(
        npz_out,
        a=a,
        tau=tau,
        tx=args.tx,
        rx=args.rx,
        scene="simple_wedge",
        scene_path=sionna.rt.scene.simple_wedge,
        frequency_hz=args.frequency,
        max_depth=args.max_depth,
        diffraction=args.diffraction,
        edge_diffraction=args.edge_diffraction,
        specular_reflection=args.specular_reflection,
        diffuse_reflection=args.diffuse_reflection,
        n_paths=n_paths,
        total_power=total_power,
        solve_time_ms=t_rt * 1e3,
    )

    plot_path = out_dir / f"{args.out_prefix}_path_powers.png"
    save_path_power_plot(plot_path, power, "simple_wedge", tau0)

    rf_out = save_rf_outputs(
        paths=paths,
        a=a,
        tau=tau,
        out_dir=out_dir,
        prefix=args.out_prefix,
        tx=np.array(args.tx, dtype=np.float32),
        rx=np.array(args.rx, dtype=np.float32),
        frequency=args.frequency,
        sampling_frequency=args.sampling_frequency,
        tx_power_dbm=args.tx_power_dbm,
        num_subcarriers=args.num_subcarriers,
        subcarrier_spacing=args.subcarrier_spacing,
        solve_ms=t_rt * 1e3,
    )

    print(
        f"[pass] Pathloss = {rf_out['pathloss_db']:.2f} dB, "
        f"Rx power = {rf_out['rx_power_dbm']:.2f} dBm, CSI shape = {rf_out['csi_shape']}"
    )
    print(f"[pass] Saved plot: {plot_path}")
    print(f"[pass] Saved results: {npz_out}")
    print(f"[pass] Saved RF: {rf_out['npz']}, {rf_out['pathloss_plot']}, {rf_out['csi_magnitude_plot']}")


if __name__ == "__main__":
    main()
