#!/usr/bin/env python3
"""Compare floor_wall floor scene outputs with available official references."""

from __future__ import annotations

import argparse
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

import sionna  # noqa: E402
from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene
from rt_output_utils import save_rf_outputs


OFFICIAL_FLOOR_WALL_PREVIEW_URL = "https://nvlabs.github.io/sionna/_images/floor_wall.png"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare local floor_wall outputs with official references when available.")
    p.add_argument("--tx", nargs=3, type=float, default=(0.0, 0.0, 2.0),
                   help="TX position x y z")
    p.add_argument("--rx", nargs=3, type=float, default=(-2.0, 0.0, 2.0),
                   help="RX position x y z")
    p.add_argument("--frequency", type=float, default=28e9, help="Carrier frequency (Hz)")
    p.add_argument("--sampling-frequency", type=float, default=122.88e6,
                   help="CIR sampling frequency (Hz)")
    p.add_argument("--tx-power-dbm", type=float, default=0.0, help="Transmit power in dBm")
    p.add_argument("--num-subcarriers", type=int, default=64, help="Number of CSI/CFR subcarriers to save")
    p.add_argument("--subcarrier-spacing", type=float, default=30e3, help="CSI subcarrier spacing in Hz")
    p.add_argument("--max-depth", type=int, default=1, help="Path depth")
    p.add_argument("--out-dir", default="Blender Preview Images/floor_wall", help="Directory for outputs")
    p.add_argument("--out-prefix", default="floor_wall_pipeline", help="File prefix for floor_wall pipeline outputs")
    p.add_argument("--official-scene-preview", default=OFFICIAL_FLOOR_WALL_PREVIEW_URL,
                   help="Path or URL to official scene preview image")
    p.add_argument("--official-csi-npz", default="", help="Optional official npz to compare CSI/pathloss")
    return p.parse_args()


def _is_url(value: str) -> bool:
    return value.startswith("http://") or value.startswith("https://")


def _ensure_official_image(path: Path, source: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path

    if _is_url(source):
        urllib.request.urlretrieve(source, path)
    else:
        src = Path(source)
        if not src.exists():
            raise FileNotFoundError(f"Official scene reference not found: {src}")
        path.write_bytes(src.read_bytes())
    return path


def solve_floor_wall(tx: np.ndarray, rx: np.ndarray, freq: float, max_depth: int, sampling_frequency: float):
    scene = load_scene(sionna.rt.scene.floor_wall)
    scene.frequency = float(freq)
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
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
    scene.add(Transmitter(name="tx", position=tx, orientation=[0, 0, 0]))
    scene.add(Receiver(name="rx", position=rx, orientation=[0, 0, 0]))

    solver = PathSolver()
    t0 = time.perf_counter()
    paths = solver(
        scene,
        max_depth=max_depth,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        diffraction=False,
        edge_diffraction=False,
        refraction=False,
        synthetic_array=False,
    )
    t_ms = (time.perf_counter() - t0) * 1e3
    a, tau = paths.cir(
        sampling_frequency=float(sampling_frequency),
        normalize_delays=False,
        out_type="numpy",
    )

    return scene, paths, a, tau, t_ms


def _compare_metric_series(local: np.ndarray, official: np.ndarray, metric_name: str, out_path: Path) -> dict[str, float]:
    local = np.asarray(local, dtype=np.float64).reshape(-1)
    official = np.asarray(official, dtype=np.float64).reshape(-1)
    n = min(local.size, official.size)
    local_v = local[:n]
    official_v = official[:n]

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = np.arange(n)
    ax.plot(x, local_v, label="local", linewidth=1.2)
    ax.plot(x, official_v, "--", label="official", linewidth=1.2)
    ax.set_title(f"{metric_name} comparison")
    ax.set_xlabel("Index")
    ax.set_ylabel(metric_name)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)

    local_padded = np.pad(local_v, (0, max(0, n - local_v.size)))
    official_padded = np.pad(official_v, (0, max(0, n - official_v.size)))
    diff = float(np.max(np.abs(local_padded - official_padded))) if n > 0 else float("nan")
    return {
        "n_points": int(n),
        "max_abs_diff": diff,
        "l2_diff": float(np.linalg.norm(local_v - official_v)),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tx = np.asarray(args.tx, dtype=np.float32)
    rx = np.asarray(args.rx, dtype=np.float32)

    scene, paths, a, tau, solve_ms = solve_floor_wall(
        tx=tx,
        rx=rx,
        freq=args.frequency,
        max_depth=args.max_depth,
        sampling_frequency=args.sampling_frequency,
    )

    rf = save_rf_outputs(
        paths=paths,
        a=a,
        tau=tau,
        out_dir=out_dir,
        prefix=args.out_prefix,
        tx=tx,
        rx=rx,
        frequency=args.frequency,
        sampling_frequency=args.sampling_frequency,
        tx_power_dbm=args.tx_power_dbm,
        num_subcarriers=args.num_subcarriers,
        subcarrier_spacing=args.subcarrier_spacing,
        solve_ms=solve_ms,
    )

    official_scene_preview = out_dir / "official_floor_wall_scene_preview.png"
    has_official_scene_reference = False
    official_scene_ref = Path(args.official_scene_preview)

    try:
        if _is_url(args.official_scene_preview):
            _ensure_official_image(official_scene_preview, args.official_scene_preview)
        else:
            if not official_scene_ref.exists():
                raise FileNotFoundError(f"Official scene preview not found: {official_scene_ref}")
            official_scene_preview.write_bytes(official_scene_ref.read_bytes())
        has_official_scene_reference = True
    except Exception as exc:
        print(f"[warn] Could not fetch official scene reference image: {exc}")

    compare_summary = {
        "scene": "floor_wall",
        "scene_path": str(sionna.rt.scene.floor_wall),
        "solve_ms": float(solve_ms),
        "tx": tx.tolist(),
        "rx": rx.tolist(),
        "frequency_hz": float(args.frequency),
        "max_depth": args.max_depth,
        "outputs": {
            "local_csi_pathloss_npz": str(rf["npz"]),
            "local_pathloss_plot": str(rf["pathloss_plot"]),
            "local_csi_magnitude_plot": str(rf["csi_magnitude_plot"]),
            "official_scene_reference": str(official_scene_preview),
        },
        "official_comparison": {
            "has_official_scene_reference": has_official_scene_reference,
            "official_scene_reference_source": args.official_scene_preview,
            "csi_metrics": None,
            "pathloss_metrics": None,
            "status": "pending",
        },
        "notes": [
            "Sionna does not publish official floor_wall CSI or pathloss plot references in the public tutorial API.",
            "This script saves local floor_wall outputs and creates comparison artifacts when official references are supplied.",
        ],
    }

    official_npy = args.official_csi_npz.strip() if isinstance(args.official_csi_npz, str) else str(args.official_csi_npz)
    official_npy_path = Path(official_npy) if official_npy else None
    if official_npy_path and official_npy_path.exists() and official_npy_path.is_file():
        data = np.load(str(official_npy_path))
        local_npz = np.load(rf["npz"])

        if "pathloss_db" in data:
            local_pathloss = np.asarray(local_npz["pathloss_db"], dtype=np.float64).reshape(-1)
            official_pathloss = np.asarray(data["pathloss_db"], dtype=np.float64).reshape(-1)
            metrics = _compare_metric_series(local_pathloss, official_pathloss, "Pathloss (dB)", out_dir / "compare_floor_wall_pathloss.png")
            compare_summary["official_comparison"]["pathloss_metrics"] = metrics

        if "csi" in data:
            local_csi = np.asarray(local_npz["csi"], dtype=np.complex64).reshape(-1)
            official_csi = np.asarray(data["csi"], dtype=np.complex64).reshape(-1)
            local_mag = 20.0 * np.log10(np.maximum(np.abs(local_csi), 1e-24))
            official_mag = 20.0 * np.log10(np.maximum(np.abs(official_csi), 1e-24))
            metrics = _compare_metric_series(local_mag, official_mag, "|CSI| (dB)", out_dir / "compare_floor_wall_csi_magnitude.png")
            compare_summary["official_comparison"]["csi_metrics"] = metrics

        compare_summary["official_comparison"]["status"] = "completed"
        compare_summary["official_comparison"]["official_reference_npz"] = str(official_npy_path)
    else:
        compare_summary["official_comparison"]["status"] = "no_official_numerical_reference"

    if has_official_scene_reference:
        # No local scene preview is generated here; this is a documentation layout comparison.
        # Still create a one-panel note artifact to make the intent explicit.
        compare_text = out_dir / "compare_floor_wall_scene_preview_note.png"
        fig, ax = plt.subplots(figsize=(10.2, 5.3))
        ax.axis("off")
        ax.text(
            0.05,
            0.7,
            "Official scene reference available: floor_wall.png",
            fontsize=14,
        )
        ax.text(
            0.05,
            0.56,
            "Sionna does not expose an official CSI/pathloss plot reference file for floor_wall.",
            fontsize=12,
        )
        ax.text(
            0.05,
            0.42,
            f"Reference saved at: {official_scene_preview}",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(compare_text, dpi=150)
        plt.close(fig)
        compare_summary["outputs"]["official_scene_preview_note"] = str(compare_text)

    summary_path = out_dir / "compare_floor_wall_official.json"
    summary_path.write_text(json.dumps(compare_summary, indent=2), encoding="utf-8")

    print(f"[pass] Saved floor_wall outputs in: {out_dir}")
    print(f"[pass] Saved floor_wall comparison summary: {summary_path}")


if __name__ == "__main__":
    main()
