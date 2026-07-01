#!/usr/bin/env python3
"""Quick validation for uploaded XML scenes extracted from meshes.zip."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mitsuba as mi
mi.set_variant("llvm_ad_mono_polarized")

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sionna  # noqa: E402
from sionna.rt import ITURadioMaterial, PathSolver, PlanarArray, Receiver, Transmitter, load_scene
from rt_output_utils import save_path_power_plot, save_rf_outputs, scene_output_dir
from run_xml_rt_quick import _ensure_radio_materials, _flatten_mesh_groups, _prepare_scene_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate uploaded mesh scene.")
    p.add_argument(
        "--scene",
        default="test_scenes/meshes/normalized_scene_sionna_materials_flat_shapes.xml",
        help="Built-in Sionna scene name, path to scene XML, or zip package."
    )
    p.add_argument("--scene-xml", default="", help="For zip input, XML path/name to select.")
    p.add_argument("--workdir", default="test_scenes", help="Directory for extracted zip scenes.")
    p.add_argument("--tx", nargs=3, type=float, default=(0.0, 0.0, 2.0))
    p.add_argument("--rx", nargs=3, type=float, default=(20.0, 0.0, 2.0))
    p.add_argument("--max-depth", type=int, default=1)
    p.add_argument("--frequency", type=float, default=28e9)
    p.add_argument("--sampling-frequency", type=float, default=122.88e6)
    p.add_argument("--tx-power-dbm", type=float, default=0.0)
    p.add_argument("--num-subcarriers", type=int, default=64)
    p.add_argument("--subcarrier-spacing", type=float, default=30e3)
    p.add_argument("--diffuse", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--specular", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--diffraction", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--edge-diffraction", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--refraction", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--merge-shapes", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--material", action="append", default=[],
                   help="Override object material as object:itu_type[:thickness], e.g. wedge:metal:100")
    p.add_argument("--out-dir", default="Blender Preview Images")
    p.add_argument("--out-prefix", default="uploaded_meshes")
    p.add_argument("--scene-name", default="", help="Output folder name under --out-dir")
    return p.parse_args()


def resolve_scene(scene_arg: str, scene_xml: str, workdir: str) -> tuple[str, str, bool]:
    """Return loadable scene path/name, output scene name, and whether it is built-in."""
    built_in = getattr(sionna.rt.scene, scene_arg, None)
    if built_in is not None:
        return built_in, scene_arg, True

    scene_path = Path(scene_arg)
    work_root = Path(workdir)
    prepared = _prepare_scene_path(scene_path, scene_xml, work_root)
    prepared, n_converted = _ensure_radio_materials(prepared)
    prepared, n_flat = _flatten_mesh_groups(prepared)

    if n_converted:
        print(f"[info] Auto-converted {n_converted} materials to Sionna-compatible radio materials.")
    if n_flat:
        print(f"[info] Flattened {n_flat} grouped mesh entries into top-level ply shapes.")

    output_scene_name = scene_path.stem if scene_path.suffix.lower() == ".zip" else prepared.stem
    return str(prepared), output_scene_name, False


def apply_material_overrides(scene, overrides: list[str]) -> None:
    for raw in overrides:
        parts = raw.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"Expected material override as object:itu_type[:thickness], got {raw!r}")

        obj_name, itu_type = parts[0], parts[1]
        thickness = float(parts[2]) if len(parts) == 3 else 0.1
        if obj_name not in scene.objects:
            raise KeyError(f"Scene object {obj_name!r} not found. Available: {list(scene.objects.keys())}")

        mat_name = f"{obj_name}_{itu_type}_{thickness:g}"
        scene.objects[obj_name].radio_material = ITURadioMaterial(
            mat_name,
            itu_type=itu_type,
            thickness=thickness,
        )


def run_case(
    scene_path: str,
    tx,
    rx,
    max_depth,
    freq,
    sampling_frequency,
    tx_power_dbm,
    num_subcarriers,
    subcarrier_spacing,
    diffuse,
    specular,
    diffraction,
    edge_diffraction,
    refraction,
    merge_shapes,
    material_overrides,
    out_dir,
    prefix,
    scene_name="",
    is_built_in=False,
    plot_scene_label="",
):
    output_scene_name = scene_name or Path(str(scene_path)).stem
    plot_scene_label = plot_scene_label or output_scene_name
    out_dir = scene_output_dir(out_dir, output_scene_name)

    scene = load_scene(scene_path, merge_shapes=merge_shapes)
    apply_material_overrides(scene, material_overrides)
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
    scene.add(
        Transmitter(
            name="tx",
            position=np.array(tx, dtype=np.float32),
            power_dbm=float(tx_power_dbm),
            orientation=[0, 0, 0],
        )
    )
    scene.add(
        Receiver(
            name="rx",
            position=np.array(rx, dtype=np.float32),
            orientation=[0, 0, 0],
        )
    )

    solver = PathSolver()
    t0 = time.perf_counter()
    paths = solver(
        scene,
        max_depth=max_depth,
        los=True,
        specular_reflection=specular,
        diffuse_reflection=diffuse,
        diffraction=diffraction,
        edge_diffraction=edge_diffraction,
        refraction=refraction,
        synthetic_array=False,
    )
    solve_ms = (time.perf_counter() - t0) * 1e3

    a, tau = paths.cir(
        sampling_frequency=sampling_frequency,
        normalize_delays=False,
        out_type="numpy",
    )
    a0 = np.asarray(a[0, 0, 0, 0, :, 0], dtype=np.complex128)
    tau0 = np.asarray(tau[0, 0, 0, 0, :], dtype=np.float64)
    power = np.abs(a0) ** 2

    n_paths = int(a0.shape[0])
    total_power = float(np.sum(power))
    dmin = float(np.min(tau0)) if n_paths else float("nan")
    dmax = float(np.max(tau0)) if n_paths else float("nan")

    # persist concise result
    npz_path = out_dir / f"{prefix}_result.npz"
    np.savez_compressed(
        npz_path,
        a=a,
        tau=tau,
        tx=np.array(tx, dtype=np.float32),
        rx=np.array(rx, dtype=np.float32),
        scene=str(scene_path),
        is_built_in=is_built_in,
        n_paths=n_paths,
        total_power=total_power,
        solve_ms=solve_ms,
        max_depth=max_depth,
        frequency_hz=freq,
        diffuse_reflection=diffuse,
        specular_reflection=specular,
        diffraction=diffraction,
        edge_diffraction=edge_diffraction,
        refraction=refraction,
    )

    out_png = out_dir / f"{prefix}_path_powers.png"
    save_path_power_plot(out_png, power, plot_scene_label, tau0)

    rf_out = save_rf_outputs(
        paths=paths,
        a=a,
        tau=tau,
        out_dir=out_dir,
        prefix=prefix,
        tx=np.array(tx, dtype=np.float32),
        rx=np.array(rx, dtype=np.float32),
        frequency=freq,
        sampling_frequency=sampling_frequency,
        tx_power_dbm=tx_power_dbm,
        num_subcarriers=num_subcarriers,
        subcarrier_spacing=subcarrier_spacing,
        solve_ms=solve_ms,
    )

    return {
        "n_paths": n_paths,
        "total_power": total_power,
        "pathloss_db": rf_out["pathloss_db"],
        "rx_power_dbm": rf_out["rx_power_dbm"],
        "solve_ms": solve_ms,
        "delay_range": (dmin, dmax),
        "npz": npz_path,
        "png": out_png,
        "rf_npz": rf_out["npz"],
        "pathloss_png": rf_out["pathloss_plot"],
        "csi_png": rf_out["csi_magnitude_plot"],
        "csi_shape": rf_out["csi_shape"],
        "a_shape": a.shape,
        "tau_shape": tau.shape,
        "scene": scene_path,
        "is_built_in": is_built_in,
    }


def main() -> None:
    args = parse_args()
    resolved_scene, default_scene_name, is_built_in = resolve_scene(args.scene, args.scene_xml, args.workdir)
    tx = np.array(args.tx, dtype=np.float32)
    rx = np.array(args.rx, dtype=np.float32)
    out = run_case(
        resolved_scene,
        tx,
        rx,
        max_depth=args.max_depth,
        freq=args.frequency,
        sampling_frequency=args.sampling_frequency,
        tx_power_dbm=args.tx_power_dbm,
        num_subcarriers=args.num_subcarriers,
        subcarrier_spacing=args.subcarrier_spacing,
        diffuse=args.diffuse,
        specular=args.specular,
        diffraction=args.diffraction,
        edge_diffraction=args.edge_diffraction,
        refraction=args.refraction,
        merge_shapes=args.merge_shapes,
        material_overrides=args.material,
        out_dir=args.out_dir,
        prefix=args.out_prefix,
        scene_name=args.scene_name or default_scene_name,
        is_built_in=is_built_in,
        plot_scene_label=default_scene_name,
    )
    print(f"[pass] scene={out['scene']}")
    print(f"[pass] built_in={is_built_in}")
    print(f"[pass] tx={tx.tolist()} rx={rx.tolist()}")
    print(
        f"[pass] max_depth={args.max_depth}, diffuse={args.diffuse}, specular={args.specular}, "
        f"diffraction={args.diffraction}, edge_diffraction={args.edge_diffraction}, "
        f"refraction={args.refraction}, "
        f"solve_ms={out['solve_ms']:.2f}"
    )
    print(
        f"[pass] paths={out['n_paths']} total_power={out['total_power']:.6e} "
        f"delay=({out['delay_range'][0]:.3e}, {out['delay_range'][1]:.3e})"
    )
    print(
        f"[pass] pathloss={out['pathloss_db']:.2f} dB, "
        f"rx_power={out['rx_power_dbm']:.2f} dBm, csi_shape={out['csi_shape']}"
    )
    print(f"[pass] saved: {out['npz']} and {out['png']}")
    print(f"[pass] saved RF: {out['rf_npz']}, {out['pathloss_png']}, {out['csi_png']}")


if __name__ == "__main__":
    main()
