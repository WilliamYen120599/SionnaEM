#!/usr/bin/env python3
"""Quick RT solve for one scene.

Examples:
  source .venv/bin/activate
  python SionnaEM/tools/run_xml_rt_quick.py --scene test_scenes/meshes.zip
  python SionnaEM/tools/run_xml_rt_quick.py --scene test_scenes/meshes.zip --scene-xml meshes/test.xml
"""

from __future__ import annotations

import argparse
import os
import shutil
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET
from typing import Tuple
import time


_RT_SYMBOLS = None  # Imported lazily after setting Mitsuba variant.


def parse_list_float(raw: str, n: int) -> Tuple[float, ...]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) != n:
        raise argparse.ArgumentTypeError(f"Expected {n} values, got {len(vals)}")
    return tuple(vals)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a quick Sionna-RT solve and save plots")
    p.add_argument("--scene", required=True, help="XML file path or zip package")
    p.add_argument(
        "--scene-xml",
        default="",
        help="For zip input, which xml to use (defaults to first xml in archive)",
    )
    p.add_argument("--workdir", default="test_scenes", help="Directory for extracted zip")
    p.add_argument("--out-dir", default="Blender Preview Images", help="Directory for output plots")
    p.add_argument("--out-prefix", default="quick_rt", help="Filename prefix for output plots")
    p.add_argument("--scene-name", default="", help="Output folder name under --out-dir")
    p.add_argument("--tx", type=lambda s: parse_list_float(s, 3), default=(0.0, 0.0, 2.0), help="Tx position x,y,z")
    p.add_argument("--rx", type=lambda s: parse_list_float(s, 3), default=(20.0, 0.0, 1.5), help="Rx position x,y,z")
    p.add_argument("--frequency", type=float, default=28e9, help="Carrier frequency in Hz")
    p.add_argument("--sampling-frequency", type=float, default=122.88e6, help="CIR sampling frequency")
    p.add_argument("--tx-power-dbm", type=float, default=0.0, help="Transmit power in dBm")
    p.add_argument("--num-subcarriers", type=int, default=64, help="Number of CSI/CFR subcarriers to save")
    p.add_argument("--subcarrier-spacing", type=float, default=30e3, help="CSI/CFR subcarrier spacing in Hz")
    p.add_argument("--max-depth", type=int, default=2, help="Path solver max depth")
    p.add_argument("--diffuse-reflection", action=argparse.BooleanOptionalAction, default=True, help="Enable diffuse reflection")
    p.add_argument("--specular-reflection", action=argparse.BooleanOptionalAction, default=True, help="Enable specular reflection")
    p.add_argument(
        "--mitsuba-variant",
        default="llvm_ad_rgb",
        help=(
            "Mitsuba variant. Use 'llvm_ad_rgb' (CPU) on headless servers; "
            "try 'cuda_ad_rgb' only if OptiX/CUDA is available."
        ),
    )
    p.add_argument(
        "--fallback-variant",
        default="scalar_rgb",
        help="Fallback variant used if requested variant fails during scene load."
    )
    return p.parse_args()


def _configure_mitsuba_variant(requested: str, fallback: str):
    """Configure Mitsuba variant with a minimal fallback strategy."""
    import mitsuba as mi

    variants = []
    if requested:
        variants.append(requested)
    if fallback and fallback != requested:
        variants.append(fallback)

    last_err = None
    for var in variants:
        try:
            mi.set_variant(var)
            print(f"[info] Mitsuba variant set to: {var}")
            return var
        except Exception as err:  # pragma: no cover - backend-specific
            last_err = err
            print(f"[warn] Failed to set Mitsuba variant '{var}': {err}")
    raise RuntimeError(f"Unable to configure Mitsuba variant. Last error: {last_err}")


def _prepare_scene_path(scene_input: Path, xml_hint: str, work_root: Path) -> Path:
    if scene_input.suffix.lower() != ".zip":
        if not scene_input.is_file():
            raise FileNotFoundError(f"Scene file not found: {scene_input}")
        return scene_input

    if not zipfile.is_zipfile(scene_input):
        raise ValueError(f"Invalid zip file: {scene_input}")

    extract_dir = work_root / scene_input.stem
    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(scene_input) as zf:
        zf.extractall(extract_dir)

    xml_candidates = sorted(extract_dir.rglob("*.xml"))
    if not xml_candidates:
        raise FileNotFoundError("No xml file found in zip package")

    if xml_hint:
        selected = None
        for c in xml_candidates:
            if c.name == xml_hint or c.as_posix().endswith(xml_hint):
                selected = c
                break
        if selected is None:
            raise FileNotFoundError(f"Could not find xml '{xml_hint}' in archive")
    else:
        # prefer top-level candidate if present
        selected = None
        for c in xml_candidates:
            if c.parent == extract_dir:
                selected = c
                break
        if selected is None:
            selected = xml_candidates[0]

    # If XML is inside a folder like "meshes/test.xml" but references "meshes/..." files,
    # move a copy to the package root so those relative paths resolve.
    if selected.parent != extract_dir:
        normalized = extract_dir / "normalized_scene.xml"
        normalized.write_text(selected.read_text())
        return normalized

    return selected


def _ensure_radio_materials(xml_path: Path) -> tuple[Path, int]:
    """Normalize non-ITU visual BSDF IDs to Sionna-compatible radio materials.

    This keeps mesh/shape references untouched by ID remap while making materials
    discoverable by Sionna's `process_xml` helper.
    """
    text = xml_path.read_text(encoding="utf-8")
    root = ET.fromstring(text)

    material_map = {}
    used_ids = set()

    for bsdf in root.findall("bsdf"):
        mat_id = bsdf.attrib.get("id")
        if mat_id is None:
            continue

        name = mat_id[4:] if mat_id.startswith("mat-") else mat_id
        is_radio_naming = name.startswith("itu_") or name.startswith("itu-")
        if is_radio_naming:
            used_ids.add(mat_id)
            continue

        base = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name)
        new_id = f"mat-itu_{base}"
        while new_id in used_ids:
            new_id = f"{new_id}_x"

        if new_id == mat_id:
            continue

        material_map[mat_id] = new_id
        used_ids.add(new_id)

        # Keep optional color to retain rough visual appearance in Blender previews.
        color_value = None
        for pname in ("color", "reflectance", "base_color"):
            color_node = bsdf.find(f".//rgb[@name='{pname}']")
            if color_node is not None:
                color_value = color_node.attrib.get("value")
                break

        bsdf.clear()
        bsdf.attrib["type"] = "itu-radio-material"
        bsdf.attrib["id"] = new_id
        bsdf.attrib["name"] = new_id

        if color_value is not None:
            ET.SubElement(bsdf, "rgb", {"name": "color", "value": color_value})
        ET.SubElement(bsdf, "string", {"name": "type", "value": "concrete"})
        ET.SubElement(bsdf, "float", {"name": "thickness", "value": "0.1"})

    if not material_map:
        return xml_path, 0

    for ref in root.findall(".//ref[@name='bsdf']"):
        ref_id = ref.attrib.get("id")
        if ref_id in material_map:
            ref.attrib["id"] = material_map[ref_id]

    out = xml_path.with_name(f"{xml_path.stem}_sionna_materials.xml")
    out.write_text(ET.tostring(root, encoding="utf-8").decode("utf-8"), encoding="utf-8")
    return out, len(material_map)


def _flatten_mesh_groups(xml_path: Path) -> tuple[Path, int]:
    """Flatten simple `shapegroup` + `instance` constructs to top-level `ply` shapes."""
    tree_root = ET.parse(xml_path).getroot()

    shapegroups = [s for s in list(tree_root) if s.tag == "shape" and s.attrib.get("type") == "shapegroup"]
    if not shapegroups:
        return xml_path, 0

    existing = {s.attrib.get("id") for s in list(tree_root) if s.tag == "shape" and s.attrib.get("type") == "ply" and "id" in s.attrib}
    flattened = 0
    removed_groups = set()

    for group in shapegroups:
        group_id = group.attrib.get("id")
        for child in list(group):
            if child.tag != "shape":
                continue
            if child.attrib.get("type") != "ply":
                continue
            new_shape = ET.fromstring(ET.tostring(child, encoding="unicode"))
            child_id = new_shape.attrib.get("id")
            if child_id is None:
                base_id = group_id if group_id else "mesh"
                child_id = base_id
                while child_id in existing:
                    child_id = f"{child_id}_x"
                new_shape.attrib["id"] = child_id
            elif child_id in existing:
                base_id = child_id
                while base_id in existing:
                    base_id = f"{base_id}_x"
                new_shape.attrib["id"] = base_id
                child_id = base_id
            existing.add(child_id)

            flattened += 1
            tree_root.append(new_shape)

        removed_groups.add(group_id)
        tree_root.remove(group)

    # Remove instances that only point to removed groups.
    for s in list(tree_root):
        if s.tag != "shape" or s.attrib.get("type") != "instance":
            continue
        ref = s.find("ref")
        if ref is not None and ref.attrib.get("id") in removed_groups:
            tree_root.remove(s)

    out = xml_path.with_name(f"{xml_path.stem}_flat_shapes.xml")
    out.write_text(ET.tostring(tree_root, encoding="unicode"), encoding="utf-8")
    return out, flattened


def _extract_path_values(a: np.ndarray, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a_abs = np.abs(a)
    if a_abs.ndim < 6:
        raise ValueError(f"Unexpected a shape {a_abs.shape}")

    # Use first TX/RX chain; this matches Sionna quick tests and built-in examples.
    # a shape: [n_rx, n_time, n_tx, n_tx_ant, n_paths, n_rx_ant]
    amp = a_abs[0, 0, 0, 0, :, 0]
    if amp.size == 0:
        raise ValueError("Solver returned zero paths")
    power = amp ** 2

    if tau.ndim >= 5:
        delays = tau[0, 0, 0, :, 0]
    elif tau.ndim == 2:
        delays = tau[0, :]
    elif tau.ndim == 1:
        delays = tau
    else:
        delays = tau.reshape(-1)[: amp.size]

    n = min(power.size, delays.size)
    return power[:n], delays[:n]


def _save_plots(power: np.ndarray, delay: np.ndarray, out_prefix: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = []
    idx = np.arange(1, len(power) + 1)

    fig = plt.figure(figsize=(9, 4))
    plt.plot(idx, 10 * np.log10(np.maximum(power, 1e-15)), marker="o")
    plt.xlabel("Path index")
    plt.ylabel("Path power (dB)")
    plt.title("SionnaRT path powers")
    plt.grid(alpha=0.3)
    p1 = out_prefix.with_name(f"{out_prefix.name}_path_power.png")
    fig.tight_layout()
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    out.append(p1)

    fig = plt.figure(figsize=(9, 4))
    order = np.argsort(delay)
    plt.plot(np.asarray(delay)[order] * 1e9, 10 * np.log10(np.maximum(np.asarray(power)[order], 1e-15)), marker="o")
    plt.xlabel("Delay (ns)")
    plt.ylabel("Path power (dB)")
    plt.title("SionnaRT power vs path delay")
    plt.grid(alpha=0.3)
    p2 = out_prefix.with_name(f"{out_prefix.name}_path_delay.png")
    fig.tight_layout()
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    out.append(p2)

    return out


def _save_no_paths_plot(out_prefix: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = []
    fig = plt.figure(figsize=(6, 3))
    plt.text(0.5, 0.5, "No paths found for this configuration", ha="center", va="center")
    plt.axis("off")
    p = out_prefix.with_name(f"{out_prefix.name}_no_paths.png")
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)
    return out


def _extract_or_zero(a: np.ndarray, tau: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        return _extract_path_values(a, tau)
    except ValueError:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)


def _print_summary(
    scene_path: Path,
    n_paths: int,
    t_solve: float,
    t_cir: float,
    out_pngs: list[Path],
    rf_out: dict[str, object] | None = None,
) -> None:
    print("\n[done]")
    print(f"  scene: {scene_path}")
    print(f"  paths: {n_paths}")
    print(f"  solve_time_s: {t_solve:.3f}")
    print(f"  cir_time_s: {t_cir:.3f}")
    if rf_out is not None:
        print(f"  pathloss_db: {rf_out['pathloss_db']:.2f}")
        print(f"  rx_power_dbm: {rf_out['rx_power_dbm']:.2f}")
        print(f"  csi_shape: {rf_out['csi_shape']}")
    print("  plots:")
    for p in out_pngs:
        print(f"    {p}")
    if rf_out is not None:
        print("  rf_outputs:")
        print(f"    {rf_out['npz']}")
        print(f"    {rf_out['pathloss_plot']}")
        print(f"    {rf_out['csi_magnitude_plot']}")


def main() -> None:
    global np

    args = parse_args()
    os.environ.setdefault("DRJIT_LIBOPTIX_PATH", "")

    # Import sionna.rt after variant selection to avoid OptiX-only default selection.
    _configure_mitsuba_variant(args.mitsuba_variant, args.fallback_variant)

    import numpy as np
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene
    from rt_output_utils import save_rf_outputs, scene_output_dir

    repo_root = Path(__file__).resolve().parents[2]

    work_root = (repo_root / args.workdir).resolve()
    work_root.mkdir(parents=True, exist_ok=True)

    out_dir_base = (repo_root / args.out_dir).resolve()

    scene_arg = Path(args.scene)
    scene_xml = _prepare_scene_path(scene_arg, args.scene_xml, work_root)
    scene_xml, n_converted = _ensure_radio_materials(scene_xml)
    scene_xml, n_flat = _flatten_mesh_groups(scene_xml)
    output_scene_name = args.scene_name or (scene_arg.stem if scene_arg.suffix.lower() == ".zip" else scene_xml.stem)
    out_dir = scene_output_dir(out_dir_base, output_scene_name)

    if n_converted:
        print(f"[info] Auto-converted {n_converted} materials to Sionna-compatible radio materials.")
    if n_flat:
        print(f"[info] Flattened {n_flat} grouped mesh entries into top-level ply shapes.")

    print(f"Loading scene from: {scene_xml}")
    scene = load_scene(str(scene_xml), remove_duplicate_vertices=False)

    scene.frequency = float(args.frequency)
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
            position=np.array(args.tx, dtype=np.float32),
            orientation=[0, 0, 0],
            power_dbm=float(args.tx_power_dbm),
        )
    )
    scene.add(
        Receiver(
            name="rx",
            position=np.array(args.rx, dtype=np.float32),
            orientation=[0, 0, 0],
        )
    )

    solver = PathSolver()
    t0 = time.perf_counter()
    solver_kwargs = dict(
        max_depth=args.max_depth,
        los=True,
        specular_reflection=args.specular_reflection,
        diffuse_reflection=args.diffuse_reflection,
        diffraction=False,
        edge_diffraction=False,
        refraction=False,
        synthetic_array=False,
    )
    t_solve = 0.0
    paths = None
    try:
        paths = solver(scene, **solver_kwargs)
        t_solve = time.perf_counter() - t0
    except Exception as err:
        # Quick fallback: try LOS-only first, which avoids problematic
        # material-dependent reflection branches in exported scenes.
        print(f"[warn] solver failed with requested settings: {err}")
        if args.diffuse_reflection or args.specular_reflection:
            print("[info] retrying with LOS-only.")
            t0 = time.perf_counter()
            try:
                paths = solver(
                    scene,
                    max_depth=0,
                    los=True,
                    specular_reflection=False,
                    diffuse_reflection=False,
                    diffraction=False,
                    edge_diffraction=False,
                    refraction=False,
                    synthetic_array=False,
                )
                t_solve = time.perf_counter() - t0
                solver_kwargs["max_depth"] = 0
                solver_kwargs["diffuse_reflection"] = False
                solver_kwargs["specular_reflection"] = False
                args.diffuse_reflection = False
                args.specular_reflection = False
            except Exception as los_err:
                print(f"[warn] LOS-only fallback also failed: {los_err}")

    if paths is None:
        raise RuntimeError("Ray tracing failed. See previous warnings for details.")

    t1 = time.perf_counter()
    a, tau = paths.cir(
        sampling_frequency=args.sampling_frequency,
        normalize_delays=False,
        out_type="numpy",
    )
    t_cir = time.perf_counter() - t1

    power, delay = _extract_or_zero(a, tau)
    out_prefix = out_dir / args.out_prefix
    if power.size == 0:
        out_pngs = _save_no_paths_plot(out_prefix)
    else:
        out_pngs = _save_plots(power, delay, out_prefix)

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
        solve_ms=t_solve * 1e3,
    )

    _print_summary(scene_xml, int(power.shape[0]), t_solve, t_cir, out_pngs, rf_out)


if __name__ == "__main__":
    main()
