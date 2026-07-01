#!/usr/bin/env python3
"""Compare local etoile outputs with Sionna documentation references."""

from __future__ import annotations

import csv
import json
import math
import re
import struct
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image


OUT_DIR = Path("etoile/comparison with documentation")
MOBILITY_ROOT = Path("etoile/mobility_single_vs_multi_bs")
SCENE_XML = Path(".venv/lib/python3.12/site-packages/sionna/rt/scenes/etoile/etoile.xml")
MESH_DIR = SCENE_XML.parent / "meshes"
OFFICIAL_DOC_URL = "https://nvlabs.github.io/sionna/rt/api/scene.html#sionna.rt.scene.etoile"
OFFICIAL_SOURCE_URL = "https://nvlabs.github.io/sionna/_modules/sionna/rt/scene.html"
OFFICIAL_IMAGE_URL = "https://nvlabs.github.io/sionna/_images/etoile.png"


def read_ply_bounds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    size_map = {
        "float": 4,
        "float32": 4,
        "double": 8,
        "float64": 8,
        "uchar": 1,
        "uint8": 1,
        "char": 1,
        "int8": 1,
        "int": 4,
        "int32": 4,
        "uint": 4,
        "uint32": 4,
    }
    with path.open("rb") as f:
        n_vertices = 0
        props = []
        in_vertex = False
        while True:
            line = f.readline().decode("ascii", "ignore").strip()
            if line.startswith("element "):
                parts = line.split()
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    n_vertices = int(parts[2])
            elif in_vertex and line.startswith("property "):
                props.append(line.split()[1:])
            if line == "end_header":
                break
        stride = sum(size_map.get(p[0], 4) for p in props if p[0] != "list")
        mn = np.array([math.inf, math.inf, math.inf], dtype=np.float64)
        mx = np.array([-math.inf, -math.inf, -math.inf], dtype=np.float64)
        for _ in range(n_vertices):
            data = f.read(stride)
            xyz = np.array(struct.unpack_from("<fff", data, 0), dtype=np.float64)
            mn = np.minimum(mn, xyz)
            mx = np.maximum(mx, xyz)
    return mn, mx


def mesh_bounds() -> dict[str, dict[str, object]]:
    bounds: dict[str, dict[str, object]] = {}
    for path in sorted(MESH_DIR.glob("*.ply")):
        if path.name == "Plane.ply":
            continue
        name = re.sub(r"-itu_.*$", "", path.stem)
        mn, mx = read_ply_bounds(path)
        if name in bounds:
            old_mn = np.array(bounds[name]["min"], dtype=np.float64)
            old_mx = np.array(bounds[name]["max"], dtype=np.float64)
            mn = np.minimum(old_mn, mn)
            mx = np.maximum(old_mx, mx)
            bounds[name]["mesh_parts"] = int(bounds[name]["mesh_parts"]) + 1
        else:
            bounds[name] = {"mesh_parts": 1}
        bounds[name].update(
            {
                "min": mn.tolist(),
                "max": mx.tolist(),
                "center": ((mn + mx) / 2.0).tolist(),
                "size": (mx - mn).tolist(),
            }
        )
    return bounds


def xml_metadata() -> dict[str, object]:
    tree = ET.parse(SCENE_XML)
    root = tree.getroot()
    shapes = root.findall(".//shape")
    bsdfs = root.findall(".//bsdf")
    files = []
    for shape in shapes:
        for string in shape.findall(".//string"):
            if string.attrib.get("name") == "filename":
                files.append(string.attrib.get("value", ""))
    material_refs = [ref.attrib.get("id", "") for ref in root.findall(".//ref") if ref.attrib.get("name") == "bsdf"]
    return {
        "scene_xml": str(SCENE_XML),
        "xml_exists": SCENE_XML.exists(),
        "xml_shape_count": len(shapes),
        "xml_bsdf_count": len(bsdfs),
        "xml_referenced_mesh_count": len(files),
        "xml_unique_referenced_mesh_count": len(set(files)),
        "xml_referenced_materials": sorted(set(material_refs)),
    }


def save_official_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(OFFICIAL_IMAGE_URL, path)


def save_local_scene_map(path: Path, bounds: dict[str, dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    mins = []
    maxs = []
    for name, b in bounds.items():
        mn = np.array(b["min"], dtype=float)
        mx = np.array(b["max"], dtype=float)
        w, h = mx[0] - mn[0], mx[1] - mn[1]
        if w <= 0.0 or h <= 0.0:
            continue
        mins.append(mn[:2])
        maxs.append(mx[:2])
        if name == "Arc_de_Triomphe":
            face, edge, alpha, lw = "#c8b081", "#6b5530", 0.9, 0.8
        else:
            face, edge, alpha, lw = "#d8dde3", "#aeb7c2", 0.56, 0.25
        ax.add_patch(Rectangle((mn[0], mn[1]), w, h, facecolor=face, edgecolor=edge, linewidth=lw, alpha=alpha))
    if mins and maxs:
        mn_all = np.min(np.vstack(mins), axis=0)
        mx_all = np.max(np.vstack(maxs), axis=0)
        pad = 0.04 * max(float(mx_all[0] - mn_all[0]), float(mx_all[1] - mn_all[1]))
        ax.set_xlim(float(mn_all[0] - pad), float(mx_all[0] + pad))
        ax.set_ylim(float(mn_all[1] - pad), float(mx_all[1] + pad))
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Local installed etoile scene footprint")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(alpha=0.16)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_side_by_side(out_path: Path, left_path: Path, right_path: Path, title: str, left_title: str, right_title: str) -> None:
    left = plt.imread(left_path)
    right = plt.imread(right_path)
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.2))
    axes[0].imshow(left)
    axes[0].set_title(left_title)
    axes[1].imshow(right)
    axes[1].set_title(right_title)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def save_na_panel(path: Path, title: str, note: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.axis("off")
    ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=18, fontweight="bold")
    ax.text(0.5, 0.42, note, ha="center", va="center", fontsize=11, wrap=True)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def collect_local_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for summary_path in sorted(MOBILITY_ROOT.glob("*GHz/*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        data_path = Path(summary["outputs"]["data"])
        csi_shape = None
        csi_abs_min_db = None
        csi_abs_max_db = None
        delay_doppler_shape = None
        if data_path.exists():
            with np.load(data_path) as data:
                csi = np.asarray(data["csi"])
                csi_shape = list(csi.shape)
                csi_db = 20.0 * np.log10(np.maximum(np.abs(csi), 1e-30))
                csi_abs_min_db = float(np.min(csi_db))
                csi_abs_max_db = float(np.max(csi_db))
                delay_doppler_shape = list(np.asarray(data["delay_doppler"]).shape)
        link = summary["link"]
        delays = np.asarray(link["delays_ns"], dtype=np.float64)
        doppler = np.asarray(link["doppler_hz"], dtype=np.float64)
        rows.append(
            {
                "band": summary["band"],
                "condition": summary["condition"],
                "tx": link["tx"],
                "rx": link["rx"],
                "frequency_hz": summary["frequency_hz"],
                "n_paths_local": link["n_paths"],
                "pathloss_db_local": link["pathloss_db"],
                "total_power_linear_local": link["total_power_linear"],
                "delay_ns_min_local": float(np.min(delays)) if delays.size else None,
                "delay_ns_max_local": float(np.max(delays)) if delays.size else None,
                "doppler_hz_min_local": float(np.min(doppler)) if doppler.size else None,
                "doppler_hz_max_local": float(np.max(doppler)) if doppler.size else None,
                "csi_shape_local": csi_shape,
                "csi_mag_db_min_local": csi_abs_min_db,
                "csi_mag_db_max_local": csi_abs_max_db,
                "delay_doppler_shape_local": delay_doppler_shape,
                "official_n_paths": None,
                "official_pathloss_db": None,
                "official_csi_shape": None,
                "official_status": "not published for etoile in Sionna public scene docs",
                "summary_path": str(summary_path),
                "data_path": str(data_path),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "band",
        "condition",
        "tx",
        "rx",
        "frequency_hz",
        "n_paths_local",
        "official_n_paths",
        "pathloss_db_local",
        "official_pathloss_db",
        "total_power_linear_local",
        "delay_ns_min_local",
        "delay_ns_max_local",
        "doppler_hz_min_local",
        "doppler_hz_max_local",
        "csi_shape_local",
        "official_csi_shape",
        "csi_mag_db_min_local",
        "csi_mag_db_max_local",
        "delay_doppler_shape_local",
        "official_status",
        "summary_path",
        "data_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def fmt(value: object, ndigits: int = 3) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{ndigits}f}"
    return str(value)


def write_markdown(path: Path, metadata: dict[str, object], rows: list[dict[str, object]], outputs: dict[str, str]) -> None:
    lines = [
        "# Etoile: Comparison with Sionna Documentation",
        "",
        "## Documentation Reference",
        "",
        f"- Official scene page: {OFFICIAL_DOC_URL}",
        f"- Official source page: {OFFICIAL_SOURCE_URL}",
        f"- Official image: {OFFICIAL_IMAGE_URL}",
        "- Documented loading pattern: `scene = load_scene(sionna.rt.scene.etoile); scene.preview()`",
        "- Documented description: area around the Arc de Triomphe in Paris, created from OpenStreetMap data with Blender, Blender-OSM, and Mitsuba Blender; ODbL licensing is stated in the Sionna scene docs.",
        "",
        "Important limitation: the public Sionna scene docs provide the built-in scene description and preview image, but do not publish an etoile path-count, pathloss, CIR, CFR/CSI, or delay-Doppler numerical benchmark. Numerical comparisons below therefore mark official numerical values as `N/A` rather than inventing a reference.",
        "",
        "## Scene/Image Comparison",
        "",
        f"![Official docs vs local footprint]({Path(outputs['scene_side_by_side']).name})",
        "",
        "## Installed Scene Metadata",
        "",
        "| Item | Official docs | Local installed scene |",
        "|---|---:|---:|",
        f"| Scene constant | `sionna.rt.scene.etoile` | `{metadata['scene_xml']}` |",
        "| Description | Arc de Triomphe / Paris area | Same installed etoile XML |",
        f"| XML shape count | N/A | {metadata['xml_shape_count']} |",
        f"| XML BSDF count | N/A | {metadata['xml_bsdf_count']} |",
        f"| Referenced mesh count | N/A | {metadata['xml_referenced_mesh_count']} |",
        f"| Unique referenced mesh count | N/A | {metadata['xml_unique_referenced_mesh_count']} |",
        f"| Consolidated object footprints | N/A | {metadata['consolidated_object_count']} |",
        "",
        "## Path/CSI Side-by-Side Summary",
        "",
        "| Band | Condition | Local paths | Official paths | Local pathloss (dB) | Official pathloss | Local CSI shape | Official CSI |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["band"]),
                    str(row["condition"]),
                    fmt(row["n_paths_local"], 0),
                    "N/A",
                    fmt(row["pathloss_db_local"], 3),
                    "N/A",
                    f"`{row['csi_shape_local']}`",
                    "N/A",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Representative Local Artifacts",
            "",
            "The figure below compares our local path/CSI artifacts for one representative link with explicit `N/A` official panels. The official docs do not provide equivalent etoile path or CSI plots.",
            "",
            f"![Path/CSI comparison note]({Path(outputs['path_csi_side_by_side']).name})",
            "",
            "Full machine-readable local-vs-official table:",
            "",
            f"- `{Path(outputs['numerical_csv']).name}`",
            f"- `{Path(outputs['summary_json']).name}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    official_image = OUT_DIR / "official_etoile.png"
    local_scene_map = OUT_DIR / "local_etoile_scene_map.png"
    scene_side_by_side = OUT_DIR / "scene_image_side_by_side.png"
    official_path_na = OUT_DIR / "official_path_csi_not_published.png"
    path_csi_side_by_side = OUT_DIR / "path_csi_side_by_side.png"
    numerical_csv = OUT_DIR / "numerical_side_by_side.csv"
    numerical_md = OUT_DIR / "numerical_side_by_side.md"
    summary_json = OUT_DIR / "documentation_comparison_summary.json"

    save_official_image(official_image)
    bounds = mesh_bounds()
    save_local_scene_map(local_scene_map, bounds)
    save_side_by_side(
        scene_side_by_side,
        official_image,
        local_scene_map,
        "Etoile scene image: Sionna docs vs local installed footprint",
        "Sionna documentation",
        "Local installed scene footprint",
    )

    save_na_panel(
        official_path_na,
        "No official etoile path/CSI benchmark",
        "Sionna's public etoile scene docs publish the scene description and image, not pathloss/CIR/CSI reference numbers.",
    )
    representative_paths = MOBILITY_ROOT / "5GHz/one_bs__bs_arc_to_rx_mobile_west/paths.png"
    representative_csi = MOBILITY_ROOT / "5GHz/one_bs__bs_arc_to_rx_mobile_west/csi.png"
    if representative_paths.exists() and representative_csi.exists():
        fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.2))
        axes[0, 0].imshow(plt.imread(representative_paths))
        axes[0, 0].set_title("Local paths")
        axes[0, 1].imshow(plt.imread(official_path_na))
        axes[0, 1].set_title("Official paths")
        axes[1, 0].imshow(plt.imread(representative_csi))
        axes[1, 0].set_title("Local CSI magnitude")
        axes[1, 1].imshow(plt.imread(official_path_na))
        axes[1, 1].set_title("Official CSI")
        for ax in axes.reshape(-1):
            ax.axis("off")
        fig.suptitle("Representative etoile path/CSI: local vs official documentation")
        fig.tight_layout()
        fig.savefig(path_csi_side_by_side, dpi=170)
        plt.close(fig)

    rows = collect_local_rows()
    write_csv(numerical_csv, rows)

    metadata = xml_metadata()
    metadata["consolidated_object_count"] = len(bounds)
    metadata["mesh_part_file_count"] = len(list(MESH_DIR.glob("*.ply")))
    outputs = {
        "official_image": str(official_image),
        "local_scene_map": str(local_scene_map),
        "scene_side_by_side": str(scene_side_by_side),
        "path_csi_side_by_side": str(path_csi_side_by_side),
        "numerical_csv": str(numerical_csv),
        "numerical_md": str(numerical_md),
        "summary_json": str(summary_json),
    }
    summary = {
        "scene": "etoile",
        "official_docs": {
            "scene_page": OFFICIAL_DOC_URL,
            "source_page": OFFICIAL_SOURCE_URL,
            "image": OFFICIAL_IMAGE_URL,
            "published_scene_image": True,
            "published_numerical_path_or_csi_reference": False,
            "note": "The Sionna public scene docs expose etoile as a built-in scene with a preview image and source description, but no pathloss/CIR/CSI benchmark table.",
        },
        "local_metadata": metadata,
        "local_link_rows": rows,
        "outputs": outputs,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(numerical_md, metadata, rows, outputs)
    print(json.dumps({"out_dir": str(OUT_DIR), "rows": len(rows), "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
