#!/usr/bin/env python3
"""Render all point-drone trajectory frames with Sionna's 3D renderer."""

from __future__ import annotations

import json
import shutil
import argparse
from pathlib import Path

import mitsuba as mi

mi.variants()
mi.set_variant("llvm_ad_mono_polarized")

import numpy as np
import sionna
from PIL import Image, ImageDraw
from sionna.rt import Camera, PlanarArray, Receiver, Transmitter, load_scene


ROOT = Path("etoile/point_drone_tx")
DEFAULT_CONDITION = "point_drone_tx_to_rx_rooftop_e089__cam_x58p9_y32p0_z44p0"
TWO_MOBILE_RX_CONDITION = "point_drone_tx_to_two_mobile_rx__cam_x58p9_y32p0_z44p0"
BANDS_HZ = {
    "5GHz": 5.0e9,
    "28GHz": 28.0e9,
}

TX_NAME = "drone_tx_point"
RX_NAME = "rx_rooftop_e089"
CAMERA_MARKER_NAME = "fixed_camera_marker_visual_only"
RESOLUTION = (720, 480)
BIRDSEYE_RESOLUTION = (960, 720)
NUM_SAMPLES = 8
FOV_DEG = 70.0
BIRDSEYE_FOV_DEG = 72.0


def array() -> PlanarArray:
    return PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )


def as_list(arr: np.ndarray) -> list[float]:
    return [float(x) for x in np.asarray(arr, dtype=float).reshape(-1)]


def fmt_coord(value: float) -> str:
    return f"{value:.1f}".replace("-", "m").replace(".", "p")


def coord_slug(prefix: str, pos: np.ndarray) -> str:
    return f"{prefix}_x{fmt_coord(pos[0])}_y{fmt_coord(pos[1])}_z{fmt_coord(pos[2])}"


def load_pose_data(source_run: Path) -> dict[str, np.ndarray]:
    with np.load(source_run / "rt_data.npz") as data:
        tx_positions = data["tx_positions_m"].astype(np.float64)
        n_frames = tx_positions.shape[0]
        if "rx_positions_m" in data:
            rx_positions = data["rx_positions_m"].astype(np.float64)
            rx_velocities = data["rx_velocity_mps"].astype(np.float64)
            rx_names = data["rx_names"].astype(str).tolist()
        else:
            rx_positions = np.repeat(data["rx_position_m"].astype(np.float64)[None, None, :], n_frames, axis=1)
            rx_velocities = np.zeros_like(rx_positions)
            rx_names = [RX_NAME]
        return {
            "tx_positions_m": tx_positions,
            "tx_velocity_mps": data["tx_velocity_mps"].astype(np.float64),
            "rx_names": rx_names,
            "rx_positions_m": rx_positions,
            "rx_velocity_mps": rx_velocities,
            "camera_position_m": data["camera_position_m"].astype(np.float64),
            "camera_look_at_m": data["camera_look_at_m"].astype(np.float64),
            "frame_time_s": data["frame_time_s"].astype(np.float64),
        }


def setup_scene(pose: dict[str, np.ndarray], freq_hz: float):
    scene = load_scene(sionna.rt.scene.etoile, merge_shapes=True)
    scene.frequency = freq_hz
    scene.tx_array = array()
    scene.rx_array = array()

    tx = Transmitter(TX_NAME, position=as_list(pose["tx_positions_m"][0]), power_dbm=0.0)
    camera_marker = Receiver(CAMERA_MARKER_NAME, position=as_list(pose["camera_position_m"]))
    scene.add(tx)
    for rx_i, rx_name in enumerate(pose["rx_names"]):
        scene.add(Receiver(rx_name, position=as_list(pose["rx_positions_m"][rx_i, 0])))
        scene.get(rx_name).velocity = as_list(pose["rx_velocity_mps"][rx_i, 0])
    scene.add(camera_marker)
    scene.get(TX_NAME).velocity = as_list(pose["tx_velocity_mps"][0])
    scene.get(CAMERA_MARKER_NAME).velocity = [0.0, 0.0, 0.0]
    scene.get(TX_NAME).look_at(scene.get(pose["rx_names"][0]))
    return scene


def birdseye_camera(pose: dict[str, np.ndarray]) -> tuple[Camera, list[float], list[float]]:
    rx_points = pose["rx_positions_m"].reshape(-1, 3)
    points = np.vstack([pose["tx_positions_m"], rx_points, pose["camera_position_m"]])
    xy_min = np.min(points[:, :2], axis=0)
    xy_max = np.max(points[:, :2], axis=0)
    xy_center = (xy_min + xy_max) / 2.0
    look_at = np.array([xy_center[0], xy_center[1], 48.0], dtype=np.float64)
    position = np.array([xy_center[0] - 85.0, xy_center[1] - 120.0, 245.0], dtype=np.float64)
    return Camera(position=as_list(position), look_at=as_list(look_at)), as_list(position), as_list(look_at)


def render_frame(
    scene,
    camera: Camera,
    pose: dict[str, np.ndarray],
    frame_idx: int,
    out_path: Path,
    *,
    view_name: str,
    resolution: tuple[int, int],
    fov_deg: float,
    condition_label: str,
    birdseye_note: bool = False,
) -> None:
    tx_pos = pose["tx_positions_m"][frame_idx]
    tx_vel = pose["tx_velocity_mps"][frame_idx]
    cam_pos = pose["camera_position_m"]
    t_s = pose["frame_time_s"][frame_idx]

    scene.get(TX_NAME).position = as_list(tx_pos)
    scene.get(TX_NAME).velocity = as_list(tx_vel)
    for rx_i, rx_name in enumerate(pose["rx_names"]):
        scene.get(rx_name).position = as_list(pose["rx_positions_m"][rx_i, frame_idx])
        scene.get(rx_name).velocity = as_list(pose["rx_velocity_mps"][rx_i, frame_idx])
    scene.get(TX_NAME).look_at(scene.get(pose["rx_names"][0]))

    bitmap = scene.render(
        camera=camera,
        num_samples=NUM_SAMPLES,
        resolution=resolution,
        show_devices=True,
        show_orientations=False,
        fov=fov_deg,
        return_bitmap=True,
    )
    arr = np.clip(np.asarray(bitmap)[..., :3], 0.0, 1.0)
    img = Image.fromarray((arr * 255.0).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    lines = [
        f"Sionna 3D etoile {condition_label} {view_name}  frame {frame_idx:03d}  t={t_s:.1f}s",
        (
            f"Tx(point drone)=({tx_pos[0]:.1f},{tx_pos[1]:.1f},{tx_pos[2]:.1f})  "
            f"V=({tx_vel[0]:.1f},{tx_vel[1]:.1f},{tx_vel[2]:.1f}) m/s"
        ),
    ]
    for rx_i, rx_name in enumerate(pose["rx_names"]):
        rx_pos = pose["rx_positions_m"][rx_i, frame_idx]
        lines.append(f"{rx_name}=({rx_pos[0]:.1f},{rx_pos[1]:.1f},{rx_pos[2]:.1f})")
    lines.append(f"Cam fixed=({cam_pos[0]:.1f},{cam_pos[1]:.1f},{cam_pos[2]:.1f})")
    label = "\n".join(lines)
    if birdseye_note:
        label += "\nBird's-eye includes visual-only camera marker; RF data unchanged."
    box_height = 20 + 16 * (label.count("\n") + 1)
    if birdseye_note:
        box = (8, 8, min(resolution[0] - 8, 910), min(resolution[1] - 8, box_height))
    else:
        box = (8, 8, min(resolution[0] - 8, 700), min(resolution[1] - 8, box_height))
    draw.rectangle(box, fill=(255, 255, 255), outline=(185, 28, 28))
    draw.text((14, 14), label, fill=(0, 0, 0))
    img.save(out_path)


def write_gif(frame_paths: list[Path], gif_path: Path, duration_ms: int = 100) -> None:
    images = [Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in frame_paths]
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=duration_ms, loop=0)
    for image in images:
        image.close()


def make_rx_slug(pose: dict[str, np.ndarray]) -> str:
    if len(pose["rx_names"]) == 1:
        return coord_slug("rx", pose["rx_positions_m"][0, 0])
    suffix = "_".join(name.replace("rx_", "") for name in pose["rx_names"])
    return f"rx{len(pose['rx_names'])}_{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--band", choices=sorted(BANDS_HZ), default="5GHz")
    parser.add_argument(
        "--condition",
        default=DEFAULT_CONDITION,
        choices=[DEFAULT_CONDITION, TWO_MOBILE_RX_CONDITION],
    )
    args = parser.parse_args()

    freq_hz = BANDS_HZ[args.band]
    source_run = ROOT / args.band / args.condition
    out_dir = source_run / "sionna_3d_renderer"
    frame_dir = out_dir / "frames"
    birdseye_frame_dir = out_dir / "birds_eye_frames"

    pose = load_pose_data(source_run)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    frame_dir.mkdir(parents=True)
    birdseye_frame_dir.mkdir(parents=True)

    scene = setup_scene(pose, freq_hz)
    camera = Camera(position=as_list(pose["camera_position_m"]), look_at=as_list(pose["camera_look_at_m"]))
    bird_camera, bird_camera_position, bird_camera_look_at = birdseye_camera(pose)

    cam_slug = coord_slug("cam", pose["camera_position_m"])
    look_slug = coord_slug("look", pose["camera_look_at_m"])
    rx_slug_value = make_rx_slug(pose)
    condition_label = "two-moving-Rx" if args.condition == TWO_MOBILE_RX_CONDITION else "fixed-Rx"

    frame_paths: list[Path] = []
    birdseye_frame_paths: list[Path] = []
    for frame_idx in range(len(pose["frame_time_s"])):
        tx_slug = coord_slug("tx", pose["tx_positions_m"][frame_idx])
        out_path = frame_dir / f"sionna_3d_frame_{frame_idx:03d}__{tx_slug}__{rx_slug_value}__{cam_slug}.png"
        render_frame(
            scene,
            camera,
            pose,
            frame_idx,
            out_path,
            view_name="fixed-camera view",
            resolution=RESOLUTION,
            fov_deg=FOV_DEG,
            condition_label=condition_label,
        )
        frame_paths.append(out_path)
        print(f"rendered {frame_idx + 1:02d}/{len(pose['frame_time_s'])}: {out_path}")

        bird_path = (
            birdseye_frame_dir
            / f"sionna_3d_birds_eye_frame_{frame_idx:03d}__{tx_slug}__{rx_slug_value}__{cam_slug}.png"
        )
        render_frame(
            scene,
            bird_camera,
            pose,
            frame_idx,
            bird_path,
            view_name="bird's-eye view",
            resolution=BIRDSEYE_RESOLUTION,
            fov_deg=BIRDSEYE_FOV_DEG,
            condition_label=condition_label,
            birdseye_note=True,
        )
        birdseye_frame_paths.append(bird_path)
        print(f"rendered bird's-eye {frame_idx + 1:02d}/{len(pose['frame_time_s'])}: {bird_path}")

    gif_path = out_dir / f"sionna_3d_motion__{rx_slug_value}__{cam_slug}__{look_slug}.gif"
    birdseye_gif_path = out_dir / f"sionna_3d_birds_eye_motion__{rx_slug_value}__{cam_slug}__{look_slug}.gif"
    full_gif_path = out_dir / "sionna_3d_drone_motion_full_5s_10fps.gif"
    birdseye_full_gif_path = out_dir / "sionna_3d_birds_eye_motion_full_5s_10fps.gif"
    write_gif(frame_paths, gif_path)
    write_gif(birdseye_frame_paths, birdseye_gif_path)
    write_gif(frame_paths, full_gif_path, duration_ms=100)
    write_gif(birdseye_frame_paths, birdseye_full_gif_path, duration_ms=100)

    metadata = {
        "scene": "etoile",
        "source_run": str(source_run),
        "condition": args.condition,
        "condition_label": condition_label,
        "note": "Visual-only Sionna 3D renderer pass. RF data remain in the per-band rt_data.npz files.",
        "point_drone_note": "The drone is modeled as a moving point Transmitter, not a physical drone mesh.",
        "n_frames": len(frame_paths),
        "frame_dt_s": float(np.diff(pose["frame_time_s"]).mean()),
        "renderer_frequency_hz": freq_hz,
        "resolution": list(RESOLUTION),
        "num_samples": NUM_SAMPLES,
        "fov_deg": FOV_DEG,
        "birds_eye_resolution": list(BIRDSEYE_RESOLUTION),
        "birds_eye_fov_deg": BIRDSEYE_FOV_DEG,
        "birds_eye_camera_position_m": bird_camera_position,
        "birds_eye_camera_look_at_m": bird_camera_look_at,
        "camera_marker_note": (
            "The bird's-eye render adds fixed_camera_marker_visual_only as an extra rendered "
            "Receiver marker at the fixed camera pose. This is for visualization only."
        ),
        "rx_names": pose["rx_names"],
        "first_rx_positions_m": pose["rx_positions_m"][:, 0, :].astype(float).tolist(),
        "last_rx_positions_m": pose["rx_positions_m"][:, -1, :].astype(float).tolist(),
        "camera_position_m": as_list(pose["camera_position_m"]),
        "camera_look_at_m": as_list(pose["camera_look_at_m"]),
        "first_tx_position_m": as_list(pose["tx_positions_m"][0]),
        "last_tx_position_m": as_list(pose["tx_positions_m"][-1]),
        "frames_dir": str(frame_dir),
        "gif": str(gif_path),
        "full_5s_gif": str(full_gif_path),
        "birds_eye_frames_dir": str(birdseye_frame_dir),
        "birds_eye_gif": str(birdseye_gif_path),
        "birds_eye_full_5s_gif": str(birdseye_full_gif_path),
    }
    (out_dir / "render_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
