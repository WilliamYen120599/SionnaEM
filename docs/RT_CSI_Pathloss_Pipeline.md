# Sionna RT CSI/Pathloss Pipeline

This note summarizes the simulation pipeline we had before switching attention to physical scene import. Weather-condition post-processing is intentionally excluded here.

## Scope

The completed work so far used Sionna RT built-in scenes rather than an imported physical/custom scene.

Main built-in scenes used:

- `simple_wedge`
- `floor_wall`
- `simple_reflector`
- `simple_street_canyon_with_cars`
- `etoile` / Paris scene for Micro-Doppler integration checks and point-drone Tx dataset generation

The pipeline outputs are not only pathloss. We save and analyze:

- Raw CIR path coefficients and delays
- Path powers and path components
- CFR/CSI over OFDM-like subcarriers
- Path gain, pathloss, and received power
- Per-link and per-antenna separation
- Pilot recovery checks after CSI generation
- Doppler and delay-Doppler for mobility scenes
- Visualization artifacts for scene/path sanity checks

## Core Static RT Pipeline

The baseline Sionna RT workflow is:

1. Load a Sionna scene.
2. Configure carrier frequency.
3. Configure TX/RX antenna arrays.
4. Place transmitters and receivers.
5. Run `PathSolver`.
6. Extract CIR with `paths.cir(...)`.
7. Extract CFR/CSI with `paths.cfr(...)`.
8. Compute path powers and pathloss from CIR coefficients.
9. Save NPZ data and diagnostic plots.

The common output helper is:

- `SionnaEM/tools/rt_output_utils.py`

Its main function is:

- `save_rf_outputs(...)`

That helper saves:

- `cir_a`: raw CIR path coefficients
- `cir_tau`: raw CIR delays
- `first_link_path_coefficients`: first TX/RX link path coefficients
- `first_link_path_power_linear`: per-path linear powers
- `first_link_path_delay_s`: per-path delays
- `csi`: CFR/CSI tensor over subcarriers
- `csi_frequencies_hz`: subcarrier frequency grid
- `path_gain_linear`: sum of path powers
- `pathloss_db`: `-10*log10(path_gain_linear)`
- `rx_power_dbm`: `tx_power_dbm - pathloss_db`
- TX/RX positions and carrier/sampling metadata

The usual output files are:

- `*_result.npz`: raw RT solve metadata and CIR
- `*_csi_pathloss.npz`: CSI/CFR plus pathloss bundle
- `*_path_powers.png`: per-path power plot
- `*_pathloss.png`: pathloss/Rx-power plot
- `*_csi_magnitude.png`: CSI magnitude over subcarriers

## CSI Tensor Meaning

Sionna preserves TX/RX/antenna dimensions in the CFR output. We used this to separate links and antennas.

The key indexing convention is:

```text
csi[rx, rx_ant, tx, tx_ant, time, subcarrier]
```

For a two-BS, one-RX, single-antenna setup:

```text
BS0 channel: csi[0, 0, 0, 0, 0, :]
BS1 channel: csi[0, 0, 1, 0, 0, :]
```

Pathloss is scalar per link, while CSI is a frequency-domain vector for that same link. The `multi_bs_csi` outputs explicitly pair each pathloss number with its corresponding CSI vector.

## Pilot Handling

Sionna RT does not output pilot symbols.

The correct workflow is:

1. Generate physical channel/CFR `H` with Sionna RT.
2. Choose a pilot matrix `X`.
3. Simulate received pilots with `Y = XH + N`.
4. Estimate channels with least squares or another estimator.

We verified:

- Orthogonal pilots recover separated channels in the noiseless checks.
- Identical pilots are rank deficient and cannot separate the channels.

## Static Built-In Scene Checks Completed

### `simple_wedge`

Purpose:

- Validate diffraction-style built-in scene behavior.
- Compare local output against Sionna documentation.
- Exercise pathloss, CSI, multi-BS separation, antenna separation, and pilot recovery.

Completed outputs:

- `Blender Preview Images/simple_wedge/` (core 2-BS scene outputs)
- `Blender Preview Images/simple_wedge/official_reference/`
- `Blender Preview Images/simple_wedge/antenna_pilot_comparison/`
- `Blender Preview Images/simple_wedge/visualization/`
- `Blender Preview Images/All Wedge Tests Different Freq Bands/`
- `Blender Preview Images/multi_bs_csi/`

Key results:

- Local reproduction matched the Sionna documentation simple-wedge setup closely.
- Documentation-style receiver index 400 pathloss difference was about `-0.0027 dB`.
- Frequency sweep behaved as expected:
  - `3.5 GHz`: about `74.53 dB` single-BS pathloss
  - `28 GHz`: about `92.59 dB` single-BS pathloss
  - Difference: about `18.06 dB`
- Multi-BS CSI and pathloss can be separated by TX dimension.
- Per-antenna dimensions are preserved.
- Orthogonal pilots recover separated channels; identical pilots do not.

Main scripts:

- `SionnaEM/tools/verify_simple_wedge_pipeline.py`
- `SionnaEM/tools/compare_simple_wedge_official.py`
- `SionnaEM/tools/wedge_checks_experiments.py`
- `SionnaEM/tools/visualize_wedge_general_scene.py`
- `SionnaEM/tools/create_wedge_scene_only_blend.py`

### `floor_wall`

Purpose:

- Basic known-scene smoke test.
- Save local CSI/pathloss outputs.
- Check against available official scene reference.

Completed outputs:

- `Blender Preview Images/floor_wall/`

Key result:

- Sionna public docs provide a scene reference image for `floor_wall`, but not official CSI/pathloss benchmark files.

Main scripts:

- `SionnaEM/tools/verify_floor_wall_pipeline.py`
- `SionnaEM/tools/compare_floor_wall_official.py`

## Mobility/Doppler Pipeline

For mobility scenes, we extended the same CIR/CFR/pathloss workflow with Doppler checks.

Documentation status:

- The Sionna Mobility tutorial provides explicit `street_canyon` settings and reference figures.
- It does not publish an official `street_canyon` output table for pathloss, CSI, path count, or RMS error.

Additional outputs:

- Path Doppler values from Sionna where available
- Delay-Doppler maps
- Doppler-vs-explicit-movement comparisons
- Moving-scene visual frames
- Single-TX vs multi-TX pathloss and CSI
- Pilot checks on dynamic-scene CSI

Completed scenes:

- `simple_reflector`
- `simple_street_canyon_with_cars`
- `etoile`

Completed outputs:

- `street_canyon/mobility/`
- `Blender Preview Images/mobility/`
- `Blender Preview Images/motion frames/`
- `etoile/point_drone_tx/`

Key results:

- `simple_reflector` is the clean sanity check:
  - One LoS path plus one reflected path
  - Delays remain unchanged across carrier frequency
  - Doppler scales with carrier frequency
- `street_canyon` is the richer dynamic case:
  - Moving cars
  - Multipath
  - Doppler-vs-explicit-movement agreement
  - Delay-Doppler heatmaps at `3.5 GHz` and `28 GHz`
  - CSI/pathloss/pilot checks
- `etoile/point_drone_tx` is the first drone dataset pass:
  - Built-in `etoile` urban geometry
  - Moving point-drone Tx
  - Fixed rooftop Rx baseline
  - Two-moving-Rx extension with `mobile_east` and `mobile_west`
  - Fixed visualization camera near the Rx
  - Curved `70 m` altitude trajectory over the denser east-side building cluster
  - Nominal speed about `18.15 m/s` over about `88.91 m`
  - Explicit frame-by-frame `PathSolver` runs for `50` trajectory samples at `0.1 s` spacing
  - `5 GHz` and `28 GHz` outputs

The current progress and output-organization note is:

- `SIONNA_RT_PIPELINE_PROGRESS_AND_ORGANIZATION.md`

Important `street_canyon/mobility` data and visual conventions:

- This bundle is separate from `etoile/point_drone_tx`.
- Each condition folder has RF outputs in `rt_data.npz`, `summary.json`, `csi.png`, `doppler.png`, and `paths.png`.
- Each condition folder has an `images/` folder with `scene_map.png`, `motion_map.png`, `link_paths_map.png`, `motion.gif`, and sampled motion frames `motion_frame_00.png`, `motion_frame_03.png`, and `motion_frame_06.png`.
- The condensed street-canyon bundle does not include full `sionna_3d_renderer/frames` or `sionna_3d_renderer/birds_eye_frames` folders.
- The Sionna Mobility tutorial provides `street_canyon` settings and reference figures, but not an official output table for pathloss, CSI, path count, or RMS error.

Important `etoile/point_drone_tx` data and visual conventions:

- The drone is a point transmitter, not a physical drone mesh.
- The moving receivers in the two-Rx extension are point receivers, not physical user/car/drone meshes.
- The current route is start-to-end along a curved cubic arc, not end-to-start and not the old short straight segment.
- Fixed-Rx baseline `pathloss_db` is `[50, 1, 1]`; two-moving-Rx `pathloss_db` is `[50, 2, 1]`.
- Fixed-Rx baseline `csi` is `[1, 1, 1, 1, 50, 512]`; two-moving-Rx `csi` is `[2, 1, 1, 1, 50, 512]`.
- Two-moving-Rx `csi_explicit_frames` is `[50, 2, 512]`, indexed as `[frame, rx, subcarrier]`.
- Two-moving-Rx receiver names are saved in `rx_names`; receiver trajectories are saved in `rx_positions_m` and `rx_velocity_mps`.
- `tx_velocity_mps` is saved per frame with shape `[50, 3]`; speed is effectively constant while direction changes along the curve.
- Motion PNG/GIF files are visual artifacts; CSI is complex channel data, not image data.
- `0.1 s` sampling is the pose/image sampling interval. Per-path Doppler is saved from Sionna geometry and velocity; delay-Doppler from 50 low-rate frames is only a trajectory diagnostic.
- `images/` contains `50` top-down motion frames plus `scene_map.png`, `motion_map.png`, and a motion GIF.
- `camera_drone_keyframes/` contains sparse Sionna fixed-camera keyframes at frames `000`, `025`, and `049`.
- `sionna_3d_renderer/frames/` contains `50` fixed-camera Sionna renderer frames for each band and condition.
- `sionna_3d_renderer/birds_eye_frames/` contains `50` bird's-eye Sionna renderer frames for each band and condition.
- `sionna_3d_drone_motion_full_5s_10fps.gif` and `sionna_3d_birds_eye_motion_full_5s_10fps.gif` provide the full 5-second renderer GIFs.
- Latest two-moving-Rx pathloss ranges are about `76.17-84.20 dB` and `76.35-93.56 dB` at `5 GHz`, and `91.20-99.33 dB` and `91.33-109.03 dB` at `28 GHz`, for `mobile_east` and `mobile_west` respectively.

Additional street-canyon delay-Doppler outputs:

- `Blender Preview Images/mobility/3p5GHz/street_canyon/local_delay_doppler.png`
- `Blender Preview Images/mobility/3p5GHz/street_canyon/local_delay_doppler_heatmap_zoom.png`
- `Blender Preview Images/mobility/3p5GHz/street_canyon/local_delay_doppler_3d.png`
- `Blender Preview Images/mobility/3p5GHz/street_canyon/local_delay_doppler_path_summary.png`
- `Blender Preview Images/mobility/3p5GHz/street_canyon/delay_doppler.npz`
- `Blender Preview Images/mobility/28GHz/street_canyon/local_delay_doppler.png`
- `Blender Preview Images/mobility/28GHz/street_canyon/local_delay_doppler_heatmap_zoom.png`
- `Blender Preview Images/mobility/28GHz/street_canyon/local_delay_doppler_3d.png`
- `Blender Preview Images/mobility/28GHz/street_canyon/local_delay_doppler_path_summary.png`
- `Blender Preview Images/mobility/28GHz/street_canyon/delay_doppler.npz`

Extra readable delay-Doppler views can be regenerated with:

```bash
.venv/bin/python SionnaEM/tools/plot_delay_doppler_views.py "Blender Preview Images/mobility/3p5GHz/street_canyon" --title "3.5GHz street_canyon"
.venv/bin/python SionnaEM/tools/plot_delay_doppler_views.py "Blender Preview Images/mobility/28GHz/street_canyon" --title "28GHz street_canyon"
```

Main scripts:

- `SionnaEM/tools/dynamic_mobility_experiments.py`
- `SionnaEM/tools/mobility_visual_export.py`
- `SionnaEM/tools/build_mobility_blends.py`

## Rooftop-BS Mobility Extension

Purpose:

- Place base stations on building rooftops in the built-in street-canyon scene.
- Place receivers on cars and rooftops.
- Evaluate one-BS and two-BS link behavior.

Completed outputs:

- `Blender Preview Images/mobility/rooftop_bs/one_bs/`
- `Blender Preview Images/mobility/rooftop_bs/two_bs/`

Saved data and plots:

- RT result NPZ files
- Per-link pathloss
- Per-link CSI magnitude
- Delay-Doppler overview
- Scene/path renderings
- Motion frames

Key result:

- Diffraction was required for rooftop-to-street links. Specular-only tracing produced no useful paths.

Main scripts:

- `SionnaEM/tools/street_canyon_rooftop_bs.py`
- `SionnaEM/tools/build_rooftop_bs_blends.py`

## Micro-Doppler Pipeline

This is a related but separate pipeline under `SionnaEM/src`.

Project idea:

1. Use Sionna RT once to get a static CIR for the environment.
2. Apply time-varying Micro-Doppler phase modulation to the CIR coefficients in post-processing.
3. Convert the dynamic CIR into a baseband signal.
4. Generate STFT spectrograms.
5. Compare integrated RT-based output against an analytic scatterer model.

Main module:

- `SionnaEM/src/micro_doppler_modulator.py`

Main integration demo:

- `SionnaEM/src/micro_doppler_integration_demo.py`

Completed steps:

- Step 1: standard Doppler baseline
- Step 2: single rotating scatterer Micro-Doppler
- Step 3: quadrotor UAV spectrogram
- Step 4: reusable `MicroDopplerModulator`
- Step 5: Sionna CIR integration demo

Still pending:

- Step 6: paper-parameter benchmarking / method comparison

Key parameters used in the integration demo:

- Carrier: `28 GHz`
- Quadrotor: `4` rotors, `2` blades per rotor
- Blade radius: `0.15 m`
- Rotation frequency: `100 Hz`
- Body velocity: `5 m/s`
- Sampling rate: `50 kHz`
- Observation duration: `0.2 s`

Completed Micro-Doppler outputs:

- `SionnaEM/figures/baseline_doppler_etoile.png`
- `SionnaEM/figures/micro_doppler_single_scatterer.png`
- `SionnaEM/figures/micro_doppler_quadrotor_spectrogram.png`
- `SionnaEM/figures/micro_doppler_modulator_validation.png`
- `SionnaEM/figures/integration_paris_los.png`
- `SionnaEM/figures/integration_paris_multipath.png`
- `SionnaEM/figures/integration_floor_wall.png`
- `SionnaEM/figures/integration_summary.png`

## Physical Scene Import Status

Physical/custom scene import is not counted as completed in this summary.

Existing helper tooling:

- `SionnaEM/tools/run_xml_rt_quick.py`
- `SionnaEM/tools/test_uploaded_meshes.py`

Those scripts can:

- Accept an XML scene or zip package.
- Extract a zip scene.
- Select an XML file.
- Convert visual BSDF materials into Sionna-compatible radio materials.
- Flatten simple mesh groups into top-level PLY shapes.
- Run a quick RT solve.
- Save the same CIR/CSI/pathloss plots as the built-in scene pipeline.

Current custom-scene test assets:

- `test_scenes/meshes.zip`
- `test_scenes/meshes/normalized_scene_sionna_materials_flat_shapes.xml`

The next milestone is to replace the built-in `load_scene(sionna.rt.scene...)` inputs with a validated imported physical scene and then rerun the same CSI/pathloss/Doppler pipeline.

## Main Report Artifact

The high-level saved report for the built-in-scene experiments is:

- `Blender Preview Images/experiment_summary.md`
