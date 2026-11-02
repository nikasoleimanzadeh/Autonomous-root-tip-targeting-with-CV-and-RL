"""reinforcement learning control: record the REAL frozen SAC policy in the original OT-2 PyBullet simulator.

Run from the repository root:
    python -B task11_rl_controller/create_rl_movement_gifs.py

Uses fixed, previously successful cases from the untouched 100-case holdout.
SAC.predict() controls every robot action: no PID, no teacher, no replay, no training.
Only writes under a NEW task11_rl_controller/evidence/rl_gifs_* directory.
Does not modify the model, holdout, environment, or their directories.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

TASK = Path(__file__).resolve().parent
REPO = TASK.parent
PATH_Y = TASK / "Path_Y"
for folder in (REPO / "src", PATH_Y):
    if not folder.is_dir():
        raise FileNotFoundError(f"Required code directory missing: {folder}")
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from path_y_env import PathYEnv

RUN = PATH_Y / "runs" / "final_push_20261012_155636_757034"
MODEL = RUN / "best_learned_sac_model.zip"
CASES_FILE = RUN / "holdout_cases_DO_NOT_TUNE_ON.json"
RESULTS_FILE = RUN / "final_holdout.csv"

W, H = 640, 400
PANEL_W = 290


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for block in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def project(world, view, projection):
    """Project a 3-D world coordinate into the PyBullet camera's image."""
    v = np.asarray(view, dtype=np.float64).reshape((4, 4), order="F")
    p = np.asarray(projection, dtype=np.float64).reshape((4, 4), order="F")
    clip = p @ v @ np.array([*world, 1.0], dtype=np.float64)
    if clip[3] <= 1e-9:
        return None
    ndc = clip[:3] / clip[3]
    if not np.all(np.isfinite(ndc)):
        return None
    return int((ndc[0] + 1) * W / 2), int((1 - ndc[1]) * H / 2)


def camera_for_case(pbullet, start, goal):
    """Use one fixed 3-D camera per episode so robot motion is visible."""
    center = 0.5 * (np.asarray(start) + np.asarray(goal))
    center[2] -= 0.025
    distance = max(0.70, 1.6 * float(np.linalg.norm(np.asarray(start) - goal)))
    projection = pbullet.computeProjectionMatrixFOV(
        fov=58, aspect=W / H, nearVal=0.01, farVal=5.0
    )
    # Try two useful robot-viewing angles. Do not change the physics.
    angles = [(40, -35), (65, -45), (15, -55)]
    best = None
    for yaw, pitch in angles:
        view = pbullet.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=center.tolist(), distance=distance,
            yaw=yaw, pitch=pitch, roll=0, upAxisIndex=2,
        )
        picture = pbullet.getCameraImage(
            width=W, height=H, viewMatrix=view, projectionMatrix=projection,
            renderer=pbullet.ER_TINY_RENDERER,
        )
        rgba = np.asarray(picture[2], dtype=np.uint8).reshape(H, W, 4)
        segmentation = np.asarray(picture[4]).reshape(H, W)
        visible_fraction = float(np.mean(segmentation >= 0))
        # Prefer some visible geometry, without filling the entire frame.
        score = -abs(visible_fraction - 0.32)
        if best is None or score > best[0]:
            best = (score, view, rgba[:, :, :3].copy(), visible_fraction)
    _, view, initial_rgb, visible_fraction = best
    if visible_fraction < 0.003:
        raise RuntimeError(
            "PyBullet camera could not see robot geometry. "
            "No misleading animation was created. Model remains untouched."
        )
    return view, projection, initial_rgb, visible_fraction


def capture(pbullet, view, projection):
    picture = pbullet.getCameraImage(
        width=W, height=H, viewMatrix=view, projectionMatrix=projection,
        renderer=pbullet.ER_TINY_RENDERER,
    )
    return np.asarray(picture[2], dtype=np.uint8).reshape(H, W, 4)[:, :, :3].copy()


def frame_image(rgb, case_id, start, goal, pos, error_mm, step, path, errors,
                view, projection, settled=False):
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (W + PANEL_W, H), (248, 250, 253))
    canvas.paste(Image.fromarray(rgb, mode="RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    # 3-D ground-truth tip positions projected over the actual PyBullet render.
    for j in range(1, len(path)):
        a = project(path[j - 1], view, projection)
        b = project(path[j], view, projection)
        if a is not None and b is not None:
            draw.line((a[0], a[1], b[0], b[1]), fill=(0, 214, 235), width=3)
    target_pixel = project(goal, view, projection)
    if target_pixel is not None:
        x, y = target_pixel
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), outline=(240, 60, 65), width=4)
        draw.line((x - 14, y, x + 14, y), fill=(240, 60, 65), width=2)
        draw.line((x, y - 14, x, y + 14), fill=(240, 60, 65), width=2)
    tip_pixel = project(pos, view, projection)
    if tip_pixel is not None:
        x, y = tip_pixel
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(20, 214, 228), outline="black", width=2)

    draw.rectangle((0, 0, W - 1, 29), fill=(19, 31, 49))
    draw.text((12, 9), f"REAL PyBullet simulation | SAC policy only | {case_id}",
              fill="white", font=font)
    draw.rectangle((W, 0, W + PANEL_W, H), fill=(246, 249, 254))
    tx = W + 16
    textlines = [
        (16, "OT-2  |  Trained SAC", (24, 42, 67)),
        (46, f"Holdout case: {case_id}", (35, 57, 81)),
        (72, f"Simulation step: {step}", (35, 57, 81)),
        (99, f"Error: {error_mm:.3f} mm", (8, 96, 149)),
        (128, "Settled: YES" if settled else "Settled: not yet", (0, 121, 83) if settled else (101, 101, 101)),
        (158, "START (metres)", (35, 57, 81)),
        (176, "  " + ", ".join(f"{x:+.3f}" for x in start), (58, 72, 91)),
        (202, "TARGET (metres)", (35, 57, 81)),
        (220, "  " + ", ".join(f"{x:+.3f}" for x in goal), (58, 72, 91)),
        (246, "CURRENT TIP (metres)", (35, 57, 81)),
        (264, "  " + ", ".join(f"{x:+.3f}" for x in pos), (58, 72, 91)),
        (300, "Distance to target (mm)", (35, 57, 81)),
    ]
    for y, t, color in textlines:
        draw.text((tx, y), t, fill=color, font=font)

    # Fixed chart axis over the whole episode's initial error range.
    x0, y0, x1, y1 = tx + 8, 328, W + PANEL_W - 16, 377
    draw.rectangle((x0, y0, x1, y1), outline=(168, 181, 195), width=1)
    bound = max(1.1, float(errors[0]) * 1.07)
    pts = [(int(x0 + k / 600 * (x1 - x0)), int(y1 - min(e / bound, 1) * (y1 - y0)))
           for k, e in enumerate(errors)]
    if len(pts) > 1:
        draw.line(pts, fill=(6, 112, 187), width=2)
    draw.text((x0 + 5, y0 + 5), "1 mm target", fill=(118, 126, 141), font=font)
    return canvas


def run_case(model, env, case, expected, output_dir, frame_stride=2, fps=12):
    import pybullet as p

    case_id = str(case["case"])
    start = np.asarray(case["start"], dtype=np.float64)
    goal = np.asarray(case["target"], dtype=np.float64)
    obs, info = env.reset(options={"start": start.tolist(), "target": goal.tolist()})
    view, projection, image0, visible = camera_for_case(p, start, goal)
    print(f"{case_id}: camera visible geometry {visible:.1%}", flush=True)

    path = [np.asarray(info["pipette"], dtype=float)]
    errors = [float(info["distance_mm"])]
    frames = [frame_image(image0, case_id, start, goal, path[-1], errors[-1],
                          0, path, errors, view, projection)]
    timeseries = [{"step": 0, "error_mm": errors[-1], "pipette_xyz_m": json.dumps(path[-1].tolist())}]
    settled = False
    unsafe = False
    final_step = None

    for step in range(1, 601):
        action, _ = model.predict(obs, deterministic=True)
        action = np.asarray(action, dtype=np.float32).reshape(3)
        if not np.all(np.isfinite(action)):
            raise RuntimeError(f"Nonfinite SAC action in {case_id} step {step}")
        obs, _, terminated, truncated, info = env.step(action)
        pos = np.asarray(info["pipette"], dtype=np.float64)
        path.append(pos)
        errors.append(float(info["distance_mm"]))
        timeseries.append({"step": step, "error_mm": errors[-1],
                           "pipette_xyz_m": json.dumps(pos.tolist())})
        settled = bool(info["settled"])
        unsafe = bool(info["unsafe"])
        if step % frame_stride == 0 or terminated or truncated:
            frames.append(frame_image(capture(p, view, projection), case_id, start, goal,
                                      pos, errors[-1], step, path, errors,
                                      view, projection, settled))
        if terminated or truncated:
            final_step = step
            break

    if not settled or unsafe:
        raise RuntimeError(
            f"STOP: {case_id} did not reproduce successful settling; "
            f"settled={settled}, unsafe={unsafe}, final_error={errors[-1]:.4f} mm. "
            "No success GIF will be saved. Model untouched."
        )
    reference_error = float(expected["final_error_mm"])
    if abs(errors[-1] - reference_error) > 0.05:
        raise RuntimeError(
            f"STOP: {case_id} final error {errors[-1]:.4f} mm differs from "
            f"original holdout {reference_error:.4f} mm. Model untouched."
        )
    if final_step != int(expected["steps"]):
        raise RuntimeError(f"STOP: {case_id} step count differs from verified holdout")

    # Hold final success frame so reviewers can see the result.
    frames.extend([frames[-1]] * (fps + 3))
    out = output_dir / f"{case_id}_SAC_success.gif"
    frames[0].save(
        out, format="GIF", save_all=True, append_images=frames[1:],
        duration=int(1000 / fps), loop=0, optimize=False,
    )
    frames[0].save(output_dir / f"{case_id}_start.png")
    frames[-1].save(output_dir / f"{case_id}_settled.png")
    with (output_dir / f"{case_id}_trajectory.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["step", "error_mm", "pipette_xyz_m"])
        writer.writeheader()
        writer.writerows(timeseries)
    result = {
        "case": case_id,
        "group": case["group"],
        "source": "Original frozen 100-case holdout; successful case chosen for visual demonstration",
        "controller": "Best SAC checkpoint (RL-initialized, imitation-fine-tuned); deterministic inference",
        "initial_distance_mm": round(errors[0], 6),
        "final_error_mm": round(errors[-1], 6),
        "steps": final_step,
        "settled": settled,
        "unsafe": unsafe,
        "source_reference_final_error_mm": reference_error,
        "gif": out.name,
    }
    print(f"GIF SUCCESS: {out.name} | {errors[0]:.2f} -> {errors[-1]:.4f} mm | {final_step} steps", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description="Record REAL SAC robot GIFs safely")
    parser.add_argument("--cases", nargs="+", default=["wide_04", "near_19"],
                        help="Successful original holdout case IDs to record")
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()
    if args.frame_stride < 1 or args.fps < 1 or args.fps > 30:
        parser.error("frame-stride must be positive and fps must be 1..30")
    if len(set(args.cases)) != len(args.cases):
        parser.error("Duplicate case IDs")

    from stable_baselines3 import SAC
    from PIL import Image
    import pybullet as p

    for needed in (MODEL, CASES_FILE, RESULTS_FILE):
        if not needed.is_file():
            raise FileNotFoundError(f"STOP: Protected file not found: {needed}")
    cases_all = json.loads(CASES_FILE.read_text(encoding="utf-8"))
    with RESULTS_FILE.open("r", encoding="utf-8", newline="") as fp:
        original_rows = list(csv.DictReader(fp))
    if len(cases_all) != 100 or len(original_rows) != 100:
        raise RuntimeError("STOP: Expected unchanged 100-case holdout")
    by_case = {x["case"]: x for x in cases_all}
    results = {x["case"]: x for x in original_rows}
    for case_id in args.cases:
        if case_id not in by_case or case_id not in results:
            raise RuntimeError(f"STOP: Unknown holdout case {case_id}")
        if results[case_id]["settled"].lower() != "true":
            raise RuntimeError(f"STOP: Case {case_id} did not settle in original holdout")
    old_hash = sha256(MODEL)
    print("MODEL SHA256 BEFORE:", old_hash, flush=True)
    print("Rendering genuine PyBullet frames; no training and no model edits.", flush=True)
    model = SAC.load(str(MODEL), device="cpu")
    out = TASK / "evidence" / ("rl_gifs_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    out.mkdir(parents=True, exist_ok=False)
    env = PathYEnv(start_strategy="uniform", max_steps=600)
    results_all = []
    try:
        for case_id in args.cases:
            results_all.append(run_case(model, env, by_case[case_id], results[case_id], out,
                                        frame_stride=args.frame_stride, fps=args.fps))
    finally:
        env.close()
        new_hash = sha256(MODEL)
        print("MODEL SHA256 AFTER:", new_hash, flush=True)
        if old_hash != new_hash:
            raise RuntimeError("STOP: Model hash changed unexpectedly!")
        print("MODEL UNCHANGED: PASS", flush=True)
    (out / "gif_evidence_summary.json").write_text(
        json.dumps({"model_sha256": old_hash, "examples": results_all,
                    "note": "These GIFs are illustrative successful episodes, NOT the full success rate. "
                            "Full holdout is 87/100 settled with 0.6411 mm mean final error."},
                   indent=2), encoding="utf-8"
    )
    print("GIF EVIDENCE COMPLETE:", out.resolve(), flush=True)


if __name__ == "__main__":
    main()
