"""GIF showing the pipette's SIMULATED, MEASURED motion over a Task 5 crop.

The dot is a visualization, NOT physical dispensing. Per-frame pipette locations
are sourced from OT-2 PyBullet via Task 9, then projected into CV image pixels.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import numpy as np


class PlateGif:
    def __init__(self, plate: str, per_plate: list[dict], plate_map,
                 image_path: Path, gif_path: Path, every: int = 45,
                 show_3d: bool = False):
        self.plate = plate
        self.targets = per_plate
        self.map = plate_map
        self.gif_path = gif_path
        self.every = max(1, int(every))
        self.show_3d = show_3d
        if show_3d:
            import pybullet as p
            self._bullet = p
            self._view = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0.07, 0.055, 0.13], distance=0.77,
                yaw=48, pitch=-35, roll=0, upAxisIndex=2)
            self._projection = p.computeProjectionMatrixFOV(
                fov=55, aspect=600 / 520, nearVal=0.015, farVal=3.0)
        self.frame_counter = 0
        self.frames = []
        self._visual_markers = []
        self.deposited: list[dict] = []
        self.active_target = None
        self.photo = Image.open(image_path).convert('RGB')
        # Fast small previews: original CV coordinates remain unchanged.
        self.photo.thumbnail((720, 620))
        self.font = ImageFont.load_default()

    def _pixel_to_display(self, x: float, y: float):
        return (int(round(x * self.photo.width / self.map.width_px)),
                int(round(y * self.photo.height / self.map.height_px)))

    def _frame(self, world_xyz, target_xyz, steps: int, error_mm: float,
               event: str = "MOVING"):
        rgb = self.photo.copy()
        d = ImageDraw.Draw(rgb)
        for t in self.targets:
            px, py = self._pixel_to_display(t['tip_x_px'], t['tip_y_px'])
            d.line([(px - 5, py), (px + 5, py)], fill=(40, 230, 80), width=2)
            d.line([(px, py - 5), (px, py + 5)], fill=(40, 230, 80), width=2)
        for t in self.deposited:
            px, py = self._pixel_to_display(t['tip_x_px'], t['tip_y_px'])
            d.ellipse((px - 7, py - 7, px + 7, py + 7),
                      fill=(50, 175, 245), outline='white', width=2)
        tx, ty = self.map.world_to_pixel(target_xyz[0], target_xyz[1])
        tx, ty = self._pixel_to_display(tx, ty)
        d.ellipse((tx - 11, ty - 11, tx + 11, ty + 11), outline='yellow', width=3)
        px, py = self.map.world_to_pixel(world_xyz[0], world_xyz[1])
        px, py = self._pixel_to_display(px, py)
        d.line((px, py, tx, ty), fill='red', width=2)
        d.ellipse((px - 8, py - 8, px + 8, py + 8), fill=(255, 130, 20), outline='white', width=2)
        bar = Image.new('RGB', (rgb.width, 68), color=(16, 24, 38))
        a = ImageDraw.Draw(bar)
        a.text((8, 5), f'{self.plate} | PID / Task 12 SIMULATION | {event}', font=self.font, fill=(230, 242, 255))
        a.text((8, 23), f'step={steps} | distance to target={error_mm:.3f} mm | Z={world_xyz[2]:.4f} m', font=self.font, fill='white')
        a.text((8, 42), 'green=CV tip  orange=robot position  blue=SIMULATED deposit', font=self.font, fill=(196, 212, 225))
        composed = Image.new('RGB', (rgb.width, bar.height + rgb.height))
        composed.paste(bar, (0, 0))
        composed.paste(rgb, (0, bar.height))
        if self.show_3d:
            _, _, rgba, _, _ = self._bullet.getCameraImage(
                width=600, height=520,
                viewMatrix=self._view, projectionMatrix=self._projection,
                renderer=self._bullet.ER_TINY_RENDERER)
            camera = Image.fromarray(np.asarray(rgba, dtype=np.uint8).reshape(520, 600, 4)[:, :, :3])
            camera.thumbnail((600, composed.height))
            paired = Image.new('RGB', (composed.width + camera.width, composed.height), (20, 20, 20))
            paired.paste(camera, (0, (composed.height - camera.height) // 2))
            paired.paste(composed, (camera.width, 0))
            return paired
        return composed

    def on_step(self, world_xyz, target_xyz, step, error_mm):
        self.frame_counter += 1
        if self.frame_counter == 1 or self.frame_counter % self.every == 0:
            if len(self.frames) < 180:
                self.frames.append(self._frame(world_xyz, target_xyz, step, error_mm))

    def mark_deposit(self, world_xyz, row):
        # A static visual marker in the PyBullet viewport: not fluid dispense.
        if self.show_3d and self.map.plate_surface_z_m is not None:
            p = self._bullet
            marker = p.createVisualShape(p.GEOM_SPHERE, radius=0.004,
                                         rgbaColor=[0.12, 0.8, 0.5, 0.95])
            body = p.createMultiBody(baseMass=0, baseVisualShapeIndex=marker,
                                     basePosition=[float(world_xyz[0]), float(world_xyz[1]),
                                                   float(self.map.plate_surface_z_m) + 0.004])
            self._visual_markers.append(body)
        self.deposited.append(row)
        self.frames.append(self._frame(world_xyz, world_xyz, 0, 0.0, event='SIMULATED DEPOSIT'))

    def save(self) -> None:
        if not self.frames:
            return
        self.gif_path.parent.mkdir(parents=True, exist_ok=True)
        self.frames[0].save(self.gif_path, save_all=True,
                            append_images=self.frames[1:], duration=95, loop=0, optimize=True)


def preview_image(preview_dir: Path, plate: str) -> Path | None:
    for extension in ('.jpg', '.jpeg', '.png'):
        f = preview_dir / f'{plate}_petri{extension}'
        if f.is_file():
            return f
    return None
