#!/usr/bin/env python3
"""Build task5_init_drift_zoom_labeled.png: zoomed bowl-region comparison of the
task-5 initial state under healthy (3.2.7) vs broken (3.8.1) mujoco physics.

Layout: two full-width labeled bands, top = mujoco 3.2.7, bottom = 3.8.1; each
band holds frame-5 crops of the same two episodes (identical init layouts across
versions, so columns pair up). Frame 5 is before the policy has moved anything.

Needs ffmpeg on PATH and the eval videos under results/failure_videos/.
"""
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent / "results" / "failure_videos"
OUT = ROOT / "task5_init_drift_zoom_labeled.png"

FRAME_IDX = 4               # 5th frame, 0-based
CROP = (25, 85, 195, 255)   # bowl + ramekin window in the 360x360 agentview
SCALE = 3
EPS = [0, 1]
FONT = "/System/Library/Fonts/Helvetica.ttc"  # swap for a DejaVu path on Linux

BANDS = [
    ("mj327_task5", "mujoco 3.2.7: bowl seated on the ramekin"),
    ("mj381_task5", "mujoco 3.8.1: bowl tilted on the rim, half overhanging"),
]


def frame(video: Path, idx: int, dst: Path) -> Image.Image:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(video),
         "-vf", f"select=eq(n\\,{idx})", "-vframes", "1", "-y", str(dst)],
        check=True,
    )
    return Image.open(dst)


def main() -> None:
    cell = (CROP[2] - CROP[0]) * SCALE
    band_h = 44
    canvas = Image.new("RGB", (cell * len(EPS), (band_h + cell) * len(BANDS)), "black")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, 30)

    with tempfile.TemporaryDirectory() as td:
        y = 0
        for run, label in BANDS:
            draw.rectangle([0, y, canvas.width, y + band_h], fill=(20, 20, 20))
            draw.text((12, y + band_h // 2), label, font=font, fill="white", anchor="lm")
            y += band_h
            for i, ep in enumerate(EPS):
                im = frame(ROOT / run / f"eval_episode_{ep}.mp4", FRAME_IDX, Path(td) / "f.png")
                im = im.crop(CROP).resize((cell, cell), Image.LANCZOS)
                canvas.paste(im, (i * cell, y))
            y += cell

    canvas.save(OUT)
    print(f"saved {OUT} {canvas.size}")


if __name__ == "__main__":
    main()
