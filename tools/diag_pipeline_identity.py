"""Diagnostic: pipe source through both paths with NO denoise.

Reference: vspipe, no filters → y4m
Candidate: NVEncC, no vpp, --lossless → ffmpeg decode → y4m

If mean |diff| is near zero on the Y plane, pipeline is clean and any
subsequent divergence is from the denoise math. If large, there's a
y4m parsing or lossless-encode issue.
"""
from __future__ import annotations
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from dfttest_validate import parse_y4m, NVENCC, VSPIPE, FFMPEG, SOURCE
import numpy as np


def main():
    workdir = Path(tempfile.mkdtemp(prefix="diag_pipeline_"))
    print(f"workdir: {workdir}")
    nframes = 5

    # Reference: identity vspipe.
    vpy = workdir / "ident.vpy"
    vpy.write_text(f"""\
import vapoursynth as vs
core = vs.core
clip = core.lsmas.LWLibavSource(r"{SOURCE}")
clip = clip[:{nframes}]
clip.set_output()
""", encoding="utf-8")
    ref = workdir / "ref.y4m"
    subprocess.run([VSPIPE, "-c", "y4m", str(vpy), str(ref)], check=True)

    # Candidate: NVEncC --lossless, no vpp.
    hevc = workdir / "cand.hevc"
    subprocess.run([
        NVENCC, "--avsw", "-i", SOURCE,
        "--trim", f"0:{nframes - 1}",
        "--codec", "hevc", "--profile", "main10", "--lossless",
        "-o", str(hevc),
    ], check=True, capture_output=True)

    cand = workdir / "cand.y4m"
    subprocess.run([
        FFMPEG, "-y", "-loglevel", "warning",
        "-i", str(hevc),
        "-pix_fmt", "yuv422p10le",
        "-strict", "-1",
        "-f", "yuv4mpegpipe",
        str(cand),
    ], check=True)

    ref_frames, ri = parse_y4m(ref)
    cand_frames, ci = parse_y4m(cand)
    print(f"ref: {len(ref_frames)} x {ri['width']}x{ri['height']} {ri['cs']}")
    print(f"cand: {len(cand_frames)} x {ci['width']}x{ci['height']} {ci['cs']}")
    n = min(len(ref_frames), len(cand_frames))
    for i in range(n):
        r = ref_frames[i].astype(np.int32)
        c = cand_frames[i].astype(np.int32)
        diff = np.abs(r - c)
        # Find location of max diff for drill-down
        flat_idx = int(np.argmax(diff))
        y = flat_idx // r.shape[1]
        x = flat_idx % r.shape[1]
        print(f"  frame {i}: mean={diff.mean():.3f}, max={diff.max()} at (y={y},x={x}) ref={r[y,x]} cand={c[y,x]}")
    shutil.rmtree(workdir)


if __name__ == "__main__":
    main()
