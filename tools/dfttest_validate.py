"""
Phase 5 parity validator — compares NVEncC --vpp-fft3d sigma_curve output
against dfttest2.Backend.CPU() reference on the same slocation curve.

Both paths decode the same source, apply the same denoise curve at the same
block_size / overlap / tbsize, and emit Y4M frames. We compare Y planes
numerically.

The candidate (NVEncC) goes through a lossless HEVC encode + ffmpeg decode
because NVEncC has no "filter-only" output mode. Lossless HEVC is bitwise
reversible for 10-bit content, so this doesn't add error.

Usage:
    python dfttest_validate.py [--frames N] [--gain 24] [--tier 1.0]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


NVENCC = r"D:/python_playground/NVEnc/_build/x64/RelStatic/NVEncC64.exe"
SOURCE = r"E:/temp/video-filters-testing/source-intermediates/sony ax700/C0054.mov"
PROFILE_JSON = r"D:/python_playground/media_processor/media_processor/config/noise_profiles/sony_fdrax700_bundled.json"
VSPIPE = shutil.which("vspipe") or r"C:/Program Files/VapourSynth/core/vspipe.exe"
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"


# ------------------------------- Y4M parsing -------------------------------

def parse_y4m(path: Path, max_frames: int | None = None) -> tuple[list[np.ndarray], dict]:
    """Parse a Y4M file and return Y-plane frames as a list of np.uint16 (or uint8) arrays.

    Returns (frames, header_info).
    """
    with open(path, "rb") as f:
        # Read the stream header (ends at \n)
        hdr = b""
        while not hdr.endswith(b"\n"):
            b = f.read(1)
            if not b:
                raise RuntimeError(f"unexpected EOF in Y4M header: {path}")
            hdr += b

        assert hdr.startswith(b"YUV4MPEG2 "), f"not a y4m: {hdr[:20]!r}"
        tokens = hdr[10:].strip().split()
        info = {}
        for tok in tokens:
            tag = chr(tok[0])
            val = tok[1:].decode()
            info.setdefault(tag, []).append(val)
        width = int(info["W"][0])
        height = int(info["H"][0])
        # Colorspace, e.g. 422p10 or 420mpeg2
        cs = info.get("C", ["420jpeg"])[0]

        # Determine bytes per Y sample + chroma subsampling
        if "10" in cs or "12" in cs or "16" in cs or "p10" in cs:
            bytes_per_sample = 2
            dtype = np.uint16
        else:
            bytes_per_sample = 1
            dtype = np.uint8

        if cs.startswith("420"):
            uv_w = width // 2
            uv_h = height // 2
        elif cs.startswith("422"):
            uv_w = width // 2
            uv_h = height
        elif cs.startswith("444"):
            uv_w = width
            uv_h = height
        else:
            raise RuntimeError(f"unknown chroma format: {cs}")

        y_bytes = width * height * bytes_per_sample
        uv_bytes = uv_w * uv_h * bytes_per_sample

        frames = []
        while True:
            # Frame header: "FRAME\n" or "FRAME ...\n"
            fhdr = b""
            while not fhdr.endswith(b"\n"):
                b = f.read(1)
                if not b:
                    return frames, {**info, "width": width, "height": height, "cs": cs}
                fhdr += b
            if not fhdr.startswith(b"FRAME"):
                raise RuntimeError(f"expected FRAME, got {fhdr!r}")
            y_buf = f.read(y_bytes)
            if len(y_buf) < y_bytes:
                break
            # Skip UV planes (we only compare luma for now — that's where denoise matters most).
            f.read(uv_bytes * 2)
            y_arr = np.frombuffer(y_buf, dtype=dtype).reshape(height, width)
            frames.append(y_arr.copy())
            if max_frames is not None and len(frames) >= max_frames:
                break

        return frames, {**info, "width": width, "height": height, "cs": cs}


# ------------------------------- Reference path -------------------------------

def render_reference(src: str, slocation_pairs: list[tuple[float, float]],
                     nframes: int, workdir: Path, block_size: int = 16,
                     tbsize: int = 3, overlap: float = 0.0,
                     zmean: bool = True) -> Path:
    """Run vspipe with dfttest2.Backend.CPU and return path to emitted Y4M."""
    # dfttest2 slocation format: flat list [f1, s1, f2, s2, ...]
    slocation_flat = []
    for f, s in slocation_pairs:
        slocation_flat.extend([float(f), float(s)])

    # dfttest sosize: number of overlapping pixels per block. overlap=0 -> sosize=0.
    sosize = int(round(block_size * overlap))

    vpy = workdir / "reference.vpy"
    vpy.write_text(f"""\
import vapoursynth as vs
import dfttest2
core = vs.core

clip = core.lsmas.LWLibavSource(r"{src}")
clip = clip[:{nframes}]
# dfttest2 prefers planar YUV 10-bit or 16-bit. ProRes 4:2:2 decodes to 16-bit via lsmas.
clip = dfttest2.DFTTest(
    clip,
    slocation={slocation_flat},
    sbsize={block_size},
    sosize={sosize},
    tbsize={tbsize},
    zmean={zmean},
    backend=dfttest2.Backend.CPU(),
)
clip.set_output()
""", encoding="utf-8")

    out = workdir / "reference.y4m"
    print(f"  [ref] vspipe -> {out.name}")
    cmd = [VSPIPE, "-c", "y4m", str(vpy), str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("vspipe stderr:", r.stderr)
        raise RuntimeError("vspipe failed")
    return out


# ------------------------------- Candidate path -------------------------------

def render_candidate(src: str, slocation_pairs: list[tuple[float, float]],
                     nframes: int, workdir: Path, block_size: int = 16,
                     overlap: float = 0.0, temporal: int = 1,
                     prec: str = "auto") -> Path:
    """NVEncC --vpp-fft3d sigma_curve=... + ffmpeg decode to Y4M."""
    # sigma_curve string form: f1/s1;f2/s2;...
    sigma_curve_str = ";".join(f"{f:.6f}/{s:.6f}" for f, s in slocation_pairs)

    fft3d_args = ",".join([
        f"sigma=8",  # ignored when sigma_curve is set, but parser wants it
        f"amount=1.0",
        f"block_size={block_size}",
        f"overlap={overlap}",
        f"temporal={temporal}",
        f"method=0",
        f"prec={prec}",
        f"sigma_curve={sigma_curve_str}",
    ])

    hevc = workdir / "candidate.hevc"
    print(f"  [nvencc] fft3d sigma_curve ({len(slocation_pairs)} pts) -> {hevc.name}")
    cmd = [
        NVENCC, "--avsw", "-i", src,
        "--trim", f"0:{nframes - 1}",
        "--vpp-fft3d", fft3d_args,
        "--codec", "hevc", "--profile", "main10",
        "--lossless",
        "-o", str(hevc),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("nvencc stderr:", r.stderr[-2000:])
        raise RuntimeError("nvencc failed")

    out = workdir / "candidate.y4m"
    print(f"  [ffmpeg] decode -> {out.name}")
    # Force same pixel format as reference (yuv422p10le for 10-bit ProRes source).
    cmd = [FFMPEG, "-y", "-loglevel", "warning",
           "-i", str(hevc),
           "-pix_fmt", "yuv422p10le",
           "-strict", "-1",
           "-f", "yuv4mpegpipe",
           str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("ffmpeg stderr:", r.stderr)
        raise RuntimeError("ffmpeg failed")
    return out


# ------------------------------- Comparison -------------------------------

def compare(ref_y4m: Path, cand_y4m: Path) -> dict:
    ref_frames, ref_info = parse_y4m(ref_y4m)
    cand_frames, cand_info = parse_y4m(cand_y4m)

    print(f"  ref:  {len(ref_frames)} frames, {ref_info['width']}x{ref_info['height']} {ref_info['cs']}")
    print(f"  cand: {len(cand_frames)} frames, {cand_info['width']}x{cand_info['height']} {cand_info['cs']}")

    n = min(len(ref_frames), len(cand_frames))
    if n == 0:
        raise RuntimeError("no frames decoded")

    per_frame = []
    for i in range(n):
        r = ref_frames[i].astype(np.int32)
        c = cand_frames[i].astype(np.int32)
        if r.shape != c.shape:
            raise RuntimeError(f"frame {i} shape mismatch: ref {r.shape} vs cand {c.shape}")
        diff = np.abs(r - c)
        per_frame.append((float(diff.mean()), int(diff.max())))

    means = np.array([m for m, _ in per_frame])
    maxes = np.array([x for _, x in per_frame])
    return {
        "frames_compared": n,
        "mean_diff_overall": float(means.mean()),
        "mean_diff_worst_frame": float(means.max()),
        "max_diff_overall": int(maxes.max()),
        "per_frame": per_frame,
    }


# ------------------------------- Main -------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--gain", type=str, default="24", help="HLG3 gain level key (0,3,6,...33)")
    ap.add_argument("--tier", type=float, default=1.0, help="sigma_scale multiplier")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--overlap", type=float, default=0.0)
    ap.add_argument("--tbsize", type=int, default=3)
    ap.add_argument("--zmean", action="store_true", default=True)
    ap.add_argument("--no-zmean", dest="zmean", action="store_false")
    ap.add_argument("--prec", default="auto", choices=["auto", "fp16", "fp32"])
    ap.add_argument("--keep", action="store_true", help="keep workdir")
    args = ap.parse_args()

    with open(PROFILE_JSON) as f:
        profile = json.load(f)
    slocation_flat = profile["picture_profiles"]["hlg3"]["noise_profiles"][args.gain]["slocation"]
    pairs_base = list(zip(slocation_flat[::2], slocation_flat[1::2]))
    pairs = [(f, s * args.tier) for f, s in pairs_base]
    print(f"slocation gain={args.gain} dB, tier x{args.tier}:")
    for f, s in pairs:
        print(f"  {f:.3f} -> {s:.3f}")

    workdir = Path(tempfile.mkdtemp(prefix="nvenc_dfttest_validate_"))
    print(f"workdir: {workdir}")
    try:
        temporal = 1 if args.tbsize >= 2 else 0
        ref = render_reference(SOURCE, pairs, args.frames, workdir,
                               block_size=args.block_size, tbsize=args.tbsize,
                               overlap=args.overlap, zmean=args.zmean)
        cand = render_candidate(SOURCE, pairs, args.frames, workdir,
                                block_size=args.block_size, overlap=args.overlap,
                                temporal=temporal, prec=args.prec)
        result = compare(ref, cand)
        print()
        print("=== RESULT ===")
        print(f"frames compared:       {result['frames_compared']}")
        print(f"mean |diff| overall:   {result['mean_diff_overall']:.3f}")
        print(f"mean |diff| worst frm: {result['mean_diff_worst_frame']:.3f}")
        print(f"max  |diff| overall:   {result['max_diff_overall']}")
        print()
        # 10-bit values are 0..1023, so "N LSB" = N raw units
        thr = 8
        if result["mean_diff_overall"] < thr:
            print(f"PASS (mean < {thr} LSB on 10-bit)")
        else:
            print(f"FAIL (mean >= {thr} LSB on 10-bit)")
    finally:
        if args.keep:
            print(f"(kept workdir {workdir})")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
