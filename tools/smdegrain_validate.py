"""
Phase 7 parity validator for nvenc-smdegrain.

Compares, on the same source clip:
  - CPU reference: vspipe -> havsfunc.SMDegrain(...) -> y4m
  - GPU candidate: NVEncC --vpp-smdegrain ... --lossless -> hevc -> ffmpeg decode -> y4m

Metrics per run:
  - frames compared
  - mean |ref - cand| and max |ref - cand| on Y plane (LSB, bit-depth-native)
  - VMAF(ref, cand)        perceptual parity GPU vs CPU
  - VMAF(cand, source)     how far the GPU filter moves content from the noisy original
  - VMAF(ref, source)      how far the CPU filter moves content

Usage:
    python smdegrain_validate.py --camera xh2s --preset medium_cs --bit-depth 10 --frames 30
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


NVENCC = r"D:/python_playground/NVEnc-smdegrain/_build/x64/RelStatic/NVEncC64.exe"
VSPIPE = shutil.which("vspipe") or r"C:/Users/Eduard/AppData/Local/Programs/VapourSynth/core/vspipe.exe"
FFMPEG = shutil.which("ffmpeg") or "ffmpeg"

FIXTURES = {
    "xh2s":  r"Q:/selectcode/20260416 ifh hannover/denoise real world samples xh2s/fuji_xh2s_flog2c_4k_iso3200_XH2S3466.MOV",
    "ax700": r"E:/temp/video-filters-testing/source-intermediates/sony ax700/C0054.mov",
}

# media_processor SMDEGRAIN_PRESETS (config/denoise.py)
PRESETS = {
    "light_cs":  {"tr": 1, "thSAD": 200, "limit": 100, "contrasharp": True, "thSCD1": 9999, "search": 3},
    "medium_cs": {"tr": 2, "thSAD": 300, "limit": 160, "contrasharp": True, "thSCD1": 9999, "search": 3},
    "heavy_cs":  {"tr": 2, "thSAD": 500, "limit": 255, "contrasharp": True, "thSCD1": 9999, "search": 3},
    "extreme":   {"tr": 3, "thSAD": 600, "limit": 255, "contrasharp": True, "thSCD1": 9999, "search": 3},
}


# ------------------------------- Y4M parsing -------------------------------
# Same parser as dfttest_validate — only luma compared.

def parse_y4m_luma(path: Path, max_frames: int | None = None):
    with open(path, "rb") as f:
        hdr = b""
        while not hdr.endswith(b"\n"):
            ch = f.read(1)
            if not ch:
                raise RuntimeError(f"EOF in Y4M header: {path}")
            hdr += ch
        assert hdr.startswith(b"YUV4MPEG2 "), f"not y4m: {hdr[:20]!r}"
        tokens = hdr[10:].strip().split()
        info = {}
        for tok in tokens:
            info.setdefault(chr(tok[0]), []).append(tok[1:].decode())
        width = int(info["W"][0])
        height = int(info["H"][0])
        cs = info.get("C", ["420jpeg"])[0]

        if "10" in cs or "12" in cs or "16" in cs or "p10" in cs:
            bps, dtype = 2, np.uint16
        else:
            bps, dtype = 1, np.uint8

        if cs.startswith("420"):
            uv_w, uv_h = width // 2, height // 2
        elif cs.startswith("422"):
            uv_w, uv_h = width // 2, height
        elif cs.startswith("444"):
            uv_w, uv_h = width, height
        else:
            raise RuntimeError(f"unknown chroma: {cs}")

        y_bytes = width * height * bps
        uv_bytes = uv_w * uv_h * bps

        frames = []
        while True:
            fhdr = b""
            while not fhdr.endswith(b"\n"):
                ch = f.read(1)
                if not ch:
                    return frames, {**info, "width": width, "height": height, "cs": cs}
                fhdr += ch
            if not fhdr.startswith(b"FRAME"):
                raise RuntimeError(f"expected FRAME, got {fhdr!r}")
            y_buf = f.read(y_bytes)
            if len(y_buf) < y_bytes:
                break
            f.read(uv_bytes * 2)  # skip UV
            frames.append(np.frombuffer(y_buf, dtype=dtype).reshape(height, width).copy())
            if max_frames is not None and len(frames) >= max_frames:
                break
        return frames, {**info, "width": width, "height": height, "cs": cs}


# ------------------------------- Reference: haf.SMDegrain -------------------------------

def render_reference_y4m(src: str, preset: dict, nframes: int, out_format: str, workdir: Path) -> Path:
    """Run vspipe with havsfunc.SMDegrain. out_format = '420p10' | '420p8'."""
    vs_format = {
        "420p10": "vs.YUV420P10",
        "420p8":  "vs.YUV420P8",
    }[out_format]

    vpy = workdir / "reference.vpy"
    vpy.write_text(f"""\
import vapoursynth as vs
import havsfunc as haf
core = vs.core

src = core.lsmas.LWLibavSource(r"{src}")
src = src[:{nframes}]
# SMDegrain accepts 8/16-bit YUV420; ProRes decodes to YUV422P16 via lsmas.
# Convert to the target planar 4:2:0 format matching NVEncC's internal smdegrain CSP
# so the ref and candidate frames line up byte-for-byte in shape.
clip = core.resize.Bicubic(src, format={vs_format})
clip = haf.SMDegrain(
    clip,
    tr={preset["tr"]},
    thSAD={preset["thSAD"]},
    limit={preset["limit"]},
    contrasharp={preset["contrasharp"]},
    thSCD1={preset["thSCD1"]},
    search={preset["search"]},
)
clip.set_output()
""", encoding="utf-8")

    out = workdir / "reference.y4m"
    print(f"  [ref] vspipe haf.SMDegrain -> {out.name}")
    r = subprocess.run([VSPIPE, "-c", "y4m", str(vpy), str(out)], capture_output=True, text=True)
    if r.returncode != 0:
        print("vspipe stderr:", r.stderr[-4000:])
        raise RuntimeError("vspipe reference failed")
    return out


# ------------------------------- Candidate: --vpp-smdegrain -------------------------------

def render_candidate_y4m(src: str, preset: dict, nframes: int, bit_depth: int, workdir: Path) -> Path:
    """NVEncC --vpp-smdegrain --lossless -> decode -> y4m at the target bit depth."""
    smd_args = ",".join([
        "enable=true",
        f"tr={preset['tr']}",
        f"thSAD={preset['thSAD']}",
        f"limit={preset['limit']}",
        f"contrasharp={'true' if preset['contrasharp'] else 'false'}",
        f"thscd1={preset['thSCD1']}",
        f"search={preset['search']}",
    ])

    if bit_depth == 10:
        codec_args = ["--codec", "hevc", "--profile", "main10", "--output-depth", "10", "--lossless"]
        pix_fmt = "yuv420p10le"
    elif bit_depth == 8:
        codec_args = ["--codec", "hevc", "--profile", "main", "--lossless"]
        pix_fmt = "yuv420p"
    else:
        raise RuntimeError(f"unsupported bit_depth {bit_depth}")

    hevc = workdir / "candidate.hevc"
    print(f"  [nvencc] --vpp-smdegrain {smd_args} --lossless -> {hevc.name}")
    cmd = [NVENCC, "--avsw", "-i", src, "--trim", f"0:{nframes - 1}",
           "--vpp-smdegrain", smd_args, *codec_args, "-o", str(hevc)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("nvencc stderr:", r.stderr[-4000:])
        raise RuntimeError("nvencc candidate failed")

    out = workdir / "candidate.y4m"
    print(f"  [ffmpeg] decode -> {out.name}")
    cmd = [FFMPEG, "-y", "-loglevel", "warning", "-i", str(hevc),
           "-pix_fmt", pix_fmt, "-strict", "-1", "-f", "yuv4mpegpipe", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print("ffmpeg stderr:", r.stderr)
        raise RuntimeError("ffmpeg decode failed")
    return out


# ------------------------------- Nofilter source -> y4m (for VMAF vs source) ----------

def render_source_y4m(src: str, nframes: int, bit_depth: int, workdir: Path) -> Path:
    """Decode source to y4m at the same bit depth / 4:2:0 format we're evaluating,
    using vspipe so the format conversion matches the reference pipeline exactly."""
    vs_format = "vs.YUV420P10" if bit_depth == 10 else "vs.YUV420P8"
    vpy = workdir / "source.vpy"
    vpy.write_text(f"""\
import vapoursynth as vs
core = vs.core
src = core.lsmas.LWLibavSource(r"{src}")[:{nframes}]
clip = core.resize.Bicubic(src, format={vs_format})
clip.set_output()
""", encoding="utf-8")
    out = workdir / "source.y4m"
    print(f"  [source] vspipe decode -> {out.name}")
    r = subprocess.run([VSPIPE, "-c", "y4m", str(vpy), str(out)], capture_output=True, text=True)
    if r.returncode != 0:
        print("vspipe stderr:", r.stderr[-4000:])
        raise RuntimeError("vspipe source decode failed")
    return out


# ------------------------------- VMAF -----------------------------------------

def vmaf(reference_y4m: Path, distorted_y4m: Path) -> dict:
    """Run ffmpeg libvmaf filter. reference is the 'ground truth' side; distorted is the side under test.
    In ffmpeg libvmaf semantics the filter takes 'main' (distorted) and 'reference' inputs in that order."""
    # Single-pass vmaf via ffmpeg libvmaf. Windows colons in log_path break the
    # filtergraph parser; sidestep it by cd'ing into distorted_y4m's dir and using
    # a bare filename — no colon in the argument.
    workdir = distorted_y4m.parent
    vmaf_json_name = distorted_y4m.stem + ".vmaf.json"
    cmd = [
        FFMPEG, "-hide_banner", "-y",
        "-i", distorted_y4m.name,
        "-i", reference_y4m.name,
        "-lavfi", f"[0:v][1:v]libvmaf=log_path={vmaf_json_name}:log_fmt=json:n_threads=4",
        "-f", "null", "-",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(workdir))
    vmaf_json = workdir / vmaf_json_name
    if r.returncode != 0:
        print("vmaf stderr tail:", r.stderr[-2000:])
        raise RuntimeError("libvmaf failed")
    with open(vmaf_json) as f:
        data = json.load(f)
    pooled = data.get("pooled_metrics", {}).get("vmaf", {})
    return {
        "mean":  pooled.get("mean"),
        "min":   pooled.get("min"),
        "max":   pooled.get("max"),
        "harmonic_mean": pooled.get("harmonic_mean"),
    }


# ------------------------------- Compare --------------------------------------

def pixel_diff(ref_y4m: Path, cand_y4m: Path) -> dict:
    ref_frames, ref_info = parse_y4m_luma(ref_y4m)
    cand_frames, cand_info = parse_y4m_luma(cand_y4m)
    n = min(len(ref_frames), len(cand_frames))
    if n == 0:
        raise RuntimeError("no frames decoded")

    means, maxes = [], []
    for i in range(n):
        r = ref_frames[i].astype(np.int32)
        c = cand_frames[i].astype(np.int32)
        if r.shape != c.shape:
            raise RuntimeError(f"frame {i} shape mismatch: ref {r.shape} vs cand {c.shape}")
        d = np.abs(r - c)
        means.append(float(d.mean()))
        maxes.append(int(d.max()))
    return {
        "frames":            n,
        "mean_diff_overall": float(np.mean(means)),
        "mean_diff_worst":   float(np.max(means)),
        "max_diff_overall":  int(np.max(maxes)),
        "ref_shape":         ref_frames[0].shape,
        "ref_cs":            ref_info["cs"],
        "cand_cs":           cand_info["cs"],
    }


# ------------------------------- Main -----------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", choices=list(FIXTURES), required=True)
    ap.add_argument("--preset", choices=list(PRESETS), required=True)
    ap.add_argument("--bit-depth", type=int, choices=[8, 10], default=10)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--keep", action="store_true", help="keep workdir (default discards)")
    ap.add_argument("--workdir", type=str, default=None, help="use this dir instead of tmpdir")
    ap.add_argument("--skip-vmaf", action="store_true")
    args = ap.parse_args()

    src = FIXTURES[args.camera]
    preset = PRESETS[args.preset]
    out_format = "420p10" if args.bit_depth == 10 else "420p8"
    lsb_range = (1 << args.bit_depth) - 1

    print(f"=== smdegrain parity: camera={args.camera} preset={args.preset} bit_depth={args.bit_depth} frames={args.frames} ===")
    print(f"source: {src}")
    print(f"preset: {preset}")

    if args.workdir:
        workdir = Path(args.workdir)
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        workdir = Path(tempfile.mkdtemp(prefix="smdegrain_validate_"))
    print(f"workdir: {workdir}")

    try:
        ref = render_reference_y4m(src, preset, args.frames, out_format, workdir)
        cand = render_candidate_y4m(src, preset, args.frames, args.bit_depth, workdir)

        px = pixel_diff(ref, cand)
        print()
        print("--- pixel diff (Y plane) ---")
        print(f"frames:            {px['frames']}")
        print(f"mean |ref-cand|:   {px['mean_diff_overall']:.3f} LSB  (of 0..{lsb_range})")
        print(f"mean worst frame:  {px['mean_diff_worst']:.3f} LSB")
        print(f"max  |ref-cand|:   {px['max_diff_overall']} LSB")
        print(f"ref cs / cand cs:  {px['ref_cs']} / {px['cand_cs']}")

        if not args.skip_vmaf:
            src_y4m = render_source_y4m(src, args.frames, args.bit_depth, workdir)
            print()
            print("--- VMAF (netflix perceptual metric) ---")
            v_parity = vmaf(reference_y4m=ref,     distorted_y4m=cand)    # GPU vs CPU
            v_gpu    = vmaf(reference_y4m=src_y4m, distorted_y4m=cand)    # GPU vs source
            v_cpu    = vmaf(reference_y4m=src_y4m, distorted_y4m=ref)     # CPU vs source
            print(f"VMAF(GPU cand vs CPU ref) [parity]: {v_parity['mean']:.3f}   min={v_parity['min']:.2f}")
            print(f"VMAF(GPU cand vs source)  [denoise drift]: {v_gpu['mean']:.3f}   min={v_gpu['min']:.2f}")
            print(f"VMAF(CPU ref  vs source)  [denoise drift]: {v_cpu['mean']:.3f}   min={v_cpu['min']:.2f}")
            print(f"gap(GPU-CPU vs source):   {v_gpu['mean'] - v_cpu['mean']:+.3f}  (close to 0 = GPU denoises about as much as CPU)")

    finally:
        if args.keep or args.workdir:
            print(f"(kept workdir: {workdir})")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
