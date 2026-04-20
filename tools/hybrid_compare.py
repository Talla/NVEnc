"""
Render two real-world clips through both hybrid denoise paths for visual comparison.

  1. CPU hybrid  — vspipe: dfttest2.Backend.CPU() + havsfunc.SMDegrain  → y4m → ffmpeg HEVC
  2. NVEncC hybrid — NVEncC --vpp-fft3d sigma_curve=... --vpp-smdegrain ...  → HEVC

Both paths use the same dfttest tier (heavy) + smdegrain tier (medium_cs) + noise profile,
so outputs are semantically comparable. Uses media_processor's noise profile JSONs as the
source of sigma_curves.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


NVENCC          = r"D:/python_playground/NVEnc-combined/_build/x64/RelStatic/NVEncC64.exe"
VSPIPE          = shutil.which("vspipe") or r"C:/Users/Eduard/AppData/Local/Programs/VapourSynth/core/vspipe.exe"
FFMPEG          = shutil.which("ffmpeg") or "ffmpeg"
PROFILES_DIR    = Path(r"D:/python_playground/media_processor/media_processor/config/noise_profiles")
OUT_DIR         = Path(r"D:/python_playground/NVEnc-combined/_validation/hybrid_comparison")

# Tier params mirroring media_processor/config/denoise.py
DFTTEST_HEAVY_SIGMA_SCALE = 2.5
DFTTEST_HEAVY_BLOCK       = 16
DFTTEST_HEAVY_OVERLAP     = 0.5
DFTTEST_HEAVY_TEMPORAL    = 1          # NVEncC temporal=1 == dfttest2 tbsize=3
DFTTEST_HEAVY_METHOD      = 0
DFTTEST_HEAVY_PREC        = "fp32"

SMDEGRAIN_MEDIUM_CS = {
    "tr": 2, "thSAD": 300, "limit": 160, "contrasharp": True,
    "thSCD1": 9999, "search": 3,
}

# Two fixtures, both high-ISO.
CLIPS = [
    {
        "name":     "xh2s_iso3200",
        "src":      r"Q:/selectcode/20260416 ifh hannover/denoise real world samples xh2s/fuji_xh2s_flog2c_4k_iso3200_XH2S3466.MOV",
        "profile":  "fujifilm_xh2s_bundled.json",
        "pp":       "com.fujifilm.f-cinegamut.f-log2",
        "gain_key": "3200",
        "frames":   60,
    },
    {
        "name":     "ax700_hlg3_24",
        "src":      r"E:/temp/video-filters-testing/source-intermediates/sony ax700/C0054.mov",
        "profile":  "sony_fdrax700_bundled.json",
        "pp":       "hlg3",
        "gain_key": "24",
        "frames":   60,
    },
]


def get_slocation(profile_file: str, pp: str, gain: str) -> list[tuple[float, float]]:
    p = json.load(open(PROFILES_DIR / profile_file))
    flat = p["picture_profiles"][pp]["noise_profiles"][gain]["slocation"]
    return list(zip(flat[::2], flat[1::2]))


def format_sigma_curve(pairs_scaled: list[tuple[float, float]]) -> str:
    return ";".join(f"{f:.6f}/{s:.6f}" for f, s in pairs_scaled)


def run(cmd, **kw):
    print("  $", " ".join(f'"{c}"' if " " in str(c) else str(c) for c in cmd))
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        print(r.stderr[-4000:])
        raise RuntimeError(f"command failed ({r.returncode}): {cmd[0]}")
    return r


def render_cpu_hybrid(clip: dict, slocation_pairs: list[tuple[float, float]], out_hevc: Path):
    """vspipe dfttest2.CPU + haf.SMDegrain → y4m → ffmpeg HEVC."""
    slocation_flat = [v for pair in slocation_pairs for v in pair]
    sosize = int(round(DFTTEST_HEAVY_BLOCK * DFTTEST_HEAVY_OVERLAP))
    tbsize = 3 if DFTTEST_HEAVY_TEMPORAL == 1 else 1

    vpy_path = OUT_DIR / f"{clip['name']}_cpu.vpy"
    vpy_path.write_text(f"""\
import vapoursynth as vs
import dfttest2
import havsfunc as haf
core = vs.core

src = core.lsmas.LWLibavSource(r"{clip['src']}")
src = src[:{clip['frames']}]
# Match NVEncC pipeline's internal CSP: convert to 10-bit 4:2:0 (same as --output-depth 10 main10).
clip = core.resize.Bicubic(src, format=vs.YUV420P10)
clip = dfttest2.DFTTest(
    clip,
    slocation={slocation_flat},
    sbsize={DFTTEST_HEAVY_BLOCK},
    sosize={sosize},
    tbsize={tbsize},
    backend=dfttest2.Backend.CPU(),
)
clip = haf.SMDegrain(
    clip,
    tr={SMDEGRAIN_MEDIUM_CS['tr']},
    thSAD={SMDEGRAIN_MEDIUM_CS['thSAD']},
    limit={SMDEGRAIN_MEDIUM_CS['limit']},
    contrasharp={SMDEGRAIN_MEDIUM_CS['contrasharp']},
    thSCD1={SMDEGRAIN_MEDIUM_CS['thSCD1']},
    search={SMDEGRAIN_MEDIUM_CS['search']},
)
clip.set_output()
""", encoding="utf-8")

    # Stream vspipe → ffmpeg via pipe — no 3+ GB intermediate y4m on disk.
    print(f"[CPU] {clip['name']}: vspipe → ffmpeg HEVC 10-bit main10 (piped)")
    import sys as _sys
    vs_cmd = [VSPIPE, "-c", "y4m", str(vpy_path), "-"]
    ff_cmd = [
        FFMPEG, "-y", "-loglevel", "warning",
        "-i", "pipe:0",
        "-c:v", "libx265", "-pix_fmt", "yuv420p10le",
        "-preset", "medium", "-crf", "20",
        str(out_hevc),
    ]
    print("  $", " ".join(vs_cmd), "|", " ".join(ff_cmd))
    vs_p = subprocess.Popen(vs_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    ff_p = subprocess.Popen(ff_cmd, stdin=vs_p.stdout, stderr=subprocess.PIPE)
    vs_p.stdout.close()   # allow vspipe to receive SIGPIPE if ffmpeg dies
    ff_err = ff_p.communicate()[1]
    vs_err = vs_p.communicate()[1]
    if vs_p.returncode != 0:
        print("vspipe stderr:", (vs_err or b"").decode(errors="replace")[-2000:])
        raise RuntimeError(f"vspipe failed (rc={vs_p.returncode})")
    if ff_p.returncode != 0:
        print("ffmpeg stderr:", (ff_err or b"").decode(errors="replace")[-2000:])
        raise RuntimeError(f"ffmpeg failed (rc={ff_p.returncode})")


def render_nvencc_hybrid(clip: dict, slocation_pairs: list[tuple[float, float]], out_hevc: Path):
    """NVEncC --vpp-fft3d sigma_curve=... --vpp-smdegrain ... → HEVC (one pass, on-GPU)."""
    scaled = [(f, s * DFTTEST_HEAVY_SIGMA_SCALE) for f, s in slocation_pairs]
    sigma_curve = format_sigma_curve(scaled)

    fft3d_args = ",".join([
        "sigma=0",
        "amount=1.0",
        f"block_size={DFTTEST_HEAVY_BLOCK}",
        f"overlap={DFTTEST_HEAVY_OVERLAP}",
        f"temporal={DFTTEST_HEAVY_TEMPORAL}",
        f"method={DFTTEST_HEAVY_METHOD}",
        f"prec={DFTTEST_HEAVY_PREC}",
        f"sigma_curve={sigma_curve}",
    ])
    smd_args = ",".join([
        "enable=true",
        f"tr={SMDEGRAIN_MEDIUM_CS['tr']}",
        f"thSAD={SMDEGRAIN_MEDIUM_CS['thSAD']}",
        f"limit={SMDEGRAIN_MEDIUM_CS['limit']}",
        f"contrasharp={'true' if SMDEGRAIN_MEDIUM_CS['contrasharp'] else 'false'}",
        f"thscd1={SMDEGRAIN_MEDIUM_CS['thSCD1']}",
        f"search={SMDEGRAIN_MEDIUM_CS['search']}",
    ])

    print(f"[NVEncC] {clip['name']}: --vpp-fft3d sigma_curve + --vpp-smdegrain, all-GPU")
    run([
        NVENCC, "--avsw", "-i", clip["src"],
        "--trim", f"0:{clip['frames'] - 1}",
        "--vpp-fft3d", fft3d_args,
        "--vpp-smdegrain", smd_args,
        "-c", "hevc", "--profile", "main10", "--output-depth", "10",
        "-o", str(out_hevc),
    ])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for clip in CLIPS:
        print(f"=== {clip['name']} ({clip['frames']} frames) ===")
        pairs = get_slocation(clip["profile"], clip["pp"], clip["gain_key"])
        cpu_out = OUT_DIR / f"{clip['name']}_hybrid_cpu.hevc"
        gpu_out = OUT_DIR / f"{clip['name']}_hybrid_nvencc.hevc"
        render_cpu_hybrid(clip, pairs, cpu_out)
        render_nvencc_hybrid(clip, pairs, gpu_out)
        print(f"    CPU hybrid: {cpu_out} ({cpu_out.stat().st_size // 1024} KB)")
        print(f"    NVEncC hybrid: {gpu_out} ({gpu_out.stat().st_size // 1024} KB)")
        print()


if __name__ == "__main__":
    main()
