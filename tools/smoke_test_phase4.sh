#!/bin/bash
# Phase 4 smoke tests: flat sigma_curve should match scalar sigma=S^2 byte-identically.
# Encoder is deterministic, so SHA match proves kernel LUT lookup is correct.

set -e
BIN="D:/python_playground/NVEnc/_build/x64/RelStatic/NVEncC64.exe"
SRC="Q:/selectcode/20260416 ifh hannover/denoise real world samples xh2s/fuji_xh2s_flog2c_4k_iso3200_XH2S3466.MOV"
OUTDIR="D:/python_playground/NVEnc/_validation"
COMMON_FFT3D_ARGS="amount=1.0,block_size=32,overlap=0.5,temporal=1,method=0,prec=auto"

run_case() {
    local label="$1"
    local fft3d_args="$2"
    local out="${OUTDIR}/phase4_${label}.h264"
    echo ">>> ${label}"
    echo "    args: ${fft3d_args}"
    "$BIN" --avsw -i "$SRC" --vpp-fft3d "$fft3d_args" -c h264 -o "$out" 2>&1 | tail -2
    sha256sum "$out"
    echo ""
}

run_case "scalar_sigma25" "sigma=25,${COMMON_FFT3D_ARGS}"
run_case "lut_flat_5"     "sigma=0,${COMMON_FFT3D_ARGS},sigma_curve=0/5;1/5"

run_case "scalar_sigma1"  "sigma=1,${COMMON_FFT3D_ARGS}"
run_case "lut_flat_1"     "sigma=0,${COMMON_FFT3D_ARGS},sigma_curve=0/1;1/1"

echo "==="
echo "Flat curve at s=5 should equal scalar sigma=25 (5²=25, both produce effectiveSigma=25/255)."
echo "Flat curve at s=1 should equal scalar sigma=1."
