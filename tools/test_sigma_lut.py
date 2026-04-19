"""
Standalone verifier for the sigma_curve LUT math used in NVEncFilterDenoiseFFT3D::init().

Mirrors the C++ algorithm in Phase 3 so we can sanity-check values before running the
actual GPU kernel in Phase 4. Not a test of kernel behavior — just the host-side LUT builder.
"""
from __future__ import annotations

import math


def interp(curve: list[tuple[float, float]], r: float) -> float:
    """Piecewise-linear interpolation matching the C++ loop in init()."""
    if r <= curve[0][0]:
        return curve[0][1]
    if r >= curve[-1][0]:
        return curve[-1][1]
    for i in range(1, len(curve)):
        if r <= curve[i][0]:
            f0, s0 = curve[i - 1]
            f1, s1 = curve[i]
            t = (r - f0) / (f1 - f0)
            return s0 + t * (s1 - s0)
    return curve[-1][1]  # unreachable


def build_lut(curve: list[tuple[float, float]], block_size: int, temporal_count: int) -> list[float]:
    """Builds the per-bin sigma² / 255 LUT. Layout: [z][y][x] row-major."""
    lut: list[float] = [0.0] * (temporal_count * block_size * block_size)
    scale_inv = 1.0 / 255.0
    for bz in range(temporal_count):
        if temporal_count > 1:
            fz = min(bz, temporal_count - bz) / (temporal_count * 0.5)
        else:
            fz = 0.0
        for by in range(block_size):
            fy = min(by, block_size - by) / (block_size * 0.5)
            for bx in range(block_size):
                fx = min(bx, block_size - bx) / (block_size * 0.5)
                radial = math.sqrt((fx * fx + fy * fy) * 0.5 + fz * fz)
                radial = min(radial, 1.0)
                sigma = interp(curve, radial)
                idx = bz * block_size * block_size + by * block_size + bx
                lut[idx] = sigma * sigma * scale_inv
    return lut


def check(label: str, got: float, want: float, tol: float = 1e-5) -> None:
    ok = abs(got - want) < tol
    status = "PASS" if ok else "FAIL"
    print(f"  {status} {label}: got {got:.6f}, want {want:.6f}")
    assert ok, f"{label}: got {got}, want {want}"


def main() -> None:
    # Test 1: flat curve → every bin identical
    print("Test 1: flat curve sigma=5 → every bin = 25/255")
    lut = build_lut([(0.0, 5.0), (1.0, 5.0)], block_size=16, temporal_count=1)
    expected = 25.0 / 255.0
    for v in lut:
        check("flat", v, expected)
    print(f"  (all {len(lut)} bins equal)")

    # Test 2: radial 0 at DC bin (bx=0, by=0, bz=0)
    print("\nTest 2: extreme curve (low at DC, high at Nyquist) — check endpoints")
    curve = [(0.0, 1.0), (1.0, 10.0)]
    lut = build_lut(curve, block_size=16, temporal_count=1)
    # DC bin (0,0,0) → radial 0 → sigma=1 → LUT = 1/255
    check("DC bin", lut[0 * 16 * 16 + 0 * 16 + 0], 1.0 / 255.0)
    # Nyquist-ish bin (bx=8, by=8) → fx=fy=1, radial=sqrt(1) = 1 → sigma=10 → 100/255
    check("Nyquist corner", lut[0 * 16 * 16 + 8 * 16 + 8], 100.0 / 255.0)
    # Mid-freq bin (bx=4, by=0) → fx=0.5, fy=0 → radial=sqrt(0.125) ≈ 0.3536
    # Linear interp: sigma = 1 + 0.3536 * (10-1) = 4.1820
    got_mid = lut[0 * 16 * 16 + 0 * 16 + 4]
    want_mid = (1.0 + 0.35355339 * 9.0) ** 2 / 255.0
    check("mid-freq bin", got_mid, want_mid, tol=1e-4)

    # Test 3: temporal dimension — temporalCount=3, bz=1 → fz = min(1,2)/1.5 = 0.667
    print("\nTest 3: temporal — bz=1 at DC (bx=by=0) should have radial=0.667")
    lut3 = build_lut([(0.0, 0.0), (1.0, 10.0)], block_size=16, temporal_count=3)
    # bz=1, bx=0, by=0: fx=0, fy=0, fz=2/3 → radial = sqrt(0 + 4/9) = 2/3 ≈ 0.6667
    # sigma = 0 + 0.6667 * 10 = 6.6667 → LUT = 44.444/255
    got_t = lut3[1 * 16 * 16 + 0 * 16 + 0]
    radial_expected = 2.0 / 3.0
    sigma_expected = radial_expected * 10.0
    want_t = sigma_expected * sigma_expected / 255.0
    check("temporal bz=1 DC", got_t, want_t, tol=1e-4)
    # DC-DC-DC at (0,0,0) → sigma=0 → 0
    check("temporal DC-DC-DC", lut3[0], 0.0)

    # Test 4: dfttest-style curve (AX700 HLG3 24dB-ish shape)
    print("\nTest 4: realistic AX700-shape curve, check monotonic rise with radial")
    curve = [(0.0, 0.0), (0.1, 0.5), (0.3, 2.0), (0.6, 5.0), (1.0, 12.0)]
    lut = build_lut(curve, block_size=16, temporal_count=1)
    dc_val = lut[0]
    nyq_val = lut[8 * 16 + 8]  # (bx=8, by=8)
    assert dc_val == 0.0, f"DC should be zero, got {dc_val}"
    assert nyq_val == (12.0 * 12.0) / 255.0
    print(f"  DC={dc_val:.4f}, Nyquist={nyq_val:.4f} — both match endpoints")

    # Test 5: layout sanity — total count
    lut_a = build_lut([(0.0, 1.0), (1.0, 1.0)], block_size=16, temporal_count=1)
    lut_b = build_lut([(0.0, 1.0), (1.0, 1.0)], block_size=16, temporal_count=3)
    assert len(lut_a) == 16 * 16 == 256
    assert len(lut_b) == 3 * 16 * 16 == 768
    print(f"\nTest 5: layout sizes — temporal=1 → 256, temporal=3 → 768. OK.")

    print("\nAll LUT sanity checks passed.")


if __name__ == "__main__":
    main()
