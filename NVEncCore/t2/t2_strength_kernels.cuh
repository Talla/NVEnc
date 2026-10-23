// MCWiener nondefault lab-c ratio path; used_by: t2_native.cu
// rules: stock J4 body except explicitly scaled merge ratio; default path unchanged.
__global__ void native_t2_main_merge_scaled(const float *ref, const float *win, const float *var, MainArgs a,
                              float noise_factor, double strength, float inv_cnt, float cnt_f, int divide,
                              const float2 *ws, int centered, float w00,
                              float2 *M, float *n, float *out, int h, int w, int oy, int ox, int nx) {
    // Opt-in K4 mode retains the exact window DC without copying a scalar to the host.
    if (centered == 2) w00 = ws[0].x;
    __shared__ float2 buf[272], tmp[272], tw[15];
    int tile = blockIdx.x, tid = threadIdx.x, i = tile * 256 + tid, idx = (tid >> 4) * 17 + (tid & 15);
    if (tid < 15) tw[tid] = T2_TW[tid];
    t2_gather_window(ref, nullptr, win, tile, tid, h, w, oy, ox, nx, buf);
    __syncthreads();
    t2_fft16_shared(buf, tmp, tw, 0, 1.0f);
    float2 r = buf[idx], acc = r;
    float nn = 1.0f;
    int fx = tid & 15, fy = tid >> 4;
    if (fx >= 8) fx -= 16;
    if (fy >= 8) fy -= 16;
    for (int k = 0; k < a.k; ++k) {
        if (k > 0) __syncthreads(); // Every thread has consumed the previous DC bin.
        t2_gather_window(a.img[k], a.shift[k], win, tile, tid, h, w, oy, ox, nx, buf);
        __syncthreads();
        t2_fft16_shared(buf, tmp, tw, 0, 1.0f);
        float2 z = buf[idx], z0 = buf[0];
        float angle = (2.0f * 3.14159265358979323846f / 16) *
                      (fx * a.frac[k][2 * tile] + fy * a.frac[k][2 * tile + 1]);
        float s, c; __sincosf(angle, &s, &c);
        if (centered) {
            // t2_coarse centered_alignment: anchor the Hann-shaped brightness, ramp the detail only.
            float cx = z0.x / w00, cy = z0.y / w00;
            float2 wb = ws[tid];
            float bx = cx * wb.x - cy * wb.y, by = cx * wb.y + cy * wb.x;
            float ex = z.x - bx, ey = z.y - by;
            z = make_float2(ex * c - ey * s + bx, ex * s + ey * c + by);
        } else {
            z = make_float2(z.x * c - z.y * s, z.x * s + z.y * c);
        }
        float dx = z.x - r.x, dy = z.y - r.y, d2 = dx * dx + dy * dy;
        // Public strength is lab c, not the fixed variance multiplier8.
        // Compute only this ratio in FP64 for nondefault values. Overflow of
        // a nonnegative noise product yields the exact limiting weight0;
        // default1 uses the original production launcher, bit-for-bit.
        double noise = (double(noise_factor) * double(var[i])) * strength;
        float al = float(double(d2) / fmax(double(d2) + noise, double(1e-30f)));
        acc = make_float2(acc.x + z.x + al * (r.x - z.x), acc.y + z.y + al * (r.y - z.y));
        nn += 1.0f - al;
    }
    float2 m = divide ? make_float2(acc.x / cnt_f, acc.y / cnt_f)
                      : make_float2(acc.x * inv_cnt, acc.y * inv_cnt);
    if (M) M[i] = m;
    n[i] = nn;
    __syncthreads(); // Preserve the shared DC bin until all readers finish.
    buf[idx] = m;
    __syncthreads();
    t2_fft16_shared(buf, tmp, tw, 1, 1.0f / 256.0f);
    out[i] = buf[idx].x;
}

