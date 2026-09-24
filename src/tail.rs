//! tailM decode tail on the CPU, fused into `NativeEdgeRetriever.retrieve_tail`'s per-KV-head task.
//!
//! The Rust port of `kv_search.tailm_runtime.TailmRuntime._tail` with the `kv_search.tailm` finish
//! helpers (`standin_finish`, `gaussian_tail_mass_finish`, `merge_lse`), all f32, for ONE KV head:
//!
//!   r  = x @ packed,   packed = [Wᵀ | scaling²·Σ | scaling·μ]  (d × (2d+1), bf16, f32 accumulation)
//!   û  = unit(r[:d] / max(‖x‖, 1e-30) + bias) · scale
//!   σ² = Σ r[d:2d]·x,  μ_s = r[2d],  σ = sqrt(max(σ², 0))
//!   ln Ẑ = ln α + [ln n + μ_s + σ²/2 + (σ > 0 ? ln Φ((b − μ_s − σ²)/σ) : 0)]   (non-finite → −inf;
//!          n = 0 → −inf; α = 0 → ln α = −inf)
//!   lse' = logaddexp(lse, ln Ẑ),  out' = e^(lse − lse')·out + e^(ln Ẑ − lse')·û
//!
//! A row whose ln Ẑ is −inf (α = 0, n = 0, non-finite) keeps its kept-keys out / lse bit for bit.
//!
//! Storage (benchmark `tailm-rust-bench/src/bin/bf16.rs`, variant BF16-all): the 2d+1 columns are
//! padded to a multiple of 16 and cut into 16-column tiles; a tile is d/2 lines of 64 bytes, line j
//! holding rows 2j and 2j+1 (2 × 16 bf16), every line 64-byte aligned. bf16 → f32 by `(u as u32) << 16`.
//! Matmul: 4 rows × 16 columns in registers per tile, tile outer, software prefetch 24 lines ahead.

/// Columns per tile (two AVX2 vectors of f32).
const W: usize = 16;
/// Software prefetch distance, in 64-byte lines.
const PF: usize = 24;

/// One 64-byte line: 2 rows × 16 bf16 of one tile.
#[repr(C, align(64))]
#[derive(Clone, Copy)]
struct Line64([u16; 2 * W]);

/// The matmul / finish implementation, picked once per retriever (`Kernel::detect`). Only
/// `detect` / `from_name` construct one (the field is private), so an AVX2 kernel exists only after
/// the CPU reported AVX2 + FMA: that is what makes the `unsafe` call in `head_tail` sound.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Kernel(Kind);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    /// x86_64, `#[target_feature(enable = "avx2,fma")]`: FMA accumulation, software prefetch.
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// aarch64: NEON and FMA are baseline, so the FMA variant (`mul_add` → `fmla`) needs no
    /// runtime check; no software prefetch.
    #[cfg(target_arch = "aarch64")]
    Neon,
    /// Any CPU: the same loops with a plain multiply + add (no FMA).
    Portable,
}

impl Kernel {
    pub fn detect() -> Self {
        #[cfg(target_arch = "aarch64")]
        {
            Kernel(Kind::Neon)
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("avx2")
                && std::arch::is_x86_feature_detected!("fma")
            {
                return Kernel(Kind::Avx2);
            }
            Kernel(Kind::Portable)
        }
    }

    pub fn name(self) -> &'static str {
        match self.0 {
            #[cfg(target_arch = "x86_64")]
            Kind::Avx2 => "avx2",
            #[cfg(target_arch = "aarch64")]
            Kind::Neon => "neon",
            Kind::Portable => "portable",
        }
    }

    /// `name`'s kernel (case-insensitive), if this CPU can run it.
    pub fn from_name(name: &str) -> Option<Self> {
        let name = name.to_ascii_lowercase();
        if name == "portable" {
            return Some(Kernel(Kind::Portable));
        }
        let k = Self::detect();
        (k.0 != Kind::Portable && k.name() == name).then_some(k)
    }
}

/// One (layer, KV head)'s tail state.
pub struct HeadTail {
    d: usize,
    nt: usize,          // column tiles, ceil((2d+1)/16)
    lines: Vec<Line64>, // [tile][d/2 lines]
    bias: Vec<f32>,     // [d]
    scale: f32,
    log_alpha: f32, // −inf for α = 0
    ln_n_keys: f32, // f32(ln n_keys), as torch rounds math.log(n_keys) into an f32 op; −inf for 0
}

impl HeadTail {
    /// `packed`: row-major [d, 2d+1] bf16 bits; `d` even.
    pub fn new(
        d: usize,
        packed: &[u16],
        bias: &[f32],
        scale: f32,
        log_alpha: f32,
        n_keys: u64,
    ) -> Self {
        let c = 2 * d + 1;
        assert!(d.is_multiple_of(2) && packed.len() == d * c && bias.len() == d);
        let nt = c.div_ceil(W);
        let per_tile = d / 2;
        let mut lines = vec![Line64([0; 2 * W]); nt * per_tile];
        for t in 0..nt {
            for jl in 0..per_tile {
                let line = &mut lines[t * per_tile + jl].0;
                for rr in 0..2 {
                    let row = &packed[(2 * jl + rr) * c..][..c];
                    for j in 0..W {
                        let col = t * W + j;
                        line[rr * W + j] = if col < c { row[col] } else { 0 };
                    }
                }
            }
        }
        HeadTail {
            d,
            nt,
            lines,
            bias: bias.to_vec(),
            scale,
            log_alpha,
            ln_n_keys: (n_keys as f64).ln() as f32,
        }
    }

    pub fn dim(&self) -> usize {
        self.d
    }
}

/// The tail of one KV head's P rows (P a multiple of 4): `x` raw q [P, d], `out` [P, d], `lse` /
/// `bnd` [P] (kept keys) → `out_o` [P, d], `lse_o` [P].
///
/// The micro-kernel takes 4 rows at a time: one KV head's 4 q-heads (Qwen3.5-9B, the only model
/// `--tailm` admits) times q_len rows. Another GQA group size needs P padded to a multiple of 4.
#[allow(clippy::too_many_arguments)]
pub fn head_tail(
    kernel: Kernel,
    st: &HeadTail,
    x: &[f32],
    out: &[f32],
    lse: &[f32],
    bnd: &[f32],
    out_o: &mut [f32],
    lse_o: &mut [f32],
) {
    let (d, p) = (st.d, lse.len());
    assert!(p.is_multiple_of(4) && x.len() == p * d && out.len() == p * d && bnd.len() == p);
    assert!(out_o.len() == p * d && lse_o.len() == p);
    match kernel.0 {
        // SAFETY: a Kind::Avx2 kernel is only built by Kernel::detect after the CPU reported AVX2 + FMA
        #[cfg(target_arch = "x86_64")]
        Kind::Avx2 => unsafe { head_tail_avx2(st, x, out, lse, bnd, out_o, lse_o) },
        #[cfg(target_arch = "aarch64")]
        Kind::Neon => head_tail_impl::<true>(st, x, out, lse, bnd, out_o, lse_o),
        Kind::Portable => head_tail_impl::<false>(st, x, out, lse, bnd, out_o, lse_o),
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn head_tail_avx2(
    st: &HeadTail,
    x: &[f32],
    out: &[f32],
    lse: &[f32],
    bnd: &[f32],
    out_o: &mut [f32],
    lse_o: &mut [f32],
) {
    head_tail_impl::<true>(st, x, out, lse, bnd, out_o, lse_o)
}

#[inline(always)]
fn head_tail_impl<const FMA: bool>(
    st: &HeadTail,
    x: &[f32],
    out: &[f32],
    lse: &[f32],
    bnd: &[f32],
    out_o: &mut [f32],
    lse_o: &mut [f32],
) {
    let (d, p) = (st.d, lse.len());
    let s = st.nt * W;
    let mut r = vec![0f32; p * s];
    gemm::<FMA>(st, x, &mut r);
    let mut v = vec![0f32; d];
    for row in 0..p {
        finish_row::<FMA>(
            st,
            &x[row * d..][..d],
            &r[row * s..][..s],
            bnd[row],
            &out[row * d..][..d],
            lse[row],
            &mut v,
            &mut out_o[row * d..][..d],
            &mut lse_o[row],
        );
    }
}

#[inline(always)]
fn fmadd<const FMA: bool>(a: f32, b: f32, c: f32) -> f32 {
    // Rust never contracts a*b+c; mul_add is one FMA instruction only inside the avx2,fma kernel
    // or on aarch64 (fmla); elsewhere it would be a slow libm call, so the portable kernel
    // multiplies and adds
    if FMA { a.mul_add(b, c) } else { a * b + c }
}

#[inline(always)]
fn prefetch(p: *const Line64) {
    // SAFETY: SSE is baseline on x86_64, and a prefetch is only a hint: it never dereferences `p`
    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch::<{ std::arch::x86_64::_MM_HINT_T0 }>(p as *const i8)
    };
    #[cfg(not(target_arch = "x86_64"))] // aarch64 (and others): no software prefetch
    let _ = p;
}

/// r [P, nt·16] = x [P, d] @ packed, tile outer so a tile's lines stay in L1 across row blocks.
#[inline(always)]
fn gemm<const FMA: bool>(st: &HeadTail, x: &[f32], r: &mut [f32]) {
    let (d, s) = (st.d, st.nt * W);
    let per_tile = d / 2;
    let p = x.len() / d;
    for t in 0..st.nt {
        let lines = &st.lines[t * per_tile..][..per_tile];
        let mut row = 0;
        while row < p {
            micro::<FMA>(
                d,
                &x[row * d..][..4 * d],
                lines,
                &mut r[row * s..][..4 * s],
                s,
                t,
            );
            row += 4;
        }
    }
}

/// 4 rows × 16 columns of one tile, 8 AVX2 accumulators held across the whole d loop.
#[inline(always)]
fn micro<const FMA: bool>(
    d: usize,
    x: &[f32],
    lines: &[Line64],
    r: &mut [f32],
    s: usize,
    tile: usize,
) {
    let (x0, x1, x2, x3) = (&x[..d], &x[d..2 * d], &x[2 * d..3 * d], &x[3 * d..4 * d]);
    let mut a = [[0f32; W]; 4];
    for (jl, line) in lines.iter().enumerate() {
        prefetch((line as *const Line64).wrapping_add(PF)); // past the end is harmless: a hint
        let u = &line.0;
        for rr in 0..2 {
            let i = 2 * jl + rr;
            let mut w = [0f32; W];
            for j in 0..W {
                w[j] = f32::from_bits((u[rr * W + j] as u32) << 16); // bf16 → f32
            }
            let xs = [x0[i], x1[i], x2[i], x3[i]];
            for k in 0..4 {
                for j in 0..W {
                    a[k][j] = fmadd::<FMA>(xs[k], w[j], a[k][j]);
                }
            }
        }
    }
    for k in 0..4 {
        r[k * s + tile * W..][..W].copy_from_slice(&a[k]);
    }
}

/// Σ a·b with 8 partial sums (vectorises without reassociation licence), then the remainder.
#[inline(always)]
fn dot<const FMA: bool>(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (ca, cb) in ca.zip(cb) {
        for j in 0..8 {
            acc[j] = fmadd::<FMA>(ca[j], cb[j], acc[j]);
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        sum = fmadd::<FMA>(*x, *y, sum);
    }
    sum
}

/// torch.logaddexp on the CPU: −inf when both are −inf, else max + log1p(exp(−|a − b|)).
#[inline(always)]
fn logaddexp(a: f32, b: f32) -> f32 {
    let m = a.max(b);
    if m == f32::NEG_INFINITY {
        return m;
    }
    m + (-(a - b).abs()).exp().ln_1p()
}

/// ln Φ(x) in f32, torch's `calc_log_ndtr` split (ATen Math.h), t = x/√2:
///   x ≥ −1: log1p(−erfc(t)/2);  x < −1, y = −t: y < 4 → ln(erfc(y)/2),
///   y ≥ 4 → −y² − ln(2√π·f), f = the Laplace continued fraction of erfc (24 terms, bottom up).
/// Benchmark port (`tailm-rust-bench/src/kernel.rs`), within 1–2 f32 ulp of torch's f32 log_ndtr.
fn log_ndtr(x: f32) -> f32 {
    const Y_CF: f32 = 4.0;
    const CF_TERMS: usize = 24;
    let t = x * std::f32::consts::FRAC_1_SQRT_2;
    if x >= -1.0 {
        return (-0.5 * libm::erfcf(t)).ln_1p();
    }
    let y = -t;
    if y < Y_CF {
        return (0.5 * libm::erfcf(y)).ln();
    }
    let mut f = y;
    for k in (1..=CF_TERMS).rev() {
        f = y + (k as f32 * 0.5) / f;
    }
    -y * y - (2.0 * std::f32::consts::PI.sqrt() * f).ln()
}

/// One row: ln Ẑ, then the stand-in û and merge_lse, in the order of `TailmRuntime._tail`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn finish_row<const FMA: bool>(
    st: &HeadTail,
    x: &[f32],
    r: &[f32],
    bnd: f32,
    out: &[f32],
    lse: f32,
    v: &mut [f32],
    out_o: &mut [f32],
    lse_o: &mut f32,
) {
    let d = st.d;
    // gaussian_tail_mass_finish (+ ln α)
    let var = dot::<FMA>(&r[d..2 * d], x); // σ_s² = scaling²·qᵀΣq
    let mu_s = r[2 * d]; // μ_s = scaling·q·μ
    let lz = if st.ln_n_keys == f32::NEG_INFINITY {
        f32::NEG_INFINITY // n_keys == 0: empty tail
    } else {
        let sigma = var.max(0.0).sqrt();
        let base = st.ln_n_keys + mu_s + 0.5 * var;
        let t = if sigma > 0.0 {
            base + log_ndtr((bnd - mu_s - var) / sigma)
        } else {
            base
        };
        let t = if t.is_finite() { t } else { f32::NEG_INFINITY };
        st.log_alpha + t
    };
    if lz == f32::NEG_INFINITY {
        // no tail mass: the kept-keys partition, bit for bit
        out_o.copy_from_slice(out);
        *lse_o = lse;
        return;
    }
    // standin_finish: û = unit(W·q̂ + bias)·scale, W·q̂ = (W·q)/‖q‖
    let nq = dot::<FMA>(x, x).sqrt().max(1e-30);
    for j in 0..d {
        v[j] = r[j] / nq + st.bias[j];
    }
    let nv = dot::<FMA>(v, v).sqrt().max(1e-30);
    // merge_lse
    let l = logaddexp(lse, lz);
    let (wa, wb) = ((lse - l).exp(), (lz - l).exp());
    for j in 0..d {
        let u = v[j] / nv * st.scale;
        out_o[j] = wa * out[j] + wb * u;
    }
    *lse_o = l;
}
