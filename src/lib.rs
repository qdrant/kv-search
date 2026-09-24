use pyo3::prelude::*;

mod tail;

#[pymodule]
mod _native {
    use std::{collections::HashMap, path::Path};

    use numpy::{
        PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
        PyReadonlyArray3, PyUntypedArrayMethods, ndarray::s,
    };
    use pyo3::exceptions::{PyRuntimeError, PyValueError};
    use pyo3::prelude::*;
    use qdrant_edge::{
        EdgeConfigBuilder, EdgeShard, NamedQuery, QueryEnum, QueryRequest, ScoringQuery,
        SearchParams, VectorInternal, VectorStructInternal, WithPayloadInterface, WithVector,
    };
    use rayon::prelude::*;

    use crate::tail::{HeadTail, Kernel, head_tail};

    /// per query row of one KV head: (out, lse, boundary)
    type Rows = Vec<(Vec<f32>, f32, f32)>;
    /// one KV head's rows, plus its tail (out, lse) when it has tail state
    type HeadRows = (Rows, Option<(Vec<f32>, Vec<f32>)>);
    /// (out [q_heads, q_len, d], lse [q_heads, q_len], boundary [q_heads, q_len])
    type Partition<'py> = (
        Bound<'py, PyArray3<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
    );
    /// `Partition` plus (tail_out [q_heads, q_len, d], tail_lse [q_heads, q_len])
    type TailPartition<'py> = (
        Bound<'py, PyArray3<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray3<f32>>,
        Bound<'py, PyArray2<f32>>,
    );

    /// The decode-tail kernel this CPU selects (`"avx2"`, `"neon"` or `"portable"`).
    #[pyfunction]
    fn tailm_kernel() -> &'static str {
        Kernel::detect().name()
    }

    #[pyclass]
    struct NativeEdgeRetriever {
        shards: HashMap<(usize, usize), EdgeShard>,
        // true: exact top-n (full scan); false: HNSW search with `hnsw_ef` (raised to the limit)
        exact: bool,
        hnsw_ef: usize,
        // tailM decode tail per (layer, KV head), registered by set_tailm
        tails: HashMap<(usize, usize), HeadTail>,
        kernel: Kernel,
    }

    impl NativeEdgeRetriever {
        /// Each KV head's query rows, q-head major and token minor, as one contiguous [4·q_len, d]
        /// block. Groups of 4 q-heads per KV head (Qwen3.5-9B, the only model `--tailm` admits;
        /// `TailmRuntime.group` is 4 there too).
        fn queries(q: &PyReadonlyArray3<'_, f32>) -> Vec<(usize, Vec<f32>)> {
            let arr = q.as_array();
            let q_heads = arr.shape()[0];
            (0..q_heads / 4)
                .map(|h| {
                    (
                        h,
                        arr.slice(s![h * 4..h * 4 + 4, .., ..])
                            .iter()
                            .copied()
                            .collect(),
                    )
                })
                .collect()
        }

        /// Search of one KV head's query rows `x` [rows, d] (exact, or HNSW with `hnsw_ef`), then
        /// the kept keys' softmax merge.
        fn search_merge(
            &self,
            layer_idx: usize,
            h: usize,
            x: &[f32],
            d: usize,
            limit: usize,
            scaling: f32,
        ) -> Result<Rows, String> {
            let batch = self.shards[&(layer_idx, h)]
                .query_batch(
                    x.chunks_exact(d)
                        .map(|qv| QueryRequest {
                            prefetches: vec![],
                            query: Some(ScoringQuery::Vector(QueryEnum::Nearest(NamedQuery {
                                query: qv.to_vec().into(),
                                using: Some("key".to_string()),
                            }))),
                            filter: None,
                            score_threshold: None,
                            limit,
                            offset: 0,
                            params: Some(SearchParams {
                                exact: self.exact,
                                hnsw_ef: (!self.exact).then_some(self.hnsw_ef),
                                ..Default::default()
                            }),
                            with_vector: WithVector::Selector(vec!["value".to_string()]),
                            with_payload: WithPayloadInterface::Bool(false),
                        })
                        .collect(),
                )
                .map_err(|e| e.to_string())?;

            // logit_i = score_i * scaling
            // m = max(logit_i for all i)
            // w_i = exp(logit_i - m)
            // lse = m + ln(sum(w_i))
            // out = sum(w_i * v_i) / sum(w_i)
            batch
                .into_iter()
                .map(|points| {
                    let m = points
                        .iter()
                        .map(|p| p.score * scaling)
                        .fold(f32::NEG_INFINITY, f32::max);
                    // weakest kept logit: where tailM truncates its Gaussian tail mass. Under HNSW
                    // it is the weakest *retrieved* logit, not the exact top-n boundary the
                    // tailM gate was measured on
                    let boundary = points
                        .iter()
                        .map(|p| p.score * scaling)
                        .fold(f32::INFINITY, f32::min);

                    let mut out: Vec<f32> = Vec::new();
                    let mut sum = 0.0f32;
                    for p in points {
                        let Some(VectorStructInternal::Named(mut named)) = p.vector else {
                            return Err("scored point has no named vectors".to_string());
                        };

                        let Some(VectorInternal::Dense(v)) = named.remove("value") else {
                            return Err("no vector named 'value'".to_string());
                        };
                        if out.is_empty() {
                            out = vec![0.0; v.len()];
                        }
                        let w = (p.score * scaling - m).exp();
                        sum += w;
                        for (o, x) in out.iter_mut().zip(v.iter()) {
                            *o += w * x;
                        }
                    }
                    let inv = 1.0 / sum;
                    for o in out.iter_mut() {
                        *o *= inv;
                    }
                    Ok((out, m + sum.ln(), boundary))
                })
                .collect::<Result<Vec<_>, String>>()
        }

        /// Rows of all KV heads (q-head major, token minor) as (out, lse, boundary) arrays.
        fn arrays<'py>(
            py: Python<'py>,
            results: &[Rows],
            q_heads: usize,
            q_len: usize,
        ) -> PyResult<Partition<'py>> {
            let value_dim = results
                .iter()
                .flatten()
                .map(|(o, _, _)| o.len())
                .max()
                .unwrap_or(0);
            let mut out_flat: Vec<f32> = Vec::with_capacity(q_heads * q_len * value_dim);
            let mut lse_flat: Vec<f32> = Vec::with_capacity(q_heads * q_len);
            let mut boundary_flat: Vec<f32> = Vec::with_capacity(q_heads * q_len);

            for (out, lse, boundary) in results.iter().flatten() {
                out_flat.extend_from_slice(out);
                lse_flat.push(*lse);
                boundary_flat.push(*boundary);
            }

            let out = PyArray1::from_vec(py, out_flat).reshape([q_heads, q_len, value_dim])?;
            let lse = PyArray1::from_vec(py, lse_flat).reshape([q_heads, q_len])?;
            let boundary = PyArray1::from_vec(py, boundary_flat).reshape([q_heads, q_len])?;
            Ok((out, lse, boundary))
        }
    }

    #[pymethods]
    impl NativeEdgeRetriever {
        #[new]
        #[pyo3(signature = (shards, exact = true, hnsw_ef = 128))]
        fn new(
            shards: Vec<((usize, usize), String)>,
            exact: bool,
            hnsw_ef: usize,
        ) -> PyResult<Self> {
            let shards = shards
                .into_iter()
                .map(|(k, p)| {
                    // somewhat empirical, also might need to go back to sequential for hnsw
                    let v = EdgeShard::load(
                        Path::new(&p),
                        Some(EdgeConfigBuilder::new().max_search_threads(8).build()),
                    )
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    Ok((k, v))
                })
                .collect::<PyResult<HashMap<_, _>>>()?;

            Ok(Self {
                shards,
                exact,
                hnsw_ef,
                tails: HashMap::new(),
                kernel: Kernel::detect(),
            })
        }

        /// `"exact"` or `"hnsw ef N"`.
        #[getter]
        fn search_mode(&self) -> String {
            if self.exact {
                "exact".to_string()
            } else {
                format!("hnsw ef {}", self.hnsw_ef)
            }
        }

        fn retrieve<'py>(
            &self,
            py: Python<'py>,
            layer_idx: usize,
            q: PyReadonlyArray3<'_, f32>, // [16, q_len, 256]
            limit: usize,
            scaling: f32,
        ) -> PyResult<Partition<'py>> {
            let (q_heads, q_len, d) = (q.shape()[0], q.shape()[1], q.shape()[2]);
            let queries = Self::queries(&q);

            // per query head and query token: (out, lse, boundary)
            let results: Vec<Rows> = queries
                .into_par_iter()
                .map(|(h, x)| self.search_merge(layer_idx, h, &x, d, limit, scaling))
                .collect::<Result<Vec<_>, String>>()
                .map_err(PyRuntimeError::new_err)?;

            Self::arrays(py, &results, q_heads, q_len)
        }

        /// `retrieve` plus the tailM tail on the KV heads registered with `set_tailm`, computed in
        /// the same per-KV-head task right after that head's search: (out, lse, boundary,
        /// tail_out, tail_lse). The first three are `retrieve`'s, bit for bit; rows of heads
        /// without tail state (and rows without tail mass) repeat out / lse exactly.
        fn retrieve_tail<'py>(
            &self,
            py: Python<'py>,
            layer_idx: usize,
            q: PyReadonlyArray3<'_, f32>, // [16, q_len, d]
            limit: usize,
            scaling: f32,
        ) -> PyResult<TailPartition<'py>> {
            let (q_heads, q_len, d) = (q.shape()[0], q.shape()[1], q.shape()[2]);
            let queries = Self::queries(&q);
            let kernel = self.kernel;

            let per_head: Vec<HeadRows> = queries
                .into_par_iter()
                .map(|(h, x)| {
                    let Some(st) = self.tails.get(&(layer_idx, h)) else {
                        return Ok((
                            self.search_merge(layer_idx, h, &x, d, limit, scaling)?,
                            None,
                        ));
                    };
                    if d != st.dim() {
                        return Err(format!(
                            "query dim {d} != tail dim {} (layer {layer_idx}, head {h})",
                            st.dim()
                        ));
                    }
                    let rows = self.search_merge(layer_idx, h, &x, d, limit, scaling)?;
                    if rows.iter().any(|(o, _, _)| o.len() != d) {
                        return Err(format!(
                            "value dim != tail dim {d} (layer {layer_idx}, head {h})"
                        ));
                    }
                    let out: Vec<f32> = rows
                        .iter()
                        .flat_map(|(o, _, _)| o.iter().copied())
                        .collect();
                    let lse: Vec<f32> = rows.iter().map(|r| r.1).collect();
                    let bnd: Vec<f32> = rows.iter().map(|r| r.2).collect();
                    let mut t_out = vec![0f32; out.len()];
                    let mut t_lse = vec![0f32; lse.len()];
                    head_tail(kernel, st, &x, &out, &lse, &bnd, &mut t_out, &mut t_lse);
                    Ok((rows, Some((t_out, t_lse))))
                })
                .collect::<Result<Vec<_>, String>>()
                .map_err(|e| PyRuntimeError::new_err(format!("retrieve_tail: {e}")))?;

            let mut tail_out: Vec<f32> = Vec::new();
            let mut tail_lse: Vec<f32> = Vec::with_capacity(q_heads * q_len);
            for (rows, t) in &per_head {
                match t {
                    Some((o, l)) => {
                        tail_out.extend_from_slice(o);
                        tail_lse.extend_from_slice(l);
                    }
                    None => {
                        for (o, l, _) in rows {
                            tail_out.extend_from_slice(o);
                            tail_lse.push(*l);
                        }
                    }
                }
            }
            let results: Vec<Rows> = per_head.into_iter().map(|(rows, _)| rows).collect();
            let (out, lse, boundary) = Self::arrays(py, &results, q_heads, q_len)?;
            let value_dim = out.shape()[2];
            if tail_out.len() != q_heads * q_len * value_dim {
                return Err(PyRuntimeError::new_err(
                    "retrieve_tail: tail rows do not match the kept-keys rows",
                ));
            }
            let tail_out = PyArray1::from_vec(py, tail_out).reshape([q_heads, q_len, value_dim])?;
            let tail_lse = PyArray1::from_vec(py, tail_lse).reshape([q_heads, q_len])?;
            Ok((out, lse, boundary, tail_out, tail_lse))
        }

        /// Register the tail of `layer`'s KV heads `heads` (replacing the layer's earlier ones):
        /// `packed` [h, d, 2d+1] bf16 bits (= [Wᵀ | scaling²·Σ | scaling·μ]), `bias` [h, d],
        /// `scale` / `log_alpha` [h] (ln α, −inf for α = 0), `n_keys` prefill keys.
        #[allow(clippy::too_many_arguments)]
        fn set_tailm(
            &mut self,
            layer: usize,
            heads: Vec<usize>,
            packed: PyReadonlyArray3<'_, u16>,
            bias: PyReadonlyArray2<'_, f32>,
            scale: PyReadonlyArray1<'_, f32>,
            log_alpha: PyReadonlyArray1<'_, f32>,
            n_keys: u64,
        ) -> PyResult<()> {
            let bad = |m: String| Err(PyValueError::new_err(format!("set_tailm: {m}")));
            let n = heads.len();
            let ps = packed.shape();
            let d = ps[1];
            if n == 0 {
                return bad("no heads".into());
            }
            if ps[0] != n || ps[2] != 2 * d + 1 || d == 0 || !d.is_multiple_of(2) {
                return bad(format!(
                    "packed shape {ps:?}, want [{n}, d, 2d+1] with d even"
                ));
            }
            if bias.shape() != [n, d] {
                return bad(format!("bias shape {:?}, want [{n}, {d}]", bias.shape()));
            }
            if scale.shape() != [n] || log_alpha.shape() != [n] {
                return bad(format!(
                    "scale {:?} / log_alpha {:?}, want [{n}]",
                    scale.shape(),
                    log_alpha.shape()
                ));
            }
            for (i, &h) in heads.iter().enumerate() {
                if heads[..i].contains(&h) {
                    return bad(format!("head {h} twice"));
                }
                if !self.shards.contains_key(&(layer, h)) {
                    return bad(format!("no shard for layer {layer}, head {h}"));
                }
            }
            let (packed, bias) = (packed.as_array(), bias.as_array());
            let (scale, log_alpha) = (scale.as_array(), log_alpha.as_array());
            let states: Vec<HeadTail> = (0..n)
                .map(|i| {
                    let p: Vec<u16> = packed.slice(s![i, .., ..]).iter().copied().collect();
                    let b: Vec<f32> = bias.slice(s![i, ..]).iter().copied().collect();
                    HeadTail::new(d, &p, &b, scale[i], log_alpha[i], n_keys)
                })
                .collect();
            self.tails.retain(|&(l, _), _| l != layer);
            for (h, st) in heads.into_iter().zip(states) {
                self.tails.insert((layer, h), st);
            }
            Ok(())
        }

        /// Drop every registered tail.
        fn clear_tailm(&mut self) {
            self.tails.clear();
        }

        /// Use another decode-tail kernel (`"avx2"` / `"neon"` if this CPU has it, or `"portable"`); for tests
        /// and benchmarks.
        fn set_tailm_kernel(&mut self, name: &str) -> PyResult<()> {
            self.kernel = Kernel::from_name(name).ok_or_else(|| {
                PyValueError::new_err(format!("unknown or unsupported tail kernel {name:?}"))
            })?;
            Ok(())
        }
    }
}
