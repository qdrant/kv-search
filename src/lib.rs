use pyo3::prelude::*;

#[pymodule]
mod _native {
    use std::{collections::HashMap, path::Path};

    use numpy::{PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray3, ndarray::s};
    use pyo3::exceptions::PyRuntimeError;
    use pyo3::prelude::*;
    use qdrant_edge::{
        EdgeShard, NamedQuery, QueryEnum, QueryRequest, ScoringQuery, VectorInternal,
        VectorStructInternal, WithPayloadInterface, WithVector,
    };
    use rayon::prelude::*;

    #[pyclass]
    struct NativeEdgeRetriever {
        shards: HashMap<(usize, usize), EdgeShard>,
    }

    #[pymethods]
    impl NativeEdgeRetriever {
        #[new]
        fn new(shards: Vec<((usize, usize), String)>) -> PyResult<Self> {
            let shards = shards
                .into_iter()
                .map(|(k, p)| {
                    let v = EdgeShard::load(Path::new(&p), None)
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                    Ok((k, v))
                })
                .collect::<PyResult<HashMap<_, _>>>()?;

            Ok(Self { shards })
        }

        fn retrieve<'py>(
            &self,
            py: Python<'py>,
            layer_idx: usize,
            q: PyReadonlyArray3<'_, f32>, // [16, q_len, 256]
            limit: usize,
            scaling: f32,
        ) -> PyResult<(Bound<'py, PyArray3<f32>>, Bound<'py, PyArray2<f32>>)> {
            let arr = q.as_array();
            let q_heads = arr.shape()[0];
            let q_len = arr.shape()[1];

            let queries: Vec<(usize, Vec<f32>)> = (0..q_heads)
                .flat_map(|qh| (0..q_len).map(move |t| (qh / 4, arr.slice(s![qh, t, ..]).to_vec())))
                .collect();

            // per query head and query token
            let results: Vec<(Vec<f32>, f32)> = queries
                .into_par_iter()
                .map(|(h, qv)| {
                    let points = self.shards[&(layer_idx, h)]
                        .query(QueryRequest {
                            prefetches: vec![],
                            query: Some(ScoringQuery::Vector(QueryEnum::Nearest(NamedQuery {
                                query: qv.into(),
                                using: Some("key".to_string()),
                            }))),
                            filter: None,
                            score_threshold: None,
                            limit: limit,
                            offset: 0,
                            params: None,
                            with_vector: WithVector::Selector(vec!["value".to_string()]),
                            with_payload: WithPayloadInterface::Bool(false),
                        })
                        .map_err(|e| e.to_string())?;

                    if points.is_empty() {
                        return Ok((Vec::new(), f32::NEG_INFINITY));
                    }

                    // logit_i = score_i * scaling
                    // m = max(logit_i for all i)
                    // w_i = exp(logit_i - m)
                    // lse = m + ln(sum(w_i))
                    // out = sum(w_i * v_i) / sum(w_i)
                    let m = points
                        .iter()
                        .map(|p| p.score * scaling)
                        .fold(f32::NEG_INFINITY, f32::max);

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
                    Ok((out, m + sum.ln()))
                })
                .collect::<Result<Vec<_>, String>>()
                .map_err(|e| PyRuntimeError::new_err(e))?;

            let value_dim = results.iter().map(|(o, _)| o.len()).max().unwrap_or(0);
            let mut out_flat: Vec<f32> = Vec::with_capacity(q_heads * q_len * value_dim);
            let mut lse_flat: Vec<f32> = Vec::with_capacity(q_heads * q_len);

            for (out, lse) in results {
                out_flat.extend_from_slice(&out);
                lse_flat.push(lse);
            }

            let out = PyArray1::from_vec(py, out_flat).reshape([q_heads, q_len, value_dim])?;
            let lse = PyArray1::from_vec(py, lse_flat).reshape([q_heads, q_len])?;

            Ok((out, lse))
        }
    }
}
