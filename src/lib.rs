use pyo3::prelude::*;

#[pymodule]
mod _native {
    use std::{collections::HashMap, path::Path};

    use numpy::{PyArray3, PyReadonlyArray3, ndarray::s};
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
        ) -> PyResult<(Bound<'py, PyArray3<f32>>, Bound<'py, PyArray3<f32>>)> {
            let arr = q.as_array();
            let q_heads = arr.shape()[0];
            let q_len = arr.shape()[1];

            let queries: Vec<(usize, Vec<f32>)> = (0..q_heads)
                .flat_map(|qh| (0..q_len).map(move |t| (qh / 4, arr.slice(s![qh, t, ..]).to_vec())))
                .collect();

            let per_query: Vec<(usize, Vec<Vec<f32>>, Vec<Vec<f32>>)> = queries
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
                            with_vector: WithVector::Bool(true),
                            with_payload: WithPayloadInterface::Bool(false),
                        })
                        .map_err(|e| e.to_string())?;

                    let mut keys: Vec<Vec<f32>> = Vec::with_capacity(points.len());
                    let mut values: Vec<Vec<f32>> = Vec::with_capacity(points.len());
                    for p in points {
                        let Some(VectorStructInternal::Named(mut named)) = p.vector else {
                            return Err("scored point has no named vectors".to_string());
                        };

                        match (named.remove("key"), named.remove("value")) {
                            (Some(VectorInternal::Dense(k)), Some(VectorInternal::Dense(v))) => {
                                keys.push(k);
                                values.push(v);
                            }
                            _ => return Err("blah".to_string()),
                        }
                    }
                    Ok((h, keys, values))
                })
                .collect::<Result<Vec<_>, String>>()
                .map_err(|e| PyRuntimeError::new_err(e))?;

            let n_kv = q_heads / 4;
            let mut keys: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_kv];
            let mut values: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_kv];

            for (kv, k, v) in per_query {
                keys[kv].extend(k);
                values[kv].extend(v);
            }

            let keys = PyArray3::from_vec3(py, &keys)?;
            let values = PyArray3::from_vec3(py, &values)?;

            Ok((keys, values))
        }
    }
}
