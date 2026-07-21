use pyo3::pymodule;

#[pymodule]
mod _native {
    use std::{collections::HashMap, path::Path};

    use numpy::{PyArray2, PyArray3, PyReadonlyArray3, PyReadonlyArrayDyn, ndarray::s};
    use pyo3::{Bound, PyResult, Python, exceptions::PyRuntimeError, pyclass, pymethods};
    use qdrant_edge::{
        EdgeShard, NamedQuery, OperationError, QueryEnum, QueryRequest, ScoredPoint, ScoringQuery,
        VectorInternal, VectorStructInternal, WithPayloadInterface, WithVector,
    };

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
            // [4, 4 * q_len * lim, 256]
            let arr = q.as_array();
            let q_heads = arr.shape()[0];
            let q_len = arr.shape()[1];

            let queries: Vec<(usize, Vec<f32>)> = (0..q_heads)
                .flat_map(|qh| (0..q_len).map(move |t| (qh / 4, arr.slice(s![qh, t, ..]).to_vec())))
                .collect();

            let points: Vec<(usize, Vec<ScoredPoint>)> = queries
                .into_iter()
                .map(|(h, qv)| {
                    Ok((
                        h,
                        (&self.shards[&(layer_idx, h)]).query(QueryRequest {
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
                        })?,
                    ))
                })
                .collect::<Result<Vec<_>, OperationError>>()
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            let n_kv = q_heads / 4;
            let mut keys: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_kv];
            let mut values: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_kv];

            for (kv, ps) in points {
                for p in ps {
                    let Some(VectorStructInternal::Named(mut named)) = p.vector else {
                        return Err(PyRuntimeError::new_err("scored point has no named vectors"));
                    };

                    match (named.remove("key"), named.remove("value")) {
                        (Some(VectorInternal::Dense(k)), Some(VectorInternal::Dense(v))) => {
                            keys[kv].push(k);
                            values[kv].push(v);
                        }
                        _ => return Err(PyRuntimeError::new_err("blah")),
                    }
                }
            }

            let keys = PyArray3::from_vec3(py, &keys)?;
            let values = PyArray3::from_vec3(py, &values)?;

            Ok((keys, values))
        }
    }
}
