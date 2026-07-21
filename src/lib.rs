use pyo3::pymodule;

#[pymodule]
mod _native {
    use std::{collections::HashMap, path::Path};

    use numpy::{PyArray2, PyReadonlyArrayDyn};
    use pyo3::{
        Bound, PyResult, Python, exceptions::PyRuntimeError, pyclass, pymethods,
    };
    use qdrant_edge::{
        EdgeShard, NamedQuery, QueryEnum, QueryRequest, ScoringQuery, VectorInternal,
        VectorStructInternal, WithPayloadInterface, WithVector,
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
            head_idx: usize,
            q: PyReadonlyArrayDyn<'_, f32>,
            limit: usize,
        ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
            let shard = &self.shards[&(layer_idx, head_idx)];
            let points = shard
                .query(QueryRequest {
                    prefetches: vec![],
                    query: Some(ScoringQuery::Vector(QueryEnum::Nearest(NamedQuery {
                        query: q.as_array().iter().copied().collect::<Vec<f32>>().into(),
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
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

            let mut keys: Vec<Vec<f32>> = Vec::with_capacity(points.len());
            let mut values: Vec<Vec<f32>> = Vec::with_capacity(points.len());
            for p in points {
                let Some(VectorStructInternal::Named(mut named)) = p.vector else {
                    return Err(PyRuntimeError::new_err("scored point has no named vectors"));
                };

                match (named.remove("key"), named.remove("value")) {
                    (Some(VectorInternal::Dense(k)), Some(VectorInternal::Dense(v))) => {
                        keys.push(k);
                        values.push(v);
                    }
                    _ => return Err(PyRuntimeError::new_err("blah")),
                }
            }

            let keys = PyArray2::from_vec2(py, &keys)?;
            let values = PyArray2::from_vec2(py, &values)?;

            Ok((keys, values))
        }
    }
}
