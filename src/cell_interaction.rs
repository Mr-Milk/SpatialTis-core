use std::collections::HashMap;

use counter::Counter;
use itertools::Itertools;
use ndarray::{s, ArrayView2};
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;

use crate::{mean_f, mean_u, py_kwarg, std_f, std_u, zscore2pvalue};

pub(crate) fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<CellCombs>()?;
    m.add_function(wrap_pyfunction!(comb_bootstrap, m)?)?;
    Ok(())
}

#[pyfunction]
pub fn comb_bootstrap(
    py: Python,
    exp_matrix: PyReadonlyArray2<bool>,
    markers: Vec<&str>,
    neighbors: Vec<Vec<usize>>,
    labels: Vec<usize>,
    pval: f64,
    order: bool,
    times: usize,
) -> Result<PyObject, PyErr> {
    let exp_matrix: ArrayView2<bool> = exp_matrix.as_array();
    let neighbors = remove_rep_neighbors(neighbors, &labels);
    let size = labels.len();
    let labels_mapper: HashMap<usize, usize> =
        labels.into_iter().zip(0..size).into_iter().collect();
    let mut results = vec![];
    for comb in (0..markers.len()).combinations_with_replacement(2) {
        let x_status = exp_matrix.slice(s![comb[0], ..]).to_vec();
        let y_status = exp_matrix.slice(s![comb[1], ..]).to_vec();
        // println!("{:?} {:?}", markers[comb[0]], markers[comb[1]]);
        let p = xy_comb(
            &x_status,
            &y_status,
            &neighbors,
            &labels_mapper,
            times,
            pval,
        );
        results.push((markers[comb[0]], markers[comb[1]], p));
        if order {
            let p_ = xy_comb(
                &y_status,
                &x_status,
                &neighbors,
                &labels_mapper,
                times,
                pval,
            );
            results.push((markers[comb[1]], markers[comb[0]], p_));
        } else {
            results.push((markers[comb[1]], markers[comb[0]], p));
        }
    }

    Ok(results.to_object(py))
}

fn xy_comb(
    x_status: &Vec<bool>,
    y_status: &Vec<bool>,
    neighbors: &Vec<Vec<usize>>,
    labels_mapper: &HashMap<usize, usize>,
    times: usize,
    pval: f64,
) -> f64 {
    let real: f64 = comb_count_neighbors(x_status, y_status, &neighbors, labels_mapper) as f64;
    let perm_counts: Vec<usize> = (0..times)
        .into_par_iter()
        .map(|_| {
            let mut rng = thread_rng();
            let mut shuffle_y = y_status.to_owned();
            shuffle_y.shuffle(&mut rng);
            let perm_result = comb_count_neighbors(x_status, &shuffle_y, &neighbors, labels_mapper);
            perm_result
        })
        .collect();
    let m = mean_u(&perm_counts);
    let sd = std_u(&perm_counts);
    // println!("mean {:?} sd {:?}", m, sd);
    if sd != 0.0 {
        let z = (real - m) / sd;
        let pvalue = zscore2pvalue(z, false);
        // println!("z {:?} pvalue {:?}", z, pvalue);
        if pvalue < pval {
            z.signum()
        } else {
            0.0
        }
    } else {
        0.0
    }
}

#[pyclass]
struct CellCombs {
    real_storage: PyObject,
    sim_storage: PyObject,
}

unsafe impl Send for CellCombs {}

#[pymethods]
impl CellCombs {
    #[new]
    fn new(py: Python, types: Vec<&str>) -> PyResult<Self> {
        let uni: Vec<&str> = types.into_iter().unique().collect();
        let mut combs: Vec<(&str, &str)> = uni
            .to_owned()
            .into_iter()
            .permutations(2)
            .map(|i| (i[0], i[1]))
            .collect();

        // Add self-self relationship
        for i in &uni {
            combs.push((*i, *i))
        }

        let real_storage: HashMap<(&str, &str), Vec<usize>> =
            combs.iter().map(|comb| (*comb, vec![])).collect();

        let sim_storage: HashMap<(&str, &str), Vec<f64>> =
            combs.iter().map(|comb| (*comb, vec![])).collect();

        Ok(CellCombs {
            real_storage: real_storage.to_object(py),
            sim_storage: sim_storage.to_object(py),
        })
    }

    /// Bootstrap functions
    ///
    /// If method is 'pval', 1.0 means association, -1.0 means avoidance, 0.0 means insignificance.
    /// If method is 'zscore', results is the exact z-score value.
    ///
    /// Args:
    ///     types: List[str]; The type of all the cells
    ///     neighbors: List[List[int]]; eg. {1:[4,5], 2:[6,7]}, cell at index 1 has neighbor cells from index 4 and 5
    ///     times: int (500); How many times to perform bootstrap
    ///     pval: float (0.05); The threshold of p-value
    ///     method: str ('pval'); 'pval' or 'zscore'
    ///     ignore_self: bool (False); Whether to consider self as a neighbor
    ///
    /// Return:
    ///     List of tuples, eg.('a', 'b', 1.0), the type a and type b has a relationship as association
    ///
    fn bootstrap(
        &self,
        py: Python,
        types: Vec<&str>,
        neighbors: Vec<Vec<usize>>,
        labels: Vec<usize>,
        times: Option<usize>,
        pval: Option<f64>,
        method: Option<&str>,
    ) -> PyResult<PyObject> {
        let real_storage: &HashMap<(&str, &str), Vec<usize>> =
            &self.real_storage.extract(py).unwrap();
        let sim_storage: &HashMap<(&str, &str), Vec<f64>> = &self.sim_storage.extract(py).unwrap();
        // let order: bool = self.order;

        let times = py_kwarg(times, 1000);
        let pval = py_kwarg(pval, 0.05);
        let method = py_kwarg(method, "pval");
        // let ignore_self = py_kwarg(ignore_self, true);
        let type_counts: HashMap<&str, usize> = types
            .to_owned()
            .into_iter()
            .collect::<Counter<_>>()
            .into_map();

        let unique_neighbors = remove_rep_neighbors(neighbors, &labels);
        let real_data = count_neighbors(
            &types,
            &labels,
            real_storage,
            &unique_neighbors,
            &type_counts,
        );

        let mut simulate_data = sim_storage.clone();

        let all_data: Vec<HashMap<(&str, &str), f64>> = (0..times)
            .into_par_iter()
            .map(|_| {
                let mut rng = thread_rng();
                let mut shuffle_types = types.to_owned();
                shuffle_types.shuffle(&mut rng);
                let perm_result = count_neighbors(
                    &shuffle_types,
                    &labels,
                    real_storage,
                    &unique_neighbors,
                    &type_counts,
                );
                perm_result
            })
            .collect();

        for perm_result in all_data {
            for (k, v) in perm_result.iter() {
                simulate_data.get_mut(k).unwrap().push(*v);
            }
        }

        let mut results: Vec<(&str, &str, f64, f64)> = Vec::with_capacity(simulate_data.len());

        for (k, v) in simulate_data.into_iter() {
            let real = real_data[&k];

            if method == "pval" {
                let mut gt: f64 = 0.0;
                let mut lt: f64 = 0.0;
                for i in v.iter() {
                    if i >= &real {
                        gt += 1.0
                    }
                    if i <= &real {
                        lt += 1.0
                    }
                }
                let gt: f64 = gt / (times.to_owned() as f64 + 1.0);
                let lt: f64 = lt / (times.to_owned() as f64 + 1.0);
                let dir: f64 = (gt < lt) as i32 as f64;
                let udir: f64 = -dir;
                let p: f64 = gt * dir + lt * udir;
                let sig: f64 = (p < pval) as i32 as f64;
                let sigv: f64 = sig * (dir - 0.5).signum();
                results.push((k.0, k.1, p, sigv));
            } else {
                let m = mean_f(&v);
                let sd = std_f(&v);
                let mut p = 1.0;
                let sigv = if sd != 0.0 {
                    let z = (real - m) / sd;
                    p = zscore2pvalue(z, false);
                    let dir: f64 = (z > 0.0) as i32 as f64;
                    let sig: f64 = (p < pval) as i32 as f64;
                    sig * (dir - 0.5).signum()
                } else {
                    0.0
                };
                results.push((k.0, k.1, p, sigv));
            }
        }

        let results_py = results.to_object(py);
        Ok(results_py)
    }
}

pub fn count_neighbors<'a>(
    types: &Vec<&'a str>,
    labels: &Vec<usize>,
    storage_ptr: &HashMap<(&'a str, &'a str), Vec<usize>>,
    unique_neighbors: &Vec<Vec<usize>>,
    type_counts: &HashMap<&str, usize>,
) -> HashMap<(&'a str, &'a str), f64> {
    let mut storage = storage_ptr.clone();
    let label_type_mapper: HashMap<usize, &str> = labels
        .into_iter()
        .zip(types)
        .into_iter()
        .map(|(label, tpy)| (*label, *tpy))
        .collect();
    for (neigh, l) in unique_neighbors.iter().zip(labels).into_iter() {
        let cent_type = label_type_mapper.get(&l).unwrap();
        let neigh_type: Counter<_> = neigh
            .iter()
            .map(|i| *(label_type_mapper.get(i).unwrap()))
            .collect();
        for (nt, count) in neigh_type.iter() {
            let comb = (*cent_type, *nt);
            storage.get_mut(&comb).unwrap().push(*count)
        }
    }
    let result: HashMap<(&str, &str), f64> = storage
        .into_iter()
        .map(|(comb, dist)| {
            let div_by = type_counts.get(comb.0).unwrap();
            let avg = (dist.iter().sum::<usize>() as f64) / (*div_by as f64);
            (comb, avg)
        })
        .collect();
    result
}

pub fn remove_rep_neighbors(
    rep_neighbors: Vec<Vec<usize>>,
    labels: &Vec<usize>,
) -> Vec<Vec<usize>> {
    let mut unique_neighbors = vec![];
    for (l, n) in labels.into_iter().zip(rep_neighbors).into_iter() {
        let mut new_neighs: Vec<usize> = vec![];
        for i in n {
            if *l < i {
                new_neighs.push(i)
            }
        }
        unique_neighbors.push(new_neighs)
    }

    unique_neighbors
}

pub fn comb_count_neighbors(
    x: &Vec<bool>,
    y: &Vec<bool>,
    neighbors: &Vec<Vec<usize>>,
    labels_mapper: &HashMap<usize, usize>,
) -> usize {
    let mut count: usize = 0;

    for (k, v) in neighbors.iter().enumerate() {
        if x[k] {
            for c in v.iter() {
                if y[labels_mapper[c]] {
                    count += 1
                }
            }
        }
    }
    count
}
