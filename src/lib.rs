use std::collections::HashMap;

use counter::Counter;
use itertools::Itertools;
use numpy::{PyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use rayon::prelude::*;

use crate::distribution_index::{clark_evans_parallel, ix_dispersion_parallel, morisita_parallel};
use crate::entropy::{altieri_parallel, leibovici_parallel};
use crate::geo::{concave, convex, point2bbox};
use crate::hotspot::hotspot;
use crate::neighbors_search::{bbox_neighbors_rtree,
                              init_bbox,
                              points_neighbors_kdtree,
                              points_neighbors_triangulation};
use crate::spatial_autocorr::{geary_c_parallel, moran_i_parallel, spatial_weights_sparse_matrix};
use crate::stat::{mean, std_dev};
use crate::utils::{comb_count_neighbors, count_neighbors, py_kwarg, remove_rep_neighbors, zscore2pvalue};

mod preprocessing;
mod utils;
mod corr;
mod stat;
mod quad_stats;
mod neighbors_search;
mod geo;
mod spatial_autocorr;
mod distribution_index;
mod hotspot;
mod entropy;

#[pymodule]
fn spatialtis_core<'py>(_py: Python, m: &PyModule) -> PyResult<()> {
    // geometry processing
    m.add_wrapped(wrap_pyfunction!(multi_points_bbox))?;
    m.add_wrapped(wrap_pyfunction!(points2bbox))?;
    m.add_wrapped(wrap_pyfunction!(points2shapes))?;

    // corr & neighbor depdent markers
    m.add_wrapped(wrap_pyfunction!(fast_corr))?;
    m.add_wrapped(wrap_pyfunction!(neighbor_components))?;

    // neighbors search
    m.add_wrapped(wrap_pyfunction!(points_neighbors))?;
    m.add_wrapped(wrap_pyfunction!(bbox_neighbors))?;

    // spatial autocorr
    m.add_wrapped(wrap_pyfunction!(spatial_autocorr))?;
    m.add_wrapped(wrap_pyfunction!(build_neighbors_matrix))?;

    // spatial distribution
    m.add_wrapped(wrap_pyfunction!(spatial_distribution_pattern))?;

    // spatial entropy
    m.add_wrapped(wrap_pyfunction!(spatial_entropy))?;

    // hotspot
    m.add_wrapped(wrap_pyfunction!(getis_ord))?;

    // boostrap for cell cell interactions
    m.add_class::<CellCombs>()?;
    m.add_wrapped(wrap_pyfunction!(comb_bootstrap))?;
    Ok(())
}

/// points2bbox(points_collections)
/// --
///
/// A utility function to return minimum bounding box list of polygons
///
/// Args:
///     points_collections: List[List[(float, float)]]; List of 2d points collections
///
/// Return:
///     A list of bounding box
#[pyfunction]
pub fn points2bbox(points_collections: Vec<Vec<(f64, f64)>>)
                   -> Vec<(f64, f64, f64, f64)> {
    points_collections.into_iter()
        .map(|p| point2bbox(p))
        .collect()
}

#[pyfunction]
pub fn multi_points_bbox(points: Vec<(f64, f64)>)
                         -> (f64, f64, f64, f64) {
    point2bbox(points)
}

#[pyfunction]
pub fn points2shapes(p: Vec<(f64, f64)>, method: Option<&str>, concavity: Option<f64>)
                     -> Vec<(f64, f64)> {
    let concavity = py_kwarg(concavity, 1.5);
    match method {
        Some("convex") => convex(p),
        Some("concave") => concave(p, concavity),
        _ => convex(p),
    }
}


#[pyfunction]
pub fn points_neighbors(points: Vec<(f64, f64)>,
                        labels: Option<Vec<usize>>,
                        method: Option<&str>,
                        r: Option<f64>,
                        k: Option<usize>, )
                        -> Vec<Vec<usize>> {
    let labels = py_kwarg(labels, (0..points.len()).into_iter().collect());
    let r = py_kwarg(r, -1.0);
    let mut k = py_kwarg(k, 0);

    let method = match method {
        Some("kdtree") => "kdtree",
        Some("delaunay") => "delaunay",
        _ => {
            k = 5;
            "kdtree"
        } // default will search for knn = 5
    };

    if (method == "kdtree") & (r < 0.0) & (k == 0) {
        panic!("Need either `r` or `k` to run the analysis.")
    }

    if method == "kdtree" {
        points_neighbors_kdtree(points, labels, r, k)
    } else {
        points_neighbors_triangulation(points, labels)
    }
}


#[pyfunction]
pub fn bbox_neighbors(bbox: Vec<(f64, f64, f64, f64)>,
                      labels: Option<Vec<usize>>,
                      expand: Option<f64>,
                      scale: Option<f64>,
)
                      -> Vec<Vec<usize>> {
    let labels = py_kwarg(labels, (0..bbox.len()).into_iter().collect());
    let expand = py_kwarg(expand, -1.0);
    let scale = py_kwarg(scale, 1.3);

    bbox_neighbors_rtree(init_bbox(bbox, labels), expand, scale)
}


// compute the number of different cells at neighbors
#[pyfunction]
pub fn neighbor_components(neighbors: HashMap<usize, Vec<usize>>, types: HashMap<usize, &str>)
                           -> (Vec<usize>, Vec<&str>, Vec<Vec<usize>>) {
    let mut uni_types: HashMap<&str, i64> = HashMap::new();
    for (_, t) in &types {
        uni_types.entry(*t).or_insert(0);
    }
    let uni_types: Vec<&str> = uni_types.keys().map(|k| *k).collect_vec();
    let mut cent_order: Vec<usize> = vec![];
    let result: Vec<Vec<usize>> = neighbors.iter().map(|(cent, neigh)| {
        let count: HashMap<&&str, usize> = neigh.iter().map(|i| &types[i]).collect::<Counter<_>>().into_map();
        let mut result_v: Vec<usize> = vec![];
        for t in &uni_types {
            let v = match count.get(t) {
                Some(v) => *v,
                None => { 0 }
            };
            result_v.push(v);
        }
        cent_order.push(*cent);
        result_v
    }).collect();

    (cent_order, uni_types, result)
}


#[pyfunction]
pub fn build_neighbors_matrix(neighbors: Vec<Vec<usize>>, labels: Vec<usize>)
                              -> (usize, Vec<usize>, Vec<usize>, Vec<usize>, Vec<f64>)
{
    spatial_weights_sparse_matrix(neighbors, labels)
}


/// comb_bootstrap(x_status, y_status, neighbors, times=500, ignore_self=False)
/// --
///
/// Bootstrap between two types
///
/// If you want to test co-localization between protein X and Y, first determine if the cell is X-positive
/// and/or Y-positive. True is considered as positive and will be counted.
///
/// Args:
///     x_status: List[bool]; If cell is type x
///     y_status: List[bool]; If cell is type y
///     neighbors: Dict[int, List[int]]; eg. {1:[4,5], 2:[6,7]}, cell at index 1 has neighbor cells from index 4 and 5
///     times: int (500); How many times to perform bootstrap
///     ignore_self: bool (False); Whether to consider self as a neighbor
///
/// Return:
///     The p-value for the spatial relationship between X and Y
///
#[pyfunction]
pub fn comb_bootstrap(
    x_status: Vec<bool>,
    y_status: Vec<bool>,
    neighbors: Vec<Vec<usize>>,
    labels: Vec<usize>,
    times: Option<usize>,
    ignore_self: Option<bool>,
)
    -> Result<f64, PyErr> {
    let times = py_kwarg(times, 1000);
    let ignore_self = py_kwarg(ignore_self, false);
    let neighbors = utils::remove_rep_neighbors(neighbors, labels, ignore_self);
    let real: f64 = comb_count_neighbors(&x_status, &y_status, &neighbors) as f64;

    let perm_counts: Vec<usize> = (0..times)
        .into_par_iter()
        .map(|_| {
            let mut rng = thread_rng();
            let mut shuffle_y = y_status.to_owned();
            shuffle_y.shuffle(&mut rng);
            let perm_result = comb_count_neighbors(&x_status, &shuffle_y, &neighbors);
            perm_result
        })
        .collect();

    let m = mean(&perm_counts);
    let sd = std_dev(&perm_counts);
    if sd != 0.0 {
        let z = (real - m) / sd;
        Ok(zscore2pvalue(z, false))
    } else { Ok(1.0) }
}


/// Constructor function
///
/// Args:
///     types: List[str]; All the type of cells in your research
///     order: bool (False); If False, A->B and A<-B is the same
///
#[pyclass]
struct CellCombs {
    #[pyo3(get)]
    cell_types: PyObject,
    #[pyo3(get)]
    cell_combs: PyObject,
    #[pyo3(get)]
    order: bool,
}

unsafe impl Send for CellCombs {}

#[pymethods]
impl CellCombs {
    #[new]
    fn new(py: Python, types: PyObject, order: Option<bool>)
           -> PyResult<Self> {
        let types_data: Vec<&str> = match types.extract(py) {
            Ok(data) => data,
            Err(_) => {
                return Err(PyTypeError::new_err(
                    "Can't resolve `types`, should be list of string.",
                ));
            }
        };

        let order = py_kwarg(order, false);
        let uni: Vec<&str> = types_data.into_iter().unique().collect();
        let mut combs = vec![];

        if order {
            for i1 in uni.to_owned() {
                for i2 in uni.to_owned() {
                    combs.push((i1, i2));
                }
            }
        } else {
            for (i1, e1) in uni.to_owned().iter().enumerate() {
                for (i2, e2) in uni.to_owned().iter().enumerate() {
                    if i2 >= i1 {
                        combs.push((e1, e2));
                    }
                }
            }
        }

        Ok(CellCombs {
            cell_types: uni.to_object(py),
            cell_combs: combs.to_object(py),
            order,
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
    ///     List of tuples, eg.(('a', 'b'), 1.0), the type a and type b has a relationship as association
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
        ignore_self: Option<bool>,
    )
        -> PyResult<PyObject> {
        let cellcombs: Vec<(&str, &str)> = match self.cell_combs.extract(py) {
            Ok(data) => data,
            Err(_) => return Err(PyTypeError::new_err("Resolve cell_combs failed.")),
        };

        let times = py_kwarg(times, 1000);
        let pval = py_kwarg(pval, 0.05);
        let method = py_kwarg(method, "pval");
        let ignore_self = py_kwarg(ignore_self, false);

        let neighbors = remove_rep_neighbors(neighbors, labels, ignore_self);
        let real_data = count_neighbors(&types, &neighbors, &cellcombs, self.order);

        let mut simulate_data = cellcombs
            .iter()
            .map(|comb| (comb.to_owned(), vec![]))
            .collect::<HashMap<(&str, &str), Vec<f64>>>();

        let all_data: Vec<HashMap<(&str, &str), f64>> = (0..times)
            .into_par_iter()
            .map(|_| {
                let mut rng = thread_rng();
                let mut shuffle_types = types.to_owned();
                shuffle_types.shuffle(&mut rng);
                let perm_result =
                    count_neighbors(&shuffle_types, &neighbors, &cellcombs, self.order);
                perm_result
            })
            .collect();

        for perm_result in all_data {
            for (k, v) in perm_result.iter() {
                simulate_data.get_mut(k).unwrap().push(*v);
            }
        }

        let mut results: Vec<((&str, &str), f64)> = Vec::with_capacity(simulate_data.len());

        for (k, v) in simulate_data.iter() {
            let real = real_data[k];

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
                results.push((k.to_owned(), sigv));
            } else {
                let m = mean(v);
                let sd = std_dev(v);

                if sd != 0.0 {
                    let z = (real - m) / sd;
                    let p = zscore2pvalue(z, false);
                    let dir: f64 = (z > 0.0) as i32 as f64;
                    let sig: f64 = (p < pval) as i32 as f64;
                    let sigv: f64 = sig * (dir - 0.5).signum();
                    results.push((k.to_owned(), sigv));
                } else { results.push((k.to_owned(), 0.0)) };
            }
        }

        let results_py = results.to_object(py);

        Ok(results_py)
    }
}


#[pyfunction]
pub fn spatial_autocorr(_py: Python,
                        x: PyReadonlyArray2<f64>,
                        neighbors: Vec<Vec<usize>>,
                        labels: Vec<usize>,
                        two_tailed: Option<bool>,
                        method: Option<&str>)
                        -> Vec<(f64, f64)> {
    let x = x.as_array();
    let two_tailed = py_kwarg(two_tailed, true);
    match method {
        Some("moran_i") => moran_i_parallel(x, neighbors, labels, two_tailed),
        Some("geary_c") => geary_c_parallel(x, neighbors, labels),
        _ => moran_i_parallel(x, neighbors, labels, two_tailed), // default back to moran_i
    }
}


#[pyfunction]
pub fn fast_corr<'py>(py: Python<'py>, data1: PyReadonlyArray2<f64>, data2: PyReadonlyArray2<f64>, method: Option<&str>)
                      -> &'py PyArray1<f64> {
    let method: &str = match method {
        Some("pearson") => "p",
        Some("spearman") => "s",
        _ => "s",
    };

    let data1 = data1.as_array();
    let data2 = data2.as_array();

    corr::cross_corr(data1, data2, method).to_pyarray(py)
}


fn bbox_side_part(bbox: (f64, f64, f64, f64)) -> (f64, f64) {
    // -> (min_side, max_side)
    let x_range = bbox.2 - bbox.0;
    let y_range = bbox.3 - bbox.1;
    if x_range <= y_range { (x_range, y_range) } else { (y_range, x_range) }
}


#[pyfunction]
pub fn spatial_distribution_pattern(points_collections: Vec<Vec<(f64, f64)>>,
                                    bbox: (f64, f64, f64, f64),
                                    method: Option<&str>,
                                    r: Option<f64>,
                                    resample: Option<usize>,
                                    quad: Option<(usize, usize)>,
                                    rect_side: Option<(f64, f64)>,
                                    pval: Option<f64>,
                                    min_cells: Option<usize>) -> Vec<(f64, f64, usize)> {
    let r = py_kwarg(r, bbox_side_part(bbox).0 / 10.0);
    let resample = py_kwarg(resample, 1000);
    let pval = py_kwarg(pval, 0.05);
    let min_cells = py_kwarg(min_cells, 10);

    match method {
        Some("id") => ix_dispersion_parallel(points_collections, bbox, r, resample, pval, min_cells),
        Some("morisita") => morisita_parallel(points_collections, bbox, quad, rect_side, pval, min_cells),
        Some("clark_evans") => clark_evans_parallel(points_collections, bbox, pval, min_cells),
        _ => clark_evans_parallel(points_collections, bbox, pval, min_cells),
    }
}


#[pyfunction]
pub fn spatial_entropy(points_collections: Vec<Vec<(f64, f64)>>,
                       types_collections: Vec<Vec<usize>>,
                       d: Option<f64>,
                       cut: Option<usize>,
                       order: Option<bool>,
                       method: Option<&str>,
) -> Vec<f64> {
    let bbox = point2bbox(points_collections[0].to_owned());
    let d = py_kwarg(d, bbox_side_part(bbox).0 / 10.0);
    let cut = py_kwarg(cut, 3);
    let order = py_kwarg(order, false);
    match method {
        Some("leibovici") => leibovici_parallel(points_collections, types_collections, d, order),
        Some("altieri") => altieri_parallel(points_collections, types_collections, cut, order),
        _ => leibovici_parallel(points_collections, types_collections, d, order),
    }
}


#[pyfunction]
pub fn getis_ord(points: Vec<(f64, f64)>,
                 bbox: (f64, f64, f64, f64),
                 search_level: Option<usize>,
                 quad: Option<(usize, usize)>,
                 rect_side: Option<(f64, f64)>,
                 pval: Option<f64>,
                 min_cells: Option<usize>, ) -> Vec<bool> {
    let search_level = py_kwarg(search_level, 3);
    let pval = py_kwarg(pval, 0.05);
    let min_cells = py_kwarg(min_cells, 10);
    hotspot(points, bbox, search_level, quad, rect_side, pval, min_cells)
}


