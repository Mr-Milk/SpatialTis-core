use std::collections::HashMap;

use counter::Counter;
use pyo3::prelude::*;

use crate::stat::{mean_f, mean_u, std_f, std_u};
use crate::utils::{py_kwarg, zscore2pvalue};

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
mod io;
mod cell_interaction;
mod custom_type;

#[pymodule]
fn spatialtis_core<'py>(py: Python, m: &PyModule) -> PyResult<()> {
    // geometry processing
    geo::register(py, m)?;
    cell_interaction::register(py, m)?;
    entropy::register(py, m)?;
    // m.add_wrapped(wrap_pyfunction!(points_bbox))?;
    // m.add_wrapped(wrap_pyfunction!(multipoints_bbox))?;
    // m.add_wrapped(wrap_pyfunction!(polygons_area))?;
    // m.add_wrapped(wrap_pyfunction!(multipolygons_area))?;
    // m.add_wrapped(wrap_pyfunction!(points_shapes))?;

    // geometry io
    io::register(py, m)?;
    // m.add_wrapped(wrap_pyfunction!(dumps_wkt_points))?;
    // m.add_wrapped(wrap_pyfunction!(dumps_wkt_polygons))?;
    // m.add_wrapped(wrap_pyfunction!(reads_wkt_points))?;
    // m.add_wrapped(wrap_pyfunction!(reads_wkt_polygons))?;


    // corr & neighbor depdent markers
    corr::register(py, m)?;
    //m.add_wrapped(wrap_pyfunction!(fast_corr))?;
    m.add_wrapped(wrap_pyfunction!(neighbor_components))?;

    // neighbors search
    neighbors_search::register(py, m)?;
    // m.add_wrapped(wrap_pyfunction!(points_neighbors))?;
    // m.add_wrapped(wrap_pyfunction!(bbox_neighbors))?;

    // spatial autocorr
    spatial_autocorr::register(py, m)?;
    // m.add_wrapped(wrap_pyfunction!(spatial_autocorr))?;
    // m.add_wrapped(wrap_pyfunction!(build_neighbors_matrix))?;

    // spatial distribution
    distribution_index::register(py, m)?;
    //m.add_wrapped(wrap_pyfunction!(spatial_distribution_pattern))?;

    // spatial entropy
    // m.add_wrapped(wrap_pyfunction!(spatial_entropy))?;

    // hotspot
    // m.add_wrapped(wrap_pyfunction!(getis_ord))?;
    hotspot::register(py, m)?;

    // boostrap for cell cell interactions
    // m.add_class::<CellCombs>()?;
    // m.add_wrapped(wrap_pyfunction!(comb_bootstrap))?;

    Ok(())
}


// #[pyfunction]
// pub fn pdist<'py>(py: Python<'py>, points: PyReadonlyArray2<f64>, par: bool) -> &'py PyArray1<f64> {
//     let points = points.as_array();
//     if par {
//         pdist_2d_par(points).into_pyarray(py)
//     } else {
//         pdist_2d(points).into_pyarray(py)
//     }
// }


// #[pyfunction]
// pub fn multipoints_bbox(points_collections: Vec<Vec<(f64, f64)>>)
//                         -> Vec<(f64, f64, f64, f64)> {
//     points_collections.into_iter()
//         .map(|p| points2bbox(p))
//         .collect()
// }
//
// #[pyfunction]
// pub fn points_bbox(points: Vec<(f64, f64)>)
//                    -> (f64, f64, f64, f64) {
//     points2bbox(points)
// }
//
// #[pyfunction]
// pub fn polygons_area(points: Vec<(f64, f64)>) -> f64 {
//     polygon_area(points)
// }
//
// #[pyfunction]
// pub fn multipolygons_area(points_collections: Vec<Vec<(f64, f64)>>) -> Vec<f64> {
//     points_collections.into_iter()
//         .map(|p| polygon_area(p))
//         .collect()
// }
//
//
// #[pyfunction]
// pub fn points_shapes(p: Vec<(f64, f64)>, method: Option<&str>, concavity: Option<f64>)
//                      -> Vec<(f64, f64)> {
//     let concavity = py_kwarg(concavity, 1.5);
//     match method {
//         Some("convex") => convex(p),
//         Some("concave") => concave(p, concavity),
//         _ => convex(p),
//     }
// }

// #[pyfunction]
// pub fn dumps_wkt_points(points: Vec<(f64, f64)>) -> Vec<String> {
//     io::points_wkt(points)
// }
//
// #[pyfunction]
// pub fn dumps_wkt_polygons(polygons: Vec<Vec<(f64, f64)>>) -> Vec<String> {
//     io::polygons_wkt(polygons)
// }
//
// #[pyfunction]
// pub fn reads_wkt_points(wkt_strings: Vec<&str>) -> Vec<(f64, f64)> {
//     io::wkt_points(wkt_strings)
// }
//
// #[pyfunction]
// pub fn reads_wkt_polygons(wkt_strings: Vec<&str>) -> Vec<Vec<(f64, f64)>> {
//     io::wkt_polygons(wkt_strings)
// }

// #[pyfunction]
// pub fn points_neighbors(points: Vec<(f64, f64)>,
//                         labels: Option<Vec<usize>>,
//                         method: Option<&str>,
//                         r: Option<f64>,
//                         k: Option<usize>, )
//                         -> Vec<Vec<usize>> {
//     let labels = py_kwarg(labels, (0..points.len()).into_iter().collect());
//     let r = py_kwarg(r, -1.0);
//     let mut k = py_kwarg(k, 0);
//
//     let method = match method {
//         Some("kdtree") => "kdtree",
//         Some("delaunay") => "delaunay",
//         _ => {
//             if k == 0 { k = 5 };
//             "kdtree"
//         } // default will search for knn = 5
//     };
//
//     if (method == "kdtree") & (r < 0.0) & (k == 0) {
//         panic!("Need either `r` or `k` to run the analysis.")
//     }
//
//     if method == "kdtree" {
//         points_neighbors_kdtree(points, labels, r, k)
//     } else {
//         points_neighbors_triangulation(points, labels)
//     }
// }


// #[pyfunction]
// pub fn bbox_neighbors(bbox: Vec<(f64, f64, f64, f64)>,
//                       labels: Option<Vec<usize>>,
//                       expand: Option<f64>,
//                       scale: Option<f64>,
// )
//                       -> Vec<Vec<usize>> {
//     let labels = py_kwarg(labels, (0..bbox.len()).into_iter().collect());
//     let expand = py_kwarg(expand, -1.0);
//     let scale = py_kwarg(scale, 1.3);
//
//     bbox_neighbors_rtree(init_bbox(bbox, labels), expand, scale)
// }


// compute the number of different cells at neighbors
#[pyfunction]
pub fn neighbor_components(neighbors: Vec<Vec<usize>>, labels: Vec<usize>, types: Vec<&str>)
                           -> (Vec<&str>, Vec<Vec<usize>>) {
    let mut uni_types: HashMap<&str, i64> = HashMap::new();
    let mut types_mapper: HashMap<usize, &str> = HashMap::new();
    for (i, t) in labels.iter().zip(types.iter()) {
        types_mapper.insert(*i, t);
        uni_types.entry(t).or_insert(0);
    }
    let uni_types: Vec<&str> = uni_types.keys().map(|k| *k).collect();
    let result: Vec<Vec<usize>> = neighbors.iter().map(|neigh| {
        let count: HashMap<&&str, usize> = neigh.iter().map(|i| types_mapper.get(i).unwrap()).collect::<Counter<_>>().into_map();
        let result_v: Vec<usize> = uni_types.iter().map(|t| {
            let v: usize = match count.get(&t) {
                Some(v) => *v,
                None => { 0 }
            };
            v
        }).collect();
        result_v
    }).collect();

    (uni_types, result)
}


// Constructor function
//
// Args:
//     types: List[str]; All the type of cells in your research
//     order: bool (False); If False, A->B and A<-B is the same
//
// #[pyclass]
// struct CellCombs {
//     #[pyo3(get)]
//     cell_types: PyObject,
//     #[pyo3(get)]
//     cell_combs: PyObject,
//     #[pyo3(get)]
//     order: bool,
// }
//
// unsafe impl Send for CellCombs {}
//
// #[pymethods]
// impl CellCombs {
//     #[new]
//     fn new(py: Python, types: PyObject, order: Option<bool>)
//            -> PyResult<Self> {
//         let types_data: Vec<&str> = match types.extract(py) {
//             Ok(data) => data,
//             Err(_) => {
//                 return Err(PyTypeError::new_err(
//                     "Can't resolve `types`, should be list of string.",
//                 ));
//             }
//         };
//
//         let order = py_kwarg(order, false);
//         let uni: Vec<&str> = types_data.into_iter().unique().collect();
//         let mut combs: Vec<(&str, &str)> = if order {
//             uni.to_owned().into_iter().permutations(2).map(|i| (i[0], i[1])).collect()
//         } else {
//             uni.to_owned().into_iter().combinations(2).map(|i| (i[0], i[1])).collect()
//         };
//         // Add self-self relationship
//         for i in &uni {
//             combs.push((*i, *i))
//         };
//
//         Ok(CellCombs {
//             cell_types: uni.to_object(py),
//             cell_combs: combs.to_object(py),
//             order,
//         })
//     }
//
//     /// Bootstrap functions
//     ///
//     /// If method is 'pval', 1.0 means association, -1.0 means avoidance, 0.0 means insignificance.
//     /// If method is 'zscore', results is the exact z-score value.
//     ///
//     /// Args:
//     ///     types: List[str]; The type of all the cells
//     ///     neighbors: List[List[int]]; eg. {1:[4,5], 2:[6,7]}, cell at index 1 has neighbor cells from index 4 and 5
//     ///     times: int (500); How many times to perform bootstrap
//     ///     pval: float (0.05); The threshold of p-value
//     ///     method: str ('pval'); 'pval' or 'zscore'
//     ///     ignore_self: bool (False); Whether to consider self as a neighbor
//     ///
//     /// Return:
//     ///     List of tuples, eg.('a', 'b', 1.0), the type a and type b has a relationship as association
//     ///
//     fn bootstrap(
//         &self,
//         py: Python,
//         types: Vec<&str>,
//         neighbors: Vec<Vec<usize>>,
//         labels: Vec<usize>,
//         times: Option<usize>,
//         pval: Option<f64>,
//         method: Option<&str>,
//         ignore_self: Option<bool>,
//     )
//         -> PyResult<PyObject> {
//         let cellcombs: Vec<(&str, &str)> = match self.cell_combs.extract(py) {
//             Ok(data) => data,
//             Err(_) => return Err(PyTypeError::new_err("Resolve cell_combs failed.")),
//         };
//         let order: bool = self.order;
//
//         let times = py_kwarg(times, 1000);
//         let pval = py_kwarg(pval, 0.05);
//         let method = py_kwarg(method, "pval");
//         let ignore_self = py_kwarg(ignore_self, false);
//
//         let neighbors = remove_rep_neighbors(neighbors, labels, ignore_self);
//         let real_data = count_neighbors(&types, &neighbors, &cellcombs, self.order);
//
//         let mut simulate_data = cellcombs
//             .iter()
//             .map(|comb| (comb.to_owned(), vec![]))
//             .collect::<HashMap<(&str, &str), Vec<f64>>>();
//
//         let all_data: Vec<HashMap<(&str, &str), f64>> = (0..times)
//             .into_par_iter()
//             .map(|_| {
//                 let mut rng = thread_rng();
//                 let mut shuffle_types = types.to_owned();
//                 shuffle_types.shuffle(&mut rng);
//                 let perm_result =
//                     count_neighbors(&shuffle_types, &neighbors, &cellcombs, self.order);
//                 perm_result
//             })
//             .collect();
//
//         for perm_result in all_data {
//             for (k, v) in perm_result.iter() {
//                 simulate_data.get_mut(k).unwrap().push(*v);
//             }
//         }
//
//         let mut results: Vec<(&str, &str, f64)> = Vec::with_capacity(simulate_data.len());
//
//         for (k, v) in simulate_data.into_iter() {
//             let real = real_data[&k];
//
//             if method == "pval" {
//                 let mut gt: f64 = 0.0;
//                 let mut lt: f64 = 0.0;
//                 for i in v.iter() {
//                     if i >= &real {
//                         gt += 1.0
//                     }
//                     if i <= &real {
//                         lt += 1.0
//                     }
//                 }
//                 let gt: f64 = gt / (times.to_owned() as f64 + 1.0);
//                 let lt: f64 = lt / (times.to_owned() as f64 + 1.0);
//                 let dir: f64 = (gt < lt) as i32 as f64;
//                 let udir: f64 = !(gt < lt) as i32 as f64;
//                 let p: f64 = gt * dir + lt * udir;
//                 let sig: f64 = (p < pval) as i32 as f64;
//                 let sigv: f64 = sig * (dir - 0.5).signum();
//                 results.push((k.0, k.1, sigv));
//                 if !order { results.push((k.1, k.0, sigv)) }
//             } else {
//                 let m = mean_f(&v);
//                 let sd = std_f(&v);
//
//                 let sigv = if sd != 0.0 {
//                     let z = (real - m) / sd;
//                     let p = zscore2pvalue(z, false);
//                     let dir: f64 = (z > 0.0) as i32 as f64;
//                     let sig: f64 = (p < pval) as i32 as f64;
//                     sig * (dir - 0.5).signum()
//                 } else { 0.0 };
//                 results.push((k.0, k.1, sigv));
//                 if !order { results.push((k.1, k.0, sigv)) }
//             }
//         }
//
//         let results_py = results.to_object(py);
//
//         Ok(results_py)
//     }
// }


// #[pyfunction]
// pub fn fast_corr<'py>(py: Python<'py>, data1: PyReadonlyArray2<f64>, data2: PyReadonlyArray2<f64>, method: Option<&str>)
//                       -> &'py PyArray1<f64> {
//     let data1: ArrayView2<f64> = data1.as_array();
//     let data2: ArrayView2<f64> = data2.as_array();
//
//     let r: Array1<f64> = match method {
//         Some("spearman") => { pair2_spearman(data1, data2) },
//         _ => { pair2_pearson(data1, data2) },
//     };
//
//     r.to_pyarray(py)
// }


// #[pyfunction]
// pub fn spatial_distribution_pattern(points_collections: Vec<Vec<(f64, f64)>>,
//                                     bbox: (f64, f64, f64, f64),
//                                     method: Option<&str>,
//                                     r: Option<f64>,
//                                     resample: Option<usize>,
//                                     quad: Option<(usize, usize)>,
//                                     rect_side: Option<(f64, f64)>,
//                                     pval: Option<f64>,
//                                     min_cells: Option<usize>) -> Vec<(f64, f64, usize)> {
//     let r = py_kwarg(r, bbox_side_part(bbox).0 / 10.0);
//     let resample = py_kwarg(resample, 1000);
//     let pval = py_kwarg(pval, 0.05);
//     let min_cells = py_kwarg(min_cells, 10);
//
//     match method {
//         Some("id") => ix_dispersion_parallel(points_collections, bbox, r, resample, pval, min_cells),
//         Some("morisita") => morisita_parallel(points_collections, bbox, quad, rect_side, pval, min_cells),
//         Some("clark_evans") => clark_evans_parallel(points_collections, bbox, pval, min_cells),
//         _ => clark_evans_parallel(points_collections, bbox, pval, min_cells),
//     }
// }


// #[pyfunction]
// pub fn spatial_entropy(points_collections: Vec<Vec<Point2D>>,
//                        types_collections: Vec<Vec<&str>>,
//                        d: Option<f64>,
//                        cut: Option<usize>,
//                        order: Option<bool>,
//                        method: Option<&str>,
// ) -> Vec<f64> {
//     let bbox = points_bbox(points_collections[0].to_owned());
//     let d = py_kwarg(d, bbox_side_part(bbox).0 / 10.0);
//     let cut = py_kwarg(cut, 3);
//     let order = py_kwarg(order, false);
//     match method {
//         Some("leibovici") => leibovici_parallel(points_collections, types_collections, d),
//         Some("altieri") => altieri_parallel(points_collections, types_collections, cut),
//         _ => leibovici_parallel(points_collections, types_collections, d),
//     }
// }



