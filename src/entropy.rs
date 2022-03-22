// use std::collections::HashMap;
use std::hash::Hash;

use counter::Counter;
// use itertools::{Itertools};
use itertools_num::linspace;
use ndarray::prelude::*;
use ordered_float::OrderedFloat;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::custom_type::{Point2D, Point3D};
use crate::neighbors_search::{points_neighbors_kdtree, points_neighbors_kdtree_3d};

pub(crate) fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(leibovici_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(altieri_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(leibovici_3d_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(altieri_3d_parallel, m)?)?;
    Ok(())
}

#[pyfunction]
pub fn leibovici_parallel(
    points_collections: Vec<Vec<Point2D>>,
    types_collections: Vec<Vec<&str>>,
    d: f64,
) -> Vec<f64> {
    points_collections
        .into_par_iter()
        .zip(types_collections)
        .map(|(p, t)| leibovici_entropy(p, t, d))
        .collect()
}

#[pyfunction]
pub fn leibovici_3d_parallel(
    points_collections: Vec<Vec<Point3D>>,
    types_collections: Vec<Vec<&str>>,
    d: f64,
) -> Vec<f64> {
    points_collections
        .into_par_iter()
        .zip(types_collections)
        .map(|(p, t)| leibovici_entropy_3d(p, t, d))
        .collect()
}

#[pyfunction]
pub fn altieri_parallel(
    points_collections: Vec<Vec<Point2D>>,
    types_collections: Vec<Vec<&str>>,
    cut: usize,
) -> Vec<f64> {
    points_collections
        .into_par_iter()
        .zip(types_collections)
        .map(|(p, t)| altieri_entropy(p, t, cut))
        .collect()
}

#[pyfunction]
pub fn altieri_3d_parallel(
    points_collections: Vec<Vec<Point3D>>,
    types_collections: Vec<Vec<&str>>,
    cut: usize,
) -> Vec<f64> {
    points_collections
        .into_par_iter()
        .zip(types_collections)
        .map(|(p, t)| altieri_entropy_3d(p, t, cut))
        .collect()
}

pub fn leibovici_entropy(points: Vec<Point2D>, types: Vec<&str>, d: f64) -> f64 {
    let neighbors = points_neighbors_kdtree(points, (0..types.len()).collect(), d, 0);
    leibovici_base(neighbors, types)
}

pub fn leibovici_entropy_3d(points: Vec<Point3D>, types: Vec<&str>, d: f64) -> f64 {
    let neighbors = points_neighbors_kdtree_3d(points, (0..types.len()).collect(), d, 0);
    leibovici_base(neighbors, types)
}

fn leibovici_base(neighbors: Vec<Vec<usize>>, types: Vec<&str>) -> f64 {
    let mut pairs = vec![];
    for (i, neighs) in neighbors.into_iter().enumerate() {
        for cell in neighs {
            if cell > i {
                let p = (types[cell], types[i]);
                pairs.push(p);
            }
        }
    }
    let v = norm_counter_values(pairs);
    v.mapv(|i| i * (1.0 / i).log2()).sum()
}

pub fn altieri_entropy(points: Vec<Point2D>, types: Vec<&str>, cut: usize) -> f64 {
    let pdist = pdist(points);
    altieri_base(pdist, types, cut)
}

pub fn altieri_entropy_3d(points: Vec<Point3D>, types: Vec<&str>, cut: usize) -> f64 {
    let pdist = pdist(points);
    altieri_base(pdist, types, cut)
}

fn altieri_base(pdist: Vec<OrderedFloat<f64>>, types: Vec<&str>, cut: usize) -> f64 {
    // let bbox = points_bbox(points.clone());
    // let dist_max = (bbox.2 - bbox.0).powi(2) + (bbox.3 - bbox.1).powi(2);
    let ptypes = pair_type(types);
    // use squared-euclidean, not more sqrt operation
    // create co-occur distance array
    // create co-occur types
    // use permutation crate to co-sort the array
    let permutation = permutation::sort(&pdist);
    let sort_pdist = permutation.apply_slice(&pdist);
    let sort_ptypes = permutation.apply_slice(&ptypes);
    let dist_max = sort_pdist.last().unwrap().into_inner();
    let cut: Vec<f64> = linspace::<f64>(0., dist_max, cut).into_iter().collect();

    // cut the array based on the cut
    // perform calculations parallel
    // get the final result
    let mut zw = vec![];
    let mut w = vec![];
    let mut cutter_pairs = vec![];

    let mut cutter = cut.into_iter();
    let mut lower = cutter.next().unwrap();
    let mut upper = cutter.next().unwrap();

    for (d, pt) in sort_pdist.iter().zip(sort_ptypes.clone()) {
        if d.into_inner() < upper {
            cutter_pairs.push(pt)
        } else {
            // calculate zw and w
            let v = norm_counter_values(cutter_pairs.to_owned());
            zw.push(v);
            w.push(upper - lower);
            // update upper
            match cutter.next() {
                Some(v) => {
                    // reset pairs container
                    cutter_pairs = vec![];
                    lower = upper;
                    upper = v;
                }
                None => {}
            }
        }
    }
    let all_dist_paris_count = norm_counter_values(sort_ptypes);
    let mut w = Array::from_vec(w);
    w = &w / w.sum();
    let mut h_z = Array::zeros(zw.len());
    let mut pi_z = Array::zeros(zw.len());
    for (i, pc_v) in zw.iter().enumerate() {
        let elem = h_z.get_mut(i).unwrap();
        *elem = pc_v.mapv(|v| v * (1.0 / v).log2()).sum();

        let v2: Array1<f64> = pc_v
            .iter()
            .zip(&all_dist_paris_count)
            .into_iter()
            .map(|(v, z)| {
                let v = *v;
                v * (v / z).log2()
            })
            .collect();
        let elem = pi_z.get_mut(i).unwrap();
        *elem = v2.sum();
    }
    let residue: f64 = (&w * &h_z).sum();
    let mutual_info: f64 = (&w * &pi_z).sum();

    residue + mutual_info
}

fn norm_counter_values<T: Hash + Eq>(pairs: Vec<T>) -> Array1<f64> {
    let v = pairs
        .into_iter()
        .collect::<Counter<T>>()
        .values()
        .map(|v| *v as f64)
        .collect();
    let mut v: Array1<f64> = Array::from_vec(v);
    v = &v / v.sum();
    return v;
}

fn square_euclidean<const K: usize>(p1: [f64; K], p2: [f64; K]) -> f64 {
    p1.into_iter()
        .zip(p2.into_iter())
        .into_iter()
        .map(|(c1, c2)| (c1 - c2).powi(2))
        .sum()
}

fn pdist<const K: usize>(x: Vec<[f64; K]>) -> Vec<OrderedFloat<f64>>
// pairwise distance, skip all the self dist
{
    let n = x.len();
    let mut result = vec![OrderedFloat(0.0); n * (n - 1) / 2];
    let mut ptr: usize = 0;
    for (i, p1) in x.iter().enumerate() {
        for p2 in x.iter().skip(i + 1) {
            let d = OrderedFloat(square_euclidean(*p1, *p2));
            result[ptr] = d;
            ptr += 1;
        }
    }
    result
}

fn pair_type(types: Vec<&str>) -> Vec<(&str, &str)> {
    let n = types.len();
    let mut ptr: usize = 0;
    let mut result = vec![("1", "1"); n * (n - 1) / 2];
    for (i, t1) in types.iter().enumerate() {
        for t2 in types.iter().skip(i + 1) {
            let p = (*t1, *t2);
            result[ptr] = p;
            ptr += 1;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use ndarray::prelude::*;

    use crate::entropy::{altieri_entropy, leibovici_entropy, pdist};

    #[test]
    fn test_pdist() {
        let points = vec![[1.0, 0.0], [3.0, 0.0], [1.0, 2.0]];
        let p: Vec<f64> = pdist(points).into_iter().map(|i| i.into_inner()).collect();
        assert_eq!(p, vec![4.0, 4.0, 8.0]);

        let points = vec![[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.0, 2.0, 0.0]];
        let p: Vec<f64> = pdist(points).into_iter().map(|i| i.into_inner()).collect();
        assert_eq!(p, vec![4.0, 4.0, 8.0]);
    }

    #[test]
    fn test_leibovici() {
        let points = vec![[1.0, 0.0], [3.0, 0.0], [1.0, 6.0], [3.0, 11.0]];
        let types = vec!["1", "2", "3", "4"];
        let e = leibovici_entropy(points, types, 10.0);
        println!("leibovici entropy is {:?}", e);
    }

    #[test]
    fn test_altieri() {
        let points = vec![[1.0, 0.0], [3.0, 0.0], [1.0, 6.0], [3.0, 11.0]];
        let types = vec!["1", "2", "3", "4"];
        let e = altieri_entropy(points, types, 3);
        println!("altieri entropy is {:?}", e);
    }
}
