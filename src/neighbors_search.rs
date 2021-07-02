use pyo3::prelude::*;

use rayon::prelude::*;
use std::collections::HashMap;
use itertools::Itertools;
use kiddo::KdTree;
use kiddo::distance::squared_euclidean;

use delaunator::{Point, triangulate};
use counter::Counter;

const PARALLEL_DATA_SIZE: usize = 20000;

// compute the number of different cells at neighbors
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


pub fn points_neighbors_kdtree(points: Vec<(f64, f64)>,
                            labels: Option<Vec<usize>>,
                            r: Option<f64>,
                            k: Option<usize>,
)-> Vec<Vec<usize>> {
    let labels = match labels {
        Some(data) => data,
        _ => (0..points.len()).into_iter().collect(),
    };

    let r = match r {
        Some(data) => data,
        _ => -1.0, // if negative, will no perform radius search
    };

    let k = match k {
        Some(data) => data,
        _ => 0, // if 0, will no perform knn search
    };

    if (r < 0.0) & (k == 0) {
        panic!("Need either `r` or `k` to run the analysis.")
    }

    let mut tree = kdtree_builder(&points, &labels);
    let neighbors: Vec<Vec<usize>> = points.iter().map(|p| {
        if r > 0.0 {
            if k > 0 {
                points_neighbors_knn_within(&tree, p, r, k)
            } else {
                points_neighbors_within(&tree, p, r)
            }
        } else {
            points_neighbors_knn(&tree, p, k)
        }
    }).collect();

    neighbors

}


pub fn points_neighbors_triangulation(points: Vec<(f64, f64)>, labels: Vec<usize>) -> Vec<Vec<usize>> {
    let points: Vec<Point> = points.into_iter().map(|p| Point{x: p.0, y: p.1}).collect();
    let result = triangulate(&points).unwrap().triangles;
    let mut neighbors: Vec<Vec<usize>> = (0..labels.len()).into_iter().map(|_| vec![]).collect();

    for i in 0..(result.len() / 3 as usize) {
        let i = i * 3;
        let slice = vec![result[i], result[i+1], result[i+2]];
        for x in &slice {
            for y in &slice {
                if neighbors[*x].iter().any(|i| i!=y) {
                    neighbors[*x].push(labels[*y])
                }
            }
        }
    };

    neighbors
}


// Build a kdtree using kiddo with labels
fn kdtree_builder(points: &Vec<(f64, f64)>, labels: &Vec<usize>) -> KdTree<f64, usize, 2> {
    let mut tree: KdTree<f64, usize, 2> = KdTree::new();
    for (p, label) in points.iter().zip(labels) {
        tree.add(&[p.0, p.1], *label).unwrap();
    }
    tree
}


fn points_neighbors_within(tree: &KdTree<f64, usize, 2>, point: &(f64, f64), r: f64)
    -> Vec<usize> {
    let within = tree.within_unsorted(&[point.0, point.1], r, &squared_euclidean).unwrap();
    within.iter().map(|(_, i)| { **i }).collect()
}


fn points_neighbors_knn(tree: &KdTree<f64, usize, 2>, point: &(f64, f64), k: usize)
    -> Vec<usize> {
    let within = tree.nearest(&[point.0, point.1], k, &squared_euclidean).unwrap();
    within.iter().map(|(_, i)| **i ).collect()
}


fn points_neighbors_knn_within(tree: &KdTree<f64, usize, 2>, point: &(f64, f64), r: f64, k: usize)
    -> Vec<usize> {
    tree.best_n_within(&[point.0, point.1], r, k, &squared_euclidean).unwrap()
}