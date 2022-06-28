use std::collections::HashSet;

use delaunator::{triangulate, Point};
use kiddo::distance::squared_euclidean;
use kiddo::KdTree;
use pyo3::prelude::*;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rstar::{RTree, RTreeObject, AABB};

use crate::custom_type::{BBox, Point2D, Point3D};

pub(crate) fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(points_neighbors_kdtree, m)?)?;
    m.add_function(wrap_pyfunction!(points_neighbors_kdtree_3d, m)?)?;
    m.add_function(wrap_pyfunction!(points_neighbors_triangulation, m)?)?;
    m.add_function(wrap_pyfunction!(bbox_neighbors, m)?)?;
    m.add_function(wrap_pyfunction!(points_neighbors_kdtree_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(points_neighbors_kdtree_3d_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(points_neighbors_triangulation_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(bbox_neighbors_parallel, m)?)?;
    Ok(())
}

#[pyfunction]
pub fn points_neighbors_kdtree(
    points: Vec<Point2D>,
    labels: Vec<usize>,
    r: f64,
    k: usize,
) -> Vec<Vec<usize>> {
    let tree = kdtree_builder(&points, &labels);
    get_neighbors(tree, points, r, k)
}

#[pyfunction]
pub fn points_neighbors_kdtree_parallel(
    points_collections: Vec<Vec<Point2D>>,
    labels_collections: Vec<Vec<usize>>,
    r: f64,
    k: usize,
) -> Vec<Vec<Vec<usize>>> {
    points_collections
        .into_par_iter()
        .zip(labels_collections)
        .map(|(ps, ls)| {
            points_neighbors_kdtree(ps, ls, r, k)
        }).collect()
}

#[pyfunction]
pub fn points_neighbors_kdtree_3d(
    points: Vec<Point3D>,
    labels: Vec<usize>,
    r: f64,
    k: usize,
) -> Vec<Vec<usize>> {
    let tree = kdtree_builder(&points, &labels);
    get_neighbors(tree, points, r, k)
}

#[pyfunction]
pub fn points_neighbors_kdtree_3d_parallel(
    points_collections: Vec<Vec<Point3D>>,
    labels_collections: Vec<Vec<usize>>,
    r: f64,
    k: usize,
) -> Vec<Vec<Vec<usize>>> {
    points_collections
        .into_par_iter()
        .zip(labels_collections)
        .map(|(ps, ls)| {
            points_neighbors_kdtree_3d(ps, ls, r, k)
        }).collect()
}

#[pyfunction]
pub fn points_neighbors_triangulation(points: Vec<Point2D>, labels: Vec<usize>) -> Vec<Vec<usize>> {
    let points: Vec<Point> = points
        .into_iter()
        .map(|p| Point { x: p[0], y: p[1] })
        .collect();
    let result = triangulate(&points).triangles;
    let mut neighbors: Vec<HashSet<usize>> = labels
        .iter()
        .map(|i| {
            let mut neighbors_set = HashSet::new();
            // To ensure that the neighbor contains at least itself
            neighbors_set.insert(*i);
            neighbors_set
        })
        .collect();

    (0..result.len()).into_iter().step_by(3).for_each(|i| {
        let slice = vec![result[i], result[i + 1], result[i + 2]];
        for p1 in &slice {
            for p2 in &slice {
                neighbors[*p1].insert(labels[*p2]);
            }
        }
    });

    neighbors
        .into_iter()
        .map(|n| n.into_iter().collect())
        .collect()
}

#[pyfunction]
pub fn points_neighbors_triangulation_parallel(
    points_collections: Vec<Vec<Point2D>>,
    labels_collections: Vec<Vec<usize>>,
) -> Vec<Vec<Vec<usize>>> {
    points_collections
        .into_par_iter()
        .zip(labels_collections)
        .map(|(ps, ls)| {
            points_neighbors_triangulation(ps, ls)
        }).collect()
}

#[pyfunction]
#[pyo3(name = "bbox_neighbors_rtree")]
pub fn bbox_neighbors(
    bbox: Vec<BBox>,
    labels: Vec<usize>,
    expand: f64,
    scale: f64,
) -> Vec<Vec<usize>> {
    bbox_neighbors_rtree(init_bbox(bbox, labels), expand, scale)
}


#[pyfunction]
#[pyo3(name = "bbox_neighbors_rtree_parallel")]
pub fn bbox_neighbors_parallel(
    bbox_collections: Vec<Vec<BBox>>,
    labels_collections: Vec<Vec<usize>>,
    expand: f64,
    scale: f64,
) -> Vec<Vec<Vec<usize>>> {
    bbox_collections
        .into_par_iter()
        .zip(labels_collections)
        .map(|(bx, ls)| {
            bbox_neighbors(bx, ls, expand, scale)
        }).collect()
}

// Build a kdtree using kiddo with labels
pub fn kdtree_builder<const K: usize>(
    points: &Vec<[f64; K]>,
    labels: &Vec<usize>,
) -> KdTree<f64, usize, K> {
    let mut tree: KdTree<f64, usize, K> = KdTree::new();
    for (p, label) in points.iter().zip(labels) {
        tree.add(p, *label).unwrap();
    }
    tree
}

pub fn get_neighbors<const K: usize>(
    tree: KdTree<f64, usize, K>,
    points: Vec<[f64; K]>,
    r: f64,
    k: usize,
) -> Vec<Vec<usize>> {
    let neighbors = points
        .iter()
        .map(|p| {
            if r > 0.0 {
                if k > 0 {
                    let within = tree.within(p, r * r, &squared_euclidean).unwrap();
                    let mut neighbors = vec![];
                    for (ix, (_, i)) in within.iter().enumerate() {
                        if ix < k {
                            neighbors.push(**i)
                        }
                    }
                    neighbors
                } else {
                    let within = tree.within_unsorted(p, r * r, &squared_euclidean).unwrap();
                    within.iter().map(|(_, i)| **i).collect()
                }
            } else {
                let within = tree.nearest(p, k, &squared_euclidean).unwrap();
                within.iter().map(|(_, i)| **i).collect()
            }
        })
        .collect();
    neighbors
}

// fn points_neighbors_within(tree: &KdTree<f64, usize, 2>, point: &(f64, f64), r: f64)
//                            -> Vec<usize> {
//     let within = tree.within_unsorted(&[point.0, point.1], r*r, &squared_euclidean).unwrap();
//     within.iter().map(|(_, i)| { **i }).collect()
// }
//
//
// fn points_neighbors_knn(tree: &KdTree<f64, usize, 2>, point: &(f64, f64), k: usize)
//                         -> Vec<usize> {
//     let within = tree.nearest(&[point.0, point.1], k, &squared_euclidean).unwrap();
//     within.iter().map(|(_, i)| **i).collect()
// }
//
//
// fn points_neighbors_knn_within(tree: &KdTree<f64, usize, 2>, point: &(f64, f64), r: f64, k: usize)
//                                -> Vec<usize> {
//     tree.best_n_within(&[point.0, point.1], r*r, k, &squared_euclidean).unwrap()
// }

pub fn bbox_neighbors_rtree(bbox: Vec<BBox2D>, expand: f64, scale: f64) -> Vec<Vec<usize>> {
    let enlarge_bbox: Vec<AABB<[f64; 2]>> = if expand <= 0.0 {
        scale_bbox(&bbox, scale)
    } else {
        expand_bbox(&bbox, expand)
    };
    let tree: RTree<BBox2D> = RTree::<BBox2D>::bulk_load(bbox);
    enlarge_bbox
        .iter()
        .map(|aabb| {
            let search_result: Vec<&BBox2D> = tree.locate_in_envelope_intersecting(aabb).collect();
            let neighbors: Vec<usize> = search_result.iter().map(|r| r.label).collect();
            neighbors
        })
        .collect()
}

// customize object to insert in to R-tree
pub struct BBox2D {
    minx: f64,
    miny: f64,
    maxx: f64,
    maxy: f64,
    label: usize,
}

impl BBox2D {
    fn new(bbox: (f64, f64, f64, f64), label: usize) -> BBox2D {
        BBox2D {
            minx: bbox.0,
            miny: bbox.1,
            maxx: bbox.2,
            maxy: bbox.3,
            label,
        }
    }
}

impl RTreeObject for BBox2D {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_corners([self.minx, self.miny], [self.maxx, self.maxy])
    }
}

pub fn init_bbox(bbox: Vec<BBox>, labels: Vec<usize>) -> Vec<BBox2D> {
    bbox.iter()
        .zip(labels.iter())
        .map(|(b, l)| BBox2D::new(b.to_owned(), *l))
        .collect()
}

fn expand_bbox(bbox: &Vec<BBox2D>, expand: f64) -> Vec<AABB<[f64; 2]>> {
    bbox.iter()
        .map(|b| {
            let ebox = (
                b.minx - expand,
                b.miny - expand,
                b.maxx + expand,
                b.maxy + expand,
            );
            BBox2D::new(ebox, b.label).envelope()
        })
        .collect()
}

fn scale_bbox(bbox: &Vec<BBox2D>, scale: f64) -> Vec<AABB<[f64; 2]>> {
    bbox.iter()
        .map(|b| {
            let xexpand: f64 = (b.maxx - b.minx) * (scale - 1.0);
            let yexpand: f64 = (b.maxy - b.miny) * (scale - 1.0);
            BBox2D::new(
                (
                    b.minx - xexpand,
                    b.miny - yexpand,
                    b.maxx + xexpand,
                    b.maxy + yexpand,
                ),
                b.label,
            )
            .envelope()
        })
        .collect()
}
