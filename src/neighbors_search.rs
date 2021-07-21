use delaunator::{Point, triangulate};
use kiddo::distance::squared_euclidean;
use kiddo::KdTree;
use rstar::{AABB, RTree, RTreeObject};

pub fn points_neighbors_kdtree(points: Vec<(f64, f64)>,
                               labels: Vec<usize>,
                               r: f64,
                               k: usize,
) -> Vec<Vec<usize>>
{
    let tree = kdtree_builder(&points, &labels);
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


pub fn points_neighbors_triangulation(points: Vec<(f64, f64)>, labels: Vec<usize>) -> Vec<Vec<usize>>
{
    let points: Vec<Point> = points.into_iter().map(|p| Point { x: p.0, y: p.1 }).collect();
    let result = triangulate(&points).unwrap().triangles;
    let mut neighbors: Vec<Vec<usize>> = (0..labels.len()).into_iter().map(|_| vec![]).collect();

    for i in 0..(result.len() / 3 as usize) {
        let i = i * 3;
        let slice = vec![result[i], result[i + 1], result[i + 2]];
        for x in &slice {
            for y in &slice {
                if neighbors[*x].iter().any(|i| i != y) {
                    neighbors[*x].push(labels[*y])
                }
            }
        }
    };

    neighbors
}


// Build a kdtree using kiddo with labels
pub fn kdtree_builder(points: &Vec<(f64, f64)>, labels: &Vec<usize>) -> KdTree<f64, usize, 2>
{
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
    within.iter().map(|(_, i)| **i).collect()
}


fn points_neighbors_knn_within(tree: &KdTree<f64, usize, 2>, point: &(f64, f64), r: f64, k: usize)
                               -> Vec<usize> {
    tree.best_n_within(&[point.0, point.1], r, k, &squared_euclidean).unwrap()
}


pub fn bbox_neighbors_rtree(bbox: Vec<BBox>, expand: f64, scale: f64) -> Vec<Vec<usize>> {
    let enlarge_bbox: Vec<AABB<[f64; 2]>> = if expand < 0.0 {
        expand_bbox(&bbox, expand)
    } else {
        scale_bbox(&bbox, scale)
    };
    let tree: RTree<BBox> = RTree::<BBox>::bulk_load(bbox);
    enlarge_bbox
        .iter()
        .map(|aabb| {
            let search_result: Vec<&BBox> =
                tree.locate_in_envelope_intersecting(aabb).collect();
            let neighbors: Vec<usize> = search_result.iter().map(|r| r.label).collect();
            neighbors
        }).collect()
}

// customize object to insert in to R-tree
pub struct BBox {
    minx: f64,
    miny: f64,
    maxx: f64,
    maxy: f64,
    label: usize,
}

impl BBox {
    fn new(bbox: (f64, f64, f64, f64), label: usize) -> BBox {
        BBox {
            minx: bbox.0,
            miny: bbox.1,
            maxx: bbox.2,
            maxy: bbox.3,
            label,
        }
    }
}

impl RTreeObject for BBox {
    type Envelope = AABB<[f64; 2]>;

    fn envelope(&self) -> Self::Envelope {
        AABB::from_corners([self.minx, self.miny], [self.maxx, self.maxy])
    }
}


pub fn init_bbox(bbox: Vec<(f64, f64, f64, f64)>, labels: Vec<usize>) -> Vec<BBox> {
    bbox.iter()
        .zip(labels.iter())
        .map(|(b, l)| {
            BBox::new(b.to_owned(), *l)
        }).collect()
}


fn expand_bbox(bbox: &Vec<BBox>, expand: f64) -> Vec<AABB<[f64; 2]>> {
    bbox.iter()
        .map(|b| {
            BBox::new((b.minx - expand,
                       b.miny - expand,
                       b.maxx + expand,
                       b.maxy + expand), b.label).envelope()
        }).collect()
}


fn scale_bbox(bbox: &Vec<BBox>, scale: f64) -> Vec<AABB<[f64; 2]>> {
    bbox.iter()
        .map(|b| {
            let xexpand: f64 = (b.maxx - b.minx) * (scale - 1.0);
            let yexpand: f64 = (b.maxy - b.miny) * (scale - 1.0);
            BBox::new(
                (b.minx - xexpand, b.miny - yexpand, b.maxx + xexpand, b.maxy + yexpand),
                b.label,
            ).envelope()
        }).collect()
}