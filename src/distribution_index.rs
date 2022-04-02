use itertools::Itertools;
use kiddo::distance::squared_euclidean;
use ndarray::Array1;
use pyo3::prelude::*;
use rand::{thread_rng, Rng};
use rayon::prelude::*;

use crate::custom_type::{BBox, BBox3D, Point2D, Point3D};
use crate::neighbors_search::kdtree_builder;
use crate::quad_stats::QuadStats;
use crate::utils::{chisquare2pvalue, zscore2pvalue};

const EMPTY_RETURN: (f64, f64, usize) = (0.0, 0.0, 0);

pub(crate) fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ix_dispersion_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(ix_dispersion_3d_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(morisita_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(clark_evans_parallel, m)?)?;
    m.add_function(wrap_pyfunction!(ix_dispersion, m)?)?;
    m.add_function(wrap_pyfunction!(ix_dispersion_3d, m)?)?;
    m.add_function(wrap_pyfunction!(morisita_ix, m)?)?;
    m.add_function(wrap_pyfunction!(clark_evans_ix, m)?)?;
    Ok(())
}

fn bbox_side_part(bbox: BBox) -> (f64, f64) {
    // -> (min_side, max_side)
    let x_range = bbox.2 - bbox.0;
    let y_range = bbox.3 - bbox.1;
    if x_range <= y_range {
        (x_range, y_range)
    } else {
        (y_range, x_range)
    }
}

#[pyfunction]
pub fn ix_dispersion_parallel(
    points_collections: Vec<Vec<Point2D>>,
    bbox: BBox,
    r: f64,
    resample: usize,
    pval: f64,
    min_cells: usize,
) -> Vec<(f64, f64, usize)> {
    points_collections
        .into_par_iter()
        .map(|p| ix_dispersion(p, bbox, r, resample, pval, min_cells))
        .collect()
}

#[pyfunction]
pub fn ix_dispersion_3d_parallel(
    points_collections: Vec<Vec<Point3D>>,
    bbox: BBox3D,
    r: f64,
    resample: usize,
    pval: f64,
    min_cells: usize,
) -> Vec<(f64, f64, usize)> {
    points_collections
        .into_par_iter()
        .map(|p| ix_dispersion_3d(p, bbox, r, resample, pval, min_cells))
        .collect()
}

#[pyfunction]
pub fn morisita_parallel(
    points_collections: Vec<Vec<Point2D>>,
    bbox: BBox,
    quad: Option<(usize, usize)>,
    rect_side: Option<(f64, f64)>,
    pval: f64,
    min_cells: usize,
) -> Vec<(f64, f64, usize)> {
    points_collections
        .into_par_iter()
        .map(|p| morisita_ix(p, bbox, quad, rect_side, pval, min_cells))
        .collect()
}

#[pyfunction]
pub fn clark_evans_parallel(
    points_collections: Vec<Vec<Point2D>>,
    bbox: BBox,
    pval: f64,
    min_cells: usize,
) -> Vec<(f64, f64, usize)> {
    points_collections
        .into_par_iter()
        .map(|p| clark_evans_ix(p, bbox, pval, min_cells))
        .collect()
}

#[pyfunction]
pub fn ix_dispersion(
    points: Vec<Point2D>,
    bbox: BBox,
    r: f64,
    resample: usize,
    pval: f64,
    min_cells: usize,
) -> (f64, f64, usize) // return (index_value, p_value, pattern)
{
    let n = points.len();
    return if n < min_cells {
        EMPTY_RETURN
    } else {
        let labels: Vec<usize> = (0..n).into_iter().collect();
        let tree = kdtree_builder(&points, &labels);
        let mut counts = vec![0.0; resample];
        let mut rng = thread_rng();
        for i in 0..resample {
            let x: f64 = rng.gen_range(bbox.0..bbox.2);
            let y: f64 = rng.gen_range(bbox.1..bbox.3);
            let within = tree
                .within_unsorted(&[x, y], r, &squared_euclidean)
                .unwrap();
            counts[i] = within.len() as f64;
        }

        let counts = Array1::from_vec(counts);
        let counts_mean = counts.mean().unwrap();
        if counts_mean != 0.0 {
            let id = counts.var(0.0) / counts_mean;
            let ddof = (n - 1) as f64;
            let chi2_v = ddof * id;
            let p_value = chisquare2pvalue(chi2_v, ddof);
            let pattern = get_pattern(id, p_value, pval);
            (id, p_value, pattern)
        } else {
            EMPTY_RETURN
        } // if sample nothing, return 0
    };
}

#[pyfunction]
pub fn ix_dispersion_3d(
    points: Vec<Point3D>,
    bbox: BBox3D,
    r: f64,
    resample: usize,
    pval: f64,
    min_cells: usize,
) -> (f64, f64, usize) // return (index_value, p_value, pattern)
{
    let n = points.len();
    return if n < min_cells {
        EMPTY_RETURN
    } else {
        let labels: Vec<usize> = (0..n).into_iter().collect();
        let tree = kdtree_builder(&points, &labels);
        let mut counts = vec![0.0; resample];
        let mut rng = thread_rng();
        for i in 0..resample {
            let x: f64 = rng.gen_range(bbox.0..bbox.3);
            let y: f64 = rng.gen_range(bbox.1..bbox.4);
            let z: f64 = rng.gen_range(bbox.2..bbox.5);
            let within = tree
                .within_unsorted(&[x, y, z], r, &squared_euclidean)
                .unwrap();
            counts[i] = within.len() as f64;
        }

        let counts = Array1::from_vec(counts);
        let counts_mean = counts.mean().unwrap();
        if counts_mean != 0.0 {
            let id = counts.var(0.0) / counts_mean;
            let ddof = (n - 1) as f64;
            let chi2_v = ddof * id;
            let p_value = chisquare2pvalue(chi2_v, ddof);
            let pattern = get_pattern(id, p_value, pval);
            (id, p_value, pattern)
        } else {
            EMPTY_RETURN
        } // if sample nothing, return 0
    };
}

#[pyfunction]
pub fn morisita_ix(
    points: Vec<Point2D>,
    bbox: BBox,
    quad: Option<(usize, usize)>,
    rect_side: Option<(f64, f64)>,
    pval: f64,
    min_cells: usize,
) -> (f64, f64, usize) {
    let n = points.len();
    return if n < min_cells {
        EMPTY_RETURN
    } else {
        let counts = QuadStats::new().grid_counts(points, Option::from(bbox), quad, rect_side);
        let quad_count =
            Array1::from_vec(counts.values().into_iter().map(|x| *x as f64).collect_vec());
        let sum_x = quad_count.sum();
        let sum_x_sqr = quad_count.mapv(|i| i.powi(2)).sum();
        if sum_x > 1.0 {
            let id = n as f64 * (sum_x_sqr - sum_x) / (sum_x.powi(2) - sum_x);
            let chi2_v = id * (sum_x - 1.0) + n as f64 - sum_x;
            let p_value = chisquare2pvalue(chi2_v, (n - 1) as f64);
            let pattern = get_pattern(id, p_value, pval);
            (id, p_value, pattern)
        } else {
            EMPTY_RETURN
        }
    };
}

#[pyfunction]
pub fn clark_evans_ix(
    points: Vec<Point2D>,
    bbox: BBox,
    pval: f64,
    min_cells: usize,
) -> (f64, f64, usize) {
    let n = points.len();
    return if n < min_cells {
        EMPTY_RETURN
    } else {
        let labels: Vec<usize> = (0..n).into_iter().collect();
        let tree = kdtree_builder(&points, &labels);

        let area = (bbox.2 - bbox.0) * (bbox.3 - bbox.1);
        let r: Array1<f64> = points
            .iter()
            .map(|p| {
                let nearest = tree.nearest(&p, 2, &squared_euclidean).unwrap();
                let np = points[*nearest[1].1];
                squared_euclidean(&np, &p)
            })
            .collect();
        let intensity = n as f64 / area;
        let nnd_mean = r.mean().unwrap();
        let nnd_expected_mean = 1.0 / (2.0 * intensity.sqrt());
        let big_r = nnd_mean / nnd_expected_mean;
        let pi = std::f64::consts::PI;
        let se: f64 = (((4.0 - pi) * area) / (4.0 * pi)).sqrt() / n as f64;
        let z = (nnd_mean - nnd_expected_mean) / se;
        let p_value = zscore2pvalue(z, true);
        let reject_null = p_value < pval;
        let pattern: usize = if reject_null {
            if big_r < 1.0 {
                3
            } else if big_r == 1.0 {
                2
            } else {
                1
            }
        } else {
            1
        };
        (big_r, p_value, pattern)
    };
}

fn get_pattern(v: f64, p_value: f64, pval: f64) -> usize {
    let reject_null = p_value < pval;

    let pattern: usize = if reject_null {
        if v > 1.0 {
            3
        } else if v == 1.0 {
            2
        } else {
            1
        }
    } else {
        1
    };

    pattern
}
