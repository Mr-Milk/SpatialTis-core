use itertools::Itertools;
use kiddo::distance::squared_euclidean;
use ndarray::prelude::*;
use ndarray_stats::QuantileExt;
use pyo3::prelude::*;

use crate::custom_type::Point2D;
use crate::neighbors_search::kdtree_builder;
use crate::quad_stats::QuadStats;
use crate::utils::zscore2pvalue;

pub(crate) fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hotspot, m)?)?;
    Ok(())
}

#[pyfunction]
pub fn hotspot(points: Vec<Point2D>,
               bbox: (f64, f64, f64, f64),
               search_level: usize,
               quad: Option<(usize, usize)>,
               rect_side: Option<(f64, f64)>,
               pval: f64,
               min_cells: usize, ) -> Vec<bool> {
    let n = points.len();
    if n == 0 { return vec![]; };
    let mut q = QuadStats::new();
    let counts = q.grid_counts(points, Option::from(bbox), quad, rect_side);
    let nx = q.nx;
    let ny = q.ny;
    let quad_n = (nx * ny) as f64;
    return if (n < min_cells) | (quad_n < 9.0) {
        vec![false; n]
    } else {
        let quad_count = Array::from_shape_vec((nx, ny),
                                               counts.values().into_iter().map(|x| *x as f64).collect_vec()).unwrap();
        let mut idx_points = vec![[0.0, 0.0]; nx * ny];
        for i in 0..nx {
            for j in 0..ny {
                idx_points[i + j] = [i as f64, j as f64]
            }
        }

        let labels: Vec<usize> = (0..idx_points.len()).into_iter().collect();
        let tree = kdtree_builder(&idx_points, &labels);

        let mean_c = quad_count.mean().unwrap();
        let sum_c = quad_count.mapv(|i| i.powi(2)).sum();
        let s = (sum_c / quad_n - mean_c.powi(2)).sqrt();
        if s == 0.0 {
            vec![false; n]
        } else {
            let hot_rect: Vec<bool> = idx_points.iter().map(|p| {
                let neighbors = tree.within_unsorted(&p,
                                                     search_level as f64 * 2.0_f64.sqrt() + 0.0001,
                                                     &squared_euclidean).unwrap();
                let sum_w = neighbors.len() as f64;
                let mut ix: Array1<usize> = Array::from_vec(vec![0; sum_w as usize]);
                let mut iy: Array1<usize> = Array::from_vec(vec![0; sum_w as usize]);

                for (id, neighbor) in neighbors.iter().enumerate() {
                    let pp = idx_points[*neighbor.1];
                    ix[id] = pp[0] as usize;
                    iy[id] = pp[1] as usize;
                };
                let ix_min = *ix.min().unwrap();
                let ix_max = *ix.max().unwrap();
                let iy_min = *iy.min().unwrap();
                let iy_max = *iy.max().unwrap();

                let sum_wc = quad_count.slice(s![ix_min..ix_max, iy_min..iy_max]).sum();
                let u = ((quad_n * sum_w - sum_w.powi(2)) / (quad_n - 1.0)).sqrt();
                if u == 0.0 { false } else {
                    let z = sum_wc - (mean_c * sum_w / (s * u));

                    let p_value = zscore2pvalue(z, true);
                    p_value < pval
                }
            }).collect();

            q.cells_grid_id.iter().map(|id| {
                if hot_rect[*id] {
                    true
                } else {
                    false
                }
            }).collect()
        }
    };
}