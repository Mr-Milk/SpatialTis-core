use std::collections::HashMap;

use crate::utils::remove_rep_neighbors;
use ndarray::prelude::*;
use ndarray::{Array2, ArrayView2, ArrayView1};
use statrs::distribution::{Normal, ContinuousCDF};

// Acquire spatial weights matrix from neighbors relationships
pub fn spatial_weights_matrix(neighbors: Vec<Vec<usize>>, labels: &Vec<usize>) -> Array2<usize> {
    // let mut w_mtx: Vec<Vec<usize>> = (0..neighbors.len()).into_iter()
    //     .map(|_| vec![0; neighbors.len()])
    //     .collect(); // pre-allocate a matrix
    // let labels_map: HashMap<usize, usize> = labels.iter().enumerate().map(|(i, l)| {
    //     (*l, i)
    // }).collect(); // To query the index of labels
    // let trim_neighbors = remove_rep_neighbors(neighbors, true);
    // for (neighbors, l) in trim_neighbors.iter().zip(labels) {
    //     let x_index: usize = labels_map[*l];
    //     for n in neighbors {
    //         let y_index: usize = labels_map[n];
    //         w_mtx[x_index][y_index] = 1
    //     }
    // }

    let n = neighbors.len();
    let mut w_mtx: Array2<usize> = Array2::zeros((n, n));
    let labels_map: HashMap<usize, usize> = labels.iter().enumerate().map(|(i, l)| {
        (*l, i)
    }).collect(); // To query the index of labels
    let trim_neighbors = remove_rep_neighbors(neighbors, true);
    for (neighbors, l) in trim_neighbors.iter().zip(labels) {
        let x_index: usize = labels_map[l];
        for n in neighbors {
            let y_index: usize = labels_map[n];
            w_mtx.slice_mut(s![x_index, y_index]).fill(1)
        }
    }

    w_mtx
}


pub fn moran_i(x: ArrayView1<f64>, w: ArrayView2<usize>, two_tailed: bool) -> (f64, f64) {
    let n: f64 = x.len() as f64;
    let W: f64 = w.sum() as f64;
    let mean_x = x.mean().unwrap();
    let z = x.to_owned() - mean_x;
    let z2ss = (&z * &z).sum();

    let wx: f64 = w.indexed_iter().map(|(i, v)| {
        (*v as f64) * (x[i.0] - &mean_x) * (x[i.1] - &mean_x)
    }).sum();

    let I = (n / W) * (wx / z2ss);

    let EI = -1.0 / n - 1.0;

    let n2 = n*n;
    let s0 = W;
    let w1 = &w + &w.t();
    let s1 = ((&w1 * &w1).sum() / 2) as f64;
    let s2 = (&w.sum_axis(Axis(1)) + &w.sum_axis(Axis(0)).t())
        .mapv_into(|a:usize| a.pow(2))
        .sum() as f64;
    let s02 = s0 * s0;
    let v_num = n2 * s1 - n * s2 + 3.0 * s02;
    let v_den = (n - 1.0) * (n + 1.0) * s02;
    let VI_norm = v_num / v_den - (1.0 / (n - 1.0)).powi(2);
    let seI_norm = VI_norm.powf(1.0 / 2.0);
    let z_norm = (I - EI) / seI_norm;

    let norm_dist = Normal::new(0.0, 1.0).unwrap(); // follow the scipy's default
    let mut p_norm: f64 = if z_norm > 0.0 { 1.0 - norm_dist.cdf(z_norm)
    } else { norm_dist.cdf(z_norm) };

    if two_tailed { p_norm *= 2.0 }

    (I, p_norm)

}

// pub fn geary_c(y: &Vec<f64>, w: &Vec<Vec<usize>>, ) ->