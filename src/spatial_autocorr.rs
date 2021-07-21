use std::collections::HashMap;

use ndarray::{Array2, ArrayView1, ArrayView2};
use ndarray::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};

use crate::utils::remove_rep_neighbors;

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


pub fn moran_i_index(x: ArrayView1<f64>, w: ArrayView2<usize>, two_tailed: bool) -> (f64, f64) {
    let n: f64 = x.len() as f64;
    let w_sum: f64 = w.sum() as f64;
    let mean_x = x.mean().unwrap();
    let z = x.to_owned() - mean_x;
    let z2ss = (&z * &z).sum();

    let wx: f64 = w.indexed_iter().map(|(i, v)| {
        (*v as f64) * (x[i.0] - &mean_x) * (x[i.1] - &mean_x)
    }).sum();

    let i_value = (n / w_sum) * (wx / z2ss);

    let ei = -1.0 / n - 1.0;

    let n2 = n * n;
    let s0 = w_sum;
    let w1 = &w + &w.t();
    let s1 = ((&w1 * &w1).sum() / 2) as f64;
    let s2 = (&w.sum_axis(Axis(1)) + &w.sum_axis(Axis(0)).t())
        .mapv_into(|a: usize| a.pow(2))
        .sum() as f64;
    let s02 = s0 * s0;
    let v_num = n2 * s1 - n * s2 + 3.0 * s02;
    let v_den = (n - 1.0) * (n + 1.0) * s02;
    let vi_norm = v_num / v_den - (1.0 / (n - 1.0)).powi(2);
    let se_i_norm = vi_norm.powf(1.0 / 2.0);
    let z_norm = (i_value - ei) / se_i_norm;

    let norm_dist = Normal::new(0.0, 1.0).unwrap(); // follow the scipy's default
    let mut p_norm: f64 = if z_norm > 0.0 {
        1.0 - norm_dist.cdf(z_norm)
    } else { norm_dist.cdf(z_norm) };

    if two_tailed { p_norm *= 2.0 }

    (i_value, p_norm)
}

pub fn geary_c_index(x: ArrayView1<f64>, w: ArrayView2<usize>) -> (f64, f64) {
    let n: f64 = x.len() as f64;
    let w_sum: f64 = w.sum() as f64;
    let mean_x = x.mean().unwrap();
    let z = x.to_owned() - mean_x;
    let z2ss = (&z * &z).sum();

    let wx: f64 = w.indexed_iter().map(|(i, v)| {
        (*v as f64) * (x[i.0] - x[i.1]).powi(2)
    }).sum();

    let c_value = ((n - 1.0) / (2.0 * w_sum)) * (wx / z2ss);

    let s0 = w_sum;
    let w1 = &w + &w.t();
    let s1 = ((&w1 * &w1).sum() / 2) as f64;
    let s2 = (&w.sum_axis(Axis(1)) + &w.sum_axis(Axis(0)).t())
        .mapv_into(|a: usize| a.pow(2))
        .sum() as f64;
    let s02 = s0 * s0;

    // let n2 = n * n;
    // let z4 = &z.mapv_into(|a: f64| a.powi(4));
    // let z2 = &z.mapv_into(|a: f64| a.powi(2));
    // let k = (z4.sum() / n) / (z2.sum() / n).powi(2);
    // let A: f64 = (n - 1.0) * s1 * (n2 - 3.0 * n - 6.0 - (n2 - n + 2.0) * k);
    // let B: f64 = (1.0 / 4.0) * ((n - 1.0) * s2 * (n2 + 3.0 * n - 6.0 - (n2 - n + 2.0) * k));
    // let C: f64 = s02 * (n2 - 3.0 - (n - 1.0).powi(2) * k);
    let vc_norm = (1.0 / (2.0 * (n + 1.0) * s02)) *
        ((2.0 * s1 + s2) * (n - 1.0) - 4.0 * s02);
    let se_c_norm = vc_norm.powf(0.5);

    let de = c_value - 1.0;
    let z_norm = de / se_c_norm;

    let norm_dist = Normal::new(0.0, 1.0).unwrap();
    let p_norm = if de > 0.0 { 1.0 - norm_dist.cdf(z_norm) } else {
        norm_dist.cdf(z_norm)
    };

    (c_value, p_norm)
}