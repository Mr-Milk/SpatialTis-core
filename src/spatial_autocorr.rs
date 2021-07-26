use std::collections::HashMap;

use ndarray::{Array2, ArrayView1, ArrayView2};
use ndarray::prelude::*;
use nalgebra_sparse::CsrMatrix;

use crate::utils::zscore2pvalue;
use itertools::min;
use nalgebra_sparse::ops::serial::{spadd_pattern, spadd_csr_prealloc, spmm_csr_prealloc};
use nalgebra_sparse::ops::Op;

// Acquire spatial weights matrix from neighbors relationships
pub fn spatial_weights_matrix(neighbors: Vec<Vec<usize>>, labels: &Vec<usize>) -> Array2<usize> {

    let n = neighbors.len();
    let mut w_mtx: Array2<usize> = Array2::zeros((n, n));
    let labels_map: HashMap<usize, usize> = labels.iter().enumerate().map(|(i, l)| {
        (*l, i)
    }).collect(); // To query the index of labels
    // let trim_neighbors = remove_rep_neighbors(neighbors, true);
    for (neighbors, l) in neighbors.iter().zip(labels) {
        let x_index: usize = labels_map[l];
        for n in neighbors {
            let y_index: usize = labels_map[n];
            w_mtx.slice_mut(s![x_index, y_index]).fill(1)
        }
    }

    w_mtx
}

struct SpatialWeight {
    s0: f64,
    s1: f64,
    s2: f64,
    s02: f64,
    w_sparse: CsrMatrix<usize>,
}

impl SpatialWeight {
    pub fn from_neighbors(
                          neighbors: &Vec<Vec<usize>>,
                          labels: &Vec<usize>) -> Self {
        let n = neighbors.len();
        let min_offset = min(labels.clone()).unwrap();
        let mut ptr: usize = 0;
        let mut indptr = vec![0];
        let mut indice = vec![];
        for (i, neighs) in neighbors.iter().enumerate() {
            ptr += neighs.len();
            indptr.push(ptr);
            for j in neighs {
                indice.push(*j - min_offset)
            }
        }
        let data_num = indice.len();
        let data: Vec<usize> = vec![1;data_num];

        let w_sparse = CsrMatrix::try_from_csr_data(n, n, indptr, indice, data).unwrap();
        let s0 = data_num;
        let mut w1 = w_sparse.clone();
        spadd_csr_prealloc(1, &mut w1, 1, Op::Transpose(&w_sparse)).unwrap();
        let mut w1_zeros = CsrMatrix::zeros(n, n);
        spmm_csr_prealloc(1, &mut w1_zeros, 1, Op::NoOp(&w1), Op::NoOp(&w1)).unwrap();
        let s1 = w1_zeros.values().iter().fold(0, |acc, a| acc + a) as f64 / 2.0;
        let w_sum1: Array1<usize> = w_sparse.clone().row_iter().map(|a| a.values().len()).collect();
        let w_sum0: Array1<usize> = w_sparse.clone().transpose_as_csc().col_iter().map(|a| a.values().len()).collect();
        let s2 = (w_sum1 + w_sum0).mapv(|a| a.pow(2)).sum() as f64;
        let s02 = (s0 * s0) as f64;

        SpatialWeight {
            s0: s0 as f64,
            s1,
            s2,
            s02,
            w_sparse
        }
    }

    pub fn wx(&self, z: ArrayView1<f64>) -> f64 {
        self.w_sparse

    }
}


pub fn moran_i_index(x: ArrayView1<f64>, neighbors: &Vec<Vec<usize>>,
                          labels: &Vec<usize>, two_tailed: bool) -> (f64, f64)
{
    let w = SpatialWeight::from_neighbors(neighbors, labels);
    let n: f64 = x.len() as f64;
    // let w_sum: f64 = w.sum() as f64;
    let mean_x = x.mean().unwrap();
    let z = x.to_owned() - mean_x;
    let z2ss = (&z * &z).sum();

    let wx: f64 = w.indexed_iter().map(|(i, v)| {
        (*v as f64) * (x[i.0] - &mean_x) * (x[i.1] - &mean_x)
    }).sum();

    let i_value = (n / w_sum) * (wx / z2ss);

    let ei = -1.0 / (n - 1.0);

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
    let p_norm = zscore2pvalue(z_norm, two_tailed);

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
    let p_norm = zscore2pvalue(z_norm, false);

    (c_value, p_norm)
}