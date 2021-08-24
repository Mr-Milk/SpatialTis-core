use ndarray::prelude::*;
use ndarray::parallel::prelude::*;

use crate::stat::correlation;
use rayon::iter::IntoParallelRefIterator;

pub fn cross_corr(data1: ArrayView2<f64>, data2: ArrayView2<f64>, method: &str) -> Array1<f64> {
    let n = data1.shape()[0];
    let mut corr_result = Array1::zeros(n * (n - 1) / 2);

    for (ix1, arr1) in data1.outer_iter().enumerate() {
        for (ix2, arr2) in data2.outer_iter().enumerate().skip(ix1 + 1) {
            let corr = correlation(&arr1.to_owned().to_vec(), &arr2.to_owned().to_vec(), method);
            let elm = corr_result.get_mut(ix1 + ix2 -1).unwrap();
            *elm = corr;
        }
    }

    corr_result
}
