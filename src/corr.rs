use ndarray::prelude::*;

use crate::stat::correlation;

pub fn cross_corr(data1: ArrayView2<f64>, data2: ArrayView2<f64>, method: &str) -> Array1<f64> {
    let mut corr_result = Array1::zeros(data1.len() * data2.len());

    for (ix1, arr1) in data1.axis_iter(Axis(0)).enumerate() {
        for (ix2, arr2) in data2.axis_iter(Axis(0)).enumerate() {
            let corr = correlation(&arr1.to_owned().to_vec(), &arr2.to_owned().to_vec(), method);
            let elm = corr_result.get_mut(ix1 + ix2).unwrap();
            *elm = corr;
        }
    }

    corr_result
}
