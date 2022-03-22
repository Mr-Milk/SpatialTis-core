// use itertools::Itertools;
// use ndarray::{Array, Array1, ArrayView1, ArrayView2};
// use ndarray::prelude::*;
// use rayon::prelude::*;
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};

pub fn py_kwarg<T>(arg: Option<T>, default_value: T) -> T {
    match arg {
        Some(data) => data,
        None => default_value,
    }
}

pub fn zscore2pvalue(z: f64, two_tailed: bool) -> f64 {
    let norm_dist: Normal = Normal::new(0.0, 1.0).unwrap(); // follow the scipy's default
    let mut p: f64 = if z > 0.0 {
        1.0 - norm_dist.cdf(z)
    } else {
        norm_dist.cdf(z)
    };

    if two_tailed {
        p *= 2.0
    }

    p
}

pub fn chisquare2pvalue(chi2_value: f64, ddof: f64) -> f64 {
    let chi2_dist: ChiSquared = ChiSquared::new(ddof).unwrap();
    1.0 - chi2_dist.cdf(chi2_value)
}

// fn square_euclidean(p1: ArrayView1<f64>, p2: ArrayView1<f64>) -> f64 {
//     let s = p1.to_owned() - p2.to_owned();
//     return s.dot(&s);
// }
//
//
// pub fn pdist_2d_par(x: ArrayView2<f64>) -> Array1<f64>
// // pairwise distance, skip all the self dist
// {
//     let n = x.shape()[0];
//     let combs: Vec<(usize, usize)> = (0..n).combinations(2)
//         .into_iter()
//         .map(|i| (i[0], i[1]))
//         .collect();
//     let r: Vec<f64> = combs.into_par_iter().map(|(c1, c2)| {
//         square_euclidean(x.slice(s![c1, ..]), x.slice(s![c2, ..]))
//     }).collect();
//     Array::from_vec(r)
// }
//
//
// pub fn pdist_2d(x: ArrayView2<f64>) -> Array1<f64>
// // pairwise distance, skip all the self dist
// {
//     let n = x.shape()[0];
//     let mut result = Array::zeros(n * (n - 1) / 2);
//     let mut ptr: usize = 0;
//     for i in 0..(n - 1) {
//         for j in (i + 1)..n {
//             let d = square_euclidean(x.slice(s![i, ..]), x.slice(s![j, ..]));
//             result[ptr] = d;
//             ptr += 1;
//         }
//     }
//     result
// }

#[cfg(test)]
mod test {}
