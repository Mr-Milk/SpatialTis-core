use itertools::Itertools;
use crate::stat::correlation;
use rayon::prelude::*;


pub fn cross_corr<'a>(labels: &Vec<&'a str>, data1: &Vec<Vec<f64>>, data2: &Vec<Vec<f64>>, method: &str)
    -> Vec<(&'a str, &'a str, f64)> {
    let label_len = labels.len();
    let corr_combs: Vec<Vec<usize>> = (0..label_len).combinations_with_replacement(2).collect();
    corr_combs.par_iter().map(|comb| {
        let a = comb[0];
        let b = comb[1];
        let corr_value = correlation(&data1[a], &data2[b], method);
        (labels[a], labels[b], corr_value)
    }).collect()
}
