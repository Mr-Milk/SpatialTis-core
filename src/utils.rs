use std::collections::HashMap;

use counter::Counter;
use itertools::min;
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};

use crate::stat::mean;

pub fn py_kwarg<T>(arg: Option<T>, default_value: T) -> T {
    match arg {
        Some(data) => data,
        None => default_value,
    }
}

pub fn count_neighbors<'a>(
    types: &Vec<&'a str>,
    neighbors: &Vec<Vec<usize>>,
    cell_combs: &Vec<(&'a str, &'a str)>,
    order: bool,
)
    -> HashMap<(&'a str, &'a str), f64> {
    let mut storage = cell_combs
        .iter()
        .map(|comb| (comb.to_owned(), vec![]))
        .collect::<HashMap<(&str, &str), Vec<usize>>>();
    for (k, v) in neighbors.iter().enumerate() {
        let cent_type = types[k];
        let neigh_type: Counter<_> = { v.iter().map(|i| types[*i]).collect::<Counter<_>>() };
        for (nt, c) in neigh_type.iter() {
            let comb = (cent_type, *nt);
            let reverse_comb = (*nt, cent_type);
            let count = *c;
            if order {
                storage.get_mut(&comb).unwrap().push(count);
                storage.get_mut(&reverse_comb).unwrap().push(count);
            } else {
                match storage.get_mut(&comb) {
                    None => storage.get_mut(&reverse_comb).unwrap().push(count * 2),
                    Some(s) => s.push(count * 2),
                };
            }
        }
    }

    let mut results: HashMap<(&'a str, &'a str), f64> = HashMap::new();
    for (k, v) in storage.iter() {
        results.insert(k.to_owned(), mean(&v));
    }
    results
}

pub fn comb_count_neighbors(x: &Vec<bool>, y: &Vec<bool>, neighbors: &Vec<Vec<usize>>)
                            -> usize {
    let mut count: usize = 0;

    for (k, v) in neighbors.iter().enumerate() {
        if x[k] {
            for c in v.iter() {
                if y[*c] {
                    count += 1
                }
            }
        }
    }
    count
}

pub fn remove_rep_neighbors(rep_neighbors: Vec<Vec<usize>>, labels: Vec<usize>, ignore_self: bool)
                            -> Vec<Vec<usize>> {
    let min_offset = min(labels.iter()).unwrap();
    let mut neighbors = vec![];
    for (i, neighs) in rep_neighbors.iter().enumerate() {
        let mut new_neighs = vec![];
        let l = labels[i];
        if ignore_self {
            for cell in neighs {
                if *cell > l {
                    new_neighs.push(*cell - *min_offset)
                }
            }
        } else {
            for cell in neighs {
                if *cell >= l {
                    new_neighs.push(*cell - *min_offset)
                }
            }
        }

        neighbors.push(new_neighs)
    }

    neighbors
}


pub fn zscore2pvalue(z: f64, two_tailed: bool) -> f64 {
    let norm_dist: Normal = Normal::new(0.0, 1.0).unwrap(); // follow the scipy's default
    let mut p: f64 = if z > 0.0 {
        1.0 - norm_dist.cdf(z)
    } else { norm_dist.cdf(z) };

    if two_tailed { p *= 2.0 }

    p
}

pub fn chisquare2pvalue(chi2_value: f64, ddof: f64) -> f64 {
    let chi2_dist: ChiSquared = ChiSquared::new(ddof).unwrap();
    1.0 - chi2_dist.cdf(chi2_value)
}