use ndarray::prelude::*;
use ndarray::{stack, Zip, Data};
use std::collections::HashMap;
use num_traits::Float;

use itertools::{CombinationsWithReplacement, Itertools};


pub fn leibovici_entropy(points: Vec<(f64, f64)>, types: Vec<usize>, d: f64, order: bool) -> f64 {
    let pdist = pdist_2d(points);
    let ptypes = pair_type(types);

    let pdist_mask = dist_cutoff_mask_arr(&pdist, (0.0, d));
    let mut used_pairs = vec![];
    for (mask, t) in pdist_mask.iter().zip(ptypes.iter()) {
        if *mask { used_pairs.push(*t) }
    }

    let pairs_counts = if order { ordered_pairs_counter(used_pairs) } else { unordered_pairs_counter(used_pairs) };
    let mut v: Array1<f64> = Array::from_vec(pairs_counts.values().into_iter().map(|i| *i as f64).collect());
    v = &v / v.sum();
    v.mapv(|i| i * (1.0 / i).log2()).sum()
}


pub fn altieri_entropy(points: Vec<(f64, f64)>, types: Vec<usize>, cut: Vec<(f64, f64)>, order: bool) -> f64 {
    let pdist = pdist_2d(points);
    let ptypes = pair_type(types);

    let mut zw = vec![];
    let mut w = vec![];

    for c in cut {
        let pdist_mask = dist_cutoff_mask_arr(&pdist, (c.0, c.1));
        let mut used_pairs = vec![];
        for (mask, t) in pdist_mask.iter().zip(ptypes.iter()) {
            if *mask { used_pairs.push(*t) }
        }
        let pairs_counts = if order { ordered_pairs_counter(used_pairs) } else { unordered_pairs_counter(used_pairs) };
        let pairs_counts = normalized_pairs_counts(pairs_counts);
        zw.push(pairs_counts);
        w.push(c.1 - c.0);
    }

    let pairs_counts = if order { ordered_pairs_counter(ptypes) } else { unordered_pairs_counter(ptypes) };
    let pairs_counts = normalized_pairs_counts(pairs_counts);

    let mut w = Array::from_vec(w);
    w = &w / w.sum();

    let mut h_z = Array::zeros(zw.len());
    let mut pi_z = Array::zeros(zw.len());
    for (i, pc_v) in zw.iter().enumerate() {
        let v1: Array1<f64> = Array::from_vec(pc_v.values().into_iter().map(|i| *i).collect());
        let elem = h_z.get_mut(i).unwrap();
        *elem = v1.mapv(|v| v * (1.0 / v).log2()).sum();

        let v2: Array1<f64> = pc_v.iter().map(|(k, v)| {
            let z = *pairs_counts.get(k).unwrap();
            let v = *v;
            v * (v / z).log2()
        }).collect();
        let elem = pi_z.get_mut(i).unwrap();
        *elem = v2.sum();
    }

    let residue: f64 = (&w * &h_z).sum();
    let mutual_info: f64 = (&w * &pi_z).sum();

    residue + mutual_info
}

fn normalized_pairs_counts(pairs_counts: HashMap<(usize, usize), usize>) -> HashMap<(usize, usize), f64> {
    let mut pairs = vec![];
    let mut counts = vec![];
    for (k, v) in pairs_counts {
        pairs.push(k);
        counts.push(v as f64);
    }
    let mut v: Array1<f64> = Array::from_vec(counts);
    v = &v / v.sum();
    pairs.iter()
        .zip(v.iter()).map(|(p, c)| (*p, *c)).collect()
}


fn euclidean_distance(p1: (f64, f64), p2: (f64, f64)) -> f64 {
    ((p1.0 - p2.0).powi(2) - (p2.1 - p2.1).powi(2)).sqrt()
}

fn pdist_2d(x: Vec<(f64, f64)>) -> Vec<f64>
// pairwise distance, skip all the self dist
{
    let mut result = vec![];
    for (i, p1) in x.iter().enumerate() {
        for p2 in x.iter().skip(i + 1) {
                let d = euclidean_distance(*p1, *p2);
                result.push(d);
        }
    }
    result
}

fn pair_type(types: Vec<usize>) -> Vec<(usize, usize)> {
    let mut result = vec![];
    for (i, t1) in types.iter().enumerate() {
        for t2 in types.iter().skip(i + 1) {
            result.push((*t1, *t2))
        }
    }
    result
}


fn dist_cutoff_mask_arr(x: &Vec<f64>, range:(f64, f64)) -> Vec<bool>
{
    x.iter().map(|i| {
        if ( *i > range.0 ) & ( *i < range.1 ) { true }
        else { false }
    }).collect()
}


fn ordered_pairs_counter(pairs: Vec<(usize, usize)>) -> HashMap<(usize, usize), usize> {
    let mut result = HashMap::new();
    for pair in pairs {
        match result.get_mut(&pair) {
            Some(d) => { *d += 1 },
            None => { result.insert(pair, 1); } // init value should be 1
        }
    }
    result
}

fn unordered_pairs_counter(pairs: Vec<(usize, usize)>) -> HashMap<(usize, usize), usize> {
    let mut result = HashMap::new();
    for pair in pairs {
        match result.get_mut(&pair) {
            Some(d) => { *d += 1 },
            None => {
                match result.get_mut(&(pair.1, pair.0)) {
                    Some(d_) => { *d_ += 1 },
                    None => { result.insert(pair, 1); }
                }
            }
        }
    }
    result
}




#[cfg(test)]
mod tests {
    use ndarray::prelude::*;
    use crate::entropy::{euclidean_distance, dist_cutoff_mask_arr, pdist_2d};

    #[test]
    fn test_euclidean_dist() {
        assert_eq!(euclidean_distance((1.0, 0.0), (2.0, 0.0)), 1.0);
    }

    #[test]
    fn test_dist_cut() {
        assert_eq!(dist_cutoff_mask_arr(&vec![1.0, 1.5, 2.0], (0.0, 1.4)), vec![true, false, false]);
    }

    #[test]
    fn test_pdist() {
        assert_eq!(pdist_2d(vec![(1.0, 0.0), (2.0, 0.0)]), vec![1.0]);
    }

}