use ndarray::prelude::*;
use std::collections::HashMap;
use itertools_num::linspace;

use crate::neighbors_search::points_neighbors_kdtree;
use rayon::prelude::*;
use counter::Counter;


pub fn leibovici_parallel(points_collections: Vec<Vec<(f64, f64)>>,
                                types_collections: Vec<Vec<usize>>,
                                d: f64,
                                order: bool) -> Vec<f64> {
    points_collections.into_par_iter().zip(types_collections).map(|(p, t)| {
        leibovici_entropy(p, t, d, order)
    }).collect()
}


pub fn altieri_parallel(points_collections: Vec<Vec<(f64, f64)>>,
                                types_collections: Vec<Vec<usize>>,
                                cut: usize,
                                order: bool) -> Vec<f64> {
    points_collections.into_par_iter().zip(types_collections).map(|(p, t)| {
        altieri_entropy(p, t, cut, order)
    }).collect()
}



pub fn leibovici_entropy(points: Vec<(f64, f64)>, types: Vec<usize>, d: f64, order: bool) -> f64 {
    let n = points.len();
    let tn = types.len();
    if n != tn { panic!("The length of points must be the same as types") }
    let neighbors = points_neighbors_kdtree(points, types.clone(), d, 0);
    let mut pairs_counts: HashMap<usize, usize> = HashMap::new();
    for (i, neighs) in neighbors.into_iter().enumerate() {
        for cell in neighs {
            if cell > i {
                let p = if order {types[cell] + 2* types[i]} else {types[cell] + types[i]};
                match pairs_counts.get_mut(&p) {
                    Some(data) => *data += 1,
                    None => {pairs_counts.insert(p, 1); ()},
                }
            }
        }
    }

    let mut v: Array1<f64> = Array::from_vec(pairs_counts.into_values().map(|v| v as f64).collect());
    v = &v / v.sum();
    v.mapv(|i| i * (1.0 / i).log2()).sum()
}


pub fn altieri_entropy(points: Vec<(f64, f64)>, types: Vec<usize>, cut: usize, order: bool) -> f64 {
    let pdist = pdist_2d(points);
    let ptypes = pair_type(types, order);

    let dist_max = pdist.iter().fold(0.0, |acc, a| {if acc > *a {*a} else {acc}});
    let cut: Vec<f64> = linspace::<f64>(0., dist_max, cut).into_iter().collect();
    let mut cut_range = vec![];
    let cut_len = cut.len();
    for (i, c) in cut.iter().enumerate() {
        if i < (cut_len - 1) {
            cut_range.push((*c, cut[i+1]))
        }
    }
    let mut zw = vec![];
    let mut w = vec![];
    for c in cut_range {
        let pdist_mask = dist_cutoff_mask_arr(&pdist, (c.0, c.1));
        let mut used_pairs = vec![];
        for (mask, t) in pdist_mask.iter().zip(ptypes.iter()) {
            if *mask { used_pairs.push(*t) }
        }
        let pairs_counts = pairs_counter(used_pairs);
        let pairs_counts = normalized_pairs_counts(pairs_counts);
        zw.push(pairs_counts);
        w.push(c.1 - c.0);
    }
    let pairs_counts = pairs_counter(ptypes);
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
            let z = pairs_counts.get(k).unwrap();
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

fn euclidean_distance(p1: (f64, f64), p2: (f64, f64)) -> f64 {
    ((p1.0 - p2.0).powi(2) - (p2.1 - p2.1).powi(2)).sqrt()
}

fn pdist_2d(x: Vec<(f64, f64)>) -> Vec<f64>
// pairwise distance, skip all the self dist
{
    let n = x.len();
    let mut result = vec![0.0;n*(n-1)/2];
    for (i, p1) in x.iter().enumerate() {
        for p2 in x.iter().skip(i + 1) {
                let d = euclidean_distance(*p1, *p2);
                result.push(d);
        }
    }
    result
}

fn pair_type(types: Vec<usize>, order: bool) -> Vec<usize> {
    let n = types.len();
    let mut result = vec![0;n*(n-1)/2];
    for (i, t1) in types.iter().enumerate() {
        for t2 in types.iter().skip(i + 1) {
            let p = if order {*t1 + 2*(*t2)} else { *t1 + *t2 };
            result.push(p)
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


fn pairs_counter(pairs: Vec<usize>) -> Counter<usize> {
    pairs.into_iter().collect()
}


// fn ordered_pairs_counter(pairs: Vec<(usize, usize)>) -> HashMap<(usize, usize), usize> {
//     pairs.into_iter().collect::<Counter<(usize, usize)>>()
//         .into_iter()
//         .map(|(k, v)| (*k, *v))
//         .collect()
// }
//
// fn unordered_pairs_counter(pairs: Vec<(usize, usize)>) -> HashMap<(usize, usize), usize> {
//     let mut result = HashMap::new();
//     for pair in pairs {
//         match result.get_mut(&pair) {
//             Some(d) => { *d += 1 },
//             None => {
//                 match result.get_mut(&(pair.1, pair.0)) {
//                     Some(d_) => { *d_ += 1 },
//                     None => { result.insert(pair, 1); }
//                 }
//             }
//         }
//     }
//     result
// }

fn normalized_pairs_counts(pairs_counts: Counter<usize>) -> HashMap<usize, f64> {
    let mut pairs = vec![];
    let mut counts = vec![];
    for (k, v) in pairs_counts.iter() {
        pairs.push(*k);
        counts.push(*v as f64);
    }
    let mut v: Array1<f64> = Array::from_vec(counts);
    v = &v / v.sum();
    pairs.iter()
        .zip(v.iter()).map(|(p, c)| (*p, *c)).collect()
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