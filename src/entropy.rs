use itertools_num::linspace;
use ndarray::prelude::*;
use rayon::prelude::*;

use crate::neighbors_search::points_neighbors_kdtree;

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
    let mut pairs: Vec<usize> = Vec::new();
    let mut pmax = 0;
    for (i, neighs) in neighbors.into_iter().enumerate() {
        for cell in neighs {
            if cell > i {
                let p = if order { types[cell] + 2 * types[i] } else { types[cell] + types[i] };
                pairs.push(p);
                if p > pmax { pmax = p };
            }
        }
    }
    let mut v = vec![];
    for i in pairs_counter(pairs, pmax) { // the result is a sparse array, remove zeros to speed up
        if i != 0.0 { v.push(i) }
    }
    let mut v: Array1<f64> = Array::from_vec(v);
    // println!("{:?}", v.clone());
    v = &v / v.sum();
    v.mapv(|i| i * (1.0 / i).log2()).sum()
}


pub fn altieri_entropy(points: Vec<(f64, f64)>, types: Vec<usize>, cut: usize, order: bool) -> f64 {
    let pdist = pdist_2d(points);
    let ptypes = pair_type(types, order);

    let dist_max = pdist.iter().fold(0.0, |acc, a| { if *a > acc { *a } else { acc } });
    let cut: Vec<f64> = linspace::<f64>(0., dist_max, cut).into_iter().collect();
    let mut cut_range = vec![];
    let cut_len = cut.len();
    for (i, c) in cut.iter().enumerate() {
        if i < (cut_len - 1) {
            cut_range.push((*c, cut[i + 1]))
        }
    }

    let pmax = ptypes.iter().fold(0, |acc, a| { if *a > acc { *a } else { acc } });
    let mut zw = vec![];
    let mut w = vec![];
    for c in cut_range {
        let pdist_mask = dist_cutoff_mask_arr(&pdist, (c.0, c.1));
        let mut used_pairs = vec![];
        for (mask, t) in pdist_mask.iter().zip(ptypes.iter()) {
            if *mask { used_pairs.push(*t) }
        }
        let pairs_counts = pairs_counter(used_pairs, pmax);
        let pairs_counts = normalized_pairs_counts(ArrayBase::from(pairs_counts));
        zw.push(pairs_counts);
        w.push(c.1 - c.0);
    }
    let pairs_counts = pairs_counter(ptypes, pmax);
    let pairs_counts = normalized_pairs_counts(ArrayBase::from(pairs_counts));
    let mut w = Array::from_vec(w);
    w = &w / w.sum();
    let mut h_z = Array::zeros(zw.len());
    let mut pi_z = Array::zeros(zw.len());
    for (i, pc_v) in zw.iter().enumerate() {
        let elem = h_z.get_mut(i).unwrap();
        *elem = pc_v.mapv(|v| v * (1.0 / v).log2()).sum();

        let v2: Array1<f64> = pc_v.iter().enumerate().map(|(i, v)| {
            let z = *pairs_counts.get(i).unwrap();
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
    let mut result = vec![0.0; n * (n - 1) / 2];
    let mut ptr: usize = 0;
    for (i, p1) in x.iter().enumerate() {
        for p2 in x.iter().skip(i + 1) {
            let d = euclidean_distance(*p1, *p2);
            result[ptr] = d;
            ptr += 1;
        }
    }
    result
}

fn pair_type(types: Vec<usize>, order: bool) -> Vec<usize> {
    let n = types.len();
    let mut ptr: usize = 0;
    let mut result = vec![0; n * (n - 1) / 2];
    for (i, t1) in types.iter().enumerate() {
        for t2 in types.iter().skip(i + 1) {
            let p = if order { *t1 + 2 * (*t2) } else { *t1 + *t2 };
            result[ptr] = p;
            ptr += 1;
        }
    }
    result
}


fn dist_cutoff_mask_arr(x: &Vec<f64>, range: (f64, f64)) -> Vec<bool>
{
    x.iter().map(|i| {
        if (*i > range.0) & (*i < range.1) { true } else { false }
    }).collect()
}


fn pairs_counter(pairs: Vec<usize>, n: usize) -> Vec<f64> {
    let mut arr = vec![0.0; n + 1];
    for e in pairs {
        arr[e] += 1.0;
    }
    arr
}


fn normalized_pairs_counts(pairs_counts: Array1<f64>) -> Array1<f64> {
    let s = &pairs_counts.sum();

    let mut arr: Vec<f64> = vec![];
    if *s != 0.0 {
        for c in pairs_counts {
            if c != 0.0 { arr.push(c / *s) }
        }
    }

    Array::from_vec(arr)

}


#[cfg(test)]
mod tests {
    use ndarray::prelude::*;

    use crate::entropy::{dist_cutoff_mask_arr, euclidean_distance, pdist_2d, leibovici_entropy, altieri_entropy};

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
        assert_eq!(pdist_2d(vec![(1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]), vec![1.0, 2.0, 1.0]);
    }

    #[test]
    fn test_leibovici() {
        let points = vec![(1.0, 0.0), (3.0, 0.0), (1.0, 6.0), (3.0, 11.0)];
        let types = vec![1,2,1,1];
        let e = leibovici_entropy(points, types, 2.0, false);
        println!("leibovici entropy is {:?}", e);
    }

    #[test]
    fn test_altieri() {
        let points = vec![(1.0, 0.0), (3.0, 0.0), (1.0, 6.0), (3.0, 11.0)];
        let types = vec![1,2,1,1];
        let e = altieri_entropy(points, types, 2, false);
        println!("altieri entropy is {:?}", e);
    }
}