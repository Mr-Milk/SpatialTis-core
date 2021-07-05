use std::collections::HashMap;

use crate::utils::remove_rep_neighbors;

// Acquire spatial weights matrix from neighbors relationships
pub fn spatial_weights_matrix(neighbors: Vec<Vec<usize>>, labels: &Vec<usize>) -> Vec<Vec<usize>> {
    let mut w_mtx: Vec<Vec<usize>> = (0..neighbors.len()).into_iter()
        .map(|_| vec![0; neighbors.len()])
        .collect(); // pre-allocate a matrix
    let labels_map: HashMap<usize, usize> = labels.iter().enumerate().map(|(i, l)| {
        (*l, i)
    }).collect(); // To query the index of labels
    let trim_neighbors = remove_rep_neighbors(neighbors, true);
    for (neighbors, l) in trim_neighbors.iter().zip(labels) {
        let x_index: usize = labels_map[*l];
        for n in neighbors {
            let y_index: usize = labels_map[n];
            w_mtx[x_index][y_index] = 1
        }
    }

    w_mtx
}


// pub fn moran_i(y: &Vec<f64>, w: &Vec<Vec<usize>>, ) ->

// pub fn geary_c(y: &Vec<f64>, w: &Vec<Vec<usize>>, ) ->