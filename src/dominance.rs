use crate::utils::{mean_f, norm_vec, hashset};

use itertools::{Itertools, zip, all};
use std::collections::HashMap;
use linregress::{FormulaRegressionBuilder, RegressionDataBuilder};
use std::hash::Hash;
use std::collections::hash_map::RandomState;

pub fn dominance_analysis(y: Vec<f64>, features: Vec<Vec<f64>>) -> Vec<f64>{
    let tmp_names: Vec<usize> = (0..features.len()).collect();
    let all_combs: HashMap<String, Vec<usize>> = all_length_combinations(tmp_names.to_owned());
    let query_combs: HashMap<usize, Vec<Vec<usize>>> = (0..features.len()).into_iter().map(|name| {
        let mut combs: Vec<Vec<usize>> = vec![];
        for (_, comb) in all_combs.to_owned() {
            if !comb.contains(&name) {
                combs.push(comb)
            }
        }

        (name, combs)

    }).collect();
    let combs_rsquare = get_combs_rsquare(y, features.to_owned(), all_combs.to_owned());
    let dominance: Vec<f64> = tmp_names.iter().map(|name| {
        let contribution = query_combs[&name]
            .iter()
            .map(|query| {
                let mut query_self = query.clone();
                query_self.push(*name);
                combs_rsquare[&hashset(query_self)] - combs_rsquare[&hashset(query.to_owned())]
        }).collect();

        mean_f(&contribution)

    }).collect();

    norm_vec(dominance)
}

fn all_length_combinations(items: Vec<usize>) -> HashMap<String, Vec<usize>> {
    let num_items = items.len();
    let mut all_combs: HashMap<String, Vec<usize>> = HashMap::new();

    for l in 1..(num_items+1) {
        for comb in items.to_owned().into_iter().combinations(l) {
            all_combs.insert(hashset(comb.to_owned()), comb.clone());
        }
    }

    all_combs

}

fn get_combs_rsquare(y: Vec<f64>, features: Vec<Vec<f64>>, combs: HashMap<String ,Vec<usize>>) -> HashMap<String, f64> {

    let combs_rsquare: HashMap<String, f64> = combs
        .into_iter()
        .map(|(name, comb)| {
            let data: HashMap<String, Vec<f64>> = comb.iter().map(|i| (i.clone().to_string(), features[*i].clone())).collect();
            let rsquare: f64 = linreg(y.to_owned(), data);

            (name.to_string(), rsquare)

        }).collect();

    combs_rsquare

    }



fn linreg(y: Vec<f64>, mut features: HashMap<String, Vec<f64>>) -> f64 {
    let names: Vec<String> = features.keys().cloned().collect();
    let mut formula = String::from("Y ~ ");
    formula.push_str(&names.join(" + "));
    features.insert(String::from("Y"), y);
    let data = RegressionDataBuilder::new().build_from(features).unwrap();
    let model = FormulaRegressionBuilder::new().data(&data).formula(formula).fit().unwrap();

    model.rsquared
}