mod dominance;
mod preprocessing;
mod utils;

use dominance::dominance_analysis;

use linregress::{FormulaRegressionBuilder, RegressionDataBuilder};

fn main() {
        let Y = vec![1., 2. ,3. , 4., 5.];
        let X = vec![
            vec![5., 4., 3., 2., 1.],
            vec![729.53, 439.0367, 42.054, 1., 0.],
            vec![5., 4., 3., 2., 1.],
            vec![5., 4., 3., 2., 1.],
        ];
        let result = dominance_analysis(Y, X);



    }