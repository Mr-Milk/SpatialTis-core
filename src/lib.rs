mod dominance;
mod preprocessing;
mod utils;

use dominance::dominance_analysis;

#[cfg(test)]
mod tests {
    use crate::dominance::dominance_analysis;

    #[test]
    fn test_dominance() {
        let Y = vec![1., 2. ,3. , 4., 5.];
        let X = vec![
            vec![5., 4., 3., 2., 1.],
            vec![729.53, 439.0367, 42.054, 1., 0.],
            vec![5., 4., 3., 2., 1.],
            vec![5., 4., 3., 2., 1.],
        ];
        let result = dominance_analysis(Y, X);

    }
}