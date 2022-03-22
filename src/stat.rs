pub fn mean_f(numbers: &Vec<f64>) -> f64 {
    let l = numbers.len();
    if l > 0 {
        let sum: f64 = numbers.iter().sum();
        sum / l as f64
    } else {
        0.0
    }
}

pub fn mean_u(numbers: &Vec<usize>) -> f64 {
    let l = numbers.len();
    if l > 0 {
        let sum: usize = numbers.iter().sum();
        sum as f64 / l as f64
    } else {
        0.0
    }
}

pub fn std_u(numbers: &Vec<usize>) -> f64 {
    let l = numbers.len();
    if l > 0 {
        let m = mean_u(numbers);
        let variance = numbers
            .iter()
            .map(|value| {
                let diff = m - (*value as f64);
                diff * diff
            })
            .sum::<f64>()
            / l as f64;
        variance.sqrt()
    } else {
        0.0
    }
}

pub fn std_f(numbers: &Vec<f64>) -> f64 {
    let l = numbers.len();
    if l > 0 {
        let m = mean_f(numbers);
        let variance = numbers
            .iter()
            .map(|value| {
                let diff = m - (*value as f64);
                diff * diff
            })
            .sum::<f64>()
            / l as f64;
        variance.sqrt()
    } else {
        0.0
    }
}

#[cfg(test)]
mod test {
    use crate::stat::*;

    #[test]
    fn test_mean_u() {
        let x: Vec<usize> = vec![1, 2, 3, 4, 5];
        assert_eq!(mean_u(&x), 3.0)
    }

    #[test]
    fn test_mean_f() {
        let x: Vec<f64> = vec![1., 2., 3., 4., 5.];
        assert_eq!(mean_f(&x), 3.0)
    }

    #[test]
    fn test_std_u() {
        let x: Vec<usize> = vec![1, 2, 3, 4, 5];
        assert_eq!(std_u(&x), std::f64::consts::SQRT_2)
    }

    #[test]
    fn test_std_f() {
        let x: Vec<f64> = vec![1., 2., 3., 4., 5.];
        assert_eq!(std_f(&x), std::f64::consts::SQRT_2)
    }
}
