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
