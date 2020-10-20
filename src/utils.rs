pub fn mean_f(numbers: &Vec<f64>) -> f64 {
    let l = numbers.len();
    if l > 0 {
        let sum: f64 = numbers.iter().sum();
        sum / l as f64
    } else {
        0.0
    }
}

pub fn hashset<'a>(mut nums: Vec<usize>) -> String {
    nums.sort();
    let str_nums: Vec<String> = nums.into_iter().map(|n| n.to_string()).collect();
    str_nums.join("")
}

pub fn norm_vec(nums: Vec<f64>) -> Vec<f64> {
    let mut s: f64 = 0.0;
    for i in nums.clone() {
        s += i;
    }

    let mut normed: Vec<f64> = vec![];
    for i in nums {
        normed.push(i/s);
    }

    normed

}