mod preprocessing;
mod utils;
mod corr;
mod stat;
mod quad_stats;
mod neighbors_search;
mod geo;

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use pyo3::exceptions::PyTypeError;

#[pymodule]
fn spatialtis_core<'py>(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(fast_corr))?;
    Ok(())
}

#[pyfunction]
fn fast_corr(py: Python, labels: PyObject, data1: PyObject, data2: PyObject, method: &str)
             -> PyResult<PyObject> {
    let py_labels: Vec<&str> = match labels.extract(py) {
            Ok(data) => data,
            Err(_) => {
                return Err(PyTypeError::new_err(
                    "Can't resolve `labels`, should be list of string.",
                ));
            }
        };

    let used_method: &str = match method {
        "pearson" => "p",
        "spearman" => "s",
        _ => "s"
    };

    let py_data1: Vec<Vec<f64>> = match data1.extract(py) {
            Ok(data) => data,
            Err(_) => {
                return Err(PyTypeError::new_err(
                    "Input data should be float",
                ));
            }
        };

    let py_data2: Vec<Vec<f64>> = match data2.extract(py) {
            Ok(data) => data,
            Err(_) => {
                return Err(PyTypeError::new_err(
                    "Input data should be float",
                ));
            }
        };

    let result: Vec<(&str, &str, f64)> = corr::cross_corr(&py_labels, &py_data1, &py_data2, used_method);

    Ok(
        result.to_object(py)
    )
}




#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    // fn test_corr() {
    //     let x = vec![
    //         vec![5., 4., 3., 2., 1.],
    //         vec![729.53, 439.0367, 42.054, 1., 0.],
    //         vec![5., 4., 3., 2., 1.],
    //         vec![5., 4., 3., 2., 1.],
    //     ];
    //     let y = vec![
    //         vec![5., 4., 3., 2., 1.],
    //         vec![729.53, 439.0367, 42.054, 1., 0.],
    //         vec![5., 4., 3., 2., 1.],
    //         vec![5., 4., 3., 2., 1.],
    //     ];
    //     let labels = vec!["a", "b", "c", "d"];
    //     corr::cross_corr(&labels, &x, &y);
    // }

}