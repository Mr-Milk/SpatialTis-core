use std::convert::TryFrom;
use std::str::FromStr;

use geo::{LineString, Point};
use pyo3::prelude::*;
use wkt::{ToWkt, Wkt};

use crate::custom_type::Point2D;

pub(crate) fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(points_wkt, m)?)?;
    m.add_function(wrap_pyfunction!(wkt_points, m)?)?;
    m.add_function(wrap_pyfunction!(polygons_wkt, m)?)?;
    m.add_function(wrap_pyfunction!(wkt_polygons, m)?)?;
    Ok(())
}

#[pyfunction]
pub fn points_wkt(points: Vec<Point2D>) -> Vec<String> {
    points
        .into_iter()
        .map(|p| {
            let p: Point<f64> = p.into();
            let wkt = geo::Geometry::from(p).to_wkt();
            format!("{}", wkt.item)
        })
        .collect()
}

#[pyfunction]
pub fn wkt_points(wkt_strings: Vec<&str>) -> Vec<Point2D> {
    wkt_strings
        .into_iter()
        .map(|w| {
            let wkt_obj = match Wkt::from_str(w) {
                Ok(result) => result,
                Err(_) => panic!("Failed to parse the points, invalid format"),
            };
            let p = geo::Point::try_from(wkt_obj).unwrap();
            let (x, y) = p.x_y();
            [x, y]
        })
        .collect()
}

#[pyfunction]
pub fn polygons_wkt(polygons: Vec<Vec<Point2D>>) -> Vec<String> {
    polygons
        .into_iter()
        .map(|poly| {
            let p = geo::Polygon::new(LineString::from(poly), vec![]);
            let wkt = geo::Geometry::from(p).to_wkt();
            format!("{}", wkt.item)
        })
        .collect()
}

#[pyfunction]
pub fn wkt_polygons(wkt_strings: Vec<&str>) -> Vec<Vec<Point2D>> {
    wkt_strings
        .into_iter()
        .map(|w| {
            let wkt_obj = match Wkt::from_str(w) {
                Ok(result) => result,
                Err(_) => panic!("Failed to parse the shapes, invalid format"),
            };
            let p = geo::Polygon::try_from(wkt_obj).unwrap();
            p.exterior()
                .points()
                .map(|ip| {
                    let (x, y) = ip.x_y();
                    [x, y]
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod test {
    use crate::io::{points_wkt, polygons_wkt, wkt_points, wkt_polygons};

    #[test]
    fn test_points_wkt() {
        let points = vec![[0.0, 1.0], [0.0, 2.0]];
        let wkt = points_wkt(points);
        println!("{:?}", wkt);
    }

    #[test]
    fn test_wkt_points() {
        let points = vec!["POINT(1 2)", "POINT(0 2)"];
        let wkt = wkt_points(points);
        println!("{:?}", wkt);
    }

    #[test]
    fn test_polygons_wkt() {
        let polygons = vec![vec![[0.0, 1.0], [0.0, 2.0]]];
        let wkt = polygons_wkt(polygons);
        println!("{:?}", wkt);
    }

    #[test]
    fn test_wkt_polygons() {
        let polygons = vec!["POLYGON((0 1,0 2,0 1))"];
        let wkt = wkt_polygons(polygons);
        println!("{:?}", wkt);
    }
}
