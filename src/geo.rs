use geo::algorithm::area::Area;
use geo::algorithm::bounding_rect::BoundingRect;
use geo::algorithm::concave_hull::ConcaveHull;
use geo::algorithm::convex_hull::ConvexHull;
use geo::algorithm::coords_iter::CoordsIter;
use geo::{LineString, Polygon};
use pyo3::prelude::*;
use rstar::AABB;

use crate::custom_type::{BBox, BBox3D, Point2D, Point3D};

pub(crate) fn register(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(points_bbox, m)?)?;
    m.add_function(wrap_pyfunction!(points3d_bbox, m)?)?;
    m.add_function(wrap_pyfunction!(multipoints_bbox, m)?)?;
    m.add_function(wrap_pyfunction!(polygon_area, m)?)?;
    m.add_function(wrap_pyfunction!(multipolygons_area, m)?)?;
    m.add_function(wrap_pyfunction!(concave, m)?)?;
    m.add_function(wrap_pyfunction!(convex, m)?)?;
    Ok(())
}

#[pyfunction]
#[pyo3(name = "_points_bbox")]
pub fn points_bbox(p: Vec<Point2D>) -> BBox {
    // minx, miny, maxx, maxy
    let line_string: LineString<f64> = p.into();
    let bounding_rect = line_string.bounding_rect().unwrap();
    (
        bounding_rect.min().x,
        bounding_rect.min().y,
        bounding_rect.max().x,
        bounding_rect.max().y,
    )
}

#[pyfunction]
#[pyo3(name = "_points3d_bbox")]
pub fn points3d_bbox(p: Vec<Point3D>) -> BBox3D {
    // minx, miny, minz, maxx, maxy, maxz
    let bbox: AABB<Point3D> = AABB::from_points(p.iter());
    let lower = bbox.lower();
    let upper = bbox.upper();
    (lower[0], lower[1], lower[2], upper[0], upper[1], upper[2])
}

#[pyfunction]
#[pyo3(name = "_multipoints_bbox")]
pub fn multipoints_bbox(points_collections: Vec<Vec<Point2D>>) -> Vec<BBox> {
    points_collections
        .into_iter()
        .map(|p| points_bbox(p))
        .collect()
}

#[pyfunction]
#[pyo3(name = "_multipoints3d_bbox")]
pub fn multipoints3d_bbox(points_collections: Vec<Vec<Point3D>>) -> Vec<BBox3D> {
    points_collections
        .into_iter()
        .map(|p| points3d_bbox(p))
        .collect()
}

#[pyfunction]
#[pyo3(name = "_polygon_area")]
pub fn polygon_area(p: Vec<Point2D>) -> f64 {
    let polygon = Polygon::new(LineString::from(p), vec![]);
    polygon.unsigned_area()
}

#[pyfunction]
#[pyo3(name = "_multipolygons_area")]
pub fn multipolygons_area(points_collections: Vec<Vec<Point2D>>) -> Vec<f64> {
    points_collections
        .into_iter()
        .map(|p| polygon_area(p))
        .collect()
}

#[pyfunction]
pub fn concave(p: Vec<Point2D>, concavity: f64) -> Vec<Point2D> {
    let line_string: LineString<f64> = p.into();
    let res = line_string.concave_hull(concavity);
    res.exterior_coords_iter()
        .map(|coord| [coord.x, coord.y])
        .collect()
}

#[pyfunction]
pub fn convex(p: Vec<Point2D>) -> Vec<Point2D> {
    let line_string: LineString<f64> = p.into();
    let res = line_string.convex_hull();
    res.exterior_coords_iter()
        .map(|coord| [coord.x, coord.y])
        .collect()
}
