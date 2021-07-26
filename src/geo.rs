use geo::algorithm::bounding_rect::BoundingRect;
use geo::algorithm::concave_hull::ConcaveHull;
use geo::algorithm::convex_hull::ConvexHull;
use geo::algorithm::coords_iter::CoordsIter;
use geo::LineString;

pub fn point2bbox(p: Vec<(f64, f64)>) -> (f64, f64, f64, f64) { // minx, miny, maxx, maxy
    let line_string: LineString<f64> = p.into();
    let bounding_rect = line_string.bounding_rect().unwrap();
    (bounding_rect.min().x, bounding_rect.min().y, bounding_rect.max().x, bounding_rect.max().y)
}

pub fn concave(p: Vec<(f64, f64)>, concavity: f64) -> Vec<(f64, f64)> {
    let line_string: LineString<f64> = p.into();
    let res = line_string.concave_hull(concavity);
    res.exterior_coords_iter().map(|coord| (coord.x, coord.y)).collect()
}

pub fn convex(p: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    let line_string: LineString<f64> = p.into();
    let res = line_string.convex_hull();
    res.exterior_coords_iter().map(|coord| (coord.x, coord.y)).collect()
}