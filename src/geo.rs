use geo::algorithm::bounding_rect::BoundingRect;
use geo::LineString;
use geo::algorithm::concave_hull::ConcaveHull;
use geo::algorithm::convex_hull::ConvexHull;
use geo::algorithm::coords_iter::CoordsIter;

/// get_bbox(points_collections)
/// --
///
/// A utility function to return minimum bounding box list of polygons
///
/// Args:
///     points_collections: List[List[(float, float)]]; List of 2d points collections
///
/// Return:
///     A list of bounding box
pub fn get_bbox(points_collections: Vec<Vec<(f64, f64)>>) -> Vec<(f64, f64, f64, f64)> {
    points_collections.into_iter()
        .map(|p| points2bbox(p))
        .collect()
}


fn points2bbox(p: Vec<(f64, f64)>) -> (f64, f64, f64, f64) {
    let line_string: LineString<f64> = p.into();
    let bounding_rect = line_string.bounding_rect().unwrap();
    (bounding_rect.min().x, bounding_rect.min().y, bounding_rect.max().x, bounding_rect.max().y)
}

fn concave(p: Vec<(f64, f64)>, concavity: f64) -> Vec<(f64, f64)> {
    let line_string: LineString<f64> = p.into();
    let res = line_string.concave_hull(concavity);
    res.exterior_coords_iter().map(|coord| (coord.x, coord.y)).collect()
}

fn convex(p: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    let line_string: LineString<f64> = p.into();
    let res = line_string.convex_hull();
    res.exterior_coords_iter().map(|coord| (coord.x, coord.y)).collect()
}