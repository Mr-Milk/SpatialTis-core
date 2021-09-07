use wkt::{Wkt, ToWkt};
use geo::{Point, LineString};
use std::convert::TryFrom;


pub fn points_wkt(points: Vec<(f64, f64)>) -> Vec<String> {
    points.into_iter().map(|(x, y)| {
        let p = Point::new(x, y);
        let wkt = geo::Geometry::from(p).to_wkt();
        let wkt_linestring = wkt.items.first().unwrap();
        format!("{}", wkt_linestring)
    }).collect()
}


pub fn wkt_points(wkt_strings: Vec<&str>) -> Vec<(f64, f64)> {
    wkt_strings.into_iter().map(|w| {
        let wkt_obj = match Wkt::from_str(w) {
            Ok(result) => result,
            Err(_) => panic!("Failed to parse the points, invalid format")
        };
        let p = geo::Point::try_from(wkt_obj).unwrap();
        p.x_y()
    }).collect()
}


pub fn polygons_wkt(polygons: Vec<Vec<(f64, f64)>>) -> Vec<String> {
    polygons.into_iter().map(|poly| {
        let p = geo::Polygon::new(LineString::from(poly), vec![]);
        let wkt = geo::Geometry::from(p).to_wkt();
        let wkt_linestring = wkt.items.first().unwrap();
        format!("{}", wkt_linestring)
    }).collect()
}


pub fn wkt_polygons(wkt_strings: Vec<&str>) -> Vec<Vec<(f64, f64)>> {
    wkt_strings.into_iter().map(|w| {
        let wkt_obj = match Wkt::from_str(w) {
            Ok(result) => result,
            Err(_) => panic!("Failed to parse the shapes, invalid format")
        };
        let p = geo::Polygon::try_from(wkt_obj).unwrap();
        p.exterior().points_iter().map(|ip| ip.x_y()).collect()
    }).collect()
}


#[cfg(test)]
mod test {
    use crate::io::{points_wkt, wkt_points, polygons_wkt, wkt_polygons};

    #[test]
    fn test_points_wkt() {
        let points = vec![(0.0, 1.0), (0.0, 2.0)];
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
        let polygons = vec![vec![(0.0, 1.0), (0.0, 2.0)]];
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



