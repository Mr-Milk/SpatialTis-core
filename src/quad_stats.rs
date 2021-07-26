use std::collections::HashMap;

use crate::geo::point2bbox;
use crate::stat::floordiv;

pub struct QuadStats {
    pub(crate) nx: usize,
    pub(crate) ny: usize,
    pub(crate) cells_grid_id: Vec<usize>,
}

impl QuadStats {
    pub fn new() -> QuadStats {
        QuadStats {
            nx: 10,
            ny: 10,
            cells_grid_id: vec![],
        }
    }

    pub fn grid_counts(&mut self,
                       points: Vec<(f64, f64)>,
                       bbox: Option<(f64, f64, f64, f64)>,
                       quad: Option<(usize, usize)>,
                       rect_side: Option<(f64, f64)>,
    ) -> HashMap<usize, usize> {
        let bbox = match bbox {
            Some(data) => data,
            _ => point2bbox(points.to_owned())
        }; // if bbox is not provide, calculate it for user

        let width = bbox.2 - bbox.0;
        let height = bbox.3 - bbox.1;

        match quad {
            Some(data) => {
                self.nx = data.0;
                self.ny = data.1;
            }
            _ => {
                match rect_side {
                    Some(rect) => {
                        let nx = floordiv(width, rect.0) as usize;
                        let ny = floordiv(height, rect.1) as usize;
                        if (nx == 0) | (ny == 0) {
                            panic!("The side of the rect is bigger than the bbox")
                        } else {
                            self.nx = nx;
                            self.ny = ny;
                        }
                    }
                    _ => {}
                }
            }
        }

        if (self.nx == 0) | (self.ny == 0) { panic!("quadratic cannot perform with 0 rectangles") }

        let nx_f: f64 = self.nx as f64;
        let ny_f: f64 = self.ny as f64;
        let width = bbox.2 - bbox.0;
        let height = bbox.3 - bbox.1;

        let wx = width / nx_f;
        let hy = height / ny_f;

        let mut dict_id: Vec<usize> = vec![];
        for i in 0..self.ny {
            for j in 0..self.nx {
                dict_id.push(j + i * self.nx)
            }
        }
        let mut dict_id_count: HashMap<usize, usize> = dict_id.iter().map(|i| (*i, 0)).collect();

        for point in points {
            let mut index_x = floordiv((point.0 - bbox.0) as usize, wx as usize);
            let mut index_y = floordiv((point.1 - bbox.1) as usize, hy as usize);

            if index_x == self.nx { index_x -= 1 };
            if index_y == self.ny { index_y -= 1 };
            let id_ = index_y * self.nx + index_x;
            let id_count = dict_id_count.get_mut(&id_).unwrap();
            *id_count += 1;
            self.cells_grid_id.push(id_);
        }

        dict_id_count
    }
}

// fn quad_stats_core(points: Vec<(f64, f64)>,
//                    bbox: (f64, f64, f64, f64),
//                    nx: usize,
//                    ny: usize, )
//                    -> HashMap<usize, usize> {
//     let nx_f: f64 = nx as f64;
//     let ny_f: f64 = ny as f64;
//     let width = bbox.2 - bbox.0;
//     let height = bbox.3 - bbox.1;
//
//     let wx = width / nx_f;
//     let hy = height / ny_f;
//
//     let mut dict_id: Vec<usize> = vec![];
//     for i in 0..ny {
//         for j in 0..nx {
//             dict_id.push(j + i * nx)
//         }
//     }
//     let mut dict_id_count: HashMap<usize, usize> = dict_id.iter().map(|i| (*i, 0)).collect();
//
//     for point in points {
//         let mut index_x = floordiv((point.0 - bbox.0) as usize, wx as usize);
//         let mut index_y = floordiv((point.1 - bbox.1) as usize, hy as usize);
//
//         if index_x == nx { index_x -= 1 };
//         if index_y == ny { index_y -= 1 };
//         let id_ = index_y * nx + index_x;
//         let id_count = dict_id_count.get_mut(&id_).unwrap();
//         *id_count += 1;
//     }
//
//     dict_id_count
// }
