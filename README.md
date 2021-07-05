# spatialtis-core

This repo implement some of the computation intense steps in SpatialTis
for a better performance. This is used in favor of the old
neighborhood_analysis packages.

## Modules
- [x] Points -> BBox
- [x] Points -> Shape
  - [x] Convex hull
  - [x] Concave hull
- [x] Search points neighbors
    - [x] KD-Tree
        - Search by radius
        - Search k-nearest neighbours
        - Search k-nearest neighbours within radius
    - [x] Delaunay triangulation
- [x] Search shape neighbors
    - [x] R-Tree (Search by expand or scale the shape)
- [x] Correlation stat
- [ ] Quadratic stat
    - [ ] Rectangle
    - [ ] Hexagon
- [ ] Spatial autocorrelation
    - [ ] Moran's I
    - [ ] Geary's C