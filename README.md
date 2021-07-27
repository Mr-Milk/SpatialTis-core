# spatialtis-core

This repo implement some of the computation intense steps in SpatialTis
for a better performance.

To build the package, you need the nightly version of rustup > 1.55.0
```shell
rustup toolchain install nightly
rustup default nightly
```

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
- [x] Cell-cell interactions (Permutation test)
- [x] Spatial co-expression
    - [x] Spatial enrichment (Permutation test)
    - [x] Correlation
        - [x] Pearson
        - [x] Spearman
- [ ] Quadratic stat
    - [x] Rectangle
    - [ ] Hexagon
- [x] Spatial autocorrelation
    - [x] Moran's I
    - [x] Geary's C
- [x] Spatial distribution index
    - [x] Index of dispersion
    - [x] Morisita index
    - [x] Clarks Evans index
- [x] Getis-ord analysis
- [x] Spatial entropy
    - [x] Leibovici entropy
    - [x] Altieri entropy