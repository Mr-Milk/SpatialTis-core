from spatialtis_core import points_bbox, multipoints_bbox, polygons_area, multipolygons_area, points_shapes

points = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.5, 0.5)]
rect = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]


def test_points_bbox():
    bbox = points_bbox(points)
    assert bbox == (0.0, 0.0, 1.0, 1.0)


def test_multipoints_bbox():
    bboxs = multipoints_bbox([points])
    assert bboxs[0] == (0.0, 0.0, 1.0, 1.0)


def test_polygons_area():
    area = polygons_area(rect)
    assert area == 1.0


def test_multipolygons_area():
    areas = multipolygons_area([rect])
    assert areas[0] == 1.0


def test_points_shapes_convex():
    shape = points_shapes(points)
    for i in shape:
        assert i in rect


def test_points_shapes_concave():
    shape = points_shapes(points, method="concave")
    for i in shape:
        assert i in rect
