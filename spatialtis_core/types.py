from typing import List, Tuple, Union, NewType

import numpy as np

Points = NewType("Points", Union[List[List[float]], np.ndarray])
BoundingBox = NewType("BoundingBox", Tuple[float])
Neighbors = NewType("Neighbors", List[List[int]])
Labels = NewType("Labels", List[int])
