import numpy as np
import matplotlib.pyplot as plt


def _inflate_vertices(vertices, multiplier: float = 1.01):
    """
    inflate the vertices by a small amount

    Mechanism:
    - calculate the center of the vertices
    - move the vertices away from the center by (1-multiplier)
    """

    center = np.mean(vertices, axis=0)
    return center + (vertices - center) * multiplier


vertices = np.array([[1, 0], [1, 1], [0, 1], [0, 0]], dtype=np.float32)

inflated_vertices = _inflate_vertices(vertices, multiplier=1.2)
inflated_vertices = _inflate_vertices(inflated_vertices, multiplier=1 / 1.2)
plt.scatter(vertices[:, 0], vertices[:, 1])
plt.scatter(inflated_vertices[:, 0], inflated_vertices[:, 1])
plt.show()
print(inflated_vertices - vertices)
