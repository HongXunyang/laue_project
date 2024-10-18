from shapely.geometry import Polygon
from shapely.affinity import translate

# Create a sample polygon
original_polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

# Specify the translation offsets
dx = 2  # Offset in x-direction
dy = 3  # Offset in y-direction

# Translate the polygon
translated_polygon = translate(original_polygon, xoff=dx, yoff=dy)

# Output the coordinates of the translated polygon
print(list(translated_polygon.exterior.coords))
