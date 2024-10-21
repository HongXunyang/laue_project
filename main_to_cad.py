import ezdxf

# Create a new DXF document
doc = ezdxf.new(dxfversion="R2010")
msp = doc.modelspace()

# Your list of polygons (each polygon is a list of (x, y) tuples)
polygons = [
    [(0, 0), (1, 0), (1, 1), (0, 1)],  # Example square
    # Add more polygons here
]

for polygon in polygons:
    # Close the polygon by adding the first point at the end if not closed
    if polygon[0] != polygon[-1]:
        polygon.append(polygon[0])

    # Add the LWPOLYLINE entity to the modelspace
    msp.add_lwpolyline(polygon)

# Save the DXF document
doc.saveas("polygons.dxf")
