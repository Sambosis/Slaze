import cadquery as cq
result = cq.Workplane("XY").box(10, 20, 30)
print(result)