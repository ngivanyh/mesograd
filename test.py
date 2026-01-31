from mesograd.engine import Scalar, Activations, Vector

a = Scalar(0.4)
b = Scalar(0.2)

c = a + b
c._act = Activations.sigmoid
d = c.act()

d.backward()

print(f"sigmoid({a} + {b}) = {d}")

e = Vector(["str", 1,2])