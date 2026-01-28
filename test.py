from mesograd.engine import Value, _Act

a = Value(0.4)
b = Value(0.2)

c = a + b
c._act = _Act.sigmoid
d = c.act()

d.backward()

print(f"sigmoid({a} + {b}) = {d}")