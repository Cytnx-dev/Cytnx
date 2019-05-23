import pytor10

a = pytor10.Storage(10,pytor10.tor10type.Double,pytor10.tor10device.cpu);

print(a[3])
print(a.dtype)
print(a.dtype_str)

