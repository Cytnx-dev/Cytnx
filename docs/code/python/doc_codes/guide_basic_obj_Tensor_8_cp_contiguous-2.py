C = B.contiguous()

print(C)
print(C.is_contiguous())

print(C.same_data(B))
