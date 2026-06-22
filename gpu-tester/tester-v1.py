import cupy as cp

# Example: Matrix multiplication
a = cp.array([[1, 2], [3, 4]])
b = cp.array([[5, 6], [7, 8]])
c = cp.dot(a, b)

print(c)