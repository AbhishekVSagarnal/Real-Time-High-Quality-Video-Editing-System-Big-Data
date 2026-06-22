import cupy as cp
import time

# Define matrix dimensions
rows, cols = 10000, 10000

# Generate a large random matrix on the GPU
matrix = cp.random.random((rows, cols), dtype=cp.float32)

# Synchronize and start timing
cp.cuda.Stream.null.synchronize()
start_time = time.time()

# Perform Singular Value Decomposition
U, S, Vt = cp.linalg.svd(matrix, full_matrices=False)

# Synchronize and end timing
cp.cuda.Stream.null.synchronize()
end_time = time.time()

# Output the time taken
print(f"SVD computation time: {end_time - start_time:.2f} seconds")
