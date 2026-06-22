import os
import sys

# Print environment information
print("CUDA_PATH:", os.environ.get('CUDA_PATH', 'Not set'))
print("PATH entries:")
for path in os.environ.get('PATH', '').split(os.pathsep):
    if 'cuda' in path.lower():
        print(f"  {path}")

try:
    import cupy as cp
    print("\nCUDA Available:", cp.cuda.is_available())
    print("CUDA Version:", cp.cuda.runtime.runtimeGetVersion())

    # Example: Matrix multiplication
    a = cp.array([[1, 2], [3, 4]])
    b = cp.array([[5, 6], [7, 8]])
    c = cp.dot(a, b)
    print("\nMatrix multiplication result:")
    print(c)
except Exception as e:
    print("\nError occurred:", str(e))
    print("Python version:", sys.version)
