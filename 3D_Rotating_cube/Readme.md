# Rotating 3D ASCII Cube in Python

This project is an attempt to implement a rotating 3D ASCII cube inspired by the original C implementation from [this repository](https://github.com/servetgulnaroglu/cube.c/tree/master). The goal is to recreate the functionality in Python and explore performance optimizations.

---

## Current State

- The current Python implementation resides in the file **`spinning_cube.py`**.
- The implementation uses the following libraries:
  - **NumPy**: For numerical calculations and matrix operations.
  - **time**: For managing frame rate using `time.sleep`.
  - **os**: For clearing the screen dynamically.

---

## Observations

- **Performance Issue**: The Python implementation is **extremely slow** (less than 1 FPS), as expected due to Python's inherent limitations in handling computation-heavy tasks compared to C.
- This slowness is particularly noticeable in real-time rendering and frequent calculations within nested loops.

---

## Future Iterations

- **Optimization Goals**:
  - Improve the performance of the implementation using techniques such as:
    - **Vectorization** with NumPy.
    - **Batch Processing** to minimize function calls.
    - Efficient screen updates (e.g., buffering outputs instead of frequent `print` calls).
  - Explore the use of tools like:
    - **PyPy**: A Just-In-Time (JIT) compiled Python interpreter.
    - **Numba**: A JIT compiler to optimize specific parts of the code.
    - **Cython**: To compile critical parts of the code into C for better performance.

- **Alternative Approaches**:
  - Compare the Python implementation with its C counterpart to identify bottlenecks.
  - Experiment with faster printing methods or advanced rendering techniques.

---

## Acknowledgments

- Original inspiration and source code: [servetgulnaroglu/cube.c](https://github.com/servetgulnaroglu/cube.c/tree/master)

---

## How to Run

1. Clone this repository.
2. Run the Python script using:
   ```bash
   python spinning_cube.py
