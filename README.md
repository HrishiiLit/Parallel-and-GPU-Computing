# Matrix Multiplication using Sequential, OpenMP, MPI and CUDA

A comparative parallel computing experiment implementing $4000 \times 4000$ matrix multiplication using four different computing models:

* Sequential C
* OpenMP shared-memory parallelism
* MPI distributed-memory parallelism
* CUDA GPU parallelism

The project compares execution time, speedup, resource usage, and the practical differences between CPU, distributed, and GPU-based computation.

## 📌 Problem Definition

Two $4000 \times 4000$ matrices are multiplied:

$$A = 4000 \times 4000$$
$$B = 4000 \times 4000$$
$$C = A \times B$$

All elements of matrices A and B are initialized to $1.0$. Therefore:

$$C[i][j] = \sum_{k=0}^{3999} A[i][k] \times B[k][j] = \underbrace{1 + 1 + \dots + 1}_{\text{4000 terms}} = 4000$$

The expected verification result is:
$$C[0][0] = 4000.00$$

---

## 🧩 Implementations

### 1. Sequential
The baseline implementation performs matrix multiplication using a single CPU execution flow.

```text
Matrix A + Matrix B 
        ↓
    Single CPU
        ↓
     Matrix C
