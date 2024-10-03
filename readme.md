
# OpenCL Matrix and Vector Multiplication

This repository contains two OpenCL programs for performing matrix-vector multiplication and matrix-matrix multiplication. The programs leverage OpenCL to perform computations on the GPU for improved performance.

## Prerequisites

Before compiling and running the programs, ensure that OpenCL development libraries are installed on your system. You can install them using the following command:

```bash
sudo apt install ocl-icd-opencl-dev
```

This will install the necessary OpenCL headers and libraries for compilation.

## Files

- `vector_multiply.c`: Program to perform vector-matrix multiplication using OpenCL.
- `matrix_multiply.c`: Program to perform matrix-matrix multiplication using OpenCL.
- `Makefile`: A simple Makefile to compile both programs.

## Compilation

To compile the programs, run the following command in the root directory of the project:

```bash
make
```

This will generate two executable files:
- `vector_multiply`: The executable for vector-matrix multiplication.
- `matrix_multiply`: The executable for matrix-matrix multiplication.

## Running the Programs

Once compiled, you can run the executables as follows:

- **Vector-Matrix Multiplication**:
    ```bash
    ./vector_multiply
    ```

- **Matrix-Matrix Multiplication**:
    ```bash
    ./matrix_multiply
    ```

Both programs will perform computations on randomly initialized matrices and vectors and print the results.

## Cleaning Up

To clean up the compiled executables, run:

```bash
make clean
```

This will remove the `vector_multiply` and `matrix_multiply` executables from the directory.

