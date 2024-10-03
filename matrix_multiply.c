#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <CL/cl.h>

#define CL_TARGET_OPENCL_VERSION 300  // Define OpenCL version to avoid warning
int MATRIX_SIZE=400;

// OpenCL kernel for matrix-matrix multiplication
const char *kernelSource = 
    "__kernel void matmat_mult(__global const int* A, __global const int* B, __global int* C, int n, int p) { \n"
    "    int row = get_global_id(0); \n"
    "    int col = get_global_id(1); \n"
    "    int sum = 0; \n"
    "    for (int k = 0; k < n; k++) { \n"
    "        sum += A[row * n + k] * B[k * p + col]; \n"
    "    } \n"
    "    C[row * p + col] = sum; \n"
    "} \n";

// Software-based matrix-matrix multiplication
void software_matrix_matrix_multiplication(int *A, int *B, int *C, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            C[i * size + j] = 0;
            for (int k = 0; k < size; k++) {
                C[i * size + j] += A[i * size + k] * B[k * size + j];
            }
        }
    }
}

// Function to print a small portion of matrices (for debugging or testing)
void print_matrix(int *matrix, int size, int elements_to_print) {
    for (int i = 0; i < elements_to_print; i++) {
        for (int j = 0; j < elements_to_print; j++) {
            printf("%d ", matrix[i * size + j]);
        }
        printf("\n");
    }
}

// Main function
int main() {
    // Allocate memory for the two matrices and result
    int *matrixA = (int*)malloc(sizeof(int) * MATRIX_SIZE * MATRIX_SIZE);  // A is M x N
    int *matrixB = (int*)malloc(sizeof(int) * MATRIX_SIZE * MATRIX_SIZE);  // B is N x P
    int *result = (int*)malloc(sizeof(int) * MATRIX_SIZE * MATRIX_SIZE);   // C is M x P
    int *ocl_result = (int*)malloc(sizeof(int) * MATRIX_SIZE * MATRIX_SIZE);

    // Initialize matrices with arbitrary values
    for (int i = 0; i < MATRIX_SIZE; i++) {
        for (int j = 0; j < MATRIX_SIZE; j++) {
            matrixA[i * MATRIX_SIZE + j] = i + j;  // Example initialization of matrixA
            matrixB[i * MATRIX_SIZE + j] = i - j;  // Example initialization of matrixB
        }
    }

    double cpu_time_used = 0;
    if(MATRIX_SIZE > 400){
        printf("It's not worth computing this with CPU, it will take ages to run\n");
    } else {
    // --- SOFTWARE MULTIPLICATION ---
        clock_t start_cpu = clock();
        software_matrix_matrix_multiplication(matrixA, matrixB, result, MATRIX_SIZE);
        clock_t end_cpu = clock();
        cpu_time_used = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC;
    }

    // --- OPENCL MULTIPLICATION ---
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret;

    // Get OpenCL platform and device information
    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_devices);

    // Create OpenCL context and command queue
    cl_context context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
    cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_id, 0, &ret);

    // Create memory buffers on the device for the matrices and result
    cl_mem buffer_matrixA = clCreateBuffer(context, CL_MEM_READ_ONLY, MATRIX_SIZE * MATRIX_SIZE * sizeof(int), NULL, &ret);
    cl_mem buffer_matrixB = clCreateBuffer(context, CL_MEM_READ_ONLY, MATRIX_SIZE * MATRIX_SIZE * sizeof(int), NULL, &ret);
    cl_mem buffer_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, MATRIX_SIZE * MATRIX_SIZE * sizeof(int), NULL, &ret);

    // Copy the matrices to the GPU memory
    ret = clEnqueueWriteBuffer(command_queue, buffer_matrixA, CL_TRUE, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(int), matrixA, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, buffer_matrixB, CL_TRUE, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(int), matrixB, 0, NULL, NULL);

    // Create OpenCL program and build it
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &ret);
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "matmat_mult", &ret);

    // Set kernel arguments
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffer_matrixA);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buffer_matrixB);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buffer_result);
    ret = clSetKernelArg(kernel, 3, sizeof(int), &MATRIX_SIZE);  // size N for the multiplication
    ret = clSetKernelArg(kernel, 4, sizeof(int), &MATRIX_SIZE);  // size P for the multiplication

    // Execute the kernel with a 2D work size
    size_t global_item_size[2] = {MATRIX_SIZE, MATRIX_SIZE}; // M x P work-items
    clock_t start_ocl = clock();
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global_item_size, NULL, 0, NULL, NULL);
    ret = clFinish(command_queue);
    clock_t end_ocl = clock();
    double ocl_time_used = ((double)(end_ocl - start_ocl)) / CLOCKS_PER_SEC;

    printf("CPU matrix-matrix multiplication took: %f seconds\n", cpu_time_used);
    printf("OpenCL matrix-matrix multiplication took: %f seconds\n", ocl_time_used);
    printf("Speedup: x%.2f\n", cpu_time_used / ocl_time_used); 

    // Copy the result from the GPU memory to the host memory
    ret = clEnqueueReadBuffer(command_queue, buffer_result, CL_TRUE, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(int), ocl_result, 0, NULL, NULL);

    // Cleanup OpenCL resources
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(buffer_matrixA);
    ret = clReleaseMemObject(buffer_matrixB);
    ret = clReleaseMemObject(buffer_result);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    // Compare results (to ensure correctness)
    printf("Results comparison (first 10x10 elements):\n");
    for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
            printf("CPU: %d, GPU: %d\n", result[i * MATRIX_SIZE + j], ocl_result[i * MATRIX_SIZE + j]);
        }
    }

    // Free memory
    free(matrixA);
    free(matrixB);
    free(result);
    free(ocl_result);

    return 0;
}

