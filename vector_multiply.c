#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <CL/cl.h>

#define CL_TARGET_OPENCL_VERSION 300  // Define OpenCL version to avoid warning
int MATRIX_SIZE=4096;
int VECTOR_SIZE=4096;

// OpenCL kernel for matrix-vector multiplication
const char *kernelSource = 
    "__kernel void matvec_mult(__global const int* matrix, __global const int* vector, __global int* result, int n) { \n"
    "    int row = get_global_id(0); \n"
    "    int sum = 0; \n"
    "    for (int col = 0; col < n; col++) { \n"
    "        sum += matrix[row * n + col] * vector[col]; \n"
    "    } \n"
    "    result[row] = sum; \n"
    "} \n";

// Software-based matrix-vector multiplication
void software_matrix_vector_multiplication(int *matrix, int *vector, int *result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = 0;
        for (int j = 0; j < size; j++) {
            result[i] += matrix[i * size + j] * vector[j];
        }
    }
}

// Function to print a small portion of vectors (for debugging or testing)
void print_vector(int *vector, int size, int elements_to_print) {
    for (int i = 0; i < elements_to_print; i++) {
        printf("%d ", vector[i]);
    }
    printf("\n");
}

// Main function
int main() {
    // Allocate memory for the matrix, vector, and result
    int *matrix = (int*)malloc(sizeof(int) * MATRIX_SIZE * MATRIX_SIZE);
    int *vector = (int*)malloc(sizeof(int) * VECTOR_SIZE);
    int *result = (int*)malloc(sizeof(int) * VECTOR_SIZE);
    int *ocl_result = (int*)malloc(sizeof(int) * VECTOR_SIZE);

    // Initialize the matrix and vector with arbitrary values
    for (int i = 0; i < MATRIX_SIZE; i++) {
        vector[i] = i;  // Example initialization of vector
        for (int j = 0; j < MATRIX_SIZE; j++) {
            matrix[i * MATRIX_SIZE + j] = i + j;  // Example initialization of matrix
        }
    }

    // --- SOFTWARE MULTIPLICATION ---
    clock_t start_cpu = clock();
    software_matrix_vector_multiplication(matrix, vector, result, MATRIX_SIZE);
    clock_t end_cpu = clock();
    double cpu_time_used = ((double)(end_cpu - start_cpu)) / CLOCKS_PER_SEC;

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

    // Create memory buffers on the device for the matrix, vector, and result
    cl_mem buffer_matrix = clCreateBuffer(context, CL_MEM_READ_ONLY, MATRIX_SIZE * MATRIX_SIZE * sizeof(int), NULL, &ret);
    cl_mem buffer_vector = clCreateBuffer(context, CL_MEM_READ_ONLY, VECTOR_SIZE * sizeof(int), NULL, &ret);
    cl_mem buffer_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, VECTOR_SIZE * sizeof(int), NULL, &ret);

    // Copy the matrix and vector to the GPU memory
    ret = clEnqueueWriteBuffer(command_queue, buffer_matrix, CL_TRUE, 0, MATRIX_SIZE * MATRIX_SIZE * sizeof(int), matrix, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, buffer_vector, CL_TRUE, 0, VECTOR_SIZE * sizeof(int), vector, 0, NULL, NULL);

    // Create OpenCL program and build it
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &ret);
    ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "matvec_mult", &ret);

    // Set kernel arguments
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffer_matrix);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buffer_vector);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buffer_result);
    ret = clSetKernelArg(kernel, 3, sizeof(int), &MATRIX_SIZE);  // Passing by value

    // Execute the kernel
    size_t global_item_size = MATRIX_SIZE; // Each work item handles one row of the matrix
    clock_t start_ocl = clock();
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, NULL, 0, NULL, NULL);
    ret = clFinish(command_queue);
    clock_t end_ocl = clock();
    double ocl_time_used = ((double)(end_ocl - start_ocl)) / CLOCKS_PER_SEC;

    printf("CPU matrix-vector multiplication took: %f seconds\n", cpu_time_used);
    printf("OpenCL matrix-vector multiplication took: %f seconds\n", ocl_time_used);
    printf("Speedup: x%.2f\n", cpu_time_used / ocl_time_used); 

    // Copy the result from the GPU memory to the host memory
    ret = clEnqueueReadBuffer(command_queue, buffer_result, CL_TRUE, 0, VECTOR_SIZE * sizeof(int), ocl_result, 0, NULL, NULL);

    // Cleanup OpenCL resources
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(buffer_matrix);
    ret = clReleaseMemObject(buffer_vector);
    ret = clReleaseMemObject(buffer_result);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);

    // Compare results (to ensure correctness)
    printf("Results comparison (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("CPU: %d, GPU: %d\n", result[i], ocl_result[i]);
    }

    // Free memory
    free(matrix);
    free(vector);
    free(result);
    free(ocl_result);

    return 0;
}