/**
 * @file miller_rabin_cuda_32.cu
 *
 * @brief This file serves as a benchmark over Python GPU implementation
 */

#include <iostream>
#include <fstream>
#include <limits.h>
#include <assert.h>
#include <chrono>


/**
 * @brief Performs the square-and-multiply algorithm to calculate a**b % m.
 *
 * This function is executed on a GPU device.
 * This function works properly for all unsigned 32-bit integers - it uses 64-bit integers to prevent overflow.
 *
 * @param a The base value.
 * @param b The exponent value.
 * @param m The modulus value.
 *
 * @return The result of a^b mod m.
 */
__device__ u_int32_t squareAndMultiply(u_int32_t a, u_int32_t b, u_int32_t m) {
    a = a % m;  // Reduce a modulo m to ensure a is within the range [0, m-1]
    u_int32_t result = 1;

    while (b > 0) {
        if (b & 1) {
            // Multiply result with a, and then reduce modulo m
            result = ((uint64_t)result * (uint64_t)a) % m;
        }
        // Square a, and then reduce modulo m
        a = ((uint64_t)a * (uint64_t)a) % m;
        // Right shift b by 1 (equivalent to dividing by 2)
        b >>= 1;
        
    }

    return result;
}


/**
 * @brief Performs the Miller-Rabin primality test to check primality.
 *
 * This test is deterministic for all unsigned 32-bit integers.
 * This function is executed on the GPU device.
 *
 * @param n The number to be tested for primality.
 *
 * @return true if n is prime, false if n is composite.
 */
__device__ bool millerRabin(u_int32_t n) {

    // Predefined bases for uint32
    u_int8_t BASE[] = {2, 7, 61};

    // Check if n is a small prime number
    if (n <= 1) {
        return false;
    }

    // check if n is in the predefined bases
    for (int i = 0; i < 3; i++) {
        if (n == BASE[i]) {
            return true;
        }
    }

    // Check if n is divisible by 2
    if (n % 2 == 0) {
        return false;
    }

    // Write n-1 as 2^r * d where d is odd
    u_int32_t d = n - 1;
    u_int32_t s = 0;
    while ((d & 1) == 0) {
        d >>= 1;
        s++;
    }


    // Perform the Miller-Rabin test with the predefined bases
    for (int i = 0; i < 3; i++) {
        u_int32_t a = BASE[i];
        u_int32_t x = squareAndMultiply(a, d, n);

        if (x == 1 || x == n - 1) {
            // Continue to the next base if x is a witness
            continue;
        }

        bool continueLoop = false;

        for (int j = 0; j < s - 1; j++) {
            x = squareAndMultiply(x, 2, n);

            if (x == 1) {
                // n is composite, return false
                return false;
            }

            if (x == n - 1) {
                // x is a witness, continue to the next base
                continueLoop = true;
                break;
            }
        }

        if (continueLoop) {
            continue;
        }

        // x is not a witness, n is composite
        return false;
    }

    // n passed the Miller-Rabin test for all bases, it is probably prime
    return true;
}


/**
 * @brief CUDA kernel function for parallel prime testing using the Miller-Rabin algorithm.
 *
 * This function is executes the test over the GPU device.
 *
 * @param boolArray A pointer to the boolean array indicating whether each number is prime.
 * @param primes A pointer to the array of numbers to be tested for primality.
 * @param size The size of the array.
 */
__global__ void kernel(bool* boolArray, const uint32_t* primes, const size_t size) {
    // Calculate the global thread ID
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the thread ID is within the valid range
    if (tid < size) {
        // Test the primality of the number at the current thread ID
        boolArray[tid] = millerRabin(primes[tid]);
    }
}


int main() {
    // Load the test data (32-bit primes)
    std::cout << "loading testing data..." << std::endl;
    FILE *file = fopen("./data/primes_2**32.csv", "r");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    uint32_t *primes = NULL;
    uint32_t number;
    size_t size = 0;
    while (fscanf(file, "%u\n", &number) == 1) {
        primes = (uint32_t *)realloc(primes, (size + 1) * sizeof(uint32_t));
        primes[size++] = number;
    }
    fclose(file);
    std::cout << "about to test " << size << " numbers" << std::endl;

    // Warm-up run (32-bit primes)
    std::cout << "note - first warmup run is expected to be longer" << std::endl;

    uint32_t* d_numbers;   // Device memory pointer for numbers
    bool* d_boolArray;     // Device memory pointer for bool array

    // Allocate device memory for numbers and bools
    cudaMalloc((void**)&d_numbers, sizeof(uint32_t) * size);
    cudaMalloc((void**)&d_boolArray, sizeof(bool) * size);

    // Copy the numbers from host to device memory
    cudaMemcpy(d_numbers, primes, sizeof(uint32_t) * size, cudaMemcpyHostToDevice);

    // Set the block size for CUDA kernel execution and calculate the grid size
    constexpr size_t BLOCKSIZE = 256;
    size_t gridSize = (size + BLOCKSIZE - 1) / BLOCKSIZE;

    // Launch the CUDA kernel to test the primality of numbers
    kernel<<<gridSize, BLOCKSIZE>>>(d_boolArray, d_numbers, size);

    // Free the allocated device memory
    cudaFree(d_numbers);
    cudaFree(d_boolArray);

    std::cout << "warm up done" << std::endl;

    // Time function and print results (32-bit primes)
    auto start = std::chrono::high_resolution_clock::now();

    // Allocate host memory for the resulting bool array
    bool* boolArray = (bool*)malloc(sizeof(bool) * size);

    // Allocate device memory for numbers and bools
    cudaMalloc((void**)&d_numbers, sizeof(uint32_t) * size);
    cudaMalloc((void**)&d_boolArray, sizeof(bool) * size);

    // Copy the numbers from host to device memory
    cudaMemcpy(d_numbers, primes, sizeof(uint32_t) * size, cudaMemcpyHostToDevice);

    // Launch the CUDA kernel to test the primality of numbers
    kernel<<<gridSize, BLOCKSIZE>>>(d_boolArray, d_numbers, size);

    // Copy the bool array from device to host memory
    cudaMemcpy(boolArray, d_boolArray, sizeof(bool) * size, cudaMemcpyDeviceToHost);

    // Free the allocated device memory
    cudaFree(d_numbers);
    cudaFree(d_boolArray);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // Results
    printf("testing took: %.3f seconds\n\n", duration / 1000);

    // Test correctness (32-bit primes)
    for (int i = 0; i < size; i++) {
        assert(boolArray[i] == true); // All numbers in primes_2**32.csv are primes
    }

    // Write the result to log.csv file
    std::ofstream logFile("./data/log.csv", std::ios::app);
    if (logFile.is_open()) {
        logFile << "C/C++ CUDA," << duration.count() / 1000.0 << std::endl;
        logFile.close();
    } else {
        std::cerr << "Unable to open log.csv file for writing." << std::endl;
    }

    // Free memory
    free(boolArray);
    free(primes);

    printf("test passed\n");
    return 0;
}
