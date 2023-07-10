// This code is distributed under the terms of the GNU General Public License (GPL) version 3.
// For more information and the source code, visit:
// <https://github.com/kzeman10/CUDA_Miller_Rabin_library>

#ifndef MILLER_RABIN_H
#define MILLER_RABIN_H

#include <cstddef>
#include <cstdint>

/**
 * @file miller_rabin_cuda.h
 *
 * @brief This module provides functions for performing the deterministic Miller-Rabin primality test.
 *        This version of the test is deterministic for all unsigned 64-bit integers.
 *        The test is accelerated on a GPU through CUDA.
 *        Run the test on a wide range of numbers for maximizing performance.
 */


/**
 * @brief Tests whether each number in the given array of numbers is prime.
 *
 * This function utilizes CUDA to parallelize the prime testing on the GPU.
 *
 * @param numbers A pointer to an array of numbers to be tested for primality.
 * @param size The size of the array.
 *
 * @return A pointer to a bool array indicating whether each number is prime.
 *         The caller is responsible for freeing the memory allocated for the bool array.
 */
bool* millerRabin(uint64_t* numbers, size_t size);

#endif  // MILLER_RABIN_H
