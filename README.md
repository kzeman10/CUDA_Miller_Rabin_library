# CUDA_Miller_Rabin_library

CUDA library with minimal dependencies and ease of use implementing deterministic Miller-Rabin primality test for uint_64 optimized for testing wide ranges of numbers on GPU.

This library aims to efficiently parallelize the deterministic Miller-Rabin primality test over multiple numbers simultaneously, not to implement the quickest Miller-Rabin to test a single integer. So in order to achieve maximal efficiency, use as huge arrays as possible.

## How to run

See the example_run folder with prepared script.

## Benchmark

This implementation tested all primes in unsigned int 32 range (203 million primes) in 3 seconds on Nvidia RTX 3090.
