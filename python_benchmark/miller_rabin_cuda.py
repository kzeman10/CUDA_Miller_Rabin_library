"""Miller-Rabin primality test on the GPU using CUDA."""
import time
import numpy as np
import numba as nb
from numba import cuda


@cuda.jit
def square_and_multiply(base: nb.uint32, exponent: nb.uint32, modulus: nb.uint32) -> nb.uint32:
    """Compute (base ** exponent) % modulus efficiently"""
    result = 1
    while exponent > 0:
        if exponent % 2 == 1:
            result = nb.uint32(nb.uint64(nb.uint64(result) * nb.uint64(base)) % modulus)
        base = nb.uint32(nb.uint64((nb.uint64(base) * nb.uint64(base))) % modulus)
        exponent //= 2
    return result


@cuda.jit
def miller_rabin(n : nb.uint32) -> nb.boolean:
    """deterministic Miller-Rabin primality test"""
    if n in [2, 7, 61]:
        return True
    if n <= 1 or n % 2 == 0:
        return False

    # Write n-1 as 2^r * d where d is odd
    r = 0
    d = n - 1
    while d % 2 == 0:
        r += 1
        d //= 2

    # Perform Miller-Rabin test for each base
    for a in [2, 7, 61]:
        x = square_and_multiply(a, d, n)
        if x in [1, n - 1]:
            continue
        for _ in range(r - 1):
            x = square_and_multiply(x, 2, n)
            if x == n - 1:
                break
        else:
            # print(f'a = {a}, a**d mod n ... {a}**{d} mod {n}')
            return False
    return True


@cuda.jit
def miller_rabin_kernel(numbers, is_prime):
    """Perform Miller-Rabin primality test on each number in the array"""
    i = cuda.grid(1)
    if i < numbers.size:
        number = numbers[i]
        is_prime[i] = miller_rabin(number)


# load an array of prime numbers and copy to the device
print('loading testing data...')
primes = np.loadtxt('./data/primes_2**32.csv', dtype=np.uint32, delimiter=',')
print(f'about to test {len(primes)} primes')
primes_device = cuda.to_device(primes)

# Create an output array on device
is_prime_device = cuda.device_array(primes.shape, dtype=np.bool_)

# Set up the CUDA kernel configuration
BLOCK_SIZE = 256
grid_size = (primes.size + BLOCK_SIZE - 1) // BLOCK_SIZE

# warm up
print('warming up')
miller_rabin_kernel[grid_size, BLOCK_SIZE](primes_device, is_prime_device)
cuda.synchronize()
print('warm up done')

# start time
start_time = time.time()

# copy to device
primes_device = cuda.to_device(primes)
# Launch the CUDA kernel
miller_rabin_kernel[grid_size, BLOCK_SIZE](primes_device, is_prime_device)
# Copy result back to host
is_prime_host = is_prime_device.copy_to_host()

# end time
total_time = time.time() - start_time
print(f'testing took: {total_time:.3f} seconds')


# check if the results are correct
assert sum(is_prime_host) == len(primes), f'primes {primes[~is_prime_host]} did not pass the test'

# Write the result to log.csv file
with open('./data/log.csv', 'a') as log_file:
    log_file.write(f'CUDA by Numba,{total_time:.3f}\n')

print('all ok')
