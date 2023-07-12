#include <stdio.h>
#include <gmp.h>
#include <iostream>
#include <time.h>
#include <vector>
#include <limits>
#include <random>


#define sqrt_uint_64_max 4294967296 // (2**64 - 1) ** 0.5
#define sqrt_uint_128_max 18446744073709551615 // (2**64 - 1) ** 0.5
#define range 63

u_int64_t squareAndMultiply(u_int64_t a, u_int64_t b, u_int64_t m) {
    a = a % m;  // Reduce a modulo m to ensure a is within the range [0, m-1]
    u_int64_t result = 1;

    while (b > 0) {
        if (b & 1) {
            // Multiply result with a, and then reduce modulo m
            result = ((__uint128_t)result * (__uint128_t)a) % m;
        }
        // Square a, and then reduce modulo m
        a = ((__uint128_t)a * (__uint128_t)a) % m;
        // Right shift b by 1 (equivalent to dividing by 2)
        b >>= 1;
        
    }

    return result;
}


int main() {

    printf("GMP version: %s\n", gmp_version);

    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dis(0, (1ULL << range) - 1);

    const int size = 4096;
    std::vector<uint64_t> bases(size);
    std::vector<uint64_t> exponents(size);

    for (int i = 0; i < size; ++i) {
        bases[i] = dis(gen);
        exponents[i] = dis(gen);
    }

    // Perform base^exponent operation and compare results
    uint64_t modulus = 836387250685805521; // max
    
    std::cout << "Results: " << std::endl;
    for (int i = 0; i < size; ++i) {
        uint64_t base = bases[i];
        uint64_t exponent = exponents[i];
        
        // Calculate the result using squareAndMultiply function
        uint64_t result = squareAndMultiply(base, exponent, modulus);

        // Calculate the result using GMP's modular exponentiation function (mpz_powm_ui)
        mpz_t base_mpz, exponent_mpz, modulus_mpz, result_mpz;
        mpz_inits(base_mpz, exponent_mpz, modulus_mpz, result_mpz, NULL);

        mpz_set_ui(base_mpz, base);
        mpz_set_ui(exponent_mpz, exponent);
        mpz_set_ui(modulus_mpz, modulus);

        mpz_powm_ui(result_mpz, base_mpz, exponent, modulus_mpz);
        uint64_t gmpResult = mpz_get_ui(result_mpz);

        // Compare the results
        if (result != gmpResult) {
            std::cout << base << "**" << exponent << "mod" << modulus << " = " << result << " -- " << "ERROR" << std::endl;
            std::cout << "GMP result: " << gmpResult << std::endl;
            mpz_clears(base_mpz, exponent_mpz, modulus_mpz, result_mpz, NULL);
            return 1;
        }
        // Print the result
        std::cout << base << "**" << exponent << " mod " << modulus << " = " << gmpResult << " -- " << "OK" << std::endl;

        mpz_clears(base_mpz, exponent_mpz, modulus_mpz, result_mpz, NULL);
    }
    std::cout << "All square and multiply tests passed!" << std::endl;

    return 0;
}
