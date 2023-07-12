#include <stdio.h>
#include <iostream>
#include <fstream>
#include <cassert>
#include <time.h>
#include <gmp.h>
#include "./../miller_rabin.h"


void print_progress(size_t current, size_t total) {
    // Calculate progress percentage
    float progress = static_cast<float>(current) / total * 100;

    // Print progress with carriage return to overwrite the previous line
    std::cout << "Progress: " << current << "/" << total << " (" << progress << "%)\r";
    std::cout.flush();  // Flush output to display immediately
}


int main() {

    printf("GMP version: %s\n", gmp_version);

    printf("loading test data...\n");
    // load test data
    FILE *file = fopen("./data/primes_2**32.csv", "r");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    uint64_t *primes = NULL;
    uint64_t number;
    size_t size = 0;
    while (fscanf(file, "%llu,", &number) == 1) {
        primes = (uint64_t *)realloc(primes, (size + 1) * sizeof(uint64_t));
        primes[size++] = number;
    }

    fclose(file);

    // create new log file ./data/log.csv as c++ stream and print header
    std::ofstream log_file("./data/log.csv");
    log_file << "implementation,time" << std::endl;


    std::cout << "about to test " << size << " primes" << " on a GPU" << std::endl;
    std::cout << "warmup..." << std::endl;
    bool *result = millerRabin(primes, size);
    free(result);
    std::cout << "warmup done" << std::endl;

    clock_t start_time, end_time;
    double elapsed_time;
    start_time = clock();
    
    result = millerRabin(primes, size);

    end_time = clock();
    elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Time taken: %.3f seconds\n", elapsed_time);
    log_file << "GPU," << elapsed_time << std::endl;
    // test correctness
    for (int i = 0; i < size; i++) {
        assert (result[i] == true);
    }


    mpz_t x;
    mpz_init(x);

    // Define the interval for progress printing
    size_t interval = 100000;

    start_time = clock();

    std::cout << "about to test " << size << " primes" << " on a CPU" << std::endl;
    for (size_t i = 0; i < size; i++) {
        mpz_set_ui(x, primes[i]);
        char is_prime = mpz_millerrabin(x, 13);
        if (is_prime == 0 && primes[i] != 2) {
            std::cout << "Result: " << static_cast<int>(is_prime) << std::endl;
            std::cout << "Failed on " << primes[i] << std::endl;
            return 1;
        }

        // Print progress every interval primes
        if (i % interval == 0) {
            print_progress(i, size);
        }
    }

    end_time = clock();
    elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Time taken: %.3f seconds\n", elapsed_time);
    log_file << "CPU," << elapsed_time << std::endl;

    free(primes);
    free(result);
    mpz_clear(x);
    return 0;
}
