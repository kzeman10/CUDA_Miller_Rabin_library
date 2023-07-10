#include "./../miller_rabin.h"
#include <iostream>
#include <chrono>
#include <cassert>

// g++ -o main test.cpp -L. -lmiller_rabin -Wl,-rpath,.

int test_from_file(const char* filename, bool correctResultValue)
{
    // load the test data
    std::cout << "loading testing data..." << std::endl;
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    u_int64_t *primes = NULL;
    u_int64_t number;
    size_t size = 0;
    while (fscanf(file, "%llu\n", &number) == 1) {
        primes = (u_int64_t *)realloc(primes, (size + 1) * sizeof(u_int64_t));
        primes[size++] = number;
    }
    fclose(file);
    std::cout << "about to test " << size << " numbers" << std::endl;

    // time function and print results
    auto start = std::chrono::high_resolution_clock::now();
    bool* result = millerRabin(primes, size);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // results
    printf("testing took: %.3f sesonds\n\n", duration / 1000);

    // test correctness
    for (int i = 0; i < size; i++) {
        assert (result[i] == correctResultValue);
    }

    // free memory
    free(result);
    free(primes);

    return 0;
}



int main() {

    // load the test data
    std::cout << "note - first warmup run is expected to be longer" << std::endl;
    std::cout << "testing over non primes" << std::endl;
    test_from_file("./data/non_primes.csv", false);

    std::cout << "testing over pseudo primes" << std::endl;
    test_from_file("./data/pseudo_primes.csv", false);

    std::cout << "testing over edge cases" << std::endl;
    test_from_file("./data/edge_cases_false.csv", false);
    test_from_file("./data/edge_cases_true.csv", true);

    std::cout << "testing over sieve of erathostenes for all unsigned numbers < 2**26" << std::endl;
    test_from_file("./data/primes_2**26.csv", true);

    std::cout << "testing over primes ~ 2**43" << std::endl;
    test_from_file("./data/primes_2**43.csv", true);

    std::cout << "testing over primes ~ 2**63" << std::endl;
    test_from_file("./data/primes_2**63.csv", true);

    return 0;
}
