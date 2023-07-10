#include "./../miller_rabin.h"
#include <iostream>

int main() {
    // Example of library usage
    const size_t arraySize = 10;  // Size of the array
    uint64_t numbers[arraySize] = {2, 3, 4, 11, 13, 17, 19, 50, 97, 100};  // Example numbers

    bool* result = millerRabin(numbers, arraySize);

    for (size_t i = 0; i < arraySize; i++) {
        if (result[i]) {
            std::cout << numbers[i] << " is prime." << std::endl;
        } else {
            std::cout << numbers[i] << " is not prime." << std::endl;
        }
    }

    // Free the memory allocated for the bool array - caller is responsible for this
    free(result);

    return 0;
}
