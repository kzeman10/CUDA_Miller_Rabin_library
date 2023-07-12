#!/bin/bash
echo "running safe_modular_multiply_test_uint128.cpp"
g++ -lgmp safe_modular_multiply_test_uint128.cpp
./a.out
rm a.out

echo
echo "running miller_rabin_test.cpp"
g++ -o a.out miller_rabin_test.cpp -L./.. -lmiller_rabin -Wl,-rpath,./..
./a.out
rm a.out
