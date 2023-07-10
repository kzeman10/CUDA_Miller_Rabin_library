#!/bin/bash
g++ -o test test.cpp -L./.. -lmiller_rabin -Wl,-rpath,./..
./test
rm test 