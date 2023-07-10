#!/bin/bash
g++ -o example example.cpp -L./.. -lmiller_rabin -Wl,-rpath,./..
./example
rm example
 