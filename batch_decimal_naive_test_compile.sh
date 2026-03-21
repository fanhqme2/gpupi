#!/bin/bash
set -e

nvcc -O3 -std=c++17 -o batch_decimal_naive_test batch_decimal_naive_test.cu batch_decimal_naive.cu -lgmp -lcurand
