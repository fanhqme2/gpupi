#!/bin/bash
set -e

nvcc -O3 -std=c++17 -o batch_bitlength_test batch_bitlength_test.cu batch_bitlength.cu -lgmp -lcurand
