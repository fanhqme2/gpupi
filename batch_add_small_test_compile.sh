#!/bin/bash
set -e

nvcc -O3 -std=c++17 -o batch_add_small_test batch_add_small_test.cu batch_add_small.cu -lgmp -lcurand
