#!/bin/bash
set -e

nvcc -O3 -std=c++17 -o batch_sub_small_test batch_sub_small_test.cu batch_sub_small.cu -lgmp -lcurand
