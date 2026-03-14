#!/bin/bash
set -e

nvcc -O3 -std=c++17 -o batch_sub_test batch_sub_test.cu batch_sub.cu -lgmp -lcurand
