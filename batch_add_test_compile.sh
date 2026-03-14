#!/bin/bash
set -e

nvcc -O3 -std=c++17 -o batch_add_test batch_add_test.cu batch_add.cu -lgmp -lcurand
