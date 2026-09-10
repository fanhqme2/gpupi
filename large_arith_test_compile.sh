#!/bin/sh
set -eu
make pi_bs_gpu
${NVCC:-nvcc} -O3 large_arith_test.cu build/batch*.o build/convert_decimal.o -lgmp -o large_arith_test
