#!/bin/bash
set -euo pipefail
make -j8
${NVCC:-nvcc} -O3 shared_mul_test.cu build/batch*.o build/convert_decimal.o -lgmp -o shared_mul_test
