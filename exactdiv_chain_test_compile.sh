#!/bin/bash
set -euo pipefail
make -j8
${NVCC:-nvcc} -O3 exactdiv_chain_test.cu build/batch*.o build/convert_decimal.o -lgmp -o exactdiv_chain_test
