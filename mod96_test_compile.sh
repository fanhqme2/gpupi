#!/bin/bash
set -euo pipefail
${NVCC:-nvcc} -O3 mod96_test.cu -lgmp -o mod96_test
