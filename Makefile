NVCC ?= nvcc
TARGET := pi_bs_gpu
BUILD_DIR := build

CUDA_ARCH ?= sm_120
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(CUDA_ARCH)
CPPFLAGS ?=
LDFLAGS ?=
LDLIBS ?= -lgmp

MAIN_SRC := pi_bs_gpu.cu
LIB_SRCS := \
	batch_arith.cu \
	batch_mul_ntt.cu \
	batch_add.cu \
	batch_sub.cu \
	batch_shift_add.cu \
	batch_shift_sub.cu \
	batch_mul_small.cu \
	batch_exactdiv_small.cu \
	batch_add_small.cu \
	batch_sub_small.cu \
	batch_mul_naive.cu \
	batch_bitlength.cu \
	batch_shift.cu

LIB_SRCS += \
	convert_decimal.cu \
	batch_decimal_naive.cu

SRCS := $(MAIN_SRC) $(LIB_SRCS)
OBJS := $(patsubst %.cu,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d) $(BUILD_DIR)/pi_mgpu.d $(BUILD_DIR)/distributed_ntt.d $(BUILD_DIR)/distributed_ntt_test.d

.PHONY: all clean

all: $(TARGET) pi_mgpu

pi_mgpu: $(BUILD_DIR)/pi_mgpu.o $(BUILD_DIR)/distributed_ntt.o $(filter-out $(BUILD_DIR)/pi_bs_gpu.o,$(OBJS))
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $^ -lgmpxx $(LDLIBS) -Xcompiler -pthread -o $@

wide_int_test: wide_int_test.cpp wide_int.h host_arith.h host_topology.h
	$(CXX) -O3 -std=c++17 $< -lgmpxx -lgmp -pthread -o $@

distributed_ntt_test: $(BUILD_DIR)/distributed_ntt_test.o $(BUILD_DIR)/distributed_ntt.o $(BUILD_DIR)/batch_mul_ntt.o
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $^ -lgmpxx $(LDLIBS) -Xcompiler -pthread -o $@

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/%.o: %.cu | $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET) pi_mgpu distributed_ntt_test wide_int_test

-include $(DEPS)
