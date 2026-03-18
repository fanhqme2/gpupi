NVCC ?= nvcc
TARGET := pi_bs_gpu
BUILD_DIR := build

NVCCFLAGS ?= -O3
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

SRCS := $(MAIN_SRC) $(LIB_SRCS)
OBJS := $(patsubst %.cu,$(BUILD_DIR)/%.o,$(SRCS))
DEPS := $(OBJS:.o=.d)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR):
	mkdir -p $@

$(BUILD_DIR)/%.o: %.cu | $(BUILD_DIR)
	$(NVCC) $(CPPFLAGS) $(NVCCFLAGS) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

-include $(DEPS)
