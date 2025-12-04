##===- Makefile -----------------------------------------------------------===##
#
# This file licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

include ${srcdir}/../../makefile-common

# ---

XILINX_XRT?=/opt/xilinx/xrt

# ---
UTILS_INCLUDE := -I$(srcdir)/../../../runtime_lib/test_lib/
UTILS_LIB=$(srcdir)/../../../runtime_lib/test_lib/test_utils.cpp
XILINX_XRT_INCLUDE?=${XILINX_XRT}/include
XILINX_XRT_LIB?=${XILINX_XRT}/lib

XRT_FLAGS=-I${XILINX_XRT_INCLUDE} -L${XILINX_XRT_LIB}
XRT_LIBS=-lxrt_coreutil
CXX=g++-13 -ggdb

mlir_aie_dot?=build/aie_dot.mlir
dot_xclbin_target?=build/dot.xclbin
dot_insts_target?=build/dot_insts.bin

host_target?=build/test

DEVICE ?= $(if $(filter 1,$(NPU2)),npu2,npu)

dot_py=dot.py

.PHONY: all
all: ${dot_xclbin_target} ${host_target}

.PHONY: test
test: ${host_target}

.PHONY: test_optimized
test_optimized: ${host_opt_target}

build/dot.o: ${srcdir}/../../../aie_kernels/aie2/dot.cc
	mkdir -p ${@D}
ifeq (${DEVICE}, npu2)
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2P_FLAGS} -c $< -o ${@F}
else
	cd ${@D} && ${PEANO_INSTALL_DIR}/bin/clang++ ${PEANOWRAP2_FLAGS} -c $< -o ${@F}
endif

${mlir_aie_dot}: ${srcdir}/${dot_py}
	mkdir -p ${@D}
	python3 $< -d ${DEVICE} > $@

${dot_xclbin_target}: ${mlir_aie_dot} build/dot.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -v --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
				--no-xchesscc --no-xbridge \
				--aie-generate-npu-insts --npu-insts-name=${dot_insts_target:build/%=%} ${<:%=../%}

${host_target}: ${srcdir}/test.cpp ${dot_xclbin_target}
	mkdir -p ${@D}
	${CXX} ${XRT_FLAGS} ${UTILS_INCLUDE} ${UTILS_LIB}  -o $@ $< ${XRT_LIBS}

.PHONY: run
run: all
	./$(host_target) -i ${dot_insts_target} -x ${dot_xclbin_target} -k MLIR_AIE

.PHONY: perf
perf: all
	./$(host_target) -i ${dot_insts_target} -x ${dot_xclbin_target} -k MLIR_AIE --iters 100 --verify false --warmup 10

# Optimized targets
mlir_aie_dot_opt?=build/aie_dot_opt.mlir
dot_opt_xclbin_target?=build/dot_opt.xclbin
dot_opt_insts_target?=build/dot_opt_insts.bin
host_opt_target?=build/test_opt

dot_opt_py=dot_optimized.py

${mlir_aie_dot_opt}: ${srcdir}/${dot_opt_py}
	mkdir -p ${@D}
	python3 $< -d ${DEVICE} --size 2560 > $@

${dot_opt_xclbin_target}: ${mlir_aie_dot_opt} build/dot.o
	mkdir -p ${@D}
	cd ${@D} && aiecc.py -v --aie-generate-xclbin --no-compile-host --xclbin-name=${@F} \
				--no-xchesscc --no-xbridge \
				--aie-generate-npu-insts --npu-insts-name=${dot_opt_insts_target:build/%=%} ${<:%=../%}

${host_opt_target}: ${srcdir}/test_optimized.cpp ${dot_opt_xclbin_target}
	mkdir -p ${@D}
	${CXX} ${XRT_FLAGS} ${UTILS_INCLUDE} ${UTILS_LIB}  -o $@ $< ${XRT_LIBS}

.PHONY: perf_optimized
perf_optimized: ${dot_opt_xclbin_target} ${host_opt_target}
	@echo "Running perf for size 2560"
	./$(host_opt_target) -i ${dot_opt_insts_target} -x ${dot_opt_xclbin_target} -k MLIR_AIE --size 2560 --iters 100 --verify true --warmup 10
	
	@echo "Running perf for size 6912 (Rebuilding MLIR for 6912)"
	python3 ${srcdir}/${dot_opt_py} -d ${DEVICE} --size 6912 > build/aie_dot_opt_6912.mlir
	cd build && aiecc.py -v --aie-generate-xclbin --no-compile-host --xclbin-name=dot_opt_6912.xclbin \
				--no-xchesscc --no-xbridge \
				--aie-generate-npu-insts --npu-insts-name=dot_opt_insts_6912.bin ../build/aie_dot_opt_6912.mlir
	./$(host_opt_target) -i build/dot_opt_insts_6912.bin -x build/dot_opt_6912.xclbin -k MLIR_AIE --size 6912 --iters 100 --verify true --warmup 10

# 5-column Performance Test
host_perf_target?=build/test_perf

${host_perf_target}: ${srcdir}/test_perf.cpp ${dot_xclbin_target}
	mkdir -p ${@D}
	${CXX} ${XRT_FLAGS} ${UTILS_INCLUDE} ${UTILS_LIB}  -o $@ $< ${XRT_LIBS}

.PHONY: perf_4col
perf_4col: ${host_perf_target}
	@echo "Running perf for size 2560 (4 columns)"
	python3 ${srcdir}/${dot_py} -d ${DEVICE} --size 2560 > build/aie_dot_2560.mlir
	cd build && aiecc.py -v --aie-generate-xclbin --no-compile-host --xclbin-name=dot_2560.xclbin \
				--no-xchesscc --no-xbridge \
				--aie-generate-npu-insts --npu-insts-name=dot_insts_2560.bin ../build/aie_dot_2560.mlir
	./$(host_perf_target) -i build/dot_insts_2560.bin -x build/dot_2560.xclbin -k MLIR_AIE --size 2560 --iters 100 --verify true --warmup 10

	@echo "Running perf for size 6912 (4 columns)"
	python3 ${srcdir}/${dot_py} -d ${DEVICE} --size 6912 > build/aie_dot_6912.mlir
	cd build && aiecc.py -v --aie-generate-xclbin --no-compile-host --xclbin-name=dot_6912.xclbin \
				--no-xchesscc --no-xbridge \
				--aie-generate-npu-insts --npu-insts-name=dot_insts_6912.bin ../build/aie_dot_6912.mlir
	./$(host_perf_target) -i build/dot_insts_6912.bin -x build/dot_6912.xclbin -k MLIR_AIE --size 6912 --iters 100 --verify true --warmup 10

.PHONY: clean
clean:
	-rm -rf build
