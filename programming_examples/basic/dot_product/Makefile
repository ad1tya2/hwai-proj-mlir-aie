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
	./$(host_target) -i ${dot_insts_target} -x ${dot_xclbin_target} -k dot_product_int32_scalar

.PHONY: clean
clean:
	-rm -rf build
