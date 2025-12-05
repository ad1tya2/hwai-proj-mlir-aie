1) 
End 2 end inference
generate the libdot.so file from going to programming_examples/basic/dot_product

place this .so file and the build folder in the root of BitNet repo
rename build folder to npu_build

2) 
run programming_examples/basic/matrix_multiplication/whole_array_i8i8 with make run to get an approx baseline performance.

run standard whole_array folder for i2_i8 performance.
Use the generated binary after fixing dimensions and call it instead of the generated gpu kernel in bitnet/gpu