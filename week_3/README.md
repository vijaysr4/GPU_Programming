# CUDA Matrix Multiplication

This repository contains several implementations of single-precision matrix multiplication using CUDA. The implementations include:
  
- A CPU-based reference implementation.
- A basic GPU kernel where each thread computes one element of the result matrix.
- A tiled GPU kernel that leverages shared memory to optimize memory accesses.
- A GPU implementation using the highly optimized cuBLAS library.

All implementations follow the SGEMM (Single-precision General Matrix Multiply) formula:
  
\[
C = \alpha \times (A \times B) + \beta \times C,
\]
  
where:
- \(\alpha\) scales the product \(A \times B\).
- \(\beta\) scales the existing contents of \(C\) (allowing accumulation).

> **Note:** This project is configured for an NVIDIA RTX 4000 Ada Generation GPU. Therefore, the code is compiled with the architecture flag `-arch=sm_89` and uses C++14.

## Compilation and Execution

### Build the Executable

To compile the project, run the following command from the project directory:

```bash
nvcc -std=c++14 -arch=sm_89 -I/usr/local/cuda/include -lcublas sgemm.cu fmatrix.cu cuda_stuff.cu main.cu -o main
```

This command:

- Uses ```C++14 (-std=c++14)``` for host code.
- Targets the RTX 4000 GPU with ```-arch=sm_89``` (instead of ```-arch=sm_75``` which  would be used for a T4 GPU).
- Specifies the CUDA include directory with ```-I/usr/local/cuda/include```.
- Links the cuBLAS library with ```-lcublas```.

### Set the Library Path
Before running the executable, ensure that the dynamic linker can find the CUDA libraries:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Run the Executable
After compiling and setting the library path, run the executable with:

```bash
./main
```

### Profiling with NVIDIA Nsight Systems
To profile the application and collect detailed CUDA and OS runtime statistics, run:

```bash
nsys profile --trace=cuda,osrt --stats=true ./main
```

This command generates a detailed report (e.g., a ```.qdstrm``` or ```.sqlite``` file) that you can analyze using the Nsight Systems GUI.

### Debugging with cuda-gdb
To compile with debugging information, run:


```bash
nvcc -g -G -I /usr/local/cuda/samples/common/inc/ -L/usr/local/cuda/include -lcublas -lcusolver sgemm.cu fmatrix.cu cuda_stuff.cu main.cu -o a.out

```
Then, prepare a debugging command file:

```bash
printf "set cuda api_failures stop\ncatch throw\nr UNIT\nbt\ninfo locals\nthread 1\nbt\n" > tmp.txt
cat tmp.txt
```

Finally, run cuda-gdb in batch mode using the command file:

```bash
cuda-gdb -batch -x tmp.txt ./a.out
```

This setup will start the debugger, stop at the first CUDA API failure or exception, and print the call stack and local variables.