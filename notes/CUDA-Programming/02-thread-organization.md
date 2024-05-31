# CUDA 中的线程组织

## CUDA 中的 Hello World 程序

CUDA 程序的编译器驱动（compiler driver）`nvcc` 支持编译纯粹的 C++ 代码。`nvcc` 在编译一个 CUDA 程序时，会将纯粹的 C++ 代码交给 C++ 的编译器去处理，它自己则负责编译剩下的部分。

一个真正利用了 GPU 的 CUDA 程序既有主机代码，也有设备代码。主机对设备的调用是通过核函数（kernel function）来实现的。

一个典型的、简单的 CUDA 程序的结构具有下面的形式：

```c++
int main(void)
{
    主机代码
    核函数的调用
    主机代码
    return 0;
}
```

CUDA 中的核函数与 C++ 中的函数类似，但它必须被限定词（qualifier）`__global__` 修饰，且返回类型必须是 `void`。

```c++
__global__ void hello_from_gpu()
{
    printf("Hello World from the GPU!\n");
}
```

其中 `__global__` 和 `void` 也可调换顺序。

调用核函数的例子：

```c++
hello_from_gpu<<<1, 1>>>();
// hello_from_gpu<<<网格大小，线程块大小>>>();
```

与普通 C++ 函数的调用格式有区别。主机在调用一个核函数时，必须指明需要在设备中指派多少个线程，常常组织为若干线程块（thread block）。三括号中的第一个数字为线程块的个数，第二个数字为每个线程块中的线程数。

一个核函数的全部线程块构成一个网格（grid），线程块的个数记为网格大小（grid size）。每个线程块中含有同样数目的线程，该数目称为线程块大小（block size）。

```c++
cudaDeviceSynchronize();
```

调用 CUDA 运行时 API 函数，作用是同步主机与设备，促使缓冲区刷新。

## 线程组织

核函数允许指派很多线程，对于 `hello3.cu`，我们往往想知道每一行输出对应哪一个线程。

这些线程的组织结构是由执行配置（execution configuration）`<<<grid_size, block_size>>>` 决定。

从开普勒架构开始，最大允许的线程块大小是 $1024$，而最大允许的网格大小是 $2^{32} - 1$。一般来说，只要线程数比 GPU 中的计算核心数（几百至几千个）多几倍时，就有可能充分利用 GPU 中的全部计算资源。

每个线程在核函数中都有一个唯一的身份标识。在核函数内部，`grid_size` 和 `block_size` 的值分别保存于如下两个内建变量（built-in variable）:

- `gridDim.x` 等于 `grid_size`
- `blockDim.x` 等于 `block_size`

类似有

- `blockIdx.x` 指定在一个网格中的线程块指标，`[0, gridDim.x - 1]`.
- `threadIdx.x` 指定在一个线程块中的线程指标，`[0, blockDim.x - 1]`.

可以用 `dim3` 的定义“多维”的网络和网格块。

网格大小在 x, y 和 z 这 3 个方向的最大允许值分别为 $2^{31} - 1$, $65535$ 和 $65535$，线程块大小在 x, y 和 z 这 3 个方向的最大允许值分别为 $1024$, $1024$ 和 $64$。

## 用 nvcc 编译 CUDA

使用 `nvcc` 编译器驱动编译 `.cu` 文件时，将自动包含必要的 CUDA 头文件，如 `<cuda.h>` 和 `<cuda_runtime.h>`。

`nvcc` 将源代码分离为主机代码和设备代码，主机代码完整地支持 C++ 语法，设备代码只部分支持 C++。`nvcc` 先将设备代码编译为 PTX（Parallel Thread eXecution）伪汇编代码，再将 PTX 代码编译为二进制 cubin 目标代码。

将源代码编译为 PTX 代码时，需要用选项 `-arch=compute_XY` 指定一个虚拟架构的计算能力，用以确定代码中能够使用的 CUDA 功能。在将 PTX 代码编译为 cubin 代码时，需要用选项 `-code=sm_ZW` 指定一个真实架构的计算能力，用以确定可执行文件能够使用的 GPU。`ZW` 必须大于等于 `XY`。

选项 `-code=sm_ZW` 指定了 GPU 的真实架构为 `Z.W`，对应的可执行文件只能在主版本号为 `Z`、次版本号大于或等于 `W` 的 GPU 中运行。

使用 `-gencode`:

```bash
-gencode arch=compute_35,code=sm_35
-gencode arch=compute_50,code=sm_50
-gencode arch=compute_60,code=sm_60
-gencode arch=compute_70,code=sm_70
```

编译出的可执行文件将包含四个二进制版本，这样的可执行文件称为胖二进制文件（fatbinary）。

## nvcc 的编译流程

![nvcc-compilation](./img/cuda-compilation-from-cu-to-executable.png)

