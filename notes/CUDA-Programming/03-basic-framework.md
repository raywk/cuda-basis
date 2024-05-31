# 简单 CUDA 程序的基本框架

现实的中、大型程序中，往往使用多个源文件。我们现在只考虑一个源文件的情况，此时典型的 CUDA 程序的基本框架如：

```c++
头文件
常量定义或宏定义
C++ 自定义函数和 CUDA 核函数的声明（原型）
int main(void)
{
    分配主机与设备内存
    初始化主机中的数据
    将某些数据从主机复制到设备
    调用核函数在设备中进行计算
    将某些数据从设备复制到主机
    释放主机与设备内存
}
C++ 自定义函数和 CUDA 核函数的定义（实现）
```

`cudaMalloc()` 函数用于分配设备内存，该函数是一个 CUDA 运行时 API 函数，所有 CUDA 运行时 API 函数都以 `cuda` 开头。

该函数原型：

```c++
cudaError_t cudaMalloc(void **address, size_t size);
```

第一个参数是待分配设备内存的指针，由于内存地址本身就是一个指针，所有该参数为双重指针。第二个参数是待分配内存的字节数。返回值是一个错误代号，如果成功则返回 `cudaSuccess`。

对应地有：

```c++
cudaError_t cudaFree(void* address);
```

分配了设备内存之后，就可以将某些数据从主机传递到设备中去了。

```c++
cudaError_t cudaMemcpy
{
    void                *dst,
    const void          *src,
    size_t              count,
    enum cudaMemcpyKind kind
}
```

第一个参数是目标地址，第二个参数是源地址，第三个参数是复制数据的字节数，第四个参数为一个枚举类型的变量，标志数据传递方向。

