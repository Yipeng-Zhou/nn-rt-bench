## Shared library "libtensorflowlite.so"

To compile our source code with TensorFlow Lite and RT-Bench together, it's more convenient to link the shared library "libtensorflowlite.so" instead of repeatedly building the source code of TensorFlow Lite. Since there is no official shared library of TensorFlow Lite, we need to build it by ourself. Instead of building it on your computer, just run the the following commands on Colab and download the binaries.

### Clone TensorFlow repository

```
! git clone https://github.com/tensorflow/tensorflow.git
%cd tensorflow
```

### Install Bazel

```
! sudo apt install curl
! curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
! echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
```
```
! sudo apt update && sudo apt install bazel
! sudo apt update && sudo apt install bazel-5.1.1
```

### Compilation

```
# Compilation natively with XNNPACK
! bazel build -c opt //tensorflow/lite:libtensorflowlite.so
```
```
# Cross-compilation for AArch64(ARM64) with XNNPACK
! bazel build --config=elinux_aarch64 -c opt //tensorflow/lite:libtensorflowlite.so
```
```
# Compilation natively without XNNPACK
! bazel build -c opt --define tflite_with_xnnpack=false //tensorflow/lite:libtensorflowlite.so
```
```
# Cross-compilation for AArch64(ARM64) without XNNPACK
! bazel build --config=elinux_aarch64 -c opt --define tflite_with_xnnpack=false //tensorflow/lite:libtensorflowlite.so
```

### Download the binaries

```
from google.colab import files
files.download('bazel-bin/tensorflow/lite/libtensorflowlite.so')
```

In order to make this shared library can be found during compilation, please copy it into the folder "./nn-rt-bench/src/image_classification", "./nn-rt-bench/src/object_detection" and "/usr/lib".
<br/>

## Combining TensorFlow Lite and RT-Bench
Our source code is based on the example from the repository “tensorflow/lite/examples/label_image". According to the requirements of RT-Bench, our source code should consist of three parts: __benchmark_init()__, __benchmark_execution()__ and __benchmark_teardown()__. For more information about RT-Bench, please refer to "https://gitlab.com/bastoni/rt-bench.git".

RT-Bench uses C, while C++ is used by TensorFlow Lite, so we should use "g++" to compile. Since Tensorflow Lite uses some features of C++14 Standard, we should specify "-std=c++14" in our Makefile. Due to compiling C and C++ together, we should make some modification:
1. C++ is designed to be more type safe than C, therefore types cannot convert with each other automatically. That means, we need to convert type in RT-Bench manually.

2. To avoid the error "undefined reference to 'magic_timing_begin'", we need replace "#ifdef GCC" with "#if defined GCC || defined \_\_GNUG\_\_" in "get_cpu_timestamp.h" of "rt-bench/generator" in order to activate this condition for defining the function "magic_timing_begin()" when using g++. 

3. Since C++ has overloading of function names and C does not, the C++ compiler cannot just use the function's name as a unique id to link to. Adding extern "C" will make a function's name in C++ have C linkage, so that C code can link to the function using a C compatible header file that contains just the declaration of the function. That means we need to add extern "C" before declaration of the following functions: 
***
&emsp;"rt-bench/generator/memory_watcher.c": void \*\_\_real_malloc (), void \*\_\_wrap_malloc () and void \*\_\_real_mmap () 
<br/>
&emsp;"rt-bench/generator/periodic_benchmark.h": int benchmark_init (), void benchmark_execution () and void benchmark_teardown () 
<br/>
&emsp;"rt-bench/generator/periodic_benchmark.h": const char\* benchmark_log_header() and float benchmark_log_data() 
<br/>
&emsp;"nn-rt-bench/src/...": int benchmark_init (), void benchmark_execution () and void benchmark_teardown () 

***
4. To avoid the errors "undefined referance to 'timer_create'/'timer_settime'/'timer_delete'", "-lrt" should be after "${BASE_SRC}" in Makefile. 
