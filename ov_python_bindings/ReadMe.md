# OpenVINS Python bindings

## How to build/Run python examples

Clone the open_vins folder.

There is a top level CMakeLists.txt file in the open_vins folder. Build OpenVINS as you would a normal
cmake project without ROS.

```
mkdir build
cd build
cmake ..
```

Run `make PyOpenVINS`. This will build the python bindings. Afterwards you will find a copy of the python library in
your build folder. Should look something like `PyOpenVINS.cpython-38-x86_64-linux-gnu.so`. Copy and paste
this file into your python virtualenv/lib folder. You can now call `import PyOpenVINS` from your
virtualenv. For a tutorial on how to use it see the examples in the ov_python_bindings/python folder.

