# Gpu

# Setup

hardware
--------
PC i7 Intel core @ 3.60 GHz,  64-bits Operating System, 16 GB memory 
NVIDIA Quadro 600 old card

Software
--------
Windows 7, MSVC Community edition 2013, CUDA SDK installed
Python 3.4.3 :: Anaconda 2.3.0 (64-bit)

# Building
Open the solution with msvc and simply build the Release version.
ImageInterpolation.exe should appear in the Release directory


# Goal of this project
Benchmark interpolation algorithm running on GPU againt their CPU version

# architecture

Design considerations
A simple test applications receiving arguments performs interpolation whose parameters depends on received arguments.
The test application returns to the user a elapsed duration in seconds and a file name

A simple Python script exercise the test app and hence can benchamrk one solution against the other

C++
The Im abstract class represents our image class with two interpolation methods: NN and bilinear
ImCpu and ImGpu implement the interface, with code running respectively on the CPU and on the GPU

The test application uses ImCpu or ImGpu depending on the received arguments. All remaining code stays the same as both classes implement 
the abstract interface

# running the scripts
Once test application is built, simply execute the python script with the following command: 


# Conclusion
You can see the result with [lena]

![Lena](https://github.com/mattvend/Gpu/blob/master/Release/Lena.tiff)
