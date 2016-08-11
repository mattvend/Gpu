# Gpu

## Goal of this project
Benchmark interpolation algorithms running on GPU againt their CPU version.  
Of course, I had to use Lena for this:
![Lena][Lena]

For the moment, benchmarking is only done with the Nearest Neighbor and Bilinear interpolations

# My Setup

hardware
--------
- CPU: PC i7 Intel core @ 3.60 GHz,  64-bits Operating System, 16 GB memory 
- GPU: NVIDIA Quadro 600, which is a really old card

Software
--------
- Windows 7, MSVC Community edition 2013, CUDA SDK installed
- Python 3.4.3 :: ![Anaconda][Anaconda] distribution 2.3.0 (64-bit)

## Building

Open the solution with msvc and simply build the Release version.
ImageInterpolation.exe should appear in the Release directory

## Architecture and design considerations

Design considerations
A simple test applications receiving arguments performs interpolation whose parameters depends on received arguments.
The test application returns to the user an elapsed duration in seconds and a file name

A simple Python script exercise the test app and hence can benchamrk one solution against the other

C++
The Im abstract class represents our image class with two interpolation methods: NN and bilinear
ImCpu and ImGpu implement the interface, with code running respectively on the CPU and on the GPU

The test application uses ImCpu or ImGpu depending on the received arguments. All remaining code stays the same as both classes implement the same abstract interface

## Running the scripts
Once test application is built, simply execute the python script with the following command: 


## Conclusion
Here are the results
![Results][Results]

## Comments
1. The Gpu version of the NN interpolation is only 2 times faster when the Bilinear interpolation runs 3 times faster.
  * it is a bit disapointing as I was expecting better performances on the GPU for both interpolations
2. I only did the tests for the Lena image (512*512), interpolated to a (8000,4000) image.
3. Both interpolations done in a naive way

Possibles explanations
1. Cpu vs Gpu seems to be tricky as the figure obtained depends obviously on the setup. In my case, as the GPU used is an old one (5 years older than the CPU, huge difference in the tech world), it makes sense to have a CPU that can compete against a GPU.
2. GPU Optimisation could probably be better, using intrinsics should help.


## Bonus
I found out the ![xkcd][xkcd] style while playing with Matplotlib ... I could not help myself to use it. Don't forget to checkout the Python script to activate it.


[Lena]: http://www.cosy.sbg.ac.at/~pmeerw/Watermarking/lena_color.gif "Lena"
[Results]: /Release/CpuVsGpu.png
[xkcd]: http://xkcd.com/
[Anaconda]: https://www.continuum.io/why-anaconda
