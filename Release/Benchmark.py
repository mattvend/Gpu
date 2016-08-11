from PIL import Image
import os
import filecmp
import subprocess
import re
import numpy as np
import matplotlib.pyplot as plt


def convert_to_raw(file):

    img = Image.open(file)
    img = img.convert('L')  # convert to 8 bits per pixels
    (x, y) = img.size

    pixels = bytearray(list(img.getdata()))

    filename, file_extension = os.path.splitext(file)
    file2 = file.replace(file_extension, '.dat')
    file_name = str(x) + 'x' + str(y) + 'x8x1' + '_' + file2

    # print(file_name)

    with open(file_name, 'wb') as f:
        f.write(pixels)

    return file_name

def loop_over_binary_file(file):

    with open(file, "rb") as imageFile:
        f = imageFile.read()
        pixels = bytearray(f)

    for p in pixels:
        print(p)



def convert_to_jpg(raw_file):


    match = re.match('(\d+)x(\d+)x(\d+)x(\d+)_(\w+)', raw_file)

    if match:
        print(match.group(1))
        print(match.group(2))
        print(match.group(3))
        print(match.group(4))
        print(match.group(5))
        x = int(match.group(1))
        y = int(match.group(2))
        bpp = int(match.group(3))
        dimension = int(match.group(4))
        filename = match.group(0)

    rawData = open(raw_file, 'rb').read()
    imgSize = (x, y)
    # Use the PIL raw decoder to read the data.
    # the 'F;16' informs the raw decoder that we are reading
    # a little endian, unsigned integer 16 bit data.
    #img = Image.fromstring('L', imgSize, rawData, 'raw', 'F;32')


    img = Image.frombuffer('L', imgSize, rawData, 'raw')
    img = img.rotate(180)
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img.save(filename + ".jpg")


def interpolate(file_in, file_out, device, iterations, interpolation_type, new_width, new_height):

    command_string = 'ImageInterpolation.exe ' + device + ' ' + str(iterations) + ' ' +  interpolation_type + ' ' + file_in + ' ' + file_out + ' ' + str(new_width) + ' ' + str(new_height) 

    program_out = str(subprocess.check_output(command_string.split(), stderr=subprocess.STDOUT),'utf-8')
    print(program_out)
    program_out = program_out.splitlines()
    seconds  = float(program_out[8])
    out_file = program_out[9]

    return (seconds, out_file)


def benchmark_cpu_vs_gpu(input_raw_file):

    (cpu1, f1) = interpolate(input_raw_file, 'cpu_nn_lena.dat', 'cpu', 20, 'nn', 8000, 4000)
    (gpu1, f2) = interpolate(input_raw_file, 'gpu_nn_lena.dat', 'gpu', 20, 'nn', 8000, 4000)
    (cpu2, f3) = interpolate(input_raw_file, 'cpu_bl_lena.dat', 'cpu', 20, 'bl', 8000, 4000)
    (gpu2, f4) = interpolate(input_raw_file, 'gpu_bl_lena.dat', 'gpu', 20, 'bl', 8000, 4000)

    return ((cpu1,cpu2),(gpu1,gpu2))


def plot_graph(durations):
    
    with plt.xkcd():

        N = 2
        # cpuMeans = (1.218, 10.303)
        cpuMeans = durations[0]

        ind = np.arange(N)  # the x locations for the groups
        width = 0.35       # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(ind, cpuMeans, width, color='r')

        # gpuMeans = (0.669, 3.46)
        gpuMeans = durations[1]
        
        rects2 = ax.bar(ind + width, gpuMeans, width, color='y')

        # add some text for labels, title and axes ticks
        ax.set_ylabel('Time in sec')
        ax.set_title('Duration by interpolation type and device type')
        ax.set_xticks(ind + width)
        ax.set_xticklabels(('Nearest Neighbor', 'Bilinear'))

        ax.legend((rects1[0], rects2[0]), ('Cpu', 'Gpu'))


        def autolabel(rects):
            # attach some text labels
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                        '%.2f' % height,
                        ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)

        # plt.show()
        plt.savefig('CpuVsGpu.png')

def check_bit_exactness(input_raw_file):

    (t1, f1) = interpolate(input_raw_file, 'cpu_nn_lena.dat', 'cpu', 1, 'nn', 8000, 4000)
    (t2, f2) = interpolate(input_raw_file, 'gpu_nn_lena.dat', 'gpu', 1, 'nn', 8000, 4000)
    (t3, f3) = interpolate(input_raw_file, 'cpu_bl_lena.dat', 'cpu', 1, 'bl', 8000, 4000)
    (t4, f4) = interpolate(input_raw_file, 'gpu_bl_lena.dat', 'gpu', 1, 'bl', 8000, 4000)

    if filecmp.cmp(f1, f2, shallow=True):
        print("NN interpolation on GPU is bit exact with CPU")
    
    if filecmp.cmp(f3, f4, shallow=True):
        print("Bilinear interpolation on GPU is bit exact with CPU")        


def exercise(input_raw_file):

    device = 'gpu'
    interp = 'bl'

    (t1, f1) = interpolate(input_raw_file, device + '_' + interp + '_lena.dat', device, 1, interp, 256, 300)
    (t2, f2) = interpolate(input_raw_file, device + '_' + interp + '_lena.dat', device, 1, interp, 2000, 1000)
    (t3, f3) = interpolate(input_raw_file, device + '_' + interp + '_lena.dat', device, 1, interp, 1000, 2000)
    (t4, f4) = interpolate(input_raw_file, device + '_' + interp + '_lena.dat', device, 1, interp, 8000, 4000)

    convert_to_jpg(f1)
    convert_to_jpg(f2)
    convert_to_jpg(f3)
    convert_to_jpg(f4)

if __name__ == '__main__':
   
    raw_file = convert_to_raw('Lena.tiff')
    # exercise(raw_file)
    # quit()

    print("Checking bit-exactness between GPU processing and CPU processing")   
    check_bit_exactness(raw_file)
    
    print("Benchmarking execution time Cpu vs Gpu")    
    durations = benchmark_cpu_vs_gpu(raw_file)
    

    plot_graph(durations)
    
    quit()
