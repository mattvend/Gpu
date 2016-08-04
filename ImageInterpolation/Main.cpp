/**********************************************************************************/
/* The MIT License(MIT)                                                           */
/*                                                                                */
/* Copyright(c) 2016-2016 Matthieu Vendeville                                     */
/*                                                                                */
/* Permission is hereby granted, free of charge, to any person obtaining a copy   */
/* of this software and associated documentation files(the "Software"), to deal   */
/* in the Software without restriction, including without limitation the rights   */
/* to use, copy, modify, merge, publish, distribute, sublicense, and / or sell    */
/* copies of the Software, and to permit persons to whom the Software is          */
/* furnished to do so, subject to the following conditions :                      */
/*                                                                                */
/* The above copyright notice and this permission notice shall be included in all */
/* copies or substantial portions of the Software.                                */
/*                                                                                */
/* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     */
/* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       */
/* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE     */
/* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         */
/* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  */
/* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  */
/* SOFTWARE.                                                                      */
/**********************************************************************************/

#include <stdio.h>
#include "ImCu.h"
#include <time.h>
#include <iostream>
using namespace std;

int main()
{
	int i;
	int iterations = 10;
	ImCu Instances[10] = { ImCu("512x512x8x1_lena.dat"), ImCu("512x512x8x1_lena.dat"), ImCu("512x512x8x1_lena.dat"), ImCu("512x512x8x1_lena.dat"), ImCu("512x512x8x1_lena.dat"), ImCu("512x512x8x1_lena.dat"), ImCu("512x512x8x1_lena.dat"), ImCu("512x512x8x1_lena.dat"), ImCu("512x512x8x1_lena.dat"), ImCu("512x512x8x1_lena.dat") };
	ImCu Instances2[10] = { ImCu("512x512x8x1_lena.dat"), ImCu("512x512x8x1_lena.dat"), ImCu("512x512x8x1_lena.dat"), ImCu("512x512x8x1_lena.dat"), ImCu("512x512x8x1_lena.dat"), ImCu("512x512x8x1_lena.dat"), ImCu("512x512x8x1_lena.dat"), ImCu("512x512x8x1_lena.dat"), ImCu("512x512x8x1_lena.dat"), ImCu("512x512x8x1_lena.dat") };

	//ImCu myimage = ImCu("512x512x8x1_lena.dat");
	
	
//	for (i = 0; i < iterations; i++){
//		Instances[i] = ImCu(myimage);
//	}
	// ImCu myimage2 = ImCu(myimage);
	
	const clock_t begin_time = clock();

	for (i = 0; i < iterations; i++){
		Instances[i].CUDA_InterpolateNN(8000, 4000);
	}
//	myimage2.CUDA_InterpolateBilinear(4000, 2000);

	// do something
	std::cout << float(clock() - begin_time) / CLOCKS_PER_SEC << '\n';
	Instances[0].Save2RawFile("popo.dat");
	
	const clock_t begin_time2 = clock();

	for (i = 0; i < iterations; i++){
		Instances2[i].InterpolateNN(8000, 4000);
	}
	//	myimage2.CUDA_InterpolateBilinear(4000, 2000);

	// do something
	std::cout << float(clock() - begin_time2) / CLOCKS_PER_SEC << '\n';;

	return 0;
}

