#include <amp.h>
#include <random>
#include <iostream>
#include <amp_math.h>
#include <time.h> 
#include <fstream>
using namespace std;
using namespace concurrency;
using namespace precise_math;
#define pi 	3.14159265358979323846
void main()
{
	clock_t start = clock();
	double hrms = 0.000000001;
	double length = 0.03;
	int pointNumber = 1000;
	int frameCount = 2000;
	int halfPointNumber = (int)(pointNumber / 2) + 1;
	uniform_real_distribution<> distr(0, 1);
	double* phases;
	phases = (double*)malloc(sizeof(double) * halfPointNumber*frameCount);
	double* psd;
	psd = (double*)malloc(sizeof(double) * halfPointNumber);
	int a = -2; // <=> /100 in psd
	int b = -2;
	int seed = rand();
	//for (int i=0; i < halfPointNumber * frameCount; i++)
	//{
	//	mt19937 gen(rand());
	//	phases[i] = distr(gen);
	//	//cout << distr(gen) << endl;
	//}

	std::cout << "Initialized in " << (double)(clock() - start) / CLOCKS_PER_SEC << "sec" << endl;
	array_view<double, 1> psdDev1(halfPointNumber, psd);
	array_view<double, 2> phasesDev1(frameCount, halfPointNumber, phases);
	psdDev1.discard_data();
	phasesDev1.discard_data();
	parallel_for_each(
		phasesDev1.extent,
		[=](index<2> idx) restrict(amp) {
			phasesDev1[idx[0]][idx[1]] = precise_math::sin(precise_math::pow((double)(idx[0] * halfPointNumber + idx[1]), seed));
		}
	);

	start = clock();
	//TODO
	//psd[0] = precise_math::exp(b * precise_math::log(pi / length * halfPointNumber)) / 100;
	array_view<double, 1> psdDev(halfPointNumber, psd);
	array_view<double, 2> phasesDev(frameCount, halfPointNumber, phases);
	psdDev.discard_data();
	phasesDev.discard_data();
	parallel_for_each(
		phasesDev.extent,
		[=](index<2> idx) restrict(amp) {
			phasesDev[idx[0]][idx[1]] = precise_math::fabs(precise_math::sin(precise_math::pow((double)(idx[0]*halfPointNumber+idx[1]),seed* precise_math::fabs(precise_math::sin(precise_math::pow((double)(idx[0] * halfPointNumber + idx[1]), seed)))))) ;
		}
	);

	
	//psdDev[0] = precise_math::exp(b * precise_math::log(pi / length * halfPointNumber)) / 100;
	parallel_for_each(
		psdDev.extent,
		[=](index<1> idx) restrict(amp) {
			psdDev[idx[0]+1] = precise_math::pow((idx[0] + 1) * pi/length * halfPointNumber,b)/100;
			//psdDev[idx[0]][idx[1] + 1] = idx[1] + 1;
		}
	);
	psdDev.synchronize();
	std::cout << "Done in " << (double)(clock() - start) / CLOCKS_PER_SEC << "sec" << endl;
	for (int j = 0; j < halfPointNumber; j++)
	{
			//std::cout << psdDev[j] << '\t';
	}
	std::cout << endl;
	ofstream fout;
	fout.open("RNDTableBetter.txt");
	for (int i = 0; i < frameCount; i++)
	{
		for (int j = 0; j < halfPointNumber; j++)
		{
			//std::cout << phasesDev[i][j] << endl;
			fout << phasesDev[i][j] << endl;
		}
		//std::cout << endl;
	}
	fout.close();
	/*free(psd);
	free(phases);*/
}