/*
	This code is using Molecular dynamics to study the evolution of the electron bunch form the photo cathod in Ultrafast electron diffraction experiment.

	Goals:
	1. make this md c code working
	2. implement OpenMP with this code and check with the fortran version
	3. implement CUDA to compare the result with OpenMP
	4. try to get PPPM working with GPU

	origian Fortran code version: MD_1121.f90

*/


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

//simulation setup
#define N 1000				//number of particles
#define Ntime 3001		//number of iternations
#define newR 46022.8	//initial radius of bunch
#define cutoff 20			//lower limit of the initial distance between electrons

//parameters
#define m 5.4858E-4		//elelctron mass
#define vc 5.85E3			//speed of light
#define PI 3.141592653589793	//  \pi

double inline getrand(){
	return double(rand()/RAND_MAX;
}

int main() {
	srand(time(NULL));
	printf("\d \n",RANDMAX);
	printf("\11.3f \n",getrand());
	



}
