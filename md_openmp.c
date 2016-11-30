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
	return (double)rand()/(double)RAND_MAX;
}

int main() {
	double R[N][3] = {{0.0}}, V[N][3] = {{0.0}}, F[N][3]={{0.0}};
	int i, j, k, iter;
	int numb, check;
	double dt = 1.0, realt = 0.0;
	double r0,r1,r2,rel0,rel1,rel2;
	int plotstride = 20;
	FILE *RVo,*To,*initR;

	RVo = fopen("./RandV.dat","w+");
	To = fopen("./time.dat","w+");
	initR = fopen("./initR","w+");

	srand(time(NULL));
	numb = 0;
	while (numb < N) {
		r0 = newR*getrand();
		r1 = newR*getrand();
		r2 = newR*getrand();
		check = 0;
		if (sqrt(r0*r0+r1*r1+r2*r2) < newR) {
			for (i = 0; i < numb; i++){
				rel0 = R[i][0] - r0;
				rel1 = R[i][1] - r1;
				rel2 = R[i][2] - r2;
				if (sqrt(rel0*rel0+rel1*rel1+rel2*rel2) < cutoff) {
					check = 1;
					break;
				}				
			}
			if (check == 0) {
				R[numb][0] = r0;
				R[numb][1] = r1;
				R[numb][2] = r2;
				numb += 1;
			}
		}
	}
	
	for (i = 0; i<N; i++) {
		fprintf(initR,"%11.3f \t %11.3f \t %11.3f \n",R[i][0],R[i][1],R[i][2]);
	}
//	for (i = 0; i< N)
	

	fclose(RVo);
	fclose(To);
	fclose(initR);
}
