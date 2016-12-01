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
#define N 100				//number of particles
#define Ntime 10		//number of iternations
#define newR 46022.8	//initial radius of bunch
#define cutoff 20			//lower limit of the initial distance between electrons

//parameters
#define m 5.4858E-4		//elelctron mass
#define vc 5.85E3			//speed of light
#define PI 3.141592653589793	//  \pi

double inline getrand(){ return (double)rand()/(double)RAND_MAX; }

void getForce(double f[N][3],double r[N][3]) {
	int i,j,k;
	double rel[3], rel_c, fij[3];
	//use openmp here with (rel[3],rel_c) private
	for (j=0; j<(N-1); j++) {
		for (i=j+1; i<N; i++) {
			rel_c = 0.0;
			for (k=0; k<3; k++) {
				rel[k] = r[i][k] - r[j][k];
				rel_c += rel[k]*rel[k];
			}
			rel_c = - pow(rel_c,1.5 );
			for (k=0; k<3; k++) {
				fij[k] = rel[k]*rel_c;
				//atomic operation here
				f[j][k] -= fij[k];
				f[i][k] += fij[k];
			}
		}
	}
}

void verlet(double r[N][3],double v[N][3],double f[N][3], double dt){
	int i,j;
	double hdtm = 0.5*dt/m; 
	for (i=0; i<N; i++) {
		for (j=0; j<3; j++) {
			v[i][j] += f[i][j]*hdtm;
			r[i][j] += v[i][j]*dt;
		}
	}
	getForce(f,r);
	for (i=0; i<N; i++) {
		for (j=0; j<3; j++) {
			v[i][j] += f[i][j]*hdtm;
		}
	}
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
	initR = fopen("./initR.xyz","w+");

	srand(time(NULL));
	numb = 0;
	while (numb < N) {
		r0 = newR*(2*getrand()-1);
		r1 = newR*(2*getrand()-1);
		r2 = newR*(2*getrand()-1);
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
	
	//output initR for check
//	fprintf(initR,"%d \n", N);
//	fprintf(initR,"%d \n", 0);
//	for (i = 0; i<N; i++) {
//		fprintf(initR,"%5d %11.3f \t %11.3f \t %11.3f \n",1,R[i][0],R[i][1],R[i][2]);
//	}
	for (iter = 0; iter< Ntime; iter++) {
		verlet(R,V,F,dt);
		realt += dt;
/*
		//output
		if ((iter % plotstride) == 1) {		
		}
*/
	}

	fclose(RVo);
	fclose(To);
	fclose(initR);

	return 0;
}
