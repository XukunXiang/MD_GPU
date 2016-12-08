/*
	This code is using Molecular dynamics to study the evolution of the electron bunch form the photo cathod in Ultrafast electron diffraction experiment.

	Goals:
	1. [done] make this md c code working
	2. [done] implement OpenMP with this code 
	3. [done] check with the fortran version
	4. working CPU with PPPM
		* [done] set up Periodic boundary conditions
		* [done] particle pre-assignment for spacial decomposition on CPU
		*	data(position) move from host to device in share mem[look at cuda example]
		*	CHARGE ASSIGNMENT with weight function
		*	FIELD CALCULATION using fft library for electronic potential
				differential of potential can get the electronic field
		*	INTERPOLATION of force with the same weight function to get the force from neighboring mesh points 
	5. try to get PPPM working with GPU
	6. Other improvement
		* set up a class for neighbor list storage

	origian Fortran code version: MD_1121.f90

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <fftw3.h>

//simulation setup
#define N 500					//number of particles
#define Ntime 1000		//number of iternations
#define newR 2.0E6		//size of the simulation box ~ 100 um
#define cutoff 200.0	//lower limit of the initial distance between electrons ~ 100 nm

//PPPM setup
#define bn 10					//number of boxes per direction
#define boxcap 100		//temporary cap for particles in one box; update to class later
#define hx 2.0E5			//[real space] cell size newR/bn
#define hx3 8.0E15		//hx^3 as the cell volume
#define grid_offset 5	//move the center of grid to 0 to match the particles

//parameters
#define m 5.4858e-4		//elelctron mass
#define vc 5.85e3			//speed of light
#define PI 3.141592653589793	//  \pi

double inline getrand(){ return (double)rand()/(double)RAND_MAX; }

double PBC(double r1) {
	double pr;
	if (r1 > 0.0) {
	  r1 -= floor(r1/newR)*newR;
	  pr = (r1 < (0.5*newR)) ? r1 : (r1 - newR) ;
	}
	else if (r1 < 0.0) {
	  r1 += ceil(r1/newR)*newR;
	  pr = (r1 > (-0.5*newR))? r1 : (r1 + newR) ;
	}
	else {pr = r1;}
	if (abs(pr)> 0.5*newR) {printf("pbc failed! \n");}
	return pr;
}

double getPE(double r[N][3]){
	int i,j,k;
	double rel,vij;
	double PE_c = 0.0;
	for(j=0; j<(N-1); j++) {
		for(i=j+1; i<N; i++) {
			vij = 0.0;
			for(k=0; k<3; k++) {
				rel = r[i][k] - r[j][k];
				vij += pow(rel,2.0); //vij = r^2 for now
			}
			PE_c += pow(vij,-0.5); // r^(-1)
		}
	}
	return PE_c;
}

double getKE(double v[N][3]){
	int i,j;
	double KE_c = 0.0; //sum(v**2)
	for (i=0; i<N; i++){
		for (j=0; j<3; j++) {
			KE_c += pow(v[i][j],2.0);
		}
	}
	KE_c = 0.5*m*KE_c;
	return KE_c;
}

int inline pbc_box(int point){
	int pb;
	if (point < 0 ) {
		pb = point + bn;
	}
	else if (point > (bn-1)) {
		pb = point - bn;
	}
	else {
		pb = point;
	}
	return pb;
}

void charge_assign(int b[3], double basis[3],double rho[bn][bn][bn],int id,double w[N][2][2][2]){
	int dx,dy,dz,pbx,pby,pbz;
	double lx,ly,lz,weight;
	for (dx=0; dx<2; dx++){
		pbx = pbc_box(b[0]+dx);
	  lx = (dx == 0) ? (hx - basis[0]) : basis[0];
	  for (dy=0; dy<2; dy++){
	    pby = pbc_box(b[1]+dy);
	    ly = (dy == 0) ? (hx - basis[1]) : basis[1];
	    for (dz=0; dz<2; dz++){
	      pbz = pbc_box(b[2]+dz);
	      lz = (dz == 0) ? (hx - basis[2]) : basis[2];
	      weight = lx*ly*lz/hx3;
	      w[id][pbx][pby][pbz] = weight;
	      rho[pbx][pby][pbz] +=  weight;
	    }
	  }
	}
}

// assign electron into bn*bn*bn boxes and store their idx
void update_box(double r[N][3], int box[bn][bn][bn][boxcap],int boxid[N][3],double rho[bn][bn][bn],double w[N][2][2][2]) {
	int i,j,realb[3],b[3],num;
	double basis[3];
	for (i=0; i<N; i++) { 
		for (j=0; j<3; j++) {
			realb[j] = (int)floor(r[i][j]/hx);
			b[j] = (int)floor(r[i][j]/hx) + grid_offset;
			boxid[i][j] = b[j];
			basis[j] = r[i][j] - realb[j]*hx;
		}
		num = box[b[0]][b[1]][b[2]][0]+1;
		box[b[0]][b[1]][b[2]][0] = num;
		box[b[0]][b[1]][b[2]][num] = i;
		charge_assign(b,basis,rho,i,w);	
	}
}

void getForce(double f[N][3],double r[N][3], int box[bn][bn][bn][boxcap]) {
	int i,j,k;
	double rel[3], rel_c, fij[3];
	//use openmp here with (rel[3],rel_c,fij) private
	#pragma omp parallel for private(i,j,k,rel,rel_c,fij)
	for (j=0; j<(N-1); j++) {
		for (i=j+1; i<N; i++) {
			rel_c = 0.0;
			for (k=0; k<3; k++) {
				rel[k] = r[i][k] - r[j][k];
				rel[k] -= rint(rel[k]/newR)*newR;
				rel_c += pow(rel[k], 2.0 );
			}
			rel_c = pow(rel_c, -1.5 );
			for (k=0; k<3; k++) {
				fij[k] = rel[k]*rel_c;
				//atomic operation here
				#pragma omp atomic
				f[j][k] -= fij[k];
				#pragma omp atomic
				f[i][k] += fij[k];
			}
		}
	}
}

void verlet(double r[N][3],double v[N][3],double f[N][3], int box[bn][bn][bn][boxcap],int boxid[N][3],double rho[bn][bn][bn],double w[N][2][2][2], double dt){
	int i,j;
	double hdtm = 0.5*dt/m; 
	for (i=0; i<N; i++) {
		for (j=0; j<3; j++) {
			v[i][j] += f[i][j]*hdtm;
			r[i][j] += v[i][j]*dt;
			r[i][j] = PBC(r[i][j]);
			f[i][j] = 0.0; //setup for force calculation
		}
	}
	update_box(r,box,boxid,rho,w);
	getForce(f,r,box);
	for (i=0; i<N; i++) {
		for (j=0; j<3; j++) {
			v[i][j] += f[i][j]*hdtm;
		}
	}
}

int main() {
	double R[N][3] = {{0.0}}, V[N][3] = {{1.0}}, F[N][3]={{0.0}};
	int box[bn][bn][bn][boxcap]; // need to go to class or dynamic array later 
	// [0] is for number of electrons in the box, and then [1]-[number] is the electron id
	int boxid[N][3];
	int i, iter;
	int numb, check;
	double dt = 1.0, realt = 0.0;
	int plotstride = 20;
	double r0,r1,r2,rel0,rel1,rel2;
	double KE,PE;
	double rho[bn][bn][bn], w[N][2][2][2];
	FILE *RVo,*To,*initR;

	RVo = fopen("./RandV.dat","w+");
	To = fopen("./time.dat","w+");
	initR = fopen("./initR.xyz","w+");

	// Initialization in a sphere
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
		verlet(R,V,F,box,boxid,rho,w,dt);
		realt += dt;

		//output
		if ((iter % plotstride) == 1) {		
			PE = getPE(R);
			KE = getKE(V);
			printf("%7d \t %11.5f \t %11.5f \t %11.5f \n", (int)realt, PE, KE, PE+KE);
			
			//position output
			fprintf(initR,"%d \n", N);
			fprintf(initR,"%d \n", iter);
			for (i = 0; i<N; i++) {
				fprintf(initR,"%5d %11.3f \t %11.3f \t %11.3f \n",1,R[i][0],R[i][1],R[i][2]);
			}
		}

		if (iter == 200) dt = 5.0;
		if (iter == 500) dt = 15.0;
		if (iter == 700) dt = 50.0;
		if (iter == 2000) plotstride = 100;
	}

	fclose(RVo);
	fclose(To);
	fclose(initR);

	return 0;
}
