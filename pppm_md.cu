/*
	This code is using Molecular dynamics to study the evolution of the electron bunch form the photo cathod in Ultrafast electron diffraction experiment.

	origian Fortran code version: MD_1121.f90

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
//#include <fftw3.h>
#include "cuda.h"
#include <cufft.h>

//simulation setup
#define N 5000					//number of particles
#define Ntime 1 
#define newR 2.0E6		//size of the simulation box ~ 100 um
#define cutoff 200.0	//lower limit of the initial distance between electrons ~ 100 nm

//PPPM setup
#define bn 100					//number of boxes per direction
#define boxcap 5		//temporary cap for particles in one box; update to class later
#define hx 2.0E4			//[real space] cell size newR/bn
#define hx3 8.0E12		//hx^3 as the cell volume
#define bn3 1000000			// total grid/mesh point number

//parameters
#define m 5.4858e-4		//elelctron mass
#define vc 5.85e3			//speed of light
#define PI 3.141592653589793	//  \pi

double inline getrand(){ return (double)rand()/(double)RAND_MAX; }

//***************
__global__ void theKernel(double * r_d){
  //This is array flattening, (Array Width * Y Index + X Index)
  r_d[(gridDim.x * blockDim.x) * (blockIdx.y * blockDim.y + threadIdx.y) + (blockIdx.x * blockDim.x + threadIdx.x)] += 5;
}

void printGrid(double a[N][3]){
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 3; j++) {
      printf("%11.3f ",a[i][j]);
    }
    printf("\n");
  }
}


//****************
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
	int i,j,k,realb[3],b[3],num;
	double basis[3];
	for (i=0; i<bn; i++){
		for (j=0; j<bn; j++){
			for (k=0; k<bn; k++){
				box[i][j][k][0]=0;
				rho[i][j][k] = 0.0;
			}
		}
	}
	printf("update box starts\n");
	for (i=0; i<N; i++) {
//		printf("i = %d \n",i);
		for (j=0; j<3; j++) {
			realb[j] = (int)floor(r[i][j]/hx);
			//b[j] = (int)floor(r[i][j]/hx) + grid_offset;
			b[j] = (int)floor(r[i][j]/hx);
//			printf("%d ",b[j]);
			boxid[i][j] = b[j];
//			printf("%d ",boxid[i][j]);
			basis[j] = r[i][j] - realb[j]*hx;
		}
//		printf("\n");
		num = box[b[0]][b[1]][b[2]][0]+1;
		box[b[0]][b[1]][b[2]][0] = num;
		box[b[0]][b[1]][b[2]][num] = i;
		charge_assign(b,basis,rho,i,w);	
	}
}

void setupKpoints(float kpoints[bn]){
	float k;
	int i;
	float nyquist = (float)bn/newR/2.0;
	for (i=0; i<(bn/2+1); i++){
		kpoints[i] = 2*PI*(float)i/(float)(bn/2)*nyquist;
		if (i!=(bn/2) && i!=0) {
			kpoints[bn-i] = kpoints[i];
		}
	}
}

__global__ void OutputToInput(cufftComplex *f_fft_result,cufftComplex *b_fft_in_x,cufftComplex *b_fft_in_y,cufftComplex *b_fft_in_z,float *d_kpoints) {
	int tx = blockIdx.x*blockDim.x + threadIdx.x;
	int ty = blockIdx.y*blockDim.y + threadIdx.y;
	int tz = blockIdx.z*blockDim.z + threadIdx.z;
	__shared__ float ks[bn];
	cufftComplex in;
	float k2i,kx,ky,kz;
	int i,j,k;
	int b3 = bn/2+1;

	for (i=0; i<bn; i++){
		ks[i] = d_kpoints[i];
	}
	kx = ks[tx];
	ky = ks[ty];
	kz = ks[tz];
	int idx = 2*(tz + ty*b3 + tx*bn*b3);
	if(tx==0 && ty ==0 && tz==0) {
		k2i = 1.0;
	}
	else{
		k2i = 1.0/(kx*kx+ky*ky+kz*kz);
	}
	in.x = f_fft_result[idx].x;
	in.y = f_fft_result[idx].y;

	b_fft_in_x[idx].x = kx*k2i*in.y;
	b_fft_in_x[idx].y = -kx*k2i*in.x;
	b_fft_in_y[idx].x = ky*k2i*in.y;
	b_fft_in_y[idx].y = -ky*k2i*in.x;
	b_fft_in_z[idx].x = kz*k2i*in.y;
	b_fft_in_z[idx].y = -kz*k2i*in.x;	

}

void getGlobalField_cufft(double rho[bn][bn][bn],double field[3][bn][bn][bn]) {
	float h_input[bn][bn][bn],h_output[3][bn][bn][bn];
	float kpoints[bn];
	int i,j,k,idx;
	int b_complex = bn*bn*((int)bn/2+1);

	cufftReal *d_input,*d_output_x,*d_output_y,*d_output_z,*d_kpoints;
	cufftComplex 	*f_fft_result, *b_fft_in_x, *b_fft_in_y, *b_fft_in_z;
	cufftHandle plan_forward, plan_backward_x, plan_backward_y, plan_backward_z;
	
	float elapsed = 0.0;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	for(i=0; i<bn; i++){
		for(j=0; j<bn; j++){
			for(k=0; k<bn; k++){
				h_input[i][j][k] = rho[i][j][k];
			}
		}
	}

	printf("setting up K-points...\n");
	setupKpoints(kpoints);

	printf("cuFFT is starting...\n");
//	for(i=0; i<bn; i++){
//		printf("%11.3f ",h_input[0][0][i]);
//	}
//	printf("\n");

	//Allocate device momory
	cudaMalloc((void**)&d_input, bn3*sizeof(cufftReal));
	cudaMalloc((void**)&d_output_x, bn3*sizeof(cufftReal));
	cudaMalloc((void**)&d_output_y, bn3*sizeof(cufftReal));
	cudaMalloc((void**)&d_output_z, bn3*sizeof(cufftReal));
	cudaMalloc((void**)&f_fft_result, b_complex*sizeof(cufftComplex));
	cudaMalloc((void**)&b_fft_in_x, b_complex*sizeof(cufftComplex));
	cudaMalloc((void**)&b_fft_in_y, b_complex*sizeof(cufftComplex));
	cudaMalloc((void**)&b_fft_in_z, b_complex*sizeof(cufftComplex));
	cudaMalloc((void**)&d_kpoints, bn*sizeof(cufftReal));
	
	cudaEventRecord(start,0);
	//Copy host momory to device
	cudaMemcpy(d_input, &h_input[0][0][0], bn3, cudaMemcpyHostToDevice);
	cudaMemcpy(d_kpoints, &kpoints[0], bn, cudaMemcpyHostToDevice);
	
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);

	printf("The time for copy from host to device: %.3f ms\n", elapsed);

	//cuFFT plan
	cufftPlan3d(&plan_forward, bn, bn, bn, CUFFT_R2C);
	cufftPlan3d(&plan_backward_x, bn, bn, bn, CUFFT_C2R);
	cufftPlan3d(&plan_backward_y, bn, bn, bn, CUFFT_C2R);
	cufftPlan3d(&plan_backward_z, bn, bn, bn, CUFFT_C2R);

	printf("forward fft is starting...\n");
	cudaEventRecord(start,0);
	//forward fft
	cufftExecR2C(plan_forward, d_input, f_fft_result);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("The time for Forward FFT: %.3f ms\n", elapsed);
	
	//transforming output_of_fft to input_of_ifft
	printf("output of fft ==> input of ifft\n");
	dim3 DimGrid(bn,1,1);
	dim3 DimBlock(1,bn,bn/2+1);
	OutputToInput<<<DimGrid,DimBlock>>>(f_fft_result,b_fft_in_x,b_fft_in_y,b_fft_in_z,d_kpoints);

	printf("backward fft is starting...\n");
	cudaEventRecord(start,0);
	//ifft
	cufftExecC2R(plan_backward_x, b_fft_in_x, d_output_x);
	cufftExecC2R(plan_backward_y, b_fft_in_y, d_output_y);
	cufftExecC2R(plan_backward_z, b_fft_in_z, d_output_z);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("The time for Backward FFT: %.3f ms\n", elapsed);
	
	cudaEventRecord(start,0);
	//Copy device memory to host
	cudaMemcpy(&h_output[0][0][0][0], d_output_x, bn3, cudaMemcpyDeviceToHost);
	cudaMemcpy(&h_output[1][0][0][0], d_output_y, bn3, cudaMemcpyDeviceToHost);
	cudaMemcpy(&h_output[2][0][0][0], d_output_z, bn3, cudaMemcpyDeviceToHost);

	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	printf("The time for copy data from device to Host: %.3f ms\n", elapsed);
//	for(i=0; i<bn; i++){
//		printf("%11.3f ",h_output[0][0][i]);
//	}
//	printf("\n");
	
	for(i=0; i<bn; i++){
		for(j=0; j<bn; j++){
			for(k=0; k<bn; k++){
				field[0][i][j][k] = (double)h_output[0][i][j][k];
				field[1][i][j][k] = (double)h_output[1][i][j][k];
				field[2][i][j][k] = (double)h_output[2][i][j][k];
			}
		}
	}

	//Destory cuFFT context
//	cufftDestroy(plan_forward);
//	cufftDestroy(plan_backward_x);
//	cufftDestroy(plan_backward_y);
//	cufftDestroy(plan_backward_z);
//
//	free(d_input);
//	free(d_output);
//	free(f_fft_result);
//	free(b_fft_in_x);
//	free(b_fft_in_y);
//	free(b_fft_in_z);

}

void getForce(double f[N][3],double r[N][3], int box[bn][bn][bn][boxcap],int boxid[N][3],double rho[bn][bn][bn],double w[N][2][2][2]) {
	int i,j,k,fx,b[3],dbx,dby,dbz,pbx,pby,pbz,nb_j;
	double rel[3], rel_c, rel_c2, fij[3], field[3][bn][bn][bn], ri[3], rj[3];
	double dummy;
	int jx,jy,jz,ijx,ijy,ijz;
	double jrel2;

//	getGlobalField(rho,field); 
//******[GPU] cufft******
	getGlobalField_cufft(rho,field);
//**********
	

//	#pragma omp parallel for private(bx,by,bz,pbx,pby,pbz,dbx,dby,dbz,nb_j,rel_c2,j,k,dummy)
	for (i=0; i<N; i++) {
		for (fx=0; fx<3; fx++){
			b[fx] = boxid[i][fx];
			ri[fx] = r[i][fx];
		}
		for (dbx=0; dbx<3; dbx++){
			pbx = pbc_box(b[0]+dbx-1);
			for (dby=0; dby<3; dby++){
				pby = pbc_box(b[1]+dby-1);
				for (dbz=0; dbz<3; dbz++){
					pbz = pbc_box(b[2]+dbz-1);
					for (j=1; j<(box[pbx][pby][pbz][0]+1); j++){
						nb_j = box[pbx][pby][pbz][j];
						if (nb_j != i) {
//*******PP*******
							rel_c2 = 0.0;
							for (k=0; k<3; k++) {
								rj[k] = r[nb_j][k];
								dummy = ri[k]-rj[k];
								rel[k] = dummy - floor(dummy/newR)*newR;
								rel_c2 += rel[k]*rel[k];
							}
							for(k=0; k<3; k++) {
							//	if (rel_c2 < 1e-6) {rel_c2 = 1e-6; printf("small number!");}
								f[i][k] += rel[k]/pow(rel_c2,1.5);
							}
//*******PM*******
							//cancel nb_j on neighboring grid point for PM
							for (jx=0; jx<2; jx++){
								for (jy=0; jy<2; jy++){
									for (jz=0; jz<2; jz++){
										//get to each neighboring gird point
										ijx = pbc_box(b[0]+jx);
										ijy = pbc_box(b[1]+jy);
										ijz = pbc_box(b[2]+jz);
										jrel2 = pow((rj[0]-ijx*hx),2.0)+pow((rj[1]-ijy*hx),2.0)+pow((rj[2]-ijz*hx),2.0);
										field[0][ijx][ijy][ijz] -= (ijx*hx-rj[0])/pow(jrel2,1.5);
										field[1][ijx][ijy][ijz] -= (ijy*hx-rj[1])/pow(jrel2,1.5);	
										field[2][ijx][ijy][ijz] -= (ijz*hx-rj[2])/pow(jrel2,1.5);	
									}
								}
							}
						}
					}
				}
			}
		}
		//PM
		for (dbx=0; dbx<2; dbx++){
			pbx = pbc_box(b[0]+dbx);
			for (dby=0; dby<2; dby++){
				pby = pbc_box(b[1]+dby);
				for (dbz=0; dbz<2; dbz++){
					pbz = pbc_box(b[2]+dbz);
					for (fx=0; fx<3; fx++){
						f[i][fx] += w[i][dbx][dby][dbz]*field[fx][dbx][dby][dbz];
					}
				}
			}
		}

	}
	printf("getforce finish\n");

}

void verlet(double r[N][3],double v[N][3],double f[N][3], int box[bn][bn][bn][boxcap],int boxid[N][3],double rho[bn][bn][bn],double w[N][2][2][2], double dt){
	int i,j;
	double hdtm = 0.5*dt/m;
	for (i=0; i<N; i++) {
		for (j=0; j<3; j++) {
			v[i][j] += f[i][j]*hdtm;
			r[i][j] += v[i][j]*dt;
			r[i][j] = r[i][j]-floor(r[i][j]/newR)*newR;
			f[i][j] = 0.0; //setup for force calculation
		}
	}
	update_box(r,box,boxid,rho,w);
	getForce(f,r,box,boxid,rho,w);
	for (i=0; i<N; i++) {
		for (j=0; j<3; j++) {
			v[i][j] += f[i][j]*hdtm;
		}
	}
	printf("update v finish\n");
}

void inputbinning(double r[N][3],double rtest[N][3]){
  //STEP 1: ALLOCATE
  double *r_d;
  int N3size = sizeof(double)*N*3;
  cudaMalloc((void **) &r_d, N3size);
  
  //STEP 2: TRANSFER
  cudaMemcpy(r_d, r, N3size, cudaMemcpyHostToDevice);

  //STEP 3: SET UP
  dim3 blockSize(N,1,1);
  dim3 gridSize(1,3,1);

  //STEP 4: RUN
  theKernel<<<gridSize, blockSize>>>(r_d);
  
  //STEP 5: TRANSFER
  printGrid(r);
  cudaMemcpy(rtest, r_d, N3size, cudaMemcpyDeviceToHost);
  printf("--------------------\n");
  printGrid(r);
  printf("--------------------\n");
  printGrid(rtest);
}

int main() {
	double R[N][3] = {{0.0}}, V[N][3] = {{0.0}}, F[N][3]={{0.0}};
	int box[bn][bn][bn][boxcap]; // need to go to class or dynamic array later 
	// [0] is for number of electrons in the box, and then [1]-[number] is the electron id
	int boxid[N][3];
	int i, iter;
	int numb, check;
	double dt = 2e-4, realt = 0.0;
	double r0,r1,r2,rel0,rel1,rel2;
	double rho[bn][bn][bn], w[N][2][2][2];

	// Initialization in a sphere
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
	
	for (iter = 0; iter< Ntime; iter++) {
		printf("iter %d: \n", iter);
		verlet(R,V,F,box,boxid,rho,w,dt);
		realt += dt;
		printf("update realtime \n");
	}

//******[GPU] Input binning for local correct ******

//	double rtest[N][3]={{0.0}};
//	inputbinning(R,rtest);

//**********


}
